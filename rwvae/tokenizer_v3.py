"""RW-VAE v3 tokenizer (auto-encoder/RWVAE_V3_ALLOCATION_PLAN.md, v3.1 §2).

Encoder: frozen DINOv2 patch tokens of several blocks spread over depth (each through the model's final
LayerNorm, concatenated, 1x1 conv -> d) and a trainable RGB conv branch (-> r) are mixed per position and
per channel group by a gate alpha: h = alpha * d + (1 - alpha) * r. A grouped 1x1 head gives a diagonal
Gaussian posterior (mu, logvar) [B, latent_dim, g, g]; z = mu + exp(logvar / 2) * eps.

Decoder: the LightningDiT / VA-VAE AutoencoderKL post_quant_conv + decoder, initialised from a checkpoint.

Noise-level readouts (the core module, §2.2): z is standardised per channel with running moments
(stop-gradient) to u, noised on the LightningDiT linear path u_t = t u + (1 - t) eps (t = 1 is clean),
and two heads that see ONLY u_t predict (a) the last DINO block's patch tokens (cosine loss) and (b) a
low-resolution RGB image plus an edge map (L1). readout="split" uses t_sem for (a) and t_det for (b);
"shared" draws one t ~ U(0, 1) per sample for both heads (the RoAR-VAE protocol, arm T-f); "off" has no
heads. The DINO network itself is not part of this module: callers pass its features in.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("adaptive", "fixed", "dino_only")
READOUTS = ("split", "shared", "off")


@dataclass(frozen=True)
class V3Config:
    dino_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23)   # 0-based blocks: layers 3, 6, ..., 24
    dino_dim: int = 1024
    fusion: str = "adaptive"
    feat_dim: int = 128
    groups: int = 8
    latent_dim: int = 32
    rgb_channels: tuple[int, ...] = (32, 64, 96, 128)               # one stride-2 stage each: /16 for 4
    gate_hidden: int = 64
    logvar_min: float = -12.0
    logvar_max: float = 4.0
    readout: str = "split"
    t_sem: float = 0.35
    t_det: float = 0.80
    sem_depth: int = 4
    sem_width: int = 512
    sem_heads: int = 8
    det_width: int = 256
    moments_momentum: float = 0.99

    def __post_init__(self) -> None:
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {self.fusion!r}")
        if self.readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {self.readout!r}")
        if not self.dino_layers or len(set(self.dino_layers)) != len(self.dino_layers):
            raise ValueError(f"dino_layers must be distinct block indices, got {self.dino_layers!r}")
        if self.feat_dim % self.groups or self.latent_dim % self.groups:
            raise ValueError(f"feat_dim={self.feat_dim} and latent_dim={self.latent_dim} must be divisible by "
                             f"groups={self.groups}")
        if self.rgb_channels[-1] != self.feat_dim:
            raise ValueError(f"the last RGB stage must output feat_dim={self.feat_dim}, got {self.rgb_channels}")
        if not (0.0 < self.t_sem < 1.0 and 0.0 < self.t_det < 1.0):
            raise ValueError(f"t_sem and t_det must lie in (0, 1), got {self.t_sem}, {self.t_det}")

    @classmethod
    def from_dict(cls, d: Mapping) -> "V3Config":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"V3Config: unknown keys {unknown}; fields are {sorted(names)}")
        d = dict(d)
        for key in ("dino_layers", "rgb_channels"):
            if key in d:
                d[key] = tuple(int(v) for v in d[key])
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @property
    def target_layer(self) -> int:
        """The DINO block whose tokens the semantic readout predicts: the deepest one used."""
        return max(self.dino_layers)


# --------------------------------------------------------------------------- building blocks
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, c), c)


def channel_layer_norm(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm over channels at every position (no affine): [B, C, H, W] -> [B, C, H, W]."""
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                  _gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RGBBranch(nn.Module):
    """[B, 3, S, S] -> [B, channels[-1], S / 2^len(channels), ...]."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers += [nn.Conv2d(cin, c, 3, stride=2, padding=1), _gn(c), nn.SiLU(), ResBlock(c)]
            cin = c
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticReadout(nn.Module):
    """Noisy latent u_t [B, C, g, g] -> predicted DINO tokens [B, out_dim, g, g]: 1x1 conv, learned position
    embedding, a pre-norm Transformer encoder, linear head. RoAR-VAE (arXiv:2610.26216, Table 5) found a
    Transformer predictor far better than an MLP for this kind of readout."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, heads: int, grid: int):
        super().__init__()
        self.inp = nn.Conv2d(in_dim, width, 1)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, _, H, W = u.shape
        x = self.inp(u).flatten(2).transpose(1, 2) + self.pos
        x = self.out(self.norm(self.blocks(x)))
        return x.transpose(1, 2).reshape(B, -1, H, W)


class DetailReadout(nn.Module):
    """Noisy latent u_t [B, C, g, g] -> [B, 4, 4g, 4g]: RGB (3 channels) and an edge map (1 channel)."""

    def __init__(self, in_dim: int, width: int, scale: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_dim, width, 3, padding=1), nn.SiLU(), ResBlock(width),
                                 ResBlock(width), _gn(width), nn.SiLU(), nn.Conv2d(width, 4 * scale * scale, 1),
                                 nn.PixelShuffle(scale))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


def edge_map(images: torch.Tensor, size: int) -> torch.Tensor:
    """Soft edges of [-1, 1] images [B, 3, S, S] at size x size: gray in [0, 1] -> area resize -> 3x3
    Gaussian blur (sigma 1) -> Sobel gradient magnitude. Returns [B, 1, size, size]."""
    w = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = F.interpolate(((images + 1) / 2 * w).sum(1, keepdim=True), size=(size, size), mode="area")
    g = images.new_tensor([math.exp(-0.5), 1.0, math.exp(-0.5)])
    g = (g[:, None] * g[None, :]) / (g.sum() ** 2)
    gray = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), g.view(1, 1, 3, 3))
    sx = images.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(padded, sx), F.conv2d(padded, sx.transpose(2, 3))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def kl_elementwise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, 1)) per element, in nats."""
    return 0.5 * (mu * mu + logvar.exp() - 1.0 - logvar)


def rate_nats_per_token(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Nominal rate R_hat: KL summed over channels, averaged over positions and images (nats/token)."""
    return kl_elementwise(mu, logvar).sum(1).mean()


class RateLagrangian:
    """Augmented Lagrangian that locks R_hat at a target (plan §2.1.1, control pair only):
    delta = (R_hat - target) / target, loss = lam * delta + quad * delta^2, and every `every` steps
    lam <- lam + lr * mean(delta) over those steps (callers average delta over ranks first)."""

    def __init__(self, target: float, *, quad: float = 5.0, lr: float = 0.1, every: int = 100, lam: float = 0.0):
        if not target > 0:
            raise ValueError(f"rate target must be > 0 nats/token, got {target!r}")
        self.target, self.quad, self.lr, self.every = float(target), float(quad), float(lr), int(every)
        self.lam, self._sum, self._n = float(lam), 0.0, 0

    def loss(self, rate: torch.Tensor) -> tuple[torch.Tensor, float]:
        delta = (rate - self.target) / self.target
        return self.lam * delta + self.quad * delta * delta, float(delta.detach())

    def observe(self, delta: float) -> bool:
        """Accumulate one step's (rank-averaged) delta; returns True when lam was updated."""
        self._sum, self._n = self._sum + float(delta), self._n + 1
        if self._n < self.every:
            return False
        self.lam += self.lr * self._sum / self._n
        self._sum, self._n = 0.0, 0
        return True

    def state_dict(self) -> dict:
        return {"lam": self.lam, "sum": self._sum, "n": self._n}

    def load_state_dict(self, state: Mapping) -> None:
        self.lam, self._sum, self._n = float(state["lam"]), float(state["sum"]), int(state["n"])


# --------------------------------------------------------------------------- the tokenizer
class TokenizerV3(nn.Module):
    """Trainable part of the v3 tokenizer: encoder branches, gate, posterior head, decoder and readouts.

    `decoder_ae` is an AutoencoderKL (LightningDiT tokenizer.autoencoder) whose post_quant_conv and decoder
    are kept (its encoder is dropped); its embed_dim must equal latent_dim. `grid` is the latent side
    (16 for 256 px images and DINOv2/14 at 224 px).
    """

    def __init__(self, cfg: V3Config, decoder_ae: nn.Module, *, grid: int = 16):
        super().__init__()
        if int(decoder_ae.embed_dim) != cfg.latent_dim:
            raise ValueError(f"decoder embed_dim {decoder_ae.embed_dim} != latent_dim {cfg.latent_dim}")
        self.cfg, self.grid = cfg, int(grid)
        self.dino_proj = nn.Conv2d(len(cfg.dino_layers) * cfg.dino_dim, cfg.feat_dim, 1)
        if cfg.fusion != "dino_only":
            self.rgb = RGBBranch(cfg.rgb_channels)
        if cfg.fusion == "adaptive":
            self.gate = nn.Sequential(nn.Conv2d(2 * cfg.feat_dim, cfg.gate_hidden, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(cfg.gate_hidden, cfg.groups, 1))
            nn.init.zeros_(self.gate[-1].weight)        # alpha = sigmoid(0) = 0.5 at the start
            nn.init.zeros_(self.gate[-1].bias)
        self.posterior = nn.Conv2d(cfg.feat_dim, 2 * cfg.latent_dim, 1, groups=cfg.groups)
        self.post_quant_conv = decoder_ae.post_quant_conv
        self.decoder = decoder_ae.decoder
        if cfg.readout != "off":
            self.sem_head = SemanticReadout(cfg.latent_dim, cfg.dino_dim, cfg.sem_width, cfg.sem_depth,
                                            cfg.sem_heads, self.grid)
            self.det_head = DetailReadout(cfg.latent_dim, cfg.det_width)
        self.register_buffer("z_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("z_var", torch.ones(cfg.latent_dim))
        self.register_buffer("moments_init", torch.zeros((), dtype=torch.bool))

    # ---------------------------------------------------------------- parameter groups
    def decoder_parameters(self) -> list[nn.Parameter]:
        return list(self.post_quant_conv.parameters()) + list(self.decoder.parameters())

    def encoder_parameters(self) -> list[nn.Parameter]:
        dec = {id(p) for p in self.decoder_parameters()}
        return [p for p in self.parameters() if id(p) not in dec]

    # ---------------------------------------------------------------- encoder
    def encode(self, images: torch.Tensor, dino_layers: torch.Tensor, freeze_gate: bool = False) -> dict:
        """images [B, 3, S, S] in [-1, 1]; dino_layers [B, L, D, g, g] (the blocks of cfg.dino_layers, in that
        order, each after the DINO final norm). Returns {mu, logvar, alpha ([B, groups, g, g]; None for
        dino_only), d, r}.

        freeze_gate (adaptive fusion only) pins alpha at 0.5 while still running the gate network times 0, so
        under DDP every parameter takes part and the gate gets zero gradient. The training script sets it during
        the warm-up: regressing onto the VA-VAE posterior mean (itself DINO-aligned) otherwise drives alpha to
        ~1 before the real objective starts (seen on the first T-c run: alpha 0.9999 at step 800)."""
        cfg = self.cfg
        B, L = dino_layers.shape[:2]
        if L != len(cfg.dino_layers):
            raise ValueError(f"got {L} DINO layers, config uses {len(cfg.dino_layers)}")
        d = channel_layer_norm(self.dino_proj(dino_layers.flatten(1, 2).float()))
        if cfg.fusion == "dino_only":
            r, alpha, h = None, None, d
        else:
            r = channel_layer_norm(self.rgb(images))
            if r.shape[-2:] != d.shape[-2:]:
                raise ValueError(f"RGB grid {tuple(r.shape[-2:])} != DINO grid {tuple(d.shape[-2:])}")
            if cfg.fusion == "fixed":
                alpha = torch.full((B, cfg.groups, *d.shape[-2:]), 0.5, device=d.device, dtype=d.dtype)
            else:
                alpha = torch.sigmoid(self.gate(torch.cat([d, r], 1)))
                if freeze_gate:
                    alpha = 0.5 + 0.0 * alpha
            a = alpha.repeat_interleave(cfg.feat_dim // cfg.groups, dim=1)
            h = a * d + (1 - a) * r
        params = self.posterior(h)
        per = cfg.latent_dim // cfg.groups
        params = params.reshape(B, cfg.groups, 2 * per, *h.shape[-2:])
        mu = params[:, :, :per].reshape(B, cfg.latent_dim, *h.shape[-2:])
        logvar = params[:, :, per:].reshape(B, cfg.latent_dim, *h.shape[-2:]).clamp(cfg.logvar_min, cfg.logvar_max)
        return {"mu": mu, "logvar": logvar, "alpha": alpha, "d": d, "r": r}

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        eps = torch.randn_like(mu) if noise is None else noise
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, images: torch.Tensor, dino_layers: torch.Tensor, dino_target: torch.Tensor | None = None,
                freeze_gate: bool = False) -> dict:
        """One training forward (what DDP wraps): encode, sample z, update the running moments (training mode),
        decode, and the readout losses when readout != "off". Returns encode()'s dict plus z, x_rec and, with
        readouts, sem / det / det_rgb / det_edge."""
        out = self.encode(images, dino_layers, freeze_gate=freeze_gate)
        z = self.sample(out["mu"], out["logvar"])
        if self.training:
            self.update_moments(z)
        out.update(z=z, x_rec=self.decode(z))
        if self.cfg.readout != "off":
            if dino_target is None:
                raise ValueError("the readouts need dino_target")
            out.update(self.readout_losses(z, images, dino_target))
        return out

    # ---------------------------------------------------------------- noise-level readouts
    @torch.no_grad()
    def update_moments(self, z: torch.Tensor) -> None:
        """Running per-channel mean / variance of z (EMA with cfg.moments_momentum; the first call copies)."""
        zf = z.detach().float()
        mean, var = zf.mean((0, 2, 3)), zf.var((0, 2, 3), unbiased=False)
        if not bool(self.moments_init):
            self.z_mean.copy_(mean)
            self.z_var.copy_(var)
            self.moments_init.fill_(True)
        else:
            m = self.cfg.moments_momentum
            self.z_mean.mul_(m).add_(mean, alpha=1 - m)
            self.z_var.mul_(m).add_(var, alpha=1 - m)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean.view(1, -1, 1, 1)) / torch.sqrt(self.z_var.view(1, -1, 1, 1) + 1e-6)

    def readout_times(self, batch: int, device, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(t_sem [B], t_det [B]) for the configured readout mode."""
        if self.cfg.readout == "split":
            return (torch.full((batch,), self.cfg.t_sem, device=device),
                    torch.full((batch,), self.cfg.t_det, device=device))
        if self.cfg.readout == "shared":
            t = torch.rand(batch, generator=generator, device="cpu" if generator is not None else device)
            t = t.to(device)
            return t, t
        raise ValueError("readout is off")

    def readout_losses(self, z: torch.Tensor, images: torch.Tensor, dino_target: torch.Tensor,
                       generator: torch.Generator | None = None) -> dict:
        """{sem, det, det_rgb, det_edge}: 1 - cosine to the DINO target tokens [B, D, g, g] at t_sem, and the
        L1 of the low-resolution RGB + edge map at t_det. Gradients reach the encoder through z."""
        u = self.standardize(z)
        B = u.shape[0]
        t_sem, t_det = self.readout_times(B, u.device, generator)

        def noisy(t):
            t = t.view(B, 1, 1, 1).to(u.dtype)
            return t * u + (1 - t) * torch.randn_like(u)

        pred = self.sem_head(noisy(t_sem))
        sem = 1 - F.cosine_similarity(pred.float(), dino_target.float(), dim=1).mean()
        det = self.det_head(noisy(t_det)).float()
        size = det.shape[-1]
        rgb_t = F.interpolate(images, size=(size, size), mode="area")
        edge_t = edge_map(images, size)
        det_rgb = (det[:, :3] - rgb_t).abs().mean()
        det_edge = (det[:, 3:] - edge_t).abs().mean()
        return {"sem": sem, "det": det_rgb + det_edge, "det_rgb": det_rgb, "det_edge": det_edge}


# --------------------------------------------------------------------------- construction / checkpoints
def build_decoder_ae(embed_dim: int = 32, ch_mult: Sequence[int] = (1, 1, 2, 2, 4)) -> nn.Module:
    """A fresh LightningDiT AutoencoderKL (its decoder half is what TokenizerV3 keeps)."""
    from rwvae.integration import _import_lightningdit
    module = _import_lightningdit("tokenizer.autoencoder")
    return module.AutoencoderKL(embed_dim=int(embed_dim), ch_mult=tuple(int(c) for c in ch_mult),
                                use_variational=True, ckpt_path=None, model_type="vavae")


def load_decoder_weights(tok: TokenizerV3, state: Mapping[str, torch.Tensor]) -> dict:
    """Copy post_quant_conv.* and decoder.* of an AutoencoderKL state dict (e.g.
    integration.load_tokenizer_state_dict) into tok, strictly. Returns {"num_tensors": n}."""
    own = {**{f"post_quant_conv.{k}": v for k, v in tok.post_quant_conv.state_dict().items()},
           **{f"decoder.{k}": v for k, v in tok.decoder.state_dict().items()}}
    missing = sorted(own.keys() - state.keys())
    mismatched = sorted(k for k in own.keys() & state.keys() if own[k].shape != state[k].shape)
    if missing or mismatched:
        raise ValueError(f"decoder weights: {len(missing)} missing {missing[:3]}, "
                         f"{len(mismatched)} shape-mismatched {mismatched[:3]}")
    tok.post_quant_conv.load_state_dict({k[len("post_quant_conv."):]: state[k] for k in own
                                         if k.startswith("post_quant_conv.")})
    tok.decoder.load_state_dict({k[len("decoder."):]: state[k] for k in own if k.startswith("decoder.")})
    return {"num_tensors": len(own)}


def save_tokenizer(path, tok: TokenizerV3, *, decoder_arch: Mapping, extra: Mapping | None = None) -> None:
    """Write {"v3_config", "decoder_arch", "grid", "model", **extra} (temp file + rename)."""
    import os
    from pathlib import Path
    path = Path(path)
    payload = {"v3_config": tok.cfg.to_dict(), "decoder_arch": dict(decoder_arch), "grid": tok.grid,
               "model": tok.state_dict(), **dict(extra or {})}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_tokenizer(path, device="cpu") -> tuple[TokenizerV3, dict]:
    """(TokenizerV3 in eval mode with the checkpoint's weights, the raw checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt["decoder_arch"]
    tok = TokenizerV3(V3Config.from_dict(ckpt["v3_config"]),
                      build_decoder_ae(arch["embed_dim"], arch["ch_mult"]), grid=int(ckpt["grid"]))
    tok.load_state_dict(ckpt["model"], strict=True)
    return tok.to(device).eval(), ckpt
