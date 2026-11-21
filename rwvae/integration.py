"""Adapter that wraps LightningDiT's VA-VAE latent with the relevance rate gate.

The gate acts on the latent z = VA_VAE.encode_images(x) of shape [B, 32, 16, 16]
(f16d32). This file is the *integration seam*; it is GPU-only (LightningDiT's
VA_VAE.load() moves the model to CUDA) and is therefore exercised by the training
script's smoke path on plain tensors, not run here.

Note on training: LightningDiT ships VA-VAE *inference* only (encoder/decoder
training "provided later"). To train the gate end-to-end you either (a) fine-tune
on top of released VA-VAE weights, or (b) use REPA-E's trainable VAE (its modular
YAML loss) and insert this gate at the latent. Either way the gate and losses
below are unchanged.

The "v2" section appended at the end (plan v2.1, docs/ARCHITECTURE_V2.md §3.10) is the
only seam to third-party code -- LightningDiT tokenizer/DiT/transport, timm DINOv2, lpips
and pytorch_fid -- and imports all of them lazily, so importing this module stays cheap.
"""
from __future__ import annotations

import os
import sys

import torch

from .gate import RelevanceRateGate


def _add_lightningdit_to_path(lightningdit_root: str) -> None:
    root = os.path.abspath(lightningdit_root)
    if not os.path.isdir(os.path.join(root, "tokenizer")):
        raise FileNotFoundError(
            f"LightningDiT not found at {root!r} (expected a 'tokenizer' subdir). "
            "Download it per README (third_party/LightningDiT)."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


class GatedVAVAE:
    """VA-VAE + relevance rate gate. Encodes to a *gated* latent and decodes it.

    GPU required (VA_VAE uses .cuda()). Construct with the path to the cloned
    LightningDiT repo and a VA-VAE config yaml.
    """

    def __init__(self, lightningdit_root: str, vavae_config: str, channels: int = 32,
                 budget: float | None = None):
        _add_lightningdit_to_path(lightningdit_root)
        from tokenizer.vavae import VA_VAE  # lazy import; needs LightningDiT + CUDA

        self.vae = VA_VAE(vavae_config)
        self.gate = RelevanceRateGate(channels, budget=budget).cuda()

    def encode_gated(self, images: torch.Tensor) -> torch.Tensor:
        """images -> gated latent z' = g ⊙ encode(images)."""
        z = self.vae.encode_images(images)   # [B, C, H, W], on cuda
        return self.gate(z)

    def decode(self, z: torch.Tensor):
        return self.vae.decode_to_images(z)


# ======================================================================================
# v2 (plan v2.1; docs/ARCHITECTURE_V2.md §3.10). Appended verbatim after the v1 code, so
# it carries its own imports. Third-party code (LightningDiT tokenizer.autoencoder,
# models.lightningdit and transport; timm; lpips; pytorch_fid) is imported lazily inside
# the functions below and is never modified.
# ======================================================================================
import hashlib
import importlib
import importlib.util
import logging
import pickle
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# <repo>/../third_party/LightningDiT; $RWVAE_LIGHTNINGDIT_ROOT overrides it (read at call time).
DEFAULT_LIGHTNINGDIT_ROOT = Path(__file__).resolve().parents[1].parent / "third_party" / "LightningDiT"

_LIGHTNINGDIT_PACKAGES = ("models", "tokenizer", "transport")
_LIGHTNINGDIT_FILES = ("models/lightningdit.py", "tokenizer/autoencoder.py", "transport/__init__.py")
_TOKENIZER_PREFIXES = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
_active_root: Path | None = None     # root of the last successful add_lightningdit_to_path()
_compile_choice: bool | None = None  # decision of the last configure_torch_compile()


def add_lightningdit_to_path(root: str | os.PathLike | None = None) -> Path:
    """Put a LightningDiT checkout at sys.path[0] and return its resolved root.

    root=None reuses the root of the last successful call; on first use it is
    $RWVAE_LIGHTNINGDIT_ROOT if set, else DEFAULT_LIGHTNINGDIT_ROOT. LightningDiT imports its
    own code by generic top-level names (models, tokenizer, transport, datasets, tools), which
    therefore shadow same-named modules for the rest of the process; this module imports only
    tokenizer.autoencoder, models.lightningdit and transport from it.

    Raises FileNotFoundError for an incomplete checkout, and ImportError if a module named
    models, tokenizer or transport is already imported from another location (the LightningDiT
    imports would silently bind to that code).
    """
    global _active_root
    if root is None:
        root = _active_root or os.environ.get("RWVAE_LIGHTNINGDIT_ROOT") or DEFAULT_LIGHTNINGDIT_ROOT
    root = Path(root).resolve()
    missing = [f for f in _LIGHTNINGDIT_FILES if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"no complete LightningDiT checkout at {str(root)!r} (missing {missing}); clone it to "
            f"{str(DEFAULT_LIGHTNINGDIT_ROOT)!r} or set RWVAE_LIGHTNINGDIT_ROOT")
    for name in _LIGHTNINGDIT_PACKAGES:
        if name not in sys.modules:
            continue
        found = getattr(sys.modules[name], "__file__", None)
        if found is None or Path(found).resolve() != (root / name / "__init__.py").resolve():
            raise ImportError(
                f"a module named {name!r} is already imported from {found!r}, not from the LightningDiT "
                f"checkout {str(root)!r}; LightningDiT's generic package names would bind to the wrong "
                "code. Use a fresh process, or import LightningDiT only through rwvae.integration.")
    entry = str(root)
    if sys.path[:1] != [entry]:
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    _active_root = root
    return root


def configure_torch_compile(enable: bool | None = None) -> bool:
    """Decide whether LightningDiT's @torch.compile decorators compile; returns the decision.

    enable=None decides automatically: compile only on Linux with CUDA and an installed triton
    (Windows has no triton, so a compiled LightningDiT would fail at its first forward). The
    decision is applied process-wide via torch._dynamo.config.disable, and is frozen once
    models.lightningdit is imported so one process never mixes compiled and eager LightningDiT:
    asking for the other setting afterwards raises ValueError. build_lightningdit() calls this
    with enable=None before that import unless the caller already chose.
    """
    global _compile_choice
    if enable is not None and not isinstance(enable, bool):
        raise ValueError(f"enable must be True, False or None (auto), got {enable!r}")
    import torch._dynamo

    if enable is None:
        enable = (sys.platform.startswith("linux") and torch.cuda.is_available()
                  and importlib.util.find_spec("triton") is not None)
    if "models.lightningdit" in sys.modules:
        imported_with = not torch._dynamo.config.disable
        if enable != imported_with:
            raise ValueError(
                f"models.lightningdit is already imported with torch.compile "
                f"{'enabled' if imported_with else 'disabled'}; refusing to switch to enable={enable}. "
                "Call configure_torch_compile() before the first LightningDiT import.")
    else:
        torch._dynamo.config.disable = not enable
    _compile_choice = enable
    logger.info("LightningDiT torch.compile %s", "enabled" if enable else "disabled")
    return enable


def _import_lightningdit(name: str):
    """Import tokenizer.autoencoder, models.lightningdit or transport from the LightningDiT checkout."""
    add_lightningdit_to_path()
    if name == "models.lightningdit" and _compile_choice is None and name not in sys.modules:
        configure_torch_compile()
    return importlib.import_module(name)


# --------------------------------------------------------------------------- tokenizer
def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tokenizer_checkpoint(ckpt_path) -> tuple[dict, dict]:
    """(autoencoder state dict, load info); see load_tokenizer_state_dict."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer checkpoint not found: {str(path)!r}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        weights_only = True
    except pickle.UnpicklingError as err:
        detail = [line.strip() for line in str(err).splitlines() if "Unsupported global" in line]
        logger.warning(
            "torch.load(weights_only=True) rejected %s (%s); retrying with weights_only=False, which "
            "unpickles arbitrary objects and is acceptable ONLY for a trusted checkpoint.",
            path, detail[0] if detail else type(err).__name__)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        weights_only = False
    if isinstance(obj, Mapping) and "state_dict" in obj:
        state, fmt = obj["state_dict"], "lightning"  # PyTorch-Lightning checkpoint
    else:
        state, fmt = obj, "state_dict"  # plain state dict (e.g. the released LDM f16d32 file)
    if not (isinstance(state, Mapping) and state and all(isinstance(v, torch.Tensor) for v in state.values())):
        keys = f" with keys {list(obj)[:5]}" if isinstance(obj, Mapping) else ""
        raise ValueError(f"{str(path)!r} is neither a state dict nor a Lightning checkpoint with a "
                         f"'state_dict' (got {type(obj).__name__}{keys})")
    kept = {k: v for k, v in state.items() if k.startswith(_TOKENIZER_PREFIXES)}
    dropped = Counter(k.split(".", 1)[0] for k in state if not k.startswith(_TOKENIZER_PREFIXES))
    info = {"checkpoint_format": fmt, "weights_only_load": weights_only,
            "num_tensors": len(kept), "dropped_keys": dict(sorted(dropped.items()))}
    return kept, info


def load_tokenizer_state_dict(ckpt_path) -> dict:
    """Autoencoder weights of a tokenizer checkpoint, on CPU.

    Accepts a PyTorch-Lightning checkpoint ({'state_dict': ...}) or a plain state dict (the
    released ldm-imagenet256-f16d32-50ep.ckpt is the latter) and keeps only the encoder.,
    decoder., quant_conv. and post_quant_conv. tensors, dropping e.g. the training loss
    (LPIPS + discriminator) under loss.*. Uses torch.load(weights_only=True) and falls back to
    weights_only=False with a logged warning: acceptable only for trusted checkpoints.
    """
    return _read_tokenizer_checkpoint(ckpt_path)[0]


def _check_images(images: torch.Tensor, multiple: int = 1) -> None:
    """A non-empty float batch [B, 3, H, W], H and W divisible by `multiple`, values in [-1, 1]
    (ToTensor + Normalize(0.5, 0.5); docs/ARCHITECTURE_V2.md §1)."""
    if not (isinstance(images, torch.Tensor) and images.dim() == 4 and images.shape[0] > 0
            and images.shape[1] == 3 and images.is_floating_point()):
        raise ValueError(f"expected float images [B, 3, H, W], got {type(images).__name__} "
                         f"{tuple(getattr(images, 'shape', ()))} {getattr(images, 'dtype', '')}")
    H, W = images.shape[-2:]
    if H % multiple or W % multiple:
        raise ValueError(f"image size {H}x{W} is not a multiple of the tokenizer downsampling factor {multiple}")
    lo, hi = (float(v) for v in torch.aminmax(images))
    if not (lo >= -1.0 and hi <= 1.0):  # also rejects NaN
        raise ValueError(f"images must lie in [-1, 1]; got values in [{lo:.4g}, {hi:.4g}]")


class FrozenTokenizer(nn.Module):
    """Frozen LightningDiT AutoencoderKL (the LDM / VA-VAE f16d32 architecture) on [-1, 1] images.

    Build it with from_checkpoint(): the autoencoder is constructed with ckpt_path=None and
    loaded strictly here, because upstream init_from_ckpt keeps every non-'loss' key (its
    foundation_model filter is a no-op) and loads with strict=False, so a mismatched file would
    load silently. The latent of every v2 arm is the posterior mean (encode_mean; plan §1.2),
    never a posterior sample. post_quant_conv and decoder are exposed for rwvae.decoder_adapter.
    """

    def __init__(self, autoencoder: nn.Module, meta: dict | None = None):
        super().__init__()
        self.autoencoder = autoencoder.requires_grad_(False)
        self.embed_dim = int(autoencoder.embed_dim)
        self.downsample = 2 ** (autoencoder.encoder.num_resolutions - 1)
        self.meta = dict(meta or {})
        self.eval()

    @classmethod
    def from_checkpoint(cls, ckpt_path, *, embed_dim: int = 32, ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
                        device: str | torch.device = "cpu") -> "FrozenTokenizer":
        """AutoencoderKL(embed_dim, ch_mult, use_variational=True, model_type='vavae') loaded strictly
        from ckpt_path: missing, unexpected or shape-mismatched tensors raise ValueError. meta records
        the architecture, the checkpoint path and sha256, and how the file was read."""
        ch_mult = tuple(int(c) for c in ch_mult)
        if int(embed_dim) != embed_dim or embed_dim < 1 or not ch_mult or min(ch_mult) < 1:
            raise ValueError(f"need embed_dim >= 1 and a non-empty ch_mult of positive ints, "
                             f"got embed_dim={embed_dim!r}, ch_mult={ch_mult!r}")
        module = _import_lightningdit("tokenizer.autoencoder")
        state, info = _read_tokenizer_checkpoint(ckpt_path)
        model = module.AutoencoderKL(embed_dim=int(embed_dim), ch_mult=ch_mult, use_variational=True,
                                     ckpt_path=None, model_type="vavae")
        own = model.state_dict()
        missing = sorted(own.keys() - state.keys())
        unexpected = sorted(state.keys() - own.keys())
        mismatched = sorted(k for k in own.keys() & state.keys() if own[k].shape != state[k].shape)
        if missing or unexpected or mismatched:
            shapes = [(k, tuple(state[k].shape), tuple(own[k].shape)) for k in mismatched[:3]]
            raise ValueError(
                f"{str(ckpt_path)!r} does not match AutoencoderKL(embed_dim={embed_dim}, ch_mult={ch_mult}): "
                f"{len(missing)} missing {missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}, "
                f"{len(mismatched)} shape-mismatched (key, checkpoint, model) {shapes}")
        model.load_state_dict(state, strict=True)
        path = Path(ckpt_path).resolve()
        meta = {
            "tokenizer": "LightningDiT tokenizer.autoencoder.AutoencoderKL",
            "lightningdit_root": str(Path(module.__file__).resolve().parents[1]),
            "embed_dim": int(embed_dim), "ch_mult": list(ch_mult), "use_variational": True,
            "model_type": "vavae", "downsample": 2 ** (len(ch_mult) - 1),
            "checkpoint_path": str(path), "checkpoint_sha256": _sha256_file(path), **info,
        }
        return cls(model, meta).to(device)

    @torch.no_grad()
    def encode_moments(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (mean, logvar) of images [B, 3, H, W] in [-1, 1], each [B, embed_dim, H/f, W/f].
        logvar is clamped to [-30, 20] by LightningDiT's DiagonalGaussianDistribution."""
        _check_images(images, self.downsample)
        posterior = self.autoencoder.encode(images)
        return posterior.mean, posterior.logvar

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        """Posterior mean mu(x) [B, embed_dim, H/f, W/f] (never posterior.sample())."""
        return self.encode_moments(images)[0]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Images [B, 3, f*h, f*w] from latents [B, embed_dim, h, w]; nominally [-1, 1], not clamped."""
        if z.dim() != 4 or z.shape[1] != self.embed_dim:
            raise ValueError(f"expected latents [B, {self.embed_dim}, h, w], got {tuple(z.shape)}")
        return self.autoencoder.decode(z)

    @property
    def post_quant_conv(self) -> nn.Conv2d:
        """First decoder layer (1x1 conv, embed_dim -> embed_dim) into which P6 folds the LMMSE map."""
        return self.autoencoder.post_quant_conv

    @property
    def decoder(self) -> nn.Module:
        return self.autoencoder.decoder


# --------------------------------------------------------------------------- DINOv2
class DinoPatchFeatures(nn.Module):
    """Frozen timm DINOv2 patch tokens on the latent grid: images [B, 3, S, S] in [-1, 1] -> [B, D, g, g].

    Preprocessing (recorded in meta; plan §1.2, §3.2): x -> (x + 1) / 2 in [0, 1] -> bilinear resize
    (antialias=True, align_corners=False) to input_size -> (x - mean) / std with the model's own
    pretrained_cfg (ImageNet) statistics -> forward_features (tokens after the final norm) -> drop
    the num_prefix_tokens class/register tokens -> row-major (h, w) reshape, g = input_size // patch.
    The defaults give g = 224 // 14 = 16: each DINO patch covers the same normalised image cell
    (1/16 x 1/16) as one f16 latent token of the 256 px input, so z and y pair token by token in
    normalised coordinates. Unlike VA-VAE's auxiliary DINOv2 (vavae/ldm/models/foundation_models.py
    feeds it the [-1, 1] input without ImageNet normalisation or antialiasing), the inputs here are
    normalised as DINOv2 was trained. The output dtype follows the model / autocast.
    """

    def __init__(self, model_name: str = "vit_large_patch14_dinov2.lvd142m", *, input_size: int = 224,
                 device: str | torch.device = "cpu", checkpoint_path: str | os.PathLike | None = None):
        super().__init__()
        import timm

        kwargs = {"pretrained": True, "dynamic_img_size": True}
        if checkpoint_path is not None:
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(f"DINO checkpoint not found: {str(checkpoint_path)!r}")
            kwargs["pretrained_cfg_overlay"] = {"file": str(checkpoint_path)}
        model = timm.create_model(model_name, **kwargs)
        patch_h, patch_w = model.patch_embed.patch_size
        if patch_h != patch_w or int(input_size) != input_size or input_size % patch_h:
            raise ValueError(f"input_size={input_size!r} must be a multiple of the square patch size "
                             f"{(patch_h, patch_w)} of {model_name}")
        self.model = model.requires_grad_(False)
        self.input_size = int(input_size)
        self.grid = self.input_size // patch_h
        self.num_prefix_tokens = int(model.num_prefix_tokens)
        mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.meta = {
            "model_name": model_name,
            "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path).resolve()),
            "input_size": self.input_size, "patch_size": int(patch_h), "grid": self.grid,
            "num_prefix_tokens": self.num_prefix_tokens, "embed_dim": int(model.num_features),
            "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "preprocess": "[-1,1] -> [0,1] -> bilinear(antialias=True, align_corners=False) -> (x-mean)/std",
            "tokens": "forward_features (after final norm), prefix tokens dropped, row-major (h, w)",
        }
        self.eval().to(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _check_images(images)
        if images.shape[-1] != images.shape[-2]:
            raise ValueError(f"expected square images (one DINO grid per latent grid), got {tuple(images.shape[-2:])}")
        x = F.interpolate((images + 1) / 2, size=(self.input_size, self.input_size), mode="bilinear",
                          align_corners=False, antialias=True)
        tokens = self.model.forward_features((x - self.mean) / self.std)
        B, N, D = tokens.shape
        g, p = self.grid, self.num_prefix_tokens
        if N != p + g * g:
            raise ValueError(f"{self.meta['model_name']} returned {N} tokens for a {self.input_size} px input; "
                             f"expected {p} prefix + {g}x{g} patch tokens")
        return tokens[:, p:].reshape(B, g, g, D).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- LightningDiT
_CUSTOM_ARCH_KEYS = ("depth", "hidden_size", "num_heads", "patch_size")
_TRANSPORT_KEYS = ("path_type", "prediction", "loss_weight", "train_eps", "sample_eps")  # required, as train.py
_TRANSPORT_OPTIONAL = ("use_cosine_loss", "use_lognorm")  # default False, as train.py
_TRANSPORT_CHOICES = {"path_type": ("Linear", "GVP", "VP"), "prediction": ("velocity", "noise", "score"),
                      "loss_weight": (None, "velocity", "likelihood")}
_SAMPLE_ODE_KEYS = ("mode", "sampling_method", "num_sampling_steps", "atol", "rtol", "reverse")


def build_lightningdit(model_type: str = "LightningDiT-B/1", *, input_size: int = 16, in_channels: int = 32,
                       num_classes: int = 1000, use_qknorm: bool = False, use_swiglu: bool = True,
                       use_rope: bool = True, use_rmsnorm: bool = True, wo_shift: bool = False,
                       class_dropout_prob: float = 0.1, use_checkpoint: bool = False,
                       custom_arch: Mapping | None = None) -> nn.Module:
    """A freshly initialised LightningDiT on CPU, constructed as LightningDiT's train.py does
    (learn_sigma=False).

    model_type picks an entry of models.lightningdit.LightningDiT_models; custom_arch =
    {depth, hidden_size, num_heads, patch_size} instead builds LightningDiT directly (small test
    models; model_type is then unused). The torch.compile decision (configure_torch_compile) is
    made before the first import. In train mode the CFG label dropout (class_dropout_prob) draws
    from the global torch RNG.
    """
    lightningdit = _import_lightningdit("models.lightningdit")
    kwargs = dict(input_size=input_size, in_channels=in_channels, num_classes=num_classes,
                  use_qknorm=use_qknorm, use_swiglu=use_swiglu, use_rope=use_rope, use_rmsnorm=use_rmsnorm,
                  wo_shift=wo_shift, class_dropout_prob=class_dropout_prob, use_checkpoint=use_checkpoint)
    if custom_arch is None:
        if model_type not in lightningdit.LightningDiT_models:
            raise ValueError(f"unknown model_type {model_type!r}; choose from {sorted(lightningdit.LightningDiT_models)}")
        return lightningdit.LightningDiT_models[model_type](**kwargs)
    arch = dict(custom_arch)
    if set(arch) != set(_CUSTOM_ARCH_KEYS) or any(int(v) != v or v < 1 for v in arch.values()):
        raise ValueError(f"custom_arch needs exactly the positive integers {_CUSTOM_ARCH_KEYS}, got {arch!r}")
    if arch["hidden_size"] % arch["num_heads"] or input_size % arch["patch_size"]:
        raise ValueError(f"custom_arch {arch!r}: hidden_size must be divisible by num_heads and "
                         f"input_size={input_size} by patch_size")
    return lightningdit.LightningDiT(**arch, **kwargs)


def build_transport(tcfg: Mapping):
    """LightningDiT Transport from a transport config block, via create_transport as train.py calls it.

    Required keys: path_type, prediction, loss_weight, train_eps, sample_eps (values may be null);
    optional: use_cosine_loss, use_lognorm (default False). Unknown keys, and values that
    create_transport would silently map to a default (e.g. prediction="eps" -> velocity), raise
    ValueError. Transport.training_losses draws t and the noise from the global torch RNGs and,
    with use_lognorm, t from the global NumPy RNG (scipy.stats.norm.rvs): seed both to reproduce.
    """
    cfg = dict(tcfg)
    missing = [k for k in _TRANSPORT_KEYS if k not in cfg]
    unknown = sorted(set(cfg) - set(_TRANSPORT_KEYS) - set(_TRANSPORT_OPTIONAL))
    if missing or unknown:
        raise ValueError(f"transport config: missing keys {missing}, unknown keys {unknown}")
    for key, choices in _TRANSPORT_CHOICES.items():
        if cfg[key] not in choices:
            raise ValueError(f"transport.{key}={cfg[key]!r} is not one of {choices}")
    transport = _import_lightningdit("transport")
    return transport.create_transport(
        cfg["path_type"], cfg["prediction"], cfg["loss_weight"], cfg["train_eps"], cfg["sample_eps"],
        use_cosine_loss=cfg.get("use_cosine_loss", False), use_lognorm=cfg.get("use_lognorm", False))


def build_ode_sample_fn(transport, scfg: Mapping) -> Callable:
    """sample_fn(x0, model_fn, **model_kwargs) from Sampler(transport).sample_ode, as inference.py builds it.

    Reads mode (must be "ODE"), sampling_method, num_sampling_steps, atol, rtol, reverse and
    timestep_shift (default 0, as inference.py); the sample block's other keys (batch size, seeds,
    sample counts, ...) belong to the caller. The time grid has num_sampling_steps points over the
    transport's interval (t = 0 noise -> 1 data for Linear/velocity), warped by a shift s > 0 as
    t -> s t / (1 + (s - 1) t); a fixed-grid solver such as euler thus makes num_sampling_steps - 1
    model calls. sample_fn returns the trajectory [num_sampling_steps, *x0.shape] (index 0 is x0);
    the sample is its last entry.
    """
    missing = [k for k in _SAMPLE_ODE_KEYS if k not in scfg]
    if missing:
        raise ValueError(f"sample config: missing keys {missing}")
    if scfg["mode"] != "ODE":
        raise ValueError(f"sample.mode={scfg['mode']!r}: only 'ODE' is supported (as in LightningDiT inference.py)")
    steps = scfg["num_sampling_steps"]
    if int(steps) != steps or steps < 2:
        raise ValueError(f"num_sampling_steps must be an integer >= 2 (time-grid points), got {steps!r}")
    sampler = _import_lightningdit("transport").Sampler(transport)
    return sampler.sample_ode(sampling_method=scfg["sampling_method"], num_steps=int(steps), atol=scfg["atol"],
                              rtol=scfg["rtol"], reverse=scfg["reverse"],
                              timestep_shift=scfg.get("timestep_shift", 0))


# --------------------------------------------------------------------------- metric networks
def load_inception(device) -> nn.Module:
    """pytorch_fid InceptionV3 pool3 (2048-d) extractor, eval mode, frozen.

    Input: float images [B, 3, H, W] in [0, 1] (resized to 299 bilinearly and mapped to [-1, 1]
    inside); output: a list holding one [B, 2048, 1, 1] tensor. The FID weights
    (pt_inception-2015-12-05) are downloaded into $TORCH_HOME/hub/checkpoints on first use.
    """
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
    return model.eval().requires_grad_(False).to(device)


def load_lpips(device) -> nn.Module:
    """LPIPS(net='vgg'), eval mode, frozen: lpips_fn(x, y) on [-1, 1] images -> [B, 1, 1, 1].
    The VGG16 backbone is downloaded into $TORCH_HOME/hub/checkpoints on first use."""
    import lpips

    return lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(device)
