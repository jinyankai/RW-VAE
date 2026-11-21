"""RW-VAE v2: decoder side of the channel -- analytic LMMSE prelayer and decoder fine-tuning.

Plan v2.1 Sec. 2.2 P6, Sec. 3.2 (`decode_with_adapter`) and milestone M4.

The tokenizer decodes a latent z as decoder(post_quant_conv(z)), where post_quant_conv is
a 1x1 conv (W0, b0). After the channel z' = m + A (mu - m) + B eps (rwvae.channel), the
linear MMSE estimate of mu is (P6: Cov(mu, z') Cov(z')^{-1} = Sigma A^T Sigma^{-1} = A)

    z_hat = m + A (z' - m) = A z' + (I - A) m,

which equals E[mu | z'] only when (mu, z') is jointly Gaussian. This affine prelayer is
absorbed exactly into the 1x1 conv, W0' = W0 A and b0' = b0 + W0 (I - A) m, so the
analytic adapter adds no layer and no inference cost.

`AdaptedDecoder` holds *copies* of post_quant_conv and decoder (the tokenizer itself is
never modified) in one of three modes:
  identity   original weights (B0 reference);
  lmmse      P6 prelayer fused into post_quant_conv (analytic, no training);
  finetuned  weights changed by `finetune_decoder` or read by `load_weights`;
             `base_mode` records the analytic mode the fine-tuning started from.

`finetune_decoder` is the one fine-tuning recipe shared by all arms: L1 + LPIPS (the
reconstruction term of the LDM autoencoder loss, without discriminator) and Adam(W)
with betas (0.5, 0.9). Each sample gets fresh channel noise seeded by
noise_seed(base, step, index). The encoder is never involved (mu comes from the cache),
and there is no GAN loss (candidate pool, plan Sec. 9).

z' lives in the raw tokenizer latent space (the channel output), not in the
DiT-normalized space. Outputs are images in [-1, 1] and are not clamped; clamping happens
when evaluation quantizes to uint8.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .channel import RWChannel, noise_seed, sample_noise

BASE_MODES = ("identity", "lmmse")   # analytic starting points; "finetuned" is reached by training/loading
MIXED_PRECISION = ("bf16", "no")


def _f64(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} has non-finite entries")
    return a


def lmmse_affine(A, m) -> tuple[np.ndarray, np.ndarray]:
    """P6 LMMSE prelayer as an affine map: z_hat = m + A (z' - m) = W_pre z' + b_pre.

    Returns float64 (W_pre, b_pre) = (A, (I - A) m). Pass the float64 `Allocation.A` and
    `Allocation.m`, not the float32 buffers of an `RWChannel`.
    """
    A, m = _f64(A, "A"), _f64(m, "m")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or m.shape != (A.shape[0],):
        raise ValueError(f"expected A [C, C] and m [C], got shapes {A.shape} and {m.shape}")
    return A.copy(), (np.eye(len(m)) - A) @ m


def _check_conv1x1(conv) -> None:
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f"expected an nn.Conv2d, got {type(conv).__name__}")
    geom = {"kernel_size": conv.kernel_size, "stride": conv.stride, "padding": conv.padding,
            "dilation": conv.dilation, "groups": conv.groups}
    plain = {"kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1}
    if geom != plain:
        raise ValueError(f"expected a plain 1x1 conv {plain}, got {geom}")


def fuse_affine_into_conv1x1(conv: nn.Conv2d, W_pre, b_pre) -> nn.Conv2d:
    """New 1x1 conv equal to `conv` applied after the per-pixel affine map z -> W_pre z + b_pre.

    Weight W0 W_pre and bias b0 + W0 b_pre (b0 = 0 if `conv` has no bias; the result always
    has a bias), computed in float64 and cast to conv's dtype/device (P6 fusion). W_pre is
    [in_channels, k], so the new conv has k input channels. `conv` is not modified; the new
    conv copies its requires_grad and training flags.
    """
    _check_conv1x1(conv)
    W_pre, b_pre = _f64(W_pre, "W_pre"), _f64(b_pre, "b_pre")
    c_in = conv.in_channels
    if W_pre.ndim != 2 or W_pre.shape[0] != c_in or b_pre.shape != (c_in,):
        raise ValueError(f"expected W_pre [{c_in}, k] and b_pre [{c_in}] for a conv with {c_in} input "
                         f"channels, got shapes {W_pre.shape} and {b_pre.shape}")
    W0 = conv.weight.detach().to("cpu", torch.float64).numpy()[:, :, 0, 0]
    b0 = np.zeros(conv.out_channels) if conv.bias is None else conv.bias.detach().to("cpu", torch.float64).numpy()
    # skip_init: no random initialization (and no global-RNG draw) for weights overwritten below
    fused = nn.utils.skip_init(nn.Conv2d, W_pre.shape[1], conv.out_channels, 1,
                               device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(torch.from_numpy(W0 @ W_pre)[:, :, None, None])
        fused.bias.copy_(torch.from_numpy(b0 + W0 @ b_pre))
    fused.requires_grad_(conv.weight.requires_grad)
    return fused.train(conv.training)


class AdaptedDecoder(nn.Module):
    """decode_with_adapter of plan Sec. 3.2: z' -> decoder(post_quant_conv(z')) on module copies.

    mode "identity" keeps the original weights; "lmmse" (needs the channel's A and m) fuses
    the P6 prelayer z_hat = m + A (z' - m) into the copied post_quant_conv. Mode "finetuned"
    cannot be requested here: `finetune_decoder` and `load_weights` set it.
    """

    def __init__(self, post_quant_conv: nn.Conv2d, decoder: nn.Module, *, mode: str = "identity", A=None, m=None):
        super().__init__()
        if mode not in BASE_MODES:
            raise ValueError(f"mode must be one of {BASE_MODES} (fine-tuned weights come from "
                             f"finetune_decoder or load_weights), got {mode!r}")
        _check_conv1x1(post_quant_conv)
        if mode == "lmmse":
            if A is None or m is None:
                raise ValueError("mode 'lmmse' needs the channel matrix A and mean m")
            self.post_quant_conv = fuse_affine_into_conv1x1(post_quant_conv, *lmmse_affine(A, m))
        else:
            if A is not None or m is not None:
                raise ValueError("A and m are only used in mode 'lmmse'")
            self.post_quant_conv = copy.deepcopy(post_quant_conv)
        self.decoder = copy.deepcopy(decoder)
        self.train(decoder.training)  # start in the copied decoder's train/eval state
        self.mode = mode
        self.base_mode = mode

    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Images [B, 3, H', W'] in [-1, 1] (not clamped) from channel outputs z' [B, C, H, W]."""
        c = self.post_quant_conv.in_channels
        if z_prime.dim() != 4 or z_prime.shape[1] != c:
            raise ValueError(f"expected z' of shape [B, {c}, H, W], got {tuple(z_prime.shape)}")
        return self.decoder(self.post_quant_conv(z_prime))

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the copied post_quant_conv and decoder (everything this module holds)."""
        yield from self.post_quant_conv.parameters()
        yield from self.decoder.parameters()

    def save(self, path: str | Path, meta: dict) -> None:
        """Write fine-tuned weights (post_quant_conv + decoder) to a safetensors file.

        File metadata: {"meta": json.dumps(meta), "base_mode": base_mode}. Only mode
        "finetuned" is saved; analytic adapters are rebuilt from (A, m) instead.
        """
        if self.mode != "finetuned":
            raise ValueError(f"only fine-tuned weights are saved, but mode is {self.mode!r}; "
                             "rebuild analytic adapters from the allocation instead")
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, str(path), metadata={"meta": json.dumps(meta, sort_keys=True), "base_mode": self.base_mode})

    def load_weights(self, path: str | Path) -> dict:
        """Strictly load weights written by `save`; sets mode "finetuned" and base_mode from
        the file, and returns the saved meta."""
        with safe_open(str(path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            state = {k: f.get_tensor(k) for k in f.keys()}
        if "meta" not in md or md.get("base_mode") not in BASE_MODES:
            raise ValueError(f"{path} is not an AdaptedDecoder weights file (metadata keys {sorted(md)})")
        self.load_state_dict(state, strict=True)
        self.mode, self.base_mode = "finetuned", md["base_mode"]
        return json.loads(md["meta"])

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, base_mode={self.base_mode!r}"


def decode_with_adapter(adapter: AdaptedDecoder, z_prime: torch.Tensor) -> torch.Tensor:
    """Plan Sec. 3.2 entry point: images [B, 3, H', W'] in [-1, 1] from channel outputs z'."""
    return adapter(z_prime)


def _cycle(loader: Iterable) -> Iterator:
    """Endless batches: re-iterate `loader` when exhausted, calling dataset.set_epoch(epoch)
    (if the dataset has it) before every pass, including the first (epoch 0)."""
    set_epoch = getattr(getattr(loader, "dataset", None), "set_epoch", None)
    epoch = 0
    while True:
        if set_epoch is not None:
            set_epoch(epoch)
        n = 0
        for batch in loader:
            n += 1
            yield batch
        if n == 0:
            raise ValueError("loader yielded no batches")
        epoch += 1


def finetune_decoder(
    adapter: AdaptedDecoder,
    loader: Iterable,
    channel: RWChannel,
    *,
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int,
    lr: float,
    l1_weight: float = 1.0,
    lpips_weight: float = 1.0,
    betas: tuple[float, float] = (0.5, 0.9),
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    mixed_precision: str = "bf16",
    device: torch.device | str,
    noise_base_seed: int,
    log_fn: Callable[[int, dict], None] | None = None,
    log_every: int = 50,
) -> dict:
    """Fine-tune the adapter's post_quant_conv + decoder on channel outputs (plan M4).

    `loader` yields (images [B,3,H',W'] in [-1,1], mu [B,C,H,W], label, index), where index
    is the dataset row of each sample; it is re-iterated when exhausted (see `_cycle`), so its
    order must be seeded by the caller. For optimizer step t (0-based) sample i sees

        z' = channel(mu, eps),   eps = sample_noise(seed=noise_seed(noise_base_seed, t, index_i)),
        loss = l1_weight * mean|x_hat - x| + lpips_weight * mean(lpips_fn(x_hat, x)),  x_hat = adapter(z').

    With both weights at 1 this is the LDM autoencoder reconstruction term without
    discriminator. Optimizer: AdamW(lr, betas, weight_decay), i.e. Adam when weight_decay=0.
    bf16 autocast is used only on CUDA; CPU runs in the parameters' dtype. Optional grad-norm
    clipping. log_fn(step, {"loss", "l1", "lpips"}) gets means over the steps since the
    previous log, every `log_every` steps and at the last step.

    Side effects: the adapter is moved to `device`, its parameters get requires_grad=True,
    its mode becomes "finetuned" after the first update, and its train/eval flag is restored
    at the end. Returns {"steps", "final_loss", "final_l1", "final_lpips"} (means over the
    last log window) and "mixed_precision", the precision that actually ran ("bf16" or "no").
    """
    if int(steps) != steps or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if int(log_every) != log_every or log_every < 1:
        raise ValueError(f"log_every must be a positive integer, got {log_every!r}")
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError(f"lr must be positive and finite, got {lr!r}")
    if not (math.isfinite(l1_weight) and math.isfinite(lpips_weight) and l1_weight >= 0 and lpips_weight >= 0) \
            or l1_weight + lpips_weight == 0:
        raise ValueError(f"loss weights must be finite, >= 0 and not both 0, got "
                         f"l1_weight={l1_weight!r}, lpips_weight={lpips_weight!r}")
    if grad_clip is not None and not (math.isfinite(grad_clip) and grad_clip > 0):
        raise ValueError(f"grad_clip must be None or positive and finite, got {grad_clip!r}")
    if mixed_precision not in MIXED_PRECISION:
        raise ValueError(f"mixed_precision must be one of {MIXED_PRECISION}, got {mixed_precision!r}")
    steps, log_every, device = int(steps), int(log_every), torch.device(device)
    use_bf16 = mixed_precision == "bf16" and device.type == "cuda"

    adapter.to(device)
    params = list(adapter.trainable_parameters())
    for p in params:
        p.requires_grad_(True)
    dtype = params[0].dtype
    opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    was_training = adapter.training
    adapter.train()
    batches = _cycle(loader)
    window: list[torch.Tensor] = []
    last: dict[str, float] = {}
    try:
        for step in range(steps):
            images, mu, _, index = next(batches)
            images, mu = images.to(device, dtype), mu.to(device, dtype)
            seeds = [noise_seed(noise_base_seed, step, i) for i in torch.as_tensor(index).tolist()]
            with torch.no_grad():  # the channel is fixed; the encoder is never involved
                z_prime = channel(mu, noise=sample_noise(mu.shape[1:], seeds, device=device))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                x_hat = adapter(z_prime).to(dtype)
                if x_hat.shape != images.shape:
                    raise ValueError(f"decoded shape {tuple(x_hat.shape)} != image shape {tuple(images.shape)}")
                lp = lpips_fn(x_hat, images).to(dtype).mean()
            l1 = (x_hat - images).abs().mean()
            loss = l1_weight * l1 + lpips_weight * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            adapter.mode = "finetuned"
            window.append(torch.stack([loss, l1, lp]).detach())
            done = step + 1
            if done % log_every == 0 or done == steps:
                last = dict(zip(("loss", "l1", "lpips"), torch.stack(window).mean(0).tolist()))
                window.clear()
                if not all(math.isfinite(v) for v in last.values()):
                    raise FloatingPointError(f"non-finite training loss at step {done}: {last}")
                if log_fn is not None:
                    log_fn(done, last)
    finally:
        batches.close()
        adapter.train(was_training)
    return {"steps": steps, "final_loss": last["loss"], "final_l1": last["l1"], "final_lpips": last["lpips"],
            "mixed_precision": "bf16" if use_bf16 else "no"}
