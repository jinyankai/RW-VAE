"""RW-VAE v2: variance-preserving Gaussian test channel on frozen tokenizer latents.

Plan v2.1, Sec. 2.2 P4/P6. Every token mu in R^C (channel dim = 1 of a [B, C, H, W]
latent) passes independently through

    z' = m + A (mu - m) + B eps,        eps ~ N(0, I_C),

with A = T^{-1} diag(sqrt(keep)) T and B = T^{-1} diag(sqrt((1 - keep) s)) produced by
`rwvae.allocation`. For any source with E mu = m and Cov(mu) = Sigma this gives
Cov(z') = A Sigma A^T + B B^T = Sigma per token, and E_x KL(q(z'|x) || N(m, Sigma)) = R
(the nominal rate). Cross-token covariances are NOT preserved: for tokens p != q,
Cov(z'_p, z'_q) = A Cov(mu_p, mu_q) A^T because eps is white across tokens. Measure the
resulting spatial-spectrum change with `rwvae.evaluation.dct2_power`.

The linear MMSE estimate of mu from z' is m + A (z' - m) (P6: Cov(mu, z') Cov(z')^{-1}
= Sigma A^T Sigma^{-1} = A for this channel); `RWChannel.lmmse` implements it.

Reproducible noise: `noise_seed(base, epoch, index)` mixes the three integers with
splitmix64, and `sample_noise` draws each sample's eps from its own CPU generator, so a
sample's noise does not depend on batch composition, DataLoader worker count or device.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

_MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """One splitmix64 step (Steele et al. 2014): a fast, well-mixed 64-bit hash."""
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def noise_seed(base_seed: int, epoch: int, index: int) -> int:
    """Seed for sample `index` in `epoch`, in [0, 2^63). Platform independent."""
    for name, v in (("base_seed", base_seed), ("epoch", epoch), ("index", index)):
        if int(v) != v or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
    h = splitmix64(int(base_seed))
    h = splitmix64(h ^ int(epoch))
    h = splitmix64(h ^ int(index))
    return h >> 1


def sample_noise(
    shape_chw: Sequence[int],
    seeds: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard-normal noise [len(seeds), *shape_chw]; row i depends only on seeds[i]."""
    shape = tuple(int(s) for s in shape_chw)
    out = torch.empty((len(seeds),) + shape, dtype=dtype)
    gen = torch.Generator(device="cpu")
    for i, s in enumerate(seeds):
        gen.manual_seed(int(s))
        out[i] = torch.randn(shape, generator=gen, dtype=dtype)
    return out if device is None else out.to(device)


def _as_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype, device="cpu").clone()
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype).clone()


class RWChannel(nn.Module):
    """Per-token affine Gaussian channel z' = m + A (mu - m) + B eps.

    Buffers: m [C], A [C, C], B [C, C], rate (nominal nats/token; inf for identity).
    Inputs are [batch, C, *spatial]; half-precision inputs are computed in float32.
    Build the buffers in float64 (`dtype=torch.float64`) when exactness matters.
    """

    def __init__(self, m, A, B, rate: float = float("inf"), *, dtype: torch.dtype = torch.float32, meta: dict | None = None):
        super().__init__()
        m_t, A_t, B_t = _as_tensor(m, dtype), _as_tensor(A, dtype), _as_tensor(B, dtype)
        C = m_t.numel()
        if m_t.shape != (C,) or A_t.shape != (C, C) or B_t.shape != (C, C):
            raise ValueError(f"expected m [C], A [C,C], B [C,C]; got {tuple(m_t.shape)}, {tuple(A_t.shape)}, {tuple(B_t.shape)}")
        for name, t in (("m", m_t), ("A", A_t), ("B", B_t)):
            if not torch.isfinite(t).all():
                raise ValueError(f"channel parameter {name} has non-finite entries")
        self.register_buffer("m", m_t)
        self.register_buffer("A", A_t)
        self.register_buffer("B", B_t)
        self.register_buffer("rate", torch.tensor(float(rate), dtype=torch.float64))
        self.meta = dict(meta or {})

    @classmethod
    def from_allocation(cls, alloc, dtype: torch.dtype = torch.float32) -> "RWChannel":
        """Channel for an `rwvae.allocation.Allocation` (rule "none" -> identity)."""
        meta = {"rule": alloc.rule, "R": float(alloc.R), "w": float(alloc.w), "name": alloc.name}
        if hasattr(alloc, "fingerprint"):
            meta["allocation_fingerprint"] = alloc.fingerprint()
        if alloc.rule == "none":
            ch = cls.identity(alloc.C, m=alloc.m, dtype=dtype)
            ch.meta.update(meta)
            return ch
        return cls(alloc.m, alloc.A, alloc.B, rate=float(alloc.R), dtype=dtype, meta=meta)

    @classmethod
    def identity(cls, C: int, m=None, dtype: torch.dtype = torch.float32) -> "RWChannel":
        """B0 reference: z' = mu exactly (A = I, B = 0, no rate constraint)."""
        m = np.zeros(C) if m is None else m
        return cls(m, np.eye(C), np.zeros((C, C)), rate=float("inf"), dtype=dtype, meta={"rule": "none"})

    @property
    def C(self) -> int:
        return int(self.m.numel())

    def _a_is_eye(self) -> bool:
        return bool(torch.equal(self.A, torch.eye(self.C, dtype=self.A.dtype, device=self.A.device)))

    @property
    def is_identity(self) -> bool:
        return self._a_is_eye() and not bool(self.B.any())

    def _check(self, x: torch.Tensor) -> torch.dtype:
        if x.dim() < 2 or x.shape[1] != self.C:
            raise ValueError(f"expected a [batch, {self.C}, ...] tensor, got shape {tuple(x.shape)}")
        return torch.promote_types(x.dtype, torch.float32)

    def _mix(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bj...->bi...", M.to(device=x.device, dtype=x.dtype), x)

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.to(device=x.device, dtype=x.dtype).view(1, self.C, *([1] * (x.dim() - 2)))

    def signal(self, mu: torch.Tensor) -> torch.Tensor:
        """Deterministic part m + A (mu - m) (the posterior mean of q(z'|x))."""
        mu = mu.to(self._check(mu))
        if self._a_is_eye():  # skip the (mu - m) + m round trip: B0 must return mu bit-exactly
            return mu
        m = self._mean(mu)
        return m + self._mix(self.A, mu - m)

    def forward(self, mu: torch.Tensor, noise: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply the channel. `noise`, if given, must be standard normal with mu's shape;
        otherwise it is drawn with `generator` (which must live on mu's device)."""
        z = self.signal(mu)
        if noise is not None and tuple(noise.shape) != tuple(mu.shape):
            raise ValueError(f"noise shape {tuple(noise.shape)} != mu shape {tuple(mu.shape)}")
        if not bool(self.B.any()):
            return z
        if noise is None:
            noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        return z + self._mix(self.B, noise.to(device=z.device, dtype=z.dtype))

    def lmmse(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Linear MMSE estimate of mu from z' (P6): m + A (z' - m)."""
        return self.signal(z_prime)

    def extra_repr(self) -> str:
        return f"C={self.C}, rate={float(self.rate):g}, identity={self.is_identity}"
