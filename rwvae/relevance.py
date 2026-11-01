"""Per-channel generation-relevance signals that drive the rate gate.

Two signals, compared as an ablation (PROPOSAL.md, Pillar 2):

  R1 denoising_sensitivity  -- how much each latent channel matters to a
       generation (denoising/diffusion) objective. This is the thesis's primary,
       most novel signal: relevance is defined operationally by the generation
       objective, NOT by variance or frequency.

  R2 alignment_relevance    -- how much each channel aligns with a frozen semantic
       encoder (DINOv2-style). Baseline/ablation; explicitly NOT claimed as novel.

Both return a non-negative tensor of shape [C] suitable for
RelevanceRateGate.relevance_alignment_loss.
"""
from __future__ import annotations

from typing import Callable

import torch


@torch.no_grad()
def denoising_sensitivity(
    latent: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    noise_std: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """R1: per-channel sensitivity of a denoising objective to each channel.

    For each channel c we zero that channel and measure how much the denoiser's
    reconstruction loss of the *clean* latent increases. Larger increase =>
    the channel carries more generation-relevant information.

    Args:
        latent: [B, C, H, W] clean latents.
        denoiser: maps a noisy latent [B,C,H,W] to a prediction of the clean one.
        noise_std: std of Gaussian noise added before denoising.
    Returns:
        relevance: [C] non-negative tensor.
    """
    B, C = latent.shape[:2]
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    noisy = latent + noise_std * noise

    def loss_of(x_latent: torch.Tensor) -> torch.Tensor:
        pred = denoiser(noisy)
        return torch.mean((pred - x_latent) ** 2)

    base = loss_of(latent)
    rel = torch.zeros(C, device=latent.device, dtype=latent.dtype)
    for c in range(C):
        ablated = latent.clone()
        ablated[:, c] = 0.0
        rel[c] = loss_of(ablated) - base
    return rel.clamp_min(0.0)


@torch.no_grad()
def variance_allocation(latent: torch.Tensor) -> torch.Tensor:
    """Baseline allocation by per-channel variance -- i.e. what plain reconstruction
    favors (Main a: capacity to high-variance directions). Used as the 'wrong'
    control to show generation-relevance != variance."""
    B, C = latent.shape[:2]
    return latent.reshape(B, C, -1).var(dim=(0, 2)).clamp_min(0.0)


@torch.no_grad()
def frequency_allocation(latent: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    """Baseline allocation by per-channel high-frequency energy fraction -- the
    frequency axis (FA-VAE / FreqWarm). Used as a control to show generation-
    relevance != frequency. latent [B,C,H,W]; returns [C]."""
    if latent.dim() != 4:
        raise ValueError("frequency_allocation expects a spatial latent [B,C,H,W]")
    B, C, H, W = latent.shape
    fft = torch.fft.fftshift(torch.fft.fft2(latent, dim=(-2, -1)), dim=(-2, -1))
    power = (fft.abs() ** 2)                                   # [B,C,H,W]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float())
    r = r / r.max()
    high = (r >= cutoff).to(power)                            # high-frequency mask
    hi_energy = (power * high).sum(dim=(-2, -1))             # [B,C]
    total = power.sum(dim=(-2, -1)) + 1e-8
    return (hi_energy / total).mean(dim=0).clamp_min(0.0)    # [C]


@torch.no_grad()
def alignment_relevance(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R2: per-channel alignment with a frozen semantic target.

    Relevance of channel c = mean absolute correlation between that channel's
    spatially-pooled activation and the target feature dimensions.

    Args:
        latent: [B, C, H, W].
        target: [B, F] semantic features (e.g. pooled DINOv2).
    Returns:
        relevance: [C] non-negative tensor.
    """
    B, C = latent.shape[:2]
    z = latent.reshape(B, C, -1).mean(dim=2)          # [B, C] spatial pool
    z = z - z.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)         # [B, F]
    z = z / (z.std(0, keepdim=True) + 1e-6)
    t = t / (t.std(0, keepdim=True) + 1e-6)
    corr = (z.T @ t) / max(B - 1, 1)                  # [C, F]
    return corr.abs().mean(dim=1).clamp_min(0.0)      # [C]
