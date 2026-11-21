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
v2 note: the v1 code above, including the R1 signal, is kept unchanged as L0 legacy; the
plan-v2.1 R2 explained covariance / CCA spectrum / alignment scan is the "v2" section at the end.
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


# ======================================================================================
# v2 (plan v2.1; contract docs/ARCHITECTURE_V2.md Sec. 3.3)
# R2 explained covariance (P1), whitened-relevance / CCA spectrum and the patch-grid
# alignment scan. numpy float64 only; the torch import at the top belongs to v1 and is
# used here only to accept tensor inputs in `alignment_shift_scan`.
# ======================================================================================
import numpy as np  # noqa: E402  (v2 section appended after the unchanged v1 code)

_SYM_REL_TOL = 1e-8   # largest accepted relative asymmetry ||M - M^T||_F / ||M||_F
_PD_REL_TOL = 1e-12   # Sigma counts as positive definite iff min eig > _PD_REL_TOL * max eig


def _matrix(a, name: str) -> np.ndarray:
    """Finite, non-empty 2-D float64 copy of `a`."""
    M = np.array(a, dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} has {int((~np.isfinite(M)).sum())} non-finite entries")
    return M


def _symmetric(a, name: str) -> np.ndarray:
    """(M + M^T) / 2 after checking that M is square and symmetric to _SYM_REL_TOL."""
    M = _matrix(a, name)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got shape {M.shape}")
    asym = np.linalg.norm(M - M.T) / max(np.linalg.norm(M), np.finfo(float).tiny)
    if asym > _SYM_REL_TOL:
        raise ValueError(f"{name} is not symmetric: relative asymmetry {asym:.3g} > {_SYM_REL_TOL:g}")
    return 0.5 * (M + M.T)


def ridge_lambda(Sigma_y, ridge_rel: float) -> float:
    """Ridge lambda = ridge_rel * tr(Sigma_y) / D (P1, `ridge_rel` mode); ridge_rel >= 0."""
    Sy = _matrix(Sigma_y, "Sigma_y")
    if Sy.shape[0] != Sy.shape[1]:
        raise ValueError(f"Sigma_y must be square, got shape {Sy.shape}")
    r = float(ridge_rel)
    if not (np.isfinite(r) and r >= 0):
        raise ValueError(f"ridge_rel must be finite and >= 0, got {ridge_rel!r}")
    tr = float(np.trace(Sy))
    if tr < 0:
        raise ValueError(f"tr(Sigma_y) = {tr:.3g} < 0: not a covariance")
    return r * tr / Sy.shape[0]


def explained_covariance(Sigma_zy, Sigma_y, ridge_rel: float = 0.0,
                         rank_tol: float = 1e-12) -> tuple[np.ndarray, dict]:
    """R2 explained covariance (P1): Sigma_expl = Sigma_zy (Sigma_y + lambda I)^{-1} Sigma_yz.

    lambda = ridge_lambda(Sigma_y, ridge_rel). With Sigma_y = V diag(e) V^T (eigh) and
    P = Sigma_zy V, Sigma_expl = P diag(f) P^T with f_i = 1 / (e_i + lambda). At lambda = 0
    this is the Moore-Penrose pseudo-inverse: f_i = 1 / e_i if e_i > rank_tol * max(e),
    else 0 (for a PSD joint covariance Sigma_zy vanishes on the null space of Sigma_y);
    info["rank_y"] < info["dim_y"] reports that directions were dropped. A merely
    ill-conditioned direction below that floor is dropped too, so the truncated result may
    lie Loewner-below a tiny-ridge one; P1's bounds still hold. A lambda > 0 at or below the
    floor is rejected rather than silently used. For a PSD joint covariance
    0 <= Sigma_expl <= Sigma, and Sigma_expl decreases (Loewner) as lambda grows.

    Returns (Sigma_expl [C, C], exactly symmetric; info) with info = {lambda, ridge_rel,
    rank_tol, rank_y, dim_y, cond_y (inf if rank_y < dim_y), eig_min_y, eig_max_y}.
    Raises ValueError if Sigma_y is not symmetric, has an eigenvalue below
    -rank_tol * max(e) (not PSD) or has no positive eigenvalue.
    """
    Szy = _matrix(Sigma_zy, "Sigma_zy")
    Sy = _symmetric(Sigma_y, "Sigma_y")
    if Szy.shape[1] != Sy.shape[0]:
        raise ValueError(f"Sigma_zy {Szy.shape} does not match Sigma_y {Sy.shape}")
    rank_tol = float(rank_tol)
    if not 0 <= rank_tol < 1:
        raise ValueError(f"rank_tol must lie in [0, 1), got {rank_tol}")
    lam = ridge_lambda(Sy, ridge_rel)
    e, V = np.linalg.eigh(Sy)
    if not e[-1] > 0:
        raise ValueError(f"Sigma_y has no positive eigenvalue (max eig {e[-1]:.3g})")
    floor = rank_tol * e[-1]
    if e[0] < -floor:
        raise ValueError(f"Sigma_y is not PSD: min eig {e[0]:.3g} < -rank_tol * max eig = {-floor:.3g}")
    kept = e > floor
    if lam == 0:
        f = np.zeros_like(e)
        f[kept] = 1.0 / e[kept]
    elif lam > floor:
        f = 1.0 / (e + lam)
    else:
        raise ValueError(f"ridge lambda={lam:.3g} is not above the numerical-rank floor rank_tol * max eig "
                         f"= {floor:.3g}; use ridge_rel=0 (pseudo-inverse) or a larger ridge_rel")
    P = Szy @ V
    S = (P * f) @ P.T
    rank = int(kept.sum())
    info = {"lambda": lam, "ridge_rel": float(ridge_rel), "rank_tol": rank_tol, "rank_y": rank,
            "dim_y": int(Sy.shape[0]), "cond_y": float(e[-1] / e[0]) if rank == e.size else float("inf"),
            "eig_min_y": float(e[0]), "eig_max_y": float(e[-1])}
    return 0.5 * (S + S.T), info


def cca_spectrum(Sigma, Sigma_zy, Sigma_y, ridge_rel: float = 0.0) -> np.ndarray:
    """Eigenvalues of K = Sigma^{-1/2} Sigma_expl Sigma^{-1/2}, descending (P1).

    They lie in [0, 1] for a PSD joint covariance. At ridge_rel = 0 they are the squared
    canonical correlations rho_i^2 of (z0, y), zero-padded to C; at ridge_rel > 0 they are
    a shrunken version and must not be reported as CCA^2. Returned raw (not clipped).
    Sigma must be positive definite (min eig > 1e-12 * max eig), else ValueError.
    """
    S = _symmetric(Sigma, "Sigma")
    Szy = _matrix(Sigma_zy, "Sigma_zy")
    if Szy.shape[0] != S.shape[0]:
        raise ValueError(f"Sigma {S.shape} does not match Sigma_zy {Szy.shape}")
    e, V = np.linalg.eigh(S)
    if not e[0] > _PD_REL_TOL * e[-1]:
        raise ValueError(f"Sigma must be positive definite: eigenvalues in [{e[0]:.3g}, {e[-1]:.3g}]")
    W = (V / np.sqrt(e)) @ V.T  # Sigma^{-1/2}
    K = W @ explained_covariance(Szy, Sigma_y, ridge_rel)[0] @ W
    return np.linalg.eigvalsh(0.5 * (K + K.T))[::-1].copy()


def explained_fraction(Sigma, Sigma_expl) -> float:
    """tr(Sigma_expl) / tr(Sigma): the share of latent variance linearly explained by y."""
    S, E = _matrix(Sigma, "Sigma"), _matrix(Sigma_expl, "Sigma_expl")
    if S.shape != E.shape or S.shape[0] != S.shape[1]:
        raise ValueError(f"need two square matrices of the same shape, got {S.shape} and {E.shape}")
    tr = float(np.trace(S))
    if not tr > 0:
        raise ValueError(f"tr(Sigma) must be positive, got {tr}")
    return float(np.trace(E)) / tr


def alignment_shift_scan(z: torch.Tensor, y: torch.Tensor, max_shift: int = 2,
                         ridge_rel: float = 1e-3) -> dict[tuple[int, int], float]:
    """Explained fraction of z0 by y for every relative grid shift (plan Sec. 1.2 patch check).

    z [B, C, H, W] and y [B, D, H, W] come from the same images. Shift (dy, dx), for
    dy, dx in [-max_shift, max_shift], pairs the token z[b, :, h, w] with
    y[b, :, h + dy, w + dx] over the valid overlap; the unbiased joint covariance of the
    pairs (float64, two-pass) gives explained_fraction(Sigma, Sigma_expl(ridge_rel)).

    Sign convention: if the feature describing z[..., h, w] sits at y[..., h + a, w + b]
    (y displaced by (a, b) tokens, e.g. y = torch.roll(y0, (a, b), dims=(2, 3))), the scan
    peaks at (dy, dx) = (a, b). A correctly aligned grid must peak at (0, 0).
    The fit is in-sample (bias ~ D / pairs) and larger shifts have fewer pairs, so use
    B (H - max_shift)(W - max_shift) >> C + D token pairs.
    """
    for name, t in (("z", z), ("y", y)):
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            raise ValueError(f"{name} must be a [B, ch, H, W] torch tensor, got {getattr(t, 'shape', type(t))}")
    B, C, H, W = z.shape
    D = y.shape[1]
    if y.shape[0] != B or tuple(y.shape[2:]) != (H, W):
        raise ValueError(f"z {tuple(z.shape)} and y {tuple(y.shape)} must share batch size and grid")
    if isinstance(max_shift, bool) or not isinstance(max_shift, (int, np.integer)) or not 0 <= max_shift < min(H, W):
        raise ValueError(f"max_shift must be an integer in [0, {min(H, W)}), got {max_shift!r}")
    k = int(max_shift)
    if B * (H - k) * (W - k) <= C + D:
        raise ValueError(f"the smallest overlap has {B * (H - k) * (W - k)} token pairs; "
                         f"need more than C + D = {C + D}")
    zn = z.detach().to("cpu", torch.float64).numpy()
    yn = y.detach().to("cpu", torch.float64).numpy()
    out = {}
    for dy in range(-k, k + 1):
        hz, hy = slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy))
        for dx in range(-k, k + 1):
            wz, wy = slice(max(0, -dx), W - max(0, dx)), slice(max(0, dx), W - max(0, -dx))
            zt = zn[:, :, hz, wz].transpose(0, 2, 3, 1).reshape(-1, C)  # flattening of stats.tokens
            yt = yn[:, :, hy, wy].transpose(0, 2, 3, 1).reshape(-1, D)
            zc, yc = zt - zt.mean(0), yt - yt.mean(0)
            n1 = zt.shape[0] - 1
            Sigma_expl, _ = explained_covariance(zc.T @ yc / n1, yc.T @ yc / n1, ridge_rel)
            out[(dy, dx)] = explained_fraction(zc.T @ zc / n1, Sigma_expl)
    return out
