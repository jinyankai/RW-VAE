"""Evaluation metrics for Pillars 3 (Accessibility Gap) and 4 (cheap latent metrics).

All pure numpy so they run on CPU without a GPU.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Pillar 3: Accessibility Gap decomposition (exact telescoping identity)
# --------------------------------------------------------------------------- #

def accessibility_gap(risk_z: float, risk_phi_x: float, risk_phi_xhat: float) -> dict:
    """Decompose the total probe-risk gap into a reconstruction-information-loss
    term and the accessibility gap (PROPOSAL.md, Pillar 3):

        R_V(z) - R_V(phi(x)) = [R_V(phi(x_hat)) - R_V(phi(x))] + [R_V(z) - R_V(phi(x_hat))]
             total                 recon info loss (bounded by Lemma 1)        accessibility gap

    Args are the three optimal bounded-probe risks. Returns the three terms; by
    construction recon_info_loss + accessibility_gap == total (exact identity).
    """
    recon_info_loss = risk_phi_xhat - risk_phi_x
    accessibility = risk_z - risk_phi_xhat
    total = risk_z - risk_phi_x
    return {
        "total": total,
        "recon_info_loss": recon_info_loss,
        "accessibility_gap": accessibility,
    }


def linear_probe_risk(features: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> float:
    """Optimal bounded (ridge) linear-probe risk: MSE of the least-squares fit of
    `targets` from `features`. The ridge term implements the norm bound on the
    probe family that Lemma 1 requires."""
    X = np.asarray(features, float)
    Y = np.asarray(targets, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    d = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(d), Xc.T @ Yc)
    resid = Yc - Xc @ beta
    return float(np.mean(np.sum(resid ** 2, axis=1)))


# --------------------------------------------------------------------------- #
# Pillar 4: cheap, training-free latent-quality metrics
# --------------------------------------------------------------------------- #

def effective_rank(features: np.ndarray, eps: float = 1e-12) -> float:
    """RankMe / effective rank: exp(Shannon entropy of the normalized singular
    value spectrum) of the feature matrix [N, D]. No training, no hyperparameters
    (Garrido et al., ICML 2023). Higher = more dimensions actually used."""
    X = np.asarray(features, float)
    s = np.linalg.svd(X, compute_uv=False)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# backward-compatible alias
rankme = effective_rank


def latent_power_spectrum(latent: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """Radially-averaged 2D power spectrum of a latent [C, H, W] (or [B, C, H, W],
    averaged over B and C). A high-frequency-heavy spectrum signals a latent that
    is hard to diffuse (Skorokhodov et al. 2025). Returns a 1D radial profile."""
    x = np.asarray(latent, float)
    if x.ndim == 4:
        x = x.mean(axis=0)
    if x.ndim != 3:
        raise ValueError(f"expected [C,H,W] or [B,C,H,W], got shape {x.shape}")
    C, H, W = x.shape
    fft = np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
    power = (np.abs(fft) ** 2).mean(axis=0)            # [H, W], averaged over channels
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    n_bins = n_bins or (min(H, W) // 2)
    profile = np.zeros(n_bins)
    for b in range(n_bins):
        mask = r == b
        if mask.any():
            profile[b] = power[mask].mean()
    return profile
