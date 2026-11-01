"""Tests for the evaluation metrics (AG decomposition + cheap latent metrics)."""
import numpy as np

from rwvae import metrics


def test_accessibility_gap_is_exact_identity():
    d = metrics.accessibility_gap(risk_z=0.9, risk_phi_x=0.2, risk_phi_xhat=0.3)
    assert abs(d["recon_info_loss"] - 0.1) < 1e-12
    assert abs(d["accessibility_gap"] - 0.6) < 1e-12
    # telescoping: the two parts sum exactly to the total
    assert abs(d["recon_info_loss"] + d["accessibility_gap"] - d["total"]) < 1e-12


def test_linear_probe_risk_zero_for_linear_target():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8))
    W = rng.standard_normal((8, 3))
    Y = X @ W                                   # perfectly linear target
    assert metrics.linear_probe_risk(X, Y, ridge=1e-8) < 1e-6


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    N, D = 400, 10
    # rank-1 matrix -> effective rank ~ 1
    u = rng.standard_normal((N, 1))
    v = rng.standard_normal((1, D))
    assert metrics.effective_rank(u @ v) < 1.5
    # isotropic full-rank -> effective rank close to D
    assert metrics.effective_rank(rng.standard_normal((N, D))) > 0.7 * D


def test_power_spectrum_shape_and_highfreq_order():
    rng = np.random.default_rng(0)
    C, H, W = 4, 16, 16
    smooth = np.cumsum(rng.standard_normal((C, H, W)), axis=-1)  # low-freq heavy
    noise = rng.standard_normal((C, H, W))                      # white / high-freq
    ps_smooth = metrics.latent_power_spectrum(smooth)
    ps_noise = metrics.latent_power_spectrum(noise)
    assert ps_smooth.shape == (H // 2,)
    # smooth latent concentrates power at low radial frequency more than white noise
    assert ps_smooth[0] / ps_smooth.sum() > ps_noise[0] / ps_noise.sum()
