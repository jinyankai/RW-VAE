"""Tests for the v2 section of rwvae.relevance (plan v2.1 P1; ARCHITECTURE_V2.md Sec. 3.3).

All covariances are *sample* covariances, i.e. blocks of one joint np.cov, which is PSD,
so P1 (0 <= Sigma_expl <= Sigma) must hold up to rounding. Exact identities are checked
at float64 relative error <= 1e-10 on well-conditioned data (contract: <= 1e-8).
"""
import numpy as np
import pytest
import torch

from rwvae.relevance import (alignment_shift_scan, cca_spectrum, explained_covariance,
                             explained_fraction, ridge_lambda)

TOL = 1e-10       # exact identities, well-conditioned inputs (observed ~1e-15)
PSD_TOL = 1e-12   # allowed negative eigenvalue relative to ||Sigma||_2 (observed <= 1e-16)
RIDGES = [0.0, 1e-8, 1e-4, 1e-2, 1.0, 100.0]
CASES = ["generic", "strongly_correlated", "rank_deficient_y", "few_samples", "near_singular_y", "isotropic_y"]


def _correlated(rng, n, d):
    L = np.eye(d) + rng.standard_normal((d, d)) / np.sqrt(d)
    return rng.standard_normal(d) + rng.standard_normal((n, d)) @ L


def _isotropic(rng, n, d):
    """Rows whose sample covariance is exactly I up to rounding (a fully repeated spectrum)."""
    Y = rng.standard_normal((n, d))
    Q, _ = np.linalg.qr(Y - Y.mean(0))
    return 3.0 + np.sqrt(n - 1) * Q


def _make(case, seed=0):
    rng = np.random.default_rng(seed)
    if case == "generic":
        z = _correlated(rng, 400, 5)
        y = z @ rng.standard_normal((5, 7)) + rng.standard_normal((400, 7))
    elif case == "strongly_correlated":          # y = invertible mix of z + tiny noise
        z = _correlated(rng, 400, 5)
        y = z @ rng.standard_normal((5, 5)) + 1e-3 * rng.standard_normal((400, 5))
    elif case == "rank_deficient_y":             # 8 features of rank 4: Sigma_y exactly singular
        z = _correlated(rng, 400, 5)
        y = (z[:, :4] @ rng.standard_normal((4, 4)) + rng.standard_normal((400, 4))) @ rng.standard_normal((4, 8))
    elif case == "few_samples":                  # n - 1 = 5 < D = 10: perfect in-sample fit
        z = _correlated(rng, 6, 3)
        y = rng.standard_normal((6, 10))
    elif case == "near_singular_y":              # cond(Sigma_y) ~ 1e10, above the rank floor
        z = _correlated(rng, 400, 5)
        y = z @ rng.standard_normal((5, 6)) + rng.standard_normal((400, 6))
        y[:, -1] = y[:, 0] + 1e-4 * rng.standard_normal(400)
    elif case == "isotropic_y":                  # Sigma_y = I: repeated eigenvalues
        y = _isotropic(rng, 400, 6)
        z = _correlated(rng, 400, 5) + 0.7 * y[:, :5]
    else:
        raise KeyError(case)
    return z, y


def _blocks(z, y):
    J = np.cov(np.hstack([z, y]), rowvar=False)
    C = z.shape[1]
    return J[:C, :C], J[:C, C:], J[C:, C:]


def _min_eig(M):
    return np.linalg.eigvalsh(0.5 * (M + M.T)).min()


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------------------
# explained covariance (P1)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_p1_bounds_on_sample_covariances(case):
    S, Szy, Sy = _blocks(*_make(case))
    scale = np.linalg.eigvalsh(S).max()
    for r in RIDGES:
        E, info = explained_covariance(Szy, Sy, r)
        assert np.array_equal(E, E.T)                            # exactly symmetric output
        assert _min_eig(E) >= -PSD_TOL * scale                   # 0 <= Sigma_expl
        assert _min_eig(S - E) >= -PSD_TOL * scale               # Sigma_expl <= Sigma
        assert info["ridge_rel"] == r and info["lambda"] == ridge_lambda(Sy, r)


@pytest.mark.parametrize("case", CASES)
def test_ridge_shrinks_explained_covariance_in_loewner_order(case):
    S, Szy, Sy = _blocks(*_make(case))
    scale = np.linalg.eigvalsh(S).max()
    Es = [explained_covariance(Szy, Sy, r)[0] for r in RIDGES]
    for a, b in zip(Es, Es[1:]):
        assert _min_eig(a - b) >= -PSD_TOL * scale               # E(lambda_small) >= E(lambda_large)
    traces = [np.trace(E) for E in Es]
    assert all(t1 > t2 for t1, t2 in zip(traces, traces[1:]))


def test_explained_covariance_matches_direct_formula():
    """Sigma_zy (Sigma_y + lambda I)^{-1} Sigma_yz via np.linalg.solve; cond(Sigma_y) ~ 60
    here (for cond ~ 1e10 both computations carry ~eps*cond error, so no exact check)."""
    S, Szy, Sy = _blocks(*_make("generic"))
    D = Sy.shape[0]
    for r in (0.0, 0.37):
        lam = r * np.trace(Sy) / D
        assert ridge_lambda(Sy, r) == pytest.approx(lam, rel=1e-15)
        E, info = explained_covariance(Szy, Sy, r)
        assert _rel(E, Szy @ np.linalg.solve(Sy + lam * np.eye(D), Szy.T)) <= TOL
        assert (info["rank_y"], info["dim_y"], info["rank_tol"]) == (D, D, 1e-12)
        assert info["cond_y"] == pytest.approx(np.linalg.cond(Sy), rel=1e-8)
        assert (info["eig_min_y"], info["eig_max_y"]) == pytest.approx(
            (np.linalg.eigvalsh(Sy).min(), np.linalg.eigvalsh(Sy).max()), rel=1e-12)


def test_y_equal_to_z_is_fully_explained():
    z, _ = _make("generic")
    S, Szy, Sy = _blocks(z, z)
    E, _ = explained_covariance(Szy, Sy, 0.0)
    assert _rel(E, S) <= TOL
    assert np.allclose(cca_spectrum(S, Szy, Sy), 1.0, rtol=0, atol=TOL)
    assert explained_fraction(S, E) == pytest.approx(1.0, abs=TOL)


def test_rank_deficient_sigma_y_uses_pseudo_inverse():
    rng = np.random.default_rng(5)
    z = _correlated(rng, 500, 3)
    y0 = z @ rng.standard_normal((3, 4)) + rng.standard_normal((500, 4))
    y = np.hstack([y0, y0 @ rng.standard_normal((4, 3))])       # 3 exactly dependent features
    S, Szy, Sy = _blocks(z, y)
    _, Szy0, Sy0 = _blocks(z, y0)
    E, info = explained_covariance(Szy, Sy, 0.0)
    E0, info0 = explained_covariance(Szy0, Sy0, 0.0)
    assert (info["rank_y"], info["dim_y"], info["cond_y"]) == (4, 7, float("inf"))
    assert (info0["rank_y"], info0["dim_y"]) == (4, 4) and np.isfinite(info0["cond_y"])
    assert abs(info["eig_min_y"]) <= 1e-12 * info["eig_max_y"]   # the dropped eigenvalues are reported
    assert _rel(E, E0) <= TOL                                    # dependent copies explain nothing new
    assert _rel(E, Szy @ np.linalg.pinv(Sy, rtol=1e-12, hermitian=True) @ Szy.T) <= TOL
    assert _min_eig(S - E) >= -PSD_TOL * np.linalg.eigvalsh(S).max()
    assert explained_covariance(Szy, Sy, 1e-3)[1]["rank_y"] == 4   # rank is reported at ridge > 0 too


# ---------------------------------------------------------------------------------------
# CCA spectrum and explained fraction
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("C,D,isotropic_z", [(4, 6, False), (6, 3, False), (5, 5, False), (4, 6, True)])
def test_cca_spectrum_equals_squared_canonical_correlations(C, D, isotropic_z):
    rng = np.random.default_rng(10 + C + D)
    z = _isotropic(rng, 2000, C) if isotropic_z else _correlated(rng, 2000, C)   # Sigma = I: repeated
    y = z @ rng.standard_normal((C, D)) + 1.5 * rng.standard_normal((2000, D))
    S, Szy, Sy = _blocks(z, y)
    Lz, Ly = np.linalg.cholesky(S), np.linalg.cholesky(Sy)       # independent whitening
    rho = np.linalg.svd(np.linalg.solve(Ly, np.linalg.solve(Lz, Szy).T).T, compute_uv=False)
    spec = cca_spectrum(S, Szy, Sy)
    k = min(C, D)
    assert spec.shape == (C,) and np.all(np.diff(spec) <= 0)
    assert _rel(spec[:k], rho[:k] ** 2) <= TOL
    assert np.abs(spec[k:]).max(initial=0.0) <= 1e-12             # rank(Sigma_zy) <= D
    assert spec.min() >= -1e-12 and spec.max() <= 1 + 1e-12       # P1: eig(K) in [0, 1]
    shrunk = cca_spectrum(S, Szy, Sy, ridge_rel=0.1)              # ridge > 0: shrunken, not CCA^2
    assert np.all(shrunk <= spec + 1e-12) and shrunk[0] < spec[0]


def test_cca_spectrum_recovers_planted_canonical_correlations():
    """Population rho = (0.9, 0.6, 0.3, 0), hidden behind invertible mixings of z and y
    (CCA is invariant to them). n = 20000: sd(rho^2_hat) <= 2 rho (1 - rho^2) / sqrt(n)
    <= 0.006 and the zero correlation is biased up by ~(sqrt(C/n) + sqrt(D/n))^2 ~ 1e-3,
    so 0.03 is > 5 sd."""
    rng = np.random.default_rng(21)
    n, rho = 20_000, np.array([0.9, 0.6, 0.3, 0.0])
    x = rng.standard_normal((n, 4))
    y = np.hstack([rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, 4)), rng.standard_normal((n, 1))])
    z = x @ (np.eye(4) + rng.standard_normal((4, 4)))
    y = y @ (np.eye(5) + rng.standard_normal((5, 5)))
    spec = cca_spectrum(*_blocks(z, y))
    assert np.abs(spec - rho ** 2).max() < 0.03


def test_explained_fraction():
    S = np.diag([3.0, 1.0])
    assert explained_fraction(S, np.diag([1.0, 0.0])) == pytest.approx(0.25, rel=1e-15)
    z, y = _make("generic")
    S, Szy, Sy = _blocks(z, y)
    fracs = [explained_fraction(S, explained_covariance(Szy, Sy, r)[0]) for r in RIDGES]
    assert all(0 <= f <= 1 for f in fracs) and fracs == sorted(fracs, reverse=True)


def test_invalid_inputs_raise():
    S, Szy, Sy = _blocks(*_make("generic"))
    e = np.linalg.eigvalsh(Sy)
    asym = Sy.copy()
    asym[0, 1] += 1e-3 * np.abs(Sy).max()
    indefinite = Sy - (e[0] + 0.1 * (e[-1] - e[0])) * np.eye(len(Sy))   # mixed-sign spectrum
    calls = [
        lambda: ridge_lambda(Sy, -1e-3),
        lambda: ridge_lambda(Sy, float("nan")),
        lambda: ridge_lambda(Sy[:, :-1], 1e-3),
        lambda: ridge_lambda(-np.eye(3), 1e-3),                  # negative trace
        lambda: explained_covariance(Szy, Sy, ridge_rel=float("inf")),
        lambda: explained_covariance(Szy[:, :-1], Sy),            # Sigma_zy / Sigma_y mismatch
        lambda: explained_covariance(Szy[0], Sy),                 # not 2-D
        lambda: explained_covariance(np.full_like(Szy, np.nan), Sy),
        lambda: explained_covariance(Szy, asym),                  # not symmetric
        lambda: explained_covariance(Szy, indefinite),            # not PSD
        lambda: explained_covariance(Szy, np.zeros_like(Sy)),     # no positive eigenvalue
        lambda: explained_covariance(Szy, Sy, rank_tol=-1.0),
        lambda: explained_covariance(Szy, Sy, rank_tol=1.0),
        lambda: explained_covariance(Szy, Sy, ridge_rel=1e-30),   # ridge below the rank floor
        lambda: cca_spectrum(np.diag([1.0, 1.0, 0.0, 1.0, 1.0]), Szy, Sy),   # Sigma singular
        lambda: cca_spectrum(-S, Szy, Sy),                        # Sigma negative definite
        lambda: cca_spectrum(S[:-1, :-1], Szy, Sy),               # Sigma / Sigma_zy mismatch
        lambda: explained_fraction(S, S[:-1, :-1]),
        lambda: explained_fraction(S[:, :-1], S[:, :-1]),         # not square
        lambda: explained_fraction(np.zeros_like(S), S),          # zero trace
    ]
    for i, call in enumerate(calls):
        with pytest.raises(ValueError):
            call()


# ---------------------------------------------------------------------------------------
# alignment shift scan
# ---------------------------------------------------------------------------------------
def _aligned_pair(seed, smooth=False, B=16, C=4, D=6, H=8, W=8):
    """y[b, :, h, w] = W z[b, :, h, w] + noise: correctly aligned token grids."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, C, H, W, generator=g, dtype=torch.float64)
    if smooth:                                   # spatially correlated, like real latents
        z = z + 0.8 * torch.roll(z, 1, dims=3) + 0.5 * torch.roll(z, 1, dims=2)
    Wm = torch.randn(D, C, generator=g, dtype=torch.float64)
    y = torch.einsum("dc,bchw->bdhw", Wm, z) + 0.5 * torch.randn(B, D, H, W, generator=g, dtype=torch.float64)
    return z, y


@pytest.mark.parametrize("smooth", [False, True])
def test_alignment_scan_peaks_at_zero_for_aligned_grids(smooth):
    z, y = _aligned_pair(0, smooth)
    scan = alignment_shift_scan(z.float(), y, max_shift=2)       # mixed input dtypes are fine
    assert set(scan) == {(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)}
    assert all(type(k[0]) is int and type(v) is float for k, v in scan.items())
    best_other = max(v for k, v in scan.items() if k != (0, 0))
    assert max(scan, key=scan.get) == (0, 0) and scan[(0, 0)] > 0.7
    if smooth:
        assert scan[(0, 0)] - best_other > 0.5                   # neighbours explain only part of z
    else:
        assert best_other < 0.05                                 # in-sample bias ~ D / pairs = 6/576


@pytest.mark.parametrize("shift", [(1, 0), (0, 1), (-1, 0), (1, -1)])
def test_alignment_scan_finds_a_one_token_displacement(shift):
    z, y = _aligned_pair(1)
    y_shifted = torch.roll(y, shifts=shift, dims=(2, 3))         # y_shifted[h + a, w + b] = y[h, w]
    scan = alignment_shift_scan(z, y_shifted, max_shift=2)
    assert max(scan, key=scan.get) == shift                      # the documented sign convention
    assert scan[shift] > 0.7 and scan[(0, 0)] < 0.05


def test_alignment_scan_rejects_invalid_inputs():
    z, y = _aligned_pair(2)
    bad = [
        dict(z=z[0], y=y),                                       # not 4-D
        dict(z=z.numpy(), y=y),                                  # not a tensor
        dict(z=z, y=y[:, :, :7]),                                # grid mismatch
        dict(z=z, y=y[:3]),                                      # batch mismatch
        dict(z=z, y=y, max_shift=8),                             # no overlap left
        dict(z=z, y=y, max_shift=-1),
        dict(z=z, y=y, max_shift=1.0),
        dict(z=z[:1, :, :3, :3], y=y[:1, :, :3, :3], max_shift=1),   # 4 pairs <= C + D = 10
        dict(z=z, y=y, ridge_rel=-1.0),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            alignment_shift_scan(**kw)
