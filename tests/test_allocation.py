"""Tests for rwvae.allocation (plan v2.1 P1-P7; docs/ARCHITECTURE_V2.md Sec. 3.2).

Reference values come from independent plain-numpy constructions (direct eigendecompositions,
a bisection water-filling solver, spectral projectors, Monte Carlo), not from the module itself.
Exact identities are checked at float64 with relative error <= 1e-8; Monte Carlo checks use
6 standard errors estimated from the samples.
"""
import dataclasses
import math
import subprocess
import sys

import numpy as np
import pytest

from rwvae import allocation as al
from rwvae.allocation import Allocation, solve_allocation

FAMILIES = ("random_pd", "repeated", "correlated", "near_singular")
LOSSY = (("equal_snr", None), ("variance", None), ("r2", 1.5), ("r2", 3.0))


# --------------------------------------------------------------------------- independent helpers

def _rotation(rng, C):
    Q, R = np.linalg.qr(rng.standard_normal((C, C)))
    return Q * np.sign(np.diag(R))


def _spd(U, eig):
    S = (U * np.asarray(eig, dtype=float)) @ U.T
    return 0.5 * (S + S.T)


def _sym_pow(S, p):
    e, U = np.linalg.eigh(S)
    return _spd(U, e ** p)


def _explained(Sigma, k, rng):
    """Sigma^{1/2} K Sigma^{1/2} with eig(K) = k, so 0 <= Sigma_expl <= Sigma (P1)."""
    Sh = _sym_pow(Sigma, 0.5)
    E = Sh @ _spd(_rotation(rng, len(k)), k) @ Sh
    return 0.5 * (E + E.T)


def _family(name, C=32):
    """The four spectrum families of plan Sec. 3.3: (Sigma, Sigma_expl)."""
    rng = np.random.default_rng(FAMILIES.index(name) + 1)
    if name == "random_pd":
        S = _spd(_rotation(rng, C), rng.uniform(0.1, 5.0, C))
        return S, _explained(S, rng.uniform(0.0, 0.95, C), rng)
    if name == "repeated":  # multiplicity-8 eigenvalues; Sigma_expl = Sigma / 2 keeps s repeated for any w
        S = _spd(_rotation(rng, C), np.repeat([4.0, 2.0, 1.0, 0.5], C // 4))
        return S, 0.5 * S
    if name == "correlated":  # one strong common factor: channel correlations ~0.9999, cond ~1e6
        v = rng.standard_normal(C)
        v /= np.linalg.norm(v)
        S = 1e3 * np.outer(v, v) + np.diag(rng.uniform(1e-3, 1e-2, C))
        return S, _explained(S, rng.uniform(0.05, 0.9, C), rng)
    S = _spd(_rotation(rng, C), np.logspace(0.0, -9.0, C))  # near_singular: min eig = 1e-9 max eig
    return S, _explained(S, rng.uniform(0.05, 0.9, C), rng)


def _stats(S, E=None, m=None):
    return {"m": np.zeros(len(S)) if m is None else m, "Sigma": S, "Sigma_expl": E}


def _bisect_waterfill(s, R):
    """Independent reference: bisection on ln(theta) of 0.5 sum ln+(s / theta) = R."""
    ls = np.log(np.asarray(s, dtype=float))
    lo, hi = ls.max() - 2.0 * R - 1.0, ls.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 0.5 * np.sum(np.maximum(ls - mid, 0.0)) > R else (lo, mid)
    lt = 0.5 * (lo + hi)
    return 0.5 * np.maximum(ls - lt, 0.0), math.exp(lt)


def _all_active_rate(s):
    """Rate at which theta = min(s) / 2, i.e. every direction active with keep >= 1/2."""
    return 0.5 * float(np.sum(np.log(s / (0.5 * s.min()))))


# --------------------------------------------------------------------------- spectrum families

def test_spectrum_families_have_the_intended_structure():
    e = {f: np.linalg.eigvalsh(_family(f)[0]) for f in FAMILIES}
    assert e["random_pd"][0] > 0.05
    assert len(np.unique(np.round(e["repeated"], 9))) == 4
    assert 1e5 < e["correlated"][-1] / e["correlated"][0] < 1e7
    assert e["near_singular"][0] / e["near_singular"][-1] == pytest.approx(1e-9, rel=1e-3)
    S = _family("correlated")[0]
    corr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    assert np.median(np.abs(corr[np.triu_indices(32, 1)])) > 0.99


@pytest.mark.parametrize("rule,w", LOSSY)
@pytest.mark.parametrize("family", FAMILIES)
def test_exact_identities_on_all_spectrum_families(family, rule, w):
    S, E = _family(family)
    stats = _stats(S, E, m=np.linspace(-1.0, 1.0, len(S)))
    w_eff = 1.0 if w is None else w
    s_ref = np.sort(np.linalg.eigvalsh(S + (w_eff ** 2 - 1.0) * E))[::-1]
    s = al.weighted_klt(S, E if rule == "r2" else None, w_eff).s
    # gaussian_mutual_information is first-order sensitive to the eigensolver backward error
    # (~eps * cond(Sigma) ~ 1e-7 relative in the smallest direction at cond 1e9); expected_kl
    # cancels it to first order and meets 1e-8 everywhere (see allocation docstrings).
    mi_tol = 1e-6 if family == "near_singular" else 1e-8
    R_all = _all_active_rate(s)
    for R in (0.0, 1e-6, 1.0, 11.0, 44.0, R_all):
        a = solve_allocation(stats, rule, w, R)
        res = al.identity_residuals(a, S)
        assert max(res.values()) <= 1e-8, (R, res)
        assert np.max(np.abs(a.s - s_ref)) <= 1e-8 * s_ref[0]  # P2: s = eig(Sigma_w), descending
        assert np.all(np.diff(a.s) <= 0)
        assert abs(al.expected_kl(a, S) - R) <= 1e-8 * max(1.0, R)
        assert abs(al.gaussian_mutual_information(a, S) - R) <= mi_tol * max(1.0, R)
        assert al.nominal_rate(a) == pytest.approx(R, rel=1e-12, abs=1e-15)
        if R == 0.0:
            assert np.all(a.keep == 0) and np.all(a.A == 0)
        if R == R_all:
            assert np.all(a.keep > 0)
        if rule != "equal_snr":
            assert a.theta == pytest.approx(s.max() if R == 0 else _bisect_waterfill(a.s, R)[1], rel=1e-10)


@pytest.mark.parametrize("family", FAMILIES)
def test_equal_snr_is_basis_independent(family):
    S, _ = _family(family)
    C, R = len(S), 7.0
    a = solve_allocation(_stats(S), "equal_snr", None, R)
    np.testing.assert_array_equal(a.rates, np.full(C, R / C))
    keep = -math.expm1(-2.0 * R / C)
    np.testing.assert_array_equal(a.keep, np.full(C, keep))
    assert np.max(np.abs(a.A - math.sqrt(keep) * np.eye(C))) <= 1e-12
    assert a.theta is None


@pytest.mark.parametrize("family", FAMILIES)
def test_r2_with_unit_weight_equals_variance(family):
    S, E = _family(family)
    v = solve_allocation(_stats(S, E), "variance", None, 11.0)
    r = solve_allocation(_stats(S, E), "r2", 1.0, 11.0)
    for name in ("T", "Tinv", "s", "rates", "keep", "A", "B", "m"):
        np.testing.assert_array_equal(getattr(r, name), getattr(v, name), err_msg=name)
    assert r.theta == v.theta and r.w == v.w == 1.0
    assert v.meta["K"] is None and r.meta["K"] is not None  # r2 still validates/records K


@pytest.mark.parametrize("rule,w", [("variance", None), ("r2", 2.0)])
def test_repeated_spectrum_channel_is_basis_free(rule, w):
    """Inside a repeated eigenspace the KLT basis is arbitrary, but A and B B^T are spectral
    functions: A = sum_g sqrt(keep_g) P_g and B B^T = sum_g (1 - keep_g) lam_g P_g."""
    rng = np.random.default_rng(21)
    U, lam = _rotation(rng, 12), np.repeat([3.0, 1.0, 0.25], 4)
    S = _spd(U, lam)
    a = solve_allocation(_stats(S, 0.5 * S), rule, w, 4.0)  # Sigma_expl = Sigma/2 -> M = c I
    c = 1.0 if w is None else 1.0 + (w * w - 1.0) * 0.5
    rates_ref, _ = _bisect_waterfill(c * lam, 4.0)
    keep_ref = -np.expm1(-2.0 * rates_ref)
    np.testing.assert_allclose(a.s, np.sort(c * lam)[::-1], rtol=1e-12)
    assert np.linalg.norm(a.A - _spd(U, np.sqrt(keep_ref))) <= 1e-10
    assert np.linalg.norm(a.B @ a.B.T - _spd(U, (1.0 - keep_ref) * lam)) <= 1e-10 * np.linalg.norm(S)


# --------------------------------------------------------------------------- P3 reverse water-filling

@pytest.mark.parametrize("R", [1e-9, 0.3, 2.0, 10.0, 60.0])
def test_waterfill_kkt_and_bisection_reference(R):
    s = np.exp(np.random.default_rng(10).uniform(-6.0, 3.0, 20))
    rates, keep, theta = al.reverse_waterfill(s, R)
    active = rates > 0
    assert active.any() and np.all(rates >= 0)
    assert abs(rates.sum() - R) <= 1e-12 * max(1.0, R)
    np.testing.assert_allclose(s[active] * np.exp(-2.0 * rates[active]), theta, rtol=1e-12)  # KKT, active
    assert np.all(s[~active] <= theta * (1.0 + 1e-12))                                        # KKT, inactive
    np.testing.assert_array_equal(keep, -np.expm1(-2.0 * rates))
    np.testing.assert_allclose(keep, np.maximum(1.0 - theta / s, 0.0), rtol=0, atol=1e-12)
    rates_ref, theta_ref = _bisect_waterfill(s, R)
    np.testing.assert_allclose(rates, rates_ref, rtol=0, atol=1e-10)
    assert theta == pytest.approx(theta_ref, rel=1e-10)


def test_waterfill_beats_random_feasible_allocations():
    rng = np.random.default_rng(11)
    s, R = np.exp(rng.uniform(-4.0, 2.0, 12)), 5.0
    rates, _, theta = al.reverse_waterfill(s, R)
    D_opt = float(np.sum(s * np.exp(-2.0 * rates)))
    assert D_opt == pytest.approx(np.sum(np.minimum(s, theta)), rel=1e-12)
    for alpha in (0.05, 1.0, 20.0):  # sparse, uniform and near-equal feasible splits (r >= 0, sum = R)
        r = R * rng.dirichlet(np.full(len(s), alpha), size=5000)
        assert D_opt <= np.min(np.sum(s * np.exp(-2.0 * r), axis=1)) * (1.0 + 1e-12)


def test_waterfill_keep_is_nested_in_rate():
    s = np.exp(np.random.default_rng(12).uniform(-3.0, 3.0, 16))
    grid = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    out = [al.reverse_waterfill(s, R) for R in grid]
    for (_, k1, t1), (_, k2, t2) in zip(out, out[1:]):
        assert np.all(k2 >= k1) and t2 < t1


def test_waterfill_is_input_order_invariant():
    rng = np.random.default_rng(13)
    s = np.concatenate([np.exp(rng.uniform(-2.0, 2.0, 9)), [1.5, 1.5, 1.5]])  # include ties
    rates, keep, theta = al.reverse_waterfill(s, 3.0)
    for _ in range(5):
        p = rng.permutation(len(s))
        r_p, k_p, t_p = al.reverse_waterfill(s[p], 3.0)
        np.testing.assert_array_equal(r_p, rates[p])
        np.testing.assert_array_equal(k_p, keep[p])
        assert t_p == theta


def test_waterfill_ties_are_all_in_or_all_out():
    s = np.array([1.0, 4.0, 1.0, 4.0, 1.0, 0.25])
    R_enter = math.log(4.0)  # rate at which the three 1's start to receive rate (theta = 1)
    rates, _, theta = al.reverse_waterfill(s, R_enter)
    assert theta == 1.0 and np.all(rates[[0, 2, 4, 5]] == 0) and rates[1] == rates[3]
    rates, _, theta = al.reverse_waterfill(s, 1.5 * R_enter)
    assert rates[1] == rates[3] and rates[0] == rates[2] == rates[4] > 0 and rates[5] == 0
    assert theta > 0.25 and rates.sum() == pytest.approx(1.5 * R_enter, rel=1e-15)


def test_waterfill_zero_and_tiny_rate():
    s = np.array([0.5, 3.0, 2.0])
    rates, keep, theta = al.reverse_waterfill(s, 0.0)
    assert np.all(rates == 0) and np.all(keep == 0) and theta == 3.0
    rates, keep, theta = al.reverse_waterfill(s, 1e-12)
    assert rates[1] == 1e-12 and rates[0] == rates[2] == 0.0
    assert keep[1] == pytest.approx(2e-12, rel=1e-9)
    assert theta == pytest.approx(3.0 * math.exp(-2e-12), rel=1e-15)


@pytest.mark.parametrize("s", [[1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [1.0, np.inf], [], [[1.0, 2.0]]])
def test_waterfill_rejects_bad_spectrum(s):
    with pytest.raises(ValueError):
        al.reverse_waterfill(np.array(s, dtype=float), 1.0)


@pytest.mark.parametrize("R", [-1e-12, -1.0, np.nan, np.inf])
def test_waterfill_rejects_bad_rate(R):
    with pytest.raises(ValueError, match="rate R"):
        al.reverse_waterfill(np.ones(3), R)


def test_equal_snr_rates_and_channel_matrices_validate_inputs():
    np.testing.assert_array_equal(al.equal_snr_rates(4, 2.0), np.full(4, 0.5))
    for C in (0, 1.5):
        with pytest.raises(ValueError):
            al.equal_snr_rates(C, 1.0)
    with pytest.raises(ValueError):
        al.equal_snr_rates(4, -1.0)
    I = np.eye(3)
    with pytest.raises(ValueError, match="rates must be >= 0"):
        al.channel_matrices(I, I, np.ones(3), np.array([0.1, -0.1, 0.0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        al.channel_matrices(I, I, np.ones(2), np.zeros(3))
    with pytest.raises(ValueError, match="s must be finite and > 0"):
        al.channel_matrices(I, I, np.array([1.0, -1.0, 1.0]), np.zeros(3))


# --------------------------------------------------------------------------- P1, P2 building blocks

def test_sym_and_spd_helpers():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((6, 6))
    S = X @ X.T + np.eye(6)
    np.testing.assert_array_equal(al.sym(S), al.sym(S).T)
    with pytest.raises(ValueError, match="not symmetric"):
        al.sym(S + 1e-3 * np.triu(np.ones((6, 6)), 1))
    with pytest.raises(ValueError, match="square"):
        al.sym(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        al.sym(np.full((2, 2), np.nan))
    info = al.check_spd(S)
    e = np.linalg.eigvalsh(S)
    assert info["min_eig"] == pytest.approx(e[0]) and info["cond"] == pytest.approx(e[-1] / e[0])
    assert 1.0 <= info["eff_rank"] <= 6.0
    U = _rotation(rng, 6)
    al.check_spd(_spd(U, np.logspace(0, -9, 6)))  # near-singular but PD at rel_tol 1e-12: accepted
    for bad in (_spd(U, np.logspace(0, -13, 6)), _spd(U, [1, 1, 1, 1, 1, 0]), -S):
        with pytest.raises(ValueError, match="not positive definite"):
            al.check_spd(bad)
    R2 = al.sym_sqrt(S)
    np.testing.assert_allclose(R2 @ R2, S, atol=1e-12 * np.linalg.norm(S))
    Wi = al.sym_inv_sqrt(S)
    np.testing.assert_allclose(Wi @ S @ Wi, np.eye(6), atol=1e-12)
    with pytest.raises(ValueError):
        al.sym_sqrt(_spd(U, [1, 1, 1, 1, 1, -1e-3]))
    with pytest.raises(ValueError):
        al.sym_inv_sqrt(_spd(U, [1, 1, 1, 1, 1, 0]))


def test_whitened_relevance_recovers_K_and_records_clipping():
    rng = np.random.default_rng(15)
    S = _spd(_rotation(rng, 8), rng.uniform(0.2, 3.0, 8))
    k0, U = rng.uniform(0.1, 0.9, 8), _rotation(rng, 8)
    K0 = _spd(U, k0)
    Sh = _sym_pow(S, 0.5)
    K, info = al.whitened_relevance(S, Sh @ K0 @ Sh)
    assert np.linalg.norm(K - K0) <= 1e-12 * np.linalg.norm(K0)
    assert info["clipped"] is False and info["eig_min_raw"] == pytest.approx(k0.min(), rel=1e-10)
    # fully explained directions (eig(K) = 1 up to rounding) are accepted and kept <= 1
    K1, info1 = al.whitened_relevance(S, S)
    assert info1["eig_max_raw"] == pytest.approx(1.0, abs=1e-12) and np.linalg.eigvalsh(K1).max() <= 1.0 + 1e-12
    # a violation inside the tolerance is clipped and reported ...
    K2, info2 = al.whitened_relevance(S, (1.0 + 1e-10) * S)
    assert info2["clipped"] is True and info2["eig_max_raw"] > 1.0
    assert np.linalg.eigvalsh(K2).max() <= 1.0 + 1e-12
    # ... outside the tolerance it is an error (P1: 0 <= Sigma_expl <= Sigma)
    for bad in ((1.0 + 1e-6) * S, -1e-6 * S):
        with pytest.raises(ValueError, match="violates P1"):
            al.whitened_relevance(S, bad)
    with pytest.raises(ValueError, match="shape"):
        al.whitened_relevance(S, np.eye(7))


def test_weighted_klt_structure():
    rng = np.random.default_rng(16)
    S = _spd(_rotation(rng, 8), rng.uniform(0.2, 3.0, 8))
    E = _explained(S, rng.uniform(0.0, 0.9, 8), rng)
    pca = al.weighted_klt(S)
    assert pca.K is None and pca.info["K"] is None
    np.testing.assert_array_equal(pca.M_half, np.eye(8))
    np.testing.assert_array_equal(pca.T, pca.Q.T)
    np.testing.assert_array_equal(pca.Tinv, pca.Q)
    np.testing.assert_allclose(pca.T @ pca.T.T, np.eye(8), atol=1e-14)  # w = 1: orthogonal PCA
    assert al.weighted_klt(S, E, 1.0).K is not None
    klt = al.weighted_klt(S, E, 2.0)
    np.testing.assert_allclose(klt.T @ klt.Tinv, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(klt.Q.T @ klt.Q, np.eye(8), atol=1e-14)
    np.testing.assert_allclose(klt.T @ S @ klt.T.T, np.diag(klt.s), atol=1e-12 * klt.s[0])
    assert np.linalg.norm(klt.T @ klt.T.T - np.eye(8)) > 0.1  # w > 1: T^T is NOT T^{-1}
    assert np.all(np.diff(klt.s) <= 0)
    rows = np.argmax(np.abs(klt.Q), axis=0)
    assert np.all(klt.Q[rows, np.arange(8)] > 0)  # sign convention
    again = al.weighted_klt(S, E, 2.0)
    for name in ("T", "Tinv", "s", "Q", "M_half", "K"):
        np.testing.assert_array_equal(getattr(again, name), getattr(klt, name))  # reproducible
    for w in (0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="w must be"):
            al.weighted_klt(S, E, w)
    with pytest.raises(ValueError, match="needs Sigma_expl"):
        al.weighted_klt(S, None, 2.0)


# --------------------------------------------------------------------------- solve_allocation contract

def _invalid_cases():
    rng = np.random.default_rng(17)
    S = _spd(_rotation(rng, 5), [3.0, 2.0, 1.0, 0.5, 0.2])
    E = _explained(S, [0.1, 0.3, 0.5, 0.7, 0.9], rng)
    asym = S.copy()
    asym[0, 1] += 0.1
    ok = _stats(S, E)
    return [
        (_stats(asym, E), "variance", None, 1.0, "not symmetric"),
        (_stats(S - 0.5 * np.eye(5), E), "variance", None, 1.0, "not positive definite"),
        (ok, "r2", 0.9, 1.0, "w must be"),
        (ok, "r2", None, 1.0, "needs a relevance weight"),
        (_stats(S), "r2", 2.0, 1.0, "Sigma_expl"),
        (_stats(S, 1.5 * E + 0.5 * S), "r2", 2.0, 1.0, "violates P1"),
        (ok, "variance", None, -1.0, "rate R"),
        (ok, "variance", None, np.nan, "rate R"),
        (ok, "equal_snr", None, np.inf, "rate R"),
        (ok, "variance", None, None, "needs a rate"),
        (ok, "variance", 2.0, 1.0, "takes no relevance weight"),
        (ok, "equal_snr", 1.5, 1.0, "takes no relevance weight"),
        (ok, "none", 2.0, None, "takes no relevance weight"),
        (ok, "bogus", None, 1.0, "unknown rule"),
        ({"m": np.zeros(4), "Sigma": S}, "variance", None, 1.0, "stats.m"),
        ({"m": np.zeros(5)}, "variance", None, 1.0, "no 'Sigma'"),
    ]


@pytest.mark.parametrize("stats,rule,w,R,match", _invalid_cases())
def test_solve_allocation_rejects_invalid_inputs(stats, rule, w, R, match):
    with pytest.raises(ValueError, match=match):
        solve_allocation(stats, rule, w, R)


class _StatsObject:
    """Duck-typed LatentStats: attributes, meta and fingerprint()."""

    def __init__(self, S, E):
        self.m, self.Sigma, self.Sigma_expl = np.zeros(len(S)), S, E
        self.meta = {"ridge": {"lambda": np.float64(0.25), "ridge_rel": 1e-3, "rank_y": 48}, "cov_ddof": 1}

    def fingerprint(self):
        return "0123456789abcdef"


def test_solve_allocation_accepts_objects_and_records_meta():
    S, E = _family("random_pd")
    a = solve_allocation(_StatsObject(S, E), "r2", 2.0, 11.0)
    assert a.name == "r2_R11_w2" and a.C == 32
    meta = a.meta
    assert (meta["rule"], meta["R"], meta["w"], meta["C"]) == ("r2", 11.0, 2.0, 32)
    assert meta["stats_fingerprint"] == "0123456789abcdef"
    assert meta["ridge"] == {"lambda": 0.25, "ridge_rel": 1e-3, "rank_y": 48}  # numpy scalar -> JSON
    assert "cov_ddof" not in meta
    assert set(meta["K"]) == {"eig_min_raw", "eig_max_raw", "clipped", "tol"}
    assert meta["Sigma_spectrum"]["cond"] == pytest.approx(np.linalg.cond(S), rel=1e-8)
    assert "created_utc" in meta
    d = solve_allocation(_stats(S, E), "variance", None, 11.0)
    assert d.name == "variance_R11" and "stats_fingerprint" not in d.meta and d.meta["K"] is None
    assert solve_allocation(_stats(S, E), "equal_snr", 1, 0.5).name == "equal_snr_R0.5"


def test_rule_none_is_the_rate_free_identity_reference():
    S, _ = _family("random_pd")
    m = np.linspace(-1.0, 1.0, 32)
    a = solve_allocation(_stats(S, m=m), "none", None, None)
    assert a.name == "none" and a.R == math.inf and a.w == 1.0 and a.theta is None
    np.testing.assert_array_equal(a.A, np.eye(32))
    np.testing.assert_array_equal(a.B, np.zeros((32, 32)))
    assert np.all(a.rates == math.inf) and np.all(a.keep == 1.0)
    np.testing.assert_array_equal(a.T, solve_allocation(_stats(S), "variance", None, 1.0).T)  # PCA basis
    assert al.nominal_rate(a) == al.expected_kl(a, S) == al.gaussian_mutual_information(a, S) == math.inf
    assert al.latent_mse_lmmse(a) == 0.0 and al.weighted_distortion(a) == 0.0
    res = al.identity_residuals(a, S)
    assert res["rate"] == res["kl"] == res["variance"] == 0.0 and max(res.values()) <= 1e-12
    np.testing.assert_allclose(al.retention_matrix_whitened(a, S), np.eye(32), atol=1e-12)
    ignored = solve_allocation(_stats(S, m=m), "none", 1, 11.0)  # R is ignored for B0
    assert ignored.R == math.inf and ignored.fingerprint() == a.fingerprint()


# --------------------------------------------------------------------------- persistence

@pytest.mark.parametrize("rule,w,R", [("none", None, None), ("equal_snr", None, 5.0),
                                      ("variance", None, 0.0), ("r2", 2.0, 11.0)])
def test_save_load_roundtrip_is_exact(tmp_path, rule, w, R):
    S, E = _family("random_pd")
    a = solve_allocation(_StatsObject(S, E), rule, w, R)
    path = tmp_path / "allocations" / f"{a.name}.npz"
    a.save(path)
    with np.load(path, allow_pickle=False) as z:  # pickle-free file
        assert all(z[k].dtype != object for k in z.files)
    b = Allocation.load(path)
    for name in ("m", "T", "Tinv", "s", "rates", "keep", "A", "B"):
        np.testing.assert_array_equal(getattr(b, name), getattr(a, name), err_msg=name)
    assert (b.rule, b.R, b.w, b.theta, b.name) == (a.rule, a.R, a.w, a.theta, a.name)
    assert b.meta == a.meta and b.fingerprint() == a.fingerprint()
    assert (a == b) is False  # eq=False: identity semantics, no ambiguous array comparison
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.R = 1.0


def test_fingerprint_is_deterministic_and_sensitive():
    S, E = _family("random_pd")
    a = solve_allocation(_stats(S, E), "r2", 2.0, 11.0)
    fp = a.fingerprint()
    assert len(fp) == 16 and int(fp, 16) >= 0
    assert solve_allocation(_stats(S, E), "r2", 2.0, 11.0).fingerprint() == fp  # meta (time) excluded
    assert dataclasses.replace(a, meta={"note": "x"}).fingerprint() == fp
    others = {solve_allocation(_stats(S, E), "r2", 2.0, 11.5).fingerprint(),
              solve_allocation(_stats(S, E), "r2", 3.0, 11.0).fingerprint(),
              solve_allocation(_stats(S, E), "variance", None, 11.0).fingerprint()}
    # every hashed array matters on its own (e.g. -B is the same channel law but other noise draws)
    others |= {dataclasses.replace(a, m=a.m + 1.0).fingerprint(), dataclasses.replace(a, A=a.A.T).fingerprint(),
               dataclasses.replace(a, B=-a.B).fingerprint(), dataclasses.replace(a, keep=a.keep[::-1]).fingerprint()}
    assert fp not in others and len(others) == 7


def test_load_rejects_foreign_files_and_construction_validates(tmp_path):
    np.savez(tmp_path / "foreign.npz", a=np.zeros(3))
    with pytest.raises(ValueError, match="not an allocation file"):
        Allocation.load(tmp_path / "foreign.npz")
    a = solve_allocation(_stats(*_family("random_pd")), "variance", None, 1.0)
    with pytest.raises(ValueError, match="unknown rule"):
        dataclasses.replace(a, rule="bogus")
    with pytest.raises(ValueError, match="Allocation.A has shape"):
        dataclasses.replace(a, A=np.eye(3))


# --------------------------------------------------------------------------- P7 and retention diagnostics

def test_closed_form_distortions():
    S, E = _family("random_pd")
    v = solve_allocation(_stats(S), "variance", None, 11.0)
    assert al.weighted_distortion(v) == pytest.approx(np.sum(np.minimum(v.s, v.theta)), rel=1e-12)
    assert al.latent_mse_lmmse(v) == pytest.approx(al.weighted_distortion(v), rel=1e-12)  # orthogonal T
    r = solve_allocation(_stats(S, E), "r2", 3.0, 11.0)
    D_direct = np.trace(r.Tinv @ np.diag((1.0 - r.keep) * r.s) @ r.Tinv.T)
    assert al.latent_mse_lmmse(r) == pytest.approx(D_direct, rel=1e-12)
    assert al.weighted_distortion(r) == pytest.approx(np.sum(np.minimum(r.s, r.theta)), rel=1e-12)
    assert abs(al.latent_mse_lmmse(r) / al.weighted_distortion(r) - 1.0) > 1e-3  # T not orthogonal
    zero = solve_allocation(_stats(S, E), "r2", 3.0, 0.0)
    assert al.latent_mse_lmmse(zero) == pytest.approx(np.trace(S), rel=1e-12)  # nothing kept
    assert al.bpp_equivalent(11.0) == pytest.approx(11.0 / (256 * math.log(2.0)), rel=1e-15)
    with pytest.raises(ValueError):
        al.bpp_equivalent(-1.0)


def test_retention_diagnostics_are_bounded_and_consistent():
    S, E = _family("random_pd")
    K, _ = al.whitened_relevance(S, E)
    allocs = [solve_allocation(_stats(S, E), rule, w, R)
              for rule, w, R in (("equal_snr", None, 5.0), ("variance", None, 5.0), ("r2", 2.0, 5.0), ("r2", 3.0, 12.0))]
    Gs = [al.retention_matrix_whitened(a, S) for a in allocs]
    for a, G in zip(allocs, Gs):
        V = al.whitened_basis(a, S)
        np.testing.assert_allclose(V.T @ V, np.eye(32), atol=1e-12)
        Sw = S + (a.w ** 2 - 1.0) * E
        np.testing.assert_allclose(V.T @ Sw @ V, np.diag(a.s), atol=1e-12 * a.s[0])  # V = eigvecs(Sigma_w)
        g = np.linalg.eigvalsh(G)
        assert g.min() >= -1e-12 and g.max() <= 1.0 + 1e-12
        Aw = al.sym_inv_sqrt(S) @ a.A @ al.sym_sqrt(S)
        np.testing.assert_allclose(G, Aw @ Aw.T, atol=1e-12)  # whitened covariance of A (z0 - m)
        assert al.retention_overlap(G, G) == pytest.approx(1.0, abs=1e-14)
        assert 0.0 <= al.semantic_retention(a, S, K) <= 1.0
        k = al.direction_relevance(a, S, K)
        assert k.min() >= -1e-12 and k.max() <= 1.0 + 1e-12
    for i in range(len(Gs)):
        for j in range(len(Gs)):
            o = al.retention_overlap(Gs[i], Gs[j])
            assert 0.0 <= o <= 1.0 + 1e-14 and o == pytest.approx(al.retention_overlap(Gs[j], Gs[i]), abs=1e-15)
    with pytest.raises(ValueError, match="does not belong"):
        al.whitened_basis(allocs[2], 2.0 * S)
    with pytest.raises(ValueError, match="zero retention"):
        al.retention_overlap(Gs[0], np.zeros((32, 32)))
    with pytest.raises(ValueError, match="tr\\(K\\) > 0"):
        al.semantic_retention(allocs[0], S, np.zeros((32, 32)))


# --------------------------------------------------------------------------- theory model (plan v2 App. A, property 7)

D_S, D_N, SIG_S2, SIG_N2, RHO_Y2 = 6, 10, 1.0, 6.0, 0.8


def _theory_model():
    """Sigma = sigma_s^2 P_s + sigma_n^2 P_n (orthogonal subspaces), Sigma_expl = rho_y^2 sigma_s^2 P_s."""
    U = _rotation(np.random.default_rng(5), D_S + D_N)
    Ps, Pn = U[:, :D_S] @ U[:, :D_S].T, U[:, D_S:] @ U[:, D_S:].T
    S, E = SIG_S2 * Ps + SIG_N2 * Pn, RHO_Y2 * SIG_S2 * Ps
    return 0.5 * (S + S.T), 0.5 * (E + E.T)


@pytest.mark.parametrize("w", [1.0, 1.5, 2.0])
def test_theory_model_rate_threshold(w):
    S, E = _theory_model()
    K, _ = al.whitened_relevance(S, E)
    rho, gain = SIG_N2 / SIG_S2, 1.0 + (w * w - 1.0) * RHO_Y2
    assert rho > gain  # nuisance is still ranked first: a positive threshold exists
    R_star = 0.5 * D_N * math.log(rho / gain)
    if w == 1.0:
        from rwvae.theory import rd_threshold  # v1 closed form of Theorem B
        assert R_star == pytest.approx(rd_threshold(D_N, rho), rel=1e-15)
    a = solve_allocation(_stats(S, E), "r2", w, 1.0)
    expected_s = np.array([SIG_N2] * D_N + [SIG_S2 * gain] * D_S)
    np.testing.assert_allclose(a.s, expected_s, rtol=0, atol=1e-12)  # eigenvalues with multiplicities
    k = al.direction_relevance(a, S, K)
    signal = k > RHO_Y2 / 2
    assert signal.sum() == D_S
    np.testing.assert_allclose(k[signal], RHO_Y2, atol=1e-12)
    np.testing.assert_allclose(k[~signal], 0.0, atol=1e-12)
    for R in (0.5 * R_star, R_star * (1.0 - 1e-6)):  # R <= R*_w: the signal subspace gets zero rate
        a = solve_allocation(_stats(S, E), "r2", w, R)
        assert np.all(a.rates[signal] == 0.0)
        assert a.theta == pytest.approx(SIG_N2 * math.exp(-2.0 * R / D_N), rel=1e-12)  # Theorem B
    for R in (R_star * (1.0 + 1e-6), 2.0 * R_star):  # R > R*_w: every signal direction gets rate
        a = solve_allocation(_stats(S, E), "r2", w, R)
        assert np.all(a.rates[signal] > 0.0)


def test_theory_model_large_weight_ranks_signal_first():
    S, E = _theory_model()
    K, _ = al.whitened_relevance(S, E)
    a = solve_allocation(_stats(S, E), "r2", 3.0, 1e-3)  # gain 1 + 8 * 0.8 = 7.4 > rho = 6
    k = al.direction_relevance(a, S, K)
    assert np.all(a.rates[k > RHO_Y2 / 2] > 0) and np.all(a.rates[k < RHO_Y2 / 2] == 0)


def test_theory_model_r2_retains_more_semantics_than_variance():
    S, E = _theory_model()
    K, _ = al.whitened_relevance(S, E)
    R_star_variance = 0.5 * D_N * math.log(SIG_N2 / SIG_S2)
    for R in (4.0, 6.0, 8.0, 12.0, 20.0):
        sem_var = al.semantic_retention(solve_allocation(_stats(S, E), "variance", None, R), S, K)
        sem_r2 = al.semantic_retention(solve_allocation(_stats(S, E), "r2", 2.0, R), S, K)
        assert 0.0 <= sem_var < sem_r2 <= 1.0
        if R <= R_star_variance:
            assert sem_var <= 1e-12  # variance order drops the whole signal subspace (Theorem B)


# --------------------------------------------------------------------------- Monte Carlo (C = 6, N = 200k)

N_MC = 200_000


def _mc_setup():
    rng = np.random.default_rng(31)
    S = _spd(_rotation(rng, 6), [4.0, 2.5, 1.5, 1.0, 0.6, 0.3])
    E = _explained(S, rng.uniform(0.1, 0.9, 6), rng)
    m = rng.standard_normal(6)
    return rng, S, m, solve_allocation(_stats(S, E, m), "r2", 2.0, 4.0)


def _source(rng, kind):
    if kind == "gaussian":
        return rng.standard_normal((N_MC, 6))
    return rng.exponential(1.0, (N_MC, 6)) - 1.0  # mean 0, variance 1, skewness 2


@pytest.mark.parametrize("kind", ["gaussian", "exponential"])
def test_monte_carlo_lmmse_mse_and_covariance(kind):
    rng, S, m, a = _mc_setup()
    z0 = m + _source(rng, kind) @ np.linalg.cholesky(S).T
    zp = m + (z0 - m) @ a.A.T + rng.standard_normal((N_MC, 6)) @ a.B.T
    err = np.sum((z0 - (m + (zp - m) @ a.A.T)) ** 2, axis=1)  # LMMSE estimate m + A (z' - m), P6
    assert abs(err.mean() - al.latent_mse_lmmse(a)) <= 6 * err.std() / math.sqrt(N_MC)  # P7
    d = zp - m  # P4: Cov(z') = Sigma, entrywise within 6 standard errors
    emp = d.T @ d / N_MC
    se = np.sqrt(((d ** 2).T @ (d ** 2) / N_MC - emp ** 2) / N_MC)
    assert np.all(np.abs(emp - S) <= 6 * se)


def test_monte_carlo_expected_kl_with_non_gaussian_source():
    """P5 needs only E z0 = m and Cov z0 = Sigma: average the per-sample Gaussian KL."""
    rng, S, m, a = _mc_setup()
    xi = _source(rng, "exponential")
    assert np.mean(xi ** 3) > 1.5  # clearly non-Gaussian (population skewness 2)
    z0 = m + xi @ np.linalg.cholesky(S).T
    Sinv, BBt = np.linalg.inv(S), a.B @ a.B.T
    const = 0.5 * (np.trace(Sinv @ BBt) - 6 + np.linalg.slogdet(S)[1] - np.linalg.slogdet(BBt)[1])
    d = (z0 - m) @ a.A.T  # posterior mean offset of q(z'|x) = N(m + A (z0 - m), B B^T)
    kl = const + 0.5 * np.sum((d @ Sinv) * d, axis=1)
    assert abs(kl.mean() - a.R) <= 6 * kl.std() / math.sqrt(N_MC)
    assert al.expected_kl(a, S) == pytest.approx(a.R, abs=1e-10)


# --------------------------------------------------------------------------- module contract

def test_allocation_module_imports_numpy_only():
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('alloc_isolated', {al.__file__!r})\n"
            "mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "sys.exit(int('torch' in sys.modules or 'scipy' in sys.modules))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_rwchannel_consumes_a_saved_allocation(tmp_path):
    torch = pytest.importorskip("torch")
    from rwvae.channel import RWChannel

    S, E = _family("random_pd")
    a = solve_allocation(_stats(S, E), "r2", 2.0, 11.0)
    a.save(tmp_path / "a.npz")
    ch = RWChannel.from_allocation(Allocation.load(tmp_path / "a.npz"), dtype=torch.float64)
    np.testing.assert_array_equal(ch.A.numpy(), a.A)
    np.testing.assert_array_equal(ch.B.numpy(), a.B)
    assert float(ch.rate) == 11.0 and ch.meta["allocation_fingerprint"] == a.fingerprint()
    assert ch.meta["name"] == "r2_R11_w2"
    assert RWChannel.from_allocation(solve_allocation(_stats(S), "none", None, None)).is_identity
