"""00 -- M0/D0 numerical verification of the channel mathematics P1-P7 on synthetic data.

Plan v2.1 Sec. 2 (P1-P7, Sec. 2.2 claims), Sec. 3.3 step 1, Sec. 6 M0/M1; docs/ARCHITECTURE_V2.md Sec. 2 and 4.1.
Every check calls the library under test (rwvae.allocation, rwvae.relevance, rwvae.stats, rwvae.channel)
and compares it with an independent numpy re-derivation, a closed form or a Monte-Carlo estimate.

  A  exact identities (float64, C = 32) on four spectrum families -- random PD, repeated spectrum,
     strongly correlated (cond ~1e6), near-singular (cond 1e9) -- with a synthetic Sigma_expl satisfying
     P1, x rules equal_snr / variance / r2 (w = 1.5, 3) x R in {0, 1e-3, 11, 22, 44, R_all} (R_all puts
     every direction at keep >= 1/2): identity_residuals <= 1e-8, s = eig(Sigma + (w^2-1) Sigma_expl),
     expected_kl = R, gaussian_mutual_information = R (1e-6 on the cond-1e9 family: documented floor),
     water-filling KKT + an independent bisection solver, weighted_distortion = sum min(s, theta),
     equal_snr A = sqrt(keep) I, whitened basis orthogonal (1e-7 on the cond-1e9 family: its error is
     ~eps * cond(Sigma)), overlap(G, G) = 1, r2 at w = 1 == variance (bitwise).
  B  Monte Carlo (C = 8, N = 4e5; --quick 6e4) for a Gaussian and a Laplace source with the same mean and
     covariance, pushed through rwvae.channel.RWChannel (float64): Cov(z') = Sigma, the average
     per-sample KL(N(m + A(z0 - m), BB^T) || N(m, Sigma)) = R (P5 without Gaussianity), the empirical LMMSE
     coefficient = A, the empirical LMMSE latent MSE = latent_mse_lmmse, and (documented limitation) the
     cross-token covariance of two correlated tokens becomes A Cov(z0_p, z0_q) A^T.
     Tolerance: |estimate - theory| <= 5 standard errors, SEs estimated from the same samples (empirical
     fourth moments; heteroskedasticity-robust HC0 SEs for regression coefficients).
  C  theory model (auto-encoder/THEORY.md Prop. C, plan v2 appendix A property 7): Sigma = sigma_s^2 P_s +
     sigma_n^2 P_n (d_s = 6, d_n = 10, rho = sigma_n^2 / sigma_s^2 = 6), Sigma_expl = rho_y^2 sigma_s^2 P_s.
     Spectrum of Sigma_w; zero signal rate iff R <= R*_w = (d_n/2) ln(rho / (1 + (w^2-1) rho_y^2)) on a
     (w, R) grid; at R = 0.8 R* (R* = (d_n/2) ln rho), rho_y = 1 the signal keep (= linear-probe R^2 of s
     from z') equals the two-level water-filling closed form and 0, .24, .47, .68, .83 for w = 1, 1.5, 2, 3, 5
     (+-0.01), confirmed by a Monte-Carlo probe; Sigma_expl estimated from samples (GroupedMomentAccumulator
     + explained_covariance) recovers the signal subspace (projector distance < 0.05 at n = 4e5).
  D  guards: non-PD Sigma, Sigma_expl violating P1, R < 0 and w < 1 raise ValueError.

Inputs   none (synthetic data from --seed); --config is optional (if it defines experiment.run_root/name the
         default output goes to <run_root>/<name>/00_verify_math, otherwise to <repo>/runs/00_verify_math).
Outputs  <out>/verify_math.json   every check {section, name, value, tolerance, passed, detail} + meta
         <out>/verify_math.md     PASS/FAIL table grouped by section, with what is and is not proved
         <out>/figures/phase_diagram.png, keep_curves.png   (--figures)
         <out>/config.yaml, log.txt, metrics.jsonl, summary.json  (rwvae.runtime.Run)
Exit     0 iff every check passes, 1 otherwise.

Examples (CPU only, no data needed; identical for ImageNet, SVP and local Imagenette)
    python scripts/00_verify_math.py --figures
    python scripts/00_verify_math.py --quick --out runs/tmp_verify --registry runs/tmp_registry.jsonl
    python scripts/00_verify_math.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        [--config configs/server_a800.yaml --config configs/svp.yaml] --figures
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rwvae import allocation as al
from rwvae import relevance, runtime, stats
from rwvae.channel import RWChannel

FAMILIES = ("random_pd", "repeated", "correlated", "near_singular")
LOSSY = (("equal_snr", None), ("variance", None), ("r2", 1.5), ("r2", 3.0))
EXACT_TOL = 1e-8
MI_FLOOR = 1e-6          # documented: gaussian_mutual_information on the cond-1e9 family (allocation docstring)
ORTHO_FLOOR = 1e-7       # max|V^T V - I| on the cond-1e9 family with w > 1 (~eps * cond: V = Sigma^{-1/2} T^{-1} ...)
ZERO_RATE = 1e-12        # 'zero signal rate' in nats/token (exactly at R = R*_w rounding leaves ~4e-15)
Z_TOL = 5.0              # Monte-Carlo tolerance in standard errors
HAND_KEEP = {1.0: 0.0, 1.5: 0.24, 2.0: 0.47, 3.0: 0.68, 5.0: 0.83}   # plan / THEORY.md, R = 0.8 R*, rho_y = 1
THEORY = {"d_s": 6, "d_n": 10, "rho": 6.0, "sigma_s2": 1.0}

SECTION_TITLES = {
    "A": "A. Exact identities (float64, C=32, four spectrum families)",
    "B": "B. Monte Carlo (C=8, Gaussian and Laplace sources, RWChannel float64)",
    "C": "C. Theory model (Prop. C generalisation, d_s=6, d_n=10, rho=6)",
    "D": "D. Guards (invalid input raises ValueError)",
}

HEADER = """\
**What this shows.** On synthetic data, in float64, the library implementation of P1-P7 (plan Sec. 2.2)
satisfies the exact identities (A), the distribution-free second-moment statements P4-P6 and the expected-KL
identity P5 hold for a non-Gaussian source in Monte Carlo (B), the weighted water-filling reproduces the
closed-form phase diagram of the theory model and a sample-estimated Sigma_expl recovers its signal
subspace (C), and invalid inputs are rejected (D). These are numerical checks of the implementation and
of the formulas, not proofs.

**What this does not show** (plan Sec. 2.2, claims that may not be made): that I(X;Z') equals R for real
data (only I(X;Z') <= R; equality needs the aggregate posterior to be N(m, Sigma)); that the cross-token
covariance is preserved (it becomes A Cov(z0_p, z0_q) A^T, see B); any closed form for pixel PSNR, LPIPS or
rFID; that DINO linear relevance is a sufficient statistic for downstream tasks; or any gFID improvement of
a finite-step DiT.
"""


# --------------------------------------------------------------------------- check bookkeeping

class Checks:
    """Every check is `value <= tolerance` (value is an error, residual, z-score or mismatch count)."""

    def __init__(self, logger):
        self.items: list[dict] = []
        self.logger = logger

    def add(self, section: str, name: str, value, tolerance, **detail) -> bool:
        value, tolerance = float(value), float(tolerance)
        passed = bool(math.isfinite(value) and value <= tolerance)
        self.items.append({"section": section, "name": name, "value": value, "tolerance": tolerance,
                           "passed": passed, "detail": detail})
        if not passed:
            self.logger.warning("FAIL [%s] %s: value %.3e > tolerance %.3e  %s", section, name, value, tolerance, detail)
        return passed


# --------------------------------------------------------------------------- synthetic matrices (independent numpy)

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
    """Sigma^{1/2} K Sigma^{1/2} with eig(K) = k in [0, 1], so 0 <= Sigma_expl <= Sigma (P1)."""
    Sh = _sym_pow(Sigma, 0.5)
    E = Sh @ _spd(_rotation(rng, len(k)), k) @ Sh
    return 0.5 * (E + E.T)


def _family(name, seed, C=32):
    """(Sigma, Sigma_expl) of one spectrum family (plan Sec. 3.3 step 1)."""
    rng = np.random.default_rng([seed, FAMILIES.index(name)])
    if name == "random_pd":
        S = _spd(_rotation(rng, C), rng.uniform(0.1, 5.0, C))
        return S, _explained(S, rng.uniform(0.0, 0.95, C), rng)
    if name == "repeated":  # 4 eigenvalues of multiplicity 8; Sigma_expl = Sigma / 2 keeps s repeated for any w
        S = _spd(_rotation(rng, C), np.repeat([4.0, 2.0, 1.0, 0.5], C // 4))
        return S, 0.5 * S
    if name == "correlated":  # one strong common factor: channel correlations ~0.9999, cond ~1e6
        v = rng.standard_normal(C)
        v /= np.linalg.norm(v)
        S = 1e3 * np.outer(v, v) + np.diag(rng.uniform(1e-3, 1e-2, C))
        return S, _explained(S, rng.uniform(0.05, 0.9, C), rng)
    S = _spd(_rotation(rng, C), np.logspace(0.0, -9.0, C))  # near_singular: min eig = 1e-9 max eig
    return S, _explained(S, rng.uniform(0.05, 0.9, C), rng)


def _bisect_waterfill(s, R):
    """Independent reference for P3: bisection on ln(theta) of 0.5 sum ln+(s / theta) = R."""
    ls = np.log(np.asarray(s, dtype=float))
    lo, hi = ls.max() - 2.0 * R - 1.0, ls.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 0.5 * np.sum(np.maximum(ls - mid, 0.0)) > R else (lo, mid)
    lt = 0.5 * (lo + hi)
    return 0.5 * np.maximum(ls - lt, 0.0), lt


# --------------------------------------------------------------------------- A: exact identities

def section_a(chk: Checks, seed: int) -> None:
    for fam in FAMILIES:
        S, E = _family(fam, seed)
        C = S.shape[0]
        st = {"m": np.linspace(-1.0, 1.0, C), "Sigma": S, "Sigma_expl": E}
        cond = float(np.linalg.cond(S))
        mi_tol = MI_FLOOR if fam == "near_singular" else EXACT_TOL
        ortho_tol = ORTHO_FLOOR if fam == "near_singular" else EXACT_TOL
        for rule, w in LOSSY:
            w_eff = 1.0 if w is None else w
            s_ref = np.sort(np.linalg.eigvalsh(S + (w_eff ** 2 - 1.0) * E))[::-1]
            s_klt = al.weighted_klt(S, E if rule == "r2" else None, w_eff).s
            R_all = 0.5 * float(np.sum(np.log(s_klt / (0.5 * s_klt.min()))))
            for R in (0.0, 1e-3, 11.0, 22.0, 44.0, R_all):
                a = al.solve_allocation(st, rule, w, R)
                case = {"family": fam, "cond": cond, "rule": rule, "w": w_eff, "R": R}
                res = al.identity_residuals(a, S)
                chk.add("A", "identity_residuals: max(klt, variance, rate, kl, lmmse)", max(res.values()), EXACT_TOL,
                        residuals=res, **case)
                chk.add("A", "s == eig(Sigma + (w^2-1) Sigma_expl), descending",
                        max(np.max(np.abs(a.s - s_ref)) / s_ref[0], float(np.any(np.diff(a.s) > 0))), EXACT_TOL, **case)
                chk.add("A", "expected_kl == R", abs(al.expected_kl(a, S) - R) / max(1.0, R), EXACT_TOL, **case)
                chk.add("A", "gaussian_mutual_information == R (1e-6 floor at cond 1e9)",
                        abs(al.gaussian_mutual_information(a, S) - R) / max(1.0, R), mi_tol, **case)
                if rule == "equal_snr":
                    k0 = -math.expm1(-2.0 * R / C)
                    chk.add("A", "equal_snr: A == sqrt(keep) I, keep = 1 - exp(-2R/C)",
                            max(np.max(np.abs(a.A - math.sqrt(k0) * np.eye(C))), np.max(np.abs(a.keep - k0))),
                            EXACT_TOL, **case)
                    wd_ref = math.exp(-2.0 * R / C) * float(np.sum(a.s))
                    chk.add("A", "weighted_distortion == sum (1-keep) s", abs(al.weighted_distortion(a) - wd_ref) / wd_ref,
                            EXACT_TOL, **case)
                else:
                    _kkt_checks(chk, a, R, case)
                V = al.whitened_basis(a, S)
                chk.add("A", "whitened basis V orthogonal: max|V^T V - I| (1e-7 floor at cond 1e9)",
                        np.max(np.abs(V.T @ V - np.eye(C))), ortho_tol, **case)
                G = al.retention_matrix_whitened(a, S)
                if R > 0:
                    chk.add("A", "retention overlap(G, G) == 1", abs(al.retention_overlap(G, G) - 1.0), 1e-12, **case)
                else:  # overlap is undefined for G = 0; check the R = 0 channel instead
                    chk.add("A", "R = 0: A == 0 and G_ret == 0 (overlap undefined)",
                            max(np.max(np.abs(a.A)), np.max(np.abs(G))), 0.0, **case)
        for R in (0.0, 1e-3, 11.0, 22.0, 44.0):
            a, b = al.solve_allocation(st, "r2", 1.0, R), al.solve_allocation(st, "variance", None, R)
            diff = max(float(np.max(np.abs(getattr(a, f) - getattr(b, f)))) for f in ("T", "Tinv", "s", "rates", "A", "B"))
            chk.add("A", "r2 at w=1 == variance (bitwise)", diff, 0.0, family=fam, R=R)


def _kkt_checks(chk: Checks, a, R: float, case: dict) -> None:
    """P3 KKT conditions of the water-filling solution + agreement with the bisection reference."""
    s, r, keep, theta = a.s, a.rates, a.keep, a.theta
    act = r > 0
    kkt = [np.max(np.abs(s[act] * np.exp(-2.0 * r[act]) - theta)) / theta if act.any() else 0.0,  # water level
           max(0.0, float(np.max((s[~act] - theta) / theta))) if (~act).any() else 0.0,      # inactive below level
           max(0.0, -float(np.min(r))),                                                     # r >= 0
           abs(float(np.sum(r)) - R) / max(1.0, R),                                         # budget
           float(np.max(np.abs(keep - np.maximum(0.0, 1.0 - theta / s))))]                  # keep = (1 - theta/s)+
    if R == 0.0:
        kkt.append(abs(theta - s.max()) / s.max())
    chk.add("A", "water-filling KKT (level, complementary slackness, budget, keep=(1-theta/s)+)",
            max(kkt), EXACT_TOL, theta=theta, active=int(act.sum()), **case)
    r_ref, lt_ref = _bisect_waterfill(s, R)
    chk.add("A", "water-filling == independent bisection solver",
            max(np.max(np.abs(r - r_ref)) / max(1.0, R), abs(math.log(theta) - lt_ref)), EXACT_TOL, **case)
    wd_ref = float(np.sum(np.minimum(s, theta)))
    chk.add("A", "weighted_distortion == sum min(s, theta)", abs(al.weighted_distortion(a) - wd_ref) / wd_ref,
            EXACT_TOL, **case)


# --------------------------------------------------------------------------- B: Monte Carlo

def _cov_z(X, Y, target):
    """(max |Cov_hat(X, Y) - target| / SE, Cov_hat); SE from the empirical second moment of the products."""
    n = X.shape[0]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Chat = Xc.T @ Yc / (n - 1)
    var = (Xc ** 2).T @ (Yc ** 2) / n - Chat ** 2
    return float(np.max(np.abs(Chat - target) / np.sqrt(var / n))), Chat


def _mean_z(x, target):
    return abs(float(x.mean()) - target) / (float(x.std(ddof=1)) / math.sqrt(x.size)), float(x.mean())


def _draw(rng, source: str, shape):
    if source == "gaussian":
        return rng.standard_normal(shape)
    return rng.laplace(0.0, 1.0 / math.sqrt(2.0), shape)  # unit variance, excess kurtosis 3


def section_b(chk: Checks, seed: int, N: int, R: float = 4.0, w: float = 2.0, rho_pq: float = 0.6) -> dict:
    C = 8
    rng = np.random.default_rng([seed, 100])
    S = _spd(_rotation(rng, C), rng.uniform(0.2, 4.0, C))
    E = _explained(S, rng.uniform(0.0, 0.9, C), rng)
    m = rng.normal(size=C)
    a = al.solve_allocation({"m": m, "Sigma": S, "Sigma_expl": E}, "r2", w, R)
    ch = RWChannel.from_allocation(a, dtype=torch.float64)
    Gamma = rho_pq * S                                   # Cov(z0_p, z0_q) of two neighbouring tokens
    L = np.linalg.cholesky(np.block([[S, Gamma], [Gamma, S]]))
    Sinv = np.linalg.inv(S)
    BBt = a.B @ a.B.T
    kl_const = 0.5 * (np.trace(Sinv @ BBt) - C + np.linalg.slogdet(S)[1] - np.linalg.slogdet(BBt)[1])
    D_lat = al.latent_mse_lmmse(a)
    info = {"C": C, "N": N, "rule": "r2", "R": R, "w": w, "rho_pq": rho_pq, "D_latent": D_lat,
            "cross_token_change_rel_frobenius": float(np.linalg.norm(a.A @ Gamma @ a.A.T - Gamma) / np.linalg.norm(Gamma))}
    for i, source in enumerate(("gaussian", "laplace")):
        z = np.tile(m, 2) + _draw(rng, source, (N, 2 * C)) @ L.T   # [N, 2C] = [token p | token q]
        z0 = torch.from_numpy(np.ascontiguousarray(z.reshape(N, 2, C).transpose(0, 2, 1)))  # [N, C, 2 tokens]
        gen = torch.Generator().manual_seed(seed * 1000 + i)
        zp = ch(z0, generator=gen)
        zhat = ch.lmmse(zp).numpy()
        zp = zp.numpy()
        z0p, zpp, zpq = z[:, :C], zp[:, :, 0], zp[:, :, 1]
        case = {"source": source, "N": N}

        zs, Chat = _cov_z(zpp, zpp, S)
        chk.add("B", "Cov(z') == Sigma (P4): max |entry error| / SE", zs, Z_TOL,
                rel_frobenius=float(np.linalg.norm(Chat - S) / np.linalg.norm(S)), **case)

        d = (z0p - m) @ a.A.T                                       # mean of q(z'|x) minus m
        kl = kl_const + 0.5 * np.sum((d @ Sinv) * d, axis=1)        # per-sample KL(q(z'|x) || N(m, Sigma))
        zs, kl_mean = _mean_z(kl, R)
        chk.add("B", "mean per-sample KL(q(z'|x) || N(m, Sigma)) == R (P5): |error| / SE", zs, Z_TOL,
                estimate=kl_mean, R=R, **case)

        Xc, Yc = zpp - zpp.mean(0), z0p - z0p.mean(0)
        XtX_inv = np.linalg.inv(Xc.T @ Xc)
        coef = (Yc.T @ Xc) @ XtX_inv                                # empirical Cov(z0, z') Cov(z')^{-1}
        e = Yc - Xc @ coef.T
        se = np.stack([np.sqrt(np.diag(XtX_inv @ ((Xc * e[:, [j]] ** 2).T @ Xc) @ XtX_inv)) for j in range(C)])
        chk.add("B", "empirical LMMSE coefficient == A (P6): max |entry error| / HC0 SE",
                float(np.max(np.abs(coef - a.A) / se)), Z_TOL, max_abs_error=float(np.max(np.abs(coef - a.A))), **case)

        sq = np.sum((z0p - zhat[:, :, 0]) ** 2, axis=1)
        zs, mse = _mean_z(sq, D_lat)
        chk.add("B", "empirical LMMSE latent MSE == latent_mse_lmmse (P7): |error| / SE", zs, Z_TOL,
                estimate=mse, D_latent=D_lat, **case)

        target = a.A @ Gamma @ a.A.T
        zs, Chat = _cov_z(zpp, zpq, target)
        zs_old = _cov_z(zpp, zpq, Gamma)[0]
        chk.add("B", "cross-token Cov(z'_p, z'_q) == A Cov(z0_p, z0_q) A^T (limitation): max |error| / SE", zs, Z_TOL,
                z_vs_unchanged_Gamma=zs_old, change_rel_frobenius=info["cross_token_change_rel_frobenius"], **case)
    return info


# --------------------------------------------------------------------------- C: theory model

def R_star_w(w: float, ry2: float) -> float:
    t = THEORY
    return 0.5 * t["d_n"] * math.log(t["rho"] / (1.0 + (w * w - 1.0) * ry2))


def closed_signal_keep(w: float, ry2: float, R: float) -> float:
    """Two-level reverse water-filling (independent closed form): signal keep at rate R."""
    t = THEORY
    a, b = t["sigma_s2"] * (1.0 + (w * w - 1.0) * ry2), t["sigma_s2"] * t["rho"]
    (hi, d_hi), lo = ((a, t["d_s"]), b) if a > b else ((b, t["d_n"]), a)
    if R <= 0.5 * d_hi * math.log(hi / lo):
        theta = hi * math.exp(-2.0 * R / d_hi)
    else:
        theta = math.exp((t["d_s"] * math.log(a) + t["d_n"] * math.log(b) - 2.0 * R) / (t["d_s"] + t["d_n"]))
    return max(0.0, 1.0 - theta / a)


def _signal(a, S, K, ry2):
    """(signal rate, signal keep) of an allocation: KLT directions with relevance k_i ~ rho_y^2."""
    sig = al.direction_relevance(a, S, K) > 0.5 * ry2
    return float(np.sum(a.rates[sig])), float(np.mean(a.keep[sig])), int(sig.sum())


def _probe_r2(s, zp):
    """Mean over the d_s coordinates of the in-sample linear-probe R^2 of s from [1, z']."""
    X = np.hstack([np.ones((zp.shape[0], 1)), zp])
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    resid = s - X @ coef
    return float(np.mean(1.0 - resid.var(0) / s.var(0)))


def section_c(chk: Checks, seed: int, N: int, n_stats: int, quick: bool) -> dict:
    t = THEORY
    d_s, d_n, rho, ss2 = t["d_s"], t["d_n"], t["rho"], t["sigma_s2"]
    C = d_s + d_n
    rng = np.random.default_rng([seed, 200])
    U = _rotation(rng, C)
    Us, Un = U[:, :d_s], U[:, d_s:]
    Ps = Us @ Us.T
    S = ss2 * Ps + ss2 * rho * (np.eye(C) - Ps)
    R_star = 0.5 * d_n * math.log(rho)
    w_grid = np.linspace(1.0, 4.0, 13 if quick else 31)
    r_frac = np.linspace(0.0, 1.5, 61 if quick else 151)[1:]
    grids = {}
    for ry2 in (1.0, 0.8):
        E = ry2 * ss2 * Ps
        K, _ = al.whitened_relevance(S, E)
        st = {"m": np.zeros(C), "Sigma": S, "Sigma_expl": E}
        for w in HAND_KEEP:
            s = al.weighted_klt(S, E, w).s
            ref = np.sort(np.r_[np.full(d_s, ss2 * (1.0 + (w * w - 1.0) * ry2)), np.full(d_n, ss2 * rho)])[::-1]
            chk.add("C", "spectrum of Sigma_w = {sigma_s^2 (1 + (w^2-1) rho_y^2) x d_s, sigma_n^2 x d_n}",
                    np.max(np.abs(s - ref)) / ref[0], 1e-12, rho_y2=ry2, w=w)
        keep_grid = np.zeros((len(w_grid), len(r_frac)))
        mismatch, keep_err, n_pts = 0, 0.0, 0
        for i, w in enumerate(w_grid):
            Rw = R_star_w(w, ry2)
            Rs = list(r_frac * R_star) + ([Rw * (1 - 1e-6), Rw * (1 + 1e-6)] if Rw > 0 else [])
            for j, R in enumerate(Rs):
                a = al.solve_allocation(st, "r2", float(w), float(R))
                rate, keep, n_sig = _signal(a, S, K, ry2)
                mismatch += int((rate <= ZERO_RATE) != (R <= Rw)) + int(n_sig != d_s)
                keep_err = max(keep_err, abs(keep - closed_signal_keep(w, ry2, R)))
                n_pts += 1
                if j < len(r_frac):
                    keep_grid[i, j] = keep
        chk.add("C", "signal subspace gets zero rate iff R <= R*_w (w, R grid incl. R*_w (1 +- 1e-6)): mismatches",
                mismatch, 0, rho_y2=ry2, grid_points=n_pts, zero_rate_nats=ZERO_RATE, w_range=[1.0, 4.0],
                R_over_Rstar=[float(r_frac[0]), 1.5])
        chk.add("C", "signal keep == two-level closed form over the (w, R) grid", keep_err, 1e-10,
                rho_y2=ry2, grid_points=n_pts)
        grids[ry2] = keep_grid

    # R = 0.8 R*, rho_y = 1: closed form, hand values, Monte-Carlo probe R^2
    ry2, R = 1.0, 0.8 * R_star
    E = ss2 * Ps
    K, _ = al.whitened_relevance(S, E)
    st = {"m": np.zeros(C), "Sigma": S, "Sigma_expl": E}
    keeps = {}
    for k, (w, hand) in enumerate(HAND_KEEP.items()):
        a = al.solve_allocation(st, "r2", w, R)
        _, keep, _ = _signal(a, S, K, ry2)
        cf = closed_signal_keep(w, ry2, R)
        keeps[w] = keep
        chk.add("C", "R=0.8R*: signal keep == closed form", abs(keep - cf), 1e-10, w=w, keep=keep, closed_form=cf)
        chk.add("C", "R=0.8R*: signal keep == 0 / .24 / .47 / .68 / .83 (+-0.01)", abs(keep - hand), 0.01,
                w=w, keep=keep, hand=hand)
        prng = np.random.default_rng([seed, 300, k])
        sig = prng.standard_normal((N, d_s)) * math.sqrt(ss2)
        z0 = sig @ Us.T + prng.standard_normal((N, d_n)) * math.sqrt(ss2 * rho) @ Un.T
        zp = RWChannel.from_allocation(a, dtype=torch.float64)(
            torch.from_numpy(z0), generator=torch.Generator().manual_seed(seed * 1000 + 300 + k)).numpy()
        r2 = _probe_r2(sig, zp)
        se = math.sqrt(4.0 * keep * (1.0 - keep) ** 2 / (N * d_s))
        tol = Z_TOL * se + 3.0 * (C + 1) / N   # 5 SE (asymptotic Var R^2 = 4 R^2 (1-R^2)^2 / n) + in-sample bias ~p/n
        chk.add("C", "Monte-Carlo linear-probe R^2 of s from z' == signal keep", abs(r2 - keep), tol,
                w=w, probe_r2=r2, keep=keep, N=N, tolerance_rule="5 SE + 3 (C+1)/N")

    # Sigma_expl estimated from samples recovers the signal subspace
    D, sig_y = 12, 0.5
    srng = np.random.default_rng([seed, 400])
    Lq, _ = np.linalg.qr(srng.standard_normal((D, d_s)))           # orthonormal columns: rho_y^2 = 1 / (1 + sig_y^2)
    acc = stats.GroupedMomentAccumulator(C + D, num_groups=1)
    for start in range(0, n_stats, 50_000):
        n = min(50_000, n_stats - start)
        sig = srng.standard_normal((n, d_s)) * math.sqrt(ss2)
        z0 = sig @ Us.T + srng.standard_normal((n, d_n)) * math.sqrt(ss2 * rho) @ Un.T
        y = sig @ Lq.T + sig_y * srng.standard_normal((n, D))
        acc.update(torch.from_numpy(np.hstack([z0, y])), 0)
    cov = acc.total().cov()
    S_hat, Szy, Sy = cov[:C, :C], cov[:C, C:], cov[C:, C:]
    E_hat, ridge_info = relevance.explained_covariance(Szy, Sy, ridge_rel=0.0)
    K_hat, k_info = al.whitened_relevance(S_hat, E_hat)
    k_eig, k_vec = np.linalg.eigh(K_hat)
    P_hat = k_vec[:, -d_s:] @ k_vec[:, -d_s:].T
    dist = float(np.linalg.norm(P_hat - Ps))
    angles = stats.principal_angles(k_vec[:, -d_s:], Us)
    tol = 0.12 if quick else 0.05
    chk.add("C", "estimated K: top-d_s eigenvectors recover the signal subspace, ||P_hat - P_s||_F", dist, tol,
            n=n_stats, D=D, rho_y2_true=1.0 / (1.0 + sig_y ** 2), eig_K_top=k_eig[::-1][:d_s + 1],
            max_principal_angle_rad=float(angles.max()), ridge=ridge_info, K_clip=k_info,
            tolerance_rule="0.05 at n=4e5; --quick (n=6e4) uses 0.12 (error scales ~1/sqrt(n))")
    return {"R_star": R_star, "keep_at_0.8Rstar": keeps, "w_grid": w_grid, "R_over_Rstar": r_frac, "grids": grids}


# --------------------------------------------------------------------------- D: guards

def section_d(chk: Checks, seed: int) -> None:
    rng = np.random.default_rng([seed, 500])
    C = 6
    U = _rotation(rng, C)
    S = _spd(U, [3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
    E = _explained(S, rng.uniform(0.1, 0.9, C), rng)
    m = np.zeros(C)
    cases = {
        "non-PD Sigma (one negative eigenvalue)": lambda: al.solve_allocation(
            {"m": m, "Sigma": _spd(U, [3.0, 2.0, 1.5, 1.0, 0.7, -0.1])}, "variance", None, 1.0),
        "singular Sigma (one zero eigenvalue)": lambda: al.solve_allocation(
            {"m": m, "Sigma": _spd(U, [3.0, 2.0, 1.5, 1.0, 0.7, 0.0])}, "variance", None, 1.0),
        "Sigma_expl violating P1 (Sigma_expl = 1.5 Sigma)": lambda: al.solve_allocation(
            {"m": m, "Sigma": S, "Sigma_expl": 1.5 * S}, "r2", 2.0, 1.0),
        "negative R": lambda: al.solve_allocation({"m": m, "Sigma": S}, "variance", None, -1.0),
        "w < 1 (w = 0.5)": lambda: al.solve_allocation({"m": m, "Sigma": S, "Sigma_expl": E}, "r2", 0.5, 1.0),
    }
    for name, fn in cases.items():
        try:
            fn()
            outcome = "no exception"
        except ValueError as e:
            outcome = f"ValueError: {e}"
        except Exception as e:  # noqa: BLE001  (any other exception type is a failed guard)
            outcome = f"{type(e).__name__}: {e}"
        chk.add("D", f"raises ValueError: {name}", 0.0 if outcome.startswith("ValueError") else 1.0, 0.0, outcome=outcome)


# --------------------------------------------------------------------------- outputs

def write_markdown(path: Path, checks: list[dict], meta: dict) -> None:
    lines = ["# 00_verify_math: numerical verification of P1-P7", "", HEADER,
             f"Seed {meta['seed']}, quick={meta['quick']}, Monte-Carlo N={meta['N_mc']}, "
             f"stats n={meta['n_stats']}, git {meta['git_commit']}. "
             f"**{sum(c['passed'] for c in checks)}/{len(checks)} checks passed.**", ""]
    for sec, title in SECTION_TITLES.items():
        groups = defaultdict(list)
        for c in checks:
            if c["section"] == sec:
                groups[c["name"]].append(c)
        lines += [f"## {title}", "", "| check | cases | passed | worst value | its tolerance | status |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, cs in groups.items():
            worst = max(cs, key=lambda c: (not c["passed"], c["value"] / c["tolerance"] if c["tolerance"] > 0
                                           else (math.inf if c["value"] > 0 else 0.0)))
            ok = all(c["passed"] for c in cs)
            cell = name.replace("|", "\\|")  # keep norms like ||x|| from splitting the table row
            lines.append(f"| {cell} | {len(cs)} | {sum(c['passed'] for c in cs)} | {worst['value']:.3g} | "
                         f"{worst['tolerance']:.3g} | {'PASS' if ok else 'FAIL'} |")
        lines.append("")
    failed = [c for c in checks if not c["passed"]]
    if failed:
        lines += ["## Failed checks", ""] + [f"- [{c['section']}] {c['name']}: value {c['value']:.3e} > "
                                             f"{c['tolerance']:.3e}; {runtime.to_jsonable(c['detail'])}" for c in failed]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(fig_dir: Path, sec_c: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig_dir.mkdir(parents=True, exist_ok=True)
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    w_grid, r_frac, R_star = sec_c["w_grid"], sec_c["R_over_Rstar"], sec_c["R_star"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    im = ax.pcolormesh(w_grid, r_frac, sec_c["grids"][1.0].T, cmap=blues, vmin=0, vmax=1, shading="nearest")
    fig.colorbar(im, ax=ax, label="signal keep  (= probe $R^2$ of s from z')")
    w_fine = np.linspace(1.0, 4.0, 400)
    for ry2, style in ((1.0, "-"), (0.8, "--")):
        ax.plot(w_fine, [R_star_w(w, ry2) / R_star if R_star_w(w, ry2) > 0 else np.nan for w in w_fine], style, color="#eb6834", lw=2,
                label=rf"$R^*_w/R^*$, $\rho_y^2={ry2:g}$")
    ax.axhline(0.8, color="#52514b", lw=1, ls=":")
    ax.set(xlabel="relevance weight w", ylabel="R / R*", ylim=(0, 1.5),
           title=r"Theory model: signal gets rate only above $R^*_w$ ($\rho_y^2=1$ shading)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "phase_diagram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    colors = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal: lighter = smaller w
    r = np.linspace(0.0, 1.5, 301)
    for (w, hand), col in zip(HAND_KEEP.items(), colors):
        ax.plot(r, [closed_signal_keep(w, 1.0, x * R_star) for x in r], color=col, lw=2)
        ax.plot([0.8], [sec_c["keep_at_0.8Rstar"][w]], "o", color=col, ms=8, mec="white", mew=2)
        ax.annotate(f"w={w:g}", (1.5, closed_signal_keep(w, 1.0, 1.5 * R_star)), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#2b2a27")
    ax.axvline(0.8, color="#52514b", lw=1, ls=":")
    ax.set(xlabel="R / R*", ylabel="signal keep", xlim=(0, 1.68), ylim=(-0.02, 1.0),
           title=r"Signal keep vs rate ($\rho=6$, $\rho_y=1$; dots: library at 0.8 R*)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "keep_curves.png")
    plt.close(fig)
    return ["figures/phase_diagram.png", "figures/keep_curves.png"]


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = _common.make_parser("M0/D0 numerical verification of P1-P7 on synthetic data (plan Sec. 2).")
    parser.add_argument("--quick", action="store_true", help="smaller Monte-Carlo sizes (N=6e4) and grids")
    parser.add_argument("--figures", action="store_true", help="write figures/phase_diagram.png, keep_curves.png")
    parser.add_argument("--registry", default=None, help="run registry JSONL (default manifests/run_registry.jsonl)")
    args = parser.parse_args(argv)
    cfg = _common.load_cfg(args)
    exp = cfg.get("experiment") or {}
    if args.out is not None:
        out_dir = Path(args.out)
    elif "run_root" in exp and "name" in exp:
        out_dir = _common.stage_dir(cfg, "00_verify_math")
    else:
        out_dir = _common.REPO / "runs" / "00_verify_math"
    cfg.setdefault("experiment", {"name": "00_verify_math"})
    cfg["script"] = "00"
    cfg.setdefault("seed", 0)
    if args.registry is not None:
        cfg["registry"] = args.registry
    N = 60_000 if args.quick else 400_000
    cfg["verify_math"] = {"quick": bool(args.quick), "N_mc": N, "n_stats": N, "figures": bool(args.figures)}
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        seed = int(cfg["seed"])
        runtime.seed_everything(seed)
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        run.logger.info("all checks run on CPU in float64 (selected device %s is not used)", device)
        chk = Checks(run.logger)
        t0 = time.time()
        section_a(chk, seed)
        run.logger.info("A done (%d checks, %.1fs)", len(chk.items), time.time() - t0)
        info_b = section_b(chk, seed, N)
        run.logger.info("B done (%d checks, %.1fs)", len(chk.items), time.time() - t0)
        sec_c = section_c(chk, seed, N, N, args.quick)
        run.logger.info("C done (%d checks, %.1fs)", len(chk.items), time.time() - t0)
        section_d(chk, seed)
        checks = chk.items
        by_section = {s: {"checks": sum(c["section"] == s for c in checks),
                          "passed": sum(c["passed"] for c in checks if c["section"] == s)} for s in SECTION_TITLES}
        for s, v in by_section.items():
            run.log_metrics(None, section=s, **v)
        all_ok = all(c["passed"] for c in checks)
        meta = {"script": "00_verify_math", "plan_sections": ["2", "3.3", "6 M0/M1"], "inputs": "synthetic (no files)",
                "seed": seed, "quick": bool(args.quick), "N_mc": N, "n_stats": N, "dtype": "float64", "device": "cpu",
                "tf32": tf32, "git_commit": runtime.git_commit(), "tolerances": {
                    "exact": EXACT_TOL, "mi_floor_cond_1e9": MI_FLOOR, "monte_carlo_z": Z_TOL},
                "theory_model": THEORY, "monte_carlo": info_b, "R_star": sec_c["R_star"],
                "keep_at_0.8Rstar": {str(k): v for k, v in sec_c["keep_at_0.8Rstar"].items()},
                "elapsed_s": time.time() - t0}
        figures = write_figures(run.out_dir / "figures", sec_c) if args.figures else []
        run.save_json("verify_math.json", {"meta": meta, "all_passed": all_ok, "by_section": by_section,
                                           "figures": figures, "checks": checks})
        write_markdown(run.out_dir / "verify_math.md", checks, meta)
        n_pass = sum(c["passed"] for c in checks)
        run.logger.info("%d/%d checks passed -> %s", n_pass, len(checks), "ALL PASS" if all_ok else "FAILURES")
        run.finish("ok" if all_ok else "checks_failed", passed=n_pass, total=len(checks), by_section=by_section)
        return 0 if all_ok else 1
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
