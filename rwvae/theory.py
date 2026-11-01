"""Closed forms and simulators for the rate-distortion phase-diagram theory.

Model (PROPOSAL.md, Main a/b/c): centered data x = U_s s + U_n n, with column-
orthonormal U_s, U_n and U_s^T U_n = 0; s ~ N(0, sigma_s^2 I_{d_s}),
n ~ N(0, sigma_n^2 I_{d_n}) independent. Variance ratio rho = sigma_n^2 / sigma_s^2 > 1
(nuisance has higher variance). All formulas use units sigma_s^2 = 1.

Everything here is pure numpy so the numerical verification runs without a GPU.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Main (a): linear AE / PCA phase diagram
# --------------------------------------------------------------------------- #

def pca_relative_error(k: int, d_s: int, d_n: int, rho: float) -> float:
    """Relative reconstruction error of the optimal rank-k linear AE (PCA).

    For k <= d_n the top-k directions are all nuisance; the error limit as
    rho -> inf is 1 - k/d_n (NOT 0; only k = d_n drives it to 0).
    """
    total = d_s + d_n * rho
    k = int(k)
    if k <= 0:
        captured = 0.0
    elif k <= d_n:
        captured = k * rho
    else:
        captured = d_n * rho + min(k - d_n, d_s) * 1.0
    return (total - captured) / total


def pca_task_r2(k: int, d_s: int, d_n: int) -> float:
    """Task linear-probe R^2 of the optimal rank-k code: 0 for k <= d_n,
    else min(k - d_n, d_s) / d_s."""
    k = int(k)
    if k <= d_n:
        return 0.0
    return min(k - d_n, d_s) / d_s


# --------------------------------------------------------------------------- #
# Main (b): Gaussian rate-distortion reverse water-filling
# --------------------------------------------------------------------------- #

def reverse_water_filling(lambdas: np.ndarray, rate: float) -> tuple[float, np.ndarray]:
    """Reverse water-filling for independent Gaussians (Cover & Thomas Sec 10.3.3).

    Given eigenvalues `lambdas` and total rate `rate` (nats), return the water
    level theta and per-component distortion D_i = min(theta, lambda_i), where
    theta solves sum_i 0.5 * ln(lambda_i / D_i) = rate.
    """
    lambdas = np.asarray(lambdas, dtype=float)

    def rate_at(theta: float) -> float:
        D = np.minimum(theta, lambdas)
        return 0.5 * np.sum(np.log(lambdas / D))

    hi = float(lambdas.max())
    if rate <= 0:
        theta = hi
    else:
        lo = 1e-15
        for _ in range(200):  # bisection; rate_at is decreasing in theta
            mid = 0.5 * (lo + hi)
            if rate_at(mid) > rate:
                lo = mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)
    return theta, np.minimum(theta, lambdas)


def rd_threshold(d_n: int, rho: float) -> float:
    """Rate threshold R* = (d_n / 2) ln(rho) (nats): I(Z;S)=0 iff R <= R*."""
    return 0.5 * d_n * np.log(rho)


def rd_relative_distortion(rate: float, d_s: int, d_n: int, rho: float) -> float:
    """Closed-form relative distortion for R <= R* (all rate on nuisance).

    theta = rho * exp(-2R/d_n); D = d_s*1 + d_n*theta; normalized by total variance.
    At R = R* this equals (d_s + d_n) / (d_s + d_n*rho).
    """
    theta = rho * np.exp(-2.0 * rate / d_n)
    distortion = d_s * 1.0 + d_n * theta
    return distortion / (d_s + d_n * rho)


# --------------------------------------------------------------------------- #
# Main (c): weighted reconstruction threshold shift (amplitude convention)
# --------------------------------------------------------------------------- #

def weighted_pca_threshold(w: float) -> float:
    """PCA threshold rho* under amplitude weighting W=w*P_s+P_n (W inside the
    norm): nuisance stops dominating the signal when rho = w^2."""
    return w ** 2


def weighted_rd_threshold(d_n: int, rho: float, w: float) -> float:
    """Weighted rate threshold R*_w = (d_n/2) ln(rho / w^2) (amplitude convention)."""
    return 0.5 * d_n * np.log(rho / (w ** 2))


# --------------------------------------------------------------------------- #
# Bridge 4: interventional variance ratio  rho_emp = (d_n / d_s) * rho
# --------------------------------------------------------------------------- #

def rho_emp_closed(d_s: int, d_n: int, rho: float) -> float:
    """In-model identity: rho_emp = (d_n / d_s) * rho."""
    return (d_n / d_s) * rho


# --------------------------------------------------------------------------- #
# Simulators (generate the actual model and measure, to check the closed forms)
# --------------------------------------------------------------------------- #

def sample_model(n: int, D: int, d_s: int, d_n: int, rho: float, rng: np.random.Generator):
    """Draw n samples x = U_s s + U_n n and return (X, S, N, U_s, U_n).

    U_s, U_n are random column-orthonormal with U_s^T U_n = 0 (joint QR), sigma_s^2 = 1.
    """
    Q, _ = np.linalg.qr(rng.standard_normal((D, d_s + d_n)))
    U_s, U_n = Q[:, :d_s], Q[:, d_s:]
    S = rng.standard_normal((n, d_s))                      # sigma_s^2 = 1
    N = rng.standard_normal((n, d_n)) * np.sqrt(rho)       # sigma_n^2 = rho
    X = S @ U_s.T + N @ U_n.T
    return X, S, N, U_s, U_n


def pca_simulate(n, D, d_s, d_n, rho, k, weight_w=1.0, rng=None):
    """Fit PCA on sampled data (optionally amplitude-weighted on the signal
    subspace by weight_w) and return empirical (relative_error, task_R2).

    task_R2 is the linear-probe R^2 predicting s from the k-dim code.
    """
    rng = rng or np.random.default_rng(0)
    X, S, N, U_s, U_n = sample_model(n, D, d_s, d_n, rho, rng)

    if weight_w != 1.0:  # amplitude weighting W = w*P_s + P_n  (y = W x)
        W = np.eye(D) + (weight_w - 1.0) * (U_s @ U_s.T)
        Xw = X @ W.T
    else:
        W, Xw = np.eye(D), X

    mu = Xw.mean(0)
    Xc = Xw - mu
    # principal directions of the (weighted) data
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Pk = Vt[:k].T                              # D x k projection basis
    code = Xc @ Pk                             # n x k
    recon = code @ Pk.T + mu
    rel_err = np.sum((Xw - recon) ** 2) / np.sum(Xc ** 2)

    # linear-probe R^2 for s from the code (closed-form least squares)
    Sc = S - S.mean(0)
    beta, *_ = np.linalg.lstsq(code, Sc, rcond=None)
    resid = Sc - code @ beta
    r2 = 1.0 - np.sum(resid ** 2) / np.sum(Sc ** 2)
    return rel_err, max(r2, 0.0)


def rho_emp_simulate(n, D, d_s, d_n, rho, rng=None):
    """Monte-Carlo estimate of rho_emp via interventions (hold s fixed, vary n;
    and vice versa). Returns the empirical ratio, which should approach
    (d_n/d_s)*rho."""
    rng = rng or np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((D, d_s + d_n)))
    U_s, U_n = Q[:, :d_s], Q[:, d_s:]

    def draw_s():
        return rng.standard_normal((n, d_s))

    def draw_n():
        return rng.standard_normal((n, d_n)) * np.sqrt(rho)

    s, s2 = draw_s(), draw_s()
    nn, nn2 = draw_n(), draw_n()
    x_sn = s @ U_s.T + nn @ U_n.T
    x_sn2 = s @ U_s.T + nn2 @ U_n.T            # same s, different n
    x_s2n = s2 @ U_s.T + nn @ U_n.T            # different s, same n
    num = np.mean(np.sum((x_sn - x_sn2) ** 2, axis=1))
    den = np.mean(np.sum((x_sn - x_s2n) ** 2, axis=1))
    return num / den
