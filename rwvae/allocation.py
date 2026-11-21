"""RW-VAE v2: weighted KLT, reverse water-filling and the closed-form channel (numpy float64).

Plan v2.1 Sec. 2 / docs/ARCHITECTURE_V2.md Sec. 2 (P1-P7). For one latent token z0 in R^C
with mean m and covariance Sigma > 0:

  P1  K = Sigma^{-1/2} Sigma_expl Sigma^{-1/2} (symmetric root), eig(K) in [0, 1]
  P2  M = I + (w^2 - 1) K,  G = M^{1/2} Sigma M^{1/2} = Q diag(s) Q^T (s descending),
      T = Q^T M^{1/2},  T^{-1} = M^{-1/2} Q,  T Sigma T^T = diag(s),  s = eig(Sigma_w),
      Sigma_w = Sigma + (w^2 - 1) Sigma_expl. T is not orthogonal for w > 1 (T^T != T^{-1}).
  P3  min sum_i s_i exp(-2 r_i)  s.t.  r_i >= 0, sum_i r_i = R
      =>  r_i = 0.5 ln+(s_i / theta),  keep_i = 1 - exp(-2 r_i) = (1 - theta / s_i)+
  P4  A = T^{-1} diag(sqrt(keep)) T,  B = T^{-1} diag(sqrt((1 - keep) s)):
      z' = m + A (z0 - m) + B eps  has  A Sigma A^T + B B^T = Sigma (no Gaussianity needed)
  P5  E_x KL(N(m + A (z0 - m), B B^T) || N(m, Sigma)) = sum_i r_i = R for any z0 with mean m
      and covariance Sigma; it upper-bounds I(X; Z')
  P6  Cov(z0, z') Cov(z')^{-1} = Sigma A^T Sigma^{-1} = A  (LMMSE estimate m + A (z' - m))
  P7  D_latent = tr[T^{-1} diag((1 - keep) s) T^{-T}];  sum_i (1 - keep_i) s_i = sum_i min(s_i, theta)

Rules at equal nominal rate R (nats/token): "equal_snr" (B1: PCA basis, r_i = R / C),
"variance" (B2: w = 1 water-filling), "r2" (B3: w >= 1, Sigma_expl from the DINO ridge
regression); "none" (B0: A = I, B = 0) is the rate-free reference, not an equal-rate arm.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

RULES = ("none", "equal_snr", "variance", "r2")

_FORMAT = "rwvae.allocation/v2.1"
_SYM_TOL = 1e-8        # relative Frobenius asymmetry accepted by `sym`
_PD_REL_TOL = 1e-12    # positive definite means min eig > _PD_REL_TOL * max eig
_ORTHO_TOL = 1e-6      # max |V^T V - I| accepted by `whitened_basis`
_ARRAY_FIELDS = ("m", "T", "Tinv", "s", "rates", "keep", "A", "B")


# --------------------------------------------------------------------------- linear algebra

def _square(M, name: str) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} has {int(np.sum(~np.isfinite(M)))} non-finite entries")
    return M


def sym(M, name: str = "M") -> np.ndarray:
    """Return (M + M^T) / 2 for a finite square M with ||M - M^T||_F <= 1e-8 ||M||_F."""
    M = _square(M, name)
    norm, asym = np.linalg.norm(M), np.linalg.norm(M - M.T)
    if asym > _SYM_TOL * norm:
        raise ValueError(f"{name} is not symmetric: ||M - M^T||_F / ||M||_F = {asym / norm:.3e} > {_SYM_TOL:g}")
    return 0.5 * (M + M.T)


def check_spd(Sigma, name: str = "Sigma", rel_tol: float = _PD_REL_TOL) -> dict:
    """Require a symmetric Sigma with min eig > rel_tol * max eig; return its spectrum summary.

    Returns {"min_eig", "max_eig", "cond", "eff_rank"}, eff_rank = exp(-sum p_i ln p_i) with
    p = eig / sum(eig) (entropy effective rank).
    """
    e = np.linalg.eigvalsh(sym(Sigma, name))
    if not e[0] > rel_tol * e[-1]:  # also rejects max eig <= 0
        raise ValueError(f"{name} is not positive definite: min eig {e[0]:.6e} <= {rel_tol:g} * max eig {e[-1]:.6e}")
    p = e / e.sum()
    return {"min_eig": float(e[0]), "max_eig": float(e[-1]), "cond": float(e[-1] / e[0]),
            "eff_rank": float(np.exp(-np.sum(p * np.log(p))))}


def sym_sqrt(M) -> np.ndarray:
    """Principal square root M^{1/2} of a symmetric positive semidefinite matrix (eigh).

    Any negative eigenvalue raises ValueError; nothing is clipped.
    """
    e, U = np.linalg.eigh(sym(M))
    if e[0] < 0:
        raise ValueError(f"sym_sqrt needs a positive semidefinite matrix, min eig = {e[0]:.6e}")
    X = (U * np.sqrt(e)) @ U.T
    return 0.5 * (X + X.T)


def sym_inv_sqrt(M) -> np.ndarray:
    """Inverse principal square root M^{-1/2} of a symmetric positive definite matrix (eigh).

    Positive definite as in `check_spd`: min eig > 1e-12 * max eig.
    """
    e, U = np.linalg.eigh(sym(M))
    if not e[0] > _PD_REL_TOL * e[-1]:
        raise ValueError(f"sym_inv_sqrt needs a positive definite matrix: min eig {e[0]:.6e} <= "
                         f"{_PD_REL_TOL:g} * max eig {e[-1]:.6e}")
    X = (U / np.sqrt(e)) @ U.T
    return 0.5 * (X + X.T)


# --------------------------------------------------------------------------- P1, P2: weighted KLT

def whitened_relevance(Sigma, Sigma_expl, tol: float = 1e-9) -> tuple[np.ndarray, dict]:
    """P1: K = Sigma^{-1/2} Sigma_expl Sigma^{-1/2} (symmetrised); eig(K) must lie in [-tol, 1 + tol].

    eig(K) outside that band means 0 <= Sigma_expl <= Sigma is violated -> ValueError. Eigenvalues
    outside [0, 1] but inside the band are rounding: they are clipped to [0, 1] and K is rebuilt;
    the returned info {"eig_min_raw", "eig_max_raw", "clipped", "tol"} records it.
    """
    S = sym(Sigma, "Sigma")
    check_spd(S, "Sigma")
    E = sym(Sigma_expl, "Sigma_expl")
    if E.shape != S.shape:
        raise ValueError(f"Sigma_expl shape {E.shape} != Sigma shape {S.shape}")
    W = sym_inv_sqrt(S)
    K = W @ E @ W
    K = 0.5 * (K + K.T)
    k, U = np.linalg.eigh(K)
    lo, hi = float(k[0]), float(k[-1])
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(f"Sigma_expl violates P1 (0 <= Sigma_expl <= Sigma): eig(K) spans [{lo:.6e}, {hi:.6e}], "
                         f"outside [-{tol:g}, 1 + {tol:g}]")
    clipped = lo < 0.0 or hi > 1.0
    if clipped:
        K = (U * np.clip(k, 0.0, 1.0)) @ U.T
        K = 0.5 * (K + K.T)
    return K, {"eig_min_raw": lo, "eig_max_raw": hi, "clipped": bool(clipped), "tol": float(tol)}


class KLT(NamedTuple):
    """Weighted Karhunen-Loeve transform (P2): u = T (z0 - m) has Cov(u) = diag(s)."""
    T: np.ndarray           # [C, C]  Q^T M^{1/2}
    Tinv: np.ndarray        # [C, C]  M^{-1/2} Q = T^{-1}
    s: np.ndarray           # [C]     eig(G) = eig(Sigma_w), descending
    Q: np.ndarray           # [C, C]  orthonormal eigenvectors of G
    M_half: np.ndarray      # [C, C]  M^{1/2} (identity for w == 1)
    K: np.ndarray | None    # [C, C]  whitened relevance (None without Sigma_expl)
    info: dict              # {"Sigma": check_spd(...), "K": whitened_relevance info or None}


def _check_w(w) -> float:
    w = float(w)
    if not (math.isfinite(w) and w >= 1.0):
        raise ValueError(f"relevance weight w must be finite and >= 1, got {w!r}")
    return w


def _sign_fix(Q: np.ndarray) -> np.ndarray:
    """Flip columns so that each column's largest-|entry| is positive (reproducible signs)."""
    rows = np.argmax(np.abs(Q), axis=0)
    return Q * np.where(Q[rows, np.arange(Q.shape[1])] < 0, -1.0, 1.0)


def weighted_klt(Sigma, Sigma_expl=None, w: float = 1.0) -> KLT:
    """P2: weighted KLT for the metric M = I + (w^2 - 1) K.

    w == 1 is plain PCA: M = I, T = Q_Sigma^T, Tinv = Q_Sigma, s = eig(Sigma). w > 1 needs
    Sigma_expl. Whenever Sigma_expl is given, K is computed (and P1 checked), also for w == 1.

    Eigenvector signs follow `_sign_fix`, so the same input on the same LAPACK build always gives
    the same T. With repeated eigenvalues the basis inside an eigenspace is not unique: A and
    B B^T are still unique, B only up to a rotation inside that eigenspace.
    """
    w = _check_w(w)
    if w > 1.0 and Sigma_expl is None:
        raise ValueError(f"w = {w:g} > 1 needs Sigma_expl (rule 'r2')")
    S = sym(Sigma, "Sigma")
    info: dict[str, Any] = {"Sigma": check_spd(S, "Sigma"), "K": None}
    K = None
    if Sigma_expl is not None:
        K, info["K"] = whitened_relevance(S, Sigma_expl)
    C = S.shape[0]
    if w == 1.0:
        s, Q = np.linalg.eigh(S)
        s, Q = s[::-1].copy(), _sign_fix(Q[:, ::-1])
        return KLT(Q.T.copy(), Q.copy(), s, Q, np.eye(C), K, info)
    M = np.eye(C) + (w * w - 1.0) * K
    M_half, M_inv_half = sym_sqrt(M), sym_inv_sqrt(M)
    G = M_half @ S @ M_half
    s, Q = np.linalg.eigh(0.5 * (G + G.T))
    s, Q = s[::-1].copy(), _sign_fix(Q[:, ::-1])
    return KLT(Q.T @ M_half, M_inv_half @ Q, s, Q, M_half, K, info)


# --------------------------------------------------------------------------- P3, P4: rates and channel

def _check_rate(R) -> float:
    R = float(R)
    if not (math.isfinite(R) and R >= 0.0):
        raise ValueError(f"rate R must be finite and >= 0 nats/token, got {R!r}")
    return R


def reverse_waterfill(s, R) -> tuple[np.ndarray, np.ndarray, float]:
    """P3: closed-form reverse water-filling; returns (rates, keep, theta) in the input order of s.

    With ln s sorted descending, R_k = 0.5 sum_{i<=k} (ln s_i - ln s_k) is the total rate at
    which direction k starts to receive rate (R_1 = 0, non-decreasing). The active set is the
    first k = #{k : R_k < R} directions and, solving 0.5 sum_{i<=k} (ln s_i - ln theta) = R,

        r_i = 0.5 (ln s_i - ln s_k) + (R - R_k) / k,   ln theta = ln s_k - 2 (R - R_k) / k,

    so s_i exp(-2 r_i) = theta on the active set and s_i <= theta elsewhere; tied s_i are all in
    or all out. R == 0 gives rates = keep = 0 and theta = max(s). keep = -expm1(-2 r).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"s must be a non-empty 1-D array, got shape {s.shape}")
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError(f"s must be finite and > 0, got min {np.min(s)!r}, max {np.max(s)!r}")
    R = _check_rate(R)
    C = s.size
    if R == 0.0:
        return np.zeros(C), np.zeros(C), float(s.max())
    log_s = np.log(s)
    order = np.argsort(-log_s, kind="stable")
    ls = log_s[order]
    Rk = np.concatenate(([0.0], np.cumsum(0.5 * np.arange(1, C) * (ls[:-1] - ls[1:]))))
    k = int(np.searchsorted(Rk, R, side="left"))  # >= 1 because Rk[0] = 0 < R
    excess = (R - Rk[k - 1]) / k
    r_sorted = np.zeros(C)
    r_sorted[:k] = 0.5 * (ls[:k] - ls[k - 1]) + excess
    rates = np.empty(C)
    rates[order] = r_sorted
    return rates, -np.expm1(-2.0 * rates), float(np.exp(ls[k - 1] - 2.0 * excess))


def equal_snr_rates(C: int, R) -> np.ndarray:
    """B1: the same rate R / C in every direction (equal keep, so A = sqrt(keep) I in any basis)."""
    if int(C) != C or C < 1:
        raise ValueError(f"C must be a positive integer, got {C!r}")
    return np.full(int(C), _check_rate(R) / C)


def channel_matrices(T, Tinv, s, rates) -> tuple[np.ndarray, np.ndarray]:
    """P4: A = T^{-1} diag(sqrt(keep)) T and B = T^{-1} diag(sqrt(exp(-2 r) s)).

    keep = -expm1(-2 r) and 1 - keep = exp(-2 r) are evaluated separately (no cancellation).
    """
    T, Tinv = _square(T, "T"), _square(Tinv, "Tinv")
    s, rates = np.asarray(s, dtype=np.float64), np.asarray(rates, dtype=np.float64)
    C = T.shape[0]
    if Tinv.shape != (C, C) or s.shape != (C,) or rates.shape != (C,):
        raise ValueError(f"shape mismatch: T {T.shape}, Tinv {Tinv.shape}, s {s.shape}, rates {rates.shape}")
    if not np.all(rates >= 0):  # also rejects NaN
        raise ValueError(f"rates must be >= 0, got min {np.min(rates)!r}")
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError(f"s must be finite and > 0, got min {np.min(s)!r}")
    A = (Tinv * np.sqrt(-np.expm1(-2.0 * rates))) @ T
    B = Tinv * np.sqrt(np.exp(-2.0 * rates) * s)
    return A, B


# --------------------------------------------------------------------------- Allocation

def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"meta value of type {type(x).__name__} is not JSON serialisable")


def _json_roundtrip(meta: Mapping) -> dict:
    return json.loads(json.dumps(dict(meta), sort_keys=True, default=_json_default))


@dataclass(frozen=True, eq=False)
class Allocation:
    """One channel arm: KLT, per-direction rates/keep and the channel matrices (all float64).

    rates are nats/token per KLT direction (sum = R), keep = 1 - exp(-2 rates), theta is the
    water level (None for "equal_snr" and "none"). meta must be JSON-serialisable; it is passed
    through JSON on construction so that save/load round-trips it exactly.
    """
    rule: str
    R: float
    w: float
    m: np.ndarray
    T: np.ndarray
    Tinv: np.ndarray
    s: np.ndarray
    rates: np.ndarray
    keep: np.ndarray
    theta: float | None
    A: np.ndarray
    B: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; expected one of {RULES}")
        for name in _ARRAY_FIELDS:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        C = self.m.shape[0] if self.m.ndim == 1 else -1
        for name in _ARRAY_FIELDS:
            want = (C,) if name in ("m", "s", "rates", "keep") else (C, C)
            if getattr(self, name).shape != want:
                raise ValueError(f"Allocation.{name} has shape {getattr(self, name).shape}, expected {want}")
        object.__setattr__(self, "R", float(self.R))
        object.__setattr__(self, "w", float(self.w))
        object.__setattr__(self, "theta", None if self.theta is None else float(self.theta))
        object.__setattr__(self, "meta", _json_roundtrip(self.meta))

    @property
    def C(self) -> int:
        return int(self.m.shape[0])

    @property
    def name(self) -> str:
        if self.rule == "none":
            return "none"
        if self.rule == "r2":
            return f"r2_R{self.R:g}_w{self.w:g}"
        return f"{self.rule}_R{self.R:g}"

    def save(self, path) -> None:
        """Write a pickle-free .npz; R, w, theta are float arrays (theta None -> empty array)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        theta = np.empty(0) if self.theta is None else np.array(self.theta, dtype=np.float64)
        with open(path, "wb") as fh:
            np.savez(fh, format=np.array(_FORMAT), rule=np.array(self.rule), R=np.array(self.R),
                     w=np.array(self.w), theta=theta, meta=np.array(json.dumps(self.meta, sort_keys=True)),
                     **{name: getattr(self, name) for name in _ARRAY_FIELDS})

    @classmethod
    def load(cls, path) -> "Allocation":
        with np.load(path, allow_pickle=False) as z:
            fmt = str(z["format"]) if "format" in z.files else None
            if fmt != _FORMAT:
                raise ValueError(f"{path}: not an allocation file of format {_FORMAT!r} (found {fmt!r})")
            theta = z["theta"]
            return cls(rule=str(z["rule"]), R=float(z["R"]), w=float(z["w"]),
                       theta=None if theta.size == 0 else float(theta),
                       meta=json.loads(str(z["meta"])), **{name: z[name] for name in _ARRAY_FIELDS})

    def fingerprint(self) -> str:
        """sha256 over (rule, R, w) and the little-endian float64 bytes of m, A, B, keep (16 hex)."""
        h = hashlib.sha256(json.dumps([self.rule, repr(self.R), repr(self.w)]).encode())
        for a in (self.m, self.A, self.B, self.keep):
            a = np.ascontiguousarray(a, dtype="<f8")
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()[:16]


def _stats_field(stats, key: str, required: bool = True):
    value = stats.get(key) if isinstance(stats, Mapping) else getattr(stats, key, None)
    if value is None and required:
        raise ValueError(f"stats has no {key!r}")
    return value


def solve_allocation(stats, rule: str, w: float | None, R: float | None) -> Allocation:
    """Build one arm's allocation from token statistics (attributes or dict keys m, Sigma, and
    Sigma_expl for "r2", e.g. a LatentStats).

    - "none" (B0): A = I, B = 0, R = inf, rates = inf, keep = 1, theta = None, PCA basis; R ignored.
    - "equal_snr" (B1): PCA basis, rates = R / C, theta = None.
    - "variance" (B2): PCA basis, reverse water-filling on eig(Sigma).
    - "r2" (B3): weighted KLT with w >= 1 and Sigma_expl; w == 1 reproduces "variance" exactly.
    Rules other than "r2" take no weight: w must be None or 1.
    meta: rule, R, w, C, Sigma spectrum, K clipping info, creation time, stats.fingerprint() if
    available, and stats.meta["ridge"] (ridge / lambda behind Sigma_expl) if present.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    S = sym(_stats_field(stats, "Sigma"), "Sigma")
    C = S.shape[0]
    m = np.asarray(_stats_field(stats, "m"), dtype=np.float64)
    if m.shape != (C,) or not np.all(np.isfinite(m)):
        raise ValueError(f"stats.m must be a finite vector of length {C}, got shape {m.shape}")
    Sigma_expl = None
    if rule == "r2":
        if w is None:
            raise ValueError("rule 'r2' needs a relevance weight w >= 1")
        w = _check_w(w)
        Sigma_expl = _stats_field(stats, "Sigma_expl")
    elif w is not None and w != 1:
        raise ValueError(f"rule {rule!r} takes no relevance weight; w must be None or 1, got {w!r}")
    else:
        w = 1.0
    klt = weighted_klt(S, Sigma_expl, w)

    if rule == "none":
        R, theta = math.inf, None
        rates, keep = np.full(C, math.inf), np.ones(C)
        A, B = np.eye(C), np.zeros((C, C))
    else:
        if R is None:
            raise ValueError(f"rule {rule!r} needs a rate R (nats/token)")
        R = _check_rate(R)
        if rule == "equal_snr":
            rates, theta = equal_snr_rates(C, R), None
            keep = -np.expm1(-2.0 * rates)
        else:
            rates, keep, theta = reverse_waterfill(klt.s, R)
        A, B = channel_matrices(klt.T, klt.Tinv, klt.s, rates)

    meta: dict[str, Any] = {"rule": rule, "R": R, "w": w, "C": C, "Sigma_spectrum": klt.info["Sigma"],
                            "K": klt.info["K"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if callable(getattr(stats, "fingerprint", None)):
        meta["stats_fingerprint"] = stats.fingerprint()
    stats_meta = _stats_field(stats, "meta", required=False)
    if isinstance(stats_meta, Mapping) and "ridge" in stats_meta:
        meta["ridge"] = stats_meta["ridge"]
    return Allocation(rule=rule, R=R, w=w, m=m, T=klt.T, Tinv=klt.Tinv, s=klt.s, rates=rates, keep=keep,
                      theta=theta, A=A, B=B, meta=meta)


# --------------------------------------------------------------------------- closed-form diagnostics

def _whitening(alloc: Allocation, Sigma) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Sigma, Sigma^{-1/2}, Sigma^{1/2}) for a PD Sigma of the allocation's dimension."""
    S = sym(Sigma, "Sigma")
    if S.shape != (alloc.C, alloc.C):
        raise ValueError(f"Sigma shape {S.shape} does not match allocation C = {alloc.C}")
    check_spd(S, "Sigma")
    return S, sym_inv_sqrt(S), sym_sqrt(S)


def _log_abs_det(M: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(M)
    return -math.inf if sign == 0 else float(logdet)


def nominal_rate(alloc: Allocation) -> float:
    """sum_i r_i in nats/token (inf for rule "none")."""
    return float(np.sum(alloc.rates))


def expected_kl(alloc: Allocation, Sigma) -> float:
    """P5 from A, B and Sigma only (rates unused): E_x KL(N(m + A (z0 - m), B B^T) || N(m, Sigma))

        = 0.5 [tr(Sigma^{-1} B B^T) + tr(Sigma^{-1} A Sigma A^T) - C + logdet Sigma - logdet B B^T]

    for any z0 with mean m and covariance Sigma. Evaluated in Sigma-whitened coordinates,
    Aw = Sigma^{-1/2} A Sigma^{1/2} and Bw = Sigma^{-1/2} B, as 0.5 (||Aw||_F^2 + ||Bw||_F^2 - C)
    - ln|det Bw|: the same expression, but its rounding errors cancel to first order instead of
    being amplified by cond(Sigma) through Sigma^{-1}. inf if B B^T is singular (rule "none").
    """
    _, W, W_half = _whitening(alloc, Sigma)
    Aw, Bw = W @ alloc.A @ W_half, W @ alloc.B
    logdet = _log_abs_det(Bw)
    if logdet == -math.inf:
        return math.inf
    return float(0.5 * (np.sum(Aw * Aw) + np.sum(Bw * Bw) - alloc.C) - logdet)


def gaussian_mutual_information(alloc: Allocation, Sigma) -> float:
    """0.5 (logdet Sigma - logdet B B^T) = -ln|det Sigma^{-1/2} B|; = I(Z0; Z') for Gaussian z0.

    Unlike `expected_kl` this is first-order sensitive to the eigensolver's backward error in the
    allocation: |MI - R| reaches ~1e-8 nats at cond(Sigma) = 1e9 with w > 1 (E KL stays < 1e-10).
    """
    _, W, _ = _whitening(alloc, Sigma)
    logdet = _log_abs_det(W @ alloc.B)
    return math.inf if logdet == -math.inf else -logdet


def latent_mse_lmmse(alloc: Allocation) -> float:
    """P7: D_latent = tr[T^{-1} diag((1 - keep) s) T^{-T}] = sum_i exp(-2 r_i) s_i ||Tinv[:, i]||^2."""
    return float(np.sum(np.exp(-2.0 * alloc.rates) * alloc.s * np.sum(alloc.Tinv ** 2, axis=0)))


def weighted_distortion(alloc: Allocation) -> float:
    """P7: metric-M distortion sum_i (1 - keep_i) s_i (= sum_i min(s_i, theta) for water-filling)."""
    return float(np.sum(np.exp(-2.0 * alloc.rates) * alloc.s))


def bpp_equivalent(R, tokens: int = 256, pixels: int = 256 * 256) -> float:
    """Theoretical bits per pixel of R nats/token: R * tokens / (pixels * ln 2)."""
    R = float(R)
    if not R >= 0:
        raise ValueError(f"R must be >= 0, got {R!r}")
    return R * tokens / (pixels * math.log(2.0))


def whitened_basis(alloc: Allocation, Sigma) -> np.ndarray:
    """V = Sigma^{-1/2} T^{-1} diag(sqrt(s)): orthogonal, its columns are the eigenvectors of Sigma_w.

    Raises ValueError if max|V^T V - I| > 1e-6, i.e. Sigma does not belong to this allocation.
    """
    _, W, _ = _whitening(alloc, Sigma)
    V = (W @ alloc.Tinv) * np.sqrt(alloc.s)
    err = float(np.max(np.abs(V.T @ V - np.eye(alloc.C))))
    if err > _ORTHO_TOL:
        raise ValueError(f"Sigma does not belong to this allocation: max|V^T V - I| = {err:.3e} > {_ORTHO_TOL:g}")
    return V


def retention_matrix_whitened(alloc: Allocation, Sigma) -> np.ndarray:
    """G_ret = V diag(keep) V^T: whitened covariance of the signal part A (z0 - m) (I for "none")."""
    V = whitened_basis(alloc, Sigma)
    G = (V * alloc.keep) @ V.T
    return 0.5 * (G + G.T)


def retention_overlap(Ga, Gb) -> float:
    """tr(Ga Gb) / sqrt(tr Ga^2 tr Gb^2), in [0, 1] for PSD retention matrices (1 = same retention)."""
    Ga, Gb = sym(Ga, "Ga"), sym(Gb, "Gb")
    if Ga.shape != Gb.shape:
        raise ValueError(f"shape mismatch {Ga.shape} vs {Gb.shape}")
    na, nb = np.linalg.norm(Ga), np.linalg.norm(Gb)
    if na == 0 or nb == 0:
        raise ValueError("retention overlap is undefined for a zero retention matrix (R = 0)")
    return float(np.sum(Ga * Gb) / (na * nb))


def _check_K(K, C: int) -> np.ndarray:
    K = sym(K, "K")
    if K.shape != (C, C):
        raise ValueError(f"K shape {K.shape} does not match allocation C = {C}")
    return K


def semantic_retention(alloc: Allocation, Sigma, K) -> float:
    """tr(G_ret K) / tr(K) in [0, 1]: share of the whitened relevance K that the channel keeps."""
    K = _check_K(K, alloc.C)
    tr = float(np.trace(K))
    if not tr > 0:
        raise ValueError(f"semantic retention needs tr(K) > 0, got {tr!r}")
    return float(np.sum(retention_matrix_whitened(alloc, Sigma) * K) / tr)


def direction_relevance(alloc: Allocation, Sigma, K) -> np.ndarray:
    """k_i = v_i^T K v_i in [0, 1]: relevance of KLT direction i (column v_i of the whitened basis)."""
    K = _check_K(K, alloc.C)
    V = whitened_basis(alloc, Sigma)
    return np.sum(V * (K @ V), axis=0)


def _rate_residual(x: float, R: float) -> float:
    if x == R:  # includes inf == inf (rule "none")
        return 0.0
    if math.isinf(x) or math.isinf(R):
        return math.inf
    return abs(x - R) / max(1.0, R)


def identity_residuals(alloc: Allocation, Sigma) -> dict:
    """Scale-free residuals of the exact identities (~1e-15..1e-9 in float64 for consistent input):

    klt       ||T Sigma T^T - diag(s)||_F / ||Sigma||_F                      (P2)
    variance  ||A Sigma A^T + B B^T - Sigma||_F / ||Sigma||_F                (P4)
    rate      |sum(rates) - R| / max(1, R)                                   (P3)
    kl        |expected_kl - R| / max(1, R)                                  (P5)
    lmmse     ||Sigma^{-1/2} (Sigma A^T Sigma^{-1} - A) Sigma^{1/2}||_F / sqrt(C)
              = ||Aw^T - Aw||_F / sqrt(C),  Aw = Sigma^{-1/2} A Sigma^{1/2}  (P6)

    P6 is measured in Sigma-whitened coordinates (where Aw has norm <= 1 and the identity reads
    "Aw is symmetric"): in raw coordinates float64 rounding of A alone is amplified by
    cond(Sigma) (~1e-7 at cond 1e9). For rule "none" the rate and kl residuals are 0 (inf == inf).
    """
    S, W, W_half = _whitening(alloc, Sigma)
    nS = np.linalg.norm(S)
    Aw = W @ alloc.A @ W_half
    return {
        "klt": float(np.linalg.norm(alloc.T @ S @ alloc.T.T - np.diag(alloc.s)) / nS),
        "variance": float(np.linalg.norm(alloc.A @ S @ alloc.A.T + alloc.B @ alloc.B.T - S) / nS),
        "rate": _rate_residual(nominal_rate(alloc), alloc.R),
        "kl": _rate_residual(expected_kl(alloc, S), alloc.R),
        "lmmse": float(np.linalg.norm(Aw - Aw.T) / math.sqrt(alloc.C)),
    }
