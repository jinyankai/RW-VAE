"""Numerical verification of the rate-distortion phase-diagram theory (E3, synthetic).

Runs on CPU in seconds, no GPU. For each theorem it *simulates* the model (or runs
a numeric solver) and checks the result against the closed form in rwvae.theory.
Exits 0 iff every check passes.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows numpy(MKL)+torch OpenMP guard

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwvae import theory  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append(condition)
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    rng = np.random.default_rng(0)
    D, d_s, d_n, rho = 40, 6, 10, 6.0
    n = 200_000

    print("=" * 72)
    print(f"Model: D={D}, d_s={d_s}, d_n={d_n}, rho={rho}  (sigma_s^2=1, sigma_n^2=rho)")
    print("=" * 72)

    # --- Main (a): PCA relative error and task R^2 vs simulation ---------- #
    print("\nMain (a) linear AE / PCA phase diagram (simulation vs closed form):")
    for k in [d_n // 2, d_n, d_n + d_s // 2, d_n + d_s]:
        rel_sim, r2_sim = theory.pca_simulate(n, D, d_s, d_n, rho, k, rng=rng)
        rel_cf = theory.pca_relative_error(k, d_s, d_n, rho)
        r2_cf = theory.pca_task_r2(k, d_s, d_n)
        check(f"k={k:2d} rel_err  sim={rel_sim:.4f} vs cf={rel_cf:.4f}", abs(rel_sim - rel_cf) < 0.02)
        check(f"k={k:2d} task R^2 sim={r2_sim:.4f} vs cf={r2_cf:.4f}", abs(r2_sim - r2_cf) < 0.03)
    # the key pathology: at k=d_n, error -> 0 as rho grows, yet task R^2 stays 0
    big = theory.pca_relative_error(d_n, d_s, d_n, 1e6)
    check("k=d_n: rel_err -> 0 as rho->inf while R^2=0", big < 1e-4 and theory.pca_task_r2(d_n, d_s, d_n) == 0.0,
          f"rel_err(rho=1e6)={big:.2e}")

    # --- Main (b): reverse water-filling solver vs closed form ------------ #
    print("\nMain (b) rate-distortion reverse water-filling (solver vs closed form):")
    lambdas = np.concatenate([np.ones(d_s), np.full(d_n, rho)])
    Rstar = theory.rd_threshold(d_n, rho)
    check(f"R* = (d_n/2)ln(rho) = {Rstar:.4f}", Rstar > 0)
    for frac in [0.25, 0.5, 0.9, 1.0]:
        R = frac * Rstar
        theta, D_i = theory.reverse_water_filling(lambdas, R)
        rel_dist_solver = D_i.sum() / lambdas.sum()
        rel_dist_cf = theory.rd_relative_distortion(R, d_s, d_n, rho)
        signal_rate = 0.5 * np.sum(np.log(lambdas[:d_s] / D_i[:d_s]))  # rate spent on signal
        check(f"R={frac:.2f}R* distortion solver={rel_dist_solver:.4f} vs cf={rel_dist_cf:.4f}",
              abs(rel_dist_solver - rel_dist_cf) < 1e-3)
        check(f"R={frac:.2f}R* signal rate ~ 0 (I(Z;S)=0)", signal_rate < 1e-6, f"signal_rate={signal_rate:.2e}")
    # at R=R*, relative distortion equals (d_s+d_n)/(d_s+d_n*rho)
    rel_at_star = theory.rd_relative_distortion(Rstar, d_s, d_n, rho)
    check("R=R* distortion = (d_s+d_n)/(d_s+d_n*rho)",
          abs(rel_at_star - (d_s + d_n) / (d_s + d_n * rho)) < 1e-9, f"{rel_at_star:.4f}")
    # above R*, signal must start receiving rate
    _, D_above = theory.reverse_water_filling(lambdas, 1.1 * Rstar)
    sig_rate_above = 0.5 * np.sum(np.log(lambdas[:d_s] / D_above[:d_s]))
    check("R>R*: signal starts getting rate (I(Z;S)>0)", sig_rate_above > 1e-6, f"signal_rate={sig_rate_above:.4f}")

    # --- Main (c): weighted PCA threshold rho* = w^2 ---------------------- #
    print("\nMain (c) weighted reconstruction threshold (amplitude convention):")
    w = 2.0
    rho_star = theory.weighted_pca_threshold(w)  # = w^2 = 4
    check(f"weighted PCA threshold rho* = w^2 = {rho_star}", abs(rho_star - w * w) < 1e-9)
    # below threshold (rho < w^2): top-k becomes signal -> task recoverable at small k
    rho_lo = rho_star - 1.0
    _, r2_lo = theory.pca_simulate(n, D, d_s, d_n, rho_lo, k=d_s, weight_w=w, rng=rng)
    # above threshold (rho > w^2): nuisance still dominates -> task lost at k=d_s<=d_n region
    rho_hi = rho_star + 2.0
    _, r2_hi = theory.pca_simulate(n, D, d_s, d_n, rho_hi, k=d_s, weight_w=w, rng=rng)
    check(f"rho<w^2 ({rho_lo}): weighting recovers task (R^2={r2_lo:.3f} high)", r2_lo > 0.9)
    check(f"rho>w^2 ({rho_hi}): task still lost (R^2={r2_hi:.3f} low)", r2_hi < 0.1)
    # weighted RD threshold shifts by ln(rho/w^2)
    Rw = theory.weighted_rd_threshold(d_n, rho, w)
    check("weighted RD threshold = (d_n/2)ln(rho/w^2)",
          abs(Rw - 0.5 * d_n * np.log(rho / w ** 2)) < 1e-9, f"R*_w={Rw:.4f}")

    # --- Bridge 4: rho_emp = (d_n/d_s) * rho ------------------------------ #
    print("\nBridge 4 interventional variance ratio (Monte-Carlo vs identity):")
    emp = theory.rho_emp_simulate(n, D, d_s, d_n, rho, rng=rng)
    cf = theory.rho_emp_closed(d_s, d_n, rho)
    check(f"rho_emp MC={emp:.4f} vs (d_n/d_s)*rho={cf:.4f}", abs(emp - cf) / cf < 0.02)

    print("\n" + "=" * 72)
    ok = all(results)
    print(f"{sum(results)}/{len(results)} checks passed -> {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
