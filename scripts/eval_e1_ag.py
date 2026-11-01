"""E1 -- encoder survey + Accessibility-Gap (AG) decomposition, CPU.

The real pipeline takes cached features for a set of encoders and computes, for
each, the exact telescoping identity (PROPOSAL.md Pillar 3)

    R_V(z) - R_V(phi(x)) = [R_V(phi(x_hat)) - R_V(phi(x))] + [R_V(z) - R_V(phi(x_hat))]
         total                 reconstruction info loss              accessibility gap

with R_V the optimal bounded (ridge) linear-probe risk. `ag_report` is that
computation; `demo` manufactures a case with small reconstruction info loss but a
large accessibility gap (z is an invertible but linearly-scrambled view of phi(x_hat)),
mirroring the thesis claim and running fully on CPU.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwvae import metrics  # noqa: E402
from rwvae.bijection import CouplingFlow  # noqa: E402


def ag_report(z, phi_x, phi_xhat, targets, ridge=1e-3):
    """Accessibility-Gap report for one encoder.

    Args are numpy arrays: z (latent features), phi_x (reference features of the
    original), phi_xhat (reference features of the reconstruction), targets.
    Returns the three risks and the decomposition (recon_info_loss + AG == total).
    """
    r_z = metrics.linear_probe_risk(z, targets, ridge)
    r_phi_x = metrics.linear_probe_risk(phi_x, targets, ridge)
    r_phi_xhat = metrics.linear_probe_risk(phi_xhat, targets, ridge)
    decomp = metrics.accessibility_gap(r_z, r_phi_x, r_phi_xhat)
    return {"risk_z": r_z, "risk_phi_x": r_phi_x, "risk_phi_xhat": r_phi_xhat, **decomp}


def demo(verbose=True, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n, d = 3000, 16

    phi_x = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d, 1)).astype(np.float32)
    targets = phi_x @ w + 0.05 * rng.standard_normal((n, 1)).astype(np.float32)

    # reconstruction loses a little information: phi(x_hat) = phi(x) + small noise
    phi_xhat = phi_x + 0.1 * rng.standard_normal((n, d)).astype(np.float32)

    # the latent z is an invertible, non-linear re-encoding of phi(x_hat): it holds
    # the SAME information (bijection) but is harder to read linearly -> AG > 0
    flow = CouplingFlow(d, n_layers=6)
    with torch.no_grad():
        z = flow(torch.from_numpy(phi_xhat)).numpy()

    rep = ag_report(z, phi_x, phi_xhat, targets)
    if verbose:
        print("E1 Accessibility-Gap decomposition (demo):")
        print(f"  R_V(phi(x))      = {rep['risk_phi_x']:.4f}")
        print(f"  R_V(phi(x_hat))  = {rep['risk_phi_xhat']:.4f}")
        print(f"  R_V(z)           = {rep['risk_z']:.4f}")
        print(f"  total gap        = {rep['total']:.4f}")
        print(f"   = recon info loss {rep['recon_info_loss']:.4f}  +  accessibility gap {rep['accessibility_gap']:.4f}")
    return rep


def main():
    rep = demo()
    # exact telescoping identity
    assert abs(rep["recon_info_loss"] + rep["accessibility_gap"] - rep["total"]) < 1e-9
    # the manufactured case: small recon info loss, large accessibility gap
    assert rep["accessibility_gap"] > 5 * abs(rep["recon_info_loss"])
    print("[PASS] AG identity holds exactly; accessibility gap dominates the small recon-info-loss term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
