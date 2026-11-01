"""E2 -- bijection intervention (verifies Lemma 2), CPU.

Take a latent z whose target is linearly decodable. Apply an invertible
CouplingFlow f. Then:
  (i)  reconstruction is invariant: dec(f^{-1}(f(z))) == dec(z) exactly, because f
       is a bijection (we check max|f^{-1}(f(z)) - z|);
  (ii) a non-linear f can be refined to push the *optimal linear probe* R^2 toward
       chance, while
  (iii) the information is preserved (a small MLP probe still recovers the target).

This is the controlled intervention the thesis uses to show reconstruction does not
determine latent usability.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwvae.bijection import CouplingFlow  # noqa: E402


def _linear_r2(z, t, ridge=1e-2):
    """Optimal (ridge) linear-probe R^2 predicting t from z. Differentiable."""
    zc = z - z.mean(0)
    tc = t - t.mean(0)
    A = zc.T @ zc + ridge * torch.eye(z.shape[1])
    beta = torch.linalg.solve(A, zc.T @ tc)
    resid = tc - zc @ beta
    return (1 - (resid ** 2).sum() / ((tc ** 2).sum() + 1e-8)).item()


def _mlp_probe_r2(z, t, steps=300, hidden=64):
    """R^2 of a small MLP probe trained on (z, t): measures recoverable info."""
    net = torch.nn.Sequential(
        torch.nn.Linear(z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, t.shape[1])
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    tc = t - t.mean(0)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((net(z) - t) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        resid = t - net(z)
        return (1 - (resid ** 2).sum() / ((tc ** 2).sum() + 1e-8)).item()


def run(seed=0, n=2000, dim=16, adversarial_steps=400, verbose=True):
    torch.manual_seed(seed)
    z = torch.randn(n, dim)
    w0 = torch.randn(dim, 1)
    t = z @ w0 + 0.05 * torch.randn(n, 1)       # target linearly decodable from z
    dec = torch.randn(dim, dim)                  # a fixed linear "decoder" (stand-in)

    flow = CouplingFlow(dim, n_layers=6, hidden=64)

    # (i) reconstruction invariance via exact invertibility
    with torch.no_grad():
        inv_err = (flow.inverse(flow(z)) - z).abs().max().item()
        recon_gap = (dec.T @ (flow.inverse(flow(z)) - z).T).abs().max().item()

    r2_lin_before = _linear_r2(z, t)
    with torch.no_grad():
        z_rand = flow(z)
    r2_lin_random = _linear_r2(z_rand, t)

    # (ii) adversarially refine f to minimize the optimal linear probe R^2
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    for _ in range(adversarial_steps):
        opt.zero_grad()
        zf = flow(z)
        # differentiable optimal-linear-probe R^2 (same formula, as a loss)
        zc, tc = zf - zf.mean(0), t - t.mean(0)
        A = zc.T @ zc + 1e-2 * torch.eye(dim)
        beta = torch.linalg.solve(A, zc.T @ tc)
        r2 = 1 - ((tc - zc @ beta) ** 2).sum() / ((tc ** 2).sum() + 1e-8)
        r2.backward()
        opt.step()

    with torch.no_grad():
        z_adv = flow(z)
        inv_err_adv = (flow.inverse(z_adv) - z).abs().max().item()
    r2_lin_adv = _linear_r2(z_adv, t)
    r2_mlp_adv = _mlp_probe_r2(z_adv.detach(), t)

    out = {
        "invertibility_error": inv_err,
        "invertibility_error_adv": inv_err_adv,
        "recon_invariance_gap": recon_gap,
        "linear_r2_before": r2_lin_before,
        "linear_r2_random_flow": r2_lin_random,
        "linear_r2_adversarial": r2_lin_adv,
        "mlp_r2_adversarial": r2_mlp_adv,
    }
    if verbose:
        print("E2 bijection intervention (Lemma 2):")
        print(f"  invertibility error (max|f^-1(f(z))-z|)   : {inv_err:.2e}  (adv {inv_err_adv:.2e})")
        print(f"  reconstruction invariance gap             : {recon_gap:.2e}  (== 0 by bijection)")
        print(f"  linear-probe R^2  original z              : {r2_lin_before:.3f}")
        print(f"  linear-probe R^2  after adversarial f     : {r2_lin_adv:.3f}  (pushed toward 0)")
        print(f"  MLP-probe   R^2  after adversarial f      : {r2_mlp_adv:.3f}  (info preserved)")
    return out


def main():
    out = run()
    assert out["invertibility_error"] < 1e-4 and out["invertibility_error_adv"] < 1e-4, "flow not invertible"
    assert out["recon_invariance_gap"] < 1e-4, "reconstruction changed (should be invariant)"
    assert out["linear_r2_adversarial"] < out["linear_r2_before"] - 0.3, "linear probe did not drop"
    assert out["mlp_r2_adversarial"] > 0.6, "information was lost (should be preserved)"
    print("[PASS] reconstruction invariant; linear usability collapses; information preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
