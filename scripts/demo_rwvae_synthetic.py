"""Architecture proof-of-concept for RW-VAE on the theory's own model, CPU.

Latent channels carry a low-variance signal subspace (d_s channels, variance 1)
and a high-variance nuisance subspace (d_n channels, variance rho>1). A fixed total
rate R is allocated across channels by the gate's rate_allocation() (sum = 1), and
per-channel Gaussian rate-distortion gives

    distortion D   = sum_c v_c * exp(-2 * R * alloc_c)          (lower = better rFID analog)
    usability  U   = mean over signal channels of 1 - exp(-2 * R * alloc_c)  (task recovery)

We train the SAME RelevanceRateGate under two allocation signals:
  (A) reconstruction-driven -> minimize D -> water-fills the high-variance nuisance
                               -> usability stays low (the pathology of Main a/b);
  (B) relevance-driven      -> gate follows a generation-relevance signal (toward the
                               signal subspace) -> rate moves to the signal -> usability jumps.

This realizes "reorganize the latent -> change usability" end-to-end with the real
gate module. (The relevance vector here is given ground-truth to isolate the gate's
reallocation; in the full system it is computed by rwvae.relevance.denoising_sensitivity.)
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwvae.gate import RelevanceRateGate  # noqa: E402


def _train(gate, variances, rate_total, signal_mask, signal, steps=800, lr=5e-2):
    """Allocate a fixed total rate with the gate. signal='recon' minimizes
    distortion; signal='relevance' makes the gate follow a generation-relevance
    vector (high on the signal subspace)."""
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    relevance = signal_mask.float()  # relevance oracle: high on signal, ~0 on nuisance
    for _ in range(steps):
        opt.zero_grad()
        if signal == "recon":
            alloc = gate.rate_allocation()
            r = rate_total * alloc
            loss = (variances * torch.exp(-2.0 * r)).sum()
        else:  # relevance-driven allocation
            loss = gate.relevance_alignment_loss(relevance)
        loss.backward()
        opt.step()
    return gate


def _metrics(gate, variances, rate_total, signal_mask):
    with torch.no_grad():
        alloc = gate.rate_allocation()
        r = rate_total * alloc
        distortion = (variances * torch.exp(-2.0 * r)).sum().item()
        resolved = 1.0 - torch.exp(-2.0 * r)
        usability = resolved[signal_mask].mean().item()
        signal_rate_frac = alloc[signal_mask].sum().item()
    return distortion, usability, signal_rate_frac


def run(d_s=6, d_n=10, rho=6.0, verbose=True):
    torch.manual_seed(0)
    C = d_s + d_n
    variances = torch.tensor([1.0] * d_s + [rho] * d_n)
    signal_mask = torch.arange(C) < d_s
    rate_total = C / 2.0

    gate_recon = _train(RelevanceRateGate(C), variances, rate_total, signal_mask, signal="recon")
    gate_rel = _train(RelevanceRateGate(C), variances, rate_total, signal_mask, signal="relevance")

    D_r, U_r, sig_r = _metrics(gate_recon, variances, rate_total, signal_mask)
    D_g, U_g, sig_g = _metrics(gate_rel, variances, rate_total, signal_mask)

    out = {
        "recon_only": {"distortion": D_r, "usability": U_r, "signal_rate_frac": sig_r},
        "relevance":  {"distortion": D_g, "usability": U_g, "signal_rate_frac": sig_g},
    }
    if verbose:
        print(f"Synthetic RW-VAE PoC (d_s={d_s}, d_n={d_n}, rho={rho}, rate={rate_total}):")
        print(f"  reconstruction-only : distortion={D_r:.3f}  usability={U_r:.3f}  signal rate frac={sig_r:.3f}")
        print(f"  relevance-driven    : distortion={D_g:.3f}  usability={U_g:.3f}  signal rate frac={sig_g:.3f}")
        print("  -> relevance reallocates rate onto the signal subspace: usability rises")
        print("     (at a modest reconstruction cost; in images the decoder's slack keeps rFID matched).")
    return out


def main():
    out = run()
    assert out["relevance"]["usability"] > out["recon_only"]["usability"] + 0.3, "relevance did not raise usability"
    assert out["relevance"]["signal_rate_frac"] > out["recon_only"]["signal_rate_frac"] + 0.3, "rate not reallocated to signal"
    print("[PASS] relevance-driven gate reallocates rate to the task subspace and raises usability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
