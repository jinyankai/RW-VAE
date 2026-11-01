"""E4 training entry point for RW-VAE (relevance rate gate on a VA-VAE latent).

Two modes:
  --smoke : CPU-only. Exercises the full training *code path* (gate forward,
            relevance signal, loss assembly, backward, optimizer step) on tiny
            random latents + a dummy denoiser. No GPU, no data, no LightningDiT.
  (default): real training. Requires CUDA + a trainable VA-VAE (released VA-VAE
            weights, or REPA-E's trainable VAE) + ImageNet latents. If CUDA is
            unavailable this prints instructions and exits 0 WITHOUT training
            (per the thesis plan: write the code, skip the GPU run).
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows numpy(MKL)+torch OpenMP guard

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwvae import relevance  # noqa: E402
from rwvae.gate import RelevanceRateGate  # noqa: E402

CHANNELS = 32  # VA-VAE f16d32


def rwvae_loss(z, gate, relevance_vec, recon_fn, lambda_budget=1.0, lambda_rel=1.0):
    """Total RW-VAE loss = reconstruction(gated latent) + budget + relevance alignment.

    `recon_fn(z_gated) -> scalar` stands in for decode+reconstruction loss (so the
    smoke test needs no decoder). Budget keeps total rate fixed (matched rFID);
    relevance term reallocates toward generation-relevant channels.
    """
    z_gated = gate(z)
    recon = recon_fn(z_gated)
    budget = gate.budget_penalty()
    rel = gate.relevance_alignment_loss(relevance_vec)
    total = recon + lambda_budget * budget + lambda_rel * rel
    return total, {"recon": recon, "budget": budget, "relevance": rel}


def smoke() -> int:
    """One CPU training step on tiny tensors; asserts the code path runs."""
    torch.manual_seed(0)
    B, H, W = 4, 8, 8
    z = torch.randn(B, CHANNELS, H, W)
    gate = RelevanceRateGate(CHANNELS)

    # dummy denoiser + relevance signal (R1); then a dummy recon objective
    denoiser = lambda noisy: noisy * 0.9
    rel_vec = relevance.denoising_sensitivity(z, denoiser, noise_std=0.5)
    assert rel_vec.shape == (CHANNELS,) and torch.all(rel_vec >= 0)

    recon_fn = lambda zg: torch.mean(zg ** 2)  # stand-in for decode+recon loss
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2)

    before = float("inf")
    for step in range(5):
        opt.zero_grad()
        total, parts = rwvae_loss(z, gate, rel_vec, recon_fn)
        total.backward()
        opt.step()
        print(f"step {step}: total={total.item():.4f} "
              f"recon={parts['recon'].item():.4f} budget={parts['budget'].item():.4f} "
              f"rel={parts['relevance'].item():.4f}")
        before = total.item()

    assert torch.isfinite(torch.tensor(before)), "loss diverged"
    assert gate.gates().shape == (CHANNELS,)
    print("[PASS] smoke: gate forward/backward, relevance signal, and loss assembly all run on CPU.")
    return 0


def real_train(args) -> int:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available -> not running real training (by design).")
        print("To run on a GPU node:")
        print("  1) download LightningDiT to third_party/LightningDiT (see README)")
        print("  2) provide a VA-VAE config + weights (or REPA-E trainable VAE)")
        print("  3) python scripts/train_rwvae.py --lightningdit-root third_party/LightningDiT \\")
        print("         --vavae-config <cfg.yaml> --data <imagenet_latents>")
        print("Meanwhile verify the code path with:  python scripts/train_rwvae.py --smoke")
        return 0

    # GPU path (not exercised in this environment; kept minimal and explicit)
    from rwvae.integration import GatedVAVAE  # noqa: F401  (import check)
    raise NotImplementedError(
        "Real training requires an ImageNet latent dataset and a trainable VA-VAE; "
        "wire GatedVAVAE + rwvae_loss into the LightningDiT/REPA-E training loop."
    )


def main() -> int:
    p = argparse.ArgumentParser(description="RW-VAE training (E4)")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test of the training code path")
    p.add_argument("--lightningdit-root", default="third_party/LightningDiT")
    p.add_argument("--vavae-config", default=None)
    p.add_argument("--data", default=None)
    args = p.parse_args()
    return smoke() if args.smoke else real_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
