"""RW-VAE: theory-anchored relevance-reweighting autoencoder for image generation.

v1 (L0 legacy, behaviour unchanged):
  theory      -- closed forms + simulators for the rate-distortion phase diagram (numpy)
  metrics     -- Accessibility-Gap decomposition, effective rank / RankMe, power spectrum (numpy)
  gate        -- RelevanceRateGate: the per-channel latent rate gate (torch)
  relevance   -- per-channel generation-relevance signals (torch); its v2 section adds the R2
                 explained covariance, CCA spectrum and patch-grid alignment scan (numpy, P1)
  integration -- GatedVAVAE adapter onto LightningDiT's VA-VAE; its v2 section is the only seam
                 to LightningDiT, timm DINOv2, lpips and pytorch-fid (torch)

v2.1 (plan RWVAE_V2_CORE_PLAN.md; module contracts in docs/ARCHITECTURE_V2.md):
  runtime         -- configs, seeds, logging, run directory + registry, file hashes
  allocation      -- weighted KLT, reverse water-filling, the B0-B3 rules, A/B, closed forms (numpy f64, P2-P7)
  stats           -- grouped streaming float64 token moments, LatentStats, stability tools (torch)
  channel         -- RWChannel z' = m + A (mu - m) + B eps with reproducible per-sample noise (torch, P4/P6)
  decoder_adapter -- LMMSE prelayer fused into post_quant_conv, decoder fine-tuning (torch, P6)
  datasets        -- split manifest, image datasets, mu cache, channel / pair datasets (torch)
  evaluation      -- PSNR, LPIPS, FID, linear probe, latent DCT spectra, bootstrap CIs (torch)
  generation      -- LightningDiT training loop, EMA, class-balanced sampling (torch)

theory and metrics are numpy-only and imported eagerly. Every other module is imported lazily on
first attribute access (PEP 562), so `import rwvae` never pulls in torch.
"""
import importlib

from . import metrics, theory

_LAZY_MODULES = ("gate", "relevance", "integration", "runtime", "allocation", "stats", "channel",
                 "decoder_adapter", "datasets", "evaluation", "generation")

__all__ = ["theory", "metrics", *_LAZY_MODULES, "RelevanceRateGate"]
__version__ = "0.2.0"


def __getattr__(name):  # PEP 562 lazy submodule / symbol loading
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    if name == "RelevanceRateGate":
        return importlib.import_module(".gate", __name__).RelevanceRateGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
