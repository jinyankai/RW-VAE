"""RW-VAE: theory-anchored relevance-reweighting autoencoder for image generation.

Package layout:
  theory      -- closed forms + simulators for the rate-distortion phase diagram (numpy)
  metrics     -- Accessibility-Gap decomposition, effective rank / RankMe, power spectrum (numpy)
  gate        -- RelevanceRateGate: the per-channel latent rate gate (torch)
  relevance   -- per-channel generation-relevance signals (torch)
  integration -- GatedVAVAE adapter onto LightningDiT's VA-VAE (torch, GPU-only)

theory and metrics are numpy-only and imported eagerly. The torch-dependent
modules are loaded lazily, so `import rwvae.theory` never pulls in torch.
"""
import importlib

from . import metrics, theory

__all__ = ["theory", "metrics", "relevance", "gate", "integration", "RelevanceRateGate"]
__version__ = "0.1.0"


def __getattr__(name):  # PEP 562 lazy submodule / symbol loading
    if name in ("gate", "relevance", "integration"):
        return importlib.import_module(f".{name}", __name__)
    if name == "RelevanceRateGate":
        return importlib.import_module(".gate", __name__).RelevanceRateGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
