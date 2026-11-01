"""Adapter that wraps LightningDiT's VA-VAE latent with the relevance rate gate.

The gate acts on the latent z = VA_VAE.encode_images(x) of shape [B, 32, 16, 16]
(f16d32). This file is the *integration seam*; it is GPU-only (LightningDiT's
VA_VAE.load() moves the model to CUDA) and is therefore exercised by the training
script's smoke path on plain tensors, not run here.

Note on training: LightningDiT ships VA-VAE *inference* only (encoder/decoder
training "provided later"). To train the gate end-to-end you either (a) fine-tune
on top of released VA-VAE weights, or (b) use REPA-E's trainable VAE (its modular
YAML loss) and insert this gate at the latent. Either way the gate and losses
below are unchanged.
"""
from __future__ import annotations

import os
import sys

import torch

from .gate import RelevanceRateGate


def _add_lightningdit_to_path(lightningdit_root: str) -> None:
    root = os.path.abspath(lightningdit_root)
    if not os.path.isdir(os.path.join(root, "tokenizer")):
        raise FileNotFoundError(
            f"LightningDiT not found at {root!r} (expected a 'tokenizer' subdir). "
            "Download it per README (third_party/LightningDiT)."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


class GatedVAVAE:
    """VA-VAE + relevance rate gate. Encodes to a *gated* latent and decodes it.

    GPU required (VA_VAE uses .cuda()). Construct with the path to the cloned
    LightningDiT repo and a VA-VAE config yaml.
    """

    def __init__(self, lightningdit_root: str, vavae_config: str, channels: int = 32,
                 budget: float | None = None):
        _add_lightningdit_to_path(lightningdit_root)
        from tokenizer.vavae import VA_VAE  # lazy import; needs LightningDiT + CUDA

        self.vae = VA_VAE(vavae_config)
        self.gate = RelevanceRateGate(channels, budget=budget).cuda()

    def encode_gated(self, images: torch.Tensor) -> torch.Tensor:
        """images -> gated latent z' = g ⊙ encode(images)."""
        z = self.vae.encode_images(images)   # [B, C, H, W], on cuda
        return self.gate(z)

    def decode(self, z: torch.Tensor):
        return self.vae.decode_to_images(z)
