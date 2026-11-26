"""PatchGAN pieces for the v3 decoder-only GAN fine-tune (RWVAE_V3_ALLOCATION_PLAN.md §2.4).

Ported from taming-transformers (Esser, Rombach, Ommer; MIT licence), the dependency of VA-VAE's
LPIPSWithDiscriminator (third_party/LightningDiT/vavae/ldm/modules/losses/contperceptual.py), which is not
installed here: NLayerDiscriminator (pix2pix PatchGAN, BatchNorm, no ActNorm), weights_init, hinge_d_loss.
adaptive_gan_weight follows LPIPSWithDiscriminator.calculate_adaptive_weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator: [B, input_nc, H, W] -> patch logits [B, 1, h, w] (taming defaults: ndf 64, 3 layers)."""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev, nf_mult = nf_mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                    nn.BatchNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*seq)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def weights_init(m: nn.Module) -> None:
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def adaptive_gan_weight(rec_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor,
                        disc_weight: float) -> torch.Tensor:
    """disc_weight * ||d rec / d last_layer|| / (||d g / d last_layer|| + 1e-4), clamped to [0, 1e4], detached:
    the adversarial term gets disc_weight times the gradient magnitude of the reconstruction term at the
    decoder's output layer, whatever the scale of either loss."""
    rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    w = torch.norm(rec_grad) / (torch.norm(g_grad) + 1e-4)
    return torch.clamp(w, 0.0, 1e4).detach() * float(disc_weight)
