"""RW-VAE: the relevance rate gate (the thesis's new architecture).

A per-channel diagonal gate on the VAE latent z of shape [B, C, H, W]:

    z' = g ⊙ z,    g = softplus(raw) >= 0   (one weight per channel)

This is the diagonal W = diag(g) of Main(c). Two pieces make it a *reallocation*
(not an addition): (i) `budget_penalty` keeps the total gate mass fixed, so the
gate cannot simply add capacity; (ii) `relevance_alignment_loss` pulls the gate
distribution toward a per-channel *generation-relevance* signal, moving capacity
from the variance ordering toward the relevance ordering. Theory (c) predicts the
weight w needed to retain a relevance subspace at a given rho and rate; the gate
learns it.

Minimal by design: one parameter vector of length C, three small methods. The
relevance signal itself lives in rwvae.relevance.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    # x such that softplus(x) = y, for initializing the gate at a chosen value
    return math.log(math.expm1(y))


class RelevanceRateGate(nn.Module):
    """Per-channel latent rate gate with a fixed total-rate budget.

    Args:
        channels: number of latent channels C (32 for VA-VAE f16d32).
        budget: target for sum(g); defaults to channels * init (i.e. the gate
            starts at the identity and may only *reallocate* mass).
        init: initial per-channel gate value.
    """

    def __init__(self, channels: int, budget: float | None = None, init: float = 1.0):
        super().__init__()
        self.channels = channels
        self.raw = nn.Parameter(torch.full((channels,), float(_inv_softplus(init))))
        self.register_buffer(
            "budget", torch.tensor(float(budget if budget is not None else channels * init))
        )

    def gates(self) -> torch.Tensor:
        """Current non-negative per-channel gate vector g, shape [C]."""
        return F.softplus(self.raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the gate to a latent [B, C, H, W] (or [B, C])."""
        g = self.gates()
        shape = [1, self.channels] + [1] * (z.dim() - 2)
        return z * g.view(*shape)

    def rate_allocation(self) -> torch.Tensor:
        """Fraction of the (fixed) rate budget assigned to each channel, g / sum(g),
        shape [C]. This is the quantity the relevance signal reshapes; comparing it
        across allocation signals is the core 'latent organization' readout."""
        g = self.gates()
        return g / (g.sum() + 1e-8)

    def budget_penalty(self) -> torch.Tensor:
        """Squared deviation of total gate mass from the fixed budget.

        Keeping this small makes the gate a reallocation of a fixed rate rather
        than an addition of capacity (so reconstruction / rFID stays matched)."""
        return (self.gates().sum() - self.budget) ** 2

    def relevance_alignment_loss(self, relevance: torch.Tensor) -> torch.Tensor:
        """Pull the (normalized) gate distribution toward a per-channel relevance
        vector. `relevance` is a non-negative tensor of shape [C]."""
        if relevance.shape != (self.channels,):
            raise ValueError(f"relevance must have shape ({self.channels},), got {tuple(relevance.shape)}")
        g = self.gates()
        gn = g / (g.sum() + 1e-8)
        rn = relevance / (relevance.sum() + 1e-8)
        return F.mse_loss(gn, rn)
