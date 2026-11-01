"""Invertible coupling flow for the bijection intervention (experiment E2).

Lemma 2: for a bijection f, the model (f∘enc, dec∘f⁻¹) has identical
reconstruction (dec∘f⁻¹∘f∘enc = dec∘enc) while a linear probe's risk can be
pushed toward chance. A CouplingFlow is a smooth, exactly-invertible f; applying
it to a frozen latent lets us change linear decodability at *strictly unchanged*
reconstruction. The flow must be non-linear (a linear/orthogonal map is absorbed
by a linear probe), which is why we use affine coupling layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    """One RealNVP-style affine coupling layer on a vector [B, dim].

    Half the coordinates condition an MLP that scales+shifts the other half;
    `flip` swaps which half is transformed so stacked layers mix all coords.
    tanh-bounded log-scale keeps the map well-conditioned and invertible.
    """

    def __init__(self, dim: int, hidden: int = 64, flip: bool = False):
        super().__init__()
        self.flip = flip
        self.d0 = dim // 2
        self.d1 = dim - self.d0
        cond, trans = (self.d1, self.d0) if flip else (self.d0, self.d1)
        self.trans = trans
        self.net = nn.Sequential(
            nn.Linear(cond, hidden), nn.Tanh(), nn.Linear(hidden, 2 * trans)
        )

    def _split(self, x):
        a, b = x[:, : self.d0], x[:, self.d0:]
        return (b, a) if self.flip else (a, b)  # (conditioner, transformed)

    def _merge(self, cond, trans):
        a, b = (trans, cond) if self.flip else (cond, trans)
        return torch.cat([a, b], dim=1)

    def forward(self, x):
        c, t = self._split(x)
        s, sh = self.net(c).chunk(2, dim=1)
        t = t * torch.exp(torch.tanh(s)) + sh
        return self._merge(c, t)

    def inverse(self, y):
        c, t = self._split(y)
        s, sh = self.net(c).chunk(2, dim=1)
        t = (t - sh) * torch.exp(-torch.tanh(s))
        return self._merge(c, t)


class CouplingFlow(nn.Module):
    """Stack of alternating affine coupling layers: an exactly invertible map
    on vectors [B, dim]."""

    def __init__(self, dim: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            AffineCoupling(dim, hidden, flip=(i % 2 == 1)) for i in range(n_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y
