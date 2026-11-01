"""Tests for the architecture additions: rate_allocation, allocation signals,
and the end-to-end synthetic PoC."""
import torch

import demo_rwvae_synthetic as demo
from rwvae import relevance
from rwvae.gate import RelevanceRateGate

C = 16


def test_rate_allocation_is_a_distribution():
    gate = RelevanceRateGate(C)
    alloc = gate.rate_allocation()
    assert alloc.shape == (C,)
    assert torch.all(alloc >= 0)
    assert abs(alloc.sum().item() - 1.0) < 1e-5


def test_variance_allocation_tracks_channel_variance():
    torch.manual_seed(0)
    z = torch.randn(64, C, 4, 4)
    z[:, 0] *= 5.0                                   # channel 0 made high-variance
    v = relevance.variance_allocation(z)
    assert v.shape == (C,)
    assert v.argmax().item() == 0


def test_frequency_allocation_shape_and_range():
    torch.manual_seed(0)
    z = torch.randn(8, C, 16, 16)
    f = relevance.frequency_allocation(z)
    assert f.shape == (C,)
    assert torch.all(f >= 0) and torch.all(f <= 1)


def test_synthetic_poc_relevance_beats_reconstruction():
    out = demo.run(verbose=False)
    assert out["relevance"]["usability"] > out["recon_only"]["usability"] + 0.3
    assert out["relevance"]["signal_rate_frac"] > out["recon_only"]["signal_rate_frac"] + 0.3
