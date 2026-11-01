"""Tests for the CouplingFlow and the E2 bijection intervention."""
import torch

import eval_e2_bijection as e2
from rwvae.bijection import CouplingFlow


def test_coupling_flow_is_invertible():
    torch.manual_seed(0)
    for dim in (8, 15, 16):                     # even and odd dims
        flow = CouplingFlow(dim, n_layers=6)
        z = torch.randn(32, dim)
        assert (flow.inverse(flow(z)) - z).abs().max().item() < 1e-4


def test_e2_reconstruction_invariant_usability_drops_info_kept():
    out = e2.run(adversarial_steps=150, verbose=False)
    assert out["invertibility_error"] < 1e-4           # reconstruction invariant
    assert out["recon_invariance_gap"] < 1e-4
    assert out["linear_r2_before"] > 0.9               # originally linearly decodable
    assert out["linear_r2_adversarial"] < 0.3          # linear usability collapses
    assert out["mlp_r2_adversarial"] > 0.5             # information preserved
