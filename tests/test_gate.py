"""Tests for the RelevanceRateGate (the new architecture)."""
import torch

from rwvae.gate import RelevanceRateGate

C = 32


def test_forward_shape_and_identity_init():
    gate = RelevanceRateGate(C, init=1.0)
    z = torch.randn(4, C, 8, 8)
    out = gate(z)
    assert out.shape == z.shape
    # init at identity -> gates ~ 1 -> output ~ input
    assert torch.allclose(out, z, atol=1e-5)


def test_budget_penalty_zero_at_init():
    gate = RelevanceRateGate(C, init=1.0)  # budget defaults to C*init = C
    assert gate.budget_penalty().item() < 1e-6


def test_relevance_alignment_shape_check():
    gate = RelevanceRateGate(C)
    good = torch.ones(C)
    assert gate.relevance_alignment_loss(good).item() >= 0.0
    try:
        gate.relevance_alignment_loss(torch.ones(C + 1))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_backward_runs():
    gate = RelevanceRateGate(C)
    z = torch.randn(2, C, 4, 4)
    loss = gate(z).pow(2).mean() + gate.budget_penalty()
    loss.backward()
    assert gate.raw.grad is not None and torch.isfinite(gate.raw.grad).all()


def test_reallocation_moves_mass_toward_relevance():
    """Optimizing the gate toward a peaked relevance vector, under the budget
    constraint, should raise the high-relevance channel's gate above the rest."""
    torch.manual_seed(0)
    gate = RelevanceRateGate(C, init=1.0)
    relevance = torch.full((C,), 0.1)
    relevance[0] = 10.0  # channel 0 is highly generation-relevant
    opt = torch.optim.Adam(gate.parameters(), lr=5e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = gate.relevance_alignment_loss(relevance) + 1e-2 * gate.budget_penalty()
        loss.backward()
        opt.step()
    g = gate.gates().detach()
    assert g[0] > g[1:].max()                      # mass moved to channel 0
    assert abs(g.sum().item() - gate.budget.item()) < 0.5  # budget roughly preserved
