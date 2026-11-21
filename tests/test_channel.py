"""Tests for rwvae.channel (plan v2.1 P4/P6).

The channel matrices are built here with an *independent* numpy construction of the
weighted KLT + reverse water-filling (bisection), so these tests also cross-check the
formulas that rwvae.allocation implements in closed form.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rwvae.channel import RWChannel, noise_seed, sample_noise, splitmix64

C = 8


def _random_spd(C, rng, cond=50.0):
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    eig = np.logspace(0, np.log10(cond), C)
    return (Q * eig) @ Q.T


def _sym_pow(M, p):
    e, V = np.linalg.eigh(M)
    return (V * e ** p) @ V.T


def _reference_channel(Sigma, Sigma_expl, w, R):
    """Independent construction: weighted KLT + bisection water-filling."""
    K = _sym_pow(Sigma, -0.5) @ Sigma_expl @ _sym_pow(Sigma, -0.5)
    M = np.eye(len(Sigma)) + (w ** 2 - 1.0) * (K + K.T) / 2
    Mh = _sym_pow(M, 0.5)
    s, Q = np.linalg.eigh(Mh @ Sigma @ Mh)
    s, Q = s[::-1], Q[:, ::-1]
    T, Tinv = Q.T @ Mh, _sym_pow(M, -0.5) @ Q
    lo, hi = 1e-300, s.max()
    for _ in range(400):
        mid = np.sqrt(lo * hi)
        rate = 0.5 * np.sum(np.log(np.maximum(s / mid, 1.0)))
        lo, hi = (mid, hi) if rate > R else (lo, mid)
    theta = np.sqrt(lo * hi)
    keep = np.maximum(1.0 - theta / s, 0.0)
    A = Tinv @ np.diag(np.sqrt(keep)) @ T
    B = Tinv @ np.diag(np.sqrt((1.0 - keep) * s))
    return A, B, keep


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    Sigma = _random_spd(C, rng)
    L = np.linalg.cholesky(Sigma)
    Sigma_expl = 0.6 * L[:, :3] @ L[:, :3].T          # 0 <= Sigma_expl <= Sigma
    m = rng.standard_normal(C)
    A, B, keep = _reference_channel(Sigma, Sigma_expl, w=2.0, R=6.0)
    return SimpleNamespace(rng=rng, Sigma=Sigma, L=L, m=m, A=A, B=B, keep=keep)


def test_reference_channel_sanity(setup):
    # the reference construction itself satisfies P4: A Sigma A^T + B B^T = Sigma
    S, A, B = setup.Sigma, setup.A, setup.B
    assert np.allclose(A @ S @ A.T + B @ B.T, S, atol=1e-10 * np.abs(S).max())
    assert abs(np.sum(-0.5 * np.log(1.0 - setup.keep)) - 6.0) < 1e-9
    assert setup.keep.min() == 0.0  # at R=6 nats some directions are dropped (soft pruning)


def test_forward_matches_matrix_formula(setup):
    ch = RWChannel(setup.m, setup.A, setup.B, rate=6.0, dtype=torch.float64)
    mu = torch.randn(3, C, 4, 5, dtype=torch.float64)
    eps = torch.randn(3, C, 4, 5, dtype=torch.float64)
    out = ch(mu, noise=eps).numpy()
    mu_n, eps_n, m = mu.numpy(), eps.numpy(), setup.m[None, :, None, None]
    ref = m + np.einsum("ij,bjhw->bihw", setup.A, mu_n - m) + np.einsum("ij,bjhw->bihw", setup.B, eps_n)
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("source", ["gaussian", "uniform"])
def test_per_token_covariance_preserved(setup, source):
    # P4 holds for any source with matching first two moments (not only Gaussian)
    n = 200_000
    rng = setup.rng
    u = rng.standard_normal((n, C)) if source == "gaussian" else rng.uniform(-np.sqrt(3), np.sqrt(3), (n, C))
    mu = setup.m + u @ setup.L.T
    ch = RWChannel(setup.m, setup.A, setup.B, rate=6.0, dtype=torch.float64)
    z = ch(torch.from_numpy(mu)[:, :, None, None], generator=torch.Generator().manual_seed(1))
    z = z[:, :, 0, 0].numpy()
    emp = np.cov(z, rowvar=False)
    assert np.linalg.norm(emp - setup.Sigma) / np.linalg.norm(setup.Sigma) < 0.02
    assert np.allclose(z.mean(0), setup.m, atol=0.05 * np.sqrt(np.diag(setup.Sigma)).max())


def test_lmmse_matches_empirical_regression(setup):
    # P6: Cov(mu, z') Cov(z')^{-1} = A
    n = 200_000
    mu = setup.m + setup.rng.standard_normal((n, C)) @ setup.L.T
    ch = RWChannel(setup.m, setup.A, setup.B, rate=6.0, dtype=torch.float64)
    z = ch(torch.from_numpy(mu)[:, :, None, None], generator=torch.Generator().manual_seed(2))[:, :, 0, 0].numpy()
    zc, mc = z - z.mean(0), mu - mu.mean(0)
    coef = (mc.T @ zc) @ np.linalg.inv(zc.T @ zc)
    assert np.linalg.norm(coef - setup.A) / np.linalg.norm(setup.A) < 0.02
    # and RWChannel.lmmse is exactly m + A (z' - m)
    zt = torch.from_numpy(z[:5])[:, :, None, None]
    ref = setup.m + (z[:5] - setup.m) @ setup.A.T
    assert np.allclose(ch.lmmse(zt)[:, :, 0, 0].numpy(), ref, atol=1e-12)


def test_identity_channel_is_exact():
    ch = RWChannel.identity(C, m=np.arange(C, dtype=float))
    mu = torch.randn(2, C, 3, 3)
    assert ch.is_identity and float(ch.rate) == float("inf")
    assert torch.equal(ch(mu), mu)
    assert torch.equal(ch(mu, noise=torch.randn_like(mu)), mu)


def test_noise_seed_properties():
    seeds = {noise_seed(7, e, i) for e in range(5) for i in range(2000)}
    assert len(seeds) == 5 * 2000                      # no collisions on a realistic grid
    assert all(0 <= s < 2 ** 63 for s in seeds)
    assert noise_seed(7, 3, 11) == noise_seed(7, 3, 11)
    assert noise_seed(7, 3, 11) != noise_seed(8, 3, 11)
    assert splitmix64(0) == 0xE220A8397B1DCDAF        # reference splitmix64 output for state 0
    with pytest.raises(ValueError):
        noise_seed(-1, 0, 0)


def test_sample_noise_independent_of_batch_composition():
    seeds = [noise_seed(1, 0, i) for i in range(4)]
    full = sample_noise((C, 4, 4), seeds)
    alone = sample_noise((C, 4, 4), [seeds[2]])
    assert torch.equal(full[2], alone[0])
    assert full.shape == (4, C, 4, 4) and full.dtype == torch.float32
    assert not torch.equal(full[0], full[1])


def test_shape_validation(setup):
    ch = RWChannel(setup.m, setup.A, setup.B)
    with pytest.raises(ValueError):
        ch(torch.randn(2, C + 1, 4, 4))
    with pytest.raises(ValueError):
        ch(torch.randn(2, C, 4, 4), noise=torch.randn(2, C, 4, 3))
    with pytest.raises(ValueError):
        RWChannel(np.zeros(C), np.eye(C), np.zeros((C, C + 1)))


def test_half_precision_input_is_upcast(setup):
    ch = RWChannel(setup.m, setup.A, setup.B)
    out = ch(torch.randn(2, C, 2, 2, dtype=torch.bfloat16), generator=torch.Generator().manual_seed(0))
    assert out.dtype == torch.float32


def test_cross_token_covariance_is_not_preserved(setup):
    """Documented limitation (plan v2.1 P4): the white channel noise changes the
    cross-token covariance from Cov(mu_p, mu_q) to A Cov(mu_p, mu_q) A^T."""
    n, rho = 200_000, 0.8
    rng, L, m = setup.rng, setup.L, setup.m
    p = rng.standard_normal((n, C)) @ L.T
    q = rho * p + np.sqrt(1 - rho ** 2) * (rng.standard_normal((n, C)) @ L.T)
    mu = np.stack([p, q], axis=-1)[:, :, None, :] + m[None, :, None, None]   # [n, C, 1, 2]
    ch = RWChannel(m, setup.A, setup.B, rate=6.0, dtype=torch.float64)
    z = ch(torch.from_numpy(mu), generator=torch.Generator().manual_seed(3)).numpy()
    zp, zq = z[:, :, 0, 0] - m, z[:, :, 0, 1] - m
    cross_emp = zp.T @ zq / n
    cross_in = rho * setup.Sigma
    cross_pred = setup.A @ cross_in @ setup.A.T
    scale = np.linalg.norm(cross_in)
    assert np.linalg.norm(cross_emp - cross_pred) / scale < 0.02
    assert np.linalg.norm(cross_pred - cross_in) / scale > 0.05  # a real, measurable change


def test_from_allocation_duck_typed(setup):
    alloc = SimpleNamespace(rule="r2", R=6.0, w=2.0, name="r2_R6_w2", C=C, m=setup.m, A=setup.A, B=setup.B)
    ch = RWChannel.from_allocation(alloc, dtype=torch.float64)
    assert np.allclose(ch.A.numpy(), setup.A) and float(ch.rate) == 6.0 and ch.meta["name"] == "r2_R6_w2"
    none = SimpleNamespace(rule="none", R=float("inf"), w=1.0, name="none", C=C, m=setup.m,
                           A=np.eye(C), B=np.zeros((C, C)))
    assert RWChannel.from_allocation(none).is_identity
