"""Tests for rwvae.stats (plan v2.1 Sec. 1.2-1.3, 3.2-3.3; ARCHITECTURE_V2.md Sec. 3.4).

References are independent numpy two-pass float64 computations. Exact identities are
checked at float64 relative error <= 1e-10 (the contract asks for <= 1e-8).
"""
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from rwvae.relevance import explained_covariance
from rwvae.stats import (GroupedMomentAccumulator, LatentStats, Moments, bootstrap_group_weights,
                         image_group_id, principal_angles, relative_frobenius, spectrum_report,
                         split_half_groups, stream_stats, tokens)

TOL = 1e-10


def _rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _moments(x):
    """Independent two-pass reference: (n, mean, sum (x - mean)(x - mean)^T)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    xc = x - mean
    return Moments(len(x), mean, xc.T @ xc)


def _assert_close(a: Moments, b: Moments, tol=TOL):
    assert a.n == b.n
    assert _rel(a.mean, b.mean) <= tol
    assert _rel(a.m2, b.m2) <= tol


def _correlated(rng, n, d, offset=0.0):
    L = np.eye(d) + rng.standard_normal((d, d)) / np.sqrt(d)
    return offset + rng.standard_normal(d) + rng.standard_normal((n, d)) @ L


def _filled(rng, d=4, G=3, sizes=(1, 7, 30, 2, 61, 13, 1), offset=3.0, device="cpu"):
    x = _correlated(rng, sum(sizes), d, offset)
    gid = rng.integers(0, G, size=len(x))
    acc = GroupedMomentAccumulator(d, G, device=device)
    start = 0
    for s in sizes:
        acc.update(torch.from_numpy(x[start:start + s]), torch.from_numpy(gid[start:start + s]))
        start += s
    return acc, x, gid


# ---------------------------------------------------------------------------------------
# Moments and the accumulator
# ---------------------------------------------------------------------------------------
def test_accumulator_matches_two_pass_on_uneven_batches():
    rng = np.random.default_rng(0)
    acc, x, gid = _filled(rng, d=5, G=3)
    for g in range(3):
        _assert_close(acc.group(g), _moments(x[gid == g]))
    tot = acc.total()
    _assert_close(tot, _moments(x))
    assert _rel(tot.cov(), np.cov(x, rowvar=False)) <= TOL
    assert _rel(tot.cov(ddof=0), np.cov(x, rowvar=False, ddof=0)) <= TOL
    assert np.array_equal(tot.m2, tot.m2.T)                     # kept exactly symmetric
    assert acc.n.dtype == acc.mean.dtype == acc.m2.dtype == torch.float64


def test_update_with_int_group_and_empty_batch():
    acc = GroupedMomentAccumulator(3, 2)
    x = torch.randn(10, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    acc.update(x, 1)
    acc.update(x[:0], 1)                                         # an empty batch is a no-op
    _assert_close(acc.group(1), _moments(x.numpy()))
    assert acc.group(0).n == 0 and not acc.group(0).m2.any()


def test_float32_input_with_large_offset_stays_accurate():
    """1e3 +- 1 in float32: raw moments E[xx^T] - mm^T lose all accuracy in float32, while
    the float64 centered merge must still match the two-pass reference."""
    rng = np.random.default_rng(1)
    x32 = (1e3 + rng.standard_normal((20_000, 4)) * [1.0, 0.5, 2.0, 1.0]).astype(np.float32)
    gid = torch.from_numpy(rng.integers(0, 2, size=len(x32)))
    acc = GroupedMomentAccumulator(4, 2)
    xt = torch.from_numpy(x32)
    for i in range(0, len(x32), 1999):
        acc.update(xt[i:i + 1999], gid[i:i + 1999])
    ref = np.cov(x32.astype(np.float64), rowvar=False)
    assert _rel(acc.total().cov(), ref) <= TOL
    naive = x32.T @ x32 / np.float32(len(x32)) - np.outer(x32.mean(0), x32.mean(0))
    assert _rel(naive, ref) > 1e-2                               # the regime really is ill-conditioned


def test_moments_merge_is_associative_and_commutative():
    rng = np.random.default_rng(2)
    a, b, c = (_correlated(rng, n, 4, offset=50.0) for n in (5, 17, 40))
    A, B, C = _moments(a), _moments(b), _moments(c)
    ref = _moments(np.concatenate([a, b, c]))
    for m in (A.merge(B).merge(C), A.merge(B.merge(C)), C.merge(A).merge(B), B.merge(C).merge(A),
              Moments.combine([C, A, B])):
        _assert_close(m, ref)
    empty = Moments(0, np.zeros(4), np.zeros((4, 4)))
    for m in (A.merge(empty), empty.merge(A)):                   # the empty sample is an exact identity
        assert m.n == A.n and np.array_equal(m.mean, A.mean) and np.array_equal(m.m2, A.m2)
    assert empty.merge(empty).n == 0


def test_subset_with_bootstrap_weights_equals_replicated_data():
    rng = np.random.default_rng(3)
    G, d = 6, 3
    data = [_correlated(rng, n, d, offset=10.0 * g) for g, n in enumerate([3, 9, 1, 14, 6, 8])]
    acc = GroupedMomentAccumulator(d, G)
    for g, x in enumerate(data):
        acc.update(torch.from_numpy(x), g)
    W = bootstrap_group_weights(G, 5, seed=7)
    assert (W == 0).any()                                        # some groups are left out
    for k in W:
        replicated = np.concatenate([x for g, x in enumerate(data) for _ in range(k[g])])
        _assert_close(acc.subset(range(G), weights=k), _moments(replicated))
    _assert_close(acc.subset([1, 4]), _moments(np.concatenate([data[1], data[4]])))
    _assert_close(acc.subset([1, 1, 4]), acc.subset([1, 4], weights=[2, 1]))   # listing twice = weight 2


def test_state_dict_round_trip_through_npz(tmp_path):
    rng = np.random.default_rng(4)
    acc, _, _ = _filled(rng)
    acc.add_images(torch.tensor([0, 2, 2, 1]))
    sd = acc.state_dict()
    acc.update(torch.zeros(3, 4, dtype=torch.float64), 0)        # the state dict is a copy
    assert float(sd["n"][0]) == float(acc.n[0]) - 3
    np.savez(tmp_path / "acc.npz", **{k: np.asarray(v) for k, v in sd.items()})
    with np.load(tmp_path / "acc.npz", allow_pickle=False) as f:
        loaded = {k: f[k] for k in f.files}
    for src in (sd, loaded):
        acc2 = GroupedMomentAccumulator(4, 3)
        acc2.load_state_dict(src)
        for g in range(3):
            m = acc2.group(g)
            assert m.n == float(sd["n"][g])
            assert np.array_equal(m.mean, sd["mean"][g].numpy()) and np.array_equal(m.m2, sd["m2"][g].numpy())
        assert acc2.n_images == 4 and acc2.images_per_group.tolist() == [1, 1, 2]
    for bad in (GroupedMomentAccumulator(5, 3), GroupedMomentAccumulator(4, 2)):
        with pytest.raises(ValueError):
            bad.load_state_dict(sd)
    with pytest.raises(ValueError):
        GroupedMomentAccumulator(4, 3).load_state_dict({k: v for k, v in sd.items() if k != "m2"})


def test_merge_of_shards_equals_single_pass():
    rng = np.random.default_rng(5)
    x = _correlated(rng, 300, 4, offset=-7.0)
    gid = rng.integers(0, 4, size=300)
    whole, part_a, part_b = (GroupedMomentAccumulator(4, 4) for _ in range(3))
    whole.update(torch.from_numpy(x), torch.from_numpy(gid))
    part_a.update(torch.from_numpy(x[:110]), torch.from_numpy(gid[:110]))
    part_b.update(torch.from_numpy(x[110:]), torch.from_numpy(gid[110:]))
    part_a.add_images([0, 1])
    part_b.add_images([1, 3, 3])
    part_a.merge_(part_b)
    for g in range(4):
        _assert_close(part_a.group(g), whole.group(g))
    _assert_close(part_a.total(), _moments(x))
    assert part_a.images_per_group.tolist() == [1, 2, 0, 2]
    with pytest.raises(ValueError):
        part_a.merge_(GroupedMomentAccumulator(4, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_accumulator_on_cuda_matches_cpu():
    rng = np.random.default_rng(6)
    gpu, x, gid = _filled(rng, device="cuda")
    assert gpu.m2.device.type == "cuda"
    for g in range(3):
        _assert_close(gpu.group(g), _moments(x[gid == g]))
    cpu = GroupedMomentAccumulator(4, 3)
    cpu.update(torch.from_numpy(x), torch.from_numpy(gid))
    gpu.merge_(cpu)                                              # cross-device merge
    _assert_close(gpu.total(), _moments(np.concatenate([x, x])))


def test_accumulator_rejects_invalid_inputs():
    acc = GroupedMomentAccumulator(3, 2)
    bad_updates = [
        (torch.zeros(4, 2), 0),                                  # wrong dim
        (torch.zeros(4, 3), 2),                                  # group id out of range
        (torch.zeros(4, 3), torch.tensor([0, 1, 0])),            # one id per row
        (torch.zeros(4, 3), torch.tensor([0.0, 1.0, 0.0, 1.0])),  # ids must be integers
        (torch.zeros(4, 3, dtype=torch.int64), 0),               # x must be floating point
        (torch.full((4, 3), float("nan")), 0),                   # non-finite rows
        (np.zeros((4, 3)), 0),                                   # not a tensor
    ]
    for x, g in bad_updates:
        with pytest.raises(ValueError):
            acc.update(x, g)
    assert acc.total().n == 0                                    # nothing was half-applied
    with pytest.raises(ValueError):
        acc.add_images(5)                                        # an int count is ambiguous for G > 1
    with pytest.raises(ValueError):
        acc.subset([0, 1], weights=[1, -1])
    with pytest.raises(ValueError):
        acc.subset([0, 1], weights=[1])
    with pytest.raises(ValueError):
        acc.group(2)
    for args in ((0, 1), (3, 0), (2.5, 1)):
        with pytest.raises(ValueError):
            GroupedMomentAccumulator(*args)
    single = GroupedMomentAccumulator(3)
    single.add_images(5)                                         # contract form add_images(k)
    single.add_images(torch.zeros(2, dtype=torch.long))
    assert single.n_images == 7
    with pytest.raises(ValueError):
        Moments(-1, np.zeros(2), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Moments(2, np.zeros(2), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        Moments(1, np.zeros(2), np.zeros((2, 2))).cov()           # n <= ddof
    with pytest.raises(ValueError):
        Moments(3, np.zeros(2), np.eye(2)).merge(Moments(3, np.zeros(3), np.eye(3)))


# ---------------------------------------------------------------------------------------
# image ids, tokens, LatentStats, stream_stats
# ---------------------------------------------------------------------------------------
def test_image_group_id_is_stable_and_roughly_uniform():
    ids = [f"n{c:08d}_{i}" for c in range(20) for i in range(1000)]
    G = 16
    gids = [image_group_id(s, G) for s in ids]
    for s, g in zip(ids[:200], gids[:200]):                      # the documented formula
        digest = hashlib.sha1(("rwvae-v2\0" + s).encode()).digest()
        assert g == int.from_bytes(digest[:8], "big") % G
    assert gids == [image_group_id(s, G) for s in ids]
    counts = np.bincount(gids, minlength=G)
    expected = len(ids) / G
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 60.0                                           # chi2_15 tail P(>60) = 2.5e-7
    assert sum(image_group_id(s, G, salt="other") != g for s, g in zip(ids[:100], gids[:100])) > 50
    for args in ((123, 16), ("a", 0), ("a", 2.0)):
        with pytest.raises(ValueError):
            image_group_id(*args)


def test_tokens_flattening_convention():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    t = tokens(x)
    assert t.shape == (2 * 4 * 5, 3)
    for b, h, w in [(0, 0, 0), (0, 1, 2), (1, 3, 4), (1, 0, 4)]:
        assert torch.equal(t[b * 20 + h * 5 + w], x[b, :, h, w])
    with pytest.raises(ValueError):
        tokens(torch.zeros(2, 3, 4))


def _joint(rng, n=3000, C=4, D=3):
    z = _correlated(rng, n, C, offset=1.0)
    y = z @ rng.standard_normal((C, D)) + 0.5 * rng.standard_normal((n, D)) - 2.0
    return z, y


def test_latent_stats_from_joint_blocks_and_explained():
    rng = np.random.default_rng(8)
    z, y = _joint(rng)
    n, C, D = len(z), z.shape[1], y.shape[1]
    st = LatentStats.from_joint(_moments(np.hstack([z, y])), C, n_images=12, meta={"split": "train_fit"})
    assert (st.C, st.D, st.n_tokens, st.n_images) == (C, D, n, 12)
    assert st.Sigma_expl is None and st.meta == {"split": "train_fit", "cov_ddof": 1}
    assert _rel(st.m, z.mean(0)) <= TOL
    assert _rel(st.Sigma, np.cov(z, rowvar=False)) <= TOL
    assert _rel(st.Sigma_y, np.cov(y, rowvar=False)) <= TOL
    assert _rel(st.Sigma_zy, (z - z.mean(0)).T @ (y - y.mean(0)) / (n - 1)) <= TOL
    ex = st.with_explained(1e-3)
    ref, info = explained_covariance(st.Sigma_zy, st.Sigma_y, 1e-3)
    assert np.array_equal(ex.Sigma_expl, ref) and ex.meta["ridge"] == info
    assert ex.meta["ridge"]["ridge_rel"] == 1e-3 and ex.meta["split"] == "train_fit"
    assert st.Sigma_expl is None and "ridge" not in st.meta     # the original is untouched
    m, std = ex.channel_norm()
    assert np.array_equal(m, st.m) and _rel(std ** 2, np.diag(st.Sigma)) <= 1e-14


def test_latent_stats_save_load_and_fingerprint(tmp_path):
    rng = np.random.default_rng(9)
    z, y = _joint(rng, n=500)
    st = LatentStats.from_joint(_moments(np.hstack([z, y])), 4, n_images=2,
                                meta={"source": "unit-test", "seed": 9}).with_explained(1e-2)
    path = tmp_path / "stats.npz"
    st.save(path)
    back = LatentStats.load(path)
    for k in ("m", "Sigma", "Sigma_zy", "Sigma_y", "Sigma_expl"):
        assert np.array_equal(getattr(back, k), getattr(st, k))
    assert back.meta == st.meta and back.fingerprint() == st.fingerprint()
    assert (back.n_tokens, back.n_images) == (st.n_tokens, st.n_images)
    S2 = st.Sigma.copy()
    S2[1, 2] = np.nextafter(S2[1, 2], np.inf)                    # one ulp in one entry
    assert replace(st, Sigma=S2).fingerprint() != st.fingerprint()
    assert st.with_explained(0.0).fingerprint() != st.fingerprint()
    assert replace(st, meta={"other": 1}).fingerprint() == st.fingerprint()   # meta is not hashed
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["Sigma"] = arrays["Sigma"] * 2
    with open(tmp_path / "tampered.npz", "wb") as fh:
        np.savez(fh, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        LatentStats.load(tmp_path / "tampered.npz")
    with pytest.raises(ValueError):
        replace(st, meta={"bad": object()}).save(tmp_path / "bad.npz")


def test_latent_stats_without_y_and_invalid_inputs(tmp_path):
    rng = np.random.default_rng(10)
    z = _correlated(rng, 200, 3)
    st = LatentStats.from_joint(_moments(z), 3, n_images=4)
    assert st.D == 0 and st.Sigma_zy is None and st.Sigma_y is None
    with pytest.raises(ValueError):
        st.with_explained(1e-3)
    st.save(tmp_path / "z_only.npz")
    assert LatentStats.load(tmp_path / "z_only.npz").fingerprint() == st.fingerprint()
    dead = z.copy()
    dead[:, 1] = 5.0                                             # a constant channel
    with pytest.raises(ValueError, match="channels \\[1\\]"):
        LatentStats.from_joint(_moments(dead), 3, n_images=4).channel_norm()
    mom = _moments(z)
    for C in (0, 4, 1.5):
        with pytest.raises(ValueError):
            LatentStats.from_joint(mom, C, n_images=1)
    with pytest.raises(ValueError):                              # non-integral token count
        LatentStats.from_joint(Moments(2.5, mom.mean, mom.m2), 3, n_images=1)
    with pytest.raises(ValueError):                              # Sigma_zy without Sigma_y
        LatentStats(st.m, st.Sigma, np.zeros((3, 2)), None, None, 1, 1)
    with pytest.raises(ValueError):                              # Sigma_zy / Sigma_y shape mismatch
        LatentStats(st.m, st.Sigma, np.zeros((3, 2)), np.eye(3), None, 1, 1)
    with pytest.raises(ValueError):                              # non-square Sigma_y
        LatentStats(st.m, st.Sigma, np.zeros((3, 2)), np.zeros((2, 3)), None, 1, 1)


def test_stream_stats_end_to_end():
    rng = np.random.default_rng(11)
    C, D, H, W, G, N = 3, 2, 4, 5, 4, 23
    gid = np.array([image_group_id(f"img{i:04d}", G) for i in range(N)])
    mu = (rng.standard_normal((N, C, H, W)) + 2.0).astype(np.float32)          # cache-like float32
    y = (np.einsum("dc,bchw->bdhw", rng.standard_normal((D, C)), mu)
         + rng.standard_normal((N, D, H, W))).astype(np.float32)
    cuts = [0, 5, 6, 15, N]
    batches = ((torch.from_numpy(mu[a:b]), torch.from_numpy(y[a:b]), torch.from_numpy(gid[a:b]))
               for a, b in zip(cuts, cuts[1:]))
    acc = stream_stats(batches, C, D, num_groups=G)
    tok = lambda a: a.transpose(0, 2, 3, 1).reshape(-1, a.shape[1])          # independent flattening
    joint = np.hstack([tok(mu), tok(y)]).astype(np.float64)
    token_group = gid[np.repeat(np.arange(N), H * W)]          # every token goes to its image's group
    for g in range(G):
        assert (token_group == g).any()
        _assert_close(acc.group(g), _moments(joint[token_group == g]))
    _assert_close(acc.total(), _moments(joint))
    assert acc.n_images == N and acc.images_per_group.tolist() == np.bincount(gid, minlength=G).tolist()
    st = LatentStats.from_joint(acc.total(), C, acc.n_images)
    assert (st.C, st.D, st.n_tokens, st.n_images) == (C, D, N * H * W, N)
    z_only = stream_stats([(torch.from_numpy(mu), None, gid.tolist())], C, 0, num_groups=G)
    _assert_close(z_only.total(), _moments(tok(mu)))
    bad = [
        ((torch.from_numpy(mu), None, gid), D),                          # y missing although D > 0
        ((torch.from_numpy(mu), torch.from_numpy(y), gid), 0),           # y given although D = 0
        ((torch.from_numpy(mu), torch.from_numpy(y[:, :1]), gid), D),    # wrong y channels
        ((torch.from_numpy(mu[:, :2]), torch.from_numpy(y), gid), D),    # wrong mu channels
        ((torch.from_numpy(mu), torch.from_numpy(y), gid[:-1]), D),      # one group id per image
    ]
    for batch, d in bad:
        with pytest.raises(ValueError):
            stream_stats([batch], C, d, num_groups=G)


# ---------------------------------------------------------------------------------------
# spectrum and stability tools
# ---------------------------------------------------------------------------------------
def test_spectrum_report_repeated_near_singular_and_singular():
    rep = spectrum_report(np.diag([2.0, 4.0, 1.0, 2.0]))
    assert np.allclose(rep["eig"], [4, 2, 2, 1], rtol=1e-14, atol=0)   # descending, repeated
    p = np.array([4, 2, 2, 1]) / 9.0
    assert rep["eff_rank"] == pytest.approx(np.exp(-(p * np.log(p)).sum()), rel=1e-12)
    assert rep["cond"] == pytest.approx(4.0, rel=1e-14) and rep["n_nonpositive"] == 0
    assert (rep["min"], rep["max"]) == pytest.approx((1.0, 4.0), rel=1e-14)
    assert spectrum_report(np.eye(6))["eff_rank"] == pytest.approx(6.0, rel=1e-12)
    Q, _ = np.linalg.qr(np.random.default_rng(12).standard_normal((3, 3)))
    near = spectrum_report((Q * [1.0, 1.0, 1e-10]) @ Q.T)
    assert near["cond"] == pytest.approx(1e10, rel=1e-4)        # eig error ~ eps * ||S|| = 2e-16
    sing = spectrum_report(np.diag([3.0, 0.0, 0.0]))
    assert sing["cond"] == float("inf") and sing["n_nonpositive"] == 2
    assert sing["eff_rank"] == pytest.approx(1.0)
    for bad in (np.array([[1.0, 0.1], [0.0, 1.0]]), np.ones((2, 3)), np.diag([1.0, np.nan]), np.zeros((0, 0))):
        with pytest.raises(ValueError):
            spectrum_report(bad)


def test_principal_angles():
    rng = np.random.default_rng(13)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    U, O = Q[:, :3], Q[:, 3:6]
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    same = principal_angles(U, U @ R)                            # same subspace, another basis
    assert same.shape == (3,) and np.abs(same).max() <= 1e-12
    assert np.allclose(principal_angles(U, O), np.pi / 2, rtol=0, atol=1e-12)
    t = np.array([1.2, 1e-9, 0.3])
    V = U * np.cos(t) + O * np.sin(t)                            # column i rotated by t_i
    # 1e-9 rad: arccos(cos t) alone would give 0 (cos t rounds to 1); arcsin keeps it
    assert np.allclose(principal_angles(U, V), np.sort(t), rtol=1e-9, atol=1e-14)
    assert np.allclose(principal_angles(V, U), np.sort(t), rtol=1e-9, atol=1e-14)   # symmetric
    assert np.abs(principal_angles(U, U[:, :2])).max() <= 1e-12   # nested subspaces: p != q
    with pytest.raises(ValueError):
        principal_angles(2 * U, U)                               # not orthonormal
    with pytest.raises(ValueError):
        principal_angles(U, Q[:7, :2])                           # row mismatch


def test_relative_frobenius_and_split_half():
    B = np.arange(1.0, 7.0).reshape(2, 3)
    assert relative_frobenius(1.1 * B, B) == pytest.approx(0.1, rel=1e-12)
    with pytest.raises(ValueError):
        relative_frobenius(B, np.zeros_like(B))
    with pytest.raises(ValueError):
        relative_frobenius(B, B.T)
    assert split_half_groups(5) == ([0, 2, 4], [1, 3])
    even, odd = split_half_groups(16)
    assert sorted(even + odd) == list(range(16)) and not set(even) & set(odd)
    with pytest.raises(ValueError):
        split_half_groups(1)


def test_bootstrap_group_weights():
    w = bootstrap_group_weights(16, 200, seed=3)
    assert w.shape == (200, 16) and w.dtype == np.int64 and (w >= 0).all()
    assert (w.sum(axis=1) == 16).all()                           # G groups drawn with replacement
    assert np.array_equal(w, bootstrap_group_weights(16, 200, seed=3))
    assert not np.array_equal(w, bootstrap_group_weights(16, 200, seed=4))
    # each count is Binomial(16, 1/16): mean 1, sd of a 200-row mean 0.068 -> 5 sd = 0.35
    assert np.abs(w.mean(axis=0) - 1.0).max() < 0.35
    # P(count = 0) = (15/16)^16 = 0.356; sd over 3200 cells 0.0085 -> 5 sd = 0.043
    assert abs((w == 0).mean() - (15 / 16) ** 16) < 0.043
    for args in ((0, 5, 1), (4, 0, 1), (4, 5, -1), (4, 5, 1.5)):
        with pytest.raises(ValueError):
            bootstrap_group_weights(*args)
