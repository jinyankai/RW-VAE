"""Cross-module tests: the v2 modules chained the way scripts 02-07 chain them (plan v2.1 Sec. 3.2-3.3).

Every module has its own contract tests. These check the seams between them, on CPU:

    stats.stream_stats -> LatentStats.with_explained -> allocation.solve_allocation
      -> channel.RWChannel.from_allocation -> decoder_adapter (P6 fold, fine-tuning)
                                           -> datasets.ChannelLatentDataset -> generation (train, sample)
                                           -> evaluation (uint8, FID statistics)

plus the save/load round trips between stages. Synthetic data: i.i.d. Gaussian latent tokens with
a known covariance, and "DINO" features that depend only on the two lowest-variance latent
directions, so the semantic rule (r2) and the variance rule must allocate differently.
Monte-Carlo tolerances are 5 standard deviations of the estimator at the stated sample size.
The real_assets test repeats the data path on Imagenette with the real tokenizer and DINOv2.
"""
import json
import logging
import math
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from rwvae.allocation import (Allocation, identity_residuals, latent_mse_lmmse, nominal_rate, retention_matrix_whitened,
                              retention_overlap, semantic_retention, solve_allocation, whitened_relevance)
from rwvae.channel import RWChannel, noise_seed
from rwvae.datasets import (ChannelLatentDataset, ImageLatentPairDataset, LatentCache, LatentCacheWriter,
                            ManifestImageDataset, build_manifest)
from rwvae.decoder_adapter import AdaptedDecoder, finetune_decoder
from rwvae.evaluation import FIDStatistics, frechet_distance, load_fid_stats, save_fid_stats, to_uint8
from rwvae.generation import DiTTrainConfig, class_balanced_labels, denormalize, sample_latents, train_dit
from rwvae.relevance import alignment_shift_scan, cca_spectrum
from rwvae.runtime import to_jsonable
from rwvae.stats import (LatentStats, bootstrap_group_weights, image_group_id, relative_frobenius, split_half_groups,
                         stream_stats, tokens)

ASSETS = os.environ.get("RWVAE_ASSETS")
REPO = Path(__file__).resolve().parents[1]
C, D, H, W = 6, 10, 4, 4                 # latent channels, "DINO" dim, token grid
N_IMAGES, NUM_GROUPS = 8192, 8            # 131072 tokens for the Monte-Carlo checks
N_CACHE, NUM_CLASSES = 1024, 3            # the latent cache behind the dataset / DiT tests
R, W_R2, RIDGE, MULT = 6.0, 3.0, 1e-3, 0.8  # nats/token, relevance weight, ridge_rel, latent_multiplier
RULES = ("equal_snr", "variance", "r2")
TCFG = dict(path_type="Linear", prediction="velocity", loss_weight=None, train_eps=None, sample_eps=None,
            use_cosine_loss=True, use_lognorm=True)       # configs/baseline_common.yaml transport block
SCFG = dict(mode="ODE", sampling_method="euler", num_sampling_steps=3, atol=1e-6, rtol=1e-3, reverse=False)


# ----------------------------------------------------------------------------- helpers

def _synthetic_pairs(n_images: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired float64 (mu [n, C, H, W], y [n, D, H, W]) with i.i.d. tokens:
    mu = m0 + Q diag(sqrt(eig)) u, eig = logspace(1, -1, C), u ~ N(0, I_C);  y = G u[-2:] + 0.5 noise,
    i.e. y sees only the two lowest-variance directions of mu. The law depends on seed only."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    eig, m0, G = np.logspace(1.0, -1.0, C), rng.standard_normal(C), rng.standard_normal((D, 2))
    u = rng.standard_normal((n_images, H, W, C))
    z = m0 + (u * np.sqrt(eig)) @ Q.T
    y = u[..., -2:] @ G.T + 0.5 * rng.standard_normal((n_images, H, W, D))
    to_bchw = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))
    return to_bchw(z), to_bchw(y)


def _group_ids(rows) -> torch.Tensor:
    return torch.tensor([image_group_id(f"img{int(i):05d}", NUM_GROUPS) for i in rows])


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _tiny_decoder(seed: int, dtype: torch.dtype = torch.float64) -> tuple[nn.Conv2d, nn.Module]:
    """Stand-in tokenizer decoder: post_quant_conv (1x1, C -> C) and a decoder to 3 x (4H) x (4W)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pqc = nn.Conv2d(C, C, 1).to(dtype)
        dec = nn.Sequential(nn.Conv2d(C, 3, 3, padding=1), nn.Upsample(scale_factor=4), nn.Tanh()).to(dtype)
    return pqc, dec


class _MeanColour(nn.Module):
    """Inception stand-in for FIDStatistics: images [b, 3, h, w] in [0, 1] -> features [b, 3]."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x.mean(dim=(2, 3), keepdim=True)]   # pytorch-fid's layout: a list of [b, d, 1, 1]


@pytest.fixture(scope="module")
def synthetic():
    """Statistics streamed from N_IMAGES synthetic images, and the allocation of every lossy rule."""
    mu, y = _synthetic_pairs(N_IMAGES, seed=0)
    gid = _group_ids(range(N_IMAGES))
    acc = stream_stats(((mu[a:a + 1000], y[a:a + 1000], gid[a:a + 1000]) for a in range(0, N_IMAGES, 1000)),
                       C, D, num_groups=NUM_GROUPS)
    stats = LatentStats.from_joint(acc.total(), C, acc.n_images, meta={"source": "synthetic"}).with_explained(RIDGE)
    allocs = {rule: solve_allocation(stats, rule, W_R2 if rule == "r2" else None, R) for rule in RULES}
    return SimpleNamespace(mu=mu, acc=acc, stats=stats, allocs=allocs)


@pytest.fixture(scope="module")
def cached_arm(tmp_path_factory):
    """The data path of one arm: a float32 latent cache with flips -> statistics streamed from the
    cache (as 03_estimate_stats) -> r2 allocation -> RWChannel -> ChannelLatentDataset."""
    out = tmp_path_factory.mktemp("cache")
    mu, y = _synthetic_pairs(N_CACHE, seed=1)
    labels = torch.arange(N_CACHE) % NUM_CLASSES
    writer = LatentCacheWriter(out, split="train_fit", meta={"source": "synthetic", "latent_source": "posterior_mean"},
                               dtype="float32", shard_size=96)
    for k in range(writer.num_shards(N_CACHE)):
        rows = writer.shard_rows(k, N_CACHE)
        sl = slice(rows.start, rows.stop)
        writer.write_shard(k, mu[sl], labels[sl], torch.arange(rows.start, rows.stop), latents_flip=mu[sl].flip(-1))
    writer.finalize(N_CACHE)
    cache = LatentCache(out)
    gid = _group_ids(range(N_CACHE))
    acc = stream_stats(((m, y[idx], gid[idx]) for m, _, idx in cache.iter_batches(100)), C, D, num_groups=NUM_GROUPS)
    stats = LatentStats.from_joint(acc.total(), C, acc.n_images, meta=dict(cache.meta)).with_explained(RIDGE)
    alloc = solve_allocation(stats, "r2", W_R2, R)
    mean, std = stats.channel_norm()
    ds = ChannelLatentDataset(cache, RWChannel.from_allocation(alloc), mean, std, base_seed=5, flip_prob=0.5,
                              latent_multiplier=MULT)
    return SimpleNamespace(mu=mu, labels=labels, stats=stats, alloc=alloc, mean=mean, std=std, ds=ds)


@pytest.fixture
def restore_global_rngs():
    """train_dit and LightningDiT draw from (and seed) the global RNGs: restore them afterwards."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


# ----------------------------------------------------------------------------- package

def test_package_loads_every_module_lazily():
    """`import rwvae` stays numpy-only (theory, metrics); every other module is an attribute (PEP 562)."""
    code = (
        "import sys, rwvae\n"
        "assert rwvae.__version__ == '0.2.0', rwvae.__version__\n"
        "assert 'torch' not in sys.modules, 'import rwvae imported torch'\n"
        "rwvae.allocation\n"
        "assert 'torch' not in sys.modules, 'rwvae.allocation imported torch'\n"
        "names = [n for n in rwvae.__all__ if n != 'RelevanceRateGate']\n"
        "assert all(getattr(rwvae, n).__name__ == 'rwvae.' + n for n in names), names\n"
        "assert rwvae.RelevanceRateGate is rwvae.gate.RelevanceRateGate\n"
        "assert not hasattr(rwvae, 'no_such_module')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr


# ----------------------------------------------------------------------------- stats -> allocation -> channel

def test_streamed_statistics_feed_every_rule(synthetic):
    st, allocs = synthetic.stats, synthetic.allocs
    tok = tokens(synthetic.mu).numpy()
    assert (st.n_tokens, st.n_images, st.C, st.D) == (N_IMAGES * H * W, N_IMAGES, C, D)
    assert relative_frobenius(st.Sigma, np.cov(tok, rowvar=False)) <= 1e-10
    assert np.abs(st.m - tok.mean(0)).max() <= 1e-12 * np.abs(tok).max()
    # relevance.cca_spectrum and the K that allocation whitens with are the same matrix (P1)
    K, info = whitened_relevance(st.Sigma, st.Sigma_expl)
    assert not info["clipped"]
    np.testing.assert_allclose(cca_spectrum(st.Sigma, st.Sigma_zy, st.Sigma_y, RIDGE), np.linalg.eigvalsh(K)[::-1],
                               rtol=0, atol=1e-10)
    for rule, a in allocs.items():
        assert a.meta["stats_fingerprint"] == st.fingerprint() and a.meta["ridge"] == st.meta["ridge"]
        assert a.C == C and nominal_rate(a) == pytest.approx(R, rel=1e-12)
        res = identity_residuals(a, st.Sigma)
        assert max(res.values()) <= 1e-8, (rule, res)
    # y is driven by the two lowest-variance directions, so at equal R the r2 rule keeps far more
    # of the whitened relevance K than the variance rule (the H1 mechanism, by construction)
    kept = {rule: semantic_retention(a, st.Sigma, K) for rule, a in allocs.items()}
    assert kept["r2"] > kept["variance"] + 0.3, kept


def test_group_subsets_give_stable_allocations(synthetic):
    """The 04_solve_channel stability chain: image-level split halves and a group bootstrap replicate
    -> per-subset LatentStats -> allocations -> whitened retention overlaps (plan Sec. 3.3 item 3)."""
    acc = synthetic.acc

    def retention(mom, n_images: int) -> dict:
        st = LatentStats.from_joint(mom, C, n_images).with_explained(RIDGE)
        return {rule: retention_matrix_whitened(solve_allocation(st, rule, W_R2 if rule == "r2" else None, R), st.Sigma)
                for rule in ("variance", "r2")}

    even, odd = (retention(acc.subset(g), int(acc.images_per_group[g].sum())) for g in split_half_groups(NUM_GROUPS))
    weights = bootstrap_group_weights(NUM_GROUPS, 1, seed=0)[0]
    boot = retention(acc.subset(range(NUM_GROUPS), weights), int((acc.images_per_group * torch.from_numpy(weights)).sum()))
    full = retention(acc.total(), acc.n_images)
    # ~65k tokens per half for a 6x6 covariance: subset retention matrices agree to ~1e-5, far closer than
    # the two rules agree with each other (the split-half / inter-rule ratio of the M3 go/no-go gate)
    between = 1.0 - retention_overlap(full["variance"], full["r2"])
    for rule in ("variance", "r2"):
        within = 1.0 - retention_overlap(even[rule], odd[rule])
        assert within < 1e-3 and 1.0 - retention_overlap(boot[rule], full[rule]) < 1e-3
        assert within / between < 0.05, (rule, within, between)


@pytest.mark.parametrize("rule", RULES)
def test_channel_from_allocation_acts_per_token(synthetic, rule):
    a, S, mu = synthetic.allocs[rule], synthetic.stats.Sigma, synthetic.mu
    ch = RWChannel.from_allocation(a, dtype=torch.float64)
    for name in ("m", "A", "B"):
        assert torch.equal(getattr(ch, name), torch.from_numpy(getattr(a, name))), name
    assert float(ch.rate) == R and ch.meta["name"] == a.name and ch.meta["allocation_fingerprint"] == a.fingerprint()

    eps = torch.randn(mu.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    out = ch(mu, noise=eps)
    tok, z = tokens(mu).numpy(), tokens(out).numpy()
    ref = a.m + (tok - a.m) @ a.A.T + tokens(eps).numpy() @ a.B.T    # z' = m + A (mu - m) + B eps per token
    assert relative_frobenius(z, ref) <= 1e-12
    # P4 empirically: a sample covariance of n Gaussian vectors has E||S_hat - S||_F^2 = (tr(S)^2 + ||S||_F^2) / n
    n = z.shape[0]
    sd = math.sqrt((np.trace(S) ** 2 + np.sum(S * S)) / n) / np.linalg.norm(S)
    assert relative_frobenius(np.cov(z, rowvar=False), S) <= 5 * sd
    assert np.all(np.abs(z.mean(0) - a.m) <= 5 * np.sqrt(np.diag(S) / n))
    # P6/P7: the error e of ch.lmmse is Gaussian with Cov e = E, so E||e||^2 = tr E = D_latent and
    # Var||e||^2 = 2 tr(E^2)
    err = np.sum((tok - tokens(ch.lmmse(out)).numpy()) ** 2, axis=1)
    E = (a.Tinv * (np.exp(-2.0 * a.rates) * a.s)) @ a.Tinv.T
    assert np.trace(E) == pytest.approx(latent_mse_lmmse(a), rel=1e-12)
    assert abs(err.mean() - latent_mse_lmmse(a)) <= 5 * math.sqrt(2.0 * np.trace(E @ E) / n)


@pytest.mark.parametrize("rule", RULES)
def test_lmmse_adapter_decodes_the_channel_lmmse(synthetic, rule):
    a = synthetic.allocs[rule]
    ch = RWChannel.from_allocation(a, dtype=torch.float64)
    pqc, dec = _tiny_decoder(0)
    plain, lmmse = AdaptedDecoder(pqc, dec), AdaptedDecoder(pqc, dec, mode="lmmse", A=a.A, m=a.m)
    mu = synthetic.mu[:8]
    z = ch(mu, noise=torch.randn(mu.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64))
    with torch.no_grad():
        # P6 fold W0' = W0 A, b0' = b0 + W0 (I - A) m  ==  post_quant_conv(m + A (z' - m))
        assert _rel(lmmse.post_quant_conv(z), pqc(ch.lmmse(z))) <= 1e-12
        out = lmmse(z)
        assert out.shape == (8, 3, 4 * H, 4 * W)
        assert _rel(out, plain(ch.lmmse(z))) <= 1e-10
        assert _rel(out, plain(z)) > 1e-3                # the analytic adapter does change the decoding
        # B0: the "none" allocation folds into the original decoder bit for bit
        none = solve_allocation(synthetic.stats, "none", None, None)
        assert torch.equal(AdaptedDecoder(pqc, dec, mode="lmmse", A=none.A, m=none.m)(z), plain(z))


def test_saved_stats_and_allocations_reproduce_the_channel(synthetic, tmp_path):
    st = synthetic.stats
    st.save(tmp_path / "stats.npz")
    loaded = LatentStats.load(tmp_path / "stats.npz")
    assert loaded.fingerprint() == st.fingerprint() and loaded.meta == st.meta
    mu = synthetic.mu[:16]
    eps = torch.randn(mu.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    for rule in ("none", *RULES):
        w, rate = (W_R2 if rule == "r2" else None), (None if rule == "none" else R)
        a = solve_allocation(loaded, rule, w, rate)
        assert a.fingerprint() == solve_allocation(st, rule, w, rate).fingerprint()
        assert a.meta["stats_fingerprint"] == st.fingerprint()
        a.save(tmp_path / f"{a.name}.npz")
        b = Allocation.load(tmp_path / f"{a.name}.npz")
        assert (b.fingerprint(), b.name, b.meta) == (a.fingerprint(), a.name, a.meta)
        ch_a, ch_b = RWChannel.from_allocation(a, torch.float64), RWChannel.from_allocation(b, torch.float64)
        assert ch_b.meta == ch_a.meta and torch.equal(ch_b(mu, noise=eps), ch_a(mu, noise=eps))
        json.dumps(to_jsonable(b.meta), allow_nan=False)  # run summaries can carry the meta (R = inf for B0)
    b0 = RWChannel.from_allocation(Allocation.load(tmp_path / "none.npz"))
    assert b0.is_identity and torch.equal(b0(mu.float()), mu.float())


# ----------------------------------------------------------------------------- datasets / decoder / generation

def test_channel_dataset_serves_the_solved_arm(cached_arm):
    arm = cached_arm
    a, ds = arm.alloc, arm.ds
    mu32 = arm.mu.float().double()                      # what the float32 cache stores
    mean, std = arm.mean[:, None, None], arm.std[:, None, None]
    for epoch in (0, 3):
        ds.set_epoch(epoch)
        for index in (0, 95, 96, N_CACHE - 1):          # shard edges (shard_size 96)
            x, label, idx = ds[index]
            # the documented per-sample protocol, recomputed in float64 from the allocation
            g = torch.Generator().manual_seed(noise_seed(5, epoch, index))
            flip = bool(torch.rand((), generator=g) < 0.5)
            eps = torch.randn((C, H, W), generator=g, dtype=torch.float32).double().numpy()
            mu = (mu32[index].flip(-1) if flip else mu32[index]).numpy()
            z = (a.m[:, None, None] + np.einsum("ij,jhw->ihw", a.A, mu - a.m[:, None, None])
                 + np.einsum("ij,jhw->ihw", a.B, eps))
            assert (label, idx, x.dtype) == (int(arm.labels[index]), index, torch.float32)
            np.testing.assert_allclose(x.numpy(), (z - mean) / std * MULT, rtol=0, atol=2e-5)
            # generation.denormalize maps the DiT input back to z'
            back = denormalize(x[None].double(), arm.mean, arm.std, MULT)[0].numpy()
            np.testing.assert_allclose(back, z, rtol=0, atol=2e-5)
    # one epoch of DiT inputs is standardised per channel by the shared LatentStats.channel_norm
    ds.set_epoch(0)
    t = tokens(torch.stack([ds[i][0] for i in range(len(ds))]).double()).numpy()
    n = t.shape[0]
    assert np.all(np.abs(t.mean(0)) <= 5 * MULT / math.sqrt(n))
    assert np.all(np.abs(t.var(0, ddof=1) / MULT ** 2 - 1.0) <= 5 * math.sqrt(2.0 / n))
    clone = pickle.loads(pickle.dumps(ds))              # spawned DataLoader workers get a pickled copy
    for i in (0, N_CACHE - 1):
        assert torch.equal(clone[i][0], ds[i][0])


def _image_folder(root: Path) -> None:
    """2 classes x (5 train, 3 val) random 24x20 RGB PNGs in the ImageFolder layout."""
    rng = np.random.default_rng(0)
    for wnid in ("n001", "n002"):
        for split, count in (("train", 5), ("val", 3)):
            for j in range(count):
                path = root / split / wnid / f"{wnid}_{split}{j}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)).save(path)


def test_decoder_finetuning_on_image_latent_pairs_of_a_solved_arm(tmp_path):
    _image_folder(tmp_path / "data")
    manifest = build_manifest(tmp_path / "data" / "train", tmp_path / "data" / "val", dev_per_class=1, seed=0,
                              name="tiny", image_size=4 * H)
    n = manifest.split_size("train_fit")
    mu, _ = _synthetic_pairs(n, seed=2)
    writer = LatentCacheWriter(tmp_path / "cache", split="train_fit", dtype="float32", shard_size=n,
                               meta={"manifest_fingerprint": manifest.fingerprint(), "latent_source": "posterior_mean"})
    writer.write_shard(0, mu, manifest.splits["train_fit"]["label"], torch.arange(n))
    writer.finalize(n)
    cache = LatentCache(tmp_path / "cache", manifest=manifest)
    acc = stream_stats(((m, None, _group_ids(idx)) for m, _, idx in cache.iter_batches(4)), C, 0,
                       num_groups=NUM_GROUPS)
    alloc = solve_allocation(LatentStats.from_joint(acc.total(), C, acc.n_images), "variance", None, R)
    channel = RWChannel.from_allocation(alloc)
    pairs = ImageLatentPairDataset(manifest, cache, image_size=4 * H)
    pqc, dec = _tiny_decoder(0, torch.float32)
    before = [p.clone() for p in (*pqc.parameters(), *dec.parameters())]
    lpips_stub = lambda x_hat, x: (x_hat - x).abs().mean(dim=(1, 2, 3), keepdim=True)   # [B, 1, 1, 1]

    def run():
        adapter = AdaptedDecoder(pqc, dec, mode="lmmse", A=alloc.A, m=alloc.m)
        loader = DataLoader(pairs, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
        return adapter, finetune_decoder(adapter, loader, channel, lpips_fn=lpips_stub, steps=4, lr=1e-3,
                                         device="cpu", noise_base_seed=7, log_every=2)

    (a1, r1), (a2, r2) = run(), run()
    assert r1 == r2 and r1["steps"] == 4 and r1["mixed_precision"] == "no"   # bf16 is CUDA-only
    assert all(math.isfinite(r1[k]) for k in ("final_loss", "final_l1", "final_lpips"))
    assert (a1.mode, a1.base_mode) == ("finetuned", "lmmse")
    assert all(torch.equal(p, q) for p, q in zip(a1.state_dict().values(), a2.state_dict().values()))
    assert all(torch.equal(p, q) for p, q in zip(before, (*pqc.parameters(), *dec.parameters())))  # copies trained
    meta = {"allocation_fingerprint": alloc.fingerprint(), "manifest_fingerprint": manifest.fingerprint(),
            "finetune": r1}
    a1.save(tmp_path / "decoder.safetensors", meta)
    fresh = AdaptedDecoder(pqc, dec, mode="lmmse", A=alloc.A, m=alloc.m)
    assert fresh.load_weights(tmp_path / "decoder.safetensors") == meta
    z = channel(mu[:2].float(), noise=torch.randn((2, C, H, W), generator=torch.Generator().manual_seed(4)))
    with torch.no_grad():
        assert torch.equal(fresh(z), a1(z))


@pytest.fixture
def lightningdit():
    """build_lightningdit / build_transport need the LightningDiT checkout and its dependencies."""
    for name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy"):
        pytest.importorskip(name)
    from rwvae import integration
    try:
        integration.add_lightningdit_to_path()
    except FileNotFoundError as err:
        pytest.skip(str(err))
    return integration


def test_dit_trains_on_the_arm_and_its_samples_reach_fid_statistics(cached_arm, lightningdit, restore_global_rngs,
                                                                    tmp_path):
    arm = cached_arm
    transport = lightningdit.build_transport(TCFG)

    def tiny_dit(seed: int) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return lightningdit.build_lightningdit(input_size=H, in_channels=C, num_classes=NUM_CLASSES,
                                                   custom_arch=dict(depth=1, hidden_size=32, num_heads=2, patch_size=1))

    cfg = DiTTrainConfig(max_steps=2, global_batch_size=8, ema_decay=0.5, log_every=1, ckpt_every=2, keep_last=1,
                         num_workers=0, seed=0)
    meta = {"allocation_fingerprint": arm.alloc.fingerprint(), "stats_fingerprint": arm.stats.fingerprint()}
    steps = []
    final = train_dit(tiny_dit(0), transport, arm.ds, cfg, tmp_path / "dit", device="cpu",
                      logger=logging.getLogger("test_cross_module"), log_fn=lambda s, m: steps.append(s), meta=meta)
    ckpt = torch.load(final, map_location="cpu", weights_only=True)
    assert (ckpt["step"], ckpt["meta"], steps) == (2, meta, [1, 2])
    ema = tiny_dit(1)
    ema.load_state_dict(ckpt["ema"])

    labels = class_balanced_labels(16, NUM_CLASSES, seed=0)
    sample_fn = lightningdit.build_ode_sample_fn(transport, SCFG)
    sample = lambda: sample_latents(ema, sample_fn, labels, range(16), base_seed=11, batch_size=8, device="cpu",
                                    latent_shape=(C, H, W))
    x = sample()
    assert x.shape == (16, C, H, W) and torch.isfinite(x).all() and torch.equal(x, sample())

    pqc, dec = _tiny_decoder(0, torch.float32)
    decoder = AdaptedDecoder(pqc, dec, mode="lmmse", A=arm.alloc.A, m=arm.alloc.m)

    def fid_stats(latents: torch.Tensor):
        fid = FIDStatistics(_MeanColour(), "cpu", batch_size=8)
        with torch.no_grad():
            for b in latents.split(8):
                fid.update(to_uint8(decoder(b)))
        return fid.compute()

    gen = fid_stats(denormalize(x, arm.mean, arm.std, MULT))        # samples -> tokenizer latent space -> images
    arm.ds.set_epoch(0)
    ref_latents = torch.stack([arm.ds[i][0] for i in range(64)])
    save_fid_stats(tmp_path / "ref.npz", *fid_stats(denormalize(ref_latents, arm.mean, arm.std, MULT)),
                   meta={"split": "train_fit"})
    mu_r, s_r, n_r, meta_r = load_fid_stats(tmp_path / "ref.npz")
    assert (gen[2], n_r, meta_r) == (16, 64, {"split": "train_fit"})
    fd, info = frechet_distance(gen[0], gen[1], mu_r, s_r, return_info=True)
    assert math.isfinite(fd) and fd >= -1e-9 and not info["eps_offset_applied"]
    assert abs(frechet_distance(mu_r, s_r, mu_r, s_r)) <= 1e-9


# ----------------------------------------------------------------------------- real assets

@pytest.mark.real_assets
@pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")
def test_real_dino_patch_grid_pairs_with_the_latent_grid():
    """Plan Sec. 1.2 data gate on Imagenette: DINOv2 patches (224 px input, 16x16 grid) pair with the
    f16d32 latent tokens (256 px input) at grid shift (0, 0); the real statistics give exact channels."""
    pytest.importorskip("timm")
    from rwvae.integration import DinoPatchFeatures, FrozenTokenizer

    root = Path(ASSETS)
    snapshots = root / "hf_cache" / "hub" / "models--timm--vit_large_patch14_dinov2.lvd142m" / "snapshots"
    weights = sorted(snapshots.glob("*/model.safetensors"))
    if not weights:
        pytest.skip(f"DINOv2-L weights not found under {snapshots}")
    data = root / "datasets" / "imagenette2-320"
    manifest = build_manifest(data / "train", data / "val", dev_per_class=20, seed=2026, name="imagenette2-320")
    labels = np.asarray(manifest.splits["train_dev"]["label"])
    rows = np.concatenate([np.flatnonzero(labels == k)[:8] for k in range(len(manifest.classes))])   # 80 images
    images = ManifestImageDataset(manifest, "train_dev", indices=rows)
    x = torch.stack([images[k][0] for k in range(len(images))])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = FrozenTokenizer.from_checkpoint(root / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt", device=device)
    dino = DinoPatchFeatures(checkpoint_path=weights[0], device=device)
    with torch.no_grad():
        mu = torch.cat([tokenizer.encode_mean(b.to(device)).cpu() for b in x.split(16)])
        y = torch.cat([dino(b.to(device)).float().cpu() for b in x.split(16)])
    assert mu.shape == (80, 32, 16, 16) and y.shape == (80, 1024, 16, 16)

    scan = alignment_shift_scan(mu, y, max_shift=1, ridge_rel=1e-3)
    print("explained fraction by grid shift (dy, dx):", {k: round(v, 4) for k, v in scan.items()})
    assert max(scan, key=scan.get) == (0, 0), scan

    gid = torch.tensor([image_group_id(manifest.record("train_dev", int(i))["image_id"], 4) for i in rows])
    acc = stream_stats([(mu, y, gid)], 32, 1024, num_groups=4)
    stats = LatentStats.from_joint(acc.total(), 32, acc.n_images).with_explained(1e-3)
    for rule, w in (("equal_snr", None), ("variance", None), ("r2", 2.0)):
        res = identity_residuals(solve_allocation(stats, rule, w, 22.0), stats.Sigma)
        assert max(res.values()) <= 1e-8, (rule, res)
