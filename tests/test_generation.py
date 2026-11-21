"""Tests for rwvae.generation (docs/ARCHITECTURE_V2.md §3.9).

The training/sampling tests use a tiny LightningDiT (depth 2, hidden 64, patch 1; ~0.2M parameters)
on fixed random latents [48, 32, 16, 16] and LightningDiT's transport with the baseline recipe
(cosine loss + lognormal t, so the NumPy global RNG is exercised as well as torch's). The DDP test
spawns two CPU processes that talk over gloo; the CUDA test runs only where a GPU is available.
"""
import dataclasses
import logging
import math
import os
import random
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

from rwvae.channel import noise_seed, sample_noise
from rwvae.generation import (
    DiTTrainConfig,
    class_balanced_labels,
    denormalize,
    initial_noise,
    sample_latents,
    train_dit,
    update_ema,
)
from rwvae.integration import build_lightningdit, build_ode_sample_fn, build_transport

TINY = dict(depth=2, hidden_size=64, num_heads=2, patch_size=1)
TCFG = dict(path_type="Linear", prediction="velocity", loss_weight=None, train_eps=None, sample_eps=None,
            use_cosine_loss=True, use_lognorm=True)  # the transport block of configs/baseline_common.yaml
SCFG = dict(mode="ODE", sampling_method="euler", num_sampling_steps=3, atol=1e-6, rtol=1e-3, reverse=False)
N, NUM_CLASSES, SHAPE = 48, 10, (32, 16, 16)
LOG = logging.getLogger("test_generation")


class ToyLatents(Dataset):
    """Fixed random 'latents' with labels; items (x, y, index) like ChannelLatentDataset.
    Records its set_epoch calls (observable with num_workers=0)."""

    def __init__(self, n: int = N):
        g = torch.Generator().manual_seed(1234)
        self.x = torch.randn((n, *SHAPE), generator=g)
        self.y = torch.arange(n) % NUM_CLASSES
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i]), i


class PairLatents(ToyLatents):
    """Items (x, y): the other batch layout train_dit accepts."""

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)  # the global RNGs are restored after every test (fixture below)
    return build_lightningdit(input_size=16, in_channels=32, num_classes=NUM_CLASSES, custom_arch=TINY)


def make_cfg(**overrides) -> DiTTrainConfig:
    base = dict(max_steps=6, global_batch_size=8, log_every=2, ckpt_every=2, keep_last=3, num_workers=0, seed=3)
    return DiTTrainConfig(**{**base, **overrides})


def assert_same(a, b, path="ckpt"):
    """Bitwise equality of nested checkpoint structures."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b)):
            assert_same(u, v, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture
def dit_deps():
    """Third-party modules that models.lightningdit and transport import."""
    for name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy"):
        pytest.importorskip(name)


@pytest.fixture(autouse=True)
def restore_global_rngs():
    """train_dit (and tiny_model) seed the global Python/NumPy/torch RNGs: restore them afterwards."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


# --------------------------------------------------------------------------- update_ema
def test_update_ema_decay_zero_copies_and_decay_mixes():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).double()
    ema = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).double()
    model[0].weight.requires_grad_(False)  # frozen parameters are averaged too (LightningDiT's pos_embed)
    update_ema(ema, model, 0.0)
    for (_, e), (_, p) in zip(ema.named_parameters(), model.named_parameters()):
        assert torch.equal(e, p)

    old = {n: e.detach().clone() for n, e in ema.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    d = 0.9
    update_ema(ema, model, d)
    for name, p in model.named_parameters():
        expected = d * old[name] + (1 - d) * p.detach()
        rel = (dict(ema.named_parameters())[name] - expected).norm() / expected.norm()
        assert rel <= 1e-12, name
    frozen = {n: e.detach().clone() for n, e in ema.named_parameters()}
    update_ema(ema, model, 1.0)  # decay 1 freezes the EMA
    assert all(torch.equal(e, frozen[n]) for n, e in ema.named_parameters())


def test_update_ema_strips_the_ddp_prefix_and_rejects_mismatches():
    class Wrapped(nn.Module):  # parameter names "module.*", as DistributedDataParallel exposes them
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    torch.manual_seed(0)
    inner, ema = nn.Linear(3, 4), nn.Linear(3, 4)
    update_ema(ema, Wrapped(inner), 0.0)
    assert torch.equal(ema.weight, inner.weight) and torch.equal(ema.bias, inner.bias)

    before = ema.weight.detach().clone()
    with pytest.raises(ValueError, match="parameters differ"):
        update_ema(ema, nn.Linear(3, 4, bias=False), 0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        update_ema(ema, nn.Linear(2, 4), 0.0)
    assert torch.equal(ema.weight, before)  # nothing was touched
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="decay"):
            update_ema(ema, inner, bad)


# --------------------------------------------------------------------------- labels / noise / denormalize
@pytest.mark.parametrize("n,k", [(23, 5), (30, 10), (3, 10), (1000, 1000)])
def test_class_balanced_labels_counts_and_determinism(n, k):
    labels = class_balanced_labels(n, k, seed=11)
    assert labels.dtype == torch.int64 and labels.shape == (n,)
    counts = torch.bincount(labels, minlength=k)
    q, r = divmod(n, k)
    assert counts.tolist() == [q + 1] * r + [q] * (k - r)  # the first N mod K classes get the extra one
    assert torch.equal(labels, class_balanced_labels(n, k, seed=11))
    other = class_balanced_labels(n, k, seed=12)
    assert torch.equal(torch.bincount(other, minlength=k), counts)
    if n >= 23:
        assert not torch.equal(other, labels)  # a different seed permutes differently


def test_class_balanced_labels_rejects_bad_inputs():
    for args in ((0, 10, 0), (10, 0, 0), (10, 5, -1), (10.0, 5, 0), (10, 5, 2 ** 64), (True, 5, 0)):
        with pytest.raises(ValueError):
            class_balanced_labels(*args)


def test_initial_noise_is_per_index_and_independent_of_the_batch():
    idx = [3, 7, 11]
    x0 = initial_noise(idx, SHAPE, 5, "cpu")
    assert x0.shape == (3, *SHAPE) and x0.dtype == torch.float32
    assert torch.equal(x0, sample_noise(SHAPE, [noise_seed(5, 0, i) for i in idx]))
    assert torch.equal(x0[1], initial_noise([7], SHAPE, 5, "cpu")[0])            # batch composition
    assert torch.equal(x0.flip(0), initial_noise(torch.tensor(idx[::-1]), SHAPE, 5, "cpu"))
    assert torch.equal(x0, initial_noise(idx, SHAPE, 5, "cpu"))                  # determinism
    assert not torch.equal(x0, initial_noise(idx, SHAPE, 6, "cpu"))              # base seed matters
    assert not torch.equal(x0[0], x0[1])
    # standard normal: n = 3 * 8192 draws, so mean and std are within 5 / sqrt(n) ~ 0.03 of 0 and 1
    assert abs(float(x0.mean())) < 0.03 and abs(float(x0.std()) - 1.0) < 0.03
    with pytest.raises(ValueError):
        initial_noise([-1], SHAPE, 5, "cpu")
    with pytest.raises(ValueError):
        initial_noise([0], (32, 16), 5, "cpu")
    with pytest.raises(ValueError):
        initial_noise(torch.zeros(2, 2, dtype=torch.int64), SHAPE, 5, "cpu")


def test_denormalize_inverts_the_dataset_normalisation():
    z = torch.randn((4, 32, 16, 16), generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 3 + 1
    rng = np.random.default_rng(0)
    mean, std, mult = rng.standard_normal(32), rng.uniform(0.5, 2.0, 32), 0.7
    m, s = torch.from_numpy(mean).view(1, -1, 1, 1), torch.from_numpy(std).view(1, -1, 1, 1)
    x = (z - m) / s * mult                                    # ChannelLatentDataset's normalisation
    back = denormalize(x, mean, std, mult)
    assert (back - z).norm() / z.norm() <= 1e-12
    assert torch.equal(back, (x * s) / mult + m)              # LightningDiT inference.py's expression
    # float32 with LightningDiT's [1, C, 1, 1] latent-stats layout
    x32 = x.float()
    out = denormalize(x32, m.float(), s.float(), mult)
    assert out.dtype == torch.float32 and torch.equal(out, (x32 * s.float()) / mult + m.float())
    assert torch.equal(denormalize(x, mean, std), x * s + m)   # default multiplier 1
    bad_std = std.copy()
    bad_std[3] = 0.0
    for args in ((x, mean, bad_std, mult), (x, mean[:31], std, mult), (x, mean, std, 0.0),
                 (x[0], mean, std, mult), (x, np.full(32, np.nan), std, mult)):
        with pytest.raises(ValueError):
            denormalize(*args)


# --------------------------------------------------------------------------- DiTTrainConfig
def test_dit_train_config_defaults_validation_and_from_dict():
    cfg = DiTTrainConfig(max_steps=10, global_batch_size=4)
    assert (cfg.lr, cfg.beta2, cfg.weight_decay, cfg.ema_decay, cfg.grad_clip, cfg.mixed_precision) == \
        (2e-4, 0.95, 0.0, 0.9999, None, "bf16")
    assert (cfg.log_every, cfg.ckpt_every, cfg.keep_last, cfg.num_workers, cfg.seed) == (100, 5000, 3, 4, 0)
    assert DiTTrainConfig.from_dict(dataclasses.asdict(cfg)) == cfg
    with pytest.raises(ValueError, match="unknown keys \\['flip_prob'\\]"):
        DiTTrainConfig.from_dict({"max_steps": 10, "global_batch_size": 4, "flip_prob": 0.5})
    with pytest.raises(ValueError, match="missing keys \\['global_batch_size'\\]"):
        DiTTrainConfig.from_dict({"max_steps": 10})
    bad = dict(max_steps=0, global_batch_size=2.0, lr=0.0, beta2=1.0, weight_decay=-1.0, ema_decay=1.1,
               grad_clip=0.0, mixed_precision="fp32", log_every=0, ckpt_every=True, keep_last=0, num_workers=-1,
               seed=-1)
    for key, value in bad.items():
        with pytest.raises(ValueError, match=key):
            DiTTrainConfig(**{"max_steps": 10, "global_batch_size": 4, key: value})


# --------------------------------------------------------------------------- train_dit
def test_train_dit_runs_logs_and_keeps_the_newest_checkpoints(dit_deps, tmp_path, caplog):
    logged, data, cfg = [], ToyLatents(), make_cfg(keep_last=2, ema_decay=0.5)
    model = tiny_model()
    init = {k: v.clone() for k, v in model.state_dict().items()}
    with caplog.at_level(logging.INFO, logger=LOG.name):
        final = train_dit(model, build_transport(TCFG), data, cfg, tmp_path, device="cpu", logger=LOG,
                          log_fn=lambda step, m: logged.append((step, m)), meta={"arm": "B2", "R": 22.0})
    ckpt_dir = tmp_path / "checkpoints"
    assert final == ckpt_dir / "0000006.pt"
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["0000004.pt", "0000006.pt"]  # keep_last, no .tmp
    assert [s for s, _ in logged] == [2, 4, 6] and data.epochs == [0]
    for _, m in logged:
        assert set(m) == {"loss_mse", "loss_cos", "loss", "grad_norm", "lr", "steps_per_sec", "samples_per_sec"}
        assert all(math.isfinite(v) for v in m.values()) and m["lr"] == 2e-4
        # per-step loss = mse + cos is a float32 sum, so the window means agree to float32 rounding
        assert m["loss"] == pytest.approx(m["loss_mse"] + m["loss_cos"], rel=1e-6)
        assert m["samples_per_sec"] == pytest.approx(8 * m["steps_per_sec"])
    assert "step 0000006" in caplog.text and "0000006.pt" in caplog.text

    ckpt = load(final)
    assert (ckpt["step"], ckpt["epoch"], ckpt["batches_done_in_epoch"]) == (6, 0, 6)
    assert ckpt["cfg"] == dataclasses.asdict(cfg) and ckpt["scaler"] is None
    assert ckpt["meta"] == {"arm": "B2", "R": 22.0} and ckpt["invariants"]["dataset_len"] == N
    assert len(ckpt["rng"]) == 1 and set(ckpt["rng"][0]) == {"python", "numpy", "torch", "cuda"}
    assert_same(ckpt["model"], model.state_dict())                 # the model was trained in place
    w = "final_layer.linear.weight"                                # zero-initialised upstream
    assert not torch.equal(ckpt["model"][w], init[w])
    assert not torch.equal(ckpt["ema"][w], init[w]) and not torch.equal(ckpt["ema"][w], ckpt["model"][w])


@pytest.mark.parametrize("batch,epochs_after_resume", [
    (8, [0]),        # 6 steps/epoch: resume 4 batches into epoch 0
    (16, [1]),       # 3 steps/epoch: resume 1 batch into epoch 1
    (24, [1, 2]),    # 2 steps/epoch: the checkpoint closes epoch 1, resume starts epoch 2
])
def test_train_dit_resume_reproduces_the_uninterrupted_run_exactly(dit_deps, tmp_path, batch, epochs_after_resume):
    transport = build_transport(TCFG)
    full = train_dit(tiny_model(0), transport, ToyLatents(), make_cfg(global_batch_size=batch), tmp_path / "a",
                     device="cpu", logger=LOG)
    train_dit(tiny_model(0), transport, ToyLatents(), make_cfg(global_batch_size=batch, max_steps=4), tmp_path / "b",
              device="cpu", logger=LOG)
    data = ToyLatents()
    resumed = train_dit(tiny_model(1), transport, data, make_cfg(global_batch_size=batch), tmp_path / "b",
                        device="cpu", logger=LOG, resume=True)  # a different init: everything must come from 4.pt
    assert data.epochs == epochs_after_resume
    # model, EMA, optimizer, RNG states, counters: bitwise identical
    assert_same(load(full), load(resumed))
    # resuming a finished run is a no-op that returns the final checkpoint
    again = train_dit(tiny_model(2), transport, ToyLatents(), make_cfg(global_batch_size=batch), tmp_path / "b",
                      device="cpu", logger=LOG, resume=True)
    assert again == resumed and sorted(p.name for p in resumed.parent.iterdir()) == \
        ["0000002.pt", "0000004.pt", "0000006.pt"]


def test_train_dit_accepts_xy_batches_and_is_independent_of_the_worker_count(dit_deps, tmp_path):
    """num_workers=2 (spawn workers on Windows: the dataset is pickled) must not change the run."""
    transport = build_transport(TCFG)
    a = train_dit(tiny_model(), transport, PairLatents(), make_cfg(max_steps=3), tmp_path / "w0", device="cpu",
                  logger=LOG)
    b = train_dit(tiny_model(), transport, PairLatents(), make_cfg(max_steps=3, num_workers=2), tmp_path / "w2",
                  device="cpu", logger=LOG)
    ca, cb = load(a), load(b)
    for key in ("model", "ema", "opt", "rng", "step", "epoch", "batches_done_in_epoch"):
        assert_same(ca[key], cb[key], key)


def test_train_dit_resume_without_checkpoint_starts_fresh_and_warns_on_meta_change(dit_deps, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOG.name):
        train_dit(tiny_model(), build_transport(TCFG), ToyLatents(), make_cfg(max_steps=2), tmp_path, device="cpu",
                  logger=LOG, resume=True, meta={"arm": "B2"})
    assert "holds no checkpoint" in caplog.text and (tmp_path / "checkpoints" / "0000002.pt").is_file()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOG.name):
        train_dit(tiny_model(), build_transport(TCFG), ToyLatents(), make_cfg(max_steps=3), tmp_path, device="cpu",
                  logger=LOG, resume=True, meta={"arm": "B3"})
    assert any(r.levelno == logging.WARNING and "['arm']" in r.getMessage() for r in caplog.records)
    assert load(tmp_path / "checkpoints" / "0000003.pt")["meta"] == {"arm": "B3"}


def test_train_dit_refuses_inconsistent_runs(dit_deps, tmp_path, monkeypatch):
    transport = build_transport(TCFG)
    run = dict(device="cpu", logger=LOG)
    train_dit(tiny_model(), transport, ToyLatents(), make_cfg(max_steps=2), tmp_path, **run)
    with pytest.raises(ValueError, match="already holds checkpoints"):  # never mix two runs in one directory
        train_dit(tiny_model(), transport, ToyLatents(), make_cfg(max_steps=2), tmp_path, **run)
    with pytest.raises(ValueError, match="'lr': \\(0.0002, 0.001\\)"):
        train_dit(tiny_model(), transport, ToyLatents(), make_cfg(max_steps=4, lr=1e-3), tmp_path, resume=True, **run)
    with pytest.raises(ValueError, match="dataset_len"):
        train_dit(tiny_model(), transport, ToyLatents(40), make_cfg(max_steps=4), tmp_path, resume=True, **run)
    with pytest.raises(ValueError, match="exceeds max_steps=1"):
        train_dit(tiny_model(), transport, ToyLatents(), make_cfg(max_steps=1), tmp_path, resume=True, **run)
    with pytest.raises(ValueError, match="fewer than one batch"):
        train_dit(tiny_model(), transport, ToyLatents(4), make_cfg(), tmp_path / "small", **run)
    with pytest.raises(ValueError, match="DiTTrainConfig"):
        train_dit(tiny_model(), transport, ToyLatents(), dict(max_steps=2, global_batch_size=8), tmp_path / "d", **run)
    with pytest.raises(ValueError, match="JSON"):
        train_dit(tiny_model(), transport, ToyLatents(), make_cfg(), tmp_path / "m", meta={"x": object()}, **run)

    class Singles(ToyLatents):
        def __getitem__(self, i):
            return self.x[i]

    with pytest.raises(ValueError, match="\\(x, y\\) or \\(x, y, index\\)"):
        train_dit(tiny_model(), transport, Singles(), make_cfg(), tmp_path / "s", **run)
    monkeypatch.setenv("WORLD_SIZE", "3")  # checked before any process group is created
    with pytest.raises(ValueError, match="not divisible by WORLD_SIZE=3"):
        train_dit(tiny_model(), transport, ToyLatents(), make_cfg(), tmp_path / "w", **run)


class _Diverging(nn.Module):
    """Returns NaN velocities: the run must stop instead of training on NaN gradients."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(self, x, t=None, y=None):
        return x * self.w * float("nan")


def test_train_dit_stops_on_a_non_finite_gradient(dit_deps, tmp_path):
    with pytest.raises(FloatingPointError, match="step 1"):
        train_dit(_Diverging(), build_transport(TCFG), ToyLatents(), make_cfg(), tmp_path, device="cpu", logger=LOG)
    assert not (tmp_path / "checkpoints" / "0000002.pt").exists()


# --------------------------------------------------------------------------- DDP (2 CPU processes, gloo)
def _free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _ddp_worker(rank, world, ports, root):
    """One torchrun-like process: train_dit creates and destroys the gloo process group itself."""
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1")
    torch.set_num_threads(1)
    for port, (sub, steps, resume) in zip(ports, [("a", 6, False), ("b", 4, False), ("b", 6, True)]):
        os.environ["MASTER_PORT"] = str(port)
        model = tiny_model(10 + rank)  # different init per rank: DDP must broadcast rank 0's weights
        train_dit(model, build_transport(TCFG), ToyLatents(), make_cfg(max_steps=steps, global_batch_size=16),
                  Path(root) / sub, device="cpu", logger=logging.getLogger(f"ddp{rank}"), resume=resume)
        torch.save(model.state_dict(), Path(root) / f"rank{rank}_{sub}{steps}.pt")


@pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()), reason="needs torch.distributed + gloo")
def test_train_dit_ddp_two_processes_keep_replicas_identical_and_resume_exactly(dit_deps, tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_ddp_worker, args=(2, _free_ports(3), str(tmp_path)), nprocs=2, start_method="spawn")
    full = load(tmp_path / "a" / "checkpoints" / "0000006.pt")
    resumed = load(tmp_path / "b" / "checkpoints" / "0000006.pt")
    assert_same(full, resumed)
    # per-process batch 8 of 24 samples per rank: 3 steps per epoch, the run ends with epoch 1
    assert (full["epoch"], full["batches_done_in_epoch"], full["invariants"]["world_size"]) == (1, 3, 2)
    assert len(full["rng"]) == 2 and not torch.equal(full["rng"][0]["torch"], full["rng"][1]["torch"])
    for sub, steps in (("a", 6), ("b", 6)):  # both replicas end with rank 0's checkpointed weights
        assert_same(load(tmp_path / f"rank0_{sub}{steps}.pt"), full["model"])
        assert_same(load(tmp_path / f"rank1_{sub}{steps}.pt"), full["model"])


# --------------------------------------------------------------------------- sample_latents
def test_sample_latents_shape_determinism_and_shared_initial_noise(dit_deps):
    sample_fn = build_ode_sample_fn(build_transport(TCFG), SCFG)
    arms = [tiny_model(0), tiny_model(1)]
    for arm in arms:
        nn.init.normal_(arm.final_layer.linear.weight, std=0.02)  # zero-initialised upstream, i.e. v = 0
        arm.train()
    labels = class_balanced_labels(5, NUM_CLASSES, seed=0)
    idx = [10, 11, 12, 13, 14]
    seen = []

    def spy(model):
        def fn(x0, model_fn, **kwargs):
            assert not model.training and set(kwargs) == {"y"}  # eval mode, no CFG kwargs
            seen.append((x0.clone(), kwargs["y"].clone()))
            return sample_fn(x0, model_fn, **kwargs)
        return fn

    out = sample_latents(arms[0], spy(arms[0]), labels, idx, base_seed=7, batch_size=2, device="cpu")
    assert out.shape == (5, *SHAPE) and out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.isfinite(out).all() and arms[0].training  # the previous mode is restored
    x0_a = torch.cat([x for x, _ in seen])
    assert [len(y) for _, y in seen] == [2, 2, 1] and torch.equal(torch.cat([y for _, y in seen]), labels)
    assert torch.equal(x0_a, initial_noise(idx, SHAPE, 7, "cpu")) and not torch.equal(out, x0_a)
    assert torch.equal(out, sample_latents(arms[0], sample_fn, labels, idx, base_seed=7, batch_size=2, device="cpu"))

    seen.clear()
    out_b = sample_latents(arms[1], spy(arms[1]), labels, idx, base_seed=7, batch_size=2, device="cpu")
    assert torch.equal(torch.cat([x for x, _ in seen]), x0_a)  # two arms share every sample's x0
    assert not torch.equal(out_b, out)
    # a sample does not depend on its batch (up to float32 GEMM blocking differences)
    whole = sample_latents(arms[0], sample_fn, labels, idx, base_seed=7, batch_size=5, device="cpu")
    torch.testing.assert_close(whole, out, rtol=1e-5, atol=1e-6)
    assert not torch.equal(out, sample_latents(arms[0], sample_fn, labels, idx, base_seed=8, batch_size=2,
                                               device="cpu"))


def test_sample_latents_rejects_bad_inputs(dit_deps):
    model = tiny_model()
    fn = build_ode_sample_fn(build_transport(TCFG), SCFG)
    kw = dict(base_seed=0, batch_size=2, device="cpu")
    for labels, idx, extra in (([1, 2], [0], {}), ([1.0, 2.0], [0, 1], {}), ([], [], {}), ([-1, 2], [0, 1], {}),
                               ([1, 2], [0, -1], {}), ([1, 2], [0, 1], {"batch_size": 0}),
                               ([1, 2], [0, 1], {"mixed_precision": "fp32"}),
                               ([1, 2], [0, 1], {"latent_shape": (32, 16)})):
        with pytest.raises(ValueError):
            sample_latents(model, fn, labels, idx, **{**kw, **extra})


# --------------------------------------------------------------------------- CUDA (production path)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_train_resume_and_sample_on_cuda(dit_deps, tmp_path, precision):
    logged = []
    transport = build_transport(TCFG)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        full = train_dit(tiny_model(0), transport, ToyLatents(), make_cfg(max_steps=4, mixed_precision=precision),
                         tmp_path / "a", device="cuda", logger=LOG, log_fn=lambda s, m: logged.append(m))
        train_dit(tiny_model(0), transport, ToyLatents(), make_cfg(max_steps=2, mixed_precision=precision),
                  tmp_path / "b", device="cuda", logger=LOG)
        resumed = train_dit(tiny_model(1), transport, ToyLatents(), make_cfg(max_steps=4, mixed_precision=precision),
                            tmp_path / "b", device="cuda", logger=LOG, resume=True)
        ca, cb = load(full), load(resumed)
        model = tiny_model(5).cuda()
        model.load_state_dict(ca["ema"])
        nn.init.normal_(model.final_layer.linear.weight, std=0.02)
        labels = class_balanced_labels(6, NUM_CLASSES, seed=0)
        sample_fn = build_ode_sample_fn(transport, SCFG)
        out = sample_latents(model, sample_fn, labels, range(6), base_seed=1, batch_size=4, device="cuda",
                             mixed_precision=precision)
        again = sample_latents(model, sample_fn, labels, range(6), base_seed=1, batch_size=4, device="cuda",
                               mixed_precision=precision)
    assert all(m["max_mem_gb"] > 0 and math.isfinite(m["loss"]) for m in logged)
    assert (ca["scaler"] is not None) == (precision == "fp16")
    if precision == "fp16":
        assert ca["scaler"]["scale"] == cb["scaler"]["scale"]
    # the RNG streams (CPU, NumPy, CUDA) are restored exactly ...
    assert_same(ca["rng"], cb["rng"])
    assert ca["rng"][0]["cuda"] is not None
    # ... and the weights up to non-deterministic CUDA kernels: an Adam step moves a weight by at most about
    # lr (Kingma & Ba 2015, Sec. 2.1), so two runs drift apart by at most 2 * lr per step after the resume
    # at step 2. (On an RTX 4060 these runs are bitwise equal, as is LightningDiT-B/1 at batch 8; two
    # *uninterrupted* B/1 runs at batch 32, with the 8 GB card oversubscribed, differ by ~1e-4 after 12 steps.)
    bound = 2 * 2e-4 * 2
    for key in ("model", "ema"):
        for name in ca[key]:
            torch.testing.assert_close(cb[key][name], ca[key][name], rtol=0, atol=bound, msg=f"{key}.{name}")
    assert out.shape == (6, *SHAPE) and out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.isfinite(out).all() and torch.equal(out, again)
