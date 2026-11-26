"""RW-VAE v2: LightningDiT training loop, EMA and class-conditional sampling (ARCHITECTURE_V2 §3.9).

`train_dit` re-implements third_party/LightningDiT/train.py (do_train) in one file, without
accelerate, for one process (e.g. Windows, one GPU) or torchrun DDP (WORLD_SIZE > 1):

    loss = mean_b L_mse(b) + mean_b L_cos(b)       (L_cos only if the transport has a cosine loss)
    AdamW(lr, betas=(0.9, beta2), weight_decay), optional clipping of the global gradient norm
    theta_ema <- d * theta_ema + (1 - d) * theta   after every optimizer step (d = 0 once at the start)

with L_mse, L_cos the per-sample terms transport.training_losses(model, x, dict(y=y))["loss"] and
["cos_loss"]. The model forward runs under autocast on CUDA and returns float32, as accelerate wraps
it upstream, so the loss terms themselves are float32.

Reproducibility (plan §3.3 item 5, §8): the process seed cfg.seed * world_size + rank seeds Python,
NumPy and torch, which LightningDiT draws from (the transport's t and noise, scipy's lognormal t, the
CFG label dropout). The data order is the DistributedSampler(seed=cfg.seed) permutation of each epoch,
the dataset's own randomness is keyed by its set_epoch(epoch), and every epoch's DataLoader gets its
own seeded generator, so creating an iterator never draws from the global RNG. A checkpoint therefore
holds everything an exact resume needs: weights, EMA, optimizer, GradScaler, (epoch, batches done in
it) and every rank's global RNG states. On CPU a resumed run reproduces the uninterrupted one bit for
bit; on CUDA it does so up to non-deterministic kernels (e.g. the attention backward).

Sampling (plan §5.1): no CFG, class-balanced labels (`class_balanced_labels`) and an initial noise
that depends only on (base_seed, sample index) (`initial_noise`), so every arm sees the same labels
and the same x0 per sample index (common random numbers), whatever the batching or sharding.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .channel import noise_seed, sample_noise

MIXED_PRECISION = ("bf16", "fp16", "no")
_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}
# DiTTrainConfig fields that shape the optimisation trajectory: a resume must not change them.
_TRAJECTORY_FIELDS = ("global_batch_size", "lr", "beta2", "weight_decay", "ema_decay", "grad_clip",
                      "mixed_precision", "seed")
_CKPT_NAME = re.compile(r"(\d+)\.pt")


def _check_int(name: str, value, minimum: int, maximum: int | None = None) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum
            or (maximum is not None and value > maximum)):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be an integer {bound}, got {value!r}")
    return int(value)


def _check_real(name: str, value, ok: Callable[[float], bool], expected: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value) or not ok(value)):
        raise ValueError(f"{name} must be a finite number {expected}, got {value!r}")
    return float(value)


def _silent(*args, **kwargs) -> None:
    """Logger stand-in on DDP ranks > 0 (only rank 0 logs, as upstream)."""


# --------------------------------------------------------------------------- EMA
@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """theta_ema <- decay * theta_ema + (1 - decay) * theta for every parameter (LightningDiT update_ema).

    Parameters are paired by named_parameters(), with a DDP "module." prefix stripped from model's names.
    As upstream, frozen parameters (LightningDiT's fixed pos_embed) are averaged too and buffers are left
    alone; decay = 0 copies the weights. Name or shape mismatches raise ValueError before any update.
    """
    decay = _check_real("decay", decay, lambda d: 0.0 <= d <= 1.0, "in [0, 1]")
    ema_params = OrderedDict(ema.named_parameters())
    model_params = OrderedDict((name.removeprefix("module."), p) for name, p in model.named_parameters())
    if ema_params.keys() != model_params.keys():
        only_model = sorted(model_params.keys() - ema_params.keys())
        only_ema = sorted(ema_params.keys() - model_params.keys())
        raise ValueError(f"EMA and model parameters differ: {len(only_model)} only in the model {only_model[:3]}, "
                         f"{len(only_ema)} only in the EMA {only_ema[:3]}")
    mismatched = [(n, tuple(ema_params[n].shape), tuple(p.shape)) for n, p in model_params.items()
                  if ema_params[n].shape != p.shape]
    if mismatched:
        raise ValueError(f"EMA/model shape mismatch (name, EMA, model): {mismatched[:3]}")
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# --------------------------------------------------------------------------- sampling protocol
def class_balanced_labels(num_samples: int, num_classes: int, seed: int) -> torch.Tensor:
    """N labels (int64 [N]) in which class k appears floor(N/K) + [k < N mod K] times, in the order of
    torch.randperm(N, generator=torch.Generator().manual_seed(seed)) (plan §5.1: class-balanced labels,
    identical for every arm)."""
    n = _check_int("num_samples", num_samples, 1)
    k = _check_int("num_classes", num_classes, 1)
    seed = _check_int("seed", seed, 0, 2 ** 64 - 1)
    labels = torch.arange(n, dtype=torch.int64) % k
    return labels[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]


def initial_noise(indices, shape_chw: Sequence[int], base_seed: int, device) -> torch.Tensor:
    """Initial ODE state x0 [len(indices), *shape_chw], float32 standard normal: row i is drawn from
    torch.Generator().manual_seed(noise_seed(base_seed, 0, indices[i])), so a sample index gets the same
    x0 in every arm, batch composition and process."""
    if isinstance(indices, torch.Tensor):
        if indices.dim() != 1:
            raise ValueError(f"indices must be 1-D, got shape {tuple(indices.shape)}")
        indices = indices.tolist()
    shape = tuple(shape_chw)
    if len(shape) != 3 or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1 for s in shape):
        raise ValueError(f"shape_chw must be three positive integers (C, H, W), got {shape_chw!r}")
    return sample_noise(shape, [noise_seed(base_seed, 0, i) for i in indices], device=device)


@torch.no_grad()
def sample_latents(model: nn.Module, sample_fn: Callable, labels, noise_indices, *, base_seed: int,
                   batch_size: int, device, latent_shape: Sequence[int] = (32, 16, 16),
                   mixed_precision: str = "bf16") -> torch.Tensor:
    """Class-conditional samples without CFG, in the NORMALISED latent space (see `denormalize`).

    Sample i starts from initial_noise([noise_indices[i]], latent_shape, base_seed) and is the last state
    of sample_fn(x0, model_fn, y=labels_batch) (e.g. rwvae.integration.build_ode_sample_fn), where
    model_fn is model.forward run under autocast(mixed_precision) on CUDA (float32 on CPU) with a float32
    output, so the ODE integrates in float32. The model is in eval mode while sampling (no label dropout)
    and gets its previous mode back. Returns a CPU float32 tensor [N, *latent_shape].
    """
    labels = _int_vector("labels", labels)
    indices = _int_vector("noise_indices", noise_indices).tolist()
    if len(indices) != len(labels):
        raise ValueError(f"got {len(labels)} labels but {len(indices)} noise indices")
    batch_size = _check_int("batch_size", batch_size, 1)
    if mixed_precision not in MIXED_PRECISION:
        raise ValueError(f"mixed_precision must be one of {MIXED_PRECISION}, got {mixed_precision!r}")
    device = torch.device(device)
    amp = device.type == "cuda" and mixed_precision != "no"

    def model_fn(x, t, **kwargs):
        with torch.autocast(device.type, dtype=_AMP_DTYPE[mixed_precision], enabled=amp):
            return model.forward(x, t, **kwargs).float()

    was_training = model.training
    model.eval()
    try:
        out = []
        for start in range(0, len(indices), batch_size):
            x0 = initial_noise(indices[start:start + batch_size], latent_shape, base_seed, device)
            y = labels[start:start + batch_size].to(device)
            out.append(sample_fn(x0, model_fn, y=y)[-1].float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(out)


def _int_vector(name: str, values) -> torch.Tensor:
    t = torch.as_tensor(values)
    if (t.dim() != 1 or t.numel() == 0 or t.is_floating_point() or t.is_complex() or t.dtype == torch.bool
            or int(t.min()) < 0):
        raise ValueError(f"{name} must be a non-empty 1-D sequence of non-negative integers, got "
                         f"{t.dtype} {tuple(t.shape)}")
    return t.to(torch.int64)


def _per_channel(name: str, value, C: int) -> torch.Tensor:
    t = (value.detach().to("cpu", torch.float64) if isinstance(value, torch.Tensor)
         else torch.from_numpy(np.asarray(value, dtype=np.float64)))
    if t.numel() != C or not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must hold {C} finite per-channel values, got shape {tuple(t.shape)}")
    return t.reshape(1, C, 1, 1)


def denormalize(x: torch.Tensor, mean, std, multiplier: float = 1.0) -> torch.Tensor:
    """Normalised DiT latents -> latent space: x * std / multiplier + mean per channel (dim 1).

    Inverse of ChannelLatentDataset's (z - mean) / std * multiplier and the same expression as
    LightningDiT inference.py. mean/std hold C values each (e.g. LatentStats.channel_norm(), or
    LightningDiT's [1, C, 1, 1] latent stats); the result has x's dtype and device.
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 4 or not x.is_floating_point():
        raise ValueError(f"expected a float tensor [B, C, H, W], got {type(x).__name__} "
                         f"{tuple(getattr(x, 'shape', ()))} {getattr(x, 'dtype', '')}")
    C = x.shape[1]
    mean_t, std_t = _per_channel("mean", mean, C), _per_channel("std", std, C)
    if bool((std_t <= 0).any()):
        raise ValueError(f"std must be > 0, got min {float(std_t.min())}")
    multiplier = _check_real("multiplier", multiplier, lambda v: v > 0, "> 0")
    return (x * std_t.to(x)) / multiplier + mean_t.to(x)


# --------------------------------------------------------------------------- training
@dataclass(frozen=True)
class DiTTrainConfig:
    """LightningDiT train.py recipe (configs/baseline_common.yaml `train`) with an explicit seed and EMA
    decay. mixed_precision applies on CUDA only; CPU training is always float32."""

    max_steps: int
    global_batch_size: int
    lr: float = 2e-4
    beta2: float = 0.95
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    grad_clip: float | None = None          # max global gradient L2 norm; None = no clipping
    mixed_precision: str = "bf16"           # "bf16" | "fp16" (with GradScaler) | "no"
    log_every: int = 100
    ckpt_every: int = 5000
    keep_last: int = 3
    num_workers: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        for name, minimum in (("max_steps", 1), ("global_batch_size", 1), ("log_every", 1), ("ckpt_every", 1),
                              ("keep_last", 1), ("num_workers", 0), ("seed", 0)):
            _check_int(name, getattr(self, name), minimum)
        _check_real("lr", self.lr, lambda v: v > 0, "> 0")
        _check_real("beta2", self.beta2, lambda v: 0 <= v < 1, "in [0, 1)")
        _check_real("weight_decay", self.weight_decay, lambda v: v >= 0, ">= 0")
        _check_real("ema_decay", self.ema_decay, lambda v: 0 <= v <= 1, "in [0, 1]")
        if self.grad_clip is not None:
            _check_real("grad_clip", self.grad_clip, lambda v: v > 0, "> 0 (or None)")
        if self.mixed_precision not in MIXED_PRECISION:
            raise ValueError(f"mixed_precision must be one of {MIXED_PRECISION}, got {self.mixed_precision!r}")

    @classmethod
    def from_dict(cls, d: Mapping) -> "DiTTrainConfig":
        """Build from a mapping of these fields only: unknown keys (e.g. the dataset keys of a config's
        `train` block, which the caller must drop) and missing required keys raise ValueError."""
        names = [f.name for f in fields(cls)]
        unknown = sorted(set(d) - set(names))
        missing = [n for n in ("max_steps", "global_batch_size") if n not in d]
        if unknown or missing:
            raise ValueError(f"DiTTrainConfig: unknown keys {unknown}, missing keys {missing}; fields are {names}")
        return cls(**d)


# --------------------------------------------------------------------------- REPA semantic anchor (v3)
@dataclass(frozen=True)
class REPAConfig:
    """REPA (arXiv:2410.06940) as the v3 semantic anchor (RWVAE_V3_ALLOCATION_PLAN.md §2.3): the output tokens
    of DiT block `depth` (1-based) go through an MLP projector (Linear-SiLU-Linear-SiLU-Linear) and are pulled
    towards frozen teacher patch tokens by loss weight * mean(1 - cos)."""

    depth: int
    weight: float = 0.5
    proj_hidden: int = 2048
    target_dim: int = 1024

    def __post_init__(self) -> None:
        for name in ("depth", "proj_hidden", "target_dim"):
            _check_int(name, getattr(self, name), 1)
        _check_real("weight", self.weight, lambda v: v > 0, "> 0")


def build_repa_projector(hidden_size: int, cfg: REPAConfig) -> nn.Module:
    return nn.Sequential(nn.Linear(hidden_size, cfg.proj_hidden), nn.SiLU(),
                         nn.Linear(cfg.proj_hidden, cfg.proj_hidden), nn.SiLU(),
                         nn.Linear(cfg.proj_hidden, cfg.target_dim))


def repa_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean over tokens of 1 - cos(projected [B, T, D], target [B, D, H, W]) with T = H * W (row-major)."""
    tgt = target.flatten(2).transpose(1, 2)
    if tgt.shape != projected.shape:
        raise ValueError(f"REPA target tokens {tuple(tgt.shape)} != projected DiT tokens {tuple(projected.shape)}")
    return 1 - nn.functional.cosine_similarity(projected.float(), tgt.float(), dim=-1).mean()


def train_dit(model: nn.Module, transport, dataset: Dataset, cfg: DiTTrainConfig, out_dir, *, device,
              logger: logging.Logger, resume: bool = False, log_fn: Callable[[int, dict], None] | None = None,
              meta: Mapping | None = None, repa: REPAConfig | None = None,
              repa_target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Path:
    """Train `model` in place with LightningDiT's recipe; returns out_dir/checkpoints/{max_steps:07d}.pt.

    Processes: one, or DDP when WORLD_SIZE > 1 (torchrun; RANK and LOCAL_RANK from the environment). The
    process group (nccl on Linux + CUDA, else gloo) is created here unless one is already initialised,
    and is then destroyed here too; on CUDA each process trains on cuda:LOCAL_RANK. global_batch_size
    must be divisible by the world size. Windows builds of torch lack the libuv that torchrun's
    rendezvous needs: there, start the processes yourself with WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR
    and MASTER_PORT set (as tests/test_generation.py does).

    Data: (x, y) or (x, y, index) items of `dataset` in the order of DistributedSampler(num_replicas=world,
    rank, shuffle=True, seed=cfg.seed, drop_last=True), with sampler.set_epoch(e) and dataset.set_epoch(e)
    (if defined) before every epoch.

    Logging (rank 0), every log_every steps and at the end: means over the steps since the last log,
    averaged over ranks, of loss_mse, loss_cos (with a cosine loss only), loss and grad_norm (before
    clipping), plus lr, steps_per_sec, samples_per_sec and, on CUDA, max_mem_gb (peak allocated GiB) ->
    logger.info and log_fn(step, metrics). The first window after a resume is partial.

    Checkpoints (rank 0, every ckpt_every steps and at max_steps; temp file + os.replace; only the
    newest keep_last kept): {step:07d}.pt = {model, ema (unwrapped state dicts), opt, scaler (None unless
    fp16 on CUDA), step, epoch, batches_done_in_epoch, cfg (dict), invariants (what a resume must not
    change), meta, rng: one {python, numpy, torch, cuda} state per rank}. meta (JSON-serialisable, e.g.
    data/channel fingerprints and git commit; §1) is stored verbatim.

    REPA (optional, v3): with `repa` set, dataset items must be (x, y, index, image) and repa_target_fn(images
    on `device`) must return teacher tokens [B, target_dim, H, W] matching the DiT's token grid; the loss gains
    repa.weight * repa_loss, the projector is trained with the DiT (same optimizer settings, DDP-synchronised),
    is logged as loss_repa and stored under "repa_projector" in every checkpoint. The EMA covers the DiT only.

    resume=True continues the newest checkpoint (by step) exactly: weights, EMA, optimizer, scaler and
    RNG states are restored and the consumed batches of the interrupted epoch are skipped. It raises
    ValueError if that checkpoint is past max_steps or was written with another world size, device type,
    dataset length or trajectory field (global_batch_size, lr, beta2, weight_decay, ema_decay, grad_clip,
    mixed_precision, seed), warns if meta differs, and starts from scratch if there is no checkpoint.
    resume=False refuses an out_dir that already holds checkpoints. A non-finite gradient norm raises
    FloatingPointError (with fp16 the GradScaler skips such steps instead).
    """
    if not isinstance(cfg, DiTTrainConfig):
        raise ValueError(f"cfg must be a DiTTrainConfig (see DiTTrainConfig.from_dict), got {type(cfg).__name__}")
    if isinstance(model, DDP):
        raise ValueError("pass the unwrapped model: train_dit wraps it in DistributedDataParallel itself")
    meta = dict(meta or {})
    try:
        json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError(f"meta must be JSON-serialisable (it is stored in every checkpoint): {err}") from None
    device = torch.device(device)
    world = _check_int("WORLD_SIZE", int(os.environ.get("WORLD_SIZE", 1)), 1)
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if not 0 <= rank < world:
        raise ValueError(f"RANK={rank} is outside [0, WORLD_SIZE={world})")
    if cfg.global_batch_size % world:
        raise ValueError(f"global_batch_size={cfg.global_batch_size} is not divisible by WORLD_SIZE={world}")
    if cfg.seed * world + rank >= 2 ** 32:
        raise ValueError(f"process seed seed*world+rank = {cfg.seed * world + rank} must be < 2**32 (NumPy)")
    if (repa is None) != (repa_target_fn is None):
        raise ValueError("pass both repa and repa_target_fn, or neither")
    if repa is not None and not 1 <= repa.depth <= len(model.blocks):
        raise ValueError(f"repa.depth={repa.depth} outside the DiT's 1..{len(model.blocks)} blocks")

    own_group = False
    if world > 1:
        if device.type == "cuda":
            if device.index not in (None, local_rank):
                raise ValueError(f"under DDP each process trains on cuda:LOCAL_RANK (={local_rank}), got {device}")
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if sys.platform.startswith("linux") and device.type == "cuda"
                                    else "gloo")
            own_group = True
        if (dist.get_world_size(), dist.get_rank()) != (world, rank):
            raise ValueError(f"process group (world {dist.get_world_size()}, rank {dist.get_rank()}) does not "
                             f"match WORLD_SIZE={world}, RANK={rank}")
    try:
        return _train(model, transport, dataset, cfg, Path(out_dir), device=device, world=world, rank=rank,
                      logger=logger, resume=resume, log_fn=log_fn, meta=meta, repa=repa,
                      repa_target_fn=repa_target_fn)
    finally:
        if own_group:
            dist.destroy_process_group()


def _train(model, transport, dataset, cfg: DiTTrainConfig, out_dir: Path, *, device: torch.device, world: int,
           rank: int, logger, resume: bool, log_fn, meta: dict, repa: REPAConfig | None = None,
           repa_target_fn=None) -> Path:
    """Body of train_dit (the process group, if any, is set up)."""
    distributed = world > 1
    info, warn = (logger.info, logger.warning) if rank == 0 else (_silent, _silent)
    ckpt_dir = out_dir / "checkpoints"
    batch = cfg.global_batch_size // world

    process_seed = cfg.seed * world + rank          # DiT / LightningDiT convention: global_seed * world + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)                 # CPU and every CUDA generator

    # Model, EMA and optimizer as train.py: the EMA is synced (decay 0) after DDP broadcast rank 0's weights.
    model.to(device)
    ema = deepcopy(model).requires_grad_(False).eval()
    net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if distributed else model
    update_ema(ema, model, 0.0)
    net.train()                                     # enables the label dropout that CFG relies on
    params = list(model.parameters())
    projector = proj_net = None
    hidden = {}
    if repa is not None:                            # built after the EMA copy: the EMA is the DiT alone
        projector = build_repa_projector(model.pos_embed.shape[-1], repa).to(device)
        proj_net = DDP(projector, device_ids=[device.index] if device.type == "cuda" else None) \
            if distributed else projector
        params += list(projector.parameters())
        model.blocks[repa.depth - 1].register_forward_hook(lambda mod, inp, out: hidden.__setitem__("h", out))
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, cfg.beta2), weight_decay=cfg.weight_decay)
    amp = device.type == "cuda" and cfg.mixed_precision != "no"
    scaler = torch.amp.GradScaler(device.type, enabled=amp and cfg.mixed_precision == "fp16")

    def forward(x, t, **kwargs):                    # accelerate semantics: autocast the forward, float32 out
        with torch.autocast(device.type, dtype=_AMP_DTYPE[cfg.mixed_precision], enabled=amp):
            return net(x, t, **kwargs).float()

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg.seed,
                                 drop_last=True)
    steps_per_epoch = len(sampler) // batch
    if steps_per_epoch == 0:
        raise ValueError(f"{len(dataset)} items give {len(sampler)} per process: fewer than one batch of {batch}")

    invariants = {**{k: getattr(cfg, k) for k in _TRAJECTORY_FIELDS}, "world_size": world,
                  "device_type": device.type, "dataset_len": len(dataset)}
    if repa is not None:                            # absent (None) for runs without REPA, as in older checkpoints
        invariants["repa"] = asdict(repa)
    step = epoch = batches_done = 0
    rng_state = None
    found = _list_checkpoints(ckpt_dir)
    if found and not resume:
        raise ValueError(f"{ckpt_dir} already holds checkpoints (newest {found[-1][1].name}); pass resume=True "
                         "to continue that run, or use a fresh out_dir")
    if resume and found:
        path = found[-1][1]
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        changed = {k: (ckpt["invariants"].get(k), v) for k, v in invariants.items()
                   if ckpt["invariants"].get(k) != v}
        if changed:
            raise ValueError(f"cannot resume {path}: changed since the checkpoint (checkpoint, now): {changed}")
        if ckpt["step"] > cfg.max_steps:
            raise ValueError(f"cannot resume {path}: its step {ckpt['step']} exceeds max_steps={cfg.max_steps}")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if projector is not None:
            projector.load_state_dict(ckpt["repa_projector"])
        opt.load_state_dict(ckpt["opt"])
        if scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        step, epoch, batches_done = ckpt["step"], ckpt["epoch"], ckpt["batches_done_in_epoch"]
        rng_state = ckpt["rng"][rank]
        if ckpt["meta"] != meta:
            changed_keys = sorted(k for k in ckpt["meta"].keys() | meta.keys() if ckpt["meta"].get(k) != meta.get(k))
            warn("resuming %s whose meta differs from this run's in %s", path, changed_keys)
        info("resuming from %s: step %d, epoch %d, %d of %d batches of the epoch done", path, step, epoch,
             batches_done, steps_per_epoch)
    elif resume:
        info("resume=True but %s holds no checkpoint: starting from scratch", ckpt_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    info("LightningDiT %.2fM parameters | AdamW lr=%g betas=(0.9, %g) weight_decay=%g | EMA decay %g | "
         "grad_clip %s", sum(p.numel() for p in model.parameters()) / 1e6, cfg.lr, cfg.beta2, cfg.weight_decay,
         cfg.ema_decay, cfg.grad_clip)
    info("world %d | device %s | batch %d per process, %d global | %d items, %d steps per epoch | mixed "
         "precision %s | seed %d (process seed %d) | max_steps %d", world, device, batch, cfg.global_batch_size,
         len(dataset), steps_per_epoch, cfg.mixed_precision if amp else "off (float32)", cfg.seed, process_seed,
         cfg.max_steps)
    if rng_state is not None:
        _set_rng_state(rng_state, device)           # last: nothing below draws from them before the next step

    sums = torch.zeros(5, dtype=torch.float64, device=device)   # window sums: mse, cos, loss, grad norm, repa
    n_window, t_window, has_cos = 0, time.perf_counter(), False
    while step < cfg.max_steps:
        sampler.set_epoch(epoch)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        order = list(sampler)[batches_done * batch:]            # resume: skip the batches already consumed
        loader = DataLoader(dataset, batch_size=batch, sampler=order, num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda", drop_last=True, persistent_workers=False,
                            generator=torch.Generator().manual_seed(noise_seed(cfg.seed, epoch, rank)))
        for items in loader:
            x, y, images = _split_batch(items)
            x = x.to(device, torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if repa is not None:
                if images is None:
                    raise ValueError("REPA needs dataset items (x, y, index, image)")
                with torch.no_grad():
                    target = repa_target_fn(images.to(device, non_blocking=True))
            terms = transport.training_losses(forward, x, dict(y=y))
            mse = terms["loss"].mean()
            has_cos = "cos_loss" in terms
            cos = terms["cos_loss"].mean() if has_cos else torch.zeros_like(mse)
            loss = mse + cos if has_cos else mse
            if repa is not None:
                with torch.autocast(device.type, dtype=_AMP_DTYPE[cfg.mixed_precision], enabled=amp):
                    projected = proj_net(hidden.pop("h"))
                loss_repa = repa_loss(projected, target)
                loss = loss + repa.weight * loss_repa
            else:
                loss_repa = torch.zeros_like(mse)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
            if not scaler.is_enabled() and not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm {float(grad_norm)} at step {step + 1} "
                                         f"(loss {float(loss.detach())})")
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grads_with_norm_(params, cfg.grad_clip, grad_norm)
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model, cfg.ema_decay)
            step += 1
            batches_done += 1
            sums += torch.stack([mse.detach(), cos.detach(), loss.detach(), grad_norm, loss_repa.detach()]).double()
            n_window += 1

            if step % cfg.log_every == 0 or step == cfg.max_steps:
                means = sums / n_window
                if distributed:
                    dist.all_reduce(means)
                    means /= world
                loss_mse, loss_cos, loss_total, grad_norm_mean, loss_rp = means.tolist()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t_window
                metrics = {"loss_mse": loss_mse, **({"loss_cos": loss_cos} if has_cos else {}),
                           **({"loss_repa": loss_rp} if repa is not None else {}), "loss": loss_total,
                           "grad_norm": grad_norm_mean, "lr": opt.param_groups[0]["lr"],
                           "steps_per_sec": n_window / elapsed,
                           "samples_per_sec": n_window * cfg.global_batch_size / elapsed}
                if device.type == "cuda":
                    metrics["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                info("step %07d | %s", step, " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
                if log_fn is not None and rank == 0:
                    log_fn(step, metrics)
                sums.zero_()
                n_window, t_window = 0, time.perf_counter()

            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                rng_states = [_rng_state(device)]
                if distributed:                     # every rank's own streams, saved by rank 0
                    local, rng_states = rng_states[0], [None] * world
                    dist.all_gather_object(rng_states, local)
                if rank == 0:
                    path = _save_checkpoint(ckpt_dir, step, {
                        "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "step": step, "epoch": epoch, "batches_done_in_epoch": batches_done, "cfg": asdict(cfg),
                        "invariants": invariants, "meta": meta, "rng": rng_states,
                        **({"repa_projector": projector.state_dict()} if projector is not None else {})})
                    removed = _prune_checkpoints(ckpt_dir, cfg.keep_last, newest=path)
                    info("saved %s (epoch %d, %d/%d batches)%s", path, epoch, batches_done, steps_per_epoch,
                         f"; removed {[p.name for p in removed]}" if removed else "")
                if distributed:
                    dist.barrier()
            if step >= cfg.max_steps:
                break
        else:                                       # epoch exhausted
            epoch, batches_done = epoch + 1, 0

    final = ckpt_dir / f"{cfg.max_steps:07d}.pt"
    info("training finished at step %d: %s", step, final)
    return final


def _split_batch(items) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(x, y, images or None) of an (x, y), (x, y, index) or (x, y, index, image) batch."""
    if isinstance(items, (list, tuple)) and len(items) in (2, 3, 4):
        return items[0], items[1], items[3] if len(items) == 4 else None
    size = f" of length {len(items)}" if isinstance(items, (list, tuple)) else ""
    raise ValueError(f"dataset items must be (x, y) or (x, y, index) (or (x, y, index, image) with REPA), got a "
                     f"{type(items).__name__}{size} batch")


def _rng_state(device: torch.device) -> dict:
    """The global RNG states a training step draws from: Python, NumPy (list form, loadable with
    weights_only=True), torch CPU and the CUDA generator of `device`."""
    kind, key, pos, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [kind, key.tolist(), int(pos), int(has_gauss), float(cached)],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _set_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    kind, key, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """(step, path) of every {step}.pt file in ckpt_dir, ascending by step."""
    if not ckpt_dir.is_dir():
        return []
    return sorted((int(m.group(1)), p) for p in ckpt_dir.iterdir()
                  if (m := _CKPT_NAME.fullmatch(p.name)) and p.is_file())


def _save_checkpoint(ckpt_dir: Path, step: int, state: dict) -> Path:
    """Write ckpt_dir/{step:07d}.pt atomically: torch.save to a temp file, then os.replace."""
    path = ckpt_dir / f"{step:07d}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def _prune_checkpoints(ckpt_dir: Path, keep_last: int, newest: Path) -> list[Path]:
    """Delete all but the newest keep_last checkpoints, never `newest` (the one just written)."""
    stale = [p for _, p in _list_checkpoints(ckpt_dir)[:-keep_last] if p != newest]
    for p in stale:
        p.unlink()
    return stale
