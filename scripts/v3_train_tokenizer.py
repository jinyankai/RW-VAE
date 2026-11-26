"""v3 -- train the RW-VAE v3 tokenizer of one arm (RWVAE_V3_ALLOCATION_PLAN.md v3.1 §2.1-2.4, §3.1 stage B).

Loss (after the warm-up):  l1 * |x_rec - x| + lpips * LPIPS(x_rec, x) + w_sem * L_sem + w_det * L_det + KL term
  KL term: kl.mode=beta -> beta * mean_b sum_{c,h,w} KL   (VA-VAE convention, kl.beta = 1e-6)
           kl.mode=lock -> RateLagrangian(kl.target) on R_hat (nats/token), lambda updated from the rank mean
Warm-up (tok_train.warmup_steps): loss = MSE(mu, mu_init) with mu_init the posterior mean of the init tokenizer
(init.ckpt); every other term is computed and multiplied by 0 so DDP sees all parameters. The decoder is
initialised from init.ckpt (post_quant_conv + decoder) and trained jointly afterwards.
Data: train split images, flipped iff rand < 0.5 drawn from Generator(noise_seed(seed, epoch, index)).

Outputs  <run_root>/<name>/tok/<arm>/{checkpoints/{step:07d}.pt, tokenizer.pt, config.yaml, log.txt, metrics.jsonl}
Resume   --resume continues the newest checkpoint (weights, optimizer, Lagrangian, step / epoch position).

Examples
  torchrun --nproc_per_node 8 scripts/v3_train_tokenizer.py --config configs/dataset_imagenet256.yaml \
      --config configs/baseline_common.yaml --config configs/v3.yaml --config configs/server_a800.yaml \
      --config configs/v3_server.yaml --arm T-c
  ... --arm L-c --set kl.target=24        (rate-locked control arm)
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import _v3
from rwvae import datasets, integration, runtime
from rwvae import tokenizer_v3 as tv3
from rwvae.channel import noise_seed

_CKPT = re.compile(r"(\d+)\.pt")
_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16}


def build_parser():
    parser = _common.make_parser("Train the RW-VAE v3 tokenizer of one arm.")
    _v3.add_arm_arg(parser)
    return parser


class RandomFlipImages(Dataset):
    """Images of a manifest split; item i in epoch e is flipped iff rand < p from Generator(noise_seed(seed, e, i))."""

    def __init__(self, manifest, split: str, image_size: int, seed: int, p: float):
        self.plain = datasets.ManifestImageDataset(manifest, split, image_size=image_size)
        self.flipped = datasets.ManifestImageDataset(manifest, split, image_size=image_size, hflip=True)
        self.seed, self.p, self.epoch = int(seed), float(p), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.plain)

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(noise_seed(self.seed, self.epoch, i))
        flip = bool(torch.rand((), generator=g) < self.p)
        return (self.flipped if flip else self.plain)[i]


def _checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    return sorted((int(m.group(1)), p) for p in (ckpt_dir.iterdir() if ckpt_dir.is_dir() else [])
                  if (m := _CKPT.fullmatch(p.name)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    rank, world = _common.rank_world()
    out_dir = Path(args.out) if args.out else _v3.tok_dir(cfg, args.arm)
    run = _common.open_run(__file__, cfg, out_dir, args) if rank == 0 else None
    try:
        final = train(cfg, args, out_dir, run, rank, world)
        if run is not None:
            run.finish("ok", arm=args.arm, tokenizer=final)
    except BaseException as e:
        if run is not None:
            run.finish("failed", error=repr(e))
        raise
    return 0


def train(cfg: dict, args, out_dir: Path, run, rank: int, world: int) -> Path:
    t, kcfg = cfg["tok_train"], cfg["kl"]
    log = run.logger if run is not None else runtime.get_logger("rwvae.v3_train_tokenizer")
    info = log.info if rank == 0 else (lambda *a, **k: None)
    device = runtime.select_device(args.device)
    own_group = False
    if world > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
            torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group("nccl" if sys.platform.startswith("linux") and device.type == "cuda" else "gloo")
            own_group = True
    try:
        return _train(cfg, t, kcfg, args, out_dir, rank, world, device, info, run)
    finally:
        if own_group:
            dist.destroy_process_group()


def _train(cfg, t, kcfg, args, out_dir, rank, world, device, info, run) -> Path:
    seed = int(t["seed"]) * world + rank
    runtime.seed_everything(seed)
    if t["global_batch_size"] % world:
        raise ValueError(f"global_batch_size={t['global_batch_size']} is not divisible by world size {world}")
    batch = int(t["global_batch_size"]) // world
    amp = device.type == "cuda" and t["mixed_precision"] in _AMP
    amp_dtype = _AMP.get(t["mixed_precision"], torch.bfloat16)

    v3cfg = tv3.V3Config.from_dict(cfg["v3"])
    init = cfg["init"]
    teacher = integration.FrozenTokenizer.from_checkpoint(_common.resolve(init["ckpt"]), embed_dim=int(init["embed_dim"]),
                                                          ch_mult=tuple(init["ch_mult"]), device=device)
    dino = _v3.make_dino(cfg, device)
    tok = tv3.TokenizerV3(v3cfg, tv3.build_decoder_ae(init["embed_dim"], init["ch_mult"]), grid=dino.grid)
    tv3.load_decoder_weights(tok, integration.load_tokenizer_state_dict(_common.resolve(init["ckpt"])))
    tok.to(device).train()
    lpips_fn = integration.load_lpips(device)
    net = DDP(tok, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else tok
    opt = torch.optim.AdamW([{"params": tok.encoder_parameters(), "lr": float(t["lr_encoder"])},
                             {"params": tok.decoder_parameters(), "lr": float(t["lr_decoder"])}],
                            betas=tuple(t["betas"]), weight_decay=float(t["weight_decay"]))
    lag = None
    if kcfg["mode"] == "lock":
        lag = tv3.RateLagrangian(float(kcfg["target"]), quad=float(kcfg["quad"]), lr=float(kcfg["lam_lr"]),
                                 every=int(kcfg["update_every"]))
    elif kcfg["mode"] != "beta":
        raise ValueError(f"kl.mode must be beta or lock, got {kcfg['mode']!r}")

    manifest = _common.load_manifest(cfg)
    data = RandomFlipImages(manifest, t["split"], int(cfg["dataset"]["image_size"]), int(t["seed"]),
                            0.5 if t["hflip"] else 0.0)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=int(t["seed"]), drop_last=True)
    steps_per_epoch = len(sampler) // batch
    if steps_per_epoch == 0:
        raise ValueError(f"{len(data)} images give fewer than one batch of {batch} per process")

    ckpt_dir = out_dir / "checkpoints"
    step = epoch = done = 0
    found = _checkpoints(ckpt_dir)
    if found and not args.resume:
        raise ValueError(f"{ckpt_dir} already holds checkpoints; pass --resume or use another --out")
    if args.resume and found:
        ckpt = torch.load(found[-1][1], map_location="cpu", weights_only=True)
        tok.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if lag is not None:
            lag.load_state_dict(ckpt["lagrangian"])
        step, epoch, done = int(ckpt["step"]), int(ckpt["epoch"]), int(ckpt["batches_done_in_epoch"])
        info("resumed %s at step %d", found[-1][1], step)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    arch = {"embed_dim": int(init["embed_dim"]), "ch_mult": [int(c) for c in init["ch_mult"]]}
    n_enc = sum(p.numel() for p in tok.encoder_parameters())
    info("arm %s | %s | trainable %.2fM encoder-side + %.2fM decoder | world %d, batch %d/process | %d steps/epoch",
         cfg["arm"], v3cfg.to_dict(), n_enc / 1e6, sum(p.numel() for p in tok.decoder_parameters()) / 1e6, world,
         batch, steps_per_epoch)

    keys = ("loss", "l1", "lpips", "sem", "det", "rate", "warm_mse", "alpha", "grad_norm")
    sums = torch.zeros(len(keys), dtype=torch.float64, device=device)
    n_win, t_win = 0, time.perf_counter()
    max_steps = int(t["max_steps"])
    while step < max_steps:
        sampler.set_epoch(epoch)
        data.set_epoch(epoch)
        order = list(sampler)[done * batch:]
        loader = DataLoader(data, batch_size=batch, sampler=order, num_workers=int(t["num_workers"]), drop_last=True,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            warm = step < int(t["warmup_steps"])
            layers, target = _v3.dino_inputs(dino, images, v3cfg, amp)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                out = net(images, layers, target)
                lp = lpips_fn(out["x_rec"].float(), images).mean()
            x_rec = out["x_rec"].float()
            l1 = (x_rec - images).abs().mean()
            mu, logvar = out["mu"].float(), out["logvar"].float()
            rate = tv3.rate_nats_per_token(mu, logvar)
            delta = None
            if lag is None:
                kl_term = float(kcfg["beta"]) * tv3.kl_elementwise(mu, logvar).sum((1, 2, 3)).mean()
            else:
                kl_term, delta = lag.loss(rate)
            zero = torch.zeros((), device=device)
            sem, det = out.get("sem", zero), out.get("det", zero)
            main = (float(t["l1_weight"]) * l1 + float(t["lpips_weight"]) * lp + float(t["w_sem"]) * sem
                    + float(t["w_det"]) * det + kl_term)
            if warm:
                with torch.no_grad():
                    mu_init = teacher.encode_mean(images)
                warm_mse = F.mse_loss(mu, mu_init.float())
                loss = warm_mse + 0.0 * main
            else:
                warm_mse, loss = zero, main
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for p in tok.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
            if t.get("grad_clip"):
                torch.nn.utils.clip_grads_with_norm_(params, float(t["grad_clip"]), grad_norm)
            opt.step()
            if lag is not None and not warm:
                d = torch.tensor(delta, device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(d)
                    d /= world
                lag.observe(float(d))
            step += 1
            done += 1
            alpha = out["alpha"].float().mean() if out["alpha"] is not None else torch.ones((), device=device)
            sums += torch.stack([loss.detach(), l1.detach(), lp.detach(), sem.detach(), det.detach(), rate.detach(),
                                 warm_mse.detach(), alpha.detach(), grad_norm]).double()
            n_win += 1
            if step % int(t["log_every"]) == 0 or step == max_steps:
                means = sums / n_win
                if world > 1:
                    dist.all_reduce(means)
                    means /= world
                metrics = dict(zip(keys, means.tolist()))
                metrics.update(warmup=warm, lam=None if lag is None else lag.lam,
                               steps_per_sec=n_win / (time.perf_counter() - t_win))
                if device.type == "cuda":
                    metrics["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                info("step %07d | %s", step, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in metrics.items()))
                if run is not None:
                    run.log_metrics(step, **metrics)
                sums.zero_()
                n_win, t_win = 0, time.perf_counter()
            if step % int(t["ckpt_every"]) == 0 or step == max_steps:
                if rank == 0:
                    path = ckpt_dir / f"{step:07d}.pt"
                    tv3.save_tokenizer(path, tok, decoder_arch=arch, extra={
                        "opt": opt.state_dict(), "step": step, "epoch": epoch, "batches_done_in_epoch": done,
                        "lagrangian": None if lag is None else lag.state_dict(), "arm": cfg["arm"]})
                    for _, old in _checkpoints(ckpt_dir)[:-int(t["keep_last"])]:
                        old.unlink()
                if world > 1:
                    dist.barrier()
            if step >= max_steps:
                break
        else:
            epoch, done = epoch + 1, 0

    final = out_dir / "tokenizer.pt"
    if rank == 0:
        tv3.save_tokenizer(final, tok, decoder_arch=arch, extra={"step": step, "arm": cfg["arm"]})
        info("tokenizer saved to %s", final)
    return final


if __name__ == "__main__":
    raise SystemExit(main())
