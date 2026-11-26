"""v3 -- decoder-only GAN fine-tune of a trained v3 tokenizer (RWVAE_V3_ALLOCATION_PLAN.md §2.4).

The encoder (and so the latent, the posterior cache and every DiT trained on it) is untouched: only
post_quant_conv + decoder are trained, from the arm's tokenizer.pt, on fresh posterior samples
z = mu + exp(logvar/2) eps read from the arm's cache (rwvae.datasets.PosteriorLatentDataset with the paired image
in the same flip view), i.e. the distribution the DiT learns.

Generator loss  l1 * |x_rec - x| + lpips * LPIPS(x_rec, x) + w * (-mean D(x_rec)),
                w = adaptive_gan_weight(...) at decoder.conv_out.weight times dec_gan.disc_weight (VA-VAE's
                LPIPSWithDiscriminator rule), switched on at dec_gan.disc_warmup steps.
Discriminator   PatchGAN (rwvae.gan, ported from taming), hinge loss on x vs x_rec.detach(), trained from step 0
                so it is already informative when the adversarial term starts.

Outputs  <run_root>/<name>/tok/<arm>/gan/{checkpoints/, config.yaml, log.txt, metrics.jsonl}
         <run_root>/<name>/tok/<arm>/tokenizer_gan.pt  (same encoder weights + "encoder_from_sha256" of tokenizer.pt,
         which v3_evaluate accepts in place of the tokenizer a DiT was trained on: pass --tokenizer)

Example
  torchrun --nproc_per_node 8 scripts/v3_finetune_decoder.py <configs...> --arm T-c
"""
from __future__ import annotations

import _common  # noqa: I001

import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import _v3
from rwvae import datasets, gan, integration, runtime
from rwvae import tokenizer_v3 as tv3

_CKPT = re.compile(r"(\d+)\.pt")
_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16}


def build_parser():
    parser = _common.make_parser("Decoder-only GAN fine-tune of a v3 tokenizer (encoder frozen).")
    _v3.add_arm_arg(parser)
    return parser


class _Decoder(torch.nn.Module):
    """post_quant_conv + decoder of a TokenizerV3, as one module for DDP."""

    def __init__(self, tok: tv3.TokenizerV3):
        super().__init__()
        self.post_quant_conv, self.decoder = tok.post_quant_conv, tok.decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


def _checkpoints(d: Path):
    return sorted((int(m.group(1)), p) for p in (d.iterdir() if d.is_dir() else []) if (m := _CKPT.fullmatch(p.name)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    rank, world = _common.rank_world()
    out_dir = Path(args.out) if args.out else _v3.tok_dir(cfg, args.arm) / "gan"
    run = _common.open_run(__file__, cfg, out_dir, args) if rank == 0 else None
    own_group = False
    try:
        device = runtime.select_device(args.device)
        if world > 1:
            if device.type == "cuda":
                device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
                torch.cuda.set_device(device)
            if not dist.is_initialized():
                dist.init_process_group("nccl" if sys.platform.startswith("linux") and device.type == "cuda" else "gloo")
                own_group = True
        final = finetune(cfg, args, out_dir, run, rank, world, device)
        if run is not None:
            run.finish("ok", arm=args.arm, tokenizer=final)
    except BaseException as e:
        if run is not None:
            run.finish("failed", error=repr(e))
        raise
    finally:
        if own_group:
            dist.destroy_process_group()
    return 0


def finetune(cfg, args, out_dir: Path, run, rank: int, world: int, device) -> Path:
    g = cfg["dec_gan"]
    info = run.logger.info if run is not None else (lambda *a, **k: None)
    runtime.seed_everything(int(g["seed"]) * world + rank)
    if int(g["global_batch_size"]) % world:
        raise ValueError(f"dec_gan.global_batch_size={g['global_batch_size']} is not divisible by {world}")
    batch = int(g["global_batch_size"]) // world
    amp = device.type == "cuda" and g["mixed_precision"] in _AMP
    amp_dtype = _AMP.get(g["mixed_precision"], torch.bfloat16)

    base_path = _v3.tokenizer_path(cfg, args.arm)
    tok, base_ckpt = tv3.load_tokenizer(base_path, device)
    base_sha = runtime.sha256_file(base_path)
    tok.requires_grad_(False)
    dec = _Decoder(tok).requires_grad_(True).train()
    disc = gan.NLayerDiscriminator().to(device).train()
    lpips_fn = integration.load_lpips(device)
    dec_net = DDP(dec, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else dec
    disc_net = DDP(disc, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else disc
    opt_g = torch.optim.Adam(dec.parameters(), lr=float(g["lr_decoder"]), betas=tuple(g["betas"]))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(g["lr_disc"]), betas=tuple(g["betas"]))
    last_layer = tok.decoder.conv_out.weight

    manifest = _common.load_manifest(cfg)
    mu_dir, lv_dir = _v3.cache_dirs(cfg, args.arm, g["split"])
    mu_cache = datasets.LatentCache(mu_dir, manifest=manifest)
    lv_cache = datasets.LatentCache(lv_dir, manifest=manifest)
    if mu_cache.meta["tokenizer_sha256"] != base_sha:
        raise ValueError(f"the cache in {mu_dir} was written by another tokenizer than {base_path}")
    stats = _v3.load_stats(cfg, args.arm)
    data = datasets.PosteriorLatentDataset(mu_cache, lv_cache, stats["mean"], stats["std"],
                                           base_seed=int(g["noise_seed"]), flip_prob=0.5 if mu_cache.has_flip else 0.0,
                                           manifest=manifest, image_size=int(cfg["dataset"]["image_size"]))
    mean = torch.tensor(stats["mean"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, -1, 1, 1)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=int(g["seed"]), drop_last=True)
    if len(sampler) // batch == 0:
        raise ValueError(f"{len(data)} items give fewer than one batch of {batch} per process")

    ckpt_dir = out_dir / "checkpoints"
    step = epoch = done = 0
    found = _checkpoints(ckpt_dir)
    if found and not args.resume:
        raise ValueError(f"{ckpt_dir} already holds checkpoints; pass --resume or use another --out")
    if args.resume and found:
        ck = torch.load(found[-1][1], map_location="cpu", weights_only=True)
        dec.load_state_dict(ck["decoder"])
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step, epoch, done = int(ck["step"]), int(ck["epoch"]), int(ck["batches_done_in_epoch"])
        info("resumed %s at step %d", found[-1][1], step)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    info("decoder GAN fine-tune of %s (arm %s): %.2fM decoder + %.2fM discriminator parameters, world %d, batch %d"
         "/process, adversarial term from step %d", base_path, args.arm, sum(p.numel() for p in dec.parameters()) / 1e6,
         sum(p.numel() for p in disc.parameters()) / 1e6, world, batch, int(g["disc_warmup"]))

    keys = ("rec", "l1", "lpips", "g_loss", "d_weight", "d_loss", "logits_real", "logits_fake")
    sums = torch.zeros(len(keys), dtype=torch.float64, device=device)
    n_win, t_win, max_steps = 0, time.perf_counter(), int(g["max_steps"])
    while step < max_steps:
        sampler.set_epoch(epoch)
        data.set_epoch(epoch)
        loader = DataLoader(data, batch_size=batch, sampler=list(sampler)[done * batch:], drop_last=True,
                            num_workers=int(g["num_workers"]), pin_memory=device.type == "cuda")
        for x_norm, _, _, images in loader:
            images = images.to(device, non_blocking=True)
            z = x_norm.to(device, non_blocking=True) * std + mean
            adv_on = step >= int(g["disc_warmup"])
            # generator (decoder) step; the discriminator is frozen and used unwrapped
            disc.requires_grad_(False)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                x_rec = dec_net(z)
                lp = lpips_fn(x_rec.float(), images).mean()
            x_rec = x_rec.float()
            l1 = (x_rec - images).abs().mean()
            rec = float(g["l1_weight"]) * l1 + float(g["lpips_weight"]) * lp
            if adv_on:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    g_loss = -disc(x_rec).float().mean()
                d_weight = gan.adaptive_gan_weight(rec, g_loss, last_layer, float(g["disc_weight"]))
                loss_g = rec + d_weight * g_loss
            else:
                g_loss = d_weight = torch.zeros((), device=device)
                loss_g = rec
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            # discriminator step
            disc.requires_grad_(True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                logits_real = disc_net(images).float()
                logits_fake = disc_net(x_rec.detach()).float()
            d_loss = gan.hinge_d_loss(logits_real, logits_fake)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            step += 1
            done += 1
            sums += torch.stack([rec.detach(), l1.detach(), lp.detach(), g_loss.detach(), d_weight.detach(),
                                 d_loss.detach(), logits_real.detach().mean(), logits_fake.detach().mean()]).double()
            n_win += 1
            if step % int(g["log_every"]) == 0 or step == max_steps:
                means = sums / n_win
                if world > 1:
                    dist.all_reduce(means)
                    means /= world
                metrics = dict(zip(keys, means.tolist()))
                metrics.update(adversarial=adv_on, steps_per_sec=n_win / (time.perf_counter() - t_win))
                info("step %07d | %s", step, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in metrics.items()))
                if run is not None:
                    run.log_metrics(step, **metrics)
                sums.zero_()
                n_win, t_win = 0, time.perf_counter()
            if step % int(g["ckpt_every"]) == 0 or step == max_steps:
                if rank == 0:
                    path = ckpt_dir / f"{step:07d}.pt"
                    tmp = path.with_name(path.name + ".tmp")
                    torch.save({"decoder": dec.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                                "opt_d": opt_d.state_dict(), "step": step, "epoch": epoch,
                                "batches_done_in_epoch": done}, tmp)
                    os.replace(tmp, path)
                    for _, old in _checkpoints(ckpt_dir)[:-int(g["keep_last"])]:
                        old.unlink()
                if world > 1:
                    dist.barrier()
            if step >= max_steps:
                break
        else:
            epoch, done = epoch + 1, 0

    final = _v3.tok_dir(cfg, args.arm) / "tokenizer_gan.pt" if not args.out else out_dir / "tokenizer_gan.pt"
    if rank == 0:
        tv3.save_tokenizer(final, tok, decoder_arch=base_ckpt["decoder_arch"],
                           extra={"arm": args.arm, "step": step, "encoder_from_sha256": base_sha,
                                  "decoder_finetune": {"kind": "gan", **{k: g[k] for k in g}}})
        info("tokenizer with the GAN-fine-tuned decoder -> %s", final)
    return final


if __name__ == "__main__":
    raise SystemExit(main())
