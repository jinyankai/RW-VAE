"""v3 -- cache the posterior (mu, logvar) of a trained v3 tokenizer and its DiT normalisation statistics.

For every split in v3_cache.splits: shards of the split rows (plain and, with v3_cache.flip, flipped crops) are
encoded and written as two rwvae.datasets latent caches, <v3_cache.dir>/<arm>/mu/<split> and .../logvar/<split>
(same rows, labels, shard layout; logvar in the cache dtype). Under torchrun each rank writes the shards
k with k % world == rank; rank 0 finalizes. Existing complete shards are skipped (resumable).

Then, on rank 0, stats.json from the stats split (both views when cached):
  m_c = E[mu_c],  s_c^2 = E[mu_c^2 + exp(logvar_c)] - m_c^2      (the posterior-sample variance)
  rate_nats_per_token = E[sum_c KL_c] per split (the measured R_hat that the plan reports for every arm).

Example
  torchrun --nproc_per_node 8 scripts/v3_cache_posterior.py <configs...> --arm T-c
"""
from __future__ import annotations

import _common  # noqa: I001

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.utils.data import DataLoader

import _v3
from rwvae import datasets, runtime
from rwvae import tokenizer_v3 as tv3


def build_parser():
    parser = _common.make_parser("Cache the posterior of a v3 tokenizer and its DiT normalisation statistics.")
    _v3.add_arm_arg(parser)
    parser.add_argument("--tokenizer", default=None, help="tokenizer.pt (default: the arm's tok run)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    rank, world = _common.rank_world()
    root = _v3.cache_root(cfg, args.arm)
    run = _common.open_run(__file__, cfg, Path(args.out) if args.out else root / "run", args) if rank == 0 else None
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
        _common.set_tf32(False)
        summary = cache(cfg, args, root, rank, world, device, run)
        if run is not None:
            run.finish("ok", **summary)
    except BaseException as e:
        if run is not None:
            run.finish("failed", error=repr(e))
        raise
    finally:
        if own_group:
            dist.destroy_process_group()
    return 0


@torch.no_grad()
def _encode(tok, dino, images, amp: bool):
    layers, _ = _v3.dino_inputs(dino, images, tok.cfg, amp)
    out = tok.encode(images, layers)
    return out["mu"].float(), out["logvar"].float()


def cache(cfg, args, root: Path, rank: int, world: int, device, run) -> dict:
    c = cfg["v3_cache"]
    log = (run.logger.info if run is not None else (lambda *a, **k: None))
    tok_path = Path(args.tokenizer) if args.tokenizer else _v3.tokenizer_path(cfg, args.arm)
    tok, ckpt = tv3.load_tokenizer(tok_path, device)
    dino = _v3.make_dino(cfg, device)
    amp = device.type == "cuda"
    manifest = _common.load_manifest(cfg)
    meta = {"manifest_fingerprint": manifest.fingerprint(), "tokenizer": str(tok_path.resolve()),
            "tokenizer_sha256": runtime.sha256_file(tok_path), "arm": args.arm, "v3_config": tok.cfg.to_dict(),
            "preprocessing": datasets.PREPROCESSING, "flip": bool(c["flip"])}
    size = int(cfg["dataset"]["image_size"])
    for split in c["splits"]:
        total = manifest.split_size(split)
        mu_dir, lv_dir = _v3.cache_dirs(cfg, args.arm, split)
        writers = [datasets.LatentCacheWriter(d, split=split, meta={**meta, "field": f}, dtype=c["dtype"],
                                              shard_size=int(c["shard_size"]))
                   for d, f in ((mu_dir, "mu"), (lv_dir, "logvar"))]
        for k in range(writers[0].num_shards(total)):
            if k % world != rank:
                continue
            rows = list(writers[0].shard_rows(k, total))
            if all(w.shard_done(k, len(rows)) for w in writers):
                continue
            views = [False, True] if c["flip"] else [False]
            parts = {v: ([], []) for v in views}
            labels = []
            for v in views:
                ds = datasets.ManifestImageDataset(manifest, split, image_size=size, hflip=v, indices=rows)
                for images, y, _ in DataLoader(ds, batch_size=int(c["batch_size"]), num_workers=int(c["num_workers"])):
                    mu, lv = _encode(tok, dino, images.to(device), amp)
                    parts[v][0].append(mu.cpu())
                    parts[v][1].append(lv.cpu())
                    if not v:
                        labels.append(y)
            y = torch.cat(labels)
            for w, j in zip(writers, (0, 1)):
                w.write_shard(k, torch.cat(parts[False][j]), y, rows,
                              latents_flip=torch.cat(parts[True][j]) if c["flip"] else None)
            log("split %s shard %d (%d rows) written", split, k, len(rows))
        if world > 1:
            dist.barrier()
        if rank == 0:
            for w in writers:
                w.finalize(total)
        if world > 1:
            dist.barrier()

    if rank != 0:
        return {}
    stats = {"arm": args.arm, "tokenizer_sha256": meta["tokenizer_sha256"], "stats_split": c["stats_split"],
             "rate_nats_per_token": {}}
    for split in c["splits"]:
        m = _moments(*_v3.cache_dirs(cfg, args.arm, split), device)
        stats["rate_nats_per_token"][split] = m["rate"]
        if split == c["stats_split"]:
            stats.update(mean=m["mean"], std=m["std"], n_positions=m["n"])
    if "mean" not in stats:
        raise ValueError(f"v3_cache.stats_split={c['stats_split']!r} is not among v3_cache.splits")
    path = _v3.stats_path(cfg, args.arm)
    path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log("stats -> %s; measured rate (nats/token) %s", path, stats["rate_nats_per_token"])
    return {"stats": path, "rate_nats_per_token": stats["rate_nats_per_token"]}


def _moments(mu_dir: Path, lv_dir: Path, device) -> dict:
    """float64 per-channel E[mu], E[mu^2 + exp(logvar)] and the mean KL per token over every cached view."""
    index = json.loads((mu_dir / datasets.CACHE_INDEX).read_text(encoding="utf-8"))
    keys = ["latents"] + (["latents_flip"] if index["has_flip"] else [])
    C = int(index["latent_shape"][0])
    s1 = torch.zeros(C, dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    kl, n = 0.0, 0
    for shard in index["shards"]:
        with safe_open(str(mu_dir / shard["file"]), "pt", device="cpu") as fm, \
                safe_open(str(lv_dir / shard["file"]), "pt", device="cpu") as fl:
            for key in keys:
                mu = fm.get_tensor(key).to(device, torch.float64)
                lv = fl.get_tensor(key).to(device, torch.float64)
                s1 += mu.sum((0, 2, 3))
                s2 += (mu * mu + lv.exp()).sum((0, 2, 3))
                kl += float(tv3.kl_elementwise(mu, lv).sum())
                n += mu.shape[0] * mu.shape[2] * mu.shape[3]
    mean = s1 / n
    var = s2 / n - mean * mean
    return {"mean": mean.tolist(), "std": var.clamp_min(1e-12).sqrt().tolist(), "n": n, "rate": kl / n}


if __name__ == "__main__":
    raise SystemExit(main())
