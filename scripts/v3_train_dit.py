"""v3 -- train LightningDiT on a v3 tokenizer's posterior, optionally with the REPA semantic anchor (§2.3).

Data: rwvae.datasets.PosteriorLatentDataset over the arm's mu / logvar caches of cfg.train.split: a fresh
posterior sample z = mu + exp(logvar/2) eps per (epoch, index) from Generator(noise_seed(v3_dit.noise_seed, e, i)),
flipped view iff rand < v3_dit.flip_prob, x = (z - m) / s with (m, s) from the arm's stats.json. With --repa the
item also carries the image (same view) and the DiT block repa.depth is aligned to the deepest DINO block the
tokenizer uses (DINOv2 final-norm patch tokens, computed online in bf16). Small splits are repeated so an epoch
has at least v3_dit.min_steps_per_epoch steps. Training recipe: cfg.train / cfg.dit / cfg.transport (as 06).

Outputs  <run_root>/<name>/dit/<arm>[_repa]/{checkpoints/, v3.json, config.yaml, log.txt, metrics.jsonl}

Example
  torchrun --nproc_per_node 8 scripts/v3_train_dit.py <configs...> --arm T-c [--repa]
"""
from __future__ import annotations

import _common  # noqa: I001

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

import _v3
from rwvae import datasets, generation, integration, runtime


def build_parser():
    parser = _common.make_parser("Train LightningDiT on a v3 tokenizer's posterior samples.")
    _v3.add_arm_arg(parser)
    parser.add_argument("--repa", action="store_true", help="add the REPA semantic anchor (cfg.repa)")
    return parser


class Repeated(Dataset):
    """`dataset` repeated k times per epoch; pass r of epoch e uses the inner epoch e * k + r."""

    def __init__(self, dataset, k: int):
        self.dataset, self.k, self.n, self.epoch = dataset, int(k), len(dataset), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.k * self.n

    def __getitem__(self, i: int):
        self.dataset.set_epoch(self.epoch * self.k + i // self.n)
        return self.dataset[i % self.n]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    rank, world = _common.rank_world()
    out_dir = Path(args.out) if args.out else _v3.dit_dir(cfg, args.arm, args.repa)
    run = _common.open_run(__file__, cfg, out_dir, args) if rank == 0 else None
    try:
        logger = run.logger if run is not None else runtime.get_logger("rwvae.v3_train_dit")
        device = runtime.select_device(args.device)
        tcfg = _common.dit_train_config(cfg)
        split = cfg["train"]["split"]
        manifest = _common.load_manifest(cfg)
        mu_dir, lv_dir = _v3.cache_dirs(cfg, args.arm, split)
        mu_cache = datasets.LatentCache(mu_dir, manifest=manifest)
        lv_cache = datasets.LatentCache(lv_dir, manifest=manifest)
        stats = _v3.load_stats(cfg, args.arm)
        if stats["tokenizer_sha256"] != mu_cache.meta["tokenizer_sha256"]:
            raise ValueError("stats.json and the cache come from different tokenizers: re-run v3_cache_posterior.py")
        d = cfg["v3_dit"]
        flip_prob = float(d["flip_prob"]) if mu_cache.has_flip else 0.0
        dataset = datasets.PosteriorLatentDataset(mu_cache, lv_cache, stats["mean"], stats["std"],
                                                  base_seed=int(d["noise_seed"]), flip_prob=flip_prob,
                                                  manifest=manifest if args.repa else None,
                                                  image_size=int(cfg["dataset"]["image_size"]))
        n = len(dataset)
        repeat = max(1, math.ceil(int(d["min_steps_per_epoch"]) * tcfg.global_batch_size / n))
        if repeat > 1:
            dataset = Repeated(dataset, repeat)

        repa, target_fn = None, None
        if args.repa:
            from rwvae.tokenizer_v3 import V3Config
            repa = generation.REPAConfig(**cfg["repa"])
            dino = _v3.make_dino(cfg, "cpu")        # moved to the rank's device on the first call
            target_layer = V3Config.from_dict(mu_cache.meta["v3_config"]).target_layer

            def target_fn(images):
                with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
                    return dino.to(images.device).forward_layers(images, [target_layer])[:, 0].float()

        latent_shape = list(mu_cache.latent_shape)
        info = {"arm": args.arm, "repa": cfg["repa"] if args.repa else None,
                "tokenizer": mu_cache.meta["tokenizer"], "tokenizer_sha256": mu_cache.meta["tokenizer_sha256"],
                "normalisation": {"mean": stats["mean"], "std": stats["std"], "latent_multiplier": 1.0},
                "data": {"split": split, "num_classes": len(manifest.classes), "latent_shape": latent_shape,
                         "items": n, "repeat": repeat, "flip_prob": flip_prob, "noise_seed": int(d["noise_seed"])},
                "measured_rate_nats_per_token": stats["rate_nats_per_token"], "world_size": world,
                "git_commit": runtime.git_commit()}
        if run is not None:
            run.save_json("v3.json", info)
        model = _common.build_dit(cfg, num_classes=len(manifest.classes))
        transport = integration.build_transport(cfg["transport"])
        log_fn = None if run is None else (lambda step, metrics: run.log_metrics(step, **metrics))
        final = generation.train_dit(model, transport, dataset, tcfg, out_dir, device=device, logger=logger,
                                     resume=args.resume, log_fn=log_fn,
                                     meta={"arm": args.arm, "repa": bool(args.repa),
                                           "tokenizer_sha256": info["tokenizer_sha256"]},
                                     repa=repa, repa_target_fn=target_fn)
        if run is not None:
            run.finish("ok", arm=args.arm, repa=bool(args.repa), final_checkpoint=final, repeat=repeat)
    except BaseException as e:
        if run is not None:
            run.finish("failed", error=repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
