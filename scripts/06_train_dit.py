"""06 -- train LightningDiT on ONE arm's channel latents (plan v2.1 Sec. 4, Sec. 6 M5, Sec. 8).

All arms share the DiT, the training recipe, the cached posterior means, the normalisation and every
seed; they differ only in the channel z' = channel(mu) of their allocation (B0 trains on mu itself).
A lossy arm (B1-B3) may only be trained at the frozen pre-registered operating point (R*, w*) of
configs/arms.yaml; --exploratory runs anywhere else and is marked as exploratory in arm.json and in the
run registry (plan Sec. 8: R*, w* come from the generator-free M3 metrics, never from gFID).

Inputs   manifest (01), the latent cache of cfg.train.split (02), stats.npz (03), the arm's allocation (04;
         built on the fly for B0).
Outputs  (default <out> = <run_root>/<name>/dit/<arm label>, e.g. .../dit/B3_r2_R22_w2)
         <out>/checkpoints/{step:07d}.pt  rwvae.generation.train_dit checkpoints (model, EMA, optimizer, RNG
                                          states; their meta = the identity block of arm.json)
         <out>/arm.json                   arm, rule, R, w, allocation / stats / cache fingerprints, normalisation,
                                          pre-registration record, noise/flip protocol, world size, exploratory flag
         <out>/config.yaml, log.txt, metrics.jsonl, summary.json  (rwvae.runtime.Run)
Data     rwvae.datasets.ChannelLatentDataset: item `index` in epoch e draws u, then eps, from
         Generator(noise_seed(train.noise_seed, e if train.resample_noise_each_epoch else 0, index)); the
         flipped latent is used iff u < flip_prob, where flip_prob = train.flip_prob, or 0 (logged and recorded)
         when the cache stores no flipped latents; x = (z' - mean) / std * train.latent_multiplier with
         (mean, std) = stats.channel_norm().
Resume   --resume continues the newest checkpoint exactly; arm.json must describe the same arm and data. To
         extend a finished run use --resume --overwrite --set train.max_steps=N (a Run accepts a changed
         config only with --overwrite; train_dit still refuses changes of lr, batch size, seed, ...).
Multi-GPU: torchrun on the Linux node; only rank 0 writes the run directory and arm.json. The Windows
         build of torch 2.12 cannot start torchrun (no libuv), so the local runs use one process.

Examples (the common flags of rwvae.runtime.add_common_args apply: --set, --out, --device, --seed, ...)
  ImageNet-1k (GPU node):
    python scripts/06_train_dit.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --arm B0
    torchrun --nproc_per_node 8 scripts/06_train_dit.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --arm B3 --R 22 --w 2
  Local Imagenette (engineering test only):
    python scripts/06_train_dit.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/local_imagenette.yaml --arm B3 --R 22 --w 2 --exploratory
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import json
from dataclasses import asdict
from pathlib import Path

import torch

from rwvae import datasets, generation, integration, runtime
from rwvae.channel import RWChannel

PROTOCOL = ("item `index` in epoch e: g = torch.Generator().manual_seed(noise_seed(base_seed, "
            "e if resample_each_epoch else 0, index)); u = torch.rand((), generator=g), the flipped latent is "
            "used iff u < flip_prob; eps = torch.randn((C, H, W), generator=g) (not drawn for B0); "
            "z' = channel(mu, noise=eps); x = (z' - mean) / std * latent_multiplier")


def build_parser():
    parser = _common.make_parser("Train LightningDiT on one arm's channel latents (plan Sec. 4, M5).")
    parser.add_argument("--arm", required=True, help="arm label from configs/arms.yaml (B0, B1, B2, B3)")
    parser.add_argument("--R", type=float, default=None, help="nominal rate in nats/token (lossy arms)")
    parser.add_argument("--w", type=float, default=None, help="relevance weight (rule r2 only)")
    parser.add_argument("--exploratory", action="store_true",
                        help="allow a lossy arm off the frozen pre-registered (R*, w*); recorded as exploratory")
    return parser


def arm_point(arm: str, rule: str, R, w) -> tuple[float | None, float | None]:
    """(R, w) as used in names and records; a flag that does not apply to the rule is an error."""
    if rule == "none":
        if R is not None or w is not None:
            raise ValueError(f"arm {arm} (rule none) is the rate-free reference: pass neither --R nor --w")
        return None, None
    if R is None or (rule == "r2" and w is None):
        raise ValueError(f"arm {arm} (rule {rule}) needs --R" + (" and --w" if rule == "r2" else ""))
    if rule != "r2" and w is not None:
        raise ValueError(f"--w applies to rule r2 only; arm {arm} has rule {rule}")
    return float(R), None if w is None else float(w)


def load_inputs(cfg: dict, arm: str, rule: str, R, w) -> tuple[datasets.ChannelLatentDataset, dict]:
    """Read-only: the arm's training dataset and its provenance record (the body of arm.json)."""
    t = cfg["train"]
    manifest = _common.load_manifest(cfg)
    cache = _common.open_cache(cfg, t["split"], manifest)
    stats = _common.load_stats(cfg)
    alloc = _common.load_allocation(cfg, arm, R, w)
    name = _common.allocation_name(rule, R, w)
    if alloc.name != name:
        raise ValueError(f"the allocation loaded for arm {arm} is {alloc.name!r}, expected {name!r}")
    solved_from = alloc.meta.get("stats_fingerprint")
    if solved_from is not None and solved_from != stats.fingerprint():
        raise ValueError(f"allocation {name} was solved from stats {solved_from}, but {_common.stats_path(cfg)} "
                         f"has fingerprint {stats.fingerprint()}: re-run scripts/04_solve_channel.py")
    mean, std = stats.channel_norm()
    flip_prob = float(t["flip_prob"]) if cache.has_flip else 0.0
    dataset = datasets.ChannelLatentDataset(
        cache, None if rule == "none" else RWChannel.from_allocation(alloc), mean, std,
        base_seed=int(t["noise_seed"]), flip_prob=flip_prob,
        resample_each_epoch=bool(t["resample_noise_each_epoch"]), latent_multiplier=float(t["latent_multiplier"]))

    alloc_file = None if rule == "none" else _common.allocations_dir(cfg) / f"{name}.npz"
    index_file = cache.cache_dir / datasets.CACHE_INDEX
    stats_file = _common.stats_path(cfg)
    info = {
        "arm": arm, "rule": rule, "R": R, "w": w, "label": _common.arm_label(arm, rule, R, w),
        "allocation": {"name": name, "fingerprint": alloc.fingerprint(), "stats_fingerprint": solved_from,
                       "file": alloc_file, "sha256": None if alloc_file is None else runtime.sha256_file(alloc_file),
                       "source": "solve_allocation(stats, 'none') built on the fly" if alloc_file is None
                       else "04_solve_channel.py"},
        "stats": {"file": stats_file, "sha256": runtime.sha256_file(stats_file), "fingerprint": stats.fingerprint(),
                  "n_images": stats.n_images, "n_tokens": stats.n_tokens},
        "data": {"split": cache.split, "cache_dir": cache.cache_dir, "cache_index_sha256": runtime.sha256_file(index_file),
                 "cache_meta": cache.meta, "num_items": len(cache), "dtype": cache.dtype, "has_flip": cache.has_flip,
                 "latent_shape": cache.latent_shape, "manifest_fingerprint": manifest.fingerprint(),
                 "num_classes": len(manifest.classes)},
        "normalisation": {"mean": mean, "std": std, "latent_multiplier": float(t["latent_multiplier"]),
                          "source": "LatentStats.channel_norm() = (m, sqrt(diag Sigma)), shared by all arms"},
        "noise_protocol": {"base_seed": int(t["noise_seed"]), "flip_prob": flip_prob,
                           "flip_prob_config": float(t["flip_prob"]), "cache_has_flip": cache.has_flip,
                           "resample_each_epoch": bool(t["resample_noise_each_epoch"]), "rule": PROTOCOL},
    }
    return dataset, runtime.to_jsonable(info)


def identity(info: dict) -> dict:
    """What a resumed run must share with the checkpointed one; stored as the meta of every checkpoint."""
    return {"arm": info["arm"], "rule": info["rule"], "R": info["R"], "w": info["w"],
            "allocation_fingerprint": info["allocation"]["fingerprint"],
            "stats_fingerprint": info["stats"]["fingerprint"],
            "cache_index_sha256": info["data"]["cache_index_sha256"],
            "manifest_fingerprint": info["data"]["manifest_fingerprint"],
            "noise_protocol": info["noise_protocol"], "normalisation": info["normalisation"],
            "exploratory": info["exploratory"], "preregistration": info["preregistration"]}


def check_resume(arm_json: Path, info: dict) -> None:
    """--resume must continue the same arm on the same data under the same pre-registration status."""
    if not arm_json.is_file():
        return
    old = identity(json.loads(arm_json.read_text(encoding="utf-8")))
    new = identity(info)
    changed = sorted(k for k in new if old.get(k) != new[k])
    if changed:
        raise ValueError(f"cannot resume {arm_json.parent}: {changed} differ from its arm.json "
                         f"(old {[old.get(k) for k in changed]}, new {[new[k] for k in changed]})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    rule = _common.arm_rule(cfg, args.arm)
    R, w = arm_point(args.arm, rule, args.R, args.w)
    prereg = _common.preregistration_check(cfg, args.arm, R, w, args.exploratory)  # raises when unregistered
    label = _common.arm_label(args.arm, rule, R, w)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "dit", label)
    rank, world = _common.rank_world()

    dataset, info = load_inputs(cfg, args.arm, rule, R, w)
    info.update(exploratory=bool(args.exploratory), preregistration=prereg, world_size=world)
    if args.resume:
        check_resume(out_dir / "arm.json", info)

    run = _common.open_run(__file__, cfg, out_dir, args) if rank == 0 else None
    logger = run.logger if run is not None else runtime.get_logger(f"rwvae.06_train_dit.rank{rank}")
    try:
        seed = int(cfg.get("seed", 0))
        runtime.seed_everything(seed)          # the DiT initialisation: identical for every arm
        device = runtime.select_device(args.device)
        noise = info["noise_protocol"]
        if noise["flip_prob"] != noise["flip_prob_config"]:
            logger.warning("train.flip_prob=%g but the %s cache stores no flipped latents: training with "
                           "flip_prob=0 (recorded in arm.json)", noise["flip_prob_config"], info["data"]["split"])
        logger.info("arm %s (rule %s, R=%s, w=%s), %s; %d training items, world size %d", label, rule, R, w,
                    "EXPLORATORY" if args.exploratory else "confirmatory (pre-registered)",
                    info["data"]["num_items"], world)
        tcfg = _common.dit_train_config(cfg)
        info.update(init_seed=seed, device=str(device), train=asdict(tcfg), dit=cfg["dit"],
                    transport=cfg["transport"], git_commit=runtime.git_commit(),
                    tf32={"tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                          "tf32_cudnn": torch.backends.cudnn.allow_tf32})  # torch defaults, as LightningDiT train.py
        if run is not None:
            run.save_json("arm.json", info)

        model = _common.build_dit(cfg, num_classes=info["data"]["num_classes"])
        transport = integration.build_transport(cfg["transport"])
        log_fn = None if run is None else (lambda step, metrics: run.log_metrics(step, **metrics))
        final = generation.train_dit(model, transport, dataset, tcfg, out_dir, device=device, logger=logger,
                                     resume=args.resume, log_fn=log_fn, meta=identity(info))
        if run is not None:
            run.finish("ok", arm=label, final_checkpoint=final, max_steps=tcfg.max_steps, world_size=world,
                       exploratory=bool(args.exploratory), registered=prereg["registered"])
    except BaseException as e:
        if run is not None:
            run.finish("failed", error=repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
