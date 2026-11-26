"""02 -- posterior-mean latent cache of the manifest splits (plan v2.1 Sec. 1.3, Sec. 3.2, Sec. 6 M2).

For each split, mu = FrozenTokenizer.encode_mean(image) (posterior mean, never a posterior sample) is
computed in float32 with TF32 off and stored by rwvae.datasets.LatentCacheWriter as deterministic
safetensors shards (shard k = split rows [k*S, min((k+1)*S, N)), S = cache.shard_size) in
cfg.cache.dir/<split>/. With cache.flip, latents_flip = encode_mean(torch.flip(image, dims=[-1])) of the
same center-cropped image, which equals the hflip transform applied after the center crop (LightningDiT's
order; tested in tests/test_scripts_data.py).

Parallelism / resume: process k of N (--num-procs N --proc-id k, or torchrun's RANK / WORLD_SIZE when the
flags are absent) writes the shards with id % N == k; with several visible CUDA devices it uses
cuda:(LOCAL_RANK or k % device_count). Shards already on disk are skipped (LatentCacheWriter.shard_done),
so an interrupted run is continued by running the same command again (add --resume for the Run directory).
The shard meta must be identical across processes and resumed runs (it includes the git commit), so the
creation time is recorded in the run summary, cache_verification.json and DATA_CARD, not in the shards.
When every shard of a split exists, process 0 (or a --finalize call) runs LatentCacheWriter.finalize and
then re-encodes cache.verify_samples seeded random rows and compares them with the cache: the error must be
within storage rounding, |d| <= u_dtype * |mu| + 1e-4 * max|mu| (u = 2^-11 fp16, 2^-8 bf16, 0 fp32; the
second term allows the fp32 re-encode to differ in batch composition / device), else the script fails.
With N processes, process 0 usually finishes before the others: run one --finalize call afterwards.

Inputs   the manifest (01), cfg.tokenizer (ckpt, embed_dim, ch_mult), cfg.cache (dir, splits, dtype, flip,
         shard_size, batch_size, num_workers, verify_samples), cfg.dataset.image_size.
Outputs  cfg.cache.dir/<split>/{<split>_shard#####.safetensors, cache_index.json, cache_verification.json}
         <run_root>/<name>/DATA_CARD.md section "## 2. Latent cache" (all finalized splits; others kept)
         run dir <run_root>/<name>/02_cache/<splits joined by +>[/proc<k>of<N>][/finalize]:
         cache_info.json (per split: shards written/skipped, finalize + verification), Run files.
Flags    --split S (repeatable; default cache.splits), --num-procs N, --proc-id k, --finalize (finalize and
         verify only), --no-verify.

Examples
  ImageNet-1k (GPU node, 8 GPUs; then one --finalize call):
    torchrun --nproc_per_node 8 scripts/02_cache_latents.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/server_a800.yaml
    python scripts/02_cache_latents.py <same configs> --finalize
  SVP:
    torchrun --nproc_per_node 8 scripts/02_cache_latents.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/server_a800.yaml --config configs/svp.yaml
    python scripts/02_cache_latents.py <same configs> --finalize
  Local Imagenette (engineering test only, one process):
    python scripts/02_cache_latents.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/local_imagenette.yaml
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rwvae import datasets, runtime

SECTION = "## 2. Latent cache"
VERIFY_FILE = "cache_verification.json"
UNIT_ROUNDOFF = {"float16": 2.0 ** -11, "bfloat16": 2.0 ** -8, "float32": 0.0}
RECOMPUTE_RTOL = 1e-4  # fp32 re-encode vs the cached encode (other batch composition / device), x max|mu|
FLIP_RULE = ("latents_flip = encode_mean(torch.flip(x, dims=[-1])) with x = image_transform(image_size)(img), "
             "i.e. the horizontal flip after the ADM center crop (LightningDiT order)")


# ----------------------------------------------------------------------------- DATA_CARD
def replace_section(path: Path, heading: str, body: str, title: str) -> None:
    """Create or replace the `heading` section (up to the next '## ' line) of a Markdown file, keeping
    every other section. A missing file starts with `title`. Duplicated in 01_prepare_manifest.py."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else [title, ""]
    new = [heading, "", *body.rstrip("\n").splitlines(), ""]
    if heading in lines:
        start = lines.index(heading)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        lines[start:end] = new
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines += new
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    tmp.replace(path)


def card_section(cfg: dict, splits: list[str]) -> str:
    """Section 2 from the caches on disk: every split of cache.splits (+ splits) that has an index."""
    rows, meta, shape = [], None, None
    for split in dict.fromkeys([*cfg["cache"]["splits"], *splits]):
        index_path = _common.cache_dir(cfg, split) / datasets.CACHE_INDEX
        if not index_path.is_file():
            rows.append(f"| {split} | not cached | | | | | |")
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        meta, shape = index["meta"], index["latent_shape"]
        sha = runtime.sha256_file(index_path)
        vpath = index_path.parent / VERIFY_FILE
        v = json.loads(vpath.read_text(encoding="utf-8")) if vpath.is_file() else None
        if v is None or v.get("cache_index_sha256") != sha:
            check = "not verified"
        else:
            check = (f"{v['num_rows']} rows: max abs {v['max_abs_err']:.3g}, max rel {v['max_rel_err']:.3g}, "
                     f"worst/bound {v['worst_ratio']:.3g} ({'PASS' if v['passed'] else 'FAIL'})")
        rows.append(f"| {split} | {index['total']} | {len(index['shards'])} x {index['shard_size']} | {index['dtype']} "
                    f"| {index['has_flip']} | `{sha}` | {check} |")
    out = [f"- cache root: `{_common.resolve(cfg['cache']['dir']).as_posix()}` (one directory per split)", ""]
    if meta is not None:
        tok = meta["tokenizer"]
        out += [f"- latent source: `{meta['latent_source']}` of `{meta['tokenizer_name']}`, checkpoint sha256 "
                f"`{tok.get('checkpoint_sha256')}`; latent shape {shape}",
                f"- encoded in float32 ({meta['tf32']}), stored as {meta['storage_dtype']}; statistics are "
                "accumulated in float64 later",
                f"- manifest fingerprint `{meta['manifest_fingerprint']}`; image size {meta['image_size']}, "
                f"preprocessing {meta['preprocessing']}",
                f"- flip: {meta['flip']} ({FLIP_RULE})", f"- code git commit `{meta['git_commit']}`", ""]
    out += ["| split | rows | shards | dtype | flip | cache_index sha256 | re-encode check |",
            "|---|---:|---|---|---|---|---|", *rows, "",
            f"Re-encode check: seeded random rows re-encoded and compared with the cache; bound "
            f"|d| <= u * |mu| + {RECOMPUTE_RTOL:g} * max|mu| (u = {UNIT_ROUNDOFF})."]
    return "\n".join(out)


# ----------------------------------------------------------------------------- encoding
def encode_rows(tokenizer, manifest, split: str, rows, *, image_size: int, flip: bool, batch_size: int,
                num_workers: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(mu, labels, indices, mu_flip or None) of `rows` (in order), mu float32 on the CPU."""
    ds = datasets.ManifestImageDataset(manifest, split, image_size=image_size, indices=list(rows))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    mus, flips, labels, indices = [], [], [], []
    for images, y, idx in loader:
        x = images.to(device, dtype=torch.float32, non_blocking=True)
        mus.append(tokenizer.encode_mean(x).float().cpu())
        if flip:
            flips.append(tokenizer.encode_mean(torch.flip(x, dims=[-1])).float().cpu())
        labels.append(y)
        indices.append(idx)
    return (torch.cat(mus), torch.cat(labels), torch.cat(indices), torch.cat(flips) if flip else None)


def verify_cache(cache: datasets.LatentCache, tokenizer, manifest, *, image_size: int, num_samples: int,
                 seed: int, batch_size: int, device) -> dict:
    """Re-encode `num_samples` seeded random rows and compare with the cache (storage rounding only)."""
    n = min(int(num_samples), cache.total)
    rows = sorted(int(i) for i in np.random.default_rng([seed, datasets.SPLITS.index(cache.split)])
                  .choice(cache.total, n, replace=False))
    mu, _, idx, mu_flip = encode_rows(tokenizer, manifest, cache.split, rows, image_size=image_size,
                                      flip=cache.has_flip, batch_size=batch_size, num_workers=0, device=device)
    if idx.tolist() != rows:
        raise RuntimeError("re-encoded rows are out of order")
    pairs = [(mu, torch.stack([cache.get(i)[0] for i in rows]))]
    if cache.has_flip:
        pairs.append((mu_flip, torch.stack([cache.get(i, flip=True)[0] for i in rows])))
    ref = torch.cat([p[0] for p in pairs])
    got = torch.cat([p[1] for p in pairs])
    d = (got - ref).abs()
    scale = float(ref.abs().max())
    bound = UNIT_ROUNDOFF[cache.dtype] * ref.abs() + RECOMPUTE_RTOL * scale
    worst = float((d / bound).max()) if scale > 0 else float(d.max() > 0)
    return {"rows": rows, "num_rows": n, "flip_checked": cache.has_flip, "max_abs_err": float(d.max()),
            "mean_abs_err": float(d.mean()), "max_rel_err": float(d.max()) / scale if scale > 0 else 0.0,
            "max_abs_mu": scale, "worst_ratio": worst, "passed": bool(worst <= 1.0),
            "bound": f"|d| <= {UNIT_ROUNDOFF[cache.dtype]:g} * |mu| + {RECOMPUTE_RTOL:g} * max|mu|",
            "storage_dtype": cache.dtype, "seed": [seed, datasets.SPLITS.index(cache.split)]}


def cache_meta(cfg: dict, tokenizer, manifest, split: str, tf32: dict) -> dict:
    """Shard meta (plan Sec. 3.2); identical for every process and resumed run of this split."""
    t, c = cfg["tokenizer"], cfg["cache"]
    if t.get("latent_source", "posterior_mean") != "posterior_mean":
        raise ValueError(f"tokenizer.latent_source = {t['latent_source']!r}: the cache stores the posterior mean only")
    return runtime.to_jsonable({
        "tokenizer_name": t["name"], "tokenizer": tokenizer.meta,
        "manifest_fingerprint": manifest.fingerprint(), "class_map_sha256": manifest.meta["class_map_sha256"],
        "dataset": manifest.meta["name"], "split": split, "latent_source": "posterior_mean",
        "encode_dtype": "float32", "storage_dtype": c["dtype"], "flip": bool(c["flip"]),
        "flip_rule": FLIP_RULE if c["flip"] else None, "image_size": int(cfg["dataset"]["image_size"]),
        "preprocessing": datasets.PREPROCESSING, "normalisation": "ToTensor + Normalize(0.5, 0.5) -> [-1, 1]",
        "tf32": tf32, "torch_version": torch.__version__, "git_commit": runtime.git_commit(),
    })


def proc_layout(args) -> tuple[int, int]:
    """(num_procs, proc_id) from the flags, else from torchrun's WORLD_SIZE / RANK."""
    if args.num_procs is None and args.proc_id is None:
        rank, world = _common.rank_world()
        num, pid = world, rank
    else:
        num, pid = (1 if args.num_procs is None else args.num_procs), (0 if args.proc_id is None else args.proc_id)
    if num < 1 or not 0 <= pid < num:
        raise ValueError(f"need --num-procs >= 1 and 0 <= --proc-id < --num-procs, got {num}, {pid}")
    return num, pid


def pick_device(pref: str, proc_id: int) -> torch.device:
    device = runtime.select_device(pref)
    if device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", proc_id)) % torch.cuda.device_count())
    return device


# ----------------------------------------------------------------------------- main
def build_parser():
    parser = _common.make_parser("Cache posterior-mean latents of the manifest splits (plan Sec. 1.3, M2).")
    parser.add_argument("--split", action="append", default=None, choices=datasets.SPLITS,
                        help="split to cache (repeatable); default cache.splits")
    parser.add_argument("--num-procs", type=int, default=None, help="number of parallel processes (default 1 or WORLD_SIZE)")
    parser.add_argument("--proc-id", type=int, default=None, help="this process: writes shards with id %% N == k")
    parser.add_argument("--finalize", action="store_true", help="only finalize and verify (all shards must exist)")
    parser.add_argument("--no-verify", action="store_true", help="skip the re-encode check after finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    c = cfg["cache"]
    splits = list(dict.fromkeys(args.split or c["splits"]))
    bad = [s for s in splits if s not in datasets.SPLITS]
    if bad:
        raise ValueError(f"cache.splits has unknown splits {bad}; known: {datasets.SPLITS}")
    num_procs, proc_id = proc_layout(args)
    parts = ["+".join(splits)] + ([f"proc{proc_id}of{num_procs}"] if num_procs > 1 else []) \
        + (["finalize"] if args.finalize else [])
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "02_cache", *parts)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        seed = int(cfg.get("seed", 0))
        runtime.seed_everything(seed)
        device = pick_device(args.device, proc_id)
        tf32 = _common.set_tf32(False)
        manifest = _common.load_manifest(cfg)
        tokenizer = _common.make_tokenizer(cfg, device)
        image_size = int(cfg["dataset"]["image_size"])
        run.logger.info("process %d of %d on %s; splits %s; tokenizer %s (sha256 %s)", proc_id, num_procs, device,
                        splits, cfg["tokenizer"]["name"], tokenizer.meta.get("checkpoint_sha256"))
        report = {}
        for split in splits:
            meta = cache_meta(cfg, tokenizer, manifest, split, tf32)
            writer = datasets.LatentCacheWriter(_common.cache_dir(cfg, split), split=split, meta=meta,
                                                dtype=c["dtype"], shard_size=int(c["shard_size"]))
            total = manifest.split_size(split)
            num = writer.num_shards(total)
            rec = {"cache_dir": writer.out_dir, "total": total, "num_shards": num, "written": [], "skipped": []}
            if not args.finalize:
                for k in range(proc_id, num, num_procs):
                    rows = writer.shard_rows(k, total)
                    if writer.shard_done(k, len(rows)):
                        rec["skipped"].append(k)
                        continue
                    mu, labels, idx, mu_flip = encode_rows(
                        tokenizer, manifest, split, rows, image_size=image_size, flip=bool(c["flip"]),
                        batch_size=int(c["batch_size"]), num_workers=int(c["num_workers"]), device=device)
                    writer.write_shard(k, mu, labels, idx, latents_flip=mu_flip)
                    rec["written"].append(k)
                    run.log_metrics(None, split=split, shard=k, rows=len(rows), max_abs_mu=float(mu.abs().max()))
                    run.logger.info("%s: wrote shard %d/%d (rows %d..%d)", split, k, num - 1, rows.start, rows.stop - 1)
                if rec["skipped"]:
                    run.logger.info("%s: %d shards already done, skipped: %s", split, len(rec["skipped"]), rec["skipped"])
            missing = [k for k in range(num) if not writer.shard_done(k, len(writer.shard_rows(k, total)))]
            rec["missing"] = missing
            if missing:
                if args.finalize:
                    raise FileNotFoundError(f"{split}: shards {missing[:10]} of 0..{num - 1} are missing in "
                                            f"{writer.out_dir}; run the caching processes first")
                rec["status"] = "incomplete"
                if proc_id == 0:
                    run.logger.warning("%s: shards %s are not written yet (other processes); run this command "
                                       "with --finalize once they are done", split, missing[:10])
                report[split] = rec
                continue
            if proc_id != 0 and not args.finalize:
                rec["status"] = "shards_done"  # process 0 or a --finalize call finalizes
                report[split] = rec
                continue

            index_path = writer.out_dir / datasets.CACHE_INDEX
            old = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
            if old is not None and old.get("meta") == writer.meta and old.get("total") == total and not rec["written"]:
                run.logger.info("%s: already finalized (%s)", split, index_path)
            else:
                writer.finalize(total)
                run.logger.info("%s: finalized %d rows in %d shards", split, total, num)
            cache = datasets.LatentCache(writer.out_dir, manifest=manifest, verify=True)
            rec.update(status="finalized", cache_index_sha256=runtime.sha256_file(index_path),
                       finalized=datetime.now().astimezone().isoformat(timespec="seconds"))
            if not args.no_verify:
                v = verify_cache(cache, tokenizer, manifest, image_size=image_size,
                                 num_samples=int(c["verify_samples"]), seed=seed,
                                 batch_size=int(c["batch_size"]), device=device)
                v.update(cache_index_sha256=rec["cache_index_sha256"], device=str(device), tf32=tf32,
                         created=rec["finalized"], git_commit=runtime.git_commit())
                tmp = writer.out_dir / (VERIFY_FILE + ".tmp")
                tmp.write_text(json.dumps(runtime.to_jsonable(v), indent=1) + "\n", encoding="utf-8")
                tmp.replace(writer.out_dir / VERIFY_FILE)
                rec["verification"] = {k: v[k] for k in ("num_rows", "max_abs_err", "max_rel_err", "worst_ratio",
                                                         "passed", "bound")}
                run.log_metrics(None, split=split, verify_max_abs_err=v["max_abs_err"],
                                verify_max_rel_err=v["max_rel_err"], verify_worst_ratio=v["worst_ratio"])
                run.logger.info("%s: re-encode check on %d rows: max abs %.3g, max rel %.3g, worst/bound %.3g",
                                split, v["num_rows"], v["max_abs_err"], v["max_rel_err"], v["worst_ratio"])
                if not v["passed"]:
                    raise ValueError(f"{split}: cached latents differ from a re-encode beyond storage rounding "
                                     f"(max abs {v['max_abs_err']:.3g}, worst/bound {v['worst_ratio']:.3g}); "
                                     f"see {writer.out_dir / VERIFY_FILE}")
            report[split] = rec

        run.save_json("cache_info.json", {"num_procs": num_procs, "proc_id": proc_id, "device": str(device),
                                          "tf32": tf32, "splits": report})
        if any(r["status"] == "finalized" for r in report.values()):
            card = _common.exp_dir(cfg) / "DATA_CARD.md"
            replace_section(card, SECTION, card_section(cfg, splits), f"# DATA CARD -- {cfg['experiment']['name']}")
            run.logger.info("updated %s (%s)", card, SECTION)
        run.finish("ok", proc_id=proc_id, num_procs=num_procs,
                   splits={s: {k: r[k] for k in ("status", "written", "skipped", "missing")} for s, r in report.items()})
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
