"""Materialise the random ImageNet subset of the SVP (sanity-validation pilot) as an ImageFolder.

Deterministic selection (numpy Generator, so the stream is numpy-version specific; the version is recorded):
    rng = np.random.default_rng(seed)
    wnids = sorted class directories of --train-root; chosen = sorted(wnids[rng.choice(len, num_classes)])
    for each chosen wnid in sorted order: train files = sorted(F)[sorted(rng.choice(len(F), train_per_class))],
    then val files likewise from --val-root/<wnid> (all choices without replacement)
Files (ImageFolder convention: recursive below <wnid>/, extensions of rwvae.datasets.IMG_EXTENSIONS) are
linked (--mode symlink, default; falls back to copy when symlinks are not permitted, e.g. on Windows) or
copied to <out>/train/<wnid>/... and <out>/val/<wnid>/..., and <out>/svp_subset.json records the source
roots, seed, wnids, per-class counts, the sorted relative file list and its sha256. A non-empty --out is
refused unless its svp_subset.json describes the same selection (then "up to date"); svp_subset.json is
written last, so an interrupted run leaves a folder that is refused until removed.
01_prepare_manifest.py copies this provenance into DATA_CARD.md when it finds <train_root>/../svp_subset.json.

Usage
  python scripts/tools/make_svp_subset.py --train-root /home/jinyankai/data/imagenet/train \
      --val-root /home/jinyankai/data/imagenet/val --out /home/jinyankai/data/svp/imagenet_c10_n100
  options: --num-classes 10 --train-per-class 100 --val-per-class 50 --seed 2026 --mode symlink|copy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")  # = rwvae.datasets
SUBSET_FILE = "svp_subset.json"


def class_files(root: Path, wnid: str) -> list[str]:
    """Sorted POSIX paths, relative to root/wnid, of the images below root/wnid (recursive)."""
    base = root / wnid
    if not base.is_dir():
        raise ValueError(f"class directory {base} does not exist")
    return sorted(Path(dirpath, f).relative_to(base).as_posix()
                  for dirpath, _, files in os.walk(base, followlinks=True)
                  for f in files if f.lower().endswith(IMG_EXTENSIONS))


def select(train_root: Path, val_root: Path, num_classes: int, train_per_class: int,
           val_per_class: int, seed: int) -> dict:
    """The selection record (everything of svp_subset.json except the materialisation mode)."""
    wnids = sorted(e.name for e in os.scandir(train_root) if e.is_dir())
    if num_classes > len(wnids):
        raise ValueError(f"--num-classes {num_classes} > {len(wnids)} classes in {train_root}")
    rng = np.random.default_rng(seed)
    chosen = sorted(wnids[i] for i in rng.choice(len(wnids), num_classes, replace=False))
    files, counts = [], {"train": {}, "val": {}}
    for wnid in chosen:
        for split, root, k in (("train", train_root, train_per_class), ("val", val_root, val_per_class)):
            avail = class_files(root, wnid)
            if len(avail) < k:
                raise ValueError(f"{root / wnid} has {len(avail)} images, fewer than the {k} requested")
            picked = [avail[i] for i in sorted(rng.choice(len(avail), k, replace=False))]
            files += [f"{split}/{wnid}/{p}" for p in picked]
            counts[split][wnid] = k
    files.sort()
    return {"tool": "scripts/tools/make_svp_subset.py",
            "source": {"train_root": train_root.as_posix(), "val_root": val_root.as_posix()},
            "seed": seed, "num_classes": num_classes, "train_per_class": train_per_class,
            "val_per_class": val_per_class, "numpy_version": np.__version__,
            "selection": "default_rng(seed): classes = sorted(choice(sorted wnids)); per chosen class (sorted): "
                         "train then val files = sorted(choice(sorted file list)); all without replacement",
            "wnids": chosen, "counts": counts, "files": files,
            "files_sha256": hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()}


def materialise(record: dict, out: Path, mode: str) -> str:
    """Link or copy every selected file into out; returns the mode actually used."""
    roots = {"train": Path(record["source"]["train_root"]), "val": Path(record["source"]["val_root"])}
    for rel in record["files"]:
        split, rest = rel.split("/", 1)
        src, dst = roots[split] / rest, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            try:
                os.symlink(src, dst)
                continue
            except OSError as e:  # e.g. Windows without the symlink privilege
                print(f"symlink not permitted ({e}); copying instead")
                mode = "copy"
        shutil.copy2(src, dst)
    return mode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Random ImageNet subset for the SVP (see module docstring).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-root", required=True, help="ImageFolder <wnid>/<file> of the source train split")
    p.add_argument("--val-root", required=True, help="ImageFolder <wnid>/<file> of the source val split")
    p.add_argument("--out", required=True, help="output folder: <out>/{train,val}/<wnid>/..., <out>/svp_subset.json")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = p.parse_args(argv)
    for name in ("num_classes", "train_per_class", "val_per_class"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1, got {getattr(args, name)}")

    record = select(Path(args.train_root).resolve(), Path(args.val_root).resolve(), args.num_classes,
                    args.train_per_class, args.val_per_class, args.seed)
    out = Path(args.out)
    subset_file = out / SUBSET_FILE
    if out.exists() and any(out.iterdir()):
        old = json.loads(subset_file.read_text(encoding="utf-8")) if subset_file.is_file() else None
        if old is None or {k: v for k, v in old.items() if k != "mode"} != record:
            raise ValueError(f"{out} is not empty and does not hold this selection "
                             f"({'no ' + SUBSET_FILE if old is None else SUBSET_FILE + ' differs'}); "
                             "remove it or choose another --out")
        missing = [f for f in record["files"] if not (out / f).is_file()]
        if missing:
            raise ValueError(f"{out}: {len(missing)} selected files are missing, e.g. {missing[:3]}; remove the folder")
        print(f"{out} is up to date ({len(record['files'])} files, sha256 {record['files_sha256'][:16]})")
        return 0

    record["mode"] = materialise(record, out, args.mode)
    tmp = subset_file.with_name(SUBSET_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    tmp.replace(subset_file)
    n = {s: sum(c.values()) for s, c in record["counts"].items()}
    print(f"wrote {out}: {record['num_classes']} classes {record['wnids']}, {n} images ({record['mode']}), "
          f"file-list sha256 {record['files_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
