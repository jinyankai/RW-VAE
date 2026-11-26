"""Convert the HF ILSVRC/imagenet-1k parquet shards into an ImageFolder layout
<out>/<split>/<wnid>/<file>.JPEG, writing the original image bytes unchanged (no re-encoding).

label k <-> the k-th key of IMAGENET2012_CLASSES in the repo's classes.py (sorted wnids; checked).
File names come from the parquet image "path" field (e.g. n01440764_10026.JPEG,
ILSVRC2012_val_00000001.JPEG); a train file name must start with its label's wnid (checked).
Re-running skips files that already exist with the same size, so an interrupted run can resume.

Usage: python -I imagenet_parquet_to_folder.py <hf_snapshot_dir> <out_dir> [--split train|validation]... [--procs 32]
Splits are written as train/ and val/ (validation -> val). The test split has no labels and is ignored.
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED = {"train": 1_281_167, "validation": 50_000}
OUT_NAME = {"train": "train", "validation": "val"}


def load_wnids(snapshot: Path) -> list[str]:
    tree = ast.parse((snapshot / "classes.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "IMAGENET2012_CLASSES" for t in node.targets):
            call = node.value  # OrderedDict([(wnid, name), ...])
            wnids = [ast.literal_eval(pair.elts[0]) for pair in call.args[0].elts]
            break
    else:
        raise SystemExit("IMAGENET2012_CLASSES not found in classes.py")
    if len(wnids) != 1000 or wnids != sorted(wnids):
        raise SystemExit("classes.py: expected 1000 sorted wnids")
    return wnids


def convert_shard(args: tuple[str, str, str, list[str]]) -> Counter:
    shard, out_split_dir, split, wnids = args
    counts = Counter()
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(columns=["image", "label"], batch_size=256):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for img, label in zip(images, labels):
            wnid = wnids[label]
            name = Path(img["path"]).name
            if not name:
                raise ValueError(f"{shard}: image without a path")
            if split == "train" and not name.startswith(wnid):
                raise ValueError(f"{shard}: {name} does not match label {label} ({wnid})")
            data = img["bytes"]
            dst = Path(out_split_dir) / wnid / name
            if not (dst.exists() and dst.stat().st_size == len(data)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")  # per process: no cross-writer tmp clash
                for attempt in range(3):                              # survive transient FS errors under heavy parallel I/O
                    try:
                        tmp.write_bytes(data)
                        tmp.replace(dst)
                        break
                    except OSError:
                        tmp.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
            counts[wnid] += 1
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("snapshot", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", action="append", choices=list(EXPECTED), help="default: train and validation")
    p.add_argument("--procs", type=int, default=32)
    a = p.parse_args(argv)
    wnids = load_wnids(a.snapshot)
    for split in a.split or list(EXPECTED):
        shards = sorted((a.snapshot / "data").glob(f"{split}-*.parquet"))
        if not shards:
            raise SystemExit(f"no {split} shards under {a.snapshot / 'data'}")
        out_dir = a.out / OUT_NAME[split]
        total = Counter()
        with ProcessPoolExecutor(a.procs) as ex:
            for i, c in enumerate(ex.map(convert_shard, [(str(s), str(out_dir), split, wnids) for s in shards])):
                total.update(c)
                print(f"{split}: {i + 1}/{len(shards)} shards, {sum(total.values())} images", flush=True)
        n = sum(total.values())
        report = {"split": split, "images": n, "classes": len(total), "expected": EXPECTED[split],
                  "min_per_class": min(total.values()), "max_per_class": max(total.values())}
        (a.out / f"{OUT_NAME[split]}_counts.json").write_text(json.dumps({"summary": report, "per_class": dict(sorted(total.items()))}, indent=1))
        print(json.dumps(report), flush=True)
        if n != EXPECTED[split] or len(total) != 1000:
            print(f"ERROR: {split} has {n} images in {len(total)} classes", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
