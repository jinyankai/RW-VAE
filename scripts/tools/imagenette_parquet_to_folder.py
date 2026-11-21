"""Convert the HF imagenette2-320 parquet (train+val concatenated, rows grouped by split then
class) into an ImageFolder layout: <out>/{train,val}/<wnid>/<id>.JPEG.

Row order was verified to match the official imagenette2 per-class counts: the first 9469
rows are the official train split, the remaining 3925 the official val split.
Usage: python -I imagenette_parquet_to_folder.py <parquet> <out_dir>
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
         "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
TRAIN_COUNTS = [963, 955, 993, 858, 941, 956, 961, 931, 951, 960]
VAL_COUNTS = [387, 395, 357, 386, 409, 394, 389, 419, 399, 390]


def main(parquet: str, out_dir: str) -> None:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    table = pq.read_table(parquet, columns=["image", "label"])
    labels = table.column("label").to_pylist()
    n_train = sum(TRAIN_COUNTS)
    assert len(labels) == n_train + sum(VAL_COUNTS), len(labels)
    assert [Counter(labels[:n_train])[k] for k in range(10)] == TRAIN_COUNTS
    assert [Counter(labels[n_train:])[k] for k in range(10)] == VAL_COUNTS
    per_split_class = Counter()
    images = table.column("image")
    for row, label in enumerate(labels):
        split = "train" if row < n_train else "val"
        data = images[row].as_py()["bytes"]
        if data[:3] == b"\xff\xd8\xff":
            ext = ".JPEG"
        elif data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        else:
            raise SystemExit(f"row {row}: unknown image format")
        wnid = WNIDS[label]
        k = per_split_class[(split, label)]
        per_split_class[(split, label)] += 1
        tag = "tr" if split == "train" else "va"
        path = out / split / wnid / f"{wnid}_{tag}{k:05d}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print({f"{s}/{WNIDS[c]}": n for (s, c), n in sorted(per_split_class.items())})
    print("done:", sum(per_split_class.values()), "images")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
