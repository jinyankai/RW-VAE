"""RW-VAE v2 data contract (ARCHITECTURE_V2 Sec. 3.7; plan v2.1 Sec. 1.2 and 3.2).

    build_manifest(train_root, val_root)    -> Manifest: image-level train_dev / train_fit / val split
    ManifestImageDataset(manifest, split)   -> (image [3, S, S] in [-1, 1], label, index)
    LatentCacheWriter(out_dir, split=...)   -> deterministic safetensors shards of mu = posterior.mean
    LatentCache(out_dir, manifest=...)      -> read-only random access to mu (and flipped mu)
    ChannelLatentDataset(cache, channel)    -> (normalized z' = channel(mu), label, index) for the DiT
    ImageLatentPairDataset(manifest, cache) -> (image, mu, label, index) for decoder adaptation / eval

`index` is always the row inside one manifest split. Together with the cache's
meta["manifest_fingerprint"] it identifies the image ID, split and label of every cached latent.
All arms read the same cache and share one normalization, so arms differ only in the channel.

Determinism: the split is a pure function of the sorted file lists and the seed. Shard k always
holds split rows [k*S, min((k+1)*S, N)). ChannelLatentDataset draws the flip decision and the
channel noise of sample `index` from a generator seeded with noise_seed(base_seed, epoch, index),
so its output does not depend on batch composition, sampling order or DataLoader worker count.

Windows: DataLoader workers are spawned, so every dataset here pickles. safetensors handles are
opened lazily per process and dropped when pickled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .channel import RWChannel, noise_seed

__all__ = [
    "IMG_EXTENSIONS", "SPLITS", "PREPROCESSING", "CACHE_INDEX", "CACHE_DTYPES",
    "center_crop_arr", "image_transform",
    "Manifest", "scan_image_folder", "build_manifest", "ManifestImageDataset",
    "LatentCacheWriter", "LatentCache", "ChannelLatentDataset", "ImageLatentPairDataset",
]

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
MANIFEST_SCHEMA_VERSION = 1
SPLITS = ("train_dev", "train_fit", "val")
PREPROCESSING = {"crop": "adm_center_crop", "resample": "box_halving_then_bicubic", "range": "[-1,1]"}
CACHE_INDEX = "cache_index.json"
CACHE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_ST_DTYPES = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}  # safetensors dtype tags
_COLUMNS = ("image_id", "path", "label")
_SHARD_KEYS = {"latents", "latents_flip", "labels", "indices"}


# ----------------------------------------------------------------------------- helpers

def _int_arg(name: str, value, minimum: int = 0) -> int:
    """`value` as a Python int; ValueError unless it is an integer (not a bool) >= minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _row_list(indices, n: int) -> list[int]:
    """Validated subset of split rows: a 1-D integer sequence with values in [0, n)."""
    arr = np.asarray(indices)
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(f"indices must be a 1-D integer sequence, got shape {arr.shape}, dtype {arr.dtype}")
    if arr.size and (arr.min() < 0 or arr.max() >= n):
        raise ValueError(f"indices must lie in [0, {n}), got min {arr.min()}, max {arr.max()}")
    return [int(i) for i in arr]


def _int64_vector(name: str, x, n: int) -> torch.Tensor:
    t = x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    if t.dtype.is_floating_point or t.dtype.is_complex or t.dtype == torch.bool or tuple(t.shape) != (n,):
        raise ValueError(f"{name} must be an integer vector of length {n}, got dtype {t.dtype}, shape {tuple(t.shape)}")
    return t.to(torch.int64, copy=True, memory_format=torch.contiguous_format)  # owns its memory (see _cast)


def _channel_vector(name: str, x, C: int) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if tuple(t.shape) != (C,) or not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must be a finite vector of length C={C}, got shape {tuple(t.shape)}")
    return t


# ----------------------------------------------------------------------------- images

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM center crop, copied verbatim from guided-diffusion (image_datasets.py#L126) as in
    LightningDiT tokenizer/vavae.py: BOX-halve while the short side is >= 2 * image_size,
    BICUBIC-resize so the short side equals image_size, then cut the central square with
    offsets floor((side - image_size) / 2)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _to_rgb(pil_image: Image.Image) -> Image.Image:
    return pil_image.convert("RGB")


def image_transform(image_size: int = 256, hflip: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    """PIL image -> float32 [3, S, S] in [-1, 1]: RGB -> center_crop_arr -> (horizontal flip)
    -> ToTensor -> Normalize(0.5, 0.5). This is LightningDiT's VA_VAE.img_transform with a
    deterministic flip: hflip=True flips every image (used to cache `latents_flip`).
    Built from module-level callables, so it pickles into spawned DataLoader workers."""
    size = _int_arg("image_size", image_size, minimum=1)
    steps = [transforms.Lambda(_to_rgb), transforms.Lambda(partial(center_crop_arr, image_size=size))]
    if hflip:
        steps.append(transforms.Lambda(TF.hflip))
    steps += [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    return transforms.Compose(steps)


# ----------------------------------------------------------------------------- manifest

@dataclass
class Manifest:
    """Image-level split manifest (plan Sec. 1.2).

    classes[label] is a class directory name (wnid; ImageFolder order). splits[split] is
    columnar, {"image_id": [...], "path": [...], "label": [...]}, with POSIX paths relative to
    meta["split_roots"][split]. Splits are made per image ID, so no image's tokens can straddle
    two splits. fingerprint() hashes classes + splits only: it is stable across re-runs and
    across relocating the data (roots and timestamps live in meta).
    """

    meta: dict
    classes: list[str]
    splits: dict[str, dict[str, list]]

    def fingerprint(self) -> str:
        """sha256 hex digest of the canonical JSON of {"classes", "splits"} (meta excluded)."""
        return _sha256_text(_canonical_json({"classes": self.classes, "splits": self.splits}))

    def _columns(self, split: str) -> dict[str, list]:
        if split not in self.splits:
            raise ValueError(f"unknown split {split!r}; the manifest has {sorted(self.splits)}")
        return self.splits[split]

    def split_size(self, split: str) -> int:
        return len(self._columns(split)["image_id"])

    def record(self, split: str, i: int) -> dict:
        """Row i of `split`: {"image_id", "path", "label"}."""
        cols, n = self._columns(split), self.split_size(split)
        if not 0 <= i < n:
            raise IndexError(f"row {i} out of range [0, {n}) for split {split!r}")
        return {c: cols[c][i] for c in _COLUMNS}

    def image_path(self, split: str, i: int) -> Path:
        roots = self.meta.get("split_roots", {})
        if split not in roots:
            raise ValueError(f"meta['split_roots'] has no root for split {split!r}: {roots}")
        return Path(roots[split]) / self.record(split, i)["path"]

    def validate(self) -> None:
        """ValueError unless class names are unique, every split has exactly the columns
        image_id / path / label of equal length, labels are ints in [0, K), and image IDs are
        unique across all splits."""
        K = len(self.classes)
        if len(set(self.classes)) != K:
            raise ValueError(f"duplicate class names among {K} classes")
        owner: dict[str, str] = {}
        for split, cols in self.splits.items():
            if set(cols) != set(_COLUMNS):
                raise ValueError(f"split {split!r} has columns {sorted(cols)}, expected {sorted(_COLUMNS)}")
            lengths = {c: len(cols[c]) for c in _COLUMNS}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"split {split!r}: column lengths differ: {lengths}")
            bad = [y for y in cols["label"] if isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < K]
            if bad:
                raise ValueError(f"split {split!r}: {len(bad)} labels are not ints in [0, {K}), e.g. {bad[:5]}")
            for image_id in cols["image_id"]:
                if image_id in owner:
                    raise ValueError(f"image_id {image_id!r} occurs in split {owner[image_id]!r} and in {split!r}")
                owner[image_id] = split

    def save(self, path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {"schema_version": MANIFEST_SCHEMA_VERSION, "meta": self.meta,
               "classes": self.classes, "splits": self.splits}
        _write_text_atomic(path, json.dumps(obj))

    @classmethod
    def load(cls, path) -> "Manifest":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {obj.get('schema_version')!r}, expected {MANIFEST_SCHEMA_VERSION}")
        missing = {"meta", "classes", "splits"} - set(obj)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        manifest = cls(meta=obj["meta"], classes=obj["classes"], splits=obj["splits"])
        manifest.validate()
        return manifest


def scan_image_folder(root) -> tuple[list[str], list[tuple[str, int]]]:
    """ImageFolder convention (torchvision): classes = sorted names of root's sub-directories,
    label = position in that list; images = files below each class directory (recursive,
    symlinks followed) whose extension is in IMG_EXTENSIONS, case-insensitively.

    Returns (classes, [(relpath, label)]), relpaths POSIX and sorted as plain strings within
    each class, so the result is platform independent. Empty class directories raise ValueError
    (as ImageFolder does)."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image folder root {str(root)!r} is not a directory")
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise ValueError(f"no class directories in {str(root)!r}")
    samples: list[tuple[str, int]] = []
    empty: list[str] = []
    for label, cls in enumerate(classes):
        files = sorted(
            Path(dirpath, fname).relative_to(root).as_posix()
            for dirpath, _, fnames in os.walk(root / cls, followlinks=True)
            for fname in fnames
            if fname.lower().endswith(IMG_EXTENSIONS)
        )
        if not files:
            empty.append(cls)
        samples.extend((rel, label) for rel in files)
    if empty:
        raise ValueError(f"{len(empty)} class directories of {str(root)!r} contain no images: {empty[:5]}")
    return classes, samples


def build_manifest(train_root, val_root, *, dev_per_class: int = 20, seed: int = 2026,
                   name: str = "imagenet1k", image_size: int = 256) -> Manifest:
    """Deterministic split manifest (plan Sec. 1.1-1.2).

    classes: sorted class directories of train_root; val class directories must be a subset and
    take the train labels. For class `label` with sorted train files F (n = len(F) must exceed
    dev_per_class):
        train_dev = F[sort(np.random.default_rng([seed, label]).choice(n, dev_per_class, replace=False))]
        train_fit = the remaining files of F, in sorted order
    val = every val file. Rows are class-major and sorted within a class. image_id = file stem,
    which must be unique across all splits.
    """
    dev_per_class = _int_arg("dev_per_class", dev_per_class, minimum=1)
    seed = _int_arg("seed", seed, minimum=0)
    image_size = _int_arg("image_size", image_size, minimum=1)
    train_root, val_root = Path(train_root).resolve(), Path(val_root).resolve()
    classes, train_samples = scan_image_folder(train_root)
    val_classes, val_samples = scan_image_folder(val_root)
    unknown = sorted(set(val_classes) - set(classes))
    if unknown:
        raise ValueError(f"{len(unknown)} val class directories are not train classes: {unknown[:5]}")
    train_label = {c: i for i, c in enumerate(classes)}

    per_class: list[list[str]] = [[] for _ in classes]
    for rel, label in train_samples:
        per_class[label].append(rel)
    rows: dict[str, list[tuple[str, int]]] = {s: [] for s in SPLITS}
    for label, files in enumerate(per_class):
        n = len(files)
        if n <= dev_per_class:
            raise ValueError(f"class {classes[label]!r} (label {label}) has {n} train images; "
                             f"need more than dev_per_class={dev_per_class}")
        is_dev = np.zeros(n, dtype=bool)
        is_dev[np.random.default_rng([seed, label]).choice(n, dev_per_class, replace=False)] = True
        rows["train_dev"] += [(f, label) for f, d in zip(files, is_dev) if d]
        rows["train_fit"] += [(f, label) for f, d in zip(files, is_dev) if not d]
    rows["val"] = [(rel, train_label[val_classes[label]]) for rel, label in val_samples]

    splits: dict[str, dict[str, list]] = {}
    owner: dict[str, str] = {}
    for split, items in rows.items():
        ids = [PurePosixPath(rel).stem for rel, _ in items]
        for image_id, (rel, _) in zip(ids, items):
            if image_id in owner:
                raise ValueError(f"duplicate image_id {image_id!r}: {owner[image_id]} and {split}:{rel}")
            owner[image_id] = f"{split}:{rel}"
        splits[split] = {"image_id": ids, "path": [rel for rel, _ in items], "label": [y for _, y in items]}

    train, val = train_root.as_posix(), val_root.as_posix()
    meta = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "train_root": train,
        "val_root": val,
        "split_roots": {"train_dev": train, "train_fit": train, "val": val},
        "seed": seed,
        "dev_per_class": dev_per_class,
        "image_size": image_size,
        "preprocessing": dict(PREPROCESSING),
        "class_map_sha256": _sha256_text(_canonical_json(classes)),
        "counts": {s: len(splits[s]["image_id"]) for s in SPLITS},
        "numpy_version": np.__version__,  # default_rng(...).choice streams are numpy-version specific
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest = Manifest(meta=meta, classes=classes, splits=splits)
    manifest.validate()
    return manifest


class ManifestImageDataset(Dataset):
    """Images of one manifest split. Item k -> (image float32 [3, S, S] in [-1, 1], label, index),
    index = indices[k] (the split row; all rows when indices is None). hflip=True flips every
    image (deterministic; used to cache latents_flip)."""

    def __init__(self, manifest: Manifest, split: str, *, image_size: int = 256, hflip: bool = False,
                 indices: Sequence[int] | None = None):
        n = manifest.split_size(split)
        self.manifest, self.split = manifest, split
        self.indices = list(range(n)) if indices is None else _row_list(indices, n)
        self.transform = image_transform(image_size, hflip)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, int, int]:
        i = self.indices[k]
        label = self.manifest.record(self.split, i)["label"]
        with Image.open(self.manifest.image_path(self.split, i)) as img:
            image = self.transform(img)
        return image, label, i


# ----------------------------------------------------------------------------- latent cache

def _num_shards(total: int, shard_size: int) -> int:
    return -(-total // shard_size)


def _shard_rows(shard_id: int, total: int, shard_size: int) -> range:
    """Split rows held by shard k: [k*S, min((k+1)*S, N))."""
    num = _num_shards(total, shard_size)
    if not 0 <= shard_id < num:
        raise ValueError(f"shard_id {shard_id} out of range [0, {num}) for {total} rows in shards of {shard_size}")
    start = shard_id * shard_size
    return range(start, min(start + shard_size, total))


def _inspect_shard(path: Path, *, shard_id: int, rows: range, dtype: str, meta: dict) -> dict:
    """Check one shard file against its expected contents: metadata (shard_id, n, dtype, meta),
    tensor keys, latent layout [n, C, H, W] in `dtype` (latents_flip identical), indices equal
    to `rows`, labels int64 [n] >= 0. Returns {"latent_shape", "has_flip", "labels"}."""
    n = len(rows)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md, keys = f.metadata() or {}, set(f.keys())
        want = {"shard_id": str(shard_id), "n": str(n), "dtype": dtype}
        got = {k: md.get(k) for k in want}
        if got != want:
            raise ValueError(f"{path.name}: metadata {got} != expected {want}")
        if json.loads(md.get("meta", "null")) != meta:
            raise ValueError(f"{path.name}: its meta differs from the expected cache meta (written by another run?)")
        if not {"latents", "labels", "indices"} <= keys <= _SHARD_KEYS:
            raise ValueError(f"{path.name}: tensor keys {sorted(keys)}, expected latents/labels/indices[/latents_flip]")
        slices = {k: f.get_slice(k) for k in ("latents", "latents_flip") if k in keys}
        layouts = {k: (tuple(s.get_shape()), s.get_dtype()) for k, s in slices.items()}
        indices, labels = f.get_tensor("indices"), f.get_tensor("labels")
    shape, st_dtype = layouts["latents"]
    if len(set(layouts.values())) != 1 or len(shape) != 4 or shape[0] != n or st_dtype != _ST_DTYPES[dtype]:
        raise ValueError(f"{path.name}: latent layouts {layouts} do not match n={n}, dtype={dtype}")
    if indices.dtype != torch.int64 or not torch.equal(indices, torch.arange(rows.start, rows.stop)):
        raise ValueError(f"{path.name}: indices are not the split rows [{rows.start}, {rows.stop})")
    if labels.dtype != torch.int64 or tuple(labels.shape) != (n,) or bool((labels < 0).any()):
        raise ValueError(f"{path.name}: labels must be int64 [{n}] >= 0, got {labels.dtype} {tuple(labels.shape)}")
    return {"latent_shape": shape[1:], "has_flip": "latents_flip" in keys, "labels": labels}


class LatentCacheWriter:
    """Writes the mu cache of one split as deterministic safetensors shards.

    Shard k holds split rows [k*S, min((k+1)*S, N)) (S = shard_size) in the file
    {split}_shard{k:05d}.safetensors: latents [n, C, H, W] in the storage dtype, optional
    latents_flip (mu of the flipped images, same shape), labels int64 [n], indices int64 [n]
    (split rows), and string metadata {"meta": json(meta), "shard_id", "n", "dtype"}. Fixed row
    ranges make writing resumable (shard_done) and parallel across processes (each writes its
    own shard ids with an identical meta); finalize then checks the split and writes
    cache_index.json, so a directory holds the cache of exactly one split. meta should carry the
    provenance (manifest_fingerprint, tokenizer checkpoint sha256, latent_source, preprocessing,
    augmentation, git commit; plan Sec. 3.2).
    """

    def __init__(self, out_dir, *, split: str, meta: dict, dtype: str = "float16", shard_size: int = 10000):
        if not isinstance(split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", split):
            raise ValueError(f"split must match [A-Za-z0-9_-]+ (it names the shard files), got {split!r}")
        if dtype not in CACHE_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(CACHE_DTYPES)}, got {dtype!r}")
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be a dict, got {type(meta).__name__}")
        try:
            self._meta_json = json.dumps(meta, sort_keys=True)
        except TypeError as e:
            raise ValueError(f"meta must be JSON-serializable: {e}") from None
        self.meta: dict = json.loads(self._meta_json)
        self.out_dir, self.split, self.dtype = Path(out_dir), split, dtype
        self.shard_size = _int_arg("shard_size", shard_size, minimum=1)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def num_shards(self, total: int) -> int:
        """ceil(total / shard_size): the shard ids of a split with `total` rows are 0..num-1."""
        return _num_shards(_int_arg("total", total, minimum=1), self.shard_size)

    def shard_rows(self, shard_id: int, total: int) -> range:
        return _shard_rows(_int_arg("shard_id", shard_id), _int_arg("total", total, minimum=1), self.shard_size)

    def shard_path(self, shard_id: int) -> Path:
        return self.out_dir / f"{self.split}_shard{_int_arg('shard_id', shard_id):05d}.safetensors"

    def shard_done(self, shard_id: int, expected_n: int) -> bool:
        """True iff the shard file exists and its metadata n equals expected_n (resume check).
        ValueError if the existing file was written with another meta, dtype or shard id: a
        different run's cache is in this directory, and it is neither mixed in nor overwritten."""
        expected_n = _int_arg("expected_n", expected_n, minimum=1)
        path = self.shard_path(shard_id)
        if not path.is_file():
            return False
        with safe_open(str(path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        if (md.get("shard_id") != str(shard_id) or md.get("dtype") != self.dtype
                or json.loads(md.get("meta", "null")) != self.meta):
            raise ValueError(f"{path} exists but was written with a different meta, dtype or shard id")
        return md.get("n") == str(expected_n)

    def _cast(self, name: str, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4 or not x.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor [n, C, H, W], got "
                             f"{type(x).__name__} of shape {tuple(getattr(x, 'shape', ()))}")
        # a fresh contiguous CPU copy: safetensors refuses tensors that share memory (e.g. two
        # views of one encoded batch passed as latents and latents_flip)
        stored = x.detach().to(device="cpu", dtype=CACHE_DTYPES[self.dtype], copy=True,
                               memory_format=torch.contiguous_format)
        if not bool(torch.isfinite(stored).all()):  # non-finite input, or an overflow in the cast
            raise ValueError(f"{name} is not finite in {self.dtype} (input max |x| = {float(x.detach().abs().max()):.6g})")
        return stored

    def write_shard(self, shard_id: int, latents, labels, indices, latents_flip=None) -> dict:
        """Atomically write shard `shard_id` (temporary file, then os.replace). Requires
        indices == [k*S, k*S + n) in order with 1 <= n <= S, labels >= 0, and latents (and
        latents_flip, same shape) finite in the storage dtype; an fp16 overflow raises
        instead of being stored as inf. Returns {"file", "shard_id", "n", "has_flip"}."""
        k = _int_arg("shard_id", shard_id)
        tensors = {"latents": self._cast("latents", latents)}
        n = tensors["latents"].shape[0]
        if not 1 <= n <= self.shard_size:
            raise ValueError(f"a shard holds 1..{self.shard_size} rows, got {n}")
        tensors["labels"] = _int64_vector("labels", labels, n)
        tensors["indices"] = _int64_vector("indices", indices, n)
        start = k * self.shard_size
        if not torch.equal(tensors["indices"], torch.arange(start, start + n)):
            got = tensors["indices"]
            raise ValueError(f"shard {k} must hold split rows [{start}, {start + n}) in order, "
                             f"got indices {int(got[0])}..{int(got[-1])}")
        if bool((tensors["labels"] < 0).any()):
            raise ValueError(f"labels must be >= 0, got min {int(tensors['labels'].min())}")
        if latents_flip is not None:
            tensors["latents_flip"] = self._cast("latents_flip", latents_flip)
            if tensors["latents_flip"].shape != tensors["latents"].shape:
                raise ValueError(f"latents_flip shape {tuple(tensors['latents_flip'].shape)} "
                                 f"!= latents shape {tuple(tensors['latents'].shape)}")
        path = self.shard_path(k)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        save_file(tensors, str(tmp), metadata={"meta": self._meta_json, "shard_id": str(k),
                                               "n": str(n), "dtype": self.dtype})
        os.replace(tmp, path)
        return {"file": path.name, "shard_id": k, "n": n, "has_flip": latents_flip is not None}

    def finalize(self, expected_total: int) -> dict:
        """Check the shards on disk form exactly the cache of a split with expected_total rows,
        then write cache_index.json (atomically) and return it.

        Checks: shard ids are exactly 0..K-1 (K = ceil(N/S)); every shard passes the per-shard
        checks (metadata, keys, layout, indices == its row range, labels int64 >= 0), so the
        indices cover [0, N) exactly once; all shards share one latent shape and either all or
        none store latents_flip. The index records the sha256 of every shard file."""
        total = _int_arg("expected_total", expected_total, minimum=1)
        index_path = self.out_dir / CACHE_INDEX
        if index_path.is_file():  # one cache (split) per directory: never replace another split's index
            old_split = json.loads(index_path.read_text(encoding="utf-8")).get("split")
            if old_split != self.split:
                raise ValueError(f"{index_path} indexes split {old_split!r}; use one directory per split")
        num = _num_shards(total, self.shard_size)
        pattern = re.compile(rf"{re.escape(self.split)}_shard(\d{{5,}})\.safetensors")
        found = {int(m.group(1)): p for p in self.out_dir.iterdir() if (m := pattern.fullmatch(p.name))}
        missing = [k for k in range(num) if k not in found]
        extra = sorted(k for k in found if k >= num)
        if missing or extra:
            raise ValueError(f"split {self.split!r} with {total} rows needs shards 0..{num - 1}: "
                             f"missing {missing[:10]}, unexpected {extra[:10]}")
        shards, layouts = [], set()
        for k in range(num):
            rows = _shard_rows(k, total, self.shard_size)
            info = _inspect_shard(found[k], shard_id=k, rows=rows, dtype=self.dtype, meta=self.meta)
            layouts.add((info["latent_shape"], info["has_flip"]))
            shards.append({"file": found[k].name, "shard_id": k, "n": len(rows), "sha256": _sha256_file(found[k])})
        if len(layouts) != 1:
            raise ValueError(f"shards disagree on (latent_shape, has_flip): {sorted(layouts)}")
        (latent_shape, has_flip), = layouts
        index = {"meta": self.meta, "split": self.split, "total": total, "dtype": self.dtype,
                 "has_flip": has_flip, "shard_size": self.shard_size, "latent_shape": list(latent_shape),
                 "shards": shards}
        _write_text_atomic(index_path, json.dumps(index, indent=1))
        return index


class LatentCache:
    """Read-only random access to a finalized latent cache (cache_index.json + shards).

    get(i) returns split row i as (mu float32 [C, H, W], label, i); row i lives in shard i // S
    at offset i % S. verify=True re-checks every shard (metadata, layout, indices covering
    [0, total) exactly once). With a manifest, meta["manifest_fingerprint"] must equal
    manifest.fingerprint() and the cached labels must equal the manifest labels of the split.
    safetensors handles are opened lazily, per process, and dropped when pickled.
    """

    def __init__(self, cache_dir, *, manifest: Manifest | None = None, verify: bool = True):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / CACHE_INDEX
        if not index_path.is_file():
            raise FileNotFoundError(f"{index_path} not found (run LatentCacheWriter.finalize first)")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.meta: dict = index["meta"]
        self.split: str = index["split"]
        self.dtype: str = index["dtype"]
        self.has_flip: bool = bool(index["has_flip"])
        self.total: int = int(index["total"])
        self.shard_size: int = int(index["shard_size"])
        self.latent_shape: tuple[int, ...] = tuple(int(s) for s in index["latent_shape"])
        shard_ids = sorted(s["shard_id"] for s in index["shards"])
        num = _num_shards(self.total, self.shard_size)
        if shard_ids != list(range(num)):
            raise ValueError(f"{index_path}: shard ids {shard_ids[:10]} != 0..{num - 1} for "
                             f"{self.total} rows in shards of {self.shard_size}")
        self._files = [s["file"] for s in sorted(index["shards"], key=lambda s: s["shard_id"])]
        labels = []
        for k, file in enumerate(self._files):
            if verify:
                rows = _shard_rows(k, self.total, self.shard_size)
                info = _inspect_shard(self.cache_dir / file, shard_id=k, rows=rows, dtype=self.dtype, meta=self.meta)
                if (info["latent_shape"], info["has_flip"]) != (self.latent_shape, self.has_flip):
                    raise ValueError(f"{file}: (latent_shape, has_flip) = {(info['latent_shape'], info['has_flip'])}, "
                                     f"index says {(self.latent_shape, self.has_flip)}")
                labels.append(info["labels"])
            else:
                with safe_open(str(self.cache_dir / file), framework="pt", device="cpu") as f:
                    labels.append(f.get_tensor("labels"))
        self._labels = torch.cat(labels).to(torch.int64)
        if self._labels.numel() != self.total:
            raise ValueError(f"shards hold {self._labels.numel()} labels, index total is {self.total}")
        if manifest is not None:
            self._check_manifest(manifest)
        self._handles: dict[int, object] = {}
        self._pid = os.getpid()

    def _check_manifest(self, manifest: Manifest) -> None:
        want, got = manifest.fingerprint(), self.meta.get("manifest_fingerprint")
        if got != want:
            raise ValueError(f"cache {str(self.cache_dir)!r} has manifest_fingerprint {got!r}, "
                             f"the given manifest has {want!r}")
        n = manifest.split_size(self.split)
        if n != self.total:
            raise ValueError(f"cache has {self.total} rows, manifest split {self.split!r} has {n}")
        expected = torch.tensor(manifest.splits[self.split]["label"], dtype=torch.int64)
        bad = torch.nonzero(expected != self._labels).flatten()
        if bad.numel():
            raise ValueError(f"{bad.numel()} cached labels differ from the manifest (first at row {int(bad[0])})")

    def __len__(self) -> int:
        return self.total

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}  # safe_open handles do not pickle; reopened lazily in the new process
        return state

    def _handle(self, k: int):
        if self._pid != os.getpid():  # forked child: never share the parent's handles
            self._handles, self._pid = {}, os.getpid()
        if k not in self._handles:
            self._handles[k] = safe_open(str(self.cache_dir / self._files[k]), framework="pt", device="cpu")
        return self._handles[k]

    def get(self, i: int, flip: bool = False) -> tuple[torch.Tensor, int, int]:
        """(mu float32 [C, H, W], label, i) for split row i; flip=True reads latents_flip."""
        if not 0 <= i < self.total:
            raise IndexError(f"row {i} out of range [0, {self.total})")
        if flip and not self.has_flip:
            raise ValueError(f"cache {str(self.cache_dir)!r} stores no latents_flip")
        i = int(i)
        k, r = divmod(i, self.shard_size)
        mu = self._handle(k).get_slice("latents_flip" if flip else "latents")[r:r + 1][0]
        return mu.float(), int(self._labels[i]), i

    def iter_batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sequential pass in storage (= split row) order yielding (mu float32 [b, C, H, W],
        labels int64 [b], indices int64 [b]); batches may span shards, only the last is short."""
        b, S = _int_arg("batch_size", batch_size, minimum=1), self.shard_size
        for start in range(0, self.total, b):
            stop = min(start + b, self.total)
            parts = [self._handle(k).get_slice("latents")[max(start, k * S) - k * S: min(stop, (k + 1) * S) - k * S]
                     for k in range(start // S, (stop - 1) // S + 1)]
            yield torch.cat(parts).float(), self._labels[start:stop].clone(), torch.arange(start, stop)


# ----------------------------------------------------------------------------- training datasets

class ChannelLatentDataset(Dataset):
    """DiT training data (plan Sec. 3.2): item `index` -> ((z' - mean) / std * latent_multiplier
    as float32 [C, H, W], label, index), z' = channel(mu); channel None is B0 (z' = mu).

    Randomness of item `index` in epoch e: g = torch.Generator().manual_seed(noise_seed(base_seed,
    e_eff, index)) with e_eff = e if resample_each_epoch else 0. First u = torch.rand((),
    generator=g), and the flipped latent is used iff u < flip_prob; then eps = torch.randn((C, H,
    W), generator=g); z' = channel(mu[None], noise=eps[None])[0]. u is drawn even when
    flip_prob = 0, so eps does not depend on the flip setting. All arms share one cache and one
    (mean, std) = LatentStats.channel_norm() (per-channel arrays [C]).

    Call set_epoch(e) before creating each epoch's DataLoader iterator, and use
    persistent_workers=False with num_workers > 0: persistent workers keep the dataset copy
    they started with and would never see the new epoch.
    """

    def __init__(self, cache: LatentCache, channel: RWChannel | None, norm_mean, norm_std, *,
                 base_seed: int, flip_prob: float = 0.5, resample_each_epoch: bool = True,
                 latent_multiplier: float = 1.0):
        C = cache.latent_shape[0]
        if channel is not None and channel.C != C:
            raise ValueError(f"channel acts on C={channel.C} channels, cached latents have C={C}")
        self.mean = _channel_vector("norm_mean", norm_mean, C).view(C, 1, 1)
        self.std = _channel_vector("norm_std", norm_std, C).view(C, 1, 1)
        if bool((self.std <= 0).any()):
            raise ValueError(f"norm_std must be > 0, got min {float(self.std.min())}")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must lie in [0, 1], got {flip_prob!r}")
        if flip_prob > 0 and not cache.has_flip:
            raise ValueError(f"flip_prob={flip_prob} but the cache stores no latents_flip")
        if not np.isfinite(latent_multiplier) or latent_multiplier <= 0:
            raise ValueError(f"latent_multiplier must be finite and > 0, got {latent_multiplier!r}")
        self.cache, self.channel = cache, channel
        self.base_seed = _int_arg("base_seed", base_seed)
        self.flip_prob = float(flip_prob)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.latent_multiplier = float(latent_multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _int_arg("epoch", epoch)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        epoch = self.epoch if self.resample_each_epoch else 0
        g = torch.Generator().manual_seed(noise_seed(self.base_seed, epoch, index))
        flip = bool(torch.rand((), generator=g) < self.flip_prob)
        mu, label, index = self.cache.get(index, flip=flip)
        if self.channel is None:
            z = mu
        else:
            eps = torch.randn(mu.shape, generator=g, dtype=torch.float32)
            z = self.channel(mu[None], noise=eps[None])[0]
        x = (z - self.mean.to(z.dtype)) / self.std.to(z.dtype) * self.latent_multiplier
        return x.float(), label, index


class ImageLatentPairDataset(Dataset):
    """Decoder adaptation / reconstruction data: item k -> (image float32 [3, S, S] in [-1, 1],
    mu float32 [C, H, W], label, index) for row index = indices[k] of the split cache.split.
    The cache must come from this manifest (fingerprint checked at init) and every item's cached
    label must equal its manifest label (checked per item)."""

    def __init__(self, manifest: Manifest, cache: LatentCache, *, image_size: int = 256,
                 indices: Sequence[int] | None = None):
        want, got = manifest.fingerprint(), cache.meta.get("manifest_fingerprint")
        if got != want:
            raise ValueError(f"cache has manifest_fingerprint {got!r}, the given manifest has {want!r}")
        n = manifest.split_size(cache.split)
        if n != len(cache):
            raise ValueError(f"cache has {len(cache)} rows, manifest split {cache.split!r} has {n}")
        self.images = ManifestImageDataset(manifest, cache.split, image_size=image_size, indices=indices)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        image, label, index = self.images[k]
        mu, cache_label, _ = self.cache.get(index)
        if cache_label != label:
            raise ValueError(f"split {self.cache.split!r} row {index}: cached label {cache_label} "
                             f"!= manifest label {label}")
        return image, mu, label, index
