"""Contract tests for rwvae.datasets (ARCHITECTURE_V2 Sec. 3.7; plan v2.1 Sec. 1.2 and 3.2).

A tiny synthetic ImageFolder (3 classes x 7 train / 2 val images with mixed sizes, aspect
ratios, PIL modes and extensions) is built in tmp_path; latent caches hold small random
latents [C=4, H=3, W=3], so everything runs on CPU in seconds. The flip / noise protocol of
ChannelLatentDataset is checked by recomputing it independently from noise_seed.
"""
import copy
import hashlib
import importlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from rwvae.channel import RWChannel, noise_seed
from rwvae.datasets import (
    CACHE_DTYPES, CACHE_INDEX, PREPROCESSING, SPLITS, ChannelLatentDataset, ImageLatentPairDataset,
    LatentCache, LatentCacheWriter, Manifest, ManifestImageDataset, build_manifest, center_crop_arr,
    scan_image_folder,
)

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["n02000002", "n01000001", "n10000010"]          # created unsorted; ImageFolder sorts
TRAIN_SIZES = [(40, 30), (30, 40), (64, 64), (100, 20), (17, 90), (50, 50), (33, 47)]   # (w, h)
FORMATS = [(".JPEG", "RGB"), (".png", "L"), (".png", "RGBA"), (".jpg", "RGB")]
S = 16                                                       # test image size
LAT = (4, 3, 3)                                              # test latent shape [C, H, W]
MEAN = np.array([0.1, -0.2, 0.3, 0.0])
STD = np.array([1.5, 0.5, 2.0, 1.0])
MEAN32, STD32 = torch.tensor(MEAN, dtype=torch.float32).view(-1, 1, 1), torch.tensor(STD, dtype=torch.float32).view(-1, 1, 1)


# ----------------------------------------------------------------------------- fixtures / helpers

def _save_image(path: Path, size, mode: str, rng) -> None:
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).convert(mode).save(path)


@pytest.fixture
def image_folder(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    for c, wnid in enumerate(CLASSES):
        for j, size in enumerate(TRAIN_SIZES):
            ext, mode = FORMATS[j % len(FORMATS)]
            _save_image(root / "train" / wnid / f"{wnid}_{j:04d}{ext}", size, mode, rng)
        for j in range(2):
            _save_image(root / "val" / wnid / f"{wnid}_val{j:04d}.JPEG", (45 + 7 * j, 25 + 11 * c), "RGB", rng)
    (root / "train" / CLASSES[0] / "notes.txt").write_text("not an image")
    (root / "train" / "README.txt").write_text("not a class directory")
    return root


def _build(root: Path, seed: int = 7) -> Manifest:
    return build_manifest(root / "train", root / "val", dev_per_class=2, seed=seed, name="synthetic", image_size=S)


@pytest.fixture
def manifest(image_folder) -> Manifest:
    return _build(image_folder)


@pytest.fixture
def lightningdit_center_crop(monkeypatch):
    """LightningDiT's tokenizer.vavae.center_crop_arr (read-only third-party reference)."""
    pytest.importorskip("omegaconf")
    root = Path(os.environ.get("RWVAE_LIGHTNINGDIT_ROOT", REPO.parent / "third_party" / "LightningDiT"))
    if not (root / "tokenizer" / "vavae.py").is_file():
        pytest.skip(f"LightningDiT not found at {root}")
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("tokenizer.vavae").center_crop_arr


def _write_cache(out_dir, *, total=11, shard_size=4, dtype="float32", flip=True, meta=None, labels=None,
                 order=None, seed=0):
    """Write a random cache of `total` rows (shards in `order`); returns (writer, lat, lat_flip, labels)."""
    gen = torch.Generator().manual_seed(seed)
    lat = 3.0 * torch.randn((total, *LAT), generator=gen)
    lat_flip = torch.flip(lat, dims=[-1]) + 0.25
    labels = torch.arange(total) % 3 if labels is None else torch.as_tensor(labels, dtype=torch.int64)
    writer = LatentCacheWriter(out_dir, split="val", meta={"source": "test"} if meta is None else meta,
                               dtype=dtype, shard_size=shard_size)
    for k in (range(writer.num_shards(total)) if order is None else order):
        rows = writer.shard_rows(k, total)
        sl = slice(rows.start, rows.stop)
        writer.write_shard(k, lat[sl], labels[sl], torch.arange(rows.start, rows.stop),
                           latents_flip=lat_flip[sl] if flip else None)
    return writer, lat, lat_flip, labels


@pytest.fixture
def latent_cache(tmp_path):
    writer, lat, lat_flip, labels = _write_cache(tmp_path / "cache", total=10, shard_size=4)
    writer.finalize(10)
    return LatentCache(tmp_path / "cache"), lat, lat_flip, labels


def _channel() -> RWChannel:
    rng = np.random.default_rng(3)
    C = LAT[0]
    return RWChannel(rng.standard_normal(C), 0.7 * np.eye(C) + 0.05 * rng.standard_normal((C, C)),
                     0.5 * rng.standard_normal((C, C)), rate=3.0)


# ----------------------------------------------------------------------------- manifest

def test_scan_image_folder_follows_imagefolder(image_folder):
    train = image_folder / "train"
    classes, samples = scan_image_folder(train)
    assert classes == sorted(CLASSES)
    assert len(samples) == 3 * 7                  # notes.txt ignored, .JPEG matched case-insensitively
    ref = ImageFolder(str(train))                 # LightningDiT's class map and sample order
    assert ref.classes == classes
    norm = lambda p: os.path.normcase(os.path.normpath(p))
    assert [(norm(p), y) for p, y in ref.samples] == [(norm(train / rel), y) for rel, y in samples]
    with pytest.raises(ValueError, match="not a directory"):
        scan_image_folder(image_folder / "missing")


def test_build_manifest_is_deterministic_and_disjoint(image_folder):
    a, b, c = _build(image_folder), _build(image_folder), _build(image_folder, seed=8)
    assert a.fingerprint() == b.fingerprint() and a.splits == b.splits and a.classes == sorted(CLASSES)
    assert a.splits["train_dev"] != c.splits["train_dev"] and a.fingerprint() != c.fingerprint()
    assert a.splits["val"] == c.splits["val"]

    counts = {"train_dev": 3 * 2, "train_fit": 3 * 5, "val": 3 * 2}
    assert {s: a.split_size(s) for s in SPLITS} == counts
    assert np.bincount(a.splits["train_dev"]["label"]).tolist() == [2, 2, 2]
    dev, fit = set(a.splits["train_dev"]["path"]), set(a.splits["train_fit"]["path"])
    _, samples = scan_image_folder(image_folder / "train")
    assert not dev & fit and dev | fit == {rel for rel, _ in samples}
    ids = [i for s in SPLITS for i in a.splits[s]["image_id"]]
    assert len(ids) == len(set(ids)) == 27

    # the documented selection rule, recomputed independently
    for label in range(3):
        files = [rel for rel, y in samples if y == label]
        pick = np.random.default_rng([7, label]).choice(len(files), 2, replace=False)
        got = [p for p, y in zip(a.splits["train_dev"]["path"], a.splits["train_dev"]["label"]) if y == label]
        assert got == [files[j] for j in sorted(pick)]
    assert a.splits["val"]["label"] == [0, 0, 1, 1, 2, 2]      # val takes the train class map

    m = a.meta
    assert (m["name"], m["seed"], m["dev_per_class"], m["image_size"], m["schema_version"]) == ("synthetic", 7, 2, S, 1)
    assert m["counts"] == counts and m["preprocessing"] == PREPROCESSING
    assert Path(m["train_root"]).is_absolute() and Path(m["val_root"]).is_absolute()
    assert m["split_roots"] == {"train_dev": m["train_root"], "train_fit": m["train_root"], "val": m["val_root"]}
    assert m["class_map_sha256"] == hashlib.sha256(json.dumps(sorted(CLASSES), separators=(",", ":")).encode()).hexdigest()
    datetime.fromisoformat(m["created"])


def test_manifest_save_load_roundtrip(manifest, tmp_path):
    path = tmp_path / "manifests" / "split.json"
    manifest.save(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["schema_version"] == 1 and set(obj) == {"schema_version", "meta", "classes", "splits"}
    loaded = Manifest.load(path)
    assert loaded == manifest and loaded.fingerprint() == manifest.fingerprint()

    # meta is not fingerprinted: relocating the data keeps the fingerprint
    moved = Manifest(meta={**manifest.meta, "split_roots": {s: "/elsewhere" for s in SPLITS}},
                     classes=manifest.classes, splits=manifest.splits)
    assert moved.fingerprint() == manifest.fingerprint()

    rec = manifest.record("val", 3)
    assert rec == {"image_id": f"{sorted(CLASSES)[1]}_val0001", "path": f"{sorted(CLASSES)[1]}/{sorted(CLASSES)[1]}_val0001.JPEG", "label": 1}
    assert manifest.image_path("val", 3).is_file()
    with pytest.raises(IndexError):
        manifest.record("val", 6)
    with pytest.raises(ValueError, match="unknown split"):
        manifest.split_size("test")

    obj["schema_version"] = 2
    (tmp_path / "v2.json").write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        Manifest.load(tmp_path / "v2.json")


def test_manifest_validate_catches_corruption(manifest):
    manifest.validate()

    def corrupt(edit):
        m = copy.deepcopy(manifest)
        edit(m.splits)
        return m

    cases = {
        "image_id": lambda s: s["val"]["image_id"].__setitem__(0, s["train_fit"]["image_id"][4]),  # across splits
        "image_id ": lambda s: s["val"]["image_id"].__setitem__(1, s["val"]["image_id"][0]),       # within a split
        "labels": lambda s: s["val"]["label"].__setitem__(0, 3),
        "labels ": lambda s: s["train_dev"]["label"].__setitem__(0, -1),
        "labels  ": lambda s: s["train_dev"]["label"].__setitem__(0, 1.0),
        "lengths differ": lambda s: s["val"]["path"].pop(),
        "columns": lambda s: s["val"].pop("path"),
    }
    for match, edit in cases.items():
        with pytest.raises(ValueError, match=match.strip()):
            corrupt(edit).validate()


def test_build_manifest_rejects_invalid_inputs(image_folder):
    train, val = image_folder / "train", image_folder / "val"
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="dev_per_class=7"):
        build_manifest(train, val, dev_per_class=7)                # every class has only 7 images
    for kwargs in (dict(dev_per_class=0), dict(seed=-1), dict(image_size=0), dict(dev_per_class=2.0)):
        with pytest.raises(ValueError):
            build_manifest(train, val, **kwargs)
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_manifest(train, train, dev_per_class=2)              # val rows would repeat train IDs

    (val / "n99999999").mkdir()
    with pytest.raises(ValueError, match="no images"):
        build_manifest(train, val, dev_per_class=2)
    _save_image(val / "n99999999" / "n99999999_val0000.JPEG", (20, 20), "RGB", rng)
    with pytest.raises(ValueError, match="not train classes"):
        build_manifest(train, val, dev_per_class=2)

    (val / "n99999999" / "n99999999_val0000.JPEG").unlink()
    (val / "n99999999").rmdir()
    wnid = sorted(CLASSES)[0]
    _save_image(train / wnid / f"{wnid}_0000.bmp", (20, 20), "RGB", rng)   # same stem as the .JPEG
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_manifest(train, val, dev_per_class=2)


# ----------------------------------------------------------------------------- images

@pytest.mark.parametrize("image_size", [256, 64])
def test_center_crop_matches_lightningdit(lightningdit_center_crop, image_size):
    rng = np.random.default_rng(image_size)
    sizes = [(600, 400), (512, 512), (257, 256), (255, 300), (1000, 37), (64, 900),
             (2 * image_size, 3 * image_size + 1), (4 * image_size - 1, 4 * image_size)]
    sizes += [tuple(int(v) for v in rng.integers(20, 700, 2)) for _ in range(6)]
    for w, h in sizes:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        ours = np.asarray(center_crop_arr(img, image_size))
        ref = np.asarray(lightningdit_center_crop(img, image_size))
        assert ours.shape == (image_size, image_size, 3)
        assert np.array_equal(ours, ref), (w, h)


def test_manifest_image_dataset(manifest):
    for split in SPLITS:          # every PIL mode (L / RGB / RGBA) and aspect ratio of the fixture
        ds = ManifestImageDataset(manifest, split, image_size=S)
        assert len(ds) == manifest.split_size(split)
        for k in range(len(ds)):
            x, label, index = ds[k]
            assert x.shape == (3, S, S) and x.dtype == torch.float32
            assert -1.0 <= float(x.min()) and float(x.max()) <= 1.0
            assert (label, index) == (manifest.splits[split]["label"][k], k)

    ds = ManifestImageDataset(manifest, "train_fit", image_size=S)
    with Image.open(manifest.image_path("train_fit", 0)) as im:      # LightningDiT pipeline, by hand
        ref = TF.normalize(TF.to_tensor(center_crop_arr(im.convert("RGB"), S)), [0.5] * 3, [0.5] * 3)
    assert torch.equal(ds[0][0], ref)

    flipped = ManifestImageDataset(manifest, "train_fit", image_size=S, hflip=True)
    for k in range(4):
        assert torch.equal(flipped[k][0], torch.flip(ds[k][0], dims=[2]))

    sub = ManifestImageDataset(manifest, "val", image_size=S, indices=np.array([5, 0]))
    assert len(sub) == 2 and [sub[k][2] for k in range(2)] == [5, 0]
    clone = pickle.loads(pickle.dumps(sub))                           # spawn-worker safe
    assert torch.equal(clone[0][0], sub[0][0])

    for kwargs in (dict(indices=[6]), dict(indices=[-1]), dict(indices=[0.5]), dict(image_size=0)):
        with pytest.raises(ValueError):
            ManifestImageDataset(manifest, "val", **kwargs)
    with pytest.raises(ValueError, match="unknown split"):
        ManifestImageDataset(manifest, "test")


# ----------------------------------------------------------------------------- latent cache

@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cache_roundtrip_out_of_order(tmp_path, dtype):
    out = tmp_path / "cache"
    writer, lat, lat_flip, labels = _write_cache(out, dtype=dtype, order=[2, 0, 1])
    assert not list(out.glob("*.tmp"))
    index = writer.finalize(11)
    assert json.loads((out / CACHE_INDEX).read_text(encoding="utf-8")) == index
    assert (index["split"], index["total"], index["dtype"], index["has_flip"], index["shard_size"]) == ("val", 11, dtype, True, 4)
    assert index["latent_shape"] == list(LAT) and index["meta"] == {"source": "test"}
    assert [(s["file"], s["shard_id"], s["n"]) for s in index["shards"]] == [
        ("val_shard00000.safetensors", 0, 4), ("val_shard00001.safetensors", 1, 4), ("val_shard00002.safetensors", 2, 3)]
    assert [s["sha256"] for s in index["shards"]] == [
        hashlib.sha256((out / s["file"]).read_bytes()).hexdigest() for s in index["shards"]]

    stored, stored_flip = lat.to(CACHE_DTYPES[dtype]).float(), lat_flip.to(CACHE_DTYPES[dtype]).float()
    if dtype == "float32":
        assert torch.equal(stored, lat)                                # exact
    else:                                                              # one rounding to the storage dtype
        unit = {"float16": 2.0 ** -11, "bfloat16": 2.0 ** -8}[dtype]
        assert bool(((stored - lat).abs() <= unit * lat.abs() + 1e-7).all())

    cache = LatentCache(out)
    assert (len(cache), cache.split, cache.has_flip, cache.latent_shape, cache.meta) == (11, "val", True, LAT, {"source": "test"})
    for i in range(11):
        mu, label, index_ = cache.get(i)
        assert mu.dtype == torch.float32 and torch.equal(mu, stored[i]) and (label, index_) == (int(labels[i]), i)
        assert torch.equal(cache.get(i, flip=True)[0], stored_flip[i])
    with pytest.raises(IndexError):
        cache.get(11)

    batches = list(cache.iter_batches(3))                              # batches span shard boundaries
    assert [b[0].shape[0] for b in batches] == [3, 3, 3, 2]
    assert torch.equal(torch.cat([b[0] for b in batches]), stored)
    assert torch.equal(torch.cat([b[1] for b in batches]), labels)
    assert torch.equal(torch.cat([b[2] for b in batches]), torch.arange(11))


def test_cache_without_flips(tmp_path):
    writer, lat, _, _ = _write_cache(tmp_path / "cache", flip=False)
    assert writer.finalize(11)["has_flip"] is False
    cache = LatentCache(tmp_path / "cache")
    assert not cache.has_flip and torch.equal(cache.get(7)[0], lat[7])
    with pytest.raises(ValueError, match="latents_flip"):
        cache.get(0, flip=True)


def test_write_shard_accepts_views_of_one_buffer(tmp_path):
    # e.g. a script encodes [x; flip(x)] in one batch and passes the two halves (shared storage)
    both = torch.randn(2, 4, *LAT)
    ints = torch.stack([torch.arange(4) % 3, torch.arange(4)])
    writer = LatentCacheWriter(tmp_path, split="val", meta={}, dtype="float32", shard_size=4)
    writer.write_shard(0, both[0], ints[0], ints[1], latents_flip=both[1])
    writer.finalize(4)
    cache = LatentCache(tmp_path)
    assert torch.equal(cache.get(2)[0], both[0, 2]) and torch.equal(cache.get(2, flip=True)[0], both[1, 2])


def test_finalize_detects_bad_shard_sets(tmp_path):
    writer, *_ = _write_cache(tmp_path / "a", order=[0, 2])
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        writer.finalize(11)
    assert not (tmp_path / "a" / CACHE_INDEX).exists()

    writer, *_ = _write_cache(tmp_path / "b")
    with pytest.raises(ValueError, match=r"unexpected \[2\]"):
        writer.finalize(7)                        # 7 rows need shards 0..1 only
    with pytest.raises(ValueError, match="metadata"):
        writer.finalize(12)                       # shard 2 would need 4 rows, it has 3
    with pytest.raises(ValueError, match="meta differs"):
        LatentCacheWriter(tmp_path / "b", split="val", meta={"source": "other"}, dtype="float32",
                          shard_size=4).finalize(11)
    with pytest.raises(ValueError, match="metadata"):
        LatentCacheWriter(tmp_path / "b", split="val", meta={"source": "test"}, dtype="float16",
                          shard_size=4).finalize(11)

    mixed = LatentCacheWriter(tmp_path / "c", split="val", meta={}, dtype="float32", shard_size=4)
    lat = torch.randn(8, *LAT)
    mixed.write_shard(0, lat[:4], torch.zeros(4, dtype=torch.int64), torch.arange(4), latents_flip=lat[:4])
    mixed.write_shard(1, lat[4:], torch.zeros(4, dtype=torch.int64), torch.arange(4, 8))
    with pytest.raises(ValueError, match="has_flip"):
        mixed.finalize(8)

    # one split per directory: another split's finalize must not replace the existing index
    writer, *_ = _write_cache(tmp_path / "d")
    writer.finalize(11)
    fit = LatentCacheWriter(tmp_path / "d", split="train_fit", meta={}, dtype="float32", shard_size=4)
    fit.write_shard(0, torch.randn(2, *LAT), torch.zeros(2, dtype=torch.int64), torch.arange(2))
    with pytest.raises(ValueError, match="one directory per split"):
        fit.finalize(2)
    assert LatentCache(tmp_path / "d").split == "val"


def test_write_shard_validation(tmp_path):
    w = LatentCacheWriter(tmp_path, split="val", meta={}, dtype="float16", shard_size=4)
    lat, labels, idx = torch.randn(4, *LAT), torch.zeros(4, dtype=torch.int64), torch.arange(4)
    nan = lat.clone()
    nan[0, 0, 0, 0] = float("nan")
    bad_calls = {
        "rows": lambda: w.write_shard(1, lat, labels, idx),                # shard 1 holds rows 4..7
        "rows ": lambda: w.write_shard(0, lat, labels, idx.flip(0)),       # out of order
        "1..4": lambda: w.write_shard(0, torch.randn(5, *LAT), torch.zeros(5, dtype=torch.int64), torch.arange(5)),
        "length": lambda: w.write_shard(0, lat, labels[:3], idx),
        "integer": lambda: w.write_shard(0, lat, labels.float(), idx),
        ">= 0": lambda: w.write_shard(0, lat, labels - 1, idx),
        "not finite": lambda: w.write_shard(0, lat * 1e5, labels, idx),    # overflows float16
        "not finite ": lambda: w.write_shard(0, nan, labels, idx),
        "latents_flip shape": lambda: w.write_shard(0, lat, labels, idx, latents_flip=lat[:, :2]),
        "floating tensor": lambda: w.write_shard(0, lat[0], labels, idx),
    }
    for match, call in bad_calls.items():
        with pytest.raises(ValueError, match=match.strip()):
            call()
    assert list(tmp_path.iterdir()) == []                                   # nothing was written

    for kwargs in (dict(dtype="float64"), dict(split="a/b"), dict(meta={"x": object()}), dict(shard_size=0)):
        with pytest.raises(ValueError):
            LatentCacheWriter(tmp_path / "x", **{"split": "val", "meta": {}, **kwargs})
    with pytest.raises(ValueError):
        w.shard_rows(3, 11)                                                 # 11 rows -> shards 0..2
    with pytest.raises(ValueError):
        w.finalize(0)


def test_shard_done_and_resume(tmp_path):
    out, meta = tmp_path / "cache", {"source": "test"}
    lat, labels = torch.randn((11, *LAT), generator=torch.Generator().manual_seed(1)), torch.arange(11) % 3
    first = LatentCacheWriter(out, split="val", meta=meta, dtype="float32", shard_size=4)
    assert first.num_shards(11) == 3 and first.shard_rows(2, 11) == range(8, 11)
    assert not first.shard_done(0, 4)
    first.write_shard(0, lat[0:4], labels[0:4], torch.arange(0, 4))         # ... then the run "crashes"
    assert first.shard_done(0, 4) and not first.shard_done(0, 3) and not first.shard_done(1, 4)

    resumed = LatentCacheWriter(out, split="val", meta=dict(meta), dtype="float32", shard_size=4)
    written = []
    for k in range(resumed.num_shards(11)):
        rows = resumed.shard_rows(k, 11)
        if not resumed.shard_done(k, len(rows)):
            resumed.write_shard(k, lat[rows.start:rows.stop], labels[rows.start:rows.stop],
                                torch.arange(rows.start, rows.stop))
            written.append(k)
    assert written == [1, 2]
    resumed.finalize(11)
    assert torch.equal(torch.cat([b[0] for b in LatentCache(out).iter_batches(5)]), lat)

    for other in (dict(meta={"source": "other"}, dtype="float32"), dict(meta=meta, dtype="float16")):
        with pytest.raises(ValueError, match="different meta"):             # never mix runs
            LatentCacheWriter(out, split="val", shard_size=4, **other).shard_done(0, 4)


def test_latent_cache_checks_against_manifest(image_folder, manifest, tmp_path):
    fp, labels = manifest.fingerprint(), manifest.splits["val"]["label"]
    _write_cache(tmp_path / "good", total=6, meta={"manifest_fingerprint": fp}, labels=labels)[0].finalize(6)
    cache = LatentCache(tmp_path / "good", manifest=manifest)
    assert [cache.get(i)[1] for i in range(6)] == labels

    with pytest.raises(ValueError, match="manifest_fingerprint"):
        LatentCache(tmp_path / "good", manifest=_build(image_folder, seed=8))   # another train_dev split
    _write_cache(tmp_path / "nofp", total=6, labels=labels)[0].finalize(6)
    with pytest.raises(ValueError, match="manifest_fingerprint"):
        LatentCache(tmp_path / "nofp", manifest=manifest)

    wrong = [(y + 1) % 3 for y in labels]
    _write_cache(tmp_path / "bad", total=6, meta={"manifest_fingerprint": fp}, labels=wrong)[0].finalize(6)
    with pytest.raises(ValueError, match="labels differ"):
        LatentCache(tmp_path / "bad", manifest=manifest)
    assert len(LatentCache(tmp_path / "bad")) == 6                          # structure alone is fine


def test_latent_cache_verify_detects_tampering(tmp_path):
    out = tmp_path / "cache"
    writer, *_ = _write_cache(out)
    index = writer.finalize(11)
    with pytest.raises(FileNotFoundError):
        LatentCache(tmp_path)

    (out / CACHE_INDEX).write_text(json.dumps({**index, "total": 12}), encoding="utf-8")
    for verify in (True, False):
        with pytest.raises(ValueError):
            LatentCache(out, verify=verify)
    (out / CACHE_INDEX).write_text(json.dumps(index), encoding="utf-8")

    # shard 1 overwritten by another run: verify catches it, verify=False only checks the index
    LatentCacheWriter(out, split="val", meta={"source": "other"}, dtype="float32", shard_size=4).write_shard(
        1, torch.zeros(4, *LAT), torch.zeros(4, dtype=torch.int64), torch.arange(4, 8))
    with pytest.raises(ValueError, match="meta differs"):
        LatentCache(out)
    assert len(LatentCache(out, verify=False)) == 11


def test_latent_cache_pickles_without_handles(tmp_path):
    writer, *_ = _write_cache(tmp_path / "cache")
    writer.finalize(11)
    cache = LatentCache(tmp_path / "cache")
    before = cache.get(5)[0]                       # opens a safetensors handle
    assert cache._handles
    clone = pickle.loads(pickle.dumps(cache))
    assert clone._handles == {}
    assert torch.equal(clone.get(5)[0], before)
    assert torch.equal(clone.get(9, flip=True)[0], cache.get(9, flip=True)[0])


# ----------------------------------------------------------------------------- channel dataset

def test_channel_dataset_reproducible_per_seed_epoch_index(latent_cache):
    cache, _, _, labels = latent_cache
    make = lambda **kw: ChannelLatentDataset(cache, _channel(), MEAN, STD, base_seed=11, **kw)
    a, b = make(), make()
    epoch0 = [a[i][0] for i in range(len(a))]
    assert all(torch.equal(b[i][0], x) for i, x in enumerate(epoch0))
    x, label, index = a[3]
    assert x.shape == LAT and x.dtype == torch.float32 and (label, index) == (int(labels[3]), 3)

    a.set_epoch(1)
    assert all(not torch.equal(a[i][0], x) for i, x in enumerate(epoch0))
    a.set_epoch(0)
    assert all(torch.equal(a[i][0], x) for i, x in enumerate(epoch0))
    assert not torch.equal(ChannelLatentDataset(cache, _channel(), MEAN, STD, base_seed=12)[0][0], epoch0[0])

    frozen = make(resample_each_epoch=False)
    frozen.set_epoch(5)                            # epoch_eff = 0
    assert all(torch.equal(frozen[i][0], x) for i, x in enumerate(epoch0))


def test_channel_dataset_follows_documented_rng_protocol(latent_cache):
    cache, lat, lat_flip, _ = latent_cache
    ch = _channel()
    ds = ChannelLatentDataset(cache, ch, MEAN, STD, base_seed=11, flip_prob=0.5, latent_multiplier=0.5)
    no_flip = ChannelLatentDataset(cache, ch, MEAN, STD, base_seed=11, flip_prob=0.0, latent_multiplier=0.5)
    ds.set_epoch(2)
    no_flip.set_epoch(2)
    flips = []
    for i in range(len(ds)):
        g = torch.Generator().manual_seed(noise_seed(11, 2, i))
        flip = bool(torch.rand((), generator=g) < 0.5)
        eps = torch.randn(LAT, generator=g, dtype=torch.float32)
        z = ch((lat_flip if flip else lat)[i][None], noise=eps[None])[0]
        assert torch.equal(ds[i][0], (z - MEAN32) / STD32 * 0.5)
        if not flip:                               # u is drawn regardless, so eps matches
            assert torch.equal(no_flip[i][0], ds[i][0])
        flips.append(flip)
    assert 0 < sum(flips) < len(flips)            # both branches exercised


@pytest.mark.parametrize("identity", [None, "RWChannel.identity"])
def test_identity_channel_output_is_normalized_mu(latent_cache, identity):
    cache, lat, lat_flip, _ = latent_cache
    ch = None if identity is None else RWChannel.identity(LAT[0])
    ds = ChannelLatentDataset(cache, ch, MEAN, STD, base_seed=0, flip_prob=0.0, latent_multiplier=0.18215)
    mean64, std64 = torch.tensor(MEAN).view(-1, 1, 1), torch.tensor(STD).view(-1, 1, 1)
    for i in range(len(ds)):
        x = ds[i][0]
        assert torch.equal(x, (lat[i] - MEAN32) / STD32 * 0.18215)
        assert torch.allclose(x.double(), (lat[i].double() - mean64) / std64 * 0.18215, rtol=1e-6, atol=1e-7)
    always = ChannelLatentDataset(cache, ch, MEAN, STD, base_seed=0, flip_prob=1.0)
    assert all(torch.equal(always[i][0], (lat_flip[i] - MEAN32) / STD32) for i in range(len(always)))


def test_channel_dataset_rejects_invalid_inputs(tmp_path, latent_cache):
    cache = latent_cache[0]
    writer, *_ = _write_cache(tmp_path / "noflip", flip=False)
    writer.finalize(11)
    noflip = LatentCache(tmp_path / "noflip")
    with pytest.raises(ValueError, match="latents_flip"):
        ChannelLatentDataset(noflip, None, MEAN, STD, base_seed=0, flip_prob=0.5)
    assert len(ChannelLatentDataset(noflip, None, MEAN, STD, base_seed=0, flip_prob=0.0)) == 11

    bad_kwargs = [dict(flip_prob=1.5), dict(flip_prob=-0.1), dict(flip_prob=float("nan")), dict(base_seed=-1),
                  dict(latent_multiplier=0.0), dict(latent_multiplier=float("inf")),
                  dict(norm_std=np.array([1.0, 0.0, 1.0, 1.0])), dict(norm_std=np.full(4, np.nan)),
                  dict(norm_mean=MEAN[:3]), dict(channel=RWChannel.identity(5))]
    for kwargs in bad_kwargs:
        args = {"channel": None, "norm_mean": MEAN, "norm_std": STD, "base_seed": 0, **kwargs}
        with pytest.raises(ValueError):
            ChannelLatentDataset(cache, args.pop("channel"), args.pop("norm_mean"), args.pop("norm_std"), **args)
    with pytest.raises(ValueError):
        ChannelLatentDataset(cache, None, MEAN, STD, base_seed=0).set_epoch(-1)


def test_channel_dataset_dataloader_worker_count_invariant(latent_cache):
    ds = ChannelLatentDataset(latent_cache[0], _channel(), MEAN, STD, base_seed=5)

    def run(num_workers):
        epochs = []
        for epoch in range(2):
            ds.set_epoch(epoch)                    # before creating this epoch's iterator
            sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(100 + epoch))
            loader = DataLoader(ds, batch_size=3, sampler=sampler, num_workers=num_workers,
                                persistent_workers=False)
            epochs.append(list(loader))
        return epochs

    serial, parallel = run(0), run(2)
    for batches_s, batches_p in zip(serial, parallel):
        assert len(batches_s) == len(batches_p) == 4
        for bs, bp in zip(batches_s, batches_p):
            assert all(torch.equal(u, v) for u, v in zip(bs, bp))
    by_index = [{int(i): x for b in ep for x, i in zip(b[0], b[2])} for ep in parallel]
    assert all(not torch.equal(by_index[0][i], by_index[1][i]) for i in range(len(ds)))   # epoch reached workers


# ----------------------------------------------------------------------------- image / latent pairs

def test_image_latent_pair_dataset(manifest, tmp_path):
    fp, labels = manifest.fingerprint(), manifest.splits["val"]["label"]
    writer, lat, _, _ = _write_cache(tmp_path / "good", total=6, meta={"manifest_fingerprint": fp}, labels=labels)
    writer.finalize(6)
    pairs = ImageLatentPairDataset(manifest, LatentCache(tmp_path / "good", manifest=manifest), image_size=S,
                                   indices=[4, 1])
    images = ManifestImageDataset(manifest, "val", image_size=S)
    assert len(pairs) == 2
    image, mu, label, index = pairs[0]
    assert (index, label) == (4, labels[4]) and torch.equal(mu, lat[4]) and torch.equal(image, images[4][0])
    assert pickle.loads(pickle.dumps(pairs))[1][3] == 1

    wrong = [(y + 1) % 3 for y in labels]
    _write_cache(tmp_path / "bad", total=6, meta={"manifest_fingerprint": fp}, labels=wrong)[0].finalize(6)
    bad = ImageLatentPairDataset(manifest, LatentCache(tmp_path / "bad"), image_size=S)
    with pytest.raises(ValueError, match="cached label"):
        bad[0]

    _write_cache(tmp_path / "other", total=6, meta={"manifest_fingerprint": "0" * 64}, labels=labels)[0].finalize(6)
    with pytest.raises(ValueError, match="manifest_fingerprint"):
        ImageLatentPairDataset(manifest, LatentCache(tmp_path / "other"), image_size=S)


# ----------------------------------------------------------------------------- real assets

@pytest.mark.real_assets
@pytest.mark.skipif(not os.environ.get("RWVAE_ASSETS"), reason="set RWVAE_ASSETS to the local assets directory")
def test_imagenette_manifest_matches_imagefolder(lightningdit_center_crop):
    """Engineering check on Imagenette (10 ImageNet classes; never a thesis result)."""
    root = Path(os.environ["RWVAE_ASSETS"]) / "datasets" / "imagenette2-320"
    if not root.is_dir():
        pytest.skip(f"{root} not found")
    m = build_manifest(root / "train", root / "val", dev_per_class=20, seed=2026, name="imagenette")
    again = build_manifest(root / "train", root / "val", dev_per_class=20, seed=2026, name="imagenette")
    assert m.fingerprint() == again.fingerprint()

    ref_train, ref_val = ImageFolder(str(root / "train")), ImageFolder(str(root / "val"))
    assert m.classes == ref_train.classes and len(m.classes) == 10
    assert m.split_size("train_dev") == 10 * 20 and m.split_size("train_fit") == len(ref_train) - 200
    norm = lambda p: os.path.normcase(os.path.normpath(p))
    assert [(norm(p), y) for p, y in ref_val.samples] == [
        (norm(m.image_path("val", i)), y) for i, y in enumerate(m.splits["val"]["label"])]

    ds = ManifestImageDataset(m, "val", image_size=256, indices=[0, len(ref_val) - 1])
    for k in range(len(ds)):
        x, _, i = ds[k]
        with Image.open(m.image_path("val", i)) as im:
            ref = lightningdit_center_crop(im.convert("RGB"), 256)
        assert torch.equal(x, TF.normalize(TF.to_tensor(ref), [0.5] * 3, [0.5] * 3))
