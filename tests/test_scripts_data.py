"""Tests for the data scripts: 01_prepare_manifest, 02_cache_latents, tools/make_svp_subset and the
svp / server_a800 config overlays (docs/ARCHITECTURE_V2.md Sec. 4.1, plan v2.1 Sec. 1.2-1.3, 3.2).

CPU tests run the scripts in-process (main([...])) on a synthetic ImageFolder (3 classes x 6 train /
2 val images of varied sizes) with a seeded random AutoencoderKL (ch_mult (1, 2): 32 px -> 16x16
latents), built exactly as in tests/test_integration_v2.py. The run registry is redirected to tmp_path.
The real_assets test caches a few real Imagenette images with the real f16d32 tokenizer on CUDA.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import _common
from rwvae import datasets, runtime
from rwvae.datasets import CACHE_INDEX, LatentCache, Manifest, ManifestImageDataset, image_transform
from rwvae.integration import FrozenTokenizer, add_lightningdit_to_path

prepare = importlib.import_module("01_prepare_manifest")
cache_script = importlib.import_module("02_cache_latents")
REPO = Path(__file__).resolve().parents[1]
# by file path: LightningDiT's top-level `tools` package may shadow scripts/tools once it is on sys.path
_spec = importlib.util.spec_from_file_location("make_svp_subset", REPO / "scripts" / "tools" / "make_svp_subset.py")
svp_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svp_tool)

BASE = [REPO / "configs" / n for n in ("dataset_imagenet256.yaml", "baseline_common.yaml", "arms.yaml")]
ASSETS = os.environ.get("RWVAE_ASSETS")
CLASSES = ["n03000003", "n01000001", "n02000002"]
SIZES = [(40, 30), (30, 41), (64, 64), (100, 20), (17, 90), (50, 50)]      # (w, h)
SHARD, BATCH = 4, 2                                                        # shard_size, cache.batch_size


# ----------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory) -> Path:
    for name in ("torchvision", "requests", "tqdm"):  # imported by tokenizer.autoencoder
        pytest.importorskip(name)
    add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=32, ch_mult=(1, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    path = tmp_path_factory.mktemp("tokenizer") / "plain.ckpt"
    torch.save(ae.state_dict(), path)
    return path


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_ckpt):
    return FrozenTokenizer.from_checkpoint(tiny_ckpt, ch_mult=(1, 2))


def make_folder(root: Path, classes, n_train: int, n_val: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    for c, wnid in enumerate(classes):
        for split, n, tag in (("train", n_train, "tr"), ("val", n_val, "va")):
            for j in range(n):
                w, h = SIZES[(j + c) % len(SIZES)]
                path = root / split / wnid / f"{wnid}_{tag}{j:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
    return root


@pytest.fixture
def setup(tmp_path, tiny_ckpt):
    """Synthetic data + a test overlay; returns (base argv, tmp_path)."""
    data = make_folder(tmp_path / "data", CLASSES, 6, 2)
    overlay = {
        "experiment": {"name": "t", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "dataset": {"name": "synthetic", "train_root": (data / "train").as_posix(),
                    "val_root": (data / "val").as_posix(), "image_size": 32, "num_classes": 3,
                    "dev_per_class": 2, "manifest": (tmp_path / "manifest.json").as_posix()},
        "tokenizer": {"ckpt": tiny_ckpt.as_posix(), "ch_mult": [1, 2], "embed_dim": 32},
        "cache": {"dir": (tmp_path / "cache").as_posix(), "dtype": "float32", "flip": True,
                  "shard_size": SHARD, "batch_size": BATCH, "num_workers": 0, "verify_samples": 5},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    argv = [a for p in [*BASE, path] for a in ("--config", str(p))] + ["--device", "cpu"]
    return argv, tmp_path


def summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["summary"]


def encode_like_script(tok, manifest, split, flip=False) -> torch.Tensor:
    """encode_mean of every row of `split` with the script's batching (shards of SHARD, batches of BATCH)."""
    ds = ManifestImageDataset(manifest, split, image_size=32, hflip=flip)
    n, out = len(ds), []
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n, (start // SHARD + 1) * SHARD)
        out.append(tok.encode_mean(torch.stack([ds[i][0] for i in range(start, stop)])))
        assert stop - start == BATCH or stop == n or stop % SHARD == 0
    return torch.cat(out)


# ----------------------------------------------------------------------------- 01
def test_manifest_created_up_to_date_and_seed_change_needs_force(setup):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    run_dir = tmp / "runs" / "t" / "01_manifest"
    m = Manifest.load(tmp / "manifest.json")
    assert m.meta["counts"] == {"train_dev": 6, "train_fit": 12, "val": 6}
    assert summary(run_dir)["manifest_status"] == "created" and summary(run_dir)["fingerprint"] == m.fingerprint()
    card = tmp / "runs" / "t" / "DATA_CARD.md"
    text = card.read_text(encoding="utf-8")
    assert "## 1. Dataset and splits" in text and m.fingerprint() in text and m.meta["class_map_sha256"] in text
    assert "terms of access" in text and "image ID" in text and "SVP" not in text
    info = json.loads((run_dir / "manifest_info.json").read_text(encoding="utf-8"))
    assert info["manifest_sha256"] == runtime.sha256_file(tmp / "manifest.json") and info["git_commit"] == runtime.git_commit()
    assert len((tmp / "registry.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    card.write_text(text + "\n## 3. Semantic statistics\n\nkept\n", encoding="utf-8")  # another script's section
    mtime = (tmp / "manifest.json").stat().st_mtime_ns
    assert prepare.main(argv + ["--overwrite"]) == 0
    assert summary(run_dir)["manifest_status"] == "up_to_date" and (tmp / "manifest.json").stat().st_mtime_ns == mtime
    text = card.read_text(encoding="utf-8")
    assert text.count("## 1. Dataset and splits") == 1 and "## 3. Semantic statistics\n\nkept" in text

    with pytest.raises(ValueError, match="--force"):
        prepare.main(argv + ["--overwrite", "--set", "dataset.split_seed=7"])
    assert Manifest.load(tmp / "manifest.json").fingerprint() == m.fingerprint()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert prepare.main(argv + ["--overwrite", "--force", "--set", "dataset.split_seed=7"]) == 0
    assert summary(run_dir)["manifest_status"] == "replaced"
    assert Manifest.load(tmp / "manifest.json").fingerprint() != m.fingerprint()


def test_manifest_rejects_wrong_num_classes(setup):
    argv, _ = setup
    with pytest.raises(ValueError, match="num_classes"):
        prepare.main(argv + ["--set", "dataset.num_classes=1000"])


# ----------------------------------------------------------------------------- 02
def test_flip_of_the_tensor_equals_hflip_after_center_crop():
    rng = np.random.default_rng(1)
    for w, h in [(40, 30), (33, 71), (300, 257), (64, 64)]:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        x = image_transform(32)(img)
        assert torch.equal(torch.flip(x, dims=[-1]), image_transform(32, hflip=True)(img))


def test_cache_all_splits_with_flips_matches_encode_mean_and_resumes(setup, tiny_tokenizer):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    assert cache_script.main(argv) == 0
    m = Manifest.load(tmp / "manifest.json")
    run_dir = tmp / "runs" / "t" / "02_cache" / "train_fit+train_dev+val"
    for split in datasets.SPLITS:
        cache = LatentCache(tmp / "cache" / split, manifest=m, verify=True)
        assert cache.has_flip and cache.dtype == "float32" and cache.latent_shape == (32, 16, 16)
        meta = cache.meta
        assert meta["latent_source"] == "posterior_mean" and meta["manifest_fingerprint"] == m.fingerprint()
        assert meta["tokenizer"]["checkpoint_sha256"] == tiny_tokenizer.meta["checkpoint_sha256"]
        assert meta["tf32"] == {"tf32_matmul": False, "tf32_cudnn": False} and meta["split"] == split
        assert meta["git_commit"] == runtime.git_commit() and meta["flip"] is True
        mu = torch.stack([cache.get(i)[0] for i in range(len(cache))])
        mu_flip = torch.stack([cache.get(i, flip=True)[0] for i in range(len(cache))])
        assert torch.equal(mu, encode_like_script(tiny_tokenizer, m, split))           # fp32 storage: exact
        assert torch.equal(mu_flip, encode_like_script(tiny_tokenizer, m, split, flip=True))
        v = json.loads((tmp / "cache" / split / cache_script.VERIFY_FILE).read_text(encoding="utf-8"))
        assert v["passed"] and v["num_rows"] == 5 and v["flip_checked"] and v["worst_ratio"] <= 1.0
        assert v["cache_index_sha256"] == runtime.sha256_file(tmp / "cache" / split / CACHE_INDEX)
    s = summary(run_dir)["splits"]
    assert s["train_fit"]["written"] == [0, 1, 2] and s["val"]["written"] == [0, 1]
    assert all(r["status"] == "finalized" for r in s.values())
    text = (tmp / "runs" / "t" / "DATA_CARD.md").read_text(encoding="utf-8")
    assert "## 1. Dataset and splits" in text and "## 2. Latent cache" in text and "PASS" in text
    assert "not verified" not in text and "not cached" not in text

    shas = {p.name: runtime.sha256_file(p) for p in (tmp / "cache" / "train_fit").glob("*.safetensors")}
    assert cache_script.main(argv + ["--resume"]) == 0                                # rerun: nothing re-encoded
    s = summary(run_dir)["splits"]
    assert s["train_fit"]["written"] == [] and s["train_fit"]["skipped"] == [0, 1, 2]
    assert shas == {p.name: runtime.sha256_file(p) for p in (tmp / "cache" / "train_fit").glob("*.safetensors")}


def test_fp16_cache_is_within_storage_rounding(setup, tiny_tokenizer):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    assert cache_script.main(argv + ["--split", "val", "--set", "cache.dtype=float16", "--set", "cache.flip=false"]) == 0
    m = Manifest.load(tmp / "manifest.json")
    cache = LatentCache(tmp / "cache" / "val", manifest=m)
    assert cache.dtype == "float16" and not cache.has_flip
    assert not (tmp / "cache" / "train_fit").exists()
    ref = encode_like_script(tiny_tokenizer, m, "val")
    got = torch.stack([cache.get(i)[0] for i in range(len(cache))])
    assert torch.equal(got, ref.half().float())                       # exactly the fp16 rounding of mu
    assert float((got - ref).abs().max()) <= 2.0 ** -11 * float(ref.abs().max())


def test_two_processes_build_the_same_cache_as_one(setup):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    split = ["--split", "train_fit"]
    two = argv + split + ["--set", f"cache.dir={(tmp / 'cache2').as_posix()}", "--num-procs", "2"]
    assert cache_script.main(two + ["--proc-id", "0"]) == 0
    assert not (tmp / "cache2" / "train_fit" / CACHE_INDEX).exists()   # proc 1's shard is missing
    proc0 = tmp / "runs" / "t" / "02_cache" / "train_fit" / "proc0of2"
    assert summary(proc0)["splits"]["train_fit"] == {"status": "incomplete", "written": [0, 2], "skipped": [],
                                                     "missing": [1]}
    with pytest.raises(FileNotFoundError, match="missing"):
        cache_script.main(two + ["--finalize"])
    assert cache_script.main(two + ["--proc-id", "1"]) == 0
    assert summary(proc0.parent / "proc1of2")["splits"]["train_fit"]["status"] == "shards_done"
    assert not (tmp / "cache2" / "train_fit" / CACHE_INDEX).exists()   # only proc 0 / --finalize finalize
    assert cache_script.main(two + ["--finalize", "--overwrite"]) == 0
    assert cache_script.main(argv + split) == 0                       # one process, cfg cache.dir
    one = json.loads((tmp / "cache" / "train_fit" / CACHE_INDEX).read_text(encoding="utf-8"))
    par = json.loads((tmp / "cache2" / "train_fit" / CACHE_INDEX).read_text(encoding="utf-8"))
    # same index up to the shard sha256 (safetensors writes its metadata keys in a random order) ...
    for index in (one, par):
        for shard in index["shards"]:
            shard.pop("sha256")
    assert one == par and len(one["shards"]) == 3
    # ... and identical tensors and shard metadata
    m = Manifest.load(tmp / "manifest.json")
    a, b = LatentCache(tmp / "cache" / "train_fit", manifest=m), LatentCache(tmp / "cache2" / "train_fit", manifest=m)
    for i in range(len(a)):
        assert torch.equal(a.get(i)[0], b.get(i)[0]) and torch.equal(a.get(i, flip=True)[0], b.get(i, flip=True)[0])


def test_torchrun_environment_selects_the_shards(setup, monkeypatch):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "3")
    assert cache_script.main(argv + ["--split", "train_fit"]) == 0
    s = summary(tmp / "runs" / "t" / "02_cache" / "train_fit" / "proc1of3")["splits"]["train_fit"]
    assert s["written"] == [1] and s["missing"] == [0, 2]


def test_cache_detects_a_corrupted_shard(setup, monkeypatch):
    argv, tmp = setup
    assert prepare.main(argv) == 0
    real = datasets.LatentCacheWriter.write_shard

    def corrupt(self, shard_id, latents, labels, indices, latents_flip=None):  # store mu + 0.01
        return real(self, shard_id, latents + 0.01, labels, indices, latents_flip=latents_flip)

    monkeypatch.setattr(datasets.LatentCacheWriter, "write_shard", corrupt)
    with pytest.raises(ValueError, match="storage rounding"):
        cache_script.main(argv + ["--split", "train_dev"])
    v = json.loads((tmp / "cache" / "train_dev" / cache_script.VERIFY_FILE).read_text(encoding="utf-8"))
    assert not v["passed"] and v["max_abs_err"] > 5e-3


# ----------------------------------------------------------------------------- make_svp_subset
def svp_args(src: Path, out: Path, *extra) -> list[str]:
    return ["--train-root", str(src / "train"), "--val-root", str(src / "val"), "--out", str(out),
            "--num-classes", "3", "--train-per-class", "5", "--val-per-class", "2", "--mode", "copy", *extra]


def test_make_svp_subset_is_deterministic_and_idempotent(tmp_path, capsys):
    assert svp_tool.IMG_EXTENSIONS == datasets.IMG_EXTENSIONS
    src = make_folder(tmp_path / "src", [f"n0{i}" for i in range(5)], 8, 4)
    assert svp_tool.main(svp_args(src, tmp_path / "a")) == 0
    assert svp_tool.main(svp_args(src, tmp_path / "b")) == 0
    a = json.loads((tmp_path / "a" / "svp_subset.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b" / "svp_subset.json").read_text(encoding="utf-8"))
    assert a == b and a["mode"] == "copy" and len(a["wnids"]) == 3 and a["wnids"] == sorted(a["wnids"])
    assert a["counts"] == {"train": {w: 5 for w in a["wnids"]}, "val": {w: 2 for w in a["wnids"]}}
    files = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*.png"))
    assert files == a["files"] and len(files) == 3 * 7
    for f in files:
        assert (tmp_path / "a" / f).read_bytes() == (src / f).read_bytes()
    capsys.readouterr()
    assert svp_tool.main(svp_args(src, tmp_path / "a")) == 0 and "up to date" in capsys.readouterr().out
    with pytest.raises(ValueError, match="differs"):
        svp_tool.main(svp_args(src, tmp_path / "a", "--seed", "1"))
    with pytest.raises(ValueError, match="fewer than"):
        svp_tool.main(svp_args(src, tmp_path / "c", "--train-per-class", "9"))
    assert svp_tool.main(svp_args(src, tmp_path / "s")[:-2] + ["--mode", "symlink"]) == 0  # falls back if needed
    s = json.loads((tmp_path / "s" / "svp_subset.json").read_text(encoding="utf-8"))
    assert s["mode"] in ("symlink", "copy") and s["files"] == a["files"]
    assert all((tmp_path / "s" / f).read_bytes() == (src / f).read_bytes() for f in s["files"])


def test_manifest_of_an_svp_subset_records_its_provenance(setup):
    argv, tmp = setup
    src = make_folder(tmp / "src", [f"n0{i}" for i in range(5)], 8, 4)
    out = tmp / "svp"
    assert svp_tool.main(svp_args(src, out)) == 0
    sub = ["--set", f"dataset.train_root={(out / 'train').as_posix()}",
           "--set", f"dataset.val_root={(out / 'val').as_posix()}"]
    assert prepare.main(argv + sub) == 0
    m = Manifest.load(tmp / "manifest.json")
    assert m.meta["counts"] == {"train_dev": 6, "train_fit": 9, "val": 6}
    text = (tmp / "runs" / "t" / "DATA_CARD.md").read_text(encoding="utf-8")
    record = json.loads((out / "svp_subset.json").read_text(encoding="utf-8"))
    assert "SVP subset provenance" in text and record["files_sha256"] in text and ", ".join(record["wnids"]) in text
    (out / "train" / record["wnids"][0] / sorted(os.listdir(out / "train" / record["wnids"][0]))[0]).unlink()
    with pytest.raises(ValueError, match="modified"):
        prepare.main(argv + sub + ["--overwrite", "--force"])


# ----------------------------------------------------------------------------- config overlays
def leaf_paths(d: dict, prefix=()) -> list[tuple]:
    return [p for k, v in d.items() for p in (leaf_paths(v, prefix + (k,)) if isinstance(v, dict) else [prefix + (k,)])]


def has_path(cfg: dict, path: tuple) -> bool:
    for k in path:
        if not isinstance(cfg, dict) or k not in cfg:
            return False
        cfg = cfg[k]
    return True


def load_stack(monkeypatch, *names) -> dict:
    for key in ("HF_HOME", "HF_ENDPOINT", "TORCH_HOME"):  # keep load_cfg's apply_env out of this process
        monkeypatch.setenv(key, os.environ.get(key, "unset-by-test"))
    argv = [a for p in [*BASE, *(REPO / "configs" / n for n in names)] for a in ("--config", str(p))]
    return _common.load_cfg(_common.make_parser("t").parse_args(argv))


def test_overlays_only_set_existing_keys():
    base = runtime.load_config(BASE)
    for name, new_blocks in (("server_a800.yaml", {"env"}), ("svp.yaml", {"svp"})):
        overlay = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
        unknown = [p for p in leaf_paths(overlay) if p[0] not in new_blocks and not has_path(base, p)]
        assert unknown == [], f"{name} sets keys that no base config defines: {unknown}"


def test_server_overlay_keeps_the_dataset_protocol(monkeypatch):
    base = runtime.load_config(BASE)
    cfg = load_stack(monkeypatch, "server_a800.yaml")
    assert cfg["env"]["HF_ENDPOINT"] == "https://hf-mirror.com" and cfg["env"]["HF_HOME"].startswith("/home/")
    assert cfg["dataset"]["train_root"] == "/home/jinyankai/data/imagenet/train"
    for key in ("name", "num_classes", "dev_per_class", "split_seed", "manifest", "image_size"):
        assert cfg["dataset"][key] == base["dataset"][key]
    assert cfg["experiment"] == base["experiment"] and cfg["train"]["global_batch_size"] == 256
    assert cfg["tokenizer"]["ckpt"].endswith("ldm-imagenet256-f16d32-50ep.ckpt")
    assert cfg["cache"]["dir"].startswith("/home/jinyankai/cache/")
    assert _common.dit_train_config(cfg).global_batch_size == 256


def test_svp_stack_gives_a_valid_overfit_pilot(monkeypatch):
    cfg = load_stack(monkeypatch, "server_a800.yaml", "svp.yaml")
    d, sub = cfg["dataset"], cfg["svp"]["subset"]
    assert cfg["experiment"]["name"] == "svp" and d["num_classes"] == sub["num_classes"] == 10
    assert d["train_root"].endswith("/train") and d["val_root"].endswith("/val") and "svp" in d["train_root"]
    assert d["manifest"] != runtime.load_config(BASE)["dataset"]["manifest"]
    assert "svp" in cfg["cache"]["dir"] and cfg["env"]["TORCH_HOME"].startswith("/home/")   # machine overlay kept
    fit_per_class = sub["train_per_class"] - d["dev_per_class"]
    assert fit_per_class == 80
    assert cfg["eval"]["probe"]["train_split"] == "train_fit" and cfg["eval"]["probe"]["train_per_class"] <= fit_per_class
    assert cfg["decoder_adapt"]["images_per_class"] <= fit_per_class
    assert cfg["eval"]["dct_images"] <= d["dev_per_class"] * d["num_classes"]
    assert cfg["stats"]["alignment_check"]["num_images"] <= fit_per_class * d["num_classes"]
    t = _common.dit_train_config(cfg)
    assert t.max_steps == 20000 and t.ema_decay == 0.999 and t.global_batch_size == 256 and t.keep_last * t.ckpt_every >= t.max_steps
    assert cfg["sample"]["num_samples_screen"] % d["num_classes"] == 0 and cfg["sample"]["num_sampling_steps"] == 50
    crit = cfg["svp"]["pass_criteria"]
    assert crit["allocation_identity_residual"]["max"] == 1e-8
    assert 0 < crit["gfid_memorisation"]["max_ratio"] < 1 and 0 < crit["nn_memorisation"]["max_ratio"] < 1
    assert crit["b0_reconstruction"]["psnr_min"] < crit["b0_reconstruction"]["psnr_max"]


# ----------------------------------------------------------------------------- real assets
@pytest.mark.real_assets
@pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_svp_pipeline_on_imagenette_subset(tmp_path):
    assets = Path(ASSETS)
    out = tmp_path / "svp"
    src = assets / "datasets" / "imagenette2-320"
    assert svp_tool.main(["--train-root", str(src / "train"), "--val-root", str(src / "val"), "--out", str(out),
                          "--num-classes", "2", "--train-per-class", "10", "--val-per-class", "3", "--mode", "copy"]) == 0
    overlay = {
        "experiment": {"name": "svp_real", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "dataset": {"name": "imagenette_svp_test", "train_root": (out / "train").as_posix(),
                    "val_root": (out / "val").as_posix(), "num_classes": 2, "dev_per_class": 4,
                    "manifest": (tmp_path / "manifest.json").as_posix()},
        "tokenizer": {"ckpt": (assets / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt").as_posix()},
        "cache": {"dir": (tmp_path / "cache").as_posix(), "shard_size": 8, "batch_size": 4, "num_workers": 2,
                  "verify_samples": 4},
    }
    path = tmp_path / "real.yaml"
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    argv = [a for p in [*BASE, path] for a in ("--config", str(p))] + ["--device", "cuda"]
    assert prepare.main(argv) == 0
    assert cache_script.main(argv) == 0
    m = Manifest.load(tmp_path / "manifest.json")
    for split in datasets.SPLITS:
        cache = LatentCache(tmp_path / "cache" / split, manifest=m)
        assert cache.dtype == "float16" and cache.has_flip and cache.latent_shape == (32, 16, 16)
        v = json.loads((tmp_path / "cache" / split / cache_script.VERIFY_FILE).read_text(encoding="utf-8"))
        assert v["passed"] and v["max_rel_err"] < 1e-3
    text = (tmp_path / "runs" / "svp_real" / "DATA_CARD.md").read_text(encoding="utf-8")
    assert "SVP subset provenance" in text and "## 2. Latent cache" in text
