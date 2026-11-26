"""In-process tests of scripts/03_estimate_stats.py and scripts/04_solve_channel.py.

CPU only: a synthetic ImageFolder, a manifest and a latent cache built with the rwvae.datasets APIs
(latents = a fixed linear map of the 8x8 average-pooled image + per-image noise), and a fake DINO
(1x1 conv + tanh on the same pooled grid) monkeypatched into _common.make_semantic, so z and y are
correlated token by token and the alignment scan peaks at (0, 0). The run registry goes to tmp.
The real_assets test runs 03 on real Imagenette images (real tokenizer + DINOv2-L, CUDA if present).
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

import _common
from rwvae.allocation import Allocation
from rwvae.datasets import LatentCacheWriter, ManifestImageDataset, build_manifest
from rwvae.stats import LatentStats

REPO = Path(__file__).resolve().parents[1]
BASE = [str(REPO / "configs" / n) for n in ("dataset_imagenet256.yaml", "baseline_common.yaml", "arms.yaml")]
ASSETS = os.environ.get("RWVAE_ASSETS")
IMG, GRID, C, D = 32, 8, 4, 6


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S03 = _load("03_estimate_stats")
S04 = _load("04_solve_channel")


# --------------------------------------------------------------------------- synthetic data
def _pool(images: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(images, GRID)


def _latents(images: torch.Tensor, rows, flip: bool) -> torch.Tensor:
    """mu = [high-variance noise (2 channels); W pool(x) + small noise (2 channels)]: z correlates with the
    image at the same token, Sigma > 0, and the DINO-relevant directions are the LOW-variance ones, so the
    variance (B2) and r2 (B3) rules keep different directions."""
    W = torch.randn(C - 2, 3, generator=torch.Generator().manual_seed(11))
    p = _pool(images)
    noise = torch.stack([torch.randn(C, GRID, GRID, generator=torch.Generator().manual_seed(1000 + int(r)))
                         for r in rows])
    if flip:
        noise = torch.flip(noise, dims=[-1])
    signal = torch.einsum("cd,bdhw->bchw", W, p) + 0.02 * noise[:, 2:]
    return torch.cat([0.5 * noise[:, :2], signal], dim=1)


class FakeDino(nn.Module):
    """1x1 conv + tanh on the pooled grid; shift rolls the output grid by `shift` tokens along w."""

    def __init__(self, shift: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(3, D, 1)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(D, 3, 1, 1, generator=g) * 3)
            self.conv.bias.copy_(torch.randn(D, generator=g) * 0.1)
        self.shift = shift
        self.meta = {"model_name": "fake-dino", "input_size": IMG, "patch_size": IMG // GRID, "grid": GRID,
                     "num_prefix_tokens": 0, "embed_dim": D, "preprocess": "avg-pool to the grid",
                     "tokens": "row-major (h, w)", "checkpoint_path": None}

    @torch.no_grad()
    def forward(self, images):
        y = torch.tanh(self.conv(_pool(images)))
        return torch.roll(y, self.shift, dims=3) if self.shift else y


def _image_folder(root: Path) -> None:
    rng = np.random.default_rng(0)
    for wnid in ("n001", "n002"):
        for split, count in (("train", 12), ("val", 3)):
            for j in range(count):
                path = root / split / wnid / f"{wnid}_{split}{j}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Synthetic manifest + train_fit cache (with flips) + overlay config; returns a helper namespace."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    _image_folder(tmp_path / "data")
    manifest = build_manifest(tmp_path / "data" / "train", tmp_path / "data" / "val", dev_per_class=2, seed=0,
                              name="tiny", image_size=IMG)
    manifest.save(tmp_path / "manifest.json")
    n = manifest.split_size("train_fit")
    images = torch.stack([ManifestImageDataset(manifest, "train_fit", image_size=IMG)[i][0] for i in range(n)])
    rows = list(range(n))
    writer = LatentCacheWriter(tmp_path / "cache" / "train_fit", split="train_fit", dtype="float32", shard_size=8,
                               meta={"manifest_fingerprint": manifest.fingerprint(), "latent_source": "posterior_mean",
                                     "source": "synthetic"})
    mu, mu_f = _latents(images, rows, False), _latents(torch.flip(images, dims=[-1]), rows, True)
    labels = manifest.splits["train_fit"]["label"]
    for k in range(writer.num_shards(n)):
        r = writer.shard_rows(k, n)
        writer.write_shard(k, mu[r.start:r.stop], labels[r.start:r.stop], torch.arange(r.start, r.stop),
                           latents_flip=mu_f[r.start:r.stop])
    writer.finalize(n)
    overlay = {
        "experiment": {"name": "t", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "dataset": {"train_root": (tmp_path / "data" / "train").as_posix(),
                    "val_root": (tmp_path / "data" / "val").as_posix(), "image_size": IMG, "num_classes": 2,
                    "dev_per_class": 2, "manifest": (tmp_path / "manifest.json").as_posix()},
        "cache": {"dir": (tmp_path / "cache").as_posix()},
        "stats": {"num_groups": 4, "batch_size": 6, "num_workers": 0, "ridge_rel": 1.0e-3,
                  "ridge_scan": [0.0, 1.0e-3, 1.0e-2], "alignment_check": {"num_images": 8, "max_shift": 1},
                  "bootstrap": 5},
        "grid": {"R_nats": [0.5, 2.0], "w": [1.5, 3.0]},
    }
    (tmp_path / "overlay.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    cfg_args = sum((["--config", c] for c in BASE + [str(tmp_path / "overlay.yaml")]), [])

    def use_dino(shift=0):
        monkeypatch.setattr(_common, "make_semantic", lambda cfg, device: FakeDino(shift).to(device))

    use_dino(0)

    class E:
        root = tmp_path
        args = cfg_args + ["--device", "cpu"]
        stats_dir = tmp_path / "runs" / "t" / "stats"
        alloc_dir = tmp_path / "runs" / "t" / "allocations"
        n_fit = n
        dino = staticmethod(use_dino)
    return E


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in ("m", "Sigma", "Sigma_zy", "Sigma_y", "Sigma_expl", "n_tokens", "n_images")}


def _registry(env) -> list[dict]:
    return [json.loads(line) for line in (env.root / "registry.jsonl").read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- 03
def test_03_single_process_matches_two_process_merge(env):
    assert S03.main(env.args + ["--figures"]) == 0
    single = _arrays(env.stats_dir / "stats.npz")
    assert int(single["n_images"]) == env.n_fit and int(single["n_tokens"]) == env.n_fit * GRID * GRID
    diag = json.loads((env.stats_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["alignment"]["pass"] and diag["alignment"]["argmax"] == [0, 0]
    assert diag["joint_dim"] == C + D and diag["tokens_per_joint_dim"] == env.n_fit * GRID * GRID / (C + D)
    assert [r["ridge_rel"] for r in diag["ridge_scan"]] == [0.0, 1e-3, 1e-2]
    assert all(0 <= e <= 1 + 1e-9 for r in diag["ridge_scan"] for e in r["eig_K"])
    assert diag["Sigma"]["shrinkage"].startswith("none") and diag["Sigma_y"]["rank"] == D
    assert len(diag["split_half"]["principal_angles_K_rad"]) == C
    assert sorted(p.name for p in (env.stats_dir / "figures").iterdir()) == \
        ["alignment_scan.png", "k_spectrum.png", "sigma_spectrum.png"]
    assert (env.stats_dir / "group_moments.pt").is_file()
    card = (env.root / "runs" / "t" / "DATA_CARD.md").read_text(encoding="utf-8")
    assert card.count("## 3. Semantic statistics") == 1 and "PASS" in card

    other = env.root / "sharded"
    for k in (0, 1):
        assert S03.main(env.args + ["--out", str(other), "--num-procs", "2", "--proc-id", str(k)]) == 0
        assert (other / "shards" / f"acc_{k}.pt").is_file()
    assert S03.main(env.args + ["--out", str(other), "--merge"]) == 0
    merged = _arrays(other / "stats.npz")
    for key, a in single.items():
        np.testing.assert_allclose(merged[key], a, rtol=1e-12, atol=1e-15, err_msg=key)
    stats = LatentStats.load(other / "stats.npz")
    assert stats.meta["mode"] == "merge" and len(stats.meta["shards"]) == 2
    card = (env.root / "runs" / "t" / "DATA_CARD.md").read_text(encoding="utf-8")
    assert card.count("## 3. Semantic statistics") == 1        # section replaced, not duplicated
    assert [r["status"] for r in _registry(env)] == ["ok"] * 4


def test_03_merge_refuses_incomplete_shards(env):
    other = env.root / "sharded"
    assert S03.main(env.args + ["--out", str(other), "--num-procs", "3", "--proc-id", "0"]) == 0
    with pytest.raises(ValueError, match="expected exactly 0..2"):
        S03.main(env.args + ["--out", str(other), "--merge"])


def test_03_refuses_val_split(env):
    with pytest.raises(ValueError, match="val"):
        S03.main(env.args + ["--set", "stats.split=val"])


def test_03_misaligned_teacher_exits_2(env):
    env.dino(1)                     # DINO grid shifted by one token along w
    assert S03.main(env.args) == 2
    diag = json.loads((env.stats_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert not diag["alignment"]["pass"] and diag["alignment"]["argmax"] == [0, 1]
    assert not (env.stats_dir / "stats.npz").exists()
    assert "FAIL" in (env.root / "runs" / "t" / "DATA_CARD.md").read_text(encoding="utf-8")
    assert _registry(env)[-1]["status"] == "failed"
    assert S03.main(env.args + ["--allow-misaligned", "--overwrite"]) == 0
    assert LatentStats.load(env.stats_dir / "stats.npz").meta["alignment_pass"] is False


def test_03_flip_and_cap_with_workers(env):
    args = env.args + ["--max-images", "10", "--set", "stats.include_flip=true", "--set", "stats.num_workers=2"]
    assert S03.main(args) == 0
    st = LatentStats.load(env.stats_dir / "stats.npz")
    assert st.n_images == 10 and st.n_tokens == 2 * 10 * GRID * GRID     # flipped tokens, images counted once
    assert st.meta["protocol"]["max_images"] == 10 and st.meta["protocol"]["include_flip"] is True


# --------------------------------------------------------------------------- 04
def test_04_solves_all_arms_and_is_deterministic(env):
    assert S03.main(env.args) == 0
    assert S04.main(env.args + ["--figures"]) == 0
    expected = {"none", "equal_snr_R0.5", "equal_snr_R2", "variance_R0.5", "variance_R2",
                "r2_R0.5_w1.5", "r2_R0.5_w3", "r2_R2_w1.5", "r2_R2_w3"}
    assert {p.stem for p in env.alloc_dir.glob("*.npz")} == expected
    summary = json.loads((env.alloc_dir / "allocations_summary.json").read_text(encoding="utf-8"))
    assert summary["identity_failures"] == []
    for name, rec in summary["allocations"].items():
        assert rec["identity_residual_max"] <= 1e-8, name
        assert 0 <= rec["semantic_retention"] <= 1 + 1e-12
    assert summary["allocations"]["none"]["semantic_retention"] == pytest.approx(1.0)
    ov = summary["overlaps"]["0.5"]
    assert ov["names"] == ["equal_snr_R0.5", "variance_R0.5", "r2_R0.5_w1.5", "r2_R0.5_w3"]
    assert np.allclose(np.diag(ov["matrix"]), 1.0)
    g = summary["go_no_go"]
    assert g["verdict"] in ("go", "no-go", "inconclusive") and len(g["points"]) == 4
    assert g["min_overlap"] == min(p["overlap_B3_B2"] for p in g["points"])
    md = (env.alloc_dir / "allocations_summary.md").read_text(encoding="utf-8")
    assert "verdict:" in md and "## R = 2 nats/token" in md
    assert {p.name for p in (env.alloc_dir / "figures").iterdir()} == {"keep_per_direction.png", "overlap_heatmap.png"}
    a = Allocation.load(env.alloc_dir / "r2_R2_w3.npz")
    assert a.meta["provenance"]["stats_fingerprint"] == LatentStats.load(env.stats_dir / "stats.npz").fingerprint()

    other = env.root / "alloc2"
    assert S04.main(env.args + ["--out", str(other)]) == 0
    again = json.loads((other / "allocations_summary.json").read_text(encoding="utf-8"))
    for key in ("go_no_go", "overlaps", "stability"):
        assert again[key] == summary[key], key
    for name in expected:
        assert Allocation.load(other / f"{name}.npz").fingerprint() == summary["allocations"][name]["fingerprint"]


def test_04_refuses_stale_ridge(env):
    assert S03.main(env.args) == 0
    with pytest.raises(ValueError, match="ridge_rel"):
        S04.main(env.args + ["--set", "stats.ridge_rel=0.01"])


# --------------------------------------------------------------------------- real assets
@pytest.mark.real_assets
@pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")
def test_03_real_imagenette_alignment_and_stats(tmp_path, monkeypatch):
    """03 on 40 real Imagenette train_fit images: real LDM f16d32 latents, real DINOv2-L (bf16 on CUDA)."""
    pytest.importorskip("timm")
    from rwvae.integration import FrozenTokenizer

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    root = Path(ASSETS)
    snapshots = root / "hf_cache" / "hub" / "models--timm--vit_large_patch14_dinov2.lvd142m" / "snapshots"
    weights = sorted(snapshots.glob("*/model.safetensors"))
    if not weights:
        pytest.skip(f"DINOv2-L weights not found under {snapshots}")
    data = root / "datasets" / "imagenette2-320"
    manifest = build_manifest(data / "train", data / "val", dev_per_class=20, seed=2026, name="imagenette2-320")
    manifest.save(tmp_path / "manifest.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 40
    images = ManifestImageDataset(manifest, "train_fit", indices=range(n))
    x = torch.stack([images[k][0] for k in range(n)])
    tok = FrozenTokenizer.from_checkpoint(root / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt", device=device)
    with torch.no_grad():
        mu = torch.cat([tok.encode_mean(b.to(device)).cpu() for b in x.split(8)])
    del tok
    if device == "cuda":
        torch.cuda.empty_cache()
    # a cache with only the first n rows is not a full split; give the manifest a train_fit of exactly n rows
    for col in manifest.splits["train_fit"]:
        manifest.splits["train_fit"][col] = manifest.splits["train_fit"][col][:n]
    manifest.save(tmp_path / "manifest.json")
    writer = LatentCacheWriter(tmp_path / "cache" / "train_fit", split="train_fit", dtype="float16", shard_size=n,
                               meta={"manifest_fingerprint": manifest.fingerprint(), "latent_source": "posterior_mean"})
    writer.write_shard(0, mu, manifest.splits["train_fit"]["label"], torch.arange(n))
    writer.finalize(n)
    overlay = {
        "experiment": {"name": "real", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "env": {"HF_HOME": (root / "hf_cache").as_posix(), "TORCH_HOME": (root / "torch_cache").as_posix()},
        "dataset": {"manifest": (tmp_path / "manifest.json").as_posix()},
        "cache": {"dir": (tmp_path / "cache").as_posix()},
        "semantic": {"checkpoint_path": weights[0].as_posix()},
        "stats": {"batch_size": 8, "num_workers": 0, "num_groups": 4, "alignment_check": {"num_images": 40, "max_shift": 1}},
    }
    (tmp_path / "overlay.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    args = sum((["--config", c] for c in BASE + [str(tmp_path / "overlay.yaml")]), []) + ["--device", device]
    assert S03.main(args) == 0
    stats_dir = tmp_path / "runs" / "real" / "stats"
    diag = json.loads((stats_dir / "diagnostics.json").read_text(encoding="utf-8"))
    print("alignment:", [(r["dy"], r["dx"], round(r["explained_fraction"], 4)) for r in diag["alignment"]["scan"]])
    print("Sigma_y rank:", diag["Sigma_y"]["rank"], "eff rank Sigma:", diag["Sigma"]["eff_rank"],
          "explained fraction:", diag["K"]["explained_fraction"])
    assert diag["alignment"]["pass"]
    assert diag["C"] == 32 and diag["D"] == 1024 and diag["n_tokens"] == n * 256
    st = LatentStats.load(stats_dir / "stats.npz")
    assert st.meta["teacher"]["autocast"] == ("bf16" if device == "cuda" else
                                              "none (float32; semantic.autocast=bf16 applies on CUDA only)")
    # 04 on the real statistics: exact identities hold on the real Sigma / Sigma_expl (C = 32, D = 1024)
    assert S04.main(args + ["--set", "stats.bootstrap=10"]) == 0
    summary = json.loads((tmp_path / "runs" / "real" / "allocations" / "allocations_summary.json").read_text(encoding="utf-8"))
    assert summary["identity_failures"] == []
    print("go/no-go:", summary["go_no_go"]["verdict"], "min overlap", summary["go_no_go"]["min_overlap"],
          summary["go_no_go"]["min_overlap_at"])
