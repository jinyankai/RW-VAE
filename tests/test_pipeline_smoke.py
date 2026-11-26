"""End-to-end CPU smoke test of the v2.1 pipeline as the SVP runs it (scripts/run_pipeline.sh svp):

make_svp_subset -> 01 manifest -> 02 cache -> 03 stats -> 04 allocations -> 05 recon/probe -> 05b decoder
-> 06 DiT (B0 and one exploratory lossy arm) -> 07 ref-stats / generate / report,

every stage through its main() on a tiny synthetic ImageFolder (4 classes; the subset keeps 3 x 8 train +
3 x 3 val), with the real configs + configs/svp.yaml layered first and a test overlay last (tiny budgets,
tmp paths, tmp run registry). Only the heavy networks are replaced: a seeded random AutoencoderKL with
ch_mult (1, 2) (32 px <-> 16x16x32 latents, as tests/test_integration_v2.py), a fake DINO (1x1 conv of the
pooled image on the same 16x16 grid), fake Inception / LPIPS, and a depth-2 LightningDiT.

It checks the interfaces between stages: every artifact the next stage reads exists where it looks for it,
the report evaluates EVERY svp.pass_criteria entry (none "not evaluated"), and the registry got one record
per stage run. It does not check metric values (random networks).
"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from rwvae import integration, runtime

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [REPO / "configs" / f for f in ("dataset_imagenet256.yaml", "baseline_common.yaml", "arms.yaml",
                                          "svp.yaml")]
TINY_DIT = "+dit.custom_arch={depth: 2, hidden_size: 64, num_heads: 2, patch_size: 1}"
R, W = 22.0, 2.0                                        # the lossy arm of the SVP (B3 @ R22, w2)

for _name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy", "torchvision", "requests", "tqdm"):
    pytest.importorskip(_name)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("smoke_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                    # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


S = {name: _load(REPO / "scripts" / f"{name}.py") for name in
     ("01_prepare_manifest", "02_cache_latents", "03_estimate_stats", "04_solve_channel", "05_eval_recon_probe",
      "05b_adapt_decoder", "06_train_dit", "07_evaluate")}
SUBSET = _load(REPO / "scripts" / "tools" / "make_svp_subset.py")
_common = importlib.import_module("_common")           # the module the scripts imported (scripts/ is on sys.path)


class FakeDino(nn.Module):
    """[B, 3, 32, 32] -> [B, 6, 16, 16]: tanh(1x1 conv) of the tiny tokenizer's own latents, so the "DINO" grid
    is aligned with the latent grid by construction (as DINOv2-L and f16d32 are; the random tokenizer's
    receptive field is off-centre, so a teacher of the raw pixels would fail the 03 alignment gate)."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            self.conv = nn.Conv2d(32, 6, 1)
        self.meta = {"model_name": "fake-dino", "input_size": 32, "patch_size": 2, "grid": 16,
                     "num_prefix_tokens": 0, "embed_dim": 6, "preprocess": "tiny tokenizer latents",
                     "tokens": "row-major (h, w)", "checkpoint_path": None}

    @torch.no_grad()
    def forward(self, images):
        return torch.tanh(self.conv(self.tokenizer.encode_mean(images)))


class FakeInception(nn.Module):
    """[0, 1] images -> [[B, 12, 1, 1]] (pytorch-fid layout): 2x2 average-pooled pixels."""

    def forward(self, x):
        return [F.adaptive_avg_pool2d(x, 2).flatten(1)[:, :, None, None]]


class FakeLPIPS(nn.Module):
    def forward(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


def _source_folder(root: Path) -> None:
    """4 classes x (10 train, 4 val) random 36x40 PNGs (i.i.d. pixels: tokens at different grid positions are
    independent, so the alignment scan of 03 has a clear peak at (0, 0))."""
    rng = np.random.default_rng(0)
    for c in range(4):
        for split, count in (("train", 10), ("val", 4)):
            for j in range(count):
                path = root / split / f"n0{c}" / f"n0{c}_{split}{j:02d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(integration, "load_inception", lambda device: FakeInception().to(device))
    monkeypatch.setattr(integration, "load_lpips", lambda device: FakeLPIPS().to(device))

    _source_folder(tmp_path / "imagenet")
    integration.add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=32, ch_mult=(1, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    torch.save(ae.state_dict(), tmp_path / "tok.ckpt")
    monkeypatch.setattr(_common, "make_semantic",
                        lambda cfg, device: FakeDino(_common.make_tokenizer(cfg, device)).to(device))

    sub = tmp_path / "svp"
    overlay = {
        "experiment": {"name": "svp_smoke", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "dataset": {"train_root": (sub / "train").as_posix(), "val_root": (sub / "val").as_posix(),
                    "image_size": 32, "num_classes": 3, "dev_per_class": 2,
                    "manifest": (tmp_path / "manifest.json").as_posix()},
        "tokenizer": {"ckpt": (tmp_path / "tok.ckpt").as_posix(), "ch_mult": [1, 2], "embed_dim": 32},
        "cache": {"dir": (tmp_path / "cache").as_posix(), "shard_size": 6, "batch_size": 6, "num_workers": 0,
                  "verify_samples": 4},
        "stats": {"num_groups": 4, "batch_size": 6, "num_workers": 0,
                  "alignment_check": {"num_images": 8, "max_shift": 1}, "bootstrap": 5},
        "grid": {"R_nats": [11.0, R], "w": [W]},
        "train": {"max_steps": 4, "global_batch_size": 6, "ema_decay": 0.5, "log_every": 1, "ckpt_every": 2,
                  "keep_last": 3, "num_workers": 0, "mixed_precision": "no"},
        "sample": {"num_sampling_steps": 3, "batch_size": 6, "num_samples_screen": 12},
        "fid": {"inception_batch": 6},
        "eval": {"batch_size": 6, "num_workers": 0, "dct_images": 6,
                 "probe": {"train_per_class": 6, "max_iter": 50}, "bootstrap": {"n_boot": 50}},
        "decoder_adapt": {"images_per_class": 2, "steps": 3, "batch_size": 3, "lr": 1.0e-3, "num_workers": 0,
                          "log_every": 1, "mixed_precision": "no"},
    }
    (tmp_path / "test.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    args = [a for p in [*CONFIGS, tmp_path / "test.yaml"] for a in ("--config", str(p))] + ["--device", "cpu"]
    cfg = runtime.load_config([*CONFIGS, tmp_path / "test.yaml"])
    return tmp_path, args, cfg


def test_svp_pipeline_end_to_end(world):
    tmp, args, cfg = world
    exp = tmp / "runs" / "svp_smoke"
    calls = []

    def run(name, *extra, sub=None):
        argv = ([sub] if sub else []) + args + list(extra)
        assert S[name].main(argv) == 0, f"{name} {sub or ''} {extra}"
        calls.append(f"{name}.py")

    # make_svp_subset (plain argparse, no Run) -> the ImageFolder that dataset.*_root points at
    assert SUBSET.main(["--train-root", str(tmp / "imagenet" / "train"), "--val-root", str(tmp / "imagenet" / "val"),
                        "--out", str(tmp / "svp"), "--num-classes", "3", "--train-per-class", "8",
                        "--val-per-class", "3", "--seed", "2026", "--mode", "copy"]) == 0
    assert (tmp / "svp" / "svp_subset.json").is_file()

    run("01_prepare_manifest")
    assert (tmp / "manifest.json").is_file() and (exp / "DATA_CARD.md").is_file()
    run("02_cache_latents")
    for split in cfg["cache"]["splits"]:
        assert (_common.cache_dir(cfg, split) / "cache_index.json").is_file()
        assert json.loads((_common.cache_dir(cfg, split) / "cache_verification.json").read_text())["passed"]
    run("03_estimate_stats")
    for f in ("stats.npz", "group_moments.pt", "diagnostics.json"):
        assert (exp / "stats" / f).is_file()
    run("04_solve_channel")
    alloc = exp / "allocations"
    assert {"none.npz", f"r2_R{R:g}_w{W:g}.npz", f"variance_R{R:g}.npz", "allocations_summary.json"} <= \
        {p.name for p in alloc.iterdir()}
    run("05_eval_recon_probe")
    assert {"results.json", "results.md", "per_image.npz"} <= {p.name for p in (exp / "recon_probe" / "train_dev").iterdir()}
    run("05b_adapt_decoder", "--arm", "B0", "--val-images", "3")
    assert (exp / "decoders" / "B0_none" / "decoder.safetensors").is_file()

    lossy = f"B3_r2_R{R:g}_w{W:g}"
    run("06_train_dit", "--set", TINY_DIT, "--arm", "B0", "--exploratory")
    run("06_train_dit", "--set", TINY_DIT, "--arm", "B3", "--R", str(R), "--w", str(W), "--exploratory")
    for label in ("B0_none", lossy):
        assert (exp / "dit" / label / "arm.json").is_file()
        assert (exp / "dit" / label / "checkpoints" / "0000004.pt").is_file()

    n_val = 9                                             # equal-size train_fit reference (as --max-images 500)
    run("07_evaluate", "--split", "train_fit", "--max-images", str(n_val), sub="ref-stats")
    run("07_evaluate", "--split", "train_fit", "--save-features", "--hflip", sub="ref-stats")
    run("07_evaluate", "--split", "val", "--save-features", sub="ref-stats")
    ref = exp / "fid_ref"
    refs = ["--ref-stats", f"train={ref / f'train_fit_n{n_val}.npz'}", "--ref-stats", f"val={ref / 'val.npz'}",
            "--nn-ref", f"train={ref / 'train_fit_features.npz'}", "--nn-baseline", str(ref / "val_features.npz")]
    run("07_evaluate", "--set", TINY_DIT, "--arm", "B0", *refs, "--save-grid", "4", sub="generate")
    run("07_evaluate", "--set", TINY_DIT, "--arm", "B3", "--R", str(R), "--w", str(W), *refs, sub="generate")
    for label in ("B0_none", lossy):
        out = exp / "gen" / label / "step4_n12"
        assert (out / "results.json").is_file() and (out / "gen_fid_stats.npz").is_file()
    assert (exp / "gen" / "B0_none" / "step4_n12" / "grid.png").is_file()
    run("07_evaluate", sub="report")

    report = (exp / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "SVP verdict" in report
    rj = json.loads((exp / "report" / "report.json").read_text(encoding="utf-8"))
    evaluated = {r["criterion"] for r in rj["svp_verdict"]}
    assert rj["svp_not_evaluated"] == [] and evaluated == set(cfg["svp"]["pass_criteria"]), rj["svp_not_evaluated"]
    by_name = {r["criterion"]: r for r in rj["svp_verdict"]}
    assert by_name["allocation_identity_residual"]["verdict"] == "PASS"   # exact math, even on random networks
    assert by_name["data_gates"]["verdict"] == "PASS"

    records = [json.loads(line) for line in (tmp / "registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["script"] for r in records] == calls and all(r["status"] == "ok" for r in records)
    assert all(isinstance(r["git_dirty"], (bool, type(None))) for r in records)
