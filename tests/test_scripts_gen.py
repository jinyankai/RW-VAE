"""In-process tests of scripts/06_train_dit.py and scripts/07_evaluate.py on tiny synthetic inputs.

Everything runs on the CPU under pytest's tmp_path: a 2-class ImageFolder of random 40x36 PNGs, its
manifest, a random float32 latent cache (with flips) of train_fit, LatentStats with a 4-d "DINO" block,
the r2 / variance allocations, a seeded random AutoencoderKL with ch_mult (1, 2) (32 px <-> 16x16 latents,
as tests/test_integration_v2.py), a tiny LightningDiT (depth 2, hidden 64) and a 4-d fake Inception.
The real configs are layered first, then a test overlay (run root, registry, paths, tiny budgets).
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from rwvae import integration
from rwvae.allocation import solve_allocation
from rwvae.datasets import LatentCacheWriter, LatentCache, build_manifest
from rwvae.evaluation import load_fid_stats
from rwvae.stats import LatentStats, image_group_id, stream_stats

REPO = Path(__file__).resolve().parents[1]
BASE = [REPO / "configs" / f for f in ("dataset_imagenet256.yaml", "baseline_common.yaml", "arms.yaml")]
ASSETS = os.environ.get("RWVAE_ASSETS")
REAL_LOAD_INCEPTION = integration.load_inception
TINY_DIT = "+dit.custom_arch={depth: 2, hidden_size: 64, num_heads: 2, patch_size: 1}"
R, W = 6.0, 2.0

for _name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy", "torchvision", "requests", "tqdm"):
    pytest.importorskip(_name)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("0", "s0", 1),
                                                  REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train06 = _load("06_train_dit.py")
eval07 = _load("07_evaluate.py")


class FakeInception(nn.Module):
    """[b, 3, h, w] in [0, 1] -> [[b, 4, 1, 1]] (pytorch-fid layout), seeded and fixed."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.conv = nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return [torch.tanh(self.conv(x * 2 - 1)).mean(dim=(2, 3), keepdim=True) + 1.0]


@pytest.fixture(autouse=True)
def fake_inception(monkeypatch):
    monkeypatch.setattr(integration, "load_inception", lambda device: FakeInception().to(device))


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)


def _random_images(data: Path) -> None:
    rng = np.random.default_rng(0)
    for wnid in ("n001", "n002"):
        for split, count in (("train", 6), ("val", 3)):
            for j in range(count):
                path = data / split / wnid / f"{wnid}_{split}{j}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)).save(path)


def _build_world(root: Path, *, image_size: int, tokenizer: str, extra: str = "", device: str = "cpu"):
    """Manifest (2 classes: 8 train_fit, 4 train_dev, 6 val images under root/data), a random latent cache of
    train_fit with flips, LatentStats with a 4-d "DINO" block, the r2 / variance allocations and the config."""
    manifest = build_manifest(root / "data" / "train", root / "data" / "val", dev_per_class=2, seed=0,
                              name="tiny", image_size=image_size)
    manifest.save(root / "manifest.json")

    n = manifest.split_size("train_fit")                      # 8 items
    g = torch.Generator().manual_seed(1)
    mu = torch.randn((n, 32, 16, 16), generator=g) * torch.linspace(0.5, 2.0, 32).view(1, 32, 1, 1)
    writer = LatentCacheWriter(root / "cache" / "train_fit", split="train_fit", dtype="float32", shard_size=n,
                               meta={"manifest_fingerprint": manifest.fingerprint(), "latent_source": "posterior_mean"})
    writer.write_shard(0, mu, manifest.splits["train_fit"]["label"], torch.arange(n), latents_flip=mu.flip(-1))
    writer.finalize(n)
    cache = LatentCache(root / "cache" / "train_fit", manifest=manifest)

    exp = root / "runs" / "tiny"
    y = torch.randn((n, 4, 16, 16), generator=g)
    gid = torch.tensor([image_group_id(i, 4) for i in manifest.splits["train_fit"]["image_id"]])
    acc = stream_stats(((m, y[idx], gid[idx]) for m, _, idx in cache.iter_batches(4)), 32, 4, num_groups=4)
    stats = LatentStats.from_joint(acc.total(), 32, acc.n_images).with_explained(1e-3)
    (exp / "stats").mkdir(parents=True)
    stats.save(exp / "stats" / "stats.npz")
    solve_allocation(stats, "r2", W, R).save(exp / "allocations" / f"r2_R{R:g}_w{W:g}.npz")
    solve_allocation(stats, "variance", None, R).save(exp / "allocations" / f"variance_R{R:g}.npz")

    overlay = f"""
experiment: {{name: tiny, run_root: {root.as_posix()}/runs}}
registry: {root.as_posix()}/registry.jsonl
dataset:
  name: tiny
  train_root: {(root / 'data' / 'train').as_posix()}
  val_root: {(root / 'data' / 'val').as_posix()}
  image_size: {image_size}
  num_classes: 2
  dev_per_class: 2
  manifest: {(root / 'manifest.json').as_posix()}
tokenizer: {tokenizer}
cache: {{dir: {(root / 'cache').as_posix()}}}
train: {{max_steps: 2, global_batch_size: 4, ema_decay: 0.5, log_every: 1, ckpt_every: 2, keep_last: 3,
         num_workers: 0, mixed_precision: "no"}}
sample: {{num_sampling_steps: 3, batch_size: 4, num_samples_screen: 8}}
fid: {{inception_batch: 4}}
eval: {{num_workers: 0}}
{extra}
"""
    (root / "test.yaml").write_text(overlay, encoding="utf-8")
    args = [a for p in (*BASE, root / "test.yaml") for a in ("--config", str(p))] + ["--set", TINY_DIT,
                                                                                     "--device", device]
    return SimpleNamespace(root=root, exp=exp, args=args, n=n, manifest=manifest)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The tiny synthetic world with a seeded random AutoencoderKL, ch_mult (1, 2): 32 px <-> 16x16 latents."""
    root = tmp_path_factory.mktemp("gen")
    _random_images(root / "data")
    integration.add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=32, ch_mult=(1, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    torch.save(ae.state_dict(), root / "tok.ckpt")
    return _build_world(root, image_size=32,
                        tokenizer=f"{{ckpt: {(root / 'tok.ckpt').as_posix()}, ch_mult: [1, 2], embed_dim: 32}}")


def _registry(world) -> list[dict]:
    path = world.root / "registry.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture(scope="module")
def trained(world):
    """06 runs used by the 07 tests: B0 and B3 (exploratory) in their default run directories."""
    assert train06.main([*world.args, "--arm", "B0"]) == 0
    assert train06.main([*world.args, "--arm", "B3", "--R", str(R), "--w", str(W), "--exploratory"]) == 0
    return SimpleNamespace(b0=world.exp / "dit" / "B0_none", b3=world.exp / "dit" / f"B3_r2_R{R:g}_w{W:g}")


# =========================================================================== 06
def test_06_confirmatory_lossy_arm_needs_the_frozen_preregistration(world, tmp_path):
    with pytest.raises(ValueError, match="pre-registered"):
        train06.main([*world.args, "--arm", "B3", "--R", str(R), "--w", str(W), "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()
    with pytest.raises(ValueError, match="neither --R nor --w"):
        train06.main([*world.args, "--arm", "B0", "--R", "6"])


def test_06_trains_writes_arm_json_checkpoints_and_metrics(world, trained):
    arm = json.loads((trained.b3 / "arm.json").read_text(encoding="utf-8"))
    assert (arm["arm"], arm["rule"], arm["R"], arm["w"], arm["exploratory"]) == ("B3", "r2", R, W, True)
    assert arm["preregistration"]["registered"] is False and arm["world_size"] == 1
    assert arm["repeat"] == 25                                    # ceil(MIN_STEPS_PER_EPOCH * 4 / 8)
    assert arm["data"]["num_items"] == world.n and arm["noise_protocol"]["flip_prob"] == 0.5
    assert len(arm["normalisation"]["mean"]) == 32 and arm["allocation"]["name"] == f"r2_R{R:g}_w{W:g}"
    ckpt = torch.load(trained.b3 / "checkpoints" / "0000002.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 2 and ckpt["meta"]["allocation_fingerprint"] == arm["allocation"]["fingerprint"]
    steps = [json.loads(line)["step"] for line in (trained.b3 / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2]
    b0 = json.loads((trained.b0 / "arm.json").read_text(encoding="utf-8"))
    assert (b0["rule"], b0["R"], b0["preregistration"]["registered"]) == ("none", None, True)
    runs = {(r["script"], r["status"], r["summary"].get("arm")) for r in _registry(world)}
    assert {("06_train_dit.py", "ok", "B0_none"), ("06_train_dit.py", "ok", f"B3_r2_R{R:g}_w{W:g}")} <= runs


def test_06_dataset_smaller_than_the_global_batch_is_repeated(world, tmp_path):
    out = tmp_path / "small"
    assert train06.main([*world.args, "--arm", "B0", "--set", "train.global_batch_size=20",
                         "--set", "train.flip_prob=0.0", "--out", str(out)]) == 0
    arm = json.loads((out / "arm.json").read_text(encoding="utf-8"))
    assert arm["repeat"] == 125 and (out / "checkpoints" / "0000002.pt").is_file()   # ceil(50 * 20 / 8)

    class Inner:                                    # records the dataset epoch each item is drawn in
        def __len__(self):
            return 4

        def set_epoch(self, e):
            self.e = e

        def __getitem__(self, i):
            return i, self.e

    ds = train06.RepeatedDataset(Inner(), 3)
    ds.set_epoch(2)
    assert len(ds) == 12 and [ds[i] for i in (0, 5, 11)] == [(0, 6), (1, 7), (3, 8)]


def test_06_resume_extends_a_finished_run(world, tmp_path):
    out = tmp_path / "resume"
    assert train06.main([*world.args, "--arm", "B0", "--out", str(out)]) == 0
    with pytest.raises(FileExistsError):
        train06.main([*world.args, "--arm", "B0", "--out", str(out)])
    assert train06.main([*world.args, "--arm", "B0", "--out", str(out), "--resume", "--overwrite",
                         "--set", "train.max_steps=4"]) == 0
    ckpt = torch.load(out / "checkpoints" / "0000004.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 4
    steps = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4]


def test_06_no_flip_cache_trains_with_flip_prob_zero(world, tmp_path, caplog):
    # a cache without latents_flip in a separate cache dir: flip_prob falls back to 0 and is recorded
    cache = tmp_path / "cache" / "train_fit"
    src = LatentCache(world.root / "cache" / "train_fit")
    mu = torch.stack([src.get(i)[0] for i in range(world.n)])
    writer = LatentCacheWriter(cache, split="train_fit", dtype="float32", shard_size=world.n, meta=src.meta)
    writer.write_shard(0, mu, world.manifest.splits["train_fit"]["label"], torch.arange(world.n))
    writer.finalize(world.n)
    out = tmp_path / "noflip"
    with caplog.at_level("WARNING"):
        assert train06.main([*world.args, "--arm", "B0", "--set", f"cache.dir={(tmp_path / 'cache').as_posix()}",
                             "--out", str(out)]) == 0
    arm = json.loads((out / "arm.json").read_text(encoding="utf-8"))
    assert arm["noise_protocol"]["flip_prob"] == 0.0 and arm["noise_protocol"]["flip_prob_config"] == 0.5


# =========================================================================== 07
@pytest.fixture(scope="module")
def refs(world):
    """ref-stats: train_fit (with features of the originals and their flips) and val (with features)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(integration, "load_inception", lambda device: FakeInception().to(device))
        assert eval07.main(["ref-stats", *world.args, "--split", "train_fit", "--save-features", "--hflip"]) == 0
        assert eval07.main(["ref-stats", *world.args, "--split", "val", "--save-features"]) == 0
    root = world.exp / "fid_ref"
    return SimpleNamespace(train=root / "train_fit.npz", val=root / "val.npz", feats=root / "train_fit_features.npz",
                           val_feats=root / "val_features.npz")


def test_07_ref_stats_writes_statistics_features_and_meta(world, refs):
    mu, sigma, n, meta = load_fid_stats(refs.train)
    assert n == world.n and mu.shape == (4,) and sigma.shape == (4, 4)
    assert meta["split"] == "train_fit" and meta["manifest_fingerprint"] == world.manifest.fingerprint()
    assert meta["inception"].startswith("pytorch-fid pt_inception-2015-12-05") and meta["tf32"]["tf32_matmul"] is False
    with np.load(refs.feats) as f:
        feats, flipped, labels = f["features"], f["flipped"], f["labels"]
        assert json.loads(f["meta"].item())["includes_hflip"] is True
    assert feats.shape == (2 * world.n, 4) and feats.dtype == np.float16 and flipped.tolist() == [False] * 8 + [True] * 8
    assert labels.tolist() == world.manifest.splits["train_fit"]["label"] * 2
    assert np.allclose(feats[:world.n].astype(np.float64).mean(0), mu, atol=1e-3)   # FID stats = the originals
    assert not np.allclose(feats[:world.n], feats[world.n:])
    assert load_fid_stats(refs.val)[2] == world.manifest.split_size("val")
    assert eval07.main(["ref-stats", *world.args, "--split", "val", "--max-images", "4"]) == 0
    assert load_fid_stats(world.exp / "fid_ref" / "val_n4.npz")[2] == 4
    with pytest.raises(ValueError, match="--hflip"):
        eval07.main(["ref-stats", *world.args, "--split", "val", "--hflip", "--out", str(world.root / "x")])


def _generate(world, arm: list[str], out: Path, refs, extra=()):
    argv = ["generate", *world.args, *arm, "--ref-stats", f"train={refs.train}", "--ref-stats", f"val={refs.val}",
            "--nn-ref", f"train={refs.feats}", "--nn-baseline", str(refs.val_feats), "--out", str(out), *extra]
    assert eval07.main(argv) == 0
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_07_generate_is_deterministic_and_arms_share_noise_and_labels(world, trained, refs, tmp_path):
    a = _generate(world, ["--arm", "B0"], tmp_path / "a", refs, ["--save-npz", "--save-grid", "5"])
    b = _generate(world, ["--arm", "B0"], tmp_path / "b", refs, ["--save-npz"])
    c = _generate(world, ["--arm", "B3", "--R", str(R), "--w", str(W)], tmp_path / "c", refs)
    assert a["fid_kind"] == "FID-8" and a["num_samples"] == 8 and a["checkpoint"]["step"] == 2
    assert set(a["gfid"]) == {"train", "val"} and all(np.isfinite(v) for v in a["gfid"].values())
    assert a["gfid"] == b["gfid"] and a["nn"] == b["nn"]
    cos, l2 = a["nn"]["train"]["cosine"], a["nn"]["train"]["l2"]
    assert -1 <= cos["p10"] <= cos["median"] <= cos["p90"] <= 1 and 0 <= cos["frac_above"] <= 1
    assert 0 <= l2["min"] <= l2["median"] and a["nn"]["train"]["includes_hflip"] is True
    base = a["nn"]["train"]["baseline"]
    assert base["n"] == 6 and a["nn"]["train"]["l2_median_ratio"] == pytest.approx(l2["median"] / base["l2"]["median"])
    with np.load(tmp_path / "a" / "samples.npz") as fa, np.load(tmp_path / "b" / "samples.npz") as fb:
        assert fa["arr_0"].shape == (8, 32, 32, 3) and fa["arr_0"].dtype == np.uint8
        assert np.array_equal(fa["arr_0"], fb["arr_0"]) and np.array_equal(fa["arr_1"], fb["arr_1"])
        assert sorted(np.bincount(fa["arr_1"]).tolist()) == [4, 4]
    assert Image.open(tmp_path / "a" / "grid.png").size == (3 * 32, 2 * 32)
    for key in ("labels_sha256", "initial_noise_sha256", "noise_seed", "label_seed"):
        assert a["sampling"][key] == c["sampling"][key]
    assert a["gfid"] != c["gfid"]
    assert a["decoder"]["mode"] == "identity" and c["decoder"]["mode"] == "lmmse" and c["exploratory"] is True
    assert a["references"]["train"]["meta"]["split"] == "train_fit"


def test_07_nearest_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    q, r = torch.rand((7, 5), generator=g), torch.rand((11, 5), generator=g)
    cos, l2 = eval07._nearest(q, r, "cpu")
    d = torch.cdist(q.double(), r.double())
    c = torch.nn.functional.cosine_similarity(q.double()[:, None], r.double()[None], dim=2)
    assert np.allclose(l2, d.min(1).values.numpy()) and np.allclose(cos, c.max(1).values.numpy())


def test_07_generate_refuses_cfg_and_a_mismatched_run(world, trained, refs, tmp_path):
    with pytest.raises(ValueError, match="cfg_scale"):
        eval07.main(["generate", *world.args, "--arm", "B0", "--set", "sample.cfg_scale=1.5",
                     "--out", str(tmp_path / "cfg")])
    assert not (tmp_path / "cfg").exists()
    with pytest.raises(ValueError, match="trained arm"):
        eval07.main(["generate", *world.args, "--arm", "B0", "--dit-run", str(trained.b3), "--out", str(tmp_path / "x")])


def _ci(m, half=1.0):
    return {"mean": m, "lo": m - half, "hi": m + half, "n": 4, "n_dropped_nonfinite": 0}


def _m3_arm(label, arm, rule, R, psnr, top1, emp, pred):
    return {"label": label, "arm": arm, "rule": rule, "R": R, "w": None, "bpp": None if R is None else R / 177.4,
            "psnr": _ci(psnr), "lpips": _ci(0.1, 0.01), "rfid": {"value": 30.0 - psnr},
            "probe": {"top1": top1, "top1_ci": _ci(top1, 0.05)}, "semantic_retention": 0.5,
            "latent_mse": {"empirical": _ci(emp, 0.01), "predicted": pred}}


SVP_BLOCK = {"svp": {"pass_criteria": {           # the schema of configs/svp.yaml, thresholds chosen for the test
    "dit_loss_drop": {"rule": "...", "max_ratio": 10.0},
    "dit_loss_plateau": {"rule": "...", "max_rel_change": 1.0e-9},                # 2-step runs: FAIL
    "gfid_memorisation": {"rule": "...", "max_ratio": 1000.0},
    "nn_memorisation": {"rule": "...", "max_ratio": 1000.0},
    "allocation_identity_residual": {"rule": "...", "max": 1.0e-8},
    "latent_mse_vs_d_latent": {"rule": "...", "max_rel_err": 0.05},
    "monotone_in_rate": {"rule": "...", "psnr_tol_db": 0.05, "probe_tol_points": 1.0},
    "b0_reconstruction": {"rule": "...", "psnr_min": 23.0, "psnr_max": 32.0, "lpips_max": 0.25},
    "data_gates": {"rule": "..."},                                                # inputs missing: not evaluated
    "mystery_check": {"rule": "...", "max": 1.0}}}}                               # unknown: not evaluated


def test_07_report_separates_fid_kinds_and_renders_the_svp_verdict(world, trained, refs, tmp_path):
    _generate(world, ["--arm", "B0"], world.exp / "gen" / "B0_none" / "step2_n8", refs)
    _generate(world, ["--arm", "B0"], world.exp / "gen" / "B0_none" / "step2_n8_seed2", refs)
    fake10k = {"label": "B2_variance_R6", "arm": "B2", "rule": "variance", "R": R, "w": None, "fid_kind": "FID-10K",
               "checkpoint": {"step": 100}, "gfid": {"val": 12.5}, "nn": {}, "references": {}}
    (world.exp / "gen" / "B2_variance_R6" / "step100_n10000").mkdir(parents=True)
    (world.exp / "gen" / "B2_variance_R6" / "step100_n10000" / "results.json").write_text(json.dumps(fake10k))
    m3 = {"meta": {"split": "train_dev", "n_images": 4, "split_role": "rule selection", "rfid_note": "small-n"},
          "arms": {a["label"]: a for a in (_m3_arm("B0_none", "B0", "none", None, 25.0, 0.5, 0.0, 0.0),
                                           _m3_arm("B2_variance_R6", "B2", "variance", 6.0, 18.0, 0.30, 1.02, 1.0),
                                           _m3_arm("B2_variance_R11", "B2", "variance", 11.0, 20.0, 0.35, 0.51, 0.5))},
          "paired": [{"R": 6.0, "w": 2.0, "a": "B3_r2_R6_w2", "b": "B2_variance_R6", "rules": "r2-variance",
                      "psnr_diff_db": _ci(-0.5), "probe_top1_diff_points": _ci(1.0)}],
          "gates": {"probe": {"evaluated": True, "pass": False, "threshold_points": 3.0},
                    "monotonicity": {"monotone": True, "violations": []}}}
    (world.exp / "recon_probe" / "train_dev").mkdir(parents=True)
    (world.exp / "recon_probe" / "train_dev" / "results.json").write_text(json.dumps(m3))
    allocs = {"allocations": {"variance_R6": {"name": "variance_R6", "rule": "variance", "R": 6.0, "w": None,
                                              "bpp_equivalent": 0.03, "n_active": 5, "latent_mse_lmmse": 1.0,
                                              "semantic_retention": 0.5, "identity_residual_max": 3e-12}},
              "go_no_go": {"verdict": "go"}}       # the 04 schema: {allocation name: record}
    (world.exp / "allocations" / "allocations_summary.json").write_text(json.dumps(allocs))
    (tmp_path / "svp.yaml").write_text(json.dumps(SVP_BLOCK))          # JSON is YAML
    out = tmp_path / "report"
    assert eval07.main(["report", *world.args, "--config", str(tmp_path / "svp.yaml"), "--out", str(out)]) == 0
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "NOT THESIS RESULTS" in text and "## SVP verdict" in text
    sections = {sec.splitlines()[0]: sec for sec in text.split("### ")}
    fid8 = next(v for k, v in sections.items() if k.startswith("FID-8 "))
    fid10k = next(v for k, v in sections.items() if k.startswith("FID-10K "))
    assert "B0_none" in fid8 and "B2_variance_R6" not in fid8
    assert "B2_variance_R6" in fid10k and "B0_none" not in fid10k
    assert "mean +- std (n)" in fid8 and "25 [24, 26]" in text and "B3_r2_R6_w2 - B2_variance_R6" in text
    report = json.loads((out / "report.json").read_text())
    verdict = {v["criterion"]: v["verdict"] for v in report["svp_verdict"]}
    assert verdict == {"dit_loss_drop": "PASS", "dit_loss_plateau": "FAIL", "gfid_memorisation": "PASS",
                       "nn_memorisation": "PASS", "allocation_identity_residual": "PASS",
                       "latent_mse_vs_d_latent": "PASS", "monotone_in_rate": "PASS", "b0_reconstruction": "PASS"}
    skipped = " ".join(report["svp_not_evaluated"])
    assert "data_gates" in skipped and "mystery_check" in skipped and "SVP overall: FAIL" in text
    assert report["gfid_aggregates"]["FID-8"]["B0_none|step2|train"]["n"] == 2
    for fig in ("dit_loss", "gfid_FID-8", "gfid_FID-10K", "rate_psnr_train_dev", "rate_probe_train_dev"):
        assert (out / "figures" / f"{fig}.png").is_file()


def test_07_knows_every_criterion_of_the_svp_config():
    path = REPO / "configs" / "svp.yaml"
    if not path.is_file():
        pytest.skip("configs/svp.yaml not written yet")
    import yaml
    names = set(yaml.safe_load(path.read_text(encoding="utf-8"))["svp"]["pass_criteria"])
    assert names <= set(eval07.SVP_CRITERIA), names - set(eval07.SVP_CRITERIA)


# =========================================================================== real assets (CUDA)
@pytest.mark.real_assets
@pytest.mark.skipif(not ASSETS or not torch.cuda.is_available(), reason="set RWVAE_ASSETS (and have CUDA)")
def test_real_assets_06_07_on_imagenette_with_the_real_tokenizer_and_inception(tmp_path, monkeypatch):
    """06 (tiny DiT, bf16, CUDA) -> 07 ref-stats / generate / report with the real f16d32 decoder and the real
    pytorch-fid Inception on 2 Imagenette classes (6 train + 3 val images each, 256 px)."""
    monkeypatch.setattr(integration, "load_inception", REAL_LOAD_INCEPTION)
    assets = Path(ASSETS)
    for split, count in (("train", 6), ("val", 3)):
        for wnid in sorted(p.name for p in (assets / "datasets" / "imagenette2-320" / split).iterdir())[:2]:
            src = sorted((assets / "datasets" / "imagenette2-320" / split / wnid).iterdir())[:count]
            (tmp_path / "data" / split / wnid).mkdir(parents=True)
            for f in src:
                (tmp_path / "data" / split / wnid / f.name).write_bytes(f.read_bytes())
    ckpt = (assets / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt").as_posix()
    w = _build_world(tmp_path, image_size=256, device="cuda",
                     tokenizer=f"{{ckpt: {ckpt}, ch_mult: [1, 1, 2, 2, 4], embed_dim: 32}}",
                     extra=f"env: {{TORCH_HOME: {(assets / 'torch_cache').as_posix()}}}")
    assert train06.main([*w.args, "--arm", "B3", "--R", str(R), "--w", str(W), "--exploratory",
                         "--set", "train.mixed_precision=bf16"]) == 0
    assert eval07.main(["ref-stats", *w.args, "--split", "train_fit", "--save-features", "--hflip"]) == 0
    assert eval07.main(["ref-stats", *w.args, "--split", "val", "--save-features"]) == 0
    fid_ref = w.exp / "fid_ref"
    _, _, n, meta = load_fid_stats(fid_ref / "train_fit.npz")
    assert n == w.n and meta["inception_module"] == "InceptionV3"
    torch.cuda.reset_peak_memory_stats()
    assert eval07.main(["generate", *w.args, "--arm", "B3", "--R", str(R), "--w", str(W),
                        "--ref-stats", f"train={fid_ref / 'train_fit.npz'}", "--ref-stats", f"val={fid_ref / 'val.npz'}",
                        "--nn-ref", f"train={fid_ref / 'train_fit_features.npz'}",
                        "--nn-baseline", str(fid_ref / "val_features.npz"), "--save-grid", "8"]) == 0
    assert torch.cuda.max_memory_allocated() < 2 * 2 ** 30
    res = json.loads((w.exp / "gen" / f"B3_r2_R{R:g}_w{W:g}" / "step2_n8" / "results.json").read_text())
    assert res["decoder"]["mode"] == "lmmse" and res["decoder"]["tokenizer"]["checkpoint_sha256"]
    assert all(np.isfinite(v) and v > 0 for v in res["gfid"].values())
    assert Image.open(w.exp / "gen" / f"B3_r2_R{R:g}_w{W:g}" / "step2_n8" / "grid.png").size == (3 * 256, 3 * 256)
    assert eval07.main(["report", *w.args]) == 0
    assert "FID-8" in (w.exp / "report" / "REPORT.md").read_text(encoding="utf-8")


def test_07_monotone_in_rate_allows_a_drop_of_exactly_the_tolerance():
    # A 1-point probe drop (2 of 200 images) must not violate a 1-point tolerance; for 0.55 -> 0.54,
    # 100*0.54 - 100*0.55 < -1.0 in float64 (05 hit the same rounding with 0.53 -> 0.52 on the SVP).
    def arm(R, top1):
        return {"label": f"B1_equal_snr_R{R:g}", "arm": "B1", "rule": "equal_snr", "R": R, "w": None,
                "psnr": {"mean": 20.0 + R}, "probe": {"top1": top1}, "latent_mse": {"empirical": {"mean": 1.0 / R}}}
    spec = {"psnr_tol_db": 0.05, "probe_tol_points": 1.0}
    c = {"m3": {"train_dev": {"data": {"arms": {"a": arm(33.0, 0.55), "b": arm(44.0, 0.54)}}}}}
    assert eval07._svp_monotone(c, spec)[2]
    c["m3"]["train_dev"]["data"]["arms"]["b"] = arm(44.0, 0.535)
    assert not eval07._svp_monotone(c, spec)[2]
