"""CPU tests of scripts/05_eval_recon_probe.py and scripts/05b_adapt_decoder.py on tiny synthetic inputs.

The fixture builds, with library APIs only (scripts 01-04 are not used): a 3-class synthetic ImageFolder
(32 px after the center crop), its manifest, a seeded random AutoencoderKL with ch_mult=(1, 2) (32 px ->
16x16x32 latents, as tests/test_integration_v2.py), float32 latent caches of all splits, a stats.npz with a
synthetic 8-d "DINO" block, and the allocations of the grid R in {11, 22}, w = 2 where 04 would save them.
Inception and LPIPS are replaced by tiny fake modules. Every run writes under tmp_path; the run registry is
redirected, so manifests/run_registry.jsonl is never touched. Tests marked real_assets run 05 on a few real
Imagenette images with the real tokenizer, Inception and LPIPS (set RWVAE_ASSETS).
"""
import importlib
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
from rwvae import allocation, datasets, evaluation, integration, runtime, stats
from rwvae.channel import RWChannel, noise_seed, sample_noise
from rwvae.decoder_adapter import AdaptedDecoder

eval05 = importlib.import_module("05_eval_recon_probe")
adapt05b = importlib.import_module("05b_adapt_decoder")

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [str(REPO / "configs" / f) for f in ("dataset_imagenet256.yaml", "baseline_common.yaml", "arms.yaml")]
ASSETS = os.environ.get("RWVAE_ASSETS")
LABELS = ["B0_none", "B1_equal_snr_R11", "B1_equal_snr_R22", "B2_variance_R11", "B2_variance_R22",
          "B3_r2_R11_w2", "B3_r2_R22_w2"]


class FakeInception(nn.Module):
    """[0, 1] images -> [[B, 48, 1, 1]] (4x4 average-pooled pixels), like pytorch-fid's list output."""

    def forward(self, x):
        return [F.adaptive_avg_pool2d(x, 4).flatten(1)[:, :, None, None]]


class FakeLPIPS(nn.Module):
    """Differentiable per-image mean |a - b| -> [B, 1, 1, 1]."""

    def forward(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


@pytest.fixture(autouse=True)
def fake_metric_nets(monkeypatch):
    monkeypatch.setattr(integration, "load_inception", lambda device: FakeInception().to(device))
    monkeypatch.setattr(integration, "load_lpips", lambda device: FakeLPIPS().to(device))


def _write_images(root: Path, n_per_class: dict, rng) -> None:
    for c, n in n_per_class.items():
        (root / c).mkdir(parents=True)
        for k in range(n):
            arr = rng.integers(0, 256, size=(36, 40, 3), dtype=np.uint8)
            arr[:, :, 0] = (arr[:, :, 0] // 4 + 64 * int(c[1:])).astype(np.uint8)  # class-dependent red channel
            Image.fromarray(arr).save(root / c / f"{root.name}_{c}_{k}.png")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    for name in ("torchvision", "requests", "tqdm"):
        pytest.importorskip(name)
    tmp = tmp_path_factory.mktemp("eval_scripts")
    rng = np.random.default_rng(0)
    classes = ["c0", "c1", "c2"]
    _write_images(tmp / "data" / "train", {c: 8 for c in classes}, rng)
    _write_images(tmp / "data" / "val", {c: 3 for c in classes}, rng)
    manifest = datasets.build_manifest(tmp / "data" / "train", tmp / "data" / "val", dev_per_class=3, seed=0,
                                       name="synthetic", image_size=32)
    manifest.save(tmp / "manifest.json")

    integration.add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=32, ch_mult=(1, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    torch.save(ae.state_dict(), tmp / "tok.ckpt")
    tok = integration.FrozenTokenizer.from_checkpoint(tmp / "tok.ckpt", ch_mult=(1, 2))

    overlay = {
        "experiment": {"name": "t", "run_root": (tmp / "runs").as_posix()},
        "registry": (tmp / "registry.jsonl").as_posix(),
        "dataset": {"train_root": (tmp / "data" / "train").as_posix(), "val_root": (tmp / "data" / "val").as_posix(),
                    "image_size": 32, "num_classes": 3, "dev_per_class": 3, "manifest": (tmp / "manifest.json").as_posix()},
        "tokenizer": {"ckpt": (tmp / "tok.ckpt").as_posix(), "ch_mult": [1, 2], "embed_dim": 32},
        "cache": {"dir": (tmp / "cache").as_posix()},
        "grid": {"R_nats": [11.0, 22.0], "w": [2.0]},
        "eval": {"batch_size": 4, "num_workers": 0, "dct_images": 5,
                 "probe": {"train_per_class": 4, "max_iter": 50}, "bootstrap": {"n_boot": 50}},
        "fid": {"inception_batch": 8},
        "decoder_adapt": {"images_per_class": 1, "steps": 6, "batch_size": 3, "lr": 1.0e-3, "num_workers": 0,
                          "log_every": 1, "mixed_precision": "no"},
    }
    (tmp / "test.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    config_args = [a for f in CONFIGS + [str(tmp / "test.yaml")] for a in ("--config", f)]
    cfg = runtime.load_config(CONFIGS + [tmp / "test.yaml"])

    mus = {}
    for split in datasets.SPLITS:
        ds = datasets.ManifestImageDataset(manifest, split, image_size=32)
        images = torch.stack([ds[k][0] for k in range(len(ds))])
        mus[split] = tok.encode_mean(images)
        writer = datasets.LatentCacheWriter(_common.cache_dir(cfg, split), split=split, dtype="float32",
                                            meta={"manifest_fingerprint": manifest.fingerprint()}, shard_size=10)
        labels = torch.tensor(manifest.splits[split]["label"])
        for k in range(writer.num_shards(len(ds))):
            r = writer.shard_rows(k, len(ds))
            writer.write_shard(k, mus[split][r.start:r.stop], labels[r.start:r.stop], torch.arange(r.start, r.stop))
        writer.finalize(len(ds))

    mu = mus["train_fit"]
    g = torch.Generator().manual_seed(1)
    y = mu[:, :8] + 0.5 * torch.randn(mu.shape[0], 8, 16, 16, generator=g)   # a synthetic, partly relevant "DINO"
    ids = [stats.image_group_id(i, 4) for i in manifest.splits["train_fit"]["image_id"]]
    acc = stats.stream_stats([(mu, y, torch.tensor(ids))], C=32, D=8, num_groups=4)
    st = stats.LatentStats.from_joint(acc.total(), 32, acc.n_images).with_explained(1e-3)
    _common.stats_path(cfg).parent.mkdir(parents=True)
    st.save(_common.stats_path(cfg))
    for R in (11.0, 22.0):
        for rule, w in (("equal_snr", None), ("variance", None), ("r2", 2.0)):
            a = allocation.solve_allocation(st, rule, w, R)
            a.save(_common.allocations_dir(cfg) / f"{a.name}.npz")
    return {"tmp": tmp, "cfg": cfg, "args": config_args, "manifest": manifest, "tok": tok, "mus": mus}


def _real_registry_state():
    p = REPO / "manifests" / "run_registry.jsonl"
    return p.stat().st_mtime_ns if p.exists() else None


def _results(out: Path) -> tuple[dict, dict]:
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    with np.load(out / "per_image.npz", allow_pickle=False) as z:
        per = {k: z[k] for k in z.files}
    return res, per


def _psnr_of(env, alloc, decode, rows):
    """Recompute one arm's per-image PSNR on train_dev rows with the documented eval noise protocol, in the
    script's batches (CPU convolutions are not bit-identical across batch sizes)."""
    cfg, manifest = env["cfg"], env["manifest"]
    ds = datasets.ManifestImageDataset(manifest, "train_dev", image_size=32)
    channel, bs, out = RWChannel.from_allocation(alloc), int(cfg["eval"]["batch_size"]), []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        mu, images = env["mus"]["train_dev"][r], torch.stack([ds[i][0] for i in r])
        eps = sample_noise(mu.shape[1:], [noise_seed(int(cfg["eval"]["noise_seed"]), 0, i) for i in r])
        with torch.no_grad():
            x_hat = decode(channel(mu, noise=eps))
        out.append(evaluation.psnr_per_image(evaluation.to_uint8(images), evaluation.to_uint8(x_hat)).numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- 05
@pytest.fixture(scope="module")
def full_run(env):
    """05 on train_dev with every arm of the grid, counting image reads (one-pass check)."""
    reads = []
    orig = datasets.ImageLatentPairDataset.__getitem__
    mp = pytest.MonkeyPatch()
    mp.setattr(integration, "load_inception", lambda device: FakeInception().to(device))
    mp.setattr(integration, "load_lpips", lambda device: FakeLPIPS().to(device))
    mp.setattr(datasets.ImageLatentPairDataset, "__getitem__", lambda self, k: reads.append(self.images.indices[k])
               or orig(self, k))
    before = _real_registry_state()
    try:
        assert eval05.main(env["args"] + ["--device", "cpu", "--figures"]) == 0
    finally:
        mp.undo()
    assert _real_registry_state() == before
    out = _common.stage_dir(env["cfg"], "recon_probe", "train_dev")
    return out, reads


def test_05_writes_every_requested_arm_and_artifact(env, full_run):
    out, _ = full_run
    res, per = _results(out)
    assert list(res["arms"]) == LABELS
    for f in ("results.md", "ref_fid.npz", "summary.json", "figures/rate_psnr.png", "figures/rate_probe.png"):
        assert (out / f).is_file(), f
    meta = res["meta"]
    assert meta["split"] == "train_dev" and meta["n_images"] == 9 and meta["split_role"] == "rule selection"
    assert meta["tf32"] == {"tf32_matmul": False, "tf32_cudnn": False} and meta["dropped"] == []
    assert meta["tokenizer"]["checkpoint_sha256"] == runtime.sha256_file(env["tmp"] / "tok.ckpt")
    assert meta["rfid_note"] == "small-n rFID, indicative only"
    for lab, r in res["arms"].items():
        assert r["n_images"] == 9 and np.isfinite(r["psnr"]["mean"]) and np.isfinite(r["lpips"]["mean"])
        assert r["rfid"]["value"] is not None and r["rfid"]["small_n"] and r["rfid"]["n"] == 9
        assert len(r["dct_band_ratio"]) == 4 and 0 <= r["semantic_retention"] <= 1 + 1e-9
        assert per[f"psnr__{lab}"].shape == per[f"lpips__{lab}"].shape == per[f"probe_correct__{lab}"].shape == (9,)
    b0 = res["arms"]["B0_none"]
    assert b0["latent_mse"]["empirical"]["mean"] == 0 and b0["latent_mse"]["predicted"] == 0
    assert b0["dct_band_ratio"] == [1.0] * 4 and b0["nominal_rate"] == "inf" and b0["bpp"] is None
    assert b0["semantic_retention"] == pytest.approx(1.0)
    assert b0["decoder"]["mode"] == "identity" and res["arms"]["B2_variance_R11"]["decoder"]["mode"] == "lmmse"
    for lab in LABELS[1:]:
        lm = res["arms"][lab]["latent_mse"]
        assert lm["empirical"]["mean"] > 0 and lm["predicted"] > 0
    # higher rate -> smaller predicted latent distortion
    assert res["arms"]["B2_variance_R22"]["latent_mse"]["predicted"] < res["arms"]["B2_variance_R11"]["latent_mse"]["predicted"]
    assert json.loads((out / "summary.json").read_text())["status"] == "ok"
    assert (env["tmp"] / "registry.jsonl").read_text().count("05_eval_recon_probe.py") >= 1


def test_05_reads_every_image_exactly_once(full_run):
    _, reads = full_run
    assert sorted(reads) == list(range(9))   # 7 arms, but each eval image is loaded and decoded-from once


def test_05_b0_equals_plain_tokenizer_decode(env, full_run):
    out, _ = full_run
    _, per = _results(out)
    b0 = _common.load_allocation(env["cfg"], "B0", None, None)
    plain = _psnr_of(env, b0, env["tok"].decode, per["rows"].tolist())   # z' = mu exactly, plain tokenizer decode
    np.testing.assert_array_equal(per["psnr__B0_none"], plain)


@pytest.mark.parametrize("name", ["variance_R11", "r2_R22_w2"])
def test_05_uses_the_documented_common_noise(env, full_run, name):
    """Recomputing one lossy arm from the protocol reproduces its per-image PSNR exactly."""
    out, _ = full_run
    _, per = _results(out)
    alloc = allocation.Allocation.load(_common.allocations_dir(env["cfg"]) / f"{name}.npz")
    adapter = AdaptedDecoder(env["tok"].post_quant_conv, env["tok"].decoder, mode="lmmse", A=alloc.A, m=alloc.m)
    label = next(lab for lab in LABELS if lab.endswith(name))
    np.testing.assert_allclose(per[f"psnr__{label}"], _psnr_of(env, alloc, adapter, per["rows"].tolist()), atol=1e-9)


def test_05_probe_comparisons_and_gates(env, full_run):
    out, _ = full_run
    res, per = _results(out)
    for lab, r in res["arms"].items():
        p = r["probe"]
        assert p["n_train"] == 12 and p["n_test"] == 9 and 0 <= p["top1"] <= 1 and p["top5"] is None
        assert p["top1"] == pytest.approx(per[f"probe_correct__{lab}"].mean())
    assert res["meta"]["probe"]["train_split"] == "train_fit" and res["meta"]["probe"]["classes_short"] == {}
    got = {(p["R"], p["rules"]) for p in res["paired"]}
    assert got == {(R, r) for R in (11.0, 22.0) for r in ("r2-variance", "r2-equal_snr", "variance-equal_snr")}
    for p in res["paired"]:
        assert {"psnr_diff_db", "probe_top1_diff_points"} <= set(p) and p["psnr_diff_db"]["n"] == 9
        a, b = per[f"psnr__{p['a']}"], per[f"psnr__{p['b']}"]
        assert p["psnr_diff_db"]["mean"] == pytest.approx(float((a - b).mean()))
    gate = res["gates"]["probe"]
    assert gate["evaluated"] and len(gate["candidates"]) == 2 and isinstance(gate["pass"], bool)
    mono = res["gates"]["monotonicity"]
    assert {(c["arm"], c["metric"]) for c in mono["checked"]} == {(a, m) for a in ("B1", "B2", "B3")
                                                                   for m in ("psnr", "probe_top1", "latent_mse")}
    assert mono["tolerances"] == {"psnr": 0.0, "probe_top1": 0.0, "latent_mse": 0.0}   # no svp block: strict
    assert all(abs(r["latent_mse"]["rel_err"]) < 0.2 for lab, r in res["arms"].items() if lab != "B0_none")
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "probe go/no-go" in md and "B3_r2_R22_w2 - B2_variance_R22" in md


def test_05_filters_balanced_subset_and_flags(env, tmp_path):
    out = tmp_path / "sub"
    assert eval05.main(env["args"] + ["--device", "cpu", "--out", str(out), "--arms", "B0,B2", "--R", "22",
                                      "--max-images", "6", "--no-rfid", "--no-lpips", "--skip-probe",
                                      "--set", "eval.num_workers=2"]) == 0   # DataLoader workers (spawn here)
    res, per = _results(out)
    assert list(res["arms"]) == ["B0_none", "B2_variance_R22"]
    assert np.bincount(per["labels"]).tolist() == [2, 2, 2]          # class-balanced, not the first 6 rows
    assert res["meta"]["dropped"] and "class-balanced" in res["meta"]["dropped"][0]
    assert "rfid" not in res["arms"]["B0_none"] and "probe" not in res["arms"]["B0_none"]
    assert not any(k.startswith("lpips__") for k in per)
    assert res["gates"]["probe"]["evaluated"] is False


def test_05_rejects_unknown_decoder_label(env, tmp_path):
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="not among the evaluated arms"):
        eval05.main(env["args"] + ["--device", "cpu", "--out", str(out), "--arms", "B0",
                                   "--decoder-ckpt", "B3_r2_R11_w2=x.safetensors"])
    assert json.loads((out / "summary.json").read_text())["status"] == "failed"


def test_05_monotonicity_tolerances_come_from_the_svp_block():
    def arm(R, psnr, top1, lmse):
        return {"arm": "B2", "w": None, "R": R, "psnr": {"mean": psnr}, "probe": {"top1": top1},
                "latent_mse": {"empirical": {"mean": lmse}}}
    res = {"a": arm(11.0, 20.0, 0.50, 1.0), "b": arm(22.0, 19.97, 0.495, 0.5), "c": arm(33.0, 21.0, 0.47, 0.6)}
    strict = eval05.monotonicity(res, {})
    assert {(v["metric"], v["R_to"]) for v in strict["violations"]} == {("psnr", 22.0), ("probe_top1", 22.0),
                                                                         ("probe_top1", 33.0), ("latent_mse", 33.0)}
    svp = {"svp": {"pass_criteria": {"monotone_in_rate": {"psnr_tol_db": 0.05, "probe_tol_points": 1.0}}}}
    tol = eval05.monotonicity(res, svp)
    assert {(v["metric"], v["R_to"]) for v in tol["violations"]} == {("probe_top1", 33.0), ("latent_mse", 33.0)}
    assert not tol["monotone"]


def test_05_monotonicity_a_drop_of_exactly_the_tolerance_is_not_a_violation():
    # SVP server run: probe 0.53 -> 0.52 (2 of 200 images) was flagged against a 1-point tolerance because
    # 0.52 - 0.53 = -0.010000000000000009 in float64.
    def arm(R, top1):
        return {"arm": "B1", "w": None, "R": R, "psnr": {"mean": 20.0 + R}, "probe": {"top1": top1},
                "latent_mse": {"empirical": {"mean": 1.0 / R}}}
    res = {"a": arm(33.0, 0.53), "b": arm(44.0, 0.52)}
    svp = {"svp": {"pass_criteria": {"monotone_in_rate": {"psnr_tol_db": 0.05, "probe_tol_points": 1.0}}}}
    assert eval05.monotonicity(res, svp)["monotone"]
    assert not eval05.monotonicity({"a": arm(33.0, 0.53), "b": arm(44.0, 0.515)}, svp)["monotone"]
    assert not eval05.monotonicity(res, {})["monotone"]       # strict mode still flags any drop


# --------------------------------------------------------------------------- 05b
def test_05b_lowers_the_loss_and_saves_loadable_weights(env):
    assert adapt05b.main(env["args"] + ["--device", "cpu", "--arm", "B0", "--val-images", "3"]) == 0
    out = _common.stage_dir(env["cfg"], "decoders", "B0_none")
    losses = [json.loads(line)["loss"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(losses) == 6 and losses[-1] < losses[0]   # the same 3 images every step (B0: no noise)
    info = json.loads((out / "adapt.json").read_text())
    assert info["meta"]["subset"]["n"] == 3 and info["meta"]["steps"] == 6 and info["meta"]["base_mode"] == "identity"
    assert info["before"]["n"] == info["after"]["n"] == 3
    adapter = AdaptedDecoder(env["tok"].post_quant_conv, env["tok"].decoder)
    meta = adapter.load_weights(out / "decoder.safetensors")
    assert adapter.mode == "finetuned" and meta["arm_label"] == "B0_none"
    assert meta["tokenizer_sha256"] == runtime.sha256_file(env["tmp"] / "tok.ckpt")
    assert meta["manifest_fingerprint"] == env["manifest"].fingerprint()


def test_05b_lossy_arm_decoder_is_accepted_by_05(env, tmp_path):
    cfg = env["cfg"]
    dec = tmp_path / "dec"
    assert adapt05b.main(env["args"] + ["--device", "cpu", "--arm", "B3", "--R", "11", "--w", "2", "--out", str(dec),
                                        "--set", "decoder_adapt.steps=2"]) == 0
    alloc = allocation.Allocation.load(_common.allocations_dir(cfg) / "r2_R11_w2.npz")
    adapter = AdaptedDecoder(env["tok"].post_quant_conv, env["tok"].decoder, mode="lmmse", A=alloc.A, m=alloc.m)
    meta = adapter.load_weights(dec / "decoder.safetensors")
    assert meta["allocation_fingerprint"] == alloc.fingerprint() and adapter.base_mode == "lmmse"

    out = tmp_path / "eval"
    assert eval05.main(env["args"] + ["--device", "cpu", "--out", str(out), "--arms", "B0,B3", "--R", "11", "--w", "2",
                                      "--no-rfid", "--decoder-ckpt",
                                      f"B3_r2_R11_w2={(dec / 'decoder.safetensors').as_posix()}"]) == 0
    res, per = _results(out)
    d = res["arms"]["B3_r2_R11_w2"]["decoder"]
    assert d["mode"] == "finetuned" and d["base_mode"] == "lmmse" and d["meta"]["steps"] == 2
    adapter.eval()
    np.testing.assert_allclose(per["psnr__B3_r2_R11_w2"], _psnr_of(env, alloc, adapter, per["rows"].tolist()), atol=1e-9)
    # a decoder fine-tuned for another allocation is refused
    with pytest.raises(ValueError, match="fine-tuned for allocation"):
        eval05.main(env["args"] + ["--device", "cpu", "--out", str(tmp_path / "wrong"), "--arms", "B3", "--R", "22",
                                   "--w", "2", "--skip-probe", "--decoder-ckpt",
                                   f"B3_r2_R22_w2={(dec / 'decoder.safetensors').as_posix()}"])


# --------------------------------------------------------------------------- real assets
@pytest.mark.real_assets
@pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")
def test_05_real_imagenette_smoke(tmp_path, monkeypatch):
    """Real tokenizer / Inception / LPIPS on CUDA: 3 classes x 4 train_dev images, B0 + B2/B3 at R = 22."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.undo()   # real Inception / LPIPS
    assets = Path(ASSETS)
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    for split, n in (("train", 10), ("val", 2)):
        src_root = assets / "datasets" / "imagenette2-320" / split
        for c in sorted(p.name for p in src_root.iterdir())[:3]:
            (data / split / c).mkdir(parents=True)
            files = sorted((src_root / c).iterdir())
            for f in [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]:
                (data / split / c / f.name).write_bytes(f.read_bytes())
    manifest = datasets.build_manifest(data / "train", data / "val", dev_per_class=4, seed=0, name="imagenette-mini")
    manifest.save(tmp_path / "manifest.json")
    overlay = {
        "experiment": {"name": "real", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "env": {"HF_HOME": (assets / "hf_cache").as_posix(), "TORCH_HOME": (assets / "torch_cache").as_posix()},
        "dataset": {"manifest": (tmp_path / "manifest.json").as_posix(), "num_classes": 3},
        "tokenizer": {"ckpt": (assets / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt").as_posix()},
        "cache": {"dir": (tmp_path / "cache").as_posix()},
        "eval": {"batch_size": 4, "num_workers": 0, "probe": {"train_per_class": 6, "max_iter": 50},
                 "bootstrap": {"n_boot": 50}},
    }
    (tmp_path / "real.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    cfg = runtime.load_config(CONFIGS + [tmp_path / "real.yaml"])
    _common.apply_env(cfg)
    tok = _common.make_tokenizer(cfg, "cuda")
    mus = {}
    for split in datasets.SPLITS:
        ds = datasets.ManifestImageDataset(manifest, split, image_size=256)
        with torch.no_grad():
            mus[split] = torch.cat([tok.encode_mean(torch.stack([ds[k][0] for k in range(s, min(s + 4, len(ds)))]).cuda())
                                    for s in range(0, len(ds), 4)]).cpu()
        writer = datasets.LatentCacheWriter(_common.cache_dir(cfg, split), split=split, dtype="float16",
                                            meta={"manifest_fingerprint": manifest.fingerprint()}, shard_size=100)
        writer.write_shard(0, mus[split], torch.tensor(manifest.splits[split]["label"]), torch.arange(len(ds)))
        writer.finalize(len(ds))
    del tok
    torch.cuda.empty_cache()
    mu = mus["train_fit"].double()
    y = mu[:, :8] + 0.5 * torch.randn(mu.shape[0], 8, 16, 16, dtype=torch.float64)
    acc = stats.stream_stats([(mu, y, torch.zeros(mu.shape[0], dtype=torch.int64))], C=32, D=8, num_groups=1)
    st = stats.LatentStats.from_joint(acc.total(), 32, acc.n_images).with_explained(1e-3)
    _common.stats_path(cfg).parent.mkdir(parents=True)
    st.save(_common.stats_path(cfg))
    for rule, w in (("variance", None), ("r2", 2.0)):
        a = allocation.solve_allocation(st, rule, w, 22.0)
        a.save(_common.allocations_dir(cfg) / f"{a.name}.npz")
    args = [a for f in CONFIGS + [str(tmp_path / "real.yaml")] for a in ("--config", f)]
    assert eval05.main(args + ["--device", "cuda", "--arms", "B0,B2,B3", "--R", "22", "--w", "2"]) == 0
    res, _ = _results(_common.stage_dir(cfg, "recon_probe", "train_dev"))
    b0 = res["arms"]["B0_none"]
    assert 20 < b0["psnr"]["mean"] < 40 and 0 < b0["lpips"]["mean"] < 0.5
    assert res["arms"]["B2_variance_R22"]["psnr"]["mean"] < b0["psnr"]["mean"]
    for r in res["arms"].values():
        assert r["rfid"]["small_n"] and (r["rfid"]["value"] is not None or "error" in r["rfid"])
