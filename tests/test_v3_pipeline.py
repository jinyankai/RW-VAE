"""End-to-end CPU smoke test of the v3 scripts on a tiny synthetic ImageFolder:

01 manifest -> v3_train_tokenizer (T-c, T-a, rate-locked L-c) -> v3_cache_posterior -> v3_train_dit (plain and
--repa, then an exact resume) -> 07 ref-stats -> v3_evaluate recon / generate (no CFG, all-channel CFG) ->
v3_analysis,

with 32 px images, 8x8x8 latents, a fake 12-d multi-layer DINO, a seeded random AutoencoderKL(ch_mult (1, 2, 2))
as the init checkpoint, fake Inception / LPIPS and a depth-2 LightningDiT. It checks interfaces and artifacts,
not metric values. Also unit-tests DinoPatchFeatures.forward_layers, PosteriorLatentDataset and REPA training.
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

from rwvae import datasets, generation, integration, runtime

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [REPO / "configs" / f for f in ("dataset_imagenet256.yaml", "baseline_common.yaml", "v3.yaml")]

for _name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy", "torchvision", "matplotlib"):
    pytest.importorskip(_name)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("v3smoke_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S = {name: _load(REPO / "scripts" / f"{name}.py") for name in
     ("01_prepare_manifest", "07_evaluate", "v3_train_tokenizer", "v3_cache_posterior", "v3_train_dit",
      "v3_evaluate", "v3_analysis")}
_common = importlib.import_module("_common")


class FakeMultiDino(nn.Module):
    """[B, 3, 32, 32] -> per block i: tanh(conv_i(8x8 average pool)) [B, 12, 8, 8]."""
    grid = 8

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            self.convs = nn.ModuleList(nn.Conv2d(3, 12, 1) for _ in range(4))

    @torch.no_grad()
    def forward_layers(self, images, indices):
        x = F.adaptive_avg_pool2d(images.float(), 8)
        return torch.stack([torch.tanh(self.convs[int(i)](x)) for i in indices], 1)

    def forward(self, images):
        return self.forward_layers(images, [3])[:, 0]


class FakeInception(nn.Module):
    def forward(self, x):
        return [F.adaptive_avg_pool2d(x, 2).flatten(1)[:, :, None, None]]


class FakeLPIPS(nn.Module):
    def forward(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


@pytest.fixture
def world(tmp_path, monkeypatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(integration, "load_inception", lambda device: FakeInception().to(device))
    monkeypatch.setattr(integration, "load_lpips", lambda device: FakeLPIPS().to(device))
    monkeypatch.setattr(_common, "make_semantic", lambda cfg, device: FakeMultiDino().to(device))
    rng = np.random.default_rng(0)
    for c in range(3):
        for split, count in (("train", 8), ("val", 3)):
            for j in range(count):
                p = tmp_path / "img" / split / f"n0{c}" / f"n0{c}_{split}{j}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)).save(p)
    integration.add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=8, ch_mult=(1, 2, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    torch.save(ae.state_dict(), tmp_path / "init.ckpt")
    overlay = {
        "experiment": {"name": "v3_smoke", "run_root": (tmp_path / "runs").as_posix()},
        "registry": (tmp_path / "registry.jsonl").as_posix(),
        "dataset": {"train_root": (tmp_path / "img" / "train").as_posix(),
                    "val_root": (tmp_path / "img" / "val").as_posix(), "image_size": 32, "num_classes": 3,
                    "dev_per_class": 2, "manifest": (tmp_path / "manifest.json").as_posix()},
        "init": {"ckpt": (tmp_path / "init.ckpt").as_posix(), "embed_dim": 8, "ch_mult": [1, 2, 2]},
        "v3": {"dino_layers": [0, 1, 3], "dino_dim": 12, "feat_dim": 16, "groups": 4, "latent_dim": 8,
               "rgb_channels": [8, 16], "gate_hidden": 8, "sem_depth": 1, "sem_width": 16, "sem_heads": 2,
               "det_width": 16},
        "kl": {"update_every": 1},
        "tok_train": {"max_steps": 3, "warmup_steps": 1, "global_batch_size": 4, "log_every": 1, "ckpt_every": 2,
                      "num_workers": 0, "mixed_precision": "no"},
        "v3_cache": {"dir": (tmp_path / "cache").as_posix(), "batch_size": 4, "num_workers": 0, "shard_size": 4},
        "v3_dit": {"min_steps_per_epoch": 2},
        "repa": {"depth": 1, "proj_hidden": 16, "target_dim": 12},
        "dit": {"input_size": 8, "in_channels": 8,
                "custom_arch": {"depth": 2, "hidden_size": 32, "num_heads": 2, "patch_size": 1}},
        "train": {"max_steps": 2, "global_batch_size": 4, "ema_decay": 0.5, "log_every": 1, "ckpt_every": 1,
                  "num_workers": 0, "mixed_precision": "no"},
        "sample": {"num_sampling_steps": 3, "batch_size": 4, "num_samples_screen": 6},
        "fid": {"inception_batch": 4},
        "eval": {"num_workers": 0},
        "v3_eval": {"max_images": 4, "batch_size": 4, "num_workers": 0, "analysis_images": 4, "probe_t": [0.3, 0.8],
                    "probe_train_per_class": 2, "probe_test_per_class": 1},
    }
    (tmp_path / "test.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    args = [a for p in [*CONFIGS, tmp_path / "test.yaml"] for a in ("--config", str(p))] + ["--device", "cpu"]
    return tmp_path, args, runtime.load_config([*CONFIGS, tmp_path / "test.yaml"])


def test_v3_pipeline_end_to_end(world):
    tmp, args, cfg = world
    exp = tmp / "runs" / "v3_smoke"

    def run(name, *extra, sub=None):
        argv = ([sub] if sub else []) + args + list(extra)
        assert S[name].main(argv) == 0, f"{name} {sub or ''} {extra}"

    run("01_prepare_manifest")
    for arm in ("T-c", "T-a"):
        run("v3_train_tokenizer", "--arm", arm)
        assert (exp / "tok" / arm / "tokenizer.pt").is_file()
    run("v3_train_tokenizer", "--arm", "L-c", "--set", "kl.target=5.0")
    metrics = [json.loads(line) for line in (exp / "tok" / "L-c" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["warmup"] is True and metrics[-1]["lam"] != 0.0      # lambda moved after the warm-up
    with pytest.raises(ValueError, match="kl.target"):
        run("v3_train_tokenizer", "--arm", "L-a")

    run("v3_cache_posterior", "--arm", "T-c")
    stats = json.loads((tmp / "cache" / "T-c" / "stats.json").read_text())
    assert len(stats["mean"]) == 8 and set(stats["rate_nats_per_token"]) == {"train_fit", "train_dev", "val"}
    mu = datasets.LatentCache(tmp / "cache" / "T-c" / "mu" / "train_fit")
    lv = datasets.LatentCache(tmp / "cache" / "T-c" / "logvar" / "train_fit")
    assert mu.has_flip and mu.latent_shape == (8, 8, 8) and len(mu) == len(lv)

    run("v3_train_dit", "--arm", "T-c")
    run("v3_train_dit", "--arm", "T-c", "--repa")
    ck = torch.load(exp / "dit" / "T-c_repa" / "checkpoints" / "0000002.pt", weights_only=True)
    assert "repa_projector" in ck and ck["invariants"]["repa"]["depth"] == 1
    assert "loss_repa" in json.loads((exp / "dit" / "T-c_repa" / "metrics.jsonl").read_text().splitlines()[-1])
    run("v3_train_dit", "--arm", "T-c", "--repa", "--resume", "--overwrite", "--set", "train.max_steps=3")
    assert (exp / "dit" / "T-c_repa" / "checkpoints" / "0000003.pt").is_file()

    run("07_evaluate", "--split", "val", sub="ref-stats")
    run("v3_evaluate", "--arm", "T-c", sub="recon")
    rec = json.loads((exp / "recon" / "T-c" / "recon.json").read_text())
    assert {"psnr", "lpips", "rfid", "rate_nats_per_token"} <= set(rec)
    assert (exp / "recon" / "T-c" / "recon_pairs.png").is_file()
    run("v3_evaluate", "--arm", "T-c", "--baseline", sub="recon")
    base = json.loads((exp / "recon" / "init_baseline" / "recon.json").read_text())
    assert base["arm"] == "init_baseline" and base["rate_nats_per_token"] > 0
    run("v3_evaluate", "--arm", "T-c", sub="generate")
    run("v3_evaluate", "--arm", "T-c", "--repa", "--cfg-scale", "2.0", sub="generate")
    run("v3_evaluate", "--arm", "T-c", "--uncond", sub="generate")
    gen = json.loads((exp / "gen" / "T-c" / "step2_n6_nocfg" / "gen.json").read_text())
    assert "val" in gen["gfid"]
    assert (exp / "gen" / "T-c_repa" / "step3_n6_cfg2_int0.11" / "samples.png").is_file()
    assert json.loads((exp / "gen" / "T-c" / "step2_n6_uncond" / "gen.json").read_text())["uncond"] is True
    with pytest.raises(FileNotFoundError, match="train_dev"):            # the sweep never tunes on val
        run("v3_evaluate", "--arm", "T-c", "--scales", "1.5", sub="cfg-sweep")
    run("07_evaluate", "--split", "train_dev", sub="ref-stats")
    run("v3_evaluate", "--arm", "T-c", "--scales", "1.5", "2", sub="cfg-sweep")
    sweep = json.loads((exp / "cfg_sweep" / "T-c" / "step2_n6_int0.11" / "sweep.json").read_text())
    assert [r["cfg_scale"] for r in sweep["rows"]] == [1.5, 2.0] and "train_dev" in sweep["reference"]

    run("v3_analysis", "--arm", "T-c")
    maps = json.loads((exp / "analysis" / "T-c" / "maps.json").read_text())
    probe = json.loads((exp / "analysis" / "T-c" / "probe.json").read_text())
    assert "corr_alpha_edge" in maps and len(probe["curve"]) == 3
    assert (exp / "analysis" / "T-c" / "maps.png").is_file()


def test_all_channel_cfg_scale_one_is_the_plain_model():
    model = integration.build_lightningdit(input_size=4, in_channels=8, num_classes=3,
                                           custom_arch={"depth": 1, "hidden_size": 16, "num_heads": 2, "patch_size": 1})
    model.eval()
    x, t, y = torch.randn(2, 8, 4, 4), torch.rand(2), torch.tensor([0, 2])
    plain = model(x, t, y)
    wrapped = S["v3_evaluate"].AllChannelCFG(model, 1.0, 3)
    assert torch.equal(wrapped(x, t, y), plain)
    guided = S["v3_evaluate"].AllChannelCFG(model, 3.0, 3)(x, t, y)
    uncond = model(x, t, torch.full_like(y, 3))
    assert torch.allclose(guided, uncond + 3.0 * (plain - uncond), atol=1e-5)
    # guidance interval: no guidance where t < start (t = 0 is noise)
    t2 = torch.tensor([0.05, 0.5])
    out = S["v3_evaluate"].AllChannelCFG(model, 3.0, 3, interval_start=0.11)(x, t2, y)
    cond2, unc2 = model(x, t2, y), model(x, t2, torch.full_like(y, 3))
    assert torch.allclose(out[0], cond2[0], atol=1e-5)
    assert torch.allclose(out[1], unc2[1] + 3.0 * (cond2[1] - unc2[1]), atol=1e-5)


def test_dino_forward_layers_last_block_equals_forward():
    import timm
    dino = integration.DinoPatchFeatures.__new__(integration.DinoPatchFeatures)
    nn.Module.__init__(dino)
    dino.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, dynamic_img_size=True,
                                   depth=3).eval()
    dino.input_size, dino.grid, dino.num_prefix_tokens = 56, 4, int(dino.model.num_prefix_tokens)
    dino.meta = {"model_name": "tiny"}
    dino.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    dino.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    feats = dino.forward_layers(x, [0, 2])
    assert feats.shape == (2, 2, 384, 4, 4)
    assert torch.allclose(feats[:, 1], dino(x), atol=1e-5)
