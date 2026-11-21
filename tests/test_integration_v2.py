"""Tests for the v2 section of rwvae.integration (docs/ARCHITECTURE_V2.md §3.10).

The CPU tests need no downloads: they use a synthetic tokenizer checkpoint, a fake timm model
and a tiny LightningDiT. Tests marked real_assets load the real LDM f16d32 tokenizer, DINOv2-L,
the FID Inception and LPIPS-VGG; they run only when RWVAE_ASSETS points to the asset directory
(checkpoints/, datasets/imagenette2-320/, hf_cache/, torch_cache/).
"""
import argparse
import hashlib
import importlib
import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import rwvae.integration as integration
from rwvae.integration import (
    DEFAULT_LIGHTNINGDIT_ROOT,
    DinoPatchFeatures,
    FrozenTokenizer,
    add_lightningdit_to_path,
    build_lightningdit,
    build_ode_sample_fn,
    build_transport,
    configure_torch_compile,
    load_inception,
    load_lpips,
    load_tokenizer_state_dict,
)

REPO = Path(__file__).resolve().parents[1]
ASSETS = os.environ.get("RWVAE_ASSETS")
needs_assets = pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")

TCFG = dict(path_type="Linear", prediction="velocity", loss_weight=None, train_eps=None, sample_eps=None,
            use_cosine_loss=True, use_lognorm=True)  # the transport block of configs/baseline_common.yaml
SCFG = dict(mode="ODE", sampling_method="euler", num_sampling_steps=3, atol=1e-6, rtol=1e-3, reverse=False)
TINY = dict(depth=2, hidden_size=64, num_heads=2, patch_size=1)


@pytest.fixture
def dit_deps():
    """Third-party modules that models.lightningdit and transport import."""
    for name in ("timm", "fairscale", "einops", "torchdiffeq", "scipy"):
        pytest.importorskip(name)


@pytest.fixture
def global_rngs():
    """LightningDiT's Transport samples from the global torch / NumPy RNGs: seed them, restore after."""
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        np.random.seed(0)
        yield
    np.random.set_state(np_state)


# --------------------------------------------------------------------------- path / compile setup
def test_import_is_cheap():
    names = ("timm", "lpips", "pytorch_fid", "models", "tokenizer", "transport", "torch._dynamo")
    code = f"import sys, rwvae.integration; print(sorted(m for m in {names!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE"), check=True)
    assert out.stdout.strip() == "[]"


def test_add_lightningdit_to_path_puts_checkout_first(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(integration, "_active_root", None)
    monkeypatch.delenv("RWVAE_LIGHTNINGDIT_ROOT", raising=False)
    root = add_lightningdit_to_path()
    assert root == DEFAULT_LIGHTNINGDIT_ROOT.resolve() and sys.path[0] == str(root)
    sys.path.insert(0, "somewhere-else")
    assert add_lightningdit_to_path() == root  # moved back to the front, never duplicated
    assert sys.path[0] == str(root) and sys.path.count(str(root)) == 1


def test_add_lightningdit_to_path_rejects_missing_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(integration, "_active_root", None)
    monkeypatch.setenv("RWVAE_LIGHTNINGDIT_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="nowhere"):
        add_lightningdit_to_path()
    (tmp_path / "partial" / "tokenizer").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="tokenizer/autoencoder.py"):
        add_lightningdit_to_path(tmp_path / "partial")


@pytest.mark.parametrize("name", ["models", "tokenizer", "transport"])
def test_add_lightningdit_to_path_rejects_foreign_module(monkeypatch, tmp_path, name):
    foreign = types.ModuleType(name)
    foreign.__file__ = str(tmp_path / name / "__init__.py")
    monkeypatch.setitem(sys.modules, name, foreign)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ImportError, match=f"named '{name}'"):
        add_lightningdit_to_path()


@pytest.mark.skipif(sys.platform != "win32", reason="the automatic decision is platform specific")
def test_configure_torch_compile_auto_is_off_on_windows():
    import torch._dynamo

    assert configure_torch_compile() is False
    assert torch._dynamo.config.disable is True


def test_configure_torch_compile_is_frozen_after_import(dit_deps):
    import torch._dynamo

    build_lightningdit(custom_arch=TINY)  # imports models.lightningdit
    current = not torch._dynamo.config.disable
    assert configure_torch_compile(current) is current
    with pytest.raises(ValueError, match="already imported"):
        configure_torch_compile(not current)
    with pytest.raises(ValueError, match="auto"):
        configure_torch_compile("auto")  # the config string must be mapped to None by the caller


# --------------------------------------------------------------------------- LightningDiT
def test_tiny_lightningdit_forward_losses_and_ode_sample(dit_deps, global_rngs):
    model = build_lightningdit(input_size=16, in_channels=32, num_classes=10, custom_arch=TINY)
    x, y = torch.randn(2, 32, 16, 16), torch.tensor([1, 7])
    assert model(x, torch.rand(2), y).shape == x.shape

    transport = build_transport(TCFG)
    rng = (torch.get_rng_state(), np.random.get_state())
    terms = transport.training_losses(model, x, dict(y=y))
    assert terms["loss"].shape == (2,) and terms["cos_loss"].shape == (2,)
    assert torch.isfinite(terms["loss"]).all() and torch.isfinite(terms["cos_loss"]).all()
    torch.set_rng_state(rng[0])
    np.random.set_state(rng[1])
    assert torch.equal(transport.training_losses(model, x, dict(y=y))["loss"], terms["loss"])  # global seeds
    (terms["loss"].mean() + terms["cos_loss"].mean()).backward()  # train.py's objective
    assert model.final_layer.linear.weight.grad.abs().sum() > 0

    nn.init.normal_(model.final_layer.linear.weight, std=0.02)  # zero-initialised upstream, i.e. v = 0
    model.eval()
    z = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        xs = build_ode_sample_fn(transport, dict(SCFG, timestep_shift=0.3))(z, model.forward, y=y)
    assert xs.shape == (3, 2, 32, 16, 16) and torch.equal(xs[0], z)
    assert torch.isfinite(xs[-1]).all() and not torch.equal(xs[-1], z)


def test_ode_sample_fn_integrates_on_the_shifted_time_grid(dit_deps):
    transport = build_transport(TCFG)
    z = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))

    def velocity(x, t):  # dx/dt = t, so Euler gives x0 + sum_i t_i (t_{i+1} - t_i) exactly
        return t.view(-1, 1, 1, 1) * torch.ones_like(x)

    s = 0.3
    grid = [s * u / (1 + (s - 1) * u) for u in (0.0, 0.5, 1.0)]  # LightningDiT's timestep shift
    shifted = sum(grid[i] * (grid[i + 1] - grid[i]) for i in range(2))
    # upstream builds the time grid in float32, so the Euler identity holds to float32 precision
    xs = build_ode_sample_fn(transport, dict(SCFG, timestep_shift=s))(z, velocity)
    torch.testing.assert_close(xs[-1], z + shifted, rtol=0, atol=1e-6)
    xs = build_ode_sample_fn(transport, SCFG)(z, velocity)  # no shift by default: grid 0, 0.5, 1
    torch.testing.assert_close(xs[-1], z + 0.25, rtol=0, atol=1e-6)


def test_build_lightningdit_passes_train_py_kwargs(dit_deps, monkeypatch):
    build_lightningdit(custom_arch=TINY)  # make sure models.lightningdit is imported
    seen = {}
    monkeypatch.setitem(sys.modules["models.lightningdit"].LightningDiT_models, "LightningDiT-B/1",
                        lambda **kw: seen.update(kw) or "model")
    assert build_lightningdit("LightningDiT-B/1", num_classes=10) == "model"
    assert seen == dict(input_size=16, in_channels=32, num_classes=10, use_qknorm=False, use_swiglu=True,
                        use_rope=True, use_rmsnorm=True, wo_shift=False, class_dropout_prob=0.1,
                        use_checkpoint=False)


def test_build_lightningdit_rejects_bad_arguments(dit_deps):
    with pytest.raises(ValueError, match="unknown model_type"):
        build_lightningdit("LightningDiT-S/1")
    with pytest.raises(ValueError, match="custom_arch needs"):
        build_lightningdit(custom_arch=dict(depth=2, hidden_size=64, num_heads=2))
    with pytest.raises(ValueError, match="divisible"):
        build_lightningdit(custom_arch=dict(TINY, num_heads=3))
    with pytest.raises(ValueError, match="divisible"):
        build_lightningdit(input_size=15, custom_arch=dict(TINY, patch_size=2))


def test_build_transport_rejects_bad_configs(dit_deps):
    with pytest.raises(ValueError, match="missing keys \\['sample_eps'\\]"):
        build_transport({k: v for k, v in TCFG.items() if k != "sample_eps"})
    with pytest.raises(ValueError, match="unknown keys \\['use_lognrom'\\]"):
        build_transport(dict(TCFG, use_lognrom=True))
    with pytest.raises(ValueError, match="prediction='eps'"):  # create_transport would silently use velocity
        build_transport(dict(TCFG, prediction="eps"))
    with pytest.raises(ValueError, match="path_type"):
        build_transport(dict(TCFG, path_type="linear"))
    with pytest.raises(ValueError, match="loss_weight"):
        build_transport(dict(TCFG, loss_weight="lambda"))


def test_build_ode_sample_fn_rejects_bad_configs(dit_deps):
    transport = build_transport(TCFG)
    with pytest.raises(ValueError, match="only 'ODE'"):
        build_ode_sample_fn(transport, dict(SCFG, mode="SDE"))
    with pytest.raises(ValueError, match="num_sampling_steps"):
        build_ode_sample_fn(transport, dict(SCFG, num_sampling_steps=1))  # 1 grid point = no integration
    with pytest.raises(ValueError, match="missing keys \\['rtol'\\]"):
        build_ode_sample_fn(transport, {k: v for k, v in SCFG.items() if k != "rtol"})


# --------------------------------------------------------------------------- tokenizer (synthetic)
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """A seeded random AutoencoderKL with ch_mult=(1, 2) (f = 2: 32 px -> 16x16 latents), saved as a
    Lightning checkpoint (loss.* extras + non-tensor metadata) and as a plain state dict."""
    for name in ("torchvision", "requests", "tqdm"):  # imported by tokenizer.autoencoder
        pytest.importorskip(name)
    add_lightningdit_to_path()
    AutoencoderKL = importlib.import_module("tokenizer.autoencoder").AutoencoderKL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ae = AutoencoderKL(embed_dim=32, ch_mult=(1, 2), use_variational=True, ckpt_path=None, model_type="vavae")
    weights = ae.state_dict()
    extras = {"loss.logvar": torch.zeros(()), "loss.discriminator.main.0.weight": torch.ones(8, 3, 4, 4)}
    root = tmp_path_factory.mktemp("tokenizer")
    torch.save({"state_dict": {**weights, **extras}, "epoch": 3, "global_step": 1200,
                "hyper_parameters": argparse.Namespace(embed_dim=32)}, root / "lightning.ckpt")
    torch.save({**weights, **extras}, root / "plain.ckpt")
    return SimpleNamespace(weights=weights, lightning=root / "lightning.ckpt", plain=root / "plain.ckpt")


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_ckpt):
    return FrozenTokenizer.from_checkpoint(tiny_ckpt.plain, ch_mult=(1, 2))


def test_load_tokenizer_state_dict_keeps_only_autoencoder_tensors(tiny_ckpt):
    for path in (tiny_ckpt.lightning, tiny_ckpt.plain):
        state = load_tokenizer_state_dict(path)
        assert state.keys() == tiny_ckpt.weights.keys()
        assert all(torch.equal(state[k], v) for k, v in tiny_ckpt.weights.items())


def test_from_checkpoint_lightning_file_warns_and_records_meta(tiny_ckpt, caplog):
    with caplog.at_level(logging.WARNING, logger="rwvae.integration"):
        tok = FrozenTokenizer.from_checkpoint(tiny_ckpt.lightning, ch_mult=(1, 2))
    assert "trusted" in caplog.text and "argparse.Namespace" in caplog.text
    meta = tok.meta
    assert meta["checkpoint_format"] == "lightning" and meta["weights_only_load"] is False
    assert meta["dropped_keys"] == {"loss": 2} and meta["num_tensors"] == len(tiny_ckpt.weights)
    assert meta["checkpoint_sha256"] == hashlib.sha256(tiny_ckpt.lightning.read_bytes()).hexdigest()
    assert (meta["embed_dim"], meta["ch_mult"], meta["downsample"]) == (32, [1, 2], 2)
    loaded = tok.autoencoder.state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in tiny_ckpt.weights.items())
    assert not tok.training and not any(p.requires_grad for p in tok.parameters())


def test_from_checkpoint_plain_state_dict_needs_no_pickle_fallback(tiny_ckpt, caplog):
    with caplog.at_level(logging.WARNING, logger="rwvae.integration"):
        tok = FrozenTokenizer.from_checkpoint(tiny_ckpt.plain, ch_mult=(1, 2))
    assert caplog.records == []
    assert tok.meta["checkpoint_format"] == "state_dict" and tok.meta["weights_only_load"] is True


@pytest.mark.parametrize("corruption, match", [("drop", "1 missing"), ("extra", "1 unexpected"),
                                               ("shape", "1 shape-mismatched")])
def test_from_checkpoint_is_strict(tiny_ckpt, tmp_path, corruption, match):
    weights = dict(tiny_ckpt.weights)
    if corruption == "drop":
        weights.pop("decoder.conv_out.bias")
    elif corruption == "extra":
        weights["encoder.extra.weight"] = torch.zeros(1)
    else:
        weights["post_quant_conv.bias"] = torch.zeros(31)
    torch.save(weights, tmp_path / "bad.ckpt")
    with pytest.raises(ValueError, match=match):
        FrozenTokenizer.from_checkpoint(tmp_path / "bad.ckpt", ch_mult=(1, 2))


def test_from_checkpoint_rejects_wrong_architecture_and_unknown_files(tiny_ckpt, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        FrozenTokenizer.from_checkpoint(tiny_ckpt.plain)  # default f16 ch_mult (1, 1, 2, 2, 4)
    with pytest.raises(ValueError, match="ch_mult"):
        FrozenTokenizer.from_checkpoint(tiny_ckpt.plain, ch_mult=())
    torch.save({"model": {"w": torch.zeros(1)}}, tmp_path / "other.ckpt")
    with pytest.raises(ValueError, match="neither a state dict"):
        load_tokenizer_state_dict(tmp_path / "other.ckpt")
    with pytest.raises(FileNotFoundError):
        load_tokenizer_state_dict(tmp_path / "absent.ckpt")


def test_encode_mean_is_the_posterior_mean(tiny_tokenizer):
    tok = tiny_tokenizer
    x = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1
    mu = tok.encode_mean(x)
    assert mu.shape == (2, 32, 16, 16) and not mu.requires_grad
    with torch.no_grad():
        posterior = tok.autoencoder.encode(x)
    assert torch.equal(mu, posterior.mean)
    assert torch.equal(tok.encode_mean(x), mu)  # deterministic: no posterior sampling
    mean, logvar = tok.encode_moments(x)
    assert torch.equal(mean, mu) and torch.equal(logvar, posterior.logvar)
    images = tok.decode(mu)
    assert images.shape == (2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(images, tok.autoencoder.decode(mu))
    assert tok.post_quant_conv is tok.autoencoder.post_quant_conv and tok.decoder is tok.autoencoder.decoder


def test_tokenizer_rejects_bad_inputs(tiny_tokenizer):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        tiny_tokenizer.encode_mean(torch.full((1, 3, 32, 32), 127.5))  # [0, 255]-scaled pixels
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        tiny_tokenizer.encode_mean(torch.full((1, 3, 32, 32), float("nan")))
    with pytest.raises(ValueError, match="3, H, W"):
        tiny_tokenizer.encode_mean(torch.zeros(1, 1, 32, 32))
    with pytest.raises(ValueError, match="multiple"):
        tiny_tokenizer.encode_mean(torch.zeros(1, 3, 33, 32))
    with pytest.raises(ValueError, match="latents"):
        tiny_tokenizer.decode(torch.zeros(1, 16, 16, 16))


# --------------------------------------------------------------------------- DINOv2 (fake timm model)
class _FakeViT(nn.Module):
    """timm-ViT stand-in. Patch token k (row-major) is (k // g, k % g, mean of its preprocessed image);
    prefix tokens are -1, so wrong prefix stripping or a transposed reshape is visible."""

    def __init__(self, num_prefix_tokens=1, extra_tokens=0, patch=14):
        super().__init__()
        self.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        self.num_prefix_tokens, self.extra_tokens, self.patch = num_prefix_tokens, extra_tokens, patch
        self.num_features = 3
        self.patch_embed = SimpleNamespace(patch_size=(patch, patch))
        self.unit = nn.Parameter(torch.ones(()))
        self.seen = None

    def forward_features(self, x):
        self.seen = x.detach().clone()
        B, _, H, W = x.shape
        k = torch.arange((H // self.patch) * (W // self.patch))
        gw = W // self.patch
        patches = torch.stack([k // gw, k % gw, torch.zeros_like(k)], -1).to(x.dtype).repeat(B, 1, 1)
        patches[..., 2] = x.mean(dim=(1, 2, 3))[:, None]
        prefix = torch.full((B, self.num_prefix_tokens + self.extra_tokens, 3), -1.0, dtype=x.dtype)
        return torch.cat([prefix, patches], dim=1) * self.unit


@pytest.fixture
def fake_timm(monkeypatch):
    """Route timm.create_model to _FakeViT(**state.options); state.calls records (name, kwargs)."""
    timm = pytest.importorskip("timm")
    state = SimpleNamespace(calls=[], options={})

    def create_model(name, **kwargs):
        state.calls.append((name, kwargs))
        return _FakeViT(**state.options)

    monkeypatch.setattr(timm, "create_model", create_model)
    return state


@pytest.mark.parametrize("num_prefix_tokens", [1, 5])  # cls only / cls + 4 registers
def test_dino_strips_prefix_tokens_and_keeps_row_major_order(fake_timm, num_prefix_tokens):
    fake_timm.options = dict(num_prefix_tokens=num_prefix_tokens)
    dino = DinoPatchFeatures()
    assert fake_timm.calls == [("vit_large_patch14_dinov2.lvd142m", dict(pretrained=True, dynamic_img_size=True))]
    images = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(0)) * 2 - 1
    out = dino(images)
    assert out.shape == (2, 3, 16, 16)
    rows, cols = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    assert torch.equal(out[:, 0], rows.expand(2, 16, 16)) and torch.equal(out[:, 1], cols.expand(2, 16, 16))
    torch.testing.assert_close(out[:, 2, 0, 0], dino.model.seen.mean(dim=(1, 2, 3)))  # batch order kept
    assert dino.meta["grid"] == 16 and dino.meta["num_prefix_tokens"] == num_prefix_tokens


def test_dino_preprocessing_and_checkpoint_argument(fake_timm, tmp_path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    dino = DinoPatchFeatures("vit_large_patch14_dinov2.lvd142m", input_size=224, checkpoint_path=ckpt)
    assert fake_timm.calls[0][1] == dict(pretrained=True, dynamic_img_size=True,
                                         pretrained_cfg_overlay=dict(file=str(ckpt)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    dino(torch.full((1, 3, 256, 256), 0.5))  # a constant image survives any resize: exact normalisation check
    torch.testing.assert_close(dino.model.seen, ((0.75 - mean) / std).expand(1, 3, 224, 224))
    images = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1)) * 2 - 1
    dino(images)

    def resized(antialias):
        x = F.interpolate((images + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False,
                          antialias=antialias)
        return (x - mean) / std

    torch.testing.assert_close(dino.model.seen, resized(antialias=True))
    assert not torch.allclose(dino.model.seen, resized(antialias=False), atol=1e-3)
    assert dino.meta["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert dino.meta["std"] == pytest.approx([0.229, 0.224, 0.225])
    assert (dino.meta["input_size"], dino.meta["patch_size"], dino.meta["embed_dim"]) == (224, 14, 3)
    assert dino.meta["checkpoint_path"] == str(ckpt.resolve())
    assert not dino.training and not dino.model.training and not any(p.requires_grad for p in dino.parameters())


def test_dino_rejects_bad_inputs(fake_timm, tmp_path):
    with pytest.raises(ValueError, match="multiple of the square patch size"):
        DinoPatchFeatures(input_size=230)
    with pytest.raises(FileNotFoundError):
        DinoPatchFeatures(checkpoint_path=tmp_path / "absent.safetensors")
    dino = DinoPatchFeatures()
    with pytest.raises(ValueError, match="square"):
        dino(torch.zeros(1, 3, 256, 192))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        dino(torch.full((1, 3, 256, 256), 2.0))
    fake_timm.options = dict(extra_tokens=1)  # e.g. register tokens missing from num_prefix_tokens
    with pytest.raises(ValueError, match="returned 258 tokens"):
        DinoPatchFeatures()(torch.zeros(1, 3, 256, 256))


# --------------------------------------------------------------------------- real assets
def _imagenette_images(n: int = 4) -> torch.Tensor:
    """First validation image of the first n Imagenette classes, ADM center crop to 256, in [-1, 1]."""
    from PIL import Image

    add_lightningdit_to_path()
    center_crop_arr = importlib.import_module("tokenizer.vavae").center_crop_arr  # ADM crop (LightningDiT)
    val = Path(ASSETS) / "datasets" / "imagenette2-320" / "val"
    paths = [sorted(d.glob("*.JPEG"))[0] for d in sorted(p for p in val.iterdir() if p.is_dir())[:n]]
    arrays = np.stack([np.asarray(center_crop_arr(Image.open(p).convert("RGB"), 256)) for p in paths])
    return torch.from_numpy(arrays).permute(0, 3, 1, 2).float() / 127.5 - 1.0  # ToTensor + Normalize(0.5, 0.5)


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(127.5 * x + 128.0, 0, 255).to(torch.uint8)  # LightningDiT's quantisation


@pytest.mark.real_assets
@needs_assets
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_tokenizer_reconstructs_imagenette_on_cuda():
    tok = FrozenTokenizer.from_checkpoint(Path(ASSETS) / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt",
                                          device="cuda")
    assert tok.meta["checkpoint_format"] == "state_dict" and tok.meta["weights_only_load"] is True
    assert tok.meta["dropped_keys"] == {"loss": 56} and len(tok.meta["checkpoint_sha256"]) == 64
    x = _imagenette_images(4).cuda()
    mu = tok.encode_mean(x)
    assert mu.shape == (4, 32, 16, 16) and torch.isfinite(mu).all()
    rec = tok.decode(mu)
    assert rec.shape == (4, 3, 256, 256)
    mse = ((_to_uint8(rec).double() - _to_uint8(x).double()) ** 2).mean(dim=(1, 2, 3))
    psnr = 20 * np.log10(255.0) - 10 * torch.log10(mse)
    print("reconstruction PSNR (dB):", [round(v, 2) for v in psnr.tolist()])
    # Mean PSNR is the usual summary; single hard textures sit near 20 dB (40 Imagenette val images:
    # mean 27.4, min 19.6, max 36.1), so the per-image floor only guards against a broken load.
    assert psnr.mean() > 20 and psnr.min() > 15, psnr


@pytest.mark.real_assets
@needs_assets
def test_real_dinov2_patch_grid_is_row_major():
    snapshots = Path(ASSETS) / "hf_cache" / "hub" / "models--timm--vit_large_patch14_dinov2.lvd142m" / "snapshots"
    weights = sorted(snapshots.glob("*/model.safetensors"))
    if not weights:
        pytest.skip(f"DINOv2-L weights not found under {snapshots}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DinoPatchFeatures(checkpoint_path=weights[0], device=device)
    assert (dino.meta["grid"], dino.meta["num_prefix_tokens"], dino.meta["embed_dim"]) == (16, 1, 1024)
    assert dino.meta["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert dino.meta["std"] == pytest.approx([0.229, 0.224, 0.225])
    x = _imagenette_images(4).to(device)
    f, f_flip = dino(x), dino(x.flip(-1))
    assert f.shape == (4, 1024, 16, 16) and torch.isfinite(f).all()
    # A horizontal image flip must flip the w axis of the grid; a transposed (column-major)
    # layout would turn it into an h flip instead.
    w_flip = F.cosine_similarity(f_flip, f.flip(-1), dim=1).mean().item()
    h_flip = F.cosine_similarity(f_flip, f.flip(-2), dim=1).mean().item()
    print(f"patch cosine similarity: w-flip {w_flip:.3f}, h-flip {h_flip:.3f}")
    assert w_flip > h_flip + 0.2


@pytest.mark.real_assets
@needs_assets
def test_real_inception_and_lpips_give_finite_outputs(monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(Path(ASSETS) / "torch_cache"))  # weights are downloaded here once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = _imagenette_images(4).to(device)
    inception = load_inception(device)
    assert not inception.training and not any(p.requires_grad for p in inception.parameters())
    with torch.no_grad():
        (pool3,) = inception((x + 1) / 2)  # expects [0, 1]
    assert pool3.shape == (4, 2048, 1, 1) and torch.isfinite(pool3).all() and (pool3 >= 0).all()
    lpips_fn = load_lpips(device)
    assert not lpips_fn.training and not any(p.requires_grad for p in lpips_fn.parameters())
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(device)
    with torch.no_grad():
        d_same, d_noisy = lpips_fn(x, x), lpips_fn(x, (x + 0.1 * noise).clamp(-1, 1))
    assert d_noisy.shape == (4, 1, 1, 1) and torch.isfinite(d_noisy).all() and (d_noisy > 0).all()
    assert d_same.abs().max() < 1e-6
