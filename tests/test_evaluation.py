"""Tests for rwvae.evaluation (docs/ARCHITECTURE_V2.md Sec. 3.8; plan v2.1 Sec. 5.1).

The metric networks are replaced by small deterministic stand-ins, so the CPU tests need no
downloads. Tests marked real_assets run the real pytorch-fid InceptionV3 and LPIPS-VGG (weights
in $RWVAE_ASSETS/torch_cache) on Imagenette images and compare against the libraries' own
reference computation; they run only when RWVAE_ASSETS points to the asset directory.
"""
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.fft
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwvae import evaluation as ev
from rwvae.evaluation import (
    FIDStatistics,
    LPIPSMeter,
    bootstrap_mean_ci,
    dct2_power,
    dct_matrix,
    frechet_distance,
    linear_probe,
    load_fid_stats,
    paired_bootstrap_diff,
    pooled_features,
    psnr_per_image,
    radial_band_power,
    save_fid_stats,
    to_uint8,
)

ASSETS = os.environ.get("RWVAE_ASSETS")
needs_assets = pytest.mark.skipif(not ASSETS, reason="set RWVAE_ASSETS to the asset directory")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _spd(d: int, rng, cond: float = 30.0) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * np.logspace(0, np.log10(cond), d)) @ Q.T


def _sym_sqrt_psd(S: np.ndarray) -> np.ndarray:
    e, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(e, 0, None))) @ V.T


def _fd_reference(mu1, s1, mu2, s2) -> float:
    """Independent Frechet distance: tr (s1 s2)^{1/2} = sum sqrt eig(s1^{1/2} s2 s1^{1/2})."""
    R = _sym_sqrt_psd(s1)
    e = np.linalg.eigvalsh(R @ s2 @ R)
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(e, 0, None)).sum())


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------- to_uint8 / PSNR

def test_to_uint8_is_lightningdit_quantisation():
    x = torch.rand(2, 3, 8, 8, generator=_gen(0)) * 2.4 - 1.2          # includes out-of-range values
    u = to_uint8(x)
    assert u.dtype == torch.uint8 and u.shape == (2, 3, 8, 8)
    assert torch.equal(u, torch.clamp(127.5 * x + 128.0, 0, 255).to(torch.uint8))
    edge = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape(1, 1, 1, 5).expand(1, 3, 1, 5)
    assert to_uint8(edge)[0, 0, 0].tolist() == [0, 0, 128, 255, 255]


def test_to_uint8_truncates_and_inverts_totensor_normalize():
    k = torch.arange(255, dtype=torch.float64)
    # 127.5 x + 128 = k + 0.9 -> k (truncation; rounding would give k + 1)
    x = ((k + 0.9 - 128.0) / 127.5).float().reshape(1, 1, 1, -1).expand(1, 3, 1, -1)
    assert torch.equal(to_uint8(x)[0, 0, 0], k.to(torch.uint8))
    # ToTensor + Normalize(0.5, 0.5) maps u to u / 127.5 - 1, and to_uint8 maps it back exactly
    u = torch.randint(0, 256, (4, 3, 16, 16), generator=_gen(1), dtype=torch.uint8)
    assert torch.equal(to_uint8(u.float() / 127.5 - 1.0), u)


def test_to_uint8_upcasts_half_precision():
    x = (torch.rand(1, 3, 4, 4, generator=_gen(2)) * 2 - 1).to(torch.bfloat16)
    assert torch.equal(to_uint8(x), to_uint8(x.float()))


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 3, 4, 4, dtype=torch.int64),   # not floating point
    torch.zeros(1, 1, 4, 4),                      # not RGB
    torch.zeros(3, 4, 4),                         # not 4-D
    torch.full((1, 3, 2, 2), float("nan")),       # NaN would become an arbitrary byte
])
def test_to_uint8_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        to_uint8(bad)


@pytest.mark.parametrize("d", [1, 7, 60])
def test_psnr_constant_offset_closed_form(d):
    ref = torch.full((2, 3, 8, 8), 100, dtype=torch.uint8)
    for test in (ref + d, ref - d):
        p = psnr_per_image(ref, test)
        assert p.dtype == torch.float64 and p.shape == (2,)
        assert np.allclose(p.numpy(), 20 * math.log10(255 / d), rtol=1e-12, atol=0)


def test_psnr_is_per_image_and_identical_images_give_inf():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (3, 3, 8, 8), dtype=np.uint8)
    test = np.clip(ref.astype(np.int64) + rng.integers(-20, 21, ref.shape), 0, 255).astype(np.uint8)
    test[2] = ref[2]
    p = psnr_per_image(torch.from_numpy(ref), torch.from_numpy(test)).numpy()
    mse = ((ref.astype(np.float64) - test) ** 2).reshape(3, -1).mean(axis=1)
    assert np.allclose(p[:2], 20 * np.log10(255 / np.sqrt(mse[:2])), rtol=1e-12, atol=0)
    assert p[2] == np.inf


def test_psnr_rejects_invalid_inputs():
    u = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        psnr_per_image(u.float(), u)
    with pytest.raises(ValueError):
        psnr_per_image(u, u[:, :, :2])


# --------------------------------------------------------------------------- LPIPSMeter

class FakeLPIPS(nn.Module):
    """Stand-in for lpips.LPIPS: per-image mean |a - b| as [B, 1, 1, 1]; records its call mode."""

    def __init__(self, out_shape=None):
        super().__init__()
        self.modes, self.out_shape = [], out_shape

    def forward(self, a, b):
        self.modes.append((self.training, torch.is_grad_enabled()))
        d = (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)
        return d if self.out_shape is None else d.reshape(-1)[:1].expand(self.out_shape)


def test_lpips_meter_collects_per_image_values_unclamped():
    x = torch.rand(5, 3, 8, 8, generator=_gen(3)) * 2 - 1
    x_hat = x + torch.randn(5, 3, 8, 8, generator=_gen(4))        # leaves [-1, 1]: not clamped
    fake = FakeLPIPS().train()
    meter = LPIPSMeter(fake)
    meter.update(x_hat[:3], x[:3])
    meter.update(x_hat[3:], x[3:])
    expected = (x_hat - x).abs().mean(dim=(1, 2, 3)).double().numpy()
    v = meter.values()
    assert v.dtype == np.float64 and v.shape == (5,)
    assert np.array_equal(v, expected)
    assert meter.mean() == pytest.approx(expected.mean(), rel=1e-12)
    assert fake.modes == [(False, False)] * 2                     # eval mode, no autograd


def test_lpips_meter_rejects_invalid_inputs():
    x = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError):
        LPIPSMeter(FakeLPIPS()).mean()                            # nothing collected yet
    with pytest.raises(ValueError):
        LPIPSMeter(FakeLPIPS()).update(x, x[:1])                  # shape mismatch
    with pytest.raises(ValueError):
        LPIPSMeter(FakeLPIPS(out_shape=(2, 2))).update(x, x)      # not one value per image
    with pytest.raises(ValueError):
        LPIPSMeter(FakeLPIPS()).update(x, torch.full_like(x, float("nan")))


# --------------------------------------------------------------------------- FIDStatistics

class FakeInception(nn.Module):
    """Stand-in for pytorch-fid's InceptionV3: [0, 1] images -> [features [b, 6, 1, 1]].

    The features are a fixed nonlinear map of per-image pixel statistics (float64, so they do
    not depend on the batch composition); every call's input and mode are recorded."""

    def __init__(self, output: str = "list"):
        super().__init__()
        self.register_buffer("proj", torch.randn(9, 6, generator=_gen(0), dtype=torch.float64))
        self.output, self.inputs, self.modes = output, [], []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().clone())
        self.modes.append((self.training, torch.is_grad_enabled()))
        x = x.double()
        s = torch.cat([x.mean(dim=(2, 3)), x.square().mean(dim=(2, 3)), x[:, :, ::2, 1::2].mean(dim=(2, 3))], dim=1)
        f = torch.tanh(s @ self.proj)
        if self.output == "tensor":
            return f
        f4 = f[:, :, None, None]
        return [f4] if self.output == "list" else (f4, 2 * f4)


def _random_u8(n: int, seed: int, size: int = 8) -> torch.Tensor:
    return torch.randint(0, 256, (n, 3, size, size), generator=_gen(seed), dtype=torch.uint8)


@pytest.mark.parametrize("output", ["list", "tuple", "tensor"])
def test_fid_statistics_equal_numpy_moments_of_the_same_features(output):
    images = _random_u8(37, seed=5)
    fake = FakeInception(output).train()
    fid = FIDStatistics(fake, "cpu", batch_size=10)
    fid.update(images[:20])
    fid.update(images[20:])
    mu, sigma, n = fid.compute()
    out = FakeInception("list")(images.float() / 255.0)[0]
    feats = out.reshape(37, -1).numpy()
    assert n == 37 and mu.dtype == sigma.dtype == np.float64
    assert _rel(mu, feats.mean(axis=0)) <= 1e-10
    assert _rel(sigma, np.cov(feats, rowvar=False)) <= 1e-10      # ddof = 1, as pytorch-fid
    # chunks of <= batch_size, fed as u8 / 255 in float32, in eval mode without autograd
    assert [t.shape[0] for t in fake.inputs] == [10, 10, 10, 7]
    seen = torch.cat(fake.inputs)
    assert seen.dtype == torch.float32 and torch.equal(seen, images.float() / 255.0)
    assert set(fake.modes) == {(False, False)}


@needs_cuda
def test_fid_statistics_on_cuda_match_cpu():
    images = _random_u8(23, seed=6)
    fake_gpu = FakeInception().cuda()
    fid_cpu, fid_gpu = FIDStatistics(FakeInception(), "cpu", 8), FIDStatistics(fake_gpu, "cuda", 8)
    fid_cpu.update(images)
    fid_gpu.update(images.cuda())                                 # e.g. decoded samples already on the GPU
    # same bits as ToTensor on the CPU (CUDA's x / 255 is x * (1/255), 1 ulp off for half the values)
    assert torch.equal(torch.cat(fake_gpu.inputs), images.float() / 255.0)
    (mu_c, s_c, n_c), (mu_g, s_g, n_g) = fid_cpu.compute(), fid_gpu.compute()
    assert n_c == n_g == 23 and _rel(mu_g, mu_c) <= 1e-10 and _rel(s_g, s_c) <= 1e-10


class _ShapeShifter(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = list(dims)

    def forward(self, x):
        return torch.ones(x.shape[0], self.dims.pop(0))


def test_fid_statistics_rejects_invalid_inputs():
    fid = FIDStatistics(FakeInception(), "cpu")
    with pytest.raises(ValueError):
        fid.compute()                                             # no images yet
    with pytest.raises(ValueError):
        fid.update(_random_u8(2, 0).float())                      # not uint8
    with pytest.raises(ValueError):
        fid.update(_random_u8(2, 0)[:, :1])                       # not RGB
    fid.update(_random_u8(1, 0))
    with pytest.raises(ValueError):
        fid.compute()                                             # a covariance needs n >= 2
    with pytest.raises(ValueError):
        FIDStatistics(FakeInception(), "cpu", batch_size=0)
    shifting = FIDStatistics(_ShapeShifter([4, 5]), "cpu")
    shifting.update(_random_u8(2, 0))
    with pytest.raises(ValueError):
        shifting.update(_random_u8(2, 0))                         # feature dimension changed


# --------------------------------------------------------------------------- Frechet distance

def test_frechet_identical_statistics_is_zero():
    rng = np.random.default_rng(0)
    S, mu = _spd(16, rng), rng.standard_normal(16)
    fd, info = frechet_distance(mu, S, mu, S, return_info=True)
    assert abs(fd) <= 1e-10 * np.trace(S)
    assert info == {"eps_offset_applied": False, "max_abs_imag": 0.0}


def test_frechet_diagonal_gaussians_closed_form():
    rng = np.random.default_rng(1)
    v1, v2 = rng.uniform(0.1, 3.0, 12), rng.uniform(0.1, 3.0, 12)
    mu1, mu2 = rng.standard_normal(12), rng.standard_normal(12)
    expected = np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2)
    fd = frechet_distance(mu1, np.diag(v1), mu2, np.diag(v2))
    assert isinstance(fd, float) and abs(fd - expected) <= 1e-12 * expected


def test_frechet_general_covariances_match_symmetric_formula():
    rng = np.random.default_rng(2)
    s1, s2 = _spd(20, rng, cond=100.0), _spd(20, rng, cond=100.0)
    mu1, mu2 = rng.standard_normal(20), rng.standard_normal(20)
    ref = _fd_reference(mu1, s1, mu2, s2)
    assert abs(frechet_distance(mu1, s1, mu2, s2) - ref) <= 1e-8 * abs(ref)


def test_frechet_rank_deficient_covariances():
    # n <= d samples (e.g. rFID on few images): singular covariances, complex sqrtm with a
    # rounding-level imaginary part that is discarded and reported
    rng = np.random.default_rng(3)
    X, Y = rng.standard_normal((10, 32)), 0.5 + 1.5 * rng.standard_normal((10, 32))
    s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    fd, info = frechet_distance(X.mean(0), s1, Y.mean(0), s2, return_info=True)
    ref = _fd_reference(X.mean(0), s1, Y.mean(0), s2)
    assert abs(fd - ref) <= 1e-6 * abs(ref)                       # ~sqrt(eps) accuracy when singular
    assert not info["eps_offset_applied"] and info["max_abs_imag"] < 1e-3
    same = frechet_distance(X.mean(0), s1, X.mean(0), s1)
    assert abs(same) <= 1e-6 * np.trace(s1)


def test_frechet_eps_offset_fallback_is_reported(monkeypatch, caplog):
    rng = np.random.default_rng(4)
    s1, s2, mu1, mu2 = _spd(6, rng), _spd(6, rng), rng.standard_normal(6), rng.standard_normal(6)
    real_sqrtm, calls = ev.linalg.sqrtm, []

    def flaky_sqrtm(a):  # the first call fails like a singular product
        calls.append(a.copy())
        return np.full_like(a, np.nan) if len(calls) == 1 else real_sqrtm(a)

    monkeypatch.setattr(ev.linalg, "sqrtm", flaky_sqrtm)
    monkeypatch.setattr(logging.getLogger("rwvae"), "propagate", True)  # a Run may have turned it off
    with caplog.at_level(logging.WARNING, logger="rwvae.evaluation"):
        fd, info = frechet_distance(mu1, s1, mu2, s2, eps=1e-3, return_info=True)
    off = 1e-3 * np.eye(6)
    assert len(calls) == 2 and np.array_equal(calls[1], (s1 + off) @ (s2 + off))
    expected = np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(real_sqrtm(calls[1]).real)
    assert abs(fd - expected) <= 1e-12 * abs(expected)
    assert info["eps_offset_applied"] is True
    assert "adding 0.001 to the diagonal" in caplog.text

    monkeypatch.setattr(ev.linalg, "sqrtm", lambda a: np.full_like(a, np.inf))
    with pytest.raises(ValueError, match="not finite"):
        frechet_distance(mu1, s1, mu2, s2)


def test_frechet_imaginary_component_handling(monkeypatch):
    rng = np.random.default_rng(5)
    s1, s2, mu = _spd(5, rng), _spd(5, rng), rng.standard_normal(5)
    clean = frechet_distance(mu, s1, mu, s2)
    real_sqrtm = ev.linalg.sqrtm
    monkeypatch.setattr(ev.linalg, "sqrtm", lambda a: real_sqrtm(a) + 1e-5j)
    fd, info = frechet_distance(mu, s1, mu, s2, return_info=True)
    assert fd == pytest.approx(clean, rel=1e-12) and info["max_abs_imag"] == pytest.approx(1e-5)
    monkeypatch.setattr(ev.linalg, "sqrtm", lambda a: real_sqrtm(a) + 1e-2j)
    with pytest.raises(ValueError, match="imaginary"):
        frechet_distance(mu, s1, mu, s2)


def test_frechet_rejects_invalid_inputs():
    S = np.eye(3)
    with pytest.raises(ValueError):
        frechet_distance(np.zeros(3), S, np.zeros(4), np.eye(4))
    with pytest.raises(ValueError):
        frechet_distance(np.zeros(3), np.eye(2), np.zeros(3), S)
    with pytest.raises(ValueError):
        frechet_distance(np.array([0, np.nan, 0]), S, np.zeros(3), S)
    with pytest.raises(ValueError):
        frechet_distance(np.zeros(3), S, np.zeros(3), S, eps=-1.0)


# --------------------------------------------------------------------------- FID stats files

def test_fid_stats_round_trip(tmp_path):
    rng = np.random.default_rng(6)
    mu, sigma = rng.standard_normal(8), _spd(8, rng)
    meta = {"split": "val", "n_images": 1000, "inception": "pt_inception-2015-12-05", "seed": 0}
    path = tmp_path / "ref_stats"                                 # no suffix: kept as given
    save_fid_stats(path, mu, sigma, 1000, meta)
    with np.load(path, allow_pickle=False) as f:
        assert set(f.files) == {"mu", "sigma", "n", "meta"} and json.loads(f["meta"].item()) == meta
    mu2, sigma2, n2, meta2 = load_fid_stats(path)
    assert np.array_equal(mu2, mu) and np.array_equal(sigma2, sigma) and n2 == 1000 and meta2 == meta


def test_load_external_reference_npz(tmp_path):
    rng = np.random.default_rng(7)
    mu, sigma = rng.standard_normal(8), _spd(8, rng)
    adm = tmp_path / "VIRTUAL_imagenet256_labeled.npz"           # ADM layout: images + pool / spatial stats
    np.savez(adm, arr_0=np.zeros((2, 4, 4, 3), np.uint8), mu=mu, sigma=sigma, mu_s=mu[:4], sigma_s=sigma[:4, :4])
    plain = tmp_path / "pytorch_fid.npz"
    np.savez(plain, mu=mu, sigma=sigma)
    for path in (adm, plain):
        m, s, n, meta = load_fid_stats(path)
        assert np.array_equal(m, mu) and np.array_equal(s, sigma) and n == -1 and meta == {"source": "external"}


def test_fid_stats_files_reject_invalid_inputs(tmp_path):
    mu, sigma = np.zeros(3), np.eye(3)
    for kwargs in (dict(n=1, meta={}), dict(n=10, meta={"x": object()}), dict(n=10, meta=[1])):
        with pytest.raises(ValueError):
            save_fid_stats(tmp_path / "bad.npz", mu, sigma, **kwargs)
    with pytest.raises(ValueError):
        save_fid_stats(tmp_path / "bad.npz", mu, np.eye(4), 10, {})
    half = tmp_path / "half.npz"
    np.savez(half, mu=mu, sigma=sigma, n=np.int64(5))           # n without meta
    nosigma = tmp_path / "nosigma.npz"
    np.savez(nosigma, mu=mu)
    mismatch = tmp_path / "mismatch.npz"
    np.savez(mismatch, mu=mu, sigma=np.eye(4))
    for path in (half, nosigma, mismatch):
        with pytest.raises(ValueError):
            load_fid_stats(path)


# --------------------------------------------------------------------------- pooled features

def test_pooled_features_shapes_and_values():
    z = torch.randn(4, 5, 16, 16, generator=_gen(7))
    f1 = pooled_features(z)
    assert f1.shape == (4, 5) and torch.allclose(f1, z.mean(dim=(2, 3)), atol=1e-6)
    f2 = pooled_features(z, grid=2)
    assert f2.shape == (4, 5 * 4)
    blocks = z.reshape(4, 5, 2, 8, 2, 8).mean(dim=(3, 5)).reshape(4, -1)   # channel-major (c, gh, gw)
    assert torch.allclose(f2, blocks, atol=1e-6)
    assert pooled_features(z.half(), grid=4).dtype == torch.float32


@pytest.mark.parametrize("z, grid", [
    (torch.zeros(2, 3, 4, 4), 0),
    (torch.zeros(2, 3, 4, 4), 5),                                 # grid larger than the latent
    (torch.zeros(3, 4, 4), 1),
    (torch.zeros(2, 3, 4, 4, dtype=torch.int64), 1),
])
def test_pooled_features_rejects_invalid_inputs(z, grid):
    with pytest.raises(ValueError):
        pooled_features(z, grid)


# --------------------------------------------------------------------------- linear probe

def _blobs(K: int, d: int, n: int, seed: int, sep: float = 4.0):
    rng = np.random.default_rng(seed)
    centers = sep * rng.standard_normal((K, d))
    y = rng.integers(0, K, 2 * n)
    X = (centers[y] + rng.standard_normal((2 * n, d))).astype(np.float32)
    return X[:n], y[:n], X[n:], y[n:]


def test_linear_probe_separable_blobs():
    Xtr, ytr, Xte, yte = _blobs(K=6, d=8, n=1500, seed=0)
    r = linear_probe(Xtr, ytr, Xte, yte, num_classes=6)
    assert r["top1"] >= 0.99 and r["train_top1"] >= 0.99 and r["top5"] >= r["top1"]
    assert r["correct"].dtype == np.bool_ and r["correct"].shape == (1500,)
    assert r["top1"] == r["correct"].mean()
    assert (r["n_train"], r["n_test"]) == (1500, 1500) and 1 <= r["n_iter"] <= 200 and r["min_train_std"] > 0


def test_linear_probe_random_labels_is_at_chance():
    rng = np.random.default_rng(1)
    K, n = 4, 2000
    Xtr, Xte = rng.standard_normal((n, 8)).astype(np.float32), rng.standard_normal((n, 8)).astype(np.float32)
    r = linear_probe(Xtr, rng.integers(0, K, n), Xte, rng.integers(0, K, n), num_classes=K)
    assert r["top5"] is None                                     # num_classes < 5
    assert abs(r["top1"] - 1 / K) < 5 * math.sqrt(0.25 * 0.75 / n)  # 5 binomial standard errors


def test_linear_probe_is_deterministic_and_accepts_tensors():
    Xtr, ytr, Xte, yte = _blobs(K=5, d=6, n=600, seed=2, sep=1.0)  # overlapping classes: top1 < 1
    a = linear_probe(Xtr, ytr, Xte, yte, num_classes=5, seed=0)
    b = linear_probe(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xte), torch.from_numpy(yte),
                     num_classes=5, seed=0)
    assert 0.3 < a["top1"] < 0.99
    assert np.array_equal(a["correct"], b["correct"])
    assert {k: v for k, v in a.items() if k != "correct"} == {k: v for k, v in b.items() if k != "correct"}


def test_linear_probe_handles_degenerate_features():
    Xtr, ytr, Xte, yte = _blobs(K=3, d=4, n=800, seed=3)
    extra = lambda X, c: np.concatenate([X, np.full((len(X), 1), c, np.float32), X[:, :1]], axis=1)
    r = linear_probe(extra(Xtr, 2.5), ytr, extra(Xte, 7.0), yte, num_classes=3)  # constant + duplicated column
    assert r["min_train_std"] == 0.0 and r["top1"] >= 0.99


def test_linear_probe_respects_max_iter():
    Xtr, ytr, Xte, yte = _blobs(K=3, d=4, n=300, seed=4)
    assert linear_probe(Xtr, ytr, Xte, yte, num_classes=3, max_iter=1)["n_iter"] == 1


@needs_cuda
def test_linear_probe_on_cuda_matches_cpu():
    Xtr, ytr, Xte, yte = _blobs(K=6, d=8, n=1000, seed=5)
    cpu = linear_probe(Xtr, ytr, Xte, yte, num_classes=6)
    gpu = linear_probe(Xtr, ytr, Xte, yte, num_classes=6, device="cuda")
    assert gpu["top1"] == cpu["top1"] and gpu["correct"].dtype == np.bool_


@pytest.mark.parametrize("case", ["label_out_of_range", "float_labels", "feature_dims_differ", "one_train_item",
                                  "nan_feature", "label_length", "one_class", "negative_weight_decay", "zero_max_iter"])
def test_linear_probe_rejects_invalid_inputs(case):
    Xtr, ytr, Xte, yte = _blobs(K=3, d=4, n=50, seed=6)
    nan_x = Xtr.copy()
    nan_x[0, 0] = np.nan
    bad = {
        "label_out_of_range": dict(train_y=np.where(ytr == 0, 3, ytr)),
        "float_labels": dict(test_y=yte.astype(np.float32)),
        "feature_dims_differ": dict(test_x=Xte[:, :3]),
        "one_train_item": dict(train_x=Xtr[:1], train_y=ytr[:1]),
        "nan_feature": dict(train_x=nan_x),
        "label_length": dict(train_y=ytr[:-1]),
        "one_class": dict(num_classes=1),
        "negative_weight_decay": dict(weight_decay=-1.0),
        "zero_max_iter": dict(max_iter=0),
    }[case]
    with pytest.raises(ValueError):
        linear_probe(**{**dict(train_x=Xtr, train_y=ytr, test_x=Xte, test_y=yte, num_classes=3), **bad})


# --------------------------------------------------------------------------- DCT spectra

@pytest.mark.parametrize("N", [1, 2, 3, 7, 16])
def test_dct_matrix_is_orthonormal_dct_ii(N):
    D = dct_matrix(N)
    assert D.dtype == torch.float64 and D.shape == (N, N)
    I = np.eye(N)
    assert np.abs(D.numpy() @ D.numpy().T - I).max() <= 1e-13 and np.abs(D.numpy().T @ D.numpy() - I).max() <= 1e-13
    x = np.random.default_rng(N).standard_normal((N, 3))
    assert np.abs(D.numpy() @ x - scipy.fft.dct(x, type=2, norm="ortho", axis=0)).max() <= 1e-13


def test_dct2_power_parseval():
    z = torch.randn(6, 4, 16, 12, generator=_gen(8), dtype=torch.float64) + 0.7
    energy = z.square().sum(dim=(2, 3))                          # [B, C]
    p = dct2_power(z)
    assert p.dtype == torch.float64 and p.shape == (16, 12)
    assert abs(float(p.sum()) / float(energy.mean()) - 1) <= 1e-12
    pc = dct2_power(z, per_channel=True)
    assert pc.shape == (4, 16, 12) and _rel(pc.sum(dim=(1, 2)), energy.mean(dim=0)) <= 1e-12


def test_dct2_power_of_a_basis_function_is_one_hot():
    H, W, k1, k2 = 8, 16, 3, 11
    z = torch.outer(dct_matrix(H)[k1], dct_matrix(W)[k2]).reshape(1, 1, H, W)
    target = torch.zeros(H, W, dtype=torch.float64)
    target[k1, k2] = 1.0
    assert torch.allclose(dct2_power(z), target, rtol=0, atol=1e-14)


def test_dct2_power_is_the_mean_of_spectra_not_the_spectrum_of_the_mean():
    pattern = torch.randn(1, 3, 8, 8, generator=_gen(9), dtype=torch.float64)
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64).reshape(4, 1, 1, 1)
    z = signs * pattern                                          # batch mean is exactly zero
    assert torch.allclose(dct2_power(z), dct2_power(pattern), rtol=1e-13, atol=0)
    assert float(dct2_power(z.mean(dim=0, keepdim=True)).abs().max()) == 0.0
    singles = torch.stack([dct2_power(z[i:i + 1]) for i in range(4)]).mean(dim=0)
    assert torch.allclose(dct2_power(z), singles, rtol=1e-13, atol=0)


def test_radial_bands_white_noise_is_flat():
    B, C, n_bands = 64, 8, 4
    z = torch.randn(B, C, 16, 16, generator=_gen(10))
    bands = radial_band_power(dct2_power(z), n_bands)
    # an orthonormal transform keeps white noise white: E X^2 = 1 in every cell; a band
    # averages B*C*cells chi^2_1 draws (variance 2), so its standard error is sqrt(2 / (B C cells))
    kh, kw = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    r = np.hypot(kh, kw).ravel()
    cells = np.bincount(np.minimum((r / r.max() * n_bands).astype(int), n_bands - 1), minlength=n_bands)
    se = np.sqrt(2.0 / (B * C * cells))
    assert bands.dtype == np.float64 and bands.shape == (n_bands,)
    assert np.all(np.abs(bands - 1.0) < 5 * se), (bands, se)


def test_radial_bands_smooth_field_is_low_frequency_heavy():
    low = torch.randn(32, 4, 4, 4, generator=_gen(11))
    smooth = F.interpolate(low, size=(16, 16), mode="bilinear", align_corners=False)
    bands = radial_band_power(dct2_power(smooth), 4)
    assert np.all(np.diff(bands) < 0) and bands[0] > 1e3 * bands[-1]


def test_radial_band_power_hand_example_and_exact_edges():
    # 3x3 grid: radii 0, 1, 1 | sqrt2 (on the edge r_max / 2 -> upper band), 2, 2, sqrt5, sqrt5, sqrt8
    p = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert np.allclose(radial_band_power(p, 2), [(0 + 1 + 3) / 3, (4 + 2 + 6 + 5 + 7 + 8) / 6], rtol=1e-15, atol=0)
    assert np.allclose(radial_band_power(torch.from_numpy(p), 1), [p.mean()], rtol=1e-15, atol=0)
    # 16x16, 3 bands: the diagonal cell (5, 5) lies exactly on e_1 = r_max / 3 and goes to band 1
    one_hot = np.zeros((16, 16))
    one_hot[5, 5] = 1.0
    bands = radial_band_power(one_hot, 3)
    assert bands[0] == 0.0 and bands[1] > 0.0 and bands[2] == 0.0


@pytest.mark.parametrize("power, n_bands", [
    (np.ones((3, 3)), 8),                                       # bands without any cell
    (np.ones((3, 3)), 0),
    (np.ones((2, 3, 3)), 2),
    (np.full((3, 3), np.nan), 2),
])
def test_radial_band_power_rejects_invalid_inputs(power, n_bands):
    with pytest.raises(ValueError):
        radial_band_power(power, n_bands)


def test_dct2_power_rejects_invalid_inputs():
    for bad in (torch.zeros(2, 3, 4, dtype=torch.float32), torch.zeros(2, 3, 4, 4, dtype=torch.int64),
                torch.full((1, 1, 2, 2), float("inf")), torch.zeros(0, 3, 4, 4)):
        with pytest.raises(ValueError):
            dct2_power(bad)
    with pytest.raises(ValueError):
        dct_matrix(0)


# --------------------------------------------------------------------------- bootstrap

def test_bootstrap_mean_ci_basic_and_deterministic():
    v = np.random.default_rng(0).exponential(size=300)
    a, b = bootstrap_mean_ci(v, n_boot=500, seed=3), bootstrap_mean_ci(v, n_boot=500, seed=3)
    assert a == b and a["mean"] == float(v.mean()) and a["lo"] < a["mean"] < a["hi"]
    assert (a["n"], a["n_boot"], a["alpha"]) == (300, 500, 0.05)
    assert bootstrap_mean_ci(v, n_boot=500, seed=4)["lo"] != a["lo"]
    const = bootstrap_mean_ci(np.full(10, 2.5), n_boot=50)       # degenerate: zero spread
    assert const["lo"] == const["hi"] == const["mean"] == 2.5


def test_bootstrap_ci_coverage_and_width():
    # 200 independent N(0, 1) samples of n = 100: a 95% interval should contain 0 about 95% of
    # the time; the binomial standard error of the coverage is sqrt(.95 * .05 / 200) = 0.015
    hits = 0
    for i in range(200):
        ci = bootstrap_mean_ci(np.random.default_rng(1000 + i).standard_normal(100), n_boot=500, seed=i)
        hits += ci["lo"] <= 0.0 <= ci["hi"]
    assert 0.89 <= hits / 200 <= 0.99
    # half-width ~ z_{0.975} * sd / sqrt(n) (the exact bootstrap SE uses the ddof = 0 sd)
    v = np.random.default_rng(7).standard_normal(400)
    ci = bootstrap_mean_ci(v, n_boot=2000, seed=0)
    assert abs((ci["hi"] - ci["lo"]) / 2 / (1.959964 * v.std() / 20.0) - 1) < 0.1


def test_paired_bootstrap_detects_a_shift_hidden_by_between_image_variance():
    rng = np.random.default_rng(8)
    base = 5.0 * rng.standard_normal(200)                        # large between-image spread
    a, b = base + 0.3 + 0.1 * rng.standard_normal(200), base
    ca, cb = bootstrap_mean_ci(a, seed=1), bootstrap_mean_ci(b, seed=1)
    assert ca["lo"] < cb["hi"] and cb["lo"] < ca["hi"]           # marginal intervals overlap
    d = paired_bootstrap_diff(a, b, seed=1)
    assert d["lo"] > 0 and d["mean"] == pytest.approx(a.mean() - b.mean(), rel=1e-12)
    assert d == bootstrap_mean_ci(a - b, seed=1)
    acc = paired_bootstrap_diff(np.ones(50, bool), np.r_[np.ones(25, bool), np.zeros(25, bool)], n_boot=200)
    assert acc["mean"] == 0.5 and acc["lo"] > 0                  # linear_probe "correct" arrays


@pytest.mark.parametrize("values, kwargs", [
    (np.array([1.0, np.inf, 2.0]), {}),                           # e.g. PSNR of an identical image
    (np.array([1.0, np.nan]), {}),
    (np.array([1.0]), {}),
    (np.ones((3, 2)), {}),
    (np.ones(5), dict(alpha=0.0)),
    (np.ones(5), dict(alpha=1.0)),
    (np.ones(5), dict(n_boot=0)),
    (np.ones(5), dict(seed=-1)),
])
def test_bootstrap_rejects_invalid_inputs(values, kwargs):
    with pytest.raises(ValueError):
        bootstrap_mean_ci(values, **kwargs)


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(np.ones(5), np.ones(6))


# --------------------------------------------------------------------------- real assets

def _imagenette_u8(n: int) -> torch.Tensor:
    """n Imagenette val images (round-robin over classes), centre-cropped to 256, uint8 [n, 3, 256, 256]."""
    from PIL import Image, ImageOps

    val = Path(ASSETS) / "datasets" / "imagenette2-320" / "val"
    classes = sorted(p for p in val.iterdir() if p.is_dir())
    paths = [sorted(classes[i % len(classes)].glob("*.JPEG"))[i // len(classes)] for i in range(n)]
    arrays = [np.asarray(ImageOps.fit(Image.open(p).convert("RGB"), (256, 256), Image.BICUBIC)) for p in paths]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


@pytest.mark.real_assets
@needs_assets
def test_real_inception_statistics_match_pytorch_fid_reference(monkeypatch):
    pytest.importorskip("pytorch_fid")
    from pytorch_fid.inception import InceptionV3
    from torchvision.transforms.functional import to_tensor

    monkeypatch.setenv("TORCH_HOME", str(Path(ASSETS) / "torch_cache"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
    inception = inception.eval().requires_grad_(False).to(device)
    images = _imagenette_u8(12)
    fid = FIDStatistics(inception, device, batch_size=6)
    fid.update(images)
    mu, sigma, n = fid.compute()
    # pytorch-fid's get_activations: ToTensor on HWC uint8 images, model(batch)[0] squeezed, np.mean / np.cov
    acts = []
    with torch.no_grad():
        for chunk in images.split(6):
            batch = torch.stack([to_tensor(img.permute(1, 2, 0).numpy()) for img in chunk]).to(device)
            acts.append(inception(batch)[0].squeeze(3).squeeze(2).cpu().numpy())
    acts = np.concatenate(acts).astype(np.float64)
    assert n == 12 and mu.shape == (2048,) and sigma.shape == (2048, 2048)
    assert _rel(mu, acts.mean(axis=0)) <= 1e-6 and _rel(sigma, np.cov(acts, rowvar=False)) <= 1e-6
    # two disjoint 6-image sets: rank-5 covariances in 2048-d -- finite, positive, imaginary part reported
    other = FIDStatistics(inception, device, batch_size=6)
    other.update(_imagenette_u8(24)[12:])
    mu2, sigma2, _ = other.compute()
    fd, info = frechet_distance(mu, sigma, mu2, sigma2, return_info=True)
    print(f"FD(12 vs 12 Imagenette images) = {fd:.2f}, info = {info}")
    assert math.isfinite(fd) and fd > 0 and info["max_abs_imag"] < 1e-3


@pytest.mark.real_assets
@needs_assets
def test_real_lpips_meter(monkeypatch):
    lpips = pytest.importorskip("lpips")
    monkeypatch.setenv("TORCH_HOME", str(Path(ASSETS) / "torch_cache"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False).to(device)
    x = (_imagenette_u8(6).float() / 127.5 - 1.0).to(device)
    noise = torch.randn(x.shape, generator=_gen(0)).to(device)
    same, mild, strong, split = LPIPSMeter(model), LPIPSMeter(model), LPIPSMeter(model), LPIPSMeter(model)
    same.update(x, x)
    mild.update((x + 0.05 * noise).clamp(-1, 1), x)
    strong.update((x + 0.3 * noise).clamp(-1, 1), x)
    split.update((x[:2] + 0.05 * noise[:2]).clamp(-1, 1), x[:2])
    split.update((x[2:] + 0.05 * noise[2:]).clamp(-1, 1), x[2:])
    print("LPIPS mild / strong:", mild.values().round(4), strong.values().round(4))
    assert same.values().shape == (6,) and np.abs(same.values()).max() < 1e-6
    assert np.all(mild.values() > 0) and np.all(strong.values() > mild.values())
    assert np.allclose(split.values(), mild.values(), rtol=1e-4, atol=1e-6)    # independent of batching
