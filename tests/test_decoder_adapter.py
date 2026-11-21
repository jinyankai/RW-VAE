"""Tests for rwvae.decoder_adapter (plan v2.1 P6, Sec. 3.2 decode_with_adapter, M4 decoder loop).

Channel matrices come from an independent numpy construction (weighted KLT + bisection
water-filling, copied from tests/test_channel.py), not from rwvae.allocation. The
tokenizer is stood in for by a tiny (post_quant_conv, decoder) pair: latent [B, C, 4, 4]
-> image [B, 3, 8, 8].
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from rwvae.channel import RWChannel, noise_seed, sample_noise
from rwvae.decoder_adapter import (AdaptedDecoder, decode_with_adapter, finetune_decoder,
                                   fuse_affine_into_conv1x1, lmmse_affine)

C = 8


# --- reference channel construction (copied from tests/test_channel.py) ---------------
def _random_spd(C, rng, cond=50.0):
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    eig = np.logspace(0, np.log10(cond), C)
    return (Q * eig) @ Q.T


def _sym_pow(M, p):
    e, V = np.linalg.eigh(M)
    return (V * e ** p) @ V.T


def _reference_channel(Sigma, Sigma_expl, w, R):
    """Independent construction: weighted KLT + bisection water-filling."""
    K = _sym_pow(Sigma, -0.5) @ Sigma_expl @ _sym_pow(Sigma, -0.5)
    M = np.eye(len(Sigma)) + (w ** 2 - 1.0) * (K + K.T) / 2
    Mh = _sym_pow(M, 0.5)
    s, Q = np.linalg.eigh(Mh @ Sigma @ Mh)
    s, Q = s[::-1], Q[:, ::-1]
    T, Tinv = Q.T @ Mh, _sym_pow(M, -0.5) @ Q
    lo, hi = 1e-300, s.max()
    for _ in range(400):
        mid = np.sqrt(lo * hi)
        rate = 0.5 * np.sum(np.log(np.maximum(s / mid, 1.0)))
        lo, hi = (mid, hi) if rate > R else (lo, mid)
    theta = np.sqrt(lo * hi)
    keep = np.maximum(1.0 - theta / s, 0.0)
    A = Tinv @ np.diag(np.sqrt(keep)) @ T
    B = Tinv @ np.diag(np.sqrt((1.0 - keep) * s))
    return A, B, keep


# --- helpers ---------------------------------------------------------------------------
def _channel_A_m(case, C, rng):
    """(A, m) of a test channel covering generic, repeated, near-singular and degenerate cases."""
    m = rng.standard_normal(C)
    if case == "identity":            # B0 / R = inf: A = I
        return np.eye(C), m
    if case == "zero_rate":           # R = 0: keep = 0, A = 0, z' carries no information on mu
        return np.zeros((C, C)), m
    if case == "generic":
        Sigma = _random_spd(C, rng)
    elif case == "repeated":          # Sigma = 2 I: repeated eigenvalues in Sigma and in G
        Sigma = 2.0 * np.eye(C)
    elif case == "near_singular":
        Sigma = _random_spd(C, rng, cond=1e8)
    L = np.linalg.cholesky(Sigma)
    A, _, keep = _reference_channel(Sigma, 0.6 * L[:, :3] @ L[:, :3].T, w=2.0, R=6.0)
    assert keep.min() == 0.0 or case == "repeated"   # a dropped direction makes A singular
    return A, m


def _tiny_tokenizer(seed, dtype=torch.float32, frozen=False):
    torch.manual_seed(seed)
    pqc = nn.Conv2d(C, C, 1).to(dtype)
    dec = nn.Sequential(nn.Conv2d(C, 16, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2),
                        nn.Conv2d(16, 3, 3, padding=1)).to(dtype)
    if frozen:  # like FrozenTokenizer: eval + requires_grad_(False)
        for mod in (pqc, dec):
            mod.requires_grad_(False).eval()
    return pqc, dec


def _snapshot(*mods):
    return [t.detach().clone() for mod in mods for t in mod.state_dict().values()]


def _unchanged(snapshot, *mods):
    now = [t for mod in mods for t in mod.state_dict().values()]
    return len(now) == len(snapshot) and all(torch.equal(a, b) for a, b in zip(snapshot, now))


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _prelayer(z, A, m):
    """m + A (z - m) per token, computed directly (the unfused P6 path)."""
    A_t, m_t = torch.as_tensor(A, dtype=z.dtype), torch.as_tensor(m, dtype=z.dtype)[None, :, None, None]
    return m_t + torch.einsum("ij,bjhw->bihw", A_t, z - m_t)


def _mse_lpips(a, b):
    """LPIPS stand-in with LPIPS's output shape [B, 1, 1, 1]."""
    return ((a - b) ** 2).mean(dim=(1, 2, 3), keepdim=True)


class _PairDataset(Dataset):
    """In-memory (image, mu, label, index) samples; records set_epoch calls."""

    def __init__(self, images, mu):
        self.images, self.mu, self.epochs = images, mu, []

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, i):
        return self.images[i], self.mu[i], i % 3, i

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def _synthetic_problem(n=16):
    """mu ~ N(m, Sigma) per token; images = tanh(teacher decode(mu)); channel at R = 8 nats."""
    rng = np.random.default_rng(0)
    Sigma = _random_spd(C, rng)
    L = np.linalg.cholesky(Sigma)
    m = rng.standard_normal(C)
    A, B, _ = _reference_channel(Sigma, 0.6 * L[:, :3] @ L[:, :3].T, w=2.0, R=8.0)
    tokens = m + rng.standard_normal((n * 16, C)) @ L.T
    mu = torch.from_numpy(tokens.reshape(n, 4, 4, C).transpose(0, 3, 1, 2).copy()).float()
    teacher_pqc, teacher_dec = _tiny_tokenizer(100)
    with torch.no_grad():
        images = torch.tanh(teacher_dec(teacher_pqc(mu)))
    return images, mu, RWChannel(m, A, B, rate=8.0), A, m


def _loader(images, mu, shuffle=True):
    ds = _PairDataset(images, mu)
    gen = torch.Generator().manual_seed(0)
    return DataLoader(ds, batch_size=4, shuffle=shuffle, generator=gen if shuffle else None), ds


FT = dict(lpips_fn=_mse_lpips, lr=3e-3, device="cpu", noise_base_seed=7)


# --- lmmse_affine / fuse_affine_into_conv1x1 ---------------------------------------------
def test_lmmse_affine_is_p6_estimator():
    rng = np.random.default_rng(1)
    A, m = _channel_A_m("generic", C, rng)
    W_pre, b_pre = lmmse_affine(A, m)
    assert W_pre.dtype == b_pre.dtype == np.float64
    assert np.array_equal(W_pre, A) and W_pre is not A
    z = rng.standard_normal((100, C))
    ref = m + (z - m) @ A.T                                   # z_hat = m + A (z' - m)
    assert np.linalg.norm(z @ W_pre.T + b_pre - ref) / np.linalg.norm(ref) <= 1e-12


@pytest.mark.parametrize("A, m", [(np.eye(3)[:2], np.zeros(3)), (np.eye(3), np.zeros(4)),
                                  (np.full((3, 3), np.nan), np.zeros(3)), (np.eye(3), [0.0, np.inf, 0.0])],
                         ids=["non_square", "m_length", "nan_A", "inf_m"])
def test_lmmse_affine_rejects_invalid(A, m):
    with pytest.raises(ValueError):
        lmmse_affine(A, m)


@pytest.mark.parametrize("bias, k", [(True, C), (False, C), (True, 5)], ids=["bias", "no_bias", "rectangular"])
def test_fuse_equals_prelayer_then_conv(bias, k):
    torch.manual_seed(0)
    conv = nn.Conv2d(C, 6, 1, bias=bias, dtype=torch.float64)
    before = _snapshot(conv)
    rng = np.random.default_rng(2)
    W_pre, b_pre = rng.standard_normal((C, k)), rng.standard_normal(C)
    fused = fuse_affine_into_conv1x1(conv, W_pre, b_pre)
    z = torch.randn(3, k, 5, 4, dtype=torch.float64)
    pre = torch.einsum("ij,bjhw->bihw", torch.from_numpy(W_pre), z) + torch.from_numpy(b_pre)[None, :, None, None]
    with torch.no_grad():
        assert _rel(fused(z), conv(pre)) <= 1e-12
    assert fused is not conv and fused.in_channels == k and fused.bias is not None
    assert fused.weight.dtype == torch.float64 and fused.weight.device == conv.weight.device
    assert _unchanged(before, conv) and (conv.bias is None) == (not bias)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def test_fuse_computes_in_float64_then_casts_and_copies_flags(device):
    torch.manual_seed(0)
    conv = nn.Conv2d(C, 6, 1).requires_grad_(False).eval().to(device)     # float32, frozen
    rng = np.random.default_rng(3)
    W_pre, b_pre = rng.standard_normal((C, C)), rng.standard_normal(C)
    fused = fuse_affine_into_conv1x1(conv, W_pre, b_pre)
    W0, b0 = conv.weight.double().cpu()[:, :, 0, 0].numpy(), conv.bias.double().cpu().numpy()
    assert fused.weight.dtype == torch.float32 and fused.weight.device == conv.weight.device
    assert torch.equal(fused.weight[:, :, 0, 0].cpu(), torch.from_numpy(W0 @ W_pre).float())
    assert torch.equal(fused.bias.cpu(), torch.from_numpy(b0 + W0 @ b_pre).float())
    assert not fused.weight.requires_grad and not fused.bias.requires_grad and not fused.training


@pytest.mark.parametrize("conv", [nn.Conv2d(C, C, 3, padding=1), nn.Conv2d(C, C, 1, stride=2),
                                  nn.Conv2d(C, C, 1, padding=1), nn.Conv2d(C, C, 1, dilation=2),
                                  nn.Conv2d(C, C, 1, groups=2), nn.Linear(C, C)],
                         ids=["kernel3", "stride2", "padding1", "dilation2", "groups2", "linear"])
def test_fuse_rejects_non_plain_1x1_conv(conv):
    with pytest.raises(ValueError):
        fuse_affine_into_conv1x1(conv, np.eye(C), np.zeros(C))


@pytest.mark.parametrize("W_pre, b_pre", [(np.eye(C + 1), np.zeros(C + 1)), (np.eye(C), np.zeros(C + 1)),
                                          (np.ones(C), np.zeros(C)), (np.full((C, C), np.nan), np.zeros(C)),
                                          (np.eye(C), np.full(C, np.inf))],
                         ids=["W_rows", "b_length", "W_1d", "nan_W", "inf_b"])
def test_fuse_rejects_bad_affine(W_pre, b_pre):
    with pytest.raises(ValueError):
        fuse_affine_into_conv1x1(nn.Conv2d(C, 6, 1), W_pre, b_pre)


# --- AdaptedDecoder ------------------------------------------------------------------------
def test_identity_adapter_equals_original_modules():
    pqc, dec = _tiny_tokenizer(0)
    adapter = AdaptedDecoder(pqc, dec)
    z = torch.randn(2, C, 4, 4)
    with torch.no_grad():
        out = decode_with_adapter(adapter, z)
        assert out.shape == (2, 3, 8, 8)
        assert torch.equal(out, dec(pqc(z))) and torch.equal(out, adapter(z))
    assert adapter.mode == adapter.base_mode == "identity"
    # the adapter holds copies: no shared storage, and training them leaves the originals alone
    originals = {p.data_ptr() for p in [*pqc.parameters(), *dec.parameters()]}
    assert originals.isdisjoint(p.data_ptr() for p in adapter.trainable_parameters())
    assert len(list(adapter.trainable_parameters())) == len(list(adapter.parameters())) == 6


@pytest.mark.parametrize("case", ["generic", "repeated", "near_singular", "zero_rate"])
def test_lmmse_adapter_equals_decoder_of_lmmse_estimate(case):
    A, m = _channel_A_m(case, C, np.random.default_rng(4))
    pqc, dec = _tiny_tokenizer(0, dtype=torch.float64)
    before = _snapshot(pqc, dec)
    adapter = AdaptedDecoder(pqc, dec, mode="lmmse", A=A, m=m)
    z1, z2 = torch.randn(2, 3, C, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        out = adapter(z1)
        assert _rel(out, dec(pqc(_prelayer(z1, A, m)))) <= 1e-10    # decoder(post_quant_conv(m + A(z' - m)))
        if case == "zero_rate":                                        # R = 0 decodes the mean for every z'
            assert torch.equal(out, adapter(z2))
    assert _unchanged(before, pqc, dec)
    assert adapter.mode == adapter.base_mode == "lmmse"


def test_lmmse_with_identity_channel_is_bit_exact():
    # A = I (B0): W0 I = W0 and b0 + W0 (I - I) m = b0 hold exactly in float64, so the fused
    # float32 conv must reproduce the original decoder bit for bit
    A, m = _channel_A_m("identity", C, np.random.default_rng(5))
    pqc, dec = _tiny_tokenizer(0)
    z = torch.randn(2, C, 4, 4)
    with torch.no_grad():
        assert torch.equal(AdaptedDecoder(pqc, dec, mode="lmmse", A=A, m=m)(z), AdaptedDecoder(pqc, dec)(z))


def test_adapter_rejects_invalid_arguments():
    pqc, dec = _tiny_tokenizer(0)
    A, m = np.eye(C), np.zeros(C)
    with pytest.raises(ValueError):
        AdaptedDecoder(pqc, dec, mode="finetuned")         # only via finetune_decoder / load_weights
    with pytest.raises(ValueError):
        AdaptedDecoder(pqc, dec, mode="LMMSE", A=A, m=m)
    with pytest.raises(ValueError):
        AdaptedDecoder(pqc, dec, mode="lmmse", A=A)         # m missing
    with pytest.raises(ValueError):
        AdaptedDecoder(pqc, dec, mode="identity", A=A, m=m)
    with pytest.raises(ValueError):
        AdaptedDecoder(pqc, dec, mode="lmmse", A=np.eye(C + 1), m=np.zeros(C + 1))
    with pytest.raises(ValueError):
        AdaptedDecoder(dec, pqc)                            # swapped arguments
    adapter = AdaptedDecoder(pqc, dec)
    with pytest.raises(ValueError):
        adapter(torch.randn(2, C + 1, 4, 4))
    with pytest.raises(ValueError):
        adapter(torch.randn(C, 4, 4))


def test_save_load_round_trip(tmp_path):
    images, mu, channel, A, m = _synthetic_problem()
    pqc, dec = _tiny_tokenizer(0, frozen=True)
    adapter = AdaptedDecoder(pqc, dec, mode="lmmse", A=A, m=m)
    with pytest.raises(ValueError):
        adapter.save(tmp_path / "analytic.safetensors", {})   # analytic adapters are rebuilt, not saved
    finetune_decoder(adapter, _loader(images, mu)[0], channel, steps=3, **FT)
    meta = {"arm": "r2_R8_w2", "steps": 3, "R": 8.0, "recipe": {"l1": 1.0, "lpips": 1.0}, "seeds": [7]}
    path = tmp_path / "decoder.safetensors"
    adapter.save(path, meta)
    with safe_open(str(path), framework="pt") as f:
        md = f.metadata()
        assert set(f.keys()) == set(adapter.state_dict())
    assert json.loads(md["meta"]) == meta and md["base_mode"] == "lmmse"

    fresh = AdaptedDecoder(pqc, dec)       # any base: load_weights replaces every tensor
    assert fresh.load_weights(path) == meta
    assert fresh.mode == "finetuned" and fresh.base_mode == "lmmse"
    z = torch.randn(3, C, 4, 4)
    with torch.no_grad():
        assert torch.equal(fresh(z), adapter(z))
        assert not torch.equal(fresh(z), AdaptedDecoder(pqc, dec, mode="lmmse", A=A, m=m)(z))

    with pytest.raises(RuntimeError):      # strict: another architecture does not load
        AdaptedDecoder(nn.Conv2d(C, C, 1), nn.Conv2d(C, 3, 1)).load_weights(path)
    foreign = tmp_path / "foreign.safetensors"
    save_file({"w": torch.zeros(1)}, str(foreign))
    with pytest.raises(ValueError):
        fresh.load_weights(foreign)


# --- finetune_decoder ------------------------------------------------------------------------
def test_finetune_decreases_loss_and_is_deterministic():
    images, mu, channel, A, m = _synthetic_problem()
    pqc, dec = _tiny_tokenizer(0, frozen=True)
    before = _snapshot(pqc, dec)
    runs = []
    for _ in range(2):
        adapter = AdaptedDecoder(pqc, dec, mode="lmmse", A=A, m=m)
        loader, ds = _loader(images, mu)
        logs = []
        out = finetune_decoder(adapter, loader, channel, steps=40, log_every=10,
                               log_fn=lambda step, d: logs.append((step, d)), **FT)
        runs.append((adapter, logs, out, ds.epochs))
    (a1, logs1, out1, epochs1), (a2, logs2, out2, _) = runs

    assert [step for step, _ in logs1] == [10, 20, 30, 40]
    assert logs1[-1][1]["loss"] < 0.7 * logs1[0][1]["loss"]          # observed ratio ~0.49
    assert out1 == {"steps": 40, "final_loss": logs1[-1][1]["loss"], "final_l1": logs1[-1][1]["l1"],
                    "final_lpips": logs1[-1][1]["lpips"], "mixed_precision": "no"}   # bf16 is CUDA-only
    assert epochs1 == list(range(10))                                  # 16 samples / batch 4 -> 10 passes
    assert a1.mode == "finetuned" and a1.base_mode == "lmmse"
    assert not a1.training and all(p.requires_grad for p in a1.parameters())
    # same seeds -> identical logs and weights
    assert logs1 == logs2 and out1 == out2
    assert all(torch.equal(x, y) for x, y in zip(a1.state_dict().values(), a2.state_dict().values()))
    # the frozen originals are neither changed nor unfrozen
    assert _unchanged(before, pqc, dec)
    assert not any(p.requires_grad for p in [*pqc.parameters(), *dec.parameters()])


def test_finetune_noise_is_seeded_per_sample_and_step():
    images, mu, channel, A, m = _synthetic_problem()
    adapter = AdaptedDecoder(*_tiny_tokenizer(0), mode="lmmse", A=A, m=m)
    seen = []
    adapter.post_quant_conv.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    finetune_decoder(adapter, _loader(images, mu, shuffle=False)[0], channel, steps=6, **{**FT, "noise_base_seed": 3})
    assert len(seen) == 6
    for step, z in enumerate(seen):
        idx = list(range(4 * (step % 4), 4 * (step % 4) + 4))        # unshuffled batches of 4
        eps = sample_noise((C, 4, 4), [noise_seed(3, step, i) for i in idx])
        assert torch.equal(z, channel(mu[idx], noise=eps))
    assert not torch.equal(seen[0], seen[4])                           # same samples, fresh noise


def test_finetune_grad_clip_is_applied():
    images, mu, channel, A, m = _synthetic_problem()
    states = []
    for clip in (None, 1e-3):
        adapter = AdaptedDecoder(*_tiny_tokenizer(0), mode="lmmse", A=A, m=m)
        out = finetune_decoder(adapter, _loader(images, mu)[0], channel, steps=5, grad_clip=clip, **FT)
        assert np.isfinite(out["final_loss"])
        states.append(adapter.state_dict())
    assert not all(torch.equal(x, y) for x, y in zip(states[0].values(), states[1].values()))


@pytest.mark.parametrize("bad", [dict(steps=0), dict(steps=2.5), dict(lr=0.0), dict(lr=float("nan")),
                                 dict(log_every=0), dict(grad_clip=0.0), dict(mixed_precision="fp16"),
                                 dict(l1_weight=0.0, lpips_weight=0.0), dict(l1_weight=-1.0)],
                         ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_finetune_rejects_invalid_arguments(bad):
    images, mu, channel, A, m = _synthetic_problem()
    adapter = AdaptedDecoder(*_tiny_tokenizer(0), mode="lmmse", A=A, m=m)
    with pytest.raises(ValueError):
        finetune_decoder(adapter, _loader(images, mu)[0], channel, **{**FT, "steps": 2, **bad})
    assert adapter.mode == "lmmse"


def test_finetune_rejects_empty_loader_and_shape_mismatch():
    images, mu, channel, A, m = _synthetic_problem()
    adapter = AdaptedDecoder(*_tiny_tokenizer(0), mode="lmmse", A=A, m=m)
    with pytest.raises(ValueError):
        finetune_decoder(adapter, [], channel, steps=2, **FT)
    with pytest.raises(ValueError):
        finetune_decoder(adapter, _loader(images[:, :, :4, :4], mu)[0], channel, steps=2, **FT)
    assert adapter.mode == "lmmse"                                     # no update happened


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_finetune_cuda_bf16():
    images, mu, channel, A, m = _synthetic_problem()
    adapter = AdaptedDecoder(*_tiny_tokenizer(0), mode="lmmse", A=A, m=m)
    logs = []
    out = finetune_decoder(adapter, _loader(images, mu)[0], channel, steps=40, log_every=10, grad_clip=1.0,
                           log_fn=lambda step, d: logs.append(d["loss"]), **{**FT, "device": "cuda"})
    assert out["mixed_precision"] == "bf16"
    assert logs[-1] < 0.7 * logs[0]
    assert all(p.device.type == "cuda" and p.dtype == torch.float32 for p in adapter.parameters())


# --- real tokenizer (plan P6: check the actual first decoder layer) ----------------------------
@pytest.mark.real_assets
@pytest.mark.skipif(not os.environ.get("RWVAE_ASSETS"), reason="set RWVAE_ASSETS to the assets directory")
def test_real_tokenizer_post_quant_conv_fusion():
    ckpt = Path(os.environ["RWVAE_ASSETS"]) / "checkpoints" / "ldm-imagenet256-f16d32-50ep.ckpt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    W, b = sd["post_quant_conv.weight"], sd["post_quant_conv.bias"]
    assert W.shape == (32, 32, 1, 1) and b.shape == (32,)          # 1x1 conv with bias, as P6 assumes
    assert sd["decoder.conv_in.weight"].shape[1] == 32               # decode(z) = decoder(post_quant_conv(z))
    torch.manual_seed(0)
    rng = np.random.default_rng(6)
    Sigma = _random_spd(32, rng, cond=1e3)
    L = np.linalg.cholesky(Sigma)
    A, _, _ = _reference_channel(Sigma, 0.6 * L[:, :8] @ L[:, :8].T, w=2.0, R=22.0)
    m = rng.standard_normal(32)
    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 1e-5)):
        conv = nn.Conv2d(32, 32, 1, dtype=dtype)
        conv.load_state_dict({"weight": W, "bias": b})
        fused = fuse_affine_into_conv1x1(conv, *lmmse_affine(A, m))
        z = torch.randn(2, 32, 16, 16, dtype=dtype)
        with torch.no_grad():
            assert _rel(fused(z), conv(_prelayer(z, A, m))) <= tol
