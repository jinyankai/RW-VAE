"""RW-VAE v2: evaluation metrics -- PSNR, LPIPS, FID, linear probe, latent DCT spectra, bootstrap CIs.

Plan v2.1 Sec. 5.1 (core metrics T1/T2) and Sec. 1.2 (images are the independent sampling
unit); contract: docs/ARCHITECTURE_V2.md Sec. 3.8. The metric networks (FID Inception,
LPIPS-VGG) are injected by the caller (`rwvae.integration.load_inception` / `load_lpips`),
so everything here runs on CPU with stand-in modules in the unit tests.

  pixels   u8 = clamp(127.5 x + 128, 0, 255) cast to uint8 (LightningDiT's quantisation);
           PSNR_i = 20 log10(255 / sqrt(MSE_i)), MSE_i over the 3 H W values of image i (float64)
  FID      Inception pool3 features f of u8 / 255, float64 streaming moments (mean, ddof=1 cov);
           d^2 = ||mu_1 - mu_2||^2 + tr(S_1 + S_2 - 2 (S_1 S_2)^{1/2})  (pytorch-fid numerics)
  probe    multinomial logistic regression on standardised pooled latents, full-batch L-BFGS
  DCT      X = D_H z D_W^T per image and channel (orthonormal DCT-II); the RW channel adds noise
           that is white across tokens (P4: cross-token covariance is not preserved), so the mean
           spectrum E X^2 of z' differs from that of mu -- `dct2_power`, `radial_band_power`
  CIs      percentile bootstrap over images; paired over the same images for arm differences
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from .stats import GroupedMomentAccumulator

logger = logging.getLogger(__name__)

PROBE_STD_EPS = 1e-6     # linear probe: z = (x - mean) / (std + PROBE_STD_EPS)
_FID_IMAG_ATOL = 1e-3    # pytorch-fid: tolerated |Im diag sqrtm(S_1 S_2)| before raising


def _check_int(v, name: str, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


def _f64(a, name: str, ndim: int) -> np.ndarray:
    """Finite float64 numpy copy of `a` with `ndim` dimensions."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().double().numpy()
    arr = np.array(a, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"{name} has {bad} non-finite entries")
    return arr


def _check_images(x, name: str, channels: int | None = 3) -> None:
    if not isinstance(x, torch.Tensor) or x.dim() != 4 or (channels is not None and x.shape[1] != channels):
        want = f"[B, {channels}, H, W]" if channels else "[B, C, H, W]"
        raise ValueError(f"{name} must be a {want} tensor, got {tuple(getattr(x, 'shape', ())) or type(x)}")


def _check_finite(x: torch.Tensor, name: str) -> None:
    bad = int((~torch.isfinite(x)).sum())
    if bad:
        raise ValueError(f"{name} has {bad} non-finite entries")


# --------------------------------------------------------------------------- pixel metrics

def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float images [B, 3, H, W] -> uint8 [B, 3, H, W]:
    clamp(127.5 x + 128, 0, 255).to(torch.uint8), LightningDiT's formula. The cast truncates
    (it does not round); on [-1, 1] that equals rounding 127.5 (x + 1) half up. Half-precision
    inputs are upcast to float32 first (bf16 cannot hold 127.5 x + 128 exactly); non-finite
    values raise (a NaN would otherwise become an arbitrary byte)."""
    _check_images(images, "images")
    if not images.is_floating_point():
        raise ValueError(f"images must be floating point in [-1, 1], got dtype {images.dtype}")
    x = images.detach().to(torch.promote_types(images.dtype, torch.float32))
    _check_finite(x, "images")
    return torch.clamp(127.5 * x + 128.0, 0, 255).to(torch.uint8)


def psnr_per_image(ref_u8: torch.Tensor, test_u8: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB, float64 [B]: 20 log10(255 / sqrt(MSE_b)) with MSE_b the mean of
    (ref - test)^2 over all channels and pixels of image b (LightningDiT's formula).
    Identical images (MSE = 0) give +inf; callers must count and report them, not average them
    (`bootstrap_mean_ci` rejects non-finite values)."""
    for name, t in (("ref_u8", ref_u8), ("test_u8", test_u8)):
        _check_images(t, name, channels=None)
        if t.dtype != torch.uint8:
            raise ValueError(f"{name} must be uint8 (use to_uint8), got dtype {t.dtype}")
    if ref_u8.shape != test_u8.shape:
        raise ValueError(f"shape mismatch: ref_u8 {tuple(ref_u8.shape)} vs test_u8 {tuple(test_u8.shape)}")
    mse = (ref_u8.double() - test_u8.double()).square().flatten(1).mean(dim=1)
    return 20.0 * torch.log10(255.0 / torch.sqrt(mse))


class LPIPSMeter:
    """Collects per-image LPIPS distances (plan Sec. 5.1 T1).

    `lpips_model(x_hat, x)` maps two [-1, 1] batches [B, 3, H, W] to B distances
    (lpips.LPIPS returns [B, 1, 1, 1]); it is run in eval mode without autograd. Inputs are
    passed unchanged -- not clamped, as in LightningDiT's evaluate_tokenizer.py -- and must be
    on the model's device. Values are kept in update order as float64.
    """

    def __init__(self, lpips_model: nn.Module):
        self.lpips_model = lpips_model
        self._values: list[np.ndarray] = []

    @torch.no_grad()
    def update(self, x_hat: torch.Tensor, x: torch.Tensor) -> None:
        _check_images(x_hat, "x_hat")
        _check_images(x, "x")
        if x_hat.shape != x.shape:
            raise ValueError(f"shape mismatch: x_hat {tuple(x_hat.shape)} vs x {tuple(x.shape)}")
        self.lpips_model.eval()
        d = self.lpips_model(x_hat, x)
        if not isinstance(d, torch.Tensor) or d.numel() != x.shape[0]:
            raise ValueError(f"lpips_model must return one value per image ({x.shape[0]}), "
                             f"got {tuple(getattr(d, 'shape', ())) or type(d)}")
        v = d.detach().reshape(-1).double().cpu().numpy()
        if not np.isfinite(v).all():
            raise ValueError(f"lpips_model returned {int((~np.isfinite(v)).sum())} non-finite values")
        self._values.append(v)

    def values(self) -> np.ndarray:
        """All per-image values so far, float64 [n]."""
        return np.concatenate(self._values) if self._values else np.zeros(0)

    def mean(self) -> float:
        v = self.values()
        if v.size == 0:
            raise ValueError("no LPIPS values yet: call update() first")
        return float(v.mean())


# --------------------------------------------------------------------------- FID

class FIDStatistics:
    """Streaming Inception statistics for rFID / gFID (plan Sec. 5.1 T1/T2).

    `inception` maps float images [b, 3, H, W] in [0, 1] to features: a tensor or a list/tuple
    whose element 0 is used (pytorch-fid's InceptionV3 returns [pool3 [b, 2048, 1, 1]] and
    resizes to 299 internally). `update` feeds uint8 images as u8 / 255 computed on the CPU
    (bit-identical to torchvision ToTensor, as in pytorch-fid) and moved to `device`, in chunks
    of `batch_size`, in eval mode without autograd, and adds the flattened features [b, d] to
    float64 moments (stats.GroupedMomentAccumulator, one group, on `device`). `compute` returns
    (mu, sigma, n) with the unbiased covariance (ddof = 1, as np.cov in pytorch-fid). Keep n
    with the statistics: FID-10K screening and FID-50K final numbers must not be mixed (plan
    Sec. 1.3).
    """

    def __init__(self, inception: nn.Module, device: torch.device | str, batch_size: int = 100):
        self.inception = inception
        self.device = torch.device(device)
        self.batch_size = _check_int(batch_size, "batch_size", 1)
        self._acc: GroupedMomentAccumulator | None = None

    @torch.no_grad()
    def update(self, images_u8: torch.Tensor) -> None:
        """Add uint8 images [B, 3, H, W]."""
        _check_images(images_u8, "images_u8")
        if images_u8.dtype != torch.uint8:
            raise ValueError(f"images_u8 must be uint8 (use to_uint8), got dtype {images_u8.dtype}")
        model = self.inception.eval()
        for chunk in images_u8.split(self.batch_size):
            # divide on the CPU like ToTensor: CUDA multiplies by 1/255 instead (1 ulp apart)
            out = model(chunk.cpu().float().div(255.0).to(self.device))
            feats = out[0] if isinstance(out, (list, tuple)) else out
            if not isinstance(feats, torch.Tensor) or feats.dim() < 2 or feats.shape[0] != chunk.shape[0]:
                raise ValueError(f"inception must return features [{chunk.shape[0]}, ...], "
                                 f"got {tuple(getattr(feats, 'shape', ())) or type(feats)}")
            feats = feats.reshape(chunk.shape[0], -1)
            if self._acc is None:
                self._acc = GroupedMomentAccumulator(feats.shape[1], 1, self.device)
            elif feats.shape[1] != self._acc.dim:
                raise ValueError(f"feature dimension changed from {self._acc.dim} to {feats.shape[1]}")
            self._acc.update(feats, 0)

    def compute(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(mu [d], sigma [d, d] (ddof = 1), n) in float64; needs n >= 2."""
        if self._acc is None:
            raise ValueError("no images yet: call update() first")
        mom = self._acc.total()
        return mom.mean, mom.cov(ddof=1), int(mom.n)


def frechet_distance(mu1, s1, mu2, s2, eps: float = 1e-6, *, return_info: bool = False):
    """Frechet distance between N(mu1, s1) and N(mu2, s2) (float64):

        d^2 = ||mu1 - mu2||^2 + tr(s1) + tr(s2) - 2 tr((s1 s2)^{1/2}),

    with the numerics of pytorch-fid's calculate_frechet_distance (= the ADM evaluator):
    covmean = scipy.linalg.sqrtm(s1 @ s2); if it is not finite, it is recomputed from
    (s1 + eps I)(s2 + eps I) (a regularisation, logged as a warning); a complex covmean whose
    diagonal imaginary parts are all within 1e-3 is replaced by its real part, otherwise
    ValueError. pytorch-fid passes disp=False, which SciPy 1.18 removed; sqrtm(A) is the same
    computation. Unlike pytorch-fid, a covmean that is still non-finite raises. With n <= d
    samples the covariances are singular: sqrtm is then accurate only to ~1e-8 relative, and
    identical statistics can give a slightly negative d^2 (returned as is, not clipped).

    Returns d^2 as a float, or (d^2, info) if return_info, with info = {"eps_offset_applied":
    bool, "max_abs_imag": largest |Im covmean| discarded (0.0 for a real result)}.
    """
    mu1, mu2 = _f64(mu1, "mu1", 1), _f64(mu2, "mu2", 1)
    s1, s2 = _f64(s1, "s1", 2), _f64(s2, "s2", 2)
    d = mu1.shape[0]
    if mu2.shape != (d,) or s1.shape != (d, d) or s2.shape != (d, d):
        raise ValueError(f"expected mu1, mu2 [d] and s1, s2 [d, d]; "
                         f"got {mu1.shape}, {mu2.shape}, {s1.shape}, {s2.shape}")
    if not (math.isfinite(eps) and eps >= 0):
        raise ValueError(f"eps must be finite and >= 0, got {eps}")
    covmean = linalg.sqrtm(s1 @ s2)
    offset_applied = not np.isfinite(covmean).all()
    if offset_applied:
        logger.warning("frechet_distance: sqrtm(s1 s2) is not finite (singular product); "
                       "adding %g to the diagonal of both covariances", eps)
        offset = eps * np.eye(d)
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
        if not np.isfinite(covmean).all():
            raise ValueError(f"sqrtm((s1 + eps I)(s2 + eps I)) is not finite either (eps={eps:g})")
    max_imag = 0.0
    if np.iscomplexobj(covmean):
        max_imag = float(np.abs(covmean.imag).max())
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=_FID_IMAG_ATOL):
            raise ValueError(f"sqrtm(s1 s2) has an imaginary component {max_imag:.3g} "
                             f"(diagonal beyond atol {_FID_IMAG_ATOL:g})")
        covmean = covmean.real
    diff = mu1 - mu2
    fd = float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    if return_info:
        return fd, {"eps_offset_applied": bool(offset_applied), "max_abs_imag": max_imag}
    return fd


def save_fid_stats(path: str | Path, mu, sigma, n: int, meta: dict) -> None:
    """Write FID statistics to an .npz readable with allow_pickle=False: mu [d], sigma [d, d]
    (float64), n (int64) and meta (JSON string). The mu / sigma keys match pytorch-fid and ADM
    reference files."""
    mu = _f64(mu, "mu", 1)
    sigma = _f64(sigma, "sigma", 2)
    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError(f"sigma must be [{mu.shape[0]}, {mu.shape[0]}] to match mu, got {sigma.shape}")
    n = _check_int(n, "n", 2)
    if not isinstance(meta, dict):
        raise ValueError(f"meta must be a dict, got {type(meta).__name__}")
    try:
        meta_json = json.dumps(meta, sort_keys=True)
    except TypeError as e:
        raise ValueError(f"meta must be JSON-serializable: {e}") from None
    with open(path, "wb") as f:  # a file object keeps np.savez from appending ".npz"
        np.savez(f, mu=mu, sigma=sigma, n=np.int64(n), meta=np.array(meta_json))


def load_fid_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Read (mu, sigma, n, meta) written by `save_fid_stats`. External reference files that
    hold mu and sigma but neither n nor meta (pytorch-fid stats, ADM's
    VIRTUAL_imagenet256_labeled.npz; extra keys such as arr_0 / mu_s / sigma_s are ignored)
    give n = -1 and meta = {"source": "external"}."""
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        if not {"mu", "sigma"} <= keys:
            raise ValueError(f"{path}: expected keys 'mu' and 'sigma', found {sorted(keys)}")
        mu, sigma = _f64(f["mu"], "mu", 1), _f64(f["sigma"], "sigma", 2)
        if {"n", "meta"} <= keys:
            n, meta = int(f["n"]), json.loads(f["meta"].item())
        elif not ({"n", "meta"} & keys):
            n, meta = -1, {"source": "external"}
        else:
            raise ValueError(f"{path}: has only one of 'n' and 'meta' ({sorted(keys)}); file is malformed")
    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError(f"{path}: sigma {sigma.shape} does not match mu {mu.shape}")
    return mu, sigma, n, meta


# --------------------------------------------------------------------------- linear probe

def pooled_features(z: torch.Tensor, grid: int = 1) -> torch.Tensor:
    """Probe features of latents z [B, C, H, W]: adaptive average pooling to grid x grid cells,
    flattened channel-major -> [B, C * grid^2] (grid = 1: per-channel spatial mean).
    Half-precision inputs are pooled in float32."""
    _check_images(z, "z", channels=None)
    if not z.is_floating_point():
        raise ValueError(f"z must be floating point, got dtype {z.dtype}")
    g = _check_int(grid, "grid", 1)
    if g > min(z.shape[2], z.shape[3]):
        raise ValueError(f"grid={g} exceeds the spatial size {tuple(z.shape[2:])}")
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    return F.adaptive_avg_pool2d(z, g).flatten(1)


def _probe_features(x, name: str, device: torch.device) -> torch.Tensor:
    t = x.detach() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    if t.dim() != 2 or t.shape[0] == 0 or not t.is_floating_point():
        raise ValueError(f"{name} must be a non-empty floating [n, d] array, got {t.dtype} {tuple(t.shape)}")
    t = t.to(device=device, dtype=torch.float64)
    _check_finite(t, name)
    return t


def _probe_labels(y, n: int, num_classes: int, name: str, device: torch.device) -> torch.Tensor:
    t = y.detach() if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y))
    if t.dim() != 1 or t.shape[0] != n:
        raise ValueError(f"{name} must be 1-D with {n} entries, got shape {tuple(t.shape)}")
    if t.is_floating_point() or t.is_complex() or t.dtype == torch.bool:
        raise ValueError(f"{name} must hold integer class ids, got dtype {t.dtype}")
    lo, hi = int(t.min()), int(t.max())
    if lo < 0 or hi >= num_classes:
        raise ValueError(f"{name} must lie in [0, {num_classes}), got range [{lo}, {hi}]")
    return t.to(device=device, dtype=torch.int64)


def linear_probe(train_x, train_y, test_x, test_y, *, num_classes: int, weight_decay: float = 1e-4,
                 max_iter: int = 200, device: torch.device | str = "cpu", seed: int = 0) -> dict:
    """Linear-probe accuracy of fixed features, e.g. `pooled_features` of latents (plan Sec. 5.1 T1).

    Protocol: standardise with the training mean and std (float64, ddof = 0):
    z = (x - mean) / (std + PROBE_STD_EPS); fit logits = z W^T + b on the full training set by
    minimising  mean_i CE(logits_i, y_i) + 0.5 * weight_decay * ||W||_F^2  (bias not penalised)
    in float32 on `device`, from W = 0, b = 0, with one torch.optim.LBFGS step: lr = 1,
    line_search_fn = "strong_wolfe", max_iter iterations at most, max_eval = 5/4 max_iter
    (torch default), history_size = 100, tolerance_grad = 1e-7, tolerance_change = 1e-9.
    The objective is convex (strictly in W for weight_decay > 0) and nothing is sampled, so the
    fit is deterministic; `seed` is part of the contract signature but no random draw uses it.
    Memory: the logits are [n_train, num_classes] float32.

    Returns {"top1", "top5" (None if num_classes < 5), "train_top1", "correct" (bool [n_test],
    top-1 hit per test item, for `paired_bootstrap_diff`), "n_train", "n_test", "n_iter"
    (L-BFGS iterations used; == max_iter means the cap was hit), "min_train_std" (smallest
    feature std before the eps; 0 flags a constant feature)}.
    """
    K = _check_int(num_classes, "num_classes", 2)
    max_iter = _check_int(max_iter, "max_iter", 1)
    _check_int(seed, "seed", 0)
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError(f"weight_decay must be finite and >= 0, got {weight_decay}")
    dev = torch.device(device)
    Xtr, Xte = _probe_features(train_x, "train_x", dev), _probe_features(test_x, "test_x", dev)
    n, d = Xtr.shape
    if n < 2 or Xte.shape[1] != d:
        raise ValueError(f"need n_train >= 2 and equal feature dims; "
                         f"got train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")
    ytr = _probe_labels(train_y, n, K, "train_y", dev)
    yte = _probe_labels(test_y, Xte.shape[0], K, "test_y", dev)

    mean, std = Xtr.mean(dim=0), Xtr.std(dim=0, correction=0)
    Ztr = ((Xtr - mean) / (std + PROBE_STD_EPS)).float()
    Zte = ((Xte - mean) / (std + PROBE_STD_EPS)).float()

    W = torch.zeros(K, d, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=100, tolerance_grad=1e-7,
                            tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(Ztr @ W.T + b, ytr) + 0.5 * weight_decay * W.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits = Zte @ W.T + b
        correct = logits.argmax(dim=1) == yte
        top5 = None if K < 5 else float((logits.topk(5, dim=1).indices == yte[:, None]).any(dim=1).double().mean())
        train_top1 = float(((Ztr @ W.T + b).argmax(dim=1) == ytr).double().mean())
    return {"top1": float(correct.double().mean()), "top5": top5, "train_top1": train_top1,
            "correct": correct.cpu().numpy(), "n_train": n, "n_test": int(Xte.shape[0]),
            "n_iter": int(opt.state[W]["n_iter"]), "min_train_std": float(std.min())}


# --------------------------------------------------------------------------- DCT spectra

def dct_matrix(N: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix [N, N] (float64): D[k, n] = a_k cos(pi (2n + 1) k / (2N)),
    a_0 = sqrt(1/N), a_k = sqrt(2/N) for k >= 1. D D^T = I, and D x equals
    scipy.fft.dct(x, type=2, norm="ortho")."""
    N = _check_int(N, "N", 1)
    k = torch.arange(N, dtype=torch.float64)[:, None]
    n = torch.arange(N, dtype=torch.float64)[None, :]
    D = math.sqrt(2.0 / N) * torch.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * N))
    D[0] = math.sqrt(1.0 / N)
    return D


def dct2_power(z: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    """Mean 2-D DCT power spectrum of latents z [B, C, H, W] (float64, on z's device).

    Coefficients X_bc = D_H z_bc D_W^T per image b and channel c (`dct_matrix`); returns
    mean_b X_bc^2 -> [C, H, W] if per_channel, else also averaged over c -> [H, W]. Index
    [kh, kw] = vertical / horizontal frequency. This is the mean of the spectra, not the
    spectrum of the mean latent. By Parseval each spectrum sums to ||z_bc||^2; z is not
    centred, so X_bc[0, 0] = sqrt(H W) * (spatial mean of z_bc)."""
    _check_images(z, "z", channels=None)
    if not z.is_floating_point() or z.shape[0] == 0:
        raise ValueError(f"z must be a non-empty floating tensor, got {z.dtype} {tuple(z.shape)}")
    z64 = z.detach().to(torch.float64)
    _check_finite(z64, "z")
    Dh, Dw = dct_matrix(z.shape[2]).to(z.device), dct_matrix(z.shape[3]).to(z.device)
    power = torch.einsum("kh,bchw,lw->bckl", Dh, z64, Dw).square().mean(dim=0)
    return power if per_channel else power.mean(dim=0)


def radial_band_power(power_hw, n_bands: int) -> np.ndarray:
    """Mean of a DCT power map [H, W] over radial frequency bands, float64 [n_bands].

    Cell (kh, kw) has radius r = sqrt(kh^2 + kw^2) on the DCT index grid. [0, r_max],
    r_max = sqrt((H-1)^2 + (W-1)^2), is cut into n_bands equal-width bands [e_i, e_{i+1}),
    e_i = i r_max / n_bands, the last band closed at r_max. Membership is decided exactly in
    integers (n_bands^2 r^2 >= i^2 r_max^2), so cells on a band edge are assigned identically
    everywhere. A band without any cell raises ValueError (use fewer bands).
    """
    p = _f64(power_hw, "power_hw", 2)
    nb = _check_int(n_bands, "n_bands", 1)
    H, W = p.shape
    kh, kw = np.meshgrid(np.arange(H, dtype=np.int64), np.arange(W, dtype=np.int64), indexing="ij")
    r2 = kh ** 2 + kw ** 2
    edges2 = np.arange(1, nb, dtype=np.int64) ** 2 * ((H - 1) ** 2 + (W - 1) ** 2)   # (i r_max)^2
    band = (nb ** 2 * r2[..., None] >= edges2).sum(axis=-1).ravel()
    counts = np.bincount(band, minlength=nb)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(f"bands {empty.tolist()} of {nb} contain no cell of the {H}x{W} DCT grid; use fewer bands")
    return np.bincount(band, weights=p.ravel(), minlength=nb) / counts


# --------------------------------------------------------------------------- bootstrap

def _check_sample(values, name: str) -> np.ndarray:
    v = _f64(values, name, 1)
    if v.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 items, got {v.shape[0]}")
    return v


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of per-item values (images are the independent unit,
    plan Sec. 1.2). Each of n_boot resamples draws n items with replacement from
    np.random.default_rng(seed); lo / hi are the alpha/2 and 1 - alpha/2 quantiles (numpy
    "linear" method) of the resampled means. Values must be finite (e.g. drop and report PSNR =
    inf first). Returns {"mean" (sample mean), "lo", "hi", "n", "n_boot", "alpha"}."""
    v = _check_sample(values, "values")
    n_boot = _check_int(n_boot, "n_boot", 1)
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(_check_int(seed, "seed", 0))
    n = v.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = v[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": n, "n_boot": n_boot, "alpha": alpha}


def paired_bootstrap_diff(a, b, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired percentile bootstrap CI for mean(a - b), where a[i] and b[i] are two arms' values on
    the same item i (same image, same noise seed). Resampling items keeps each pair together, so
    this is bootstrap_mean_ci(a - b, ...); "mean" is mean(a) - mean(b). Boolean arrays (e.g.
    linear_probe "correct") give the top-1 accuracy difference."""
    a, b = _check_sample(a, "a"), _check_sample(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b must be paired (same length), got {a.shape[0]} and {b.shape[0]}")
    return bootstrap_mean_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)
