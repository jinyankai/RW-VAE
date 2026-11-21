"""RW-VAE v2: streaming float64 token statistics, LatentStats and stability tools.

Plan v2.1 Sec. 1.2-1.3 and 3.2-3.3; contract: docs/ARCHITECTURE_V2.md Sec. 3.4.

A token is the C-vector z0 at one position of a [B, C, H, W] latent, flattened as
x.permute(0, 2, 3, 1).reshape(-1, C) (`tokens`, row-major (h, w)). The DINO patch map
y [B, D, H, W] is flattened the same way, so z0 and y tokens stay paired by position.
For the joint token x = [z0; y] we keep float64 centered moments

    n,    mean = (1/n) sum_i x_i,    m2 = sum_i (x_i - mean)(x_i - mean)^T

and merge partial results with the exact pairwise update of Chan, Golub & LeVeque
(1979). Raw second moments are never formed, so float16/32 inputs with a large common
offset stay accurate. `LatentStats.from_joint` splits the joint covariance m2 / (n - 1)
into Sigma = Cov z0, Sigma_zy = Cov(z0, y) and Sigma_y = Cov y; `with_explained` adds
Sigma_expl (P1, `rwvae.relevance.explained_covariance`).

Images are the independent sampling unit (plan Sec. 1.2): moments are kept per group,
an image's group is a stable hash of its ID (`image_group_id`), and all tokens of an
image go to that group. Groups give split-half estimates (`split_half_groups`) and a
group-level (= image-level block) bootstrap (`bootstrap_group_weights` together with
`GroupedMomentAccumulator.subset`) for the stability checks of plan Sec. 3.3 item 3.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Iterable, Sequence

import numpy as np
import torch

from .relevance import explained_covariance


def _check_int(v, name: str, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


def _as_f64(a, name: str, shape: tuple) -> np.ndarray:
    """Finite float64 copy of `a` with the given shape (None = any size)."""
    arr = np.array(a, dtype=np.float64)
    if arr.ndim != len(shape) or any(s is not None and s != t for s, t in zip(shape, arr.shape)):
        raise ValueError(f"{name} must have shape {shape} (None = any size), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} has {int((~np.isfinite(arr)).sum())} non-finite entries")
    return arr


# ---------------------------------------------------------------------------------------
# Moments (numpy float64)
# ---------------------------------------------------------------------------------------
@dataclass(eq=False)
class Moments:
    """Centered sufficient statistics of a sample of d-vectors (float64 numpy).

    n: sample size (a float, so replication-weighted combinations stay exact);
    mean [d]; m2 [d, d] = sum_i (x_i - mean)(x_i - mean)^T. Empty sample: n = 0, zeros.
    """

    n: float
    mean: np.ndarray
    m2: np.ndarray

    def __post_init__(self) -> None:
        self.n = float(self.n)
        if not (np.isfinite(self.n) and self.n >= 0):
            raise ValueError(f"n must be finite and >= 0, got {self.n}")
        self.mean = _as_f64(self.mean, "mean", (None,))
        d = self.mean.shape[0]
        self.m2 = _as_f64(self.m2, "m2", (d, d))

    def merge(self, other: Moments) -> Moments:
        """Exact pairwise merge (Chan, Golub & LeVeque 1979):
        n = n_a + n_b, delta = mean_b - mean_a, mean = mean_a + delta n_b / n,
        m2 = m2_a + m2_b + delta delta^T n_a n_b / n."""
        _check_same_dim(self, other)
        n = self.n + other.n
        if n == 0:
            return Moments(0.0, np.zeros_like(self.mean), np.zeros_like(self.m2))
        delta = other.mean - self.mean
        return Moments(n, self.mean + delta * (other.n / n),
                       self.m2 + other.m2 + np.outer(delta, delta) * (self.n * other.n / n))

    def cov(self, ddof: int = 1) -> np.ndarray:
        """Sample covariance m2 / (n - ddof)."""
        if not self.n - ddof > 0:
            raise ValueError(f"a covariance needs n > ddof, got n={self.n:g}, ddof={ddof}")
        return self.m2 / (self.n - ddof)

    @staticmethod
    def combine(items: Sequence[Moments], weights: Sequence[float] | None = None) -> Moments:
        """Weighted combination; weight k_g means "sample g replicated k_g times" (integer
        multinomial counts give one bootstrap resample):
            n = sum_g k_g n_g,   mean = sum_g k_g n_g mean_g / n,
            m2 = sum_g k_g [m2_g + n_g (mean_g - mean)(mean_g - mean)^T]."""
        items = list(items)
        if not items:
            raise ValueError("combine needs at least one Moments")
        for it in items[1:]:
            _check_same_dim(items[0], it)
        k = np.ones(len(items)) if weights is None else weights
        return _combine(np.array([it.n for it in items]), np.stack([it.mean for it in items]),
                        np.stack([it.m2 for it in items]), k)


def _check_same_dim(a: Moments, b: Moments) -> None:
    if a.mean.shape != b.mean.shape:
        raise ValueError(f"dimension mismatch: {a.mean.shape[0]} vs {b.mean.shape[0]}")


def _combine(n: np.ndarray, mean: np.ndarray, m2: np.ndarray, weights) -> Moments:
    """`Moments.combine` on stacked arrays n [G], mean [G, d], m2 [G, d, d]."""
    k = np.asarray(weights, dtype=np.float64)
    if k.shape != n.shape or not np.isfinite(k).all() or (k < 0).any():
        raise ValueError(f"weights must be {n.shape[0]} finite non-negative numbers, got {k.tolist()}")
    nk = k * n
    total = float(nk.sum())
    if total == 0:
        return Moments(0.0, np.zeros(mean.shape[1]), np.zeros(m2.shape[1:]))
    mu = nk @ mean / total
    dev = mean - mu
    out = np.tensordot(k, m2, axes=1) + (dev.T * nk) @ dev
    return Moments(total, mu, 0.5 * (out + out.T))  # symmetrize matmul rounding only


# ---------------------------------------------------------------------------------------
# Grouped streaming accumulator (torch float64)
# ---------------------------------------------------------------------------------------
def _chan_merge(n_a, mean_a, m2_a, n_b, mean_b, m2_b):
    """Chan et al. merge on torch tensors n [...], mean [..., d], m2 [..., d, d] (leading
    group axes broadcast); a group with n_a = n_b = 0 stays empty."""
    n = n_a + n_b
    frac_b = torch.where(n > 0, n_b / n, torch.zeros_like(n))
    delta = mean_b - mean_a
    mean = mean_a + delta * frac_b[..., None]
    m2 = m2_a + m2_b + delta[..., :, None] * delta[..., None, :] * (n_a * frac_b)[..., None, None]
    return n, mean, m2


def _to_tensor(v, dtype: torch.dtype, shape: tuple, name: str) -> torch.Tensor:
    t = v.detach().cpu().to(dtype).clone() if isinstance(v, torch.Tensor) else torch.tensor(np.asarray(v), dtype=dtype)
    if tuple(t.shape) != shape:
        raise ValueError(f"state entry {name} must have shape {shape}, got {tuple(t.shape)}")
    return t


class GroupedMomentAccumulator:
    """Streaming float64 moments of d-dim rows, kept separately for G groups.

    Storage on `device`: n [G], mean [G, d], m2 [G, d, d] (float64), plus
    images_per_group [G] (int64, CPU) maintained by `add_images`. `update` computes each
    group's batch moments in float64 (two-pass: batch mean, then centered scatter) and
    Chan-merges them into the running state, so results do not depend on batching (up to
    rounding) and low-precision inputs with a large offset stay accurate.
    """

    def __init__(self, dim: int, num_groups: int = 1, device: torch.device | str = "cpu"):
        self.dim = _check_int(dim, "dim", 1)
        self.num_groups = _check_int(num_groups, "num_groups", 1)
        self.device = torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.device)
        self.n = torch.zeros(self.num_groups, **f64)
        self.mean = torch.zeros(self.num_groups, self.dim, **f64)
        self.m2 = torch.zeros(self.num_groups, self.dim, self.dim, **f64)
        self.images_per_group = torch.zeros(self.num_groups, dtype=torch.int64)

    @property
    def n_images(self) -> int:
        return int(self.images_per_group.sum())

    def _check_group(self, g) -> int:
        if isinstance(g, bool) or not isinstance(g, (int, np.integer)) or not 0 <= g < self.num_groups:
            raise ValueError(f"group id must be an integer in [0, {self.num_groups}), got {g!r}")
        return int(g)

    def _group_ids(self, group_ids, n: int | None = None) -> torch.Tensor:
        """Validated int64 ids on self.device; an int means "all n rows are in that group"."""
        if isinstance(group_ids, (int, np.integer)) and not isinstance(group_ids, bool):
            return torch.full((n,), self._check_group(group_ids), dtype=torch.int64, device=self.device)
        gid = torch.as_tensor(group_ids)
        if gid.dim() != 1 or (n is not None and gid.shape[0] != n):
            raise ValueError(f"group_ids must be 1-D with {n} entries, got shape {tuple(gid.shape)}")
        if gid.numel() == 0:
            gid = gid.to(torch.int64)
        if gid.is_floating_point() or gid.is_complex() or gid.dtype == torch.bool:
            raise ValueError(f"group_ids must be integers, got dtype {gid.dtype}")
        if gid.numel() and (int(gid.min()) < 0 or int(gid.max()) >= self.num_groups):
            raise ValueError(f"group ids must lie in [0, {self.num_groups}), "
                             f"got range [{int(gid.min())}, {int(gid.max())}]")
        return gid.to(device=self.device, dtype=torch.int64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, group_ids: torch.Tensor | int) -> None:
        """Add rows x [N, dim] (any floating dtype); group_ids is an int (all rows) or [N] ints."""
        if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be a [N, {self.dim}] tensor, got {getattr(x, 'shape', type(x))}")
        if not x.is_floating_point():
            raise ValueError(f"x must be floating point, got dtype {x.dtype}")
        gid = self._group_ids(group_ids, x.shape[0])
        if x.shape[0] == 0:
            return
        x = x.detach().to(device=self.device, dtype=torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError(f"x has {int((~torch.isfinite(x)).sum())} non-finite entries")
        counts = torch.bincount(gid, minlength=self.num_groups).tolist()
        rows = x[torch.argsort(gid, stable=True)]
        for g, xg in enumerate(torch.split(rows, counts)):
            if xg.shape[0] == 0:
                continue
            mean_b = xg.mean(dim=0)
            xc = xg - mean_b
            m2_b = xc.T @ xc
            self.n[g], self.mean[g], self.m2[g] = _chan_merge(
                self.n[g], self.mean[g], self.m2[g], float(xg.shape[0]), mean_b, 0.5 * (m2_b + m2_b.T))

    def add_images(self, group_ids: torch.Tensor | Sequence[int] | int) -> None:
        """Count images: group_ids [B] holds one group id per image. For a single-group
        accumulator an int k is accepted as "k images" (contract: add_images(k))."""
        if isinstance(group_ids, (int, np.integer)) and not isinstance(group_ids, bool):
            if self.num_groups != 1:
                raise ValueError(f"with num_groups={self.num_groups} pass one group id per image, "
                                 f"got the int {group_ids}")
            self.images_per_group[0] += _check_int(group_ids, "image count", 0)
            return
        gid = self._group_ids(group_ids).cpu()
        self.images_per_group += torch.bincount(gid, minlength=self.num_groups)

    def group(self, g: int) -> Moments:
        """Moments of group g (exact copy of the stored state)."""
        g = self._check_group(g)
        return Moments(float(self.n[g]), self.mean[g].cpu().numpy(), self.m2[g].cpu().numpy())

    def subset(self, groups: Sequence[int], weights: Sequence[float] | None = None) -> Moments:
        """`Moments.combine` of the listed groups; weights[i] replicates groups[i]."""
        idx = [self._check_group(g) for g in groups]
        if not idx:
            raise ValueError("subset needs at least one group")
        sel = torch.as_tensor(idx, device=self.device)
        k = np.ones(len(idx)) if weights is None else weights
        return _combine(self.n[sel].cpu().numpy(), self.mean[sel].cpu().numpy(), self.m2[sel].cpu().numpy(), k)

    def total(self) -> Moments:
        """Moments of all rows."""
        return self.subset(range(self.num_groups))

    def state_dict(self) -> dict:
        """CPU copy of the state (tensors + ints), e.g. for torch.save or np.savez."""
        return {"dim": self.dim, "num_groups": self.num_groups, "n": self.n.cpu().clone(),
                "mean": self.mean.cpu().clone(), "m2": self.m2.cpu().clone(),
                "images_per_group": self.images_per_group.clone()}

    def load_state_dict(self, sd: dict) -> None:
        """Load a `state_dict` (tensors or numpy arrays, e.g. read back from an .npz)."""
        missing = {"dim", "num_groups", "n", "mean", "m2", "images_per_group"} - set(sd)
        if missing:
            raise ValueError(f"state dict is missing {sorted(missing)}")
        if (int(sd["dim"]), int(sd["num_groups"])) != (self.dim, self.num_groups):
            raise ValueError(f"state dict has dim={int(sd['dim'])}, num_groups={int(sd['num_groups'])}; "
                             f"this accumulator has dim={self.dim}, num_groups={self.num_groups}")
        G, d = self.num_groups, self.dim
        n = _to_tensor(sd["n"], torch.float64, (G,), "n")
        mean = _to_tensor(sd["mean"], torch.float64, (G, d), "mean")
        m2 = _to_tensor(sd["m2"], torch.float64, (G, d, d), "m2")
        images = _to_tensor(sd["images_per_group"], torch.int64, (G,), "images_per_group")
        if not all(bool(torch.isfinite(t).all()) for t in (n, mean, m2)) or (n < 0).any() or (images < 0).any():
            raise ValueError("state dict has non-finite moments or negative counts")
        self.n, self.mean, self.m2 = n.to(self.device), mean.to(self.device), m2.to(self.device)
        self.images_per_group = images

    def merge_(self, other: GroupedMomentAccumulator) -> None:
        """In-place Chan merge of another accumulator (other process / shard), group by group."""
        if not isinstance(other, GroupedMomentAccumulator) or (other.dim, other.num_groups) != (self.dim, self.num_groups):
            raise ValueError(f"can only merge an accumulator with dim={self.dim}, num_groups={self.num_groups}")
        o = (other.n.to(self.device), other.mean.to(self.device), other.m2.to(self.device))
        self.n, self.mean, self.m2 = _chan_merge(self.n, self.mean, self.m2, *o)
        self.images_per_group = self.images_per_group + other.images_per_group


def image_group_id(image_id: str, num_groups: int, salt: str = "rwvae-v2") -> int:
    """Stable group of an image: int.from_bytes(sha1(salt + "\\0" + image_id)[:8], "big") % num_groups.

    Independent of platform, process and PYTHONHASHSEED (unlike the built-in hash())."""
    if not isinstance(image_id, str) or not isinstance(salt, str):
        raise ValueError(f"image_id and salt must be str, got {type(image_id).__name__}, {type(salt).__name__}")
    G = _check_int(num_groups, "num_groups", 1)
    digest = hashlib.sha1(f"{salt}\0{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % G


def tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B*H*W, C], row-major (h, w): x.permute(0, 2, 3, 1).reshape(-1, C)."""
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        raise ValueError(f"expected a [B, C, H, W] tensor, got {getattr(x, 'shape', type(x))}")
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


# ---------------------------------------------------------------------------------------
# LatentStats
# ---------------------------------------------------------------------------------------
@dataclass(eq=False)
class LatentStats:
    """Token statistics of the frozen-tokenizer latent z0 and its paired DINO patch y.

    m = E z0 [C]; Sigma = Cov z0 [C, C]; Sigma_zy = Cov(z0, y) [C, D]; Sigma_y = Cov y [D, D]
    (both None without y); Sigma_expl = Sigma_zy (Sigma_y + lambda I)^{-1} Sigma_yz [C, C]
    (P1, set by `with_explained`). `from_joint` takes all blocks from one joint sample
    covariance, which is PSD, so 0 <= Sigma_expl <= Sigma holds for the sample (P1).
    """

    m: np.ndarray
    Sigma: np.ndarray
    Sigma_zy: np.ndarray | None
    Sigma_y: np.ndarray | None
    Sigma_expl: np.ndarray | None
    n_tokens: int
    n_images: int
    meta: dict = field(default_factory=dict)

    _ARRAYS: ClassVar[tuple[str, ...]] = ("m", "Sigma", "Sigma_zy", "Sigma_y", "Sigma_expl")

    def __post_init__(self) -> None:
        self.m = _as_f64(self.m, "m", (None,))
        C = self.m.shape[0]
        self.Sigma = _as_f64(self.Sigma, "Sigma", (C, C))
        if (self.Sigma_zy is None) != (self.Sigma_y is None):
            raise ValueError("Sigma_zy and Sigma_y must both be given or both be None")
        if self.Sigma_y is not None:
            self.Sigma_y = _as_f64(self.Sigma_y, "Sigma_y", (None, None))
            D = self.Sigma_y.shape[0]
            if self.Sigma_y.shape != (D, D):
                raise ValueError(f"Sigma_y must be square, got shape {self.Sigma_y.shape}")
            self.Sigma_zy = _as_f64(self.Sigma_zy, "Sigma_zy", (C, D))
        if self.Sigma_expl is not None:
            self.Sigma_expl = _as_f64(self.Sigma_expl, "Sigma_expl", (C, C))
        self.n_tokens = _check_int(self.n_tokens, "n_tokens", 0)
        self.n_images = _check_int(self.n_images, "n_images", 0)
        self.meta = dict(self.meta)

    @property
    def C(self) -> int:
        return self.m.shape[0]

    @property
    def D(self) -> int:
        return 0 if self.Sigma_y is None else self.Sigma_y.shape[0]

    @classmethod
    def from_joint(cls, mom: Moments, C: int, n_images: int, meta: dict | None = None) -> LatentStats:
        """Split joint token moments of x = [z0; y] (dim C + D, D >= 0) into blocks of the
        unbiased joint covariance m2 / (n - 1); meta["cov_ddof"] = 1 is recorded."""
        d = mom.mean.shape[0]
        C = _check_int(C, "C", 1)
        if C > d:
            raise ValueError(f"C={C} exceeds the joint dimension {d}")
        if not mom.n.is_integer():
            raise ValueError(f"token count must be an integer, got n={mom.n}")
        cov = mom.cov(ddof=1)
        has_y = C < d
        return cls(m=mom.mean[:C], Sigma=cov[:C, :C],
                   Sigma_zy=cov[:C, C:] if has_y else None, Sigma_y=cov[C:, C:] if has_y else None,
                   Sigma_expl=None, n_tokens=int(mom.n), n_images=n_images,
                   meta={**(meta or {}), "cov_ddof": 1})

    def with_explained(self, ridge_rel: float) -> LatentStats:
        """Copy with Sigma_expl = Sigma_zy (Sigma_y + lambda I)^{-1} Sigma_yz,
        lambda = ridge_rel tr(Sigma_y) / D (P1); the ridge info dict goes to meta["ridge"]."""
        if self.Sigma_y is None:
            raise ValueError("these statistics have no y block; Sigma_expl needs Sigma_zy and Sigma_y")
        S, info = explained_covariance(self.Sigma_zy, self.Sigma_y, ridge_rel)
        return replace(self, Sigma_expl=S, meta={**self.meta, "ridge": info})

    def channel_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """(m, sqrt(diag Sigma)): the per-channel DiT normalization shared by all arms."""
        var = np.diag(self.Sigma)
        bad = np.flatnonzero(var <= 0)
        if bad.size:
            raise ValueError(f"channels {bad.tolist()} have non-positive variance {var[bad].tolist()}")
        return self.m.copy(), np.sqrt(var)

    def fingerprint(self) -> str:
        """sha256 over the arrays (name, shape, little-endian float64 bytes; absent arrays are
        marked) and the two counts; meta is excluded. First 16 hex digits."""
        h = hashlib.sha256()
        for name in self._ARRAYS:
            a = getattr(self, name)
            h.update(f"{name}:{None if a is None else a.shape};".encode())
            if a is not None:
                h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
        h.update(f"n_tokens={self.n_tokens};n_images={self.n_images}".encode())
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        """Write an .npz readable with allow_pickle=False; meta is stored as a JSON string,
        and the fingerprint is stored and re-checked by `load`."""
        try:
            meta_json = json.dumps(self.meta, sort_keys=True)
        except TypeError as e:
            raise ValueError(f"meta must be JSON-serializable: {e}") from None
        arrays = {k: getattr(self, k) for k in self._ARRAYS if getattr(self, k) is not None}
        with open(path, "wb") as f:  # a file object keeps np.savez from renaming the path
            np.savez(f, **arrays, n_tokens=np.int64(self.n_tokens), n_images=np.int64(self.n_images),
                     meta=np.array(meta_json), fingerprint=np.array(self.fingerprint()))

    @classmethod
    def load(cls, path: str | Path) -> LatentStats:
        with np.load(path, allow_pickle=False) as f:
            missing = {"m", "Sigma", "n_tokens", "n_images", "meta"} - set(f.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            stats = cls(**{k: (f[k] if k in f.files else None) for k in cls._ARRAYS},
                        n_tokens=int(f["n_tokens"]), n_images=int(f["n_images"]),
                        meta=json.loads(f["meta"].item()))
            stored = f["fingerprint"].item() if "fingerprint" in f.files else None
        if stored is not None and stored != stats.fingerprint():
            raise ValueError(f"{path}: stored fingerprint {stored} != recomputed {stats.fingerprint()}")
        return stats


@torch.no_grad()
def stream_stats(batches: Iterable[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | Sequence[int]]],
                 C: int, D: int, num_groups: int = 16,
                 device: torch.device | str = "cpu") -> GroupedMomentAccumulator:
    """Joint token moments of x = [z0; y] (plan Sec. 3.2 `stream_stats`).

    `batches` yields (mu [B, C, H, W], y [B, D, H, W] or None, group_ids [B]); every token
    of image b goes to group group_ids[b] (get ids from `image_group_id`). With D = 0, y
    must be None and only z0 moments are kept. Returns the accumulator (dim C + D); the
    statistics are `LatentStats.from_joint(acc.total(), C, acc.n_images, meta)`.
    """
    C = _check_int(C, "C", 1)
    D = _check_int(D, "D", 0)
    acc = GroupedMomentAccumulator(C + D, num_groups, device)
    for mu, y, group_ids in batches:
        if not isinstance(mu, torch.Tensor) or mu.dim() != 4 or mu.shape[1] != C:
            raise ValueError(f"mu must be a [B, {C}, H, W] tensor, got {getattr(mu, 'shape', type(mu))}")
        B, _, H, W = mu.shape
        parts = [tokens(mu).to(device=acc.device, dtype=torch.float64)]
        if D == 0 and y is not None:
            raise ValueError("D=0 but the batch has a y tensor")
        if D > 0:
            if not isinstance(y, torch.Tensor) or tuple(y.shape) != (B, D, H, W):
                raise ValueError(f"y must be a {(B, D, H, W)} tensor paired with mu, got {getattr(y, 'shape', y)}")
            parts.append(tokens(y).to(device=acc.device, dtype=torch.float64))
        gid = torch.as_tensor(group_ids)
        if tuple(gid.shape) != (B,):
            raise ValueError(f"group_ids must have shape ({B},) (one id per image), got {tuple(gid.shape)}")
        acc.update(torch.cat(parts, dim=1), gid.repeat_interleave(H * W))
        acc.add_images(gid)
    return acc


# ---------------------------------------------------------------------------------------
# Spectrum and stability diagnostics
# ---------------------------------------------------------------------------------------
def spectrum_report(Sigma) -> dict:
    """Eigen-spectrum diagnostics of a symmetric PSD matrix (plan Sec. 3.3).

    eig: eigenvalues, descending; min/max; cond = max/min (inf if min <= 0);
    eff_rank = exp(-sum p_i ln p_i), p_i = e_i / sum e over the positive eigenvalues
    (Roy & Vetterli 2007); n_nonpositive = number of eigenvalues <= 0, which eff_rank
    excludes (reported, not hidden). Asymmetry beyond 1e-8 (relative) raises.
    """
    S = _as_f64(Sigma, "Sigma", (None, None))
    if S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError(f"Sigma must be a non-empty square matrix, got shape {S.shape}")
    asym = np.linalg.norm(S - S.T) / max(np.linalg.norm(S), np.finfo(float).tiny)
    if asym > 1e-8:
        raise ValueError(f"Sigma is not symmetric: relative asymmetry {asym:.3g} > 1e-8")
    eig = np.linalg.eigvalsh(0.5 * (S + S.T))[::-1].copy()
    pos = eig[eig > 0]
    p = pos / pos.sum() if pos.size else pos
    return {"eig": eig, "min": float(eig[-1]), "max": float(eig[0]),
            "cond": float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf"),
            "eff_rank": float(np.exp(-np.sum(p * np.log(p)))) if pos.size else 0.0,
            "n_nonpositive": int(eig.size - pos.size)}


def principal_angles(U, V) -> np.ndarray:
    """Principal angles (radians, ascending) between span(U) and span(V).

    U [n, p], V [n, q] must have orthonormal columns; returns min(p, q) angles. With
    p >= q: cos(theta) = svd(U^T V) and sin(theta) = svd(V - U U^T V). Angles with
    sin <= 1/sqrt(2) use arcsin, the rest arccos, which keeps small and large angles
    accurate (Knyazev & Argentati 2002); no argument ever needs clipping.
    """
    U = _as_f64(U, "U", (None, None))
    V = _as_f64(V, "V", (U.shape[0], None))
    for name, X in (("U", U), ("V", V)):
        if X.shape[1] == 0:
            raise ValueError(f"{name} needs at least one column")
        err = float(np.abs(X.T @ X - np.eye(X.shape[1])).max())
        if err > 1e-8:
            raise ValueError(f"{name} must have orthonormal columns: max |X^T X - I| = {err:.3g}")
    if U.shape[1] < V.shape[1]:
        U, V = V, U
    UtV = U.T @ V
    cos = np.linalg.svd(UtV, compute_uv=False)                  # descending
    sin = np.linalg.svd(V - U @ UtV, compute_uv=False)[::-1]    # ascending
    small = sin <= np.sqrt(0.5)
    theta = np.empty_like(sin)
    theta[small] = np.arcsin(sin[small])
    theta[~small] = np.arccos(cos[~small])
    return theta


def relative_frobenius(A, B) -> float:
    """||A - B||_F / ||B||_F."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    nb = float(np.linalg.norm(B))
    if not nb > 0:
        raise ValueError(f"||B||_F must be positive and finite, got {nb}")
    return float(np.linalg.norm(A - B)) / nb


def split_half_groups(num_groups: int) -> tuple[list[int], list[int]]:
    """(even, odd) group ids: two disjoint image halves (plan Sec. 3.3 item 3)."""
    G = _check_int(num_groups, "num_groups", 2)
    return list(range(0, G, 2)), list(range(1, G, 2))


def bootstrap_group_weights(num_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, G] int64 Multinomial(G, uniform) counts, i.e. G groups drawn with replacement.

    Row b is the `weights` argument of `subset(range(G), weights)`; groups are hash buckets
    of images, so this is an image-level block bootstrap (plan Sec. 1.2)."""
    G = _check_int(num_groups, "num_groups", 1)
    B = _check_int(n_boot, "n_boot", 1)
    rng = np.random.default_rng(_check_int(seed, "seed", 0))
    return rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.int64)
