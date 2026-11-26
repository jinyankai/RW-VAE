"""v3 -- allocation analysis of one arm (RWVAE_V3_ALLOCATION_PLAN.md §2.5): forward passes only.

gate / rate maps  v3_eval.analysis_images evenly spaced rows of v3_eval.split. Per latent position: the gate
                  alpha (mean over groups; per group too), the KL summed over channels (nats), the edge density
                  (mean Sobel edge map, tokenizer_v3.edge_map at 4x the latent grid, average-pooled) and the texture
                  energy (variance of the gray image inside the position's cell). Pearson correlations over all
                  positions and the mean KL of positions with alpha above / below its median -> maps.json,
                  maps.npz, maps.png (first 8 images).
probe curves      u = (mu - m) / s in the DiT coordinates (stats.json); for every t in v3_eval.probe_t,
                  u_t = t u + (1 - t) eps (eps from Generator(probe_seed)), and
                  (a) a linear class probe on the spatial mean of u_t (evaluation.linear_probe), trained on
                      probe_train_per_class images per class of train_dev and tested on probe_test_per_class per
                      class of val;
                  (b) a per-token ridge map u_t -> DINO target tokens (deepest block used), fitted on the same
                      training images; mean cosine on the test images.
                  -> probe.json, probe.png. Curves further left (higher at small t) = semantics readable earlier.

Example  python scripts/v3_analysis.py <configs...> --arm T-c
"""
from __future__ import annotations

import _common  # noqa: I001

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _v3
from rwvae import datasets, evaluation, runtime
from rwvae import tokenizer_v3 as tv3


def build_parser():
    parser = _common.make_parser("Gate / rate maps and noisy-latent probe curves of a v3 arm.")
    _v3.add_arm_arg(parser)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--skip-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "analysis", args.arm)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        _common.set_tf32(False)
        tok, _, _ = _v3.load_arm_tokenizer(cfg, args.arm, device, args.tokenizer)
        dino = _v3.make_dino(cfg, device)
        manifest = _common.load_manifest(cfg)
        summary = {"maps": gate_rate_maps(cfg, tok, dino, manifest, device, out_dir, run)}
        if not args.skip_probe:
            summary["probe"] = probe_curves(cfg, args.arm, tok, dino, manifest, device, out_dir, run)
        run.finish("ok", **summary)
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


def _loader(cfg, manifest, split, rows):
    e = cfg["v3_eval"]
    ds = datasets.ManifestImageDataset(manifest, split, image_size=int(cfg["dataset"]["image_size"]), indices=rows)
    return DataLoader(ds, batch_size=int(e["batch_size"]), num_workers=int(e["num_workers"]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return None if den == 0 else float((a * b).sum() / den)


@torch.no_grad()
def gate_rate_maps(cfg, tok, dino, manifest, device, out_dir: Path, run) -> dict:
    e = cfg["v3_eval"]
    n_split = manifest.split_size(e["split"])
    N = min(int(e["analysis_images"]), n_split)
    rows = [i * n_split // N for i in range(N)]
    amp = device.type == "cuda"
    alpha, kl, edge, tex, first = [], [], [], [], None
    for x, _, _ in _loader(cfg, manifest, e["split"], rows):
        x = x.to(device)
        layers, _ = _v3.dino_inputs(dino, x, tok.cfg, amp)
        enc = tok.encode(x, layers)
        g = enc["mu"].shape[-1]
        if enc["alpha"] is not None:
            alpha.append(enc["alpha"].float().cpu())
        kl.append(tv3.kl_elementwise(enc["mu"].float(), enc["logvar"].float()).sum(1).cpu())
        edge.append(F.adaptive_avg_pool2d(tv3.edge_map(x, 4 * g), g)[:, 0].cpu())
        gray = ((x + 1) / 2 * x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        tex.append((F.adaptive_avg_pool2d(gray * gray, g) - F.adaptive_avg_pool2d(gray, g) ** 2)[:, 0].cpu())
        if first is None:
            first = x[:8].cpu()
    kl_m, edge_m, tex_m = (torch.cat(v).numpy() for v in (kl, edge, tex))
    res = {"n_images": N, "split": e["split"], "kl_mean_per_token": float(kl_m.mean()),
           "corr_kl_edge": _pearson(kl_m, edge_m), "corr_kl_texture": _pearson(kl_m, tex_m)}
    arrays = {"kl": kl_m, "edge": edge_m, "texture": tex_m}
    if alpha:
        a = torch.cat(alpha).numpy()                       # [N, G, g, g]
        a_m = a.mean(1)
        med = np.median(a_m)
        res.update(alpha_mean=float(a_m.mean()), alpha_mean_per_group=a.mean((0, 2, 3)).tolist(),
                   corr_alpha_edge=_pearson(a_m, edge_m), corr_alpha_texture=_pearson(a_m, tex_m),
                   corr_alpha_kl=_pearson(a_m, kl_m),
                   kl_where_alpha_high=float(kl_m[a_m > med].mean()) if (a_m > med).any() else None,
                   kl_where_alpha_low=float(kl_m[a_m <= med].mean()))
        arrays["alpha"] = a
    np.savez_compressed(out_dir / "maps.npz", **arrays)
    run.save_json("maps.json", res)
    _plot_maps(first, arrays, out_dir / "maps.png")
    run.logger.info("maps: %s", res)
    return res


def _plot_maps(images, arrays, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("alpha", "kl", "edge") if k in arrays]
    n = images.shape[0]
    fig, ax = plt.subplots(1 + len(keys), n, figsize=(2 * n, 2 * (1 + len(keys))), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(((images[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
        for r, k in enumerate(keys, 1):
            m = arrays[k][i].mean(0) if k == "alpha" else arrays[k][i]
            ax[r, i].imshow(m, cmap="viridis", vmin=0 if k == "alpha" else None, vmax=1 if k == "alpha" else None)
        for r in range(1 + len(keys)):
            ax[r, i].axis("off")
    for r, k in enumerate(["image", *keys]):
        ax[r, 0].set_title(k, loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _per_class_rows(manifest, split: str, k: int) -> list[int]:
    labels = np.asarray(manifest.splits[split]["label"])
    return sorted(int(i) for c in np.unique(labels) for i in np.flatnonzero(labels == c)[:k])


@torch.no_grad()
def _encode_rows(cfg, tok, dino, manifest, split, rows, mean, std, device):
    amp = device.type == "cuda"
    us, ys, tg = [], [], []
    for x, y, _ in _loader(cfg, manifest, split, rows):
        x = x.to(device)
        layers, target = _v3.dino_inputs(dino, x, tok.cfg, amp)
        mu = tok.encode(x, layers)["mu"].float()
        us.append(((mu - mean) / std).cpu())
        ys.append(y)
        tg.append(target.cpu())
    return torch.cat(us), torch.cat(ys), torch.cat(tg)


def _ridge_cosine(u_tr, t_tr, u_te, t_te, lam: float = 1e-3) -> float:
    """Per-token ridge map u [N, C, g, g] -> target [N, D, g, g]; mean test cosine."""
    X = u_tr.permute(0, 2, 3, 1).reshape(-1, u_tr.shape[1]).double()
    Y = t_tr.permute(0, 2, 3, 1).reshape(-1, t_tr.shape[1]).double()
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    A = X1.T @ X1
    W = torch.linalg.solve(A + lam * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=A.dtype), X1.T @ Y)
    Xt = u_te.permute(0, 2, 3, 1).reshape(-1, u_te.shape[1]).double()
    pred = torch.cat([Xt, torch.ones(len(Xt), 1, dtype=Xt.dtype)], 1) @ W
    Yt = t_te.permute(0, 2, 3, 1).reshape(-1, t_te.shape[1]).double()
    return float(F.cosine_similarity(pred, Yt, dim=1).mean())


def probe_curves(cfg, arm, tok, dino, manifest, device, out_dir: Path, run) -> dict:
    e = cfg["v3_eval"]
    stats = _v3.load_stats(cfg, arm)
    mean = torch.tensor(stats["mean"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, -1, 1, 1)
    tr_rows = _per_class_rows(manifest, "train_dev", int(e["probe_train_per_class"]))
    te_rows = _per_class_rows(manifest, "val", int(e["probe_test_per_class"]))
    u_tr, y_tr, t_tr = _encode_rows(cfg, tok, dino, manifest, "train_dev", tr_rows, mean, std, device)
    u_te, y_te, t_te = _encode_rows(cfg, tok, dino, manifest, "val", te_rows, mean, std, device)
    num_classes = len(manifest.classes)
    curve = []
    for t in [float(v) for v in e["probe_t"]] + [1.0]:
        g = torch.Generator().manual_seed(int(e["probe_seed"]))
        n_tr = u_tr * t + (1 - t) * torch.randn(u_tr.shape, generator=g)
        n_te = u_te * t + (1 - t) * torch.randn(u_te.shape, generator=g)
        acc = evaluation.linear_probe(evaluation.pooled_features(n_tr), y_tr, evaluation.pooled_features(n_te), y_te,
                                      num_classes=num_classes, device=device)
        cos = _ridge_cosine(n_tr, t_tr, n_te, t_te)
        curve.append({"t": t, "acc": acc["top1"], "probe": acc, "dino_cos": cos})
        run.logger.info("probe t=%.2f acc=%s dino_cos=%.4f", t, curve[-1]["acc"], cos)
    res = {"n_train": len(tr_rows), "n_test": len(te_rows), "curve": curve}
    run.save_json("probe.json", res)
    _plot_probe(curve, out_dir / "probe.png")
    return {"n_train": len(tr_rows), "n_test": len(te_rows),
            "acc": [c["acc"] for c in curve], "dino_cos": [c["dino_cos"] for c in curve]}


def _plot_probe(curve, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [c["t"] for c in curve]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(t, [c["acc"] for c in curve], marker="o")
    ax[0].set(xlabel="t (1 = clean)", ylabel="linear probe accuracy")
    ax[1].plot(t, [c["dino_cos"] for c in curve], marker="o")
    ax[1].set(xlabel="t (1 = clean)", ylabel="ridge cosine to DINO tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
