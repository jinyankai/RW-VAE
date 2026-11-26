"""07 -- FID reference statistics, gFID of one trained arm, and the experiment report (plan v2.1 Sec. 5.1,
5.3, 7). Three subcommands; the common flags (--config, --set, --out, --device, --seed, --overwrite,
--resume) follow the subcommand. Inception, the decoder and the statistics run with TF32 off.

ref-stats --split S [--max-images N] [--save-features [--hflip]]
    Inception (pytorch-fid pt_inception-2015-12-05, pool3) statistics of the center-cropped originals of
    manifest split S, quantised exactly like reconstructions and samples (evaluation.to_uint8 of the [-1, 1]
    tensors). --max-images N keeps the evenly spaced rows floor(i * n / N) of the class-sorted split.
    Outputs (root = --out or <run_root>/<name>/fid_ref; <ref> = S or S_n<N>):
      <root>/<ref>.npz            mu, sigma, n, meta (evaluation.save_fid_stats)
      <root>/<ref>_features.npz   per-image pool features (float16 [n, d]), labels, rows, flipped (--save-features;
                                  --hflip appends the features of the flipped images, the training augmentation)
      <root>/<ref>/               run record: config.yaml, log.txt, summary.json

generate --arm A [--R R] [--w W] [--dit-run DIR] [--ckpt latest|STEP] [--num-samples N] [--decoder-ckpt PATH]
         [--ref-stats [LABEL=]PATH ...] [--nn-ref LABEL=FEATURES_NPZ ...] [--nn-baseline FEATURES_NPZ]
         [--save-npz] [--save-grid K]
    No-CFG class-conditional samples of the EMA weights of one checkpoint of the arm's 06 run (sample.cfg_scale
    must be 1.0): labels = class_balanced_labels(N, K, sample.label_seed), sample i starts from
    initial_noise(i; sample.noise_seed) -- identical in every arm (results record the sha256 of all labels and
    initial noise) -- and is integrated with the ODE protocol of the `sample` block, denormalised with the
    arm's (mean, std, latent_multiplier), decoded by the arm's decoder (LMMSE fold of its allocation, identity
    for B0, or --decoder-ckpt from 05b), quantised by to_uint8 and fed to Inception. gFID is computed against
    EVERY reference (default: fid.reference_stats, else fid_ref/<fid.reference_split>.npz).
    Memorisation (--nn-ref, SVP overfit check): for every sample, the nearest feature of the reference set
    (largest cosine similarity: mean / median / p10 / p90 and the fraction > 0.95; smallest L2 distance:
    same summaries). With --nn-baseline (features of held-out images, e.g. val) the held-out images' own L2
    NN distances to the set give l2_median_ratio = median L2(samples) / median L2(held-out).
    Sampling runs in ONE process (multi-GPU sampling is not implemented; fine for SVP and FID-10K sizes).
    Outputs (<out> = --out or <run_root>/<name>/gen/<arm label>/step<k>_n<N>):
      results.json / results.md   gFID per reference, fid_kind (FID-10K | FID-50K | FID-<N>; never mixed in
                                  one table), nn statistics, sampling protocol, decoder, references, timings
      gen_fid_stats.npz           the samples' Inception statistics (reusable against other references)
      samples.npz                 arr_0 = uint8 [N, H, W, 3] (ADM evaluator format), arr_1 = labels (--save-npz;
                                  all samples are held in RAM: ~9.8 GB for 50k at 256 px)
      grid.png                    the first K samples (--save-grid K)

report
    Collects recon_probe/*/results.json (05), allocations/allocations_summary.json (04), dit/*/ (06: arm.json
    and the loss curve of metrics.jsonl) and gen/**/results.json (07 generate) into <out> = --out or
    <run_root>/<name>/report: REPORT.md, report.json and figures/*.png (rate-PSNR / rate-rFID / rate-probe per
    split, gFID per arm per FID kind, DiT loss curves). gFID tables are grouped by fid_kind; runs sharing
    (arm, checkpoint step, reference) -- e.g. training seeds -- are aggregated as mean +- std (n). With a `svp`
    config block, an "SVP verdict" section evaluates every svp.pass_criteria entry that SVP_CRITERIA knows
    (value, threshold, PASS/FAIL) and lists the ones it could not evaluate (unknown name, missing inputs).
    Re-running needs --overwrite (the report directory is a run directory).

Examples
  ImageNet-1k (GPU node):
    C="--config configs/dataset_imagenet256.yaml --config configs/baseline_common.yaml --config configs/arms.yaml \
       --config configs/server_a800.yaml"
    python scripts/07_evaluate.py ref-stats $C --split val
    python scripts/07_evaluate.py generate $C --arm B3 --R 22 --w 2                 # FID-10K screening
    python scripts/07_evaluate.py generate $C --arm B3 --R 22 --w 2 --num-samples 50000 --save-npz
    python scripts/07_evaluate.py report $C
  SVP (svp.yaml overlay; gFID against equal-size train_fit and val references, memorisation vs train_fit):
    C="$C --config configs/svp.yaml"; F=runs/svp/fid_ref
    python scripts/07_evaluate.py ref-stats $C --split train_fit --max-images 500
    python scripts/07_evaluate.py ref-stats $C --split train_fit --save-features --hflip
    python scripts/07_evaluate.py ref-stats $C --split val --save-features
    python scripts/07_evaluate.py generate $C --arm B0 --ref-stats train=$F/train_fit_n500.npz \
        --ref-stats val=$F/val.npz --nn-ref train=$F/train_fit_features.npz --nn-baseline $F/val_features.npz \
        --save-grid 64
    python scripts/07_evaluate.py report $C
  Local Imagenette (engineering test only):
    C="--config configs/dataset_imagenet256.yaml --config configs/baseline_common.yaml --config configs/arms.yaml \
       --config configs/local_imagenette.yaml"
    python scripts/07_evaluate.py ref-stats $C --split val --max-images 1000
    python scripts/07_evaluate.py generate $C --arm B3 --R 22 --w 2 --ref-stats val=runs/local_imagenette/fid_ref/val_n1000.npz
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rwvae import runtime

INCEPTION_WEIGHTS = "pytorch-fid pt_inception-2015-12-05 (pool3, 2048-d)"
NN_THRESHOLD = 0.95
_CKPT = re.compile(r"(\d+)\.pt")


# =========================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="FID references, gFID of one arm, experiment report "
                                                 "(plan Sec. 5.1, 5.3, 7).", formatter_class=fmt)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ref-stats", help="Inception statistics of a manifest split", formatter_class=fmt)
    runtime.add_common_args(p)
    p.add_argument("--split", required=True, help="manifest split (train_fit, train_dev, val)")
    p.add_argument("--max-images", type=int, default=None, help="evenly spaced subset of N rows")
    p.add_argument("--save-features", action="store_true", help="also save per-image pool features (float16)")
    p.add_argument("--hflip", action="store_true",
                   help="with --save-features: add the features of the horizontally flipped images (the training "
                        "augmentation) to the feature file; the FID statistics stay those of the originals")

    g = sub.add_parser("generate", help="sample one arm and compute gFID", formatter_class=fmt)
    runtime.add_common_args(g)
    g.add_argument("--arm", required=True, help="arm label from configs/arms.yaml (B0, B1, B2, B3)")
    g.add_argument("--R", type=float, default=None, help="nominal rate in nats/token (lossy arms)")
    g.add_argument("--w", type=float, default=None, help="relevance weight (rule r2 only)")
    g.add_argument("--dit-run", default=None, help="06 run directory (default dit/<arm label>)")
    g.add_argument("--ckpt", default="latest", help="'latest' or a training step")
    g.add_argument("--num-samples", type=int, default=None, help="default sample.num_samples_screen")
    g.add_argument("--decoder-ckpt", default=None, help="fine-tuned decoder from 05b (default: analytic decoder)")
    g.add_argument("--ref-stats", action="append", default=None, metavar="[LABEL=]PATH",
                   help="FID reference npz; repeatable (default fid.reference_stats or fid_ref/<reference_split>)")
    g.add_argument("--nn-ref", action="append", default=None, metavar="LABEL=FEATURES_NPZ",
                   help="reference features (ref-stats --save-features) for the memorisation metric; repeatable")
    g.add_argument("--nn-baseline", default=None, metavar="FEATURES_NPZ",
                   help="features of held-out images (e.g. ref-stats --split val --save-features): their NN "
                        "distance to each --nn-ref set is the scale of the memorisation ratio")
    g.add_argument("--save-npz", action="store_true", help="write samples.npz (ADM format; N*H*W*3 bytes in RAM)")
    g.add_argument("--save-grid", type=int, default=0, metavar="K", help="write a PNG grid of the first K samples")

    r = sub.add_parser("report", help="collect all results into REPORT.md", formatter_class=fmt)
    runtime.add_common_args(r)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    return {"ref-stats": cmd_ref_stats, "generate": cmd_generate, "report": cmd_report}[args.command](args, cfg)


# =========================================================================== shared helpers
class FeatureTap(nn.Module):
    """Wraps the Inception network and keeps a CPU float32 copy [b, d] of every feature batch it returns,
    i.e. exactly the features FIDStatistics accumulates."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net, self.features = net, []

    def forward(self, x):
        out = self.net(x)
        feats = out[0] if isinstance(out, (list, tuple)) else out
        self.features.append(feats.reshape(feats.shape[0], -1).float().cpu())
        return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def _labelled_paths(items: list[str] | None, flag: str) -> dict[str, Path]:
    """['LABEL=PATH' or 'PATH', ...] -> {label: path}; a bare PATH is labelled by its stem."""
    out: dict[str, Path] = {}
    for item in items or []:
        label, sep, path = item.partition("=")
        label, path = (label, path) if sep else (Path(item).stem, item)
        if not label or not path or label in out:
            raise ValueError(f"{flag} {item!r}: need distinct LABEL=PATH entries")
        out[label] = _common.resolve(path)
        if not out[label].is_file():
            raise FileNotFoundError(f"{flag} {item!r}: {out[label]} not found")
    return out


def _arm_point(arm: str, rule: str, R, w) -> tuple[float | None, float | None]:
    """(R, w) as 06_train_dit.py names the arm; flags that do not apply to the rule are errors."""
    if rule == "none":
        if R is not None or w is not None:
            raise ValueError(f"arm {arm} (rule none) is the rate-free reference: pass neither --R nor --w")
        return None, None
    if R is None or (rule == "r2" and w is None):
        raise ValueError(f"arm {arm} (rule {rule}) needs --R" + (" and --w" if rule == "r2" else ""))
    if rule != "r2" and w is not None:
        raise ValueError(f"--w applies to rule r2 only; arm {arm} has rule {rule}")
    return float(R), None if w is None else float(w)


def fid_kind(n: int) -> str:
    return {10000: "FID-10K", 50000: "FID-50K"}.get(int(n), f"FID-{int(n)}")


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}g}"
    if isinstance(v, dict) and "mean" in v and "lo" in v and "hi" in v:    # bootstrap_mean_ci
        return f"{_fmt(v['mean'])} [{_fmt(v['lo'])}, {_fmt(v['hi'])}]"
    return str(v)


def _md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return lines


# =========================================================================== ref-stats
def cmd_ref_stats(args, cfg: dict) -> int:
    manifest = _common.load_manifest(cfg)
    n_split = manifest.split_size(args.split)
    N = args.max_images
    if args.hflip and not args.save_features:
        raise ValueError("--hflip only adds features to the --save-features file; pass both")
    if N is not None and not 2 <= N <= n_split:
        raise ValueError(f"--max-images must lie in [2, {n_split}] for split {args.split!r}, got {N}")
    name = args.split if N is None else f"{args.split}_n{N}"
    root = Path(args.out) if args.out else _common.stage_dir(cfg, "fid_ref")
    stats_file, feats_file = root / f"{name}.npz", root / f"{name}_features.npz"
    run = _common.open_run(__file__, cfg, root / name, args)
    try:
        from torch.utils.data import DataLoader
        from rwvae import datasets, evaluation, integration
        runtime.seed_everything(int(cfg.get("seed", 0)))
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        rows = list(range(n_split)) if N is None else [i * n_split // N for i in range(N)]
        size = int(cfg["dataset"]["image_size"])
        images = datasets.ManifestImageDataset(manifest, args.split, image_size=size, indices=rows)
        loader = DataLoader(images, batch_size=int(cfg["fid"]["inception_batch"]), shuffle=False,
                            num_workers=int(cfg["eval"]["num_workers"]))
        net = integration.load_inception(device)
        inception = FeatureTap(net) if args.save_features else net   # tap only when the features are kept
        fid = evaluation.FIDStatistics(inception, device, batch_size=int(cfg["fid"]["inception_batch"]))
        labels, t0 = [], time.perf_counter()
        for x, y, _ in loader:
            fid.update(evaluation.to_uint8(x))
            labels.append(y)
            run.log_metrics(None, images=sum(len(b) for b in labels))
        mu, sigma, n = fid.compute()
        labels = torch.cat(labels).numpy().astype(np.int64)
        meta = {"kind": "reference", "split": args.split, "n": n, "rows": "all" if N is None else
                "evenly spaced rows floor(i * n_split / N)", "rows_sha256": _sha256_array(np.asarray(rows, np.int64)),
                "n_split": n_split, "max_images": N, "manifest_fingerprint": manifest.fingerprint(),
                "dataset": cfg["dataset"].get("name"), "image_size": size, "preprocessing": datasets.PREPROCESSING,
                "quantisation": "evaluation.to_uint8 of the [-1, 1] tensor (same as reconstructions/samples)",
                "inception": INCEPTION_WEIGHTS, "inception_module": type(net).__name__,
                "class_counts": np.bincount(labels, minlength=len(manifest.classes)).tolist(),
                "tf32": tf32, "git_commit": runtime.git_commit(), "created": _utc_now()}
        if n <= mu.shape[0]:
            run.logger.warning("only %d reference images for %d-d features: the covariance is singular (FID "
                               "against it is a sanity number, not comparable to FID-10K/50K)", n, mu.shape[0])
        root.mkdir(parents=True, exist_ok=True)
        evaluation.save_fid_stats(stats_file, mu, sigma, n, meta)
        summary = {"stats_file": stats_file, "n": n, "split": args.split, "seconds": time.perf_counter() - t0}
        if args.save_features:
            feats, flipped = torch.cat(inception.features), np.zeros(n, dtype=bool)
            labels_f, rows_f = labels, np.asarray(rows, np.int64)
            if args.hflip:                               # same quantisation and Inception input path as FIDStatistics
                inception.features = []
                flips = datasets.ManifestImageDataset(manifest, args.split, image_size=size, indices=rows, hflip=True)
                with torch.no_grad():
                    for x, _, _ in DataLoader(flips, batch_size=loader.batch_size, num_workers=loader.num_workers):
                        inception.eval()(evaluation.to_uint8(x).float().div(255.0).to(device))
                feats = torch.cat([feats, *inception.features])
                labels_f, rows_f = np.concatenate([labels, labels]), np.concatenate([rows_f, rows_f])
                flipped = np.concatenate([flipped, np.ones(n, dtype=bool)])
            feats = feats.numpy().astype(np.float16)
            if not np.isfinite(feats).all():
                raise ValueError("pool features overflow float16")
            fmeta = {**meta, "features": "Inception pool features, float16", "includes_hflip": bool(args.hflip)}
            with open(feats_file, "wb") as f:
                np.savez(f, features=feats, labels=labels_f, rows=rows_f, flipped=flipped,
                         meta=np.array(json.dumps(runtime.to_jsonable(fmeta), sort_keys=True)))
            summary["features_file"] = feats_file
        run.logger.info("reference %s: n=%d -> %s", name, n, stats_file)
        run.finish("ok", **summary)
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


# =========================================================================== generate
def _pick_checkpoint(dit_run: Path, which: str) -> Path:
    ckpt_dir = dit_run / "checkpoints"
    found = sorted((int(m.group(1)), p) for p in (ckpt_dir.iterdir() if ckpt_dir.is_dir() else [])
                   if (m := _CKPT.fullmatch(p.name)))
    if not found:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}: run scripts/06_train_dit.py first")
    if which == "latest":
        return found[-1][1]
    path = dict(found).get(int(which))
    if path is None:
        raise FileNotFoundError(f"no checkpoint for step {which} in {ckpt_dir} (have {[s for s, _ in found]})")
    return path


def _default_refs(cfg: dict) -> dict[str, Path]:
    ext = cfg["fid"].get("reference_stats")
    if ext:
        return {"reference": _common.resolve(ext)}
    split = cfg["fid"]["reference_split"]
    return {split: _common.stage_dir(cfg, "fid_ref", f"{split}.npz")}


def _build_decoder(cfg: dict, rule: str, alloc, decoder_ckpt, device):
    """(AdaptedDecoder on device in eval mode, record): the arm's analytic decoder or a 05b fine-tuned one."""
    from rwvae.decoder_adapter import AdaptedDecoder
    tok = _common.make_tokenizer(cfg, "cpu")
    mode = "identity" if rule == "none" or decoder_ckpt else cfg["decoder"]["mode"]
    kwargs = dict(A=alloc.A, m=alloc.m) if mode == "lmmse" else {}
    dec = AdaptedDecoder(tok.post_quant_conv, tok.decoder, mode=mode, **kwargs)
    info = {"mode": mode, "tokenizer": tok.meta, "allocation_fingerprint": alloc.fingerprint()}
    if decoder_ckpt:
        path = _common.resolve(decoder_ckpt)
        saved = dec.load_weights(path)
        want = saved.get("allocation_fingerprint")
        if want is not None and want != alloc.fingerprint():
            raise ValueError(f"{path} was fine-tuned for allocation {want}, the arm has {alloc.fingerprint()}")
        info.update(mode=dec.mode, base_mode=dec.base_mode, ckpt=path, ckpt_sha256=runtime.sha256_file(path),
                    ckpt_meta=saved)
    del tok
    return dec.to(device).eval(), info


def _nearest(queries: torch.Tensor, ref: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    """Per query feature: the largest cosine similarity and the smallest L2 distance to any reference feature
    (float64; chunked so one chunk of the [Nq, Nref] product stays ~512 MB; ||q - r||^2 is formed as
    |q|^2 + |r|^2 - 2 q.r and float64 rounding below 0 is set to 0)."""
    r = ref.to(device, torch.float64)
    nr2 = (r * r).sum(1)
    cos, l2 = [], []
    for q in queries.split(max(1, 2 ** 26 // len(r))):
        q = q.to(device, torch.float64)
        nq2 = (q * q).sum(1)
        dots = q @ r.T
        cos.append((dots / torch.sqrt(nq2[:, None] * nr2[None])).max(dim=1).values.cpu())
        l2.append((nq2[:, None] + nr2[None] - 2 * dots).clamp_min(0).min(dim=1).values.sqrt().cpu())
    return torch.cat(cos).numpy(), torch.cat(l2).numpy()


def _summary(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


def _load_features(path: Path) -> tuple[torch.Tensor, dict]:
    with np.load(path, allow_pickle=False) as f:
        return torch.from_numpy(f["features"].astype(np.float32)), json.loads(f["meta"].item())


def cmd_generate(args, cfg: dict) -> int:
    s = cfg["sample"]
    if float(s["cfg_scale"]) != 1.0:
        raise ValueError(f"sample.cfg_scale={s['cfg_scale']}: the main metric is gFID without CFG "
                         "(plan Sec. 5.1); CFG sampling is not implemented")
    rule = _common.arm_rule(cfg, args.arm)
    R, w = _arm_point(args.arm, rule, args.R, args.w)
    label = _common.arm_label(args.arm, rule, R, w)
    dit_run = _common.resolve(args.dit_run) if args.dit_run else _common.stage_dir(cfg, "dit", label)
    arm_info = json.loads((dit_run / "arm.json").read_text(encoding="utf-8"))
    if (arm_info["arm"], arm_info["rule"], arm_info["R"], arm_info["w"]) != (args.arm, rule, R, w):
        raise ValueError(f"{dit_run} trained arm {arm_info['label']}, not {label}")
    ckpt_path = _pick_checkpoint(dit_run, args.ckpt)
    step = int(_CKPT.fullmatch(ckpt_path.name).group(1))
    N = int(args.num_samples or s["num_samples_screen"])
    if N < 2:
        raise ValueError(f"--num-samples must be >= 2, got {N}")
    refs = _labelled_paths(args.ref_stats, "--ref-stats") or _default_refs(cfg)
    for ref_label, path in refs.items():
        if not path.is_file():
            raise FileNotFoundError(f"reference {ref_label}: {path} not found (run `07_evaluate.py ref-stats`)")
    nn_refs = _labelled_paths(args.nn_ref, "--nn-ref")
    baseline = _common.resolve(args.nn_baseline) if args.nn_baseline else None
    if baseline is not None and (not nn_refs or not baseline.is_file()):
        raise ValueError(f"--nn-baseline {baseline} needs an existing file and at least one --nn-ref")
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "gen", label, f"step{step}_n{N}")
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        from rwvae import evaluation, generation, integration
        runtime.seed_everything(int(cfg.get("seed", 0)))
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        log = run.logger
        if args.save_npz and N * 256 * 256 * 3 > 4 * 2 ** 30:
            log.warning("--save-npz keeps all %d samples in RAM (~%.1f GB at 256 px)", N, N * 256 * 256 * 3 / 2 ** 30)

        # the arm: statistics, allocation and normalisation must be those the DiT was trained with
        stats = _common.load_stats(cfg)
        if stats.fingerprint() != arm_info["stats"]["fingerprint"]:
            raise ValueError(f"stats fingerprint {stats.fingerprint()} != {arm_info['stats']['fingerprint']} "
                             f"recorded in {dit_run / 'arm.json'}")
        alloc = _common.load_allocation(cfg, args.arm, R, w)
        if alloc.fingerprint() != arm_info["allocation"]["fingerprint"]:
            raise ValueError(f"allocation fingerprint {alloc.fingerprint()} != the trained arm's "
                             f"{arm_info['allocation']['fingerprint']}")
        mean, std = stats.channel_norm()
        multiplier = float(arm_info["normalisation"]["latent_multiplier"])
        num_classes = int(arm_info["data"]["num_classes"])
        latent_shape = tuple(int(v) for v in arm_info["data"]["latent_shape"])

        # the model: the run's own config, EMA weights of the chosen checkpoint
        train_cfg = runtime.load_config(dit_run / "config.yaml")
        model = _common.build_dit(train_cfg, num_classes=num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if int(ckpt["step"]) != step:
            raise ValueError(f"{ckpt_path} holds step {ckpt['step']}")
        model.load_state_dict(ckpt["ema"])
        model.to(device).eval()
        transport = integration.build_transport(train_cfg["transport"])
        sample_fn = integration.build_ode_sample_fn(transport, s)
        decoder, decoder_info = _build_decoder(cfg, rule, alloc, args.decoder_ckpt, device)
        net = integration.load_inception(device)
        inception = FeatureTap(net) if nn_refs else net             # N x d float32 features, only for --nn-ref
        fid = evaluation.FIDStatistics(inception, device, batch_size=int(cfg["fid"]["inception_batch"]))

        labels = generation.class_balanced_labels(N, num_classes, int(s["label_seed"]))
        base_seed, batch = int(s["noise_seed"]), int(s["batch_size"])
        noise_hash, keep, timings = hashlib.sha256(), [], {"sample": 0.0, "decode": 0.0, "inception": 0.0}
        n_keep = N if args.save_npz else int(args.save_grid)
        t_start = time.perf_counter()
        for start in range(0, N, batch):
            idx = list(range(start, min(start + batch, N)))
            noise_hash.update(generation.initial_noise(idx, latent_shape, base_seed, "cpu").numpy().tobytes())
            t0 = time.perf_counter()
            z = generation.sample_latents(model, sample_fn, labels[idx], idx, base_seed=base_seed, batch_size=batch,
                                          device=device, latent_shape=latent_shape,
                                          mixed_precision=s.get("mixed_precision", "bf16"))
            t1 = time.perf_counter()
            with torch.no_grad():
                u8 = evaluation.to_uint8(decoder(generation.denormalize(z, mean, std, multiplier).to(device)))
            t2 = time.perf_counter()
            fid.update(u8)
            t3 = time.perf_counter()
            timings["sample"] += t1 - t0
            timings["decode"] += t2 - t1
            timings["inception"] += t3 - t2
            if len(keep) * batch < n_keep:
                keep.append(u8.cpu())
            run.log_metrics(None, samples=idx[-1] + 1, seconds=time.perf_counter() - t_start)
        mu, sigma, n = fid.compute()
        evaluation.save_fid_stats(out_dir / "gen_fid_stats.npz", mu, sigma, n,
                                  {"kind": "generated", "arm": label, "step": step, "n": n,
                                   "inception": INCEPTION_WEIGHTS, "tf32": tf32})

        gfid, references = {}, {}
        for ref_label, path in refs.items():
            mu_r, s_r, n_r, meta_r = evaluation.load_fid_stats(path)
            if mu_r.shape != mu.shape:
                raise ValueError(f"reference {ref_label} has {mu_r.shape[0]}-d features, samples {mu.shape[0]}-d")
            value, info = evaluation.frechet_distance(mu, sigma, mu_r, s_r, return_info=True)
            gfid[ref_label] = value
            references[ref_label] = {"file": path, "sha256": runtime.sha256_file(path), "n": n_r, "meta": meta_r,
                                     "frechet_info": info}
            log.info("gFID vs %s (n_ref=%d): %.4f", ref_label, n_r, value)
        samples_feats = torch.cat(inception.features) if nn_refs else None
        nn_stats = {}
        base_feats, base_meta = _load_features(baseline) if baseline else (None, None)
        for ref_label, path in nn_refs.items():
            ref_feats, ref_meta = _load_features(path)
            cos, l2 = _nearest(samples_feats, ref_feats, device)
            entry = {"cosine": {**_summary(cos), "threshold": NN_THRESHOLD,
                                "frac_above": float((cos > NN_THRESHOLD).mean())},
                     "l2": _summary(l2), "n_ref": int(ref_feats.shape[0]), "file": path,
                     "sha256": runtime.sha256_file(path), "ref_split": ref_meta.get("split"),
                     "includes_hflip": ref_meta.get("includes_hflip", False),
                     "metric": "nearest reference of each sample's Inception pool feature: max cosine, min L2",
                     "baseline": None, "l2_median_ratio": None}
            if base_feats is not None:
                b_cos, b_l2 = _nearest(base_feats, ref_feats, device)
                entry["baseline"] = {"file": baseline, "sha256": runtime.sha256_file(baseline),
                                     "split": base_meta.get("split"), "n": int(base_feats.shape[0]),
                                     "cosine": _summary(b_cos), "l2": _summary(b_l2)}
                entry["l2_median_ratio"] = float(np.median(l2) / np.median(b_l2)) if np.median(b_l2) > 0 else None
            nn_stats[ref_label] = entry
            log.info("NN vs %s: cosine mean %.4f, frac > %.2f = %.3f; L2 median %.4g (ratio to baseline %s)",
                     ref_label, cos.mean(), NN_THRESHOLD, entry["cosine"]["frac_above"], np.median(l2),
                     entry["l2_median_ratio"])

        files = {"gen_fid_stats": out_dir / "gen_fid_stats.npz"}
        if keep:
            kept = torch.cat(keep)[:n_keep].permute(0, 2, 3, 1).contiguous().numpy()
            if args.save_npz:
                with open(out_dir / "samples.npz", "wb") as f:
                    np.savez(f, kept, labels.numpy())          # arr_0 images, arr_1 labels (ADM layout)
                files["samples_npz"] = out_dir / "samples.npz"
            if args.save_grid:
                files["grid"] = _save_grid(kept[:args.save_grid], out_dir / "grid.png")

        results = {
            "arm": args.arm, "rule": rule, "R": R, "w": w, "label": label,
            "exploratory": arm_info.get("exploratory"), "registered": arm_info.get("preregistration", {}).get("registered"),
            "dit_run": dit_run, "checkpoint": {"file": ckpt_path, "step": step, "sha256": runtime.sha256_file(ckpt_path),
                                               "weights": "ema"},
            "dit_train_seed": train_cfg["train"].get("seed"), "dit_init_seed": arm_info.get("init_seed"),
            "fid_kind": fid_kind(N), "num_samples": N, "num_classes": num_classes, "gfid": gfid,
            "references": references, "nn": nn_stats,
            "sampling": {k: s.get(k) for k in ("mode", "sampling_method", "num_sampling_steps", "atol", "rtol",
                                               "reverse", "timestep_shift", "cfg_scale", "label_seed", "noise_seed",
                                               "batch_size", "mixed_precision")}
                        | {"labels": "class_balanced_labels(N, K, label_seed)", "noise_indices": f"0..{N - 1}",
                           "labels_sha256": _sha256_array(labels.numpy()),
                           "initial_noise_sha256": noise_hash.hexdigest(), "processes": 1},
            "normalisation": {"mean": mean, "std": std, "latent_multiplier": multiplier,
                              "stats_fingerprint": stats.fingerprint()},
            "decoder": decoder_info, "manifest_fingerprint": arm_info["data"].get("manifest_fingerprint"),
            "inception": INCEPTION_WEIGHTS, "tf32": tf32, "device": str(device), "git_commit": runtime.git_commit(),
            "timings_sec": timings | {"total": time.perf_counter() - t_start}, "files": files,
        }
        run.save_json("results.json", results)
        _write_results_md(out_dir / "results.md", results)
        run.finish("ok", fid_kind=results["fid_kind"], gfid=gfid, nn={k: v["cosine"]["mean"] for k, v in nn_stats.items()},
                   step=step, num_samples=N)
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


def _save_grid(images_nhwc: np.ndarray, path: Path) -> Path:
    from PIL import Image
    k, h, w, c = images_nhwc.shape
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, img in enumerate(images_nhwc):
        r, q = divmod(i, cols)
        grid[r * h:(r + 1) * h, q * w:(q + 1) * w] = img
    Image.fromarray(grid).save(path)
    return path


def _write_results_md(path: Path, res: dict) -> None:
    lines = [f"# gFID {res['label']} step {res['checkpoint']['step']} ({res['fid_kind']}, no CFG)", ""]
    lines += _md_table(["reference", "n_ref", "gFID"],
                       [[k, res["references"][k]["n"], v] for k, v in res["gfid"].items()])
    if res["nn"]:
        lines += ["", f"Nearest-neighbour cosine similarity (Inception pool features; threshold {NN_THRESHOLD})", ""]
        lines += _md_table(["reference", "n_ref", "cos mean", "cos median", "cos p10", "cos p90", "frac_above",
                            "L2 median", "L2 median / baseline"],
                           [[k, v["n_ref"], v["cosine"]["mean"], v["cosine"]["median"], v["cosine"]["p10"],
                             v["cosine"]["p90"], v["cosine"]["frac_above"], v["l2"]["median"], v["l2_median_ratio"]]
                            for k, v in res["nn"].items()])
    sp = res["sampling"]
    lines += ["", f"Sampling: {sp['mode']} {sp['sampling_method']}, {sp['num_sampling_steps']} steps, shift "
              f"{sp['timestep_shift']}, cfg {sp['cfg_scale']}, label seed {sp['label_seed']}, noise seed "
              f"{sp['noise_seed']}; decoder {res['decoder']['mode']}; exploratory={res['exploratory']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================================== report
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"_error": f"{path}: {e!r}"}


def _ci(d, scale: float = 1.0) -> str:
    """'mean [lo, hi]' of a bootstrap CI dict (05_eval_recon_probe Stats.ci), scaled; '-' if absent."""
    if not isinstance(d, dict) or d.get("mean") is None:
        return "-"
    lo, hi = d.get("lo"), d.get("hi")
    text = _fmt(d["mean"] * scale)
    return text if lo is None or hi is None else f"{text} [{_fmt(lo * scale)}, {_fmt(hi * scale)}]"


def _mean(d):
    return d.get("mean") if isinstance(d, dict) else None


def _loss_curve(run_dir: Path) -> list[tuple[int, float]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    curve = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line) if line.strip() else {}
        if isinstance(rec.get("step"), int) and isinstance(rec.get("loss"), (int, float)):
            curve.append((rec["step"], float(rec["loss"])))
    return curve


def collect(cfg: dict) -> dict:
    """Every input of the report. Unreadable files are kept as {'_error': ...} and listed in the report."""
    root = _common.exp_dir(cfg)
    m3 = {p.parent.name: {"file": p, "data": _read_json(p)} for p in sorted(root.glob("recon_probe/*/results.json"))}
    alloc_file = _common.allocations_dir(cfg) / "allocations_summary.json"
    dit = {p.parent.name: {"dir": p.parent, "arm": _read_json(p), "curve": _loss_curve(p.parent)}
           for p in sorted(root.glob("dit/*/arm.json"))}
    gen = [{"file": p, **_read_json(p)} for p in sorted(root.glob("gen/**/results.json"))]
    gates = {"cache_verification": {s: _common.cache_dir(cfg, s) / "cache_verification.json"
                                    for s in cfg["cache"].get("splits", [])},
             "stats_diagnostics": _common.stats_path(cfg).parent / "diagnostics.json"}
    return {"root": root, "m3": m3, "dit": dit, "gen": gen, "gates": gates,
            "allocations": {"file": alloc_file, "data": _read_json(alloc_file)} if alloc_file.is_file() else None}


def _m3_lines(split: str, m: dict) -> list[str]:
    d, lines = m["data"], [f"### split {split}", "", f"source `{Path(m['file']).as_posix()}`"]
    if "_error" in d or not isinstance(d.get("arms"), dict):
        return lines + ["", f"(unreadable or unknown format: {d.get('_error', sorted(d)[:10])})", ""]
    meta = d.get("meta", {})
    lines += [f"{meta.get('n_images')} images ({meta.get('split_role', '')}); rFID: {meta.get('rfid_note')}; "
              f"dropped: {meta.get('dropped') or 'none'}", ""]
    rows = [[r.get("label"), r.get("R"), r.get("w"), r.get("bpp"), _ci(r.get("psnr")), _ci(r.get("lpips")),
             (r.get("rfid") or {}).get("value"), _ci((r.get("probe") or {}).get("top1_ci"), 100),
             r.get("semantic_retention"), (r.get("latent_mse") or {}).get("predicted"),
             _ci((r.get("latent_mse") or {}).get("empirical"))] for r in d["arms"].values()]
    lines += _md_table(["arm", "R", "w", "bpp", "PSNR dB", "LPIPS", "rFID", "probe top-1 %", "sem. retention",
                        "D_latent (pred.)", "latent MSE (emp.)"], rows)
    if d.get("paired"):
        lines += ["", "Paired comparisons (same images, same channel noise; bootstrap CIs)", ""]
        lines += _md_table(["R", "w", "a - b", "PSNR diff dB", "probe diff points"],
                           [[p.get("R"), p.get("w"), f"{p.get('a')} - {p.get('b')}", _ci(p.get("psnr_diff_db")),
                             _ci(p.get("probe_top1_diff_points"))] for p in d["paired"]])
    g = d.get("gates", {})
    probe, mono = g.get("probe", {}), g.get("monotonicity", {})
    lines += ["", f"Gates: probe {'PASS' if probe.get('pass') else 'not passed' if probe.get('evaluated') else 'not evaluated'}"
              f" (threshold {probe.get('threshold_points')} points); monotone in R: {mono.get('monotone')}"
              f" ({len(mono.get('violations', []))} violations)", ""]
    return lines


def _alloc_lines(a: dict | None) -> list[str]:
    if a is None:
        return ["(no allocations/allocations_summary.json)"]
    d = a["data"]
    recs = d.get("allocations") if isinstance(d, dict) else None   # 04: {allocation name: record}
    if not isinstance(recs, dict):
        return [f"(unreadable or unknown format: {d.get('_error', sorted(d)[:10])})"]
    lines = [f"source `{Path(a['file']).as_posix()}`; go/no-go: {(d.get('go_no_go') or {}).get('verdict')}", ""]
    return lines + _md_table(["allocation", "R", "w", "bpp", "active", "D_latent", "sem. retention", "max residual"],
                             [[r.get("name"), r.get("R"), r.get("w"), r.get("bpp_equivalent"), r.get("n_active"),
                               r.get("latent_mse_lmmse"), r.get("semantic_retention"), r.get("identity_residual_max")]
                              for r in recs.values()])


def _gen_lines(gen: list[dict], root: Path) -> tuple[list[str], dict]:
    good = [g for g in gen if "fid_kind" in g and "gfid" in g]
    lines, agg_out = [], {}
    for kind in sorted({g["fid_kind"] for g in good}):
        runs = [g for g in good if g["fid_kind"] == kind]
        refs = sorted({r for g in runs for r in g["gfid"]})
        nn_refs = sorted({r for g in runs for r in g.get("nn", {})})
        lines += ["", f"### {kind} (no CFG; never compare with other FID kinds)", ""]
        rows = []
        for g in runs:
            nn_cells = []
            for k in nn_refs:
                v = g.get("nn", {}).get(k)
                nn_cells.append("-" if v is None else f"{_fmt(v['cosine']['mean'])} / {_fmt(v['cosine']['frac_above'])}"
                                f" / {_fmt(v.get('l2_median_ratio'))}")
            rows.append([g["label"], g["checkpoint"]["step"], g.get("exploratory"), g.get("dit_train_seed"),
                         Path(g["file"]).parent.relative_to(root).as_posix(), *[g["gfid"].get(r) for r in refs],
                         *nn_cells])
        lines += _md_table(["arm", "step", "exploratory", "train seed", "run", *[f"gFID vs {r}" for r in refs],
                            *[f"NN vs {r}: cos mean / frac>{NN_THRESHOLD} / L2 med. ratio" for r in nn_refs]], rows)
        groups: dict[tuple, list[float]] = {}
        for g in runs:
            for r, v in g["gfid"].items():
                groups.setdefault((g["label"], g["checkpoint"]["step"], r), []).append(v)
        agg = {key: {"mean": statistics.fmean(v), "std": statistics.stdev(v) if len(v) > 1 else None, "n": len(v)}
               for key, v in sorted(groups.items())}
        if any(a["n"] > 1 for a in agg.values()):
            lines += ["", "Runs sharing (arm, step, reference), e.g. training seeds: mean +- std (n)", ""]
            lines += _md_table(["arm", "step", "reference", "gFID mean", "std", "n"],
                               [[*key, a["mean"], a["std"], a["n"]] for key, a in agg.items()])
        agg_out[kind] = {f"{a}|step{s}|{r}": v for (a, s, r), v in agg.items()}
    bad = [Path(g["file"]).as_posix() for g in gen if g not in good]
    if bad:
        lines += ["", f"Not tabulated (unreadable or without fid_kind): {bad}"]
    return lines, agg_out


def _dit_rows(dit: dict) -> list[dict]:
    rows = []
    for name, d in dit.items():
        losses = [v for _, v in d["curve"]]
        rows.append({"run": name, "label": d["arm"].get("label"), "exploratory": d["arm"].get("exploratory"),
                     "repeat": d["arm"].get("repeat"), "logged": len(losses),
                     "last_step": d["curve"][-1][0] if losses else None, "first": losses[0] if losses else None,
                     "last": losses[-1] if losses else None, "min": min(losses) if losses else None})
    return rows


# --------------------------------------------------------------------------- SVP verdict (configs/svp.yaml)
# Each evaluator gets (collected inputs, the criterion's mapping from svp.pass_criteria) and returns
# (value, threshold text, passed, detail) or raises _NotEvaluated(reason).
class _NotEvaluated(Exception):
    pass


def _need(spec: dict, *keys: str) -> list[float]:
    missing = [k for k in keys if not isinstance(spec.get(k), (int, float))]
    if missing:
        raise _NotEvaluated(f"threshold key(s) {missing} missing")
    return [float(spec[k]) for k in keys]


def _window_mean(curve, lo: float, hi: float):
    vals = [v for s, v in curve if lo < s <= hi]
    return statistics.fmean(vals) if vals else None


def _curves(c: dict) -> dict:
    curves = {k: d["curve"] for k, d in c["dit"].items() if d["curve"]}
    if not curves:
        raise _NotEvaluated("no 06 run with a loss curve (dit/*/metrics.jsonl)")
    return curves


def _svp_loss_drop(c, spec):
    (max_ratio,) = _need(spec, "max_ratio")
    worst, detail = None, {}
    for name, curve in _curves(c).items():
        last = curve[-1][0]
        win = min(1000, last // 2)       # 1k steps as specified; halved for runs shorter than 2k steps
        ratio = _window_mean(curve, last - win, last) / _window_mean(curve, 0, win)
        detail[name] = {"ratio": ratio, "window_steps": win}
        worst = ratio if worst is None else max(worst, ratio)
    return worst, f"<= {max_ratio:g}", worst <= max_ratio, detail


def _svp_loss_plateau(c, spec):
    (max_rel,) = _need(spec, "max_rel_change")
    worst, detail = None, {}
    for name, curve in _curves(c).items():
        last = curve[-1][0]
        win = min(2000, last // 2)       # 2k steps as specified; halved for runs shorter than 4k steps
        late, prev = _window_mean(curve, last - win, last), _window_mean(curve, last - 2 * win, last - win)
        if late is None or prev is None:
            raise _NotEvaluated(f"{name}: too few logged steps for two windows of {win}")
        rel = abs(late - prev) / late
        detail[name] = {"rel_change": rel, "window_steps": win}
        worst = rel if worst is None else max(worst, rel)
    return worst, f"<= {max_rel:g}", worst <= max_rel, detail


def _ref_label(table: dict, side: str, key) -> str | None:
    """Label of the reference in `table` whose split (key(info)) is a training split ('train') or 'val'."""
    for label, info in table.items():
        split = key(info) or label
        if (side == "train" and split.startswith("train")) or (side == "val" and split.startswith("val")):
            return label
    return None


def _b0_final(c: dict) -> list[dict]:
    """07 results of arm B0 at the final (largest) step of each 06 run."""
    b0 = [g for g in c["gen"] if g.get("arm") == "B0" and "gfid" in g]
    last = {}
    for g in b0:
        last[g.get("dit_run")] = max(last.get(g.get("dit_run"), 0), g["checkpoint"]["step"])
    out = [g for g in b0 if g["checkpoint"]["step"] == last[g.get("dit_run")]]
    if not out:
        raise _NotEvaluated("no 07 generate results for arm B0")
    return out


def _svp_gfid_mem(c, spec):
    (max_ratio,) = _need(spec, "max_ratio")
    worst, detail = None, {}
    for g in _b0_final(c):
        split = lambda info: (info.get("meta") or {}).get("split")
        tr, va = _ref_label(g["references"], "train", split), _ref_label(g["references"], "val", split)
        if tr is None or va is None:
            raise _NotEvaluated(f"{g['file']}: needs gFID against a train_fit and a val reference")
        ratio = g["gfid"][tr] / g["gfid"][va]
        detail[Path(g["file"]).parent.name] = {"ratio": ratio, "gfid_train": g["gfid"][tr], "gfid_val": g["gfid"][va],
                                               "n_train": g["references"][tr]["n"], "n_val": g["references"][va]["n"],
                                               "step": g["checkpoint"]["step"]}
        worst = ratio if worst is None else max(worst, ratio)
    return worst, f"<= {max_ratio:g}", worst <= max_ratio, detail


def _svp_nn_mem(c, spec):
    (max_ratio,) = _need(spec, "max_ratio")
    worst, detail = None, {}
    for g in _b0_final(c):
        tr = _ref_label(g.get("nn", {}), "train", lambda info: info.get("ref_split"))
        ratio = None if tr is None else g["nn"][tr].get("l2_median_ratio")
        if ratio is None:
            raise _NotEvaluated(f"{g['file']}: needs --nn-ref train=<train_fit features> and --nn-baseline "
                                "<val features> (ref-stats --save-features)")
        detail[Path(g["file"]).parent.name] = {"ratio": ratio, "includes_hflip": g["nn"][tr].get("includes_hflip")}
        worst = ratio if worst is None else max(worst, ratio)
    return worst, f"<= {max_ratio:g}", worst <= max_ratio, detail


def _svp_alloc_residual(c, spec):
    (tol,) = _need(spec, "max")
    recs = (c["allocations"] or {}).get("data", {}).get("allocations") if c["allocations"] else None
    if not isinstance(recs, dict) or not recs:                     # 04: {allocation name: record}
        raise _NotEvaluated("no allocations/allocations_summary.json")
    name = max(recs, key=lambda n: recs[n]["identity_residual_max"])
    worst = recs[name]["identity_residual_max"]
    return worst, f"<= {tol:g}", worst <= tol, {"worst": name, "n": len(recs)}


def _m3_arms(c: dict) -> list[tuple[str, dict]]:
    out = [(split, r) for split, m in c["m3"].items() if isinstance(m["data"].get("arms"), dict)
           for r in m["data"]["arms"].values()]
    if not out:
        raise _NotEvaluated("no recon_probe/*/results.json")
    return out


def _svp_latent_mse(c, spec):
    (max_rel,) = _need(spec, "max_rel_err")
    detail = {}
    for split, r in _m3_arms(c):
        lm = r.get("latent_mse") or {}
        emp, pred = _mean(lm.get("empirical")), lm.get("predicted")
        if r.get("rule") != "none" and emp is not None and pred:
            detail[f"{split}/{r['label']}"] = abs(emp - pred) / pred
    if not detail:
        raise _NotEvaluated("no lossy arm with empirical and predicted latent MSE")
    worst = max(detail.values())
    return worst, f"<= {max_rel:g}", worst <= max_rel, {"worst": max(detail, key=detail.get), "n": len(detail)}


def _svp_monotone(c, spec):
    psnr_tol, probe_tol = _need(spec, "psnr_tol_db", "probe_tol_points")
    lines: dict[tuple, list[dict]] = {}
    for split, r in _m3_arms(c):
        if r.get("R") is not None:
            lines.setdefault((split, r.get("arm"), r.get("w")), []).append(r)
    checks = (("psnr", lambda r: _mean(r.get("psnr")), psnr_tol, +1),
              ("probe_top1_points", lambda r: None if not r.get("probe") else 100 * r["probe"]["top1"], probe_tol, +1),
              ("latent_mse", lambda r: _mean((r.get("latent_mse") or {}).get("empirical")), 0.0, -1))
    violations, checked = [], 0
    for key, rs in lines.items():
        rs = sorted(rs, key=lambda r: r["R"])
        for metric, get, tol, sign in checks:
            vals = [get(r) for r in rs]
            if len(rs) < 2 or any(v is None for v in vals):
                continue
            checked += 1
            violations += [f"{key[0]} {key[1]} w={key[2]} {metric}: R {a['R']:g}->{b['R']:g}: {va:.4g}->{vb:.4g}"
                           for a, b, va, vb in zip(rs, rs[1:], vals, vals[1:])
                           if sign * (vb - va) < -tol - 1e-12]          # a drop of exactly tol is allowed
    if not checked:
        raise _NotEvaluated("no lossy arm evaluated at two or more rates")
    return len(violations), "0 violations", not violations, {"lines_checked": checked, "violations": violations}


def _svp_b0_recon(c, spec):
    psnr_min, psnr_max, lpips_max = _need(spec, "psnr_min", "psnr_max", "lpips_max")
    m = c["m3"].get("train_dev")
    b0 = [r for r in (m["data"].get("arms") or {}).values() if r.get("arm") == "B0"] if m else []
    if not b0 or _mean(b0[0].get("psnr")) is None or _mean(b0[0].get("lpips")) is None:
        raise _NotEvaluated("needs B0 PSNR and LPIPS in recon_probe/train_dev/results.json")
    psnr, lpips = _mean(b0[0]["psnr"]), _mean(b0[0]["lpips"])
    ok = psnr_min <= psnr <= psnr_max and lpips <= lpips_max
    return (f"PSNR {psnr:.3f}, LPIPS {lpips:.4f}", f"PSNR in [{psnr_min:g}, {psnr_max:g}], LPIPS <= {lpips_max:g}",
            ok, {"psnr": psnr, "lpips": lpips})


def _svp_data_gates(c, spec):
    detail, missing = {}, []
    for split, path in c["gates"]["cache_verification"].items():
        if path.is_file():
            detail[f"cache {split} re-encode"] = bool(_read_json(path).get("passed"))
        else:
            missing.append(path.as_posix())
    diag = c["gates"]["stats_diagnostics"]
    if diag.is_file():
        detail["alignment peak (0, 0)"] = bool((_read_json(diag).get("alignment") or {}).get("pass"))
    else:
        missing.append(diag.as_posix())
    if missing:
        raise _NotEvaluated(f"missing {missing}")
    ok = all(detail.values())
    return f"{sum(detail.values())}/{len(detail)} passed", "all passed", ok, detail


SVP_CRITERIA = {"dit_loss_drop": _svp_loss_drop, "dit_loss_plateau": _svp_loss_plateau,
                "gfid_memorisation": _svp_gfid_mem, "nn_memorisation": _svp_nn_mem,
                "allocation_identity_residual": _svp_alloc_residual, "latent_mse_vs_d_latent": _svp_latent_mse,
                "monotone_in_rate": _svp_monotone, "b0_reconstruction": _svp_b0_recon, "data_gates": _svp_data_gates}


def svp_verdict(c: dict, criteria) -> tuple[list[dict], list[str]]:
    """Evaluate cfg.svp.pass_criteria (name -> {rule, <threshold keys>}); unknown or unevaluable ones are listed."""
    rows, skipped = [], []
    if not isinstance(criteria, dict):
        return rows, [f"svp.pass_criteria is not a mapping: {criteria!r}"]
    for name, spec in criteria.items():
        fn = SVP_CRITERIA.get(name)
        if fn is None or not isinstance(spec, dict):
            skipped.append(f"{name}: no evaluator in 07_evaluate.py (known: {sorted(SVP_CRITERIA)})")
            continue
        try:
            value, threshold, ok, detail = fn(c, spec)
        except _NotEvaluated as e:
            skipped.append(f"{name}: {e}")
            continue
        rows.append({"criterion": name, "rule": spec.get("rule"), "value": value, "threshold": threshold,
                     "verdict": "PASS" if ok else "FAIL", "detail": detail})
    return rows, skipped


# --------------------------------------------------------------------------- figures
def _figures(c: dict, fig_dir: Path, logger) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed: no figures")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=120)
        plt.close(fig)
        made.append(fig_dir / name)

    specs = (("psnr", "PSNR (dB)", lambda r: _mean(r.get("psnr"))),
             ("rfid", "rFID", lambda r: (r.get("rfid") or {}).get("value")),
             ("probe", "probe top-1 (%)", lambda r: None if not r.get("probe") else 100 * r["probe"]["top1"]))
    for split, m in c["m3"].items():
        arms = m["data"].get("arms")
        if not isinstance(arms, dict):
            continue
        for key, ylabel, get in specs:
            series: dict[str, list] = {}
            refs = [r for r in arms.values() if r.get("R") is None and get(r) is not None]
            for r in arms.values():
                if r.get("R") is not None and get(r) is not None:
                    series.setdefault(r["arm"] + (f" w={r['w']:g}" if r.get("w") else ""), []).append((r["R"], get(r)))
            if not series:
                continue
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for name, pts in sorted(series.items()):
                ax.plot(*zip(*sorted(pts)), marker="o", label=name)
            for r in refs:
                ax.axhline(get(r), color="k", ls="--", lw=1, label=r["label"])
            ax.set(xlabel="R (nats/token)", ylabel=ylabel, title=f"rate-{key} ({split})")
            ax.legend(fontsize=7)
            save(fig, f"rate_{key}_{split}.png")
    for kind in sorted({g["fid_kind"] for g in c["gen"] if "fid_kind" in g and "gfid" in g}):
        runs = [g for g in c["gen"] if g.get("fid_kind") == kind and "gfid" in g]
        refs = sorted({r for g in runs for r in g["gfid"]})
        width = 0.8 / len(refs)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for j, r in enumerate(refs):
            ax.bar([i + j * width for i in range(len(runs))], [g["gfid"].get(r, np.nan) for g in runs], width, label=r)
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(runs))],
                      [f"{g['label']}@{g['checkpoint']['step']}" for g in runs], rotation=30, ha="right", fontsize=7)
        ax.set(ylabel="gFID", title=f"{kind} per arm (no CFG)")
        ax.legend(fontsize=7)
        save(fig, f"gfid_{kind}.png")
    curves = {k: d["curve"] for k, d in c["dit"].items() if d["curve"]}
    if curves:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for name, curve in sorted(curves.items()):
            ax.plot(*zip(*curve), label=name)
        ax.set(xlabel="step", ylabel="loss (mse + cos)", yscale="log", title="DiT training loss")
        ax.legend(fontsize=7)
        save(fig, "dit_loss.png")
    return made


def cmd_report(args, cfg: dict) -> int:
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "report")
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        c = collect(cfg)
        name, dataset = cfg["experiment"]["name"], cfg["dataset"].get("name")
        lines = [f"# RW-VAE v2.1 report: {name}", "",
                 f"Generated {_utc_now()} by scripts/07_evaluate.py report (git {runtime.git_commit()}); dataset "
                 f"`{dataset}`, experiment dir `{c['root'].as_posix()}`.", ""]
        if "svp" in cfg or dataset != "imagenet1k":
            lines += ["> **ENGINEERING / SANITY TEST -- NOT THESIS RESULTS.** "
                      + ("Sanity-validation pilot (SVP): a small random subset; the generator is deliberately "
                         "overfit. " if "svp" in cfg else "")
                      + f"Numbers on dataset `{dataset}` must never be mixed into ImageNet-1k tables.", ""]
        lines += ["## M3 generator-free results (05_eval_recon_probe)", ""]
        for split, m in c["m3"].items():
            lines += _m3_lines(split, m)
        if not c["m3"]:
            lines += ["(no recon_probe/*/results.json)", ""]
        lines += ["## Allocations (04_solve_channel)", ""] + _alloc_lines(c["allocations"])
        lines += ["", "## Generation (07 generate): gFID grouped by FID kind"]
        gen_lines, gen_agg = _gen_lines(c["gen"], c["root"])
        lines += gen_lines or ["", "(no gen/**/results.json)"]
        dit_rows = _dit_rows(c["dit"])
        lines += ["", "## DiT training loss (06_train_dit, metrics.jsonl window means)", ""]
        lines += _md_table(["run", "exploratory", "repeat", "logged", "last step", "first", "last", "min"],
                           [[r["run"], r["exploratory"], r["repeat"], r["logged"], r["last_step"], r["first"],
                             r["last"], r["min"]] for r in dit_rows]) if dit_rows else ["(no dit/*/arm.json)"]
        verdict, skipped = [], []
        if "svp" in cfg:
            verdict, skipped = svp_verdict(c, (cfg.get("svp") or {}).get("pass_criteria"))
            lines += ["", "## SVP verdict", "", "Criteria of svp.pass_criteria (fixed before the run); worst case "
                      "over runs/arms; memorisation criteria use B0 at the final checkpoint of each 06 run.", ""]
            lines += _md_table(["criterion", "value", "threshold", "verdict", "rule"],
                               [[v["criterion"], v["value"], v["threshold"], v["verdict"], v["rule"]] for v in verdict]) \
                if verdict else ["(no criterion evaluated)"]
            if skipped:
                lines += ["", "Not evaluated:", ""] + [f"- {s}" for s in skipped]
            overall = ("FAIL" if any(v["verdict"] == "FAIL" for v in verdict) else
                       "PASS" if verdict and not skipped else "INCOMPLETE")
            lines += ["", f"**SVP overall: {overall}**"]
        figs = _figures(c, out_dir / "figures", run.logger)
        if figs:
            lines += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
        (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.save_json("report.json", {"m3_files": {k: v["file"] for k, v in c["m3"].items()},
                                      "gen_files": [g["file"] for g in c["gen"]], "gfid_aggregates": gen_agg,
                                      "dit": dit_rows, "svp_verdict": verdict, "svp_not_evaluated": skipped,
                                      "figures": figs, "git_commit": runtime.git_commit()})
        run.finish("ok", report=out_dir / "REPORT.md", n_gen=len(c["gen"]), n_m3=len(c["m3"]),
                   svp={v["criterion"]: v["verdict"] for v in verdict})
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
