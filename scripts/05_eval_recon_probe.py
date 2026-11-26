"""05 -- generator-free evaluation of B0 and the allocation grid (plan v2.1 Sec. 3.3 step 4, 5.1, 5.3, 6 M3).

For every selected arm (B0 plus each (rule, R, w) of the grid, filtered by --arms/--R/--w) on ONE eval split:
  reconstruction  z' = channel(mu, eps), x_hat = AdaptedDecoder(z') (identity for B0, cfg.decoder.mode otherwise,
                  or a 05b fine-tuned decoder via --decoder-ckpt); per-image PSNR of to_uint8(x_hat) vs to_uint8(x),
                  LPIPS (VGG, [-1, 1] inputs), rFID of the uint8 reconstructions vs the uint8 originals;
  latent          empirical LMMSE latent MSE mean_tokens ||mu - channel.lmmse(z')||^2 vs the closed form D_latent (P7),
                  semantic retention tr(G_ret K)/tr(K) (closed form from stats), DCT radial band power ratio z'/mu;
  linear probe    T1: multinomial logistic regression on pooled_features(z') (train: probe.train_split, test: eval split);
  comparisons     paired bootstrap at each R (same images, same noise): B3(w)-B2, B3(w)-B1, B2-B1 on PSNR and probe;
  gates           probe go/no-go (some (R, w) with B3-B2 >= go_no_go.probe_diff_points, CI excluding 0) and
                  monotonicity in R per rule: PSNR / probe top-1 non-decreasing, latent MSE non-increasing (tolerances
                  from svp.pass_criteria.monotone_in_rate if present, else strict; violations reported, never fatal).
The images are read and decoded ONCE: every batch serves all arms. Every arm sees the SAME channel noise
(common random numbers): eps_i = sample_noise(seed = noise_seed(eval.noise_seed, 0, i)) for split row i; the probe
uses noise_seed(probe.noise_seed, 0, i) on its training rows and noise_seed(probe.noise_seed, 1, i) on the eval rows.
Use --split train_dev to choose R*, w* (plan Sec. 8) and --split val only once, for the final report.

Inputs   manifest (01), latent caches of the eval split and of eval.probe.train_split (02), stats.npz (03),
         allocations (04; B0 is built from the stats), the tokenizer checkpoint, optional 05b decoders.
Outputs  (default <out> = <run_root>/<name>/recon_probe/<split>)
         results.json   per-arm metrics with CIs, paired comparisons, gates, provenance (plan Sec. 3.2)
         results.md     the same as tables
         per_image.npz  rows, labels, image ids and per-arm psnr / lpips / latent_mse / probe_correct
         ref_fid.npz    Inception statistics of the original eval images (reused by a re-run with --overwrite
                        when manifest, split, rows and feature extractor match)
         figures/rate_{psnr,rfid,probe}.png  with --figures (one line per rule / w, B0 dashed)
         config.yaml, log.txt, metrics.jsonl, summary.json  (rwvae.runtime.Run)
Eval rows: all rows of the split, or with eval.max_images a class-balanced prefix (the k-th image of every class
         before any (k+1)-th; manifest rows are class-major, so the first N rows would cover only a few classes).
rFID with fewer than 2048 images uses singular covariances: it is flagged as "small-n rFID, indicative only".

Examples (common flags of rwvae.runtime.add_common_args apply: --set, --out, --device, --overwrite, ...)
  ImageNet-1k (GPU node), rule selection on train_dev, then the final report on val:
    python scripts/05_eval_recon_probe.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --split train_dev --figures
    python scripts/05_eval_recon_probe.py ... --split val --arms B0,B2,B3 --R 22 --w 2
  SVP (random ImageNet subset, overfit pilot; configs/svp.yaml overlay):
    python scripts/05_eval_recon_probe.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/server_a800.yaml --config configs/svp.yaml --figures
  Local Imagenette (engineering test only):
    python scripts/05_eval_recon_probe.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/local_imagenette.yaml \
        --max-images 200 --R 22 --w 2
  With a 05b decoder:  ... --decoder-ckpt B3_r2_R22_w2=runs/<name>/decoders/B3_r2_R22_w2/decoder.safetensors
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from rwvae import allocation as alloc_lib
from rwvae import datasets, evaluation, integration, runtime
from rwvae.channel import RWChannel, noise_seed, sample_noise
from rwvae.decoder_adapter import AdaptedDecoder

FID_FULL_RANK = 2048   # Inception pool3 dimension: fewer images give a singular covariance
NOISE_PROTOCOL = {
    "recon": "eps_i = sample_noise(seed=noise_seed(eval.noise_seed, 0, i)), i = split row; identical for every arm",
    "probe_train": "noise_seed(eval.probe.noise_seed, 0, i) on the probe training rows",
    "probe_test": "noise_seed(eval.probe.noise_seed, 1, i) on the eval rows",
}


def build_parser():
    p = _common.make_parser("Generator-free reconstruction / latent / linear-probe evaluation (plan M3).")
    p.add_argument("--split", default=None, help="eval split (default eval.split); train_dev selects rules, val is final")
    p.add_argument("--arms", default=None, help="comma list of arms, e.g. B0,B2,B3 (default: all arms of the config)")
    p.add_argument("--R", default=None, help="comma list of rates in nats/token (default grid.R_nats)")
    p.add_argument("--w", default=None, help="comma list of r2 weights (default grid.w)")
    p.add_argument("--max-images", type=int, default=None, help="cap on eval images (sets eval.max_images)")
    p.add_argument("--decoder-ckpt", action="append", default=[], metavar="ARMLABEL=PATH",
                   help="05b fine-tuned decoder for one arm label, e.g. B3_r2_R22_w2=.../decoder.safetensors; repeatable")
    p.add_argument("--no-lpips", action="store_true", help="skip LPIPS (sets eval.lpips=false)")
    p.add_argument("--no-rfid", action="store_true", help="skip rFID (sets eval.rfid=false)")
    p.add_argument("--skip-probe", action="store_true", help="skip the linear probe")
    p.add_argument("--skip-recon", action="store_true", help="skip the image pass (PSNR / LPIPS / rFID)")
    p.add_argument("--figures", action="store_true", help="write rate-PSNR / rate-rFID / rate-probe plots")
    return p


def floats(text):
    return None if text is None else [float(v) for v in text.split(",") if v.strip()]


@dataclass
class Arm:
    arm: str
    rule: str
    R: float | None
    w: float | None
    label: str
    alloc: alloc_lib.Allocation
    channel: RWChannel | None = None
    adapter: AdaptedDecoder | None = None
    decoder: dict = field(default_factory=dict)
    psnr: list = field(default_factory=list)
    lmse: list = field(default_factory=list)
    lpips: evaluation.LPIPSMeter | None = None
    fid: evaluation.FIDStatistics | None = None
    dct_sum: torch.Tensor | None = None
    probe_train: list = field(default_factory=list)
    probe_test: list = field(default_factory=list)


def select_arms(cfg: dict, arms: list[str], Rs: list[float], ws: list[float]) -> list[Arm]:
    """B0 once; equal_snr / variance at every R; r2 at every (R, w). Allocations from 04 (B0 on the fly)."""
    stats_fp = _common.load_stats(cfg).fingerprint()
    out = []
    for arm in arms:
        rule = _common.arm_rule(cfg, arm)
        points = [(None, None)] if rule == "none" else \
            [(R, w) for R in Rs for w in ws] if rule == "r2" else [(R, None) for R in Rs]
        for R, w in points:
            alloc = _common.load_allocation(cfg, arm, R, w)
            solved_from = alloc.meta.get("stats_fingerprint")
            if solved_from is not None and solved_from != stats_fp:
                raise ValueError(f"allocation {alloc.name} was solved from stats {solved_from}, but "
                                 f"{_common.stats_path(cfg)} has {stats_fp}: re-run scripts/04_solve_channel.py")
            out.append(Arm(arm, rule, R, w, _common.arm_label(arm, rule, R, w), alloc))
    return out


def balanced_prefix(labels: list[int], n: int | None) -> list[int]:
    """All rows (n None), else the first n rows in (rank within class, row) order: class-balanced, deterministic."""
    if n is None or n >= len(labels):
        return list(range(len(labels)))
    seen: dict[int, int] = {}
    rank = []
    for y in labels:
        rank.append(seen.get(y, 0))
        seen[y] = rank[-1] + 1
    return sorted(sorted(range(len(labels)), key=lambda i: (rank[i], i))[:n])


def per_class_rows(labels: list[int], per_class: int) -> tuple[list[int], dict]:
    """The first `per_class` rows of every class (row order); classes with fewer images give all of theirs."""
    taken: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        if len(taken.setdefault(y, [])) < per_class:
            taken[y].append(i)
    short = {y: len(r) for y, r in taken.items() if len(r) < per_class}
    return sorted(i for r in taken.values() for i in r), {"per_class_requested": per_class, "classes_short": short}


def rows_sha256(rows: list[int]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def iter_rows(cache: datasets.LatentCache, rows: list[int], batch_size: int):
    """(mu [b, C, H, W] f32, labels [b], rows [b]) for the given split rows, read straight from the cache."""
    for s in range(0, len(rows), batch_size):
        items = [cache.get(i) for i in rows[s:s + batch_size]]
        yield (torch.stack([m for m, _, _ in items]), torch.tensor([y for _, y, _ in items]),
               torch.tensor([i for _, _, i in items]))


def noise(index, shape, base: int, epoch: int, device) -> torch.Tensor:
    return sample_noise(shape, [noise_seed(base, epoch, int(i)) for i in index], device=device)


class Stats:
    """Bootstrap helpers with the cfg.eval.bootstrap protocol; non-finite values are dropped and counted."""

    def __init__(self, b: dict):
        self.n_boot, self.seed, self.alpha = int(b["n_boot"]), int(b["seed"]), float(b["alpha"])

    def ci(self, values) -> dict:
        v = np.asarray(values, dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size < 2:
            return {"mean": float(finite.mean()) if finite.size else None, "lo": None, "hi": None,
                    "n": int(finite.size), "n_dropped_nonfinite": int(v.size - finite.size)}
        out = evaluation.bootstrap_mean_ci(finite, self.n_boot, self.seed, self.alpha)
        return {**out, "n_dropped_nonfinite": int(v.size - finite.size)}

    def paired(self, a, b) -> dict:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < 2:
            return {"mean": None, "lo": None, "hi": None, "n": int(keep.sum()), "n_dropped_nonfinite": int((~keep).sum())}
        out = evaluation.paired_bootstrap_diff(a[keep], b[keep], self.n_boot, self.seed, self.alpha)
        return {**out, "n_dropped_nonfinite": int((~keep).sum())}


def parse_decoder_ckpts(items: list[str], labels: set[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not path:
            raise ValueError(f"--decoder-ckpt expects ARMLABEL=PATH, got {item!r}")
        if label not in labels:
            raise ValueError(f"--decoder-ckpt label {label!r} is not among the evaluated arms {sorted(labels)}")
        out[label] = _common.resolve(path)
    return out


def build_adapter(cfg: dict, arm: Arm, tokenizer, ckpt: Path | None, device) -> tuple[AdaptedDecoder, dict]:
    mode = "identity" if arm.rule == "none" else cfg["decoder"]["mode"]
    kw = {"A": arm.alloc.A, "m": arm.alloc.m} if mode == "lmmse" else {}
    adapter = AdaptedDecoder(tokenizer.post_quant_conv, tokenizer.decoder, mode=mode, **kw).to(device).eval()
    if ckpt is None:
        # analytic adapters differ only in post_quant_conv: share the frozen decoder (one copy for all arms)
        adapter.decoder = tokenizer.decoder
        return adapter, {"mode": mode, "checkpoint": None}
    meta = adapter.load_weights(ckpt)
    adapter.to(device).eval()
    if meta.get("allocation_fingerprint") != arm.alloc.fingerprint():
        raise ValueError(f"{ckpt} was fine-tuned for allocation {meta.get('allocation_fingerprint')} "
                         f"({meta.get('arm_label')}), not {arm.alloc.fingerprint()} ({arm.label})")
    return adapter, {"mode": "finetuned", "base_mode": adapter.base_mode, "checkpoint": ckpt,
                     "sha256": runtime.sha256_file(ckpt), "meta": meta}


def ref_fid_stats(out_dir: Path, key: dict, inception, device, batch: int):
    """(FIDStatistics to fill, None) or (None, (mu, sigma, n)) when ref_fid.npz matches `key`."""
    path = out_dir / "ref_fid.npz"
    if path.is_file():
        mu, sigma, n, meta = evaluation.load_fid_stats(path)
        if meta.get("key") == key:
            return None, (mu, sigma, n)
    return evaluation.FIDStatistics(inception, device, batch), None


# --------------------------------------------------------------------------- passes
@torch.no_grad()
def image_pass(run, cfg, arms, manifest, cache, rows, device, ref_fid, lpips_on, rfid_on):
    """One pass over the eval images: every batch is decoded once per arm, never re-read."""
    e = cfg["eval"]
    ds = datasets.ImageLatentPairDataset(manifest, cache, image_size=int(cfg["dataset"]["image_size"]), indices=rows)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(e["batch_size"]), shuffle=False,
                                         num_workers=int(e["num_workers"]))
    seen, base = [], int(e["noise_seed"])
    for step, (images, mu, _, index) in enumerate(loader):
        images, mu = images.to(device), mu.to(device)
        x_u8 = evaluation.to_uint8(images)
        if ref_fid is not None:
            ref_fid.update(x_u8)
        eps = noise(index, mu.shape[1:], base, 0, device)
        for a in arms:
            x_hat = a.adapter(a.channel(mu, noise=eps))
            u8 = evaluation.to_uint8(x_hat)
            a.psnr.append(evaluation.psnr_per_image(x_u8, u8).cpu().numpy())
            if lpips_on:
                a.lpips.update(x_hat, images)
            if rfid_on:
                a.fid.update(u8)
        seen += index.tolist()
        if step % 10 == 0 or len(seen) == len(rows):
            run.log_metrics(step, stage="image_pass", images_done=len(seen), images_total=len(rows))
    if seen != rows:
        raise RuntimeError("image pass did not visit the eval rows exactly once in order")


@torch.no_grad()
def latent_pass(cfg, arms, cache, rows, device, probe_on) -> torch.Tensor:
    """Latent MSE, DCT spectra (first eval.dct_images rows) and probe test features; returns the mu spectrum sum."""
    e = cfg["eval"]
    n_dct, grid = int(e["dct_images"]), int(e["probe"]["pool_grid"])
    mu_dct, done = None, 0
    for mu, _, index in iter_rows(cache, rows, int(e["batch_size"])):
        mu = mu.to(device)
        eps = noise(index, mu.shape[1:], int(e["noise_seed"]), 0, device)
        eps_probe = noise(index, mu.shape[1:], int(e["probe"]["noise_seed"]), 1, device) if probe_on else None
        k = max(0, min(mu.shape[0], n_dct - done))
        if k:
            p = evaluation.dct2_power(mu[:k]) * k
            mu_dct = p if mu_dct is None else mu_dct + p
        for a in arms:
            z = a.channel(mu, noise=eps)
            a.lmse.append((mu - a.channel.lmmse(z)).double().square().sum(1).mean((1, 2)).cpu().numpy())
            if k:
                p = evaluation.dct2_power(z[:k]) * k
                a.dct_sum = p if a.dct_sum is None else a.dct_sum + p
            if probe_on:
                a.probe_test.append(evaluation.pooled_features(a.channel(mu, noise=eps_probe), grid).cpu())
        done += k
    return mu_dct


@torch.no_grad()
def probe_train_pass(cfg, arms, cache, rows, device) -> None:
    e = cfg["eval"]
    for mu, _, index in iter_rows(cache, rows, int(e["batch_size"])):
        mu = mu.to(device)
        eps = noise(index, mu.shape[1:], int(e["probe"]["noise_seed"]), 0, device)
        for a in arms:
            a.probe_train.append(evaluation.pooled_features(a.channel(mu, noise=eps), int(e["probe"]["pool_grid"])).cpu())


# --------------------------------------------------------------------------- reporting
def comparisons(arms: list[Arm], per: dict, st: Stats) -> list[dict]:
    """Paired differences at each R: r2(w) - variance, r2(w) - equal_snr, variance - equal_snr."""
    by = {(a.rule, a.R, a.w): a for a in arms}
    out = []
    for R in sorted({a.R for a in arms if a.R is not None}):
        pairs = [(by.get(("r2", R, w)), by.get((b, R, None))) for w in sorted({a.w for a in arms if a.rule == "r2"})
                 for b in ("variance", "equal_snr")] + [(by.get(("variance", R, None)), by.get(("equal_snr", R, None)))]
        for a, b in pairs:
            if a is None or b is None:
                continue
            row = {"R": R, "w": a.w, "a": a.label, "b": b.label, "rules": f"{a.rule}-{b.rule}"}
            if "psnr" in per[a.label]:
                row["psnr_diff_db"] = st.paired(per[a.label]["psnr"], per[b.label]["psnr"])
            if "probe_correct" in per[a.label]:
                d = st.paired(per[a.label]["probe_correct"], per[b.label]["probe_correct"])
                row["probe_top1_diff_points"] = {k: (v * 100 if k in ("mean", "lo", "hi") and v is not None else v)
                                                 for k, v in d.items()}
            out.append(row)
    return out


def probe_gate(pairs: list[dict], cfg: dict) -> dict:
    thr = float(cfg["go_no_go"]["probe_diff_points"])
    cands = [{"R": p["R"], "w": p["w"], **{k: p["probe_top1_diff_points"][k] for k in ("mean", "lo", "hi")}}
             for p in pairs if p["rules"] == "r2-variance" and "probe_top1_diff_points" in p]
    for c in cands:
        c["pass"] = bool(c["mean"] is not None and c["lo"] is not None and c["mean"] >= thr and c["lo"] > 0)
    if not cands:
        return {"evaluated": False, "reason": "needs probe results for both r2 (B3) and variance (B2) arms"}
    return {"evaluated": True, "threshold_points": thr, "rule": "B3(w) - B2 probe top-1 >= threshold and CI lo > 0",
            "candidates": cands, "pass": any(c["pass"] for c in cands)}


def monotonicity(results: dict[str, dict], cfg: dict) -> dict:
    """More rate should not hurt within one (arm, w) line: PSNR and probe top-1 non-decreasing, latent MSE
    non-increasing in R. A step worse by more than the tolerance is a violation (reported, never fatal); the
    tolerances come from svp.pass_criteria.monotone_in_rate when the config has it, else 0 (strict)."""
    t = ((cfg.get("svp") or {}).get("pass_criteria") or {}).get("monotone_in_rate") or {}
    metrics = (("psnr", +1, float(t.get("psnr_tol_db", 0.0)), lambda r: (r.get("psnr") or {}).get("mean")),
               ("probe_top1", +1, float(t.get("probe_tol_points", 0.0)) / 100,
                lambda r: (r.get("probe") or {}).get("top1")),
               ("latent_mse", -1, 0.0, lambda r: r["latent_mse"]["empirical"]["mean"]))
    lines: dict[tuple, list] = {}
    for r in results.values():
        if r["R"] is not None:
            lines.setdefault((r["arm"], r["w"]), []).append(r)
    out = {"tolerances": {m: tol for m, _, tol, _ in metrics}, "violations": [], "checked": []}
    for (arm, w), rs in sorted(lines.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        rs = sorted(rs, key=lambda r: r["R"])
        for metric, sign, tol, get in metrics:
            vals = [get(r) for r in rs]
            if len(rs) < 2 or any(v is None for v in vals):
                continue
            out["checked"].append({"arm": arm, "w": w, "metric": metric, "R": [r["R"] for r in rs], "values": vals})
            for lo, hi, vlo, vhi in zip(rs, rs[1:], vals, vals[1:]):
                if sign * (vhi - vlo) < -tol - 1e-12:      # 1e-12: a drop of exactly tol is allowed
                    out["violations"].append({"arm": arm, "w": w, "metric": metric, "R_from": lo["R"], "R_to": hi["R"],
                                              "from": vlo, "to": vhi})
    out["monotone"] = not out["violations"]
    return out


def fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.{nd}f}"
    return str(v)


def ci_text(c: dict | None, nd=2, scale=1.0) -> str:
    if not c or c.get("mean") is None:
        return "-"
    if c.get("lo") is None:
        return fmt(c["mean"] * scale, nd)
    return f"{c['mean'] * scale:.{nd}f} [{c['lo'] * scale:.{nd}f}, {c['hi'] * scale:.{nd}f}]"


def markdown(res: dict) -> str:
    m = res["meta"]
    L = [f"# 05 generator-free evaluation -- split `{m['split']}` ({m['split_role']})", "",
         f"- images: {m['n_images']} (rows sha256 {m['rows_sha256'][:12]}), manifest {m['manifest_fingerprint'][:12]}, "
         f"stats {m['stats_fingerprint']}, git {m['git_commit']}",
         f"- noise: {NOISE_PROTOCOL['recon']}",
         f"- rFID: {m['rfid_note']}", "",
         "| arm | R | w | bpp | decoder | PSNR dB [95% CI] | LPIPS | rFID | latent MSE emp / pred | sem. ret. "
         "| probe top-1 % [CI] | n |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["arms"].values():
        lm = r["latent_mse"]
        L.append(f"| {r['label']} | {fmt(r['R'], 0)} | {fmt(r['w'], 1)} | {fmt(r['bpp'], 4)} | {r['decoder']['mode']} "
                 f"| {ci_text(r.get('psnr'))} | {ci_text(r.get('lpips'), 4)} | {fmt((r.get('rfid') or {}).get('value'), 2)} "
                 f"| {fmt(lm['empirical']['mean'], 4)} / {fmt(lm['predicted'], 4)} | {fmt(r['semantic_retention'])} "
                 f"| {ci_text(r.get('probe', {}).get('top1_ci'), 1, 100)} | {r['n_images']} |")
    L += ["", "## DCT radial band power ratio z'/mu (low -> high frequency)", "", "| arm | ratios |", "|---|---|"]
    L += [f"| {r['label']} | {', '.join(f'{x:.3f}' for x in r['dct_band_ratio'])} |" for r in res["arms"].values()
          if r.get("dct_band_ratio") is not None]
    L += ["", "## Paired comparisons (same images, same noise)", "",
          "| R | w | a - b | PSNR diff dB [CI] | probe diff points [CI] |", "|---|---|---|---|---|"]
    L += [f"| {fmt(p['R'], 0)} | {fmt(p['w'], 1)} | {p['a']} - {p['b']} | {ci_text(p.get('psnr_diff_db'))} "
          f"| {ci_text(p.get('probe_top1_diff_points'), 1)} |" for p in res["paired"]]
    g, mono = res["gates"]["probe"], res["gates"]["monotonicity"]
    L += ["", "## Gates", ""]
    if g["evaluated"]:
        L.append(f"- probe go/no-go ({g['rule']}, threshold {g['threshold_points']} points): "
                 f"**{'PASS' if g['pass'] else 'FAIL'}**")
        L += [f"  - R={fmt(c['R'], 0)} w={fmt(c['w'], 1)}: {fmt(c['mean'], 2)} [{fmt(c['lo'], 2)}, {fmt(c['hi'], 2)}]"
              f" -> {'pass' if c['pass'] else 'fail'}" for c in g["candidates"]]
    else:
        L.append(f"- probe go/no-go: not evaluated ({g['reason']})")
    L.append(f"- monotonicity in R: {'ok' if mono['monotone'] else str(len(mono['violations'])) + ' violation(s)'}")
    L += [f"  - {v['arm']} w={v['w']} {v['metric']}: R {fmt(v['R_from'], 0)} -> {fmt(v['R_to'], 0)}: "
          f"{fmt(v['from'])} -> {fmt(v['to'])}" for v in mono["violations"]]
    if m["dropped"]:
        L += ["", "## Dropped / capped", ""] + [f"- {d}" for d in m["dropped"]]
    return "\n".join(L) + "\n"


def figures(out_dir: Path, results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("psnr", "PSNR (dB)", lambda r: (r.get("psnr") or {}).get("mean")),
             ("rfid", "rFID", lambda r: (r.get("rfid") or {}).get("value")),
             ("probe", "probe top-1 (%)", lambda r: None if not r.get("probe") else 100 * r["probe"]["top1"])]
    (out_dir / "figures").mkdir(exist_ok=True)
    written = []
    for name, ylabel, get in specs:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        lines: dict[str, list] = {}
        for r in results.values():
            if r["R"] is None:
                if get(r) is not None:
                    ax.axhline(get(r), color="k", ls="--", lw=1, label=r["label"])
            elif get(r) is not None:
                key = f"{r['arm']} {r['rule']}" + ("" if r["w"] is None else f" w={r['w']:g}")
                lines.setdefault(key, []).append((r["R"], get(r)))
        for key, pts in sorted(lines.items()):
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="o", label=key)
        if not lines and not ax.lines:
            plt.close(fig)
            continue
        ax.set_xlabel("nominal rate R (nats/token)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "figures" / f"rate_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.relative_to(out_dir).as_posix())
    return written


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    e = cfg["eval"]
    if args.max_images is not None:
        e["max_images"] = int(args.max_images)
    if args.no_lpips:
        e["lpips"] = False
    if args.no_rfid:
        e["rfid"] = False
    split = args.split or e["split"]
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "recon_probe", split)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        log = run.logger
        runtime.seed_everything(int(cfg.get("seed", 0)))
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        recon_on, probe_on = not args.skip_recon, not args.skip_probe
        lpips_on, rfid_on = recon_on and bool(e["lpips"]), recon_on and bool(e["rfid"])
        sel_split = (cfg.get("preregistration") or {}).get("selection_split", "train_dev")
        role = "final report (val: never use it to choose R*, w*)" if split == "val" else \
            "rule selection" if split == sel_split else "diagnostic"
        log.info("eval split %s: %s", split, role)

        arm_names = args.arms.split(",") if args.arms else list(cfg["arms"])
        arms = select_arms(cfg, [a.strip() for a in arm_names if a.strip()],
                           floats(args.R) or [float(r) for r in cfg["grid"]["R_nats"]],
                           floats(args.w) or [float(w) for w in cfg["grid"]["w"]])
        ckpts = parse_decoder_ckpts(args.decoder_ckpt, {a.label for a in arms})
        log.info("%d arms: %s", len(arms), ", ".join(a.label for a in arms))

        manifest = _common.load_manifest(cfg)
        cache = _common.open_cache(cfg, split, manifest)
        labels = manifest.splits[split]["label"]
        rows = balanced_prefix(labels, e["max_images"])
        dropped = []
        if len(rows) < len(labels):
            dropped.append(f"eval.max_images={e['max_images']}: {len(rows)} of {len(labels)} {split} rows "
                           "(class-balanced prefix)")
        stats = _common.load_stats(cfg)
        K = None if stats.Sigma_expl is None else alloc_lib.whitened_relevance(stats.Sigma, stats.Sigma_expl)[0]
        C, H, W = cache.latent_shape
        S = int(cfg["dataset"]["image_size"])
        for a in arms:
            a.channel = RWChannel.from_allocation(a.alloc).to(device)

        # ---- image pass: PSNR / LPIPS / rFID
        tokenizer, ref, ref_key, ref_moments = None, None, None, None
        if recon_on:
            tokenizer = _common.make_tokenizer(cfg, device)
            lpips_model = integration.load_lpips(device) if lpips_on else None
            inception = integration.load_inception(device) if rfid_on else None
            fb = int(cfg["fid"]["inception_batch"])
            for a in arms:
                a.adapter, a.decoder = build_adapter(cfg, a, tokenizer, ckpts.get(a.label), device)
                a.lpips = evaluation.LPIPSMeter(lpips_model) if lpips_on else None
                a.fid = evaluation.FIDStatistics(inception, device, fb) if rfid_on else None
            if rfid_on:
                ref_key = {"manifest_fingerprint": manifest.fingerprint(), "split": split, "n": len(rows),
                           "rows_sha256": rows_sha256(rows), "image_size": S, "inception": type(inception).__name__}
                ref, ref_moments = ref_fid_stats(out_dir, ref_key, inception, device, fb)
                log.info("reference FID statistics: %s", "reused ref_fid.npz" if ref is None else "computed in this pass")
            image_pass(run, cfg, arms, manifest, cache, rows, device, ref, lpips_on, rfid_on)
            if ref is not None:
                ref_moments = ref.compute()
                evaluation.save_fid_stats(out_dir / "ref_fid.npz", *ref_moments,
                                          meta={"key": ref_key, "git_commit": runtime.git_commit(), "tf32": tf32})
        else:
            dropped.append("--skip-recon: no PSNR / LPIPS / rFID")

        # ---- latent pass (+ probe test features) and probe training features
        mu_dct = latent_pass(cfg, arms, cache, rows, device, probe_on)
        n_dct = min(int(e["dct_images"]), len(rows))
        probe_info = None
        if probe_on:
            p = e["probe"]
            pcache = _common.open_cache(cfg, p["train_split"], manifest)
            prow, short = per_class_rows(manifest.splits[p["train_split"]]["label"], int(p["train_per_class"]))
            if short["classes_short"]:
                dropped.append(f"probe: {len(short['classes_short'])} classes of {p['train_split']} have fewer than "
                               f"{p['train_per_class']} images (all of theirs used)")
            probe_train_pass(cfg, arms, pcache, prow, device)
            probe_info = {"train_split": p["train_split"], "n_train": len(prow), "train_rows_sha256": rows_sha256(prow),
                          **short, "test_split": split, "pool_grid": int(p["pool_grid"]),
                          "weight_decay": float(p["weight_decay"]), "max_iter": int(p["max_iter"]),
                          "noise_seed": int(p["noise_seed"])}
            ytr = torch.tensor([manifest.splits[p["train_split"]]["label"][i] for i in prow])
        yte = torch.tensor([labels[i] for i in rows])

        # ---- per-arm results
        st = Stats(e["bootstrap"])
        results, per = {}, {}
        n_classes = len(manifest.classes)
        for a in arms:
            per[a.label] = {"latent_mse": np.concatenate(a.lmse)}
            r = {"label": a.label, "arm": a.arm, "rule": a.rule, "R": a.R, "w": a.w, "allocation": a.alloc.name,
                 "allocation_fingerprint": a.alloc.fingerprint(),
                 "nominal_rate": alloc_lib.nominal_rate(a.alloc),
                 "bpp": None if a.R is None else alloc_lib.bpp_equivalent(a.R, tokens=H * W, pixels=S * S),
                 "decoder": a.decoder or {"mode": "-"}, "n_images": len(rows),
                 "latent_mse": {"empirical": st.ci(per[a.label]["latent_mse"]),
                                "predicted": alloc_lib.latent_mse_lmmse(a.alloc)},
                 "semantic_retention": None if K is None else alloc_lib.semantic_retention(a.alloc, stats.Sigma, K),
                 "dct_band_ratio": None if mu_dct is None else
                 (evaluation.radial_band_power(a.dct_sum, int(e["dct_bands"]))
                  / evaluation.radial_band_power(mu_dct, int(e["dct_bands"]))).tolist(),
                 "dct_images": n_dct}
            lm = r["latent_mse"]
            lm["rel_err"] = None if lm["predicted"] == 0 else (lm["empirical"]["mean"] - lm["predicted"]) / lm["predicted"]
            if recon_on:
                per[a.label]["psnr"] = np.concatenate(a.psnr)
                r["psnr"] = st.ci(per[a.label]["psnr"])
                if r["psnr"]["n_dropped_nonfinite"]:
                    dropped.append(f"{a.label}: {r['psnr']['n_dropped_nonfinite']} images with PSNR = inf "
                                   "(identical uint8) left out of the PSNR mean / CI")
            if lpips_on:
                per[a.label]["lpips"] = a.lpips.values()
                r["lpips"] = st.ci(per[a.label]["lpips"])
            if rfid_on:
                mu_a, s_a, n_a = a.fid.compute()
                try:
                    value, info = evaluation.frechet_distance(mu_a, s_a, ref_moments[0], ref_moments[1], return_info=True)
                    r["rfid"] = {"value": value, **info}
                except ValueError as err:   # recorded, not hidden: a singular small-n product can fail sqrtm
                    log.warning("rFID of %s failed: %s", a.label, err)
                    r["rfid"] = {"value": None, "error": str(err)}
                r["rfid"].update(n=n_a, feature_dim=int(mu_a.shape[0]),
                                 small_n=bool(n_a < max(FID_FULL_RANK, mu_a.shape[0])))
            if probe_on:
                pr = evaluation.linear_probe(torch.cat(a.probe_train), ytr, torch.cat(a.probe_test), yte,
                                             num_classes=n_classes, weight_decay=float(e["probe"]["weight_decay"]),
                                             max_iter=int(e["probe"]["max_iter"]), device=device)
                per[a.label]["probe_correct"] = pr.pop("correct")
                r["probe"] = {**pr, "top1_ci": st.ci(per[a.label]["probe_correct"].astype(np.float64))}
                if pr["n_iter"] >= int(e["probe"]["max_iter"]):
                    log.warning("probe of %s hit max_iter=%d", a.label, pr["n_iter"])
            results[a.label] = r
            log.info("%s: psnr %s, lpips %s, rfid %s, latent mse %.4g (pred %.4g), probe %s", a.label,
                     fmt((r.get("psnr") or {}).get("mean")), fmt((r.get("lpips") or {}).get("mean"), 4),
                     fmt((r.get("rfid") or {}).get("value"), 2), r["latent_mse"]["empirical"]["mean"],
                     r["latent_mse"]["predicted"], fmt((r.get("probe") or {}).get("top1")))
            run.log_metrics(None, stage="arm", label=a.label, psnr=(r.get("psnr") or {}).get("mean"),
                            probe_top1=(r.get("probe") or {}).get("top1"))

        pairs = comparisons(arms, per, st)
        small_n = rfid_on and len(rows) < FID_FULL_RANK
        res = {
            "meta": {
                "script": "05_eval_recon_probe.py", "split": split, "split_role": role, "n_images": len(rows),
                "rows_sha256": rows_sha256(rows), "max_images": e["max_images"], "image_size": S,
                "manifest_fingerprint": manifest.fingerprint(),
                "cache": {"split": split, "dir": cache.cache_dir, "dtype": cache.dtype,
                          "index_sha256": runtime.sha256_file(cache.cache_dir / datasets.CACHE_INDEX),
                          "meta": cache.meta},
                "tokenizer": None if tokenizer is None else tokenizer.meta,
                "stats_fingerprint": stats.fingerprint(), "stats_file_sha256": runtime.sha256_file(_common.stats_path(cfg)),
                "seeds": {"seed": int(cfg.get("seed", 0)), "eval_noise_seed": int(e["noise_seed"]),
                          "probe_noise_seed": int(e["probe"]["noise_seed"]), "bootstrap": e["bootstrap"]},
                "noise_protocol": NOISE_PROTOCOL, "normalisation": "none: channel and decoder act on raw mu / z'",
                "dtype": {"cache": cache.dtype, "compute": "float32", "statistics": "float64"},
                "tf32": tf32, "device": str(device), "git_commit": runtime.git_commit(),
                "flags": {"lpips": lpips_on, "rfid": rfid_on, "probe": probe_on, "recon": recon_on},
                "decoder_mode": cfg["decoder"]["mode"], "probe": probe_info,
                "rfid_note": ("small-n rFID, indicative only" if small_n else "full-rank rFID") if rfid_on else "skipped",
                "dropped": dropped,
            },
            "arms": results, "paired": pairs,
            "gates": {"probe": probe_gate(pairs, cfg), "monotonicity": monotonicity(results, cfg)},
        }
        if args.figures:
            res["meta"]["figures"] = figures(out_dir, results)
        run.save_json("results.json", res)
        (out_dir / "results.md").write_text(markdown(runtime.to_jsonable(res)), encoding="utf-8")
        flat = {f"{k}__{lab}": v for lab, d in per.items() for k, v in d.items()}
        with open(out_dir / "per_image.npz", "wb") as fh:
            np.savez(fh, rows=np.asarray(rows, dtype=np.int64), labels=yte.numpy(),
                     image_id=np.asarray([manifest.splits[split]["image_id"][i] for i in rows]), **flat)
        for d in dropped:
            log.warning("recorded: %s", d)
        gate = res["gates"]["probe"]
        run.finish("ok", split=split, n_images=len(rows), arms=[a.label for a in arms],
                   probe_gate=gate.get("pass") if gate["evaluated"] else None,
                   monotone=res["gates"]["monotonicity"]["monotone"])
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
