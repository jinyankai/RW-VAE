"""04 -- closed-form channels for every lossy arm on the (R, w) grid (plan v2.1 Sec. 2.2, 4, 6 M3).

For every arm of cfg.arms with a lossy rule (equal_snr = B1, variance = B2, r2 = B3), every R in
cfg.grid.R_nats and (rule r2 only) every w in cfg.grid.w: alloc = solve_allocation(stats, rule, w, R),
saved as <out>/<alloc.name>.npz; the B0 reference (rule none) is saved as none.npz. Per allocation the
closed-form diagnostics of ARCHITECTURE_V2 Sec. 2 are reported: bpp equivalent, active directions, theta,
LMMSE latent MSE D_latent, weighted distortion, semantic retention tr(G K)/tr K, direction relevance k_i,
keep / rate vectors and the exact-identity residuals (the run FAILS if one exceeds 1e-8; the Gaussian
mutual-information residual has the documented floor 1e-6, Sec. 2 "numerical check").

Rule separation and stability (plan Sec. 6 M3 go/no-go, thresholds cfg.go_no_go):
  overlap(R, w)   = retention_overlap(G_B3(R, w), G_B2(R)) on the full data (whitened retention matrices);
  split-half      = ||G_even - G_odd||_F, each half solved from its own statistics (even vs odd image
                    groups of group_moments.pt, the ridge of stats.npz) and whitened with its own Sigma;
  bootstrap       = ||G_b - G_full||_F over cfg.stats.bootstrap group-bootstrap replicates (seed cfg.seed);
  ratio(R, w)     = max over {B2, B3(w)} of split-half distance / ||G_B3(R, w) - G_B2(R)||_F.
  separable  iff  min overlap <= overlap_max;   no-go iff every overlap >= overlap_no_go;
  stable(R, w) iff ratio(R, w) <= stability_ratio_max;
  verdict: "go" if some (R, w) is separable and stable, "no-go" if no-go, else "inconclusive".
  (The probe gate probe_diff_points is evaluated by 05_eval_recon_probe.py, not here.)

Inputs   stats.npz and group_moments.pt of 03 (_common.stats_path(cfg) and its directory).
Outputs  <out> = --out or <run_root>/<name>/allocations  (05/06 read <run_root>/<name>/allocations)
         <name>.npz, none.npz  Allocation files (meta: solver meta + provenance of the statistics)
         allocations_summary.json  every number above, overlap matrices per R, stability, go/no-go
         allocations_summary.md    tables per R, overlap matrices, go/no-go verdict with numbers
         figures/*.png         with --figures: keep per KLT direction per R, overlap heat map (R x w)

Examples (common flags: --set, --out, --overwrite, ...)
  ImageNet-1k:
    python scripts/04_solve_channel.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --figures
  SVP (configs/svp.yaml overlay):
    python scripts/04_solve_channel.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/server_a800.yaml --config configs/svp.yaml --figures
  Local Imagenette (engineering test only):
    python scripts/04_solve_channel.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/local_imagenette.yaml
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rwvae import runtime
from rwvae import allocation as al
from rwvae.stats import GroupedMomentAccumulator, LatentStats, bootstrap_group_weights, split_half_groups

LOSSY = ("equal_snr", "variance", "r2")
IDENTITY_TOL = 1e-8       # relative residual of the exact identities (ARCHITECTURE_V2 Sec. 2)
MI_TOL = 1e-6             # documented floor of |gaussian_mutual_information - R| / max(1, R)
MOMENTS_FORMAT = "rwvae.group_moments/v1"


def build_parser():
    p = _common.make_parser("Closed-form channels B0-B3 on the (R, w) grid with stability and go/no-go (M3).")
    p.add_argument("--figures", action="store_true", help="write figures/*.png")
    return p


# --------------------------------------------------------------------------- inputs
def load_inputs(cfg: dict):
    """(stats, accumulator, provenance); the moments file must belong to stats.npz and the config ridge."""
    stats = _common.load_stats(cfg)
    stats_file = _common.stats_path(cfg)
    moments_file = stats_file.parent / "group_moments.pt"
    if not moments_file.is_file():
        raise FileNotFoundError(f"{moments_file} not found: run scripts/03_estimate_stats.py first")
    gm = torch.load(moments_file, map_location="cpu", weights_only=True)
    if gm.get("format") != MOMENTS_FORMAT or gm.get("stats_fingerprint") != stats.fingerprint():
        raise ValueError(f"{moments_file} (format {gm.get('format')}, stats {gm.get('stats_fingerprint')}) does not "
                         f"belong to {stats_file} (fingerprint {stats.fingerprint()}): re-run 03")
    if stats.Sigma_expl is None or "ridge" not in stats.meta:
        raise ValueError(f"{stats_file} has no Sigma_expl / ridge info: re-run 03")
    ridge = float(stats.meta["ridge"]["ridge_rel"])
    if ridge != float(cfg["stats"]["ridge_rel"]):
        raise ValueError(f"{stats_file} was computed at ridge_rel={ridge:g}, the config has "
                         f"stats.ridge_rel={cfg['stats']['ridge_rel']}: re-run 03 or fix the config")
    acc = GroupedMomentAccumulator(int(gm["dim"]), int(gm["num_groups"]), "cpu")
    acc.load_state_dict(gm["state"])
    prov = {"stats_file": stats_file, "stats_sha256": runtime.sha256_file(stats_file),
            "stats_fingerprint": stats.fingerprint(), "group_moments_file": moments_file,
            "group_moments_sha256": runtime.sha256_file(moments_file), "ridge_rel": ridge,
            "n_images": stats.n_images, "n_tokens": stats.n_tokens, "split": stats.meta.get("inputs", {}).get("split"),
            "manifest_fingerprint": stats.meta.get("inputs", {}).get("manifest_fingerprint"),
            "git_commit": runtime.git_commit(), "script": "04_solve_channel.py"}
    return stats, acc, int(gm["C"]), runtime.to_jsonable(prov)


def grid_points(cfg: dict) -> tuple[list[tuple[str, float, float | None]], dict[str, list[str]]]:
    """[(rule, R, w)] for every lossy rule in cfg.arms x grid, and rule -> arm labels."""
    arms: dict[str, list[str]] = {}
    for arm, spec in (cfg.get("arms") or {}).items():
        if spec["rule"] not in al.RULES:
            raise ValueError(f"arm {arm} has unknown rule {spec['rule']!r}; expected one of {al.RULES}")
        arms.setdefault(spec["rule"], []).append(arm)
    Rs = [float(R) for R in cfg["grid"]["R_nats"]]
    ws = [float(w) for w in cfg["grid"]["w"]]
    points = []
    for rule in LOSSY:
        if rule in arms:
            points += [(rule, R, w) for R in Rs for w in (ws if rule == "r2" else [None])]
    return points, arms


def name_of(rule: str, R, w) -> str:
    return _common.allocation_name(rule, R, w)


def label_of(arms: dict, rule: str, w) -> str:
    base = "/".join(arms.get(rule, [rule]))
    return f"{base}(w={w:g})" if rule == "r2" else base


# --------------------------------------------------------------------------- per allocation
def describe(alloc: al.Allocation, Sigma: np.ndarray, K: np.ndarray) -> dict:
    """Closed-form diagnostics of one allocation (ARCHITECTURE_V2 Sec. 2, P3-P7)."""
    res = al.identity_residuals(alloc, Sigma)
    if alloc.rule == "none":
        mi = None
    else:
        mi = abs(al.gaussian_mutual_information(alloc, Sigma) - alloc.R) / max(1.0, alloc.R)
    tr = float(np.trace(Sigma))
    d_lat = al.latent_mse_lmmse(alloc)
    return {
        "name": alloc.name, "rule": alloc.rule, "R": alloc.R, "w": alloc.w,
        "bpp_equivalent": math.inf if alloc.rule == "none" else al.bpp_equivalent(alloc.R),
        "n_active": int(np.sum(alloc.rates > 0)), "theta": alloc.theta,
        "latent_mse_lmmse": d_lat, "latent_mse_rel": d_lat / tr, "weighted_distortion": al.weighted_distortion(alloc),
        "semantic_retention": al.semantic_retention(alloc, Sigma, K),
        "direction_relevance": al.direction_relevance(alloc, Sigma, K),
        "s": alloc.s, "keep": alloc.keep, "rates": alloc.rates,
        "identity_residuals": res, "identity_residual_max": max(res.values()), "mi_residual": mi,
        "fingerprint": alloc.fingerprint(),
    }


def residual_failures(rec: dict) -> list[str]:
    bad = [f"{rec['name']}: {k} residual {v:.3g} > {IDENTITY_TOL:g}"
           for k, v in rec["identity_residuals"].items() if not v <= IDENTITY_TOL]
    if rec["mi_residual"] is not None and not rec["mi_residual"] <= MI_TOL:
        bad.append(f"{rec['name']}: Gaussian MI residual {rec['mi_residual']:.3g} > {MI_TOL:g}")
    return bad


# --------------------------------------------------------------------------- stability
def subset_stats(acc, C: int, ridge: float, groups, weights=None) -> LatentStats:
    g = np.asarray(list(groups))
    k = np.ones(g.size) if weights is None else np.asarray(weights, dtype=np.float64)
    n_img = int(np.round(np.sum(k * acc.images_per_group.numpy()[g])))
    return LatentStats.from_joint(acc.subset(g.tolist(), k), C, n_img).with_explained(ridge)


def retention(st: LatentStats, rule: str, R: float, w) -> np.ndarray:
    return al.retention_matrix_whitened(al.solve_allocation(st, rule, w, R), st.Sigma)


def stability(acc, C: int, ridge: float, points, G_full: dict, n_boot: int, seed: int, logger) -> dict:
    """Per allocation name: split-half distance and bootstrap distance distribution of G_ret."""
    halves = [subset_stats(acc, C, ridge, g) for g in split_half_groups(acc.num_groups)]
    weights = bootstrap_group_weights(acc.num_groups, n_boot, seed)
    boots = [subset_stats(acc, C, ridge, range(acc.num_groups), wt) for wt in weights]
    logger.info("stability: split halves of %s images, %d bootstrap replicates", [h.n_images for h in halves], n_boot)
    out = {}
    for rule, R, w in points:
        name = name_of(rule, R, w)
        d_half = float(np.linalg.norm(retention(halves[0], rule, R, w) - retention(halves[1], rule, R, w)))
        d_boot = np.array([np.linalg.norm(retention(b, rule, R, w) - G_full[name]) for b in boots])
        out[name] = {"split_half_distance": d_half, "bootstrap_mean": float(d_boot.mean()),
                     "bootstrap_p95": float(np.quantile(d_boot, 0.95)), "bootstrap_n": int(d_boot.size)}
    meta = {"split_half_images": [h.n_images for h in halves], "split_half_tokens": [h.n_tokens for h in halves],
            "bootstrap_replicates": n_boot, "bootstrap_seed": seed, "num_groups": acc.num_groups,
            "bootstrap_images": [b.n_images for b in boots], "ridge_rel": ridge,
            "distance": "Frobenius norm of the difference of whitened retention matrices, each solved and "
                        "whitened with its own statistics"}
    return {"per_allocation": out, "meta": meta}


# --------------------------------------------------------------------------- go / no-go
def go_no_go(cfg: dict, Rs, ws, G_full: dict, stab: dict) -> dict:
    gn = cfg["go_no_go"]
    thr = float(gn["stability_ratio_max"])
    points = []
    for R in Rs:
        for w in ws:
            b2, b3 = name_of("variance", R, None), name_of("r2", R, w)
            inter = float(np.linalg.norm(G_full[b3] - G_full[b2]))
            sh = {b: stab[b]["split_half_distance"] for b in (b2, b3)}
            bp = {b: stab[b]["bootstrap_p95"] for b in (b2, b3)}
            ratio = max(sh.values()) / inter if inter > 0 else math.inf
            points.append({"R": R, "w": w, "overlap_B3_B2": al.retention_overlap(G_full[b3], G_full[b2]),
                           "inter_rule_distance": inter, "split_half_distance": sh, "bootstrap_p95": bp,
                           "split_half_ratio": ratio,
                           "bootstrap_p95_ratio": max(bp.values()) / inter if inter > 0 else math.inf,
                           "stable": ratio <= thr})
    ov = [p["overlap_B3_B2"] for p in points]
    best = points[int(np.argmin(ov))]
    separable = best["overlap_B3_B2"] <= float(gn["overlap_max"])
    no_go = all(v >= float(gn["overlap_no_go"]) for v in ov)
    go_pts = [(p["R"], p["w"]) for p in points if p["overlap_B3_B2"] <= float(gn["overlap_max"]) and p["stable"]]
    return {"thresholds": {k: gn[k] for k in ("overlap_max", "overlap_no_go", "stability_ratio_max")},
            "points": points, "min_overlap": best["overlap_B3_B2"], "min_overlap_at": {"R": best["R"], "w": best["w"]},
            "separable": separable, "no_go": no_go, "stable_at_min_overlap": best["stable"],
            "stable_all": all(p["stable"] for p in points), "go_points": go_pts,
            "verdict": "go" if go_pts else ("no-go" if no_go else "inconclusive")}


# --------------------------------------------------------------------------- reporting
def _f(x, spec=".4g") -> str:
    return "-" if x is None else format(x, spec)


def summary_markdown(summary: dict) -> str:
    recs = summary["allocations"]
    stab = summary["stability"]["per_allocation"]
    L = ["# M3 closed-form channels (04_solve_channel.py)", "",
         f"Statistics: `{summary['provenance']['stats_file']}` (fingerprint `{summary['provenance']['stats_fingerprint']}`, "
         f"{summary['provenance']['n_images']} images, ridge_rel {summary['provenance']['ridge_rel']:g}). "
         "Distances are Frobenius norms of whitened retention matrices G = V diag(keep) V^T.", ""]
    b0 = recs["none"]
    L += [f"B0 (none): A = I, B = 0; identity residual max {b0['identity_residual_max']:.2e}.", ""]
    for R in summary["grid"]["R_nats"]:
        names = [n for n, r in recs.items() if r["rule"] != "none" and r["R"] == R]
        L += [f"## R = {R:g} nats/token ({al.bpp_equivalent(R):.4f} bpp equivalent)", "",
              "| allocation | arm | active | theta | D_latent | D_latent/trS | weighted dist. | sem. retention "
              "| max residual | MI residual | split-half d | boot mean | boot p95 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            r, s = recs[n], stab[n]
            L.append(f"| {n} | {r['arm']} | {r['n_active']} | {_f(r['theta'])} | {r['latent_mse_lmmse']:.4g} | "
                     f"{r['latent_mse_rel']:.4f} | {r['weighted_distortion']:.4g} | {r['semantic_retention']:.4f} | "
                     f"{r['identity_residual_max']:.1e} | {_f(r['mi_residual'], '.1e')} | {s['split_half_distance']:.4f} | "
                     f"{s['bootstrap_mean']:.4f} | {s['bootstrap_p95']:.4f} |")
        ovm = summary["overlaps"][str(R)]
        L += ["", "Retention overlap tr(GaGb)/sqrt(tr Ga^2 tr Gb^2):", "",
              "| | " + " | ".join(ovm["labels"]) + " |", "|---|" + "---|" * len(ovm["labels"])]
        for lab, row in zip(ovm["labels"], ovm["matrix"]):
            L.append(f"| {lab} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        L.append("")
    g = summary["go_no_go"]
    if g is None:
        L += ["## Go / no-go", "", "Not evaluated: the arms need both rule variance (B2) and rule r2 (B3).", ""]
        return "\n".join(L)
    t = g["thresholds"]
    L += ["## Go / no-go (plan Sec. 6, M3)", "",
          f"Thresholds: overlap_max {t['overlap_max']}, overlap_no_go {t['overlap_no_go']}, "
          f"stability_ratio_max {t['stability_ratio_max']}.", "",
          "| R | w | overlap(B3, B2) | inter-rule d | split-half d (B2 / B3) | ratio | boot p95 ratio | stable |",
          "|---|---|---|---|---|---|---|---|"]
    for p in g["points"]:
        sh = list(p["split_half_distance"].values())
        L.append(f"| {p['R']:g} | {p['w']:g} | {p['overlap_B3_B2']:.4f} | {p['inter_rule_distance']:.4f} | "
                 f"{sh[0]:.4f} / {sh[1]:.4f} | {p['split_half_ratio']:.3f} | {p['bootstrap_p95_ratio']:.3f} | "
                 f"{'yes' if p['stable'] else 'no'} |")
    L += ["", f"- min overlap {g['min_overlap']:.4f} at R={g['min_overlap_at']['R']:g}, w={g['min_overlap_at']['w']:g} "
              f"-> separable: **{g['separable']}** (<= {t['overlap_max']}); no-go (all >= {t['overlap_no_go']}): "
              f"**{g['no_go']}**",
          f"- stable at the min-overlap point: **{g['stable_at_min_overlap']}**; stable at every point: {g['stable_all']}",
          f"- separable and stable points: {g['go_points']}",
          f"- **verdict: {g['verdict']}** (the probe gate is evaluated by 05_eval_recon_probe.py)", ""]
    return "\n".join(L)


def figures(out: Path, summary: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    recs = summary["allocations"]
    Rs = summary["grid"]["R_nats"]
    paths = []
    fig, axes = plt.subplots(1, len(Rs), figsize=(3.2 * len(Rs), 3), squeeze=False, sharey=True)
    for ax, R in zip(axes[0], Rs):
        for n, r in recs.items():
            if r["rule"] != "none" and r["R"] == R:
                ax.plot(np.arange(1, len(r["keep"]) + 1), r["keep"], "o-", ms=2, lw=1, label=r["arm"])
        ax.set(title=f"R = {R:g}", xlabel="KLT direction i (s descending)")
    axes[0][0].set_ylabel("keep_i")
    axes[0][-1].legend(fontsize=6)
    paths.append(out / "keep_per_direction.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    g = summary["go_no_go"]
    if g is not None:
        ws = summary["grid"]["w"]
        M = np.array([[next(p["overlap_B3_B2"] for p in g["points"] if p["R"] == R and p["w"] == w) for w in ws]
                      for R in Rs])
        fig, ax = plt.subplots(figsize=(1.2 * len(ws) + 2, 0.6 * len(Rs) + 1.6))
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        for i in range(len(Rs)):
            for j in range(len(ws)):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
        ax.set_xticks(range(len(ws)), [f"{w:g}" for w in ws])
        ax.set_yticks(range(len(Rs)), [f"{R:g}" for R in Rs])
        ax.set(xlabel="w", ylabel="R (nats/token)", title="overlap(B3(w), B2)")
        fig.colorbar(im, ax=ax)
        paths.append(out / "overlap_heatmap.png")
        fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    out_dir = Path(args.out) if args.out else _common.allocations_dir(cfg)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        if out_dir.resolve() != _common.allocations_dir(cfg).resolve():
            run.logger.warning("--out %s: 05/06 read allocations from %s", out_dir, _common.allocations_dir(cfg))
        seed = int(cfg.get("seed", 0))
        runtime.seed_everything(seed)  # numpy float64 only; no device is used
        stats, acc, C, prov = load_inputs(cfg)
        ridge = prov["ridge_rel"]
        points, arms = grid_points(cfg)
        K, kinfo = al.whitened_relevance(stats.Sigma, stats.Sigma_expl)
        run.logger.info("stats %s: %d images, C=%d, D=%d; %d lossy allocations + B0", prov["stats_fingerprint"],
                        stats.n_images, stats.C, stats.D, len(points))

        recs, G_full, failures = {}, {}, []
        for rule, R, w in [("none", None, None)] + points:
            alloc = al.solve_allocation(stats, rule, w, R)
            alloc = replace(alloc, meta={**alloc.meta, "provenance": prov})
            path = out_dir / f"{alloc.name}.npz"
            alloc.save(path)
            rec = describe(alloc, stats.Sigma, K)
            rec.update(arm=label_of(arms, rule, w) if rule != "none" else "/".join(arms.get("none", ["B0"])),
                       file=path, sha256=runtime.sha256_file(path))
            recs[alloc.name] = rec
            G_full[alloc.name] = al.retention_matrix_whitened(alloc, stats.Sigma)
            failures += residual_failures(rec)
            run.log_metrics(None, allocation=alloc.name, semantic_retention=rec["semantic_retention"],
                            latent_mse_lmmse=rec["latent_mse_lmmse"], identity_residual_max=rec["identity_residual_max"])

        Rs = [float(R) for R in cfg["grid"]["R_nats"]]
        ws = [float(w) for w in cfg["grid"]["w"]]
        overlaps = {}
        for R in Rs:
            names = [name_of(r, R2, w) for r, R2, w in points if R2 == R]
            overlaps[str(R)] = {"names": names, "labels": [recs[n]["arm"] for n in names],
                                "matrix": [[al.retention_overlap(G_full[a], G_full[b]) for b in names] for a in names]}
        stab = stability(acc, C, ridge, points, G_full, int(cfg["stats"]["bootstrap"]), seed, run.logger)
        has_pair = "variance" in arms and "r2" in arms
        gng = go_no_go(cfg, Rs, ws, G_full, stab["per_allocation"]) if has_pair else None
        summary = {"provenance": prov, "grid": {"R_nats": Rs, "w": ws}, "arms": arms, "K": {"info": kinfo,
                   "eig": np.linalg.eigvalsh(K)[::-1], "trace": float(np.trace(K))},
                   "tolerances": {"identity": IDENTITY_TOL, "mi_floor": MI_TOL},
                   "allocations": recs, "overlaps": overlaps, "stability": stab, "go_no_go": gng,
                   "identity_failures": failures}
        run.save_json("allocations_summary.json", summary)
        (out_dir / "allocations_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
        figs = figures(out_dir / "figures", summary) if args.figures else []
        if failures:
            raise ValueError("exact-identity residuals above tolerance:\n  " + "\n  ".join(failures))
        if gng is not None:
            run.logger.info("go/no-go: %s (min overlap %.4f at R=%g, w=%g; stable there: %s)", gng["verdict"],
                            gng["min_overlap"], gng["min_overlap_at"]["R"], gng["min_overlap_at"]["w"],
                            gng["stable_at_min_overlap"])
        run.finish("ok", allocations=sorted(recs), verdict=None if gng is None else gng["verdict"],
                   min_overlap=None if gng is None else gng["min_overlap"], figures=figs)
        return 0
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
