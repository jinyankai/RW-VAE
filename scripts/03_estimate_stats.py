"""03 -- joint float64 token statistics of [mu; DINO patch] on train_fit (plan v2.1 Sec. 1.2, 3.3 step 3).

Every token of a train_fit image contributes the joint vector x = [z0; y] (z0 = cached posterior mean
token, y = the DINOv2 patch feature at the same normalised grid position) to a float64
GroupedMomentAccumulator; the image's group is stats.image_group_id(image_id, stats.num_groups), so the
split-half / bootstrap checks of 04 resample images, never tokens. With stats.include_flip the flipped
image (torch.flip(images, -1) -> DINO) and the cached flipped latent are added under the SAME group;
images are counted once. The official val split is refused (plan Sec. 1.2).

Gate (plan Sec. 1.2, "patch grid one-to-one"): before streaming, the first stats.alignment_check.num_images
selected rows are encoded and relevance.alignment_shift_scan(mu, y, max_shift, ridge_rel) must peak at grid
shift (0, 0). On FAIL the scan is written and the script exits with code 2 (no stats.npz) unless
--allow-misaligned. Every mode (single, shard, merge) runs this check first, so a misaligned run fails fast.

Inputs   manifest (01), the latent cache of stats.split (02), the DINO teacher (cfg.semantic).
Outputs  <out> = --out or <run_root>/<name>/stats  (04 reads <run_root>/<name>/stats, i.e. _common.stats_path)
         stats.npz            LatentStats (m, Sigma, Sigma_zy, Sigma_y, Sigma_expl at stats.ridge_rel) + meta
         group_moments.pt     accumulator state per image group (04: split-half and group bootstrap)
         diagnostics.json     spectra, Sigma_y rank, ridge scan (CCA spectrum, explained fraction), eig(K),
                              split-half stability, alignment scan, token/dimension ratio
         figures/*.png        with --figures: spectrum of Sigma, eig(K) per ridge, alignment heat map
         <exp>/DATA_CARD.md   section "## 3. Semantic statistics" (replaced if present)
         shards/acc_<k>.pt    with --num-procs N (N > 1): per-process accumulator states, merged by --merge;
                              each process's run directory is shards/proc_<k>/
Rows     the first min(N_split, stats.max_images | --max-images) rows of the split, in order; process k of N
         takes rows[k::N]. Under torchrun (WORLD_SIZE > 1) --num-procs/--proc-id default to WORLD_SIZE/RANK.

Examples (common flags: --set, --out, --device, --seed, --overwrite, ...)
  ImageNet-1k, one process:
    python scripts/03_estimate_stats.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml
  ImageNet-1k, 8 GPUs (one independent process per GPU), then merge:
    torchrun --nproc_per_node 8 scripts/03_estimate_stats.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml
    python scripts/03_estimate_stats.py --config ... (same configs) --merge --figures
  SVP (configs/svp.yaml overlay; same pipeline):
    python scripts/03_estimate_stats.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/server_a800.yaml --config configs/svp.yaml --figures
  Local Imagenette (engineering test only):
    python scripts/03_estimate_stats.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/local_imagenette.yaml --max-images 500
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rwvae import datasets, relevance, runtime
from rwvae.allocation import whitened_relevance
from rwvae.stats import (GroupedMomentAccumulator, LatentStats, image_group_id, principal_angles,
                         relative_frobenius, spectrum_report, split_half_groups, tokens)

SHARD_FORMAT = "rwvae.stats_shard/v1"
MOMENTS_FORMAT = "rwvae.group_moments/v1"
EXIT_MISALIGNED = 2
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
TOP_K = 8                 # eigenvectors compared by principal angles in the split-half check
LOG_EVERY = 50            # batches between progress lines
RANK_TOL = 1e-12          # numerical rank of Sigma_y: eig > RANK_TOL * max eig (as relevance.explained_covariance)


def build_parser():
    p = _common.make_parser("Joint float64 [mu; DINO] token statistics on train_fit (plan Sec. 1.2, 3.3).")
    p.add_argument("--num-procs", type=int, default=None, help="number of row shards (default: WORLD_SIZE or 1)")
    p.add_argument("--proc-id", type=int, default=None, help="this process's shard (default: RANK or 0)")
    p.add_argument("--merge", action="store_true", help="merge shards/acc_*.pt and finalise")
    p.add_argument("--max-images", type=int, default=None, help="cap: first N rows (overrides stats.max_images)")
    p.add_argument("--allow-misaligned", action="store_true",
                   help="continue (exit 0) although the alignment scan does not peak at (0, 0)")
    p.add_argument("--figures", action="store_true", help="write figures/*.png")
    return p


# --------------------------------------------------------------------------- data
def pick_device(pref: str, num_procs: int, proc_id: int) -> torch.device:
    """One GPU per shard process when several are visible: cuda:(LOCAL_RANK or proc_id % count)."""
    if num_procs > 1 and pref in ("auto", "cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        local = os.environ.get("LOCAL_RANK")
        device = runtime.select_device(f"cuda:{int(local) if local is not None else proc_id % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        return device
    return runtime.select_device(pref)


def semantic_autocast(cfg: dict, device: torch.device) -> tuple[torch.dtype | None, str]:
    """cfg.semantic.autocast (bf16 | fp16 | none) applies on CUDA only; CPU runs float32."""
    name = cfg["semantic"].get("autocast")
    if name in (None, "none", "no", "fp32", "float32"):
        return None, "none (float32)"
    if name not in AMP_DTYPES:
        raise ValueError(f"semantic.autocast must be one of {sorted(AMP_DTYPES)} or none, got {name!r}")
    if device.type != "cuda":
        return None, f"none (float32; semantic.autocast={name} applies on CUDA only)"
    return AMP_DTYPES[name], name


def iter_pairs(manifest, cache, rows: list[int], semantic, device, amp, cfg):
    """Yield (mu [B,C,H,W], y [B,D,H,W] float32, group ids [B], flipped) for the given split rows, in order.

    With stats.include_flip every batch is followed by its flipped copy (flipped image -> DINO, cached
    latents_flip) under the same group ids."""
    s = cfg["stats"]
    split = cache.split
    ids = manifest.splits[split]["image_id"]
    G = int(s["num_groups"])
    flip = bool(s["include_flip"])
    if flip and not cache.has_flip:
        raise ValueError(f"stats.include_flip=true but the {split} cache {cache.cache_dir} stores no latents_flip")
    pairs = datasets.ImageLatentPairDataset(manifest, cache, image_size=int(cfg["dataset"]["image_size"]),
                                            indices=rows)
    loader = DataLoader(pairs, batch_size=int(s["batch_size"]), shuffle=False, num_workers=int(s["num_workers"]),
                        pin_memory=device.type == "cuda", persistent_workers=False)

    def features(images):
        with torch.autocast("cuda", dtype=amp) if amp is not None else contextlib.nullcontext():
            y = semantic(images.to(device, non_blocking=True))
        return y.float()

    for images, mu, _, index in loader:
        idx = index.tolist()
        gid = torch.tensor([image_group_id(ids[i], G) for i in idx], dtype=torch.int64)
        y = features(images)
        if y.shape[0] != mu.shape[0] or tuple(y.shape[2:]) != tuple(mu.shape[2:]):
            raise ValueError(f"semantic features {tuple(y.shape)} do not pair with latents {tuple(mu.shape)}: "
                             "the DINO grid must equal the latent grid (no reshaping, plan Sec. 1.2)")
        yield mu, y, gid, False
        if flip:
            mu_f = torch.stack([cache.get(i, flip=True)[0] for i in idx])
            yield mu_f, features(torch.flip(images, dims=[-1])), gid, True


def stream(run, manifest, cache, rows, semantic, device, amp, cfg) -> GroupedMomentAccumulator:
    """Accumulate the joint token moments of `rows` (accumulator created at the first batch: dim C + D)."""
    acc = None
    G = int(cfg["stats"]["num_groups"])
    n_batches = 0
    for mu, y, gid, flipped in iter_pairs(manifest, cache, rows, semantic, device, amp, cfg):
        if acc is None:
            acc = GroupedMomentAccumulator(mu.shape[1] + y.shape[1], G, device)
        H, W = mu.shape[2:]
        acc.update(torch.cat([tokens(mu.to(device)), tokens(y)], dim=1), gid.repeat_interleave(H * W))
        if not flipped:
            acc.add_images(gid)
            n_batches += 1
            if n_batches % LOG_EVERY == 0:
                run.logger.info("stats: %d / %d images", acc.n_images, len(rows))
                run.log_metrics(n_batches, images=acc.n_images, token_rows=int(acc.n.sum()))
    if acc is None:
        raise ValueError("no rows selected: nothing to accumulate")
    run.log_metrics(n_batches, images=acc.n_images, token_rows=int(acc.n.sum()))
    return acc


def alignment_check(run, manifest, cache, rows, semantic, device, amp, cfg) -> dict:
    """Plan Sec. 1.2 gate on the first alignment_check.num_images selected rows (no flips): PASS iff the
    explained fraction of z0 by y peaks at grid shift (0, 0)."""
    a = cfg["stats"]["alignment_check"]
    keep = rows[:int(a["num_images"])]
    mus, ys = [], []
    no_flip = {**cfg, "stats": {**cfg["stats"], "include_flip": False}}
    for mu, y, _, _ in iter_pairs(manifest, cache, keep, semantic, device, amp, no_flip):
        mus.append(mu.float().cpu())
        ys.append(y.cpu())
    mu, y = torch.cat(mus), torch.cat(ys)
    ridge = float(cfg["stats"]["ridge_rel"])
    scan = relevance.alignment_shift_scan(mu, y, int(a["max_shift"]), ridge)
    best = max(scan, key=scan.get)
    others = [v for k, v in scan.items() if k != (0, 0)]
    out = {"num_images": len(keep), "rows": "first num_images selected rows, unflipped",
           "max_shift": int(a["max_shift"]), "ridge_rel": ridge,
           "scan": [{"dy": dy, "dx": dx, "explained_fraction": v} for (dy, dx), v in sorted(scan.items())],
           "argmax": list(best), "pass": best == (0, 0), "device": str(device), "autocast": str(amp),
           "margin_vs_best_other": scan[(0, 0)] - max(others) if others else None}
    run.logger.info("alignment scan: argmax %s (explained fraction %.4f at (0,0)) -> %s", best, scan[(0, 0)],
                    "PASS" if out["pass"] else "FAIL")
    return out


# --------------------------------------------------------------------------- shards
def shard_path(stats_dir: Path, k: int) -> Path:
    return stats_dir / "shards" / f"acc_{k}.pt"


def merge_shards(stats_dir: Path, num_procs: int | None, device, provenance: dict, logger):
    """Load and merge shards/acc_<k>.pt; they must cover k = 0..N-1 and agree on dim, groups and meta."""
    files = sorted((stats_dir / "shards").glob("acc_*.pt"))
    if not files:
        raise FileNotFoundError(f"no shard states under {stats_dir / 'shards'}: run with --num-procs N --proc-id k")
    loaded = [(f, torch.load(f, map_location="cpu", weights_only=True)) for f in files]
    bad = [str(f) for f, sh in loaded if sh.get("format") != SHARD_FORMAT]
    if bad:
        raise ValueError(f"not {SHARD_FORMAT} files: {bad}")
    loaded.sort(key=lambda item: int(item[1]["meta"]["proc_id"]))
    files, shards = [f for f, _ in loaded], [sh for _, sh in loaded]
    N = int(shards[0]["meta"]["num_procs"]) if num_procs in (None, 1) else int(num_procs)
    ids = [int(sh["meta"]["proc_id"]) for sh in shards]
    if ids != list(range(N)) or any(int(sh["meta"]["num_procs"]) != N for sh in shards):
        raise ValueError(f"shards cover proc ids {ids} with num_procs {[sh['meta']['num_procs'] for sh in shards]}; "
                         f"expected exactly 0..{N - 1}")
    first = shards[0]
    for sh in shards:
        for key in ("C", "dim", "num_groups"):
            if sh[key] != first[key]:
                raise ValueError(f"shard {sh['meta']['proc_id']} has {key}={sh[key]}, shard 0 has {first[key]}")
        for key in ("inputs", "protocol", "teacher"):
            if sh["meta"][key] != first["meta"][key]:
                raise ValueError(f"shard {sh['meta']['proc_id']} disagrees with shard 0 on {key}")
    for key in ("inputs", "protocol"):
        if first["meta"][key] != provenance[key]:
            raise ValueError(f"the shards were computed with different {key} than the current config/data: "
                             f"shards {first['meta'][key]} vs now {provenance[key]}")
    acc = GroupedMomentAccumulator(int(first["dim"]), int(first["num_groups"]), device)
    acc.load_state_dict(first["state"])
    for sh in shards[1:]:
        other = GroupedMomentAccumulator(int(sh["dim"]), int(sh["num_groups"]), "cpu")
        other.load_state_dict(sh["state"])
        acc.merge_(other)
    logger.info("merged %d shard states: %d images, %d token rows", N, acc.n_images, int(acc.n.sum()))
    shard_info = [{"file": f, "sha256": runtime.sha256_file(f), "proc_id": int(sh["meta"]["proc_id"]),
                   "n_rows": int(sh["meta"]["n_rows"]), "device": sh["meta"]["device"]}
                  for f, sh in zip(files, shards)]
    return acc, int(first["C"]), first["meta"]["teacher"], shard_info


# --------------------------------------------------------------------------- diagnostics
def _top_eigvecs(M: np.ndarray, k: int) -> np.ndarray:
    _, V = np.linalg.eigh(0.5 * (M + M.T))
    return V[:, ::-1][:, :k]


def _half_stats(acc, groups, C: int, ridge: float) -> LatentStats:
    n_img = int(acc.images_per_group[groups].sum())
    return LatentStats.from_joint(acc.subset(groups), C, n_img).with_explained(ridge)


def split_half(acc, C: int, ridge: float) -> dict:
    """Even vs odd image groups: relative Frobenius differences and top-k principal angles."""
    even, odd = split_half_groups(acc.num_groups)
    a, b = _half_stats(acc, even, C, ridge), _half_stats(acc, odd, C, ridge)
    Ka, Kb = whitened_relevance(a.Sigma, a.Sigma_expl)[0], whitened_relevance(b.Sigma, b.Sigma_expl)[0]
    k = min(TOP_K, C)
    ang_S = principal_angles(_top_eigvecs(a.Sigma, k), _top_eigvecs(b.Sigma, k))
    ang_K = principal_angles(_top_eigvecs(Ka, k), _top_eigvecs(Kb, k))
    return {"groups": {"even": even, "odd": odd}, "n_images": [a.n_images, b.n_images],
            "n_tokens": [a.n_tokens, b.n_tokens], "ridge_rel": ridge,
            "rel_frobenius_Sigma": relative_frobenius(a.Sigma, b.Sigma),
            "rel_frobenius_Sigma_expl": relative_frobenius(a.Sigma_expl, b.Sigma_expl),
            "rel_frobenius_K": relative_frobenius(Ka, Kb),
            "principal_angles_top_k": k,
            "principal_angles_Sigma_rad": ang_S, "principal_angles_Sigma_max_deg": float(np.degrees(ang_S.max())),
            "principal_angles_K_rad": ang_K, "principal_angles_K_max_deg": float(np.degrees(ang_K.max())),
            "note": "relative_frobenius(A, B) = ||A - B||_F / ||B||_F with A = even half, B = odd half"}


def diagnostics(st: LatentStats, acc, cfg: dict, align: dict, logger) -> dict:
    s = cfg["stats"]
    spec = spectrum_report(st.Sigma)
    spec_y = spectrum_report(st.Sigma_y)
    rank_y = int((spec_y["eig"] > RANK_TOL * spec_y["max"]).sum())
    scan = []
    for r in s["ridge_scan"]:
        try:
            S_e, info = relevance.explained_covariance(st.Sigma_zy, st.Sigma_y, float(r))
            eig = relevance.cca_spectrum(st.Sigma, st.Sigma_zy, st.Sigma_y, float(r))
        except ValueError as e:  # e.g. a ridge below the numerical-rank floor: recorded, not hidden
            logger.warning("ridge_rel=%g skipped in the ridge scan: %s", r, e)
            scan.append({"ridge_rel": float(r), "error": str(e)})
            continue
        scan.append({"ridge_rel": float(r), "kind": "CCA^2 (rho_i^2)" if r == 0 else "shrunken (ridge) -- not CCA^2",
                     "eig_K": eig, "explained_fraction": relevance.explained_fraction(st.Sigma, S_e), "ridge": info})
    K, kinfo = whitened_relevance(st.Sigma, st.Sigma_expl)
    eig_K = np.linalg.eigvalsh(K)[::-1]
    joint = st.C + st.D
    return {
        "n_images": st.n_images, "n_tokens": st.n_tokens, "C": st.C, "D": st.D, "joint_dim": joint,
        "tokens_per_joint_dim": st.n_tokens / joint, "images_per_joint_dim": st.n_images / joint,
        "images_per_group": acc.images_per_group, "tokens_per_group": acc.n.cpu(),
        "Sigma": {**spec, "shrinkage": "none (raw unbiased sample covariance; plan Sec. 3.3)"},
        "Sigma_y": {"rank": rank_y, "rank_tol": RANK_TOL, "dim": st.D, "min_eig": spec_y["min"],
                    "max_eig": spec_y["max"], "eff_rank": spec_y["eff_rank"], "cond": spec_y["cond"],
                    "n_nonpositive": spec_y["n_nonpositive"], "eig_top8": spec_y["eig"][:8],
                    "eig_bottom8": spec_y["eig"][-8:]},
        "ridge_scan": scan,
        "K": {"ridge_rel": float(s["ridge_rel"]), "eig": eig_K, "trace": float(np.trace(K)), "info": kinfo,
              "explained_fraction": relevance.explained_fraction(st.Sigma, st.Sigma_expl),
              "ridge": st.meta.get("ridge")},
        "split_half": split_half(acc, st.C, float(s["ridge_rel"])),
        "alignment": align,
    }


# --------------------------------------------------------------------------- reporting
def alignment_table(align: dict) -> str:
    k = align["max_shift"]
    val = {(r["dy"], r["dx"]): r["explained_fraction"] for r in align["scan"]}
    lines = ["| dy \\ dx | " + " | ".join(str(dx) for dx in range(-k, k + 1)) + " |",
             "|---|" + "---|" * (2 * k + 1)]
    for dy in range(-k, k + 1):
        cells = [(f"**{val[(dy, dx)]:.4f}**" if (dy, dx) == tuple(align["argmax"]) else f"{val[(dy, dx)]:.4f}")
                 for dx in range(-k, k + 1)]
        lines.append(f"| {dy} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def data_card_section(cfg, provenance, align, diag=None, stats_file=None, st=None) -> str:
    inp, prot, teach = provenance["inputs"], provenance["protocol"], provenance["teacher"]
    sem = teach["semantic_meta"]
    status = "PASS" if align["pass"] else "FAIL"
    lines = [
        "## 3. Semantic statistics", "",
        f"Written by `scripts/03_estimate_stats.py` on {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"(git {runtime.git_commit()}).", "",
        f"- Split: `{inp['split']}` (the official val split is never used for statistics, plan Sec. 1.2); "
        f"rows: the first {prot['n_rows']} of {inp['split_size']} (max_images={prot['max_images']}); "
        f"include_flip={prot['include_flip']} (flipped samples join the image's group; images counted once).",
        f"- Latents: cache `{inp['cache_dir']}` (cache_index sha256 `{inp['cache_index_sha256']}`, dtype "
        f"{inp['cache_dtype']}, shape {inp['latent_shape']}); manifest fingerprint `{inp['manifest_fingerprint']}`.",
        f"- DINO teacher: `{cfg['semantic'].get('model')}`, input size {cfg['semantic'].get('input_size')}, "
        f"patch {sem.get('patch_size')}, grid {sem.get('grid')}, prefix tokens dropped: {sem.get('num_prefix_tokens')}, "
        f"embed dim {sem.get('embed_dim')}; preprocessing: {sem.get('preprocess')}; tokens: {sem.get('tokens')}; "
        f"checkpoint: {sem.get('checkpoint_path')}; autocast: {teach['autocast']}; TF32: {teach['tf32']}.",
        f"- Pairing: z0 token (h, w) of the {cfg['dataset']['image_size']} px crop with the DINO patch (h, w) of the "
        "same crop (normalised grid positions; flattening `stats.tokens`, row-major (h, w)).",
        f"- Accumulation: float64 centred moments (Chan merge) in {prot['num_groups']} image-ID hash groups "
        "(`stats.image_group_id`); unbiased covariance (ddof = 1); no shrinkage.",
        "", f"Alignment scan (explained fraction of z0 by y per grid shift; {align['num_images']} images, "
            f"ridge_rel={align['ridge_rel']:g}): **{status}** (argmax {tuple(align['argmax'])})", "",
        alignment_table(align), "",
    ]
    if diag is not None:
        r = diag["K"]["ridge"]
        sh = diag["split_half"]
        lines += [
            f"- n_images = {diag['n_images']}, n_tokens = {diag['n_tokens']}, joint dim C + D = {diag['joint_dim']} "
            f"(tokens / dim = {diag['tokens_per_joint_dim']:.1f}, images / dim = {diag['images_per_joint_dim']:.2f}).",
            f"- Sigma: eff. rank {diag['Sigma']['eff_rank']:.2f} of {diag['C']}, cond {diag['Sigma']['cond']:.3g}; "
            f"Sigma_y: rank {diag['Sigma_y']['rank']} of {diag['D']}, eig [{diag['Sigma_y']['min_eig']:.3g}, "
            f"{diag['Sigma_y']['max_eig']:.3g}].",
            f"- Ridge: ridge_rel = {r['ridge_rel']:g}, lambda = {r['lambda']:.4g}; explained fraction "
            f"tr(Sigma_expl)/tr(Sigma) = {diag['K']['explained_fraction']:.4f}; eig(K) max {diag['K']['eig'][0]:.4f}.",
            f"- Split-half (even vs odd groups, {sh['n_images'][0]} / {sh['n_images'][1]} images): rel. Frobenius "
            f"Sigma {sh['rel_frobenius_Sigma']:.4f}, Sigma_expl {sh['rel_frobenius_Sigma_expl']:.4f}; max principal "
            f"angle (top {sh['principal_angles_top_k']}) Sigma {sh['principal_angles_Sigma_max_deg']:.2f} deg, "
            f"K {sh['principal_angles_K_max_deg']:.2f} deg.",
            f"- Stats file: `{Path(stats_file).as_posix()}` (fingerprint `{st.fingerprint()}`, sha256 `{runtime.sha256_file(stats_file)}`).",
            "",
        ]
    else:
        lines += ["- Statistics were NOT written: the alignment gate failed.", ""]
    return "\n".join(lines)


def write_data_card(cfg: dict, section: str) -> Path:
    """Replace (or append) the '## 3. Semantic statistics' section of <exp>/DATA_CARD.md."""
    path = _common.exp_dir(cfg) / "DATA_CARD.md"
    head = "## 3. Semantic statistics"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# DATA CARD -- {cfg['experiment']['name']}\n\n"
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.strip() == head), None)
    if start is None:
        new = text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        rest = "".join(lines[end:])
        new = "".join(lines[:start]) + section.rstrip("\n") + "\n" + ("\n" + rest if rest else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)
    return path


def figures(out: Path, diag: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 3.5))
    eig = np.asarray(diag["Sigma"]["eig"])
    ax.semilogy(np.arange(1, eig.size + 1), eig, "o-", ms=3)
    ax.set(xlabel="index i", ylabel="eigenvalue of Sigma", title=f"Sigma spectrum (eff. rank {diag['Sigma']['eff_rank']:.1f})")
    paths.append(out / "sigma_spectrum.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for r in diag["ridge_scan"]:
        if "eig_K" in r:
            e = np.asarray(r["eig_K"])
            ax.plot(np.arange(1, e.size + 1), e, "o-", ms=3, label=f"ridge_rel={r['ridge_rel']:g}")
    ax.set(xlabel="index i", ylabel="eig(K)  (ridge 0: CCA^2)", ylim=(0, 1), title="Whitened relevance spectrum")
    ax.legend(fontsize=7)
    paths.append(out / "k_spectrum.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    a = diag["alignment"]
    k = a["max_shift"]
    grid = np.zeros((2 * k + 1, 2 * k + 1))
    for r in a["scan"]:
        grid[r["dy"] + k, r["dx"] + k] = r["explained_fraction"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(grid, cmap="viridis", extent=(-k - 0.5, k + 0.5, k + 0.5, -k - 0.5))
    fig.colorbar(im, ax=ax, label="explained fraction")
    ax.set(xlabel="dx", ylabel="dy", title="Alignment scan (" + ("PASS" if a["pass"] else "FAIL") + ")")
    paths.append(out / "alignment_scan.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    s = cfg["stats"]
    split = s["split"]
    if split == "val":
        raise ValueError("stats.split=val is refused: statistics come from train_fit only (plan Sec. 1.2)")
    rank, world = _common.rank_world()
    num_procs = int(args.num_procs) if args.num_procs is not None else world
    proc_id = int(args.proc_id) if args.proc_id is not None else rank
    if num_procs < 1 or not 0 <= proc_id < num_procs:
        raise ValueError(f"need --num-procs >= 1 and 0 <= --proc-id < --num-procs, got {num_procs}, {proc_id}")
    mode = "merge" if args.merge else ("shard" if num_procs > 1 else "single")
    max_images = args.max_images if args.max_images is not None else s.get("max_images")
    stats_dir = Path(args.out) if args.out else _common.stats_path(cfg).parent
    out_dir = stats_dir / "shards" / f"proc_{proc_id}" if mode == "shard" else stats_dir
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        if stats_dir.resolve() != _common.stats_path(cfg).parent.resolve():
            run.logger.warning("--out %s: 04_solve_channel.py reads %s, not this directory", stats_dir,
                               _common.stats_path(cfg))
        runtime.seed_everything(int(cfg.get("seed", 0)))
        device = pick_device(args.device, num_procs, proc_id)
        tf32 = _common.set_tf32(False)
        manifest = _common.load_manifest(cfg)
        cache = _common.open_cache(cfg, split, manifest)
        if cache.split != split:
            raise ValueError(f"cache {cache.cache_dir} holds split {cache.split!r}, stats.split is {split!r}")
        n_split = len(cache)
        rows = list(range(n_split if max_images is None else min(n_split, int(max_images))))
        if max_images is not None and int(max_images) < n_split:
            run.logger.warning("max_images=%d: using the first %d of %d %s rows (recorded in meta)",
                               int(max_images), len(rows), n_split, split)
        amp, amp_label = semantic_autocast(cfg, device)
        semantic = _common.make_semantic(cfg, device)
        provenance = {
            "inputs": {"split": split, "split_size": n_split, "manifest": _common.manifest_path(cfg),
                       "manifest_fingerprint": manifest.fingerprint(), "cache_dir": cache.cache_dir,
                       "cache_index_sha256": runtime.sha256_file(cache.cache_dir / datasets.CACHE_INDEX),
                       "cache_dtype": cache.dtype, "latent_shape": cache.latent_shape, "cache_has_flip": cache.has_flip,
                       "cache_meta": cache.meta},
            "protocol": {"n_rows": len(rows), "max_images": max_images, "rows": "first n_rows rows of the split, in order",
                         "include_flip": bool(s["include_flip"]), "num_groups": int(s["num_groups"]),
                         "group_id": "stats.image_group_id(image_id, num_groups) (salt 'rwvae-v2')",
                         "image_size": int(cfg["dataset"]["image_size"]), "semantic": cfg["semantic"],
                         "joint_token": "cat(stats.tokens(mu), stats.tokens(y)); row-major (h, w)",
                         "accumulation_dtype": "float64", "latent_source": cache.meta.get("latent_source")},
            "teacher": {"semantic_meta": dict(getattr(semantic, "meta", None) or {}), "autocast": amp_label,
                        "tf32": tf32},
        }
        provenance = runtime.to_jsonable(provenance)
        run.logger.info("mode %s (proc %d of %d) on %s; %d %s rows; DINO autocast %s", mode, proc_id, num_procs,
                        device, len(rows), split, amp_label)

        align = alignment_check(run, manifest, cache, rows, semantic, device, amp, cfg)
        if not align["pass"]:
            if mode == "shard":
                run.save_json("alignment.json", align)
            else:
                run.save_json("diagnostics.json", {"alignment": align, "provenance": provenance})
                write_data_card(cfg, data_card_section(cfg, provenance, align))
            if not args.allow_misaligned:
                run.logger.error("alignment gate FAILED (argmax %s, expected (0, 0)); no statistics written. "
                                 "Fix the pairing or pass --allow-misaligned.", tuple(align["argmax"]))
                run.finish("failed", error="alignment gate failed", alignment_argmax=align["argmax"])
                return EXIT_MISALIGNED
            run.logger.warning("alignment gate FAILED but --allow-misaligned was given: continuing")

        if mode == "shard":
            acc = stream(run, manifest, cache, rows[proc_id::num_procs], semantic, device, amp, cfg)
            path = shard_path(stats_dir, proc_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {**provenance, "num_procs": num_procs, "proc_id": proc_id, "n_rows": len(rows[proc_id::num_procs]),
                    "device": str(device), "alignment_pass": align["pass"], "git_commit": runtime.git_commit()}
            tmp = path.with_name(path.name + ".tmp")
            torch.save({"format": SHARD_FORMAT, "C": int(cache.latent_shape[0]), "dim": acc.dim,
                        "num_groups": acc.num_groups, "state": acc.state_dict(), "meta": runtime.to_jsonable(meta)}, tmp)
            os.replace(tmp, path)
            run.finish("ok", mode=mode, shard=path, proc_id=proc_id, num_procs=num_procs, n_images=acc.n_images)
            return 0

        if mode == "merge":
            acc, C, teacher, shard_info = merge_shards(stats_dir, args.num_procs, device, provenance, run.logger)
            provenance["teacher"] = teacher
        else:
            acc = stream(run, manifest, cache, rows, semantic, device, amp, cfg)
            C, shard_info = int(cache.latent_shape[0]), None
        if acc.n_images != len(rows):
            raise ValueError(f"accumulated {acc.n_images} images, expected {len(rows)}")

        ridge = float(s["ridge_rel"])
        meta = runtime.to_jsonable({**provenance, "mode": mode, "num_procs": num_procs, "shards": shard_info,
                                    "seed": int(cfg.get("seed", 0)), "alignment_pass": align["pass"],
                                    "allow_misaligned": bool(args.allow_misaligned), "git_commit": runtime.git_commit(),
                                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        st = LatentStats.from_joint(acc.total(), C, acc.n_images, meta).with_explained(ridge)
        stats_file = stats_dir / "stats.npz"
        st.save(stats_file)
        moments_file = stats_dir / "group_moments.pt"
        tmp = moments_file.with_name(moments_file.name + ".tmp")
        torch.save({"format": MOMENTS_FORMAT, "C": C, "dim": acc.dim, "num_groups": acc.num_groups,
                    "state": acc.state_dict(), "stats_fingerprint": st.fingerprint(),
                    "stats_sha256": runtime.sha256_file(stats_file), "meta": meta}, tmp)
        os.replace(tmp, moments_file)

        diag = diagnostics(st, acc, cfg, align, run.logger)
        diag.update(stats_file=stats_file, stats_fingerprint=st.fingerprint(), group_moments_file=moments_file,
                    provenance=provenance)
        run.save_json("diagnostics.json", diag)
        card = write_data_card(cfg, data_card_section(cfg, provenance, align, diag, stats_file, st))
        figs = figures(stats_dir / "figures", diag) if args.figures else []
        run.logger.info("stats: %d images, %d tokens, C=%d, D=%d; eff. rank Sigma %.2f; explained fraction %.4f",
                        st.n_images, st.n_tokens, st.C, st.D, diag["Sigma"]["eff_rank"], diag["K"]["explained_fraction"])
        run.finish("ok", mode=mode, stats=stats_file, stats_fingerprint=st.fingerprint(), n_images=st.n_images,
                   n_tokens=st.n_tokens, C=st.C, D=st.D, alignment_pass=align["pass"], data_card=card, figures=figs)
        return 0
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
