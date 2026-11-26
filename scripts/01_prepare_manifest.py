"""01 -- deterministic split manifest + DATA_CARD section 1 (plan v2.1 Sec. 1.1-1.2, Sec. 6 M2).

Builds the image-level train_dev / train_fit / val manifest with rwvae.datasets.build_manifest:
per class, train_dev = np.random.default_rng([split_seed, label]).choice(n, dev_per_class) of the
sorted train files, train_fit = the rest, val = every official val image. If cfg.dataset.manifest
already exists it is rebuilt and compared: same fingerprint (classes + splits) -> "up to date" and the
file is kept (rewritten only when the data roots moved); a different fingerprint is an error unless
--force (every downstream cache/stats/allocation is tied to the manifest fingerprint).

Inputs   cfg.dataset.{train_root, val_root, name, image_size, num_classes, dev_per_class, split_seed, manifest};
         optional <train_root>/../svp_subset.json written by scripts/tools/make_svp_subset.py (SVP provenance;
         its train file list must equal the manifest's train_dev + train_fit files).
Outputs  cfg.dataset.manifest                      the manifest JSON (rwvae.datasets.Manifest)
         <run_root>/<name>/DATA_CARD.md            section "## 1. Dataset and splits" created or replaced
                                                   (other sections are kept)
         <run_root>/<name>/01_manifest/manifest_info.json  counts, fingerprints, status, SVP provenance
         <run_root>/<name>/01_manifest/{config.yaml, log.txt, summary.json}  (rwvae.runtime.Run)

Examples
  ImageNet-1k (GPU node):
    python scripts/01_prepare_manifest.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/server_a800.yaml
  SVP (10 random classes x 100 train + 50 val images, built by scripts/tools/make_svp_subset.py):
    python scripts/01_prepare_manifest.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml \
        --config configs/server_a800.yaml --config configs/svp.yaml
  Local Imagenette (engineering test only):
    python scripts/01_prepare_manifest.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/local_imagenette.yaml
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import json
from pathlib import Path

from rwvae import datasets, runtime

SECTION = "## 1. Dataset and splits"
SPLIT_RULE = ("per class (labels = sorted train class directories): train_dev = sorted files F[sort("
              "np.random.default_rng([split_seed, label]).choice(len(F), dev_per_class, replace=False))], "
              "train_fit = the remaining files of F, val = every file of the official val folder")
NO_LEAKAGE = [
    "Splits are made per image ID (file stem, unique across splits): no image's tokens straddle two splits.",
    "Latent statistics m, Sigma, Sigma_zy, Sigma_y come from train_fit only; the official val split is never "
    "used for statistics, rule selection or model fitting.",
    "train_dev serves generator-free rule selection and diagnostics (M3); val is the final evaluation (D3) "
    "and the gFID reference only. R*, w* are never chosen from gFID (plan Sec. 8).",
    "DINOv2 is a feature teacher only; ImageNet labels serve only the independent linear probe and class "
    "conditioning.",
    "Bootstrap / confidence intervals resample images (image-ID hash groups), never tokens; a flipped copy "
    "of an image stays in the group of its original.",
    "The split, the class map, the manifest fingerprint, the crop/resample parameters and the tokenizer "
    "checkpoint hash are persisted (this card, the manifest, the cache meta).",
]
LICENCE = ("ImageNet (ILSVRC-2012) images are used under the ImageNet terms of access (non-commercial "
           "research and educational use only; no redistribution of the images). This card, the manifest "
           "and the caches hold file names, labels and derived latents, not images.")


# ----------------------------------------------------------------------------- DATA_CARD
def replace_section(path: Path, heading: str, body: str, title: str) -> None:
    """Create or replace the `heading` section (up to the next '## ' line) of a Markdown file, keeping
    every other section. A missing file starts with `title`. Duplicated in 02_cache_latents.py."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else [title, ""]
    new = [heading, "", *body.rstrip("\n").splitlines(), ""]
    if heading in lines:
        start = lines.index(heading)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        lines[start:end] = new
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines += new
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    tmp.replace(path)


def svp_provenance(train_root: Path, manifest: datasets.Manifest) -> dict | None:
    """The svp_subset.json next to train_root (make_svp_subset.py), checked against the manifest."""
    path = train_root.parent / "svp_subset.json"
    if train_root.name != "train" or not path.is_file():
        return None
    subset = json.loads(path.read_text(encoding="utf-8"))
    want = sorted(f"train/{p}" for s in ("train_dev", "train_fit") for p in manifest.splits[s]["path"])
    got = sorted(f for f in subset["files"] if f.startswith("train/"))
    if got != want:
        raise ValueError(f"{path} lists {len(got)} train files but the manifest of {train_root} has {len(want)} "
                         "(or different names): the subset folder was modified after make_svp_subset.py")
    return {"file": path, "sha256": runtime.sha256_file(path), **{k: subset[k] for k in (
        "source", "seed", "num_classes", "train_per_class", "val_per_class", "wnids", "counts",
        "files_sha256", "mode", "numpy_version")}}


def card_section(manifest: datasets.Manifest, info: dict) -> str:
    m = manifest.meta
    counts = m["counts"]
    out = [
        f"- dataset: `{m['name']}`; {len(manifest.classes)} classes (ImageFolder order, labels 0..{len(manifest.classes) - 1})",
        f"- train root: `{m['train_root']}`",
        f"- val root: `{m['val_root']}`",
        f"- manifest: `{info['manifest_file']}` (file sha256 `{info['manifest_sha256']}`)",
        f"- manifest fingerprint (classes + splits): `{info['fingerprint']}`",
        f"- class map sha256: `{m['class_map_sha256']}`",
        f"- split seed {m['seed']}, dev images per class {m['dev_per_class']}; rule: {SPLIT_RULE} "
        f"(numpy {m['numpy_version']})",
        "",
        "| split | images | role |",
        "|---|---:|---|",
        f"| train_dev | {counts['train_dev']} | generator-free rule selection, diagnostics (D1) |",
        f"| train_fit | {counts['train_fit']} | statistics, decoder adaptation, generator training (D2) |",
        f"| val | {counts['val']} | final evaluation, gFID reference (D3) |",
        "",
        f"Preprocessing: RGB, ADM center crop to {m['image_size']} (box-halving while the short side >= "
        f"2x{m['image_size']}, then bicubic resize of the short side, central square), ToTensor + "
        f"Normalize(0.5, 0.5) -> [-1, 1] (`rwvae.datasets.image_transform`; {m['preprocessing']}).",
        "",
        "No-leakage protocol (plan Sec. 1.2):",
        *[f"- {b}" for b in NO_LEAKAGE],
        "",
        f"Licence: {LICENCE}",
    ]
    svp = info.get("svp_subset")
    if svp is not None:
        per_split = {s: sum(c.values()) for s, c in svp["counts"].items()}
        out += [
            "",
            "SVP subset provenance (`scripts/tools/make_svp_subset.py`; sanity-validation pilot, never thesis results):",
            f"- source train root `{svp['source']['train_root']}`, val root `{svp['source']['val_root']}`",
            f"- seed {svp['seed']}: {svp['num_classes']} classes x {svp['train_per_class']} train + "
            f"{svp['val_per_class']} val images ({per_split}); materialised as {svp['mode']} (numpy {svp['numpy_version']})",
            f"- wnids: {', '.join(svp['wnids'])}",
            f"- file-list sha256 `{svp['files_sha256']}`; `svp_subset.json` sha256 `{svp['sha256']}`",
        ]
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def build_parser():
    parser = _common.make_parser("Build the deterministic split manifest and DATA_CARD section 1 (plan Sec. 1.2, M2).")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing manifest whose fingerprint differs from the rebuilt one")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "01_manifest")
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        runtime.seed_everything(int(cfg.get("seed", 0)))
        runtime.select_device(args.device)  # unused: the manifest is CPU-only
        d = cfg["dataset"]
        train_root, val_root = _common.resolve(d["train_root"]), _common.resolve(d["val_root"])
        manifest = datasets.build_manifest(train_root, val_root, dev_per_class=int(d["dev_per_class"]),
                                           seed=int(d["split_seed"]), name=str(d["name"]),
                                           image_size=int(d["image_size"]))
        manifest.validate()
        if len(manifest.classes) != int(d["num_classes"]):
            raise ValueError(f"{train_root} has {len(manifest.classes)} classes but dataset.num_classes = "
                             f"{d['num_classes']}: wrong config stack or data root")
        fingerprint = manifest.fingerprint()
        path = _common.manifest_path(cfg)

        if not path.exists():
            status = "created"
        else:
            old = datasets.Manifest.load(path)
            if old.fingerprint() != fingerprint:
                if not args.force:
                    raise ValueError(f"{path} has fingerprint {old.fingerprint()} but the rebuilt manifest has "
                                     f"{fingerprint} (seed, dev_per_class or files changed); every cache, stats "
                                     "and allocation built on it would be invalidated. Pass --force to replace it.")
                status = "replaced"
                run.logger.warning("--force: replacing %s (fingerprint %s -> %s)", path, old.fingerprint(), fingerprint)
            elif old.meta.get("split_roots") != manifest.meta["split_roots"]:
                status = "roots_updated"  # same images and split, data moved: keep everything but the roots
                run.logger.warning("%s: same fingerprint, data roots moved %s -> %s; rewriting the roots",
                                   path, old.meta.get("split_roots"), manifest.meta["split_roots"])
            else:
                status = "up_to_date"
                manifest = old
                run.logger.info("%s is up to date (fingerprint %s)", path, fingerprint)
        if status != "up_to_date":
            manifest.save(path)
            run.logger.info("wrote %s (%s): %s", path, status, manifest.meta["counts"])

        info = {
            "status": status, "manifest_file": path, "manifest_sha256": runtime.sha256_file(path),
            "fingerprint": fingerprint, "class_map_sha256": manifest.meta["class_map_sha256"],
            "num_classes": len(manifest.classes), "counts": manifest.meta["counts"],
            "dev_per_class": manifest.meta["dev_per_class"], "split_seed": manifest.meta["seed"],
            "split_rule": SPLIT_RULE, "image_size": manifest.meta["image_size"],
            "preprocessing": manifest.meta["preprocessing"], "train_root": manifest.meta["train_root"],
            "val_root": manifest.meta["val_root"], "numpy_version": manifest.meta["numpy_version"],
            "manifest_created": manifest.meta["created"], "git_commit": runtime.git_commit(),
            "svp_subset": svp_provenance(train_root, manifest),
        }
        info = runtime.to_jsonable(info)
        run.save_json("manifest_info.json", info)
        card = _common.exp_dir(cfg) / "DATA_CARD.md"
        replace_section(card, SECTION, card_section(manifest, info), f"# DATA CARD -- {cfg['experiment']['name']}")
        run.logger.info("updated %s (%s)", card, SECTION)
        run.finish("ok", manifest_status=status, fingerprint=fingerprint, counts=info["counts"], manifest=path, data_card=card)
    except BaseException as e:
        run.finish("failed", error=repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
