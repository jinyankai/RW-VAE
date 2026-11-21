"""Shared glue for the v2 pipeline scripts (00..07). Not part of the library API.

Conventions
- Relative paths in configs are resolved against the repo root, never the CWD.
- ``cfg["env"]`` (optional) is copied into ``os.environ`` before third-party imports
  (HF_HOME / HF_ENDPOINT / TORCH_HOME for timm, lpips and pytorch-fid weights).
- Every stage writes under ``<experiment.run_root>/<experiment.name>/<stage>/``.
- An *arm* is a label from ``configs/arms.yaml`` (B0..B3) plus, for lossy rules, a nominal
  rate R (nats/token) and, for rule r2, a relevance weight w.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows numpy(MKL) + torch OpenMP guard

from rwvae import runtime  # noqa: E402


def resolve(path) -> Path:
    """Absolute paths unchanged; relative paths are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def apply_env(cfg: dict) -> None:
    """Export cfg['env'] entries that are not already set (call before heavy imports)."""
    for key, value in (cfg.get("env") or {}).items():
        os.environ.setdefault(str(key), str(value))


def exp_dir(cfg: dict) -> Path:
    exp = cfg["experiment"]
    return resolve(exp["run_root"]) / exp["name"]


def stage_dir(cfg: dict, stage: str, *parts: str) -> Path:
    return exp_dir(cfg).joinpath(stage, *parts)


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    runtime.add_common_args(parser)
    return parser


def load_cfg(args) -> dict:
    cfg = runtime.parse_config_from_args(args)
    apply_env(cfg)
    return cfg


def open_run(script: str, cfg: dict, out_dir, args) -> "runtime.Run":
    """runtime.Run for a script; cfg['registry'] (optional) redirects the run registry (tests)."""
    registry = cfg.get("registry")
    return runtime.Run(Path(script).name, cfg, out_dir, registry=None if registry is None else resolve(registry),
                       overwrite=bool(getattr(args, "overwrite", False)), resume=bool(getattr(args, "resume", False)))


def rank_world() -> tuple[int, int]:
    """(RANK, WORLD_SIZE) from the torchrun environment; (0, 1) for a plain process."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def set_tf32(enabled: bool) -> dict:
    """Set TF32 for CUDA matmul and cuDNN; returns the setting for the artifact meta.

    torch enables TF32 cuDNN convolutions by default; evaluation and latent caching turn it off
    so that decoder/encoder/Inception/LPIPS numbers do not depend on that default.
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    return {"tf32_matmul": bool(enabled), "tf32_cudnn": bool(enabled)}


# --------------------------------------------------------------------------- models
def make_tokenizer(cfg: dict, device):
    """Frozen tokenizer from cfg['tokenizer'] (strict checkpoint load, posterior-mean encoder)."""
    from rwvae import integration
    t = cfg["tokenizer"]
    return integration.FrozenTokenizer.from_checkpoint(
        resolve(t["ckpt"]), embed_dim=int(t["embed_dim"]), ch_mult=tuple(t["ch_mult"]), device=device)


def make_semantic(cfg: dict, device):
    """DINOv2 patch-feature teacher from cfg['semantic'] ([B, D, 16, 16] for 224 inputs)."""
    from rwvae import integration
    s = cfg["semantic"]
    ckpt = s.get("checkpoint_path")
    return integration.DinoPatchFeatures(s["model"], input_size=int(s["input_size"]), device=device,
                                         checkpoint_path=None if ckpt is None else resolve(ckpt))


def build_dit(cfg: dict, num_classes: int):
    """LightningDiT from cfg['dit']; dit.torch_compile 'auto' -> integration's automatic choice."""
    from rwvae import integration
    dit = dict(cfg["dit"])
    flag = dit.pop("torch_compile", "auto")
    integration.configure_torch_compile(None if flag in (None, "auto") else bool(flag))
    return integration.build_lightningdit(num_classes=int(num_classes), **dit)


def dit_train_config(cfg: dict):
    """DiTTrainConfig from cfg['train'], keeping only its fields (the other `train` keys configure
    the dataset: split, flip_prob, resample_noise_each_epoch, latent_multiplier, noise_seed)."""
    from dataclasses import fields
    from rwvae.generation import DiTTrainConfig
    names = {f.name for f in fields(DiTTrainConfig)}
    return DiTTrainConfig.from_dict({k: v for k, v in cfg["train"].items() if k in names})


def preregistration_check(cfg: dict, arm: str, R, w, exploratory: bool) -> dict:
    """Plan §8: generator training of a lossy arm only at the frozen (R*, w*), unless exploratory.

    Returns {"registered": bool, "exploratory": bool, "R_star", "w_star"} for the run record;
    raises ValueError when a confirmatory run would use an unregistered operating point.
    """
    rule = arm_rule(cfg, arm)
    pre = cfg.get("preregistration") or {}
    record = {"registered": rule == "none", "exploratory": bool(exploratory),
              "R_star": pre.get("R_star"), "w_star": pre.get("w_star")}
    if rule == "none":
        return record
    frozen = bool(pre.get("frozen"))
    ok = frozen and pre.get("R_star") is not None and float(R) == float(pre["R_star"])
    if rule == "r2":
        ok = ok and pre.get("w_star") is not None and float(w) == float(pre["w_star"])
    record["registered"] = bool(ok)
    if not ok and not exploratory:
        raise ValueError(
            f"arm {arm} at R={R}, w={w} is not the frozen pre-registered operating point "
            f"(preregistration: frozen={frozen}, R_star={pre.get('R_star')}, w_star={pre.get('w_star')}). "
            "Choose R*, w* from 05_eval_recon_probe.py results only, freeze them in configs/arms.yaml, "
            "or pass --exploratory to mark this run as exploratory in the run registry.")
    return record


# --------------------------------------------------------------------------- data access
def manifest_path(cfg: dict) -> Path:
    return resolve(cfg["dataset"]["manifest"])


def load_manifest(cfg: dict):
    from rwvae.datasets import Manifest
    return Manifest.load(manifest_path(cfg))


def cache_dir(cfg: dict, split: str) -> Path:
    return resolve(cfg["cache"]["dir"]) / split


def open_cache(cfg: dict, split: str, manifest=None):
    from rwvae.datasets import LatentCache
    return LatentCache(cache_dir(cfg, split), manifest=manifest, verify=True)


def stats_path(cfg: dict) -> Path:
    return stage_dir(cfg, "stats", "stats.npz")


def load_stats(cfg: dict):
    from rwvae.stats import LatentStats
    path = stats_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/03_estimate_stats.py first")
    return LatentStats.load(path)


# --------------------------------------------------------------------------- arms
def arm_rule(cfg: dict, arm: str) -> str:
    arms = cfg.get("arms") or {}
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}; configs define {sorted(arms)}")
    return arms[arm]["rule"]


def allocation_name(rule: str, R: float | None, w: float | None) -> str:
    if rule == "none":
        return "none"
    if rule == "r2":
        return f"r2_R{float(R):g}_w{float(w):g}"
    return f"{rule}_R{float(R):g}"


def arm_label(arm: str, rule: str, R: float | None, w: float | None) -> str:
    return f"{arm}_{allocation_name(rule, R, w)}"


def allocations_dir(cfg: dict) -> Path:
    return stage_dir(cfg, "allocations")


def load_allocation(cfg: dict, arm: str, R: float | None, w: float | None):
    """Allocation for an arm: the saved file from 04 for lossy rules, built on the fly for B0."""
    from rwvae.allocation import Allocation, solve_allocation
    rule = arm_rule(cfg, arm)
    if rule == "none":
        return solve_allocation(load_stats(cfg), "none", None, None)
    if R is None or (rule == "r2" and w is None):
        raise ValueError(f"arm {arm} (rule {rule}) needs --R" + (" and --w" if rule == "r2" else ""))
    path = allocations_dir(cfg) / f"{allocation_name(rule, R, w)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/04_solve_channel.py first")
    return Allocation.load(path)
