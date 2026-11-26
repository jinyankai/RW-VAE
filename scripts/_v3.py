"""Shared glue for the v3 scripts (v3_*.py). Not part of the library API.

Layout under <experiment.run_root>/<experiment.name>/:
  tok/<arm>/            tokenizer run: checkpoints/{step:07d}.pt, tokenizer.pt (final weights)
  dit/<arm>[_repa]/     LightningDiT run on that tokenizer's posterior (rwvae.generation checkpoints, v3.json)
  gen/, recon/, analysis/<arm>/
and the posterior caches under <v3_cache.dir>/<arm>/{mu,logvar}/<split> plus <v3_cache.dir>/<arm>/stats.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import _common  # noqa: F401  (puts the repo on sys.path)
from _common import resolve, stage_dir


def add_arm_arg(parser) -> None:
    parser.add_argument("--arm", required=True, help="arm preset from v3_arms (T-a ... T-f, L-a, L-c)")


def _merge(base: dict, new: dict) -> dict:
    for k, v in new.items():
        base[k] = _merge(dict(base.get(k) or {}), v) if isinstance(v, dict) else v
    return base


def apply_arm(cfg: dict, arm: str) -> dict:
    """cfg with the arm preset merged in (deep copy); records cfg['arm'] = arm."""
    arms = cfg.get("v3_arms") or {}
    if arm not in arms:
        raise ValueError(f"unknown v3 arm {arm!r}; configs define {sorted(arms)}")
    out = _merge(copy.deepcopy(cfg), copy.deepcopy(arms[arm] or {}))
    out["arm"] = arm
    if out["kl"]["mode"] == "lock" and out["kl"].get("target") is None:
        raise ValueError(f"arm {arm} locks the rate: set --set kl.target=<R* nats/token> (T-c's measured R_hat)")
    return out


def tok_dir(cfg: dict, arm: str) -> Path:
    return stage_dir(cfg, "tok", arm)


def tokenizer_path(cfg: dict, arm: str) -> Path:
    return tok_dir(cfg, arm) / "tokenizer.pt"


def cache_root(cfg: dict, arm: str) -> Path:
    return resolve(cfg["v3_cache"]["dir"]) / arm


def cache_dirs(cfg: dict, arm: str, split: str) -> tuple[Path, Path]:
    root = cache_root(cfg, arm)
    return root / "mu" / split, root / "logvar" / split


def stats_path(cfg: dict, arm: str) -> Path:
    return cache_root(cfg, arm) / "stats.json"


def load_stats(cfg: dict, arm: str) -> dict:
    path = stats_path(cfg, arm)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run scripts/v3_cache_posterior.py --arm {arm} first")
    return json.loads(path.read_text(encoding="utf-8"))


def dit_dir(cfg: dict, arm: str, repa: bool) -> Path:
    return stage_dir(cfg, "dit", f"{arm}_repa" if repa else arm)


class InitBaseline:
    """The init tokenizer (VA-VAE, init.ckpt) behind the TokenizerV3 interface the v3 scripts use (encode -> {mu,
    logvar, alpha=None}, decode, cfg), so the baseline goes through exactly the same cache / DiT / evaluation
    protocol as a v3 arm. Its cfg asks DINO only for the last block (the REPA / probe target)."""

    def __init__(self, frozen, target_layer: int):
        from rwvae.tokenizer_v3 import V3Config
        self.frozen = frozen
        self.cfg = V3Config(dino_layers=(int(target_layer),))

    def encode(self, images, dino_layers=None, freeze_gate: bool = False) -> dict:
        mu, logvar = self.frozen.encode_moments(images)
        return {"mu": mu, "logvar": logvar, "alpha": None}

    def decode(self, z):
        return self.frozen.decode(z)


def load_arm_tokenizer(cfg: dict, arm: str, device, path=None):
    """(tokenizer, path, checkpoint dict): `path` if given, else the arm's tokenizer.pt, or for an arm with
    baseline_init (configs/v3.yaml v3_arms) the init tokenizer wrapped in InitBaseline."""
    from rwvae import integration
    from rwvae import tokenizer_v3 as tv3
    if path is None and cfg.get("baseline_init"):
        init = cfg["init"]
        path = resolve(init["ckpt"])
        frozen = integration.FrozenTokenizer.from_checkpoint(path, embed_dim=int(init["embed_dim"]),
                                                             ch_mult=tuple(init["ch_mult"]), device=device)
        return InitBaseline(frozen, max(cfg["v3"]["dino_layers"])), path, {}
    path = Path(path) if path is not None else tokenizer_path(cfg, arm)
    tok, ckpt = tv3.load_tokenizer(path, device)
    return tok, path, ckpt


def make_dino(cfg: dict, device):
    """Frozen DINOv2 with forward_layers (rwvae.integration.DinoPatchFeatures)."""
    return _common.make_semantic(cfg, device)


def dino_inputs(dino, images, v3cfg, amp: bool):
    """(layers [B, L, D, g, g] in v3cfg.dino_layers order, target [B, D, g, g] = deepest used block)."""
    import torch
    order = sorted(set(v3cfg.dino_layers) | {v3cfg.target_layer})
    with torch.no_grad(), torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
        feats = dino.forward_layers(images, order).float()
    pos = {b: i for i, b in enumerate(order)}
    layers = feats[:, [pos[b] for b in v3cfg.dino_layers]]
    return layers, feats[:, pos[v3cfg.target_layer]]
