"""v3 -- evaluate a v3 arm: reconstruction (recon) and class-conditional generation (generate).

recon     v3_eval.max_images evenly spaced rows of v3_eval.split: encode (posterior mean), decode; PSNR, LPIPS,
          rFID (reconstructions vs the same originals) and the measured rate R_hat (nats/token) -> recon.json.
generate  EMA weights of a v3 DiT run; class-balanced labels and per-index initial noise as in 07 (sample.*);
          latents are de-normalised with the run's stats and decoded by the arm's tokenizer. gFID against the
          reference statistics of `07_evaluate.py ref-stats` (fid.reference_stats, or <run>/fid_ref/<split>.npz).
          --cfg-scale > 1 uses classifier-free guidance on ALL channels (the v3 latent has no privileged first
          three channels) inside the guidance interval t >= sample.cfg_interval_start (as LightningDiT);
          --cfg-scale 1 calls the model directly, bit-identical to the unguided path. --uncond samples every image
          with the null class (the DiT learns it from the 10% label dropout): the T3 diagnostic, where the model
          must pick the class itself. CFG and --uncond need dit.class_dropout_prob > 0.
cfg-sweep gFID at each --scales value against the train_dev reference (tuning set, never val) -> sweep.json with
          best_scale; report that scale once on val with `generate --cfg-scale <best_scale>`.

Examples
  python scripts/v3_evaluate.py recon <configs...> --arm T-c
  python scripts/v3_evaluate.py generate <configs...> --arm T-c [--repa] --num-samples 10000
  python scripts/07_evaluate.py ref-stats <configs...> --split train_dev      # tuning reference, once
  python scripts/v3_evaluate.py cfg-sweep <configs...> --arm T-c --scales 1.25 1.5 2 3
  python scripts/v3_evaluate.py generate <configs...> --arm T-c --uncond
"""
from __future__ import annotations

import _common  # noqa: I001

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import _v3
from rwvae import datasets, evaluation, generation, integration, runtime
from rwvae import tokenizer_v3 as tv3

_CKPT = re.compile(r"(\d+)\.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("recon", "generate", "cfg-sweep"):
        p = sub.add_parser(name)
        runtime.add_common_args(p)
        _v3.add_arm_arg(p)
        p.add_argument("--tokenizer", default=None, help="tokenizer.pt (default: the arm's tok run)")
    rec = sub.choices["recon"]
    rec.add_argument("--baseline", action="store_true", help="evaluate the init tokenizer (init.ckpt) instead")
    rec.add_argument("--save-grid", type=int, default=16, help="(original, reconstruction) pairs saved as recon_pairs.png")
    for name in ("generate", "cfg-sweep"):
        g = sub.choices[name]
        g.add_argument("--repa", action="store_true", help="use the arm's REPA DiT run")
        g.add_argument("--dit-run", default=None, help="DiT run directory (default: dit/<arm>[_repa])")
        g.add_argument("--ckpt", default="latest", help="checkpoint step or 'latest'")
        g.add_argument("--weights", choices=("ema", "model"), default="ema",
                       help="EMA or raw DiT weights of the checkpoint (outputs of 'model' are tagged _raw)")
        g.add_argument("--num-samples", type=int, default=None, help="default sample.num_samples_screen")
        g.add_argument("--cfg-interval-start", type=float, default=None,
                       help="no guidance for t below this (t = 0 is noise); default sample.cfg_interval_start")
        g.add_argument("--ref-stats", default=None, help="reference FID statistics npz")
    gen = sub.choices["generate"]
    gen.add_argument("--cfg-scale", type=float, default=1.0)
    gen.add_argument("--uncond", action="store_true", help="sample every image with the null class (T3 diagnostic)")
    sub.choices["cfg-sweep"].add_argument("--scales", type=float, nargs="+", default=[1.25, 1.5, 2.0, 3.0],
                                          help="CFG scales to evaluate against the train_dev reference")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    return {"recon": cmd_recon, "generate": cmd_generate, "cfg-sweep": cmd_cfg_sweep}[args.command](args, cfg)


def _tokenizer(cfg, args, device):
    """(tokenizer, path, checkpoint dict): --tokenizer, else the arm's tokenizer.pt."""
    return _v3.load_arm_tokenizer(cfg, args.arm, device, args.tokenizer)


def _tok_tag(args) -> str:
    """'' for the arm's own tokenizer.pt, else '_<file stem>' (e.g. _tokenizer_gan), plus '_raw' for raw DiT
    weights, so outputs do not collide."""
    tag = "" if not getattr(args, "tokenizer", None) else f"_{Path(args.tokenizer).stem}"
    return tag + ("_raw" if getattr(args, "weights", "ema") == "model" else "")


class AllChannelCFG(nn.Module):
    """model(x, t, y) with classifier-free guidance on every channel; scale 1 is the plain model.

    Guidance interval as LightningDiT (sample.cfg_interval_start): with t = 0 noise and t = 1 data, samples whose
    t < interval_start get the conditional prediction unguided (the noisiest steps), the others
    uncond + scale * (cond - uncond)."""

    def __init__(self, model: nn.Module, scale: float, null_class: int, interval_start: float = 0.0):
        super().__init__()
        self.model, self.scale, self.null_class = model, float(scale), int(null_class)
        self.interval_start = float(interval_start)

    def forward(self, x, t, y):
        if self.scale == 1.0:
            return self.model(x, t, y)
        out = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, self.null_class)]))
        cond, uncond = out.chunk(2)
        guided = uncond + self.scale * (cond - uncond)
        on = (t >= self.interval_start).view(-1, *([1] * (x.dim() - 1)))
        return torch.where(on, guided, cond)


def cmd_recon(args, cfg: dict) -> int:
    e = cfg["v3_eval"]
    name = "init_baseline" if args.baseline else args.arm + _tok_tag(args)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "recon", name)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        if args.baseline:                    # the init tokenizer (VA-VAE) through the same protocol
            init = cfg["init"]
            tok_path = _common.resolve(init["ckpt"])
            base = integration.FrozenTokenizer.from_checkpoint(tok_path, embed_dim=int(init["embed_dim"]),
                                                               ch_mult=tuple(init["ch_mult"]), device=device)
        else:
            tok, tok_path, _ = _tokenizer(cfg, args, device)
            dino = _v3.make_dino(cfg, device)
        manifest = _common.load_manifest(cfg)
        n_split = manifest.split_size(e["split"])
        N = n_split if e["max_images"] is None else min(int(e["max_images"]), n_split)
        rows = [i * n_split // N for i in range(N)]
        ds = datasets.ManifestImageDataset(manifest, e["split"], image_size=int(cfg["dataset"]["image_size"]),
                                           indices=rows)
        inception = integration.load_inception(device)
        fid_rec = evaluation.FIDStatistics(inception, device, batch_size=int(cfg["fid"]["inception_batch"]))
        fid_ref = evaluation.FIDStatistics(inception, device, batch_size=int(cfg["fid"]["inception_batch"]))
        lp = evaluation.LPIPSMeter(integration.load_lpips(device))
        psnr, kl_sum, n_tok, grid = [], 0.0, 0, []
        amp = device.type == "cuda"
        with torch.no_grad():
            for x, _, _ in DataLoader(ds, batch_size=int(e["batch_size"]), num_workers=int(e["num_workers"])):
                x = x.to(device)
                if args.baseline:
                    mu, logvar = base.encode_moments(x)
                    enc, x_rec = {"mu": mu, "logvar": logvar}, base.decode(mu).float()
                else:
                    layers, _ = _v3.dino_inputs(dino, x, tok.cfg, amp)
                    enc = tok.encode(x, layers)
                    x_rec = tok.decode(enc["mu"]).float()
                kl_sum += float(tv3.kl_elementwise(enc["mu"].double(), enc["logvar"].double()).sum())
                n_tok += x.shape[0] * enc["mu"].shape[2] * enc["mu"].shape[3]
                u8_rec, u8_ref = evaluation.to_uint8(x_rec), evaluation.to_uint8(x)
                psnr.append(evaluation.psnr_per_image(u8_ref, u8_rec).cpu())
                lp.update(x_rec, x)
                fid_rec.update(u8_rec)
                fid_ref.update(u8_ref)
                if sum(len(g) for g in grid) < 2 * int(args.save_grid):
                    k = int(args.save_grid) - sum(len(g) for g in grid) // 2
                    grid.append(torch.stack([u8_ref[:k].cpu(), u8_rec[:k].cpu()], 1).flatten(0, 1))
                run.log_metrics(None, images=sum(len(p) for p in psnr))
        mu1, s1, _ = fid_rec.compute()
        mu2, s2, _ = fid_ref.compute()
        if grid:                             # pairs (original, reconstruction), row-major
            _save_grid(torch.cat(grid), out_dir / "recon_pairs.png", cols=8)
        res = {"arm": name, "split": e["split"], "n": N, "tokenizer": tok_path, "latent": "posterior mean",
               "psnr": float(torch.cat(psnr).double().mean()), "lpips": lp.mean(),
               "rfid": evaluation.frechet_distance(mu1, s1, mu2, s2), "rate_nats_per_token": kl_sum / n_tok,
               "tf32": tf32}
        run.save_json("recon.json", res)
        run.logger.info("recon %s: %s", name, {k: res[k] for k in ("psnr", "lpips", "rfid", "rate_nats_per_token")})
        run.finish("ok", **{k: res[k] for k in ("psnr", "lpips", "rfid", "rate_nats_per_token", "n")})
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


def _pick(dit_run: Path, which: str) -> Path:
    found = sorted((int(m.group(1)), p) for p in (dit_run / "checkpoints").iterdir() if (m := _CKPT.fullmatch(p.name)))
    if not found:
        raise FileNotFoundError(f"no checkpoints in {dit_run / 'checkpoints'}")
    return found[-1][1] if which == "latest" else dict(found)[int(which)]


class _Generator:
    """EMA DiT of a v3 run + the arm's tokenizer: draws N samples at one guidance setting and returns their
    Inception statistics and the first 64 images. Labels and initial noise follow sample.* (the 07 protocol), so every
    scale and arm sees the same labels and x0 per sample index; uncond replaces every label by the null class."""

    def __init__(self, cfg, args, device):
        self.cfg, self.device = cfg, device
        self.dit_run = Path(args.dit_run) if args.dit_run else _v3.dit_dir(cfg, args.arm, args.repa)
        self.ckpt_path = _pick(self.dit_run, args.ckpt)
        self.step = int(_CKPT.fullmatch(self.ckpt_path.name).group(1))
        self.info = json.loads((self.dit_run / "v3.json").read_text(encoding="utf-8"))
        self.tok, tok_path, ckpt = _tokenizer(cfg, args, device)
        # the DiT's own tokenizer, or one whose encoder is that tokenizer's (a decoder-only fine-tune)
        if self.info["tokenizer_sha256"] not in (runtime.sha256_file(tok_path), ckpt.get("encoder_from_sha256")):
            raise ValueError(f"{tok_path} is not the tokenizer the DiT in {self.dit_run} was trained on, nor a "
                             "decoder fine-tune of it")
        train_cfg = runtime.load_config(self.dit_run / "config.yaml")
        self.num_classes = int(self.info["data"]["num_classes"])
        model = _common.build_dit(train_cfg, num_classes=self.num_classes)
        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu", weights_only=True)[args.weights])
        self.model = model.to(device).eval()
        self.has_null = float(model.y_embedder.dropout_prob) > 0      # the null-class row exists only then
        self.sample_fn = integration.build_ode_sample_fn(integration.build_transport(train_cfg["transport"]),
                                                         cfg["sample"])
        self.inception = integration.load_inception(device)

    def run(self, N: int, scale: float, interval_start: float, uncond: bool, log=None):
        s = self.cfg["sample"]
        if (scale != 1.0 or uncond) and not self.has_null:
            raise ValueError(f"{self.dit_run} was trained with dit.class_dropout_prob = 0: it has no null class, so "
                             "neither CFG nor unconditional sampling is defined")
        if uncond and scale != 1.0:
            raise ValueError("unconditional sampling takes no CFG scale")
        net = AllChannelCFG(self.model, scale, self.num_classes, interval_start).eval()
        labels = generation.class_balanced_labels(N, self.num_classes, int(s["label_seed"]))
        if uncond:
            labels = torch.full_like(labels, self.num_classes)
        fid = evaluation.FIDStatistics(self.inception, self.device, batch_size=int(self.cfg["fid"]["inception_batch"]))
        norm, batch, keep, t0 = self.info["normalisation"], int(s["batch_size"]), [], time.perf_counter()
        for start in range(0, N, batch):
            idx = list(range(start, min(start + batch, N)))
            z = generation.sample_latents(net, self.sample_fn, labels[idx], idx, base_seed=int(s["noise_seed"]),
                                          batch_size=batch, device=self.device,
                                          latent_shape=self.info["data"]["latent_shape"],
                                          mixed_precision=s.get("mixed_precision", "bf16"))
            with torch.no_grad():
                u8 = evaluation.to_uint8(self.tok.decode(
                    generation.denormalize(z, norm["mean"], norm["std"]).to(self.device)))
            fid.update(u8)
            if len(keep) * batch < 64:
                keep.append(u8.cpu())
            if log is not None:
                log(samples=idx[-1] + 1, seconds=time.perf_counter() - t0)
        mu, sigma, n = fid.compute()
        return mu, sigma, n, torch.cat(keep)[:64], time.perf_counter() - t0


def _refs(cfg, path_arg, split: str | None = None) -> dict:
    """{label: npz} reference statistics: --ref-stats, else <run>/fid_ref/<split>.npz when split is given, else
    fid.reference_stats, else <run>/fid_ref/<fid.reference_split>.npz."""
    if path_arg:
        refs = {"reference": _common.resolve(path_arg)}
    elif split is None and cfg["fid"].get("reference_stats"):
        refs = {"reference": _common.resolve(cfg["fid"]["reference_stats"])}
    else:
        split = split or cfg["fid"]["reference_split"]
        refs = {split: _common.stage_dir(cfg, "fid_ref", f"{split}.npz")}
    for label, path in refs.items():
        if not path.is_file():
            raise FileNotFoundError(f"reference {label}: {path} not found (run `07_evaluate.py ref-stats`)")
    return refs


def _gfid(mu, sigma, refs: dict) -> dict:
    out = {}
    for label, path in refs.items():
        mu_r, s_r, _, _ = evaluation.load_fid_stats(path)
        out[label] = evaluation.frechet_distance(mu, sigma, mu_r, s_r)
    return out


def _interval(args, cfg) -> float:
    if args.cfg_interval_start is not None:
        return float(args.cfg_interval_start)
    return float(cfg["sample"].get("cfg_interval_start", 0.0))


def _dit_step(args, cfg) -> tuple[Path, int]:
    dit_run = Path(args.dit_run) if args.dit_run else _v3.dit_dir(cfg, args.arm, args.repa)
    return dit_run, int(_CKPT.fullmatch(_pick(dit_run, args.ckpt).name).group(1))


def cmd_generate(args, cfg: dict) -> int:
    N = int(args.num_samples or cfg["sample"]["num_samples_screen"])
    start = _interval(args, cfg)
    tag = "uncond" if args.uncond else (f"cfg{args.cfg_scale:g}_int{start:g}" if args.cfg_scale != 1.0 else "nocfg")
    dit_run, step = _dit_step(args, cfg)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "gen", dit_run.name,
                                                               f"step{step}_n{N}_{tag}{_tok_tag(args)}")
    refs = _refs(cfg, args.ref_stats)                 # before open_run: a missing reference leaves no run dir
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        _common.set_tf32(False)
        gen = _Generator(cfg, args, device)
        mu, sigma, n, grid, seconds = gen.run(N, args.cfg_scale, start, args.uncond,
                                              log=lambda **kw: run.log_metrics(None, **kw))
        evaluation.save_fid_stats(out_dir / "gen_fid_stats.npz", mu, sigma, n,
                                  {"kind": "generated", "arm": args.arm, "dit_run": str(dit_run), "step": step,
                                   "cfg_scale": args.cfg_scale, "cfg_interval_start": start, "uncond": args.uncond})
        gfid = _gfid(mu, sigma, refs)
        _save_grid(grid, out_dir / "samples.png")
        res = {"arm": args.arm, "dit_run": dit_run, "step": step, "n": n, "cfg_scale": args.cfg_scale,
               "cfg_interval_start": start, "uncond": args.uncond, "gfid": gfid, "seconds": seconds}
        run.save_json("gen.json", res)
        run.logger.info("gFID %s step %d n=%d %s: %s", dit_run.name, step, n, tag, gfid)
        run.finish("ok", gfid=gfid, n=n, step=step, cfg_scale=args.cfg_scale, uncond=args.uncond)
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


def cmd_cfg_sweep(args, cfg: dict) -> int:
    """gFID at every --scales value against the TUNING reference (default <run>/fid_ref/train_dev.npz from
    `07_evaluate.py ref-stats --split train_dev`), never val; the chosen scale is then reported once on val with
    `generate --cfg-scale <best_scale>`."""
    N = int(args.num_samples or cfg["sample"]["num_samples_screen"])
    start = _interval(args, cfg)
    dit_run, step = _dit_step(args, cfg)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "cfg_sweep", dit_run.name,
                                                               f"step{step}_n{N}_int{start:g}{_tok_tag(args)}")
    refs = _refs(cfg, args.ref_stats, split="train_dev")
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        _common.set_tf32(False)
        label = next(iter(refs))
        gen = _Generator(cfg, args, device)
        rows = []
        for scale in args.scales:
            mu, sigma, n, grid, seconds = gen.run(N, float(scale), start, False)
            g = _gfid(mu, sigma, refs)[label]
            rows.append({"cfg_scale": float(scale), "gfid": g, "n": n, "seconds": seconds})
            _save_grid(grid, out_dir / f"samples_cfg{float(scale):g}.png")
            run.log_metrics(None, cfg_scale=float(scale), gfid=g)
            run.logger.info("cfg %.3g (interval start %.3g): gFID vs %s = %.4f", float(scale), start, label, g)
        best = min(rows, key=lambda r: r["gfid"])
        run.save_json("sweep.json", {"arm": args.arm, "dit_run": dit_run, "step": step, "reference": refs,
                                     "cfg_interval_start": start, "rows": rows, "best_scale": best["cfg_scale"]})
        run.finish("ok", best_scale=best["cfg_scale"], best_gfid=best["gfid"], reference=label)
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


def _save_grid(u8: torch.Tensor, path: Path, cols: int | None = None) -> None:
    from PIL import Image
    n = u8.shape[0]
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    _, _, H, W = u8.shape
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = u8[k].permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
