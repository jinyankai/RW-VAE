"""v3 -- evaluate a v3 arm: reconstruction (recon) and class-conditional generation (generate).

recon     v3_eval.max_images evenly spaced rows of v3_eval.split: encode (posterior mean), decode; PSNR, LPIPS,
          rFID (reconstructions vs the same originals) and the measured rate R_hat (nats/token) -> recon.json.
generate  EMA weights of a v3 DiT run; class-balanced labels and per-index initial noise as in 07 (sample.*);
          latents are de-normalised with the run's stats and decoded by the arm's tokenizer. gFID against the
          reference statistics of `07_evaluate.py ref-stats` (fid.reference_stats, or <run>/fid_ref/<split>.npz).
          --cfg-scale > 1 uses classifier-free guidance on ALL channels (the v3 latent has no privileged first
          three channels); --cfg-scale 1 calls the model directly, bit-identical to the unguided path.

Examples
  python scripts/v3_evaluate.py recon <configs...> --arm T-c
  python scripts/v3_evaluate.py generate <configs...> --arm T-c [--repa] --num-samples 10000
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
    for name in ("recon", "generate"):
        p = sub.add_parser(name)
        runtime.add_common_args(p)
        _v3.add_arm_arg(p)
        p.add_argument("--tokenizer", default=None, help="tokenizer.pt (default: the arm's tok run)")
    g = sub.choices["generate"]
    g.add_argument("--repa", action="store_true", help="use the arm's REPA DiT run")
    g.add_argument("--dit-run", default=None, help="DiT run directory (default: dit/<arm>[_repa])")
    g.add_argument("--ckpt", default="latest", help="checkpoint step or 'latest'")
    g.add_argument("--num-samples", type=int, default=None, help="default sample.num_samples_screen")
    g.add_argument("--cfg-scale", type=float, default=1.0)
    g.add_argument("--ref-stats", default=None, help="reference FID statistics npz (default: fid config)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _v3.apply_arm(_common.load_cfg(args), args.arm)
    return {"recon": cmd_recon, "generate": cmd_generate}[args.command](args, cfg)


def _tokenizer(cfg, args, device):
    path = Path(args.tokenizer) if args.tokenizer else _v3.tokenizer_path(cfg, args.arm)
    tok, _ = tv3.load_tokenizer(path, device)
    return tok, path


class AllChannelCFG(nn.Module):
    """model(x, t, y) with classifier-free guidance on every channel; scale 1 is the plain model."""

    def __init__(self, model: nn.Module, scale: float, null_class: int):
        super().__init__()
        self.model, self.scale, self.null_class = model, float(scale), int(null_class)

    def forward(self, x, t, y):
        if self.scale == 1.0:
            return self.model(x, t, y)
        out = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, self.null_class)]))
        cond, uncond = out.chunk(2)
        return uncond + self.scale * (cond - uncond)


def cmd_recon(args, cfg: dict) -> int:
    e = cfg["v3_eval"]
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "recon", args.arm)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        tok, tok_path = _tokenizer(cfg, args, device)
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
        psnr, kl_sum, n_tok = [], 0.0, 0
        amp = device.type == "cuda"
        with torch.no_grad():
            for x, _, _ in DataLoader(ds, batch_size=int(e["batch_size"]), num_workers=int(e["num_workers"])):
                x = x.to(device)
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
                run.log_metrics(None, images=sum(len(p) for p in psnr))
        mu1, s1, _ = fid_rec.compute()
        mu2, s2, _ = fid_ref.compute()
        res = {"arm": args.arm, "split": e["split"], "n": N, "tokenizer": tok_path, "latent": "posterior mean",
               "psnr": float(torch.cat(psnr).double().mean()), "lpips": lp.mean(),
               "rfid": evaluation.frechet_distance(mu1, s1, mu2, s2), "rate_nats_per_token": kl_sum / n_tok,
               "tf32": tf32}
        run.save_json("recon.json", res)
        run.logger.info("recon %s: %s", args.arm, {k: res[k] for k in ("psnr", "lpips", "rfid", "rate_nats_per_token")})
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


def cmd_generate(args, cfg: dict) -> int:
    s = cfg["sample"]
    dit_run = Path(args.dit_run) if args.dit_run else _v3.dit_dir(cfg, args.arm, args.repa)
    ckpt_path = _pick(dit_run, args.ckpt)
    step = int(_CKPT.fullmatch(ckpt_path.name).group(1))
    N = int(args.num_samples or s["num_samples_screen"])
    tag = f"cfg{args.cfg_scale:g}" if args.cfg_scale != 1.0 else "nocfg"
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "gen", dit_run.name, f"step{step}_n{N}_{tag}")
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        device = runtime.select_device(args.device)
        _common.set_tf32(False)
        info = json.loads((dit_run / "v3.json").read_text(encoding="utf-8"))
        tok, tok_path = _tokenizer(cfg, args, device)
        if runtime.sha256_file(tok_path) != info["tokenizer_sha256"]:
            raise ValueError(f"{tok_path} is not the tokenizer the DiT in {dit_run} was trained on")
        train_cfg = runtime.load_config(dit_run / "config.yaml")
        num_classes = int(info["data"]["num_classes"])
        model = _common.build_dit(train_cfg, num_classes=num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["ema"])
        model.to(device).eval()
        guided = AllChannelCFG(model, args.cfg_scale, num_classes).eval()
        sample_fn = integration.build_ode_sample_fn(integration.build_transport(train_cfg["transport"]), s)
        refs = {"reference": _common.resolve(args.ref_stats)} if args.ref_stats else (
            {"reference": _common.resolve(cfg["fid"]["reference_stats"])} if cfg["fid"].get("reference_stats")
            else {cfg["fid"]["reference_split"]: _common.stage_dir(cfg, "fid_ref", f"{cfg['fid']['reference_split']}.npz")})
        for label, path in refs.items():
            if not path.is_file():
                raise FileNotFoundError(f"reference {label}: {path} not found (run `07_evaluate.py ref-stats`)")
        fid = evaluation.FIDStatistics(integration.load_inception(device), device,
                                       batch_size=int(cfg["fid"]["inception_batch"]))
        norm = info["normalisation"]
        labels = generation.class_balanced_labels(N, num_classes, int(s["label_seed"]))
        batch, keep, t0 = int(s["batch_size"]), [], time.perf_counter()
        for start in range(0, N, batch):
            idx = list(range(start, min(start + batch, N)))
            z = generation.sample_latents(guided, sample_fn, labels[idx], idx, base_seed=int(s["noise_seed"]),
                                          batch_size=batch, device=device, latent_shape=info["data"]["latent_shape"],
                                          mixed_precision=s.get("mixed_precision", "bf16"))
            with torch.no_grad():
                u8 = evaluation.to_uint8(tok.decode(generation.denormalize(z, norm["mean"], norm["std"]).to(device)))
            fid.update(u8)
            if len(keep) * batch < 64:
                keep.append(u8.cpu())
            run.log_metrics(None, samples=idx[-1] + 1, seconds=time.perf_counter() - t0)
        mu, sigma, n = fid.compute()
        evaluation.save_fid_stats(out_dir / "gen_fid_stats.npz", mu, sigma, n,
                                  {"kind": "generated", "arm": args.arm, "dit_run": str(dit_run), "step": step,
                                   "cfg_scale": args.cfg_scale})
        gfid = {}
        for label, path in refs.items():
            mu_r, s_r, _, _ = evaluation.load_fid_stats(path)
            gfid[label] = evaluation.frechet_distance(mu, sigma, mu_r, s_r)
        grid = torch.cat(keep)[:64]
        _save_grid(grid, out_dir / "samples.png")
        res = {"arm": args.arm, "dit_run": dit_run, "step": step, "n": n, "cfg_scale": args.cfg_scale, "gfid": gfid,
               "seconds": time.perf_counter() - t0}
        run.save_json("gen.json", res)
        run.logger.info("gFID %s step %d n=%d %s: %s", dit_run.name, step, n, tag, gfid)
        run.finish("ok", gfid=gfid, n=n, step=step, cfg_scale=args.cfg_scale)
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


def _save_grid(u8: torch.Tensor, path: Path) -> None:
    from PIL import Image
    n = u8.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    _, _, H, W = u8.shape
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = u8[k].permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
