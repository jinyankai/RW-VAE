"""05b -- fine-tune the decoder of ONE arm on its channel outputs (plan v2.1 Sec. 4 "B0-adapt", Sec. 6 M4).

post_quant_conv + decoder of the frozen tokenizer are copied into an AdaptedDecoder initialised at the analytic
fold (cfg.decoder.mode, normally the P6 LMMSE fold; identity for B0, which gives the B0-adapt control) and trained
with rwvae.decoder_adapter.finetune_decoder: loss = l1_weight * L1 + lpips_weight * LPIPS(VGG), Adam(W), the
recipe and budget of cfg.decoder_adapt -- identical for every arm and recorded in the weights' meta. Step t sees
z' = channel(mu, eps) with eps seeded by noise_seed(decoder_adapt.noise_seed, t, row). The encoder is never used:
mu comes from the latent cache.

Data     ImageLatentPairDataset over the first decoder_adapt.images_per_class rows of every class of
         decoder_adapt.split (deterministic; classes with fewer images give all of theirs, recorded), DataLoader
         with shuffle + drop_last and a generator seeded with cfg.seed (data order is reproducible).
Inputs   manifest (01), latent cache of decoder_adapt.split (02; also train_dev for --val-images), stats.npz (03),
         the arm's allocation (04; built on the fly for B0), the tokenizer checkpoint.
Outputs  (default <out> = <run_root>/<name>/decoders/<arm label>)
         decoder.safetensors  fine-tuned weights; metadata meta = arm, allocation fingerprint, tokenizer sha256,
                              manifest fingerprint, recipe, subset (size, rows sha256), steps, seeds, git commit
         adapt.json           the same meta plus the training result and the optional before/after check
         config.yaml, log.txt, metrics.jsonl (loss / l1 / lpips), summary.json  (rwvae.runtime.Run)
--val-images N: PSNR / LPIPS of the analytic adapter before and the fine-tuned one after, on a class-balanced
         prefix of N train_dev rows with the 05 eval noise protocol (noise_seed(eval.noise_seed, 0, row)).
Evaluate the result with 05_eval_recon_probe.py --decoder-ckpt <arm label>=<out>/decoder.safetensors.

Examples (common flags of rwvae.runtime.add_common_args apply)
  ImageNet-1k (GPU node):
    python scripts/05b_adapt_decoder.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --arm B3 --R 22 --w 2 --val-images 500
    python scripts/05b_adapt_decoder.py ... --arm B0          # the B0-adapt control (same recipe and budget)
  SVP:
    python scripts/05b_adapt_decoder.py ... --config configs/server_a800.yaml --config configs/svp.yaml --arm B2 --R 22 --val-images 100
  Local Imagenette (engineering test only):
    python scripts/05b_adapt_decoder.py --config configs/dataset_imagenet256.yaml \
        --config configs/baseline_common.yaml --config configs/arms.yaml --config configs/local_imagenette.yaml \
        --arm B3 --R 22 --w 2 --val-images 32
"""
from __future__ import annotations

import _common  # noqa: I001  (first: puts the repo on sys.path, sets KMP_DUPLICATE_LIB_OK)

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rwvae import datasets, evaluation, integration, runtime
from rwvae.channel import RWChannel, noise_seed, sample_noise
from rwvae.decoder_adapter import AdaptedDecoder, finetune_decoder


def build_parser():
    p = _common.make_parser("Fine-tune one arm's decoder on its channel outputs (plan M4).")
    p.add_argument("--arm", required=True, help="arm label from configs/arms.yaml (B0 = B0-adapt control)")
    p.add_argument("--R", type=float, default=None, help="nominal rate in nats/token (lossy arms)")
    p.add_argument("--w", type=float, default=None, help="relevance weight (rule r2 only)")
    p.add_argument("--val-images", type=int, default=0, help="PSNR/LPIPS before vs after on N train_dev images")
    return p


def arm_point(arm: str, rule: str, R, w) -> tuple[float | None, float | None]:
    if rule == "none":
        if R is not None or w is not None:
            raise ValueError(f"arm {arm} (rule none) is the rate-free reference: pass neither --R nor --w")
        return None, None
    if R is None or (rule == "r2" and w is None):
        raise ValueError(f"arm {arm} (rule {rule}) needs --R" + (" and --w" if rule == "r2" else ""))
    if rule != "r2" and w is not None:
        raise ValueError(f"--w applies to rule r2 only; arm {arm} has rule {rule}")
    return float(R), None if w is None else float(w)


def per_class_rows(labels: list[int], per_class: int) -> tuple[list[int], dict]:
    """The first `per_class` rows of every class (row order); classes with fewer images give all of theirs."""
    taken: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        if len(taken.setdefault(y, [])) < per_class:
            taken[y].append(i)
    short = {y: len(r) for y, r in taken.items() if len(r) < per_class}
    return sorted(i for r in taken.values() for i in r), {"per_class_requested": per_class, "classes_short": short}


def balanced_prefix(labels: list[int], n: int) -> list[int]:
    """First n rows in (rank within class, row) order (as 05_eval_recon_probe.py)."""
    seen: dict[int, int] = {}
    rank = []
    for y in labels:
        rank.append(seen.get(y, 0))
        seen[y] = rank[-1] + 1
    return sorted(sorted(range(len(labels)), key=lambda i: (rank[i], i))[:n])


@torch.no_grad()
def check(adapter, channel, lpips_model, dataset, base_seed: int, batch_size: int, device) -> dict:
    """Mean PSNR / LPIPS of `adapter` on `dataset` with the 05 eval noise protocol."""
    was = adapter.training
    adapter.eval()
    psnr, meter = [], evaluation.LPIPSMeter(lpips_model)
    for images, mu, _, index in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images, mu = images.to(device), mu.to(device)
        eps = sample_noise(mu.shape[1:], [noise_seed(base_seed, 0, int(i)) for i in index], device=device)
        x_hat = adapter(channel(mu, noise=eps))
        psnr.append(evaluation.psnr_per_image(evaluation.to_uint8(images), evaluation.to_uint8(x_hat)).cpu().numpy())
        meter.update(x_hat, images)
    adapter.train(was)
    p = np.concatenate(psnr)
    return {"psnr": float(p[np.isfinite(p)].mean()), "n_psnr_inf": int((~np.isfinite(p)).sum()),
            "lpips": meter.mean(), "n": int(p.size)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _common.load_cfg(args)
    rule = _common.arm_rule(cfg, args.arm)
    R, w = arm_point(args.arm, rule, args.R, args.w)
    label = _common.arm_label(args.arm, rule, R, w)
    out_dir = Path(args.out) if args.out else _common.stage_dir(cfg, "decoders", label)
    run = _common.open_run(__file__, cfg, out_dir, args)
    try:
        log = run.logger
        seed = int(cfg.get("seed", 0))
        runtime.seed_everything(seed)
        device = runtime.select_device(args.device)
        tf32 = _common.set_tf32(False)
        d = cfg["decoder_adapt"]

        manifest = _common.load_manifest(cfg)
        stats = _common.load_stats(cfg)
        alloc = _common.load_allocation(cfg, args.arm, R, w)
        solved_from = alloc.meta.get("stats_fingerprint")
        if solved_from is not None and solved_from != stats.fingerprint():
            raise ValueError(f"allocation {alloc.name} was solved from stats {solved_from}, but "
                             f"{_common.stats_path(cfg)} has {stats.fingerprint()}: re-run scripts/04_solve_channel.py")
        cache = _common.open_cache(cfg, d["split"], manifest)
        rows, short = per_class_rows(manifest.splits[d["split"]]["label"], int(d["images_per_class"]))
        if short["classes_short"]:
            log.warning("%d classes of %s have fewer than %d images: all of theirs are used",
                        len(short["classes_short"]), d["split"], d["images_per_class"])
        batch = int(d["batch_size"])
        if len(rows) < batch:
            raise ValueError(f"{len(rows)} training images < decoder_adapt.batch_size={batch} (drop_last=True)")
        S = int(cfg["dataset"]["image_size"])
        dataset = datasets.ImageLatentPairDataset(manifest, cache, image_size=S, indices=rows)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True,
                                             num_workers=int(d["num_workers"]),
                                             generator=torch.Generator().manual_seed(seed))

        tokenizer = _common.make_tokenizer(cfg, device)
        mode = "identity" if rule == "none" else cfg["decoder"]["mode"]
        kw = {"A": alloc.A, "m": alloc.m} if mode == "lmmse" else {}
        adapter = AdaptedDecoder(tokenizer.post_quant_conv, tokenizer.decoder, mode=mode, **kw).to(device)
        channel = RWChannel.from_allocation(alloc).to(device)
        lpips_model = integration.load_lpips(device)

        val_set, before = None, None
        if args.val_images > 0:
            vcache = _common.open_cache(cfg, "train_dev", manifest)
            vrows = balanced_prefix(manifest.splits["train_dev"]["label"], int(args.val_images))
            val_set = datasets.ImageLatentPairDataset(manifest, vcache, image_size=S, indices=vrows)
            before = check(adapter, channel, lpips_model, val_set, int(cfg["eval"]["noise_seed"]), batch, device)
            log.info("before fine-tuning (%s, %d train_dev images): psnr %.3f, lpips %.4f", mode, before["n"],
                     before["psnr"], before["lpips"])

        recipe = {k: d[k] for k in ("steps", "batch_size", "lr", "betas", "l1_weight", "lpips_weight", "grad_clip",
                                    "mixed_precision", "noise_seed")}
        log.info("fine-tuning %s from %s on %d images of %s: %s", label, mode, len(rows), d["split"], recipe)
        result = finetune_decoder(
            adapter, loader, channel, lpips_fn=lpips_model, steps=int(d["steps"]), lr=float(d["lr"]),
            l1_weight=float(d["l1_weight"]), lpips_weight=float(d["lpips_weight"]), betas=tuple(d["betas"]),
            grad_clip=None if d["grad_clip"] is None else float(d["grad_clip"]), mixed_precision=d["mixed_precision"],
            device=device, noise_base_seed=int(d["noise_seed"]), log_every=int(d["log_every"]),
            log_fn=lambda step, m: run.log_metrics(step, **m))
        after = None
        if val_set is not None:
            after = check(adapter, channel, lpips_model, val_set, int(cfg["eval"]["noise_seed"]), batch, device)
            log.info("after fine-tuning: psnr %.3f, lpips %.4f", after["psnr"], after["lpips"])

        meta = runtime.to_jsonable({
            "arm": args.arm, "rule": rule, "R": R, "w": w, "arm_label": label, "allocation": alloc.name,
            "allocation_fingerprint": alloc.fingerprint(), "stats_fingerprint": stats.fingerprint(),
            "base_mode": mode, "tokenizer_sha256": tokenizer.meta.get("checkpoint_sha256"),
            "manifest_fingerprint": manifest.fingerprint(),
            "recipe": {**recipe, "weight_decay": 0.0, "optimizer": "AdamW", "loss": "l1_weight*L1 + lpips_weight*LPIPS(vgg)",
                       "trained": "post_quant_conv + decoder", "mixed_precision_ran": result["mixed_precision"]},
            "subset": {"split": d["split"], "n": len(rows), **short,
                       "rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                       "cache_index_sha256": runtime.sha256_file(cache.cache_dir / datasets.CACHE_INDEX)},
            "steps": result["steps"],
            "seeds": {"seed": seed, "data_order": f"DataLoader(shuffle, drop_last, generator seed {seed})",
                      "noise": f"noise_seed({d['noise_seed']}, step, row)"},
            "tf32": tf32, "image_size": S, "git_commit": runtime.git_commit(),
        })
        adapter.save(out_dir / "decoder.safetensors", meta)
        run.save_json("adapt.json", {"meta": meta, "result": result, "before": before, "after": after})
        run.finish("ok", arm=label, final_loss=result["final_loss"], before=before, after=after,
                   decoder=out_dir / "decoder.safetensors")
    except BaseException as err:
        run.finish("failed", error=repr(err))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
