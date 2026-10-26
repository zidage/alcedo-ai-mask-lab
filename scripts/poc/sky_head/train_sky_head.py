#!/usr/bin/env python3
"""Train the fine sky head on the frozen SAM 2.1 tiny encoder (issue #10).

Input:  --data-dir (prepared data), --train-split and --val-split, --sam2-repo (checkout at
        SAM2_COMMIT), --sam2-checkpoint (sam2.1_hiera_tiny.pt, CHECKPOINT_SHA256).
Output: --out-dir/run.json (arguments, versions, hashes), metrics.jsonl (one line per epoch),
        last.pt (each epoch), best.pt (highest val boundary_f1).

Three modes:

  default             training for --epochs; validation after each epoch (evaluate_sky_head.py)
  --pilot-steps N     N training steps without a checkpoint: time per step (after
                      PILOT_WARMUP_STEPS), photos per second, data wait share, peak CUDA memory,
                      and the estimated time of one epoch and of --epochs epochs -> pilot.json
  --overfit-images N  training on the first N training photos without augmentation for
                      --overfit-steps steps, then the measurements on the same photos ->
                      overfit.json (the head must learn them: iou near 1)

Precision: the encoder runs under torch.no_grad in bfloat16 autocast during training (faster; the
small difference to float32 is like a photometric augmentation); validation and evaluation run
the encoder in float32 with TF32 off. The head trains in bfloat16 autocast with float32 weights.
A stopped run continues with --resume from last.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_head_data as sd  # noqa: E402

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
PILOT_WARMUP_STEPS = 10
LOG_EVERY_STEPS = 50


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources(sam2_repo: str, checkpoint: Path) -> dict:
    commit = subprocess.run(["git", "-C", sam2_repo, "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != SAM2_COMMIT:
        raise ValueError(f"{sam2_repo} is at {commit}, expected {SAM2_COMMIT}")
    digest = file_sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"{checkpoint}: SHA-256 {digest}, expected {CHECKPOINT_SHA256}")
    return {"sam2_commit": commit, "sam2_checkpoint": checkpoint.name,
            "sam2_checkpoint_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-split", nargs="+", default=["train"])
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--widths", type=int, nargs=5, default=[128, 96, 64, 32, 16])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--val-limit", type=int, default=0, help="first N val photos (0 = all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--overfit-images", type=int, default=0)
    parser.add_argument("--overfit-steps", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import sky_head_losses as losses
    import sky_head_model as model
    from evaluate_sky_head import evaluate

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: select a GPU runtime")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = check_sources(args.sam2_repo, args.sam2_checkpoint)

    train_items = [item for split in args.train_split
                   for item in sd.split_items(args.data_dir, split)]
    val_items = sd.split_items(args.data_dir, args.val_split)
    if args.val_limit:
        val_items = val_items[:args.val_limit]
    overfit = args.overfit_images > 0
    if args.pilot_steps and args.pilot_steps <= PILOT_WARMUP_STEPS:
        raise ValueError(f"--pilot-steps must be more than the {PILOT_WARMUP_STEPS} warm-up "
                         f"steps that are not timed")
    if overfit:
        train_items = train_items[:args.overfit_images]
        val_items = train_items

    encoder = model.build_encoder(args.sam2_repo, str(args.sam2_checkpoint)).to(device)
    head = model.SkyAlphaHead(tuple(args.widths)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    dataset = sd.SkyDataset(train_items, train=not overfit, seed=args.seed)
    loader_args = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.workers, pin_memory=device.type == "cuda")
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    total_steps = (args.overfit_steps if overfit else
                   args.pilot_steps if args.pilot_steps else steps_per_epoch * args.epochs)

    def learning_rate(step: int) -> float:
        warmup = min(args.warmup_steps, max(1, total_steps // 10))
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    run = {"step": "issue #10: fine sky head", "args": {k: str(v) for k, v in vars(args).items()},
           "sources": sources, "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "head_parameters": model.parameter_count(head),
           "train_photos": len(train_items), "val_photos": len(val_items),
           "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}
    print(json.dumps(run, indent=1))
    mode_name = "overfit" if overfit else "pilot" if args.pilot_steps else "train"
    if mode_name == "train":
        sd.write_json(args.out_dir / "run.json", run)

    start_epoch, step, best = 0, 0, -1.0
    last_path = args.out_dir / "last.pt"
    if args.resume and mode_name == "train":
        if not last_path.exists():
            raise FileNotFoundError(f"--resume: {last_path} does not exist")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        print(f"[resume] epoch {start_epoch}, step {step}, best boundary_f1 {best:.4f}")

    def save(path: Path, epoch: int) -> None:
        torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "step": step, "best": best,
                    "widths": list(args.widths), "sources": sources}, path)

    def train_step(pixels, target, weight) -> dict:
        nonlocal step
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        pixels = pixels.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16,
                                  enabled=device.type == "cuda")
        with torch.no_grad(), autocast:
            embed, high_0, high_1 = encoder(pixels)
        with autocast:
            outputs = head(embed.float(), high_0.float(), high_1.float(), pixels)
        terms = losses.sky_head_loss(outputs, target, weight)
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        step += 1
        return {k: float(v) for k, v in terms.items()}

    if len(dataset) < args.batch_size:
        raise ValueError(f"{len(dataset)} training photos is less than --batch-size "
                         f"{args.batch_size}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epoch = start_epoch
    wait, compute, timed_steps, timed_seconds = 0.0, 0.0, 0, 0.0
    while step < total_steps:
        dataset.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
        sums: dict = {}
        count = 0
        mark = time.perf_counter()
        for pixels, target, weight in loader:
            loaded = time.perf_counter()
            terms = train_step(pixels, target, weight)
            if device.type == "cuda":
                torch.cuda.synchronize()
            done = time.perf_counter()
            if step > PILOT_WARMUP_STEPS:
                wait += loaded - mark
                compute += done - loaded
                timed_steps += 1
                timed_seconds += done - mark
            mark = done
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
            if step % LOG_EVERY_STEPS == 0:
                rate = (timed_steps * args.batch_size / timed_seconds) if timed_seconds else 0
                print(f"[{mode_name}] epoch {epoch} step {step}/{total_steps} "
                      f"loss {terms['total']:.4f} lr {learning_rate(step):.2e} "
                      f"{rate:.1f} photos/s", flush=True)
            if step >= total_steps:
                break
        train_terms = {k: v / max(1, count) for k, v in sums.items()}
        if mode_name != "train" and step < total_steps:
            epoch += 1
            continue

        if mode_name == "pilot":
            seconds_per_step = timed_seconds / max(1, timed_steps)
            pilot = {**run, "timed_steps": timed_steps,
                     "seconds_per_step": round(seconds_per_step, 4),
                     "photos_per_second": round(args.batch_size / seconds_per_step, 2),
                     "data_wait_share": round(wait / max(1e-9, wait + compute), 3),
                     "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                                if device.type == "cuda" else None),
                     "estimated_epoch_hours": round(steps_per_epoch * seconds_per_step / 3600, 2),
                     "estimated_run_hours": round(steps_per_epoch * args.epochs
                                                  * seconds_per_step / 3600, 2),
                     "train_terms": train_terms}
            sd.write_json(args.out_dir / "pilot.json", pilot)
            print(json.dumps({k: pilot[k] for k in (
                "seconds_per_step", "photos_per_second", "data_wait_share",
                "peak_cuda_memory_bytes", "estimated_epoch_hours", "estimated_run_hours")},
                indent=1))
            return 0

        measured = evaluate(encoder, head, val_items, device, batch_size=args.batch_size,
                            workers=args.workers, keep_per_photo=overfit)
        if mode_name == "overfit":
            sd.write_json(args.out_dir / "overfit.json",
                          {**run, "train_terms": train_terms, **measured})
            print(json.dumps(measured["summary"], indent=1))
            return 0

        summary = measured["summary"]
        line = {"epoch": epoch, "step": step, "train": train_terms, "val": summary,
                "val_seconds": measured["seconds"],
                "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                           if device.type == "cuda" else None)}
        with open(args.out_dir / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(line) + "\n")
        score = summary["boundary_f1"] if summary["boundary_f1"] is not None else -1.0
        if score > best:
            best = score
            save(args.out_dir / "best.pt", epoch)
        save(last_path, epoch)
        print(f"[epoch {epoch}] loss {train_terms['total']:.4f} val iou {summary['iou']:.4f} "
              f"boundary_f1 {summary['boundary_f1']:.4f} edge_sad {summary['edge_sad']:.3f} "
              f"(best {best:.4f})", flush=True)
        epoch += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
