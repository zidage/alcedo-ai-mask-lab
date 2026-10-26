#!/usr/bin/env python3
"""Measure a fine sky head checkpoint on a prepared split (issue #10).

Input:  --checkpoint (train_sky_head.py), --data-dir and --split (prepared data), --sam2-repo,
        --sam2-checkpoint.
Output: --out (JSON): sky_head_metrics.summary of the split and the values per photo.

The encoder runs in float32 with TF32 off (the precision of point_encoder.onnx on Windows); the
head runs in float32. The measurements are at 1024 x 1024 (sky_head_metrics.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_head_data as sd  # noqa: E402
import sky_head_metrics as sm  # noqa: E402


def evaluate(encoder, head, items: list[dict], device, batch_size: int = 4,
             workers: int = 2, keep_per_photo: bool = True) -> dict:
    import torch

    loader = torch.utils.data.DataLoader(sd.SkyDataset(items, train=False),
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=device.type == "cuda")
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    was_training = head.training
    head.eval()
    measurements = []
    started = time.perf_counter()
    try:
        with torch.no_grad():
            index = 0
            for pixels, target, weight in loader:
                pixels = pixels.to(device, non_blocking=True)
                embed, high_0, high_1 = encoder(pixels)
                logits, _, _ = head(embed, high_0, high_1, pixels)
                alpha = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
                for a, t, w in zip(alpha, target.numpy()[:, 0], weight.numpy()[:, 0]):
                    entry = sm.photo_measurements(a, t, w)
                    entry["stem"] = items[index]["stem"]
                    measurements.append(entry)
                    index += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        head.train(was_training)
    result = {"summary": sm.summary(measurements),
              "seconds": round(time.perf_counter() - started, 1)}
    if keep_per_photo:
        result["photos"] = measurements
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import sky_head_model as model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: select a GPU runtime")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = model.SkyAlphaHead(tuple(state["widths"]))
    head.load_state_dict(state["head"])
    head.to(device)
    encoder = model.build_encoder(args.sam2_repo, args.sam2_checkpoint).to(device)
    items = sd.split_items(args.data_dir, args.split)
    if args.limit:
        items = items[:args.limit]
    result = evaluate(encoder, head, items, device, args.batch_size, args.workers)
    result.update(checkpoint=str(args.checkpoint), epoch=state.get("epoch"), split=args.split,
                  device=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"))
    sd.write_json(args.out, result)
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
