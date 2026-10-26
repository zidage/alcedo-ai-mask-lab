#!/usr/bin/env python3
"""Sheets of the MVP Sky and the fine sky head on prepared photos, and the head cost (issue #10).

Input:  --data-dir and --split (prepared data), --head-measurements and --mvp-measurements (the
        per-photo JSON of evaluate_sky_head.py and evaluate_mvp_sky.py on the split),
        --sky-head-onnx (export_sky_head_onnx.py), the MVP graphs (mask_pipeline.GraphPaths),
        --provider.
Output: --out-dir/<rank>_<stem>.jpg, one row per photo at --width px per panel:
          photo | target alpha | MVP Sky mask | sky head alpha | sky head alpha on the photo (blue)
        and --out-dir/sheets.json: the selected photos with the measurements, and the head time
        per run on --provider (the first run is not counted) and the maximum alpha difference to
        the CPU provider.

Photos: --random sky photos (seed --seed), the --worst sky photos with the lowest head IoU and a
target sky share of at least 1 %, and the --false photos without sky with the largest head false
sky share. The sheets show MaSS13K photos: they stay local (`*.jpg` is in .gitignore).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ai_mask_mvp"))
import sky_head_data as sd  # noqa: E402

HEAD_INPUTS = ["image_embed", "high_res_feats_0", "high_res_feats_1", "pixel_values"]
MIN_WORST_SKY_SHARE = 0.01
TINT = np.array([40, 120, 255], np.float32)


def select(items: dict, head: dict, args) -> list[tuple[str, str]]:
    sky = sorted(stem for stem, p in head.items() if p["iou"] is not None)
    no_sky = [stem for stem, p in head.items() if p["false_sky_share"] is not None]
    chosen = [("random", stem) for stem in random.Random(args.seed).sample(sky, args.random)]
    worst = sorted((stem for stem in sky if items[stem]["sky_share"] >= MIN_WORST_SKY_SHARE),
                   key=lambda stem: head[stem]["iou"])
    chosen += [("worst", stem) for stem in worst[:args.worst]]
    false = sorted(no_sky, key=lambda stem: -head[stem]["false_sky_share"])
    chosen += [("false", stem) for stem in false[:args.false]]
    return chosen


def panel(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    gray = cv2.resize(np.clip(values, 0, 1).astype(np.float32), size,
                      interpolation=cv2.INTER_AREA)
    return np.repeat((gray * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=2)


def main() -> int:
    import cv2
    import onnxruntime as ort
    from PIL import Image, ImageDraw
    from mask_pipeline import (PROVIDERS, GraphPaths, MaskPipeline, add_provider_arguments,
                               create_session, pipeline_provider_options)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--head-measurements", type=Path, required=True)
    parser.add_argument("--mvp-measurements", type=Path, required=True)
    parser.add_argument("--sky-head-onnx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--random", type=int, default=24)
    parser.add_argument("--worst", type=int, default=8)
    parser.add_argument("--false", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4)
    add_provider_arguments(parser, default="directml")
    args = parser.parse_args()
    if args.provider not in ("directml", "cpu"):
        raise ValueError("--provider: directml or cpu")

    items = {item["stem"]: item for item in sd.split_items(args.data_dir, args.split)}
    head = {p["stem"]: p for p in sd.read_json(args.head_measurements)["photos"]}
    mvp = {p["stem"]: p for p in sd.read_json(args.mvp_measurements)["photos"]}
    chosen = select(items, head, args)

    pipeline = MaskPipeline(GraphPaths.default(), args.provider, args.threads,
                            **pipeline_provider_options(args))
    session = create_session(args.sky_head_onnx, args.provider, args.threads)
    cpu = (create_session(args.sky_head_onnx, "cpu", args.threads)
           if args.provider != "cpu" else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    times, records = [], []
    for rank, (reason, stem) in enumerate(chosen):
        item = items[stem]
        path = Path(item["split_dir"]) / "images" / f"{stem}.jpg"
        photo, alpha, weight = sd.load_sample(item)
        state = pipeline.open_photo(path)
        mvp_mask = pipeline.select_sky(state).mask(sd.MODEL_SIZE, sd.MODEL_SIZE)
        pixels = pipeline.graphs.preprocess_image(state.photo.image).numpy()
        feeds = dict(zip(HEAD_INPUTS, (*state.features, pixels)))
        started = time.perf_counter()
        head_alpha = session.run(None, feeds)[0][0, 0]
        if rank:
            times.append((time.perf_counter() - started) * 1000)
        cpu_difference = (float(np.abs(cpu.run(None, feeds)[0][0, 0] - head_alpha).max())
                          if cpu is not None else None)
        pipeline.events.clear()

        height = round(args.width * photo.shape[0] / photo.shape[1])
        size = (args.width, height)
        shown = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        head_shown = cv2.resize(head_alpha, size, interpolation=cv2.INTER_AREA)[..., None]
        tinted = (shown * (1 - 0.6 * head_shown) + TINT * 0.6 * head_shown).astype(np.uint8)
        row = np.concatenate([shown, panel(alpha, size), panel(mvp_mask.astype(np.float32), size),
                              panel(head_alpha, size), tinted], axis=1)
        image = Image.fromarray(row)
        h, m = head[stem], mvp[stem]

        def text(p, key):
            return "-" if p[key] is None else f"{p[key]:.3f}"

        label = (f"{reason} {stem}  sky {item['sky_share']:.3f} | MVP iou {text(m, 'iou')} "
                 f"bf1 {text(m, 'boundary_f1')} | head iou {text(h, 'iou')} "
                 f"bf1 {text(h, 'boundary_f1')} false {text(h, 'false_sky_share')}")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, len(label) * 6 + 8, 14), fill=(0, 0, 0))
        draw.text((4, 2), label, fill=(255, 255, 255))
        image.save(args.out_dir / f"{rank:02d}_{reason}_{stem}.jpg", quality=90)
        records.append({"rank": rank, "reason": reason, "stem": stem,
                        "sky_share": item["sky_share"], "head": h, "mvp": m,
                        "cpu_max_abs_difference": cpu_difference})
        print(label, flush=True)

    summary = {"step": "issue #10: sky head sheets", "provider": args.provider,
               "provider_name": PROVIDERS[args.provider], "onnxruntime": ort.__version__,
               "sky_head_onnx": str(args.sky_head_onnx),
               "head_ms": {"runs": len(times), "median": round(float(np.median(times)), 1),
                           "max": round(float(np.max(times)), 1)} if times else None,
               "cpu_max_abs_difference": (max(r["cpu_max_abs_difference"] for r in records)
                                          if cpu is not None else None),
               "photos": records}
    sd.write_json(args.out_dir / "sheets.json", summary)
    print(json.dumps({k: summary[k] for k in ("head_ms", "cpu_max_abs_difference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
