#!/usr/bin/env python3
"""Measure the MVP Sky selection on a prepared split (issue #10): the baseline of the sky head.

Input:  --data-dir and --split (prepared data, sky_head_data.py), the MVP graphs
        (mask_pipeline.GraphPaths.default: SAM 2.1 tiny E4, the P1 run 2 head), --provider.
Output: --out (JSON): sky_head_metrics.summary of the split and the values per photo.

Per photo: MaskPipeline.open_photo and select_sky (P1 head presence and sky regions, 3 points
per region, one decoder run per region). The Sky mask of the MVP is the union of the regions
(the 256 x 256 logits, bilinear, > 0). Here the union is made at 1,024 x 1,024 on the stretched
model input, the grid of the sky head target, so the measurements of the sky head and of the MVP
use the same target and the same code. The MVP mask is binary: alpha = 1 inside, 0 outside.
When select_sky finds no sky, the alpha is 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ai_mask_mvp"))
import sky_head_data as sd  # noqa: E402
import sky_head_metrics as sm  # noqa: E402


def mvp_alpha(pipeline, path: Path) -> tuple[np.ndarray, dict]:
    state = pipeline.open_photo(path)
    sky = pipeline.select_sky(state)
    alpha = sky.mask(sd.MODEL_SIZE, sd.MODEL_SIZE).astype(np.float32)
    return alpha, {"present": sky.present, "found": sky.found,
                   "presence": round(sky.presence, 4), "regions": len(sky.regions)}


def main() -> int:
    from mask_pipeline import (GraphPaths, MaskPipeline, add_provider_arguments,
                               pipeline_provider_options)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    add_provider_arguments(parser, default="directml")
    args = parser.parse_args()

    items = sd.split_items(args.data_dir, args.split)
    if args.limit:
        items = items[:args.limit]
    pipeline = MaskPipeline(GraphPaths.default(), args.provider, args.threads,
                            **pipeline_provider_options(args))
    measurements = []
    started = time.perf_counter()
    for index, item in enumerate(items):
        _, alpha, weight = sd.load_sample(item)
        target, use = sd.resize_target(alpha), sd.resize_weight(weight)
        output, selection = mvp_alpha(pipeline, Path(item["split_dir"]) / "images"
                                      / f"{item['stem']}.jpg")
        pipeline.events.clear()
        entry = sm.photo_measurements(output, target, use)
        entry.update(stem=item["stem"], **selection)
        measurements.append(entry)
        if (index + 1) % 100 == 0 or index + 1 == len(items):
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"[{args.split}] {index + 1} / {len(items)}, {rate:.2f} photos/s", flush=True)
    result = {"step": "issue #10: MVP Sky baseline", "split": args.split,
              "provider": args.provider, "graphs": pipeline.graph_records,
              "thresholds": pipeline.thresholds, "head_checkpoint_epoch": pipeline.head_epoch,
              "summary": sm.summary(measurements),
              "found_photos": sum(m["found"] for m in measurements),
              "seconds": round(time.perf_counter() - started, 1), "photos": measurements}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sd.write_json(args.out, result)
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
