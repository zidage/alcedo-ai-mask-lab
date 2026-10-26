#!/usr/bin/env python3
"""The "sky found" rule of the fine sky head: sky share of the head alpha against a threshold.

Input:  --data-dir and --split (prepared data), --sky-head-onnx (export_sky_head_onnx.py),
        --mvp-measurements (evaluate_mvp_sky.py on the split, for the MVP "found" values), the
        MVP graphs (mask_pipeline.GraphPaths), --provider.
Output: --out (JSON): per photo the target sky share and the head sky share (the share of the
        1,024 x 1,024 values >= 0.5), and per threshold of THRESHOLDS the photos with sky that the
        rule finds and misses, and the photos without sky that the rule finds.

A photo has sky when its target (the area average of the label, stretched to 1,024 x 1,024 as in
sky_head_metrics.py) is >= 0.5 at one or more pixels. AlcedoStudio uses the rule for "No sky
found" (AlcedoStudio issue #536); the head has no presence output.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ai_mask_mvp"))
import sky_head_data as sd  # noqa: E402

HEAD_INPUTS = ["image_embed", "high_res_feats_0", "high_res_feats_1", "pixel_values"]
ALPHA_INSIDE = 0.5
THRESHOLDS = [0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]


def share(values: np.ndarray) -> float:
    return float(np.count_nonzero(values >= ALPHA_INSIDE)) / values.size


def rule_table(photos: list[dict]) -> list[dict]:
    sky = [p for p in photos if p["has_sky"]]
    no_sky = [p for p in photos if not p["has_sky"]]
    rows = []
    for threshold in THRESHOLDS:
        found = [p for p in sky if p["head_share"] >= threshold and p["head_share"] > 0]
        false = [p for p in no_sky if p["head_share"] >= threshold and p["head_share"] > 0]
        missed = [p for p in sky if p not in found]
        rows.append({
            "threshold": threshold,
            "sky_photos_found": len(found),
            "sky_photos_missed": len(missed),
            "missed_target_share_max": max((p["target_share"] for p in missed), default=0.0),
            "missed_with_target_share_at_least_1_percent": sum(
                1 for p in missed if p["target_share"] >= 0.01),
            "photos_without_sky_found": len(false),
            "correct": len(found) + len(no_sky) - len(false),
        })
    return rows


def main() -> int:
    import onnxruntime as ort
    from mask_pipeline import (PROVIDERS, GraphPaths, MaskPipeline, add_provider_arguments,
                               create_session, pipeline_provider_options)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sky-head-onnx", type=Path, required=True)
    parser.add_argument("--mvp-measurements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    add_provider_arguments(parser, default="directml")
    args = parser.parse_args()
    if args.provider not in ("directml", "cpu"):
        raise ValueError("--provider: directml or cpu")

    mvp = {p["stem"]: p for p in sd.read_json(args.mvp_measurements)["photos"]}
    pipeline = MaskPipeline(GraphPaths.default(), args.provider, args.threads,
                            **pipeline_provider_options(args))
    session = create_session(args.sky_head_onnx, args.provider, args.threads)
    started = time.perf_counter()
    photos = []
    items = sd.split_items(args.data_dir, args.split)
    for number, item in enumerate(items):
        stem = item["stem"]
        _, alpha, _ = sd.load_sample(item)
        target = sd.resize_target(alpha)
        state = pipeline.open_photo(Path(item["split_dir"]) / "images" / f"{stem}.jpg")
        pixels = pipeline.graphs.preprocess_image(state.photo.image).numpy()
        head_alpha = session.run(None, dict(zip(HEAD_INPUTS, (*state.features, pixels))))[0][0, 0]
        pipeline.events.clear()
        photos.append({"stem": stem, "has_sky": bool((target >= ALPHA_INSIDE).any()),
                       "target_share": round(share(target), 7),
                       "head_share": round(share(head_alpha), 7),
                       "mvp_found": bool(mvp[stem]["found"])})
        if number % 100 == 0:
            print(f"{number} / {len(items)}", flush=True)

    sky = [p for p in photos if p["has_sky"]]
    no_sky = [p for p in photos if not p["has_sky"]]
    record = {"step": "AlcedoStudio issue #536: sky found rule of the sky head",
              "split": args.split, "provider": args.provider,
              "provider_name": PROVIDERS[args.provider], "onnxruntime": ort.__version__,
              "sky_head_onnx": str(args.sky_head_onnx), "seconds": round(time.perf_counter()
                                                                        - started, 1),
              "alpha_inside": ALPHA_INSIDE, "sky_photos": len(sky),
              "photos_without_sky": len(no_sky),
              "mvp": {"sky_photos_found": sum(p["mvp_found"] for p in sky),
                      "photos_without_sky_found": sum(p["mvp_found"] for p in no_sky)},
              "rules": rule_table(photos), "photos": photos}
    sd.write_json(args.out, record)
    for row in record["rules"]:
        print(row)
    print(record["mvp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
