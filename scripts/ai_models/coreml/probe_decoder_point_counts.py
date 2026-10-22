#!/usr/bin/env python3
"""Call time of the native Core ML SAM 2.1 decoder when the number of prompt points changes
(issue #8).

Input:  --model (the compiled `point_decoder.mlmodelc`, a flexible point dimension), --features
        (an encoder output `<stem>.npz` of measure_graph_cost.py features), --compute-units
        (default ALL), --runs (default 20).
Output: --out (JSON): the p50, p95, and maximum call time of four sequences, after one call of
        each point count:
          same_2        --runs calls with 2 points (a box)
          same_10       --runs calls with 10 points and a mask_input
          alternate     --runs pairs: 2 points, then 10 points (the warm loop of
                        measure_pipeline_cost.py)
          counts_1_to_8 --runs passes over the point counts 1 to 8 (the strokes and the sky
                        regions of the MVP have different counts)

The inputs are those of mask_pipeline.MaskPipeline.decode. All calls are in one process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FEATURES = ("image_embed", "high_res_feats_0", "high_res_feats_1")


def spread(values: list[float]) -> dict:
    array = np.asarray(values)
    return {"count": len(values), "p50_ms": round(float(np.percentile(array, 50)), 2),
            "p95_ms": round(float(np.percentile(array, 95)), 2),
            "max_ms": round(float(array.max()), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--compute-units", default="ALL",
                        choices=("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import coremltools as ct

    features = dict(np.load(args.features))
    model = ct.models.CompiledMLModel(args.model,
                                      compute_units=getattr(ct.ComputeUnit, args.compute_units))

    def feed(points: int, with_mask: bool) -> dict:
        coords = [[96.0 * (k + 1), 512.0] for k in range(points)]
        return {**{name: features[name] for name in FEATURES},
                "point_coords": np.array([coords], np.float32),
                "point_labels": np.ones((1, points), np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), np.float32),
                "has_mask_input": np.array([1.0 if with_mask else 0.0], np.float32)}

    def timed(inputs: dict) -> float:
        started = time.perf_counter()
        model.predict(inputs)
        return (time.perf_counter() - started) * 1000

    feeds = {count: feed(count, count == 10) for count in range(1, 11)}
    for inputs in feeds.values():
        timed(inputs)
    sequences = {
        "same_2": [timed(feeds[2]) for _ in range(args.runs)],
        "same_10": [timed(feeds[10]) for _ in range(args.runs)],
        "alternate": [timed(feeds[count]) for _ in range(args.runs) for count in (2, 10)],
        "counts_1_to_8": [timed(feeds[count]) for _ in range(args.runs)
                          for count in range(1, 9)],
    }
    report = {"step": "issue #8: native decoder call time per point count sequence",
              "model": args.model, "compute_units": args.compute_units,
              "features": Path(args.features).name,
              "sequences": {name: spread(values) for name, values in sequences.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["sequences"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
