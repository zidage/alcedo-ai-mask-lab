#!/usr/bin/env python3
"""Compare the universal matting ONNX graphs with the TensorFlow graph of the source on the CPU
(issue #6).

Check name: OnnxAlphaMatchesTensorflowWithinTolerance.

Input:  --graph (`tf_graph.pb`), one or more --graph-onnx LABEL=PATH, the photos of --prompts in
        --photos (the quality set at a long edge of at most 4096 px).
Output: --out (JSON report) and, next to it, per photo:
        reference_alpha/<stem>.npy  the TensorFlow alpha (channel 3 of `cond_3/Merge:0` / 255)
                                    at the photo size, float16, for probe_subject_masks.py (kind
                                    universal_matting);
        <label>/alpha/<stem>.npy    the same for the ONNX Runtime CPU output of each graph.

Reference: `tf_graph.pb` in a TensorFlow 1 session on the CPU with the default graph
optimizations (the ModelScope pipeline creates a default `tf.Session`), --threads intra-op threads,
fed with the photo of `LoadImage` (universal_matting_graphs.load_rgb). Per photo and graph:

- `max_abs_alpha_diff`, `mean_abs_alpha_diff`, `p999_abs_alpha_diff`: the alpha in [0, 1] at the
  photo size against the reference;
- `iou`: IoU of alpha > 0.5 against the reference;
- `max_abs_alpha_diff_grappler_off`: against the same TensorFlow graph with the grappler meta
  optimizer off.

`reference_spread` records the float32 rounding of the reference itself: the maximum absolute alpha
difference between the TensorFlow session with and without grappler. The branch tensors of the
reference (`person_rows_columns`, `human_alpha_share`, `human_result`, `person_box`) show the
branch of each photo; `output_png_is_truncated_rgba` checks that `output_png` is the uint8
truncation of the compared tensor.

Requirement for a graph with `--required LABEL` (the float32 graph): `max_abs_alpha_diff` <= 1e-3
on every photo (issue #6, acceptance criteria). The other graphs (float16) are reported without a
requirement. TensorFlow and ONNX Runtime use --threads CPU threads (default 4); the ONNX Runtime CPU
memory arena is off (AGENTS.md resource limits). The runs are sequential.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import universal_matting_graphs as graphs  # noqa: E402

TOLERANCE = 1e-3


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def parse_graph(text: str) -> tuple[str, Path]:
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", required=True, help="tf_graph.pb")
    parser.add_argument("--graph-onnx", action="append", required=True, type=parse_graph,
                        help="LABEL=PATH of an ONNX graph")
    parser.add_argument("--required", action="append", default=[],
                        help="label of a graph that must meet the tolerance")
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True, help="the quality set; only the photo list "
                                                         "is used")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import onnxruntime as ort
    import tensorflow as tf

    graph_def = graphs.read_graph_def(Path(args.graph))
    reference = graphs.tf_session(graph_def, args.threads)
    reference_off = graphs.tf_session(graph_def, args.threads, graph_optimizations=False)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    sessions = {label: ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
                for label, path in args.graph_onnx}

    out_path = Path(args.out)
    (out_path.parent / "reference_alpha").mkdir(parents=True, exist_ok=True)
    for label in sessions:
        (out_path.parent / label / "alpha").mkdir(parents=True, exist_ok=True)
    branch_names = list(graphs.TF_BRANCH_TENSORS)
    fetches = [graphs.TF_OUTPUT_RGBA, graphs.TF_OUTPUT_PNG,
               *(graphs.TF_BRANCH_TENSORS[name] for name in branch_names)]
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        image = graphs.load_rgb(Path(args.photos) / entry["file"])
        feed = graphs.model_input(image)
        started = time.perf_counter()
        rgba, png, *branch = reference.run(fetches, {graphs.TF_INPUT: feed})
        tf_seconds = time.perf_counter() - started
        rgba_off = reference_off.run(graphs.TF_OUTPUT_RGBA, {graphs.TF_INPUT: feed})
        alpha = graphs.alpha_from_rgba(rgba)
        alpha_off = graphs.alpha_from_rgba(rgba_off)
        np.save(out_path.parent / "reference_alpha" / f"{stem}.npy", alpha.astype(np.float16))
        values = {name: np.asarray(value).tolist() for name, value in zip(branch_names, branch)}
        result = {"image": entry["file"], "photo_size": [image.shape[1], image.shape[0]],
                  "branch": "human" if values["human_result"] else "object",
                  "branch_tensors": values,
                  "output_png_is_truncated_rgba": bool(np.array_equal(rgba.astype(np.uint8), png)),
                  "reference_spread": float(np.abs(alpha_off - alpha).max()),
                  "reference_foreground": float((alpha > 0.5).mean()),
                  "tensorflow_cpu_seconds": round(tf_seconds, 2), "graphs": {}}
        for label, session in sessions.items():
            started = time.perf_counter()
            (output,) = session.run(None, {graphs.ONNX_INPUT: feed})
            onnx_seconds = time.perf_counter() - started
            onnx_alpha = graphs.alpha_from_rgba(output)
            np.save(out_path.parent / label / "alpha" / f"{stem}.npy",
                    onnx_alpha.astype(np.float16))
            difference = np.abs(onnx_alpha - alpha)
            record = {
                "max_abs_alpha_diff": float(difference.max()),
                "mean_abs_alpha_diff": float(difference.mean()),
                "p999_abs_alpha_diff": float(np.quantile(difference, 0.999)),
                "iou": iou(onnx_alpha > 0.5, alpha > 0.5),
                "max_abs_alpha_diff_grappler_off": float(np.abs(onnx_alpha - alpha_off).max()),
                "finite": bool(np.isfinite(output).all()),
                "onnxruntime_cpu_seconds": round(onnx_seconds, 2),
            }
            if label in args.required:
                record["passed"] = record["max_abs_alpha_diff"] <= TOLERANCE
            result["graphs"][label] = record
            print(f"[compare] {label} {stem} ({result['branch']}): max|dalpha| "
                  f"{record['max_abs_alpha_diff']:.2e} (grappler off "
                  f"{record['max_abs_alpha_diff_grappler_off']:.2e}) IoU {record['iou']:.6f}; "
                  f"reference spread {result['reference_spread']:.2e}", flush=True)
        results.append(result)

    summary = {}
    for label, _ in args.graph_onnx:
        records = [r["graphs"][label] for r in results]
        summary[label] = {
            "required": label in args.required,
            "max_abs_alpha_diff": max(r["max_abs_alpha_diff"] for r in records),
            "max_abs_alpha_diff_grappler_off": max(r["max_abs_alpha_diff_grappler_off"]
                                                   for r in records),
            "minimum_iou": min(r["iou"] for r in records),
            "photos_above_tolerance": [r["image"] for r in results
                                       if r["graphs"][label]["max_abs_alpha_diff"] > TOLERANCE],
            "all_finite": all(r["finite"] for r in records),
        }
        if label in args.required:
            summary[label]["passed"] = all(r["passed"] for r in records)
    passed = all(s.get("passed", True) for s in summary.values())
    report = {"check": "OnnxAlphaMatchesTensorflowWithinTolerance", "tolerance": TOLERANCE,
              "onnxruntime": ort.__version__, "tensorflow": tf.__version__,
              "threads": args.threads, "graph": Path(args.graph).as_posix(),
              "graphs": {label: path.as_posix() for label, path in args.graph_onnx},
              "reference_spread": max(r["reference_spread"] for r in results),
              "output_png_is_truncated_rgba": all(r["output_png_is_truncated_rgba"]
                                                  for r in results),
              "passed": passed, "summary": summary, "results": results}
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for label, values in summary.items():
        print(f"[compare] {label}: {values}")
    print(f"[compare] {'PASS' if passed else 'FAIL'}; report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
