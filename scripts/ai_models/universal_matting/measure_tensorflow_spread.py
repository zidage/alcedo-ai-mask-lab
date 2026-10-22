#!/usr/bin/env python3
"""Spread of the TensorFlow CPU reference of the universal matting graph over TensorFlow settings
(issue #6).

Check name: TensorflowSpreadAgainstOnnx (a record, no requirement).

Input:  --graph (`tf_graph.pb`), the photos of --prompts in --photos, --onnx-alpha (the
        <label>/alpha directory of compare_onnx_outputs.py, alpha at the photo size, float16).
Output: --out (JSON) and, in --work-dir, the alpha of each setting and photo (<setting>/<stem>.npy,
        float32).

`tf_graph.pb` runs in one new process per setting (oneDNN is selected when TensorFlow is
imported, by the environment variable `TF_ENABLE_ONEDNN_OPTS`):

  SETTINGS  name -> (oneDNN on, intra-op threads, grappler meta optimizer on)

The default setting of compare_onnx_outputs.py (the reference of
`OnnxAlphaMatchesTensorflowWithinTolerance`) is `onednn_t4_grappler`. Per photo the record has the
largest absolute alpha difference between two settings (`tensorflow_spread`, the maximum over all
pairs) and between the ONNX alpha and each setting. The ONNX alpha is stored in float16, so a
difference below 4.9e-4 (the float16 step at 1.0) is not resolved. The processes run one after the
other (AGENTS.md resource limits).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import universal_matting_graphs as graphs  # noqa: E402

SETTINGS = {
    "onednn_t4_grappler": (True, 4, True),
    "onednn_t4_no_grappler": (True, 4, False),
    "onednn_t1_grappler": (True, 1, True),
    "no_onednn_t4_grappler": (False, 4, True),
    "no_onednn_t4_no_grappler": (False, 4, False),
    "no_onednn_t1_no_grappler": (False, 1, False),
}


def child(args) -> int:
    _, threads, grappler = SETTINGS[args.setting]
    out_dir = Path(args.work_dir) / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    session = graphs.tf_session(graphs.read_graph_def(Path(args.graph)), threads,
                                graph_optimizations=grappler)
    for name in photo_names(args.prompts):
        feed = graphs.model_input(graphs.load_rgb(Path(args.photos) / name))
        rgba = session.run(graphs.TF_OUTPUT_RGBA, {graphs.TF_INPUT: feed})
        np.save(out_dir / f"{Path(name).stem}.npy", graphs.alpha_from_rgba(rgba))
        print(f"[spread] {args.setting} {name}", flush=True)
    return 0


def photo_names(prompts: str) -> list[str]:
    return [entry["file"] for entry in
            json.loads(Path(prompts).read_text(encoding="utf-8"))["images"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True, help="the quality set; only the photo list "
                                                         "is used")
    parser.add_argument("--onnx-alpha", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--setting", choices=sorted(SETTINGS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.setting:
        return child(args)

    for setting, (onednn, _, _) in SETTINGS.items():
        environment = {**os.environ, "TF_ENABLE_ONEDNN_OPTS": "1" if onednn else "0",
                       "TF_CPP_MIN_LOG_LEVEL": "2"}
        command = [sys.executable, "-I", os.path.abspath(__file__), "--graph", args.graph,
                   "--photos", args.photos, "--prompts", args.prompts, "--onnx-alpha",
                   args.onnx_alpha, "--work-dir", args.work_dir, "--out", args.out,
                   "--setting", setting]
        subprocess.run(command, env=environment, check=True)

    results = []
    for name in photo_names(args.prompts):
        stem = Path(name).stem
        alphas = {s: np.load(Path(args.work_dir) / s / f"{stem}.npy") for s in SETTINGS}
        onnx_alpha = np.load(Path(args.onnx_alpha) / f"{stem}.npy").astype(np.float32)
        pairs = {f"{a}__{b}": float(np.abs(alphas[a] - alphas[b]).max())
                 for a, b in itertools.combinations(SETTINGS, 2)}
        against = {s: float(np.abs(onnx_alpha - alpha).max()) for s, alpha in alphas.items()}
        results.append({"image": name, "tensorflow_spread": max(pairs.values()),
                        "tensorflow_pairs": pairs, "onnx_against_setting": against,
                        "onnx_nearest_setting": min(against.values())})
        print(f"[spread] {name}: TensorFlow spread {max(pairs.values()):.2e}, ONNX against the "
              f"settings {min(against.values()):.2e} to {max(against.values()):.2e}", flush=True)
    import tensorflow as tf

    report = {"check": "TensorflowSpreadAgainstOnnx", "graph": Path(args.graph).as_posix(),
              "onnx_alpha": Path(args.onnx_alpha).as_posix(), "tensorflow": tf.__version__,
              "settings": {s: {"onednn": o, "threads": t, "grappler": g}
                           for s, (o, t, g) in SETTINGS.items()},
              "reference_setting": "onednn_t4_grappler",
              "largest_tensorflow_spread": max(r["tensorflow_spread"] for r in results),
              "photos_where_onnx_is_inside_the_spread": [
                  r["image"] for r in results
                  if r["onnx_nearest_setting"] <= r["tensorflow_spread"]],
              "results": results}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[spread] report {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
