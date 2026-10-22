#!/usr/bin/env python3
"""Float32 difference between runs of the same S3OD model on the CPU (issue #6).

Input:  --source, --checkpoint, --graph (the float32 ONNX graph), --photo (one or more).
Output: --out (JSON): per photo the maximum absolute difference of the three sigmoid masks and of
        the IoU scores between these runs, all on the same preprocessed pixels:

  torch_4_threads         PyTorch, 4 CPU threads (the reference of compare_onnx_outputs.py)
  torch_1_thread          PyTorch, 1 CPU thread
  torch_eager_attention   PyTorch, 4 threads, DINOv3 attention `eager` in place of `sdpa`
  onnx_all                ONNX Runtime CPU, all graph optimizations (compare_onnx_outputs.py)
  onnx_disabled           ONNX Runtime CPU, graph optimizations off

The differences between the PyTorch runs show the float32 rounding of the reference itself (other
summation order, other attention kernel). compare_onnx_outputs.py uses a tolerance of 1e-3 for
onnx_all against torch_4_threads; this script shows whether a difference near that tolerance is
in the range of the rounding differences of PyTorch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import s3od_graphs as graphs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photo", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    torch.set_grad_enabled(False)
    model = graphs.load_model(args.source, Path(args.checkpoint))
    wrapper = graphs.subject_masks_module(model)

    def onnx_session(level):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        return ort.InferenceSession(args.graph, options, providers=["CPUExecutionProvider"])

    results = []
    for photo in args.photo:
        pixels, _ = graphs.letterbox(graphs.load_rgb(Path(photo)))
        tensor = torch.from_numpy(pixels)
        runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        model.encoder.config._attn_implementation = "sdpa"
        torch.set_num_threads(4)
        runs["torch_4_threads"] = tuple(t.numpy() for t in wrapper(tensor))
        torch.set_num_threads(1)
        runs["torch_1_thread"] = tuple(t.numpy() for t in wrapper(tensor))
        torch.set_num_threads(4)
        model.encoder.config._attn_implementation = "eager"
        runs["torch_eager_attention"] = tuple(t.numpy() for t in wrapper(tensor))
        model.encoder.config._attn_implementation = "sdpa"
        for label, level in (("onnx_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                             ("onnx_disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
            runs[label] = tuple(onnx_session(level).run(None, {"pixel_values": pixels}))
        pairs = {}
        for a, b in itertools.combinations(runs, 2):
            pairs[f"{a} vs {b}"] = {
                "max_abs_masks_diff": float(np.abs(runs[a][0] - runs[b][0]).max()),
                "max_abs_iou_diff": float(np.abs(runs[a][1] - runs[b][1]).max())}
        results.append({"image": Path(photo).name, "pairs": pairs})
        for name, values in pairs.items():
            print(f"[spread] {Path(photo).stem} {name}: {values['max_abs_masks_diff']:.2e} "
                  f"{values['max_abs_iou_diff']:.2e}", flush=True)
    report = {"onnxruntime": ort.__version__, "torch": torch.__version__,
              "graph": Path(args.graph).as_posix(), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
