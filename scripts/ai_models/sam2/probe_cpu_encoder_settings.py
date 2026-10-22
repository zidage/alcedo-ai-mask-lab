#!/usr/bin/env python3
"""SAM 2.1 tiny encoder on the ONNX Runtime CPU provider against PyTorch, per session setting
(issue #8: the macOS CPU output of onnxruntime 1.24.4).

Input:  --onnx (point_encoder.onnx), --sam2-source, --sam2-checkpoint, --photo (one photo,
        preprocessed with sam2_graphs.preprocess_image), --repeats (default 3).
Output: --out (JSON): the onnxruntime and PyTorch versions, the machine; per session setting
        (graph optimization level ORT_ENABLE_ALL, ORT_ENABLE_EXTENDED, ORT_ENABLE_BASIC,
        ORT_DISABLE_ALL; intra-op threads 4 and 1) and per repeat call: for each output the
        relative error ||onnx - pytorch|| / ||pytorch|| and the largest absolute difference, and
        whether the repeat calls of one session give equal outputs.

The CPU memory arena is off in every session (AGENTS.md). PyTorch runs once (4 threads, float32).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUTPUTS = ("image_embed", "high_res_feats_0", "high_res_feats_1")
LEVELS = ("ORT_ENABLE_ALL", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_BASIC", "ORT_DISABLE_ALL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--sam2-source", default="build/tmp/explore/src/sam2")
    parser.add_argument("--sam2-checkpoint",
                        default="build/tmp/explore/weights/sam2.1_hiera_tiny.pt")
    parser.add_argument("--photo", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch
    from PIL import Image

    import sam2_graphs as graphs

    torch.set_num_threads(4)
    graphs.add_repository_to_path(args.sam2_source)
    encoder = graphs.PointEncoder(graphs.build_model("tiny", args.sam2_checkpoint)).eval()
    with Image.open(args.photo) as image:
        pixels = graphs.preprocess_image(image.convert("RGB")).numpy()
    with torch.no_grad():
        reference = [t.numpy() for t in encoder(torch.from_numpy(pixels))]

    settings = []
    for threads in (4, 1):
        for level in LEVELS:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
            session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])
            calls = [session.run(None, {"pixel_values": pixels}) for _ in range(args.repeats)]
            record = {"threads": threads, "graph_optimization_level": level, "calls": [],
                      "repeat_calls_equal": all(all(np.array_equal(a, b) for a, b in
                                                    zip(calls[0], call)) for call in calls[1:])}
            for call in calls:
                record["calls"].append({
                    name: {"relative_error": float(np.linalg.norm(value - expected)
                                                   / np.linalg.norm(expected)),
                           "max_abs_difference": float(np.abs(value - expected).max())}
                    for name, value, expected in zip(OUTPUTS, call, reference)})
            settings.append(record)
            worst = max(c[name]["relative_error"] for c in record["calls"] for name in OUTPUTS)
            print(f"[cpu encoder] threads {threads} {level}: largest relative error {worst:.2e}, "
                  f"repeat calls equal {record['repeat_calls_equal']}", flush=True)
            del session
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "issue #8: SAM 2.1 tiny encoder, ONNX Runtime CPU against PyTorch",
        "onnxruntime": ort.__version__, "torch": torch.__version__, "photo": Path(args.photo).name,
        "onnx": args.onnx, "machine": {"platform": platform.platform(),
                                       "processor": platform.processor()},
        "settings": settings}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
