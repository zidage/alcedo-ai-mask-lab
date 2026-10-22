#!/usr/bin/env python3
"""Compare the IS-Net general-use ONNX graphs with the PyTorch model of the source on the CPU
(issue #6).

Check name: OnnxD1MatchesPytorchWithinTolerance.

Input:  --source and --checkpoint (as export_onnx.py), one or more --graph LABEL=PATH, the photos
        of --prompts in --photos (the quality set at a long edge of at most 4096 px).
Output: --out (JSON report) and, next to it, per photo:
        reference_alpha/<stem>.npy  the PyTorch alpha of the source pipeline (`Inference.py`:
                                    d1 resized to the photo size, then scaled with its minimum
                                    and maximum) at the photo size, float16;
        reference_d1/<stem>.npy     the PyTorch alpha in the format of the DirectML scripts
                                    (d1 scaled with its minimum and maximum at 1024 x 1024),
                                    float16, for probe_subject_masks.py (kind isnet);
        <label>/alpha/<stem>.npy    the same for the ONNX Runtime CPU output of each graph.

Per photo and graph the script runs the graph on ONNX Runtime CPU with the pixels of the source
preprocessing (isnet_graphs.preprocess) and compares with the PyTorch model output on the same
pixels:

- `max_abs_d1_diff`: maximum absolute difference of the d1 output (sigmoid) at 1024 x 1024;
- `max_abs_alpha_diff`: the same after the scaling with the minimum and maximum;
- at the photo size (the postprocessing of the source on both outputs): the maximum absolute alpha
  difference and the IoU of alpha > 0.5.

Requirement for a graph with `--required LABEL` (the float32 graph): `max_abs_d1_diff` <= 1e-3 on
every photo (issue #6, acceptance criteria). The other graphs (float16) are reported without a
requirement.

`preprocess_max_abs_diff` is the maximum absolute difference between isnet_graphs.preprocess and
the statements of `Inference.py` (`F.upsample`, `.type(torch.uint8)`, torchvision `normalize`).
`source_pipeline_max_abs_diff` is the maximum absolute difference between the alpha of the source
order (resize, then scale) and the order of the stored alpha (scale at 1024 x 1024, then resize),
both from the PyTorch d1.

PyTorch and ONNX Runtime use --threads CPU threads (default 4); the ONNX Runtime CPU memory arena
is off (AGENTS.md resource limits). The graphs run one at a time after the PyTorch run of a photo.
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

import isnet_graphs as graphs  # noqa: E402

TOLERANCE = 1e-3


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def source_pixels(image: np.ndarray) -> np.ndarray:
    """The preprocessing statements of `Inference.py`, unchanged."""
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import normalize

    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), [graphs.IMAGE_SIZE, graphs.IMAGE_SIZE],
                           mode="bilinear").type(torch.uint8)
    pixels = torch.divide(im_tensor, 255.0)
    return normalize(pixels, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).numpy()


def parse_graph(text: str) -> tuple[str, Path]:
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph", action="append", required=True, type=parse_graph,
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
    import torch

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    wrapper = graphs.subject_d1_module(graphs.load_model(args.source, Path(args.checkpoint)))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    sessions = {label: ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
                for label, path in args.graph}

    out_path = Path(args.out)
    for directory in ("reference_alpha", "reference_d1"):
        (out_path.parent / directory).mkdir(parents=True, exist_ok=True)
    for label in sessions:
        (out_path.parent / label / "alpha").mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        image = graphs.load_rgb(Path(args.photos) / entry["file"])
        height, width = image.shape[:2]
        pixels = graphs.preprocess(image)
        started = time.perf_counter()
        reference_d1 = wrapper(torch.from_numpy(pixels)).numpy()
        torch_seconds = time.perf_counter() - started
        reference_alpha = graphs.normalize_range(reference_d1[0, 0])
        reference_photo = graphs.source_alpha(reference_d1, height, width)
        stored_order_photo = graphs.alpha_to_photo(reference_alpha, height, width)
        np.save(out_path.parent / "reference_alpha" / f"{stem}.npy",
                reference_photo.astype(np.float16))
        np.save(out_path.parent / "reference_d1" / f"{stem}.npy",
                reference_alpha.astype(np.float16))
        result = {"image": entry["file"], "photo_size": [width, height],
                  "preprocess_max_abs_diff": float(np.abs(source_pixels(image) - pixels).max()),
                  "reference_d1_range": [float(reference_d1.min()), float(reference_d1.max())],
                  "source_pipeline_max_abs_diff": float(np.abs(stored_order_photo
                                                               - reference_photo).max()),
                  "torch_cpu_seconds": round(torch_seconds, 2), "graphs": {}}
        for label, session in sessions.items():
            started = time.perf_counter()
            (d1,) = session.run(None, {graphs.INPUT_NAME: pixels})
            onnx_seconds = time.perf_counter() - started
            alpha = graphs.normalize_range(d1[0, 0])
            photo = graphs.source_alpha(d1, height, width)
            np.save(out_path.parent / label / "alpha" / f"{stem}.npy", alpha.astype(np.float16))
            record = {
                "max_abs_d1_diff": float(np.abs(d1 - reference_d1).max()),
                "mean_abs_d1_diff": float(np.abs(d1 - reference_d1).mean()),
                "max_abs_alpha_diff": float(np.abs(alpha - reference_alpha).max()),
                "d1_range": [float(d1.min()), float(d1.max())],
                "photo_max_abs_alpha_diff": float(np.abs(photo - reference_photo).max()),
                "photo_iou": iou(photo > 0.5, reference_photo > 0.5),
                "onnxruntime_cpu_seconds": round(onnx_seconds, 2),
            }
            if label in args.required:
                record["passed"] = record["max_abs_d1_diff"] <= TOLERANCE
            result["graphs"][label] = record
            print(f"[compare] {label} {stem}: max|dd1| {record['max_abs_d1_diff']:.2e} "
                  f"max|dalpha| {record['max_abs_alpha_diff']:.2e} "
                  f"IoU {record['photo_iou']:.5f} ort {onnx_seconds:.1f} s", flush=True)
        results.append(result)

    summary = {}
    for label, _ in args.graph:
        records = [r["graphs"][label] for r in results]
        summary[label] = {
            "required": label in args.required,
            "max_abs_d1_diff": max(r["max_abs_d1_diff"] for r in records),
            "max_abs_alpha_diff": max(r["max_abs_alpha_diff"] for r in records),
            "minimum_photo_iou": min(r["photo_iou"] for r in records),
            "photo_max_abs_alpha_diff": max(r["photo_max_abs_alpha_diff"] for r in records),
        }
        if label in args.required:
            summary[label]["passed"] = all(r["passed"] for r in records)
    passed = all(s.get("passed", True) for s in summary.values())
    report = {"check": "OnnxD1MatchesPytorchWithinTolerance", "tolerance": TOLERANCE,
              "onnxruntime": ort.__version__, "torch": torch.__version__,
              "threads": args.threads, "graphs": {label: path.as_posix()
                                                  for label, path in args.graph},
              "preprocess_max_abs_diff": max(r["preprocess_max_abs_diff"] for r in results),
              "source_pipeline_max_abs_diff": max(r["source_pipeline_max_abs_diff"]
                                                  for r in results),
              "passed": passed, "summary": summary, "results": results}
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for label, values in summary.items():
        print(f"[compare] {label}: {values}")
    print(f"[compare] {'PASS' if passed else 'FAIL'}; report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
