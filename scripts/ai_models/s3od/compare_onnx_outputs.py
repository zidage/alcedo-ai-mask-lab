#!/usr/bin/env python3
"""Compare the S3OD ONNX graphs with the PyTorch model of the source on the CPU (issue #6).

Check name: OnnxMasksMatchPytorchWithinTolerance.

Input:  --source and --checkpoint (as export_onnx.py), one or more --graph LABEL=PATH, the photos
        of --prompts in --photos (the quality set at a long edge of at most 4096 px).
Output: --out (JSON report) and, next to it, per photo:
        reference_alpha/<stem>.npy  the PyTorch alpha of the source pipeline
                                    (`BackgroundRemoval.remove_background`, `predicted_mask`) at
                                    the photo size, float16;
        <label>/alpha/<stem>.npy    the ONNX Runtime CPU alpha of each graph, unpadded, at the
                                    model resolution (long edge 1024), float16, for
                                    probe_subject_masks.py.

Per photo and graph the script runs the graph on ONNX Runtime CPU with the pixels of the source
preprocessing (`BackgroundRemoval._preprocess`) and compares with the PyTorch model output on the
same pixels:

- `max_abs_masks_diff`: maximum absolute difference of the three sigmoid masks at 1024 x 1024;
- `max_abs_iou_diff`: maximum absolute difference of the three sigmoid IoU scores;
- `same_selected_mask`: the largest IoU score selects the same mask;
- at the photo size (the postprocessing of the source on both outputs): the maximum absolute alpha
  difference and the IoU of alpha > 0.5.

Requirement for a graph with `--required LABEL` (the float32 graph): `max_abs_masks_diff` and
`max_abs_iou_diff` <= 1e-3 and the same selected mask on every photo (issue #6, acceptance
criteria). The other graphs (float16) are reported without a requirement.

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

import s3od_graphs as graphs  # noqa: E402

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
    s3od = graphs.import_source(args.source)
    model = graphs.load_model(args.source, Path(args.checkpoint))
    wrapper = graphs.subject_masks_module(model)
    predictor = s3od.predictor.BackgroundRemoval.__new__(s3od.predictor.BackgroundRemoval)
    predictor.image_size = graphs.IMAGE_SIZE
    predictor.device = "cpu"
    predictor.model = model
    predictor.mean = np.array(graphs.NORMALIZE_MEAN)
    predictor.std = np.array(graphs.NORMALIZE_STD)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    sessions = {label: ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
                for label, path in args.graph}

    out_path = Path(args.out)
    (out_path.parent / "reference_alpha").mkdir(parents=True, exist_ok=True)
    for label in sessions:
        (out_path.parent / label / "alpha").mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: list[dict] = []
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        image = graphs.load_rgb(Path(args.photos) / entry["file"])
        height, width = image.shape[:2]
        tensor, info = predictor._preprocess(image)
        own_pixels, own_info = graphs.letterbox(image)
        preprocess_diff = float(np.abs(tensor.numpy() - own_pixels).max())
        started = time.perf_counter()
        reference_masks, reference_iou = (t.numpy() for t in wrapper(tensor))
        torch_seconds = time.perf_counter() - started
        source_alpha = predictor.remove_background(image).predicted_mask
        reference_alpha, reference_index = graphs.select_mask(reference_masks, reference_iou)
        reference_photo = graphs.masks_to_photo(graphs.unpad(reference_alpha[None], info),
                                                height, width)[0]
        np.save(out_path.parent / "reference_alpha" / f"{stem}.npy",
                source_alpha.astype(np.float16))
        result = {"image": entry["file"], "photo_size": [width, height],
                  "pad": {"top": info["height_pad"], "left": info["width_pad"],
                          "resized": list(info["resized_size"])},
                  "preprocess_max_abs_diff": preprocess_diff,
                  "source_pipeline_max_abs_diff": float(np.abs(reference_photo
                                                               - source_alpha).max()),
                  "torch_cpu_seconds": round(torch_seconds, 2),
                  "reference_iou_scores": [round(float(v), 5) for v in reference_iou[0]],
                  "reference_selected_mask": reference_index, "graphs": {}}
        for label, session in sessions.items():
            started = time.perf_counter()
            masks, iou_scores = session.run(None, {"pixel_values": own_pixels})
            onnx_seconds = time.perf_counter() - started
            alpha, index = graphs.select_mask(masks, iou_scores)
            unpadded = graphs.unpad(alpha[None], own_info)
            photo = graphs.masks_to_photo(unpadded, height, width)[0]
            np.save(out_path.parent / label / "alpha" / f"{stem}.npy",
                    unpadded[0].astype(np.float16))
            record = {
                "max_abs_masks_diff": float(np.abs(masks - reference_masks).max()),
                "mean_abs_masks_diff": float(np.abs(masks - reference_masks).mean()),
                "max_abs_iou_diff": float(np.abs(iou_scores - reference_iou).max()),
                "selected_mask": index,
                "same_selected_mask": index == reference_index,
                "photo_max_abs_alpha_diff": float(np.abs(photo - reference_photo).max()),
                "photo_iou": iou(photo > 0.5, reference_photo > 0.5),
                "onnxruntime_cpu_seconds": round(onnx_seconds, 2),
            }
            if label in args.required:
                record["passed"] = (record["max_abs_masks_diff"] <= TOLERANCE
                                    and record["max_abs_iou_diff"] <= TOLERANCE
                                    and record["same_selected_mask"])
            result["graphs"][label] = record
            print(f"[compare] {label} {stem}: max|dmask| {record['max_abs_masks_diff']:.2e} "
                  f"max|diou| {record['max_abs_iou_diff']:.2e} mask {index}/{reference_index} "
                  f"IoU {record['photo_iou']:.5f} ort {onnx_seconds:.1f} s", flush=True)
        results.append(result)

    summary = {}
    for label, _ in args.graph:
        records = [r["graphs"][label] for r in results]
        summary[label] = {
            "required": label in args.required,
            "max_abs_masks_diff": max(r["max_abs_masks_diff"] for r in records),
            "max_abs_iou_diff": max(r["max_abs_iou_diff"] for r in records),
            "same_selected_mask": sum(r["same_selected_mask"] for r in records),
            "minimum_photo_iou": min(r["photo_iou"] for r in records),
            "photo_max_abs_alpha_diff": max(r["photo_max_abs_alpha_diff"] for r in records),
        }
        if label in args.required:
            summary[label]["passed"] = all(r["passed"] for r in records)
    passed = all(s.get("passed", True) for s in summary.values())
    report = {"check": "OnnxMasksMatchPytorchWithinTolerance", "tolerance": TOLERANCE,
              "onnxruntime": ort.__version__, "torch": torch.__version__,
              "threads": args.threads, "graphs": {label: path.as_posix()
                                                  for label, path in args.graph},
              "passed": passed, "summary": summary, "results": results}
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for label, values in summary.items():
        print(f"[compare] {label}: {values}")
    print(f"[compare] {'PASS' if passed else 'FAIL'}; report {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
