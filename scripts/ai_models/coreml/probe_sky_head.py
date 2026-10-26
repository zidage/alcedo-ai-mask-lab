#!/usr/bin/env python3
"""Sky alpha of the native Core ML sky head against ONNX Runtime CPU on photos (AlcedoStudio
issue 538).

Input:  --onnx-dir (the accepted `point_encoder.onnx` and `sky_head.onnx`), --encoder (the
        compiled `point_encoder.mlmodelc`), --sky-head LABEL=PATH (one compiled
        `sky_head.mlmodelc` per conversion, repeatable), --compute-units (one or more of ALL,
        CPU_AND_GPU, CPU_AND_NE, CPU_ONLY), --photos (directories of JPEG photos, repeatable),
        --runs (timed sky head runs per photo).
Output: --out (JSON) and one line per photo and variant on stdout.

Reference: ONNX Runtime CPU (4 threads, CPU memory arena off) runs `point_encoder.onnx` and
`sky_head.onnx` on the photo stretched to 1024 x 1024 (sam2_graphs.preprocess_image). For each
sky head conversion and compute units:

  head      the Core ML sky head on the ONNX Runtime encoder outputs and the same pixels: the
            difference of the conversion alone
  pipeline  the Core ML encoder (same compute units; the product runs the encoder with ALL, and
            with CPU_ONLY on the `cpu` device) and the Core ML sky head: the difference that the
            Alcedo Studio port check sees

Per alpha against the reference: the largest and the mean absolute difference, the count of
values that differ by more than 1/255, the IoU of alpha >= 0.5, the sky share (alpha >= 0.5) and
"sky found" (share >= 5e-4, the Alcedo Studio SM3.E rule S3). Per variant: the model load time
(first prediction included) and the median and largest time of --runs sky head runs after one
untimed run per photo.

All runs are in this process, one after the other. Run it on the Mac in the Core ML conversion
environment with onnxruntime 1.31 added (ONNX Runtime 1.24 to 1.27 miscompute the encoder on the
Mac, microsoft/onnxruntime#29071).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sam2"))

FEATURES = ("image_embed", "high_res_feats_0", "high_res_feats_1")
COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")
SKY_FOUND_SHARE = 5e-4
ALPHA_LEVEL = 1.0 / 255.0


def ort_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def compare(alpha: np.ndarray, reference: np.ndarray) -> dict:
    difference = np.abs(alpha.astype(np.float64) - reference.astype(np.float64))
    inside, reference_inside = alpha >= 0.5, reference >= 0.5
    union = int(np.logical_or(inside, reference_inside).sum())
    share = float(inside.mean())
    return {"max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "values_above_one_level": int((difference > ALPHA_LEVEL).sum()),
            "iou": (float(np.logical_and(inside, reference_inside).sum()) / union
                    if union else 1.0),
            "share": share, "found": share >= SKY_FOUND_SHARE}


def photo_paths(directories: list[str]) -> list[Path]:
    paths = []
    for directory in directories:
        paths.extend(sorted(Path(directory).glob("*.jpg")))
    if not paths:
        raise ValueError(f"no JPEG photos in {directories}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--sky-head", action="append", required=True,
                        help="LABEL=PATH of a compiled sky_head.mlmodelc")
    parser.add_argument("--compute-units", nargs="+", default=["ALL", "CPU_ONLY"],
                        choices=COMPUTE_UNITS)
    parser.add_argument("--photos", action="append", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import coremltools as ct
    import torch
    from PIL import Image

    import sam2_graphs as graphs

    torch.set_num_threads(4)
    onnx_dir = Path(args.onnx_dir)
    onnx_encoder = ort_session(onnx_dir / "point_encoder.onnx")
    onnx_sky_head = ort_session(onnx_dir / "sky_head.onnx")
    variants = {}
    for item in args.sky_head:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--sky-head {item!r}: expected LABEL=PATH")
        for units in args.compute_units:
            started = time.perf_counter()
            encoder = ct.models.CompiledMLModel(args.encoder,
                                                compute_units=getattr(ct.ComputeUnit, units))
            encoder_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            head = ct.models.CompiledMLModel(path, compute_units=getattr(ct.ComputeUnit, units))
            load_ms = (time.perf_counter() - started) * 1000
            variants[f"{label}/{units}"] = {"label": label, "units": units, "path": path,
                                            "encoder": encoder, "head": head,
                                            "encoder_load_ms": encoder_ms, "load_ms": load_ms,
                                            "first_run_ms": None, "run_ms": []}

    records = []
    for path in photo_paths(args.photos):
        with Image.open(path) as image:
            pixels = graphs.preprocess_image(image.convert("RGB")).numpy()
        features = dict(zip(FEATURES, onnx_encoder.run(None, {"pixel_values": pixels})))
        reference = onnx_sky_head.run(None, dict(features, pixel_values=pixels))[0]
        reference_share = float((reference >= 0.5).mean())
        record = {"photo": path.stem, "reference_share": reference_share,
                  "reference_found": reference_share >= SKY_FOUND_SHARE, "variants": {}}
        for name, variant in variants.items():
            head = variant["head"]
            feed = dict(features, pixel_values=pixels)
            started = time.perf_counter()
            alpha = np.asarray(head.predict(feed)["sky_alpha"], np.float32)
            elapsed = (time.perf_counter() - started) * 1000
            if variant["first_run_ms"] is None:
                variant["first_run_ms"] = elapsed
            for _ in range(args.runs):
                started = time.perf_counter()
                head.predict(feed)
                variant["run_ms"].append((time.perf_counter() - started) * 1000)
            coreml_features = variant["encoder"].predict({"pixel_values": pixels})
            pipeline_feed = {name: np.asarray(coreml_features[name], np.float32)
                             for name in FEATURES}
            pipeline_feed["pixel_values"] = pixels
            pipeline = np.asarray(head.predict(pipeline_feed)["sky_alpha"], np.float32)
            record["variants"][name] = {"head": compare(alpha, reference),
                                        "pipeline": compare(pipeline, reference)}
            values = record["variants"][name]
            print(f"[sky head] {path.stem} {name}: head max {values['head']['max_abs_difference']:.2e}"
                  f" iou {values['head']['iou']:.6f}; pipeline max "
                  f"{values['pipeline']['max_abs_difference']:.2e} iou "
                  f"{values['pipeline']['iou']:.6f}; share {reference_share:.4f}", flush=True)
        records.append(record)

    summary = {}
    for name, variant in variants.items():
        rows = [record["variants"][name] for record in records]
        summary[name] = {
            "path": variant["path"], "encoder_load_ms": round(variant["encoder_load_ms"], 1),
            "load_ms": round(variant["load_ms"], 1),
            "first_run_ms": round(variant["first_run_ms"], 1),
            "run_ms_median": round(statistics.median(variant["run_ms"]), 1),
            "run_ms_max": round(max(variant["run_ms"]), 1), "runs": len(variant["run_ms"])}
        for kind in ("head", "pipeline"):
            values = [row[kind] for row in rows]
            summary[name][kind] = {
                "max_abs_difference": max(v["max_abs_difference"] for v in values),
                "mean_abs_difference": statistics.mean(v["mean_abs_difference"] for v in values),
                "min_iou": min(v["iou"] for v in values),
                "values_above_one_level_max": max(v["values_above_one_level"] for v in values),
                "found_differs": [record["photo"] for record, v in zip(records, values)
                                  if v["found"] != record["reference_found"]]}
        print(f"[sky head] {name}: {json.dumps(summary[name])}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"step": "AlcedoStudio issue 538: Core ML sky head",
                               "onnxruntime": __import__("onnxruntime").__version__,
                               "coremltools": ct.__version__, "photos": len(records),
                               "summary": summary, "records": records}, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
