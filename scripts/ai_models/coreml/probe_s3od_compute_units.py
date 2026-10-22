#!/usr/bin/env python3
"""S3OD native Core ML output per compute units against a reference alpha (issue #8).

Input:  --model (a compiled S3OD model `.mlmodelc` of convert_models.py), --compute-units (one or
        more of ALL, CPU_AND_GPU, CPU_AND_NE, CPU_ONLY), --photos, --photo (stems, repeatable),
        --reference-alpha (a directory of reference alphas, <stem>.npy on the unpadded grid with
        the long edge 1024 px: S3OD float32 on the ONNX Runtime CPU), --repeats (default 2).
Output: --out (JSON): per compute units and photo, for each repeat: the call time, the IoU scores,
        the selected mask, the IoU of alpha > 0.5 with the reference, the largest absolute alpha
        difference, and the number of NaN and infinite values; and the record of the model
        (convert_models.py: precision, float32 op types and module scopes).

The photo is read and letterboxed with s3od_graphs (load_rgb, letterbox); the alpha is the mask
with the largest IoU score, unpadded (select_mask, unpad), as measure_directml_cost.py. A repeat
on the same photo shows whether the output of the compute units is the same on each call. The
first call includes the model preparation of Core ML.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s3od"))
import s3od_graphs  # noqa: E402

COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compute-units", nargs="+", default=["ALL"], choices=COMPUTE_UNITS)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--photo", action="append", required=True)
    parser.add_argument("--reference-alpha", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import coremltools as ct

    model_path = Path(args.model)
    record = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    prepared = {}
    for stem in args.photo:
        pixels, info = s3od_graphs.letterbox(s3od_graphs.load_rgb(Path(args.photos) / f"{stem}.jpg"))
        reference = np.load(Path(args.reference_alpha) / f"{stem}.npy").astype(np.float32)
        prepared[stem] = (pixels, info, reference)
    results = {}
    for units in args.compute_units:
        model = ct.models.CompiledMLModel(str(model_path),
                                          compute_units=getattr(ct.ComputeUnit, units))
        rows = []
        for repeat in range(args.repeats):
            for stem, (pixels, info, reference) in prepared.items():
                started = time.perf_counter()
                outputs = model.predict({"pixel_values": pixels})
                milliseconds = (time.perf_counter() - started) * 1000
                masks = np.asarray(outputs["masks"], np.float32)
                scores = np.asarray(outputs["iou_scores"], np.float32)
                alpha, index = s3od_graphs.select_mask(masks, scores)
                alpha = s3od_graphs.unpad(alpha[None], info)[0]
                a, b = alpha > 0.5, reference > 0.5
                union = int((a | b).sum())
                rows.append({"photo": stem, "repeat": repeat, "ms": round(milliseconds, 1),
                             "iou_scores": [round(float(v), 4) for v in scores[0]],
                             "selected_mask": index,
                             "iou_alpha_gt_0_5": 1.0 if union == 0 else float((a & b).sum() / union),
                             "max_abs_alpha_difference": float(np.abs(alpha - reference).max()),
                             "non_finite": int(np.count_nonzero(~np.isfinite(masks)))})
                print(f"[probe] {units} {stem} repeat {repeat}: {milliseconds:.0f} ms IoU "
                      f"{rows[-1]['iou_alpha_gt_0_5']:.5f}", flush=True)
        results[units] = {"rows": rows,
                          "minimum_iou": min(r["iou_alpha_gt_0_5"] for r in rows),
                          "ms_after_first_call": [r["ms"] for r in rows[1:]]}
        del model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "issue #8: S3OD native Core ML output per compute units",
        "model": model_path.as_posix(),
        "model_record": {k: record.get(k) for k in ("precision", "float32_op_types",
                                                    "float32_scopes", "float32_ops",
                                                    "coremltools", "torch", "mlmodelc")},
        "reference_alpha": Path(args.reference_alpha).as_posix(), "results": results,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
