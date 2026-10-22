#!/usr/bin/env python3
"""SAM 2.1 tiny encoder outputs of the native Core ML model per compute units against ONNX
Runtime CPU and PyTorch on photos (issue #8).

Input:  --model (the compiled `point_encoder.mlmodelc`), --onnx (the accepted
        `point_encoder.onnx`), --sam2-source and --sam2-checkpoint (PyTorch reference),
        --photos and --photo (stems, repeatable), --compute-units (one or more of ALL,
        CPU_AND_GPU, CPU_AND_NE, CPU_ONLY), --input (`photo`: the photo preprocessed with
        sam2_graphs.preprocess_image; `random`: a standard normal tensor of the same shape, the
        input of the convert_models.py check).
Output: --out (JSON): per photo and per output (image_embed, high_res_feats_0,
        high_res_feats_1), for each pair of ONNX Runtime CPU, PyTorch float32, and Core ML per
        compute units: the largest absolute difference, the relative error
        ||a - b|| / ||b||, and the cosine similarity; the value range of each output.

All runs are in this process, one after the other (4 CPU threads). Run it in the MVP environment
with the SAM 2 source on the path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sam2"))
sys.path.insert(0, str(HERE.parents[1] / "poc" / "ai_mask_mvp"))

OUTPUTS = ("image_embed", "high_res_feats_0", "high_res_feats_1")
COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")


def errors(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    difference = a64 - b64
    return {"max_abs_difference": float(np.abs(difference).max()),
            "relative_error": float(np.linalg.norm(difference) / max(np.linalg.norm(b64), 1e-30)),
            "cosine": float(a64 @ b64 / max(np.linalg.norm(a64) * np.linalg.norm(b64), 1e-30))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--sam2-source", default="build/tmp/explore/src/sam2")
    parser.add_argument("--sam2-checkpoint",
                        default="build/tmp/explore/weights/sam2.1_hiera_tiny.pt")
    parser.add_argument("--photos", required=True)
    parser.add_argument("--photo", action="append", required=True)
    parser.add_argument("--compute-units", nargs="+", default=list(COMPUTE_UNITS),
                        choices=COMPUTE_UNITS)
    parser.add_argument("--input", choices=("photo", "random"), default="photo")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import coremltools as ct
    import torch

    import mask_pipeline as mp
    import sam2_graphs as graphs

    torch.set_num_threads(4)
    graphs.add_repository_to_path(args.sam2_source)
    encoder = graphs.PointEncoder(graphs.build_model("tiny", args.sam2_checkpoint)).eval()
    onnx_session = mp.create_session(Path(args.onnx), "cpu", 4)
    coreml = {units: ct.models.CompiledMLModel(args.model,
                                               compute_units=getattr(ct.ComputeUnit, units))
              for units in args.compute_units}
    records = []
    generator = np.random.default_rng(0)
    for stem in args.photo:
        if args.input == "photo":
            photo = mp.load_photo(Path(args.photos) / f"{stem}.jpg")
            pixels = graphs.preprocess_image(photo.image).numpy()
        else:
            pixels = generator.standard_normal((1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE),
                                               dtype=np.float32)
        outputs = {"onnxruntime_cpu": dict(zip(OUTPUTS, onnx_session.run(None,
                                                                         {"pixel_values": pixels})))}
        with torch.no_grad():
            outputs["pytorch"] = dict(zip(OUTPUTS, (t.numpy() for t in
                                                    encoder(torch.from_numpy(pixels)))))
        for units, model in coreml.items():
            prediction = model.predict({"pixel_values": pixels})
            outputs[f"coreml_{units}"] = {name: np.asarray(prediction[name], np.float32)
                                          for name in OUTPUTS}
        record = {"photo": stem if args.input == "photo" else f"random {stem}",
                  "ranges": {name: [float(outputs["pytorch"][name].min()),
                                    float(outputs["pytorch"][name].max())] for name in OUTPUTS},
                  "pairs": {}}
        labels = list(outputs)
        for index, first in enumerate(labels):
            for second in labels[index + 1:]:
                record["pairs"][f"{first} vs {second}"] = {
                    name: errors(outputs[first][name], outputs[second][name]) for name in OUTPUTS}
        records.append(record)
        for pair, values in record["pairs"].items():
            print(f"[encoder] {record['photo']} {pair}: relative error "
                  + ", ".join(f"{name} {values[name]['relative_error']:.2e}" for name in OUTPUTS),
                  flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"step": "issue #8: native encoder outputs", "input": args.input,
                               "model": args.model, "onnx": args.onnx, "records": records},
                              indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
