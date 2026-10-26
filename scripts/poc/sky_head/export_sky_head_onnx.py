#!/usr/bin/env python3
"""Export the fine sky head to ONNX and compare ONNX Runtime with PyTorch (issue #10).

Input:  --checkpoint (best.pt of train_sky_head.py), --data-dir and --split (photos for the
        comparison), --sam2-repo, --sam2-checkpoint.
Output: --out-dir/sky_head.onnx (inputs image_embed [1,256,64,64], high_res_feats_0
        [1,32,256,256], high_res_feats_1 [1,64,128,128], pixel_values [1,3,1024,1024]; output
        sky_alpha [1,1,1024,1024]) and --out-dir/sky_head.json (SHA-256, size, the comparison).

Comparison: the encoder outputs of --compare-photos photos (float32 PyTorch) go into the
PyTorch head and into ONNX Runtime 1.31 on the CPU. Acceptance: maximum absolute alpha
difference <= TOLERANCE on every photo; the script stops with an error otherwise.

TOLERANCE is 1e-3 (accepted by the user 2026-10-26; docs/exploration/sky_head_mass13k.md,
section 6): ONNX Runtime sums the GroupNorm statistics over up to 2 M float32 values per group,
and run 1 differs from PyTorch by up to 4.0e-4 (PyTorch float32 against float64: at most 8.6e-6).
1e-3 is about a quarter of one 8-bit alpha level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_head_data as sd  # noqa: E402

OPSET = 17
TOLERANCE = 1e-3
INPUT_NAMES = ["image_embed", "high_res_feats_0", "high_res_feats_1", "pixel_values"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compare-photos", type=int, default=8)
    args = parser.parse_args()

    import onnxruntime
    import torch
    import sky_head_model as model

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = model.SkyAlphaHead(tuple(state["widths"]))
    head.load_state_dict(state["head"])
    graph = model.ExportedSkyHead(head).eval()
    encoder = model.build_encoder(args.sam2_repo, args.sam2_checkpoint)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "sky_head.onnx"
    example = (torch.zeros(1, 256, 64, 64), torch.zeros(1, 32, 256, 256),
               torch.zeros(1, 64, 128, 128), torch.zeros(1, 3, 1024, 1024))
    torch.onnx.export(graph, example, str(onnx_path), input_names=INPUT_NAMES,
                      output_names=["sky_alpha"], opset_version=OPSET, dynamo=False)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_cpu_mem_arena = False
    session = onnxruntime.InferenceSession(str(onnx_path), options,
                                           providers=["CPUExecutionProvider"])
    items = sd.split_items(args.data_dir, args.split)[:args.compare_photos]
    differences = []
    with torch.no_grad():
        for item in items:
            photo, _, _ = sd.load_sample(item)
            pixels = sd.model_pixels(photo)[None]
            embed, high_0, high_1 = encoder(pixels)
            reference = graph(embed, high_0, high_1, pixels).numpy()
            feeds = dict(zip(INPUT_NAMES, (t.numpy() for t in (embed, high_0, high_1, pixels))))
            onnx_alpha = session.run(None, feeds)[0]
            differences.append({"stem": item["stem"],
                                "max_abs": float(np.abs(onnx_alpha - reference).max())})
    worst = max(d["max_abs"] for d in differences)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    record = {"step": "issue #10: sky_head.onnx", "checkpoint": str(args.checkpoint),
              "epoch": state.get("epoch"), "sources": state.get("sources"),
              "onnx": {"path": onnx_path.name, "bytes": onnx_path.stat().st_size,
                       "sha256": digest, "opset": OPSET},
              "onnxruntime": onnxruntime.__version__, "tolerance": TOLERANCE,
              "max_abs_difference": worst, "photos": differences}
    sd.write_json(args.out_dir / "sky_head.json", record)
    print(json.dumps({k: record[k] for k in ("onnx", "max_abs_difference")}, indent=1))
    if worst > TOLERANCE:
        raise RuntimeError(f"ONNX Runtime differs from PyTorch by {worst} > {TOLERANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
