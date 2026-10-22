#!/usr/bin/env python3
"""Find the ONNX Runtime graph optimizer that changes the SAM 2.1 tiny encoder output on the CPU
provider (issue #8: onnxruntime 1.24.4 on Apple silicon).

Input:  --onnx (point_encoder.onnx), --photo (preprocessed with sam2_graphs.preprocess_image after
        the scaling of mask_pipeline.load_photo), --level (the graph optimization level of the
        test sessions, default ORT_ENABLE_BASIC), --optimizers (the optimizer names to disable,
        one at a time; default OPTIMIZERS).
Output: --out (JSON): the reference output (graph optimizations off, ORT_DISABLE_ALL); the output
        at --level with all optimizers; and at --level with each optimizer of --optimizers
        disabled (`InferenceSession(disabled_optimizers=[name])`): per output the relative error
        ||test - reference|| / ||reference||. An optimizer whose removal gives a relative error
        below 1e-5 is the optimizer that changes the output.

The tolerance 1e-5: the float32 difference between ONNX Runtime and PyTorch on this encoder is at
most 3.1e-6 (probe_cpu_encoder_settings.py, ORT_DISABLE_ALL). CPU threads 4, CPU memory arena off.
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
sys.path.insert(0, str(HERE.parents[1] / "poc" / "ai_mask_mvp"))

OUTPUTS = ("image_embed", "high_res_feats_0", "high_res_feats_1")
TOLERANCE = 1e-5
# Graph transformers of the ONNX Runtime levels basic and extended (onnxruntime/core/optimizer).
OPTIMIZERS = (
    "ConstantFolding", "ConstantSharing", "CommonSubexpressionElimination",
    "ConcatSliceElimination", "EliminateIdentity", "EliminateSlice", "UnsqueezeElimination",
    "EliminateDropout", "ExpandElimination", "CastElimination", "NoopElimination",
    "DivMulFusion", "FuseReluClip", "GemmTransposeFusion", "NotWhereFusion", "ConvAddFusion",
    "ConvMulFusion", "ConvBNFusion", "PadFusion", "MatmulTransposeFusion",
    "MatmulBNFusion", "LabelEncoderFusion", "EliminateTranspose", "ReshapeFusion",
    "FreeDimensionOverrideTransformer", "QDQPropagationTransformer", "EnsureUniqueDQForNodeUnit",
    "RocmBlasAltImpl", "TransposeOptimizer", "GeluFusion", "LayerNormFusion",
    "SimplifiedLayerNormFusion", "AttentionFusion", "EmbedLayerNormFusion", "BiasGeluFusion",
    "FastGeluFusion", "SkipLayerNormFusion", "MatMulAddFusion", "MatMulScaleFusion",
    "GatherSliceToSplitFusion", "GatherToSliceFusion", "MatMulActivationFusion",
    "ConvActivationFusion", "QuickGeluFusion", "BiasSoftmaxFusion", "BiasDropoutFusion",
    "IsInfReduceSumFusion", "SceLossGradBiasFusion", "DynamicQuantizeMatMulFusion",
)


def relative_errors(outputs: list[np.ndarray], reference: list[np.ndarray]) -> dict:
    return {name: float(np.linalg.norm(value - expected) / np.linalg.norm(expected))
            for name, value, expected in zip(OUTPUTS, outputs, reference)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--level", default="ORT_ENABLE_BASIC")
    parser.add_argument("--optimizers", nargs="*", default=list(OPTIMIZERS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    import mask_pipeline as mp
    import sam2_graphs as graphs

    pixels = graphs.preprocess_image(mp.load_photo(Path(args.photo)).image).numpy()

    def run(level: str, disabled: list[str]) -> list[np.ndarray]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 3
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
        session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"],
                                       disabled_optimizers=disabled)
        return session.run(None, {"pixel_values": pixels})

    reference = run("ORT_DISABLE_ALL", [])
    all_optimizers = relative_errors(run(args.level, []), reference)
    print(f"[optimizers] {args.level}, all optimizers: {max(all_optimizers.values()):.2e}",
          flush=True)
    rows = []
    for name in args.optimizers:
        errors = relative_errors(run(args.level, [name]), reference)
        rows.append({"disabled": name, "relative_error": errors,
                     "matches_reference": max(errors.values()) < TOLERANCE})
        print(f"[optimizers] {args.level} without {name}: {max(errors.values()):.2e}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "issue #8: ONNX Runtime CPU graph optimizers and the SAM 2.1 tiny encoder",
        "onnxruntime": ort.__version__, "onnx": args.onnx, "photo": Path(args.photo).name,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "level": args.level, "tolerance": TOLERANCE,
        "all_optimizers_relative_error": all_optimizers, "disabled_one_at_a_time": rows,
        "optimizers_that_change_the_output": [r["disabled"] for r in rows
                                              if r["matches_reference"]],
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
