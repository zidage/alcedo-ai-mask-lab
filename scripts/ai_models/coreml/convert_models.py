#!/usr/bin/env python3
"""Convert the accepted AI Mask models from PyTorch to native Core ML models (issue #8).

The models of the accepted set (acceptance record section 1) are converted with coremltools to
the Core ML MLProgram format (`.mlpackage`) and compiled to `.mlmodelc`, the form that Alcedo
Studio loads for its native Core ML SigLIP2 models (`coreml_clip.rs`, `coreml` crate,
`ModelConfiguration` with the compute units). The PyTorch modules are the modules of the ONNX
exports, so the Core ML models have the inputs and outputs of the ONNX graphs:

  point_encoder    sam2_graphs.PointEncoder (SAM 2.1 tiny, facebookresearch/sam2 at 2b90b9f5):
                   pixel_values [1, 3, 1024, 1024] -> image_embed, high_res_feats_0,
                   high_res_feats_1. The Hiera position embedding (a bicubic resize of a weight
                   to the 256 x 256 token grid, which coremltools cannot convert) is computed in
                   PyTorch and is a constant of the model
  point_decoder    sam2_graphs.PointDecoder: image_embed, high_res_feats_0, high_res_feats_1,
                   point_coords [1, N, 2], point_labels [1, N] (N = 1 to --max-points; a Core ML
                   range of shapes, or with --point-shapes enumerated one enumerated shape per
                   N), mask_input [1, 1, 256, 256], has_mask_input [1] ->
                   masks, iou_predictions, object_score_logits (the ONNX output order)
  people_sky_head  head_model.PeopleSkyHead of the P1 run 2 checkpoint (epoch 15): image_embed ->
                   the outputs of head_model.OUTPUT_NAMES
  s3od             s3od_graphs.subject_masks_module (S3OD s3od-synth, KupynOrest/s3od at
                   11385423) with the attention of the source (scaled_dot_product_attention, no
                   query chunks: the chunks of the ONNX graph are a DirectML memory measure):
                   pixel_values [1, 3, 1024, 1024] -> masks [1, 3, 1024, 1024], iou_scores [1, 3]
  sky_head         sky_head_model.ExportedSkyHead of the sky head run 1 checkpoint (epoch 15,
                   issue #10; the graph of sky_head.onnx): image_embed, high_res_feats_0,
                   high_res_feats_1, pixel_values -> sky_alpha [1, 1, 1024, 1024]

Precision (--precision): `float32` (the precision of the accepted SAM 2.1 tiny and head graphs),
`float16` (all ops in float16; the sky head measurement of AlcedoStudio issue 538),
`float16_embeddings_attention_float32` (S3OD: the DINOv3 patch embedding and the attention of all
encoder layers in float32, the other ops in float16: the patch embedding gives a wrong output on
the Neural Engine in float16, and the attention runs faster on the GPU; issue #8 report),
`float16_encoder_float32` (S3OD: the DINOv3 encoder in float32 and the DPT decoder in float16), or
`float16_layer0_float32` (S3OD: float16 with the encoder layer 0 attention in float32, the
precision of the accepted S3OD graph; the layer 0 attention scores reach 135,151 on the quality
photos, above the float16 maximum 65,504, S3OD export_onnx.py). The layer 0 attention ops are the
MIL `scaled_dot_product_attention` ops (and, when coremltools writes the attention as separate
ops, the `matmul` and `softmax` ops) whose TorchScript module scope is the encoder layer 0
attention; the script records their names and fails when it finds none. `--float32-op-types`
and `--float32-scopes` keep all ops of the given MIL op types or module paths in float32 as well
(the investigation of the Neural Engine output).

Input:  --model, --precision, the source checkouts and checkpoints, --out-dir.
Output: <out-dir>/<model>.mlpackage, <out-dir>/<model>.mlmodelc, and <out-dir>/<model>.json:
        coremltools and PyTorch versions, deployment target, precision, the float32 ops, the
        checkpoint SHA-256, the file sizes and a SHA-256 of the compiled model (over the sorted
        relative paths and contents of its files), the conversion time, and the largest absolute
        difference per output between the Core ML model (compute units CPU_ONLY and ALL) and
        PyTorch float32 on one input (recorded, no tolerance: the comparison with the ONNX graphs
        on the quality photos is compare_providers.py).

Run one model job at a time (AGENTS.md). Run Python with -I.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]
for directory in (SCRIPTS / "ai_models" / "sam2", SCRIPTS / "poc" / "people_sky_head",
                  SCRIPTS / "ai_models" / "s3od", SCRIPTS / "poc" / "sky_head"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

MODELS = ("point_encoder", "point_decoder", "people_sky_head", "s3od", "sky_head")
PRECISIONS = ("float32", "float16", "float16_layer0_float32", "float16_encoder_float32",
              "float16_embeddings_attention_float32")
# Module paths of the S3OD encoder (DPTSegmentation.encoder, DINOv3ViTModel) that stay float32 in
# the precisions float16_encoder_float32 and float16_embeddings_attention_float32.
PRECISION_SCOPES = {
    "float16_encoder_float32": (r"(^|\.)inner\.encoder(\.|$)",),
    "float16_embeddings_attention_float32": (
        r"(^|\.)inner\.encoder\.embeddings(\.|$)",
        r"(^|\.)inner\.encoder\.model\.[0-9]+\.attention(\.|$)"),
}
FLOAT32_ATTENTION_OPS = ("scaled_dot_product_attention", "matmul", "softmax")
# The module path of the S3OD encoder layer 0 attention: DPTSegmentation.encoder is the
# transformers DINOv3ViTModel, its layers are the ModuleList `model`.
LAYER0_ATTENTION = re.compile(r"(^|\.)encoder\.model\.0\.attention(\.|$)")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> tuple[str, int]:
    """SHA-256 over the sorted relative paths and file contents, and the total bytes."""
    digest = hashlib.sha256()
    total = 0
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_of(file).encode("ascii"))
        total += file.stat().st_size
    return digest.hexdigest(), total


def build_module(args):
    """The PyTorch module, its example inputs (name -> tensor), and the output names."""
    import torch

    if args.model in ("point_encoder", "point_decoder"):
        import sam2_graphs as graphs

        graphs.add_repository_to_path(args.sam2_source)
        model = graphs.build_model("tiny", args.sam2_checkpoint)
        # Hiera._get_pos_embed resizes the position embedding with a bicubic interpolate, which
        # coremltools cannot convert. The result depends only on the weights and the fixed token
        # grid, so it is computed once here and the trace has it as a constant (equal values).
        trunk = model.image_encoder.trunk
        grid = graphs.IMAGE_SIZE // 4
        with torch.no_grad():
            position_embedding = trunk._get_pos_embed((grid, grid)).detach().clone()

        def fixed_position_embedding(hw, expected=(grid, grid)):
            if tuple(hw) != expected:
                raise ValueError(f"token grid {tuple(hw)}, the Core ML model has {expected}")
            return position_embedding

        trunk._get_pos_embed = fixed_position_embedding
        if args.model == "point_encoder":
            return (graphs.PointEncoder(model).eval(),
                    {"pixel_values": torch.randn(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)},
                    ["image_embed", "high_res_feats_0", "high_res_feats_1"], args.sam2_checkpoint)
        with torch.no_grad():
            features = graphs.PointEncoder(model).eval()(
                torch.randn(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE))
        size = graphs.LOW_RES_MASK_SIZE
        inputs = {"image_embed": features[0], "high_res_feats_0": features[1],
                  "high_res_feats_1": features[2],
                  "point_coords": torch.tensor([[[400.0, 300.0], [600.0, 500.0]]]),
                  "point_labels": torch.tensor([[2.0, 3.0]]),
                  "mask_input": torch.zeros(1, 1, size, size),
                  "has_mask_input": torch.zeros(1)}
        return (graphs.PointDecoder(model).eval(), inputs,
                ["masks", "iou_predictions", "object_score_logits"], args.sam2_checkpoint)
    if args.model == "people_sky_head":
        import head_data as hd
        import head_model as hm

        state = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
        if state.get("epoch") not in (None, args.head_epoch):
            raise ValueError(f"{args.head_checkpoint}: epoch {state.get('epoch')}, expected "
                             f"{args.head_epoch}")
        config = state["config"]
        model = hm.PeopleSkyHead(config["in_channels"], config["width"],
                                 tuple(config["dilations"]), config["groups"])
        model.load_state_dict(state["model"])

        class OrderedOutputs(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, image_embed):
                outputs = self.inner(image_embed)
                return tuple(outputs[name] for name in hm.OUTPUT_NAMES)

        return (OrderedOutputs(model.eval()).eval(),
                {"image_embed": torch.randn(1, hd.EMBED_CHANNELS, hd.GRID, hd.GRID)},
                list(hm.OUTPUT_NAMES), args.head_checkpoint)
    if args.model == "sky_head":
        import sky_head_model as sm

        state = torch.load(args.sky_head_checkpoint, map_location="cpu", weights_only=False)
        if state.get("epoch") not in (None, args.sky_head_epoch):
            raise ValueError(f"{args.sky_head_checkpoint}: epoch {state.get('epoch')}, expected "
                             f"{args.sky_head_epoch}")
        head = sm.SkyAlphaHead(tuple(state["widths"]))
        head.load_state_dict(state["head"])
        # Random features and pixels of the input shapes of sky_head.onnx; the comparison with
        # ONNX Runtime on photos is probe_sky_head.py.
        inputs = {"image_embed": torch.randn(1, sm.EMBED_CHANNELS, 64, 64),
                  "high_res_feats_0": torch.randn(1, sm.HIGH_RES_0_CHANNELS, 256, 256),
                  "high_res_feats_1": torch.randn(1, sm.HIGH_RES_1_CHANNELS, 128, 128),
                  "pixel_values": torch.randn(1, 3, 1024, 1024)}
        return (sm.ExportedSkyHead(head).eval(), inputs, ["sky_alpha"],
                args.sky_head_checkpoint)
    import s3od_graphs

    model = s3od_graphs.load_model(args.s3od_source, Path(args.s3od_checkpoint))
    model.encoder.config._attn_implementation = "sdpa"
    size = s3od_graphs.IMAGE_SIZE
    return (s3od_graphs.subject_masks_module(model),
            {"pixel_values": torch.randn(1, 3, size, size)}, ["masks", "iou_scores"],
            args.s3od_checkpoint)


def layer0_attention_selector(names: list[str], scope_samples: list[str],
                              float32_op_types: tuple[str, ...] = (),
                              float32_scopes: tuple[str, ...] = ()):
    """op_selector of FP16ComputePrecision: False (keep float32) for the ops of
    FLOAT32_ATTENTION_OPS whose TorchScript module scope is the encoder layer 0 attention (the
    module path matches LAYER0_ATTENTION), for every op of `float32_op_types`, and for every op
    whose module path matches one of the regular expressions `float32_scopes`. The selected
    attention ops go to `names`; the scopes of the first attention ops go to `scope_samples` (for
    the error message when nothing is selected)."""
    from coremltools.converters.mil.mil.scope import ScopeSource

    scope_patterns = [re.compile(pattern) for pattern in float32_scopes]

    def selector(op) -> bool:
        if op.op_type in float32_op_types:
            return False
        scopes = getattr(op, "scopes", {}) or {}
        path = ".".join(str(v) for v in scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, []))
        if op.op_type in FLOAT32_ATTENTION_OPS:
            if len(scope_samples) < 8:
                scope_samples.append(f"{op.op_type}:{path}")
            if LAYER0_ATTENTION.search(path):
                names.append(f"{op.op_type}:{op.name}:{path}")
                return False
        return not any(pattern.search(path) for pattern in scope_patterns)

    return selector


def convert(args) -> dict:
    import coremltools as ct
    import torch

    torch.set_num_threads(args.threads)
    module, example, output_names, checkpoint = build_module(args)
    with torch.no_grad():
        reference = [t.numpy() for t in module(*example.values())]
        traced = torch.jit.trace(module, tuple(example.values()))
    inputs = []
    for name, tensor in example.items():
        shape = list(tensor.shape)
        if args.model == "point_decoder" and name in ("point_coords", "point_labels"):
            if args.point_shapes == "enumerated":
                shape = ct.EnumeratedShapes(
                    shapes=[[1, count] + shape[2:] for count in range(1, args.max_points + 1)],
                    default=shape)
            else:
                shape[1] = ct.RangeDim(lower_bound=1, upper_bound=args.max_points, default=2)
        inputs.append(ct.TensorType(name=name, shape=shape, dtype=np.float32))
    float32_ops: list[str] = []
    scope_samples: list[str] = []
    if args.precision == "float32":
        precision = ct.precision.FLOAT32
    elif args.precision == "float16":
        precision = ct.precision.FLOAT16
    else:
        precision = ct.transform.FP16ComputePrecision(
            op_selector=layer0_attention_selector(
                float32_ops, scope_samples, tuple(args.float32_op_types),
                tuple(args.float32_scopes) + PRECISION_SCOPES.get(args.precision, ())))
    started = time.perf_counter()
    mlmodel = ct.convert(traced, source="pytorch", convert_to="mlprogram", inputs=inputs,
                         outputs=[ct.TensorType(name=name) for name in output_names],
                         compute_precision=precision,
                         minimum_deployment_target=ct.target.macOS15)
    convert_seconds = time.perf_counter() - started
    if args.precision not in ("float32", "float16") and not float32_ops:
        raise RuntimeError("no layer 0 attention op was found for float32; module scopes of "
                           f"the first attention ops: {scope_samples}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = out_dir / f"{args.model}.mlpackage"
    compiled = out_dir / f"{args.model}.mlmodelc"
    # compile_model puts its output inside an existing directory: replace the earlier outputs.
    for earlier in (package, compiled):
        if earlier.exists():
            shutil.rmtree(earlier)
    mlmodel.save(str(package))
    compiled = Path(ct.models.utils.compile_model(str(package), str(compiled)))
    compiled_sha, compiled_bytes = directory_sha256(compiled)
    package_sha, package_bytes = directory_sha256(package)

    differences = {}
    feed = {name: tensor.numpy() for name, tensor in example.items()}
    for units in ("CPU_ONLY", "ALL"):
        model = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit,
                                                                               units))
        outputs = model.predict(feed)
        differences[units] = {name: float(np.abs(np.asarray(outputs[name], np.float32)
                                                  - value).max())
                              for name, value in zip(output_names, reference)}
        del model
    return {
        "step": "issue #8: native Core ML model", "model": args.model, "precision": args.precision,
        "float32_ops": float32_ops, "float32_op_types": list(args.float32_op_types),
        "float32_scopes": list(args.float32_scopes),
        "precision_scopes": list(PRECISION_SCOPES.get(args.precision, ())),
        "point_shapes": args.point_shapes if args.model == "point_decoder" else None,
        "coremltools": ct.__version__, "torch": torch.__version__,
        "format": "mlprogram", "minimum_deployment_target": "macOS15",
        "inputs": {name: [str(v) for v in i.shape.shape] if hasattr(i.shape, "shape") else
                   str(i.shape) for name, i in zip(example, inputs)},
        "outputs": output_names,
        "checkpoint": {"path": str(checkpoint), "bytes": Path(checkpoint).stat().st_size,
                       "sha256": sha256_of(Path(checkpoint))},
        "mlpackage": {"path": package.as_posix(), "bytes": package_bytes, "sha256": package_sha},
        "mlmodelc": {"path": compiled.as_posix(), "bytes": compiled_bytes,
                     "sha256": compiled_sha},
        "convert_seconds": round(convert_seconds, 1),
        "max_abs_difference_against_pytorch_float32": differences,
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--sam2-source", default="build/tmp/explore/src/sam2")
    parser.add_argument("--sam2-checkpoint",
                        default="build/tmp/explore/weights/sam2.1_hiera_tiny.pt")
    parser.add_argument("--head-checkpoint",
                        default="build/tmp/explore/p1/colab_run2_coco_openimages/run/best.pt")
    parser.add_argument("--head-epoch", type=int, default=15)
    parser.add_argument("--sky-head-checkpoint",
                        default="build/tmp/explore/sky_head/run1/train/best.pt")
    parser.add_argument("--sky-head-epoch", type=int, default=15)
    parser.add_argument("--s3od-source", default="build/tmp/explore/src/s3od/s3od")
    parser.add_argument("--s3od-checkpoint",
                        default="build/tmp/explore/weights/s3od_synth/s3od.pt")
    parser.add_argument("--max-points", type=int, default=64,
                        help="point_decoder: the largest number of prompt points")
    parser.add_argument("--point-shapes", choices=("range", "enumerated"), default="range",
                        help="point_decoder: a range of point counts (RangeDim) or one "
                             "enumerated shape per point count (EnumeratedShapes)")
    parser.add_argument("--float32-op-types", nargs="*", default=[],
                        help="precision float16_layer0_float32: MIL op types that stay float32 "
                             "in all layers (for example layer_norm)")
    parser.add_argument("--float32-scopes", nargs="*", default=[],
                        help="precision float16_layer0_float32: regular expressions of TorchScript "
                             "module paths whose ops stay float32 (for example inner\\.seg_head)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = convert(args)
    path = Path(args.out_dir) / f"{args.model}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("model", "precision", "convert_seconds",
                                             "max_abs_difference_against_pytorch_float32")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
