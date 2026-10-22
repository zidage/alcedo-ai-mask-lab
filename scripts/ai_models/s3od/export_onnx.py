#!/usr/bin/env python3
"""Export the S3OD `s3od-synth` model to ONNX in float32 and float16 (issue #6).

Input:  --source (the `s3od/s3od` directory of the KupynOrest/s3od checkout), --checkpoint
        (`s3od.pt` of `okupyn/s3od-synth`), the calibration photos of the float16 rule.
Output: <out-root>/float32/subject_masks.onnx, <out-root>/float16/subject_masks.onnx, and an
        export_report.json in each directory (source revisions, file sizes and SHA-256 values,
        parameter count, opset, inputs and outputs, operator counts, exporter messages).

Graph: `pixel_values` [1, 3, 1024, 1024] float32 -> `masks` [1, 3, 1024, 1024] and `iou_scores`
[1, 3], both after the sigmoid (s3od_graphs.subject_masks_module). The input size is the size of
the source (`BackgroundRemoval(image_size=1024)`).

Exporter: `torch.onnx.export(..., dynamo=True)`, opset 18. The TorchScript exporter stops with
"Exporting the operator 'aten::_upsample_bilinear2d_aa' to ONNX opset version 18 is not supported"
(the `antialias=True` resize of `MultiMaskHead`). The dynamo exporter writes it as `Resize` with
`antialias=1` (opset 18).

Reshape `allowzero`: the dynamo exporter writes `Reshape` with `allowzero=1` for `Tensor.view`.
ONNX Runtime DirectML 1.24.4 refuses the session of that graph with `80070057` (E_INVALIDARG,
`MLOperatorAuthorImpl.cpp(2853)`); the first node that causes it is the first such Reshape (graph
cut bisection, report section 4). The script sets `allowzero=0` on each Reshape whose shape is a
constant without a 0. For such a shape both values give the same output (ONNX Reshape: the
attribute only changes the meaning of a 0 in the shape), so the computation does not change. A
Reshape with a 0 or a computed shape stops the export. `--keep-reshape-allowzero` writes the graph
of the exporter without this change (to reproduce the error).

Resize to the same size: `MultiMaskHead` resizes its 1024 x 1024 feature map to the target size
1024 x 1024 with `antialias=True`. The result is equal to the input (ONNX Runtime CPU and PyTorch
give a bitwise equal tensor, report section 3). DirectML has no `Resize` with `antialias=1`, so
ONNX Runtime runs that node on the CPU and copies the [1, 64, 1024, 1024] tensor to the CPU and
back in each call. The script removes each `Resize` whose constant `sizes` are equal to the
inferred shape of its input and connects its consumers to its input.
`--keep-identity-resize` keeps the node.

Float16: `onnxruntime.transformers.float16.convert_float_to_float16` with `keep_io_types=True`
(float32 inputs and outputs, as the BiRefNet float16 graph of E2), the default operator block
list of the converter, and a node block list per encoder layer: the attention nodes of a layer
(each score MatMul, the MatMul whose output goes to a Softmax, and its Softmax; or each
`com.microsoft.MultiHeadAttention`) stay in float32 when the largest absolute attention score of
the layer on the calibration photos (PyTorch float32, `s3od_graphs.attention_score_maxima`) is
above 1/16 of the float16 maximum (65,504 / 16 = 4,094). The nodes of a layer are found with the
`pkg.torch.onnx.name_scopes` metadata of the exporter. On the 14 quality photos only encoder
layer 0 is above it: 134,699 to 135,151; the other ten layers are at most 482 (report section
4.3). With layer 0 in float16 the DirectML output is NaN on every photo. ONNX Runtime CPU does
not show the overflow: it has no float16 kernel for many operators and runs them in float32. The
converter puts a Cast to float16 and a Cast back to float32 between two blocked nodes, which
overflows in the same way; the script removes each such pair. It also merges the Cast nodes of
one tensor that the converter writes for each blocked node with the same name.

Query chunked attention (`--query-chunks N`, `--attention-kernel`): each encoder attention is
computed in N query chunks, one after the other (s3od_graphs.use_query_chunked_attention). The
output is the same attention; the score tensor of one chunk is about 1/N of the
[1, 12, 4101, 4101] tensor. Kernel `sdpa` writes MatMul, Softmax, MatMul per chunk; kernel
`multi_head_attention` writes `com.microsoft.MultiHeadAttention` per chunk. The default
(`--query-chunks 1 --attention-kernel sdpa`) exports the attention of the source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s3od_graphs as graphs  # noqa: E402

GRAPH_NAME = "subject_masks.onnx"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_onnx(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def shape_of(value_info):
        return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]

    def type_of(value_info):
        return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type).lower()

    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    files = [path] + sorted(path.parent.glob(path.name + ".data"))
    return {
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version}
                  for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": shape_of(i), "type": type_of(i)}
                   for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": shape_of(o), "type": type_of(o)}
                    for o in model.graph.output],
        "nodes": len(model.graph.node),
        "operators": dict(sorted(operator_counts.items(), key=lambda item: -item[1])),
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256_of(f)}
                  for f in files],
    }


def clear_reshape_allowzero(path: Path) -> dict:
    """Set `allowzero=0` on each Reshape with a constant shape without a 0 (see the module
    docstring). Raises ValueError for a Reshape with `allowzero=1` and a 0 or a computed shape."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    constants = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    constants.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t)
                      for n in graph.node if n.op_type == "Constant"})
    changed = 0
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        for attribute in node.attribute:
            if attribute.name != "allowzero" or attribute.i == 0:
                continue
            shape = constants.get(node.input[1])
            if shape is None or (shape == 0).any():
                raise ValueError(f"{node.name}: allowzero=1 with the shape {node.input[1]} "
                                 f"({'computed' if shape is None else shape.tolist()})")
            attribute.i = 0
            changed += 1
    onnx.save(model, str(path))
    return {"reshape_allowzero_cleared": changed}


def remove_identity_resize(path: Path) -> dict:
    """Remove each Resize with constant `sizes` equal to the inferred shape of its input (see the
    module docstring). A Resize that is a graph output is kept."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None
                       for d in v.type.tensor_type.shape.dim]
              for v in list(inferred.graph.value_info) + list(inferred.graph.input)}
    graph = model.graph
    constants = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    constants.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t)
                      for n in graph.node if n.op_type == "Constant"})
    outputs = {o.name for o in graph.output}
    removed = []
    for node in list(graph.node):
        if node.op_type != "Resize" or len(node.input) < 4 or node.output[0] in outputs:
            continue
        sizes = constants.get(node.input[3])
        shape = shapes.get(node.input[0])
        if sizes is None or shape is None or None in shape or sizes.tolist() != shape:
            continue
        for consumer in graph.node:
            for index, name in enumerate(consumer.input):
                if name == node.output[0]:
                    consumer.input[index] = node.input[0]
        graph.node.remove(node)
        removed.append({"name": node.name, "sizes": sizes.tolist(),
                        "antialias": next((a.i for a in node.attribute
                                           if a.name == "antialias"), 0)})
    onnx.save(model, str(path))
    return {"identity_resize_removed": removed}


def export_float32(args, out_dir: Path) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    model = graphs.load_model(args.source, Path(args.checkpoint))
    export_settings: dict = {}
    if args.query_chunks != 1 or args.attention_kernel != "sdpa":
        export_settings = graphs.use_query_chunked_attention(model, args.query_chunks,
                                                             args.attention_kernel)
    wrapper = graphs.subject_masks_module(model)
    pixels = torch.zeros(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)
    report: dict = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "exporter": "torch.onnx.export(dynamo=True)",
        "opset_requested": args.opset,
        "attention": {"query_chunks": args.query_chunks, "kernel": args.attention_kernel,
                      "chunk_sizes": graphs.query_chunk_sizes(graphs.ENCODER_TOKENS,
                                                              args.query_chunks)},
    }
    path = out_dir / GRAPH_NAME
    started = time.perf_counter()
    try:
        program = torch.onnx.export(wrapper, (pixels,), input_names=["pixel_values"],
                                    output_names=["masks", "iou_scores"],
                                    opset_version=args.opset, dynamo=True, external_data=False,
                                    **export_settings)
        program.save(str(path))
    except Exception as error:  # noqa: BLE001 - the report records the real exporter error
        report["error"] = "".join(traceback.format_exception(error))[-4000:]
        return report
    report["export_seconds"] = round(time.perf_counter() - started, 1)
    if args.keep_reshape_allowzero:
        report["reshape_allowzero_cleared"] = 0
    else:
        report.update(clear_reshape_allowzero(path))
    if args.keep_identity_resize:
        report["identity_resize_removed"] = []
    else:
        report.update(remove_identity_resize(path))
    report["graph"] = describe_onnx(path)
    return report


FLOAT16_MAX = 65504.0
SCORE_LIMIT = FLOAT16_MAX / 16


def convert_float16(source: Path, out_dir: Path, maxima: list[float],
                   photos: list[Path]) -> dict:
    import onnx
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    model = onnx.load(str(source))
    nodes = attention_nodes_by_layer(model)
    if sorted(nodes) != list(range(len(nodes))) or len(nodes) > len(maxima):
        raise ValueError(f"attention nodes in layers {sorted(nodes)}, score maxima of "
                         f"{len(maxima)} layers")
    blocked = [name for layer, names in sorted(nodes.items()) if maxima[layer] > SCORE_LIMIT
               for name in names]
    converted = convert_float_to_float16(model, keep_io_types=True, node_block_list=blocked)
    merged = merge_duplicate_casts(converted)
    removed = remove_float16_round_trips(converted, set(blocked))
    path = out_dir / GRAPH_NAME
    onnx.save(converted, str(path))
    return {
        "source": source.as_posix(),
        "converter": "onnxruntime.transformers.float16.convert_float_to_float16",
        "keep_io_types": True,
        "op_block_list": sorted(DEFAULT_OP_BLOCK_LIST),
        "attention_score_limit": SCORE_LIMIT,
        "attention_score_photos": [photo.name for photo in photos],
        "attention_score_maxima_per_layer": [round(value, 1) for value in maxima],
        "attention_nodes_per_layer": {layer: len(names) for layer, names in sorted(nodes.items())},
        "node_block_list": blocked,
        "duplicate_casts_merged": merged,
        "float16_round_trips_removed": removed,
        "graph": describe_onnx(path),
    }


def encoder_layer_of(node) -> int | None:
    """The encoder layer index of a node, from the `pkg.torch.onnx.name_scopes` metadata."""
    for prop in node.metadata_props:
        if prop.key != "pkg.torch.onnx.name_scopes":
            continue
        for scope in ast.literal_eval(prop.value):
            parts = scope.split(".")
            if parts[:4] == ["inner", "encoder", "model", "layer"] and len(parts) >= 5:
                return int(parts[4])
    return None


def attention_nodes_by_layer(model) -> dict[int, list[str]]:
    """The attention nodes of each encoder layer: each score MatMul (a MatMul whose output goes
    to a Softmax) and its Softmax, and each `com.microsoft.MultiHeadAttention`."""
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    nodes: dict[int, list[str]] = {}
    softmax_count = 0
    for node in model.graph.node:
        names = []
        if node.op_type == "MatMul":
            softmax = [c for c in consumers.get(node.output[0], []) if c.op_type == "Softmax"]
            if softmax:
                names = [node.name, softmax[0].name]
                softmax_count += 1
        elif node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft":
            names = [node.name]
        if not names:
            continue
        layer = encoder_layer_of(node)
        if layer is None:
            raise ValueError(f"{node.name}: no encoder layer in the name scopes")
        nodes.setdefault(layer, []).extend(names)
    if softmax_count != sum(n.op_type == "Softmax" for n in model.graph.node):
        raise ValueError("not every Softmax has a score MatMul input")
    return nodes


def calibration_score_maxima(args, photos: list[Path]) -> list[float]:
    """`s3od_graphs.attention_score_maxima` of the source model on the calibration photos."""
    import torch

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    model = graphs.load_model(args.source, Path(args.checkpoint))
    return graphs.attention_score_maxima(
        model, (graphs.letterbox(graphs.load_rgb(photo))[0] for photo in photos))


def merge_duplicate_casts(model) -> int:
    """Merge the Cast nodes with the same input and the same target type into the first one.

    The converter writes one Cast per blocked node input, with a name and an output name made
    from the input name. Blocked nodes that read the same tensor (the key and the value of the
    query chunks of one layer) get Cast nodes with the same names, which is not a valid graph.
    The merged Cast nodes compute the same tensor."""
    graph = model.graph
    first: dict[tuple[str, int], object] = {}
    renamed: dict[str, str] = {}
    merged = 0
    for node in list(graph.node):
        if node.op_type != "Cast":
            continue
        key = (node.input[0], next(a.i for a in node.attribute if a.name == "to"))
        if key not in first:
            first[key] = node
            continue
        renamed[node.output[0]] = first[key].output[0]
        graph.node.remove(node)
        merged += 1
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in renamed:
                node.input[index] = renamed[name]
    return merged


def remove_float16_round_trips(model, blocked: set[str]) -> int:
    """Remove each Cast(float32) whose input is a Cast(float16) of a blocked node output, and
    connect the consumers to that float32 output. Cast nodes without a consumer are removed."""
    from onnx import TensorProto

    graph = model.graph
    producer = {name: node for node in graph.node for name in node.output}

    def cast_type(node) -> int:
        return next(a.i for a in node.attribute if a.name == "to")

    removed = 0
    for node in list(graph.node):
        if node.op_type != "Cast" or cast_type(node) != TensorProto.FLOAT:
            continue
        to_float16 = producer.get(node.input[0])
        if (to_float16 is None or to_float16.op_type != "Cast"
                or cast_type(to_float16) != TensorProto.FLOAT16):
            continue
        source = producer.get(to_float16.input[0])
        if source is None or source.name not in blocked:
            continue
        for consumer in graph.node:
            for index, name in enumerate(consumer.input):
                if name == node.output[0]:
                    consumer.input[index] = to_float16.input[0]
        graph.node.remove(node)
        removed += 1
    used = {name for node in graph.node for name in node.input} | {o.name for o in graph.output}
    for node in list(graph.node):
        if node.op_type == "Cast" and node.output[0] not in used:
            graph.node.remove(node)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="s3od/s3od of the KupynOrest/s3od checkout")
    parser.add_argument("--checkpoint", required=True, help="s3od.pt of okupyn/s3od-synth")
    parser.add_argument("--revisions", required=True,
                        help="JSON with the Hugging Face and GitHub revisions of the files")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-reshape-allowzero", action="store_true",
                        help="keep allowzero=1 of the exporter (DirectML refuses the graph)")
    parser.add_argument("--calibration-photos", required=True,
                        help="photos for the attention score maxima of the float16 rule")
    parser.add_argument("--calibration-prompts", required=True,
                        help="the quality set; only the photo list is used")
    parser.add_argument("--keep-identity-resize", action="store_true",
                        help="keep the Resize to the same size (DirectML runs it on the CPU)")
    parser.add_argument("--query-chunks", type=int, default=1,
                        help="encoder attention in this number of query chunks")
    parser.add_argument("--attention-kernel", choices=graphs.ATTENTION_KERNELS, default="sdpa",
                        help="attention of one chunk: MatMul-Softmax-MatMul or "
                             "com.microsoft.MultiHeadAttention")
    args = parser.parse_args()

    import onnxruntime

    out_root = Path(args.out_root)
    checkpoint = Path(args.checkpoint)
    common = {
        "model": "S3OD s3od-synth",
        "revisions": json.loads(Path(args.revisions).read_text(encoding="utf-8")),
        "checkpoint": {"name": checkpoint.name, "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_of(checkpoint)},
        "model_input": {"size": graphs.IMAGE_SIZE, "resize": "long edge to size, OpenCV "
                        "INTER_LINEAR, centered on a black size x size image",
                        "mean": list(graphs.NORMALIZE_MEAN), "std": list(graphs.NORMALIZE_STD)},
        "onnxruntime": onnxruntime.__version__,
    }

    float32_dir = out_root / "float32"
    float32_dir.mkdir(parents=True, exist_ok=True)
    report32 = {**common, **export_float32(args, float32_dir)}
    (float32_dir / "export_report.json").write_text(json.dumps(report32, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    if "error" in report32:
        print(f"[export] float32 failed:\n{report32['error']}", file=sys.stderr)
        return 1
    print(f"[export] float32: {report32['graph']['nodes']} nodes, "
          f"{report32['graph']['files'][0]['bytes'] / 1e6:.1f} MB", flush=True)

    float16_dir = out_root / "float16"
    float16_dir.mkdir(parents=True, exist_ok=True)
    photos = [Path(args.calibration_photos) / entry["file"] for entry in
              json.loads(Path(args.calibration_prompts).read_text(encoding="utf-8"))["images"]]
    maxima = calibration_score_maxima(args, photos)
    report16 = {**common, "float16": convert_float16(float32_dir / GRAPH_NAME, float16_dir,
                                                     maxima, photos)}
    (float16_dir / "export_report.json").write_text(json.dumps(report16, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(f"[export] float16: {report16['float16']['graph']['nodes']} nodes, "
          f"{report16['float16']['graph']['files'][0]['bytes'] / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
