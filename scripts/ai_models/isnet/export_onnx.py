#!/usr/bin/env python3
"""Export the IS-Net general-use model to ONNX in float32 and float16 (issue #6).

Input:  --source (the `IS-Net` directory of the xuebinqin/DIS checkout), --checkpoint
        (`isnet-general-use.pth`), --revisions (JSON with the DIS commit and the weights URL),
        the calibration photos of the float16 range check.
Output: <out-root>/float32/subject_d1.onnx, <out-root>/float16/subject_d1.onnx, and an
        export_report.json in each directory (source revisions, file sizes and SHA-256 values,
        parameter count, opset, inputs and outputs, operator counts).

Graph: `pixel_values` [1, 3, 1024, 1024] float32 -> `d1` [1, 1, 1024, 1024] float32, the sigmoid
of side output d1 (isnet_graphs.subject_d1_module). The input size is the size of the source
(`Inference.py`, input_size [1024, 1024]). The other five side outputs are not graph outputs; the
exporter removes their nodes.

Exporter: the TorchScript exporter (`torch.onnx.export(..., dynamo=False)`), opset 17, as the
BiRefNet export of E2. The model has plain convolutions, max pooling, and bilinear upsampling.

Float16: `onnxruntime.transformers.float16.convert_float_to_float16` with `keep_io_types=True`
(float32 input and output, as the BiRefNet float16 graph of E2) and the default operator block
list of the converter. Before the conversion the script records the largest absolute output of
each PyTorch module (float32) on the calibration photos (`activation_maxima`) and the modules
above 1/16 of the float16 maximum (65,504 / 16 = 4,094), the limit of the S3OD float16 rule. The
convolution of each REBNCONV block (`conv_s1`) is not a graph tensor: the exporter folds the
BatchNorm after it into its weights, so the graph tensor is the `bn_s1` output. The check uses
the graph tensors and records the folded convolutions separately. ONNX Runtime CPU runs most
float16 operators in float32, so check the float16 graph on DirectML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isnet_graphs as graphs  # noqa: E402

FLOAT16_MAX = 65504.0
ACTIVATION_LIMIT = FLOAT16_MAX / 16


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
    return {
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version}
                  for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": shape_of(i), "type": type_of(i)}
                   for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": shape_of(o), "type": type_of(o)}
                    for o in model.graph.output],
        "nodes": len(model.graph.node),
        "operators": dict(sorted(operator_counts.items(), key=lambda item: -item[1])),
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}],
    }


def export_float32(args, model, out_dir: Path) -> dict:
    import torch

    wrapper = graphs.subject_d1_module(model)
    pixels = torch.zeros(1, 3, graphs.IMAGE_SIZE, graphs.IMAGE_SIZE)
    report: dict = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_in_graph": None,
        "torch": torch.__version__,
        "exporter": "torch.onnx.export(dynamo=False)",
        "opset": args.opset,
    }
    path = out_dir / graphs.GRAPH_NAME
    started = time.perf_counter()
    try:
        torch.onnx.export(wrapper, (pixels,), str(path), input_names=[graphs.INPUT_NAME],
                          output_names=[graphs.OUTPUT_NAME], opset_version=args.opset,
                          dynamo=False)
    except Exception as error:  # noqa: BLE001 - the report records the real exporter error
        report["error"] = "".join(traceback.format_exception(error))[-4000:]
        return report
    report["export_seconds"] = round(time.perf_counter() - started, 1)
    import onnx
    from onnx import numpy_helper

    exported = onnx.load(str(path))
    report["parameters_in_graph"] = int(sum(numpy_helper.to_array(t).size
                                            for t in exported.graph.initializer))
    report["graph"] = describe_onnx(path)
    return report


def activation_maxima(model, photos: list[Path]) -> dict[str, float]:
    """Largest absolute output of each leaf module and each RSU block (float32 PyTorch) over the
    photos."""
    import torch

    maxima: dict[str, float] = {}
    hooks = []
    for name, module in model.named_modules():
        if not name:
            continue

        def record(_module, _inputs, output, name=name):
            if isinstance(output, torch.Tensor):
                value = float(output.detach().abs().max())
                maxima[name] = max(maxima.get(name, 0.0), value)

        hooks.append(module.register_forward_hook(record))
    with torch.no_grad():
        for photo in photos:
            pixels = torch.from_numpy(graphs.preprocess(graphs.load_rgb(photo)))
            model(pixels)
    for hook in hooks:
        hook.remove()
    return maxima


def folded_convolutions(model) -> set[str]:
    """Names of the convolutions that the exporter folds with the next BatchNorm (REBNCONV)."""
    return {f"{name}.conv_s1" for name, module in model.named_modules()
            if hasattr(module, "conv_s1") and hasattr(module, "bn_s1")}


def convert_float16(source: Path, out_dir: Path, maxima: dict[str, float], folded: set[str],
                    photos: list[Path]) -> dict:
    import onnx
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    model = onnx.load(str(source))
    converted = convert_float_to_float16(model, keep_io_types=True)
    path = out_dir / graphs.GRAPH_NAME
    onnx.save(converted, str(path))
    largest = sorted(((n, v) for n, v in maxima.items() if n not in folded),
                     key=lambda item: -item[1])
    return {
        "source": source.as_posix(),
        "converter": "onnxruntime.transformers.float16.convert_float_to_float16",
        "keep_io_types": True,
        "op_block_list": sorted(DEFAULT_OP_BLOCK_LIST),
        "node_block_list": [],
        "activation_limit": ACTIVATION_LIMIT,
        "activation_photos": [photo.name for photo in photos],
        "activation_maximum": round(largest[0][1], 1),
        "modules_above_limit": {name: round(value, 1) for name, value in largest
                                if value > ACTIVATION_LIMIT},
        "largest_activations": {name: round(value, 1) for name, value in largest[:20]},
        "folded_convolutions": len(folded),
        "folded_convolution_maximum": round(max(maxima[name] for name in folded), 1),
        "graph": describe_onnx(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="IS-Net directory of the DIS checkout")
    parser.add_argument("--checkpoint", required=True, help="isnet-general-use.pth")
    parser.add_argument("--revisions", required=True,
                        help="JSON with the DIS commit and the weights source")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration-photos", required=True,
                        help="photos for the activation maxima of the float16 check")
    parser.add_argument("--calibration-prompts", required=True,
                        help="the quality set; only the photo list is used")
    args = parser.parse_args()

    import onnxruntime
    import torch

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    out_root = Path(args.out_root)
    checkpoint = Path(args.checkpoint)
    model = graphs.load_model(args.source, checkpoint)
    common = {
        "model": "IS-Net general-use",
        "revisions": json.loads(Path(args.revisions).read_text(encoding="utf-8")),
        "checkpoint": {"name": checkpoint.name, "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_of(checkpoint)},
        "model_input": {"size": graphs.IMAGE_SIZE,
                        "resize": "stretch to size x size, torch bilinear, align_corners=False, "
                                  "no antialias, then truncation to uint8",
                        "mean": list(graphs.NORMALIZE_MEAN), "std": list(graphs.NORMALIZE_STD)},
        "model_output": "d1: sigmoid of side output d1; the alpha is (d1 - min) / (max - min)",
        "onnxruntime": onnxruntime.__version__,
    }

    float32_dir = out_root / "float32"
    float32_dir.mkdir(parents=True, exist_ok=True)
    report32 = {**common, **export_float32(args, model, float32_dir)}
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
    maxima = activation_maxima(model, photos)
    report16 = {**common, "float16": convert_float16(float32_dir / graphs.GRAPH_NAME, float16_dir,
                                                     maxima, folded_convolutions(model),
                                                     photos)}
    (float16_dir / "export_report.json").write_text(json.dumps(report16, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(f"[export] float16: {report16['float16']['graph']['nodes']} nodes, "
          f"{report16['float16']['graph']['files'][0]['bytes'] / 1e6:.1f} MB; largest activation "
          f"{report16['float16']['activation_maximum']}, modules above "
          f"{ACTIVATION_LIMIT:.0f}: {len(report16['float16']['modules_above_limit'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
