#!/usr/bin/env python3
"""Convert the Alibaba universal matting graph to ONNX in float32 and float16 (issue #6).

Input:  --graph (`tf_graph.pb` of ModelScope `iic/cv_unet_universal-matting`), --revisions (JSON
        with the ModelScope revision and the file hashes), the calibration photos (the rewrite
        check and the float16 range check).
Output: <out-root>/tf_graph_rewritten.pb, <out-root>/float32/universal_matting.onnx,
        <out-root>/float16/universal_matting.onnx, and an export_report.json in each directory.

Graph: `input_image` [H, W, 3] float32 (the photo, values 0 to 255, any size) -> `rgba`
[H, W, 4] float32 (`cond_3/Merge:0`, the RGBA before the uint8 Cast of `output_png`). The size
rules are inside the graph (universal_matting_graphs.GRAPH_RULES); no input size is chosen here.

Steps:

1. `universal_matting_graphs.rewrite_for_onnx` (constant cond, `ResizeArea`, `ResizeBilinear`,
   `CropAndResize`). Check `RewrittenGraphMatchesTensorflowGraph`: the rewritten graph and the
   original graph, both in TensorFlow with the grappler meta optimizer off (it changes the float
   order of the MatMul form), on the calibration photos; requirement: the same RGBA output
   (maximum absolute difference <= 1e-6) and the same branch tensors.
2. Float32: `universal_matting_graphs.convert_to_onnx` (tf2onnx without grappler), opset 17,
   then `float64_scalar_divisions`: each Div outside the U-Nets with a float32 output of one
   element (the scales of the crop and resize rewrites, the box coordinates, the alpha share) is
   Cast to float64, Div, Cast to float32. This gives the correctly rounded float32 quotient, the
   value of the TensorFlow kernels and of ONNX Runtime CPU (the CPU output does not change).
   DirectML float32 Div is not correctly rounded: it gives 4095 / 1919 = 2.13392377 (one ulp
   low), so the source row of the last crop row is 4094.99976 instead of 4095.00024; that row
   then reads the photo instead of the extrapolation value 0, and the alpha of DSC00674 changes
   (IoU 0.936 against TensorFlow). DirectML has no float64 Div, so ONNX Runtime runs these
   one-element nodes on the CPU. The Div nodes of whole images and maps stay float32 on
   DirectML. `--scalar-divisions float32` keeps the first graph for the check.
3. Float16: `convert_regions_to_float16`. Only nodes of the six U-Nets are float16 (`NETWORKS`:
   the region of a U-Net is the backward closure from its last Tanh through the nodes of its name
   scope and the tf2onnx helper nodes: names with no "/" and a "__<number>" suffix, operators of
   `FLOAT16_OPERATORS`; an If node of the glue stops the closure). The glue (pixel coordinates up to
   4096, crops, resizes, the box rule, the output) stays float32: float16 has a step of 2 to 4 at
   4096. In a U-Net, a node is float16 only when its output and each of its inputs have a largest
   absolute value on the calibration photos of at most 65,504 / 16 = 4,094 (the S3OD float16
   limit). The value of a tensor is that of its TensorFlow node; a helper node has the largest
   value of its inputs; an initializer counts as 0 (its float16 copy must be in the float16
   range); the graph input and the glue tensors have no range, so a node that reads one stays
   float32. The input rule is necessary: with the output rule only, the Sub of an instance norm
   (x - mean, small) read a float16 Cast of a large float32 Conv output, and the alpha changed by
   up to 1.0 (IoU 0.70 on DSC_1188). A float16 node reads a float16
   copy of a float32 initializer, or a Cast of a float32 tensor; a float32 node (or a branch
   output) reads a Cast of a float16 tensor. The graph input and output stay float32. ONNX Runtime
   CPU runs many float16 operators in float32, so check the float16 graph on DirectML.

   `onnxruntime.transformers.float16.convert_float_to_float16` (with a node block list) is not
   used: on this graph it changes the If branch outputs to float16 and writes their Cast into the
   parent graph ("Graph output ... does not exist in the graph"), and it writes the same Cast
   output name in two branches (not SSA).

The tf2onnx optimizers merge duplicated nodes in set order, so two exports can differ by a few
Reshape and Sub nodes (3,272 to 3,298 nodes measured before step 2); three conversions gave
bitwise equal outputs on 3 photos. The reports identify each graph by its SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import universal_matting_graphs as graphs  # noqa: E402

FLOAT16_MAX = 65504.0
ACTIVATION_LIMIT = FLOAT16_MAX / 16
REWRITE_TOLERANCE = 1e-6
# U-Net name scope -> its last node (ONNX node name; the TensorFlow name of the Tanh).
NETWORKS = {
    "detect/": "detect/logits/Tanh",
    "cond_1/coarse/": "cond_1/coarse/logits/Tanh",
    "cond_1/refine_human/": "cond_1/refine_human/Tanh",
    "cond_3/object_stage0/": "cond_3/object_stage0/logits/Tanh",
    "cond_3/object_stage1/": "cond_3/object_stage1/logits/Tanh",
    "cond_3/refine_object/": "cond_3/refine_object/Tanh",
}
HELPER_SUFFIX = re.compile(r"__\d+$")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def all_graphs(graph):
    """The graph and its subgraphs (If branches), depth first."""
    import onnx

    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from all_graphs(attribute.g)


def describe_onnx(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path))

    def shape_of(value_info):
        return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]

    def type_of(value_info):
        return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type).lower()

    operators: dict[str, int] = {}
    nodes = 0
    for graph in all_graphs(model.graph):
        for node in graph.node:
            operators[node.op_type] = operators.get(node.op_type, 0) + 1
            nodes += 1
    return {
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version}
                  for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": shape_of(i), "type": type_of(i)}
                   for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": shape_of(o), "type": type_of(o)}
                    for o in model.graph.output],
        "nodes_main_graph": len(model.graph.node),
        "nodes_with_subgraphs": nodes,
        "operators": dict(sorted(operators.items(), key=lambda item: -item[1])),
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}],
    }


def check_rewrite(original, rewritten, photos: list[Path], threads: int) -> dict:
    """`RewrittenGraphMatchesTensorflowGraph` on the photos (grappler off in both sessions)."""
    fetches = [graphs.TF_OUTPUT_RGBA, *graphs.TF_BRANCH_TENSORS.values()]
    sessions = [graphs.tf_session(g, threads, graph_optimizations=False)
                for g in (original, rewritten)]
    records = []
    for photo in photos:
        feed = {graphs.TF_INPUT: graphs.model_input(graphs.load_rgb(photo))}
        before, after = (session.run(fetches, feed) for session in sessions)
        records.append({
            "image": photo.name,
            "max_abs_rgba_diff": float(np.abs(before[0] - after[0]).max()),
            "same_branch_tensors": all(np.array_equal(a, b) for a, b in zip(before[1:], after[1:])),
        })
        print(f"[rewrite] {photo.name}: max|drgba| {records[-1]['max_abs_rgba_diff']:.2e} "
              f"branch tensors equal {records[-1]['same_branch_tensors']}", flush=True)
    passed = all(r["max_abs_rgba_diff"] <= REWRITE_TOLERANCE and r["same_branch_tensors"]
                 for r in records)
    return {"check": "RewrittenGraphMatchesTensorflowGraph", "tolerance": REWRITE_TOLERANCE,
            "grappler": "off in both sessions", "passed": passed,
            "max_abs_rgba_diff": max(r["max_abs_rgba_diff"] for r in records),
            "results": records}


def activation_maxima(graph_def, photos: list[Path], threads: int) -> tuple[dict, dict]:
    """Largest absolute float32 output of each TensorFlow node in the U-Net scopes (original
    graph, default session) over the photos, and the number of photos per U-Net (a branch runs on
    some photos only). Nodes without a float32 activation output (shapes, constants, the
    Identity and Switch nodes of the weights) get 0.0."""
    import tensorflow as tf

    tf1 = tf.compat.v1
    graph = tf.Graph()
    maxima: dict[str, float] = {}
    with graph.as_default():
        tf1.import_graph_def(graph_def, name="")
        probes: dict[str, dict] = {scope: {} for scope in NETWORKS}
        for operation in graph.get_operations():
            scope = next((s for s in NETWORKS if operation.name.startswith(s)), None)
            if scope is None:
                continue
            if (operation.type in ("Const", "Identity", "Switch") or not operation.outputs
                    or operation.outputs[0].dtype != tf.float32):
                maxima[operation.name] = 0.0
                continue
            probes[scope][operation.name] = tf.reduce_max(tf.abs(operation.outputs[0]))
    config = tf1.ConfigProto(intra_op_parallelism_threads=threads, inter_op_parallelism_threads=1,
                             device_count={"GPU": 0})
    session = tf1.Session(graph=graph, config=config)
    branch = {"cond_1/": "Greater_6:0", "cond_3/": "Greater_8:0"}
    photos_per_network = {scope: 0 for scope in NETWORKS}
    for photo in photos:
        feed = {graphs.TF_INPUT: graphs.model_input(graphs.load_rgb(photo))}
        human_ran, human_result = session.run([branch["cond_1/"], branch["cond_3/"]], feed)
        for scope, tensors in probes.items():
            if scope.startswith("cond_1/") and not human_ran:
                continue
            if scope.startswith("cond_3/") and human_result:
                continue
            photos_per_network[scope] += 1
            for name, value in session.run(tensors, feed).items():
                maxima[name] = max(maxima.get(name, 0.0), float(value))
        print(f"[ranges] {photo.name}: human branch {bool(human_ran)}, human result "
              f"{bool(human_result)}", flush=True)
    return maxima, photos_per_network


def float16_nodes(model, maxima: dict[str, float]) -> tuple[set[str], dict]:
    """Names of the nodes that become float16 (module docstring, step 3) and a record per U-Net."""
    producers = {}
    nodes = []
    for graph in all_graphs(model.graph):
        for node in graph.node:
            nodes.append(node)
            for output in node.output:
                producers[output] = node
    by_name = {node.name: node for node in nodes}
    initializer_names = {tensor.name for graph in all_graphs(model.graph)
                         for tensor in graph.initializer}
    allowed: set[str] = set()
    record: dict[str, dict] = {}
    for scope, last in NETWORKS.items():
        if last not in by_name:
            raise ValueError(f"no ONNX node {last}")

        def in_scope(node):
            return node.name.startswith(scope) or (
                "/" not in node.name and HELPER_SUFFIX.search(node.name) is not None
                and node.op_type in FLOAT16_OPERATORS)

        region, pending = [], [by_name[last]]
        seen = set()
        while pending:
            node = pending.pop()
            if node.name in seen or not in_scope(node):
                continue
            seen.add(node.name)
            region.append(node)
            pending.extend(producers[i] for i in node.input if i in producers)
        largest: dict[str, float | None] = {}

        def tensor_max(name: str) -> float | None:
            """Largest absolute value of a tensor on the calibration photos; None if unknown (the
            graph input and the glue outputs)."""
            if name in largest:
                return largest[name]
            producer = producers.get(name)
            if producer is None:
                value = 0.0 if name in initializer_names else None
            elif producer.name in seen and "/" not in producer.name:
                inputs = [tensor_max(i) for i in producer.input if i]
                value = None if None in inputs else max(inputs, default=0.0)
            else:
                base = HELPER_SUFFIX.sub("", producer.name)
                if producer.name in seen and base not in maxima:
                    raise ValueError(f"{producer.name}: no TensorFlow maximum for {base}")
                value = maxima.get(base)
            largest[name] = value
            return value

        decision: dict[str, bool] = {}
        for node in region:
            values = [tensor_max(i) for i in node.input if i] + [tensor_max(node.output[0])]
            decision[node.name] = all(v is not None and v <= ACTIVATION_LIMIT for v in values)

        names = {name for name, ok in decision.items() if ok}
        allowed.update(names)
        above = {node.name: round(largest[node.output[0]], 1) for node in region
                 if not decision[node.name] and largest[node.output[0]] is not None
                 and largest[node.output[0]] > ACTIVATION_LIMIT}
        unknown_inputs = sorted({i for node in region for i in node.input
                                 if i and largest.get(i, 0.0) is None})
        operators: dict[str, int] = {}
        for node in region:
            if node.name in names:
                operators[node.op_type] = operators.get(node.op_type, 0) + 1
        record[scope] = {"region_nodes": len(region), "float16_nodes": len(names),
                         "float32_nodes": len(region) - len(names),
                         "float16_operators": dict(sorted(operators.items())),
                         "inputs_without_range": unknown_inputs,
                         "largest_float32_nodes": dict(sorted(above.items(),
                                                              key=lambda kv: -kv[1])[:12]),
                         "largest_activation": round(max(maxima[HELPER_SUFFIX.sub("", n.name)]
                                                         for n in region
                                                         if HELPER_SUFFIX.sub("", n.name)
                                                         in maxima), 1)}
    return allowed, record


# Operators that can be float16 in a region: their float inputs and outputs have one type and
# their attributes have no data type.
FLOAT16_OPERATORS = {"Add", "Sub", "Mul", "Div", "Sqrt", "Relu", "LeakyRelu", "Tanh", "Conv",
                     "ConvTranspose", "MaxPool", "GlobalAveragePool", "Concat", "Reshape",
                     "Transpose", "Squeeze", "Unsqueeze", "Identity", "Split", "Slice"}


def convert_regions_to_float16(model, allowed: set[str]) -> dict:
    """Make the `allowed` nodes float16 (module docstring, step 3). Changes `model` in place and
    returns the counts of converted nodes, Cast nodes, and float16 initializers."""
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    float_tensors: set[str] = set()
    for graph in all_graphs(inferred.graph):
        for value in list(graph.value_info) + list(graph.input) + list(graph.output):
            if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                float_tensors.add(value.name)
        for tensor in graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                float_tensors.add(tensor.name)
    initializers = {}
    for graph in all_graphs(model.graph):
        for tensor in graph.initializer:
            initializers[tensor.name] = tensor
    found: set[str] = set()
    counts = {"float16_nodes": 0, "casts_to_float16": 0, "casts_to_float32": 0,
              "float16_initializers": 0}

    def convert(graph):
        nodes = list(graph.node)
        produced_by_allowed = {o for node in nodes if node.name in allowed for o in node.output}
        produced_here = {o for node in nodes for o in node.output}
        to_float16: dict[str, str] = {}
        new_nodes, new_initializers = [], []
        float16_outputs: set[str] = set()
        for node in nodes:
            if node.name not in allowed:
                continue
            if node.op_type not in FLOAT16_OPERATORS:
                raise ValueError(f"{node.name}: operator {node.op_type} is not handled")
            found.add(node.name)
            counts["float16_nodes"] += 1
            for index, name in enumerate(node.input):
                if not name or name not in float_tensors or name in produced_by_allowed:
                    continue
                if name not in to_float16:
                    converted_name = name + "__float16"
                    if name in initializers and name not in produced_here:
                        values = numpy_helper.to_array(initializers[name])
                        if np.abs(values).max(initial=0.0) > FLOAT16_MAX:
                            raise ValueError(f"initializer {name} is out of the float16 range")
                        new_initializers.append(numpy_helper.from_array(
                            values.astype(np.float16), converted_name))
                        counts["float16_initializers"] += 1
                    else:
                        new_nodes.append(helper.make_node(
                            "Cast", [name], [converted_name], name=name + "__to_float16",
                            to=onnx.TensorProto.FLOAT16))
                        counts["casts_to_float16"] += 1
                    to_float16[name] = converted_name
                node.input[index] = to_float16[name]
            float16_outputs.update(o for o in node.output if o in float_tensors)
        # Float32 readers of float16 outputs: nodes of this graph that are not allowed, nodes of
        # the subgraphs of this graph (outer-scope names), and the outputs of this graph.
        to_float32: dict[str, str] = {}

        def float32_name(name):
            if name not in to_float32:
                to_float32[name] = name + "__float32"
                new_nodes.append(helper.make_node("Cast", [name], [to_float32[name]],
                                                  name=name + "__to_float32",
                                                  to=onnx.TensorProto.FLOAT))
                counts["casts_to_float32"] += 1
            return to_float32[name]

        def rewire_subgraph(subgraph):
            for inner in subgraph.node:
                for index, name in enumerate(inner.input):
                    if name in float16_outputs:
                        inner.input[index] = float32_name(name)
                for attribute in inner.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        rewire_subgraph(attribute.g)

        for node in nodes:
            if node.name in allowed:
                continue
            for index, name in enumerate(node.input):
                if name in float16_outputs:
                    node.input[index] = float32_name(name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    rewire_subgraph(attribute.g)
        for output in graph.output:
            if output.name in float16_outputs:
                identity = helper.make_node("Identity", [float32_name(output.name)],
                                            [output.name + "__output"],
                                            name=output.name + "__output")
                new_nodes.append(identity)
                output.name = identity.output[0]
        graph.node.extend(new_nodes)
        graph.initializer.extend(new_initializers)
        del graph.value_info[:]
        for node in nodes:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    convert(attribute.g)

    convert(model.graph)
    if found != allowed:
        raise ValueError(f"allowed nodes not found: {sorted(allowed - found)[:5]}")
    # Remove the float32 initializers that only float16 nodes read before the conversion.
    read = {name for graph in all_graphs(model.graph) for node in graph.node
            for name in node.input}
    read.update(output.name for graph in all_graphs(model.graph) for output in graph.output)
    counts["removed_initializers"] = 0
    for graph in all_graphs(model.graph):
        kept = [tensor for tensor in graph.initializer if tensor.name in read]
        counts["removed_initializers"] += len(graph.initializer) - len(kept)
        del graph.initializer[:]
        graph.initializer.extend(kept)
    return counts


def float64_scalar_divisions(model) -> list[str]:
    """Compute each scalar float32 Div outside the U-Nets in float64 (module docstring, step 2).
    Changes `model` in place and returns the names of the changed Div nodes."""
    import onnx
    from onnx import helper, shape_inference

    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    scalar_float: set[str] = set()
    for graph in all_graphs(inferred.graph):
        for value in list(graph.value_info) + list(graph.output):
            tensor = value.type.tensor_type
            dims = tensor.shape.dim
            if (tensor.elem_type == onnx.TensorProto.FLOAT and tensor.HasField("shape")
                    and all(d.HasField("dim_value") and d.dim_value == 1 for d in dims)):
                scalar_float.add(value.name)
    changed: list[str] = []
    for graph in all_graphs(model.graph):
        nodes = []
        for node in graph.node:
            if (node.op_type != "Div" or node.output[0] not in scalar_float
                    or any(node.name.startswith(scope) for scope in NETWORKS)):
                nodes.append(node)
                continue
            inputs = []
            for index, name in enumerate(node.input):
                inputs.append(f"{node.output[0]}__input{index}_float64")
                nodes.append(helper.make_node("Cast", [name], [inputs[-1]],
                                              name=f"{node.name}__input{index}_to_float64",
                                              to=onnx.TensorProto.DOUBLE))
            quotient = node.output[0] + "__float64"
            nodes.append(helper.make_node("Div", inputs, [quotient], name=node.name + "__float64"))
            nodes.append(helper.make_node("Cast", [quotient], [node.output[0]],
                                          name=node.name + "__to_float32",
                                          to=onnx.TensorProto.FLOAT))
            changed.append(node.name)
        del graph.node[:]
        graph.node.extend(nodes)
    return changed


def sort_topologically(graph) -> None:
    """Order the nodes of `graph` and of its subgraphs so that each node follows the producers of
    its inputs (the float16 converter appends its Cast nodes at the end of the main graph; the ONNX
    checker requires the order). Tensors that no node of the graph produces (inputs,
    initializers, names of an enclosing graph) are available from the start."""
    import onnx

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                sort_topologically(attribute.g)

    def inputs_of(node, acc):
        acc.update(i for i in node.input if i)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                for inner in attribute.g.node:
                    inputs_of(inner, acc)
        return acc

    produced = {o: index for index, node in enumerate(graph.node) for o in node.output if o}
    nodes = list(graph.node)
    waiting = [{produced[i] for i in inputs_of(node, set()) if i in produced} - {index}
               for index, node in enumerate(nodes)]
    consumers: dict[int, list[int]] = {}
    for index, sources in enumerate(waiting):
        for source in sources:
            consumers.setdefault(source, []).append(index)
    ready = [index for index, sources in enumerate(waiting) if not sources]
    order = []
    while ready:
        index = ready.pop(0)
        order.append(index)
        for consumer in consumers.get(index, []):
            waiting[consumer].discard(index)
            if not waiting[consumer]:
                ready.append(consumer)
    if len(order) != len(nodes):
        raise ValueError(f"graph {graph.name}: cycle or missing producer")
    del graph.node[:]
    graph.node.extend(nodes[index] for index in order)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", required=True, help="tf_graph.pb")
    parser.add_argument("--revisions", required=True, help="JSON with the ModelScope revision")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration-photos", required=True)
    parser.add_argument("--calibration-prompts", required=True,
                        help="the quality set; only the photo list is used")
    parser.add_argument("--scalar-divisions", choices=("float64", "float32"), default="float64",
                        help="float32: keep the scalar Div nodes in float32 (the first graph of "
                             "the report, for the DirectML division check)")
    args = parser.parse_args()

    import onnx
    import onnxruntime
    import tensorflow as tf
    import tf2onnx

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    source = Path(args.graph)
    photos = [Path(args.calibration_photos) / entry["file"] for entry in
              json.loads(Path(args.calibration_prompts).read_text(encoding="utf-8"))["images"]]
    original = graphs.read_graph_def(source)
    started = time.perf_counter()
    rewritten = graphs.rewrite_for_onnx(original)
    rewrite_seconds = time.perf_counter() - started
    rewritten_path = out_root / "tf_graph_rewritten.pb"
    rewritten_path.write_bytes(rewritten.SerializeToString(deterministic=True))
    common = {
        "model": "Alibaba universal matting (ModelScope iic/cv_unet_universal-matting)",
        "revisions": json.loads(Path(args.revisions).read_text(encoding="utf-8")),
        "source_graph": {"name": source.name, "bytes": source.stat().st_size,
                         "sha256": sha256_of(source),
                         "operators": graphs.operator_counts(original)},
        "rewritten_graph": {"name": rewritten_path.name, "bytes": rewritten_path.stat().st_size,
                            "sha256": sha256_of(rewritten_path),
                            "operators": graphs.operator_counts(rewritten),
                            "rewrite_seconds": round(rewrite_seconds, 1)},
        "graph_rules": graphs.GRAPH_RULES,
        "model_input": "input_image [H, W, 3] float32, RGB 0 to 255 (LoadImage of ModelScope)",
        "model_output": "rgba [H, W, 4] float32 (cond_3/Merge:0); alpha = channel 3 / 255",
        "tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__, "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }
    rewrite_check = check_rewrite(original, rewritten, photos, args.threads)
    common["rewrite_check"] = rewrite_check
    (out_root / "rewrite_check.json").write_text(json.dumps(rewrite_check, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    if not rewrite_check["passed"]:
        print("[export] RewrittenGraphMatchesTensorflowGraph failed", file=sys.stderr)
        return 1

    float32_dir = out_root / "float32"
    float32_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = graphs.convert_to_onnx(rewritten, args.opset)
    divisions = (float64_scalar_divisions(model) if args.scalar_divisions == "float64" else [])
    float32_path = float32_dir / graphs.ONNX_NAME
    onnx.save(model, str(float32_path))
    onnx.checker.check_model(str(float32_path), full_check=True)
    report32 = {**common, "converter": "tf2onnx.tfonnx.process_tf_graph and "
                                      "tf2onnx.optimizer.optimize_graph, without grappler",
                "opset": args.opset, "convert_seconds": round(time.perf_counter() - started, 1),
                "scalar_divisions": args.scalar_divisions,
                "float64_divisions": divisions,
                "graph": describe_onnx(float32_path)}
    (float32_dir / "export_report.json").write_text(json.dumps(report32, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(f"[export] float32: {report32['graph']['nodes_with_subgraphs']} nodes, "
          f"{report32['graph']['files'][0]['bytes'] / 1e6:.1f} MB, {len(divisions)} float64 "
          f"scalar divisions", flush=True)

    maxima, photos_per_network = activation_maxima(original, photos, args.threads)
    allowed, networks = float16_nodes(model, maxima)
    converted = onnx.load(str(float32_path))
    counts = convert_regions_to_float16(converted, allowed)
    sort_topologically(converted.graph)
    float16_dir = out_root / "float16"
    float16_dir.mkdir(parents=True, exist_ok=True)
    float16_path = float16_dir / graphs.ONNX_NAME
    onnx.save(converted, str(float16_path))
    onnx.checker.check_model(str(float16_path), full_check=True)
    report16 = {**common, "float16": {
        "source": float32_path.as_posix(),
        "converter": "convert_regions_to_float16", "conversion": counts,
        "activation_limit": ACTIVATION_LIMIT,
        "activation_photos": [photo.name for photo in photos],
        "photos_per_network": photos_per_network, "networks": networks},
        "graph": describe_onnx(float16_path)}
    (float16_dir / "export_report.json").write_text(json.dumps(report16, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(f"[export] float16: {report16['graph']['nodes_with_subgraphs']} nodes, "
          f"{report16['graph']['files'][0]['bytes'] / 1e6:.1f} MB; float16 nodes per U-Net "
          + ", ".join(f"{s}{v['float16_nodes']}/{v['region_nodes']}"
                      for s, v in networks.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
