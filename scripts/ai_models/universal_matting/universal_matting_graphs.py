"""Shared helpers of the Alibaba universal matting scripts (issue #6): photo reading, the TensorFlow
sessions, and the graph rewrites before the ONNX conversion.

Model: ModelScope `iic/cv_unet_universal-matting`, file `tf_graph.pb` (TensorFlow 1 frozen graph).
Pipeline of the source (`modelscope/pipelines/cv/image_matting_pipeline.py`,
`ImageMattingPipeline`):

- read the photo with Pillow, `ImageOps.exif_transpose`, `convert("RGB")`, `np.array(...)
  .astype(float)` (`LoadImage.convert_to_ndarray`), and feed it to `input_image:0` (float32
  [H, W, 3], values 0 to 255, any size);
- the output is `output_png:0` (uint8 [H, W, 4], RGBA); the pipeline converts it to BGRA for
  `cv2.imwrite`.

The graph does all resizing itself (`GRAPH_RULES`). The alpha of these scripts is channel 3 of
`cond_3/Merge:0`, the float32 RGBA tensor before the uint8 Cast of `output_png` (`alpha_from_rgba`:
value / 255). `output_png` is the truncation of that tensor to uint8.

Graph rewrites before `tf2onnx` (`rewrite_for_onnx`). Each one keeps the computation of the
TensorFlow CPU kernel (sources: `tensorflow/core/kernels/image/*.cc` at v2.21.0):

- `cond_2` is a `tf.cond` with a constant in each branch (1.0 if more than 50 person rows and
  columns, else 0.0). `tf2onnx` 1.17 cannot convert a cond whose branches have only constants.
  The rewrite replaces `cond_2/Merge` with `Cast(Greater_7, float32)`, the same value.
- `ResizeArea` (the photo to 192 x 192 for the person detector when the long edge is more than
  1920 px). `tf2onnx` 1.17 converts it to a nearest-neighbor `Resize`, which is a different
  operation. The rewrite writes the area average of `resize_area_op.cc` as two weight matrices and
  two MatMul operations (`resize_area_weights`).
- `ResizeBilinear` (8 nodes, all `align_corners=True`) and `CropAndResize` (4 nodes, bilinear, one
  box). ONNX Runtime 1.31 `Resize` (linear, `align_corners` and `tf_crop_and_resize`) gives other
  values than these kernels (1e-4 to 7e-4 on random input in [-1, 1]; 0.996 in the last row of a
  full-box crop, where the TensorFlow kernel writes the extrapolation value because the float
  source row is above H - 1). The rewrite computes the source indices and the interpolation
  weights with the float32 expressions of `resize_bilinear_op.cc` (`LegacyScaler`,
  `compute_interpolation_weights`, `compute_lerp`) and `crop_and_resize_op.cc`, and reads the four
  neighbors with Gather.

`tf2onnx` 1.17 also writes a float `scales` input for a `ResizeBilinear` with a static input shape
(224 -> 480 becomes floor(224 * 2.142857) = 479 in ONNX Runtime). After the rewrite no
`ResizeBilinear` node is left, so this does not apply.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GRAPH_NAME = "tf_graph.pb"
ONNX_NAME = "universal_matting.onnx"
TF_INPUT = "input_image:0"
TF_OUTPUT_PNG = "output_png:0"
TF_OUTPUT_RGBA = "cond_3/Merge:0"
ONNX_INPUT = "input_image"
ONNX_OUTPUT = "rgba"
# Tensors of the main graph (not in a cond branch) that record the branch selection.
TF_BRANCH_TENSORS = {"person_box": "concat:0", "person_rows_columns": "add_3:0",
                     "human_alpha_share": "truediv_13:0", "human_result": "Greater_8:0"}

# Graph rules of tf_graph.pb (read from the graph, 2026-10-21). Sizes in pixels.
GRAPH_RULES = {
    "detector_input": "photo / 127.5 - 1 resized to 192 x 192 (ResizeArea if the long edge "
                      "> 1920, else ResizeBilinear; align_corners=True)",
    "person_rule": "detector channel 0: (tanh + 1) / 2 > 0.8; rows and columns with more than 3 "
                   "person pixels; the human branch runs if rows + columns > 50",
    "human_branch": "box of the person rows and columns with a margin of 1/8 (box side > 0.33 of "
                    "the photo) or 1/2; CropAndResize to 1920 x 1920 and AvgPool 5 x 5 VALID "
                    "(long edge > 1920) or 640 x 640; ResizeBilinear to 224 (coarse U-Net); "
                    "refine U-Net at 480 x 480 on the crop and the coarse map; alpha = "
                    "((tanh + 1) / 2)^2 * 255",
    "branch_rule": "human result if the human branch ran and its alpha sum / 255 / photo area "
                   "> 0.1, else the object branch",
    "object_branch": "photo ResizeBilinear to 224 (stage 0 U-Net); box of the rows and columns "
                     "with more than 5 pixels of (tanh + 1) / 2 > 0.8, margin 1/4; CropAndResize "
                     "to 640 x 640, ResizeBilinear to 224 (stage 1 U-Net); refine U-Net at "
                     "640 x 640 on the crop and the stage 1 map; alpha = (tanh + 1) * 127.5",
    "output": "alpha ResizeBilinear (align_corners=True) from 480 or 640 to the box size in "
              "photo pixels, Pad to the photo; RGB = photo * (alpha > 0.001); RGBA",
}


def load_rgb(path: Path) -> np.ndarray:
    """The photo as `LoadImage` of ModelScope reads it: RGB uint8 H x W x 3 after
    `ImageOps.exif_transpose`."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


def model_input(rgb: np.ndarray) -> np.ndarray:
    """`input_image` [H, W, 3] float32. The source feeds `astype(float)` (float64); TensorFlow
    converts it to the float32 placeholder type, which is exact for the values 0 to 255."""
    return rgb.astype(np.float32)


def alpha_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Alpha in [0, 1] (float32 H x W) from the float RGBA output (channel 3, values 0 to 255)."""
    return rgba[..., 3].astype(np.float32) / 255.0


def alpha_to_photo(alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    """Alpha at another size (the MVP display size) with the resize of the other Subject alphas
    (PyTorch bilinear, `align_corners=False`, `antialias=True`), clipped to [0, 1]. The alpha
    itself is unchanged when the size is the same."""
    if alpha.shape == (height, width):
        return alpha.astype(np.float32)
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(alpha, np.float32))[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False,
                            antialias=True)
    return resized[0, 0].numpy().clip(0.0, 1.0)


def read_graph_def(path: Path):
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(Path(path).read_bytes())
    return graph_def


def tf_session(graph_def, threads: int = 4, graph_optimizations: bool = True):
    """A TensorFlow 1 session of `graph_def` on the CPU with `threads` intra-op threads.
    `graph_optimizations=False` turns off the grappler meta optimizer: it changes the float order
    of the MatMul form of `ResizeArea` (difference 1.6e-5 against 7.2e-7 without it)."""
    import tensorflow as tf

    tf1 = tf.compat.v1
    graph = tf.Graph()
    with graph.as_default():
        tf1.import_graph_def(graph_def, name="")
    config = tf1.ConfigProto(intra_op_parallelism_threads=threads, inter_op_parallelism_threads=1,
                             device_count={"GPU": 0})
    if not graph_optimizations:
        config.graph_options.rewrite_options.disable_meta_optimizer = True
    return tf1.Session(graph=graph, config=config)


def replace_constant_cond(graph_def):
    """`cond_2/Merge` -> `Cast(Greater_7, float32)`; the other `cond_2/` nodes are removed."""
    import tensorflow as tf
    from tensorflow.core.framework import types_pb2

    if "cond_2/Merge" not in {node.name for node in graph_def.node}:
        raise ValueError("graph has no cond_2/Merge")
    for node in graph_def.node:
        for name in node.input:
            source = name.split(":")[0].lstrip("^")
            if (not node.name.startswith("cond_2/") and source.startswith("cond_2/")
                    and source != "cond_2/Merge"):
                raise ValueError(f"{node.name} reads {name}")
    output = tf.compat.v1.GraphDef()
    output.versions.CopyFrom(graph_def.versions)
    output.library.CopyFrom(graph_def.library)
    for node in graph_def.node:
        if node.name == "cond_2/Merge":
            cast = output.node.add()
            cast.name = "cond_2/Merge"
            cast.op = "Cast"
            cast.input.append("Greater_7")
            cast.attr["SrcT"].type = types_pb2.DT_BOOL
            cast.attr["DstT"].type = types_pb2.DT_FLOAT
            cast.attr["Truncate"].b = False
        elif not node.name.startswith("cond_2/"):
            output.node.add().CopyFrom(node)
    return output


def resize_area_weights(in_size, out_size: int):
    """[out_size, in_size] float32 weights of `ResizeArea` with `align_corners=True` along one
    axis (TensorFlow ops; `in_size` is an int32 scalar tensor), and the scale.

    Kernel rule (`resize_area_op.cc`, `CalculateResizeScale`): scale = (in - 1) / (out - 1) for
    in > 1 and out > 1. Output index o covers [a, b) = [o * scale, (o + 1) * scale); input index i
    in [floor(a), ceil(b)) has the weight
      i < a:  (i + 1 > b) ? scale : i + 1 - a
      i >= a: (i + 1 > b) ? b - i : 1
    An index i >= in reads input in - 1 (`Bound`), so its weight is added to column in - 1. The
    output is the weighted sum times 1 / (height scale * width scale)."""
    import tensorflow as tf

    scale = tf.cast(in_size - 1, tf.float32) / float(out_size - 1)
    o = tf.range(out_size, dtype=tf.float32)[:, None]
    a = o * scale
    b = (o + 1.0) * scale
    columns = in_size + tf.cast(tf.math.ceil(scale), tf.int32) + 2
    i = tf.cast(tf.range(columns), tf.float32)[None, :]
    weight = tf.where(i < a, tf.where(i + 1.0 > b, tf.broadcast_to(scale, tf.shape(i + a)),
                                      i + 1.0 - a),
                      tf.where(i + 1.0 > b, b - i, tf.ones_like(i + a)))
    inside = tf.logical_and(i >= tf.math.floor(a), i < tf.math.ceil(b))
    weight = tf.where(inside, weight, tf.zeros_like(weight))
    last = in_size - 1
    folded = tf.concat([weight[:, :last], tf.reduce_sum(weight[:, last:], axis=1, keepdims=True)],
                       axis=1)
    return folded, scale


def resize_area(image, size: tuple[int, int]):
    """`ResizeArea(image, size, align_corners=True)` of a [1, H, W, C] tensor with MatMul."""
    import tensorflow as tf

    out_h, out_w = size
    shape = tf.shape(image)
    height, width, channels = shape[1], shape[2], shape[3]
    weight_h, scale_h = resize_area_weights(height, out_h)
    weight_w, scale_w = resize_area_weights(width, out_w)
    rows = tf.matmul(weight_h, tf.reshape(image[0], tf.stack([height, -1])))
    rows = tf.transpose(tf.reshape(rows, tf.stack([out_h, width, channels])), [1, 0, 2])
    both = tf.matmul(weight_w, tf.reshape(rows, tf.stack([width, -1])))
    both = tf.transpose(tf.reshape(both, tf.stack([out_w, out_h, channels])), [1, 0, 2])
    return both[None] * (1.0 / (scale_h * scale_w))


def bilinear_axis(in_size, out_size):
    """Source indices and weights of `ResizeBilinear(align_corners=True)` along one axis:
    `CalculateResizeScale`, `LegacyScaler` (in = float(o) * scale) and
    `compute_interpolation_weights` (lower = max(floor(in), 0), upper = min(ceil(in), in_size - 1),
    lerp = in - floor(in))."""
    import tensorflow as tf

    in_size = tf.cast(in_size, tf.int32)
    out_size = tf.cast(out_size, tf.int32)
    corners = tf.logical_and(in_size > 1, out_size > 1)
    scale = tf.where(corners,
                     tf.cast(in_size - 1, tf.float32) / tf.cast(tf.maximum(out_size - 1, 1),
                                                                tf.float32),
                     tf.cast(in_size, tf.float32) / tf.cast(out_size, tf.float32))
    position = tf.cast(tf.range(out_size), tf.float32) * scale
    floor = tf.math.floor(position)
    lower = tf.maximum(tf.cast(floor, tf.int32), 0)
    upper = tf.minimum(tf.cast(tf.math.ceil(position), tf.int32), in_size - 1)
    return lower, upper, position - floor


def bilinear_from_indices(image, rows: tuple, columns: tuple):
    """`compute_lerp` of a [N, H, W, C] tensor: top = tl + (tr - tl) * x_lerp, bottom = bl +
    (br - bl) * x_lerp, top + (bottom - top) * y_lerp; rows and columns are (lower, upper,
    lerp)."""
    import tensorflow as tf

    row_lower, row_upper, y_lerp = rows
    column_lower, column_upper, x_lerp = columns
    top_rows = tf.gather(image, row_lower, axis=1)
    bottom_rows = tf.gather(image, row_upper, axis=1)
    top_left = tf.gather(top_rows, column_lower, axis=2)
    top_right = tf.gather(top_rows, column_upper, axis=2)
    bottom_left = tf.gather(bottom_rows, column_lower, axis=2)
    bottom_right = tf.gather(bottom_rows, column_upper, axis=2)
    x_lerp = x_lerp[None, None, :, None]
    y_lerp = y_lerp[None, :, None, None]
    top = top_left + (top_right - top_left) * x_lerp
    bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
    return top + (bottom - top) * y_lerp


def resize_bilinear(image, size):
    """`ResizeBilinear(image, size, align_corners=True, half_pixel_centers=False)` of a
    [N, H, W, C] float32 tensor with Gather."""
    import tensorflow as tf

    shape = tf.shape(image)
    size = tf.cast(size, tf.int32)
    rows = bilinear_axis(shape[1], size[0])
    columns = bilinear_axis(shape[2], size[1])
    return bilinear_from_indices(image, rows, columns)


def crop_axis(start, end, in_size, crop_size: int):
    """Source indices, weights, and the inside mask of `CropAndResize` (bilinear) along one axis,
    crop_size > 1: scale = (end - start) * (in - 1) / (crop - 1); in = start * (in - 1) + o *
    scale; outside [0, in - 1] -> extrapolation value; lower = floor(in), upper = ceil(in) (clipped
    for the Gather of the outside positions), lerp = in - lower."""
    import tensorflow as tf

    if crop_size <= 1:
        raise ValueError("crop size 1 is not handled")
    last = tf.cast(in_size - 1, tf.float32)
    scale = (end - start) * last / float(crop_size - 1)
    position = start * last + tf.cast(tf.range(crop_size), tf.float32) * scale
    inside = tf.logical_and(position >= 0.0, position <= last)
    lower = tf.cast(tf.math.floor(position), tf.int32)
    upper = tf.cast(tf.math.ceil(position), tf.int32)
    lerp = position - tf.cast(lower, tf.float32)
    limit = in_size - 1
    return (tf.clip_by_value(lower, 0, limit), tf.clip_by_value(upper, 0, limit), lerp), inside


def crop_and_resize(image, boxes, box_index, crop_size: tuple[int, int],
                    extrapolation_value: float):
    """`CropAndResize(image, boxes, box_index, crop_size, method="bilinear")` for one box of a
    [1, H, W, C] tensor (the graph has one box per call; `box_index` must be [0])."""
    import tensorflow as tf

    shape = tf.shape(image)
    with tf.control_dependencies([tf.debugging.assert_equal(
            tf.shape(boxes)[0], 1, message="CropAndResize rewrite: one box only"),
            tf.debugging.assert_equal(box_index, tf.zeros_like(box_index),
                                      message="CropAndResize rewrite: box index 0 only")]):
        y1, x1, y2, x2 = boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3]
    rows, rows_inside = crop_axis(y1, y2, shape[1], crop_size[0])
    columns, columns_inside = crop_axis(x1, x2, shape[2], crop_size[1])
    values = bilinear_from_indices(image, rows, columns)
    inside = tf.logical_and(rows_inside[:, None], columns_inside[None, :])[None, :, :, None]
    return tf.where(tf.broadcast_to(inside, tf.shape(values)), values,
                    tf.fill(tf.shape(values), extrapolation_value))


def _constant_value(nodes: dict, name: str):
    import tensorflow as tf

    node = nodes[name.split(":")[0]]
    if node.op != "Const":
        return None
    return tf.make_ndarray(node.attr["value"].tensor)


def _replace_nodes(graph_def, kinds: dict):
    """Replace each node whose op is a key of `kinds` with the tensor of `kinds[op](graph, node,
    inputs, nodes)`. The new nodes are in the name scope `<node>_rewrite/` and have the control
    inputs of the replaced node and of its constant inputs (the cond branch of the node). The
    consumers read the new tensor; nodes that no output needs are removed."""
    import tensorflow as tf

    tf1 = tf.compat.v1
    nodes = {node.name: node for node in graph_def.node}
    targets = [node for node in graph_def.node if node.op in kinds]
    graph = tf.Graph()
    replacement: dict[str, str] = {}
    with graph.as_default():
        tf1.import_graph_def(graph_def, name="")
        for node in targets:
            data = [name for name in node.input if not name.startswith("^")]
            controls = {name.lstrip("^") for name in node.input if name.startswith("^")}
            for name in data:
                if nodes[name.split(":")[0]].op == "Const":
                    controls.update(i.lstrip("^") for i in nodes[name.split(":")[0]].input)
            tensors = [graph.get_tensor_by_name(name if ":" in name else name + ":0")
                       for name in data]
            control_ops = [graph.get_operation_by_name(name) for name in sorted(controls)]
            with tf1.name_scope(node.name + "_rewrite/"), graph.control_dependencies(control_ops):
                result = tf.identity(kinds[node.op](node, tensors, data, nodes), name="output")
            replacement[node.name] = result.op.name
        rewritten = graph.as_graph_def()
    for node in rewritten.node:
        for index, name in enumerate(node.input):
            source = name.split(":")[0]
            if source in replacement:
                if name.startswith("^"):
                    raise ValueError(f"{node.name} has a control input on {source}")
                if ":" in name and not name.endswith(":0"):
                    raise ValueError(f"{node.name} reads output {name}")
                node.input[index] = replacement[source]
    outputs = {TF_OUTPUT_PNG.split(":")[0], TF_OUTPUT_RGBA.split(":")[0]}
    outputs.update(name.split(":")[0] for name in TF_BRANCH_TENSORS.values())
    return _remove_unused_nodes(rewritten, outputs)


def _remove_unused_nodes(graph_def, outputs: set[str]):
    """Keep the nodes that `outputs` read (data and control inputs). `extract_sub_graph` keeps
    the replaced nodes because of their colocation attributes (`_class: loc:@...`), so this walks
    the inputs only and removes the colocation entries of removed nodes."""
    import tensorflow as tf

    by_name = {node.name: node for node in graph_def.node}
    needed, pending = set(), list(outputs)
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(i.split(":")[0].lstrip("^") for i in by_name[name].input)
    output = tf.compat.v1.GraphDef()
    output.versions.CopyFrom(graph_def.versions)
    output.library.CopyFrom(graph_def.library)
    for node in graph_def.node:
        if node.name not in needed:
            continue
        copy = output.node.add()
        copy.CopyFrom(node)
        if "_class" in copy.attr:
            kept = [v for v in copy.attr["_class"].list.s
                    if v.decode().removeprefix("loc:@") in needed]
            del copy.attr["_class"].list.s[:]
            copy.attr["_class"].list.s.extend(kept)
            if not kept:
                del copy.attr["_class"]
    return output


def _resize_area_node(node, tensors, data, nodes):
    if not node.attr["align_corners"].b:
        raise ValueError(f"{node.name}: align_corners=False is not handled")
    size = _constant_value(nodes, data[1])
    return resize_area(tensors[0], tuple(int(v) for v in size))


def _resize_bilinear_node(node, tensors, data, nodes):
    if not node.attr["align_corners"].b or node.attr["half_pixel_centers"].b:
        raise ValueError(f"{node.name}: only align_corners=True, half_pixel_centers=False")
    import tensorflow as tf

    return resize_bilinear(tf.cast(tensors[0], tf.float32), tensors[1])


def _crop_and_resize_node(node, tensors, data, nodes):
    if node.attr["method"].s != b"bilinear":
        raise ValueError(f"{node.name}: method {node.attr['method'].s!r} is not handled")
    size = _constant_value(nodes, data[3])
    if size is None:
        raise ValueError(f"{node.name}: crop size is not a constant")
    return crop_and_resize(tensors[0], tensors[1], tensors[2], tuple(int(v) for v in size),
                           float(node.attr["extrapolation_value"].f))


def rewrite_for_onnx(graph_def):
    """The rewrites of the module docstring: the constant cond, then `ResizeArea`,
    `ResizeBilinear`, and `CropAndResize`."""
    return _replace_nodes(replace_constant_cond(graph_def),
                          {"ResizeArea": _resize_area_node,
                           "ResizeBilinear": _resize_bilinear_node,
                           "CropAndResize": _crop_and_resize_node})


def convert_to_onnx(graph_def, opset: int):
    """ONNX model of the rewritten graph: `tf2onnx.tfonnx.process_tf_graph` and the ONNX
    optimizers of `tf2onnx`, with `continue_on_error=False` (an unsupported operator stops the
    conversion). `tf2onnx.convert.from_graph_def` runs the grappler `constfold` pass first, which
    pushes constants through Mul and RealDiv (`(x * y) / c` becomes `x * (y * (1 / c))`); that
    changes the float32 crop coordinates of `crop_axis` by one step, and the last crop row or
    column then reads the photo where TensorFlow writes the extrapolation value. This function
    does not run grappler."""
    import tensorflow as tf
    from tf2onnx import optimizer, tfonnx

    with tf.Graph().as_default() as tf_graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        graph = tfonnx.process_tf_graph(
            tf_graph, continue_on_error=False, opset=opset, input_names=[TF_INPUT],
            output_names=[TF_OUTPUT_RGBA],
            tensors_to_rename={TF_INPUT: ONNX_INPUT, TF_OUTPUT_RGBA: ONNX_OUTPUT})
        graph = optimizer.optimize_graph(graph, catch_errors=False)
        return graph.make_model("universal matting, converted from tf_graph.pb")


def operator_counts(graph_def) -> dict:
    counts: dict[str, int] = {}
    for node in graph_def.node:
        counts[node.op] = counts.get(node.op, 0) + 1
    return dict(sorted(counts.items()))
