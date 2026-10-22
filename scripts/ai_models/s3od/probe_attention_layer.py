#!/usr/bin/env python3
"""Cost of one S3OD encoder attention on DirectML in different graph forms (issue #6).

The question: does `com.microsoft.MultiHeadAttention` on DirectML keep the [1, 12, 4101, 4101]
score tensor, and does an attention in query chunks use less memory? The probe uses the real
query, key, and value tensors of one encoder layer, so the float16 overflow of layer 0 shows.

Steps (one model job at a time):

  capture  (export environment, PyTorch) the query, key, and value after RoPE of --layer on
           --photo, in the [1, tokens, 768] layout, and the PyTorch attention output
           -> <work-dir>/layer<L>.npz
  build    (export environment, onnx) the single-attention graphs of the forms below
           -> <work-dir>/<form>_<precision>_chunks<N>_layer<L>.onnx
  run      (MVP environment, onnxruntime-directml) each graph in a new process, with the MVP
           session settings: the GPU memory of the process after the first run, the first run,
           the warm p50 of 10 runs, the executed nodes per provider, and the maximum absolute
           difference to the PyTorch output -> --out (JSON)

Forms (all compute softmax(q k^T x scale) v with 12 heads; the query chunk forms split the
queries into N chunks and concatenate the outputs):

  multi_head_attention   `com.microsoft.MultiHeadAttention` (query chunks: one node per chunk)
  matmul                 MatMul, Mul (scale), Softmax, MatMul (as the exported graph)
  *_chained              chunk k + 1 adds 0 x ReduceMax(output of chunk k) to its query, so that
                         the chunks are a sequence (s3od_graphs.use_query_chunked_attention)
  add                    q + k + v (the GPU memory of the inputs and the device, for reference)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEADS = 12
FORMS = ("multi_head_attention", "matmul", "multi_head_attention_chained", "matmul_chained",
         "add")
DEFAULT_CASES = ["multi_head_attention:float32:1", "matmul:float32:1",
                 "multi_head_attention:float16:1", "matmul:float16:1", "add:float32:1",
                 "multi_head_attention:float32:4", "matmul:float32:4",
                 "multi_head_attention_chained:float32:4", "matmul_chained:float32:4",
                 "multi_head_attention:float32:16", "matmul:float32:16",
                 "multi_head_attention_chained:float32:16", "matmul_chained:float32:16"]


def case_name(form: str, precision: str, chunks: int, layer: int) -> str:
    return f"{form}_{precision}_chunks{chunks}_layer{layer}"


def parse_case(text: str) -> tuple[str, str, int]:
    form, precision, chunks = text.split(":")
    if form not in FORMS or precision not in ("float32", "float16"):
        raise ValueError(f"case {text!r}: form one of {FORMS}, precision float32 or float16")
    return form, precision, int(chunks)


def capture(args) -> None:
    import torch
    from transformers import AttentionInterface

    sys.path.insert(0, str(HERE))
    import s3od_graphs as graphs

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model = graphs.load_model(args.source, Path(args.checkpoint))
    captured = []

    def capturing_attention(module, query, key, value, attention_mask, scaling=None,
                            dropout=0.0, **kwargs):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value,
                                                                  scale=scaling)
        batch, heads, tokens, size = query.shape

        def tokens_last(tensor):
            return tensor.transpose(1, 2).reshape(batch, tokens, heads * size).numpy()

        captured.append({"query": tokens_last(query), "key": tokens_last(key),
                         "value": tokens_last(value), "output": tokens_last(output),
                         "scale": np.float32(scaling)})
        return output.transpose(1, 2).contiguous(), None

    AttentionInterface.register("s3od_capture", capturing_attention)
    model.encoder.config._attn_implementation = "s3od_capture"
    pixels, _ = graphs.letterbox(graphs.load_rgb(Path(args.photo)))
    model(torch.from_numpy(pixels))
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / f"layer{args.layer}.npz", **captured[args.layer])
    print(f"[capture] layer {args.layer}: {captured[args.layer]['query'].shape}", flush=True)


def build(args) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    data = np.load(Path(args.work_dir) / f"layer{args.layer}.npz")
    batch, tokens, width = data["query"].shape
    size = width // HEADS
    scale = float(data["scale"])
    for text in args.case or DEFAULT_CASES:
        form, precision, chunks = parse_case(text)
        element = TensorProto.FLOAT16 if precision == "float16" else TensorProto.FLOAT
        dtype = np.float16 if precision == "float16" else np.float32
        inputs = [helper.make_tensor_value_info(n, element, [batch, tokens, width])
                  for n in ("query", "key", "value")]
        output = helper.make_tensor_value_info("output", element, [batch, tokens, width])
        initializers = [
            numpy_helper.from_array(np.array([tokens // chunks + (1 if i < tokens % chunks else 0)
                                              for i in range(chunks)], np.int64), "chunk_sizes"),
            numpy_helper.from_array(np.array([batch, -1, HEADS, size], np.int64), "heads_shape"),
            numpy_helper.from_array(np.array([batch, -1, width], np.int64), "tokens_shape"),
            numpy_helper.from_array(np.array(scale, dtype), "scale"),
            numpy_helper.from_array(np.array(0, dtype), "zero")]
        if form == "add":
            nodes = [helper.make_node("Add", ["query", "key"], ["query_key"]),
                     helper.make_node("Add", ["query_key", "value"], ["output"])]
        else:
            nodes = attention_nodes(helper, form, chunks, scale)
        graph = helper.make_graph(nodes, "attention_probe", inputs, [output], initializers)
        model = helper.make_model(graph, ir_version=10, opset_imports=[
            helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)])
        path = Path(args.work_dir) / f"{case_name(form, precision, chunks, args.layer)}.onnx"
        onnx.save(model, str(path))
        print(f"[build] {path.name}: {len(nodes)} nodes", flush=True)


def attention_nodes(helper, form: str, chunks: int, scale: float) -> list:
    node = helper.make_node
    queries = [f"query_{i}" for i in range(chunks)]
    nodes = [node("Split", ["query", "chunk_sizes"], queries, axis=1)]
    if form.startswith("matmul"):
        nodes += [node("Reshape", ["key", "heads_shape"], ["key_heads"]),
                  node("Transpose", ["key_heads"], ["key_t"], perm=[0, 2, 3, 1]),
                  node("Reshape", ["value", "heads_shape"], ["value_heads"]),
                  node("Transpose", ["value_heads"], ["value_t"], perm=[0, 2, 1, 3])]
    outputs = []
    for i in range(chunks):
        query = queries[i]
        if form.endswith("_chained") and i > 0:
            nodes += [node("ReduceMax", [outputs[-1]], [f"maximum_{i}"], keepdims=0),
                      node("Mul", [f"maximum_{i}", "zero"], [f"zero_{i}"]),
                      node("Add", [query, f"zero_{i}"], [f"query_after_{i}"])]
            query = f"query_after_{i}"
        if form.startswith("multi_head_attention"):
            nodes.append(node("MultiHeadAttention", [query, "key", "value"], [f"output_{i}"],
                              domain="com.microsoft", num_heads=HEADS, scale=scale))
        else:
            nodes += [node("Reshape", [query, "heads_shape"], [f"query_heads_{i}"]),
                      node("Transpose", [f"query_heads_{i}"], [f"query_t_{i}"],
                           perm=[0, 2, 1, 3]),
                      node("MatMul", [f"query_t_{i}", "key_t"], [f"products_{i}"]),
                      node("Mul", [f"products_{i}", "scale"], [f"scores_{i}"]),
                      node("Softmax", [f"scores_{i}"], [f"weights_{i}"], axis=-1),
                      node("MatMul", [f"weights_{i}", "value_t"], [f"output_heads_{i}"]),
                      node("Transpose", [f"output_heads_{i}"], [f"output_t_{i}"],
                           perm=[0, 2, 1, 3]),
                      node("Reshape", [f"output_t_{i}", "tokens_shape"], [f"output_{i}"])]
        outputs.append(f"output_{i}")
    nodes.append(node("Concat", outputs, ["output"], axis=1))
    return nodes


def run_one(args) -> dict:
    """One graph in this process (the `run` step starts a process per graph)."""
    sys.path.insert(0, str(HERE.parents[1] / "poc" / "ai_mask_mvp"))
    import onnxruntime as ort
    from measure_birefnet_memory_phases import gpu_memory_mb

    form, precision, chunks = parse_case(args.case[0])
    data = np.load(Path(args.work_dir) / f"layer{args.layer}.npz")
    dtype = np.float16 if precision == "float16" else np.float32
    path = Path(args.work_dir) / f"{case_name(form, precision, chunks, args.layer)}.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(args.work_dir) / f"profile_{path.stem}")
    session = ort.InferenceSession(str(path), options, providers=["DmlExecutionProvider"])
    feeds = {name: data[name].astype(dtype) for name in ("query", "key", "value")}
    started = time.perf_counter()
    result = session.run(None, feeds)[0]
    first_ms = (time.perf_counter() - started) * 1000
    memory = gpu_memory_mb(os.getpid())
    times = []
    for _ in range(10):
        started = time.perf_counter()
        result = session.run(None, feeds)[0]
        times.append((time.perf_counter() - started) * 1000)
    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    providers: dict[str, int] = {}
    for event in profile:
        provider = event.get("args", {}).get("provider")
        if event.get("cat") == "Node" and provider and event["name"].endswith("_kernel_time"):
            providers[provider] = providers.get(provider, 0) + 1
    values = result.astype(np.float64)
    record = {"form": form, "precision": precision, "query_chunks": chunks,
              "layer": args.layer, "graph": path.name, "first_run_ms": round(first_ms, 1),
              "warm_p50_ms": round(float(np.median(times)), 1),
              "after_first_run": memory, "executed_nodes_per_provider_in_11_runs": providers,
              "finite": bool(np.isfinite(values).all())}
    if form != "add":
        record["max_abs_diff_to_pytorch"] = float(np.nanmax(np.abs(values - data["output"])))
    return record


def run(args) -> None:
    records = []
    for text in args.case or DEFAULT_CASES:
        command = [sys.executable, "-I", __file__, "run-one", "--work-dir", args.work_dir,
                   "--layer", str(args.layer), "--case", text]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{text}: exit {completed.returncode}\n{completed.stderr[-3000:]}")
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        records.append(record)
        print(f"[run] {text}: {record['after_first_run']['dedicated_mb']} MB, "
              f"{record['warm_p50_ms']} ms, finite {record['finite']}", flush=True)
        time.sleep(2)
    import onnxruntime

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "step": "issue #6: one S3OD encoder attention on DirectML in different graph forms",
        "onnxruntime": onnxruntime.__version__, "layer": args.layer,
        "session_settings": {"intra_op_threads": 4, "inter_op_threads": 1,
                             "cpu_memory_arena": False, "memory_pattern": False,
                             "execution_mode": "sequential"},
        "records": records}, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("capture", "build", "run", "run-one"))
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--case", action="append",
                        help="form:precision:chunks (default: the cases of the report)")
    parser.add_argument("--source", help="capture: s3od/s3od of the KupynOrest/s3od checkout")
    parser.add_argument("--checkpoint", help="capture: s3od.pt")
    parser.add_argument("--photo", help="capture: the photo")
    parser.add_argument("--out", help="run: the JSON record")
    args = parser.parse_args()
    if args.step == "capture":
        capture(args)
    elif args.step == "build":
        build(args)
    elif args.step == "run":
        run(args)
    else:
        print(json.dumps(run_one(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
