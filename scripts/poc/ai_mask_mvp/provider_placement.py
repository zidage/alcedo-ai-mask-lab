"""Operator placement of one ONNX graph on an ONNX Runtime execution provider (issues #6 and #8).

Input:  a graph, the provider (mask_pipeline.PROVIDERS), the Core ML model format, a function
        that gives the inputs of one run for the session, the number of runs, and a profile
        directory.
Output: a record with:
          graph_operators         node counts of the graph file per operator type (onnx package)
          coreml_capability       Core ML only: the lines of the ONNX Runtime log
                                  "CoreMLExecutionProvider::GetCapability, number of partitions
                                  supported by CoreML: P number of nodes in the graph: N number
                                  of nodes supported by CoreML: S" as numbers; ONNX Runtime writes
                                  one line each time that it partitions the graph (before and
                                  after graph optimizations); the last line is the final graph
          executed_nodes          from the ONNX Runtime profile: the number of executed nodes per
                                  provider (a DirectML or Core ML node is one fused partition)
          kernel_ms_per_run       the kernel time per provider and run
          cpu_nodes_per_operator  the CPU nodes per operator type
        The profile file stays in the profile directory.

The session is `mask_pipeline.create_session` with profiling on and the log severity "info"
(the Core ML capability line has the severity "warning" when Core ML takes nodes, and "info"
when it takes none). ONNX Runtime writes its log to the native error
stream (file descriptor 2), so `capture_native_stderr` sends that stream to a file during the
session creation. Profiling changes the times; the cost measurements use other processes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

ORT_INFO_LOG_LEVEL = 1
CAPABILITY_PATTERN = re.compile(
    r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
    r"number of nodes supported by CoreML: (\d+)")


def capture_native_stderr(function):
    """Run `function()` with file descriptor 2 sent to a temporary file. Returns the result and
    the text that native code wrote to the error stream. An exception of `function` is raised
    after the stream is restored; its text is in the exception notes."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as capture:
        os.dup2(capture.fileno(), 2)
        error = None
        try:
            result = function()
        except Exception as caught:  # noqa: BLE001 - raised again below with the log text
            error, result = caught, None
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        capture.seek(0)
        text = capture.read().decode("utf-8", errors="replace")
    if error is not None:
        error.add_note(text[-4000:])
        raise error
    return result, text


def coreml_capability(log_text: str) -> list[dict]:
    return [{"partitions": int(p), "graph_nodes": int(n), "coreml_nodes": int(c)}
            for p, n, c in CAPABILITY_PATTERN.findall(log_text)]


def graph_operators(path: Path) -> dict:
    """Operator counts of the ONNX file; empty when the onnx package is not installed."""
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(path), load_external_data=False)
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def profile_placement(profile_path: Path, runs: int) -> dict:
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    nodes: dict[str, set[str]] = defaultdict(set)
    kernel_us: Counter = Counter()
    cpu_operators: Counter = Counter()
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        event_args = event.get("args", {})
        provider = event_args.get("provider", "unknown")
        name = event["name"][: -len("_kernel_time")]
        if name not in nodes[provider] and provider == "CPUExecutionProvider":
            cpu_operators[event_args.get("op_name", "unknown")] += 1
        nodes[provider].add(name)
        kernel_us[provider] += int(event.get("dur", 0))
    return {"executed_nodes": {p: len(n) for p, n in sorted(nodes.items())},
            "kernel_ms_per_run": {p: round(us / 1000 / runs, 1)
                                  for p, us in sorted(kernel_us.items())},
            "cpu_nodes_per_operator": dict(cpu_operators.most_common())}


def measure_placement(graph: Path, provider: str, make_inputs: Callable[[object], dict], runs: int,
                      profile_dir: Path, label: str, threads: int = 4,
                      coreml_model_format: str = mp.COREML_MODEL_FORMATS[0]) -> dict:
    """Create a profiled session, run it `runs` times with `make_inputs(session)`, and return the
    record."""
    import onnxruntime as ort

    profile_dir.mkdir(parents=True, exist_ok=True)
    session, log_text = capture_native_stderr(lambda: mp.create_session(
        graph, provider, threads, coreml_model_format, str(profile_dir / label),
        ORT_INFO_LOG_LEVEL))
    inputs = make_inputs(session)
    for _ in range(runs):
        session.run(None, inputs)
    profile_path = Path(session.end_profiling())
    record = {"graph": mp.repository_path(graph), "provider": provider,
              "provider_settings": mp.provider_settings(provider, coreml_model_format),
              "onnxruntime": ort.__version__, "runs": runs,
              "graph_operators": graph_operators(graph)}
    if provider == "coreml":
        record["coreml_capability"] = coreml_capability(log_text)
    record.update(profile_placement(profile_path, runs))
    record["profile"] = mp.repository_path(profile_path)
    return record
