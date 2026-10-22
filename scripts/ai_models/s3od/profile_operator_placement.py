#!/usr/bin/env python3
"""Operator placement of one Subject graph on ONNX Runtime DirectML (issue #6).

Input:  --model (s3od, birefnet, isnet, or universal_matting), --graph, --photo (the model
        input), --runs (default 2).
Output: --out (JSON): the node counts of the graph file per operator type; from the ONNX Runtime
        profile of the session: the number of executed nodes per execution provider (a DirectML
        node is one fused partition after the DirectML graph fusion), the kernel time per
        provider, and the CPU nodes per operator type. The profile file stays in --profile-dir.

The session has the MVP settings (subject_runtime.create_session) and profiling on. Profiling
changes the times, so this is a separate process from measure_directml_cost.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subject_runtime as rt  # noqa: E402


def graph_operators(path: Path) -> dict:
    """Operator counts of the ONNX file; empty when the onnx package is not installed (the MVP
    environment has no onnx package; export_report.json has the same counts)."""
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(path), load_external_data=False)
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=rt.MODELS)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    graph = Path(args.graph)
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    model = rt.SubjectModel(args.model)
    prepared = model.prepare(Path(args.photo))
    session = rt.create_session(graph, "directml",
                                profile_prefix=str(profile_dir / f"{args.model}_{graph.parent.name}"))
    input_name = session.get_inputs()[0].name
    for _ in range(args.runs):
        session.run(None, {input_name: prepared.pixels})
    profile_path = Path(session.end_profiling())

    events = json.loads(profile_path.read_text(encoding="utf-8"))
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
    report = {
        "model": args.model, "graph": rt.mp.repository_path(graph),
        "onnxruntime": ort.__version__, "photo": Path(args.photo).name, "runs": args.runs,
        "graph_operators": graph_operators(graph),
        "executed_nodes": {p: len(n) for p, n in sorted(nodes.items())},
        "kernel_ms_per_run": {p: round(us / 1000 / args.runs, 1)
                              for p, us in sorted(kernel_us.items())},
        "cpu_nodes_per_operator": dict(cpu_operators.most_common()),
        "profile": rt.mp.repository_path(profile_path),
    }
    rt.mp.save_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("executed_nodes", "kernel_ms_per_run",
                                             "cpu_nodes_per_operator")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
