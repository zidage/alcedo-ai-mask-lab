#!/usr/bin/env python3
"""Operator placement of one Subject graph on ONNX Runtime DirectML (issue #6) or Core ML (issue
#8, `--provider coreml`).

Input:  --model (s3od, birefnet, isnet, or universal_matting), --graph, --photo (the model
        input), --runs (default 2), --provider (default directml), --coreml-model-format.
Output: --out (JSON): the record of `provider_placement.measure_placement`: the node counts of
        the graph file per operator type; on Core ML the partitions and the supported nodes of
        the ONNX Runtime capability log; from the ONNX Runtime profile of the session: the number
        of executed nodes per execution provider (a DirectML or Core ML node is one fused
        partition), the kernel time per provider, and the CPU nodes per operator type. The
        profile file stays in --profile-dir.

The session has the MVP settings (subject_runtime.create_session) and profiling on. Profiling
changes the times, so this is a separate process from measure_directml_cost.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subject_runtime as rt  # noqa: E402
from provider_placement import measure_placement  # noqa: E402  (MVP directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=rt.MODELS)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--provider", default="directml", choices=sorted(rt.mp.PROVIDERS))
    parser.add_argument("--coreml-model-format", default=rt.mp.COREML_MODEL_FORMATS[0],
                        choices=rt.mp.COREML_MODEL_FORMATS)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    graph = Path(args.graph)
    model = rt.SubjectModel(args.model)
    prepared = model.prepare(Path(args.photo))
    label = f"{args.model}_{graph.parent.name}"
    if args.provider == "coreml":
        label += f"_coreml_{args.coreml_model_format}"
    report = {"model": args.model, "photo": Path(args.photo).name,
              **measure_placement(graph, args.provider,
                                  lambda session: {session.get_inputs()[0].name: prepared.pixels},
                                  args.runs, Path(args.profile_dir), label,
                                  coreml_model_format=args.coreml_model_format)}
    rt.mp.save_json(Path(args.out), report)
    print(json.dumps({k: report.get(k) for k in ("coreml_capability", "executed_nodes",
                                                 "kernel_ms_per_run", "cpu_nodes_per_operator")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
