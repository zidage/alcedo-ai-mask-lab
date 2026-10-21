#!/usr/bin/env python3
"""Session creation time of the BiRefNet graph on DirectML (issue #5, MVP report section 5.3).

Input:  the BiRefNet graph (positional), --work-dir (build/tmp: the saved optimized graph).
Output: --out (JSON): the creation time per graph optimization level, and the creation time of a
        graph that ONNX Runtime optimized once at the extended level on the CPU provider and saved
        (no provider-specific nodes in the file).

No inference runs. The session settings are the DirectML settings of
mask_pipeline.create_session (4 CPU threads, CPU memory arena off, sequential execution, no
memory patterns).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject_onnx")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    levels = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
              "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
              "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}

    def options(level: str, save: Path | None = None):
        result = ort.SessionOptions()
        result.intra_op_num_threads = 4
        result.inter_op_num_threads = 1
        result.enable_cpu_mem_arena = False
        result.log_severity_level = mp.ORT_ERROR_LOG_LEVEL
        result.enable_mem_pattern = False
        result.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        result.graph_optimization_level = levels[level]
        if save:
            result.optimized_model_filepath = str(save)
        return result

    def create(path: Path, level: str, provider: str, save: Path | None = None) -> float:
        started = time.perf_counter()
        session = ort.InferenceSession(str(path), options(level, save), providers=[provider])
        seconds = time.perf_counter() - started
        del session
        return round(seconds, 2)

    graph = Path(args.subject_onnx)
    record = {"onnxruntime": ort.__version__, "graph": mp.repository_path(graph),
              "directml_seconds": {}, "saved_extended_cpu": {}}
    for level in levels:
        record["directml_seconds"][level] = create(graph, level, "DmlExecutionProvider")
        print(f"[load] {level}: {record['directml_seconds'][level]} s", flush=True)
    saved = Path(args.work_dir) / "subject_matte_extended_cpu.onnx"
    saved.parent.mkdir(parents=True, exist_ok=True)
    record["saved_extended_cpu"]["save_seconds"] = create(graph, "extended", "CPUExecutionProvider",
                                                          saved)
    record["saved_extended_cpu"]["bytes"] = saved.stat().st_size
    for level in ("all", "basic", "disable_all"):
        seconds = create(saved, level, "DmlExecutionProvider")
        record["saved_extended_cpu"][f"directml_seconds_{level}"] = seconds
        print(f"[load] saved graph, {level}: {seconds} s", flush=True)
    mp.save_json(Path(args.out), record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
