#!/usr/bin/env python3
"""GPU memory of the BiRefNet DirectML session per phase (issue #5, MVP report section 5.3).

Input:  the BiRefNet graph (positional), one photo (positional).
Output: --out (JSON): per phase the time and the dedicated and shared GPU memory of this process.

Phases: session load, run 1, run 2, a run with the ONNX Runtime run option
`memory.enable_memory_arena_shrinkage` (device strings `gpu:0` and `DML:0`), session release,
session load again, run. The session settings are the settings of mask_pipeline.create_session
(DirectML only, 4 CPU threads, CPU memory arena off). The GPU memory comes from the Windows
"GPU Process Memory" counters of this process (os.getpid(): the interpreter, not the virtual
environment launcher), read by PowerShell after each phase.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402


def gpu_memory_mb(pid: int) -> dict:
    counter = f"\\GPU Process Memory(pid_{pid}_*)"
    command = (f"$s=(Get-Counter -Counter '{counter}\\Dedicated Usage','{counter}\\Shared Usage' "
               f"-ErrorAction SilentlyContinue).CounterSamples; "
               f"$d=($s|?{{$_.Path -like '*dedicated usage'}}|Measure-Object CookedValue -Sum).Sum; "
               f"$h=($s|?{{$_.Path -like '*shared usage'}}|Measure-Object CookedValue -Sum).Sum; "
               f"\"$d $h\"")
    text = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                          text=True).stdout.split()
    values = [float(v) if v else 0.0 for v in text] + [0.0, 0.0]
    return {"dedicated_mb": round(values[0] / 2**20), "shared_mb": round(values[1] / 2**20)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject_onnx")
    parser.add_argument("photo")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    pid = os.getpid()
    records = []

    def phase(name: str, milliseconds: float | None = None) -> None:
        record = {"phase": name, "ms": None if milliseconds is None else round(milliseconds, 1),
                  **gpu_memory_mb(pid)}
        records.append(record)
        print(record, flush=True)

    subject_onnx = Path(args.subject_onnx)
    birefnet = mp._load_birefnet_graphs()
    pixels = birefnet.preprocess(mp.load_photo(Path(args.photo)).image, 1024).numpy()
    phase("before load")
    started = time.perf_counter()
    session = mp.create_session(subject_onnx, "directml", 4)
    phase("session load", (time.perf_counter() - started) * 1000)
    for index in (1, 2):
        started = time.perf_counter()
        session.run(None, {"pixel_values": pixels})
        phase(f"run {index}", (time.perf_counter() - started) * 1000)
    for device in ("gpu:0", "DML:0"):
        options = ort.RunOptions()
        options.add_run_config_entry("memory.enable_memory_arena_shrinkage", device)
        started = time.perf_counter()
        try:
            session.run(None, {"pixel_values": pixels}, options)
            phase(f"run with arena shrinkage {device}", (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001 - records which device string ONNX Runtime refuses
            records.append({"phase": f"run with arena shrinkage {device}", "error": str(exc)[:300]})
            print(records[-1], flush=True)
    del session
    gc.collect()
    phase("session released")
    started = time.perf_counter()
    session = mp.create_session(subject_onnx, "directml", 4)
    phase("session load again", (time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    session.run(None, {"pixel_values": pixels})
    phase("run after load again", (time.perf_counter() - started) * 1000)
    mp.save_json(Path(args.out), {"onnxruntime": ort.__version__,
                                  "graph": mp.repository_path(subject_onnx),
                                  "photo": Path(args.photo).name, "records": records})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
