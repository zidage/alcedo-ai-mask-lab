#!/usr/bin/env python3
"""Run a measurement program on macOS and sample its memory (issue #8): the macOS equivalent of
`run_with_gpu_memory.ps1`.

Input:  --out (JSON), --log (the program output; the error output goes to <log>.stderr),
        --interval-ms (default 200), and the program with its arguments after `--`.
Output: --out: the exit code (a negative value is the signal that stopped the program), the
        duration, the peak physical footprint of the program process and its child processes, the
        system memory in use before the run, its peak, and the difference, the samples, and the
        crash reports of the run window.

Memory values (macos_records.py): the physical footprint of each process of the program (unified
memory: CPU and GPU allocations of the process) and its lifetime peak, which the kernel keeps, so
the peak does not depend on the sampling interval. The system value (`vm_stat`, Activity Monitor
"Memory Used") also contains memory that other processes, for example the Neural Engine process
`aned`, allocate for the program; other programs on the machine change it too, so it is an upper
estimate. Run one model job at a time (AGENTS.md).

Example:
  python run_with_memory_sampling.py --out cost_memory.json --log cost.log -- \
      venv/bin/python -I measure_directml_cost.py --provider coreml ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macos_records  # noqa: E402

CRASH_REPORT_MARGIN = timedelta(seconds=10)


def process_tree(pid: int) -> list[int]:
    import psutil

    try:
        root = psutil.Process(pid)
        return [pid] + [child.pid for child in root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the program after --")

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    system_before = macos_records.system_used_mb()
    started_at = datetime.now()
    started = time.perf_counter()
    peaks: dict[int, float] = {}
    samples = []
    with log.open("w", encoding="utf-8") as stdout, \
            Path(f"{log}.stderr").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while True:
            exit_code = process.poll()
            footprint_mb = 0.0
            for pid in process_tree(process.pid):
                try:
                    record = macos_records.process_footprint(pid)
                except OSError:
                    continue
                footprint_mb += record["phys_footprint_mb"]
                peaks[pid] = max(peaks.get(pid, 0.0), record["lifetime_max_phys_footprint_mb"])
            samples.append({"s": round(time.perf_counter() - started, 2),
                            "phys_footprint_mb": round(footprint_mb, 1),
                            "system_used_mb": macos_records.system_used_mb()})
            if exit_code is not None:
                break
            time.sleep(args.interval_ms / 1000)
    ended_at = datetime.now()
    if exit_code != 0:
        # ReportCrash writes the report some seconds after the process stops.
        time.sleep(CRASH_REPORT_MARGIN.total_seconds())
    system_peak = max(s["system_used_mb"] for s in samples)
    report = {
        "command": command, "exit_code": exit_code,
        "start": started_at.isoformat(timespec="seconds"),
        "end": ended_at.isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1), "interval_ms": args.interval_ms,
        "peak_phys_footprint_mb_per_process": {str(pid): value for pid, value in peaks.items()},
        "peak_phys_footprint_mb": max(peaks.values(), default=None),
        "peak_sampled_phys_footprint_sum_mb": max(s["phys_footprint_mb"] for s in samples),
        "system_used_before_mb": system_before, "system_used_peak_mb": system_peak,
        "system_used_peak_increase_mb": round(system_peak - system_before, 1),
        "crash_reports": macos_records.crash_reports(started_at,
                                                     ended_at + CRASH_REPORT_MARGIN),
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[memory] exit {exit_code}, peak footprint {report['peak_phys_footprint_mb']} MB, "
          f"system +{report['system_used_peak_increase_mb']} MB, "
          f"{len(report['crash_reports'])} crash reports -> {out}", flush=True)
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
