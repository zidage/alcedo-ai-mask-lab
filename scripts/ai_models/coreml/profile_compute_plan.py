#!/usr/bin/env python3
"""Compute device placement of a native Core ML model (issue #8): the Core ML compute plan.

Input:  --model (a compiled model `.mlmodelc` of convert_models.py), --compute-units (ALL,
        CPU_AND_GPU, CPU_AND_NE, CPU_ONLY).
Output: --out (JSON): from `MLComputePlan` (coremltools `MLComputePlan.load_from_path`, the Core
        ML API `MLComputePlan` of macOS 14.4 and later), for the ops of the `main` function
        without `const`: the number of ops per preferred compute device (CPU, GPU, NeuralEngine),
        the share of the estimated cost per device (`MLComputePlan` estimated cost, a relative
        weight), the op types per device, and the ops that have no device.

The compute plan is the plan of Core ML for this machine and these compute units; it does not
run the model, and it states where Core ML prefers to run each op (the ONNX Runtime profile of
provider_placement.py has no equivalent for native models). This is a separate process from the
cost measurements.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")


def device_name(device) -> str:
    from coremltools.models.compute_device import (MLCPUComputeDevice, MLGPUComputeDevice,
                                                   MLNeuralEngineComputeDevice)

    if isinstance(device, MLNeuralEngineComputeDevice):
        return "NeuralEngine"
    if isinstance(device, MLGPUComputeDevice):
        return "GPU"
    if isinstance(device, MLCPUComputeDevice):
        return "CPU"
    return "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compute-units", default="ALL", choices=COMPUTE_UNITS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(path=args.model,
                                        compute_units=getattr(ct.ComputeUnit, args.compute_units))
    program = plan.model_structure.program
    if program is None:
        raise ValueError(f"{args.model}: not an ML program")
    operations = program.functions["main"].block.operations
    ops: Counter = Counter()
    cost: defaultdict = defaultdict(float)
    types: defaultdict = defaultdict(Counter)
    supported: defaultdict = defaultdict(Counter)
    without_device = []
    for operation in operations:
        if operation.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = device_name(usage.preferred_compute_device) if usage else "none"
        if device == "none":
            without_device.append(operation.operator_name)
        estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
        ops[device] += 1
        cost[device] += estimate.weight if estimate else 0.0
        types[device][operation.operator_name] += 1
        if usage:
            for supported_device in usage.supported_compute_devices:
                supported[device_name(supported_device)][operation.operator_name] += 1
    total_cost = sum(cost.values()) or 1.0
    report = {
        "model": Path(args.model).as_posix(), "compute_units": args.compute_units,
        "coremltools": ct.__version__, "macos": platform.mac_ver()[0],
        "ops": sum(ops.values()), "ops_per_preferred_device": dict(ops),
        "estimated_cost_share_per_preferred_device": {d: round(c / total_cost, 4)
                                                      for d, c in cost.items()},
        "op_types_per_preferred_device": {d: dict(c.most_common()) for d, c in types.items()},
        "ops_without_device": dict(Counter(without_device)),
        "supported_op_types_per_device": {d: dict(c.most_common()) for d, c in supported.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("compute_units", "ops", "ops_per_preferred_device",
                                             "estimated_cost_share_per_preferred_device")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
