"""Native Core ML models in the MVP pipeline (issue #8): the `coreml_native` provider.

Input:  a compiled Core ML model (`.mlmodelc`) of scripts/ai_models/coreml/convert_models.py and
        its record `<model>.json` in the same directory (the input and output names), and the
        compute units.
Output: `CoreMLSession`, an object with the methods of `onnxruntime.InferenceSession` that the
        pipeline and the measurement scripts use: `run(None, feed)` returns the outputs as
        float32 arrays in the order of the record, `get_inputs()` and `get_outputs()` give the
        names and the shapes, `get_providers()` names the runtime and the compute units.

This is the runtime form of the Alcedo Studio native Core ML SigLIP2 models (`coreml_clip.rs`):
a precompiled `.mlmodelc` loaded with an `MLModelConfiguration` whose compute units are
`ComputeUnits::All` by default. Here coremltools `CompiledMLModel` loads it (macOS only).
Compute units: ALL (the production default), CPU_AND_GPU, CPU_AND_NE, CPU_ONLY. The compute
units are given explicitly; the module does not change them after an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")
RUNTIME = "CoreMLNative"


@dataclass
class TensorInfo:
    name: str
    shape: list


def model_record(model: Path) -> dict:
    """The convert_models.py record of a compiled model (`<stem>.json` next to it)."""
    path = Path(model).with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the record of convert_models.py is missing")
    return json.loads(path.read_text(encoding="utf-8"))


class CoreMLSession:
    """A compiled Core ML model with the `InferenceSession` methods of the pipeline."""

    def __init__(self, model: Path, compute_units: str = COMPUTE_UNITS[0]) -> None:
        import coremltools as ct

        if compute_units not in COMPUTE_UNITS:
            raise ValueError(f"compute units {compute_units!r}: use one of {COMPUTE_UNITS}")
        self.path = Path(model)
        self.record = model_record(self.path)
        self.compute_units = compute_units
        self.model = ct.models.CompiledMLModel(str(self.path),
                                               compute_units=getattr(ct.ComputeUnit, compute_units))
        self.input_infos = [TensorInfo(name, shape) for name, shape in
                            self.record["inputs"].items()]
        self.output_names = list(self.record["outputs"])

    def run(self, output_names, feed: dict) -> list[np.ndarray]:
        outputs = self.model.predict({name: np.ascontiguousarray(value, np.float32)
                                      for name, value in feed.items()})
        names = output_names or self.output_names
        return [np.asarray(outputs[name], np.float32) for name in names]

    def get_inputs(self) -> list[TensorInfo]:
        return self.input_infos

    def get_outputs(self) -> list[TensorInfo]:
        return [TensorInfo(name, []) for name in self.output_names]

    def get_providers(self) -> list[str]:
        return [f"{RUNTIME}:{self.compute_units}"]
