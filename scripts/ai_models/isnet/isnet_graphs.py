"""Shared helpers of the IS-Net general-use scripts (issue #6): model loading, preprocessing, and
postprocessing.

The model code comes from the DIS repository (`xuebinqin/DIS`, file `IS-Net/models/isnet.py`,
argument `--source` is the `IS-Net` directory). `load_model` builds `ISNetDIS()` with the default
arguments of `IS-Net/Inference.py` and loads `isnet-general-use.pth` with `weights_only=True`.

Preprocessing rule of the source (`IS-Net/Inference.py`, input_size [1024, 1024]):

- read the photo as RGB uint8 (`skimage.io.imread`; for JPEG files it decodes with Pillow, the
  same decoder as `load_rgb`);
- float32 NCHW tensor, `F.upsample(..., [1024, 1024], mode="bilinear")` (the deprecated name of
  `F.interpolate`, `align_corners=False`, no antialias), then `.type(torch.uint8)` (truncation
  toward zero);
- `value / 255`, then `normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])`: `value / 255 - 0.5`.

Postprocessing rule of the source: the output is `result[0][0]`, the sigmoid of side output d1
at 1024 x 1024. The source resizes it to the photo size (bilinear, `align_corners=False`), then
scales it to [0, 1] with its minimum and maximum (`(result - min) / (max - min)`).

The stored alpha of these scripts is the d1 output scaled with its minimum and maximum at
1024 x 1024 (`normalize_range`), as the BiRefNet alpha of E2 is stored at 1024 x 1024.
`alpha_to_photo` resizes it to the photo size. The order (scale, then resize) is not the order of
the source (resize, then scale); `compare_onnx_outputs.py` records the difference
(`source_pipeline_max_abs_diff`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

IMAGE_SIZE = 1024
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (1.0, 1.0, 1.0)
CHECKPOINT_NAME = "isnet-general-use.pth"
GRAPH_NAME = "subject_d1.onnx"
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "d1"


def load_rgb(path: Path) -> np.ndarray:
    """The photo as RGB uint8 H x W x 3."""
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def preprocess(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB uint8 H x W x 3 -> `pixel_values` [1, 3, S, S] float32 (the rule of `Inference.py`)."""
    import torch
    import torch.nn.functional as F

    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear",
                            align_corners=False).type(torch.uint8)
    pixels = torch.divide(resized, 255.0)
    mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)
    return ((pixels - mean) / std).numpy().astype(np.float32)


def normalize_range(alpha: np.ndarray) -> np.ndarray:
    """`(alpha - min) / (max - min)`, the scaling of `Inference.py`."""
    alpha = alpha.astype(np.float32)
    low, high = float(alpha.min()), float(alpha.max())
    return (alpha - low) / (high - low)


def alpha_to_photo(alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize (`align_corners=False`) of an S x S alpha to the photo size."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(alpha, np.float32))[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def source_alpha(d1: np.ndarray, height: int, width: int) -> np.ndarray:
    """The alpha of `Inference.py` at the photo size: resize d1 [1, 1, S, S], then scale it to
    [0, 1] with its minimum and maximum."""
    return normalize_range(alpha_to_photo(d1[0, 0], height, width))


def import_isnet(source: str):
    """`IS-Net/models/isnet.py` of the DIS checkout, by its file path."""
    path = Path(source) / "models" / "isnet.py"
    spec = importlib.util.spec_from_file_location("dis_isnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(source: str, checkpoint: Path):
    """`ISNetDIS()` with the weights of `checkpoint`, in evaluation mode, on the CPU."""
    import torch

    isnet = import_isnet(source)
    model = isnet.ISNetDIS()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def subject_d1_module(model):
    """Export wrapper: `pixel_values` -> the sigmoid of side output d1 [1, 1, S, S] (`result[0][0]`
    of `Inference.py`). The other side outputs are not graph outputs."""
    import torch

    class SubjectD1(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            side_outputs, _ = self.inner(pixel_values)
            return side_outputs[0]

    return SubjectD1(model).eval()
