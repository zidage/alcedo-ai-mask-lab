"""Shared helpers of the S3OD scripts (issue #6): model loading, preprocessing, and postprocessing.

The model code comes from the S3OD inference package (`KupynOrest/s3od`, directory `s3od/s3od`,
argument `--source`). `load_model` builds `DPTSegmentation` with the arguments of
`s3od.predictor.BackgroundRemoval._load_model` (DINOv3 ViT-B/16 encoder, DPT decoder, three mask
outputs and three IoU outputs) and loads the `state_dict` of `s3od.pt` with `weights_only=True`.

The functions without PyTorch (`letterbox`, `select_mask`, `unpad`) are also used by the DirectML
measurement in the MVP environment.

Preprocessing rule of the source (`BackgroundRemoval._preprocess`, `s3od.utils.get_pad_info`):

- scale the RGB photo so that its long edge is S (S = 1024), `round` of each side, OpenCV
  `INTER_LINEAR`;
- put it in the middle of an S x S black image (padding `(S - side) // 2` on the top and the left);
- `(value / 255 - mean) / std` with the ImageNet mean and std, NCHW, float32.

Postprocessing rule of the source (`BackgroundRemoval.remove_background`): sigmoid of the masks and
of the IoU scores, remove the padding, resize the masks to the photo size (bilinear,
`align_corners=False`, `antialias=True`), take the mask with the largest IoU score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 1024
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
MASK_COUNT = 3
CHECKPOINT_NAME = "s3od.pt"


def pad_info(height: int, width: int, size: int = IMAGE_SIZE) -> dict:
    """`s3od.utils.get_pad_info` for a photo of height x width."""
    scale = size / max(height, width)
    new_height = round(height * scale)
    new_width = round(width * scale)
    return {"height_pad": (size - new_height) // 2, "width_pad": (size - new_width) // 2,
            "original_size": (height, width), "resized_size": (new_height, new_width)}


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    """RGB uint8 H x W x 3 -> `pixel_values` [1, 3, S, S] float32 and the pad information."""
    info = pad_info(image.shape[0], image.shape[1], size)
    resized = cv2.resize(image, info["resized_size"][::-1], interpolation=cv2.INTER_LINEAR)
    height, width = resized.shape[:2]
    top, left = info["height_pad"], info["width_pad"]
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[top:top + height, left:left + width] = resized
    normalized = (padded.astype(np.float32) / 255.0 - np.array(NORMALIZE_MEAN)) \
        / np.array(NORMALIZE_STD)
    return normalized.astype(np.float32).transpose(2, 0, 1)[None].copy(), info


def unpad(masks: np.ndarray, info: dict) -> np.ndarray:
    """[..., S, S] -> [..., resized height, resized width] (`s3od.utils.remove_padding`)."""
    height, width = info["resized_size"]
    top, left = info["height_pad"], info["width_pad"]
    return masks[..., top:top + height, left:left + width]


def select_mask(masks: np.ndarray, iou_scores: np.ndarray) -> tuple[np.ndarray, int]:
    """`masks` [1, 3, S, S] and `iou_scores` [1, 3] -> the mask with the largest score and its
    index (`pred_ious.argmax()`)."""
    index = int(np.argmax(iou_scores[0]))
    return masks[0, index], index


def load_rgb(path: Path) -> np.ndarray:
    """The photo as RGB uint8, as `BackgroundRemoval.remove_background` reads a PIL image."""
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def import_source(source: str):
    """Import the S3OD package from the checkout directory `s3od/s3od` of KupynOrest/s3od."""
    root = Path(source).resolve()
    if not (root / "s3od" / "model.py").is_file():
        raise FileNotFoundError(f"{root} has no s3od/model.py; pass the s3od/s3od directory "
                                "of the KupynOrest/s3od checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import s3od.model
    import s3od.predictor

    return s3od


def load_model(source: str, checkpoint: Path):
    """`DPTSegmentation` with the weights of `checkpoint`, in eval mode, on the CPU."""
    import torch

    s3od = import_source(source)
    model = s3od.model.DPTSegmentation(num_classes=1, num_outputs=MASK_COUNT,
                                       encoder_name="dinov3_base", features=256, use_bn=True,
                                       use_clstoken=False)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["state_dict"]
    state = s3od.predictor.BackgroundRemoval._remap_state_dict(state, model)
    model.load_state_dict(state, strict=True)
    return model.eval()


def subject_masks_module(model):
    """`pixel_values` -> (`masks` [1, 3, S, S], `iou_scores` [1, 3]), both after the sigmoid."""
    import torch

    class SubjectMasks(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            outputs = self.inner(pixel_values)
            return outputs["pred_masks"].sigmoid(), outputs["pred_iou"].sigmoid()

    return SubjectMasks(model).eval()


def masks_to_photo(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Unpadded masks [N, h, w] -> [N, height, width] with the resize of the source (PyTorch
    bilinear, `align_corners=False`, `antialias=True`)."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(masks, np.float32))[None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False,
                            antialias=True)
    return resized[0].numpy()
