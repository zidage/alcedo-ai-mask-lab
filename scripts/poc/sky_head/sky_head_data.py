"""Shared layout of the fine sky head (issue #10): prepared data, sample loading, augmentation.

Prepared data (prepare_mass13k.py), one directory per split:

  <prepared>/<split>/images/<stem>.jpg    photo, long edge PREPARED_LONG_EDGE, JPEG quality 95
  <prepared>/<split>/alpha/<stem>.png     sky coverage of each pixel, uint8 (255 = all sky), the
                                          area average of the hard 4K sky label
  <prepared>/<split>/weight/<stem>.png    only when the label has ignored pixels: 255 = use,
                                          0 = ignore (the loss and the metrics skip the pixel)
  <prepared>/<split>/manifest.json        source member names, sizes, sky share, ignored share

A training sample has the layout of the product (README section 4.1): the photo is stretched to
1024 x 1024 (`sam2_graphs.preprocess_image`: bilinear with antialias, ImageNet normalization), so
the target alpha is also stretched to 1024 x 1024 (area average).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

PREPARED_LONG_EDGE = 2048
MODEL_SIZE = 1024
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# Training augmentation (the photo before the stretch to 1024 x 1024).
CROP_SCALE = (0.6, 1.0)       # side of the crop as a share of the photo side
CROP_ASPECT_JITTER = 0.1      # the two sides are scaled independently within this share
FLIP_PROBABILITY = 0.5
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.02
GAMMA = (0.8, 1.25)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8", newline="\n")


def split_items(prepared: Path, split: str) -> list[dict]:
    manifest = read_json(prepared / split / "manifest.json")
    return [dict(item, split_dir=str(prepared / split)) for item in manifest["items"]]


def load_sample(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Photo uint8 [H, W, 3], alpha float32 [H, W] in [0, 1], weight float32 [H, W] (0 or 1)."""
    from PIL import Image

    split_dir = Path(item["split_dir"])
    stem = item["stem"]
    with Image.open(split_dir / "images" / f"{stem}.jpg") as image:
        photo = np.array(image.convert("RGB"))
    with Image.open(split_dir / "alpha" / f"{stem}.png") as image:
        alpha = np.asarray(image, dtype=np.float32) / 255.0
    if item.get("has_weight"):
        with Image.open(split_dir / "weight" / f"{stem}.png") as image:
            weight = (np.asarray(image) >= 128).astype(np.float32)
    else:
        weight = np.ones(alpha.shape, dtype=np.float32)
    if photo.shape[:2] != alpha.shape or alpha.shape != weight.shape:
        raise ValueError(f"{stem}: photo {photo.shape}, alpha {alpha.shape}, weight "
                         f"{weight.shape} differ")
    return photo, alpha, weight


def resize_target(values: np.ndarray, size: int = MODEL_SIZE) -> np.ndarray:
    """Area average to size x size (the stretch of the product)."""
    import cv2

    return cv2.resize(values, (size, size), interpolation=cv2.INTER_AREA)


def resize_weight(weight: np.ndarray, size: int = MODEL_SIZE) -> np.ndarray:
    """A model pixel is used only when all its photo pixels are used."""
    return (resize_target(weight, size) >= 0.999).astype(np.float32)


def model_pixels(photo: np.ndarray):
    """uint8 photo [H, W, 3] -> normalized tensor [3, 1024, 1024] (sam2_graphs.preprocess_image)."""
    import torch
    from torchvision.transforms import Normalize, Resize

    tensor = torch.from_numpy(np.ascontiguousarray(photo)).permute(2, 0, 1).float() / 255.0
    tensor = Resize((MODEL_SIZE, MODEL_SIZE))(tensor)
    return Normalize(list(NORMALIZE_MEAN), list(NORMALIZE_STD))(tensor)


def augment(photo: np.ndarray, alpha: np.ndarray, weight: np.ndarray,
            generator: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop (scale and aspect), horizontal flip, and photometric changes of the photo."""
    from PIL import Image, ImageEnhance

    height, width = alpha.shape
    scale = generator.uniform(*CROP_SCALE)
    crop_w = min(width, round(width * scale * generator.uniform(1 - CROP_ASPECT_JITTER,
                                                                 1 + CROP_ASPECT_JITTER)))
    crop_h = min(height, round(height * scale * generator.uniform(1 - CROP_ASPECT_JITTER,
                                                                  1 + CROP_ASPECT_JITTER)))
    x = generator.randint(0, width - crop_w)
    y = generator.randint(0, height - crop_h)
    window = (slice(y, y + crop_h), slice(x, x + crop_w))
    photo, alpha, weight = photo[window], alpha[window], weight[window]
    if generator.random() < FLIP_PROBABILITY:
        photo, alpha, weight = photo[:, ::-1], alpha[:, ::-1], weight[:, ::-1]

    image = Image.fromarray(np.ascontiguousarray(photo))
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(1 - BRIGHTNESS,
                                                                     1 + BRIGHTNESS))
    image = ImageEnhance.Contrast(image).enhance(generator.uniform(1 - CONTRAST, 1 + CONTRAST))
    image = ImageEnhance.Color(image).enhance(generator.uniform(1 - SATURATION, 1 + SATURATION))
    photo = np.asarray(image, dtype=np.float32) / 255.0
    hue = generator.uniform(-HUE, HUE)
    if hue:
        photo = shift_hue(photo, hue)
    photo = np.clip(photo, 0.0, 1.0) ** generator.uniform(*GAMMA)
    photo = np.round(photo * 255.0).astype(np.uint8)
    return photo, np.ascontiguousarray(alpha), np.ascontiguousarray(weight)


def shift_hue(photo: np.ndarray, shift: float) -> np.ndarray:
    """Hue rotation of an RGB float photo in [0, 1] by `shift` turns."""
    import cv2

    hsv = cv2.cvtColor(photo.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + shift * 360.0) % 360.0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class SkyDataset:
    """Samples of one or more prepared splits as (pixels, alpha, weight) tensors at 1024."""

    def __init__(self, items: list[dict], train: bool, seed: int = 0) -> None:
        self.items, self.train, self.seed, self.epoch = items, train, seed, 0

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        import torch

        photo, alpha, weight = load_sample(self.items[index])
        if self.train:
            generator = random.Random(f"{self.seed}-{self.epoch}-{index}")
            photo, alpha, weight = augment(photo, alpha, weight, generator)
        pixels = model_pixels(photo)
        return (pixels, torch.from_numpy(resize_target(alpha))[None],
                torch.from_numpy(resize_weight(weight))[None])
