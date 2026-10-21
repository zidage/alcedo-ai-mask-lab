"""People and sky head on the frozen SAM 2.1 tiny `image_embed` (README section 5, step 1).

  input           image_embed      [B, 256, 64, 64]  (point_encoder.onnx output)
  outputs         sky_logits       [B, 1, 64, 64]    sky share of each cell (sigmoid)
                  center_logits    [B, 1, 64, 64]    person center heatmap (sigmoid)
                  center_offset    [B, 2, 64, 64]    person center inside the cell, x and y in [0, 1)
                  log_size         [B, 2, 64, 64]    log of the person box width and height in cells
                  presence_logits  [B, 2]            sky present, person present (sigmoid)

Layers: LayerNorm over the channels of each cell, a 1 x 1 reduction to `width` channels, and
residual 3 x 3 blocks with dilations 1, 2, 4 (receptive field 15 cells). The presence output reads
the mean and the maximum of the trunk features over the grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 64
# Initial center probability 0.1 (CenterNet): log(0.1 / 0.9).
CENTER_PRIOR_BIAS = -2.19


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis of [B, C, H, W] (the SAM 2 LayerNorm2d)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(groups, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.norm(self.conv(x)))


class PeopleSkyHead(nn.Module):
    def __init__(self, in_channels: int = 256, width: int = 96,
                 dilations: tuple[int, ...] = (1, 2, 4), groups: int = 8) -> None:
        super().__init__()
        self.config = {"in_channels": in_channels, "width": width,
                       "dilations": list(dilations), "groups": groups}
        self.input_norm = LayerNorm2d(in_channels)
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False),
                                    nn.GroupNorm(groups, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, d, groups) for d in dilations])
        self.sky = nn.Conv2d(width, 1, 1)
        self.center = nn.Conv2d(width, 1, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.presence = nn.Linear(2 * width, 2)
        nn.init.constant_(self.center.bias, CENTER_PRIOR_BIAS)

    def forward(self, image_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.reduce(self.input_norm(image_embed)))
        box = self.box(features)
        pooled = torch.cat([features.mean((2, 3)), features.amax((2, 3))], dim=1)
        return {
            "sky_logits": self.sky(features),
            "center_logits": self.center(features),
            "center_offset": torch.sigmoid(box[:, :2]),
            "log_size": box[:, 2:],
            "presence_logits": self.presence(pooled),
        }


OUTPUT_NAMES = ("sky_logits", "center_logits", "center_offset", "log_size", "presence_logits")


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N, 4] x [M, 4] boxes (x0, y0, x1, y1) -> [N, M] IoU."""
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def suppress_overlapping_boxes(boxes: torch.Tensor, nms_iou: float) -> torch.Tensor:
    """Greedy suppression of boxes sorted by score, highest first: a box is removed when its IoU
    with a kept box of higher score is at least `nms_iou`. Returns the kept indices."""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    overlap = box_iou(boxes, boxes)
    keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
    for i in range(len(boxes)):
        if keep[i]:
            later = torch.arange(len(boxes), device=boxes.device) > i
            keep &= ~(later & (overlap[i] >= nms_iou))
    return torch.nonzero(keep)[:, 0]


def decode_people(outputs: dict[str, torch.Tensor], top_k: int = 20,
                  min_score: float = 0.05,
                  nms_iou: float | None = None) -> list[dict[str, torch.Tensor]]:
    """Person boxes per image: local maxima (3 x 3) of the center heatmap, highest first.

    With `nms_iou`, a box whose IoU with a box of higher score is at least `nms_iou` is removed
    (a large person gives several local maxima). None keeps all local maxima (the measurements of
    the first Colab run). Returns per image `boxes` [K, 4] (x0, y0, x1, y1 in cells) and
    `scores` [K]."""
    heatmap = torch.sigmoid(outputs["center_logits"][:, 0])
    peaks = heatmap * (F.max_pool2d(heatmap[:, None], 3, stride=1, padding=1)[:, 0] == heatmap)
    batch = heatmap.shape[0]
    scores, flat = peaks.reshape(batch, -1).topk(top_k, dim=1)
    ys, xs = (flat // GRID).float(), (flat % GRID).float()
    offset = outputs["center_offset"].reshape(batch, 2, -1)
    size = outputs["log_size"].reshape(batch, 2, -1).exp()
    results = []
    for b in range(batch):
        center_x = xs[b] + offset[b, 0, flat[b]]
        center_y = ys[b] + offset[b, 1, flat[b]]
        width, height = size[b, 0, flat[b]], size[b, 1, flat[b]]
        boxes = torch.stack([center_x - width / 2, center_y - height / 2,
                             center_x + width / 2, center_y + height / 2], dim=1)
        keep = scores[b] >= min_score
        boxes, image_scores = boxes[keep], scores[b][keep]
        if nms_iou is not None:
            kept = suppress_overlapping_boxes(boxes, nms_iou)
            boxes, image_scores = boxes[kept], image_scores[kept]
        results.append({"boxes": boxes, "scores": image_scores})
    return results


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
