"""Training objective of the people and sky head (README section 5, step 1).

  total = w_sky_bce   * sky_bce        per-cell binary cross-entropy against the sky share
        + w_sky_dice  * sky_dice       soft Dice of the sky map per image
        + w_center    * center_focal   CenterNet focal loss of the person center heatmap
        + w_offset    * offset_l1      L1 of the center offset at the person center cells
        + w_size      * size_l1        L1 of the log box size at the person center cells
        + w_presence  * presence_bce   binary cross-entropy of "sky present", "person present"

Cells without a target (`sky_valid`, `heatmap_valid` False) and ignored presence labels (-1) do
not count. The center focal loss and the box losses are averaged over the person centers of the
batch; the other terms over their valid cells or images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = {"sky_bce": 1.0, "sky_dice": 1.0, "center_focal": 1.0, "offset_l1": 1.0,
                   "size_l1": 1.0, "presence_bce": 1.0}
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0


def sky_losses(logits: torch.Tensor, coverage: torch.Tensor,
               valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [B, 1, 64, 64]; coverage uint8 [B, 64, 64] (share x 255); valid bool [B, 64, 64]."""
    target = coverage.float() / 255.0
    weight = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp(min=1.0)
    probability = torch.sigmoid(logits[:, 0]) * weight
    target = target * weight
    overlap = (probability * target).sum((1, 2))
    dice = 1.0 - (2.0 * overlap + 1.0) / (probability.sum((1, 2)) + target.sum((1, 2)) + 1.0)
    return bce, dice.mean()


def center_focal_loss(logits: torch.Tensor, heatmap: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss. heatmap [B, 64, 64] has the value 1 at each center."""
    logits = logits[:, 0]
    log_p, log_not_p = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = log_p.exp()
    positive = (heatmap >= 1.0).float()
    negative = (1.0 - positive) * valid.float()
    positive_loss = -((1.0 - p) ** FOCAL_ALPHA) * log_p * positive
    negative_loss = -((1.0 - heatmap) ** FOCAL_BETA) * (p ** FOCAL_ALPHA) * log_not_p * negative
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1.0)


def gather_cells(maps: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """maps [B, C, 64, 64], cells [B, K] flat indices (>= 0) -> [B, K, C]."""
    flat = maps.flatten(2)
    index = cells[:, None, :].expand(-1, flat.shape[1], -1)
    return flat.gather(2, index).permute(0, 2, 1)


def box_losses(outputs: dict[str, torch.Tensor],
               batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    cells = batch["person_cell"]
    mask = (cells >= 0).float()[..., None]
    cells = cells.clamp(min=0).long()
    count = mask.sum().clamp(min=1.0)
    offset = gather_cells(outputs["center_offset"], cells)
    log_size = gather_cells(outputs["log_size"], cells)
    offset_l1 = ((offset - batch["person_offset"]).abs() * mask).sum() / count
    size_l1 = ((log_size - batch["person_log_size"]).abs() * mask).sum() / count
    return offset_l1, size_l1


def presence_loss(logits: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """logits [B, 2]; presence int8 [B, 2] with 1, 0, or -1 (ignored)."""
    valid = (presence >= 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, presence.clamp(min=0).float(),
                                             reduction="none")
    return (bce * valid).sum() / valid.sum().clamp(min=1.0)


def total_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
               weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    sky_bce, sky_dice = sky_losses(outputs["sky_logits"], batch["sky_coverage"],
                                   batch["sky_valid"])
    offset_l1, size_l1 = box_losses(outputs, batch)
    terms = {
        "sky_bce": sky_bce,
        "sky_dice": sky_dice,
        "center_focal": center_focal_loss(outputs["center_logits"], batch["center_heatmap"],
                                          batch["heatmap_valid"]),
        "offset_l1": offset_l1,
        "size_l1": size_l1,
        "presence_bce": presence_loss(outputs["presence_logits"], batch["presence"]),
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, {name: float(value.detach()) for name, value in terms.items()}
