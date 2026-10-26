"""Training objective of the fine sky head (issue #10).

All terms use the weight map (1 = use the pixel, 0 = ignore it):

  alpha_l1       mean |alpha - target| at 1024
  gradient_l1    mean |Sobel(alpha) - Sobel(target)| at 1024 (edge position and sharpness)
  laplacian_l1   L1 of the Laplacian pyramid difference, LAPLACIAN_LEVELS levels, level k
                 weighted by 2^k (the matting loss of the matting literature, for example
                 Hou and Liu 2019)
  coarse_bce     BCE of the 256 and 64 deep supervision logits against the area-averaged target

total = alpha_l1 + W_GRADIENT * gradient_l1 + W_LAPLACIAN * laplacian_l1 + W_COARSE * coarse_bce
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

W_GRADIENT = 1.0
W_LAPLACIAN = 1.0
W_COARSE = 0.25
LAPLACIAN_LEVELS = 5


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def sobel(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                            device=x.device, dtype=x.dtype) / 8.0
    kernels = torch.stack([kernel_x, kernel_x.t()])[:, None]
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernels)


def gauss_down(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype)
    kernel = (kernel[:, None] * kernel[None, :]) / 256.0
    x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), kernel[None, None])
    return x[..., ::2, ::2]


def laplacian_pyramid(x: torch.Tensor) -> list[torch.Tensor]:
    levels = []
    for _ in range(LAPLACIAN_LEVELS - 1):
        down = gauss_down(x)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        levels.append(x - up)
        x = down
    levels.append(x)
    return levels


def weight_pyramid(weight: torch.Tensor) -> list[torch.Tensor]:
    """A level pixel is used only when all its source pixels are used (min pooling)."""
    levels = [weight]
    for _ in range(LAPLACIAN_LEVELS - 1):
        levels.append(-F.max_pool2d(-levels[-1], 2))
    return levels


def coarse_target(target: torch.Tensor, weight: torch.Tensor, size: int):
    factor = target.shape[-1] // size
    return (F.avg_pool2d(target, factor), -F.max_pool2d(-weight, factor))


def sky_head_loss(outputs, target: torch.Tensor, weight: torch.Tensor) -> dict:
    logits, logits_256, logits_64 = (o.float() for o in outputs)
    alpha = torch.sigmoid(logits)
    terms = {"alpha_l1": weighted_mean((alpha - target).abs(), weight)}
    gradient_weight = -F.max_pool2d(-weight, 3, stride=1, padding=1)
    terms["gradient_l1"] = weighted_mean((sobel(alpha) - sobel(target)).abs(),
                                         gradient_weight.expand(-1, 2, -1, -1))
    laplacian = 0.0
    for level, (a, t, w) in enumerate(zip(laplacian_pyramid(alpha), laplacian_pyramid(target),
                                          weight_pyramid(weight))):
        laplacian = laplacian + (2.0 ** level) * weighted_mean((a - t).abs(), w)
    terms["laplacian_l1"] = laplacian
    coarse = 0.0
    for coarse_logits in (logits_256, logits_64):
        t, w = coarse_target(target, weight, coarse_logits.shape[-1])
        coarse = coarse + weighted_mean(
            F.binary_cross_entropy_with_logits(coarse_logits, t, reduction="none"), w)
    terms["coarse_bce"] = coarse
    terms["total"] = (terms["alpha_l1"] + W_GRADIENT * terms["gradient_l1"]
                      + W_LAPLACIAN * terms["laplacian_l1"] + W_COARSE * terms["coarse_bce"])
    return terms
