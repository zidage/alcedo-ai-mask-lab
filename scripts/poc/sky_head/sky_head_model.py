"""Fine sky head on the frozen SAM 2.1 tiny encoder (issue #10, README U17).

Inputs (the outputs of point_encoder.onnx and its input, so the product runs no second encoder):

  image_embed       [B, 256, 64, 64]     semantics (stride 16)
  high_res_feats_1  [B, 64, 128, 128]    stride 8
  high_res_feats_0  [B, 32, 256, 256]    stride 4
  pixel_values      [B, 3, 1024, 1024]   the normalized encoder input (edges)

Output: sky_logits [B, 1, 1024, 1024] (sigmoid = sky alpha) and two coarse logits for deep
supervision (training only): [B, 1, 256, 256] and [B, 1, 64, 64].

Structure: a semantic trunk on image_embed; a decoder that doubles the resolution four times and
at each step concatenates the SAM 2.1 feature or an RGB feature of the same resolution (the
global semantic branch and the local structure branch of MaSSFormer, CVPR 2025). The RGB branch
is a small convolution stack on pixel_values (1024, 512, 256). The edge comes from the model
output; there is no post-processing (U3).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EMBED_CHANNELS = 256
HIGH_RES_1_CHANNELS = 64
HIGH_RES_0_CHANNELS = 32
GROUPS = 8


def conv_block(in_channels: int, out_channels: int, stride: int = 1,
               dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation,
                  dilation=dilation, bias=False),
        nn.GroupNorm(min(GROUPS, out_channels), out_channels),
        nn.GELU())


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_block(channels, channels, dilation=dilation),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(min(GROUPS, channels), channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.body(x))


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of a [B, C, H, W] tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class UpFuse(nn.Module):
    """x2 bilinear upsampling, concatenation with a skip feature, two convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(conv_block(in_channels + skip_channels, out_channels),
                                  conv_block(out_channels, out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class SkyAlphaHead(nn.Module):
    def __init__(self, widths: tuple[int, int, int, int, int] = (128, 96, 64, 32, 16)) -> None:
        super().__init__()
        w64, w128, w256, w512, w1024 = widths
        self.trunk = nn.Sequential(
            LayerNorm2d(EMBED_CHANNELS),
            nn.Conv2d(EMBED_CHANNELS, w64, 1, bias=False), nn.GroupNorm(GROUPS, w64), nn.GELU(),
            ResidualBlock(w64, 1), ResidualBlock(w64, 2), ResidualBlock(w64, 4))
        self.rgb_1024 = nn.Sequential(conv_block(3, w1024), conv_block(w1024, w1024))
        self.rgb_512 = nn.Sequential(conv_block(w1024, w512, stride=2), conv_block(w512, w512))
        self.rgb_256 = nn.Sequential(conv_block(w512, w256, stride=2), conv_block(w256, w256))
        self.up_128 = UpFuse(w64, HIGH_RES_1_CHANNELS, w128)
        self.up_256 = UpFuse(w128, HIGH_RES_0_CHANNELS + w256, w256)
        self.up_512 = UpFuse(w256, w512, w512)
        self.up_1024 = UpFuse(w512, w1024, w1024)
        self.out_1024 = nn.Conv2d(w1024, 1, 1)
        self.out_256 = nn.Conv2d(w256, 1, 1)
        self.out_64 = nn.Conv2d(w64, 1, 1)

    def forward(self, image_embed: torch.Tensor, high_res_feats_0: torch.Tensor,
                high_res_feats_1: torch.Tensor, pixel_values: torch.Tensor):
        x64 = self.trunk(image_embed)
        rgb_1024 = self.rgb_1024(pixel_values)
        rgb_512 = self.rgb_512(rgb_1024)
        rgb_256 = self.rgb_256(rgb_512)
        x128 = self.up_128(x64, high_res_feats_1)
        x256 = self.up_256(x128, torch.cat([high_res_feats_0, rgb_256], dim=1))
        x512 = self.up_512(x256, rgb_512)
        x1024 = self.up_1024(x512, rgb_1024)
        return self.out_1024(x1024), self.out_256(x256), self.out_64(x64)


class ExportedSkyHead(nn.Module):
    """The graph of sky_head.onnx: the alpha only."""

    def __init__(self, head: SkyAlphaHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, image_embed, high_res_feats_0, high_res_feats_1, pixel_values):
        logits, _, _ = self.head(image_embed, high_res_feats_0, high_res_feats_1, pixel_values)
        return torch.sigmoid(logits)


def build_encoder(sam2_repo: str, checkpoint: str):
    """The frozen SAM 2.1 tiny encoder (sam2_graphs.PointEncoder, the graph of
    point_encoder.onnx). Returns (image_embed, high_res_feats_0, high_res_feats_1)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai_models" / "sam2"))
    import sam2_graphs as graphs

    graphs.add_repository_to_path(sam2_repo)
    encoder = graphs.PointEncoder(graphs.build_model("tiny", checkpoint)).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
