"""Matting Anything (MAM) inference in PyTorch: SAM ViT-B with a box or point prompt, then the
Mask-to-Matte (M2M) module and the MAM output fusion.

Input:  the MAM source tree (`--mam-source`, the clone of SHI-Labs/Matting-Anything, for the
        modified `segment_anything` package and `networks/m2ms/conv_sam.py`) and the checkpoint
        `mam_sam_vitb.pth` of huggingface.co/shi-labs/Matting-Anything (`state_dict` with the
        frozen SAM ViT-B as `module.seg_model.*` and the M2M module as `module.m2m.*`).
Output: `MamResult`: the alpha matte at the image size, the binary SAM mask that guides M2M, and
        the SAM IoU scores.

Step: MAM proof of concept (`check_mam.py`).

The computation follows `gradio_app.py` and `inference_benchmark.py` of the MAM repository
(commit eb01e1c):

1. Resize the long edge to 1024 (`ResizeLongestSide`), normalize with the SAM mean and standard
   deviation, pad to 1024 x 1024 at the bottom and the right.
2. SAM image encoder; prompt encoder and mask decoder with `multimask_output=True`; the selected
   output has the largest IoU score. The guide mask is its 256 x 256 logits > 0; the photo-size SAM
   mask (`post_mask`) is the same logits upsampled as `Sam.postprocess_masks` and > 0.
3. M2M on (image embedding, padded image, guide mask) gives alpha at output strides 8, 4, and 1
   on the 1024 grid; each is unpadded and upsampled (bilinear) to the image size.
4. Output fusion (MAM paper section 3, "iterative refinement"), one of the three modes of the
   repository (`GUIDANCE_MODES`):
   - `alpha` (gradio_app.py default, and inference_benchmark.py `--postprocess`): alpha = os8,
     set to the SAM mask outside the bounding rectangle of the SAM mask; then os4 where the
     dilated (ellipse 10 px) area of alpha >= 1/255 is, and os1 where the dilated (ellipse 5 px)
     area of the result is; these two steps do not replace pixels with alpha > 1 - 1/255.
   - `mask_one_side` (gradio_app.py "mask", inference_benchmark.py `--maskguide`): alpha = the SAM
     mask; os8 replaces it in the band outside the SAM mask (the background minus its erosion by an
     ellipse of 5 px); then os4 and os1 as in `alpha`.
   - `mask_two_side` (inference_benchmark.py `--maskguide --twoside`): alpha = the SAM mask; os8
     replaces it in the band on both sides of the SAM edge (all pixels minus the erosions of the
     mask and of the background by an ellipse of 5 px); then os4 in the dilated (10 px) area of
     1/255 <= alpha <= 1 - 1/255, and os1 in the dilated (5 px) area of the result.
   The widths are the defaults of both scripts (os8 10, os4 20, os1 10; the dilation and the
   erosion use half of the width).

Several points form one prompt (the demo mode "merge"), as the MVP gives the 3 points of a sky
region to one SAM 2.1 decoder call.

`target = "complement"` gives M2M the complement of the selected SAM mask (guide mask and SAM mask
are 1 - the selection): the matte of everything that is not the prompted region. For Sky, the
prompt selects the sky, M2M mattes the non-sky content, and Sky = 1 - alpha.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_SIZE = 1024
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)
# gradio_app.py: get_unknown_tensor_from_pred_oneside(rand_width=20 and 10, train_mode=False)
# dilates with an ellipse of rand_width // 2 pixels.
OS8_EROSION = 5
OS4_DILATION = 10
OS1_DILATION = 5
ALPHA_UNKNOWN_LOW = 1.0 / 255.0
GUIDANCE_MODES = ("alpha", "mask_one_side", "mask_two_side")
TARGETS = ("selection", "complement")


@dataclass
class MamResult:
    alpha: np.ndarray        # float32 [H, W] in [0, 1]
    sam_mask: np.ndarray     # float32 [H, W], 0 or 1: the SAM mask that guides M2M
    iou_scores: list[float]  # the three SAM IoU scores
    selected: int            # the SAM output with the largest IoU score
    milliseconds: float      # prompt, decoder, M2M, and fusion (not the encoder)


@dataclass
class EncodedImage:
    pixels: object           # torch [1, 3, 1024, 1024], normalized and padded
    embedding: object        # torch [1, 256, 64, 64]
    original_size: tuple[int, int]   # (H, W)
    input_size: tuple[int, int]      # (h, w) after the resize, before the padding
    milliseconds: float


class MamModel:
    def __init__(self, mam_source: Path, checkpoint: Path, threads: int = 4) -> None:
        import torch

        torch.set_num_threads(threads)
        for directory in (mam_source / "segment-anything", mam_source):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        from networks.m2ms import sam_decoder_deep

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        seg = {k[len("module.seg_model."):]: v.float() for k, v in state.items()
               if k.startswith("module.seg_model.")}
        m2m = {k[len("module.m2m."):]: v.float() for k, v in state.items()
               if k.startswith("module.m2m.")}
        if len(seg) + len(m2m) != len(state):
            raise ValueError(f"{checkpoint}: keys outside module.seg_model and module.m2m")
        self.sam = sam_model_registry["vit_b"](checkpoint=None)
        self.sam.load_state_dict(seg, strict=True)
        self.m2m = sam_decoder_deep(nc=256)
        self.m2m.load_state_dict(m2m, strict=True)
        self.sam.eval()
        self.m2m.eval()
        self.transform = ResizeLongestSide(MODEL_SIZE)
        self.torch = torch
        self.seg_state = seg

    def encode(self, rgb8: np.ndarray) -> EncodedImage:
        """Steps 1 and 2 (encoder) for an RGB uint8 [H, W, 3] image."""
        torch = self.torch
        F = torch.nn.functional
        started = time.perf_counter()
        image = torch.as_tensor(self.transform.apply_image(rgb8)).permute(2, 0, 1).float()
        mean = torch.tensor(PIXEL_MEAN).view(3, 1, 1)
        std = torch.tensor(PIXEL_STD).view(3, 1, 1)
        image = (image - mean) / std
        h, w = image.shape[-2:]
        pixels = F.pad(image, (0, MODEL_SIZE - w, 0, MODEL_SIZE - h))[None]
        with torch.no_grad():
            embedding = self.sam.image_encoder(pixels)
        return EncodedImage(pixels, embedding, tuple(rgb8.shape[:2]), (int(h), int(w)),
                            (time.perf_counter() - started) * 1000)

    def predict(self, encoded: EncodedImage, box: list[float] | None = None,
                points: list[list[float]] | None = None, guidance: str = "alpha",
                target: str = "selection") -> MamResult:
        """Steps 2 to 4 for one prompt. `box`: (x0, y0, x1, y1) in image pixels. `points`:
        (x, y) foreground points in image pixels, one prompt. `guidance`: one of
        GUIDANCE_MODES. `target`: one of TARGETS."""
        torch = self.torch
        F = torch.nn.functional
        if (box is None) == (points is None):
            raise ValueError("give a box or points")
        if guidance not in GUIDANCE_MODES or target not in TARGETS:
            raise ValueError(f"guidance {guidance!r}, target {target!r}")
        started = time.perf_counter()
        size = encoded.original_size
        with torch.no_grad():
            if box is not None:
                boxes = torch.as_tensor(self.transform.apply_boxes(np.array([box], float), size),
                                        dtype=torch.float)
                sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
            else:
                coords = torch.as_tensor(self.transform.apply_coords(np.array(points, float), size),
                                         dtype=torch.float)[None]
                labels = torch.ones((1, coords.shape[1]), dtype=torch.int)
                sparse, dense = self.sam.prompt_encoder(points=(coords, labels), boxes=None,
                                                        masks=None)
            low_res, iou = self.sam.mask_decoder(
                image_embeddings=encoded.embedding, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True)
            selected = int(iou[0].argmax())
            guide = (low_res[:, selected:selected + 1] > 0).float()
            post = self.sam.postprocess_masks(low_res, encoded.input_size, size)
            post = (post[:, selected:selected + 1] > 0).float()
            if target == "complement":
                guide, post = 1.0 - guide, 1.0 - post
            out = self.m2m(encoded.embedding, encoded.pixels, guide)
            h, w = encoded.input_size
            alphas = {key: F.interpolate(out[key][..., :h, :w], size, mode="bilinear",
                                         align_corners=False)[0, 0].numpy()
                      for key in ("alpha_os1", "alpha_os4", "alpha_os8")}
        sam_mask = post[0, 0].numpy()
        alpha = fuse_outputs(alphas, sam_mask, guidance)
        return MamResult(alpha=alpha, sam_mask=sam_mask,
                         iou_scores=[float(v) for v in iou[0]], selected=selected,
                         milliseconds=(time.perf_counter() - started) * 1000)


def ellipse_kernel(size: int) -> np.ndarray:
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def unknown_area_one_side(alpha: np.ndarray, dilation: int) -> np.ndarray:
    """`get_unknown_tensor_from_pred_oneside` at inference: the pixels with alpha >= 1/255,
    dilated with an ellipse of `dilation` pixels, without the pixels with alpha > 1 - 1/255."""
    import cv2

    area = (alpha >= ALPHA_UNKNOWN_LOW).astype(np.uint8)
    area = cv2.dilate(area, ellipse_kernel(dilation))
    area[alpha > 1.0 - ALPHA_UNKNOWN_LOW] = 0
    return area.astype(bool)


def unknown_area_two_side(alpha: np.ndarray, dilation: int) -> np.ndarray:
    """`get_unknown_tensor_from_pred` at inference: the pixels with
    1/255 <= alpha <= 1 - 1/255, dilated with an ellipse of `dilation` pixels."""
    import cv2

    area = ((alpha >= ALPHA_UNKNOWN_LOW) & (alpha <= 1.0 - ALPHA_UNKNOWN_LOW)).astype(np.uint8)
    return cv2.dilate(area, ellipse_kernel(dilation)).astype(bool)


def mask_band(sam_mask: np.ndarray, erosion: int, two_side: bool) -> np.ndarray:
    """`get_unknown_tensor_from_mask` (two side) and `get_unknown_tensor_from_mask_oneside` at
    inference: all pixels without the eroded background and without the (eroded, for two side)
    SAM mask."""
    import cv2

    mask = (sam_mask > 0.5).astype(np.uint8)
    band = np.ones(mask.shape, bool)
    band[cv2.erode(1 - mask, ellipse_kernel(erosion)) == 1] = False
    band[(cv2.erode(mask, ellipse_kernel(erosion)) if two_side else mask) == 1] = False
    return band


def fuse_outputs(alphas: dict[str, np.ndarray], sam_mask: np.ndarray,
                 guidance: str = "alpha") -> np.ndarray:
    """Step 4 in one of GUIDANCE_MODES. For `alpha`, an empty SAM mask has no bounding rectangle
    (the demo stops with an error there); the result is then the SAM mask (all 0)."""
    if guidance == "alpha":
        alpha = alphas["alpha_os8"].copy()
        rows, columns = np.nonzero(sam_mask)
        outside = np.ones(sam_mask.shape, bool)
        if rows.size:
            # get_unknown_box_from_mask: the slice [y_min:y_max, x_min:x_max] (end exclusive).
            outside[rows.min():rows.max(), columns.min():columns.max()] = False
        alpha[outside] = sam_mask[outside]
    else:
        alpha = sam_mask.astype(np.float32).copy()
        band = mask_band(sam_mask, OS8_EROSION, guidance == "mask_two_side")
        alpha[band] = alphas["alpha_os8"][band]
    unknown = unknown_area_two_side if guidance == "mask_two_side" else unknown_area_one_side
    for key, dilation in (("alpha_os4", OS4_DILATION), ("alpha_os1", OS1_DILATION)):
        area = unknown(alpha, dilation)
        alpha[area] = alphas[key][area]
    return alpha.astype(np.float32)
