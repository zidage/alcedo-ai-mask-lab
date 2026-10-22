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
# Encoder tokens at IMAGE_SIZE: 1 class token, 4 register tokens, (1024 / 16)^2 patches.
ENCODER_TOKENS = 1 + 4 + (IMAGE_SIZE // 16) ** 2
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


ATTENTION_KERNELS = ("sdpa", "multi_head_attention")
CHUNKED_ATTENTION = "s3od_query_chunked"


def query_chunk_sizes(tokens: int, chunks: int) -> list[int]:
    """Sizes of `chunks` query chunks of `tokens` tokens; the first chunks get one more token."""
    if not 1 <= chunks <= tokens:
        raise ValueError(f"{chunks} query chunks for {tokens} tokens")
    return [tokens // chunks + (1 if index < tokens % chunks else 0) for index in range(chunks)]


_multi_head_attention_op = None


def register_multi_head_attention_schema() -> None:
    """Register the part of the `com.microsoft.MultiHeadAttention` schema of ONNX Runtime that the
    export uses (query, key, value; `num_heads`, `scale`). The onnx package has no schemas of the
    ONNX Runtime contrib operators, and the exporter needs one to write the node."""
    import onnx
    from onnx.defs import OpSchema

    parameter = OpSchema.FormalParameter
    onnx.defs.register_schema(OpSchema(
        "MultiHeadAttention", "com.microsoft", 1,
        doc="ONNX Runtime contrib operator (query [B, S, E], key and value [B, L, E]).",
        inputs=[parameter("query", "T"), parameter("key", "T"), parameter("value", "T")],
        outputs=[parameter("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
        attributes=[OpSchema.Attribute("num_heads", OpSchema.AttrType.INT, "heads"),
                    OpSchema.Attribute("scale", OpSchema.AttrType.FLOAT, "score scale")]))


def multi_head_attention_op():
    """`torch.ops.s3od.multi_head_attention(query, key, value, num_heads, scale)`: attention of
    [B, Sq, E] queries on [B, Sk, E] keys and values (E = heads x head size), the inputs and the
    output of ONNX Runtime `com.microsoft.MultiHeadAttention`. PyTorch computes it with
    `scaled_dot_product_attention`. Returns (the operator, its ONNX translation)."""
    global _multi_head_attention_op
    if _multi_head_attention_op is not None:
        return _multi_head_attention_op
    import onnxscript
    import torch
    import torch.nn.functional as F

    # The schema is explicit: `from __future__ import annotations` makes the annotations strings.
    @torch.library.custom_op("s3od::multi_head_attention", mutates_args=(),
                             schema="(Tensor query, Tensor key, Tensor value, int num_heads, "
                                    "float scale) -> Tensor")
    def multi_head_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             num_heads: int, scale: float) -> torch.Tensor:
        batch, length, width = query.shape

        def heads(tensor):
            return tensor.view(batch, -1, num_heads, width // num_heads).transpose(1, 2)

        output = F.scaled_dot_product_attention(heads(query), heads(key), heads(value),
                                                scale=scale)
        return output.transpose(1, 2).reshape(batch, length, width)

    @multi_head_attention.register_fake
    def _(query, key, value, num_heads, scale):
        return query.new_empty(query.shape)

    microsoft_attention = onnxscript.values.Opset("com.microsoft", 1)["MultiHeadAttention"]
    if microsoft_attention is None:
        register_multi_head_attention_schema()
        microsoft_attention = onnxscript.values.Opset("com.microsoft", 1)["MultiHeadAttention"]

    def multi_head_attention_onnx(query, key, value, num_heads: int, scale: float):
        return microsoft_attention(query, key, value, num_heads=num_heads, scale=scale)

    _multi_head_attention_op = (torch.ops.s3od.multi_head_attention.default,
                                multi_head_attention_onnx)
    return _multi_head_attention_op


def use_query_chunked_attention(model, chunks: int, kernel: str) -> dict:
    """Replace the attention of each encoder layer with an attention in `chunks` query chunks.

    Each query row has its own Softmax, so the attention of a query chunk on all keys gives the
    rows of the full attention for these queries; the concatenation of the chunks is the full
    output. The score tensor of one chunk is [B, heads, chunk, tokens] instead of
    [B, heads, tokens, tokens].

    Chunk k + 1 reads the output of chunk k: `query_{k+1} + 0 * max(output_k)`. The added value
    is 0 for each finite output. The dependency makes the chunks a sequence in the graph, so the
    DirectML graph can use the memory of one score tensor for all chunks (without it, DirectML
    keeps the score tensors of all chunks at the same time).

    kernel `sdpa`: `scaled_dot_product_attention` per chunk (the exporter writes MatMul, Softmax,
    MatMul). kernel `multi_head_attention`: `com.microsoft.MultiHeadAttention` per chunk
    (`multi_head_attention_op`). Returns the export settings (the translation table)."""
    import torch
    import torch.nn.functional as F
    from transformers import AttentionInterface

    if kernel not in ATTENTION_KERNELS:
        raise ValueError(f"attention kernel {kernel!r} is not one of {ATTENTION_KERNELS}")
    operator, translation = (multi_head_attention_op() if kernel == "multi_head_attention"
                             else (None, None))

    def query_chunked_attention(module, query, key, value, attention_mask, scaling=None,
                                dropout=0.0, **kwargs):
        if attention_mask is not None or dropout:
            raise ValueError("the query chunked attention has no mask and no dropout")
        batch, heads, tokens, size = query.shape
        if operator is not None:
            def tokens_last(tensor):
                return tensor.transpose(1, 2).reshape(batch, -1, heads * size)

            key, value = tokens_last(key), tokens_last(value)
        outputs = []
        for part in query.split(query_chunk_sizes(tokens, chunks), dim=2):
            if outputs:
                part = part + outputs[-1].max() * 0
            if operator is None:
                outputs.append(F.scaled_dot_product_attention(part, key, value, scale=scaling)
                               .transpose(1, 2))
            else:
                output = operator(tokens_last(part), key, value, heads, float(scaling))
                outputs.append(output.view(batch, -1, heads, size))
        return torch.cat(outputs, dim=1).contiguous(), None

    AttentionInterface.register(CHUNKED_ATTENTION, query_chunked_attention)
    model.encoder.config._attn_implementation = CHUNKED_ATTENTION
    return {"custom_translation_table": {operator: translation}} if operator is not None else {}


def attention_score_maxima(model, pixel_batches) -> list[float]:
    """The largest absolute attention score (query . key x scale) of each encoder layer over the
    pixel batches, with the attention of the source (PyTorch, float32)."""
    import torch
    from transformers import AttentionInterface

    maxima: list[float] = []
    layer = [0]

    def scored_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                         **kwargs):
        scores = (query @ key.transpose(-1, -2)) * scaling
        if layer[0] == len(maxima):
            maxima.append(0.0)
        maxima[layer[0]] = max(maxima[layer[0]], float(scores.abs().max()))
        layer[0] += 1
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value,
                                                                  scale=scaling)
        return output.transpose(1, 2).contiguous(), None

    previous = model.encoder.config._attn_implementation
    AttentionInterface.register("s3od_score_maxima", scored_attention)
    model.encoder.config._attn_implementation = "s3od_score_maxima"
    try:
        for pixels in pixel_batches:
            layer[0] = 0
            model(torch.from_numpy(pixels))
    finally:
        model.encoder.config._attn_implementation = previous
    return maxima


def masks_to_photo(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Unpadded masks [N, h, w] -> [N, height, width] with the resize of the source (PyTorch
    bilinear, `align_corners=False`, `antialias=True`)."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(masks, np.float32))[None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False,
                            antialias=True)
    return resized[0].numpy()
