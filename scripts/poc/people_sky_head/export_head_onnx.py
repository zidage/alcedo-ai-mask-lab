#!/usr/bin/env python3
"""Export the trained people and sky head to ONNX and compare it with PyTorch (step 1).

Input:  --checkpoint (best.pt of train_head.py), --val-dir (a split with image_embed.npy, for
        the comparison inputs).
Output: --out-dir/people_sky_head.onnx (input image_embed [1, 256, 64, 64], the outputs of
        head_model.OUTPUT_NAMES) and --out-dir/people_sky_head.json (hash, size, parameters,
        validation measurements of the checkpoint, comparison result).

Comparison: ONNX Runtime CPU against PyTorch CPU float32 on --compare-images validation images.
Each output must agree within an absolute difference of 1e-4; otherwise the script fails.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head_data as hd  # noqa: E402
import head_model as hm  # noqa: E402

MAX_ABS_DIFFERENCE = 1e-4


class OrderedOutputs(torch.nn.Module):
    """The head with a tuple output in OUTPUT_NAMES order (ONNX output order)."""

    def __init__(self, model: hm.PeopleSkyHead) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_embed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(image_embed)
        return tuple(outputs[name] for name in hm.OUTPUT_NAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compare-images", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import onnxruntime as ort

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = hm.PeopleSkyHead(config["in_channels"], config["width"], tuple(config["dilations"]),
                             config["groups"])
    model.load_state_dict(state["model"])
    model.eval()
    wrapper = OrderedOutputs(model).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "people_sky_head.onnx"
    example = torch.zeros(1, hd.EMBED_CHANNELS, hd.GRID, hd.GRID)
    with torch.no_grad():
        torch.onnx.export(wrapper, (example,), str(onnx_path), input_names=["image_embed"],
                          output_names=list(hm.OUTPUT_NAMES), opset_version=args.opset,
                          dynamo=False)

    embeds = np.load(hd.embed_path(Path(args.val_dir)), mmap_mode="r")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    differences = {name: 0.0 for name in hm.OUTPUT_NAMES}
    for index in range(min(args.compare_images, len(embeds))):
        embed = np.asarray(embeds[index:index + 1], np.float32)
        onnx_outputs = session.run(list(hm.OUTPUT_NAMES), {"image_embed": embed})
        with torch.no_grad():
            torch_outputs = wrapper(torch.from_numpy(embed))
        for name, onnx_value, torch_value in zip(hm.OUTPUT_NAMES, onnx_outputs, torch_outputs):
            difference = float(np.abs(onnx_value - torch_value.numpy()).max())
            differences[name] = max(differences[name], difference)
    worst = max(differences.values())
    data = onnx_path.read_bytes()
    hd.write_json(out_dir / "people_sky_head.json", {
        "step": "README section 5, step 1: people and sky head export",
        "onnx": {"path": onnx_path.name, "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest(), "opset": args.opset},
        "model": config,
        "parameters": hm.parameter_count(model),
        "checkpoint_epoch": state["epoch"],
        "validation": state["val"],
        "comparison": {"rule": "max absolute difference per output, ONNX Runtime CPU against "
                               "PyTorch CPU float32", "tolerance": MAX_ABS_DIFFERENCE,
                       "images": min(args.compare_images, len(embeds)),
                       "max_abs_difference": differences},
    })
    print(f"[export] {onnx_path}: {len(data) / 1e6:.2f} MB, max abs difference {worst:.2e}")
    if worst > MAX_ABS_DIFFERENCE:
        raise ValueError(f"ONNX and PyTorch outputs differ by {worst:.2e} "
                         f"(tolerance {MAX_ABS_DIFFERENCE}): {differences}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
