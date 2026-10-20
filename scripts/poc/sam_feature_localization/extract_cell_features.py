#!/usr/bin/env python3
"""Cell features of the SAM 2.1 tiny encoder on the 64 x 64 encoder grid (issue #1, step 3).

Input:  --photos (the quality photos at the 4096 px application input, E4 input set) and
        --onnx-dir with `point_encoder.onnx` (SAM 2.1 tiny, float32, E1/E4).
Output: --out-dir/<stem>.npz per photo with three float32 arrays on the 64 x 64 grid:

  image_embed       [256, 64, 64]  the encoder output as it is (stride 16)
  high_res_feats_1  [64, 64, 64]   high_res_feats_1 [64, 128, 128], average of each 2 x 2 block
  high_res_feats_0  [32, 64, 64]   high_res_feats_0 [32, 256, 256], average of each 4 x 4 block

and --out-dir/features.json with the encoder time per photo, the model hash, the machine, and
the peak working set. The encoder runs once per photo on ONNX Runtime CPU with --threads intra-op
threads and the CPU memory arena off. The preprocessing is `sam2_graphs.preprocess_image` (stretch
to 1024 x 1024, bilinear with antialias, ImageNet normalization), the same as E1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_localization as fl  # noqa: E402


def average_blocks(features: np.ndarray, block: int) -> np.ndarray:
    """[C, H, W] -> [C, H / block, W / block], mean of each block x block square."""
    channels, height, width = features.shape
    return features.reshape(channels, height // block, block, width // block, block).mean(
        axis=(2, 4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    fl.limit_threads(args.threads)
    fl.add_sam2_scripts_to_path()
    import onnxruntime as ort
    from PIL import Image

    import sam2_graphs as graphs

    encoder_path = Path(args.onnx_dir) / "point_encoder.onnx"
    session = fl.create_cpu_session(encoder_path, args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = []
    for path in sorted(Path(args.photos).glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        if max(image.size) > 4096:
            raise ValueError(f"{path.name}: long edge {max(image.size)} px is above 4096 px")
        pixel_values = graphs.preprocess_image(image).numpy()
        started = time.perf_counter()
        image_embed, feats_0, feats_1 = session.run(None, {"pixel_values": pixel_values})
        encoder_ms = (time.perf_counter() - started) * 1000
        if image_embed.shape != (1, 256, fl.GRID, fl.GRID):
            raise ValueError(f"{path.name}: image_embed shape {image_embed.shape}")
        np.savez_compressed(
            out_dir / f"{path.stem}.npz",
            image_embed=image_embed[0].astype(np.float32),
            high_res_feats_1=average_blocks(feats_1[0], 2).astype(np.float32),
            high_res_feats_0=average_blocks(feats_0[0], 4).astype(np.float32))
        photos.append({"stem": path.stem, "width": image.size[0], "height": image.size[1],
                       "encoder_cpu_ms": encoder_ms})
        print(f"[encoder] {path.name} {image.size[0]}x{image.size[1]}: {encoder_ms:.0f} ms")

    times = [p["encoder_cpu_ms"] for p in photos]
    fl.write_json(out_dir / "features.json", {
        "step": "issue #1, step 3: cell features",
        "encoder": {"path": str(encoder_path), "sha256": fl.file_sha256(encoder_path)},
        "onnxruntime": ort.__version__,
        "session": {"provider": "CPUExecutionProvider", "intra_op_threads": args.threads,
                    "cpu_memory_arena": False},
        "machine": fl.machine_description(),
        "photos": photos,
        "encoder_cpu_ms_median": float(np.median(times)),
        "peak_working_set_bytes": fl.peak_working_set_bytes(),
    })
    print(f"[done] {len(photos)} photos, median encoder {np.median(times):.0f} ms, "
          f"peak working set {fl.peak_working_set_bytes() / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
