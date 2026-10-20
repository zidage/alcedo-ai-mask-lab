#!/usr/bin/env python3
"""image_embed cache of the SAM 2.1 tiny encoder for one split (step 1).

Input:  --split-dir (manifest.json of prepare_coco_targets.py), --image-dir (the photos),
        --sam2-repo (checkout of facebookresearch/sam2 at SAM2_COMMIT), --checkpoint
        (sam2.1_hiera_tiny.pt, SHA-256 CHECKPOINT_SHA256).
Output: --split-dir/image_embed.npy, float16 [N, 256, 64, 64], in the image order of the
        manifest; --split-dir/image_embed_progress.json (images done, run description).

The encoder is `sam2_graphs.PointEncoder` (the graph of point_encoder.onnx: Hiera trunk, FPN neck,
`no_mem_embed` on the stride-16 map) and the preprocessing is `sam2_graphs.preprocess_image`
(stretch to 1024 x 1024, bilinear with antialias, ImageNet normalization), the same as E1 and P0.
The encoder runs in float32 on CUDA with TF32 off; the cache stores float16. A stopped run
continues from the progress file.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ai_models" / "sam2"))
import head_data as hd  # noqa: E402

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
FLOAT16_MAX = 65504.0


class PhotoTensors:
    """Preprocessed photo tensors [3, 1024, 1024] in manifest order, from `start`."""

    def __init__(self, paths: list[Path], start: int) -> None:
        self.paths, self.start = paths, start

    def __len__(self) -> int:
        return len(self.paths) - self.start

    def __getitem__(self, offset: int):
        from PIL import Image

        import sam2_graphs as graphs

        with Image.open(self.paths[self.start + offset]) as image:
            return graphs.preprocess_image(image.convert("RGB"))[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda",
                        help="cuda (Colab) or cpu (a check of the script on a few images)")
    args = parser.parse_args()

    import torch

    import sam2_graphs as graphs

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: select a GPU runtime")
    device = torch.device(args.device)
    commit = subprocess.run(["git", "-C", args.sam2_repo, "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != SAM2_COMMIT:
        raise ValueError(f"{args.sam2_repo} is at {commit}, expected {SAM2_COMMIT}")
    checkpoint = Path(args.checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(f"{checkpoint}: SHA-256 {checkpoint_sha256}, expected {CHECKPOINT_SHA256}")

    split_dir = Path(args.split_dir)
    manifest = hd.read_json(split_dir / "manifest.json")
    paths = [Path(args.image_dir) / entry["file_name"] for entry in manifest["images"]]
    count = len(paths)
    progress_path = split_dir / "image_embed_progress.json"
    cache_path = hd.embed_path(split_dir)
    shape = (count, hd.EMBED_CHANNELS, hd.GRID, hd.GRID)
    if progress_path.exists():
        progress = hd.read_json(progress_path)
        cache = np.lib.format.open_memmap(cache_path, mode="r+")
        if cache.shape != shape or cache.dtype != np.float16:
            raise ValueError(f"{cache_path}: {cache.dtype} {cache.shape}, expected float16 {shape}")
    else:
        progress = {"done": 0, "seconds": 0.0}
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
    start = progress["done"]
    if start == count:
        print(f"[done] {cache_path} already has all {count} images")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graphs.add_repository_to_path(args.sam2_repo)
    encoder = graphs.PointEncoder(graphs.build_model("tiny", str(checkpoint))).to(device).eval()
    loader = torch.utils.data.DataLoader(PhotoTensors(paths, start), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers,
                                         pin_memory=device.type == "cuda")
    run = {
        "step": "README section 5, step 1: image_embed cache",
        "sam2_commit": commit,
        "checkpoint": {"path": checkpoint.name, "sha256": checkpoint_sha256},
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "torch": torch.__version__,
        "encoder_dtype": "float32, TF32 off",
        "cache_dtype": "float16",
        "preprocessing": "sam2_graphs.preprocess_image: stretch to 1024 x 1024, bilinear with "
                         "antialias, ImageNet normalization",
    }
    index = start
    started = time.perf_counter()
    with torch.no_grad():
        for pixels in loader:
            image_embed, _, _ = encoder(pixels.to(device))
            values = image_embed.float().cpu().numpy()
            if not np.isfinite(values).all() or np.abs(values).max() >= FLOAT16_MAX:
                raise ValueError(f"images {index} to {index + len(values) - 1}: image_embed has "
                                 f"values that float16 cannot store (max abs "
                                 f"{np.nanmax(np.abs(values))})")
            cache[index:index + len(values)] = values.astype(np.float16)
            index += len(values)
            if index % (args.batch_size * 100) < args.batch_size or index == count:
                cache.flush()
                elapsed = progress["seconds"] + time.perf_counter() - started
                hd.write_json(progress_path, {**run, "done": index, "count": count,
                                              "seconds": round(elapsed, 1)})
                rate = (index - start) / (time.perf_counter() - started)
                print(f"[encode] {index} / {count}, {rate:.1f} images/s")
    cache.flush()
    hd.write_json(progress_path, {**run, "done": index, "count": count,
                                  "seconds": round(progress["seconds"] + time.perf_counter()
                                                   - started, 1),
                                  "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                                             if device.type == "cuda" else None)})
    print(f"[done] {count} images -> {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
