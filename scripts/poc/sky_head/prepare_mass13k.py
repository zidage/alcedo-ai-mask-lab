#!/usr/bin/env python3
"""MaSS13K archives -> prepared sky data (fine sky head, issue #10).

Input:  --photo-archive (zip files with the photos, for example val.zip test.zip
        train.zip.001), --label-archive (zip files with the labels: masks.zip), --inspect or
        --out-dir. For split volumes give the first volume (train.zip.001): the script reads
        train.zip.001, .002, ... as one file without joining them on the disk. Only JPEG files of
        the photo archives and PNG files of the label archives are used.
Output: --inspect: a description of the archives (top directories, file types, image sizes, the
        label values of sample labels) on stdout and in --inspect-json.
        --out-dir: <out-dir>/<split>/{images,alpha,weight}/ and manifest.json per split
        (layout: sky_head_data.py).

MaSS13K (CVPR 2025, https://github.com/xiechenxi99/MaSS13K, README decision U18) labels each
pixel of a 4K photo with one of 7 classes (MaSSDataset in its mmsegmentation fork):
0 background, 1 person, 2 building, 3 tree, 4 ground, 5 sky, 6 water. A label PNG can hold the
class index (mode L or P) or the class color of the MaSSDataset palette (mode RGB); the script
accepts both and stops on any other value, except IGNORE_INDEX (255), which becomes an ignored
pixel in the weight map.

A photo and its label are paired by the file stem; the split is the first path component that
is train, val, or test. Copies that a Jupyter checkpoint left (stem ending in "-checkpoint") are
skipped and listed in the manifest. The script stops when a photo has no label, a label has no photo, a
member has no split, or a photo and its label have different sizes. The photo is scaled to a
long edge of sky_head_data.PREPARED_LONG_EDGE (Lanczos); the sky coverage is the area average of
the hard sky label to the same size.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_head_data as sd  # noqa: E402

CLASSES = ("background", "person", "building", "tree", "ground", "sky", "water")
PALETTE = ((0, 0, 0), (255, 0, 0), (64, 34, 64), (0, 255, 0), (93, 64, 0), (0, 0, 255),
           (0, 255, 255))
SKY_INDEX = CLASSES.index("sky")
IGNORE_INDEX = 255
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg")
LABEL_SUFFIX = ".png"
INSPECT_SAMPLES = 8
JPEG_QUALITY = 95


class JoinedFile(io.RawIOBase):
    """The volumes <name>.001, <name>.002, ... (a byte split of one file) as one seekable file."""

    def __init__(self, first: Path) -> None:
        super().__init__()
        self.paths = volume_paths(first)
        self.sizes = [path.stat().st_size for path in self.paths]
        self.starts = [sum(self.sizes[:k]) for k in range(len(self.sizes))]
        self.size = sum(self.sizes)
        self.position = 0
        self.files = [open(path, "rb") for path in self.paths]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self.position < self.size:
            index = max(k for k, start in enumerate(self.starts) if start <= self.position)
            file = self.files[index]
            file.seek(self.position - self.starts[index])
            count = min(len(view) - done, self.starts[index] + self.sizes[index] - self.position)
            read = file.readinto(view[done:done + count])
            if not read:
                break
            done += read
            self.position += read
        return done

    def close(self) -> None:
        for file in self.files:
            file.close()
        super().close()


def volume_paths(first: Path) -> list[Path]:
    stem = first.name[:-len(".001")]
    paths = []
    number = 1
    while (first.parent / f"{stem}.{number:03d}").exists():
        paths.append(first.parent / f"{stem}.{number:03d}")
        number += 1
    return paths


def is_first_volume(path: Path) -> bool:
    return path.name.endswith(".001")


@contextlib.contextmanager
def open_archive(path: Path):
    """zipfile.ZipFile of a zip file or of split volumes; closes the volume files on exit
    (ZipFile does not close a file object that it did not open)."""
    if is_first_volume(path):
        with JoinedFile(path) as joined, zipfile.ZipFile(joined) as archive:
            yield archive
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive


def archive_bytes(path: Path) -> int:
    paths = volume_paths(path) if is_first_volume(path) else [path]
    return sum(item.stat().st_size for item in paths)


def member_split(name: str) -> str | None:
    for part in PurePosixPath(name).parts:
        if part.lower() in SPLITS:
            return part.lower()
    return None


def is_copy_of_notebook_checkpoint(name: str) -> bool:
    """Files that a Jupyter checkpoint left in the archives (for example
    annotations/test/<stem>-checkpoint.png): copies, not dataset items."""
    path = PurePosixPath(name)
    return path.stem.endswith("-checkpoint") or ".ipynb_checkpoints" in path.parts


def archive_members(photo_archives: list[Path],
                    label_archives: list[Path]) -> tuple[dict, dict, list[str]]:
    """(split, stem) -> (archive index, member name) for the photos (JPEG files of the photo
    archives) and the labels (PNG files of the label archives); the archive index is the position
    in photo_archives + label_archives. Also returns the skipped checkpoint copies."""
    photos: dict = {}
    labels: dict = {}
    skipped: list[str] = []
    archives = photo_archives + label_archives
    for index, path in enumerate(archives):
        table, wanted = ((photos, IMAGE_SUFFIXES) if index < len(photo_archives)
                         else (labels, (LABEL_SUFFIX,)))
        with open_archive(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or "__MACOSX" in info.filename:
                    continue
                if PurePosixPath(info.filename).suffix.lower() not in wanted:
                    continue
                if is_copy_of_notebook_checkpoint(info.filename):
                    skipped.append(f"{path.name}:{info.filename}")
                    continue
                split = member_split(info.filename)
                if split is None:
                    raise ValueError(f"{path.name}: {info.filename} has no train, val, or test "
                                     f"directory")
                key = (split, PurePosixPath(info.filename).stem)
                if key in table:
                    raise ValueError(f"{key} is in {archives[table[key][0]].name} and "
                                     f"{path.name}")
                table[key] = (index, info.filename)
    return photos, labels, skipped


def label_classes(label) -> np.ndarray:
    """Label PIL image -> class index per pixel (uint8), IGNORE_INDEX kept."""
    if label.mode in ("L", "P", "I", "I;16"):
        values = np.asarray(label)
        if label.mode in ("I", "I;16"):
            values = values.astype(np.int64)
        unknown = np.setdiff1d(np.unique(values), list(range(len(CLASSES))) + [IGNORE_INDEX])
        if unknown.size:
            raise ValueError(f"label values {unknown[:10].tolist()} are not class indexes")
        return values.astype(np.uint8)
    if label.mode in ("RGB", "RGBA"):
        rgb = np.asarray(label.convert("RGB"), dtype=np.int32)
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        classes = np.full(code.shape, IGNORE_INDEX, dtype=np.uint8)
        known = np.zeros(code.shape, dtype=bool)
        for index, (r, g, b) in enumerate(PALETTE):
            match = code == ((r << 16) | (g << 8) | b)
            classes[match] = index
            known |= match
        if not known.all():
            colors = np.unique(code[~known])[:10]
            raise ValueError(f"label colors {[f'#{c:06x}' for c in colors.tolist()]} are not "
                             f"in the MaSSDataset palette")
        return classes
    raise ValueError(f"label mode {label.mode} is not supported")


def inspect(archives: list[Path]) -> dict:
    from PIL import Image

    report: dict = {"archives": []}
    for path in archives:
        with open_archive(path) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            tops = Counter("/".join(PurePosixPath(i.filename).parts[:3]) for i in infos)
            suffixes = Counter(PurePosixPath(i.filename).suffix.lower() for i in infos)
            samples = []
            for kind, wanted in (("photo", IMAGE_SUFFIXES), ("label", (LABEL_SUFFIX,))):
                members = [i for i in infos
                           if PurePosixPath(i.filename).suffix.lower() in wanted]
                for info in members[:INSPECT_SAMPLES]:
                    with Image.open(io.BytesIO(archive.read(info))) as image:
                        entry = {"kind": kind, "member": info.filename, "mode": image.mode,
                                 "size": list(image.size)}
                        if kind == "label":
                            values, counts = np.unique(np.asarray(image).reshape(
                                -1, *np.asarray(image).shape[2:]), axis=0, return_counts=True)
                            order = np.argsort(-counts)[:12]
                            entry["values"] = [[np.atleast_1d(values[k]).tolist(),
                                                int(counts[k])] for k in order]
                            entry["palette"] = (image.getpalette()[:3 * len(CLASSES)]
                                                if image.mode == "P" else None)
                    samples.append(entry)
            report["archives"].append({
                "archive": path.name, "bytes": archive_bytes(path), "members": len(infos),
                "suffixes": dict(suffixes), "top_directories": dict(tops.most_common(20)),
                "samples": samples})
    return report


def prepare_pair(task: tuple) -> dict:
    """One photo and its label -> prepared files; returns the manifest item."""
    from PIL import Image

    archives, split, stem, photo_member, label_member, out_dir = task
    with open_archive(archives[photo_member[0]]) as archive:
        photo_bytes = archive.read(photo_member[1])
    with open_archive(archives[label_member[0]]) as archive:
        label_bytes = archive.read(label_member[1])
    with Image.open(io.BytesIO(photo_bytes)) as image:
        photo = image.convert("RGB")
    with Image.open(io.BytesIO(label_bytes)) as image:
        classes = label_classes(image)
    width, height = photo.size
    if classes.shape != (height, width):
        raise ValueError(f"{split}/{stem}: photo {width} x {height}, label "
                         f"{classes.shape[1]} x {classes.shape[0]}")
    scale = sd.PREPARED_LONG_EDGE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale))) if scale < 1 else \
        (width, height)
    sky = (classes == SKY_INDEX).astype(np.float32)
    ignore = classes == IGNORE_INDEX
    import cv2

    alpha = cv2.resize(sky, size, interpolation=cv2.INTER_AREA)
    split_dir = Path(out_dir) / split
    photo.resize(size, Image.Resampling.LANCZOS).save(split_dir / "images" / f"{stem}.jpg",
                                                      quality=JPEG_QUALITY)
    Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(
        split_dir / "alpha" / f"{stem}.png")
    has_weight = bool(ignore.any())
    if has_weight:
        use = cv2.resize((~ignore).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        Image.fromarray(np.where(use >= 0.999, 255, 0).astype(np.uint8)).save(
            split_dir / "weight" / f"{stem}.png")
    return {"stem": stem, "photo_member": photo_member[1], "label_member": label_member[1],
            "source_size": [width, height], "size": list(size),
            "sky_share": round(float(sky.mean()), 5),
            "ignored_share": round(float(ignore.mean()), 5), "has_weight": has_weight}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photo-archive", type=Path, nargs="+", required=True)
    parser.add_argument("--label-archive", type=Path, nargs="+", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N pairs per split only (a check of the script)")
    args = parser.parse_args()
    photo_archives = [path.resolve() for path in args.photo_archive]
    label_archives = [path.resolve() for path in args.label_archive]
    archives = photo_archives + label_archives

    if args.inspect:
        report = inspect(archives)
        photos, labels, skipped = archive_members(photo_archives, label_archives)
        report["skipped_checkpoint_copies"] = skipped
        report["pairs"] = {split: {
            "photos": sum(1 for s, _ in photos if s == split),
            "labels": sum(1 for s, _ in labels if s == split),
            "paired": sum(1 for key in photos if key[0] == split and key in labels)}
            for split in SPLITS}
        text = json.dumps(report, indent=1)
        print(text)
        if args.inspect_json:
            args.inspect_json.write_text(text + "\n", encoding="utf-8", newline="\n")
        return 0

    if args.out_dir is None:
        raise ValueError("give --inspect or --out-dir")
    photos, labels, skipped = archive_members(photo_archives, label_archives)
    missing_labels = sorted(set(photos) - set(labels))
    missing_photos = sorted(set(labels) - set(photos))
    if missing_labels or missing_photos:
        raise ValueError(f"{len(missing_labels)} photos without a label (for example "
                         f"{missing_labels[:3]}), {len(missing_photos)} labels without a photo "
                         f"(for example {missing_photos[:3]})")
    started = time.perf_counter()
    for split in SPLITS:
        keys = sorted(key for key in photos if key[0] == split)
        if args.limit:
            keys = keys[:args.limit]
        if not keys:
            continue
        for sub in ("images", "alpha", "weight"):
            (args.out_dir / split / sub).mkdir(parents=True, exist_ok=True)
        tasks = [(archives, split, stem, photos[(split, stem)], labels[(split, stem)],
                  str(args.out_dir)) for _, stem in keys]
        items = []
        with ProcessPoolExecutor(args.workers) as pool:
            for count, item in enumerate(pool.map(prepare_pair, tasks, chunksize=4), start=1):
                items.append(item)
                if count % 500 == 0 or count == len(tasks):
                    rate = count / (time.perf_counter() - started)
                    print(f"[{split}] {count} / {len(tasks)}, {rate:.1f} pairs/s", flush=True)
        sd.write_json(args.out_dir / split / "manifest.json", {
            "step": "issue #10: prepared MaSS13K split",
            "source": "MaSS13K (CVPR 2025), https://github.com/xiechenxi99/MaSS13K, README U18",
            "archives": [{"name": path.name, "bytes": archive_bytes(path)} for path in archives],
            "long_edge": sd.PREPARED_LONG_EDGE, "sky_index": SKY_INDEX,
            "ignore_index": IGNORE_INDEX, "count": len(items),
            "skipped_checkpoint_copies": [name for name in skipped
                                          if member_split(name.split(":", 1)[1]) == split],
            "photos_with_sky": sum(1 for item in items if item["sky_share"] > 0),
            "items": items})
    print(f"[done] {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
