# Proof of concept: Matting Anything (MAM)

Report: [Matting Anything (MAM) for Sky and object boxes](../../../docs/exploration/mam_poc.md).

| File | Purpose |
| --- | --- |
| `mam_model.py` | MAM inference in PyTorch (SAM ViT-B prompt, M2M, the three output fusion modes of the MAM repository, the complement target). |
| `export_mvp_references.py` | The reference masks of the accepted model set on the `box_s3od` photos and boxes (S3OD on the box crop, SAM 2.1 tiny box prompt, MVP one-click Sky) and the MVP sky points. MVP environment, DirectML. |
| `check_mam.py` | MAM on the same photos, boxes, and sky points; check pages with 1:1 zoom windows and `summary.json`. MAM environment, PyTorch CPU. |
| `requirements.txt` | The MAM environment. |

The photos are the 17 Wikimedia Commons photos of [`../box_s3od/boxes.json`](../box_s3od/boxes.json),
downloaded to `build/tmp/box_s3od_poc/photos/` (box S3OD report, section 6).

## Model files (not committed, not distributed)

```text
W=build/tmp/mam_poc
git clone https://github.com/SHI-Labs/Matting-Anything $W/src/Matting-Anything   # commit eb01e1c
curl -L -o $W/weights/mam_sam_vitb.pth \
  https://huggingface.co/shi-labs/Matting-Anything/resolve/main/checkpoints/mam_sam_vitb.pth
curl -L -o $W/weights/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

`sam_vit_b_01ec64.pth` is only used to verify that the SAM weights in the MAM checkpoint are the
official SAM ViT-B weights.

## Commands

```text
E=build/tmp/explore
py -3.12 -m venv $E/venv_mam
$E/venv_mam/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1 torchvision==0.29.1
$E/venv_mam/Scripts/python -m pip install -r scripts/poc/mam/requirements.txt
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/mam/export_mvp_references.py
$E/venv_mam/Scripts/python -I scripts/poc/mam/check_mam.py --out build/tmp/mam_poc/check_run3
# pages: build/tmp/mam_poc/check_run3/index.html
```
