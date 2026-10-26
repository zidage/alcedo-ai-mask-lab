# Proof of concept: S3OD on the crop of a box

Report: [S3OD on the crop of a box (Sky and objects)](../../../docs/exploration/box_s3od_poc.md).

| File | Purpose |
| --- | --- |
| `check_box_s3od.py` | S3OD on the crop of each box of `boxes.json` (three outputs and IoU scores), the Sky mask 1 - alpha, one SAM 2.1 box prompt with the same box, the MVP one-click Sky, and the photo Background; check pages and `summary.json`. |
| `boxes.json` | The 17 Wikimedia Commons photos (file, source page, license, author, SHA-256) and the hand-drawn boxes. |

The photos are not committed. Download them as the report section 6 states, to
`build/tmp/box_s3od_poc/photos/`.

```text
E=build/tmp/explore
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/box_s3od/check_box_s3od.py
```

The script uses the MVP environment (`scripts/poc/ai_mask_mvp/requirements.txt`) and the graphs
of `mask_pipeline.GraphPaths.default`, with the accepted S3OD graph
`s3od/onnx_mha_chunk8_final/float16/subject_masks.onnx` (`--subject-onnx`). Provider: DirectML
(`--provider`).
