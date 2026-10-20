# Locate people and sky from SAM 2.1 tiny image features (proof of concept)

Issue [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1),
[README section 5](../../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features).
Report: [P0: SAM 2.1 feature localization](../../../docs/exploration/p0_sam_feature_localization.md).

The scripts check, without training, whether the cell features of the SAM 2.1 tiny encoder
separate people and sky from the rest of a photo. A class prototype is the mean feature of the
labeled cells of the other 13 photos (leave one photo out). The score of a cell is
`cosine(class prototype) - cosine(other prototype)`.

## Files

| File | Purpose |
| --- | --- |
| `feature_localization.py` | Shared helpers: cell grid, ONNX Runtime CPU session (4 threads, CPU memory arena off), reference masks, peak working set. |
| `class_presence.json` | Presence of people and sky per photo (referenced, present without a reference, uncertain, absent), the reference masks, and the ignore boxes of unreferenced instances. Classified by eye. |
| `extract_cell_features.py` | Step 3: one encoder run per photo; `image_embed` and the pooled high-resolution features on the 64 x 64 grid. |
| `make_label_grids.py` | Step 2: label grid per photo (person, sky, other, excluded) and the evaluation sets per class; contact sheet. |
| `score_prototypes.py` | Steps 4 and 5: prototypes, score maps, area under the ROC curve, peak precision, separation, and the acceptance check. |
| `decode_top_cells.py` | Step 6: top-scoring cells and region boxes as SAM 2.1 tiny prompts; IoU with the reference. |
| `make_score_sheets.py` | Step 7: score map overlays per photo (local only). |

## Inputs

All inputs are local and are not committed. They are under `build/tmp/` of this repository:

| Input | Path |
| --- | --- |
| Photos and E1 reference masks at the 4096 px application input (E4 input set) | `build/tmp/explore/e4/input_4096/` |
| SAM 2.1 tiny graphs, float32 (`point_encoder.onnx` SHA-256 `17a554b4...`, `point_decoder.onnx` SHA-256 `6f9a0ef2...`; byte-identical to E1) | `build/tmp/explore/e4/onnx_float32_batch16/tiny/` |
| E1 predicted IoU per mask (selects the best multimask sky reference) | `build/tmp/explore/e1/compare/base_plus/compare.json` |
| Prompts of the quality set | `build/tmp/sm1/quality/prompts.json` |
| Python environment (the SAM 2.1 environment of [`scripts/ai_models/sam2/`](../../ai_models/sam2/README.md)) | `build/tmp/explore/venv/` |

## Outputs

| Output | Path | Committed |
| --- | --- | --- |
| Cell features (71 MB) | `build/tmp/explore/p0/features/` | no |
| Decoder logits of step 6 | `build/tmp/explore/p0/decoded_logits/` | no |
| Measurements and tables (`*.json`, `summary.md`) | `results/p0_sam_feature_localization/` | yes |
| Label grids and score maps (`labels.npz`, `score_maps.npz`) | `results/p0_sam_feature_localization/` | no (`results/**/*.npz` is in `.gitignore`) |
| Sheets (show the photos) | `results/p0_sam_feature_localization/sheets/` | no (`*.jpg` is in `.gitignore`) |

## Commands

Run one command at a time (Git Bash, Windows). Only `extract_cell_features.py` and
`decode_top_cells.py` run a model (ONNX Runtime CPU, 4 threads, CPU memory arena off).

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"; I=$E/e4/input_4096
R=results/p0_sam_feature_localization; P=scripts/poc/sam_feature_localization
# Step 2: label grids and the contact sheet.
$PY $P/make_label_grids.py --reference-masks $I/reference_masks \
  --reference-scores $E/e1/compare/base_plus/compare.json --photos $I/photos \
  --out-dir $R --sheet $R/sheets/label_grids.jpg
# Step 3: encoder, once per photo.
$PY $P/extract_cell_features.py --photos $I/photos --onnx-dir $E/e4/onnx_float32_batch16/tiny \
  --out-dir $E/p0/features --threads 4
# Steps 4 and 5: prototypes, score maps, measurements (metrics.json, summary.md).
$PY $P/score_prototypes.py --features $E/p0/features --labels $R --out-dir $R
# Step 6: SAM 2.1 tiny decoder on the top cells (end_to_end.json).
$PY $P/decode_top_cells.py --photos $I/photos --onnx-dir $E/e4/onnx_float32_batch16/tiny \
  --scores $R --reference-masks $I/reference_masks \
  --reference-scores $E/e1/compare/base_plus/compare.json \
  --prompts build/tmp/sm1/quality/prompts.json --out-dir $R --logit-dir $E/p0/decoded_logits
# Step 7: sheets (local only).
$PY $P/make_score_sheets.py --photos $I/photos --scores $R --labels $R \
  --logits $E/p0/decoded_logits --out-dir $R/sheets
```
