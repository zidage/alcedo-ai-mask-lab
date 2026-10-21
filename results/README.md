# Results

Measurement results of the prototypes under `scripts/poc/`. Each step has one directory. The
report of the step explains the values; the script README states the commands that make them.

Only the text results (JSON, Markdown) are committed. Images that show the quality photos (sheets,
overlays) are written to the same directories but are not committed: `*.jpg` and `*.png` are in
`.gitignore`, because the photos must not be shared again. The binary arrays (`*.npz`) are also
local; run the commands of the script README to make them again.

| Directory | Step | Report |
| --- | --- | --- |
| [`p0_sam_feature_localization/`](p0_sam_feature_localization/) | P0, issue [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1) | [P0 report](../docs/exploration/p0_sam_feature_localization.md) |
| [`p1_people_sky_head/`](p1_people_sky_head/) | P1 (README section 5, step 1), issue [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1) | [P1 report](../docs/exploration/p1_people_sky_head.md) |

## `p0_sam_feature_localization/`

| File | Content | Script |
| --- | --- | --- |
| `labels.json` | Per photo: class status, reference files, cell counts | `make_label_grids.py` |
| `labels.npz` (local only) | Per photo: label grid, class fraction, positive, negative, and center cells (64 x 64) | `make_label_grids.py` |
| `score_maps.npz` (local only) | Score map per feature source, photo, and class (64 x 64, float32) | `score_prototypes.py` |
| `metrics.json` | AUC, peak precision, peak scores, separation, acceptance check, prototype cell counts | `score_prototypes.py` |
| `summary.md` | The tables of `metrics.json` for all feature sources | `score_prototypes.py` |
| `end_to_end.json` | Prompts, predicted IoU, and IoU with the reference of every decoder run | `decode_top_cells.py` |
| `sheets/` (local only) | Label grid contact sheet and score map sheets per photo | `make_label_grids.py`, `make_score_sheets.py` |

## `p1_people_sky_head/`

| File | Content | Script |
| --- | --- | --- |
| `colab_run/run.json`, `colab_run/metrics.jsonl` | Training arguments, per-epoch loss terms and validation measurements of the first Colab run | `train_head.py` |
| `colab_run/people_sky_head.json` | ONNX hash and size, the validation measurements of `best.pt`, the ONNX Runtime against PyTorch comparison | `export_head_onnx.py` |
| `colab_run/*_image_embed_progress.json` | Encoder source, checkpoint hash, device, time, CUDA memory peak per split | `encode_image_embed.py` |
| `colab_run/*_manifest_summary.json` | The split manifest without the photo list, with the SHA-256 of the full manifest and the selected photos per license | `prepare_coco_targets.py` |
| `quality_photos.json` | Per quality photo: presence, sky regions and IoU, People list with rank, score, box, IoU; timings | `evaluate_quality_photos.py` |
| `sheets/` (local only) | Sky and people overlays per photo | `make_quality_sheets.py` |
