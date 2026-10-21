# People and sky head on SAM 2.1 tiny `image_embed` (step 1)

[README section 5](../../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features),
step 1. The proof of concept [P0](../../../docs/exploration/p0_sam_feature_localization.md) shows
that `image_embed` holds the person and sky information, but that a prototype without training
cannot give a calibrated "found" score, separate people from animals, or give one box per person.
This step trains a small head on the frozen `image_embed` of SAM 2.1 tiny. The training runs on
Google Colab, so the dataset is not downloaded to the development machine.

## Files

| File | Purpose |
| --- | --- |
| `head_data.py` | Shared layout: grid, target arrays, CenterNet Gaussian, dataset of the `image_embed` cache, block-shuffle sampler. |
| `prepare_coco_targets.py` | COCO 2017 panoptic annotations -> photos with an allowed license (downloaded) and per-cell targets of one split. |
| `encode_image_embed.py` | SAM 2.1 tiny encoder -> float16 `image_embed` cache of one split. Continues a stopped run. |
| `head_model.py` | The head and the person box decoding. |
| `head_losses.py` | The training objective. |
| `head_metrics.py` | The validation measurements and the checkpoint selection. |
| `train_head.py` | Training loop, validation each epoch, `best.pt`, `last.pt`, `metrics.jsonl`. |
| `export_head_onnx.py` | `people_sky_head.onnx` and its comparison with PyTorch. |
| `evaluate_head.py` | The validation measurements of an existing checkpoint, without training. |
| `evaluate_quality_photos.py` | End to end on the 14 quality photos: head prompts -> SAM 2.1 tiny decoder -> IoU with the P0 references. |
| `make_quality_sheets.py` | Overlay sheets of the end-to-end run (local only). |
| `colab_train_people_sky_head.ipynb` | The Colab run of all steps. |
| `requirements_colab.txt` | Packages that the Colab runtime does not have. |

## Model

```text
image_embed [1, 256, 64, 64]  (point_encoder.onnx output, the same input as the SAM 2.1 decoder)
  -> LayerNorm over channels -> 1 x 1 conv 256 -> 96, GroupNorm, GELU
  -> 3 residual blocks: 3 x 3 conv 96 -> 96, dilation 1, 2, 4, GroupNorm, GELU
  -> sky_logits [1, 1, 64, 64]       sky share of each cell
     center_logits [1, 1, 64, 64]    person center heatmap
     center_offset [1, 2, 64, 64]    center inside the cell (x, y in [0, 1))
     log_size [1, 2, 64, 64]         log of the person box width and height in cells
     presence_logits [1, 2]          sky present, person present (from the mean and maximum
                                     of the trunk features)
```

275,656 parameters, 1.12 MB as float32 ONNX. The receptive field of the 3 x 3 blocks is 15
cells; `image_embed` already has global context from the Hiera attention.

Use in the product (README section 4.1): Sky = `presence[sky]` above a threshold, then points on
the confident cells of the sky map, one set per connected sky region. People =
`presence[person]` above a threshold, then the local maxima of the center heatmap, one box per
maximum (`decode_people` with `nms_iou = 0.5`: a large person gives several local maxima, and the
suppression keeps the box with the highest score), one SAM 2.1 box prompt per person, with the
box clipped to the photo. This is also the People list. The list threshold is
`person_list_best_f1_nms` of the COCO validation split.

## Training data and licenses

| Item | License | Use |
| --- | --- | --- |
| SAM 2.1 Hiera tiny checkpoint and code (`facebookresearch/sam2` at `2b90b9f5`) | Apache-2.0 | Frozen encoder |
| COCO 2017 panoptic annotations (`person`, `sky-other-merged`) | CC BY 4.0 | Targets |
| COCO 2017 photos (Flickr) | Per photo, as the COCO `images` list names it | Training only with license 4 (CC BY 2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions), 8 (United States Government Work). Licenses 1 to 3 (non-commercial) are excluded by decision U2. License 6 (CC BY-ND 2.0) is excluded by default because the effect of "no derivatives" on training is not clear |

The validation split uses the photos of all licenses, because it is only measured. The
`manifest.json` of each split records the photo ids, the Flickr URL, and the license of each
photo, and the photo count per license. COCO states the license of each photo; COCO did not
verify it.

## Targets

All targets are on the 64 x 64 stretched grid: cell (x, y) covers 1/64 of the photo width and
1/64 of the photo height, the same as the SAM 2.1 preprocessing.

| Target | Value |
| --- | --- |
| `sky_coverage` | Share of the cell pixels in `sky-other-merged` segments (pixel-area average) |
| `sky_valid` | The cell has at most 50 percent unlabeled pixels |
| `center_heatmap` | Gaussian with peak 1 at the box center cell of each non-crowd person; radius from the CenterNet rule (IoU 0.7) |
| `heatmap_valid` | False on the cells that a crowd person segment covers (except the peaks) |
| `person_offset`, `person_log_size` | Box center minus the cell corner; log of the box width and height in cells (minimum 1/16 cell) |
| `presence` | Sky: 1 if the sky area is at least 1 percent, 0 if there is no sky pixel, ignored otherwise. Person: 1 if a non-crowd person segment has at least one cell area, 0 if there is no person segment, ignored otherwise |

## Objective

```text
L = L_sky_bce + L_sky_dice + L_center + L_offset + L_size + L_presence      (all weights 1)

L_sky_bce   = mean over valid cells of BCE(sigmoid(sky_logit), sky_coverage)
L_sky_dice  = mean over images of 1 - (2 sum(p t) + 1) / (sum(p) + sum(t) + 1),  p, t on valid cells
L_center    = (1 / N_centers) sum over cells of
                 -(1 - p)^2 log(p)                    at the centers (target 1)
                 -(1 - y)^4 p^2 log(1 - p)            at the other valid cells (target y < 1)
L_offset    = (1 / N_centers) sum over centers of |offset_pred - offset_target|_1
L_size      = (1 / N_centers) sum over centers of |log_size_pred - log_size_target|_1
L_presence  = mean over non-ignored labels of BCE(sigmoid(presence_logit), presence)
```

- The sky map learns the sky share of each cell. The soft target keeps the tree and building
  horizons as partial cells. The Dice term keeps small sky areas from being lost against the
  many non-sky cells.
- The center loss is the CenterNet focal loss. It makes one peak per person at the box center,
  not at the face or the hair (the P0 failure). The `(1 - y)^4` factor reduces the penalty near
  a center.
- The size loss is L1 of the log size, so the error is relative to the person size.
- The presence loss gives the "found / not found" score that P0 could not calibrate. The
  validation photos give its threshold.

Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4, one epoch linear warmup, cosine decay,
batch 32, 20 epochs, float32.

## Measurements (validation split, each epoch)

| Measurement | Rule |
| --- | --- |
| `sky_cell_iou` | IoU over all valid cells: sky probability >= 0.5 against sky share >= 0.5 |
| `presence_auc_sky`, `presence_auc_person` | Area under the ROC curve of the presence output |
| `map_max_auc_*` | The same with the map maximum as the score, for comparison |
| `presence_best_f1_*` | Threshold with the highest F1 of "found", with precision and recall |
| `person_ap50` | Average precision of the person boxes at IoU 0.5; an unmatched box with at least half of its area in a crowd box is not counted |
| `person_top1_hit` | Share of the photos with a person whose highest-scoring box has IoU >= 0.5 with a person (decision U5) |
| `person_ap50_nms` | `person_ap50` after the suppression of overlapping boxes (IoU >= 0.5). `person_top1_hit` does not change, because the top-1 box is never suppressed |
| `person_list_best_f1_nms` | The box score threshold with the highest F1 of the matched boxes after the suppression (the People list threshold), with precision and recall |
| `person_box_fit_nms` | Per person height (small < 8 cells, medium 8 to 24, large >= 24): mean best box IoU, share with IoU >= 0.75, median log ratio of box height and width to the person (negative: too small) |

`best.pt` is the epoch with the highest mean of `sky_cell_iou` and `person_ap50`. These are COCO
measurements. The acceptance on the 14 quality photos (P0 criteria, SAM 2.1 decoder end to end)
runs on the development machine with `people_sky_head.onnx` and the P0 features.

## Colab run

1. Open `colab_train_people_sky_head.ipynb` in Colab (File > Open notebook > GitHub,
   `zidage/alcedo-ai-mask-lab`, branch `main`) and select a GPU runtime. The notebook clones
   the branch that `REPO_BRANCH` names.
2. Run all cells. The outputs go to `MyDrive/alcedo_people_sky_head/`: `run/` (`best.pt`,
   `last.pt`, `metrics.jsonl`, `run.json`), `export/` (`people_sky_head.onnx`,
   `people_sky_head.json`), `splits/` (the manifests).
3. After a runtime reset, run all cells again with `RESUME = True`.
4. To measure an existing `run/best.pt` with the current `head_metrics.py` (for example after a
   change of the measurements), set `TRAIN = False`. The notebook then prepares and encodes only
   the validation split, does not train or export, and writes
   `export/people_sky_head_val_measurements.json` (`evaluate_head.py`). The scripts come from
   `REPO_BRANCH`, so push the changes first.

Disk: the `image_embed` cache is 2 MiB per photo on the local disk of the runtime. A cell checks
the free space before the encoder starts.

## Checks on the development machine (2026-10-20)

A synthetic COCO layout of 8 noise photos (`build/tmp/people_sky_head_check/`, the generator is
local) ran through all scripts on CPU (`--device cpu`), with the SAM 2.1 tiny checkpoint of E1:

| Check | Rule | Result |
| --- | --- | --- |
| License selection | Photos with licenses 1 and 6 are not selected for `4,5,7,8` | 6 of 8 selected |
| Targets | Sky share of the boundary row, person center cell, offset, and size against values computed by hand | Equal (row 21: 51 / 255 = 0.2; center (10, 40), offset 0.667, size 10.67 x 16 cells) |
| `image_embed` cache against `point_encoder.onnx` (E4 float32, SHA-256 `17a554b4...`) on 2 photos | Maximum absolute difference at most the float16 step at the largest value (about 1e-3) | 4.9e-4 (largest value 1.48); lowest per-cell cosine 0.999999 |
| Overfit of 8 photos (train = validation, 300 epochs) | All measurements reach 1 | sky IoU, AP50, top-1, both presence AUCs: 1.000 |
| Resume | `--resume` continues at the next epoch | Continued at epoch 2 |
| ONNX export | Maximum absolute difference per output at most 1e-4 | 6.2e-6 |

Not checked locally: the Colab notebook cells, the COCO download, and the CUDA run.

Checks of the later changes (2026-10-20, CPU):

| Check | Rule | Result |
| --- | --- | --- |
| `head_metrics.py` with `person_ap50_nms`, `person_list_best_f1_nms`, `person_box_fit_nms` | The committed and the changed `Evaluation` give equal values for all fields of the first run, on the 8 synthetic photos, with the 300-epoch overfit checkpoint and with the Colab `best.pt` | Equal |
| `person_box_fit_nms` | On the overfit checkpoint of epoch 299: mean best IoU 1 and log size ratio 0 | 1.000, +0.000 |
| `evaluate_head.py` | Runs on CPU on the synthetic split and writes the JSON | Done |
| Notebook | All code cells are valid Python with the `!` and `%` lines replaced | Valid |

## Results of the first Colab run

[P1 report](../../../docs/exploration/p1_people_sky_head.md): COCO validation, and end to end on
the quality photos (commands in its section 3.4).
