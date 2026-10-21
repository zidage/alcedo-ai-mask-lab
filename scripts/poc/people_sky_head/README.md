# People and sky head on SAM 2.1 tiny `image_embed` (step 1)

[README section 5](../../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features),
step 1. The proof of concept [P0](../../../docs/exploration/p0_sam_feature_localization.md) shows
that `image_embed` holds the person and sky information, but that a prototype without training
cannot give a calibrated "found" score, separate people from animals, or give one box per person.
This step trains a small head on the frozen `image_embed` of SAM 2.1 tiny. The training runs on
Google Colab, so the dataset is not downloaded to the development machine. The training data are
COCO 2017 (dense sky and person targets) and, from the second run, Open Images V7 (sky point
labels and person boxes on more varied photos), for a head that generalizes to more scenes.

## Files

| File | Purpose |
| --- | --- |
| `head_data.py` | Shared layout: grid, target arrays, CenterNet Gaussian, dataset of the `image_embed` cache, block-shuffle sampler. |
| `prepare_coco_targets.py` | COCO 2017 panoptic annotations -> photos with an allowed license (downloaded) and per-cell targets of one split. |
| `prepare_openimages_targets.py` | Open Images V7 point labels, boxes, and verified image-level labels -> photos with an allowed license and rotation 0 (downloaded) and per-cell targets of one split. |
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
| Open Images V7 annotations (point labels of `Sky` and `Cloud`; boxes of `Person`, `Man`, `Woman`, `Boy`, `Girl`; human-verified image-level labels) | CC BY 4.0 (Google LLC) | Targets |
| Open Images V7 photos (Flickr; the CVDF copy with the long edge 1024 px) | Per photo, as the image list names it (all photos of the validation list: CC BY 2.0). Open Images asks the user to verify the license of each photo | Only CC BY 2.0 and the rotation 0. The manifest records the license, the author, and the source page of each photo |

The validation split uses the photos of all licenses, because it is only measured. The
`manifest.json` of each split records the photo ids, the Flickr URL, and the license of each
photo, and the photo count per license. COCO states the license of each photo; COCO did not
verify it.

Open Images photos: the image list has no rotation for 12.8 percent of the validation photos and a
rotation for 1.2 percent. For these photos the orientation of the labels and the pixels is not
known, so only the photos with the rotation 0 are used. The photos come from the copy of the
Common Visual Data Foundation (`open-images-dataset` on Amazon S3): the long edge is 1024 px and
the EXIF data are removed. Some Flickr originals are removed (HTTP 404). The `OriginalMD5` of the
image list is the MD5 of the Flickr original, so it does not apply to the copy; the check is that
the photo decodes.

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
| `sky_dense` | True: the sky target is a segmentation (COCO) |

Open Images V7 targets (`prepare_openimages_targets.py`), on the same grid:

| Target | Value |
| --- | --- |
| `sky_coverage`, `sky_valid` | Only the cells of point labels: a `Sky` or `Cloud` "yes" point counts 1, a `Sky` "no" point counts 0, the cell share is the mean of its points. A `Cloud` "no" point is not used (the point can be on clear sky), and "unsure" points are not used. The "no" points are often on the areas where the model of the Open Images annotation was not sure (foliage, walls, water, snow) |
| `sky_dense` | False: no Dice term, and the cells count in the `sky_point_*` measurements |
| Person targets | The boxes of the five person classes; `IsGroupOf` and `IsDepiction` boxes are ignored areas (as COCO crowd segments). Boxes of different person classes with IoU >= 0.7 are one person. Open Images annotates every instance of each positive label of a box-annotated photo, so the targets are complete when the photo has a person box or a verified negative `Person` label; otherwise the person losses ignore the photo |
| `presence` | Sky: 1 with a `Sky` or `Cloud` "yes" point or a verified positive `Sky` label, 0 with a verified negative `Sky` label and no "yes" point, ignored otherwise. Person: 1 with a person box of at least one cell area, 0 when the targets are complete and the photo has no person box, ignored otherwise |

## Objective

```text
L = L_sky_bce + L_sky_dice + L_center + L_offset + L_size + L_giou + L_presence   (all weights 1;
                                                                     runs 1 and 2: L_giou weight 0)

L_sky_bce   = mean over valid cells of BCE(sigmoid(sky_logit), sky_coverage)
L_sky_dice  = mean over the images with sky_dense of 1 - (2 sum(p t) + 1) / (sum(p) + sum(t) + 1),
              p, t on valid cells
L_center    = (1 / N_centers) sum over cells of
                 -(1 - p)^2 log(p)                    at the centers (target 1)
                 -(1 - y)^4 p^2 log(1 - p)            at the other valid cells (target y < 1)
L_offset    = (1 / N_centers) sum over centers of |offset_pred - offset_target|_1
L_size      = (1 / N_centers) sum over centers of |log_size_pred - log_size_target|_1
L_giou      = (1 / N_centers) sum over centers of 1 - GIoU(clip(box_pred), clip(box_target)),
              box = (cell + offset) -/+ exp(log_size) / 2, clip to [0, 64] (the photo)
L_presence  = mean over non-ignored labels of BCE(sigmoid(presence_logit), presence)
```

- The sky map learns the sky share of each cell. The soft target keeps the tree and building
  horizons as partial cells. The Dice term keeps small sky areas from being lost against the
  many non-sky cells.
- The center loss is the CenterNet focal loss. It makes one peak per person at the box center,
  not at the face or the hair (the P0 failure). The `(1 - y)^4` factor reduces the penalty near
  a center.
- The size loss is L1 of the log size, so the error is relative to the person size.
- The GIoU loss measures the box edges that SAM 2.1 gets. Run 2 gave boxes of large persons that
  are 3 to 4 percent too short on COCO and Open Images, and on `DSC00674` a bottom edge 5.6 cells
  above the photo edge changed the SAM 2.1 mask from the person to the head (P1 report, section
  10). Both boxes are clipped to the photo, so an edge past the photo edge has no GIoU loss: it
  selects the same area as an edge at the photo edge. The log-size L1 still pulls such an edge
  back.
- The presence loss gives the "found / not found" score that P0 could not calibrate. The
  validation photos give its threshold.
- The point cells of an Open Images photo give the BCE term only: a few cells do not give the
  sky area of the photo, so they have no Dice term.

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
| `person_edge_fit_nms` | The persons that a photo edge cuts (a box edge within 0.5 cell of the photo edge), per person height: mean best IoU of the boxes after the suppression, clipped to the photo; for the cut edges of the matched persons (best IoU >= 0.5), the median gap between the box edge and the photo edge in cells, the share of edges with a gap <= 1 cell, and the share with a gap > 4 cells |
| `sky_point_auc`, `sky_point_accuracy`, `sky_point_false_sky_share` | Only the point cells (Open Images): AUC of the sky probability against the point target, accuracy at probability 0.5, and the share of the "not sky" cells with probability >= 0.5. Null on COCO |

`best.pt` is the epoch with the highest mean of `sky_cell_iou` and `person_ap50` on the first
validation split (COCO `val`). The other validation splits (Open Images `oi_val`) are measured
each epoch (`val_extra` in `metrics.jsonl`). The acceptance on the 14 quality photos (P0 criteria, SAM 2.1 decoder end to end)
runs on the development machine with `people_sky_head.onnx` and the P0 features.

## Colab run

1. Open `colab_train_people_sky_head.ipynb` in Colab (File > Open notebook > GitHub,
   `zidage/alcedo-ai-mask-lab`, branch `main`) and select a GPU runtime. The notebook clones
   the branch that `REPO_BRANCH` names.
2. Run all cells. The outputs go to `MyDrive/alcedo_people_sky_head/RUN_NAME/`: `run/`
   (`best.pt`, `last.pt`, `metrics.jsonl`, `run.json`), `export/` (`people_sky_head.onnx`,
   `people_sky_head.json`), `splits/` (the manifests). The first run (COCO only) is in
   `MyDrive/alcedo_people_sky_head/run/` and `export/`; a new `RUN_NAME` does not change it.
3. After a runtime reset, run all cells again with `RESUME = True`.
4. The last cell measures `RUN_NAME/run/best.pt` and the checkpoints of `EVALUATE_CHECKPOINTS`
   (default: runs 1 and 2) on COCO `val` and Open Images `oi_val` (`evaluate_head.py`, result
   `<checkpoint>_val_measurements.json`). With `TRAIN = False`, the notebook prepares and encodes
   only the validation splits and does only this measurement. The scripts come from
   `REPO_BRANCH`, so push the changes first.

Disk: the `image_embed` cache is 2 MiB per photo on the local disk of the runtime: about 117 GiB
for 30,836 + 2,500 COCO and at most 24,000 + 2,500 Open Images photos. The Open Images training
annotations are about 11 GB. A cell checks the free space before the encoder starts.

Time (L4): COCO download and targets about 25 minutes (first run), Open Images annotations and
photos not measured, encoder about 13 photos/s (24,000 Open Images photos in 1,737 s), training
448 s per epoch for 54,836 photos (run 2, median of 20 epochs).

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

Checks of the Open Images source (2026-10-20, CPU, real Open Images V7 validation files,
`build/tmp/openimages_check/`):

| Check | Rule | Result |
| --- | --- | --- |
| Selection and download | 30 sky and 30 person photos of the validation list; all CC BY 2.0, rotation 0, every photo decodes | 60 photos; of 13,506 pool checks (2,280 sky, 11,226 person; a photo in both pools counts twice), 2,039 rejected for the rotation, 0 for the license |
| CVDF copy | Size against the Flickr original of 3 photos | Long edge 1024 px, same aspect ratio, no EXIF; one Flickr original is removed (404) |
| Targets by eye | Sky point cells and person boxes drawn on 8 photos | "Yes" cells on sky, "no" cells on foliage, cars, walls; boxes on the people |
| Targets against the CSV | One photo: sky cells recomputed from the point rows; its group box | Equal; the group box is an ignored area |
| Sparse sky loss | An Open Images batch with logits 0 | BCE ln 2 = 0.6931, Dice 0, gradient only on the point cells |
| COCO unchanged | The committed and the changed `head_losses.py` and `head_metrics.py` on the synthetic COCO split, with the Colab `best.pt` and the overfit checkpoint | Loss terms and all fields of the first run equal |
| Chain | `encode_image_embed.py` (CPU) on the 60 photos, `train_head.py` with two train and two validation splits (3 epochs), `evaluate_head.py` with two splits | Runs; `val_extra` has the `oi_val_small` measurements |

Checks of the GIoU term and `person_edge_fit_nms` (2026-10-21, CPU, `build/tmp/p1_run3/`):

| Check | Rule | Result |
| --- | --- | --- |
| Runs 1 and 2 unchanged | The committed and the changed `head_losses.py` and `head_metrics.py`, with `w_box_giou` 0: synthetic COCO split with the overfit checkpoint and the run 2 `best.pt`; the 60 Open Images photos with the run 1 and run 2 `best.pt`. Loss terms and metric fields exactly equal, total within 1e-6 | Equal (total difference 0) |
| GIoU term, constructed boxes | Target box: 0. The target box with the bottom edge past the photo edge: 0. The `DSC00674` box (bottom edge 5.6 cells short): 1 - IoU = 0.092869. Two boxes without overlap: 1.9. Tolerance 1e-5. Log size 50: finite | All as expected |
| GIoU gradient | The short box: the gradient of the log height is negative (the training makes the box higher) | -0.042 |
| `person_edge_fit_nms`, constructed boxes | The `DSC00674` box: gap 5.6 (share > 4 cells: 1). A box past the left photo edge: gap 0 after the clipping. A person that no photo edge cuts: not counted | All as expected |
| Chain | `train_head.py` (3 epochs, `w_box_giou` 1) and `evaluate_head.py` on the synthetic COCO split and the 60 Open Images photos | Runs; `run.json` records `w_box_giou` 1.0; `train_terms` has `box_giou` |

On the 60 Open Images photos, the large persons that a photo edge cuts have a gap > 4 cells at 30
percent (run 1) and 26 percent (run 2) of their cut edges (27 edges). This set is small; the
Colab run measures `oi_val` and COCO `val`.

## Results of the Colab runs

[P1 report](../../../docs/exploration/p1_people_sky_head.md): COCO validation, and end to end on
the quality photos (commands in its sections 3.4 and 10.2).
