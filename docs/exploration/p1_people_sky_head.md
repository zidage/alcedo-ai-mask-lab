# P1: People and sky head on SAM 2.1 tiny `image_embed` (step 1)

**Issue:** [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1); run 2:
[#4](https://github.com/zidage/alcedo-ai-mask-lab/issues/4).
**Design:** [README section 5](../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features),
step 1, and [section 4.1](../../README.md#41-how-people-and-sky-work).
**Previous step:** [P0](p0_sam_feature_localization.md).
**Date:** 2026-10-20; runs 2 and 3: 2026-10-21.
**Status:** the first Colab run and the end-to-end run on the quality photos are done. Run 2 (COCO
and Open Images V7) is done and measured (section 10). Run 3 (run 2 with a GIoU term on the person
box) is done and measured (section 11). The prompt rules and the People list threshold are user
decisions (section 11.7).

> **Images are local only.** The sheets show photos that must not be shared again. They are
> under `results/p1_people_sky_head/sheets/` (`*.jpg` is in `.gitignore`). To make them again,
> run the commands of section 3.4 with the local evidence under `build/tmp/explore/`.

## 1. Question

P0 showed that `image_embed` of SAM 2.1 tiny holds the person and sky information, but a
prototype without training cannot give a calibrated "found" score, separate people from animals,
or give one box per person. Step 1 trains a small head on the frozen `image_embed`. This report
answers:

1. Does the head reach usable measurements on COCO 2017 (validation split)?
2. On the 14 quality photos, end to end with the SAM 2.1 tiny decoder: does the head find people
   and sky on the photos that have them and not on the photos that do not, and do its prompts
   select the referenced person or the sky?
3. Which P0 failures does the head correct, and which failures remain?

## 2. Result summary

- **Run 3 (GIoU term on the person box, section 11):** all criteria of issue #4 on COCO `val` and
  `oi_val` pass, and the boxes of large persons fit better on COCO (IoU >= 0.75: 0.490 to 0.528;
  cut edges with a gap > 4 cells: 0.163 to 0.132). On the quality photos run 3 is worse: 3 of 7
  referenced persons (run 2: 5 of 7) and 2 of 3 sky masks within 0.02 of E1. `DSC00674` is not
  corrected (0.000): its box is almost the full photo, and SAM 2.1 tiny selects the background.
  Box edge changes of 2 to 4 cells change the SAM 2.1 selection. Recommendation: keep run 2; the
  next step is a prompt rule change (box with the center point), not a box loss change.
- **Run 2 (COCO and Open Images V7, section 10):** all criteria of issue #4 on COCO `val` and
  `oi_val` pass (`oi_val` person AP50 0.473 to 0.533). On the quality photos, 5 of 7 referenced
  persons are within 0.02 of the E1 box (run 1: 4 of 7), but `DSC00674` changed from pass to fail
  (0.992 to 0.731): the box bottom edge is 5.6 cells above the photo edge. The waterfall stays sky
  (0.999). Recommendation: use run 2, and train the next run with an IoU loss on the boxes.

The bullets below are for run 1.

- **COCO validation (2,500 photos, checkpoint epoch 13):** sky cell IoU 0.895; presence AUC
  0.992 for sky and 0.989 for people (best F1 0.955 and 0.958); person AP50 0.623; top-1 box hit
  0.845. The validation measurements stop to increase after about epoch 8 to 13, while the
  training loss continues to decrease.
- **Person presence passes on the quality photos.** At the COCO threshold (0.602), people are
  found on all 7 photos with a referenced person (0.968 to 1.000) and on none of the 4 photos
  without people (0.002 to 0.098). The squirrel of `DSC_1188` scores 0.002. In P0 no threshold
  separated these photos (margin -0.177); now the margin is 0.870.
- **Sky presence fails on one photo.** At the COCO threshold (0.650), sky is found on the 3
  photos with referenced sky (0.995 to 0.999) and not on `DSC_1188` (0.003) and `DSC_1535`
  (0.005). The waterfall of `DSC00936` scores 0.942: the head marks the whole falling-water curtain
  (82 percent of the photo) as sky. By eye, the area is falling water, not fog.
- **Sky masks are good.** The 3 most probable cells of each sky region as SAM 2.1 points give
  IoU 0.980 to 0.994 against the reference (E1 sky point: 0.993 to 0.997; P0 prototype points:
  0.969 to 0.995).
- **People: 4 of 7 referenced persons get a mask within 0.02 of the E1 box IoU or better.** The
  head box selects `DSC00936` (0.989), `DSC00674` (0.992), `DSC00527` (0.948, rank 11 in a crowd),
  and the small boarder of `2088064441` (0.689; the E1 box gives 0.269). Three fail:
  - `DSC_1535`: the box of the main person is at rank 7 (score 0.42) after 6 background
    pedestrians (0.47 to 0.65). Its top edge is 5.8 cells below the top of the head, so the mask
    has no head and no shirt (IoU 0.855; E1 box 0.993).
  - `P1024439`: the bassist box is too small (top 4.4 cells low, bottom 3.7 cells high, right
    edge 3.8 cells short). The mask selects only parts of the person (IoU 0.325; E1 box 0.934).
  - `DSC00326`: IoU 0.911 at rank 1 (E1 box 0.973).
- **Box rules that the product needs:** a large person gives several local maxima of the center
  heatmap (9 boxes with IoU >= 0.5 on `DSC_1535`). The suppression of overlapping boxes (IoU >= 0.5)
  removes them. A box that extends past the photo edge selects the wrong area (`DSC00674`: IoU
  0.000 without clipping, 0.992 with the box clipped to the photo).
- **Cost:** the head runs in 19.5 ms on CPU (median of 14 photos, 4 threads); it has 275,656
  parameters (1.1 MB ONNX).
- **Recommendation:** keep the head for sky and people (open decision O4: the head is better than
  the P0 prototypes on every P0 failure, except the waterfall). Before a second training run:
  measure the People list threshold and the box fit per person size on COCO validation with
  `evaluate_head.py` (section 9). The training changes (box size of large persons, waterfall
  against sky) need a user decision.

## 3. Method

### 3.1 Training (Colab)

| Item | Value |
| --- | --- |
| Encoder | SAM 2.1 Hiera tiny, frozen, float32 with TF32 off (`facebookresearch/sam2` at `2b90b9f5ceec907a1c18123530e92e794ad901a4`, checkpoint `sam2.1_hiera_tiny.pt` SHA-256 `7402e0d8...`). Apache-2.0. `image_embed` cached as float16 |
| Head | [`head_model.py`](../../scripts/poc/people_sky_head/head_model.py): LayerNorm, 1 x 1 conv 256 -> 96, 3 residual 3 x 3 blocks (dilation 1, 2, 4), sky map, center heatmap, offset, log size, 2 presence scores. 275,656 parameters |
| Objective | [`head_losses.py`](../../scripts/poc/people_sky_head/head_losses.py), all weights 1 |
| Optimizer | AdamW, learning rate 1e-3, weight decay 1e-4, 1 epoch warmup, cosine decay, batch 32, 20 epochs |
| Training data | COCO 2017 panoptic `train2017` (annotations CC BY 4.0), only the photos with license 4 (19,470), 5 (10,901), or 7 (465): 30,836 photos (decision U2). License 8 has no panoptic photo. Licenses 1 to 3 (non-commercial) and 6 (no derivatives) are excluded |
| Validation data | The first 2,500 photos of `val2017` after a shuffle with seed 0, all licenses (measured only) |
| Machine | Google Colab, NVIDIA L4, PyTorch 2.11.0+cu130 |
| Time | Encoder 2,353 s (train) and 193 s (validation); training about 225 s per epoch; CUDA memory peak of the encoder 2.9 GB |
| Checkpoint | `best.pt`: epoch 13 (highest mean of sky cell IoU and AP50) |
| Export | `people_sky_head.onnx`, opset 17, 1,119,131 bytes, SHA-256 `af44affc0440961205da91da8d029e485c99c0fe3ee9e8883b6f3ebba8ebbc55`. ONNX Runtime CPU against PyTorch CPU on 16 validation photos: maximum absolute difference 2.1e-5 (tolerance 1e-4) |
| Records | [`results/p1_people_sky_head/colab_run/`](../../results/p1_people_sky_head/colab_run/): `run.json`, `metrics.jsonl`, `people_sky_head.json`, the split summaries with the SHA-256 of the full manifests |

### 3.2 Quality photos (development machine)

| Item | Value |
| --- | --- |
| Photos | The 14 quality photos at the 4096 px application input (E4 input set). Not committed |
| Presence of the classes | The P0 [`class_presence.json`](../../scripts/poc/sam_feature_localization/class_presence.json) (by eye) |
| References | The P0 references (E1 SAM 2.1 base plus masks): one person or one sky per photo |
| SAM 2.1 tiny | The float32 ONNX graphs of E1 and E4 (`point_encoder.onnx` SHA-256 `17a554b4...`, `point_decoder.onnx` `6f9a0ef2...`) |
| Thresholds | Fixed before this run: the best-F1 presence thresholds of the COCO validation split in `people_sky_head.json` (sky 0.650, person 0.602). The 14 photos do not set a threshold |
| Machine | Windows 11 build 22635, Intel Core i7-12700H (20 logical CPUs), 31.8 GB. Python 3.12.4, ONNX Runtime 1.31.0 (CPU, 4 intra-op threads, CPU memory arena off), PyTorch 2.14.1+cpu (box decoding) |

### 3.3 Rules

- **Sky:** found when `presence[sky] >= 0.650`. The prompts are one set per 8-connected region of
  cells with sky probability >= 0.5 and at least 3 cells: the centers of the 3 most probable cells
  of the region, as positive points. The sky mask is the union of the region masks.
- **People:** found when `presence[person] >= 0.602`. The People list is
  `head_model.decode_people` with `nms_iou = 0.5`: the local maxima (3 x 3) of the center heatmap
  with a score >= 0.05, at most 20, highest score first, then a box whose IoU with a box of higher
  score is >= 0.5 is removed. Each box, clipped to the photo, is one SAM 2.1 box prompt.
- **Mask selection:** the E1 rule (`select_single_mask`) for a box or several points. IoU at the
  photo size (bilinear upsampling of the logits, threshold 0) against the reference.
- **The People list threshold is not set.** COCO validation gives it (`person_list_best_f1_nms`,
  section 9). This report gives the rank and the score of the referenced person. Each photo has
  one reference: a box on another person has an IoU near 0, although it selects a person.
- **Proposed acceptance criteria** (the user confirms them):
  - presence: the COCO threshold finds the class on every referenced photo and on no photo
    without the class;
  - sky: IoU at most 0.02 below the E1 sky point on every referenced sky photo;
  - people: the referenced person is in the People list with an IoU at most 0.02 below the E1 box.

### 3.4 Commands

Run one command at a time (Git Bash, Windows). Only `evaluate_quality_photos.py` runs a model.

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"; I=$E/e4/input_4096
H=$E/p1/colab_run_2026-10-20/export; R=results/p1_people_sky_head; S=scripts/poc/people_sky_head
# End to end: encoder, head, SAM 2.1 tiny decoder (quality_photos.json).
$PY $S/evaluate_quality_photos.py --head-onnx $H/people_sky_head.onnx \
  --head-json $H/people_sky_head.json --photos $I/photos \
  --onnx-dir $E/e4/onnx_float32_batch16/tiny --reference-masks $I/reference_masks \
  --reference-scores $E/e1/compare/base_plus/compare.json --out-dir $R \
  --logit-dir $E/p1/quality_logits --threads 4
# Sheets (local only).
$PY $S/make_quality_sheets.py --photos $I/photos --results $R/quality_photos.json \
  --logits $E/p1/quality_logits --out-dir $R/sheets
```

The Colab output of the training run is under `build/tmp/explore/p1/colab_run_2026-10-20/`
(`run/best.pt`, `run/last.pt`, `export/people_sky_head.onnx`, the full split manifests).

## 4. Results: COCO validation

From [`metrics.jsonl`](../../results/p1_people_sky_head/colab_run/metrics.jsonl).

| Epoch | Train loss | Sky cell IoU | Presence AUC sky | Presence AUC person | AP50 | Top-1 hit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.742 | 0.822 | 0.986 | 0.980 | 0.376 | 0.559 |
| 4 | 2.134 | 0.872 | 0.991 | 0.986 | 0.579 | 0.803 |
| 8 | 1.902 | 0.885 | 0.992 | 0.988 | 0.606 | 0.824 |
| 12 | 1.689 | 0.897 | 0.992 | 0.988 | 0.611 | 0.842 |
| **13** | 1.639 | **0.895** | **0.992** | **0.989** | **0.623** | **0.845** |
| 16 | 1.505 | 0.893 | 0.992 | 0.988 | 0.610 | 0.843 |
| 19 | 1.450 | 0.891 | 0.992 | 0.988 | 0.604 | 0.838 |

- The validation measurements stop to increase after about epoch 8 to 13. The training loss
  continues to decrease (presence term 0.061 at epoch 13, 0.039 at epoch 19). More epochs of the
  same data do not help.
- The presence output is better than the map maximum: person AUC 0.989 against 0.969, sky 0.992
  against 0.987.
- The best-F1 presence threshold changes much between epochs (sky 0.13 to 0.65, person 0.53 to
  0.75). The threshold belongs to one checkpoint; it must be measured again after each training
  run.

## 5. Results: presence on the quality photos

| Photo | Sky status | Sky presence | Sky found | Person status | Person presence | People found |
| --- | --- | ---: | --- | --- | ---: | --- |
| `2088064441` | unreferenced | 0.991 | yes | referenced (small) | 0.995 | yes |
| `5761824892` | unreferenced (foliage gaps) | 0.036 | no | absent | 0.026 | no |
| `DSC00326` | unreferenced (small gap) | 0.988 | yes | referenced | 0.968 | yes |
| `DSC00527` | referenced | 0.999 | yes | referenced (crowd) | 1.000 | yes |
| `DSC00562` | unreferenced | 0.998 | yes | unreferenced (very small) | 0.974 | yes |
| `DSC00602` | unreferenced | 0.995 | yes | unreferenced (passengers) | 0.207 | no |
| `DSC00674` | uncertain (sunset backdrop) | 0.130 | no | referenced (close-up) | 1.000 | yes |
| `DSC00728` | referenced | 0.999 | yes | absent | 0.004 | no |
| `DSC00936` | **absent (waterfall)** | **0.942** | **yes** | referenced | 0.998 | yes |
| `DSC_1188` | absent | 0.003 | no | absent (squirrel) | 0.002 | no |
| `DSC_1456` | unreferenced | 0.994 | yes | unreferenced | 1.000 | yes |
| `DSC_1535` | absent | 0.005 | no | referenced | 1.000 | yes |
| `Nikon-D800e-raw-00007` | referenced | 0.995 | yes | absent | 0.098 | no |
| `P1024439` | unreferenced | 0.995 | yes | referenced | 1.000 | yes |

- **People:** lowest referenced 0.968, highest absent 0.098. The criterion passes. The passengers
  of `DSC00602` are behind the tram windows; the head does not find them (0.207).
- **Sky:** lowest referenced 0.995, highest absent 0.942 (the waterfall). The criterion fails at
  the COCO threshold. A threshold between 0.942 and 0.988 would separate these 14 photos, but this
  set has one hard negative; a threshold from these photos is not a product threshold.
- Sky is found on 7 of the 8 photos with unreferenced sky; `5761824892` (small gaps in the
  foliage) is not found, as in P0.

## 6. Results: sky masks

| Photo | Regions (cells) | Sky cell IoU | IoU | E1 sky point | P0 3 points |
| --- | --- | ---: | ---: | ---: | ---: |
| `DSC00527` | 1 (1,707) | 0.981 | 0.988 | 0.997 | 0.995 |
| `DSC00728` | 1 (1,467) | 0.978 | 0.994 | 0.997 | 0.992 |
| `Nikon-D800e-raw-00007` | 1 (1,059) | 0.992 | 0.980 | 0.993 | 0.969 |

All three are within 0.02 of the E1 sky point. On the photos with unreferenced sky the region
count is 1, except `DSC00602` (2 regions: the sky above the buildings and a piece at the left,
807 and 19 cells) and `P1024439` (4 regions between the trees and the high-rise buildings, 206,
64, 15, and 8 cells). On the sheets, the region masks follow the tree and building horizons and
leave out the overhead wires of `DSC00602`. On `DSC00936` the one region (3,343 cells) is the
waterfall: the sky mask covers 82 percent of the photo.

## 7. Results: people

IoU against the one reference of each photo. Rank and score in the People list (after the
suppression of overlapping boxes).

| Photo | Local maxima | List entries (score >= 0.05) | Top-1 IoU (score) | Referenced person: rank, score, IoU | E1 box | Within 0.02 of E1 |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `DSC00936` | 4 | 2 | 0.989 (0.72) | 1, 0.72, **0.989** | 0.989 | yes |
| `DSC00674` | 18 | 6 | 0.992 (0.66) | 1, 0.66, **0.992** | 0.988 | yes |
| `DSC00527` | 20 | 20 | 0.000 (0.75, another person) | 11, 0.62, **0.948** | 0.957 | yes |
| `2088064441` | 18 | 18 | 0.689 (0.33) | 1, 0.33, **0.689** | 0.269 | yes |
| `DSC00326` | 10 | 9 | 0.911 (0.45) | 2, 0.11, 0.922 | 0.973 | no |
| `DSC_1535` | 20 | 18 | 0.000 (0.65, a pedestrian) | 7, 0.42, 0.855 | 0.993 | no |
| `P1024439` | 20 | 10 | 0.325 (0.29) | 1, 0.29, 0.325 | 0.934 | no |

Observations from the sheets:

- `DSC_1535`: the 6 boxes before the main person are pedestrians in the background (the P0
  ignore boxes). The main person box (rank 7) begins 5.8 cells below the top of the head; SAM 2.1
  selects the suit without the head and the shirt.
- `P1024439`: the bassist box is short at the top (4.4 cells), at the bottom (3.7 cells), and at
  the right (3.8 cells). SAM 2.1 selects the head, an arm, and the legs. A larger box at rank 5
  (score 0.23) includes the person and the amplifier.
- `DSC00527`: the list has people of the crowd from rank 1 to 20 (scores 0.61 to 0.75). The
  kneeling woman (the referenced person, in the foreground) is at rank 11. U5 accepts
  missed people in crowds, but the order puts the small people first.
- `2088064441`: the boarder box excludes the board, so the mask is the person; the E1 box
  includes the board and a bag.
- **Duplicates:** without the suppression, `DSC_1535` has 9 boxes with a box IoU >= 0.5 against
  the reference box of the main person, and `DSC00674` has 9 on the close-up. After the
  suppression at 0.5, one list entry of each photo gives a mask with IoU >= 0.5. `DSC00326` keeps
  3 such entries (ranks 1 to 3, scores 0.45, 0.11, 0.09) and `2088064441` 2 (ranks 1 and 3,
  scores 0.33 and 0.28); the list threshold decides if the low-score entries are shown.
- **Clipping:** the close-up box of `DSC00674` extends from x = -7.0 to 75.8 cells. Without
  clipping, SAM 2.1 selects an area outside the person (27 percent of the photo, IoU 0.000; first
  run of this report, local only:
  `build/tmp/explore/p1/quality_run_without_box_clipping/quality_photos.json`);
  with the box clipped to the photo, it selects the person (0.992).

## 8. Cost

From `quality_photos.json` (one run, CPU, 4 threads, arena off).

| Item | Median | Note |
| --- | ---: | --- |
| Head (`people_sky_head.onnx`) | 19.5 ms | 14 photos |
| SAM 2.1 tiny decoder | 60.0 ms | 97 prompts |
| SAM 2.1 tiny encoder | 2,798 ms | 14 photos |
| Peak working set | 1,463 MB | encoder, head, decoder in one process |

The encoder and decoder times of this run are about twice the P0 values for the same graphs and
settings (1,375 ms and 31.5 ms). The first run of this report (without the box clipping, same
photos) gave 1,733 ms, 40.5 ms, and 13.7 ms for the head. The cause of the difference between the
runs was not checked; these values are not a cost measurement of the encoder or the decoder. The encoder runs anyway for the stroke and
box tools, so People and Sky add the head and the decoder runs. GPU and DirectML were not
measured.

## 9. Recommendation and next steps

**Keep the trained head for People and Sky (O4).** The head corrects the P0 failures: the
calibrated presence (people separate with a margin of 0.870), person against animal (the squirrel
0.002), and one box per person (the box selects the whole person on 4 of 7 photos). Sky masks are
within 0.02 of the E1 sky point.

Next steps:

1. **COCO measurements of `best.pt`** (Colab, about 15 minutes, `TRAIN = False` in the notebook):
   `evaluate_head.py` gives the People list threshold after the suppression
   (`person_list_best_f1_nms`) and the box fit per person height (`person_box_fit_nms`: mean best
   IoU, share with IoU >= 0.75, median log ratio of box height and width). These show whether the
   short boxes of section 7 are a general error for large persons on COCO, or only on these photos.
2. **Product rules for the People list** (user decision): the order of the list (score order
   puts the small background people before the main person on `DSC_1535` and `DSC00527`), and the
   list threshold from step 1.
3. **Training changes** (user decision, after step 1):
   - box size of large persons: an IoU loss (GIoU) on the decoded boxes at the center cells, in
     addition to the log-size L1;
   - the waterfall as sky: COCO has few photos of falling water. A fix needs photos of
     waterfalls, spray, and fog with a sky label, under a license that permits commercial use
     (U2). Open Images V7 has no sky masks. The COCO-Stuff `fog` class was not checked against
     `sky-other-merged`, because the false sky on `DSC00936` is falling water, not fog.
4. **Open Images V7 as a second training source (implemented, not run).** The user decided to
   improve the generalization in general, not only the waterfall case. Open Images V7 has point
   labels for `Sky` (575,362 points, 284,093 "yes") and `Cloud`, and person boxes, on photos
   listed as CC BY 2.0 (annotations CC BY 4.0). The second Colab run adds 12,000 photos with sky
   points and 12,000 photos with person boxes to the COCO training set, and measures an Open
   Images validation split (`oi_val`, 2,500 photos) each epoch. The first run is measured on
   `oi_val` in the same session. Method and checks:
   [script README](../../scripts/poc/people_sky_head/README.md#training-data-and-licenses).

## 10. Run 2: COCO and Open Images V7

**Issue:** [#4](https://github.com/zidage/alcedo-ai-mask-lab/issues/4). **Date:** 2026-10-21.

### 10.1 Question and result summary

Does a second training run on COCO and Open Images V7 generalize better than the first run (COCO
only), without a loss on COCO and on the quality photos? The acceptance criteria are the
proposal of issue #4 (the user confirms the thresholds).

| Criterion (issue #4) | Run 1 | Run 2 | Result |
| --- | ---: | ---: | --- |
| COCO `val` sky cell IoU, at most 0.01 lower | 0.8947 | 0.8941 | pass (-0.0006) |
| COCO `val` presence AUC sky, at most 0.01 lower | 0.9922 | 0.9930 | pass (+0.0007) |
| COCO `val` presence AUC person, at most 0.01 lower | 0.9886 | 0.9883 | pass (-0.0003) |
| COCO `val` person AP50, at most 0.01 lower | 0.6234 | 0.6394 | pass (+0.0159) |
| `oi_val` person AP50, higher | 0.4732 | 0.5334 | pass (+0.0602) |
| `oi_val` person presence AUC, higher | 0.9769 | 0.9807 | pass (+0.0037) |
| `oi_val` sky point false sky share, not higher | 0.0201 (78 of 3,879 cells) | 0.0199 (77 of 3,879 cells) | pass (1 cell) |
| Quality photos: person presence | pass, margin 0.870 | pass, margin 0.881 | as good |
| Quality photos: sky presence on the photos without sky | fails on the waterfall (0.942) | fails on the waterfall (0.999) | as in run 1 (recorded, not required) |
| Quality photos: sky masks within 0.02 of E1 | 3 of 3 | 3 of 3 | as good |
| Quality photos: referenced person within 0.02 of E1 | 4 of 7 | 5 of 7 | **2 new passes, 1 new failure** (`DSC00674`) |

- Run 2 passes all criteria on COCO `val` and `oi_val`. On `oi_val`, person AP50 increases by
  0.060 and the box fit of large persons improves. The false sky share passes by 1 cell; the
  value changes between 0.008 and 0.066 from epoch to epoch, so this criterion gives no signal.
- On the quality photos, the people result is better on 3 photos (`DSC_1535`, `P1024439`,
  `DSC00326`) and worse on 1 photo (`DSC00674`: IoU 0.992 to 0.731). The cause is the box: its
  bottom edge is 5.6 cells above the photo edge, but the person continues to the photo edge. With
  the bottom edge at the photo edge, the same run 2 box gives 0.992 (section 10.5). The criterion
  "as good as in the first run" is not met per photo.

### 10.2 Inputs and method

| Item | Value |
| --- | --- |
| Training | `train_head.py` on COCO `train` (30,836 photos) and `oi_train` (24,000), 20 epochs, all other arguments as in run 1. `best.pt` is selected on COCO `val` (mean of sky cell IoU and person AP50): epoch 15. Colab, NVIDIA L4, PyTorch 2.11.0+cu130 |
| Encoder cache | The same SAM 2.1 tiny commit, checkpoint SHA-256, and preprocessing as run 1 (`*_image_embed_progress.json`) |
| Validation | `evaluate_head.py` measures run 1 `best.pt` (epoch 13) and run 2 `best.pt` (epoch 15) on COCO `val` and `oi_val` in the same Colab session (`best_val_measurements.json`) |
| Quality photos | `evaluate_quality_photos.py` with the run 2 ONNX (SHA-256 `e02f2a1e...`), the same photos, references, SAM 2.1 tiny graphs, and machine as section 3.2. The presence thresholds come from the run 2 export: sky 0.423, person 0.533 |
| Run 1 values | Run 1 on COCO `val` and `oi_val`: [`colab_run/best_val_measurements.json`](../../results/p1_people_sky_head/colab_run/best_val_measurements.json). Run 1 on the quality photos: `results/p1_people_sky_head/quality_photos.json` (not run again) |
| Records | [`results/p1_people_sky_head/run2_coco_openimages/`](../../results/p1_people_sky_head/run2_coco_openimages/) (`colab_run/`, `quality_photos.json`); the sheets stay local |

The COCO manifests of run 2 are equal to the run 1 manifests (only the `seconds` field is
different). The Colab output is under `build/tmp/explore/p1/colab_run2_coco_openimages/`.

Photo count per license of each split (`colab_run/*_manifest_summary.json`):

| Split | Use | Photos | Licenses |
| --- | --- | ---: | --- |
| COCO `train` | training | 30,836 | Attribution 19,470; Attribution-ShareAlike 10,901; no known copyright restrictions 465 |
| COCO `val` | measurement only | 2,500 | Attribution-NonCommercial-ShareAlike 707; Attribution-NonCommercial 330; Attribution-NonCommercial-NoDerivs 720; Attribution 408; Attribution-ShareAlike 200; Attribution-NoDerivs 132; no known copyright restrictions 3 |
| `oi_train` | training | 24,000 | CC BY 2.0 24,000 (10,265 authors) |
| `oi_val` | measurement only | 2,500 | CC BY 2.0 2,500 (2,125 authors) |

Annotations: COCO 2017 CC BY 4.0, Open Images V7 CC BY 4.0. The COCO `val` photos with a
non-commercial license are only measured, as in run 1.

Commands (Git Bash, development machine, one at a time):

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"; I=$E/e4/input_4096
H=$E/p1/colab_run2_coco_openimages/export; R=results/p1_people_sky_head/run2_coco_openimages
S=scripts/poc/people_sky_head; L=$E/p1/colab_run2_coco_openimages/quality_logits
# End to end: encoder, head, SAM 2.1 tiny decoder (quality_photos.json).
$PY $S/evaluate_quality_photos.py --head-onnx $H/people_sky_head.onnx \
  --head-json $H/people_sky_head.json --photos $I/photos \
  --onnx-dir $E/e4/onnx_float32_batch16/tiny --reference-masks $I/reference_masks \
  --reference-scores $E/e1/compare/base_plus/compare.json --out-dir $R \
  --logit-dir $L --threads 4
# Sheets (local only).
$PY $S/make_quality_sheets.py --photos $I/photos --results $R/quality_photos.json \
  --logits $L --out-dir $R/sheets
```

### 10.3 Results: COCO `val` and `oi_val`

Per epoch, from [`metrics.jsonl`](../../results/p1_people_sky_head/run2_coco_openimages/colab_run/metrics.jsonl)
(`val` is COCO, `val_extra` is `oi_val`):

| Epoch | Train loss | COCO sky cell IoU | COCO AP50 | COCO top-1 hit | `oi_val` AP50 | `oi_val` presence AUC person | `oi_val` false sky share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.004 | 0.810 | 0.447 | 0.658 | 0.301 | 0.969 | 0.018 |
| 4 | 2.110 | 0.861 | 0.603 | 0.849 | 0.453 | 0.981 | 0.009 |
| 7 | 1.943 | 0.812 | 0.657 | 0.878 | 0.590 | 0.984 | 0.031 |
| 10 | 1.805 | 0.888 | 0.622 | 0.864 | 0.471 | 0.979 | 0.020 |
| **15** | 1.634 | **0.894** | **0.639** | **0.871** | **0.533** | **0.981** | **0.020** |
| 19 | 1.586 | 0.895 | 0.630 | 0.868 | 0.523 | 0.980 | 0.023 |

Both checkpoints, from `best_val_measurements.json` (run 1 epoch 13, run 2 epoch 15):

| Measurement | COCO `val` run 1 | COCO `val` run 2 | `oi_val` run 1 | `oi_val` run 2 |
| --- | ---: | ---: | ---: | ---: |
| Sky cell IoU | 0.895 | 0.894 | 0.966 | 0.974 |
| Sky point AUC / accuracy | - | - | 0.994 / 0.978 | 0.996 / 0.983 |
| Presence AUC sky | 0.992 | 0.993 | not defined | not defined |
| Presence AUC person | 0.989 | 0.988 | 0.977 | 0.981 |
| Person AP50 | 0.623 | 0.639 | 0.473 | 0.533 |
| Person AP50 after the suppression (`nms`) | 0.647 | 0.673 | 0.494 | 0.568 |
| Top-1 box hit | 0.845 | 0.871 | 0.760 | 0.806 |
| People list best F1 (threshold) | 0.702 (0.342) | 0.717 (0.313) | 0.605 (0.360) | 0.645 (0.349) |
| Large persons: mean best IoU | 0.701 | 0.729 | 0.667 | 0.703 |
| Large persons: share with IoU >= 0.75 | 0.404 | 0.490 | 0.329 | 0.450 |
| Large persons: median log ratio, height / width | -0.049 / -0.060 | -0.028 / -0.044 | -0.052 / -0.040 | -0.039 / -0.047 |

- `oi_val` has no photo with sky absent (1,042 present, 1,458 ignored), so the sky presence AUC
  is not defined there. The person presence AUC uses 1,360 present and 154 absent photos.
- The boxes of large persons are still short on both splits: the median height is 3 to 4 percent
  too small (log ratio -0.028 and -0.039), and the width 4 to 5 percent.
- The best-F1 presence thresholds of COCO `val` changed from 0.650 to 0.423 (sky) and from 0.602
  to 0.533 (person). The quality photos use the thresholds of run 2.

### 10.4 Results: presence and sky on the quality photos

| Photo | Sky status | Sky run 1 | Sky run 2 | Person status | Person run 1 | Person run 2 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `2088064441` | unreferenced | 0.991 | 0.997 | referenced (small) | 0.995 | 0.999 |
| `5761824892` | unreferenced (foliage gaps) | 0.036 | 0.312 | absent | 0.026 | 0.032 |
| `DSC00326` | unreferenced (small gap) | 0.988 | 0.941 | referenced | 0.968 | 0.999 |
| `DSC00527` | referenced | 0.999 | 1.000 | referenced (crowd) | 1.000 | 1.000 |
| `DSC00562` | unreferenced | 0.998 | 0.999 | unreferenced (very small) | 0.974 | 0.876 |
| `DSC00602` | unreferenced | 0.995 | 0.998 | unreferenced (passengers) | 0.207 | 0.293 |
| `DSC00674` | uncertain (sunset backdrop) | 0.130 | **0.773** | referenced (close-up) | 1.000 | 1.000 |
| `DSC00728` | referenced | 0.999 | 1.000 | absent | 0.004 | 0.116 |
| `DSC00936` | **absent (waterfall)** | 0.942 | **0.999** | referenced | 0.998 | 0.997 |
| `DSC_1188` | absent | 0.003 | 0.006 | absent (squirrel) | 0.002 | 0.004 |
| `DSC_1456` | unreferenced | 0.994 | 0.990 | unreferenced | 1.000 | 1.000 |
| `DSC_1535` | absent | 0.005 | 0.008 | referenced | 1.000 | 1.000 |
| `Nikon-D800e-raw-00007` | referenced | 0.995 | 1.000 | absent | 0.098 | 0.012 |
| `P1024439` | unreferenced | 0.995 | 0.987 | referenced | 1.000 | 1.000 |

- **People:** lowest referenced 0.997, highest absent 0.116 (`DSC00728`), threshold 0.533. The
  criterion passes with the margin 0.881 (run 1: 0.870). The squirrel scores 0.004.
- **Sky:** the referenced photos score 1.000. The waterfall of `DSC00936` scores 0.999 (run 1:
  0.942), and its sky mask covers 81 percent of the photo: the Open Images data does not correct
  the waterfall. `DSC00674` (uncertain) is now found as sky: 2 regions (639 and 153 cells, 22
  percent of the photo) on the blue background at the two sides of the head.
- **Sky masks:** IoU 0.993 (`DSC00527`), 0.990 (`DSC00728`), 0.982 (`Nikon-D800e-raw-00007`);
  E1 sky point 0.997, 0.997, 0.993. All are within 0.02 (run 1: 0.988, 0.994, 0.980).

### 10.5 Results: people on the quality photos

The People list threshold is the COCO `val` best F1 after the suppression of each run (run 1
0.342, run 2 0.313). "Shown" tells if the referenced person is in the list at that threshold.

| Photo | E1 box | Run 1: rank, score, IoU | Run 2: rank, score, IoU | Run 2 within 0.02 of E1 | Run 2 shown |
| --- | ---: | --- | --- | --- | --- |
| `DSC00936` | 0.989 | 1, 0.72, 0.989 | 1, 0.47, **0.989** | yes | yes |
| `DSC00674` | 0.988 | 1, 0.66, 0.992 | 1, 0.73, **0.731** | **no** (run 1: yes) | yes |
| `DSC00527` | 0.957 | 11, 0.62, 0.948 | 6, 0.54, **0.943** | yes | yes |
| `2088064441` | 0.269 | 1, 0.33, 0.689 | 1, 0.31, **0.686** | yes | no (0.3127) |
| `DSC00326` | 0.973 | 2, 0.11, 0.922 | 1, 0.46, **0.976** | **yes** (run 1: no) | yes |
| `DSC_1535` | 0.993 | 7, 0.42, 0.855 | 2, 0.54, **0.971** | no (0.022 below) | yes |
| `P1024439` | 0.934 | 1, 0.29, 0.325 | 1, 0.36, **0.938** | **yes** (run 1: no) | yes |

Observations from the local sheets (`results/p1_people_sky_head/run2_coco_openimages/sheets/`):

- `DSC_1535`: the main person is at rank 2, after one background pedestrian (score 0.60). The box
  now includes the head and the shoes; the mask misses parts of the hair and of the dark shirt
  (IoU 0.971, 0.002 below the limit).
- `P1024439`: the bassist box now includes the full person (IoU 0.938; run 1: 0.325).
- `DSC00674`: the box is `[1.1, 2.5, 63.9, 58.4]` cells; the woman continues to the bottom photo
  edge (64). SAM 2.1 selects the head and the neck without the shoulders and the jacket (mask 50
  percent of the photo; reference 67 percent). In run 1 the box extended past the photo edge, and
  the clipping put it at the photo edge. A local check decodes edited boxes on this photo with
  the same graphs and the same mask selection
  (`build/tmp/explore/p1/colab_run2_coco_openimages/dsc00674_box_check.py`, output
  `dsc00674_box_check.txt`):

  | Box (cells) | IoU |
  | --- | ---: |
  | Run 2 box `[1.1, 2.5, 63.9, 58.4]` | 0.731 |
  | Run 2 box, bottom edge at 64 | 0.992 |
  | Run 2 box, left and right edges at 0 and 64 | 0.730 |
  | Run 1 box after the clipping `[0, 3.7, 64, 63.3]` | 0.992 |

  The bottom edge alone causes the failure. This is the short box of large persons of section 7
  and section 10.3, on a person that the photo edge cuts. The check is a diagnosis, not a
  product rule.
- `2088064441`: the boarder is at rank 1, but its score (0.3127) is just below the run 2 list
  threshold (0.3132). In run 1 it was also below the threshold.

### 10.6 Cost

| Item | Run 1 | Run 2 | Note |
| --- | ---: | ---: | --- |
| Training time per epoch (L4, median) | 228 s | 448 s | 30,836 against 54,836 photos |
| Encoder cache of `oi_train` (L4) | - | 1,737 s | 24,000 photos, 13.8 photos/s |
| Head (`people_sky_head.onnx`), CPU | 19.5 ms | 12.4 ms | 14 photos, the same graph (275,656 parameters) |
| SAM 2.1 tiny decoder, CPU | 60.0 ms | 35.9 ms | 97 prompts in each run |
| SAM 2.1 tiny encoder, CPU | 2,798 ms | 1,576 ms | 14 photos |
| Peak working set | 1,463 MB | 1,466 MB | encoder, head, decoder in one process |

The two heads have the same graph. The CPU times of the development machine changed between the
runs (section 8). These values are not a cost comparison of the two heads.

### 10.7 Recommendation

**Use run 2 as the head.** It is better than run 1 on all measurements of `oi_val` and equal or
better on COCO `val`. It corrects `P1024439` and `DSC00326`, and it puts the main person of
`DSC_1535` at rank 2 (run 1: rank 7).

**Do the next training run with an IoU loss on the boxes (section 9, item 3).** The
`person_box_fit_nms` values show that the boxes of large persons are short in general (median
height 3 to 4 percent too small on COCO and Open Images), not only on the quality photos.
`DSC00674` shows the effect on SAM 2.1: a box edge 5.6 cells inside the photo, on a person that
the photo edge cuts, changes the selection from the person to the head. The run 2 records give
the values for the comparison.

The Open Images data does not correct the waterfall of `DSC00936` (0.999). It needs photos of
falling water with a "not sky" label (section 9, item 3).

User decisions:

1. Confirm the acceptance criteria of issue #4, and the result of the quality-photo criterion:
   5 of 7 persons (run 1: 4 of 7), but `DSC00674` changed from pass to fail.
2. The GIoU loss for the next run. **Decided 2026-10-21: run 3 adds it.** The term is
   1 - GIoU of the person box, with the predicted and the target box clipped to the photo (the
   box that SAM 2.1 gets), weight 1. A new measurement, `person_edge_fit_nms`, gives the gap
   between the box edge and the photo edge for the persons that the photo edge cuts (the
   `DSC00674` case). The notebook measures runs 1 and 2 again with it. Code and checks:
   [script README](../../scripts/poc/people_sky_head/README.md#objective).
3. The order of the People list and the list threshold (run 2: 0.313 on COCO `val`; the boarder
   of `2088064441` is 0.0005 below it).

## 11. Run 3: GIoU term on the person box

**Issue:** [#4](https://github.com/zidage/alcedo-ai-mask-lab/issues/4). **Date:** 2026-10-21.

### 11.1 Question and result summary

Does the GIoU term on the person box (section 10.7, decision 2) correct the short boxes of large
persons, and does it correct `DSC00674` without a loss on the other quality photos? Run 3 is run 2
with one change: `w_box_giou` 1 (runs 1 and 2: 0).

| Criterion | Run 1 | Run 2 | Run 3 | Run 3 result |
| --- | ---: | ---: | ---: | --- |
| COCO `val` sky cell IoU, at most 0.01 lower than run 1 | 0.8947 | 0.8941 | 0.8918 | pass (-0.0029) |
| COCO `val` presence AUC sky, at most 0.01 lower | 0.9922 | 0.9930 | 0.9918 | pass (-0.0004) |
| COCO `val` presence AUC person, at most 0.01 lower | 0.9886 | 0.9883 | 0.9879 | pass (-0.0007) |
| COCO `val` person AP50, at most 0.01 lower | 0.6234 | 0.6394 | 0.6446 | pass (+0.0212) |
| `oi_val` person AP50, higher than run 1 | 0.4732 | 0.5334 | 0.5297 | pass (+0.0565; run 2 -0.0037) |
| `oi_val` person presence AUC, higher than run 1 | 0.9769 | 0.9807 | 0.9802 | pass |
| `oi_val` sky point false sky share, not higher than run 1 | 0.0201 | 0.0199 | 0.0196 | pass (76 of 3,879 cells) |
| Quality photos: person presence | margin 0.870 | margin 0.881 | margin 0.937 | pass |
| Quality photos: sky presence on the photos without sky | fails on the waterfall | fails on the waterfall | fails on the waterfall (0.999) | as before (recorded, not required) |
| Quality photos: sky masks within 0.02 of E1 | 3 of 3 | 3 of 3 | **2 of 3** | **fail** (`Nikon-D800e-raw-00007` 0.942) |
| Quality photos: referenced person within 0.02 of E1 | 4 of 7 | 5 of 7 | **3 of 7** | **fail** |

- **The GIoU term does what it should on the validation splits, but the effect is small.** For
  large persons on COCO `val`, the share of boxes with IoU >= 0.75 increases from 0.490 to 0.528,
  the median width error from -4.4 to -2.6 percent, and the share of cut edges with a gap > 4
  cells decreases from 0.163 to 0.132. On `oi_val` the box fit of large persons is almost equal
  (IoU >= 0.75: 0.450 to 0.452), and the gap > 4 cells decreases from 0.224 to 0.202. AP50 and
  the top-1 hit do not change (differences of 0.01 or less in both directions).
- **On the quality photos run 3 is worse than run 2.** `DSC00674` is not corrected: it goes from
  0.731 to 0.000. `DSC_1535`, `DSC00326`, and `P1024439` fall below the limit. The box edges move
  by 2 to 4 cells between run 2 and run 3, and SAM 2.1 tiny changes its selection with such
  changes (section 11.5). The sky mask failure of `Nikon-D800e-raw-00007` comes from the point
  rule, not from the sky map (section 11.4).
- **Recommendation: keep run 2 as the head (section 11.7).** A box loss alone does not make the
  box prompt reliable on large persons.

### 11.2 Inputs and method

| Item | Value |
| --- | --- |
| Training | `train_head.py` as in run 2 (COCO `train` 30,836 and `oi_train` 24,000 photos, 20 epochs, seed 0), with `w_box_giou` 1. `best.pt`: epoch 15 (as run 2). Colab, NVIDIA L4, PyTorch 2.11.0+cu130 |
| Data | The four split manifests are equal to run 2 (only the `seconds` field is different). The encoder cache has the same SAM 2.1 tiny commit, checkpoint SHA-256, and preprocessing (`*_image_embed_progress.json`) |
| Validation | `evaluate_head.py` measures run 1 (epoch 13), run 2 (epoch 15), and run 3 (epoch 15) `best.pt` on COCO `val` and `oi_val` in the same Colab session, with the `head_metrics.py` of b4f530b (`best_val_measurements.json`). The values of runs 1 and 2 are equal to the earlier records; the files now also have `person_edge_fit_nms` |
| Export | `people_sky_head.onnx`, 1,119,131 bytes, SHA-256 `ce84bc6430860381aa9045396807f13ab5e1ad21eb9e7fd15e60d6d4d490258f`. ONNX Runtime CPU against PyTorch CPU on 16 validation photos: maximum absolute difference 2.3e-5 (tolerance 1e-4) |
| Quality photos | `evaluate_quality_photos.py` with the run 3 ONNX and the same photos, references, SAM 2.1 tiny graphs, rules (section 3.3), and machine as section 3.2. Presence thresholds from the run 3 export: sky 0.420, person 0.644. People list threshold (COCO `val` best F1 after the suppression): 0.3145 |
| Records | [`results/p1_people_sky_head/run3_box_giou/`](../../results/p1_people_sky_head/run3_box_giou/) (`colab_run/`, `quality_photos.json`); the sheets stay local |

The Colab output is under `build/tmp/explore/p1/colab_run3_box_giou/`. Run 1 and run 2 on the
quality photos were not run again: their values come from `quality_photos.json` of sections 3 and
10. The run 3 split summaries were made from the full manifests with the rule that gives the
committed run 2 summaries byte for byte (`build/tmp/p1_run3/scripts/make_manifest_summaries.py`).

Commands (Git Bash, development machine, one at a time):

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"; I=$E/e4/input_4096
H=$E/p1/colab_run3_box_giou/export; R=results/p1_people_sky_head/run3_box_giou
S=scripts/poc/people_sky_head; L=$E/p1/colab_run3_box_giou/quality_logits
# End to end: encoder, head, SAM 2.1 tiny decoder (quality_photos.json).
$PY $S/evaluate_quality_photos.py --head-onnx $H/people_sky_head.onnx \
  --head-json $H/people_sky_head.json --photos $I/photos \
  --onnx-dir $E/e4/onnx_float32_batch16/tiny --reference-masks $I/reference_masks \
  --reference-scores $E/e1/compare/base_plus/compare.json --out-dir $R \
  --logit-dir $L --threads 4
# Sheets (local only).
$PY $S/make_quality_sheets.py --photos $I/photos --results $R/quality_photos.json \
  --logits $L --out-dir $R/sheets
# Edited boxes on DSC00674 (section 11.5).
$PY $E/p1/colab_run3_box_giou/dsc00674_box_check.py .
```

### 11.3 Results: COCO `val` and `oi_val`

Per epoch, from [`metrics.jsonl`](../../results/p1_people_sky_head/run3_box_giou/colab_run/metrics.jsonl)
(large persons of COCO `val`; "gap > 4" is the share of the cut edges with a gap > 4 cells):

| Epoch | Train loss | GIoU term | Size L1 | COCO sky cell IoU | COCO AP50 | `oi_val` AP50 | Large: IoU >= 0.75 | Large: gap > 4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.462 | 0.455 | 0.759 | 0.805 | 0.432 | 0.280 | 0.087 | 0.319 |
| 4 | 2.400 | 0.285 | 0.335 | 0.854 | 0.590 | 0.431 | 0.375 | 0.177 |
| 7 | 2.213 | 0.262 | 0.299 | 0.792 | 0.659 | 0.587 | 0.430 | 0.085 |
| 10 | 2.065 | 0.247 | 0.274 | 0.882 | 0.619 | 0.453 | 0.473 | 0.103 |
| **15** | 1.883 | 0.227 | 0.243 | **0.892** | **0.645** | **0.530** | **0.528** | **0.132** |
| 19 | 1.831 | 0.220 | 0.233 | 0.896 | 0.638 | 0.520 | 0.527 | 0.120 |

The size L1 term of run 3 is lower than in run 2 at each epoch (epoch 19: 0.233 against 0.246),
so the GIoU term does not work against it. The gap > 4 cells changes between 0.085 and 0.223 from
epoch to epoch after epoch 4, as the false sky share in section 10.1.

The three checkpoints, from `best_val_measurements.json`:

| Measurement | COCO run 1 | COCO run 2 | COCO run 3 | `oi_val` run 1 | `oi_val` run 2 | `oi_val` run 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sky cell IoU | 0.895 | 0.894 | 0.892 | 0.966 | 0.974 | 0.971 |
| Presence AUC person | 0.989 | 0.988 | 0.988 | 0.977 | 0.981 | 0.980 |
| Person AP50 | 0.623 | 0.639 | 0.645 | 0.473 | 0.533 | 0.530 |
| Person AP50 after the suppression | 0.647 | 0.673 | 0.671 | 0.494 | 0.568 | 0.560 |
| Top-1 box hit | 0.845 | 0.871 | 0.869 | 0.760 | 0.806 | 0.793 |
| People list best F1 (threshold) | 0.702 (0.342) | 0.717 (0.313) | 0.716 (0.314) | 0.605 (0.360) | 0.645 (0.349) | 0.645 (0.338) |
| Medium persons: share with IoU >= 0.75 | 0.436 | 0.496 | 0.506 | 0.389 | 0.454 | 0.467 |
| Large persons: mean best IoU | 0.701 | 0.729 | 0.730 | 0.667 | 0.703 | 0.704 |
| Large persons: share with IoU >= 0.75 | 0.404 | 0.490 | 0.528 | 0.329 | 0.450 | 0.452 |
| Large persons: median log ratio, height / width | -0.049 / -0.060 | -0.028 / -0.044 | -0.023 / -0.026 | -0.052 / -0.040 | -0.039 / -0.047 | -0.030 / -0.029 |
| Large cut persons: mean best IoU (clipped) | 0.712 | 0.738 | 0.745 | 0.697 | 0.718 | 0.724 |
| Large cut persons: median edge gap (cells) | 0.42 | 0.69 | 0.34 | 0.70 | 1.47 | 1.05 |
| Large cut persons: share of edges with gap <= 1 cell | 0.585 | 0.553 | 0.599 | 0.534 | 0.439 | 0.494 |
| Large cut persons: share of edges with gap > 4 cells | 0.161 | 0.163 | 0.132 | 0.223 | 0.224 | 0.202 |

- The cut persons are 618 large persons (760 matched cut edges) on COCO `val` and 900 (1,385) on
  `oi_val`.
- The GIoU term reduces the median size error of large persons to 2 to 3 percent, and the error
  of the width more than the error of the height. The edges with a gap > 4 cells decrease by 2
  to 3 percentage points; 13 percent (COCO) and 20 percent (`oi_val`) of the cut edges of large
  persons still have the `DSC00674` error.
- Run 2 increased the median edge gap of large cut persons against run 1 (0.42 to 0.69 cells on
  COCO, 0.70 to 1.47 on `oi_val`); run 3 reduces it again (0.34 and 1.05).
- Each configuration has one training run (seed 0). The differences between runs 2 and 3 on AP50
  and the top-1 hit (0.01 or less) are smaller than the change between neighbouring epochs of one
  run (section 10.3), so they are not a signal.

### 11.4 Results: presence and sky on the quality photos

| Photo | Sky status | Sky run 2 | Sky run 3 | Person status | Person run 2 | Person run 3 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `2088064441` | unreferenced | 0.997 | 0.997 | referenced (small) | 0.999 | 0.992 |
| `5761824892` | unreferenced (foliage gaps) | 0.312 | 0.626 | absent | 0.032 | 0.055 |
| `DSC00326` | unreferenced (small gap) | 0.941 | 0.824 | referenced | 0.999 | 1.000 |
| `DSC00527` | referenced | 1.000 | 1.000 | referenced (crowd) | 1.000 | 1.000 |
| `DSC00562` | unreferenced | 0.999 | 0.999 | unreferenced (very small) | 0.876 | 0.953 |
| `DSC00602` | unreferenced | 0.998 | 0.998 | unreferenced (passengers) | 0.293 | 0.255 |
| `DSC00674` | uncertain (sunset backdrop) | 0.773 | 0.462 | referenced (close-up) | 1.000 | 1.000 |
| `DSC00728` | referenced | 1.000 | 1.000 | absent | 0.116 | 0.044 |
| `DSC00936` | **absent (waterfall)** | 0.999 | **0.999** | referenced | 0.997 | 0.993 |
| `DSC_1188` | absent | 0.006 | 0.008 | absent (squirrel) | 0.004 | 0.016 |
| `DSC_1456` | unreferenced | 0.990 | 0.956 | unreferenced | 1.000 | 1.000 |
| `DSC_1535` | absent | 0.008 | 0.007 | referenced | 1.000 | 1.000 |
| `Nikon-D800e-raw-00007` | referenced | 1.000 | 0.999 | absent | 0.012 | 0.018 |
| `P1024439` | unreferenced | 0.987 | 0.985 | referenced | 1.000 | 1.000 |

- **People:** lowest referenced 0.992 (`2088064441`), highest absent 0.055 (`5761824892`),
  threshold 0.644. The margin is 0.937 (run 2: 0.881). The squirrel scores 0.016.
- **Sky:** the referenced photos score 0.999 or more. The waterfall stays sky (0.999, mask 79
  percent of the photo). `5761824892` is found at 0.626 but has no region of 3 cells with
  probability >= 0.5, so it gets no mask. `DSC00674` (uncertain) is found at 0.462 with 2 regions
  (804 and 9 cells, mask 17 percent of the photo; run 2: 22 percent).
- **Sky masks:** IoU 0.993 (`DSC00527`), 0.995 (`DSC00728`), **0.942**
  (`Nikon-D800e-raw-00007`); E1 sky point 0.997, 0.997, 0.993. On `Nikon-D800e-raw-00007` the
  sky map is almost the same as in run 2 (1,046 against 1,059 cells with probability >= 0.5, one
  region). The 3 most probable cells are now all in the clear sky at the top of the photo (y 24 to
  88 of 1024 model pixels); in run 2 one point was on the lit clouds above the mountains. The
  sheet shows that the run 3 mask does not include these clouds. The rule "3 most probable cells
  of the region" selects points from a small part of a large region, and small differences of
  the sky map move them. This failure is not caused by the box term.

### 11.5 Results: people on the quality photos

"Shown" tells if the referenced person is in the People list at the run 3 threshold (0.3145).

| Photo | E1 box | Run 2: rank, score, IoU | Run 3: rank, score, IoU | Run 3 within 0.02 of E1 | Run 3 shown |
| --- | ---: | --- | --- | --- | --- |
| `DSC00936` | 0.989 | 1, 0.47, 0.989 | 1, 0.48, **0.989** | yes | yes |
| `DSC00674` | 0.988 | 1, 0.73, 0.731 | 2, 0.05, **0.029** (rank 1: 0.000) | no | no |
| `DSC00527` | 0.957 | 6, 0.54, 0.943 | 9, 0.53, **0.952** | yes | yes |
| `2088064441` | 0.269 | 1, 0.31, 0.686 | 1, 0.31, **0.702** | yes | no (0.309) |
| `DSC00326` | 0.973 | 1, 0.46, 0.976 | 1, 0.61, **0.947** | **no** (run 2: yes) | yes |
| `DSC_1535` | 0.993 | 2, 0.54, 0.971 | 2, 0.56, **0.946** | no | yes |
| `P1024439` | 0.934 | 1, 0.36, 0.938 | 7, 0.12, **0.827** (rank 1: 0.393) | **no** (run 2: yes) | no |

Observations from the local sheets (`results/p1_people_sky_head/run3_box_giou/sheets/`) and the
boxes of `quality_photos.json`:

- `DSC00674`: the rank 1 box is `[-1.8, 0.7, 64.9, 60.5]` cells. Clipped to the photo, it covers
  the photo except 0.7 cells at the top and 3.5 cells at the bottom. SAM 2.1 selects the
  background (mask 28 percent of the photo, IoU 0.000). The bottom edge is 2.1 cells nearer to
  the photo edge than in run 2, but the top edge moved from 2.5 to 0.7 cells. The edited boxes
  (same graphs and mask selection; script and output
  `build/tmp/explore/p1/colab_run3_box_giou/dsc00674_box_check.py` and `.txt`):

  | Box (cells) | IoU |
  | --- | ---: |
  | Run 3 box `[-1.8, 0.7, 64.9, 60.5]` | 0.000 |
  | Run 3 box, bottom edge at 64 | 0.373 |
  | Run 3 box, top edge at 2.5 | 0.794 |
  | Run 3 box, top 2.5 and bottom 58.4 (the run 2 top and bottom edges) | 0.730 |
  | Run 2 box `[1.1, 2.5, 63.9, 58.4]` | 0.731 |
  | Run 1 box after the clipping `[0, 3.7, 64, 63.3]` | 0.992 |

  For a person that fills the photo, the box is almost the full photo, and SAM 2.1 tiny gets
  only the gaps at the photo edges as information. A change of 2 cells at one edge changes the
  IoU from 0.99 to 0.00. The head cannot predict the edges with this precision: the median edge
  gap of the large cut persons is 0.3 to 1.0 cells, and 13 to 20 percent of the edges have a gap
  > 4 cells (section 11.3).
- `P1024439`: the rank 1 box `[41.3, 22.9, 51.9, 58.2]` starts at the shoulders (run 2: 19.2,
  above the head). SAM 2.1 selects the legs (IoU 0.393). A second box of the same person,
  `[38.8, 12.7, 57.9, 52.4]` (IoU 0.827), is at rank 7 with score 0.12: the two boxes have IoU
  < 0.5, so the suppression keeps both.
- `DSC_1535`: the box is 1.9 cells narrower at the left and 0.9 cells shorter at the bottom than
  in run 2 (IoU 0.971 to 0.946). The main person stays at rank 2 after a background pedestrian.
- `DSC00326`: the right edge is 2.4 cells shorter than in run 2 (IoU 0.976 to 0.947).
- On all four photos, the run 3 box is smaller than the run 2 box on at least one side, although
  the validation splits show larger boxes on average. The 7 persons are too few to show the
  average effect; they show the sensitivity of the box prompt.

### 11.6 Cost

| Item | Run 2 | Run 3 | Note |
| --- | ---: | ---: | --- |
| Training time per epoch (L4, median) | 448 s | 432 s | 54,836 photos; the GIoU term adds no measurable time |
| Head (`people_sky_head.onnx`), CPU | 12.4 ms | 14.2 ms | 14 photos, the same graph (275,656 parameters) |
| SAM 2.1 tiny decoder, CPU | 35.9 ms | 41.6 ms | 97 and 94 prompts |
| SAM 2.1 tiny encoder, CPU | 1,576 ms | 1,881 ms | 14 photos |
| Peak working set | 1,466 MB | 1,452 MB | encoder, head, decoder in one process |

The two heads have the same graph; the CPU time differences come from the machine (section 10.6).

### 11.7 Recommendation

**Keep run 2 as the head. Do not use run 3.** Run 3 passes all criteria of issue #4 on COCO `val`
and `oi_val`, and its boxes of large persons fit better on COCO. But on the quality photos it is
worse than run 2 (3 of 7 persons, 2 of 3 sky masks) and worse than run 1 (4 of 7), so it fails
the criterion "as good as in the first run".

**The GIoU term is not the fix for `DSC00674`.** The results of runs 1 to 3 show that the box
prompt of SAM 2.1 tiny is sensitive to edge errors of 2 to 4 cells on large persons, and the head
has such errors on 13 to 20 percent of the cut edges after the GIoU term. Between runs, the
quality photos change in both directions (run 2 against run 1: 2 new passes, 1 new failure; run
3 against run 2: IoU higher on 2 persons and lower on 4, 2 new failures) with almost equal
validation measurements. A further box loss change gives a small average gain
and does not remove this sensitivity.

User decisions:

1. Confirm run 2 as the head, and the run 3 result of the quality-photo criteria (3 of 7
   persons, 2 of 3 sky masks).
2. The prompt of a person. The head gives the center of each person (the heatmap peak) in
   addition to the box. A box with a positive point at the center gives SAM 2.1 the person
   inside the box, also when the box is almost the full photo. This is a change of the prompt
   rule (section 3.3), not of the training; it can be measured with the run 2 head on the
   quality photos and on COCO `val` without a new Colab run. The decision is the user's: it
   changes what the product sends to SAM 2.1.
3. The sky points. The rule "3 most probable cells of each region" can put all points in one
   part of a large region (`Nikon-D800e-raw-00007`). Points spread over the region (for example
   the most probable cell of each part of the region) are a rule change of the same kind as
   item 2.
4. The People list order and threshold (section 10.7, item 3), and the duplicate boxes of one
   person with IoU < 0.5 (`P1024439` run 3).
5. Seed variation: each configuration has one training run. To separate the effect of a training
   change from the run-to-run variation on the quality photos, a second seed of run 2 is needed.

## 12. Not measured and open items

- **Run 2:** GPU and DirectML cost; the sky presence AUC on `oi_val` (no photo with sky absent);
  the run 1 quality-photo values were not measured again in the same session as run 2.
- **Run 3:** the run-to-run variation of one configuration (one seed per configuration); the
  box with a center point and the spread sky points (section 11.7, items 2 and 3) are not
  measured. Runs 1 and 2 on the quality photos were not measured again in the same session.
- **The People list threshold** and the box fit on COCO (step 1 of section 9).
- **The set is small:** 7 referenced persons, 3 referenced skies, 4 photos without people, 3
  without sky (one of them the waterfall). One reference per photo.
- **GPU, DirectML, and Core ML** cost of the head.
- **"All people" mask** (the union of the list masks): not measured; it needs references for all
  people of a photo.
- **The user confirms** the acceptance criteria of section 3.3 and reviews the sheets
  (`results/p1_people_sky_head/sheets/`).

## 13. Changed files

Run 3 (section 11): `results/p1_people_sky_head/run3_box_giou/` (the Colab run records, the split
summaries, `quality_photos.json`), `best_val_measurements.json` of runs 1 and 2 (the same values,
with `person_edge_fit_nms` added), this report, [the index](README.md), [README](../../README.md)
section 5, and [`results/README.md`](../../results/README.md).

Run 2 (section 10): `results/p1_people_sky_head/run2_coco_openimages/` (the Colab run records,
the split summaries, `quality_photos.json`), `results/p1_people_sky_head/colab_run/best_val_measurements.json`
(run 1 on COCO `val` and `oi_val`), this report, [the index](README.md), and
[README](../../README.md) section 5.

Run 1:

- `scripts/poc/people_sky_head/`: `evaluate_quality_photos.py`, `make_quality_sheets.py`,
  `evaluate_head.py` (new); `head_model.py` (`decode_people` with the suppression of overlapping
  boxes); `head_metrics.py` (`person_ap50_nms`, `person_list_best_f1_nms`, `person_box_fit_nms`;
  the measurements of the first run do not change); the Colab notebook (`TRAIN = False` and the
  `evaluate_head.py` cell); `README.md`.
- `results/p1_people_sky_head/`: the Colab run records and `quality_photos.json` (the sheets stay
  local).
- This report, [the index](README.md), and [README](../../README.md) section 5.
