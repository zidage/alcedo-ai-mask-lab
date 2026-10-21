# P1: People and sky head on SAM 2.1 tiny `image_embed` (step 1)

**Issue:** [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1).
**Design:** [README section 5](../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features),
step 1, and [section 4.1](../../README.md#41-how-people-and-sky-work).
**Previous step:** [P0](p0_sam_feature_localization.md).
**Date:** 2026-10-20.
**Status:** the first Colab run and the end-to-end run on the quality photos are done. The People
list threshold (COCO validation) is open. The user decides the training changes of section 9.

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

## 10. Not measured and open items

- **The People list threshold** and the box fit on COCO (step 1 of section 9).
- **The set is small:** 7 referenced persons, 3 referenced skies, 4 photos without people, 3
  without sky (one of them the waterfall). One reference per photo.
- **GPU, DirectML, and Core ML** cost of the head.
- **"All people" mask** (the union of the list masks): not measured; it needs references for all
  people of a photo.
- **The user confirms** the acceptance criteria of section 3.3 and reviews the sheets
  (`results/p1_people_sky_head/sheets/`).

## 11. Changed files

- `scripts/poc/people_sky_head/`: `evaluate_quality_photos.py`, `make_quality_sheets.py`,
  `evaluate_head.py` (new); `head_model.py` (`decode_people` with the suppression of overlapping
  boxes); `head_metrics.py` (`person_ap50_nms`, `person_list_best_f1_nms`, `person_box_fit_nms`;
  the measurements of the first run do not change); the Colab notebook (`TRAIN = False` and the
  `evaluate_head.py` cell); `README.md`.
- `results/p1_people_sky_head/`: the Colab run records and `quality_photos.json` (the sheets stay
  local).
- This report, [the index](README.md), and [README](../../README.md) section 5.
