# P0: Locate people and sky from SAM 2.1 tiny image features (no training)

**Issue:** [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1).
**Design:** [README section 5](../../README.md#5-next-step-locate-people-and-sky-from-sam-21-image-features)
and [section 4.1](../../README.md#41-how-people-and-sky-work).
**Date:** 2026-10-20.
**Status:** measurements done. The user confirms the acceptance thresholds and decides step 1
and open decision O4.

> **Images are local only.** The sheets show photos that must not be shared again. They are
> under `results/p0_sam_feature_localization/sheets/` (`*.jpg` is in `.gitignore`). To make them
> again, run the commands of the
> [script README](../../scripts/poc/sam_feature_localization/README.md#commands) with the local
> evidence under `build/tmp/explore/`.

## 1. Question

People and Sky need one click (decision U4). The SAM3 text detector can find them, but it is too
large for this task (U11). The SAM 2.1 tiny encoder already computes a 64 x 64 feature map of the
whole photo for the stroke and box tools. Step P0 answers, without training:

1. Do the SAM 2.1 tiny features separate people and sky from the rest of the photo?
2. Do the highest-scoring cells give SAM 2.1 prompts that select the person or the sky?
3. Is it worth the work to train a small head on these features (step 1)?

## 2. Result summary

- **`image_embed` carries class information for both classes.** Within a photo, the prototype
  score separates the class cells from the other cells almost perfectly. The area under the ROC
  curve is 0.994 to 1.000 for people (6 photos) and 1.000 for sky (3 photos). The pooled
  high-resolution features do not carry it. `high_res_feats_1` has a median person AUC of 0.516
  and `high_res_feats_0` 0.784; they follow texture and color.
- **Sky: the acceptance criteria pass on `image_embed`.** On the three sky photos the 3
  highest-scoring cells are sky. One threshold (0.243) finds the sky on all three sky photos and
  on none of the three photos without sky, but the margin is small (0.026). The waterfall mist of
  `DSC00936` (0.230) is the nearest false result. At this threshold, the sky is also found on 6 of
  the 7 photos with unreferenced sky, and the 3 top cells lie on the sky in all 6. The 3 top
  cells as SAM 2.1 points give the whole sky: IoU 0.969 to 0.995, against 0.993 to 0.997 for the
  E1 sky point.
- **People: the acceptance criteria fail on every feature source.** On `image_embed`:
  - the 3 top cells lie on the referenced person on 5 of 7 photos. On `P1024439` they lie on the
    arm of the second musician, which is a person but not the referenced one. On `2088064441`
    (the boarder, about 1 cell wide) 2 of 3 lie on the person and one on a wooden post;
  - **no threshold separates the photos.** The squirrel of `DSC_1188` (0.293) scores higher than
    five of the seven referenced people. The prototype responds to "person or animal", and the
    absolute score is not comparable between photos. Without the small boarder, the margin is
    still negative (-0.056).
- **End to end, the prototype maps do not give good person prompts.** The 3 top cells cluster on
  one body part (face, hair), so SAM 2.1 selects that part (IoU 0.00 to 0.64). The box of the
  largest region works only where the person is the largest region: IoU 0.99 on `DSC_1535`,
  `DSC00674`, and `DSC00936` with the rule "score > 0". It selects another person, the bridge,
  or a rock on the other photos. The person score map itself marks the people well, including
  the unreferenced people in `DSC00527`, `DSC_1456`, `DSC00562`, and `P1024439` (sheets, by eye).
- **Cost:** the score of one feature source is a matrix product: 2.4 ms per photo for
  `image_embed` in NumPy. The three prototypes have 256 values each (3 KB).
- **Recommendation (step 1):** train a small head on the frozen `image_embed` of SAM 2.1 tiny,
  for sky and for people. Do not use the high-resolution features as the input. The head must
  learn what the prototypes cannot do: a calibrated "found" score, people against animals, mist
  and water against sky, and one output per person (center and box) instead of the densest body
  part. Section 7 has the details and the decision for O4.

## 3. Method

### 3.1 Inputs

| Item | Value |
| --- | --- |
| Photos | The 14 photos of the quality set at the 4096 px application input (E4 input set, `build/tmp/explore/e4/input_4096/photos`). Not committed. |
| Model | SAM 2.1 Hiera tiny, float32 ONNX of E1/E4: `point_encoder.onnx` (SHA-256 `17a554b420d3c51a109567f784ff3792b354c78dc8c2c09c585403329b412d61`) and `point_decoder.onnx` (SHA-256 `6f9a0ef2c217a0abccc8b0c01e631c42c6a78a2038a7944cc7fcb4f536439278`). Checkpoint `sam2.1_hiera_tiny.pt` (SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`), code `facebookresearch/sam2` at `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Apache-2.0. |
| References | The E1 reference masks of SAM 2.1 base plus (PyTorch) at the 4096 px input. People: the single-mask output of the E1 box prompt. Sky: the multimask output of the E1 sky point with the highest predicted IoU (`compare.json`). For `2088064441` the board mask of the board point is removed from the boarder box mask (the box mask includes the board and a bag). |
| Class presence | [`class_presence.json`](../../scripts/poc/sam_feature_localization/class_presence.json): presence per photo and class, classified by eye, and ignore boxes for people and sky without a reference (section 3.2). |
| Scripts | [`scripts/poc/sam_feature_localization/`](../../scripts/poc/sam_feature_localization/README.md) |
| Results | [`results/p0_sam_feature_localization/`](../../results/p0_sam_feature_localization/): `labels.json`, `metrics.json`, `summary.md`, `end_to_end.json` (committed); `labels.npz`, `score_maps.npz` (local only) |
| Training | None. No new model download. |

### 3.2 Reference cells (step 2)

The cell grid is the grid of `image_embed`: the photo is stretched to 1024 x 1024, so one cell is
1/64 of the photo width by 1/64 of the photo height (64 x 43 px on a 4096 x 2731 photo).

- A **class cell** has at least 90 percent of its pixels in the class reference.
- An **other cell** has at most 10 percent of its pixels in a person or sky reference, and no
  pixel in an ignore box.
- The remaining cells (mixed cells, unreferenced people and sky) are excluded from the
  prototypes.

The issue asks for a label grid of three values. The photos have more people and sky than the
references, so the evaluation needs more than that grid. The E1 references cover one person or
one sky per photo, but 10 photos have people or sky without a reference (pedestrians, a second
musician, overcast sky behind buildings). Without the ignore boxes, these cells would count as
"other" in the prototypes and as false results in the AUC. The measurements of one class
therefore use two sets per class:

- **positive:** the class cells;
- **negative:** cells with at most 10 percent of the class, outside the ignore boxes of that
  class. The cells of the other class are negatives.

The close-up `DSC00674` has a blurred sunset color gradient behind the person. It is not clear
whether this is sky, so its sky status is "uncertain". The background still counts as "not a
person".

| Photo | Person | Person cells (center inside) | Sky | Sky cells (center inside) |
| --- | --- | ---: | --- | ---: |
| `2088064441` | referenced (boarder, small); reflection ignored | 0 (5) | unreferenced | - |
| `5761824892` | absent | - | unreferenced (gaps in the foliage) | - |
| `DSC00326` | referenced | 77 (109) | unreferenced (small gap) | - |
| `DSC00527` | referenced (kneeling woman); many other people ignored | 25 (37) | referenced | 1,578 (1,674) |
| `DSC00562` | unreferenced (very small people) | - | unreferenced | - |
| `DSC00602` | unreferenced (passengers, one person at the edge) | - | unreferenced | - |
| `DSC00674` | referenced (close-up) | 2,666 (2,773) | uncertain | - |
| `DSC00728` | absent | - | referenced | 1,351 (1,435) |
| `DSC00936` | referenced | 71 (89) | absent (mist) | - |
| `DSC_1188` | absent (squirrel) | - | absent | - |
| `DSC_1456` | unreferenced (pedestrians, driver) | - | unreferenced (small) | - |
| `DSC_1535` | referenced; pedestrians ignored | 817 (898) | absent | - |
| `Nikon-D800e-raw-00007` | absent | - | referenced | 967 (1,049) |
| `P1024439` | referenced (bassist); second musician ignored | 127 (177) | unreferenced | - |

The contact sheet `sheets/label_grids.jpg` shows the grids over the photos. A check by eye shows:
the class cells cover the referenced person or sky, and the excluded cells are the mixed edge
cells and the ignore boxes.

### 3.3 Features (step 3)

One encoder run per photo, ONNX Runtime CPU, 4 intra-op threads, CPU memory arena off. The
preprocessing is the E1 preprocessing (stretch to 1024 x 1024, bilinear with antialias, ImageNet
normalization). Four feature sources on the 64 x 64 grid, each cell vector normalized to unit
length:

| Source | Channels | Content |
| --- | ---: | --- |
| `image_embed` | 256 | The encoder output that the decoder reads (stride 16, with `no_mem_embed`) |
| `high_res_feats_1` | 64 | Stride 8 features, mean of each 2 x 2 block |
| `high_res_feats_0` | 32 | Stride 4 features, mean of each 4 x 4 block |
| `concatenated` | 352 | The three normalized sources, concatenated and normalized again |

### 3.4 Prototypes and measurements (steps 4 and 5)

For each photo, the person, sky, and other prototypes are the means of the labeled cells of the
other 13 photos (a mean over cells). For example, the person prototype for `DSC_1535` uses
2,966 cells, sky 3,896, and other 33,008. The score of a cell is
`cosine(cell, class prototype) - cosine(cell, other prototype)`.

- **AUC:** score map against the positive and negative cells of the class, on the photos where the
  class is referenced.
- **Peak precision:** the share of the 3 and the 10 highest-scoring cells of the whole photo that
  are class cells ("label") or whose center pixel is in the reference ("center").
- **Separation:** margin = the lowest third-highest score on the referenced photos minus the
  highest score on the photos without the class. A positive margin gives one threshold (the
  midpoint) that finds the class by its 3 best cells on every referenced photo and on no photo
  without it. Photos with unreferenced or uncertain presence are reported against the
  threshold, but they do not set it.

**Acceptance criteria** (proposal of the issue; the user confirms the thresholds):

- **peak:** precision at 3 is 1 on every referenced photo. The "label" rule applies. A photo with
  fewer than 3 class cells (only `2088064441`) uses the "center" rule, because the label rule
  cannot be met there.
- **separation:** margin > 0.
- The features are usable for a class when both pass.

### 3.5 End to end (step 6)

On each photo where the class is referenced, for each feature source:

- `top3_points`: the centers of the 3 top cells as positive points.
- People only, `region_box`: the bounding box of the largest 8-connected region above a
  threshold. Because the separation failed, no common threshold exists. This report gives two
  rules and does not select one per photo: `above_zero` (score > 0) and `half_max` (score >= 0.5 x the photo
  maximum). Each box also runs with the 3 points.
- `e1_prompt`: the E1 prompt of the reference (box or sky point), decoded with tiny, for
  comparison.

The mask selection is the E1 rule: the best multimask output for one point, the single-mask
selection for a box or several points. IoU at the photo size against the reference (logits
bilinear, threshold 0). The IoU is against one reference: a prompt that selects another person
gets an IoU near 0, although it found a person.

### 3.6 Machine

Windows 11 build 22635, Intel Core i7-12700H (20 logical CPUs), 31.8 GB. Python 3.12.4,
onnxruntime 1.31.0 (CPU), NumPy 2.5.2. No GPU run.

## 4. Results: acceptance criteria

| Feature source | Class | Precision at 3 = 1 | Separation margin | Usable |
| --- | --- | --- | ---: | --- |
| `image_embed` | person | 5 / 7 (fails: `P1024439`, `2088064441`) | -0.177 | no |
| `image_embed` | sky | 3 / 3 | **0.026** | **yes** |
| `high_res_feats_1` | person | 1 / 7 | -0.061 | no |
| `high_res_feats_1` | sky | 2 / 3 | -0.046 | no |
| `high_res_feats_0` | person | 3 / 7 | -0.248 | no |
| `high_res_feats_0` | sky | 1 / 3 | -0.075 | no |
| `concatenated` | person | 5 / 7 | -0.100 | no |
| `concatenated` | sky | 3 / 3 | -0.029 | no |

| Feature source | Person AUC, median (minimum) | Sky AUC, median (minimum) |
| --- | --- | --- |
| `image_embed` | 0.999 (0.994) | 1.000 (1.000) |
| `high_res_feats_1` | 0.516 (0.062) | 0.890 (0.871) |
| `high_res_feats_0` | 0.784 (0.197) | 0.998 (0.971) |
| `concatenated` | 0.944 (0.743) | 1.000 (0.989) |

The high-resolution sources add no class information: concatenating them to `image_embed`
lowers the AUC and the sky margin. The rest of this report shows `image_embed`. The tables of all
sources are in [`summary.md`](../../results/p0_sam_feature_localization/summary.md).

## 5. Results: people

### 5.1 Referenced photos (`image_embed`)

| Photo | Person cells | AUC | Precision at 3 (label / center) | Precision at 10 (label / center) | Highest / third-highest score | Where the 3 top cells lie (sheet) |
| --- | ---: | ---: | --- | --- | --- | --- |
| `DSC_1535` | 817 | 1.000 | 1.00 / 1.00 | 1.0 / 1.0 | 0.359 / 0.341 | Face |
| `DSC00326` | 77 | 0.999 | 1.00 / 1.00 | 1.0 / 1.0 | 0.271 / 0.262 | Hair |
| `DSC00527` | 25 | 0.999 | 1.00 / 1.00 | 1.0 / 1.0 | 0.305 / 0.246 | Kneeling woman |
| `DSC00674` | 2,666 | 0.998 | 1.00 / 1.00 | 1.0 / 1.0 | 0.243 / 0.237 | Face and ear |
| `DSC00936` | 71 | 1.000 | 1.00 / 1.00 | 1.0 / 1.0 | 0.256 / 0.245 | Torso |
| `P1024439` | 127 | 0.994 | 0.00 / 0.00 | 0.0 / 0.0 | 0.332 / 0.327 | Arm of the second musician (ignored, unreferenced) |
| `2088064441` | 0 | - | - / 0.67 | - / 0.2 | 0.129 / 0.116 | 2 on the boarder, 1 on a wooden post |

### 5.2 Separation (`image_embed`)

| Photo | Status | Highest | Third-highest |
| --- | --- | ---: | ---: |
| `DSC_1188` (squirrel) | absent | **0.293** | 0.273 |
| `5761824892` (car) | absent | 0.243 | 0.212 |
| `DSC00728` (balloon) | absent | 0.164 | 0.120 |
| `Nikon-D800e-raw-00007` | absent | 0.078 | 0.068 |
| `DSC00602` | unreferenced (passengers) | 0.229 | 0.222 |
| `DSC_1456` | unreferenced (pedestrians) | 0.213 | 0.187 |
| `DSC00562` | unreferenced (very small people) | 0.212 | 0.165 |

The lowest third-highest score on a referenced photo is 0.116 (`2088064441`), and the next are
0.237 (`DSC00674`) and 0.245 (`DSC00936`). The squirrel (0.293) is above all of them except
`DSC_1535` and `P1024439`. Margin: -0.177; without the small boarder: -0.056. No threshold exists.

On the photos with unreferenced people, the 3 top cells lie on people on `DSC_1456`
(pedestrians) and `DSC00562` (the small people in the park), and on the tram on `DSC00602`.

### 5.3 Observations from the sheets

- The person score map of `image_embed` marks whole people, not only faces: the man of
  `DSC_1535` from head to shoes and the pedestrian behind him, all visitors in the park of
  `DSC00527`, both musicians of `P1024439`.
- It also marks the squirrel of `DSC_1188` completely. A prototype cannot separate "person" from
  "animal" in this feature space. A trained head has examples of both.
- On the close-up `DSC00674` the whole frame has a high score, also the background. A whole-frame
  person makes the "other" side of the score weak.
- `high_res_feats_0` and `high_res_feats_1` mark edges, water, and asphalt texture more than the
  people.

## 6. Results: sky

### 6.1 Per photo (`image_embed`)

| Photo | Status | AUC | Precision at 3 | Highest / third-highest | Found at 0.243 |
| --- | --- | ---: | --- | --- | --- |
| `DSC00527` | referenced | 1.000 | 1.00 | 0.286 / 0.279 | yes |
| `DSC00728` | referenced | 1.000 | 1.00 | 0.264 / 0.256 | yes |
| `Nikon-D800e-raw-00007` | referenced | 1.000 | 1.00 | 0.280 / 0.278 | yes |
| `DSC00936` (mist) | absent | - | - | **0.230** / 0.226 | no |
| `DSC_1188` | absent | - | - | 0.144 / 0.136 | no |
| `DSC_1535` | absent | - | - | 0.124 / 0.087 | no |
| `DSC_1456` | unreferenced | - | - | 0.312 / 0.310 | yes |
| `P1024439` | unreferenced | - | - | 0.299 / 0.297 | yes |
| `DSC00326` | unreferenced (small gap) | - | - | 0.277 / 0.275 | yes |
| `DSC00562` | unreferenced | - | - | 0.276 / 0.273 | yes |
| `2088064441` | unreferenced | - | - | 0.273 / 0.264 | yes |
| `DSC00602` | unreferenced | - | - | 0.265 / 0.258 | yes |
| `5761824892` | unreferenced (small gaps in the foliage) | - | - | 0.139 / 0.126 | no |
| `DSC00674` | uncertain (sunset backdrop) | - | - | 0.190 / 0.186 | no |

On all 6 unreferenced photos that are found, the 3 top cells lie in the sky ignore box (the
sky). The margin of 0.026 comes from 3 sky photos and 3 photos without sky. It is a decision
signal, not a threshold for the product.

### 6.2 Observations from the sheets

- The sky score of `image_embed` follows the horizon of trees and buildings at the cell
  resolution (`DSC00527`, `DSC00602`).
- The waterfall of `DSC00326` and the mist of `DSC00936` get a medium sky score. These bright,
  textureless areas are the hard negatives for sky.

## 7. Results: end to end (step 6)

IoU against the reference. The E1 column is the E1 prompt decoded with tiny. For `2088064441`,
the E1 box selects the boarder, the board, and the bag, but the reference has no board, so its
IoU is low (0.269).

### 7.1 People (`image_embed`)

| Photo | E1 box | 3 points | Box `above_zero` | Box `above_zero` + points | Box `half_max` | Box `half_max` + points | Selection by eye (sheet) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `DSC_1535` | 0.993 | 0.043 | **0.993** | 0.991 | 0.117 | 0.112 | Points: head. `above_zero`: whole person. `half_max`: head and torso |
| `DSC00674` | 0.988 | 0.283 | **0.992** | 0.989 | 0.060 | 0.129 | Points: face. `above_zero`: whole person |
| `DSC00936` | 0.989 | 0.636 | **0.989** | 0.983 | **0.989** | 0.964 | Points: part of the body |
| `DSC00326` | 0.973 | 0.164 | 0.261 | 0.399 | **0.896** | 0.915 | Points: hair. `above_zero`: person and the rock below her |
| `DSC00527` | 0.957 | 0.234 | 0.000 | 0.004 | 0.680 | 0.774 | `above_zero`: a different person (woman with a child) |
| `P1024439` | 0.934 | 0.000 | 0.000 | 0.000 | 0.195 | 0.112 | Points and `above_zero`: the second musician and his guitar |
| `2088064441` | 0.269 | 0.452 | 0.000 | 0.005 | 0.000 | 0.000 | Points: the person. Boxes: the bridge |

The other feature sources do not give better prompts in general. In four single cases their 3
points give a higher IoU than `image_embed`, because the points happen to be spread over the
body: `high_res_feats_1` on `DSC00326` (0.954) and `DSC_1535` (0.781), `concatenated` on
`DSC00936` (0.710) and `DSC00527` (0.525). All values are in `end_to_end.json`.

### 7.2 Sky (3 top cells as points)

| Photo | E1 sky point | `image_embed` | `high_res_feats_1` | `high_res_feats_0` | `concatenated` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DSC00527` | 0.997 | 0.995 | 0.997 | 0.992 | 0.994 |
| `DSC00728` | 0.997 | 0.992 | 0.995 | 0.996 | 0.996 |
| `Nikon-D800e-raw-00007` | 0.993 | 0.969 | 0.967 | 0.000 | 0.951 |

For sky, 3 points in the sky are enough: SAM 2.1 selects the whole sky. For people, 3 points in
one body part select that part (the single-point ambiguity of E1). People need a box per person,
or points spread over the body.

## 8. Cost

| Item | Value | Run |
| --- | --- | --- |
| Encoder, CPU, 4 threads, arena off, median of 14 photos | 1,375 ms | `extract_cell_features.py`, `build/tmp/explore/p0/features/features.json` |
| Peak working set, encoder run | 1,327 MB | same |
| Score of one source and photo (`image_embed`, 3 cosine maps, NumPy) | 2.4 ms | `score_prototypes.py`, `metrics.json` |
| Peak working set, scoring | 615 MB | same |
| Decoder, CPU, median of 162 prompts | 31.5 ms | `decode_top_cells.py`, `end_to_end.json` |
| Peak working set, encoder and decoder run | 1,353 MB | same |

The encoder time is from this Python run with the arena off. E1 measured 1,118 ms on CPU with
the production session settings, and 76 ms on DirectML. The encoder runs anyway for the stroke
and box tools, so People and Sky only add the head. GPU memory was not measured (no GPU run).

## 9. Recommendation

**Step 1: train a small head on the frozen `image_embed` of SAM 2.1 tiny, for sky and for
people.** Reasons:

- The class information is in `image_embed`. Without training, a mean vector per class already
  ranks the class cells first within a photo (AUC 0.994 to 1.000). A head of one to three
  convolution layers on 256 channels can learn more than one mean vector.
- The prototype failures are the failures that training corrects: a calibrated presence score
  that is comparable between photos (people separation), person against animal (`DSC_1188`), and
  mist and water against sky (`DSC00936`, `DSC00326`).
- The high-resolution features are not an input for the head (AUC 0.52 and 0.78 for people).

Requirements for the head, from the end-to-end results:

- **Sky:** a score map per cell and a presence score. Points on the confident sky regions are
  sufficient prompts (IoU 0.97 to 0.995 with 3 points). Each separate sky piece needs its own
  points (README section 5).
- **People:** one output per person, not a score map only. The top cells cluster on the face or
  the hair, and the largest region can be another person, a rock, or a bridge. A center heatmap
  with a box size per center (CenterNet type) gives one box per person. A box selects the whole
  person (IoU 0.99 where the box was correct). This also gives the People list of F4a.
- **Limits of the grid:** a person narrower than about one cell (16 model pixels; the boarder of
  `2088064441`, 64 px wide at the 4096 px input) cannot get a reliable center or box from a
  64 x 64 grid. U5 accepts missed people in crowds; a single small person in a landscape is a
  case for the user to decide.

**Open decision O4 (person localizer):** the proof of concept does not reject the SAM 2.1 feature
head. It shows that the features hold the information, but also that a prototype is not enough
for people. The decision between the trained head and YOLOX-Nano needs the trained head of step 1
and its result on these photos. Sky has no alternative candidate with a commercial license
(SegFormer is non-commercial, U2), and the sky result here is positive.

## 10. Not measured and open items

- **Training data and its license** for step 1 (README section 5, U2): COCO-Stuff or Open Images
  V7 annotations (CC BY 4.0) and the image licenses.
- **The set is small.** 7 photos with a referenced person, 3 with referenced sky, 4 without people,
  3 without sky. The margin of the sky (0.026) is a decision signal, not a product threshold.
- **The class presence is classified by eye** (`class_presence.json`), including the ignore boxes.
  The user can check it on the contact sheet.
- **One reference per photo.** A prompt that selects another person counts as IoU 0 (`DSC00527`,
  `P1024439`).
- **No other aggregation was tested:** photo-weighted prototypes, mean-centered features, or
  several prototypes per class. These are variants of a no-training method. A trained head
  replaces them.
- **No GPU run, no Core ML run.** The head adds a few milliseconds on CPU; its GPU cost is
  measured in step 1.
- **The user confirms** the acceptance thresholds of section 3.4 and reviews the sheets
  (`results/p0_sam_feature_localization/sheets/`).

## 11. Changed files

- `scripts/poc/sam_feature_localization/`: the scripts, `class_presence.json`, `README.md`.
- `results/p0_sam_feature_localization/`: the measurements (the sheets stay local).
- This report, [the index](README.md), and [README](../../README.md) sections 5 and 7 (O4).
