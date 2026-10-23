# People masks from S3OD on person crops (AlcedoStudio SM3.D)

**Issue:** [AlcedoStudio #469](https://github.com/zidage/AlcedoStudio/issues/469). **Date:**
2026-10-23. **Plan:** [SM3.D phase plan](https://github.com/zidage/AlcedoStudio/blob/main/docs/roadmap/alcedo_studio/edit/ai_segmentation_mask/phase_sm3d_people_from_s3od_plan.md).
**Code:** [`scripts/poc/ai_mask_mvp/`](../../scripts/poc/ai_mask_mvp/README.md)
(`mask_pipeline.py`, `ai_mask_app.py`, `check_people_from_s3od.py`).
**Results:** [`results/sm3d_people_from_s3od/summary.json`](../../results/sm3d_people_from_s3od/summary.json).
**Status:** measured; the person edges are accepted by the user (2026-10-23).

> **No images.** The check pages show the quality photos, so they stay in
> `build/tmp/explore/sm3d/` (section 5 gives the command).

## 1. Question

The SM3.C People task made one SAM 2.1 box prompt per listed person. The user asked for People
masks from S3OD, as for Subject (AlcedoStudio master plan D21):

1. The People result starts with the Subject alpha (S3OD on the photo). S3OD usually selects all
   people of a group in one run.
2. A click on a person box adds or removes that person. The person alpha is S3OD on the crop of
   the person box with a margin, from the session image (long edge up to 4096 px).
3. Each person box shows whether the person is already in the Subject alpha, without an S3OD run
   per person (decision E8: `subject_fraction` and `PERSON_IN_SUBJECT_FRACTION`).

Does S3OD on a person crop give acceptable person edges, and which value of
`PERSON_IN_SUBJECT_FRACTION` agrees with the person alphas?

## 2. Rules

- People list: `decode_people` with an overlap limit of 0.5; the boxes with a score >= the list
  threshold, else the best box when the presence passed (C2); left to right by the box center
  (C3). This is the SM3.C list; the lab adds the C2 rule.
- Person crop (E3): the clipped box, widened by 10 % of the box width on the left and on the
  right and by 10 % of the box height on the top and on the bottom, clipped to the photo; pixel
  edges `floor(left * width)` and `ceil(right * width)` (with a tolerance of 1e-4 px for float32
  box values), at least one pixel.
- Stored crop (E4): the crop pixels when the long edge is at most 1024, else the crop resized with
  the S3OD resize (OpenCV `INTER_LINEAR` to the letterbox size). The crops are made at the photo
  open.
- Person alpha: S3OD letterbox of the stored crop, the mask with the largest IoU score, unpadded:
  the alpha on the crop grid. A crop alpha without a value >= 0.5 is "not found" (E9).
- `subject_fraction` (E8): the share of the Subject grid pixels inside the clipped box (not
  widened) whose Subject alpha is >= 0.5.
- Lab rule of E8: a person is in the Subject when at least half of the pixels with a person alpha
  >= 0.5 have a Subject alpha >= 0.5 (both sampled bilinearly at the photo size with the crop
  rectangle, section 2.5 of the plan). The selected value is the value in steps of 0.05 that agrees
  with the lab rule for the most people; on equal counts the smaller value.
- Composition (`PeopleSteps`, the session rule of SM5.2): the Subject alpha, then each click in
  order: Add is the maximum with the person alpha, Subtract the minimum with 1 - person alpha,
  inside the crop rectangle only. A click on an included box subtracts the person; a click on
  another box adds it.

## 3. Results

DirectML, onnxruntime-directml 1.24.4, RTX 3080 Laptop; the AlcedoStudio SM3.C model directory
(`subject_masks.onnx` SHA-256 `0f070b86...`); the 14 quality photos at a long edge of 4096 px.
Five photos have no people (person presence below 0.533).

| Photo | People | Included (0.15) | Not found (E9) | Subject S3OD ms | Person S3OD ms | Crops ms | Crops MB | Largest crop MB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2088064441 | 1 | 1 | 0 | 275 (first run) | 210 | 24.4 | 0.19 | 0.19 |
| DSC00326 | 1 | 1 | 0 | 209 | 196 | 27.5 | 2.12 | 2.12 |
| DSC00527 | 20 | 3 | 0 | 195 | 195-197 | 19.1 | 2.91 | 0.67 |
| DSC00562 | 6 | 3 | 0 | 196 | 196-198 | 21.4 | 0.52 | 0.22 |
| DSC00674 | 1 | 1 | 0 | 197 | 196 | 21.5 | 2.10 | 2.10 |
| DSC00936 | 1 | 1 | 0 | 196 | 197 | 26.3 | 1.10 | 1.10 |
| DSC_1456 | 12 | 0 | 1 | 197 | 197-210 | 21.1 | 3.19 | 0.62 |
| DSC_1535 | 9 | 2 | 2 | 198 | 196-211 | 28.5 | 3.91 | 1.30 |
| P1024439 | 2 | 1 | 0 | 196 | 197-198 | 32.2 | 2.83 | 1.46 |

- A person crop S3OD run takes the time of a photo run (195 to 211 ms): the input is always
  1024 x 1024.
- The crops at the photo open take 19 to 32 ms of CPU time (PIL photo, NumPy slice, OpenCV resize)
  and at most 2.12 MB per person (3 MB limit of E4).

### 3.1 Value of `PERSON_IN_SUBJECT_FRACTION`

50 of the 53 listed people have a crop alpha (3 are "not found", E9).

| Value | 0.05 | 0.10 | **0.15** | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 0.70 to 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| People that agree with the lab rule | 47 | 47 | **48** | 47 | 47 | 47 | 46 | 46 | 46 | 44 | 41 | 40 to 35 |

The selected value is **0.15** (48 of 50). The two people that do not agree are `DSC_1535`
persons 4 and 5 (section 3.2).

### 3.2 Findings

1. **Person edges:** on the subject people, the group, and the background people, the S3OD crop
   alpha follows the person edges (hair, hats, feet, bags) better than the SM3.C SAM 2.1 box mask,
   which often cuts the head or the feet at the box.
2. **Small blurred people behind a sharp subject (`DSC_1535` persons 4, 5, 6):** the crop holds a
   part of the sharp subject (an arm), and S3OD selects that part, not the blurred person. The lab
   rule then counts these people as "in the Subject" (shares 0.83 to 0.98). Person 6 has a
   `subject_fraction` of 0.186 because the arm of the subject is inside its box, so it is shown as
   included; a click on it subtracts the alpha of the arm from the Subject. Persons 4 and 5 have a
   `subject_fraction` of 0.03 and 0 and are shown as not included (correct by eye).
3. **Not found (E9):** `DSC_1456` person 6 and `DSC_1535` persons 7 and 8 (small blurred people):
   no crop alpha value reaches 0.5. The click gives the message and no step.
4. Overlapping neighbors are in the crops of the group photos (`DSC00527`, `DSC_1456`). S3OD
   selects the person at the center of the crop, and can also select the part of a neighbor that
   is inside the crop (`DSC00527` person 17: the left neighbor at the crop edge). There is no
   special rule: the host applies the Add and Subtract steps in click order (user decision of
   SM3.D section 1.1).

## 4. User check

Manual verification by the user on 2026-10-23 (the pages of section 5, with the findings of
section 3.2): the person edges are **accepted**; port the rule with a margin of 0.1 and
`PERSON_IN_SUBJECT_FRACTION` 0.15.

## 5. Commands

```text
E=build/tmp/explore
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/ai_mask_mvp/check_people_from_s3od.py \
  --model-dir <AlcedoStudio build/tmp/sm3c/models/alcedo-ai-mask-v1>
# pages: build/tmp/explore/sm3d/index.html; values: build/tmp/explore/sm3d/summary.json
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/ai_mask_mvp/ai_mask_app.py \
  --subject-onnx <model directory>/subject_masks.onnx <photo>
```
