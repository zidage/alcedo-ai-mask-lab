# Proof of concept: S3OD on the crop of a box (Sky and objects)

**Date:** 2026-10-24. **Code:** [`scripts/poc/box_s3od/`](../../scripts/poc/box_s3od/README.md)
(`check_box_s3od.py`, `boxes.json`).
**Results:** [`results/box_s3od_poc/summary.json`](../../results/box_s3od_poc/summary.json).
**Status:** measured; user check open.

> **Photos.** This check does not use the quality photos. It uses 17 photos from Wikimedia
> Commons (CC0, public domain, CC BY, CC BY-SA). `boxes.json` records the source page, the
> license, the author, and the SHA-256 of each downloaded file. The check pages stay local in
> `build/tmp/box_s3od_poc/check/` (section 6 gives the commands).

## 1. Question

The one-click Sky of the MVP (P1 head and SAM 2.1 point prompts) gives coarse sky edges, as the
SAM examples on the [ZIM page](https://naver-ai.github.io/ZIM/) show. SAM 2.1 selects parts well,
but not a whole subject at photographic quality. The user proposed:

1. Sky: a head gives a Sky box. S3OD runs on the crop of the box, and the Sky mask is the
   background of the crop (1 - the S3OD alpha).
2. Objects: a box around a tree (or another object) and S3OD on the crop of the box.

Does S3OD on a hand-drawn box give a usable Sky mask and usable object masks, compared with one
SAM 2.1 box prompt on the same box?

## 2. Result summary

- **Objects: yes.** On 16 of 18 object boxes, S3OD on the crop selects the object in the box with
  much better edges than the SAM 2.1 box prompt. On trees, S3OD keeps the sky between the branches
  and the fine twigs. SAM 2.1 fills the crown as one area. On cars, dogs, the windmill, the
  lighthouse, the tower, and the church tower, both models select the object, and the S3OD edges
  are finer (antenna, hair, sails). One box is ambiguous (a tree row, `c070`: the selected output has
  only a part of the row). One box fails: a group of trees at a forest edge (`c072` tree);
  S3OD selects almost nothing (area 0.6 % of the photo against SAM 24 %).
- **Sky as 1 - S3OD alpha: no.** The result is correct only when all non-sky content in the box is
  one salient silhouette (a single tree, the windmill). S3OD selects the salient object, so all
  other non-sky content in the Sky box becomes sky: hills (`c005`, `c011`), mountains (`c038`,
  `c059`), dunes (`c066`), roofs and the city skyline (`c084`, `c099`), tree lines and the forest
  (`c070`, `c072`, `c077`), and the second tree (`c056`). In `c072` the full forest edge is sky.
  None of the three S3OD outputs selects the sky in any crop. A Sky box from a head cannot fix this,
  because the Sky box must reach the horizon and so contains the non-sky content.

## 3. Inputs

| Item | Value |
| --- | --- |
| Photos | 17 Wikimedia Commons photos, thumbnails with a long edge of 3072 px (`boxes.json`). 15 have a Sky box; 18 object boxes (10 trees, 2 cars, 2 dogs, windmill, lighthouse, tower, church tower) |
| Boxes | Drawn by hand on a 0.1 grid, normalized `x0, y0, x1, y1`. Sky box: full width, from the top to the lowest visible sky. Object box: tight around the object |
| S3OD | `s3od/onnx_mha_chunk8_final/float16/subject_masks.onnx`, 219,301,899 bytes, SHA-256 `0f070b8641676d68faaafc0b70816be1458fcc976d8d6d119e67038527cbc288` (the accepted graph, README U14) |
| SAM 2.1 tiny and head | `mask_pipeline.GraphPaths.default` (E4 float32 encoder and decoder, P1 run 2 head) |
| Runtime | onnxruntime-directml 1.24.4, DirectML, RTX 3080 Laptop GPU 8 GB, CPU threads 4 |

## 4. Method

- Crop rectangle: the box widened by a margin of the box size on each side, clipped to the photo
  (`person_crop_rect` rule). Objects: margin 0.1 (the People value E3). Sky: margin 0, because a
  margin below the Sky box adds ground to the crop.
- Crop image and S3OD: `person_crop_image` and the S3OD letterbox to 1024; the three output masks
  and IoU scores; the selected mask has the largest IoU score.
- Object mask: the selected alpha in the crop rectangle. Sky mask: 1 - the selected alpha in the
  crop rectangle. Both are 0 outside the rectangle.
- References: one SAM 2.1 box prompt with the same box (E1 single mask, threshold 0); for Sky also
  the MVP one-click Sky and the photo Background (1 - S3OD alpha of the photo).
- Check: by eye on the check pages (the photos have no reference masks). The summary records the
  area of each mask (share of the photo pixels >= 0.5).

## 5. Results

S3OD on a crop: 189 to 391 ms (the cause of the longer runs was not examined; SM3.D measured 195
to 211 ms per crop). The memory peak was not measured.

### 5.1 Objects

| Photo | Box | S3OD IoU scores (selected) | Area S3OD | Area SAM | S3OD on the crop |
| --- | --- | --- | ---: | ---: | --- |
| c005 | tree | 0.931 (0), 0.922, 0.921 | 0.157 | 0.171 | Tree with open crown; output 0 also has a part of a small tree at the left |
| c011 | bare tree | 0.931 (0), 0.924, 0.927 | 0.115 | 0.209 | All twigs, sky open; SAM fills the crown |
| c013 | backlit tree | 0.846, 0.872 (1), 0.836 | 0.107 | 0.138 | Silhouette with open crown; sun area correct |
| c017 | frosted tree in fog | 0.856, 0.859, 0.860 (2) | 0.095 | 0.179 | Fine branches; SAM fills the crown |
| c018 | bare tree | 0.896 (0), 0.890, 0.881 | 0.145 | 0.211 | Fine branches; SAM fills the crown and adds a block at the right |
| c038 | rowan | 0.928 (0), 0.920, 0.921 | 0.202 | 0.226 | Correct, gaps in the crown open |
| c056 | windmill | 0.951, 0.960 (1), 0.949 | 0.166 | 0.175 | Correct, with the thin sails |
| c056 | spruce | 0.875, 0.914 (1), 0.873 | 0.008 | 0.013 | Upper spruce correct; the lower part behind the hedge is not selected |
| c066 | lighthouse | 0.969, 0.972 (1), 0.959 | 0.039 | 0.041 | Correct with the lamp and the antenna |
| c068 | car | 0.990, 0.993 (1), 0.988 | 0.317 | 0.314 | Correct |
| c070 | tree row | 0.624, 0.744, 0.756 (2) | 0.099 | 0.119 | Ambiguous: output 0 has all trees of the row, output 2 (selected) the tall tree and a part of the row; SAM has the tall tree only |
| c072 | car | 0.987, 0.990 (1), 0.985 | 0.087 | 0.088 | Correct |
| c072 | trees at a forest edge | 0.914, 0.929 (1), 0.905 | 0.006 | 0.244 | **Fails**: no tree selected (only the car at the crop edge) |
| c077 | pine | 0.902, 0.916 (1), 0.898 | 0.159 | 0.181 | Correct, gaps open |
| c084 | tower | 0.757, 0.946 (1), 0.938 | 0.049 | 0.050 | Correct |
| c099 | church tower | 0.954, 0.957 (1), 0.928 | 0.016 | 0.017 | Tower and the attached church; SAM similar |
| c123 | dog | 0.989 (0), 0.984, 0.982 | 0.180 | 0.178 | Correct, fur edge |
| c131 | long-haired dog | 0.987, 0.988 (1), 0.986 | 0.099 | 0.099 | Correct, hair edge finer than SAM |

### 5.2 Sky

Area of the box: the Sky box covers the full width down to the lowest visible sky, so "1 - S3OD"
cannot be larger than the box area minus the selected object.

| Photo | Area 1 - S3OD | Area SAM box | Area MVP Sky | Non-sky content that becomes sky |
| --- | ---: | ---: | ---: | --- |
| c005 | 0.628 | 0.437 | 0.384 | Rock walls, small trees |
| c011 | 0.716 | 0.528 | 0.752 | Wooded hill with the castle |
| c013 | 0.513 | 0.448 | 0.448 | Bushes and cacti at the horizon |
| c017 | 0.681 | 0.592 | 0.760 | None visible (fog; correct) |
| c018 | 0.481 | 0.361 | 0.351 | Hills and the village at the horizon |
| c038 | 0.434 | 0.350 | 0.346 | Mountains |
| c056 | 0.652 | 0.587 | 0.590 | The spruce at the left |
| c059 | 0.450 | 0.365 | 0.360 | Mountain range |
| c066 | 0.580 | 0.517 | 0.103 | Dunes |
| c068 | 0.401 | 0.340 | 0.338 | Tree line at the horizon |
| c070 | 0.493 | 0.428 | 0.410 | Tree line at the horizon |
| c072 | 0.640 | 0.563 | 0.304 | The full forest edge |
| c077 | 0.469 | 0.384 | 0.386 | Tree line |
| c084 | 0.470 | 0.426 | 0.035 | The city skyline and the hills |
| c099 | 0.584 | 0.480 | 0.475 | Roofs and houses |

Edges where the box content is one silhouette (`c011` twigs, `c017`, `c018`, `c077` crown) are
better than the SAM box and the MVP Sky. The MVP Sky also fails on two photos: `c084` (3.5 % of
the photo, the sky is about 45 %) and `c066` (one cloud only).

## 6. Commands

```text
E=build/tmp/explore
W=build/tmp/box_s3od_poc
# photo candidates, download (the photos and sources.json; the files in boxes.json)
$E/venv_ai_mask_mvp/Scripts/python -I $W/tools/commons_candidates.py $W/candidates \
  Featured_pictures_of_trees Featured_pictures_of_clouds Featured_pictures_of_cityscapes \
  Featured_pictures_of_dogs
$E/venv_ai_mask_mvp/Scripts/python -I $W/tools/commons_download.py $W/candidates/candidates.json \
  $W/photos 5,11,13,17,18,38,70,77,56,66,68,72,84,99,59,123,131
# check pages and summary.json
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/box_s3od/check_box_s3od.py
# pages: build/tmp/box_s3od_poc/check/index.html
```

The two download tools are local (`build/tmp/box_s3od_poc/tools/`); `boxes.json` records the
downloaded files and their SHA-256 values.

## 7. Recommendation

1. **Object box with S3OD.** Use S3OD on the crop of a box for whole objects (trees, buildings,
   vehicles, animals). It is the People rule (SM3.D) with a box from the user. It replaces the
   SAM 2.1 box prompt where the user wants the whole object; SAM 2.1 stays for parts (click,
   stroke). For groups (tree row, forest edge), show the three S3OD outputs as candidates: in
   `c070` output 0 is the full row.
2. **Do not use 1 - S3OD for Sky.** It is not a box problem: S3OD is trained for salient objects,
   and sky is a region class ("stuff"). The non-sky content at the horizon is usually not salient.
3. **Sky needs a model that outputs sky.** ZIM (the model of the reference page) is CC BY-NC 4.0
   and is not a candidate. The closest path with the accepted model set: a sky output trained on
   the S3OD encoder (DINOv3 ViT-B/16, frozen) with a DPT decoder at 1024 px, as the S3OD masks.
   The edge quality then comes from the same decoder type that gives the S3OD edges. The training
   data must permit commercial use (README U13 is open), and sky labels with fine edges are the
   main risk: COCO-Stuff sky labels are coarse.

## 8. Open items

- User check of the pages.
- The memory peak of S3OD on crops was not measured (the runs use the accepted graph, as SM3.D).
- The boxes are drawn by hand. A box from a head (the user proposal) was not trained.
- Sky training: data source and its license, label edge quality, the decoder size and cost.
