# Sky matte labels from COCO and closed-form matting

**Date:** 2026-10-25. **Code:** [`scripts/poc/sky_matte_labels/`](../../scripts/poc/sky_matte_labels/README.md).
**Results:** [`results/sky_matte_labels/label_stats.json`](../../results/sky_matte_labels/label_stats.json),
[`results/sky_matte_labels/selection_manifest.json`](../../results/sky_matte_labels/selection_manifest.json).
**Status:** labels of 200 photos made and checked by eye by the agent; user check open; head
training not started.

> **Photos.** This step does not use the quality photos. It uses 200 COCO 2017 train photos with
> a license that permits commercial use (CC BY 2.0, CC BY-SA 2.0, no known copyright
> restrictions). The manifest records the COCO id, the license, and the Flickr URL of each photo.
> The photos, labels, and check sheets stay local in `build/tmp/explore/sky_labels/`
> (section 8 gives the commands).

## 1. Question

The MVP Sky (P1 head and SAM 2.1 point prompts) does not select the full sky: the head map has
64 x 64 cells (64 px at a 4096 px photo), a sky region needs 3 cells to get points, and the SAM
2.1 decoder gives 256 x 256 logits with a threshold (16 px cells, no partial alpha). Sky between
branches and leaves is lost. README decision U17 selects an own sky head with a fine alpha output
on the SAM 2.1 features. Such a head needs fine sky labels, and no public dataset with fine sky
labels permits commercial use (ADE20K images, Mapillary Vistas, BDD100K, Cityscapes, and SA-1B do
not; COCO-Stuff permits it but its sky edges are coarse).

Can classical matting without learned weights turn the coarse COCO sky into fine sky labels
that are correct enough for training, and at which cost?

## 2. Result summary

- **Yes, with rules for the known COCO-Stuff errors.** On the final rules (run 6), the labels
  keep the sky between branches, twigs, wires, poles, hair, and kite lines, and they correct
  coarse COCO edges (heads, roofs, distant animals on the horizon). The best case is the target
  case: bare branches against an overcast sky (`539283`): COCO marks the whole crown as tree and
  covers branches with sky polygons; the label has crisp branches and 60 % of the tree pixels as
  sky.
- **The plain trimap from the COCO segments is not sufficient.** Six failure classes were found
  on the check sheets and fixed by rules (section 5): JPEG blocks in flat sky, whole bare crowns
  labeled as tree, sky polygons over branches and forest, a dark night sky, snow on branches, and
  large objects without a COCO class (a balloon). Each rule has a check photo.
- **Uncertain pixels are not labeled; they are ignored.** The training loss ignores a pixel where
  two matting settings disagree or where the alpha is a semi-transparent plateau. This is 1.3 %
  of all pixels, but a median of 14.6 % of the unknown (edge) area of a photo.
- **Random check by eye** (agent, run 3 to run 5 samples): 9 of 10 photos of a random sample
  were correct in run 3; the failures found in the samples (balloon, dim sign plateau, snow) were
  fixed in runs 4 to 6. The user has not checked the sheets.
- **Cost:** 7.4 s per photo on 4 CPU threads (1.5 s of matting); about 15 hours for all 7,164
  candidate photos (estimate, not measured).

## 3. Inputs

| Item | Value |
| --- | --- |
| Annotations | COCO 2017 panoptic, `panoptic_annotations_trainval2017.zip` (SHA-256 `c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370`), `panoptic_train2017.json` (SHA-256 `560a90a2...bfbbb`, the P1 file). CC BY 4.0 |
| Photo license rule | COCO license 4 (CC BY 2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions), 8 (United States Government Work): the P1 rule (U2). Selected: 120, 73, 7, 0 photos |
| Candidates | Coarse sky 5 to 85 % of the photo, at most 10 % unlabeled: 7,164 photos (tree 4,573, building 1,088, other 1,503) |
| Selection | 100 tree, 60 building, 40 other; each group shuffled with seed 0. 60 downloads rejected (44 removed from Flickr, 16 with a long edge below 800 px) |
| Photos | Flickr size `_b` of the COCO `flickr_url` (long edge 1,024 px; COCO has 640 px), same aspect ratio within 1.5 % |
| Matting | PyMatting 1.1.16 (MIT), `estimate_alpha_cf` (closed-form matting, Levin et al. 2008); no learned weights |
| Machine | Intel Core i7-12700H, CPU only, 4 threads (Numba, OpenMP); Python 3.12.4 |

## 4. Method

Per photo, at the 1,024 px photo size (`sky_label_data.py`, `make_sky_matte_labels.py`):

1. Coarse sky: `sky-other-merged`, plus each unlabeled component of at most 1 % of the photo
   whose outer ring is at least half sky (label gaps).
2. Sky color distance: the RGB distance to the 400th nearest of 4,096 samples of the flat coarse
   sky (5 x 5 luminance standard deviation < 0.02), after a 16 px erosion of the coarse sky.
3. Trimap:
   - unknown: a 16 px band on both sides of the coarse sky edge; unlabeled pixels within 48 px
     of the sky; each tree component that comes within 16 px of the sky; coarse sky pixels with
     a sky color distance > 0.25 and a 4 px ring around them.
   - in these tree components: distance > 0.25 is not sky, < 0.03 is sky (eroded by 1 px). No
     color rule when the sky is dark (median sample luminance < 0.2) or the photo has a `snow`
     segment.
4. Closed-form matting with epsilon 1e-7 and 1e-5; `alpha_label` is the mean.
5. `label_weight` = 0 in the unknown area where the two alphas differ by more than 0.15, and on
   semi-transparent plateaus (`alpha_label` in (0.15, 0.85) after a 3 px erosion).

Check: by eye on sheets with the photo, the coarse sky, the trimap, the label with the ignored
pixels in red, and a 256 x 256 zoom at the window with the most semi-transparent pixels
(`make_sky_label_sheets.py`, `index.html` per run). The photos have no reference labels, so the
statistics describe the labels, not their accuracy.

## 5. Rule development

Each run labeled all 200 photos; the trials labeled the photos of a failure. The rule was kept
when the failure photo was correct and the earlier correct photos stayed correct.

| Run | Rule change | Failure that the change fixes (check photo) |
| --- | --- | --- |
| Trial | Closed-form alone, KNN matting as a second estimate | KNN smears edges and is 7 s per photo; not a useful reference. Closed-form shows 8 x 8 JPEG blocks in flat grey sky (`200796`) |
| Trial | Denoised input (bilateral, NL-means) against two epsilons | Denoising removed thin branches from the alpha (`539283`); not used. Two epsilons: blocks only in 1e-7, transparent dark parts only in 1e-5; the difference marks unreliable pixels |
| 1 | Band + tree and unlabeled pixels within 48 px of the sky | The interior of a bare crown, labeled as tree, was fixed as not sky, and the matting spread it over the sky between the branches (`539283`) |
| 2 | Tree components next to the sky by sky color (nearest sample) | The coarse sky polygons cover branches, so branch colors were in the sky samples: all branches became sky (`539283`) |
| 3 | Flat samples, 400th neighbor, dark sky rule | 64th and 128th neighbors: a forest under the sky polygons became sky (`312153`); night sky: a dark tree became sky (`450860`) |
| 4 | Unlabeled component to sky only up to 1 % of the photo | A balloon without a COCO class became sky (`2148`) |
| 5 | Plateau rule for `label_weight` | A dim green sign in an overcast dusk sky got alpha 0.5 (`173132`) |
| 6 | No color rule with a `snow` segment | Snow on branches under an overcast sky became sky (`122199`) |

## 6. Results

Run 6 (final rules), median | 90th percentile | maximum over the photos of the group:

| Statistic | Tree (100) | Building (60) | Other (40) | All (200) |
| --- | --- | --- | --- | --- |
| Coarse sky share | 0.229 \| 0.522 \| 0.759 | 0.342 \| 0.590 \| 0.818 | 0.357 \| 0.729 \| 0.828 | 0.276 \| 0.590 \| 0.828 |
| Unknown share of the photo | 0.068 \| 0.133 \| 0.260 | 0.087 \| 0.146 \| 0.219 | 0.062 \| 0.136 \| 0.187 | 0.074 \| 0.138 \| 0.260 |
| IoU (label >= 0.5, coarse sky) | 0.953 \| 0.986 \| 0.994 | 0.970 \| 0.989 \| 0.995 | 0.982 \| 0.995 \| 0.998 | 0.967 \| 0.988 \| 0.998 |
| Semi-transparent share of the unknown area (0.05 to 0.95) | 0.429 \| 0.567 \| 0.781 | 0.331 \| 0.471 \| 0.658 | 0.324 \| 0.502 \| 0.755 | 0.374 \| 0.546 \| 0.781 |
| Ignored share of the unknown area | 0.180 \| 0.305 \| 0.545 | 0.102 \| 0.273 \| 0.415 | 0.101 \| 0.268 \| 0.546 | 0.146 \| 0.297 \| 0.546 |
| Tree pixels labeled sky | 0.006 \| 0.063 \| 0.604 | | | |
| Matting time, s | 1.15 \| 1.95 \| 4.68 | 1.57 \| 2.94 \| 6.47 | 1.73 \| 2.73 \| 3.13 | 1.37 \| 2.60 \| 6.47 |

- Ignored pixels: 1.30 % of all pixels of the 200 photos. 36 photos have more than 25 % of the
  unknown area ignored, 8 more than 40 %.
- 14 of 100 tree photos have more than 5 % of their tree pixels labeled as sky (sky through the
  crown that COCO marks as tree).
- 21 photos have a `snow` segment and get no color rule for the trees.
- Wall time: 1,484 s for 200 photos (7.4 s per photo, with the JSON load, the KD tree queries,
  and the PNG writes); matting 1.53 s on average.

Photos checked by eye on run 6, or on an earlier run with an `alpha_label` equal to run 6 (pixel
comparison, maximum difference 0): `539283` (bare branches,
correct), `454382` (bare tree against clouds, correct), `500568` (eucalyptus on the horizon, sky
between the trees found), `312153` (forest and signs, correct; the signs are ignored),
`450860` (night, correct), `2148` (balloon, correct), `173132` (signs ignored), `122199` (snow,
correct), `30519` (drilling rig and wires, correct), `386623` (black-and-white poles, correct),
`246848`, `197163`, `267521`, `387812`, `535483`, `407437`, `248221`, `64816`, `337342`,
`127450` (correct). `376185` (backlit forest, sky blown out to white, JPEG blocks) has patchy
labels between dense twigs; part of them is ignored. `407437` (snow, tree line) shows the cost of
the snow rule: the sky between the outer twigs of the crowns is not found; the edge is ignored
in blobs.

What was not measured: the label accuracy against hand-made fine labels (there are none), and
the effect of the labels on a trained head.

## 7. Cost

| Item | Value |
| --- | --- |
| Per photo | 7.4 s wall, 1.5 s matting (4 CPU threads) |
| All 7,164 candidates | About 15 hours (estimate from run 6, one process) |
| Disk per photo | Photo about 0.1 to 0.3 MB; five label PNGs |
| Panoptic annotations | 861 MB download; `panoptic_train2017.json` 165 MB, PNG archive 808 MB |

## 8. Commands

The commands are in the [script README](../../scripts/poc/sky_matte_labels/README.md#commands).
The check sheets of run 6 are in `build/tmp/explore/sky_labels/run6_sheets/index.html`.

## 9. Recommendation

1. **User check** of the run 6 sheets (`index.html`, by group). The most useful photos to check
   are the 36 photos with more than 25 % of the unknown area ignored and the 14 tree photos with
   sky in the crown.
2. **Label all 7,164 candidates** with the run 6 rules (about 15 hours on the development
   machine, or on Colab). The tree group is the most important: it holds the fine edges.
3. **Train the sky head with a fine output** on the frozen SAM 2.1 tiny encoder: `image_embed`
   (64 x 64), `high_res_feats_1` (128 x 128), `high_res_feats_0` (256 x 256), and a small RGB
   branch at 1024 x 1024; output a 1024 x 1024 sky alpha (4 px at a 4096 px photo, the S3OD
   output resolution). Loss on `alpha_label` with `label_weight`; the presence output and the
   People outputs of the P1 head stay. The encoder is shared, so no second image encoder runs.
4. **Make a small hand-labeled test set** of sky edges (trees, wires, buildings) on photos with a
   license that permits commercial use. Without it, the head can only be compared by eye.

## 10. Open items

- User check of the labels and of the U3 reading in U17 (offline matting for labels is not edge
  post-processing).
- The ignored share of the unknown area (median 14.6 %) is large. The head does not learn from
  these pixels; most are haze at the horizon, low-contrast objects, and patchy sky between dense
  twigs.
- Known weak cases: a backlit sky blown out to white with dense twigs (`376185`), unlabeled
  objects inside the sky with sky colors, and scenes where foliage and sky have similar colors
  without a `snow` segment. In photos with a `snow` segment, the sky inside the crowns is not
  found (`407437`); the 21 snow photos teach only the coarse crown edge.
- The labels have the resolution of the Flickr `_b` photos (1,024 px). Larger Flickr sizes need
  another URL secret and were not used.
- COCO photos are mostly everyday scenes; landscapes with large skies are fewer than in the
  photos of Alcedo Studio users.
- Flickr removes photos over time (44 of 260 tried downloads failed); a new selection can differ
  from the recorded manifest.
