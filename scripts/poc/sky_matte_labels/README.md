# Sky matte labels from COCO and closed-form matting

Report: [Sky matte labels](../../../docs/exploration/sky_matte_labels.md).

The sky head gives a 64 x 64 sky map, and the SAM 2.1 decoder gives 256 x 256 sky logits. Both
are too coarse for the sky between branches and leaves. A sky head with a 1024 x 1024 alpha output
needs fine sky labels, and no public dataset with fine sky labels permits commercial use. This
step makes these labels offline from photos with a license that permits commercial use (README
decision U17):

```text
COCO 2017 panoptic train photo, license 4, 5, 7, or 8 (the P1 rule, decision U2)
  -> Flickr size "_b" of the same photo (long edge 1024 px)
  -> coarse sky = `sky-other-merged` + unlabeled components of at most 1 percent of the photo
     that the sky encloses (label gaps)
  -> trimap (sizes at 1024 px):
       unknown: 16 px band on both sides of the coarse edge; unlabeled pixels within 48 px of
                the sky; the tree segments next to the sky; coarse sky pixels with a color that
                is not a sky color, and a 4 px ring around them
       sky color distance = RGB distance to the 400th nearest of 4,096 samples of the flat
       coarse sky pixels (5 x 5 luminance standard deviation < 0.02)
       in the tree segments next to the sky: distance > 0.25 not sky, < 0.03 sky; no color
       rule when the sky is dark (median sample luminance < 0.2) or the photo has a COCO
       `snow` segment
  -> closed-form matting (PyMatting) with epsilon 1e-7 and 1e-5
  -> alpha_label = mean of the two
  -> label_weight = 0 (ignored by the loss) where the two differ by more than 0.15, and on
     semi-transparent plateaus (alpha_label in (0.15, 0.85) after a 3 px erosion)
```

Why these rules (from the check sheets of the first runs, report section 5):

- COCO-Stuff labels a bare or sparse tree, with the sky between the branches, as one tree
  segment, and its sky polygons also cover branches. A trimap from the segments alone marks
  this sky as "not sky" and these branches as "sky". The sky color distance gives the matting
  the cores of the branches as "not sky" and the clear gaps as "sky".
- The coarse sky polygons also cover dark forest, signs, and branches, so the nearest sky
  sample can have their colors. The samples are only flat pixels (the sky is smooth, foliage is
  not), and a color is a sky color only when 400 of 4,096 samples (about 10 percent) are near it.
  With the nearest sample, the 64th, or the 128th, branches or a forest became sky.
- A large unlabeled component inside the sky is an object without a COCO class (a balloon), not
  a label gap, so only small components become sky.
- A night sky has the colors of dark foliage; with the color rule, a dark tree became sky. A dark
  sky gets no color rule for the trees. The same holds for snow on branches under an overcast
  sky (21 of the 200 photos have a `snow` segment).
- Epsilon 1e-7 alone shows the 8 x 8 JPEG blocks of the Flickr photos in flat grey sky; 1e-5
  alone makes dark opaque parts semi-transparent. Where they differ, the label is not reliable
  and the loss ignores the pixel. A denoised input (bilateral filter, NL-means) removed thin
  branches from the alpha and was not used.
- Mixed pixels of a real edge are a few pixels wide. A wider semi-transparent area comes from
  similar colors (a dim green sign in an overcast dusk sky) and is not a reliable label.

No learned model makes the labels, so the labels have no training data lineage of another model
(proposal U13 of issue #6). The matting runs only offline; the product has no edge
post-processing (U3).

| File | Purpose |
| --- | --- |
| `sky_label_data.py` | Shared layout: license ids, panoptic decoding, enclosed unlabeled areas, trimap, file names. |
| `select_sky_photos.py` | Photo selection in three groups (tree, building, other) and the download of the 1024 px Flickr photos with checks. |
| `make_sky_matte_labels.py` | Trimap, the two closed-form alphas, `alpha_label`, `label_weight`, and `label_stats.json`. |
| `make_sky_label_sheets.py` | Check sheets (local only). |
| `requirements.txt` | The environment of this step. |

## Licenses

| Item | License | Use |
| --- | --- | --- |
| COCO 2017 panoptic annotations (`panoptic_annotations_trainval2017.zip`, SHA-256 `c05f76d2...a370`; `panoptic_train2017.json` SHA-256 `560a90a2...bfbbb`, the same file as P1) | CC BY 4.0 | Coarse sky, tree, and unlabeled areas |
| Flickr photos of COCO 2017 train | Per photo, as `panoptic_train2017.json` names it | Only licenses 4 (CC BY 2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions), 8 (United States Government Work). The manifest records the license and the Flickr URL of each photo |
| PyMatting 1.1.16 | MIT | Closed-form matting (no learned weights) |

## Commands

```text
E=build/tmp/explore
S=$E/sky_labels
py -3.12 -m venv $E/venv_sky_labels
$E/venv_sky_labels/Scripts/python -m pip install -r scripts/poc/sky_matte_labels/requirements.txt
mkdir -p $S/download $S/coco_panoptic
curl -L -o $S/download/panoptic_annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip -j $S/download/panoptic_annotations_trainval2017.zip \
  annotations/panoptic_train2017.json annotations/panoptic_train2017.zip -d $S/coco_panoptic
A="--manifest $S/selection/manifest.json --photo-dir $S/selection/photos \
   --panoptic-zip $S/coco_panoptic/panoptic_train2017.zip \
   --panoptic-json $S/coco_panoptic/panoptic_train2017.json"
$E/venv_sky_labels/Scripts/python -I scripts/poc/sky_matte_labels/select_sky_photos.py \
  --panoptic-json $S/coco_panoptic/panoptic_train2017.json --out-dir $S/selection
$E/venv_sky_labels/Scripts/python -I scripts/poc/sky_matte_labels/make_sky_matte_labels.py \
  $A --out-dir $S/run6
$E/venv_sky_labels/Scripts/python -I scripts/poc/sky_matte_labels/make_sky_label_sheets.py \
  $A --label-dir $S/run6 --sheet-dir $S/run6_sheets
```

Flickr removes photos over time, so a new selection run can give other photos than the recorded
manifest. The download checks are in the docstring of `select_sky_photos.py`.
