# Proof of concept: Matting Anything (MAM) for Sky and object boxes

**Date:** 2026-10-24. **Code:** [`scripts/poc/mam/`](../../scripts/poc/mam/README.md)
(`mam_model.py`, `export_mvp_references.py`, `check_mam.py`).
**Results:** [`results/mam_poc/summary.json`](../../results/mam_poc/summary.json),
[`results/mam_poc/references.json`](../../results/mam_poc/references.json).
**Status:** measured; not recommended; user check open.

> **Photos.** This check does not use the quality photos. It uses the 17 Wikimedia Commons photos
> and the hand-drawn boxes of the [box S3OD check](box_s3od_poc.md)
> ([`boxes.json`](../../scripts/poc/box_s3od/boxes.json)). The check pages stay local in
> `build/tmp/mam_poc/check_run3/` (section 6 gives the commands).

## 1. Question

The [box S3OD check](box_s3od_poc.md) showed that S3OD cannot give Sky (S3OD selects salient
objects, and sky is a region class). ZIM is CC BY-NC 4.0. The user asked for a check of
[Matting Anything](https://github.com/SHI-Labs/Matting-Anything) (MAM): SAM with a box or point
prompt, then a Mask-to-Matte (M2M) module that changes the SAM mask into an alpha matte.

1. Does MAM give a Sky mask with fine edges (twigs, roofs, and towers against the sky)?
2. Are the MAM object masks on a box better than S3OD on the box crop?

License: the code and the Hugging Face weights are MIT. The training data of M2M is AIM,
Distinctions-646, AM-2K, Human-2K, and RefMatte foregrounds on COCO and BG-20K backgrounds
(`GETTING_STARTED.md`). The user decision U16 (README) applies: the user downloads the weights,
Alcedo Studio does not distribute them, and Alcedo Studio supplies only the inference. The terms
of the training datasets were not checked in this step.

## 2. Result summary

- **Sky: no.** None of the three output fusion modes of the MAM repository gives Sky:
  - Demo default (`alpha` guidance): SAM ViT-B selects the sky correctly (for example 24 % of
    `c018`), but M2M ignores the sky mask and mattes the salient object. Inside the sky mask, the
    M2M alpha is 0.002 on average (`c018`). On `c011`, the result is the tree, not the sky. Sky area
    0 % to 20 %; the MVP Sky has 30 % to 76 % on 13 of 15 photos.
  - Demo "mask" mode (`mask_one_side`): the result is the SAM ViT-B mask, with M2M only in a thin
    band outside it. IoU with the MVP one-click Sky (SAM 2.1, same points) is 0.95 to 0.99 on 12 of
    15 photos. The edges are as coarse as the MVP Sky, and false sky spots occur in forests (`c011`).
  - Complement (M2M mattes the non-sky part, Sky = 1 - alpha): the edges of a single tree against
    the sky are the best of all methods (`c011`, `c018`, `c077`: twigs and gaps in the crown are
    open). But M2M gives the ground, the hills, the forest, and the city a partial alpha, so these
    areas become partly sky (Sky area 65 % to 99 %). This is the S3OD failure of the box S3OD check
    again: the M2M module is trained for foreground objects.
- **Objects: not better than S3OD.**
  - Demo default (`alpha` guidance) on the box crop: trees are selected with the crown gaps open,
    and the forest edge `c072` (where S3OD fails) gets the trees. But at 1:1 the alpha is blurred:
    twigs and dog hair become soft halos, and S3OD on the same crop resolves them more sharply.
    Rigid objects fail: the tower `c084` keeps only its outline (area 0.2 % against 4.9 % for
    S3OD), the church tower `c099` only its spire, the windmill `c056` body is transparent, and the
    lighthouse `c066` and the two cars are partly transparent.
  - Demo "mask" mode: rigid objects are complete (IoU with S3OD 0.91 to 1.00), but the edges are
    the SAM ViT-B edges, and the tree crowns are filled (the sky between the branches is selected).
  - The two dogs are correct in all modes (IoU with S3OD 0.98 to 0.99). The MAM fur edge is soft
    but blurred.
- **Cost:** MAM adds a third image encoder (SAM ViT-B; 96 M parameters with M2M, a checkpoint of 408 MB)
  beside SAM 2.1 tiny and S3OD. The cost on DirectML was not measured, because the quality result
  is negative.

## 3. Inputs

| Item | Value |
| --- | --- |
| Photos and boxes | 17 Wikimedia Commons photos (long edge 3072 px), 15 Sky boxes, 18 object boxes (`boxes.json`) |
| MAM source | [SHI-Labs/Matting-Anything](https://github.com/SHI-Labs/Matting-Anything) commit `eb01e1c1f0016e2a4d00615b115f3b5ddcd031e4`, MIT |
| MAM weights | [`shi-labs/Matting-Anything/checkpoints/mam_sam_vitb.pth`](https://huggingface.co/shi-labs/Matting-Anything) (repository revision `c22773e`), 407,858,479 bytes, SHA-256 `b7d0781aa6edeab7b0d8195edf5b894402f9fa5b7220759825ff750c75450f08`, MIT. Holds the frozen SAM ViT-B (`module.seg_model.*`) and M2M (`module.m2m.*`, 2.7 M parameters) |
| SAM ViT-B check | `sam_vit_b_01ec64.pth` (Apache-2.0), SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`: the SAM weights in the MAM checkpoint are equal (largest absolute difference 0.0) |
| References | `export_mvp_references.py` on DirectML (onnxruntime-directml 1.24.4): S3OD on the box crop (accepted graph, README U14; selected output; 1 - alpha for Sky), one SAM 2.1 tiny box prompt, the MVP one-click Sky and its sky points (P1 run 2 head, 3 points per region) |
| MAM runtime | PyTorch 2.14.1 CPU, 4 threads (`venv_mam`) |

## 4. Method

- Inference as `gradio_app.py` and `inference_benchmark.py` of the MAM repository
  (`mam_model.py`): resize the long edge to 1024, SAM ViT-B encoder, prompt, mask decoder with three
  outputs (the selected output has the largest IoU score), M2M on (image embedding, image, binary
  SAM mask), alpha at output strides 8, 4, and 1 on the 1024 grid, upsampled to the image size, then
  the output fusion of the repository (the "iterative refinement" of the MAM paper):
  - `alpha` (demo default): os8, with the SAM mask outside the bounding rectangle of the SAM mask;
    then os4 and os1 in dilated bands around the soft area.
  - `mask_one_side` (demo "mask" mode): the SAM mask; os8 only in the band outside the SAM mask;
    then os4 and os1 as in `alpha`.
  - `complement`: the guide mask and the SAM mask are 1 - the selected SAM mask; fusion `alpha`;
    Sky = 1 - alpha. This is not a mode of the repository; it gives M2M the non-sky part, which is
    a foreground for the M2M training data.
- Sky: the MVP sky points (one prompt of 3 points per region, the maximum of the regions) with the
  three variants above, and the Sky box as a box prompt (demo default).
- Objects: MAM on the box crop (the S3OD crop rectangle, margin 0.1; box prompt in crop pixels) in
  `alpha` and `mask_one_side`, and MAM on the photo with the box prompt (`alpha`).
- Check: by eye on the check pages, with a zoom window of 384 x 384 photo pixels at 1:1 for each
  method (the window inside the box with the most soft MAM and S3OD pixels). Indicators: area
  (share of the photo pixels with alpha >= 0.5), IoU of the binarized masks, and the soft share
  (share of the box pixels with 0.02 < alpha < 0.98).

## 5. Results

Run `check_run3`: 499 s for 17 photos on the CPU. The SAM ViT-B encoder takes 5.9 to 7.7 s per
image, and the decoder, M2M, and fusion take 0.9 to 1.8 s per prompt (PyTorch CPU; these are not
product values). The memory was not measured.

### 5.1 Sky

Area = share of the photo with Sky >= 0.5. IoU MVP = IoU with the MVP one-click Sky.

| Photo | MAM `alpha` | MAM `mask_one_side` (IoU MVP) | MAM complement | MVP Sky | 1 - S3OD (box crop) | Remark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c005 | 0.100 | 0.440 (0.87) | 0.789 | 0.384 | 0.628 | `alpha`: the tree; complement: rocks and ground partly sky |
| c011 | 0.107 | 0.761 (0.99) | 0.895 | 0.752 | 0.716 | `alpha`: the tree in place of the sky; complement: hill and castle sky |
| c013 | 0.044 | 0.450 (0.99) | 0.855 | 0.448 | 0.513 | |
| c017 | 0.068 | 0.757 (0.98) | 0.986 | 0.760 | 0.681 | Fog |
| c018 | 0.003 | 0.357 (0.98) | 0.834 | 0.351 | 0.481 | Complement: best tree edges, field partly sky |
| c038 | 0.040 | 0.355 (0.97) | 0.758 | 0.346 | 0.434 | |
| c056 | 0.086 | 0.591 (0.98) | 0.909 | 0.590 | 0.652 | |
| c059 | 0.201 | 0.360 (0.98) | 0.766 | 0.360 | 0.450 | |
| c066 | 0.062 | 0.512 (0.20) | 0.695 | 0.103 | 0.580 | SAM ViT-B selects the sky that SAM 2.1 misses |
| c068 | 0.098 | 0.344 (0.98) | 0.647 | 0.338 | 0.401 | |
| c070 | 0.124 | 0.432 (0.95) | 0.825 | 0.410 | 0.493 | |
| c072 | 0.091 | 0.306 (0.99) | 0.872 | 0.304 | 0.640 | |
| c077 | 0.099 | 0.389 (0.99) | 0.648 | 0.386 | 0.469 | Complement: best crown edges, heath partly sky |
| c084 | 0.000 | 0.029 (0.82) | 0.918 | 0.035 | 0.470 | The sky points fail for SAM ViT-B as for SAM 2.1 |
| c099 | 0.000 | 0.480 (0.99) | 0.916 | 0.475 | 0.584 | |

The Sky box prompt gives the same result as the `alpha` sky points (area 0.000 to 0.201).

### 5.2 Objects

Area = share of the photo with alpha >= 0.5. MAM on the box crop.

| Photo | Box | MAM `alpha` (IoU S3OD) | MAM `mask_one_side` (IoU S3OD) | S3OD crop | SAM 2.1 box | Remark (MAM `alpha`) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c005 | tree | 0.132 (0.77) | 0.190 (0.78) | 0.157 | 0.171 | Crown open; at 1:1 blurred |
| c011 | bare tree | 0.103 (0.80) | 0.214 (0.52) | 0.115 | 0.209 | Twigs as blurred halos (zoom); S3OD sharper |
| c013 | backlit tree | 0.117 (0.82) | 0.151 (0.67) | 0.107 | 0.138 | |
| c017 | frosted tree in fog | 0.066 (0.64) | 0.187 (0.51) | 0.095 | 0.179 | Fine soft branches; S3OD has frost spots |
| c018 | bare tree | 0.144 (0.84) | 0.220 (0.66) | 0.145 | 0.211 | Twigs as blurred halos (zoom) |
| c038 | rowan | 0.201 (0.91) | 0.230 (0.87) | 0.202 | 0.226 | |
| c056 | windmill | 0.057 (0.31) | 0.172 (0.91) | 0.166 | 0.175 | **Body transparent** |
| c056 | spruce | 0.010 (0.74) | 0.012 (0.64) | 0.008 | 0.013 | |
| c066 | lighthouse | 0.033 (0.79) | 0.040 (0.97) | 0.039 | 0.041 | **Tower partly transparent** |
| c068 | car | 0.303 (0.95) | 0.317 (1.00) | 0.317 | 0.314 | Lower body part transparent |
| c070 | tree row | 0.088 (0.63) | 0.136 (0.57) | 0.099 | 0.119 | The tall tree; the row soft |
| c072 | car | 0.065 (0.74) | 0.087 (0.99) | 0.087 | 0.088 | Body partly transparent |
| c072 | trees at a forest edge | 0.092 (0.00) | 0.245 (0.00) | 0.006 | 0.244 | Trees selected, soft (S3OD fails) |
| c077 | pine | 0.166 (0.91) | 0.186 (0.85) | 0.159 | 0.181 | |
| c084 | tower | 0.002 (0.04) | 0.049 (0.98) | 0.049 | 0.050 | **Only the outline** |
| c099 | church tower | 0.001 (0.02) | 0.017 (0.97) | 0.016 | 0.017 | **Only the spire** |
| c123 | dog | 0.180 (0.99) | 0.180 (0.99) | 0.180 | 0.178 | Fur soft, blurred |
| c131 | long-haired dog | 0.101 (0.98) | 0.101 (0.98) | 0.099 | 0.099 | Hair soft, blurred; S3OD sharp, hard |

MAM on the photo with the box prompt gives the same failures (church tower area 0.000, tower
0.002, the spruce `c056` 0.000).

## 6. Commands

```text
E=build/tmp/explore
W=build/tmp/mam_poc
# source and weights: scripts/poc/mam/README.md
$E/venv_ai_mask_mvp/Scripts/python -I scripts/poc/mam/export_mvp_references.py
$E/venv_mam/Scripts/python -I scripts/poc/mam/check_mam.py --out $W/check_run3
# pages: build/tmp/mam_poc/check_run3/index.html
```

`results/mam_poc/summary.json` is `check_run3/summary.json`; `results/mam_poc/references.json` is
`references/references.json`.

## 7. Recommendation

1. **Do not use MAM for Sky.** M2M is trained to matte foreground objects (people, animals,
   objects). It does not matte a sky region, and as a complement it does not keep the ground
   opaque. The fine tree edges of the complement are real, but they come with the S3OD failure
   (non-salient content becomes sky).
2. **Do not replace S3OD on the box crop with MAM for objects.** MAM is better only on the forest
   edge `c072`; it is worse on rigid objects (transparent inside in the demo mode, SAM edges in the
   mask mode) and blurs twigs and hair. It also adds a third image encoder.
3. Sky stays open. The closest path is the one of the box S3OD report (section 7.3): a sky output
   trained on the S3OD encoder, with training data that permits the use (U13 is open), or a
   pretrained sky matting model whose weights the user downloads (U16).

## 8. Open items

- User check of the pages.
- The DirectML cost (an export of SAM ViT-B and M2M to ONNX) was not measured.
- The MAM ViT-L and ViT-H checkpoints (Google Drive of the repository) were not checked. They
  change the SAM encoder, not the M2M training data, so the Sky result is not expected to change.
- The combination of the SAM ViT-B sky mask and the complement alpha (the complement edges only
  near the SAM sky edge) was not tried. It would be an edge combination rule outside the model
  (README U3) and needs a user decision.
- The terms of the M2M training datasets were not checked (U16 makes the user responsible for the
  use of the weights).
