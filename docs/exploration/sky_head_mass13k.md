# Fine sky head on SAM 2.1 tiny features, trained on MaSS13K

**Date:** 2026-10-26. **Issue:** [#10](https://github.com/zidage/alcedo-ai-mask-lab/issues/10).
**Code:** [`scripts/poc/sky_head/`](../../scripts/poc/sky_head/README.md).
**Results:** [`results/sky_head/`](../../results/sky_head/).
**Status:** run 1 trained, measured, and exported; export tolerance 1e-3 accepted by the user
2026-10-26 (section 6); user check of the sheets open.

> **Photos.** MaSS13K photos (README U18). The comparison sheets show MaSS13K photos and stay
> local in `build/tmp/explore/sky_head/sheets_test/` (`*.jpg` is in `.gitignore`). Only text
> results are committed.

## 1. Question

The MVP Sky (P1 head and SAM 2.1 point prompts) does not select the full sky: the decoder gives
256 x 256 logits, sky between branches is lost, and some sky regions get no points. README U17
selects an own sky head with a fine alpha output on the SAM 2.1 features; U18 allows MaSS13K as
training data.

Does a small head on the frozen SAM 2.1 tiny encoder outputs, trained on MaSS13K, give a better
Sky than the MVP, and at which cost in the product?

## 2. Result summary

- **Yes, much better than the MVP Sky on the MaSS13K test split** (1,500 photos; same target, same
  measurement code; table in section 5): IoU 0.875 against 0.642, boundary F1 0.896 against
  0.272, edge SAD 5.05 against 24.3 on the 938 sky photos. The head is better on 915 of 938 sky
  photos. Median IoU 0.963 against 0.803.
- **Fine edges:** the head keeps the sky between bare branches and twigs (sheets, for example
  `36388`: MVP boundary F1 0.029, head 0.988). The MVP mask has the blocks of the 256 x 256
  logits.
- **Photos without sky:** a false sky share of 0.009 % (mean of 562 photos) against 0.26 % for
  the MVP. 10 of 562 photos have more than 0.1 % false sky (MVP: 11). The worst false photo
  (`06934`, 1.3 %) has a small visible sky area that the label does not have.
- **Failures:** 58 of 938 sky photos have a head IoU below 0.5 (MVP: 269). The worst photos with
  at least 1 % sky are mostly sky seen through windows and glass doors from inside a building:
  MaSS13K labels it as sky, and the head and the MVP do not select it. Others are a dark night sky
  (`11038`) and small overexposed sky through a gate (`06526`).
- **Product cost:** 30 ms per head run on DirectML (median, NVIDIA GeForce RTX 3080 Laptop GPU;
  section 7); `sky_head.onnx` is 5.8 MB (1.44 M parameters). The head uses the outputs
  of the encoder that the photo open already runs, so Sky needs no second image encoder.
- **Export:** ONNX Runtime 1.31 on the CPU differs from PyTorch by up to 4.0e-4 (alpha) on 8 val
  photos. The cause is the float32 reductions of ONNX Runtime in GroupNorm over up to 2 M values
  (section 6). The difference is about 1/10 of one 8-bit alpha level. The user accepted it
  (2026-10-26): the export tolerance is now 1e-3, and the export passes.

## 3. Inputs

| Item | Value |
| --- | --- |
| Data | MaSS13K (README U18): train 11,348, val 500, test 1,500 photos (4,032 x 3,024), labels from `masks.zip` (class 5 = sky). 3 Jupyter checkpoint copies skipped. Photos with sky (target >= 0.5 at one or more pixels): val 306, test 938; train 7,128 with a sky label pixel (manifest) |
| Prepared data | Photo long edge 2,048 px (JPEG 95), sky coverage alpha (area average of the 4K hard label), 17 GB |
| Encoder | SAM 2.1 tiny (`sam2_graphs.PointEncoder`, commit `2b90b9f5`, checkpoint SHA-256 `7402e0d8...be69`), frozen; bfloat16 autocast in training, float32 without TF32 in validation |
| Head | `SkyAlphaHead`, 1,435,523 parameters: trunk on `image_embed`, decoder with `high_res_feats_1`, `high_res_feats_0`, and an RGB branch at 256, 512, 1,024; output 1,024 x 1,024 logits |
| Loss | Alpha L1, Sobel gradient L1, Laplacian pyramid (5 levels), coarse BCE x 0.25 (`sky_head_losses.py`) |
| Training | 20 epochs, batch 8, AdamW lr 1e-3, weight decay 0.01, 500 warm-up steps, cosine; augmentation: crop 0.6 to 1.0, flip, colour, gamma, hue |
| Machine | RunPod, NVIDIA RTX 4000 Ada (20 GB), PyTorch 2.8.0+cu128, about 15 CPU cores |
| MVP baseline | `ai_mask_mvp/mask_pipeline.py` Sky selection, E4 graphs, P1 run 2 head, DirectML (onnxruntime-directml 1.24.4) |

## 4. Runs and changes during the run

1. **Data stages** (2026-10-26, pod time): download 14 min (rclone, 58 GB), prepare 16 min,
   pilot: 0.77 s per step, 10.4 photos/s, data wait 0 %, CUDA memory peak 16.8 GB, estimate 6.0
   hours for 20 epochs.
2. **Overfit check failed at 300 steps** (16 photos without augmentation): IoU 0.84, MAE 0.095,
   loss still decreasing. Two checks: a full-resolution BCE term at 300 steps gave the same
   values (no change to the loss); 1,200 steps gave MAE 0.014, boundary F1 0.91, IoU 0.87 to
   0.997 on the 15 sky photos. The check had too few steps; the default is now 1,200 steps
   (`results/sky_head/overfit_checks/`).
3. **Measurement change after epoch 2:** on a photo without sky, one false sky pixel set IoU and
   boundary F1 to 0, so the mean showed the count of these photos (194 of 500 val photos), and
   best.pt followed it. IoU and boundary F1 are now on sky photos only, and photos without sky
   have the false sky share. Training continued from epoch 3 (`--resume`) with the new code and
   the best value reset. `metrics.jsonl` epochs 0 to 2 have the old values
   (`results/sky_head/run1/metrics_note.txt`).
4. **Training:** epochs 3 to 19, 6.0 hours; validation 105 to 112 s per epoch.

| Epoch | Train loss | Val IoU | Val boundary F1 | Val edge SAD | Val false sky |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.1026 | 0.842 | 0.871 | 6.46 | 0.028 % |
| 6 | 0.0863 | 0.858 | 0.893 | 5.84 | 0.046 % |
| 10 | 0.0773 | 0.860 | 0.886 | 5.78 | 0.027 % |
| 15 (best.pt) | 0.0681 | 0.874 | 0.902 | 5.33 | 0.029 % |
| 19 | 0.0642 | 0.871 | 0.899 | 5.36 | 0.027 % |

The values change little after epoch 12; more epochs of this setup are not expected to give much.

## 5. Results on the test split

Same target (sky coverage at 1,024 x 1,024 on the stretched model input), same code
(`sky_head_metrics.py`). Head: best.pt, encoder float32 (PyTorch, CUDA). MVP: binary mask, the
union of the region masks (logits > 0).

| Measurement | MVP Sky | Sky head |
| --- | --- | --- |
| IoU (938 sky photos) | 0.642 | **0.875** |
| Boundary F1, 3 px (938) | 0.272 | **0.896** |
| Edge SAD (938) | 24.3 | **5.05** |
| Edge MSE (938) | 0.216 | **0.032** |
| Edge gradient (938) | 0.149 | **0.067** |
| MAE (1,500) | 0.0192 | **0.0035** |
| False sky share (562 photos without sky) | 0.26 % | **0.009 %** |
| Sky photos with IoU < 0.5 | 269 | **58** |
| Sky found (MVP: presence and a region) | 789 of 938 sky photos, 11 of 562 others | - |

The MVP mask is binary, so its edge SAD and MSE also contain the missing partial alpha; IoU and
boundary F1 do not depend on that.

**Sheets** (`make_sky_head_sheets.py`, 24 random sky photos, 8 worst sky photos with at least 1 %
sky, 6 photos without sky with the most false sky; local): checked by eye by the agent. The head
follows tree crowns, bare branches, poles, and roof lines; the MVP follows the 16 px blocks and
misses sky regions between trees. The worst photos are sky through glass from inside a building
(labeled as sky), a dark night sky, and overexposed sky through a gate.

## 6. ONNX export

`export_sky_head_onnx.py` (opset 17) wrote `sky_head.onnx` (5,821,904 bytes, SHA-256
`a6c555e7...4788`): ONNX Runtime 1.31 (CPU) against PyTorch (CPU, float32) maximum 4.0e-4 on 8 val
photos. The first export used a tolerance of 1e-4 and stopped with an error; after the user
decision below, the export ran again with 1e-3 and passed (the same file, the same SHA-256;
`results/sky_head/run1/sky_head.json`).

Checks (local scripts on the pod, results here):

| Check | Result |
| --- | --- |
| PyTorch float32 against PyTorch float64 | at most 8.6e-6 |
| ONNX Runtime against PyTorch float64 | 8.3e-5 to 4.1e-4; 0 to 141 pixels above 1e-4 per photo, at alpha 0.2 to 0.6 (edges) |
| ONNX Runtime graph optimization disabled, basic, extended, all | same difference (4.0e-4, 3.2e-4, 4.2e-4 on 3 photos) |
| Per module, real inputs | GroupNorm (exported as `InstanceNormalization`) has 100 to 1,300 times the relative error of PyTorch; largest at 1,024 x 1,024 (2 M values per group) |
| GroupNorm as two-pass `ReduceMean` operations | same difference |
| DirectML (onnxruntime-directml 1.24.4) against CPU | at most 7.6e-4 on 38 test photos |

The difference comes from float32 sums over many values in ONNX Runtime; the graph is correct.
4e-4 is about 1/10 of one 8-bit alpha level (1/255 = 3.9e-3).

**User decision (2026-10-26):** the difference is in the permitted range. The export tolerance of
the sky head is 1e-3 (about a quarter of one 8-bit alpha level). The head is not changed.

## 7. Cost

| Item | Value |
| --- | --- |
| Head on DirectML | 30.0 ms median, 45.1 ms maximum, 37 runs (first run not counted); NVIDIA GeForce RTX 3080 Laptop GPU, onnxruntime-directml 1.24.4 |
| Head on the pod CPU (ONNX Runtime 1.31) | 6 to 10 s per run (not a product target) |
| Graph size | 5.8 MB (float32) |
| Training | 0.77 s per step on an RTX 4000 Ada; 20 epochs 7.0 hours with validation |

GPU memory and macOS Core ML are not measured.

## 8. Limits and open points

- **Data domain:** MaSS13K photos are smartphone photos of Chinese cities (Chengdu, Tianjin,
  Wuhan, Yinchuan), mostly street level. Landscapes, mountains, sunsets, night sky, and RAW
  photos with a high dynamic range are rare. The quality photos of Alcedo Studio users are not
  measured.
- **Labels:** MaSS13K labels are hard class labels at 4K; the alpha target is their area average,
  not a matting alpha. Sky through glass is labeled as sky.
- **License:** README U18 (notice to the user in the Sky function).
- **The COCO matting labels** (U17, [report](sky_matte_labels.md)) are not used in run 1.
- **Product:** Sky in Alcedo Studio uses `sky_head.onnx` after the encoder. The head has no
  presence output; the "sky found" rule is a threshold on the sky share of the alpha (section 10).
- macOS Core ML conversion of the head is not done.

## 9. Commands

On the pod (`scripts/poc/sky_head/README.md`):

```text
bash run.sh setup download inspect prepare pilot overfit
bash run.sh train evaluate export
```

On Windows (MVP environment, prepared test split copied from the pod):

```bash
PY="build/tmp/explore/venv_ai_mask_mvp/Scripts/python -I"; S=scripts/poc/sky_head
$PY $S/evaluate_mvp_sky.py --data-dir build/tmp/mass13k_prepared --split test \
  --out results/sky_head/mvp_sky_test.json
$PY $S/make_sky_head_sheets.py --data-dir build/tmp/mass13k_prepared --split test \
  --head-measurements results/sky_head/run1/best_test_measurements.json \
  --mvp-measurements results/sky_head/mvp_sky_test.json \
  --sky-head-onnx build/tmp/explore/sky_head/run1/export/sky_head.onnx \
  --out-dir build/tmp/explore/sky_head/sheets_test
$PY $S/measure_sky_found_rule.py --data-dir build/tmp/mass13k_prepared --split test \
  --sky-head-onnx build/tmp/explore/sky_head/run1/export/sky_head.onnx \
  --mvp-measurements results/sky_head/mvp_sky_test.json \
  --out results/sky_head/sky_found_rule_test.json
```

## 10. "Sky found" rule for Alcedo Studio

**Date:** 2026-10-26. **Issue:** AlcedoStudio
[#536](https://github.com/zidage/AlcedoStudio/issues/536). **Code:**
[`measure_sky_found_rule.py`](../../scripts/poc/sky_head/measure_sky_found_rule.py). **Result:**
[`results/sky_head/sky_found_rule_test.json`](../../results/sky_head/sky_found_rule_test.json).

The head has no presence output, so Alcedo Studio needs a rule for "No sky found". The rule is:
sky is found when the share of the 1,024 x 1,024 head alpha values >= 0.5 is at least a threshold.
The script ran the MVP encoder and the head on DirectML (onnxruntime-directml 1.24.4) on the 1,500
test photos (409 s). A photo has sky when its target is >= 0.5 at one or more pixels (938 photos).

| Sky share threshold | Sky photos found (of 938) | Sky photos missed, target share >= 1 % | Photos without sky found (of 562) | Correct decisions (of 1,500) |
| --- | --- | --- | --- | --- |
| > 0 | 927 | 4 | 46 | 1,443 |
| 1e-5 | 924 | 5 | 38 | 1,448 |
| 1e-4 | 917 | 5 | 28 | 1,451 |
| 3e-4 | 907 | 5 | 18 | 1,451 |
| **5e-4** | **906** | **5** | **15** | **1,453** |
| 1e-3 | 897 | 6 | 10 | 1,449 |
| 2e-3 | 884 | 7 | 7 | 1,439 |
| 1e-2 | 793 | 23 | 2 | 1,353 |
| MVP (presence and a region) | 789 | - | 11 | 1,340 |

**Result:** 5e-4 (0.05 % of the image) gives the most correct decisions. Alcedo Studio uses it
(`task_masks::SKY_FOUND_SHARE`). Of the 32 missed sky photos, 27 have a target sky share below
1 %. The 5 misses with more sky are the failure cases of section 5: sky through glass from inside
a building (`25795`, `57765`, `25733`, `38447`) and the dark night sky (`11038`). The 15 photos
without sky that the rule finds are not checked by eye; the largest share is 1.3 % (`06934`, the
photo of section 2 with a small sky area that the label does not have).
