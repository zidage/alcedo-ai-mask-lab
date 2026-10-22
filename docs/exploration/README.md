# AI Mask candidate models (#458)

Reports of the model re-evaluation for the AI Segmentation Mask. The steps and the evidence rules
are in [the master plan, section 8.16.4](../background/model_re_evaluation.md#8164-exploration-steps-458).
Weights, ONNX files, and full-size outputs stay under `build/tmp/explore/` and are not committed.

The comparison images of the reports are under `images/`. They show photos that must not be
shared again, so `images/` is in `.gitignore` and is not distributed with the GitHub repository.
The image links in the reports work only on a machine that has the images; each report states the
command that makes them from the local evidence.

| Step | Report | Status |
| --- | --- | --- |
| E1 | [SAM 2.1 point and box prompts](e1_sam2_1_point_box_evaluation.md) | Measured 2026-10-19; user review open |
| E2 | [BiRefNet one-click Subject and matting](e2_birefnet_subject_matting.md) | Measured 2026-10-19; stopped by the user; only lite kept |
| E3 | [Stroke selection](e3_stroke_selection.md) (sample points along a stroke; replaces the Sky and People models, which are not done for now) | Measured 2026-10-20; user review open |
| E4 | [Click selection with candidates and hover highlight](e4_click_candidates.md) | Measured 2026-10-19; user review open |
| E5 | [Edge stage of the stroke selection (people)](e5_edge_stage.md) (SAM 2.1 tiny; guided-filter upsampling against the SM1 projection matte, on the photo, rectangle, and tile passes) | Measured 2026-10-20; user review open |
| E6 | Report per tool and recommendation | Not started |
| P0 | [Locate people and sky from SAM 2.1 tiny image features (no training)](p0_sam_feature_localization.md) ([#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1)) | Measured 2026-10-20; sky usable on `image_embed`, people not usable without training; user decision on step 1 and O4 open |
| P1 | [People and sky head on SAM 2.1 tiny `image_embed` (step 1)](p1_people_sky_head.md) ([#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1)) | Measured 2026-10-20; run 2 (COCO and Open Images, [#4](https://github.com/zidage/alcedo-ai-mask-lab/issues/4)) measured 2026-10-21: passes the COCO and Open Images criteria, 5 of 7 persons pass on the quality photos, `DSC00674` fails (short box), the waterfall is found as sky; run 3 (GIoU term on the person box) measured 2026-10-21: better box fit on COCO, but 3 of 7 persons and 2 of 3 sky masks pass on the quality photos; keep run 2; prompt rules and People list threshold open |
| MVP | [AI Mask with the selected model set](mvp_ai_mask.md) ([#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5)) | Measured 2026-10-21: same prompts and masks as the P1 and E3 scripts; the Subject alpha differs by up to 0.035 in onnxruntime-directml 1.24.4 (within 2.4e-4 in onnxruntime 1.31.0); 4.5 GB dedicated GPU memory with all models on DirectML; BiRefNet keeps about 3 GB from its first run, and 4 of 9 first BiRefNet runs reset the display driver (cause open, the user suspects the BiRefNet selection); user check of the interaction open |
| Subject alternatives | [Subject models in place of BiRefNet lite](subject_alternatives.md) ([#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6)) | S3OD `s3od-synth` measured 2026-10-21: export with three graph changes for DirectML (Reshape `allowzero`, identity Resize, layer 0 attention in float32 for float16); float32 ONNX within 1.09e-3 of PyTorch (one photo above the 1e-3 tolerance, less than the PyTorch rounding); 643 ms and 3.5 GB (float32), 0 driver resets in 70 sessions against 5 of 11 for BiRefNet; equal or better masks with hard edges. With the attention in 8 chained query chunks (MHA, float16): 191 ms and 1.4 GB, the MVP pipeline with S3OD 2.7 GB peak, so U6 is met; license check of the training data open. IS-Net general-use and universal matting not measured |
