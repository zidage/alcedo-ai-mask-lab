# Alcedo AI Mask Lab

Research and Python prototypes for the **AI Segmentation Mask** of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio), a RAW photo editor. In the editor, the user
selects an object, a person, the sky, or the main subject, and a model makes a Mask that follows the
object edges. The Mask joins the existing Mask system of a Color Grade node.

This README records the product requirements, the user decisions, and the current model selection.
The evidence is in the step reports under [`docs/exploration/`](docs/exploration/README.md).

**Status (2026-10-22):** the Windows DirectML exploration is complete. The accepted model set is
SAM 2.1 tiny, the people and sky head (P1 run 2), and S3OD `s3od-synth` for Subject (U14). The
[acceptance record](docs/exploration/acceptance_ai_mask_model_set.md) gives the evidence for each
requirement and the open items. The [AI Mask MVP](docs/exploration/mvp_ai_mask.md) (issue
[#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5)) runs the model set end to end with a
desktop UI. The [macOS Core ML verification](docs/exploration/macos_coreml_verification.md)
(issue [#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8)) is measured: native Core ML
models of the set meet the output criteria on an Apple M4; the ONNX Runtime Core ML provider does
not run the set as it is, and ONNX Runtime 1.24 (the version of Alcedo Studio) computes the SAM 2.1
tiny encoder wrongly (microsoft/onnxruntime#29071, fixed in 1.28.0). The macOS decisions are U15.
Open: the user check by eye. The section
[Final selection and MVP architecture](#final-selection-and-mvp-architecture-reference-for-alcedo-studio)
gives the summary to port the AI Mask into Alcedo Studio.

## Main project

| Item | Link |
| --- | --- |
| Repository | [zidage/AlcedoStudio](https://github.com/zidage/AlcedoStudio) |
| Re-evaluation issue | [AlcedoStudio#458](https://github.com/zidage/AlcedoStudio/issues/458) |
| Master plan | [AI Segmentation Mask master plan](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md) (decisions D1 to D11 in section 1.2) |
| Branch of the research | [`feature/ai-segmentation-sm1-quality-review`](https://github.com/zidage/AlcedoStudio/tree/feature/ai-segmentation-sm1-quality-review) |
| Runtime measurement program | [`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs) |

The production integration (sidecar inference, C++ client, editor UI) is done in Alcedo Studio.
This repository decides the models and the interaction, and proves them in Python.

## Final selection and MVP architecture (reference for Alcedo Studio)

This section is the summary to port the AI Mask into Alcedo Studio. It contains no new
measurements: each value comes from the
[acceptance record](docs/exploration/acceptance_ai_mask_model_set.md), the
[MVP report](docs/exploration/mvp_ai_mask.md), and the
[macOS Core ML report](docs/exploration/macos_coreml_verification.md). The reference code is
[`scripts/poc/ai_mask_mvp/mask_pipeline.py`](scripts/poc/ai_mask_mvp/mask_pipeline.py) (no UI);
the PySide6 UI [`ai_mask_app.py`](scripts/poc/ai_mask_mvp/ai_mask_app.py) only calls it.

### Models and runtime per platform

| Model (function) | Windows: ONNX Runtime DirectML | macOS: native Core ML (U15) | License |
| --- | --- | --- | --- |
| SAM 2.1 Hiera tiny image encoder (F3 to F6) | `point_encoder.onnx`, float32, 109.4 MB | `point_encoder.mlmodelc`, float32, compute units ALL, 134.3 MB | Apache-2.0 |
| SAM 2.1 tiny prompt encoder and mask decoder (F3 to F6) | `point_decoder.onnx`, float32, 21.2 MB | `point_decoder.mlmodelc`, float32, **CPU_ONLY**, 21.1 MB; point count N = 1 to 64 | Apache-2.0 |
| People and sky head, P1 run 2, epoch 15 (F3, F4a) | `people_sky_head.onnx` and `people_sky_head.json` (thresholds), float32, 1.1 MB | `people_sky_head.mlmodelc`, float32, ALL, 1.1 MB | Own model (COCO 2017, Open Images V7) |
| Fine sky head, run 1, epoch 15 (Sky; U17, [report](docs/exploration/sky_head_mass13k.md)) | `sky_head.onnx`, float32, 5.8 MB | `sky_head.mlmodelc`, float32, ALL, 5.9 MB (report section 11) | Own model; training data MaSS13K (U18) |
| S3OD `s3od-synth` (F1, F2) | `subject_masks.onnx`: attention in 8 chained query chunks (`com.microsoft.MultiHeadAttention`), float16, layer 0 attention float32, 219.3 MB | `s3od.mlmodelc`: `float16_embeddings_attention_float32` (patch embedding and encoder attention float32, other ops float16), ALL, 271.2 MB | MIT; DINOv3 License for the encoder (ship the license text with the weights) |

- The native Core ML models have the same input names, output names, and shapes as the ONNX
  graphs. The pipeline code is the same on both platforms.
- The SHA-256 values are in the acceptance record, section 1 (ONNX), and in
  `results/macos_coreml/native_*_convert.json` (Core ML). The export and conversion commands are
  in [`scripts/ai_models/sam2/`](scripts/ai_models/sam2/README.md),
  [`scripts/poc/people_sky_head/`](scripts/poc/people_sky_head/README.md),
  [`scripts/ai_models/s3od/`](scripts/ai_models/s3od/README.md), and
  [`scripts/ai_models/coreml/`](scripts/ai_models/coreml/README.md).
- Do not use the ONNX Runtime Core ML execution provider on macOS: it runs S3OD on the CPU (7.9 s,
  10.3 GB), and the decoder stops the process in the MLProgram format.
- Use ONNX Runtime 1.28.0 or later. ONNX Runtime 1.24.4 to 1.27.0 (Alcedo Studio uses 1.24,
  `ort` 2.0.0-rc.12 with `api-24`) fold the two `If` nodes of the SAM 2.1 tiny encoder into freed
  memory (microsoft/onnxruntime#29071). On the Mac the encoder output was wrong in every process;
  on Windows the result depends on the memory reuse.
- ONNX Runtime session settings of the MVP: one execution provider per session and no other
  provider after an error; 4 intra-op CPU threads, 1 inter-op thread; CPU memory arena off;
  DirectML with sequential execution and memory patterns off.
- Core ML: load compiled `.mlmodelc` models, as Alcedo Studio loads its SigLIP2 models
  (`rust/puerh_mind/src/service/coreml_clip.rs`, `coreml` crate), with a compute units setting per
  model. On the GPU, Core ML prepares the decoder again for each new point count (about 52 ms); on
  the CPU the decoder takes about 30 ms for each call.

### Data flow

```text
Open the AI Mask tool on a photo
  photo -> 8-bit RGB, EXIF orientation applied -> long edge at most 4096 px (Lanczos)
        -> stretch to 1024 x 1024, ImageNet mean and std
        -> SAM 2.1 encoder (once per photo)
           -> image_embed [1,256,64,64], high_res_feats_0 [1,32,256,256], high_res_feats_1 [1,64,128,128]
        -> head(image_embed) (once per photo)
           -> sky_logits, center_logits, center_offset, log_size [1,C,64,64], presence_logits [1,2]

Sky      presence (sigmoid of presence_logits[0]) < 0.4232 -> "not found"
         else 8-connected regions of cells with sky probability >= 0.5 and >= 3 cells
         -> 3 most probable cell centers per region -> decoder per region -> union of the masks
People   presence (presence_logits[1]) < 0.5332 -> "not found"
         else decode_people: 3 x 3 local maxima of the center heatmap, top 20, score >= 0.05,
         box = center + offset, size = exp(log_size), boxes with IoU >= 0.5 with a higher box removed
         -> People list = boxes with score >= 0.3132 -> one box prompt per person (clipped to the
         photo) -> decoder per person -> one person or the union of all
Stroke   4 points at equal spacing along the pointer path (photo pixels, both ends); a click = 1 point
Box      2 corners (labels 2, 3), always first in the prompt; a new box replaces the box
Add/sub  label 1 or 0 (Alt); each interaction decodes ALL points and the box of the selection in
         one decoder call, with the previous selected logits as mask_input (has_mask_input = 1)
Subject  letterbox: long edge 1024 (bilinear), centered zero padding, ImageNet mean and std
         -> S3OD -> masks [1,3,1024,1024], iou_scores [1,3] (both after the sigmoid)
         -> the mask with the largest iou_score -> remove the padding -> alpha (once per photo)
Background = 1 - Subject alpha (no model run)
```

Rules of the mask outputs:

| Item | Rule |
| --- | --- |
| Decoder prompt coordinates | Normalized photo coordinates x 1024 (the encoder input is stretched, so x and y scale separately) |
| Decoder mask selection, box or several points | Token 0 when its stability (area of logits > 0.05 / area of logits > -0.05) is >= 0.98, else the token 1 to 3 with the best `iou_predictions`; logits clipped to +-32 |
| Decoder mask selection, one point without a box (click) | The token 1 to 3 with the best `iou_predictions`; the three tokens, sorted by area, are the levels of `[` and `]` (no model run) |
| SAM 2.1 mask at the photo size | 256 x 256 logits, bilinear to the photo size, threshold 0 |
| S3OD alpha at the photo size | Bilinear with antialiasing (PyTorch `interpolate`, `align_corners=False`), clipped to [0, 1] |
| Edge, Feather (F8) | Edge: dilate (> 0) or erode (< 0) by \|edge\| px with alternating 3 x 3 cross and square steps; Feather: Gaussian blur, sigma in px. No edge post-processing (U3) |

The thresholds come from `people_sky_head.json` (best F1 on COCO validation). The decoder state of
a selection is: all points and labels, the box, the selected logits of the last call, and the
click levels.

### Lifecycle and memory

- Load all four models when the AI Mask tool opens. The encoder and the head run once per photo;
  the Subject alpha is computed at the first Subject or Background selection and kept for the
  photo. U6 asks to release the models when the user goes back to adjusting the photo; the MVP
  does not do this.
- Memory with all models loaded: Windows 2,666 MB dedicated GPU memory (budget 3 GB, U6); macOS
  3,190 MB peak physical footprint of the Python process (value 3.5 GB, U15). The macOS value
  includes the Python runtime, PyTorch, and the photo arrays of the MVP.
- In the MVP UI, the model jobs run in one worker thread with a job queue, and the results go to the
  UI thread with a queued signal. coremltools predictions in a PySide `QThread` stopped the
  process (SIGSEGV); this is a Python binding problem, but keep all Core ML calls of a model on one
  worker thread.

### Cost of the MVP pipeline

All models loaded in one process, warm p50 (records:
`results/subject_alternatives/mvp_pipeline_s3od_mha_chunk8_float16_cost.json`,
`results/macos_coreml/mvp_pipeline_native_ALL_decoder_CPU_ONLY_cost.json`):

| Stage | Windows, RTX 3080 Laptop, DirectML | macOS, Apple M4, native Core ML |
| --- | ---: | ---: |
| Session load: encoder, decoder, head, S3OD | 2,623, 249, 112, 910 ms | 1,197, 69, 9, 43 ms (a new process with one model: 0.8 to 1.7 s) |
| Read and scale a 4096 px photo | 65.9 ms | 29 ms |
| Encoder (once per photo) | 70.5 ms | 191 ms |
| Head (once per photo) | 2.0 ms | 2.1 ms |
| Decoder: box; 10 points with `mask_input` | 23.2; 22.2 ms | 33.5; 34.1 ms |
| S3OD (once per photo) | 191.1 ms | 531 ms |

### Port checks

The port is correct when it gives the results of the MVP on the 14 quality photos.
[`compare_providers.py`](scripts/poc/ai_mask_mvp/compare_providers.py) has the checks of
issue #8 against ONNX Runtime 1.31.0 on the CPU: the same Sky and People presence decisions, the
same People list, the same number of sky regions, SAM 2.1 masks with IoU >= 0.999 at the photo
size, the S3OD alpha > 0.5 with IoU >= 0.998 on the 1024 grid, and no NaN or infinite output. The
checks of the MVP against the measurement scripts are in the MVP report, section 3.3.

### Open items for the port

- The user check of the interaction by eye on Windows and macOS (U1).
- Rules that are not done: Subject "not found" (F1), the People list order and threshold (F4a),
  the decoder point limit of a long selection (the Core ML decoder takes 1 to 64 points; each
  stroke adds 4).
- User decisions: F7 and F9 (O1), the text prompts (O3), the training data rule U13, and the S3OD
  float32 tolerance (acceptance record, section 6).

## 1. Background

The first plan used EfficientSAM3 for point, box, and text prompts. Its quality review failed: the
point and box path makes regular grid holes on large regions and fails on close-up portraits,
because the student image encoder gives a 32 x 32 feature map that is resized to the 72 x 72 grid of
SAM3. Single points are ambiguous, and the text path misses many common classes. Details:
[model re-evaluation](docs/background/model_re_evaluation.md) and
[SM1 record](docs/background/sm1_efficientsam3_record.md).

The re-evaluation (Alcedo Studio issue
[#458](https://github.com/zidage/AlcedoStudio/issues/458)) measured these candidates:

| Step | Report | Result |
| --- | --- | --- |
| E1 | [SAM 2.1 point and box prompts](docs/exploration/e1_sam2_1_point_box_evaluation.md) | No grid holes in any size; the close-up box selects the whole person. Tiny: 76 ms encoder, 10.5 ms decoder, 1.36 GB on DirectML |
| E2 | [BiRefNet one-click Subject](docs/exploration/e2_birefnet_subject_matting.md) | Lite finds the expected subject on 12 of 13 photos without a prompt, with a clean but hard hair edge. Matting variants miss rigid objects |
| E3 | [Stroke selection](docs/exploration/e3_stroke_selection.md) | A stroke across the region with 3 or 4 points selects the intended region without a level change on 23 of 25 targets |
| E4 | [Click selection with candidates](docs/exploration/e4_click_candidates.md) | Precomputed candidates are slower and not better than the live point decoder. The user rejected an interaction that follows the pointer |
| E5 | [Edge stage](docs/exploration/e5_edge_stage.md) | A tile pass with guided-filter upsampling improves the edge but costs 0.15 to 1.5 s. The user rejected this edge post-processing |

The interaction of other applications is in
[AI mask interaction in Lightroom and Capture One](docs/research/interaction_practice.md).

## 2. User decisions

These decisions come from the user. Do not change them without a new user decision.

| ID | Decision | Date |
| --- | --- | --- |
| U1 | The interaction must not be worse than Lightroom and Capture One. | 2026-10-20 |
| U2 | Model licenses must permit commercial use. Alcedo Studio is GPLv3 and the user does not sell it, but many users are professional photographers whose use is commercial. Non-commercial weights are excluded. AGPL-3.0, Apache-2.0, MIT, and the Meta SAM License are acceptable. | 2026-10-20 |
| U3 | Edge post-processing is rejected: the tile pass, guided-filter upsampling, and the projection matte are too slow and give poor results. The edge quality comes from the model output. | 2026-10-20 |
| U4 | One-click People and one-click Sky are required. | 2026-10-20 |
| U5 | The selection of one person is more important than recall in crowds. | 2026-10-20 |
| U6 | GPU memory: the editor uses about 5 GB on a 24 MP photo. The 3 GB budget for the models is soft, because the driver can swap part of the editor memory (caches). Models run while the user selects; they are unloaded when the user goes back to adjusting the photo. **macOS (2026-10-22):** the budget can be somewhat larger because of the 16 GB unified memory, but not much larger; U15 sets 3.5 GB (the MVP pipeline with the native Core ML models has a peak footprint of 3,190 MB, [macOS report](docs/exploration/macos_coreml_verification.md#9-cost-and-memory)). | 2026-10-20 |
| U7 | An interaction that follows the pointer (hover highlight) is rejected. The stroke is the prompt tool (E3). | 2026-10-20 |
| U8 | Float16 models are acceptable (lower precision for a shorter processing time). | 2026-10-19 |
| U9 | If BiRefNet is used, use the 512 px variant; efficiency first. See open decision O2. | 2026-10-20 |
| U10 | A comparison experiment of prompt sources for People and Sky ("E7") is not done. | 2026-10-20 |
| U11 | The SAM3 detector is too large for its task (locating people and sky). Evaluate a small model that gives points or a box to SAM 2.1; a model trained by the user (for example on Colab) is acceptable. | 2026-10-20 |
| U12 | The AI Mask MVP (issue #5) uses BiRefNet at 1024 px (lite). The MVP uses the float16 graph of E2 (U8). Replaced by U14 for the Subject model. | 2026-10-21 |
| U14 | F1 and F2 use S3OD `s3od-synth` (the 8 chunk MHA float16 graph of issue #6) in place of BiRefNet lite. The model set is SAM 2.1 tiny, the people and sky head, and S3OD. The exploration on Windows DirectML is complete; the verification on macOS follows. (U13 is the proposed training data rule of issue #6; it is not decided.) | 2026-10-22 |
| U15 | macOS uses native Core ML models (`.mlmodelc`), not the ONNX Runtime Core ML provider (issue #8). The user selected CPU_ONLY for the SAM 2.1 decoder (about 30 ms for each call, no cost for a change of the point count) and gave the other decisions to the exploration: the encoder, the head, and S3OD use the compute units ALL; S3OD on macOS is `float16_embeddings_attention_float32` (the DINOv3 patch embedding and the encoder attention in float32; the accepted precision gives a wrong alpha on the Neural Engine); the macOS memory value is 3.5 GB peak physical footprint of the process with all models loaded (measured 3,190 MB). [macOS report](docs/exploration/macos_coreml_verification.md#13-recommendation) | 2026-10-22 |
| U16 | A model whose weights are not distributed with Alcedo Studio can be a candidate when its weights or training data do not permit commercial use: the user downloads the weights from the source site, Alcedo Studio supplies only the inference, and the user is responsible for a compliant use. First case: Matting Anything (MAM, [report](docs/exploration/mam_poc.md)). U2 stays the rule for weights that Alcedo Studio distributes. | 2026-10-24 |
| U17 | Sky: train an own sky head with a fine alpha output on the SAM 2.1 features; SAM 2.1 stays the interactive model. ZIM is not a candidate, because a second matting model beside S3OD makes the role of S3OD unclear. The fine sky training labels come from photos with a license that permits commercial use (COCO 2017, the P1 license rule) and are made offline with classical matting without learned weights from the coarse COCO sky ([report](docs/exploration/sky_matte_labels.md)). The offline label generation is not edge post-processing in the sense of U3. Rendered synthetic data is not used (too much time). | 2026-10-25 |
| U18 | The sky head can be trained on MaSS13K (CVPR 2025, [repository](https://github.com/xiechenxi99/MaSS13K); 13,348 4K smartphone photos with matting-level labels, sky is a class). The repository README states "This project is released under the Apache 2.0 license"; the linked `LICENSE` file is not in the repository and the dataset has no separate license, so the license of the training data is not certain. In the Sky function, Alcedo Studio tells the user that the training data of the model may not permit commercial use and that the user must not use it for commercial work. This is an exception to U2 for Sky only. | 2026-10-25 |

## 3. Product requirements (proposal)

The requirement set is a proposal. The user has not approved each row.

| ID | Function | Reference | Proposed priority | Acceptance |
| --- | --- | --- | --- | --- |
| F1 | Select Subject (one click) | Lightroom, Capture One | Required | No prompt. All salient subjects in one mask. A photo without a subject gives a "not found" message, not an empty mask |
| F2 | Select Background | Lightroom, Capture One | Required | The inverted F1 mask; no extra model |
| F3 | Select Sky (one click) | Lightroom | Required (U4) | No blocks or glow along tree or building horizons. A photo without sky gives a "not found" message |
| F4a | Select People (one click): all people or one person | Lightroom People list | Required (U4, U5) | A list of the detected people; the user can select one person. Missed people in crowds are acceptable |
| F4b | People parts (skin, hair, clothes, eyes) | Lightroom, Capture One | Not in this release | Needs a human parsing model |
| F5 | Object selection: stroke and box | Lightroom Objects, Capture One AI Select box | Required | A click is a stroke with one point. No pointer-follow highlight (U7) |
| F6 | Add and subtract within one AI Mask | Lightroom, Capture One | Required | Strokes and boxes as model prompts; Alt subtracts (E3) |
| F7 | Combination across mask types (Subject minus a stroke, People intersect Sky) | Lightroom, Capture One 16.7 | Required; needs a user decision | Alcedo Studio decision D6 has only Union (per-pixel maximum). Subtract and Intersect need a composition of the components into one R8 coverage at commit |
| F8 | Feather and Edge sliders | Lightroom 15.5 | Required | Blur and erode or dilate of the final mask; adjustable after commit. Unlike Lightroom, they stay available when a brush component is added |
| F9 | Store the recipe (task and prompts) and recompute | Lightroom paste and presets | Needs a user decision | Store the recipe at least; recompute on paste or sync is an extension |
| F10 | Overlay: color, opacity, toggle key | Lightroom Y key | Required | Alcedo Studio decision D3 |
| F11 | Text prompts | Neither application | Proposed for removal | Removes the text requirement of Alcedo Studio decision D2 |

## 4. Model selection (proposal)

Principle: **SAM 2.1 is the only interactive model.** A task model either makes its own mask
(Subject) or gives prompts (points or a box) to SAM 2.1 (People, Sky).

| Function | Model | License | Download | Evidence and open items |
| --- | --- | --- | --- | --- |
| F5, F6 (and the final mask of F3, F4a) | SAM 2.1 Hiera tiny | Apache-2.0 | 130.6 MB (encoder 109.4 MB, decoder 21.2 MB, float32) | E1, E3. Mask logits are 256 x 256: one logit cell is 16 px at the 4096 px application input |
| F1, F2 | **S3OD `s3od-synth`** (U14), attention in 8 chained query chunks, float16 | MIT; DINOv3 License for the encoder (the license text must go with the weights); training data lineage: proposal U13 | 219.3 MB | [Acceptance record](docs/exploration/acceptance_ai_mask_model_set.md): 191 ms, 1,358 MB, 1.1 s load, 0 driver resets in 88 sessions; the set with SAM 2.1 tiny and the head has a peak of 2,666 MB (U6 met). History: E2 measured BiRefNet lite at 1024 px (float16: 175 ms, 3.2 GB). The 512 px variant is not measured. Alternative measured in issue #6: S3OD `s3od-synth` (MIT, DINOv3 License; 433.8 MB float32): 643 ms, 3.5 GB, 1.0 s load, 0 driver resets in 70 sessions (BiRefNet: 5 in 11), equal or better masks on the 14 photos with hard edges ([Subject alternatives report](docs/exploration/subject_alternatives.md)). With the attention in 8 chained query chunks (`com.microsoft.MultiHeadAttention` per chunk, float16, report section 5.5): 191 ms, 1,358 MB; the MVP pipeline with this S3OD graph has a peak of 2,666 MB (with BiRefNet: 4,528 MB). IS-Net general-use (code Apache-2.0, weights without a license statement; 88.2 MB float16): 14 ms, 437 MB, MVP pipeline peak 1,748 MB, 0 driver resets in 23 sessions, but worse masks than BiRefNet and S3OD on 7 of 14 photos (mean subject recall 0.80; report section 11). Alibaba universal matting (Apache License 2.0; 173.3 MB float32; reads the photo at its size): 116 ms and 2,678 MB at 4096 px, 0 driver resets in 46 sessions, but MVP pipeline peak 3,982 MB (U6 not met) and worse masks than BiRefNet and S3OD on 5 of 14 photos (mean subject recall 0.84; report section 12). Of the three candidates, the report recommends S3OD |
| F3, F4a localization | **People and sky head** on the SAM 2.1 tiny `image_embed`, P1 run 2, epoch 15 (section 5) | Own model; training data COCO 2017 and Open Images V7 | 1.1 MB | [P0](docs/exploration/p0_sam_feature_localization.md): `image_embed` holds the class information (AUC 0.994 to 1.000). [P1](docs/exploration/p1_people_sky_head.md): run 2 is used (section 5). Person alternative without training: YOLOX-Nano (Apache-2.0) |

Total download of the accepted set: 351 MB (SAM 2.1 tiny 130.6 MB, head 1.1 MB, S3OD 219.3 MB).

### 4.1 How People and Sky work

```text
Open the AI Mask tool: SAM 2.1 encoder (once per photo) -> image features
Select Sky:    head reads the features -> sky score map -> a few confident points -> SAM 2.1 decoder -> mask
Select People: head reads the features -> person score map -> one region per person
               -> box or points per person -> SAM 2.1 decoder -> one mask per person -> People list
No sky or no person: the highest score is below the threshold -> "not found" message
```

The SAM 2.1 encoder already reads the whole photo for the interaction. The head only reads the
features, so People and Sky add no second image encoder.

### 4.2 Candidates that were replaced or rejected

| Candidate | Reason |
| --- | --- |
| EfficientSAM3 point and box path | Grid holes, close-up failures (SM1, section 1) |
| EfficientSAM3 text detector as the People and Sky localizer | 201.6 MB (plus a 169.2 MB text encoder), 254 ms, 2.4 s session load, Core ML profile did not finish; misses people in crowds. Replaced by the proposal of U11 |
| BiRefNet general, HR, and matting variants | Speed and memory (E2), matting variants miss rigid objects; user decision 2026-10-19 |
| BiRefNet lite 1024 (MVP Subject model until U14) | 4.5 GB with the set (U6), 7.5 to 9.4 s load, display driver resets in the first run (issue #6) |
| IS-Net general-use | Masks worse than S3OD and BiRefNet on 7 of 14 photos; no weights license (issue #6) |
| Alibaba universal matting | Reads the photo at its full size: 4.0 GB with the set (U6); masks worse on 5 of 14 photos (issue #6) |
| SAM 2.1 automatic mask generator candidates | 0.8 to 3.2 s and 1.3 GB more GPU memory, not better than the live decoder (E4) |
| Tile pass, guided filter, projection matte | U3 |
| SegFormer (ADE20K) for Sky | NVIDIA non-commercial license (U2) |
| Sapiens for People parts | Non-commercial license (U2) |
| ViTMatte and other trimap matting models | Trained on Composition-1k (research use only); also an edge post-processing stage (U3) |
| HQ-SAM 2 | Only the Hiera large size (3.1 GB, 322 ms encoder in E1). Kept as an option for better object edges |

### 4.3 SAM 2.1 derivatives to check

| Model | License | Reason to check |
| --- | --- | --- |
| UnSAMv2 | Apache-2.0 | Granularity control for single-point ambiguity (part or whole object). The backbone size is not confirmed |
| EdgeTAM | Apache-2.0 | RepViT encoder of 8.3M parameters (33 MB). Only if tiny is too slow on a target machine. Check the grid-hole probe of E1 first |
| EfficientTAM | Apache-2.0 | Not preferred: plain single-scale ViT, the same cause as the EfficientSAM3 grid holes is possible |

## 5. Next step: locate people and sky from SAM 2.1 image features

Issue: [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1).

The SAM3 detector is open-vocabulary: it can find any word. The product uses two fixed words, so it
pays for generality that it does not use. People and Sky need only a coarse answer: "is it there,
and where are a few reliable points". The SAM 2.1 encoder already computes a 64 x 64 feature map of
the whole photo.

The proof of concept checks, without training, whether these features separate people and sky
from the rest of the photo. If they do, a small head (one to three convolution layers) is trained on
the frozen SAM 2.1 tiny encoder. If they do not, the person localizer changes to a small detector,
and the sky needs a different feature source.

**Result of the proof of concept (P0, 2026-10-20,
[report](docs/exploration/p0_sam_feature_localization.md)).** No training: one mean feature
vector per class from the other 13 photos (leave one photo out), on the 14 quality photos.

- `image_embed` (256 channels, stride 16) holds the class information. Within a photo the class
  cells score first: AUC 0.994 to 1.000 for people, 1.000 for sky. The pooled high-resolution
  features do not (person AUC 0.52 and 0.78).
- **Sky passes** the proposed acceptance criteria on `image_embed`: the 3 top cells are sky on
  the 3 sky photos, and one threshold separates them from the 3 photos without sky (margin
  0.026; the waterfall mist is the nearest false result). The 3 top cells as SAM 2.1 points give
  the whole sky (IoU 0.97 to 0.995).
- **People fail.** The 3 top cells are on the referenced person on 5 of 7 photos, but no
  threshold separates the photos: the squirrel photo scores higher than five of the seven people.
  The top cells cluster on the face or hair, so the points select a part. A box per person selects
  the whole person (IoU 0.99), but the largest score region is not always the person.
- **Step 1, proposed:** train a head on the frozen `image_embed` of SAM 2.1 tiny: a sky score map
  with a presence score, and for people one center and box per person (this also gives the
  People list). The head must learn a calibrated presence score, person against animal, and mist
  and water against sky.

**Result of step 1 (P1, 2026-10-20, [report](docs/exploration/p1_people_sky_head.md)).** The head
(275,656 parameters, 19.5 ms on CPU) was trained on Colab on 30,836 COCO 2017 photos with a
license that permits commercial use. COCO validation: sky cell IoU 0.895, presence AUC 0.992 (sky)
and 0.989 (people), person AP50 0.623. On the 14 quality photos, end to end with the SAM 2.1 tiny
decoder:

- **People presence passes:** one COCO threshold finds people on all 7 referenced photos and on
  none of the 4 photos without people; the squirrel scores 0.002 (P0 margin -0.177, now 0.870).
- **Sky masks pass:** IoU 0.980 to 0.994, within 0.02 of the E1 sky point. **Sky presence
  fails** on the waterfall of `DSC00936` (0.942, the whole water curtain is marked as sky).
- **People boxes:** 4 of 7 referenced persons get a mask within 0.02 of the E1 box or better
  (also the small boarder, 0.689 against 0.269). On `DSC_1535` and `P1024439` the box of the main
  person is too short (head or legs cut off), and on `DSC_1535` it is ranked after 6 background
  pedestrians.
- **Run 2 (2026-10-21, [report section 10](docs/exploration/p1_people_sky_head.md#10-run-2-coco-and-open-images-v7)):**
  trained on COCO and 24,000 Open Images V7 photos (CC BY 2.0, annotations CC BY 4.0). COCO
  validation stays equal (AP50 0.639); on the Open Images validation split, person AP50 increases
  from 0.473 to 0.533. On the quality photos, 5 of 7 referenced persons pass (`DSC_1535`,
  `P1024439`, `DSC00326` improve), but the close-up `DSC00674` fails: its box ends 5.6 cells
  above the photo edge. The waterfall stays sky (0.999).
- **Run 3 (2026-10-21, [report section 11](docs/exploration/p1_people_sky_head.md#11-run-3-giou-term-on-the-person-box)):**
  run 2 with a GIoU loss on the person box clipped to the photo. On COCO validation the boxes of
  large persons fit better (IoU >= 0.75: 0.490 to 0.528; median size error 2 to 3 percent), AP50
  is equal (0.645). On the quality photos it is worse: 3 of 7 referenced persons and 2 of 3 sky
  masks pass. `DSC00674` is not corrected: its box is almost the full photo, and SAM 2.1 tiny
  selects the background. Box edge changes of 2 to 4 cells change the SAM 2.1 selection.
- **Next:** use run 2. The open decisions are prompt rules (a box with the center point of the
  person, sky points spread over the region) and the order and threshold of the People list.

Risks:

- SAM features carry more shape and edge information than class information. (P0: true for the
  high-resolution features; `image_embed` holds the class information.) Linear probes of the
  SAM encoder for classification are clearly worse than CLIP or DINOv2.
- Two people next to each other give one region in a score map. A center-point output can separate
  them; U5 makes this less important.
- Branches cut the sky into pieces. Each confident piece needs its own points.
- Training data: COCO-Stuff (CC BY 4.0 annotations, Flickr images with mixed licenses) or
  Open Images V7 (CC BY 4.0 annotations, CC BY 2.0 images). The license of each dataset must be
  checked against U2.

## 6. Repository layout

| Path | Content |
| --- | --- |
| [`docs/background/`](docs/background/) | Copied Alcedo Studio plan sections: SM1 record, model re-evaluation |
| [`docs/research/`](docs/research/) | Interaction research of other applications |
| [`docs/exploration/`](docs/exploration/README.md) | Step reports E1 to E5 (images local only) |
| [`scripts/ai_models/sam2/`](scripts/ai_models/sam2/README.md) | SAM 2.1 export, probes, stroke and candidate tools |
| [`scripts/ai_models/birefnet/`](scripts/ai_models/birefnet/README.md) | BiRefNet export, float16 conversion, matte probes |
| [`scripts/ai_models/efficientsam3/`](scripts/ai_models/efficientsam3/README.md) | EfficientSAM3 export and comparison (SM1) |
| [`scripts/ai_models/s3od/`](scripts/ai_models/s3od/README.md) | S3OD export, CPU comparison, DirectML cost, first runs, and quality probes of Subject models (issue #6) |
| [`scripts/ai_models/isnet/`](scripts/ai_models/isnet/README.md) | IS-Net general-use export and CPU comparison (issue #6) |
| [`scripts/ai_models/universal_matting/`](scripts/ai_models/universal_matting/README.md) | Alibaba universal matting conversion (TensorFlow to ONNX), CPU comparison, and TensorFlow spread (issue #6) |
| [`scripts/poc/sam_feature_localization/`](scripts/poc/sam_feature_localization/README.md) | P0: people and sky from SAM 2.1 tiny features |
| [`scripts/poc/people_sky_head/`](scripts/poc/people_sky_head/README.md) | Step 1: people and sky head on SAM 2.1 tiny `image_embed`, trained on Colab |
| [`scripts/poc/ai_mask_mvp/`](scripts/poc/ai_mask_mvp/README.md) | AI Mask MVP: the selected model set with a PySide6 UI (issue #5) |
| [`results/`](results/README.md) | Measurement results of the prototypes (sheets with photos stay local) |
| `build/tmp/` (not committed) | Local evidence, weights, ONNX files, photos, and the Python environments of all steps |

The scripts came from Alcedo Studio commit `d1e0d8c7` (branch
`feature/ai-segmentation-sm1-quality-review`). Issue numbers in the reports (for example [AlcedoStudio#458](https://github.com/zidage/AlcedoStudio/issues/458))
refer to the Alcedo Studio repository. The runtime measurements use
[`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs) in Alcedo Studio.

Rules for contributors and agents: [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## 7. Open decisions

| ID | Question |
| --- | --- |
| O1 | Approve the requirement set of section 3, especially F7 (combination across mask types) and F9 (recipe and recompute). |
| O2 | **Closed by U14 (2026-10-22): S3OD replaces BiRefNet.** Record: BiRefNet variant. U9 asks for the 512 px variant. Because U3 removed edge post-processing, the output resolution is the final edge resolution: lite at 1024 px gives 4 px cells at the 4096 px input (measured: 175 ms, 96.4 MB) and `BiRefNet_512x512` gives 8 px cells (Swin-L, 444 MB, not measured). The recommendation is lite at 1024 px. **The MVP uses lite at 1024 px (U12).** With all models loaded, the MVP needs 4.5 GB dedicated GPU memory, more than the 3 GB budget of U6 ([MVP report, section 5.2](docs/exploration/mvp_ai_mask.md#52-memory)). BiRefNet keeps about 3 GB of DirectML working memory from its first run, and 4 of 9 first BiRefNet runs reset the display driver (`887A0006`); the user suspects the BiRefNet selection ([section 5.3](docs/exploration/mvp_ai_mask.md#53-birefnet-memory-and-gpu-resets)). **Issue #6, S3OD `s3od-synth`** ([report](docs/exploration/subject_alternatives.md)): no driver reset in 70 S3OD sessions, while BiRefNet reset the driver in 5 of 11 sessions in the same runs; with the exported attention S3OD takes 643 ms (float32) to 895 ms (float16) per call against 154 ms, and 3.5 to 3.9 GB. With the attention in 8 chained query chunks (`com.microsoft.MultiHeadAttention` per chunk, float16, layer 0 in float32), S3OD takes 191 ms and 1,358 MB, and the MVP pipeline with S3OD has a peak of 2,666 MB dedicated GPU memory, in the 3 GB budget of U6 ([report, section 5.5](docs/exploration/subject_alternatives.md#55-attention-in-query-chunks)). **Issue #6, IS-Net general-use** ([report, section 11](docs/exploration/subject_alternatives.md#11-is-net-general-use)): float16 14 ms and 437 MB per call, MVP pipeline peak 1,748 MB, 0 driver resets in 23 sessions (the BiRefNet reference reset the driver in 2 of 2 sessions in the same period); but its masks are worse than BiRefNet and S3OD on 7 of 14 photos (soft background areas, one subject not found, mean subject recall 0.80 against 0.92 and 0.99), and `isnet-general-use.pth` has no license statement (DIS#150 has no answer). The report does not recommend IS-Net. **Issue #6, Alibaba universal matting** ([report, section 12](docs/exploration/subject_alternatives.md#12-alibaba-universal-matting)): the TensorFlow graph reads the photo at its size; 116 ms and 2,678 MB per call at 4096 px, 0 driver resets in 46 sessions (the BiRefNet reference run reset the driver again), but the MVP pipeline with it has a peak of 3,982 MB (U6 not met), and its masks are worse than BiRefNet and S3OD on 5 of 14 photos (background areas on 2 photos, mean subject recall 0.84). The report does not recommend it; S3OD stays the recommendation of the three candidates. Open: the float32 tolerance (S3OD 1.09e-3 on one photo, universal matting 6.3e-3; both less than the float32 rounding of the framework itself), the training data rule U13 (FLUX.1 [dev] output terms, DIS5K-trained teacher labels), and the check by eye of the hard S3OD edges and of the IS-Net and universal matting masks. |
| O3 | Remove the text prompt requirement from Alcedo Studio decision D2 (F11). |
| O4 | Person localizer: the SAM 2.1 feature head, or YOLOX-Nano. P0 does not reject the head: `image_embed` holds the person information, but prototypes without training are not sufficient (no threshold separates the photos). The decision needs the trained head of step 1. Sky has no alternative candidate with a commercial license; the P0 sky result is positive. |

## License

GPL-3.0, the same as Alcedo Studio. See [LICENSE](LICENSE). Model weights are not part of this
repository; each model keeps its own license.
