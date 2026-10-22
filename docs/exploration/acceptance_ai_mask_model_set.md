# Acceptance: AI Mask model set (SAM 2.1 tiny and S3OD)

**Date:** 2026-10-22. **Decision:** user, 2026-10-22 (README U14).
**Issues:** [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1),
[#2](https://github.com/zidage/alcedo-ai-mask-lab/issues/2),
[#4](https://github.com/zidage/alcedo-ai-mask-lab/issues/4),
[#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5),
[#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6); macOS verification:
[#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8).
**Status:** the Windows DirectML exploration is complete. The macOS Core ML verification is open.

This document records the model set that the exploration accepts, the evidence for each
requirement, and the items that stay open. It does not add measurements: each number comes from
the report that it names.

## 1. Decision

The AI Mask uses this model set:

| Function | Model | Graph (ONNX) | Bytes | SHA-256 | License |
| --- | --- | --- | ---: | --- | --- |
| F5, F6, and the masks of F3, F4a | SAM 2.1 Hiera tiny, image encoder | `e4/onnx_float32_batch16/tiny/point_encoder.onnx` (float32) | 109,381,695 | `17a554b4...` | Apache-2.0 (`facebookresearch/sam2` at `2b90b9f5`) |
| as above | SAM 2.1 Hiera tiny, prompt encoder and mask decoder | `e4/onnx_float32_batch16/tiny/point_decoder.onnx` (float32) | 21,188,206 | `6f9a0ef2...` | as above |
| F3, F4a localization | People and sky head on SAM 2.1 tiny `image_embed`, P1 run 2, epoch 15 | `p1/colab_run2_coco_openimages/export/people_sky_head.onnx` and `people_sky_head.json` | 1,119,131 | `e02f2a1e...` | Own model; training data COCO 2017 and Open Images V7 (CC BY 4.0 annotations; photos with licenses that permit commercial use) |
| F1, F2 | **S3OD `s3od-synth`**, attention in 8 chained query chunks (`com.microsoft.MultiHeadAttention` per chunk), float16 with the encoder layer 0 attention in float32 | `s3od/onnx_mha_chunk8/float16/subject_masks.onnx` | 219,301,899 | `0f070b86...` | Weights and code MIT; the DINOv3 encoder under the DINOv3 License (the license text must go with the weights) |

Total download: 350,990,931 bytes (351 MB). The graphs stay under `build/tmp/explore/`; the
export commands are in the script READMEs of section 7.

S3OD replaces BiRefNet lite at 1024 px for F1 (Subject) and F2 (Background = 1 - Subject). This
closes open decision O2 and replaces U12 (the MVP Subject model).

## 2. Why S3OD

Issue #6 measured three models in place of BiRefNet lite, with the MVP settings on ONNX Runtime
DirectML ([Subject alternatives report](subject_alternatives.md)):

| Item | BiRefNet lite float16 | **S3OD 8 chunk MHA float16** | IS-Net general-use float16 | Universal matting float32 |
| --- | ---: | ---: | ---: | ---: |
| Session load | 9,442 ms | 1,111 ms | 834 ms | 1,513 ms |
| Warm p50 per call | 154 ms | 191 ms | 24 ms (photos p50 13.8 ms) | 116 ms |
| Dedicated GPU memory of the Subject session | 3,226 MB | 1,358 MB | 437 MB | 2,678 MB |
| MVP pipeline peak dedicated GPU memory (U6: 3 GB) | 4,528 MB | **2,666 MB** | 1,748 MB | 3,982 MB |
| Display driver resets | 8 in 14 sessions | **0 in 88 sessions** | 0 in 23 | 0 in 46 |
| Mean subject recall (14 photos) | 0.922 | **0.986** | 0.802 | 0.838 |
| Weights license | MIT | MIT (DINOv3 License for the encoder) | none stated | Apache License 2.0 |

- S3OD is the only candidate that meets U6, has no driver reset, and gives masks that are equal to
  or better than BiRefNet on the 14 quality photos (report sections 5.5, 6, 8).
- IS-Net general-use costs less, but its masks are worse on 7 of 14 photos, and its weights have
  no license statement (report section 11).
- Universal matting reads the photo at its full size: the MVP pipeline with it needs 3,982 MB, and
  its masks are worse on 5 of 14 photos (report section 12).

## 3. Acceptance against the requirements

Status: **met** (measured, the report names the run), **open** (a user check or a user decision),
**not met**, **not measured**.

### 3.1 User decisions

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| U2 | Licenses permit commercial use | **met** for the weights and code; **open** for the S3OD training data lineage | SAM 2.1 Apache-2.0; head: own model on COCO and Open Images V7 ([P1](p1_people_sky_head.md)); S3OD MIT and DINOv3 License. The S3OD training images come from FLUX.1 Krea [dev], and its labels come from a teacher decoder that was supervised on DIS5K, HRSOD, UHRSD, and DUTS ([report section 7](subject_alternatives.md#7-license)). The training data rule is the proposal U13 of issue #6; the user has not recorded it |
| U3 | No edge post-processing | **met** | The S3OD mask is the final mask. Its edge is almost binary (edge width 0.03 to 0.13 px on the 1024 grid; 4 px steps at the 4096 px photo); the user checks it by eye ([report section 6.2](subject_alternatives.md#62-results)) |
| U6 | Models in a 3 GB budget | **met** on Windows DirectML: 2,666 MB peak with all graphs loaded (406 MB margin) | [Report section 5.5](subject_alternatives.md#55-attention-in-query-chunks), record `mvp_pipeline_s3od_mha_chunk8_float16_cost*.json`. The unload rule of U6 (release the sessions when the user goes back to adjusting) is not in the MVP |
| U7 | No pointer-follow highlight | **met** | The MVP has the stroke, box, and click tools only ([MVP report](mvp_ai_mask.md)) |
| U8 | Float16 is acceptable | **met** | S3OD float16 against S3OD float32 on the CPU: minimum IoU 0.99824 of alpha > 0.5 on DirectML ([report section 5.5](subject_alternatives.md#55-attention-in-query-chunks)). SAM 2.1 tiny and the head stay float32 |
| — | No display driver reset (issue #6) | **met** | S3OD: 0 resets in 88 DirectML sessions, among them 9 first runs of the accepted float16 graph and 9 of its float32 form. SAM 2.1 tiny: no failure in E1, E3, E4, and the MVP |
| U1 | Interaction not worse than Lightroom and Capture One | **open** | The user checks the MVP by eye ([MVP report section 7](mvp_ai_mask.md#7-open-items)) |

### 3.2 Functions (README section 3)

| ID | Acceptance (README section 3) | Status | Evidence |
| --- | --- | --- | --- |
| F1 | No prompt; all salient subjects in one mask | **met** on the 14 photos | S3OD selects the same subjects as BiRefNet on 12 photos (IoU 0.86 to 0.995), the bassist only on `P1024439` (BiRefNet adds 75 % background), and the rock on `Nikon-D800e-raw-00007` (BiRefNet: empty); mean subject recall 0.986 ([report section 6.2](subject_alternatives.md#62-results)) |
| F1 | "Not found" message for a photo without a subject | **not measured** | The quality set has no photo without a subject. The largest S3OD IoU score (0.933 to 0.993 on the 14 photos) is a possible input of the rule ([report section 9](subject_alternatives.md#9-open-items)) |
| F2 | The inverted F1 mask; no extra model | **met** | `mask_pipeline.py`: Background = 1 - Subject alpha for every Subject graph |
| F3 | Sky masks without blocks or glow; "not found" message | **met** for the masks and the presence on 13 photos; the waterfall of `DSC00936` is found as sky (not a target, issue #5) | Sky masks IoU 0.982 to 0.993 against the E1 sky point ([P1 section 10.4](p1_people_sky_head.md#104-results-presence-and-sky-on-the-quality-photos)) |
| F4a | People list; one person can be selected | **met** on 5 of 7 referenced persons; **open**: the list threshold and order | The close-up `DSC00674` fails (box too short); the boarder of `2088064441` is below the list threshold by 0.0005 ([P1 section 10.5](p1_people_sky_head.md#105-results-people-on-the-quality-photos), [MVP report section 2](mvp_ai_mask.md#2-result-summary)) |
| F5 | Stroke and box; a click is a stroke with one point | **met** | E3: 23 of 25 targets without a level change ([E3](e3_stroke_selection.md)); the MVP gives the same decoder inputs as the E3 scripts in 149 of 149 calls |
| F6 | Add and subtract in one AI Mask | **met** | E3 sequences with `mask_input` and subtract steps, 149 of 149 calls equal in the MVP ([MVP report section 4](mvp_ai_mask.md#4-results-comparison-with-the-measurement-scripts)) |
| F7 | Combination across mask types | **open** (user decision O1) | Not in the MVP |
| F8 | Feather and Edge | **met** in the MVP | [MVP report section 3.5](mvp_ai_mask.md#35-functional-run-of-the-ui); user check open |
| F9 | Recipe and recompute | **open** (user decision O1) | Not in the MVP |
| F10 | Overlay color, opacity, toggle | **met** in the MVP | [MVP report section 3.5](mvp_ai_mask.md#35-functional-run-of-the-ui) |
| F11 | Text prompts | Proposed for removal (O3) | No model of the set takes text |

### 3.3 Numerical checks

| Check | Result | Status |
| --- | --- | --- |
| S3OD float32 ONNX against PyTorch, tolerance 1e-3 (issue #6) | 1.06e-3 on `P1024439`, 6.7e-4 or less on 13 photos; the same selected mask on 14 photos; minimum IoU at the photo size 0.999999 | **not met by 6e-5**. PyTorch differs from itself by 1.09e-3 and ONNX Runtime by 2.19e-3 ([report section 4.4](subject_alternatives.md#44-comparison-with-pytorch-on-the-cpu)). The user decides the tolerance |
| MVP Sky and People against the P1 scripts | 14 of 14 photos; 45 masks IoU >= 0.99993 | **met** |
| MVP strokes against the E3 scripts | 149 of 149 calls, `mask_input` difference 0.0 | **met** |
| S3OD DirectML float16 against S3OD float32 CPU | minimum IoU 0.99824 of alpha > 0.5 | **met** (no stated tolerance; DirectML float32: 0.99998) |

## 4. Cost of the accepted set

Windows, Intel Core i7-12700H, NVIDIA GeForce RTX 3080 Laptop GPU 8 GB, onnxruntime-directml
1.24.4, all graphs loaded in one process (`measure_pipeline_cost.py --subject-onnx <S3OD graph>`,
record `results/subject_alternatives/mvp_pipeline_s3od_mha_chunk8_float16_cost*.json`).

| Item | Value |
| --- | ---: |
| Peak dedicated / shared GPU memory | 2,666 MB / 459 MB |
| Peak working set | 1,675 MB |
| Session load: encoder, decoder, head, S3OD | 2,623 ms, 249 ms, 112 ms, 910 ms |
| Read and scale a 4096 px photo (p50) | 65.9 ms |
| SAM 2.1 tiny encoder, warm p50 (once per photo) | 70.5 ms |
| Head, warm p50 | 2.0 ms |
| Decoder, warm p50: box; 10 points with `mask_input` | 23.2 ms; 22.2 ms |
| Decoder per sky region / per person, p50 | 8.4 ms / 24.4 ms |
| S3OD, first call / warm p50 | 250 ms / 191.1 ms |

## 5. Rejected candidates of this phase

| Candidate | Reason | Evidence |
| --- | --- | --- |
| BiRefNet lite 1024 float16 (MVP Subject model until this decision) | 4,528 MB with the set (U6), 7.5 to 9.4 s session load, display driver resets in 8 of 14 sessions of issue #6 | [MVP report section 5.3](mvp_ai_mask.md#53-birefnet-memory-and-gpu-resets), [report section 12.4](subject_alternatives.md#124-cost-on-directml) |
| IS-Net general-use | Masks worse on 7 of 14 photos; no weights license | [Report section 11](subject_alternatives.md#11-is-net-general-use) |
| Alibaba universal matting | 3,982 MB with the set (U6); masks worse on 5 of 14 photos | [Report section 12](subject_alternatives.md#12-alibaba-universal-matting) |
| EfficientSAM3 and the SAM3 detector | Grid holes, close-up failures; the detector is too large for People and Sky (U11) | README section 1, [SM1 record](../background/sm1_efficientsam3_record.md) |

## 6. Open items

1. **macOS Core ML verification** of the accepted set (master plan section 8.16.4: "record macOS
   Core ML results when a candidate is selected"). No model of the set is measured on macOS. The
   SM1 record has Core ML defects of the earlier models (wrong encoder output, NaN detector output,
   a decoder load that stopped the process). The S3OD graph uses `com.microsoft.MultiHeadAttention`,
   a contrib operator of ONNX Runtime. Issue
   [#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8).
2. **The user checks by eye** the MVP interaction (U1) and the hard S3OD edge on hair and fur.
3. **Decisions:** the S3OD float32 tolerance (section 3.3), the training data rule U13 (section
   3.1), F7 and F9 (O1), the text prompts (O3).
4. **Rules that are not done:** the Subject "not found" rule (F1), the People list threshold and
   order (F4a), the unload rule of U6.
5. **The MVP default Subject graph** is still BiRefNet lite (`mask_pipeline.GraphPaths.default`);
   S3OD runs with `--subject-onnx`. The UI labels name BiRefNet.

## 7. Evidence index

| Item | Location |
| --- | --- |
| Reports | [E1](e1_sam2_1_point_box_evaluation.md), [E3](e3_stroke_selection.md), [P0](p0_sam_feature_localization.md), [P1](p1_people_sky_head.md), [MVP](mvp_ai_mask.md), [Subject alternatives](subject_alternatives.md) |
| Export and measurement scripts | [`scripts/ai_models/sam2/`](../../scripts/ai_models/sam2/README.md), [`scripts/poc/people_sky_head/`](../../scripts/poc/people_sky_head/README.md), [`scripts/ai_models/s3od/`](../../scripts/ai_models/s3od/README.md), [`scripts/poc/ai_mask_mvp/`](../../scripts/poc/ai_mask_mvp/README.md) |
| Results | [`results/`](../../results/README.md): `p1_people_sky_head/`, `ai_mask_mvp/`, `subject_alternatives/` |
| Local evidence (not committed) | `build/tmp/explore/` (graphs, weights, photos, alphas), `docs/exploration/images/` (sheets) |
