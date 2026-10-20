# Grid holes and model re-evaluation

Source: section 8.16 of the [AI Segmentation Mask master plan](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md) in Alcedo Studio, commit `d1e0d8c7` on `feature/ai-segmentation-sm1-quality-review`, copied on 2026-10-20. Section numbers in the text (for example "section 8.6") refer to that plan. The step reports E1 to E5 are in [`../exploration/`](../exploration/README.md).

**Date and status:** 2026-10-19. The user stopped the EfficientSAM3 follow-up work after the quality
review. Closed as not planned: #449, #450, #452, #453, #454. #451 stays open because it also affects
the CLIP encoders. The exploration of section 8.16.4 is tracked in #458. The user excluded the
SAM3 teacher image encoder from the exploration; SAM 2.1 and the task mask models are tested.

## 8.16.1 Cause of the grid holes

The holes are also in the PyTorch output (the ONNX graphs match PyTorch, section 8.15). They come
from the student image encoder:

| Student trunk | Native output at a 1008 px input |
| --- | --- |
| EfficientViT-B1 | 256 x 32 x 32 |
| RepViT-M1.1 | 512 x 32 x 32 |
| TinyViT-11M | 1024 tokens (32 x 32) |

- The SAM3 teacher ViT (patch 14) gives a native 72 x 72 grid. `ImageStudentEncoder`
  (`sam3/model_builder.py`) resizes the 32 x 32 student output to 72 x 72 with bilinear
  interpolation. One student cell covers about 31.5 input pixels.
- Stage 1 distillation uses only MSE and cosine loss on the final 72 x 72 embedding
  (`stage1/configs/base_stage1.yaml`). The point and box decoder is never trained on student
  features.
- On difficult prompts the decoder logits stay near 0. The neck and the decoder upscale with
  transposed convolutions (kernel 2, stride 2, no overlap), which add a fixed pattern per cell. Near
  0, this pattern decides the sign of each pixel and makes the regular dots and holes.

Measurement (EfficientViT-M, PyTorch, 288 x 288 logits; script `build/tmp/sm1/probe/grid_probe.py`,
not committed):

| Prompt | Share of \|logit\| < 2 | Share of logit > 4 | Grid energy |
| --- | ---: | ---: | ---: |
| `DSC00674` box (close-up, fails) | 0.826 | 0.000 | 0.288 |
| `DSC00728` sky point (grid on the cliff) | 0.213 | 0.105 | 0.374 |
| `DSC00326` box (correct) | 0.012 | 0.019 | 0.031 |
| `DSC_1535` box (correct) | 0.049 | 0.089 | 0.022 |

Grid energy is the share of the high-pass logit spectrum on the lines of period 4 and 9 cells.

Mitigations that do not correct the mask:

- Four encoder passes with half-cell shifts, logits aligned and averaged: grid energy 0.29 to 0.16
  on `DSC00674`, but the hole and island area stays the same.
- A box filter on `high_res_feats_0` and `high_res_feats_1`: no improvement.
- Zero high-resolution features: the grid stays (grid energy 0.46), so it comes from the image
  embedding.
- A Gaussian blur of the logits (1.5 cells): no dots, but the mask is still wrong (the face is left
  out).
- A 2304 px input (native 72 x 72 student output): the face is included, but new holes appear.

Result: post-processing cannot correct the EfficientSAM3 point and box masks. The model is uncertain
on these regions.

## 8.16.2 Prompt dependence

- One point is ambiguous (`DSC_1535` torso: suit only; `DSC_1188`: squirrel without the tail).
- The text path misses squirrels, ducks, flowers, trees, trams, kayaks, and most people in crowds at
  the 0.5 score threshold.
- The user must know which prompt type works for which subject. This is not acceptable for the 0.3.4
  main feature.

## 8.16.3 Practice in other photo applications

This review does not use RapidRAW or similar new raw editors.

| Application | Main entry | Other objects | Edge stage |
| --- | --- | --- | --- |
| Lightroom | One click: Subject, Sky, Background, People with body parts (skin, hair, eyes, lips, teeth, clothes), Landscape | Objects (brush or box) | Feather and Edge controls |
| Capture One | One click: Subject, Background (inverted Subject) | AI Select | not documented |
| Photoshop | Select Subject (Device or Cloud quality) | Object Selection tool | Select and Mask |
| Apple Photos (Vision) | Subject lift: all foreground instances in one pass; a tap selects an instance (`VNInstanceMaskObservation`, `instanceAtPoint`) | not applicable | 512 x 512 network, then guided-filter upsampling to full resolution on the GPU; a confidence branch hides unreliable results |
| DaVinci Resolve | Magic Mask with Person and Object modes, Faster and Better quality | add and subtract strokes | consistency controls |

Common pattern:

- Task masks with dedicated models are the main entry. The user selects a task, not a prompt type.
- A click or box tool is the second entry. Candidates are computed before the click, so a click
  selects a candidate instead of asking the model to resolve an ambiguous point.
- The coarse mask and the edge quality are separate stages.
- A confidence check hides an unreliable mask instead of showing it.

## 8.16.4 Exploration steps (#458)

Evidence rules: use the quality set and prompts of section 8.6 (`build/tmp/sm1/quality/`). Keep
weights, ONNX files, and reports under `build/tmp/explore/`; do not commit them. Run on the Windows
DirectML machine first; record macOS Core ML results when a candidate is selected. Record the
license of each model; a license that does not permit commercial use excludes the model.

| Step | Work | Question that it answers |
| --- | --- | --- |
| E1 | Run the point and box prompts of section 8.6 with SAM 2.1 (Hiera tiny, small, base plus, and large; 1024 px input). SAM 2.1 takes its high-resolution decoder features from the Hiera stages at stride 4 and 8, not from transposed convolutions of one low-resolution map. Record the probe metrics of section 8.16.1, the encoder and decoder latency and memory on DirectML, the file size, and the license. | Does an encoder with native multi-scale features remove the grid holes and the close-up failure? Which size is affordable? |
| E2 | Run dichotomous image segmentation and matting models on the 14 photos: BiRefNet lite, BiRefNet 1024, and the BiRefNet high-resolution and matting variants (MIT license). Measure latency, memory, file size, and the soft matte on hair and fur. Record photos with more than one subject (crowd, ducks). | Can one model supply the one-click Subject mask and the hair edge without a separate matte step? |
| E3 | Stroke selection, like the Quick Selection tool of Photoshop: the user drags a short stroke inside the region to select. The host places 4 to 5 sample points along the stroke as positive points of the SAM 2.1 point decoder; a stroke with a modifier key gives negative points. Measure the selection quality against the E1 reference masks compared with one click (E4), the point count and spacing, add and subtract strokes, and the decoder time per stroke. Replaces the Sky and People body-part step: the Sky and People task masks are not done for now (user decision, 2026-10-20). | Does a short stroke inside the region select the intended mask without a level change? |
| E4 | Compute instance candidates when the tool opens (the SAM 2.1 automatic mask generator with the E1 size, or a class-agnostic instance model). A click selects a candidate; the pointer position highlights it. Measure the candidate time and the single-point cases of section 8.16.2. | Does candidate selection remove the single-point ambiguity at an acceptable time? |
| E5 | Only if E2 is not sufficient: compare guided-filter upsampling of a soft matte (the Apple method) with the SM1 projection matte (section 8.15, method (d)). | Which edge stage is robust on textured backgrounds? |
| E6 | Write one table per task (Subject, object click, stroke selection): model, latency, file size, license, quality, recommendation. | The user decides the revision of D2 and the interaction set. SM2 and later phases are replanned after that decision. |

Reports of the steps are in [`ai_mask_candidate/`](../exploration/README.md).

- E1 (2026-10-19, [report](../exploration/e1_sam2_1_point_box_evaluation.md)): SAM 2.1
  removes the grid holes in all four sizes (largest grid energy 0.04 to 0.11, EfficientSAM3 0.54)
  and gives the whole person on the `DSC00674` close-up box. One point still selects a part (E4).
  DirectML encoder warm p50: tiny 76 ms, small 87 ms, base plus 161 ms, large 322 ms; decoder
  about 10.5 ms. Peak dedicated GPU memory 1.36 to 3.10 GB. Apache-2.0. Recommendation: tiny,
  with base plus as the quality alternative. The user review is open.
- E2 (2026-10-19, [report](../exploration/e2_birefnet_subject_matting.md)): BiRefNet lite
  finds the expected subject on 12 of 13 photos without a prompt (not the ducks; all salient
  subjects in one mask) with a clean but hard hair edge. The matting variants give soft strands
  but miss rigid objects (car, tram, balloon). The tutorial ONNX export runs out of memory on
  DirectML; a GridSample form of `deform_conv2d` runs. DirectML warm p50: lite 218 ms (float16
  175 ms, 3.2 GB), Swin-L 3 to 15 s (float16 351 ms), HR fails on 8 GB. MIT. User assessment: the
  speed is not usable. User decision: only BiRefNet lite stays a candidate; the other five
  variants are dropped. E5 and the Subject task of E6 stay open.
- E4 (2026-10-19, [report](../exploration/e4_click_candidates.md)): measured on the photos
  scaled to the 4096 px application input. The live point decoder at the pointer (three size
  levels, default: best predicted IoU) reaches the intended mask more often than candidates
  precomputed with the `SAM2AutomaticMaskGenerator` point grid of the same model. Base plus live:
  default is the target on 20 of 25 single-point prompts, the rest one or two level steps away,
  0 not reachable (tiny: 16, 6, 3 not reachable). The candidates need 0.8 to 3.2 s after the
  encoder and 1.3 GB more GPU memory (a batch of 64 prompts: 5.2 GB, and the encoder becomes
  17 times slower), and reach 18 (base plus) or 17 (tiny) targets by default with 2 or 4 not
  reachable. Point decoder at the pointer: 12 ms; keeping the highlight while the pointer is
  inside it gives a same-highlight share of 0.96 to 0.98 for nearby positions. Recommendation:
  live point decoder with that hover rule, a level change by key or mouse wheel, base plus (tiny
  as the lower-memory alternative). User decision (2026-10-19): float16 is acceptable (lower
  model precision for a shorter processing time); the float16 forms of the encoder and the point
  decoder are not measured yet. User assessment (2026-10-20): a selection that follows the
  pointer movement is probably not a good interaction; step E3 measures stroke selection
  instead.
- E3 (2026-10-20, [report](../exploration/e3_stroke_selection.md)): synthetic straight
  strokes inside the E1 reference masks, SAM 2.1 tiny and base plus. A short stroke selects the
  region that it covers, like a click: half of the region gives 18 of 25 targets with base plus
  (one click: 20) and 17 with tiny (16). A stroke across the whole region along its long axis
  gives 23 of 25 without a level change (base plus 25 of 25 with 2 or 4 points). 3 or 4 points
  per stroke are best; 8 points and dense points on small parts are worse. Add and subtract
  strokes work with all points in one decoder run and the previous mask as `mask_input`: 0 of 58
  corrections lowered the IoU (base plus); from the largest mask of a click, one subtract stroke
  reaches the part on 7 of 14 part prompts. On DirectML, a decoder session runs only the point
  count of its first run at 10.5 ms; other counts take 22 to 34 ms. Padding to a fixed point
  count breaks the masks (also in PyTorch); one session per point count gives 12 ms at about
  90 MB each. `StrokePromptLogitsMatchPytorchWithinTolerance` passed (tiny 390, base plus 324
  decoder calls). Recommendation: the click stays the main entry; a stroke tool decodes once per
  stroke with 3 to 4 points and the previous mask, base plus. The user review is open.
- E5 (2026-10-20, [report](../exploration/e5_edge_stage.md)): E2 is not sufficient (hard hair
  edge, speed not usable), so E5 ran on the six people of the quality set with SAM 2.1 tiny (user
  decision: tiny, no model matrix) and the E3 stroke selection. The resolution of the coarse
  logits limits the edge: at the 16 px logit cell of the photo pass (256 x 256 logits at the
  4096 px input), guided-filter upsampling makes a glow of 30 to 45 px and the SM1 projection
  matte speckles on asphalt, rock, and bricks (561 specks in 23 regions). A second encoder pass
  at 4 px per cell fixes both: a rectangle pass helps only subjects below half of the photo; a
  new tile pass (1024 px tiles along the edge, the photo logits as `mask_input`, tile logits only
  within 2 photo cells of the edge) works for every size. On the tile pass, guided-filter
  upsampling of the smoothed mask is the robust stage (6 specks, 9 to 10 px edge); the projection
  matte is sharper (4.4 px) with 116 specks. Raw `sigmoid(logits)` is not a matte. Fine fly-away
  strands are not captured by any variant. Tile cost: 2 to 19 encoder runs, about 0.15 to 1.5 s
  on DirectML (estimate). Recommendation: tile pass and guided filter at commit. The user review
  is open.
