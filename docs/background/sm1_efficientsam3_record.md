# SM1 completion record: EfficientSAM3 export and runtime

Source: section 8.15 of the [AI Segmentation Mask master plan](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md) in Alcedo Studio, commit `d1e0d8c7` on `feature/ai-segmentation-sm1-quality-review`, copied on 2026-10-20. Section numbers in the text (for example "section 8.6") refer to that plan. The step reports E1 to E5 are in [`../exploration/`](../exploration/README.md).

**Phase / date / status:** SM1, 2026-10-19. Export and runtime proof of concept done. The user
review (#449) and the user decisions (#454) are open. Issue: [AlcedoStudio#448](https://github.com/zidage/AlcedoStudio/issues/448).

**Source revision and branch:** `3c378d9c` on `feature/ai-segmentation-sm1-runtime-poc`.

**EfficientSAM3 repository commit:** `bd0936c788fed8d51fa799437f05abd97b401b06`.

**Checkpoints inspected** (`inspect_checkpoint.py`; every file loads with 0 missing and 0
unexpected keys, without a `facebook/sam3` checkpoint):

| File | Parameters | Point and box decoder | Text encoder |
| --- | ---: | --- | --- |
| `stage1_all_converted/efficient_sam3_efficientvit_m_geo_ft.pt` | 428.2M | yes (4.2M) | SAM3 (353.7M) |
| `stage1_all_converted/efficient_sam3_efficientvit_m.pt` | 428.2M | yes | SAM3 |
| `stage1_all_converted/efficient_sam3_repvit_m_geo_ft.pt` | not inspected separately; exported | yes | SAM3 |
| `stage1_all_converted/efficient_sam3_tinyvit_m_geo_ft.pt` | not inspected separately; exported | yes | SAM3 |
| `efficientsam3_ft/efficientsam3_efficientvit.pt` | 97.5M | no | MobileCLIP-S0 (42.5M) |
| `efficientsam3_ft/efficientsam3_repvit.pt` | 100.9M | no | MobileCLIP-S0 |
| `efficientsam3_ft/efficientsam3_tinyvit.pt` | 103.6M | no | MobileCLIP-S0 |

Tensor group origin: only the image encoder trunk is an EfficientSAM3 student (14.4M for
EfficientViT-M). The `geo_ft` and the plain stage 1 file differ only in the image encoder. The
detector transformer, geometry encoder, scoring, and segmentation head are byte-identical in the
stage 1 and stage 3 files. The stage 3 (`efficientsam3_ft`) files have no point and box decoder, so
point and box prompts need a stage 1 file. This confirms the inference in section 5.6: the full
checkpoints contain SAM3 components.

**Actual changed modules:**

- `scripts/ai_models/efficientsam3/`: `esam3_graphs.py` (shared graph wrappers, not in the
  section 8.3 list), `inspect_checkpoint.py`, `export_onnx.py`, `compare_onnx_outputs.py`,
  `refine_edges_prototype.py`, `requirements.txt`, `README.md`.
- [`rust/puerh_mind/src/service/ort_runtime.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/src/service/ort_runtime.rs): items are `pub`; new `session_builder` returns the
  production builder and `load_session` commits it. No behavior change.
- [`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs): `runtime` and `ping` modes.

**Implemented behavior:** four ONNX graphs (`point_encoder`, `point_decoder`, `text_encoder`,
`detector_grounding`; tensor layout in the README), opset 17. The decoder returns the four mask
tokens; the host selects the masks as `MaskDecoder.forward` does. The Rust program reproduces the
model preprocessing, the prompt mapping, the mask selection, the bilinear upsampling, and the CLIP
BPE tokens, and compares its masks with the PyTorch masks.

**Explicitly unimplemented items:** user review of the quality set (#449); Ping with the C++ client, GPU memory, 40 MP photo (#453); a correct Core ML
configuration (#450); text path with one shared image embedding (#452).

**Primary success call chain:**

```text
export_onnx.py -> PointEncoder / PointDecoder / TextEncoder / DetectorGrounding -> torch.onnx.export
compare_onnx_outputs.py -> Sam3Processor.set_image + predict_inst | set_text_prompt (PyTorch)
                        -> onnxruntime CPU -> IoU and max |logit difference| -> reference_masks/*.png
segmentation_runtime_report runtime -> ort_runtime::load_session (production settings)
  -> preprocess (stretch to 1008, normalize) -> point_encoder -> point_decoder (prompts x 1008)
  -> select_masks -> upsample_to_mask -> compare_with_reference -> write_overlay
  -> tokenize -> text_encoder -> detector_grounding -> score > 0.5 -> masks
  -> profile_graphs (production settings + with_profiling) -> node count per provider -> JSON
```

**Primary failure and restore call chain:** a graph load failure is recorded in
`SegmentationGraphsLoadWithProductionSessionSettings` with the ONNX Runtime error, and the run
continues with the other graphs. A failed profiling pass is recorded per graph. SM1 changes no
product state.

**Build and test commands with exit codes:**

- `cargo_msvc.cmd build --release --example segmentation_runtime_report`: 0, no warnings.
- `cargo_msvc.cmd clippy --release --example segmentation_runtime_report`: 0; no warning in the
  changed files (existing warnings in other files).
- `cargo_msvc.cmd test -- --skip live_`: 0.
- macOS `cargo build --release --example segmentation_runtime_report`: 0.

**Discovered / passed / failed / skipped counts:** `cargo test`: 293 passed, 0 failed, 8 filtered
(`live_`). `OnnxMaskLogitsMatchPytorchWithinTolerance`: EfficientViT-M 33/33 (28 point and box,
5 text prompts), RepViT-M 28/28, TinyViT-M 28/28; minimum IoU 0.99994, maximum absolute logit
difference 0.0002. `DecoderOutputShapeMatchesExportedSpecification` and
`HostTokenizerMatchesPythonBpeTokens` (5/5 prompts) passed on every run.
`SegmentationGraphsLoadWithProductionSessionSettings` passed for all graphs on `cpu`, `directml`,
and `coreml`. Rust pipeline masks compared with the PyTorch masks (different resize code): median
IoU 0.997 to 0.998; 35 of 39 at 0.99 or more for EfficientViT-M (minimum 0.79, caused by a
different single-mask token selection on `child_center`), 28 of 28 for RepViT-M and TinyViT-M.

**Machines:**

- Windows: Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce RTX 3080 Laptop GPU 8 GB, driver
  610.62, Windows 11 build 22635.
- macOS: Apple M4, 16 GB, macOS 27.0.1.
- ONNX Runtime 1.24 through `ort` 2.0.0-rc.12. PyTorch 2.14.1 (CPU), onnxruntime 1.31.0 (Python).

**Latency and memory results** (production settings unless stated; milliseconds, warm p50 / p95;
EfficientViT-M unless stated):

| Graph | Windows CPU | Windows DirectML | macOS CPU | macOS Core ML |
| --- | --- | --- | --- | --- |
| `point_encoder` | 739 / 768 | 47.7 / 92.5 | 600 / 624 | 372 / 382, wrong output |
| `point_decoder` | 38.8 / 44.3 | 12.6 / 14.0 | 18.2 / 18.3 | 74.2 / 78.6 |
| `text_encoder` | 3.8 / 4.3 | 1.0 / 1.2 | 1.9 / 1.9 | not measured (run stopped) |
| `detector_grounding` | 2667 / 2861 | 254.5 / 263.7 | 2383 / 2487 | about 11 900, NaN scores |

- Session load on DirectML: encoder 2.8 s, decoder 0.4 s, detector 2.4 s. Core ML detector load
  22.3 s.
- RepViT-M and TinyViT-M encoder on DirectML: 48.0 / 51.1 and 60.8 / 63.4. CPU: 748 and 978.
- Core ML MLProgram format (not production): encoder 322 / 326 with correct output; the decoder
  load stops the process (#450).
- Memory after inference: Windows DirectML working set 1.1 GB, private bytes 4.8 GB; Windows CPU
  working set 0.6 GB, peak 1.9 GB; macOS CPU peak 5.1 GB. GPU memory is unavailable (#453).

**Operator placement results:** DirectML runs `point_encoder`, `text_encoder`, and
`detector_grounding` completely (fused DirectML nodes, 0 CPU nodes). `point_decoder` places 121
nodes on the CPU provider (Gather 37, Unsqueeze 22, Sqrt 14, Concat 11, Slice 10, Cast 7, Div 7,
Mul 7, Reshape 3, Equal 1, OneHot 1, Where 1), 6.4 ms of kernel time in two runs. Core ML
(production): `point_encoder` 72 Core ML and 50 CPU nodes, `point_decoder` 56 and 220,
`text_encoder` 30 and 42. The Core ML detector profile did not finish in 30 minutes.

**Quality review (user result per case):** user result not recorded yet (#449). The quality set has
14 user photos in `build/tmp/sm1/quality/` (prompts in `quality/prompts.json`, 48 point and box
prompts, 28 text prompts). No panorama (case 9): the user removed it from the set on 2026-10-19.
Case 7 uses `DSC00674.jpg` (ISO 1250, visible grain at 100 percent) as a weak low-light substitute.
Two extra hard photos: `P1024439.jpg` (backlit hair, microphone stand and cable in front of the
subject) and `Nikon-D800e-raw-00007.jpg` (lupine field, mountain horizon).

| Case | Photo | Developer observation (EfficientViT-M; RepViT-M and TinyViT-M are similar unless stated) |
| --- | --- | --- |
| 1 | `DSC_1535.jpg` | Box and three points: whole person. One torso point: suit only. Text `person`: the subject and one pedestrian; `hair` correct. |
| 2 | `DSC00326.jpg` | Box: coarse edge and holes in the legs. The rectangle pass gives a clean body with legs. Text `person` and `hair` correct. |
| 3 | `DSC_1188.jpg` | Box and two points: whole squirrel. One body point misses the tail. Text `squirrel`: no detection (top score 0.03). |
| 4 | `DSC00527.jpg` | Sky point: whole sky, the tree horizon is coarse. Tree point: one tree. Text `sky` correct; `tree` no detection; `person` finds 1 of about 30 people. |
| 5 | `2088064441.jpg` | Box: person and board as one blob. Rectangle IoU (box) 0.661 EfficientViT-M, 0.663 RepViT-M, 0.635 TinyViT-M; the rectangle pass is clearly better. |
| 6 | `DSC00728.jpg` | Box: balloon with a wrong edge. Rectangle IoU (box) 0.800 EfficientViT-M, 0.791 RepViT-M, 0.826 TinyViT-M; the rectangle pass gives the correct outline. Sky point: sky with a gap above the mountains and a false patch on the cliff with grid artifacts. |
| 7 | `DSC00674.jpg` | Head-and-shoulders close-up. Box and points fail in all three variants: the face is left out and the mask has grid holes. Letterbox input fixes the box prompt. Text `person` and `hair` correct. |
| 8 | `5761824892.jpg` (45.6 MP) | Box: whole car. Kayak point correct. Text `car` correct; `kayak` and `tree` no detection. |
| 10 | `DSC00936.jpg` | Box and points: whole person; the edge is about 20 px outside the body, and the rectangle pass and the matte correct it. |
| 11 | `DSC00562.jpg` | One duck correct. Three duck points: ducks and the hedge behind them. Text `duck`: no detection; `building`: 10 buildings, correct. |
| 12 | `DSC00602.jpg` (42.3 MP) | Box: tram, ragged bottom edge near the shadow. Text `tram`: no detection (top score 0.23). |
| extra | `DSC_1456.jpg` | Car box and point correct. Text `car` correct; `person` finds 1 of many pedestrians. |
| extra | `P1024439.jpg` | Box and three points: bassist with the bass. Text `person`: both musicians. The rectangle pass loses part of the right arm. |
| extra | `Nikon-D800e-raw-00007.jpg` | Sky point: empty mask (EfficientViT-M), sky with grid artifacts in the clouds (RepViT-M), correct (TinyViT-M). Rock point: the whole lupine field. Text `sky` and `mountain` correct; `flower` no detection. |

Findings across the set:

- The point and box decoder often shows a regular grid of holes and islands (the 72 x 72 embedding
  grid) on large or low-contrast regions, in all three variants and on CPU and DirectML.
- Encoder preprocessing (`compare_preprocessing.py`): letterbox input (aspect ratio kept, gray
  padding) fixes the close-up portrait and the Nikon sky, but makes some single-point prompts worse
  (`DSC_1535` torso, car points). "Dual" (both encoder passes, the mask with the higher decoder IoU
  score) picks the better input in most prompts, but not for the close-up hair point.
- A second decoder pass with the selected logits as `mask_input` makes single-point masks smaller
  or wrong on this model. It is not recommended.
- The text path is the most reliable path for people, hair, sky, buildings, and mountains. It
  misses animals, ducks, flowers, trees, trams, kayaks, and most people in crowds at the 0.5 score
  threshold.

**Edge refinement review (user result):** user result not recorded yet (#449).
`refine_edges_prototype.py` compares (a) bilinear, (b) the gray guided filter (radius 8, eps 1e-3),
(c) a rectangle pass, and (d) a projection matte:

- (b) changes 0.00 to 1.04 percent of the mask pixels on the quality set; the change is not visible
  on hair or fur.
- (c) encodes a square of the source photo around the subject when the square covers less than half
  of the photo. It gives the largest quality gain for small and medium subjects (cases 2, 5, 6, 10).
  It does not change D7: the whole photo is still the first input.
- (d) projects each pixel color in a band of one logit cell around the (c) edge onto the line
  between the local mean colors of the sure foreground and the sure background, then smooths the
  result with a small color guided filter. It keeps single hairs and fur against smooth
  backgrounds (cases 3, 7, `P1024439`). It adds speckle on textured backgrounds such as asphalt
  (case 1, shoes). The Python prototype takes 0.04 to 12 s per mask on the CPU at 4096 pixels.

**Ping results** (Rust tonic client, 1000 calls, Windows loopback): persistent channel p50 0.20,
p95 0.32, maximum 0.60 ms; new channel per call p50 0.52, p95 13.4, maximum 26.6 ms.

**Publish list** (`point_decoder.onnx` is byte-identical for the three variants):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| EfficientViT-M `point_encoder.onnx` | 85,471,721 | `f6f6f60b0794656a6524322389c6fa828b4c0cc998e566bf46a34aac2f99ccd1` |
| RepViT-M `point_encoder.onnx` | 98,952,875 | `fd47ab621081a29ee742cfaff287b7a6a75945ba6e4178e905be65ce12bcc90b` |
| TinyViT-M `point_encoder.onnx` | 117,556,605 | `ef28ac407b50f7ab45b1cff591c27dfcafbb8037becfcf71a568dbcf196b156f` |
| `point_decoder.onnx` | 22,302,046 | `201e25c652b342625fa6a2f7f48f744843f0edeb61217bcb316b3b2965abafc0` |
| EfficientViT `text_encoder.onnx` | 169,171,248 | `24325d3cf107ef585ba6d29d74e643355962bf2caf5d0e9c67ca7032318137a9` |
| EfficientViT `detector_grounding.onnx` | 201,571,788 | `d37a94b23eb5dbf7f8fd6daf47a833a8590b949e0e8ea06ad1d69910c14ab582` |
| SAM License (`sam3/LICENSE`) | 7,353 | `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be` |
| Apache-2.0 text | missing upstream | [AlcedoStudio#454](https://github.com/zidage/AlcedoStudio/issues/454) |

**Recommendations and user decisions:** recommendations are in [AlcedoStudio#454](https://github.com/zidage/AlcedoStudio/issues/454). No user decision yet.

**Evidence path:** `build/tmp/sm1/` on the Windows machine: `inspect/`, `onnx/`, `compare*/`,
`runtime/`, `profiles*/`, `overlays/`, `refine/`, `variant_contact_sheet.png`, and `quality/`
(photos, prompts, runtime reports, overlays and contact sheets per variant, `preprocess/`, `refine/`,
`review.html`).

**Remaining defects or unavailable platforms:**

- Core ML gives wrong encoder output and NaN detector output with the production settings (#450).
- `ort_runtime` runs a session on the CPU provider without an error when the GPU provider fails to
  register (#451).
- The SM1 diff is about 2,550 lines (Rust example 1,406 after `rustfmt`, Python and README about
  1,100), more than the 900-1500 estimate of section 8.14.
