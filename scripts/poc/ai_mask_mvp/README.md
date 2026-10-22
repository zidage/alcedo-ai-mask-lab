# AI Mask MVP: the selected model set with a desktop UI (issue #5)

Issue [#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5). The MVP runs the model set of
[README section 4](../../../README.md#4-model-selection-proposal) end to end on one photo, as the
product will use it: one-click Subject, Background, Sky, and People, and the stroke, click, and
box selection with add and subtract. The report is
[MVP: AI Mask with the selected model set](../../../docs/exploration/mvp_ai_mask.md).

Weights, ONNX files, photos, logs, and exported masks are not committed. Keep them under
`build/tmp/`.

## Files

| File | Purpose |
| --- | --- |
| `mask_pipeline.py` | Pipeline without UI: graph loading on one provider, photo reading and scaling, encoder and head once per photo, one function per selection (Sky, People, stroke, click, box, Subject), Feather and Edge, mask and JSON export, the timing record. |
| `ai_mask_app.py` | PySide6 UI: canvas with the photo and the overlay, one-click buttons, People list, stroke and box tools with Alt for subtract, level change of a click, Feather and Edge, overlay color and opacity, export, stage times. |
| `compare_with_measurement_scripts.py` | The comparison of the pipeline with `evaluate_quality_photos.py` (run 2), `compare_stroke_outputs.py` (E3 sequences), and the E2 BiRefNet CPU output. |
| `measure_pipeline_cost.py` | Stage times and memory with all models loaded together (run it under `run_with_gpu_memory.ps1`). `--subject-onnx` selects another Subject graph, for example S3OD, IS-Net general-use, or Alibaba universal matting (issue #6). |
| `measure_birefnet_memory_phases.py` | GPU memory of the BiRefNet DirectML session per phase (load, runs, the arena shrinkage run option, release, load again). |
| `measure_birefnet_load_time.py` | BiRefNet session creation time per graph optimization level, and with a saved optimized graph (no inference). |
| `requirements.txt` | Packages of the MVP environment (onnxruntime-directml, PySide6). |

## Models

The graphs are the ONNX files of the earlier steps under `build/tmp/explore/`
(`mask_pipeline.GraphPaths.default`). The pipeline records the SHA-256 of each graph that it loads.

| Function | Graph | Bytes | SHA-256 | Source and license |
| --- | --- | ---: | --- | --- |
| F3, F4a, F5, F6 | `e4/onnx_float32_batch16/tiny/point_encoder.onnx` (SAM 2.1 tiny encoder, float32) | 109,381,695 | `17a554b420d3c51a109567f784ff3792b354c78dc8c2c09c585403329b412d61` | `facebookresearch/sam2` at `2b90b9f5`, Apache-2.0 ([E1](../../../docs/exploration/e1_sam2_1_point_box_evaluation.md), [E4](../../../docs/exploration/e4_click_candidates.md)) |
| F3, F4a, F5, F6 | `e4/onnx_float32_batch16/tiny/point_decoder.onnx` (SAM 2.1 tiny prompt encoder and mask decoder) | 21,188,206 | `6f9a0ef2c217a0abccc8b0c01e631c42c6a78a2038a7944cc7fcb4f536439278` | as above |
| F3, F4a | `p1/colab_run2_coco_openimages/export/people_sky_head.onnx` (run 2, epoch 15) and `people_sky_head.json` (thresholds) | 1,119,131 | `e02f2a1eaace94e0efe2b3c8db0a0391be210b477a95e49063e80758f12abf35` | Own model ([P1](../../../docs/exploration/p1_people_sky_head.md)); training data COCO 2017 (CC BY 4.0 annotations, photos with a license that permits commercial use) and Open Images V7 (CC BY 4.0 annotations, CC BY 2.0 photos) |
| F1, F2 | `e2/onnx_fp16/lite/subject_matte.onnx` (BiRefNet lite, 1024 x 1024, float16 inside, float32 input and output) | 96,437,277 | `b27fc5c46a703a5de0ab17a43f212d7b0579a58421c4139b8f6c7f36be91e7cf` | `ZhengPeng7/BiRefNet_lite` at `aa62cd87`, MIT ([E2](../../../docs/exploration/e2_birefnet_subject_matting.md)) |

`--subject-onnx` selects another BiRefNet lite 1024 graph, for example the float32 graph
`e2/onnx/lite/subject_matte.onnx` (183,439,077 bytes, SHA-256 `dcb89537...`), or an S3OD graph of
issue #6, for example `s3od/onnx_mha_chunk8/float16/subject_masks.onnx` (219,301,899 bytes,
SHA-256 `0f070b86...`, [Subject alternatives report, section 5.5](../../../docs/exploration/subject_alternatives.md#55-attention-in-query-chunks)),
or the IS-Net general-use graph `isnet/onnx/float16/subject_d1.onnx` (88,155,011 bytes, SHA-256
`31cf8b2d...`, [report, section 11](../../../docs/exploration/subject_alternatives.md#11-is-net-general-use)),
or the Alibaba universal matting graph `universal_matting/onnx/float32/universal_matting.onnx`
(173,323,991 bytes, SHA-256 `eda5d60e...`, [report, section 12](../../../docs/exploration/subject_alternatives.md#12-alibaba-universal-matting)).
The pipeline selects the Subject processing by the graph outputs: `alpha` is BiRefNet, `masks` and
`iou_scores` are S3OD (letterbox, mask with the largest IoU score, unpadding, and the resize of the
source), `d1` is IS-Net (stretch to 1024 x 1024 with truncation to uint8, the scaling of d1 with
its minimum and maximum, and the bilinear resize), `rgba` is universal matting (the scaled
photo at its size as the input, alpha = channel 3 / 255 at the photo size). The pipeline
refuses a BiRefNet, S3OD, or IS-Net graph with an input size other than 1024.

UI toolkit: PySide6 6.12 (LGPL-3.0). ONNX Runtime: onnxruntime-directml 1.24.4 (MIT).

## Data flow and rules

```text
Open a photo -> long edge at most 4096 px (make_input_set.scaled_size, Lanczos; EXIF orientation)
  -> SAM 2.1 tiny encoder (once) -> image_embed, high_res_feats_0, high_res_feats_1
  -> head(image_embed) (once)    -> presence, sky map, person centers and boxes
Sky      found when presence >= 0.423; the 3 most probable cells of each 8-connected region of
         >= 3 cells -> decoder per region -> union
People   found when presence >= 0.533; decode_people (nms_iou 0.5); the People list holds the
         boxes with score >= 0.313; one box per person, clipped to the photo -> decoder
Stroke   4 points at equal spacing along the stroke (both ends); a click is one point
Add, subtract   all points of the selection (labels 1 and 0) and the box in one decoder run, with
         the previous selected logits as mask_input
Click    multimask output, best predicted IoU; [ and ] change to the smaller or larger output
Box      box corners (labels 2 and 3) first; replaces the box of the selection
Subject  BiRefNet lite 1024 (stretch to 1024 x 1024, ImageNet normalization) -> alpha;
         Background = 1 - alpha
Feather  Gaussian blur, sigma in photo px; Edge: dilate (> 0) or erode (< 0) in photo px
```

- The presence thresholds are the best-F1 thresholds of the run 2 COCO `val` split, and the People
  list threshold is `person_list_best_f1_nms`, all from `people_sky_head.json`.
- SAM 2.1 masks are the 256 x 256 logits upsampled bilinearly to the photo size, threshold 0 (the
  rule of `evaluate_quality_photos.py`). The BiRefNet alpha is resized bilinearly to the photo size
  (`birefnet_graphs.alpha_to_photo`).
- A selection is one mask type at a time. A stroke or box after a one-click selection starts a new
  prompt selection (F7, the combination across mask types, is not in the MVP).
- A subtract stroke without a selection gives a message. A box with Alt gives a message: the
  decoder takes one box as a positive prompt.
- "Not found": Sky and People give a message with the presence and the threshold, not an empty
  mask.

## Runtime

- `--provider directml` (default of the UI) or `--provider cpu`. Each session has only the
  selected execution provider. If the provider fails, the error stops the program; the MVP does not
  change the provider.
- CPU threads 4 (ONNX Runtime intra-op threads, PyTorch, OpenCV), CPU memory arena off. DirectML
  sessions use sequential execution without memory patterns.
- All four graphs load together at the start. The model calls run one at a time in one worker
  thread.
- Each stage time (photo read, encoder, head, each decoder run, BiRefNet) shows in the UI. `--log`
  writes a JSON record of the times and the graph hashes.

## Commands

```bash
# Python environment (Git Bash, Windows)
E=build/tmp/explore
py -3.12 -m venv $E/venv_ai_mask_mvp
$E/venv_ai_mask_mvp/Scripts/python -m pip install -r scripts/poc/ai_mask_mvp/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
PY="$E/venv_ai_mask_mvp/Scripts/python -I"; M=scripts/poc/ai_mask_mvp

# The UI (DirectML). Optional: a photo path, --provider cpu, --log <json>.
$PY $M/ai_mask_app.py $E/e4/input_4096/photos/DSC_1535.jpg --log build/tmp/ai_mask_mvp/ui_log.json
```

The comparison needs the E2 BiRefNet CPU output of the same graph on the same photos (the
4096 px input set). Make it with the E2 script, then run the comparison (one job at a time):

```bash
E=build/tmp/explore; T=build/tmp/ai_mask_mvp; M=scripts/poc/ai_mask_mvp
$E/venv_birefnet/Scripts/python -I scripts/ai_models/birefnet/compare_onnx_outputs.py \
  --weights $E/weights/birefnet --variant lite --onnx-dir $E/e2/onnx_fp16/lite \
  --photos $E/e4/input_4096/photos --prompts build/tmp/sm1/quality/prompts.json \
  --out $T/e2_reference/lite_fp16/compare.json --threads 4
$E/venv_ai_mask_mvp/Scripts/python -I $M/compare_with_measurement_scripts.py --provider cpu \
  --photos $E/e4/input_4096/photos \
  --quality-results results/p1_people_sky_head/run2_coco_openimages/quality_photos.json \
  --quality-logits $E/p1/colab_run2_coco_openimages/quality_logits \
  --strokes-report $E/e3/runtime/tiny_directml.json \
  --subject-alpha $T/e2_reference/lite_fp16/alpha --out results/ai_mask_mvp/comparison_cpu.json
```

```powershell
# Cost on DirectML with the GPU memory of the process (PowerShell, from the repository root).
$e = "build\tmp\explore"; $py = "$e\venv_ai_mask_mvp\Scripts\python.exe"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
  -Arguments @('-I', 'scripts\poc\ai_mask_mvp\measure_pipeline_cost.py', '--provider', 'directml',
               '--photos', "$e\e4\input_4096\photos",
               '--strokes-report', "$e\e3\runtime\tiny_directml.json",
               '--out', 'results\ai_mask_mvp\cost_directml.json') `
  -Out results\ai_mask_mvp\cost_directml_gpu_memory.json -Log build\tmp\ai_mask_mvp\cost_directml.log
```

## Limits

- Input: JPEG, PNG, or TIFF (8 or 16 bit), scaled to a long edge of at most 4096 px. No RAW
  decoding.
- Not in the MVP: F4b (people parts), F7 (combination across mask types), F9 (stored recipe),
  F11 (text prompts), edge post-processing (U3), a "not found" rule for Subject.
- BiRefNet on DirectML: the first run of a session allocates about 3 GB of working memory, which
  stays until the session is released (the MVP keeps it for the program lifetime), and 4 of 9 first
  runs reset the display driver (`887A0006`) in the MVP report, section 5.3. The MVP shows the error
  as it occurs.
