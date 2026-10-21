# MVP: AI Mask with the selected model set

**Issue:** [#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5). **Date:** 2026-10-21.
**Code:** [`scripts/poc/ai_mask_mvp/`](../../scripts/poc/ai_mask_mvp/README.md).
**Results:** [`results/ai_mask_mvp/`](../../results/ai_mask_mvp/).
**Status:** implemented and measured. The interaction check by eye (U1) is open: the user does it.

> **No images.** This report names photos and describes them in words. The UI screenshots of the
> functional run stay in `build/tmp/ai_mask_mvp/ui_check/` (section 3.5).

## 1. Question

Each model of [README section 4](../../README.md#4-model-selection-proposal) was measured alone
(E1 to E5, P0, P1). Issue #5 asks for one program that runs the selected models together as the
product will use them, with a desktop UI:

1. Does the MVP give the same prompts and masks as the measurement scripts?
2. What do all models cost together on the development machine (time per stage, working set, GPU
   memory), compared with U6?
3. Does the interaction (one click, stroke, box, add and subtract, People list, Feather and Edge,
   overlay) work end to end? (The user checks it by eye.)

## 2. Result summary

- **The MVP exists:** `scripts/poc/ai_mask_mvp/` has the pipeline module (no UI), the PySide6 UI,
  the comparison script, the cost script, `requirements.txt`, and a README. The UI runs all tools
  of issue #5 on DirectML (section 3.5).
- **BiRefNet variant (user decision, 2026-10-21): 1024 px.** The MVP uses BiRefNet lite at 1024 px,
  the float16 graph of E2 (U8 permits float16). This closes O2 for the MVP.
- **Same results as the measurement scripts (`--provider cpu`):**
  - Sky and People: 14 of 14 photos. The presence decisions are equal, the sky points are equal,
    every People box is within 2.1e-5 cell, and 45 masks have IoU >= 0.99993 with the masks of
    `evaluate_quality_photos.py` (run 2).
  - Stroke, add, subtract: 149 of 149 decoder calls of the E3 sequences have the same inputs as
    `compare_stroke_outputs.py` (`mask_input` difference 0.0; 49 calls with `mask_input`, 10
    subtract steps).
  - "Not found": 8 of 8 cases give a message and no mask (for example `DSC_1188`).
  - Subject: **fails the 1e-3 tolerance in the MVP runtime** (maximum 0.035, 13 of 14 photos above
    1e-3). The cause is the ONNX Runtime build, not the MVP code: the MVP needs
    onnxruntime-directml 1.24.4 (the newest DirectML build), and the E2 output comes from
    onnxruntime 1.31.0. The two builds have different float16 CPU kernels. With onnxruntime 1.31.0,
    the same MVP code gives a maximum difference of 2.4e-4 (the float16 storage of the E2 file) and
    passes on 14 of 14 photos. With 1.24.4, the mean difference is at most 3.2e-5 and the IoU of
    alpha > 0.5 at the photo size is at least 0.99986.
- **Cost on DirectML (RTX 3080 Laptop GPU 8 GB), all models loaded together:**
  - warm p50: encoder 74 ms, head 2 ms, decoder 22 to 24 ms per prompt (box, stroke, 10 points
    with `mask_input`), sky decoder 9 ms per region, BiRefNet 156 ms. Read and scale a 4096 px
    photo: 67 ms.
  - Peak dedicated GPU memory **4,528 MB**, peak shared 486 MB, peak working set 1,873 MB. The
    peak is reached at the first BiRefNet run (BiRefNet lite float16 alone: 3,226 MB in E2).
  - Against U6: **the models need 1.5 GB more than the soft 3 GB budget.** With the editor
    (about 5 GB on a 24 MP photo), the total is more than the 8 GB of this GPU.
- **BiRefNet memory and GPU resets (section 5.3, recorded on 2026-10-21; the cause is open):**
  - The GPU memory increases by about 3 GB at the first BiRefNet run, not at the load: the session
    load holds 227 MB, the first run 3,226 MB. More runs do not increase it (not a leak). DirectML
    keeps this working memory of the intermediate tensors in its pool until the session is
    released (then 107 MB). ONNX Runtime has no option that releases the DirectML pool of a session
    (`memory.enable_memory_arena_shrinkage` refuses the DirectML device). A new session load takes
    7.5 to 8 s (3.7 to 4.5 s with a saved optimized graph).
  - `887A0006` (`DXGI_ERROR_DEVICE_HUNG`) is a reset of the display driver: Windows recorded 4
    events "nvlddmkm stopped responding and has successfully recovered" (event 4101). All 4
    occurred in the first run of a BiRefNet session: **4 of 9 first runs, 0 of 100 later runs**.
    A failed first run takes about 64 s. The SAM 2.1 tiny graphs did not fail in this step or in
    E1, E3, and E4. The MVP stopped with the error each time.
  - The facts point to the BiRefNet graph as the common factor (the user suspects the model
    selection): only BiRefNet fails, E2 recorded the same error with the float32 and float16
    graphs and with the Rust program, and the BiRefNet graph has a special structure (300
    GridSample nodes from the deformable convolution, 601 DirectML partitions). A test that shows
    the cause was not done.
- **Finding on the People list threshold:** on `2088064441` (paddle boarder) the person presence
  is 0.999, but the best box has the score 0.31267, below the list threshold 0.31321 (run 2 COCO
  `val` best F1). The People list is empty and the MVP gives the message "No people found:
  presence 0.999, but no person box has a score >= 0.313". The box of that person gives IoU 0.686
  (P1 run 2). The list threshold is an open decision of P1 (section 7, item 2).

## 3. Method

### 3.1 Inputs

| Item | Value |
| --- | --- |
| Photos | The 14 quality photos at the application input size (`build/tmp/explore/e4/input_4096/photos/`, long edge 4096 px, E4) |
| SAM 2.1 tiny | `e4/onnx_float32_batch16/tiny/point_encoder.onnx` (SHA-256 `17a554b4...`) and `point_decoder.onnx` (`6f9a0ef2...`), float32, the graphs of P1 |
| Head | P1 run 2, epoch 15: `p1/colab_run2_coco_openimages/export/people_sky_head.onnx` (`e02f2a1e...`) and its JSON (thresholds sky 0.4232, person 0.5332, People list 0.3132) |
| BiRefNet | lite, 1024 px, float16 inside, float32 input and output: `e2/onnx_fp16/lite/subject_matte.onnx` (96,437,277 bytes, `b27fc5c4...`) |
| Sky and People reference | `results/p1_people_sky_head/run2_coco_openimages/quality_photos.json` and the saved logits of that run (`p1/colab_run2_coco_openimages/quality_logits/`, float16) |
| Stroke reference | The E3 SAM 2.1 tiny strokes report (`e3/runtime/tiny_directml.json`), sequences `click_then_clicks`, `click_then_strokes`, `stroke_then_strokes`, `full_stroke_then_strokes` (the MVP rule: all points with the previous mask) |
| Subject reference | E2 `compare_onnx_outputs.py` with the float16 lite graph on the same 4096 px photos, ONNX Runtime CPU (`build/tmp/ai_mask_mvp/e2_reference/lite_fp16/`). The E2 run of 2026-10-19 used the camera-size photos, so it is not the same input. The new run also compares the graph with PyTorch: 14 of 14 photos pass, minimum IoU 0.99938 |
| Licenses | SAM 2.1 Apache-2.0; BiRefNet MIT; head: own model, training data COCO 2017 (CC BY 4.0 annotations, photos with a license that permits commercial use) and Open Images V7 (CC BY 4.0, CC BY 2.0 photos); PySide6 LGPL-3.0; ONNX Runtime MIT |

### 3.2 Pipeline and rules

`mask_pipeline.py` reuses the code of the measurement scripts: `make_input_set.scaled_size`
(photo scaling), `sam2_graphs.preprocess_image` and `model_prompt`, `select_single_mask` of
`compare_onnx_outputs.py`, `sky_regions` and `region_points` of `evaluate_quality_photos.py`,
`head_model.decode_people`, and `birefnet_graphs.preprocess` and `alpha_to_photo`. The rules are
the rules of issue #5:

| Function | Rule |
| --- | --- |
| Sky | presence >= 0.4232; 3 most probable cells of each 8-connected region of >= 3 cells; union of the region masks |
| People | presence >= 0.5332; `decode_people`, `nms_iou = 0.5`; People list: score >= 0.3132; one box per person, clipped to the photo; one person or all people (union) |
| Stroke | 4 points at equal spacing along the stroke in photo pixels, both ends included; a click is one point |
| Add, subtract | Alt subtracts (label 0); all points and the box in one decoder run, previous selected logits as `mask_input` |
| Click | Multimask output, best predicted IoU; `[` and `]` select the smaller or larger of the three outputs (no model run) |
| Box | Box corners first (labels 2, 3); a new box replaces the box of the selection. Alt with a box gives a message (the decoder takes one positive box) |
| Subject, Background | BiRefNet alpha; Background = 1 - alpha |
| Masks | SAM 2.1: 256 x 256 logits, bilinear to the photo size, threshold 0. BiRefNet: alpha, bilinear to the photo size |
| Feather, Edge | Gaussian blur (sigma in photo px); dilate or erode in photo px (alternating 3 x 3 cross and square steps, an octagon) |
| Runtime | One execution provider per session (`DmlExecutionProvider` or `CPUExecutionProvider`); a failure stops the program. CPU threads 4, CPU memory arena off |

### 3.3 Checks

`compare_with_measurement_scripts.py`, `--provider cpu`, all graphs loaded in one pipeline:

| Check | Behavior and tolerance |
| --- | --- |
| `SkyPeoplePromptsMatchQualityScript` | Presence decision equal; sky points within 1e-6 model px; every `decode_people` box within 1e-4 cell; sky union and People list masks IoU >= 0.999 against the script logits (where the script saved them) |
| `NotFoundGivesMessage` | Script presence below the threshold: the MVP gives "not found", a message, and no mask |
| `StrokeDecoderInputsMatchProbeScript` | `point_coords` within 1e-4 model px, `point_labels` and `has_mask_input` equal, `mask_input` within 1e-4, against the inputs of `compare_stroke_outputs.py` (`interactions`, `OnnxDecoder`) on separate CPU sessions; `stroke_points(start, end, N)` gives the recorded E3 points within 1e-6 |
| `SubjectAlphaMatchesE2Output` | Maximum absolute alpha difference <= 1e-3 at 1024 x 1024 |

The issue names `probe_strokes.py` as the stroke reference. That script only makes tables from the
Rust report; the decoder inputs of the replay come from `compare_stroke_outputs.py`, so the check
uses it.

### 3.4 Cost measurement

`measure_pipeline_cost.py` under `run_with_gpu_memory.ps1`, `--provider directml`. All four graphs
load first. Per photo: read and scale, encoder, head, Sky, People (one decoder run per listed
person), the E3 `full_stroke_then_strokes` sequences (stroke, then add and subtract strokes with
`mask_input`), Subject. Then 20 warm runs on the first photo: encoder, head, decoder with a box,
decoder with 10 points and `mask_input`, BiRefNet. 104 decoder runs over the photos.

`run_with_gpu_memory.ps1` sampled only the started process. A virtual environment `python.exe`
is a launcher that runs the interpreter as a child process, so the first run recorded 0 MB. The
script now adds the counters of all child processes and writes LF line endings. The Windows
counter query takes about 1.3 s, so the samples are 1.3 s apart (as in E3), not 250 ms. The
DirectML memory does not decrease during the run (the samples after the first BiRefNet run are all
4,528 MB).

| Run | Result |
| --- | --- |
| 1 | Passed; GPU memory not recorded (launcher process only) |
| 2 | Stopped with `887A0006` at the first BiRefNet call of the first photo (dedicated 4,016 MB at that time). Log: `build/tmp/ai_mask_mvp/cost_directml_attempt2_887A0006.log.stderr` |
| 3 | Passed (process tree query in each sample, samples 1.6 s apart) |
| 4 | Passed; **the values of this report** (`results/ai_mask_mvp/cost_directml.json`, `cost_directml_gpu_memory.json`) |

### 3.5 Functional run of the UI

`build/tmp/ai_mask_mvp/ui_run_check.py` (local, not committed) opens the real window
(`QT_QPA_PLATFORM=offscreen`, DirectML) on `DSC00527` and calls the UI handlers in sequence. All
steps passed:

| Step | Result |
| --- | --- |
| Open | 4096 x 2729, encoder and head once |
| Subject, Background | foreground 2.4 percent (three people), Background 97.6 percent |
| Sky | 1 region, 40.3 percent of the photo |
| People | 20 people in the list (the `decode_people` limit is 20); person 1 alone and all people |
| Click | level 2 of 3 by default; `]` and `[` change the level without a model run |
| Add stroke, subtract stroke (Alt) | 5 and 9 decoder points (the click and 4 points per stroke), `mask_input` from the previous step |
| Box | 11 decoder points (2 box corners and 9 points) |
| Box with Alt | message, no model run |
| Subtract without a selection | message "Nothing to subtract: make a selection first." |
| Feather 10, Edge -5, overlay toggle | the overlay changes; `Y` hides it |
| Export | 8-bit PNG at 4096 x 2729 and the prompts JSON |

The stage times show in the UI panel (for example click decoder 40 ms, stroke 27 ms, box 33 ms in
this run).

### 3.6 Machine

Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce RTX 3080 Laptop GPU 8 GB,
driver 610.62. Python 3.12.4. MVP environment: onnxruntime-directml 1.24.4, PySide6 6.12.0,
PyTorch 2.14.1 CPU (preprocessing and `decode_people` only), NumPy 2.5.2, OpenCV 5.0.0.93. The
reference environment of the SAM 2.1 and head scripts has onnxruntime 1.31.0; the BiRefNet
environment has onnxruntime 1.31.0 and Python 3.14.2.

## 4. Results: comparison with the measurement scripts

Records: [`comparison_cpu.json`](../../results/ai_mask_mvp/comparison_cpu.json) (MVP environment,
onnxruntime-directml 1.24.4) and
[`comparison_cpu_onnxruntime_1_31.json`](../../results/ai_mask_mvp/comparison_cpu_onnxruntime_1_31.json)
(the same script in the reference environment, onnxruntime 1.31.0). The graph paths in the
records were absolute after the runs; they were changed to paths relative to the repository, the
form that `mask_pipeline.py` now writes. No other value changed.

| Check | onnxruntime-directml 1.24.4 (MVP) | onnxruntime 1.31.0 |
| --- | --- | --- |
| Sky and People prompts and masks | Pass: 14 / 14 photos; boxes within 2.1e-5 cell; presence within 7.7e-7; 45 masks, minimum IoU 0.99993 | Pass: 14 / 14; boxes and presence equal (0.0); minimum IoU 0.99993 |
| "Not found" | Pass: 8 / 8 | Pass: 8 / 8 |
| Stroke decoder inputs | Pass: 149 / 149 calls; `mask_input` difference 0.0; selections equal | Pass: 149 / 149 |
| Subject alpha <= 1e-3 | **Fail:** 1 / 14 photos; maximum 0.035 (`DSC00527`) | Pass: 14 / 14; maximum 2.4e-4 |

- The minimum mask IoU of 0.99993 comes from the float16 storage of the script logits: it is the
  same with equal boxes (onnxruntime 1.31.0).
- The People masks are compared where the script decoded the boxes (the 7 photos with a
  referenced person). On `DSC00562` (6 listed) and `DSC_1456` (12 listed) only the boxes are
  compared.
- Per photo with onnxruntime-directml 1.24.4 (Subject): maximum alpha difference 0.006 to 0.035,
  mean at most 3.2e-5, IoU of alpha > 0.5 at the photo size 0.99986 to 1.0. The lupine photo
  (almost no foreground) has 2.5e-5.
- `build/tmp/ai_mask_mvp/subject_ort_version_check.py` (local) runs the MVP session and
  preprocessing in the onnxruntime 1.31.0 environment on `DSC00527` and `2088064441`: maximum
  difference 2.4e-4, and 0.0 after float16 rounding of the MVP alpha (the E2 file is float16).

Sky and People per photo (MVP; equal to the script):

| Photo | Sky | People |
| --- | --- | --- |
| `2088064441` | 1 region | presence 0.999, list empty (best score 0.31267 < 0.31321): message |
| `5761824892` | not found (0.312) | not found (0.032) |
| `DSC00326` | 1 region | 1 person |
| `DSC00527` | 1 region | 20 people |
| `DSC00562` | 1 region | 6 people (people on the wall photo, not referenced) |
| `DSC00602` | 3 regions | not found (0.293) |
| `DSC00674` | 2 regions | 1 person |
| `DSC00728` | 1 region | not found (0.116) |
| `DSC00936` | 1 region (the waterfall; not a target of this step) | 1 person |
| `DSC_1188` | not found (0.006) | not found (0.004) |
| `DSC_1456` | 1 region | 12 people |
| `DSC_1535` | not found (0.008) | 9 people |
| `Nikon-D800e-raw-00007` | 1 region | not found (0.012) |
| `P1024439` | 4 regions | 2 people |

## 5. Results: cost on DirectML

Record: [`cost_directml.json`](../../results/ai_mask_mvp/cost_directml.json) and
[`cost_directml_gpu_memory.json`](../../results/ai_mask_mvp/cost_directml_gpu_memory.json) (run 4).

### 5.1 Time per stage

Milliseconds. "Photos" is one call per photo (13 photos after the first); warm is 20 runs on the
first photo (`2088064441`).

| Stage | Session load | First call | Photos p50 / maximum | Warm p50 / p95 / maximum |
| --- | ---: | ---: | --- | --- |
| Read and scale the photo | — | 84 | 66 / 81 | — |
| SAM 2.1 tiny encoder | 2,162 | 374 | 74 / 81 | 73.7 / 75.0 / 75.1 |
| Head | 106 | 7.8 | 2.1 / 3.0 | 2.1 / 3.0 / 4.3 |
| Decoder (all 99 calls after the first photo) | 248 | 95.5 | 22.2 / 42.6 | — |
| Decoder, box | | | | 23.2 / 29.2 / 34.3 |
| Decoder, 10 points and `mask_input` | | | | 23.4 / 34.2 / 38.6 |
| BiRefNet lite 1024 float16 | 7,998 | 417 | 160 / 175 | 155.8 / 167.8 / 178.9 |

Decoder per use (all photos): sky region 8.9 ms p50 (17 runs), person 22.6 ms (52 runs), stroke
22.3 to 24.2 ms (35 runs). What the user waits for (warm): open a photo about 145 ms (read,
encoder, head); Sky 9 to 36 ms (1 to 4 regions); People 23 ms per person (`DSC00527`, 20 people:
about 0.5 s); one stroke, click, or box 22 to 40 ms; Subject 156 to 175 ms. The first call of
each graph after the load is slower (encoder 374 ms, BiRefNet 417 ms; 3.6 s in run 1).

The decoder with one session and mixed point counts takes 22 to 24 ms per call, between the E3
values (10.5 ms for the first point count of a session, about 27 ms with mixed counts). The sky
regions (3 points each) take 9 ms.

### 5.2 Memory

| Item | Value |
| --- | ---: |
| Peak dedicated GPU memory | 4,528 MB |
| Peak shared GPU memory | 486 MB |
| Dedicated GPU memory samples on the first photo | 230 MB after the load, 383 to 387 MB after the encoder, then 1,460 MB and 4,528 MB in the 2.6 s with the first BiRefNet run |
| Peak working set | 1,873 MB |
| Working set after the load of the four graphs | 1,018 MB |
| Private bytes at the end | 6,600 MB |

Earlier values of the single models: SAM 2.1 tiny 1,355 MB (E1); BiRefNet lite float16 3,226 MB
(E2). The MVP peak (4,528 MB) is about their sum.

**Comparison with U6.** U6 gives a soft budget of 3 GB for the models; the editor uses about 5 GB
on a 24 MP photo, and the driver can swap part of the editor caches. The MVP models use 4.5 GB
dedicated (1.5 GB over the budget). The samples are 1.3 s apart, so the share of each graph is
not separated; E1 and E2 give 1.4 GB for SAM 2.1 tiny and 3.2 GB for BiRefNet. U6 permits the models to be unloaded when the user goes back to adjusting; the
MVP keeps all four loaded, as issue #5 asks for the measurement.

### 5.3 BiRefNet memory and GPU resets

The user asked why the GPU memory increases suddenly when Subject is selected. This section
records the measurements. The cause of the GPU resets is not known; the user suspects the
selection of BiRefNet.

**Memory per phase of the BiRefNet session.** `measure_birefnet_memory_phases.py`, float16 lite
graph, `DSC_1535`, DirectML, the session settings of the MVP. Record:
[`birefnet_memory_phases.json`](../../results/ai_mask_mvp/birefnet_memory_phases.json). The run
used a copy of the script in `build/tmp/ai_mask_mvp/birefnet_memory/` with the same behavior; the
record has the repository path form and the photo name added.

| Phase | Time | Dedicated GPU memory | Shared GPU memory |
| --- | ---: | ---: | ---: |
| Before the load | — | 0 MB | 0 MB |
| Session load | 7,860 ms | 227 MB | 180 MB |
| Run 1 | 781 ms | 3,226 MB | 206 MB |
| Run 2 | 148 ms | 3,226 MB | 234 MB |
| Run option `memory.enable_memory_arena_shrinkage` = `gpu:0` | refused: "Did not find an arena based allocator registered for device-id combination" | — | — |
| The same option with `DML:0` | refused: "Unsupported device specified in the memory arena shrink list: DML" | — | — |
| Session released | — | 107 MB | 164 MB |
| Session load again | 7,500 ms | 230 MB | 218 MB |
| Run after the second load | 63,866 ms (GPU reset, see below) | 3,227 MB | 428 MB |

- The weights take about 0.2 GB. The about 3 GB come at the first run: DirectML allocates the
  working memory of the intermediate tensors (1024 x 1024 input, decoder maps up to 1024 x 1024)
  and keeps it in the pool of the session for the next runs. The float32 graph needs 5.7 GB (E2),
  so the memory is proportional to the tensor data, not a fixed cost per session.
- It is not a leak: 20 warm runs and 14 photos keep 4,528 MB in the cost run (section 5.2), and run
  2 here keeps 3,226 MB.
- Only the release of the session releases the memory. The MVP keeps the session for the program
  lifetime; it has no release step (U6: unload when the user goes back to adjusting).

**Session load time.** `measure_birefnet_load_time.py` (no inference). Record:
[`birefnet_load_time.json`](../../results/ai_mask_mvp/birefnet_load_time.json).

| Graph | All optimizations | Extended | Basic | No optimization |
| --- | ---: | ---: | ---: | ---: |
| E2 float16 lite graph | 8.03 s | 7.58 s | 6.87 s | 17.90 s |
| The same graph, optimized once at the extended level on the CPU provider and saved (12.5 s, 94.8 MB) | 4.45 s | — | 3.72 s | 6.02 s |

A release after each Subject call therefore costs 3.7 to 8 s more on the next Subject call, and
every call becomes a first run (see the resets below).

**GPU resets.** Windows System event log, 2026-10-21 15:00 to 17:30. Record:
[`gpu_reset_events.json`](../../results/ai_mask_mvp/gpu_reset_events.json). Each reset has
`nvlddmkm` event 153 at the start of the run, again 60 s and 64 s later, then event 4101 "the
display driver nvlddmkm stopped responding and has successfully recovered". ONNX Runtime then
reports `887A0006` (`DXGI_ERROR_DEVICE_HUNG`) for that run or the next run.

| BiRefNet DirectML session (2026-10-21) | First run |
| --- | --- |
| Cost run 1 | passed (3,572 ms) |
| Cost run 2 | reset (event 4101 at 16:03:08) |
| Cost run 3 | passed |
| Cost run 4 | passed (417 ms) |
| UI functional run (section 3.5) | passed |
| Memory phases, attempt 1 | reset (16:54:16) |
| Memory phases, attempt 2 | reset (16:56:01) |
| Memory phases, attempt 3, first session | passed (781 ms) |
| Memory phases, attempt 3, second session | reset (16:58:48) |

- 4 of 9 first runs reset the driver; 0 of 100 later runs did (33 runs in each of the cost runs
  1, 3, and 4, and run 2 of the memory phases).
- A failed first run takes about 64 s: the driver waits 60 s before the reset.
- The SAM 2.1 tiny sessions did not fail: 6 first runs in this step (the cost runs 1 to 4, the UI
  run, and cost run 2, which ran SAM 2.1 before BiRefNet), and no such error in E1, E3, and E4.
- E2 recorded the same error with BiRefNet: the first float16 runs of lite and general, the float16
  run of lite matting, and one float32 lite run, in the Rust program (`ort`, ONNX Runtime 1.24).
  So the error does not depend on float16, on the Python package, or on the MVP code.
- Other programs used the GPU during the runs (desktop programs, about 2.8 GB dedicated memory in
  total); the dedicated memory of the MVP process was at most 4,016 MB in the failed cost run.

**Facts that point to the BiRefNet graph (the user suspects the model selection).**

- Only the BiRefNet graph fails, with all precisions and both runtimes (E2 and this step).
- The export decomposes `deform_conv2d` of the BiRefNet decoder into 300 GridSample nodes and one
  1 x 1 convolution per kernel point (E2 section 3.3). DirectML runs the graph in 601 fused
  partitions; the float16 graph puts 724 nodes on the CPU (the casts around the float32 GridSample
  and Range nodes), so data goes between the GPU and the CPU at the partition borders.
- The working memory of BiRefNet lite at 1024 px is 3.2 GB (float16) and 5.7 GB (float32), against
  1.4 GB for SAM 2.1 tiny.

Not shown: which part of the graph causes the reset. Tests that can show it: the same BiRefNet
graph with the decoder without the deformable blocks (a graph change, not a product candidate), a
Subject model without deformable convolution, and the reset rate of `BiRefNet_512x512` (O2; it has
the same decoder, so the same structure). These tests need a user decision on the Subject model
(O2, U9, U12).

## 6. Recommendation

- **Use the MVP for the interaction check (U1).** The pipeline gives the measured prompts and
  masks of P1 and E3, and the UI has all functions of issue #5.
- **Subject criterion.** Accept the Subject comparison with the stated cause, or change the
  criterion: the MVP code gives the E2 alpha within 2.4e-4 in the same ONNX Runtime build, and the
  DirectML package cannot be at 1.31.0 (the newest DirectML build is 1.24.4). A criterion for the
  shipped runtime can be "IoU of alpha > 0.5 >= 0.999 at the photo size" (measured: >= 0.99986).
  This is the user's decision.
- **GPU memory and GPU resets.** The models need 4.5 GB with BiRefNet lite 1024 float16, and the
  first BiRefNet run reset the display driver in 4 of 9 sessions (section 5.3). Both come from the
  BiRefNet session. A release after each Subject call frees 3.1 GB but costs 3.7 to 8 s per call
  and makes each call a first run, so it is not recommended before the cause of the resets is
  known. The next step is a decision on the Subject model (O2): the tests of section 5.3 show
  whether the BiRefNet graph structure is the cause.
- **People list threshold.** The boarder of `2088064441` is lost by a margin of 0.0005. The list
  threshold (best F1 of COCO `val`) is part of the open People list decision of P1 (section 11.7,
  item 4).

## 7. Open items

1. **The user checks the interaction by eye** on the quality photos and on own photos (U1). The
   acceptance criteria of issue #5 are a proposal; the user confirms them.
2. **People list threshold and order** (P1 section 11.7, items 2 to 4): a box with the center
   point, spread sky points, and the threshold. The rules are in `mask_pipeline.py` and can change
   without a UI change.
3. **Subject "not found" rule** (F1): not in the MVP. On the lupine photo BiRefNet gives almost
   nothing (E2: 0.3 percent at the camera size).
4. **GPU memory over U6** (sections 5.2 and 5.3) and the unload rule of U6: the MVP has no
   release step for the sessions.
5. **GPU resets with BiRefNet** (section 5.3): 4 of 9 first runs of a BiRefNet DirectML session
   reset the display driver (`887A0006`); the cause is not known. The user suspects the selection
   of BiRefNet (O2).
6. **F7** (combination across mask types): a stroke after a one-click selection starts a new
   selection in the MVP.

## 8. Not measured

- macOS Core ML and other GPUs.
- CPU cost of the MVP (the CPU runs of section 4 are comparisons; the float16 BiRefNet graph takes
  about 9 s per photo on the CPU, because the CPU runs float16 slowly).
- The UI with a real user, on own photos, and with 16-bit TIFF files (the reader is implemented, not
  run on a file).
- The People masks of `DSC00562` and `DSC_1456` against a script (the script decodes only the
  photos with a referenced person).
- The float32 BiRefNet lite graph in the MVP (`--subject-onnx` selects it; not measured).

## 9. Commands

The commands are in the [MVP README](../../scripts/poc/ai_mask_mvp/README.md#commands): the E2
reference run on the 4096 px photos, the comparison (`--provider cpu`), the cost run under
`run_with_gpu_memory.ps1`, and the UI. The comparison in the reference environment is the same
command with `build/tmp/explore/venv/Scripts/python` and
`--out results/ai_mask_mvp/comparison_cpu_onnxruntime_1_31.json`. The UI screenshots of section 3.5:

```bash
E=build/tmp/explore
$E/venv_ai_mask_mvp/Scripts/python -I build/tmp/ai_mask_mvp/ui_run_check.py \
  $E/e4/input_4096/photos/DSC00527.jpg directml build/tmp/ai_mask_mvp/ui_check
```

Section 5.3 (one at a time; the memory phases run BiRefNet on DirectML and can reset the display
driver):

```bash
E=build/tmp/explore; PY="$E/venv_ai_mask_mvp/Scripts/python -I"; M=scripts/poc/ai_mask_mvp
$PY $M/measure_birefnet_memory_phases.py $E/e2/onnx_fp16/lite/subject_matte.onnx \
  $E/e4/input_4096/photos/DSC_1535.jpg --out results/ai_mask_mvp/birefnet_memory_phases.json
$PY $M/measure_birefnet_load_time.py $E/e2/onnx_fp16/lite/subject_matte.onnx \
  --work-dir build/tmp/ai_mask_mvp/birefnet_memory --out results/ai_mask_mvp/birefnet_load_time.json
```

```powershell
# GPU reset events (Windows System event log).
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-10-21 15:00';
  EndTime=[datetime]'2026-10-21 17:30'} |
  Where-Object { $_.Id -eq 4101 -or ($_.ProviderName -eq 'nvlddmkm' -and $_.Id -eq 153) }
```

## 10. Changed files

- `scripts/poc/ai_mask_mvp/` (new): `mask_pipeline.py`, `ai_mask_app.py`,
  `compare_with_measurement_scripts.py`, `measure_pipeline_cost.py`,
  `measure_birefnet_memory_phases.py`, `measure_birefnet_load_time.py`, `requirements.txt`,
  `README.md`.
- `scripts/ai_models/birefnet/compare_onnx_outputs.py`: `--threads` (default 4) for PyTorch and
  ONNX Runtime, CPU memory arena off.
- `scripts/ai_models/sam2/run_with_gpu_memory.ps1`: the counters of the child processes; LF line
  endings in the JSON output.
- `results/ai_mask_mvp/` (new), this report, [the index](README.md),
  [`results/README.md`](../../results/README.md), and [README](../../README.md) (status, O2,
  section 6).
