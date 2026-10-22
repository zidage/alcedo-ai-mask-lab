# Subject models in place of BiRefNet lite

**Issue:** [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6). **Date:** 2026-10-21.
**Code:** [`scripts/ai_models/s3od/`](../../scripts/ai_models/s3od/README.md),
[`scripts/ai_models/isnet/`](../../scripts/ai_models/isnet/README.md),
[`scripts/ai_models/universal_matting/`](../../scripts/ai_models/universal_matting/README.md).
**Results:** [`results/subject_alternatives/`](../../results/subject_alternatives/).
**Status:** S3OD `s3od-synth` is measured (sections 2 to 8), also with the attention in query
chunks (section 5.5, 2026-10-21). IS-Net general-use is measured (section 11, 2026-10-21).
Alibaba universal matting is measured (section 12, 2026-10-22). The check of the masks by eye is
open: the user does it.

> **No images.** This report names photos and describes them in words. The comparison sheets are
> local only; section 6.3 gives the command that makes them.

## 1. Question

The AI Mask MVP (#5) uses BiRefNet lite at 1024 px float16 for F1 (Subject) and F2 (Background).
It is the most expensive model of the set ([MVP report, sections 5.1 to 5.3](mvp_ai_mask.md#5-results-cost-on-directml)):
3,226 MB dedicated GPU memory from the first run, 7.5 to 8.0 s session load, and 4 of 9 first
runs reset the display driver (`887A0006`). Issue #6 asks for three candidates, measured with the
MVP settings on ONNX Runtime DirectML and compared with BiRefNet lite:

1. Can the candidate be exported to ONNX, and does the float32 ONNX output agree with the
   framework output on the CPU (maximum absolute difference <= 1e-3 on the 14 quality photos)?
2. What does it cost on DirectML (load, first run, warm runs, GPU memory per phase, CPU nodes,
   driver resets in at least 9 first runs)?
3. How do its masks compare with BiRefNet lite (E2 metrics, IoU)?
4. Does it meet U6 (the Subject model and SAM 2.1 tiny, about 1.4 GB, together in 3 GB) without
   a driver reset?

## 2. Result summary

- **Export: possible, with three graph changes that keep the computation.** `s3od.pt` is a
  `state_dict` (not TorchScript). The TorchScript exporter has no symbolic for the `antialias=True`
  resize of the mask head; the dynamo exporter (opset 18) exports the model. Three changes were
  necessary for DirectML (section 4):
  1. The exporter writes 51 Reshape nodes with `allowzero=1`. ONNX Runtime DirectML 1.24.4
     refuses that graph at session creation (`80070057`, E_INVALIDARG). `allowzero=0` gives the
     same output because no shape has a 0.
  2. The mask head resizes a 1024 x 1024 map to 1024 x 1024 with antialias. The result is bitwise
     equal to the input, and DirectML runs the node on the CPU. The export removes it.
  3. Float16: the attention scores of encoder layer 0 reach 134,699 to 135,151, more than the
     float16 maximum (65,504). With them in float16 the DirectML output is NaN on 14 of 14 photos.
     The float16 graph keeps that MatMul and its Softmax in float32. The other ten layers are at
     most 482.
- **Float32 ONNX against PyTorch: fails the 1e-3 tolerance on 1 of 14 photos.** `P1024439` has
  a maximum mask difference of 1.09e-3; the other 13 are between 4.0e-5 and 9.3e-4. The selected
  mask is the same on 14 of 14 photos, and the IoU at the photo size is 1.00000. PyTorch differs
  from itself by up to 1.09e-3 (its own `eager` attention against its `sdpa` attention), and ONNX
  Runtime with and without graph optimizations differs by 2.19e-3. So 1e-3 is less than the float32
  rounding differences of the reference. The tolerance is the user's decision (section 8).
- **Cost on DirectML (warm p50, dedicated GPU memory after the first run):**

  | Graph | Session load | Warm p50 | Dedicated GPU memory | Driver resets in first runs |
  | --- | ---: | ---: | ---: | --- |
  | BiRefNet lite 1024 float16 (MVP) | 9,442 ms | 154 ms | 3,226 MB | 4 of 9 |
  | S3OD float32, CPU Resize kept | 1,014 ms | 643 ms | 3,549 MB | 0 of 9 |
  | S3OD float16 (layer 0 attention float32) | 988 ms | 895 ms | 3,916 MB (+784 MB shared) | 0 of 9 |
  | S3OD float32, one partition | 1,151 ms | 4,013 ms | 3,805 MB (+1,504 MB shared) | 0 of 9 |

  No S3OD run reset the display driver: 0 of 54 first runs in new processes (6 graph forms), and
  0 in 8 cost runs and 8 profiling runs. BiRefNet reset the driver in 4 of 9 first runs again, and
  in its profiling run. S3OD loads 8 times faster but is 4 to 6 times slower per call. It uses more
  GPU memory than BiRefNet. The cause is the ViT-B attention at 1024 px: 4,101 tokens, so a
  [1, 12, 4101, 4101] score tensor per layer (403 MB in float16, 807 MB in float32).
- **Quality (14 photos, the 1024 grid, against BiRefNet lite float16):** IoU 0.86 to 0.995 on 12
  photos. On the two others S3OD is better: on `P1024439` BiRefNet adds the background (75 % of
  its foreground is outside the subject box; S3OD 2 %), and on `Nikon-D800e-raw-00007` BiRefNet
  gives an empty mask (S3OD finds the rock, recall 0.969). Mean subject recall 0.986 (BiRefNet
  0.922). The S3OD mask is almost binary: edge width 0.03 to 0.13 px (BiRefNet 0.99 to 5.44 px),
  so the edge at the 4096 px photo shows the 4 px steps of the 1024 grid. Hair and fur have no
  soft band. The user checks the sheets by eye.
- **U6: not met with the exported attention.** S3OD (3.5 to 3.9 GB) plus SAM 2.1 tiny (1.4 GB)
  is more than 3 GB, as with BiRefNet (3.2 GB). The "no driver reset" criterion is met.
- **Attention in query chunks (section 5.5): U6 met.** `com.microsoft.MultiHeadAttention` alone
  does not decrease the memory on DirectML: one layer uses 1,159 MB in both forms, and the float16
  layer 0 overflows in both forms. An attention in 8 query chunks, where each chunk reads the
  output of the chunk before it, gives the same output with a score tensor of 1/8 size. With
  `MultiHeadAttention` per chunk and float16 (layer 0 in float32): **191 ms warm, 1,358 MB
  dedicated GPU memory** (before: 895 ms and 3,916 MB; BiRefNet: 154 ms and 3,226 MB), 1.1 s
  load, one DirectML partition, 0 driver resets in 18 first runs. **The MVP pipeline with S3OD in
  place of BiRefNet (SAM 2.1 tiny, head, S3OD in one process) has a peak of 2,666 MB dedicated
  GPU memory** (with BiRefNet: 4,528 MB). The masks are the same (IoU >= 0.998 against S3OD
  float32 on the CPU); the float32 graph has the same CPU difference as before (1.06e-3 on
  `P1024439`).
- **IS-Net general-use (section 11): the lowest cost, the lowest quality, and no weights
  license.** The export is direct (TorchScript exporter, opset 17, no graph change); the float32
  ONNX output is within 9.1e-6 of PyTorch (requirement 1e-3 met). On DirectML: one partition,
  0 CPU nodes, float16 **14 ms per photo and 437 MB** dedicated GPU memory (float32: 27 ms,
  1,000 MB), 0.8 to 1.0 s load, 0 driver resets in 18 first runs. The MVP pipeline with IS-Net
  float16 has a peak of **1,748 MB** (U6 met). But the masks are worse than BiRefNet and S3OD on 7
  of 14 photos: mean subject recall 0.80 (BiRefNet 0.92, S3OD 0.99), large soft areas on the
  background (the rock under the squirrel, the cliff next to the balloon), the wall of
  `DSC00562` not found, and the hair of `DSC00674` partly below 0.5. The source scales the output
  with its minimum and maximum, so a photo without a detected subject (`Nikon-D800e-raw-00007`,
  d1 maximum 0.047) gets a soft mask of noise. The weights have no license statement
  ([DIS#150](https://github.com/xuebinqin/DIS/issues/150), no answer). In the same period the
  BiRefNet reference run reset the display driver in 2 of 2 sessions.
- **Alibaba universal matting (section 12): no reset, but U6 not met and worse masks.** The
  TensorFlow graph reads the photo at its size. The conversion needs rewrites of `ResizeArea`,
  `ResizeBilinear`, and `CropAndResize` (`tf2onnx` 1.17 converts them to other operations) and
  float64 scalar divisions (DirectML float32 division moved a crop row and changed one photo).
  The float32 ONNX output fails 1e-3 against the default TensorFlow session (6.3e-3), but
  TensorFlow differs from itself by up to 7.2e-3 over its CPU settings. On DirectML: 116 ms warm
  and 2,678 MB at 4096 px, 1.5 s load, 0 driver resets in 46 sessions; float16 gives no gain. The
  MVP pipeline with it has a peak of **3,982 MB** (U6 not met). Its masks are worse than BiRefNet
  and S3OD on 5 of 14 photos (mean subject recall 0.84; the bridge and the water of
  `2088064441`, the buildings of `DSC00562`). Weights license: Apache License 2.0. The BiRefNet
  reference run of this step reset the driver again.
- **License:** weights and code MIT; the encoder is DINOv3 (DINOv3 License, the license text must
  go with the weights). The training images come from FLUX.1 Krea [dev]. The FLUX.1 [dev]
  Non-Commercial License permits commercial use of outputs in section 2(d), but section 4(a)
  forbids commercial use of "any data produced by" the model. The masks come from a decoder that
  was supervised on DIS5K, HRSOD, UHRSD, and DUTS. A legal check is necessary (section 7).

## 3. Inputs

| Item | Value |
| --- | --- |
| Photos | The 14 quality photos at a long edge of at most 4096 px (`build/tmp/explore/e4/input_4096/photos`, `make_input_set.py`), list from `build/tmp/sm1/quality/prompts.json`. No photo has an EXIF orientation tag other than 1 |
| S3OD weights | `okupyn/s3od-synth` at `4b40ce663c31fdfbb446a347282960033e624b4d`, `s3od.pt` 464,526,126 bytes, SHA-256 `c67f26415ad5493edaac48c8f2a08fecb1d9933aed2ed473e8ad713b93b8e5f3`, model card `README.md` (`license: mit`) SHA-256 `d8d7a46d...` |
| S3OD code | `KupynOrest/s3od` at `11385423e8d04d78b19e71cb166971b79c49f4fe` (`s3od/s3od/model.py` SHA-256 `9fb241b3...`, `predictor.py` `a83e0be3...`, `dinov3_config/config.json` `256b03d5...`) |
| BiRefNet reference | The MVP graph `e2/onnx_fp16/lite/subject_matte.onnx` (96,437,277 bytes, SHA-256 `b27fc5c46a70...`) and its CPU alpha of the MVP comparison (`build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha`) |
| Subject reference masks | E1 SAM 2.1 base+ box masks (`build/tmp/explore/e1/compare/base_plus/reference_masks`), the first box prompt of each photo |

S3OD model: DINOv3 ViT-B/16 encoder (Hugging Face `DINOv3ViTModel`, 4 register tokens, RoPE),
DPT decoder, three mask heads and three IoU scores; 116,097,990 parameters. Input 1024 x 1024:
the source scales the long edge to 1024 and centers the photo on black (letterbox). The alpha is
the mask with the largest IoU score, unpadded and resized to the photo size.

## 4. Export

### 4.1 Exporter and preprocessing

- `s3od.pt` is a `torch.save` dictionary with one `state_dict` (371 tensors, float32). The issue
  text "TorchScript" is not correct. `torch.load(..., weights_only=True)` reads it, and
  `load_state_dict(strict=True)` accepts it (transformers 5.20.2).
- The preprocessing of `s3od_graphs.letterbox` gives the same tensor as
  `BackgroundRemoval._preprocess` (difference 0.0 on 14 photos). The postprocessing gives the same
  alpha as `BackgroundRemoval.remove_background` (difference 0.0 on 14 photos).
- TorchScript exporter: "Exporting the operator 'aten::_upsample_bilinear2d_aa' to ONNX opset
  version 18 is not supported". The dynamo exporter (`torch.onnx.export(..., dynamo=True)`,
  torch 2.14.1, opset 18) exports the model in 12 s: 716 nodes, 433.8 MB, no GridSample. The last
  encoder layer is not in the graph: the DPT decoder reads layers 2, 5, 8, and 11 of
  `hidden_states`, and `hidden_states[11]` is the output of the eleventh layer.

### 4.2 Changes for DirectML (float32)

**Reshape `allowzero`.** A DirectML session of the exported graph fails at creation:
`MLOperatorAuthorImpl.cpp(2853) ... 80070057` (E_INVALIDARG). Each operator type of the graph
creates a session when it is alone (33 single-node graphs), so the cause is in the graph context.
Bisection with graph cuts (`onnx.utils.Extractor`, input to the output of node k) found the first
failing node: `node_view_8`, a Reshape with `allowzero=1` after the first attention. With
`allowzero=0` on all 51 such Reshape nodes, the session is created. No shape constant of these
nodes has a 0, so the attribute does not change the output (ONNX Reshape). During these session
tests the System log has four `nvlddmkm` events 153 (18:13:33 to 18:14:27) and no event 4101.
`export_onnx.py --keep-reshape-allowzero` writes the graph without the change.

**Resize to the same size.** `MultiMaskHead` resizes its 1024 x 1024 map to the target size
1024 x 1024 with `antialias=True`. ONNX Runtime and PyTorch both give a bitwise equal tensor
(random [1, 64, 1024, 1024] input). DirectML has no antialias Resize, so the profile shows this
node on the CPU, with 4 DirectML partitions. The [1, 64, 1024, 1024] float32 tensor (256 MB) goes
to the CPU and back in each call. `export_onnx.py` removes the node (715 nodes, one DirectML
partition, 0 CPU nodes). `--keep-identity-resize` keeps it; this report calls that graph the
"CPU Resize" form. Both forms compute the same output; their cost is different (section 5).

### 4.3 Float16

`convert_float_to_float16` with `keep_io_types=True` (as E2) gave a graph whose CPU output agrees
with float32 (IoU >= 0.9994), but whose DirectML output is NaN on 14 of 14 photos, the IoU scores
included. Investigation:

- Module outputs of the encoder reach about 12,000 (layers 10 and 11, residual stream): within
  the float16 range.
- All 735 float32 intermediate tensors on `DSC_1535` (ONNX Runtime CPU): only one is above 65,504,
  `node_MatMul_148`, the attention scores of encoder layer 0 (134,850).
- Score maxima per attention layer on the 14 photos (float32): layer 0 134,699 to 135,151; the
  other ten layers 27 to 482 (`export_report.json`, `attention_score_maxima`).
- Bisection of the float16 graph on DirectML: the first non-finite tensor is `node_MatMul_148`.
- ONNX Runtime CPU hides the overflow: it has no float16 kernel for many operators and runs them in
  float32 (warnings "Could not find a CPU kernel").
- The converter puts a Cast to float16 and a Cast back to float32 between two blocked nodes. With
  the score MatMul and the Softmax both in float32, this pair makes Inf again. The script removes
  it.

Rule of `export_onnx.py`: the score MatMul and its Softmax stay in float32 when their largest
absolute score on the calibration photos (the 14 quality photos) is above 65,504 / 16 = 4,094.
Result: layer 0 only (2 nodes, 1 round trip removed; 722 nodes, 217.9 MB). The other layers have
a margin of 136 or more to the float16 maximum on these photos. A graph with all 11 attention
layers in float32 gives the same output and costs more (2,253 ms warm, the same memory; records
`s3od_float16_all_attention_float32_*`).

The layer 0 scores cannot be kept in float16 with a scale factor: at a magnitude of 135,000 the
float16 step is 64 to 128, which is much more than the score differences that the Softmax uses.

### 4.4 Comparison with PyTorch on the CPU

`compare_onnx_outputs.py`, check `OnnxMasksMatchPytorchWithinTolerance`: the three sigmoid masks
at 1024 x 1024 and the three IoU scores, ONNX Runtime 1.31.0 CPU (all optimizations, 4 threads)
against PyTorch 2.14.1 CPU (4 threads) on the same pixels. Requirement for float32: maximum
absolute difference <= 1e-3 for masks and scores, and the same selected mask. Record:
[`s3od_comparison_cpu.json`](../../results/subject_alternatives/s3od_comparison_cpu.json).

| Graph | Maximum mask difference | Maximum IoU score difference | Same selected mask | Minimum IoU of alpha > 0.5 at the photo size | Result |
| --- | ---: | ---: | --- | ---: | --- |
| float32 (both forms) | 1.09e-3 (`P1024439`); 13 photos 4.0e-5 to 9.3e-4 | 5.1e-6 | 14 of 14 | 1.00000 | **fails** on `P1024439` |
| float16 (both forms, CPU) | 0.819 | 8.5e-3 | 14 of 14 | 0.99954 | no requirement |

The two float32 forms give the same values (the removed Resize is an identity). The float16
graphs on the CPU do not show the DirectML behavior (section 4.3). Section 6 compares the
DirectML outputs with PyTorch.

**Float32 rounding of the reference** (`measure_float32_spread.py`, record
[`s3od_float32_spread.json`](../../results/subject_alternatives/s3od_float32_spread.json)),
maximum mask difference:

| Pair | `P1024439` | `DSC00562` |
| --- | ---: | ---: |
| PyTorch 4 threads against PyTorch 1 thread | 1.0e-4 | 1.6e-4 |
| PyTorch `sdpa` attention against PyTorch `eager` attention | 1.2e-4 | **1.09e-3** |
| ONNX Runtime all optimizations against ONNX Runtime no optimization | **2.19e-3** | 7.3e-4 |
| PyTorch (reference) against ONNX Runtime all optimizations | 1.09e-3 | 9.3e-4 |

The two attention kernels of PyTorch give the same mathematics and differ by more than 1e-3. The
large differences are at pixels where the sigmoid changes fast (steep logits at the edge); the
selected mask and the thresholded alpha do not change.

## 5. Cost on DirectML

### 5.1 Method and machine

- Intel Core i7-12700H, 31.8 GB; NVIDIA GeForce RTX 3080 Laptop GPU 8 GB, driver 610.62; Windows
  11. Other desktop programs used about 2.8 to 3.0 GB of dedicated GPU memory.
- onnxruntime-directml 1.24.4 (MVP environment), the session settings of the MVP
  (`mask_pipeline.create_session`: DirectML only, 4 CPU threads, CPU memory arena off, sequential
  execution, no memory patterns, all graph optimizations). One model job at a time.
- `measure_directml_cost.py` under `run_with_gpu_memory.ps1`: session load, first run, 20 warm
  runs on `DSC_1535`, one run per photo (14), release; GPU memory and working set after each
  phase. Times are the `session.run` calls only.
- `profile_operator_placement.py`: executed nodes per execution provider from an ONNX Runtime
  profile (2 runs). The profiled kernel time of a fused DirectML partition does not follow the wall
  time (31 ms profiled against 266 ms measured for one graph), so it is not used.
- `measure_first_runs.py`: 9 new processes per graph, each with a new session and one run, 5 s
  apart; System events 153 and 4101 in the time window of each process.

### 5.2 Results

Records: `results/subject_alternatives/<graph>_cost.json`, `<graph>_cost_gpu_memory.json`,
`<graph>_placement.json`, `<graph>_first_runs.json`. Times in ms; memory in MB (dedicated /
shared) of the process.

| Graph | File | Placement (DirectML partitions, CPU nodes) | Session load | First run | Warm p50 / p95 / maximum (20) | Photos p50 / maximum (14) | After load | After first run | After 20 runs | After release | Peak working set |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |
| BiRefNet lite 1024 float16 (MVP) | 96.4 MB | E2: 601 partitions, 724 CPU nodes (this profiling run stopped with `887A0006`) | 9,442 | 817 | 154 / 178 / 187 | 153 / 200 | 227 / 180 | 3,226 / 204 | 3,226 / 301 | 107 / 232 | 1,144 |
| S3OD float32, CPU Resize | 433.8 MB | 4, 1 (Resize) | 1,014 | 863 | 643 / 673 / 675 | 651 / 679 | 698 / 450 | 3,549 / 976 | 3,549 / 976 | 78 / 38 | 1,944 |
| S3OD float32, one partition | 433.8 MB | 1, 0 | 1,151 | 4,346 | 4,013 / 4,090 / 4,101 | 3,998 / 4,028 | 698 / 449 | 3,805 / 1,503 | 3,805 / 1,504 | 78 / 38 | 2,080 |
| S3OD float16, one partition | 217.9 MB | 1, 0 | 988 | 1,111 | 895 / 947 / 971 | 890 / 933 | 295 / 242 | 3,916 / 784 | 3,916 / 784 | 87 / 38 | 1,278 |
| S3OD float16, CPU Resize | 217.9 MB | 4, 3 (Resize, 2 Cast) | 935 | 1,549 | 1,339 / 1,405 / 1,408 | 1,315 / 1,380 | 295 / 243 | 4,047 / 2,048 | 4,047 / 2,048 | 87 / 38 | 3,043 |

- The peak GPU memory of the sampling script is equal to the value after the first run in all
  rows.
- The one-partition float32 graph and both float16 graphs also use 0.8 to 2.0 GB of shared GPU
  memory: DirectML pages part of the working memory to system memory, because the dedicated
  memory of the process plus the other programs (about 3 GB) is near the 8 GB of the GPU. These
  are the slow rows (895 to 4,013 ms).
- The float16 graph with layer 0 attention in float16 (NaN output, not a candidate) took 266 ms
  and 2,379 MB in one partition. The difference to 895 ms and 3,916 MB is the float32 score and
  Softmax tensors of layer 0 (807 MB each).
- The session load of S3OD is 0.9 to 1.2 s for all forms; BiRefNet takes 7.9 to 9.4 s.

### 5.3 First runs and driver resets

| Graph | First runs (new processes) | Failed | Driver resets (event 4101) | First run time |
| --- | ---: | ---: | ---: | --- |
| S3OD float32, CPU Resize | 9 | 0 | 0 | 736 to 754 ms |
| S3OD float32, one partition | 9 | 0 | 0 | 7,965 to 8,366 ms |
| S3OD float16, one partition | 9 | 0 | 0 | 613 to 1,082 ms |
| S3OD float16, CPU Resize | 9 | 0 | 0 | 1,458 to 1,543 ms |
| S3OD float16 forms with layer 0 in float16 (NaN output) | 18 | 0 | 0 | 326 to 575 ms |
| BiRefNet lite 1024 float16 | 9 | 0 | **4** (runs 2, 3, 5, 8) | 495 to 638 ms; 63,802 to 64,383 ms with a reset |

BiRefNet: each reset has `nvlddmkm` event 153 at the start of the run and 60 s and 64 s later,
then event 4101, as in the MVP report. ONNX Runtime returned from these runs without an error;
the run took 64 s. The BiRefNet profiling run (18:30:48 to 18:31:53) stopped with
`DmlFusedNode_0_0 ... 887A0006` and event 4101; the BiRefNet cost run passed. In total in this
step: 5 resets in 11 BiRefNet sessions, 0 in 70 S3OD sessions (54 first runs, 8 cost runs, 8
profiling runs). Record:
[`gpu_reset_events.json`](../../results/subject_alternatives/gpu_reset_events.json).

### 5.4 Against U6

| Model set | Dedicated GPU memory | U6 (3 GB) |
| --- | ---: | --- |
| SAM 2.1 tiny (E1) + BiRefNet lite float16 | 1,355 + 3,226 MB (MVP measured 4,528 MB together) | over |
| SAM 2.1 tiny + S3OD float32, CPU Resize | 1,355 + 3,549 MB | over |
| SAM 2.1 tiny + S3OD float16, one partition | 1,355 + 3,916 MB (+784 MB shared) | over |

No S3OD form of the exported attention meets U6 at 1024 px. The S3OD session releases its memory
when it is deleted (78 to 88 MB left), and a new session takes about 1 s plus the first run (0.7
to 1.1 s). With the U6 rule "unload when the user goes back to adjusting", a Subject call after an
unload costs about 2 s with S3OD and 9 to 10 s with BiRefNet (and a 4 in 9 reset rate). The
attention in query chunks meets U6 (section 5.5).

### 5.5 Attention in query chunks

**Question.** The first version of this report (section 8, item 3) proposed to replace the
materialized attention with `com.microsoft.MultiHeadAttention` (MHA), with the expectation that
DirectML has a fused kernel without the [1, 12, 4101, 4101] score tensor. Does it decrease the
cost?

**One layer** (`probe_attention_layer.py`, record
[`attention_layer0_probe.json`](../../results/subject_alternatives/attention_layer0_probe.json)):
the query, key, and value of encoder layer 0 on `DSC_1535` (after RoPE, from PyTorch), one
attention per graph, the MVP session settings, a new process per graph. Dedicated GPU memory of
the process after the first run; warm p50 of 10 runs. All graphs run as one DirectML node.

| Graph | Float32 memory | Float32 warm | Float16 memory | Float16 warm | Float16 output |
| --- | ---: | ---: | ---: | ---: | --- |
| q + k + v (inputs and device only) | 80 MB | 10.4 ms | | | |
| MHA, one node | 1,159 MB | 29.8 ms | 621 MB | 16.0 ms | not finite |
| MatMul, Mul, Softmax, MatMul | 1,159 MB | 36.6 ms | 621 MB | 25.1 ms | not finite |
| MHA in 4 / 16 query chunks, independent | 1,159 / 2,183 MB | 32.6 / 33.0 ms | | | |
| MatMul form in 4 / 16 query chunks, independent | 1,159 / 1,159 MB | 36.5 / 35.4 ms | | | |
| MHA in 4 / 16 query chunks, chained | **391 / 263 MB** | 29.4 / 31.1 ms | | | |
| MatMul form in 4 / 16 query chunks, chained | 391 / 263 MB | 37.5 / 37.2 ms | | | |

- The DirectML MHA uses the same memory as the materialized attention, and its float16 output
  is not finite on layer 0, as the MatMul form. So on this GPU and driver it does not keep the
  scores out of memory, and it does not compute them in float32. It takes 19 to 36 % less time.
- All float32 graphs have the same maximum difference to PyTorch (2.89e-3 at values up to 23.2).
- Independent chunks do not help: DirectML keeps the score tensors of all chunks of one fused
  graph at the same time. When chunk k + 1 reads the output of chunk k (it adds
  0 x ReduceMax(output of chunk k) to its query, which is 0 for a finite output), DirectML uses
  the memory of one chunk for all of them.

**Graph change.** `s3od_graphs.use_query_chunked_attention` replaces the attention of each encoder
layer of the PyTorch model with this chained form before the export (`export_onnx.py
--query-chunks N --attention-kernel multi_head_attention|sdpa`). The kernel
`multi_head_attention` writes one MHA node per chunk (a `torch.library` operator with an ONNX
translation, `num_heads` 12, `scale` 0.125); `sdpa` writes the MatMul form per chunk. The PyTorch
model with the change gives the same masks as the source within 6.5e-5 (`DSC_1535`). Float16:
the same rule as section 4.3, per encoder layer (the layer of a node comes from the
`pkg.torch.onnx.name_scopes` metadata): the 8 MHA nodes of layer 0 stay in float32. The float16
converter writes one Cast per blocked node for the shared key and value with the same node name,
which ONNX Runtime refuses ("two nodes with same node name"); the script merges these 14 Cast
nodes. The 8 chunk graph with MHA: 1,243 nodes, 435.2 MB (float32); 1,265 nodes, 219.3 MB
(float16), one DirectML partition and 0 CPU nodes (records `s3od_mha_chunk8_*_export.json`,
`*_placement.json`). Two exports give the same file (SHA-256 `d85a33e3...` float32,
`0f070b86...` float16).

**Comparison with PyTorch on the CPU** (record
[`s3od_chunk8_comparison_cpu.json`](../../results/subject_alternatives/s3od_chunk8_comparison_cpu.json)):
the same result as section 4.4. The float32 MHA graph has a maximum mask difference of 1.06e-3 on
`P1024439` (the MatMul chunk graph 1.09e-3); the other 13 photos are at most 6.7e-4. The selected
mask is the same on 14 of 14 photos; the minimum IoU at the photo size is 0.999999.

**Cost on DirectML** (`measure_directml_cost.py` under `run_with_gpu_memory.ps1`, records
`results/subject_alternatives/s3od_{mha,sdpa}_chunk<N>_<precision>_cost*.json`). Times in ms;
memory in MB (dedicated / shared) after the first run, which is also the peak of the sampling
script in all rows.

| Graph | File | Session load | First run | Warm p50 / p95 / maximum (20) | Photos p50 / maximum (14) | After load | After first run | Peak working set |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| BiRefNet lite 1024 float16 (section 5.2) | 96.4 MB | 9,442 | 817 | 154 / 178 / 187 | 153 / 200 | 227 / 180 | 3,226 / 204 | 1,144 |
| S3OD float16, exported attention (section 5.2) | 217.9 MB | 988 | 1,111 | 895 / 947 / 971 | 890 / 933 | 295 / 242 | 3,916 / 784 | 1,278 |
| MHA, 4 chunks, float16 | 218.5 MB | 1,060 | 379 | 185 / 187 / 187 | 185 / 188 | 296 / 244 | 2,380 / 273 | 778 |
| **MHA, 8 chunks, float16** | 219.3 MB | 1,111 | 234 | **191 / 193 / 198** | 192 / 213 | 297 / 245 | **1,358 / 274** | 800 |
| MHA, 16 chunks, float16 | 221.0 MB | 1,402 | 283 | 224 / 243 / 243 | 220 / 222 | 300 / 244 | 1,360 / 273 | 806 |
| MHA, 32 chunks, float16 | 224.4 MB | 1,742 | 578 | 278 / 281 / 282 | 280 / 296 | 303 / 246 | 1,363 / 275 | 835 |
| MatMul form, 8 chunks, float16 | 219.6 MB | 1,381 | 449 | 296 / 312 / 328 | 293 / 311 | 296 / 243 | 1,356 / 272 | 792 |
| MHA, 4 chunks, float32 | 434.4 MB | 1,200 | 1,113 | 420 / 435 / 444 | 420 / 432 | 698 / 450 | 2,779 / 479 | 956 |
| MHA, 8 chunks, float32 | 435.2 MB | 1,450 | 1,028 | 413 / 500 / 937 | 410 / 412 | 699 / 450 | 2,780 / 479 | 962 |
| MHA, 16 chunks, float32 | 436.9 MB | 1,401 | 632 | 420 / 423 / 424 | 423 / 435 | 700 / 451 | 2,781 / 480 | 975 |
| MHA, 32 chunks, float32 | 440.3 MB | 1,729 | 631 | 442 / 447 / 448 | 443 / 447 | 703 / 453 | 2,784 / 482 | 1,013 |
| MatMul form, 8 chunks, float32 | 435.5 MB | 1,350 | 608 | 510 / 566 / 619 | 502 / 538 | 699 / 450 | 2,780 / 479 | 975 |

- From 8 chunks (float16) and 4 chunks (float32), more chunks do not decrease the memory: the
  attention is then not the largest part of the working memory. More chunks add time (more nodes).
- MHA takes 35 % less time than the MatMul form at 8 chunks in float16 (191 against 296 ms).
- The float32 graphs no longer page to shared memory as the one-partition float32 graph of
  section 5.2 did (4,013 ms, 1,504 MB shared); one partition and 0 CPU nodes.
- The 8 chunk MHA float16 graph is 1.2 times slower than BiRefNet per call (191 against 154 ms),
  uses 42 % of its GPU memory, and loads 8 times faster.

**First runs** (`measure_first_runs.py`, 9 new processes per graph, records
`s3od_mha_chunk8_*_first_runs.json`): 0 failed runs, 0 driver resets, and no event 153 or 4101 for
the float16 graph (first runs 224 to 247 ms) and the float32 graph (616 to 674 ms).

**The MVP pipeline with S3OD** (`scripts/poc/ai_mask_mvp/measure_pipeline_cost.py --subject-onnx
<8 chunk MHA float16 graph>` under `run_with_gpu_memory.ps1`, records
`mvp_pipeline_s3od_mha_chunk8_float16_cost*.json`): the same sequence as the MVP measurement
(SAM 2.1 tiny encoder and decoder, people and sky head, Sky, People, the E3 strokes, Subject on 14
photos, 20 warm runs), with S3OD as the Subject graph (`mask_pipeline.py` selects the S3OD
processing by the graph outputs `masks` and `iou_scores`).

| Item | MVP with BiRefNet lite float16 ([MVP report 5.2](mvp_ai_mask.md#52-memory)) | MVP with S3OD (8 chunk MHA float16) |
| --- | ---: | ---: |
| Peak dedicated GPU memory | 4,528 MB | **2,666 MB** |
| Peak shared GPU memory | 486 MB | 459 MB |
| Peak working set | 1,873 MB | 1,675 MB |
| Subject graph load | 7,998 ms | 910 ms |
| Subject first call / warm p50 | 417 ms / 155.8 ms | 250 ms / 191.1 ms |
| SAM 2.1 tiny encoder warm p50, decoder box warm p50 | 73.7 ms, 23.2 ms | 70.5 ms, 23.2 ms |

The dedicated memory samples (1.3 s apart) are 230 MB, 455 MB, 1,511 MB, then 2,665 to 2,666 MB
until the end of the run. **2,666 MB is in the 3 GB budget of U6 (406 MB
margin)**, with all graphs loaded.

**Quality** (`probe_subject_masks.py`, record
[`quality_chunked_attention.json`](../../results/subject_alternatives/quality_chunked_attention.json)):
the DirectML alphas of the 8 chunk MHA graphs against S3OD float32 on the CPU: minimum IoU 0.99824
(float16) and 0.99998 (float32), as the float16 and float32 graphs of section 6.1. The means of
the E2 metrics are equal to S3OD float32 CPU within 0.001 (subject recall 0.986, edge width
0.08 px, IoU with BiRefNet 0.845).

## 6. Quality

### 6.1 Method

`probe_subject_masks.py`: all alphas on one grid per photo, the photo aspect with the long edge
1024 px (the S3OD model grid; the BiRefNet 1024 x 1024 alpha is resized to it). The E2 metrics on
alpha > 0.5, and the IoU with the BiRefNet lite float16 CPU mask and with the S3OD float32 CPU
mask. Record: [`quality.json`](../../results/subject_alternatives/quality.json).

**DirectML output against the CPU reference (minimum IoU of 14 photos):** S3OD float32 both forms
0.99998; S3OD float16 both forms 0.99829 (against S3OD float32); BiRefNet lite DirectML 0.9842
(against its CPU output, the onnxruntime-directml difference of the MVP report).

### 6.2 Results

S3OD float32 CPU against BiRefNet lite float16 CPU; the S3OD DirectML outputs give the same
values within 0.001.

| Photo (subject box) | Foreground BiRefNet / S3OD | Subject recall BiRefNet / S3OD | Outside box BiRefNet / S3OD | Edge width px BiRefNet / S3OD | Parts | IoU |
| --- | --- | --- | --- | --- | --- | ---: |
| `DSC_1535` (person) | 0.222 / 0.223 | 0.998 / 0.999 | 0.000 / 0.000 | 1.57 / 0.05 | 1 / 1 | 0.992 |
| `DSC00326` (person) | 0.026 / 0.026 | 0.989 / 0.991 | 0.000 / 0.000 | 1.32 / 0.08 | 1 / 1 | 0.957 |
| `DSC00674` (person) | 0.679 / 0.680 | 0.999 / 1.000 | 0.005 / 0.005 | 5.44 / 0.08 | 1 / 1 | 0.995 |
| `DSC00936` (person) | 0.022 / 0.022 | 0.996 / 0.992 | 0.000 / 0.000 | 1.23 / 0.04 | 1 / 1 | 0.982 |
| `P1024439` (bassist) | 0.262 / 0.061 | 0.996 / 0.985 | **0.753** / 0.019 | 1.67 / 0.13 | 3 / 1 | 0.230 |
| `DSC_1188` (squirrel) | 0.110 / 0.107 | 0.996 / 0.994 | 0.003 / 0.001 | 2.05 / 0.08 | 1 / 1 | 0.970 |
| `DSC00527` (kneeling woman) | 0.024 / 0.023 | 0.997 / 0.993 | 0.558 / 0.538 | 0.99 / 0.09 | 2 / 2 | 0.928 |
| `2088064441` (boarder) | 0.005 / 0.004 | 0.981 / 0.957 | 0.001 / 0.000 | 1.38 / 0.08 | 1 / 1 | 0.863 |
| `DSC00728` (balloon) | 0.003 / 0.003 | 0.992 / 0.977 | 0.000 / 0.000 | 0.99 / 0.03 | 1 / 1 | 0.959 |
| `5761824892` (car) | 0.241 / 0.241 | 1.000 / 1.000 | 0.075 / 0.073 | 1.45 / 0.09 | 1 / 1 | 0.991 |
| `DSC00562` (duck on the wall) | 0.370 / 0.369 | 0.970 / 0.952 | 0.995 / 0.995 | 1.94 / 0.11 | 1 / 1 | 0.988 |
| `DSC00602` (tram) | 0.273 / 0.270 | 1.000 / 0.998 | 0.014 / 0.012 | 1.58 / 0.09 | 1 / 1 | 0.985 |
| `DSC_1456` (car) | 0.065 / 0.065 | 0.997 / 0.999 | 0.002 / 0.000 | 1.35 / 0.05 | 1 / 1 | 0.986 |
| `Nikon-D800e-raw-00007` (rock) | **0.000** / 0.015 | **0.000** / 0.969 | 0.000 / 0.017 | — / 0.12 | 0 / 1 | 0.000 |
| Mean | 0.164 / 0.151 | 0.922 / 0.986 | 0.172 / 0.119 | 1.64 / 0.08 | 1.14 / 1.07 | 0.845 |

- Both models select the same subjects on 12 photos (IoU 0.86 to 0.995). "Outside box" is high
  for both on `DSC00562` (all ducks) and `DSC00527` (several people); both take all salient
  subjects, as F1 requires.
- `P1024439`: BiRefNet adds a large background area (3 parts, 75 % outside the bassist box);
  S3OD takes the bassist only. `Nikon-D800e-raw-00007`: BiRefNet finds nothing; S3OD takes the
  rock.
- Small subjects: S3OD recall is a little lower (boarder 0.957 against 0.981, balloon 0.977
  against 0.992).
- **Edge.** S3OD gives an almost binary alpha: soft share 0.000 (BiRefNet 0.006), edge width 0.03
  to 0.13 px against 0.99 to 5.44 px. In the hair and fur regions of E2 the soft share is 0.0005
  to 0.0023 (BiRefNet 0.024 to 0.039); the mean alpha in these regions is the same within 0.05.
  On `DSC_1535` S3OD follows the curly hair outline closely with a hard edge; at the 4096 px
  photo the edge has the 4 px steps of the 1024 grid. BiRefNet has a soft band of about 10 to
  20 px around the hair that also covers background between the curls. Edge post-processing is
  rejected (U3), so the S3OD edge is the final edge.

### 6.3 Sheets for the check by eye (local only)

```bash
E=build/tmp/explore; S=$E/s3od; D=$S/directml
$E/venv_birefnet/Scripts/python -I scripts/ai_models/s3od/make_subject_sheets.py \
  --photos $E/e4/input_4096/photos --prompts build/tmp/sm1/quality/prompts.json \
  --alpha "BiRefNet lite float16 (DirectML)=birefnet:$D/birefnet_lite_float16/alpha" \
  --alpha "S3OD float16 (DirectML)=s3od:$D/s3od_float16/alpha" \
  --alpha "S3OD float32 (DirectML)=s3od:$D/s3od_cpu_resize_float32/alpha" \
  --crop "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --crop "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
  --crop "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --crop "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
  --crop "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --crop "DSC00936.jpg:person:0.47,0.63,0.68,0.93" \
  --out-dir docs/exploration/images/subject_alternatives
```

The sheets (14 photos, 6 detail crops at the photo resolution; red: alpha > 0.5, yellow: 0.05 to
0.95) are in `docs/exploration/images/subject_alternatives/` (in `.gitignore`).

## 7. License

| Item | License | Commercial use | Evidence |
| --- | --- | --- | --- |
| S3OD weights `s3od-synth` | MIT (model card `license: mit`) | yes, for the weights as a work | `README.md` SHA-256 `d8d7a46d...` |
| S3OD code | MIT, "Copyright (c) 2024 S3OD Team" | yes | `LICENSE` 1,067 bytes, SHA-256 `bcb7a447f69786d7...` |
| DINOv3 ViT-B/16 (the encoder in the weights, fine-tuned) | DINOv3 License: use, reproduce, distribute, and derivative works permitted; a distribution must include the license text (section 1.b.i) | yes, with the license text | `facebookresearch_dinov3_LICENSE.md` 7,503 bytes, SHA-256 `25d122eb...` (survey evidence `build/tmp/subject_alternatives/sod/`) |
| Training images of `s3od-synth` | Generated with FLUX.1 Krea [dev] (paper, appendix; model README: "Trained exclusively on synthetic data") | **legal check necessary**: the FLUX.1 [dev] Non-Commercial License permits commercial use of outputs (section 2.d) but its restrictions forbid commercial use of "any data produced by the FLUX.1 [dev] Model" (section 4.a) | `FLUX1-dev_LICENSE.txt` SHA-256 `6f7b3d94...`; a separate FLUX.1 Krea [dev] license file was not downloaded |
| Labels of the training images | From a teacher decoder "supervising it with DIS-5K, HR-SOD, UHRSOD, and DUTS datasets" (paper section 4.1) | DIS5K: non-commercial terms; DUTS: ImageNet research terms (survey 2026-10-21) | `2510.21605.txt` |

The weights have a license that permits commercial use. The training data lineage has the same
problem as BiRefNet (DIS5K) and adds the FLUX output terms. This is open decision U13 of issue #6.

## 8. Recommendation

**User decision (2026-10-22, README U14): F1 and F2 use S3OD, the 8 chunk MHA float16 graph of
section 5.5.** The [acceptance record](acceptance_ai_mask_model_set.md) gives the evidence of the
model set and the open items.

- **The "no driver reset" criterion: S3OD meets it** (0 resets in 70 S3OD sessions; BiRefNet 5
  in 11 in the same period). This supports the user's suspicion that the BiRefNet graph causes the
  resets: S3OD has no deformable convolution and no GridSample, and its graph is one DirectML
  partition.
- **The U6 criterion: S3OD meets it with the attention in query chunks (section 5.5).** The
  best form is MHA in 8 chained query chunks, float16 with layer 0 in float32: 191 ms warm,
  1,358 MB, 1.1 s load; the MVP pipeline with this graph has a peak of 2,666 MB. With the
  exported attention, S3OD did not meet U6 (643 to 895 ms, 3.5 to 3.9 GB).
- **Quality: equal or better on the 14 photos**, with hard edges. Whether the hard edge on hair is
  acceptable is the user's check by eye (U3 forbids edge post-processing).
- **IS-Net general-use is not a replacement for BiRefNet lite** (section 11.8): it meets the
  cost and reset criteria with the largest margin (437 MB, 14 ms, MVP pipeline 1,748 MB), but its
  masks are worse on 7 of 14 photos, and its weights have no license. Of the measured candidates,
  S3OD (8 chunk MHA float16) stays the recommendation.
- **Alibaba universal matting is not a replacement for BiRefNet lite** (section 12.8): no driver
  reset, but the MVP pipeline with it has a peak of 3,982 MB (U6 not met), and its masks are worse
  on 5 of 14 photos. S3OD stays the recommendation of the three candidates.
- **Decisions for the user:**
  1. The float32 tolerance: accept 1.09e-3 on `P1024439` with the cause of section 4.4 (the
     rounding of the reference is larger than 1e-3), or set a tolerance for this model, for
     example "same selected mask and IoU of alpha > 0.5 >= 0.999 at the photo size" (measured:
     1.00000).
  2. The training data rule U13 (FLUX output terms and teacher labels, section 7).
  3. The S3OD graph for the Subject model: the 8 chunk MHA float16 graph of section 5.5 (or
     the float32 graph: 413 ms, 2,780 MB, the same output as the CPU float32 graph). The
     expectation of the first version of this item (a fused DirectML MHA kernel without the
     score tensors) was not correct: the DirectML MHA keeps the full score tensor (section 5.5);
     the decrease comes from the chained query chunks.

## 9. Open items

- All three candidates of issue #6 are measured. The open items of IS-Net general-use are in
  section 11.9, those of Alibaba universal matting in section 12.9.
- Check of the masks by eye (sheets of section 6.3).
- The tolerance decision (section 8, item 1) and U13.
- Not measured: S3OD at another input size (the issue forbids a smaller input as a substitute).
  The part of the 1,358 MB working memory of the 8 chunk float16 graph that is not attention
  (probably the DPT decoder at 512 and 1024 px) is not separated. The chained query chunks are
  not measured with the CPU Resize form of section 4.2. `ai_mask_app.py` with `--subject-onnx`
  and the S3OD graph uses the S3OD processing of `mask_pipeline.py`; the app is not tested with
  it.
- `P1024439` and `Nikon-D800e-raw-00007` show the largest differences between the models; the
  quality set has no photo without a subject, so the "not found" rule of F1 is not measured for
  S3OD (the largest IoU score of S3OD is 0.933 to 0.993 on the 14 photos; it could be the input
  of such a rule).

## 10. Commands

The IS-Net commands are in
[`scripts/ai_models/isnet/README.md`](../../scripts/ai_models/isnet/README.md#commands), the
universal matting commands in
[`scripts/ai_models/universal_matting/README.md`](../../scripts/ai_models/universal_matting/README.md#commands).
The S3OD
commands are in [`scripts/ai_models/s3od/README.md`](../../scripts/ai_models/s3od/README.md#commands):
sources and weights, export of both forms, the CPU comparison, the float32 spread, placement,
first runs, cost with GPU memory, quality metrics, and the attention in query chunks of section
5.5 (the one layer probe, the export, and the MVP pipeline cost). The investigation of section 4 used ad hoc
scripts in `build/tmp/s3od_synth/` (single-node DirectML tests, graph cut bisection for the
session error and for the first non-finite tensor, the intermediate tensor ranges); their results
are in this report, and `export_onnx.py` records the attention score maxima in
`export_report.json`.

## 11. IS-Net general-use

**Date:** 2026-10-21. **Code:** [`scripts/ai_models/isnet/`](../../scripts/ai_models/isnet/README.md)
(export, CPU comparison) and the shared DirectML and quality scripts of
[`scripts/ai_models/s3od/`](../../scripts/ai_models/s3od/README.md) with `--model isnet`. The
questions are the questions of section 1, with the same photos, machine, and settings (sections 3
and 5.1).

### 11.1 Result summary

| Item | IS-Net general-use float16 | IS-Net float32 | S3OD 8 chunk MHA float16 (section 5.5) | BiRefNet lite float16 (section 5.2) |
| --- | ---: | ---: | ---: | ---: |
| Float32 ONNX against PyTorch, maximum difference (14 photos) | | 9.1e-6 (passes 1e-3) | 1.06e-3 (fails on 1 photo) | |
| DirectML partitions, CPU nodes | 1, 0 | 1, 0 | 1, 0 | 601, 724 (E2) |
| Session load | 834 ms | 908 ms | 1,111 ms | 9,442 ms |
| Per photo, p50 (14 photos) | **13.8 ms** | 27.4 ms | 192 ms | 153 ms |
| Dedicated GPU memory after the first run (peak) | **437 MB** | 1,000 MB | 1,358 MB | 3,226 MB |
| Driver resets in 9 first runs | 0 | 0 | 0 | 4 |
| MVP pipeline peak dedicated GPU memory | **1,748 MB** | | 2,666 MB | 4,528 MB |
| Mean subject recall (14 photos) | 0.802 (PyTorch) | | 0.986 (S3OD float32) | 0.922 |
| Weights license | **none stated** | | MIT | MIT |

- IS-Net meets U6 with the largest margin of the measured models (1,748 MB with SAM 2.1 tiny and
  the head) and has no driver reset in 23 sessions.
- Its masks are worse than both BiRefNet lite and S3OD on 7 of the 14 photos (section 11.6).
- The weights have no license statement (section 11.7).

### 11.2 Sources

| Item | Value |
| --- | --- |
| Weights | `isnet-general-use.pth`, 176,579,397 bytes, SHA-256 `9e1aafea58f0b55d0c35077e0ceade6ba1ba2bce372fd4f8f77215391f3fac13`, Google Drive file `1XHIzgTzY5BQHw140EDIgwIb53K659ENH` (server file name `isnet-general-use.pth`), downloaded 2026-10-21 |
| Code | `xuebinqin/DIS` at `b6764e20381f6f42a70f83fa3324181529ed1403`: `IS-Net/models/isnet.py` (SHA-256 `83505426...`), `IS-Net/Inference.py` (`e4d4c3b7...`), `LICENSE.md` Apache-2.0 (`c71d239d...`) |
| Model | `ISNetDIS()`: RSU U-Net (U2-Net family), 6 encoder and 5 decoder stages, 6 side outputs; 44,046,790 parameters; plain convolutions, max pooling, bilinear upsampling |

The DIS README links the two weight files in a way that is not consistent: line 145 names file
`1XHIzgTz...` `isnet.pth` and file `1klUkUnQ...` `isnet-general-use.pth`, but the server sends
`isnet-general-use.pth` for `1XHIzgTz...` and `isnet.pth` (176,597,693 bytes, SHA-256
`ea088974...`) for `1klUkUnQ...`. Line 28 names `1XHIzgTz...` `isnet-general-use.pth`. This step
uses the file that the server names `isnet-general-use.pth`.

**Processing of the source** (`Inference.py`): stretch the RGB photo to 1024 x 1024 with
`F.upsample(mode="bilinear")` (no antialias, so a 4096 px photo is sampled without a low-pass
filter), truncate to uint8, `value / 255 - 0.5`. The output is `result[0][0]`, the sigmoid of side
output d1. The source resizes it to the photo size and scales it to [0, 1] with its minimum and
maximum. `isnet_graphs.preprocess` gives the same tensor as the statements of `Inference.py`
(difference 0.0 on 14 photos). The stored alpha of this step is d1 scaled at 1024 x 1024 and
resized after that; the source order (resize, then scale) differs by at most 3e-4 on 13 photos
and by 3.1e-2 on `Nikon-D800e-raw-00007`, where the d1 maximum is 0.047 (record
`isnet_comparison_cpu.json`, `source_pipeline_max_abs_diff`).

### 11.3 Export and comparison with PyTorch

- TorchScript exporter (`torch.onnx.export(..., dynamo=False)`, torch 2.14.1, opset 17), 3.2 s,
  no graph change: 593 nodes (Conv 114, Relu 112, Concat 84, Resize 34, MaxPool 33, Shape 34,
  Slice 34, Add 11, Sigmoid 1, Constant 136), no GridSample. The exporter folds the BatchNorm
  layers into the convolutions and removes the side outputs d2 to d6 (44,004,737 parameters in the
  graph). Float32 176.1 MB (SHA-256 `f3622a65...`), float16 88.2 MB (`31cf8b2d...`). Records:
  [`isnet_float32_export.json`](../../results/subject_alternatives/isnet_float32_export.json),
  [`isnet_float16_export.json`](../../results/subject_alternatives/isnet_float16_export.json).
- **Float16 range.** `convert_float_to_float16` with `keep_io_types=True` and the default block
  list; no node block list. The largest PyTorch module output on the 14 photos that is a graph
  tensor is the d1 logit, 1,789 (limit of the S3OD rule: 4,094). The convolutions before their
  BatchNorm reach 29,832, but these tensors are not in the graph (folded). The DirectML float16
  output is finite on all photos (section 11.5).
- **`OnnxD1MatchesPytorchWithinTolerance`** (`compare_onnx_outputs.py`, ONNX Runtime 1.31.0 CPU
  against PyTorch CPU, 4 threads, same pixels; record
  [`isnet_comparison_cpu.json`](../../results/subject_alternatives/isnet_comparison_cpu.json)):

  | Graph | Maximum d1 difference | Maximum alpha difference at the photo size | Minimum IoU of alpha > 0.5 at the photo size | Result |
  | --- | ---: | ---: | ---: | --- |
  | float32 | 9.1e-6 (`P1024439`); 13 photos 3.6e-7 to 5.0e-6 | 8.9e-6 | 0.999997 | **passes** (1e-3) |
  | float16 (CPU) | 6.2e-3 | 6.1e-3 | 0.99682 (`DSC00527`) | no requirement |

### 11.4 Cost on DirectML

`profile_operator_placement.py`, `measure_directml_cost.py` under `run_with_gpu_memory.ps1`, and
`measure_first_runs.py` (section 5.1). Records
`results/subject_alternatives/isnet_<precision>_*.json`. Times in ms; memory in MB (dedicated /
shared) of the process.

| Graph | File | Placement | Session load | First run | Warm p50 / p95 / maximum (20) | Photos p50 / maximum (14) | After load | After first run | After release | Peak working set |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| BiRefNet lite 1024 float16 (section 5.2) | 96.4 MB | 601, 724 (E2) | 9,442 | 817 | 154 / 178 / 187 | 153 / 200 | 227 / 180 | 3,226 / 204 | 107 / 232 | 1,144 |
| S3OD 8 chunk MHA float16 (section 5.5) | 219.3 MB | 1, 0 | 1,111 | 234 | 191 / 193 / 198 | 192 / 213 | 297 / 245 | 1,358 / 274 | | 800 |
| IS-Net float32 | 176.1 MB | 1, 0 | 908 | 63 | 29.2 / 57.4 / 58.6 | 27.4 / 28.2 | 467 / 203 | 1,000 / 220 | 74 / 36 | 827 |
| **IS-Net float16** | 88.2 MB | 1, 0 | 834 | 32 | 23.9 / 38.9 / 39.0 | **13.8 / 14.6** | 161 / 120 | **437 / 137** | 77 / 36 | 763 |

- The peak dedicated memory of the sampling script is equal to the value after the first run.
- The warm runs on `DSC_1535` get faster during the 20 runs (float16: 37 to 39 ms for the first 5
  runs, then 23 to 25 ms, then 13.4 to 14.4 ms from run 14; float32: 58 ms, then 28 to 29 ms from
  run 8). The per-photo runs after them are 13.2 to 14.6 ms (float16) and 26.9 to 28.2 ms
  (float32). The cause is probably the GPU clock; it was not measured.
- The profiled kernel time is 5.6 ms (float16) and 10.4 ms (float32) per run; it is not used
  (section 5.1).

**First runs** (9 new processes per graph): 0 failed runs, 0 driver resets, and no event 153 or
4101. Load 856 to 988 ms and first run 24.8 to 26.1 ms (float16); load 848 to 985 ms and first run
49.0 to 54.7 ms (float32). In total in this step: 0 resets in 23 IS-Net sessions (18 first runs,
2 cost runs, 2 profiling runs, 1 pipeline run).

**BiRefNet reference in the same period.** Two runs of `measure_directml_cost.py` with the
BiRefNet lite float16 graph (the issue asks for the reference in the same runs) both stopped in
the warm runs with `DmlFusedNode_0_0 ... 887A0006` after a display driver reset (event 4101):

- run 1 (22:27:50): first run 63,906 ms with events 153 at 22:28:00, 22:29:00, and 22:29:04,
  event 4101 at 22:29:05, then the error on the first warm run;
- run 2 (22:30:08): first run 551 ms, then a warm run hung (event 153 at 22:30:21, event 4101 at
  22:31:26) and stopped with the same error.

The script wrote no cost JSON. Record:
[`birefnet_lite_float16_isnet_step_failed_runs.json`](../../results/subject_alternatives/birefnet_lite_float16_isnet_step_failed_runs.json)
and the GPU memory records `birefnet_lite_float16_isnet_step_attempt{1,2}_cost_gpu_memory.json`
(peak 3,226 and 3,227 MB). The BiRefNet values of the tables are from section 5.2 (same day, same
machine). Run 2 shows that a reset can also occur after a first run that passed. Issue #6 in
total: BiRefNet 7 resets in 13 sessions; S3OD 0 in 88; IS-Net 0 in 23.

### 11.5 MVP pipeline with IS-Net and DirectML output

`measure_pipeline_cost.py --subject-onnx <IS-Net float16 graph>` under `run_with_gpu_memory.ps1`
(the sequence of section 5.5; `mask_pipeline.py` selects the IS-Net processing by the graph output
`d1`). Records `mvp_pipeline_isnet_float16_cost*.json`.

| Item | MVP with BiRefNet lite float16 | MVP with S3OD (8 chunk MHA float16) | MVP with IS-Net float16 |
| --- | ---: | ---: | ---: |
| Peak dedicated GPU memory | 4,528 MB | 2,666 MB | **1,748 MB** |
| Peak shared GPU memory | 486 MB | 459 MB | 325 MB |
| Peak working set | 1,873 MB | 1,675 MB | 1,476 MB |
| Subject graph load | 7,998 ms | 910 ms | 507 ms |
| Subject first call / warm p50 | 417 ms / 155.8 ms | 250 ms / 191.1 ms | 30.7 ms / 13.9 ms |
| SAM 2.1 tiny encoder warm p50, decoder box warm p50 | 73.7 ms, 23.2 ms | 70.5 ms, 23.2 ms | 72.8 ms, 21.2 ms |

1,748 MB is in the 3 GB budget of U6 with a margin of 1,324 MB. No event 153 or 4101 occurred
during the run.

**DirectML output against PyTorch** (stored alphas, float16): float32 maximum alpha difference
4.9e-4 (the float16 storage step), minimum IoU of alpha > 0.5 on the 1024 grid 0.99999. Float16:
maximum alpha difference 2.9e-2, minimum IoU 0.9601 (`DSC00527`, where most of the alpha is near
0.5) and 0.9643 (`Nikon-D800e-raw-00007`, foreground 0.008 %); the other 12 photos 0.9957 or more.
The means of the E2 metrics of float16 DirectML are equal to PyTorch within 0.004 (the edge width
within 1.5 px of 88 px).

### 11.6 Quality

`probe_subject_masks.py` (section 6.1; kind `isnet`: the 1024 x 1024 alpha resized to the 1024
grid of the photo aspect). Record
[`quality_isnet.json`](../../results/subject_alternatives/quality_isnet.json). The IS-Net values
are the PyTorch alpha; the ONNX float32 alphas (CPU and DirectML) give the same values within
0.001. B: BiRefNet lite float16 CPU; S: S3OD float32 CPU; I: IS-Net.

| Photo (subject box) | Foreground B / S / I | Subject recall B / S / I | Outside box B / S / I | Edge width px B / S / I | Parts B / S / I | IoU I with B / S |
| --- | --- | --- | --- | --- | --- | --- |
| `DSC_1535` (person) | 0.222 / 0.223 / 0.223 | 0.998 / 0.999 / 0.999 | 0.000 / 0.000 / 0.000 | 1.57 / 0.05 / 1.72 | 1 / 1 / 1 | 0.995 / 0.992 |
| `DSC00326` (person) | 0.026 / 0.026 / 0.026 | 0.989 / 0.991 / 0.986 | 0.000 / 0.000 / 0.000 | 1.32 / 0.08 / 2.90 | 1 / 1 / 1 | 0.979 / 0.953 |
| `DSC00674` (person) | 0.679 / 0.680 / 0.662 | 0.999 / 1.000 / **0.962** | 0.005 / 0.005 / 0.005 | 5.44 / 0.08 / 31.31 | 1 / 1 / 1 | 0.944 / 0.943 |
| `DSC00936` (person) | 0.022 / 0.022 / 0.022 | 0.996 / 0.992 / 0.994 | 0.000 / 0.000 / 0.000 | 1.23 / 0.04 / 1.47 | 1 / 1 / 1 | 0.992 / 0.978 |
| `P1024439` (bassist) | 0.262 / 0.061 / 0.243 | 0.996 / 0.985 / 0.996 | 0.753 / 0.019 / 0.725 | 1.67 / 0.13 / 16.01 | 3 / 1 / 4 | 0.724 / 0.247 |
| `DSC_1188` (squirrel) | 0.110 / 0.107 / **0.359** | 0.996 / 0.994 / 0.998 | 0.003 / 0.001 / **0.675** | 2.05 / 0.08 / 32.58 | 1 / 1 / 1 | 0.301 / 0.297 |
| `DSC00527` (kneeling woman) | 0.024 / 0.023 / 0.015 | 0.997 / 0.993 / **0.474** | 0.558 / 0.538 / 0.644 | 0.99 / 0.09 / 6.29 | 2 / 2 / 4 | 0.606 / 0.620 |
| `2088064441` (boarder) | 0.005 / 0.004 / 0.006 | 0.981 / 0.957 / 0.930 | 0.001 / 0.000 / **0.232** | 1.38 / 0.08 / 8.34 | 1 / 1 / 1 | 0.673 / 0.607 |
| `DSC00728` (balloon) | 0.003 / 0.003 / 0.003 | 0.992 / 0.977 / 0.986 | 0.000 / 0.000 / 0.000 | 0.99 / 0.03 / **818** | 1 / 1 / 1 | 0.986 / 0.962 |
| `5761824892` (car) | 0.241 / 0.241 / 0.223 | 1.000 / 1.000 / **0.911** | 0.075 / 0.073 / 0.086 | 1.45 / 0.09 / 4.64 | 1 / 1 / 1 | 0.903 / 0.900 |
| `DSC00562` (duck on the wall) | 0.370 / 0.369 / **0.018** | 0.970 / 0.952 / **0.000** | 0.995 / 0.995 / 1.000 | 1.94 / 0.11 / 69.24 | 1 / 1 / 1 | 0.000 / 0.000 |
| `DSC00602` (tram) | 0.273 / 0.270 / 0.272 | 1.000 / 0.998 / 0.993 | 0.014 / 0.012 / 0.017 | 1.58 / 0.09 / 14.00 | 1 / 1 / 1 | 0.955 / 0.958 |
| `DSC_1456` (car) | 0.065 / 0.065 / 0.065 | 0.997 / 0.999 / 0.996 | 0.002 / 0.000 / 0.012 | 1.35 / 0.05 / 5.77 | 1 / 1 / 1 | 0.984 / 0.974 |
| `Nikon-D800e-raw-00007` (rock) | 0.000 / 0.015 / 0.000 | 0.000 / 0.969 / 0.000 | 0.000 / 0.017 / 1.000 | — / 0.12 / 219 | 0 / 1 / 1 | 0.000 / 0.000 |
| Mean | 0.164 / 0.151 / 0.153 | 0.922 / 0.986 / **0.802** | 0.172 / 0.119 / **0.314** | 1.64 / 0.08 / 88.0 | 1.14 / 1.07 / 1.43 | 0.717 / 0.674 |

Soft share (0.05 < alpha < 0.95), mean: B 0.006, S 0.0002, I **0.096**. Mean hole fraction: B
0.004, S 0.008, I 0.020.

What the local sheets show (section 11.10), in words:

- **Soft areas on the background.** `DSC_1188`: IS-Net takes the squirrel and also the tree
  stump under it, most of the stump with an alpha of 0.05 to 0.95 and a part above 0.5.
  `DSC00728`: the balloon is correct, but the cliff on the left has an alpha between 0.05 and 0.5
  over its full area (24 % of the photo is soft; hence the edge width of 818 px). `2088064441`:
  the reflection of the boarder in the water is in the mask, the boarder is partly soft, and dry
  grass stems on the left have a soft alpha.
- **Missed or partial subjects.** `DSC00562`: BiRefNet and S3OD take the wall with the waterfall
  and the ducks; IS-Net takes a part of a tower in the background and soft spots on the water,
  not the wall. `DSC00527`: the alpha of the three people is mostly between 0.05 and 0.95, so the
  mask > 0.5 has half of the kneeling woman. `5761824892`: the background seen through the car
  windows and an area next to the kayak are soft; holes 8.9 %.
- **Hair.** `DSC_1535`, `DSC00326`, `DSC00936`: the same outline as BiRefNet with a soft edge of
  1.5 to 2.9 px. `DSC00674`: the left side of the hair and the ear are below 0.5 (soft share 0.177
  in the hair region; BiRefNet 0.039).
- **No subject.** `Nikon-D800e-raw-00007`: the d1 maximum is 0.047 (BiRefNet also finds
  nothing). The scaling with the minimum and maximum of the source multiplies the output by 21,
  so the mountains and the clouds get soft areas, and 0.008 % of the photo is above 0.5. A "not
  found" rule of F1 must use d1 before the scaling (the d1 maximum is 0.988 to 1.0 on the other 13
  photos).
- `P1024439`: IS-Net adds the background as BiRefNet does (72.5 % outside the box; S3OD 1.9 %).

IS-Net is equal to BiRefNet on 5 photos (`DSC_1535`, `DSC00326`, `DSC00936`, `DSC00602`,
`DSC_1456`: IoU 0.955 to 0.995), similar to BiRefNet on 2 (`P1024439`, `Nikon-D800e-raw-00007`),
and worse than both models on 7 (`DSC00674`, `DSC_1188`, `DSC00527`, `2088064441`, `DSC00728`,
`5761824892`, `DSC00562`).

### 11.7 License

| Item | License | Commercial use | Evidence |
| --- | --- | --- | --- |
| IS-Net code | Apache-2.0 ("Our code and evaluation metric use Apache License 2.0", README line 170) | yes | `LICENSE.md` 11,357 bytes, SHA-256 `c71d239d...` |
| `isnet-general-use.pth` | **No statement.** The Apache-2.0 statement names the code and the evaluation metric only | **unknown** | [DIS#150](https://github.com/xuebinqin/DIS/issues/150) (opened 2026-08-20, open): two requests for the commercial use of these weights, no answer from the author on 2026-10-21 |
| Training data of the general-use weights | Not stated ("for general use, this is NOT DIS V2.0") | unknown | README line 28 |
| DIS5K (training data of the academic `isnet.pth`) | DIS5K Terms of Use, non-commercial | no | `DIS5K-Dataset-Terms-of-Use.pdf` SHA-256 `d509ad22...` |

Without a weights license, the weights are not a candidate under the license rule of README
section 2. The weights license is an open decision of issue #6; this step does not change it.

### 11.8 Recommendation

- **IS-Net general-use is not a replacement for BiRefNet lite.** It meets U6 and the "no driver
  reset" criterion with the largest margin of the measured models (437 MB, 14 ms, 0.8 s load, MVP
  pipeline 1,748 MB, 0 resets in 23 sessions). But:
  1. the masks are worse than BiRefNet and S3OD on 7 of 14 photos (mean subject recall 0.80,
     soft background areas, one subject not found). The soft alpha is not a matte of the edge but
     a low confidence over large areas, and U3 (no edge post-processing) does not permit a
     correction;
  2. the weights have no license statement.
- **S3OD (8 chunk MHA float16) stays the recommended alternative** of the measured candidates
  (section 8).
- The low cost of IS-Net comes from its plain convolutions at 1024 px (one DirectML partition, no
  attention). This supports section 8: the BiRefNet cost and resets come from its graph
  (deformable convolution as GridSample, 724 CPU nodes), not from the 1024 px input.

### 11.9 Open items

- The weights license of `isnet-general-use.pth` (DIS#150).
- The check of the masks by eye (sheets of section 11.10).
- Not measured: IS-Net with an antialiased downscaling of the photo (the source has none, and the
  issue requires the preprocessing of the source); the academic `isnet.pth`; the rembg ONNX file
  `isnet-general-use.onnx` (the issue asks for the export from the official checkpoint). The
  cause of the faster warm runs (section 11.4) is not measured. `ai_mask_app.py` with the IS-Net
  graph is not tested (its labels name BiRefNet); the pipeline code is tested through
  `measure_pipeline_cost.py`.
- The BiRefNet first runs were not repeated in this step (section 5.3 has 9 runs of the same
  day); its two cost runs of this step reset the driver (section 11.4).

### 11.10 Sheets for the check by eye (local only)

The command is in
[`scripts/ai_models/isnet/README.md`](../../scripts/ai_models/isnet/README.md#commands) (last
block). The sheets (14 photos and 6 detail crops; panels: BiRefNet lite float16 CPU, S3OD 8 chunk
MHA float16 DirectML, IS-Net float16 DirectML; red: alpha > 0.5, yellow: 0.05 to 0.95) are in
`docs/exploration/images/subject_alternatives_isnet/` (in `.gitignore`).

## 12. Alibaba universal matting

**Date:** 2026-10-22. **Code:**
[`scripts/ai_models/universal_matting/`](../../scripts/ai_models/universal_matting/README.md)
(conversion, CPU comparison, TensorFlow spread) and the shared DirectML and quality scripts of
[`scripts/ai_models/s3od/`](../../scripts/ai_models/s3od/README.md) with
`--model universal_matting`. The questions are the questions of section 1, with the same photos,
machine, and settings (sections 3 and 5.1).

### 12.1 Result summary

| Item | Universal matting float32 | Universal matting float16 | S3OD 8 chunk MHA float16 | BiRefNet lite float16 |
| --- | ---: | ---: | ---: | ---: |
| Input | the photo (4096 px) | the photo (4096 px) | 1024 x 1024 | 1024 x 1024 |
| Session load | 1,513 ms | 1,827 ms | 1,111 ms | 9,442 ms |
| Per photo, p50 / maximum (14 photos) | 126 / 248 ms | 130 / 167 ms | 192 / 213 ms | 153 / 200 ms |
| Dedicated GPU memory, peak of the cost run | 2,678 MB | 2,664 MB | 1,358 MB | 3,226 MB |
| Driver resets in 9 first runs | 0 | 0 | 0 | 4 |
| MVP pipeline peak dedicated GPU memory | **3,982 MB** | | 2,666 MB | 4,528 MB |
| Mean subject recall (14 photos) | 0.838 (TensorFlow) | | 0.986 (S3OD float32) | 0.922 |
| Float32 ONNX against the framework (1e-3) | **fails** (6.3e-3; TensorFlow against itself 7.2e-3) | | fails (1.09e-3) | |
| Weights license | Apache License 2.0 (model card) | | MIT | MIT |

- **U6 is not met.** The graph reads the photo at its size and keeps tensors of the photo size,
  so the MVP pipeline with it has a peak of 3,982 MB dedicated GPU memory (3 GB budget). No
  driver reset occurred in 46 sessions.
- **The masks are worse than BiRefNet lite and S3OD on 5 of the 14 photos**: large background
  areas on 2 photos (the bridge and the water of `2088064441`, the buildings and trees of
  `DSC00562`), a wide soft band on the squirrel tail, parts of the mountains on the photo where
  BiRefNet finds no subject (section 12.6).
- **The conversion needs graph rewrites.** `tf2onnx` 1.17 converts `ResizeArea`,
  `ResizeBilinear`, and `CropAndResize` to other operations and fails on a constant `tf.cond`.
  The rewrites keep the TensorFlow kernel arithmetic (0.0 difference against the original graph).
  DirectML float32 division is not correctly rounded: one division moved the last row of a crop
  and changed the alpha of `DSC00674` (IoU 0.936). The export now computes the scalar divisions in
  float64 (section 12.3).
- **Float16 gives no gain.** The instance norms reach 5.2e8, so most U-Net nodes stay float32. On
  DirectML float16 saves 14 MB and no time, and it changes the crop box of two photos (IoU 0.98).

### 12.2 Sources

| Item | Value |
| --- | --- |
| Graph | `tf_graph.pb`, 204,740,209 bytes, SHA-256 `afcad611b6707f8a7e815f5702216af69fa77a72ef470d0b2c25a6afc340ea1f`, ModelScope `iic/cv_unet_universal-matting` master `2e7fe175a9a4a8ce234324b7cccd0a756c75c229` (tag v1.0.0 `de441e1b...`), downloaded 2026-10-21 |
| Model card | `README.md` 4,334 bytes, SHA-256 `d21254bf...` (`license: Apache License 2.0`; tags "CVPR 2020", "semantic human matting"; citation Liu et al., "Boosting semantic human matting with coarse annotations") |
| Pipeline code | `modelscope/modelscope` at `958c3f7da68b700cc671134e267c356a2444398b`: `image_matting_pipeline.py` (SHA-256 `b0333642...`), `preprocessors/image.py` (`ba44792d...`), `LICENSE` Apache-2.0 (`cfc7749b...`) |
| Model | TensorFlow 1 frozen graph: six U-Nets with instance normalization and branch rules in `tf.cond` (`universal_matting_graphs.GRAPH_RULES`); 173 MB of float32 weights |

**Processing of the source** (`ImageMattingPipeline`): read the photo with Pillow,
`exif_transpose`, RGB, `astype(float)`; feed it at its size to `input_image:0`; the output
`output_png:0` is the uint8 RGBA. The graph does all resizing: a person detector at 192 px; for
photos with a person, a crop of 1920 px (AvgPool to 384 px) for a coarse U-Net at 224 px and a
refine U-Net at 480 px; for other photos, or when the human alpha covers less than 10 % of the
photo, an object branch at 224 px and a refine U-Net at 640 px. The alpha is resized to the box
and padded to the photo. The scripts use the float32 RGBA before the uint8 Cast (`output_png` is
its truncation, checked on 14 photos). On the 14 photos the human result is used on `DSC_1535`
and `DSC00674`; the human branch also runs on 4 more photos.

The model card states that the expected result is for images below 2000 x 2000 and that larger
images can have blurred edges. The quality photos have a long edge of 4096 px; the issue requires
the input size of the source, which is the photo.

### 12.3 Conversion and comparison with TensorFlow

- **Rewrites before `tf2onnx` 1.17** (`universal_matting_graphs.rewrite_for_onnx`). `tf2onnx`
  cannot convert the `tf.cond` with a constant in each branch (`cond_2`), converts `ResizeArea`
  to a nearest-neighbor Resize, and writes float scales for a static `ResizeBilinear` (224 to 480
  becomes 479 in ONNX Runtime). ONNX Runtime `Resize` (linear, `align_corners`,
  `tf_crop_and_resize`) also differs from the TensorFlow kernels (1e-4 to 7e-4, and 0.996 in the
  last row of a full-box crop, where the kernel writes the extrapolation value because the float
  source row is above H - 1). The rewrites write the kernel arithmetic of TensorFlow v2.21.0
  (`resize_area_op.cc`, `resize_bilinear_op.cc`, `crop_and_resize_op.cc`) with Cast, MatMul, and
  Gather. Check `RewrittenGraphMatchesTensorflowGraph` (grappler off in both sessions): 0.0 RGBA
  difference and equal branch tensors on the 14 photos.
- **No grappler in the conversion.** The grappler constant folding that
  `tf2onnx.from_graph_def` runs changes `(x * y) / c` to another float order. That moved one crop
  position by one ulp and changed the alpha of `DSC00674` (IoU 0.934). `convert_to_onnx` calls
  `process_tf_graph` and `optimize_graph` without grappler.
- **Float64 scalar divisions.** On DirectML, the first float32 graph gave IoU 0.936 against
  TensorFlow on `DSC00674` (CPU: 1.0000). A diagnosis with the If branches inlined (ad hoc
  scripts in `build/tmp/universal_matting/`) found the first different tensor: the 1920 px crop
  of the human branch, in its last row only. DirectML computes 4095 / 1919 as 2.13392377, one ulp
  below the correctly rounded 2.13392401 of the CPU and TensorFlow. The source row of the last
  crop row is then 4094.99976 (not above 4095), so DirectML reads the photo where TensorFlow
  writes 0. With the DirectML metacommands off or the graph optimizations at the basic level, the
  result did not change. The export now computes each of the 42 one-element Div nodes outside the
  U-Nets in float64 (Cast, Div, Cast to float32). DirectML has no float64 Div, so ONNX Runtime
  runs these nodes on the CPU: 22 of them run on `DSC_1535`, 2.3 ms CPU kernel time in total per
  run, warm p50 +1.7 ms. The CPU output is bit-identical to the first graph on the 14 photos. The
  records of the first graph are `*_float32_divisions.json`.
- **Float16** (`convert_regions_to_float16`): only nodes of the six U-Nets, and only when the
  output and each input of the node have a largest absolute value of at most 4,094 on the 14
  photos (the S3OD limit, 65,504 / 16). The instance norms of the detector, coarse, and object
  U-Nets reach 5.2e8 (squared deviations), so 888 of 1,954 U-Net nodes are float16 (refine U-Nets:
  38 to 40 of 55; their maximum is 10,135). The input rule is necessary: with the output rule
  only, the `x - mean` of an instance norm read a float16 copy of a large Conv output (IoU 0.70 on
  `DSC_1188`). The ONNX Runtime float16 converter cannot convert this graph: it moves the If branch
  outputs to float16 with the Cast in the parent graph and writes one Cast name in two branches.
- Graphs: float32 173.3 MB, 3,400 nodes with the If branches (SHA-256 `eda5d60e...`); float16
  154.8 MB, 4,023 nodes (`66dfb827...`). Records
  [`universal_matting_float32_export.json`](../../results/subject_alternatives/universal_matting_float32_export.json),
  [`universal_matting_float16_export.json`](../../results/subject_alternatives/universal_matting_float16_export.json).
- **`OnnxAlphaMatchesTensorflowWithinTolerance`** (`compare_onnx_outputs.py`, ONNX Runtime 1.31.0
  CPU against TensorFlow 2.21.0 CPU with its default session, 4 threads; record
  [`universal_matting_comparison_cpu.json`](../../results/subject_alternatives/universal_matting_comparison_cpu.json)):

  | Graph | Maximum alpha difference | Against TensorFlow with grappler off | Minimum IoU of alpha > 0.5 | Result |
  | --- | ---: | ---: | ---: | --- |
  | float32 | 6.3e-3 (`DSC00527`); above 1e-3 on 7 photos | 3.4e-3 (`DSC00674`); the other 13 photos at most 9.0e-4 | 0.99991 | **fails** (1e-3) |
  | float16 (CPU) | 1.0 | 1.0 | 0.9769 (`DSC00674`), 0.9798 (`DSC00326`) | no requirement |

- **The reference is not stable to 1e-3**
  ([`universal_matting_tensorflow_spread.json`](../../results/subject_alternatives/universal_matting_tensorflow_spread.json),
  `measure_tensorflow_spread.py`). TensorFlow with six CPU settings (oneDNN on or off, 1 or 4
  threads, grappler on or off) differs from itself by up to 7.2e-3 (`5761824892`), 6.6e-3
  (`DSC00527`), 4.1e-3 (`DSC_1188`), and 3.4e-3 (`DSC00674`). The ONNX float32 alpha is within
  1.3e-3 of the nearest TensorFlow setting on every photo (the stored ONNX alpha is float16, so
  values below 2.4e-4 are not resolved). Two runs of the default setting also differ: on
  `2088064441` the same ONNX alpha is 1.19e-3 from one run and 1.27e-3 from the other. The
  largest differences are in a few edge pixels (p99.9 at most 6.3e-4; mean at most 3.0e-6).
- **The model is sensitive on two photos** (ad hoc check). Gaussian noise with a standard
  deviation of 0.5 (of 255) on the input changes the IoU against TensorFlow to 0.989 to 0.997 on
  `DSC00674` and to 0.980 to 0.999 on `DSC00326` (`DSC_1535`: 0.9999). The float16 change of
  these photos comes from one U-Net each (ad hoc, one U-Net in float16 at a time): the coarse
  U-Net on `DSC00674` and the object stage 0 U-Net on `DSC00326`. Both decide a crop box. The
  other U-Nets in float16 change at most a few edge pixels (IoU >= 0.998).

### 12.4 Cost on DirectML

Records `results/subject_alternatives/universal_matting_<precision>_*.json`. Times in ms; memory
in MB (dedicated / shared) of the process. The photos are fed at their size (4096 px long edge).

| Graph | File | Placement (DirectML nodes, CPU nodes) | Session load | First run | Warm p50 / p95 / maximum (20) | Photos p50 / maximum (14) | After load | After first run | After the photos | After release |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| BiRefNet lite 1024 float16 (section 5.2) | 96.4 MB | 601, 724 (E2) | 9,442 | 817 | 154 / 178 / 187 | 153 / 200 | 227 / 180 | 3,226 / 204 | | 107 / 232 |
| S3OD 8 chunk MHA float16 (section 5.5) | 219.3 MB | 1, 0 | 1,111 | 234 | 191 / 193 / 198 | 192 / 213 | 297 / 245 | 1,358 / 274 | | |
| Universal matting float32 | 173.3 MB | 1,404, 174 | 1,513 | 336 | 116 / 127 / 131 | 126 / 248 | 364 / 124 | 1,905 / 638 | **2,678 / 895** | 76 / 44 |
| Universal matting float16 | 154.8 MB | 1,769, 174 | 1,827 | 352 | 115 / 120 / 120 | 130 / 167 | 345 / 132 | 1,880 / 646 | 2,664 / 903 | 85 / 46 |

- **No DirectML graph fusion.** The DirectML nodes are single operators, not fused partitions:
  the input has no static size, and the If branches split the graph. A test graph with the If
  branches inlined fails in the DirectML graph fusion at session creation (`80070057`,
  E_INVALIDARG). The 174 CPU nodes are shape and box computations (outputs of at most 4 KB) and
  the float64 divisions; together 2.3 ms per run (DirectML 253 ms in the profiling run).
- **The memory comes from tensors of the photo size.** One 4096 x 2731 photo is 134 MB as the
  float32 input and 179 MB as the RGBA output, and the graph keeps more tensors of that size (the
  scaled photo, the crops, the padded alpha, the masked RGB). The dedicated memory rises from
  1,905 MB after the first photo to 2,678 MB after the 14 photos of different sizes. Float16 in
  the U-Nets does not change this (2,664 MB). The split per tensor was not measured.
- The time per photo depends on the branch: 110 ms (`DSC_1535`, human result, small box), 248 ms
  (`DSC00674`, human result, full-photo box with the 1920 px crop), 123 to 177 ms (object branch;
  the human branch also runs on 4 of these photos).
- The working set of the cost script includes the 14 prepared photo inputs (1.9 GB before the
  load); it is not comparable with the other models.

**First runs** (9 new processes per graph, `DSC_1535`): 0 failed runs, 0 driver resets, and no
event 153 or 4101. Float32: load 1,586 to 1,772 ms, first run 231 to 240 ms. Float16: load 1,875
to 2,030 ms, first run 255 to 267 ms. In total: 0 resets in 46 universal matting sessions (per
graph form: 18 first runs, 2 cost runs, 2 profiling runs, 1 pipeline run; two graph forms).

**BiRefNet reference in the same period.** One run of `measure_directml_cost.py` with the
BiRefNet lite float16 graph (13:09:22): first run 64,239 ms with events 153 at 13:09:31,
13:10:32, and 13:10:35 and event 4101 at 13:10:36, then `DmlFusedNode_0_0 ... 887A0006` on the
first warm run. Record
[`birefnet_lite_float16_universal_matting_step_failed_runs.json`](../../results/subject_alternatives/birefnet_lite_float16_universal_matting_step_failed_runs.json)
(peak 3,226 MB). The BiRefNet values of the tables are from section 5.2. Issue #6 in total:
BiRefNet 8 resets in 14 sessions; S3OD 0 in 88; IS-Net 0 in 23; universal matting 0 in 46.

### 12.5 MVP pipeline with universal matting and DirectML output

`measure_pipeline_cost.py --subject-onnx <universal matting float32 graph>` under
`run_with_gpu_memory.ps1` (the sequence of section 5.5; `mask_pipeline.py` selects the universal
matting processing by the graph output `rgba` and feeds the scaled photo of the MVP). Records
`mvp_pipeline_universal_matting_float32_cost*.json`.

| Item | MVP with BiRefNet lite float16 | MVP with S3OD (8 chunk MHA float16) | MVP with IS-Net float16 | MVP with universal matting float32 |
| --- | ---: | ---: | ---: | ---: |
| Peak dedicated GPU memory | 4,528 MB | 2,666 MB | 1,748 MB | **3,982 MB** |
| Peak shared GPU memory | 486 MB | 459 MB | 325 MB | 1,082 MB |
| Peak working set | 1,873 MB | 1,675 MB | 1,476 MB | 2,687 MB |
| Subject graph load | 7,998 ms | 910 ms | 507 ms | 1,221 ms |
| Subject first call / warm p50 | 417 ms / 155.8 ms | 250 ms / 191.1 ms | 30.7 ms / 13.9 ms | 240 ms / 115.9 ms |
| SAM 2.1 tiny encoder warm p50, decoder box warm p50 | 73.7 ms, 23.2 ms | 70.5 ms, 23.2 ms | 72.8 ms, 21.2 ms | 68.1 ms, 20.4 ms |

3,982 MB is 910 MB above the 3 GB budget of U6. No event 153 or 4101 occurred during the run.

**DirectML output against TensorFlow** (stored alphas, IoU of alpha > 0.5 on the 1024 grid,
[`quality_universal_matting.json`](../../results/subject_alternatives/quality_universal_matting.json)):
float32 0.9997 (`Nikon-D800e-raw-00007`) to 1.0000; float16 0.9804 (`DSC00326`), 0.9833
(`Nikon-D800e-raw-00007`), 0.9874 (`2088064441`), the other 11 photos 0.991 or more. The first
float32 graph (float32 divisions) gave 0.9359 on `DSC00674`
([`quality_universal_matting_float32_divisions.json`](../../results/subject_alternatives/quality_universal_matting_float32_divisions.json)).

### 12.6 Quality

`probe_subject_masks.py` (section 6.1; kind `universal_matting`: the alpha at the photo size
resized to the 1024 grid). The universal matting values are the TensorFlow alpha; the ONNX
float32 alphas (CPU and DirectML) give the same means within 0.001. B: BiRefNet lite float16 CPU;
S: S3OD float32 CPU; U: universal matting.

| Photo (subject box) | Foreground B / S / U | Subject recall B / S / U | Outside box B / S / U | Edge width px B / S / U | Parts B / S / U | IoU U with B / S |
| --- | --- | --- | --- | --- | --- | --- |
| `DSC_1535` (person) | 0.222 / 0.223 / 0.221 | 0.998 / 0.999 / 0.995 | 0.000 / 0.000 / 0.000 | 1.57 / 0.05 / 2.29 | 1 / 1 / 1 | 0.987 / 0.985 |
| `DSC00326` (person) | 0.026 / 0.026 / 0.026 | 0.989 / 0.991 / 0.968 | 0.000 / 0.000 / 0.000 | 1.32 / 0.08 / 2.98 | 1 / 1 / 1 | 0.945 / 0.927 |
| `DSC00674` (person) | 0.679 / 0.680 / 0.723 | 0.999 / 1.000 / 0.993 | 0.005 / 0.005 / 0.012 | 5.44 / 0.08 / 6.57 | 1 / 1 / 1 | 0.925 / 0.924 |
| `DSC00936` (person) | 0.022 / 0.022 / 0.022 | 0.996 / 0.992 / 0.992 | 0.000 / 0.000 / 0.000 | 1.23 / 0.04 / 1.54 | 1 / 1 / 1 | 0.989 / 0.978 |
| `P1024439` (bassist) | 0.262 / 0.061 / 0.149 | 0.996 / 0.985 / 0.995 | 0.753 / 0.019 / 0.538 | 1.67 / 0.13 / 4.56 | 3 / 1 / 2 | 0.497 / 0.396 |
| `DSC_1188` (squirrel) | 0.110 / 0.107 / 0.097 | 0.996 / 0.994 / **0.904** | 0.003 / 0.001 / 0.000 | 2.05 / 0.08 / 7.99 | 1 / 1 / 1 | 0.867 / 0.877 |
| `DSC00527` (kneeling woman) | 0.024 / 0.023 / 0.029 | 0.997 / 0.993 / 0.996 | 0.558 / 0.538 / 0.625 | 0.99 / 0.09 / 2.25 | 2 / 2 / 3 | 0.799 / 0.754 |
| `2088064441` (boarder) | 0.005 / 0.004 / **0.204** | 0.981 / 0.957 / 1.000 | 0.001 / 0.000 / **0.939** | 1.38 / 0.08 / 8.55 | 1 / 1 / 3 | 0.022 / 0.020 |
| `DSC00728` (balloon) | 0.003 / 0.003 / 0.003 | 0.992 / 0.977 / 0.981 | 0.000 / 0.000 / 0.000 | 0.99 / 0.03 / 1.29 | 1 / 1 / 1 | 0.980 / 0.969 |
| `5761824892` (car) | 0.241 / 0.241 / 0.224 | 1.000 / 1.000 / 0.924 | 0.075 / 0.073 / 0.080 | 1.45 / 0.09 / 3.53 | 1 / 1 / 1 | 0.922 / 0.920 |
| `DSC00562` (duck on the wall) | 0.370 / 0.369 / **0.516** | 0.970 / 0.952 / **0.001** | 0.995 / 0.995 / 0.999 | 1.94 / 0.11 / 6.33 | 1 / 1 / 2 | 0.702 / 0.696 |
| `DSC00602` (tram) | 0.273 / 0.270 / 0.265 | 1.000 / 0.998 / 0.986 | 0.014 / 0.012 / 0.004 | 1.58 / 0.09 / 5.35 | 1 / 1 / 1 | 0.958 / 0.963 |
| `DSC_1456` (car) | 0.065 / 0.065 / 0.065 | 0.997 / 0.999 / 0.996 | 0.002 / 0.000 / 0.004 | 1.35 / 0.05 / 2.01 | 1 / 1 / 1 | 0.987 / 0.982 |
| `Nikon-D800e-raw-00007` (rock) | 0.000 / 0.015 / 0.005 | 0.000 / 0.969 / 0.000 | 0.000 / 0.017 / **1.000** | — / 0.12 / 4.27 | 0 / 1 / 9 | 0.000 / 0.000 |
| Mean | 0.164 / 0.151 / 0.182 | 0.922 / 0.986 / **0.838** | 0.172 / 0.119 / **0.300** | 1.64 / 0.08 / 4.25 | 1.14 / 1.07 / 2.00 | 0.756 / 0.742 |

Soft share (0.05 < alpha < 0.95), mean: B 0.006, S 0.0002, U 0.029. Mean hole fraction: B
0.004, S 0.008, U 0.026. Soft share in the hair regions: U 0.046 to 0.070 (B 0.024 to 0.039); in
the squirrel fur region: U 0.130 (B 0.028).

What the local sheets show (section 12.9), in words:

- **Background areas.** `2088064441`: the mask has the bridge, most of the water, and the roof of
  a building, partly soft (94 % of the foreground is outside the subject box). BiRefNet and S3OD
  take only the boarder. `DSC00562`: BiRefNet and S3OD take the wall with the waterfall.
  Universal matting takes the wall, the buildings behind it, a band of trees, and a floor area;
  the ducks on the wall are not in the mask (subject recall 0.001).
- **No subject found.** `Nikon-D800e-raw-00007`: BiRefNet finds nothing, and S3OD finds the rock.
  Universal matting takes parts of the mountain ridge (9 parts, all outside the box).
- **Soft bands and missed parts.** `DSC_1188`: a soft band of 0.05 to 0.95 around the tail that
  is wider than the fur, and the white belly is not in the mask. `P1024439`: the bassist and the
  guitarist in front, with parts of the microphone stand; less background than BiRefNet (53.8 %
  outside the box; BiRefNet 75.3 %, S3OD 1.9 %). `5761824892`: the windshield shows the
  background through the glass, and that area is not in the mask (BiRefNet and S3OD fill the
  windows).
- **Hair.** `DSC_1535`, `DSC00326`, `DSC00936`: the same outline as BiRefNet with a soft edge of
  1.5 to 3.0 px. `DSC00674`: an outline similar to BiRefNet; the hair at the right edge of the
  photo is softer.

Universal matting is equal to BiRefNet on 6 photos (`DSC_1535`, `DSC00326`, `DSC00936`,
`DSC00728`, `DSC00602`, `DSC_1456`: IoU 0.945 to 0.989), similar on 2 (`DSC00674`, `DSC00527`),
and worse than BiRefNet and S3OD on 5 (`2088064441`, `DSC00562`, `DSC_1188`, `5761824892`,
`Nikon-D800e-raw-00007`). On `P1024439` it is between BiRefNet and S3OD.

### 12.7 License

| Item | License | Commercial use | Evidence |
| --- | --- | --- | --- |
| `tf_graph.pb` | Apache License 2.0 (model card front matter) | yes | `README.md` SHA-256 `d21254bf...` |
| ModelScope pipeline code | Apache-2.0 | yes | `LICENSE` SHA-256 `cfc7749b...` |
| Training data | "public data sets (COCO etc.) and images from internet searches", annotated, about 200,000 composites | **unknown**: the rights of the internet images are not stated; COCO images are Flickr images with their own licenses | model card, section "training data" |

The weights have a license that permits commercial use. The training data has the same open
question as the other models (U13).

### 12.8 Recommendation

- **Universal matting is not a replacement for BiRefNet lite:**
  1. It does not meet U6: the MVP pipeline with it has a peak of 3,982 MB (budget 3 GB). The
     model reads the photo at its size, so the memory grows with the photo. The Subject graph
     alone has 2,678 MB: less than BiRefNet (3,226 MB) but twice S3OD (1,358 MB).
  2. Its masks are worse than BiRefNet and S3OD on 5 of 14 photos (mean subject recall 0.84; 30 %
     of the foreground outside the subject box), with large background areas on 2 photos.
  3. The float32 ONNX graph does not meet the 1e-3 tolerance against the default TensorFlow
     session (6.3e-3). But TensorFlow differs from itself by up to 7.2e-3, and the ONNX alpha is
     within 1.3e-3 of a TensorFlow setting on every photo.
- It meets the "no driver reset" criterion (0 resets in 46 sessions), and its warm time at the
  full photo size is 116 ms. Float16 has no use: no time or memory gain, and box changes on 2
  photos.
- **S3OD (8 chunk MHA float16) stays the recommendation** of the measured candidates (section 8).
- The DirectML division defect (section 12.3) can occur in each graph that compares float
  positions computed with a division (crop and resize rules written as operators). It was not
  checked in the S3OD and IS-Net graphs; their DirectML outputs match the CPU (sections 5.5 and
  11.5).

### 12.9 Open items

- The check of the masks by eye. The command is in
  [`scripts/ai_models/universal_matting/README.md`](../../scripts/ai_models/universal_matting/README.md#commands)
  (last block). The sheets (14 photos and 6 detail crops; panels: BiRefNet lite float16 CPU, S3OD
  8 chunk MHA float16 DirectML, universal matting float32 DirectML; red: alpha > 0.5, yellow:
  0.05 to 0.95) are in `docs/exploration/images/subject_alternatives_universal_matting/` (in
  `.gitignore`).
- Not measured: the model with a photo below 2000 x 2000 px (the range of the model card; less
  memory). The issue requires the input size of the source, so this is a user decision. The split
  of the GPU memory per tensor is not measured.
- The float32 tolerance decision (as section 8, item 1): the TensorFlow reference of this model is
  not stable to 1e-3.
- `ai_mask_app.py` with the universal matting graph is not tested (its labels name BiRefNet); the
  pipeline code is tested through `measure_pipeline_cost.py`.
- The BiRefNet first runs were not repeated in this step; its one cost run of this step reset the
  driver (section 12.4).
