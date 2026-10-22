# Subject models in place of BiRefNet lite

**Issue:** [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6). **Date:** 2026-10-21.
**Code:** [`scripts/ai_models/s3od/`](../../scripts/ai_models/s3od/README.md).
**Results:** [`results/subject_alternatives/`](../../results/subject_alternatives/).
**Status:** S3OD `s3od-synth` is measured (this report), also with the attention in query chunks
(section 5.5, 2026-10-21). IS-Net general-use and Alibaba universal matting are not measured yet
(section 9). The check of the masks by eye is open: the user does it.

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

- **IS-Net general-use and Alibaba universal matting** (issue #6) are not measured in this step.
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

The commands are in [`scripts/ai_models/s3od/README.md`](../../scripts/ai_models/s3od/README.md#commands):
sources and weights, export of both forms, the CPU comparison, the float32 spread, placement,
first runs, cost with GPU memory, quality metrics, and the attention in query chunks of section
5.5 (the one layer probe, the export, and the MVP pipeline cost). The investigation of section 4 used ad hoc
scripts in `build/tmp/s3od_synth/` (single-node DirectML tests, graph cut bisection for the
session error and for the first non-finite tensor, the intermediate tensor ranges); their results
are in this report, and `export_onnx.py` records the attention score maxima in
`export_report.json`.
