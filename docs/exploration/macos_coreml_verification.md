# macOS Core ML verification of the accepted model set

**Date:** 2026-10-22. **Issue:** [#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8).
**Model set:** [acceptance record](acceptance_ai_mask_model_set.md), section 1 (README U14).
**Status:** measured; the decisions of section 13 are made (README U15); the user checks the UI
interaction by eye.

## 1. Question

Does the accepted model set (SAM 2.1 tiny encoder and decoder, the people and sky head of P1 run 2,
S3OD `s3od-synth`) run correctly on macOS with Core ML, and what are its placement, cost, memory,
and stability? Issue #8 asks for ONNX Runtime with the Core ML execution provider. During the work
the user decided (2026-10-22) to try native Core ML models for macOS, as Alcedo Studio does for its
SigLIP2 models (`coreml_clip.rs`: a precompiled `.mlmodelc`, `ComputeUnits::All`).

## 2. Result summary

- **The ONNX Runtime Core ML provider does not run the set as it is** (section 4). With the
  production setting (NeuralNetwork format), Core ML takes no S3OD node: S3OD runs on the CPU in
  7.9 s per call with a 10.3 GB peak footprint. In the MLProgram format the 88
  `com.microsoft.MultiHeadAttention` nodes and 227 Cast nodes of S3OD stay on the CPU, and the
  SAM 2.1 decoder stops the process (SIGABRT, MPSGraph assertion) in 3 of 3 first runs.
- **ONNX Runtime 1.24.4 to 1.27.0 compute the SAM 2.1 tiny encoder wrongly on this Mac**
  (section 7). Constant folding of the two `If` nodes of the Hiera position embedding keeps a
  freed buffer (microsoft/onnxruntime#29071, fixed in 1.28.0). The relative error of the encoder
  output is 1.9e-2 to 7.7e8 and changes from process to process. Alcedo Studio uses ONNX
  Runtime 1.24 (`ort` 2.0.0-rc.12, `api-24`), on macOS and on Windows.
- **Native Core ML models of the set are correct** (section 8). Against ONNX Runtime 1.31.0 on
  the CPU of the same Mac, on the 14 quality photos: the same Sky and People decisions, the same
  People lists, 212 of 212 SAM 2.1 masks with IoU >= 0.999 (minimum 0.99959), the S3OD alpha with
  IoU >= 0.998 (minimum 0.99907), no NaN or infinite value.
- **S3OD needs another precision split on the Neural Engine** (section 6). The accepted precision
  (float16, layer 0 attention in float32) gives a wrong alpha with the compute units ALL and
  CPU_AND_NE (IoU 0.0 to 0.49), because the DINOv3 patch embedding is wrong on the Neural Engine.
  With the patch embedding and the encoder attention in float32 (they run on the GPU) and the other
  ops in float16 (Neural Engine), the alpha is correct with ALL.
- **Cost on the Apple M4** (section 9, MVP pipeline, all models loaded, the decoder with CPU_ONLY
  and the other models with ALL, U15): encoder 195 ms, head 2 ms, decoder 28 to 35 ms for each
  call, S3OD 531 ms warm (p50). Peak physical footprint of the process 3,190 MB (with the decoder
  on ALL: 3,314 MB). On Windows DirectML the models used 2,666 MB GPU memory and 1,675 MB working
  set.
- **Stability:** 0 failures in 45 first runs (9 per model, the decoder with ALL and with CPU_ONLY)
  of the native models.
- **The decoder point count:** each change of the number of prompt points costs about 52 ms on the
  GPU (section 10). Enumerated shapes do not change this; CPU_ONLY has no such cost (30 ms). The
  decoder therefore runs with CPU_ONLY (U15).
- **The UI runs on macOS** after one change: the model jobs run in a Python thread, not in a
  QThread (section 11).

## 3. Machine and runtime

| Item | Value |
| --- | --- |
| Machine | Apple M4 (10 cores), 16 GB unified memory |
| System | macOS 27.0.1 (26A434) |
| Python | 3.13.7 (Homebrew) |
| MVP environment | onnxruntime 1.24.4 (the ONNX Runtime of the production `ort` 2.0.0-rc.12 with `api-24`), coremltools 9.0 (`CompiledMLModel`), PyTorch 2.14.1, NumPy 2.5.2, PySide6 6.12.0 |
| Conversion environment | coremltools 9.0, PyTorch 2.7.0, NumPy 2.2.6, transformers 5.20.2; onnxruntime 1.31.0 for the CPU reference of section 8 |
| Session settings | 4 CPU threads, CPU memory arena off, one provider per session (`mask_pipeline.create_session`) |
| Core ML provider, production | `CoreMLExecutionProvider` with `MLComputeUnits` ALL and the default model format NeuralNetwork, then `CPUExecutionProvider`, graph optimization level 3 (`ort_runtime::session_builder` of Alcedo Studio) |
| Core ML provider, other format | the same with `ModelFormat` MLProgram (`segmentation_runtime_report.rs --session-settings coreml-mlprogram`) |
| Native models | compiled `.mlmodelc`, coremltools `CompiledMLModel`, compute units ALL (the production default of the native SigLIP2 models) unless stated |

Memory method: macOS has no per-process GPU memory counter. Apple silicon has unified memory, and
the GPU buffers of a process are in its physical footprint. The scripts record the physical
footprint and its lifetime peak from `proc_pid_rusage` (`RUSAGE_INFO_V4`,
`scripts/ai_models/sam2/macos_records.py`; equal to the value of `footprint`, 318 MB against
317.5 MB in a check). `run_with_memory_sampling.py` also samples the system memory in use
(`vm_stat`), which contains memory that other processes (for example `aned` of the Neural Engine)
allocate for the program; other programs change it too. The sampler reads the lifetime peak every
200 ms, so a peak in the last interval before the process ends is not in its record; the MVP
pipeline therefore states the value that the process records itself.

Failure records: the exit code (a negative value is the signal) and the macOS crash reports
(`.ips` in `~/Library/Logs/DiagnosticReports` and `/Library/Logs/DiagnosticReports`) of the run
window, in place of the Windows System events 153 and 4101.

## 4. ONNX Runtime Core ML provider

Placement from the Core ML capability log of ONNX Runtime and the ONNX Runtime profile (2 profiled
runs on `DSC_1535`; the first includes the Core ML preparation), records
`results/macos_coreml/<graph>_coreml_<format>_placement.json`:

| Graph | Format | Core ML partitions | Nodes on Core ML / in the graph | CPU nodes (main types) | Kernel ms per run, CPU / Core ML |
| --- | --- | ---: | ---: | --- | ---: |
| encoder | NeuralNetwork | 70 | 389 / 508 | MatMul 24, LayerNormalization 24, Reshape 36, Transpose 18, Erf 12 | 315 / 1,478 |
| encoder | MLProgram | 19 | 449 / 508 | Reshape 36, Transpose 18, Pad 5 | 7 / 2,267 |
| decoder | NeuralNetwork | 56 | 260 / 479 | MatMul 47, Gather 36, Reshape 34, Unsqueeze 29 | 10 / 163 |
| decoder | MLProgram | — | — | the process stops (below) | — |
| head | NeuralNetwork | 10 | 56 / 66 | InstanceNormalization 4, Erf 4 | 1 / 19 |
| head | MLProgram | 1 | 66 / 66 | — | 0 / 647 |
| S3OD | NeuralNetwork | 0 | 0 / 1,241 | all 1,734 executed nodes | 8,124 / 0 |
| S3OD | MLProgram | 100 | 1,127 / 1,241 | Cast 227, MultiHeadAttention 88, Neg 22 | 1,681 / 381 |

- **S3OD, NeuralNetwork (production):** Core ML takes no node of the float16 graph. The whole
  graph runs on the CPU provider: 7.9 s per call, 10.3 GB peak footprint (`/usr/bin/time -l` of
  the placement run).
- **S3OD, MLProgram:** the 88 `com.microsoft.MultiHeadAttention` nodes (8 chunks x 11 layers; the
  operator is an ONNX Runtime contrib operator) and 227 Cast nodes stay on the CPU, in 100
  partitions.
- **Decoder, MLProgram:** 3 of 3 first runs stop with SIGABRT and a crash report
  (`results/macos_coreml/point_decoder_coreml_MLProgram_first_runs.json`): `MPSGraphExecutable.mm:4419:
  failed assertion 'Error: Optimize Original Module MLIR pass manager failed ... 'mps.matmul' op
  contracting dimensions differ 1 & 256'`. The SM1 record had the same kind of decoder failure for
  EfficientSAM3.
- **Encoder:** ONNX Runtime does the constant folding of section 7 before it partitions the graph,
  so the encoder output of the provider has the same error as the CPU provider.

Not measured with the provider: the cost, the 9 first runs per graph (except the decoder in
MLProgram), and the output comparison. After these results the user chose the native models.

## 5. Native Core ML models

`scripts/ai_models/coreml/convert_models.py` converts the PyTorch modules of the ONNX exports
(`torch.jit.trace`, coremltools 9.0, MLProgram, macOS 15 deployment target) and compiles them.
The models have the inputs and outputs of the ONNX graphs, so the pipeline code is the same
(`scripts/poc/ai_mask_mvp/coreml_native.py`, provider `coreml_native`). Records:
`results/macos_coreml/native_<model>_convert.json`.

| Model | Source | Precision | Compiled bytes | SHA-256 (compiled) | Against PyTorch float32 (one input) |
| --- | --- | --- | ---: | --- | --- |
| `point_encoder.mlmodelc` | SAM 2.1 tiny (`sam2.1_hiera_tiny.pt`, `7402e0d8...`) | float32 | 134,260,018 | `d3f66d7b...` | relative error 3e-6 on photos (section 8) |
| `point_decoder.mlmodelc` | as above; point count N = 1 to 64 (range of shapes) | float32 | 21,098,274 | `f2efed04...` | masks 1.6e-4 |
| `people_sky_head.mlmodelc` | P1 run 2 `best.pt` (epoch 15, `5da8a4a2...`) | float32 | 1,138,418 | `ccb5c48d...` | 1.7e-4 |
| `s3od.mlmodelc` (`float16_embeddings_attention_float32`) | `s3od.pt` (`c67f2641...`) | float16; patch embedding and the attention of all 12 encoder layers float32 | 271,166,829 | `2c6a24e6...` | section 6 |
| `s3od.mlmodelc` (`float16_layer0_float32`) | as above | the precision of the accepted ONNX graph | 216,980,716 | `4f4dada5...` | section 6 |

Conversion facts:

- The Hiera position embedding is a bicubic resize of a weight to the 256 x 256 token grid.
  coremltools has no converter for `upsample_bicubic2d`, so the script computes it in PyTorch
  and the model has it as a constant (the same values; the encoder matches PyTorch within 3e-6).
- The S3OD attention is the attention of the source (`scaled_dot_product_attention`); coremltools
  writes it as matmul, softmax, matmul. The query chunks of the ONNX graph are a DirectML memory
  measure and are not in the Core ML model.
- coremltools 9.0 needs PyTorch 2.7.0 and NumPy 2.2 for the conversion: with NumPy 2.5 its
  PyTorch frontend fails (`TypeError: only 0-dimensional arrays can be converted to Python
  scalars`).

Placement from the Core ML compute plan (`profile_compute_plan.py`, compute units ALL, records
`results/macos_coreml/native_*_compute_plan.json`): the encoder (352 ops) and the head (69 ops) run
on the GPU (float32 models cannot run on the Neural Engine, so ALL is CPU_AND_GPU for them); the
decoder has 187 GPU ops, 36 CPU ops, and 21 ops without a device (the flexible point dimension);
the main S3OD model has 402 GPU ops and 189 Neural Engine ops (24 % of the estimated cost on the
Neural Engine).

## 6. S3OD on the Neural Engine

The S3OD model in the accepted precision gives a wrong alpha when Core ML uses the Neural Engine.
`probe_s3od_compute_units.py` on `DSC_1535` and `P1024439` against S3OD float32 on the ONNX
Runtime CPU (the reference of the DirectML criterion; records
`results/macos_coreml/native_s3od_variants/<variant>_outputs.json`), call time of the second call:

| Variant (ops in float32 besides the layer 0 attention) | Compute units | IoU of alpha > 0.5 (two photos) | ms per call |
| --- | --- | --- | ---: |
| none (`float16_layer0_float32`, accepted precision) | ALL | 0.493, 0.000 | 1,473 to 1,480 |
| none | CPU_AND_NE | 0.489, 0.314 | 1,607 to 1,609 |
| none | CPU_AND_GPU | 0.99999, 0.99983 | 943 |
| none | CPU_ONLY | 0.99913, 0.99611 | 1,046 to 1,062 |
| all `layer_norm` ops | ALL | 0.493, 0.235 | 1,631 |
| DPT decoder (`inner.seg_head`) | ALL | 0.495, 0.000 | 1,918 to 1,928 |
| encoder attention | ALL | 0.493, 0.240 | 524 to 541 |
| encoder MLP | ALL | 0.470, 0.041 | 1,754 to 1,773 |
| RoPE module (folded to constants, no op changes) | ALL | 0.493, 0.000 | 1,480 |
| final encoder norm | ALL | 0.493, 0.000 | 1,475 to 1,481 |
| **patch embedding** (`inner.encoder.embeddings`) | ALL | **0.99994, 0.99934** | 1,480 |
| **patch embedding and encoder attention** (`float16_embeddings_attention_float32`) | ALL | **0.99995, 0.99973** | 537 to 561 |
| whole encoder (`float16_encoder_float32`) | ALL | 0.99999, 0.99998 | 660 to 662 |
| whole encoder | CPU_AND_GPU / CPU_AND_NE / CPU_ONLY | 1.00000 / 0.99996 / 0.99975 | 1,026 / 1,871 to 1,879 / 2,355 to 2,361 |

- The wrong output comes from the DINOv3 patch embedding (the 16 x 16 stride 16 convolution and
  the token concatenation) on the Neural Engine in float16. With only this block in float32 (on the
  GPU) the alpha is correct; no other block changes the result.
- The Neural Engine is slower than the GPU for the encoder attention at 4,101 tokens. With the
  attention also in float32, the encoder attention runs on the GPU and the MLP blocks, norms, and
  the DPT decoder in float16 on the Neural Engine: 537 to 561 ms, the fastest correct form. It is
  the main native S3OD model of sections 8 to 11.
- The precision of these forms is higher than the accepted ONNX graph (more float32 ops); U8
  allows float16 and does not ask for it.

## 7. ONNX Runtime 1.24 CPU defect on this Mac

The first output comparison (MVP on the native models against the MVP on the ONNX Runtime 1.24.4
CPU) failed on the SAM 2.1 masks. The native encoder matched PyTorch within 3e-6, but the ONNX
Runtime CPU output did not (`probe_encoder_outputs.py`, record
`native_point_encoder_outputs_photos.json`). Investigation of the ONNX encoder graph on the CPU
provider, relative error against PyTorch float32 on `DSC_1535`:

| Run (record in `results/macos_coreml/`) | Result |
| --- | --- |
| Windows x86, onnxruntime 1.31.0 | 2.3e-6 |
| Mac, onnxruntime 1.31.0 | 3.1e-6 |
| Mac, onnxruntime 1.24.4, graph optimization off (`cpu_encoder_settings_onnxruntime_1_24_4.json`) | 3.1e-6 |
| Mac, onnxruntime 1.24.4, levels BASIC, EXTENDED, ALL; 4 or 1 threads | wrong (4.0e8 in that process); repeat calls in one session equal |
| Mac, onnxruntime 1.24.4, BASIC without one optimizer (`cpu_encoder_optimizers_onnxruntime_1_24_4.json`, 49 optimizers) | equal to the unoptimized output only without `ConstantFolding` |
| Mac, BASIC against unoptimized, onnxruntime 1.24.4 / 1.25.1 / 1.26.0 / 1.27.0 | 7.7e8 / 7.5e8 / 1.9e-2 / 3.7e-1 |
| Mac, the same, onnxruntime 1.28.0 / 1.29.0 / 1.30.0 / 1.31.0 | 0 / 0 / 0 / 0 |
| Windows x86, onnxruntime 1.24.4 (`..._directml_1_24_4_windows.json`) | 9.8e-7 |

- The folded initializer `/image_encoder/trunk/Transpose_output_0` (the position embedding,
  1 x 256 x 256 x 96 after the transpose) has wrong values on the Mac: 0.065 % of the elements,
  in the period-8 pattern of the 8 x 8 window embedding, up to 2.2e24 (comparison of the BASIC
  optimized graphs of both machines). The window embedding and the `Tile` repeats come from the
  two `If` nodes of the graph, which constant folding inlines.
- This is microsoft/onnxruntime#29071, fixed by PR #29158 (merged 2026-06-22; first release
  1.28.0): `Graph::InlineIfSubgraph` did not move the OrtValue of an initializer with in-memory
  external data, so the folded tensor points to freed memory. The error size depends on what
  overwrites the memory, so it changes from process to process; on Windows the memory was not
  overwritten in this run, but the defect is the same.
- Of the set, only the encoder has `If` nodes (2); the decoder, the head, and both S3OD graphs
  have none.
- Effect on the MVP (`cpu_onnxruntime_1_24_4_outputs_against_1_31_0.json`): the Sky decision
  differs on 1 photo, the People list on 2, the sky region count on 1; 112 of 211 SAM 2.1 masks
  have IoU >= 0.999 (minimum 0.832); People boxes move by up to 14.6 cells.

The CPU reference of section 8 is therefore ONNX Runtime 1.31.0.

## 8. Output against the CPU

`compare_providers.py` runs the MVP pipeline once per provider on the 14 quality photos (long edge
4096 px) and compares the outputs: Sky, People (all boxes, the People list, a mask per listed
person), the 4 E3 stroke sequences (149 decoder calls; each provider uses its own previous mask as
`mask_input`), and the Subject alpha. Reference: ONNX Runtime 1.31.0 CPU with the accepted SAM 2.1
and head graphs and S3OD float32 (`outputs_cpu_onnxruntime_1_31_0`). Records
`results/macos_coreml/native_ALL_outputs_against_cpu.json` and
`native_CPU_AND_GPU_outputs_against_cpu.json`.

| Check (acceptance criterion of issue #8) | Native, ALL (main S3OD model) | Native, CPU_AND_GPU (S3OD in the accepted precision) |
| --- | --- | --- |
| Sky and People presence decisions equal | 14 / 14 photos | 14 / 14 |
| People list equal (ranks and order) | 14 / 14 | 14 / 14 |
| Sky region count equal | 14 / 14 | 14 / 14 |
| SAM 2.1 masks, IoU >= 0.999 at the photo size | 212 / 212, minimum 0.99959 (sky 1.0, strokes 0.99996, persons 0.99959) | 212 / 212, minimum 0.99959 |
| S3OD alpha > 0.5, IoU >= 0.998 on the 1024 grid | 14 / 14, minimum 0.99907 | 14 / 14, minimum 0.99897 |
| No NaN or infinite output | 14 / 14 | 14 / 14 |
| Encoder outputs, largest relative error (no tolerance) | 4.0e-6 | 4.0e-6 |
| Head outputs, largest absolute difference / People box difference | 2.3e-4 / 7.2e-4 cell | the same |

With the decoder on CPU_ONLY (U15; record `native_ALL_decoder_CPU_ONLY_outputs_against_cpu.json`)
all checks pass with the same values: 212 of 212 masks with IoU >= 0.999 (minimum 0.99959), the
S3OD alpha minimum 0.99907, encoder largest relative error 4.0e-6, no NaN or infinite output.

Subject quality metrics (`probe_subject_masks.py`, record `native_subject_quality.json`, mean over
the 14 photos): the main native model has the same foreground (0.1507), subject recall (0.986), and
edge width (0.081 px) as S3OD float32 on the CPU; the accepted precision with ALL has a subject
recall of 0.352 and IoU 0.284 with the reference.

## 9. Cost and memory

Per model, one process and one session, compute units ALL (`measure_graph_cost.py cost` and
`measure_directml_cost.py --provider coreml_native`, records `native_<model>_ALL_cost*.json` and
`s3od_native_<precision>_<units>_cost*.json`). Times in ms; the peak is the lifetime peak of the
physical footprint; "before" is the footprint before the session load (the photo inputs are
already in memory).

| Model | Session load | First run | Warm p50 / p95 / max (20 runs) | Photos p50 / max | Footprint before / peak |
| --- | ---: | ---: | --- | --- | --- |
| encoder | 782 | 266 | 265 / 269 / 270 | 294 / 554 | 1,666 / 2,323 MB |
| decoder (box; 10 points with `mask_input`: p50 6.7) | 1,377 | 58 | 6.7 / 7.3 / 7.7 | 6.9 / 64 | 392 / 707 MB |
| decoder, CPU_ONLY (U15; 10 points with `mask_input`: p50 30.5) | 1,247 | 31 | 29.5 / 29.9 / 29.9 | 29.0 / 34.0 | 392 / 668 MB |
| head | 1,228 | 23 | 1.9 / 5.9 / 6.0 | 1.9 / 2.0 | 232 / 423 MB |
| S3OD `float16_embeddings_attention_float32`, ALL | 1,603 | 645 | 709 / 742 / 786 | 709 / 720 | 600 / 1,946 MB |
| S3OD `float16_encoder_float32`, ALL | 4,361 | 754 | 941 / 984 / 1,010 | 947 / 954 | 600 / 1,358 MB |
| S3OD `float16_layer0_float32`, CPU_AND_GPU | 1,742 | 1,038 | 947 / 948 / 949 | 947 / 948 | 603 / 1,431 MB |
| S3OD `float16_layer0_float32`, ALL (wrong output) | 8,504 | 1,534 | 1,483 / 1,485 / 1,490 | 1,482 / 1,487 | 600 / 1,109 MB |

The S3OD time of the same model differs between runs: 537 to 561 ms in the probe of section 6,
709 ms in this cost run, 535 ms (p50) in the MVP pipeline below.

MVP pipeline with all native models loaded (`measure_pipeline_cost.py --provider coreml_native`,
records `mvp_pipeline_native_ALL_decoder_CPU_ONLY_cost*.json` and
`mvp_pipeline_native_ALL_cost*.json`), against the Windows DirectML values of the acceptance
record, section 4:

| Item | macOS, native Core ML, decoder CPU_ONLY (U15) | macOS, native Core ML, all ALL | Windows, DirectML |
| --- | ---: | ---: | ---: |
| Session load: encoder, decoder, head, S3OD | 1,197, 69, 9, 43 ms | 1,235, 198, 9, 44 ms | 2,623, 249, 112, 910 ms |
| Read and scale a 4096 px photo (p50) | 29 ms | 29 ms | 65.9 ms |
| Encoder, warm p50 | 191 ms | 194 ms | 70.5 ms |
| Head, warm p50 | 2.1 ms | 2.2 ms | 2.0 ms |
| Decoder, warm p50: box; 10 points with `mask_input` (alternating) | 33.5; 34.1 ms | 55.5; 55.7 ms | 23.2; 22.2 ms |
| Decoder per sky region / per person, p50 | 32.7 / 28.2 ms | 55.3 / 6.8 ms | 8.4 / 24.4 ms |
| S3OD, first call / warm p50 / p95 / max | 614 / 531 / 535 / 537 ms | 929 / 540 / 1,149 / 2,493 ms | 250 / 191.1 ms |
| Memory | peak physical footprint 3,190 MB | peak physical footprint 3,314 MB | GPU 2,666 MB dedicated + 459 MB shared; working set 1,675 MB |

The macOS footprint is the process footprint (unified memory: CPU and GPU).

- The session loads in the pipeline are short because the first model pays the Core ML
  initialization and Core ML keeps compiled programs between processes; a new process with one
  model loads in 0.8 to 1.7 s (section 10).
- With ALL, the decoder time in the pipeline is the cost of the point count change (section 10).
- With the decoder on ALL, S3OD has single slow calls in the pipeline (maximum 2.5 s warm, 3.0 s on
  one photo). With the decoder on CPU_ONLY, no S3OD call took more than 614 ms. A probable cause is
  that the GPU preparation of the decoder and S3OD share the GPU; this is not measured.
- Memory: the 3,314 MB contain the Python process with PyTorch and the photo arrays. The user
  stated (2026-10-22) that the macOS budget can be somewhat larger than 3 GB because of the 16 GB
  unified memory, but not much larger. U15 sets the macOS value to 3.5 GB peak physical
  footprint of the process with all models loaded; the MVP pipeline (3,190 MB) is in it.

## 10. Stability and the decoder point count

Nine first runs per native model in new processes (`measure_graph_cost.py first-runs`,
`measure_first_runs.py --provider coreml_native`, records `native_<model>_ALL_first_runs.json`,
`s3od_native_float16_embeddings_attention_float32_ALL_first_runs.json`):

| Model | Failed | Crash reports | NaN or infinite | Session load | First run |
| --- | ---: | ---: | ---: | --- | --- |
| encoder | 0 / 9 | 0 | 0 | 771 to 781 ms | 252 to 270 ms |
| decoder | 0 / 9 | 0 | 0 | 1,363 to 1,373 ms | 48 to 60 ms |
| decoder, CPU_ONLY (`native_point_decoder_CPU_ONLY_first_runs.json`) | 0 / 9 | 0 | 0 | 1,226 to 1,243 ms | 31.0 to 31.8 ms |
| head | 0 / 9 | 0 | 0 | 1,178 to 1,193 ms | 21 to 23 ms |
| S3OD (main) | 0 / 9 | 0 | 0 | 1,602 to 1,740 ms | 630 to 699 ms |

Decoder point count (`probe_decoder_point_counts.py`, records
`native_point_decoder*_point_counts.json`, p50 per call):

| Decoder model | Compute units | Same count (2 or 10) | Alternating 2 and 10 | Counts 1 to 8 in turn |
| --- | --- | ---: | ---: | ---: |
| range of shapes (RangeDim) | ALL | 6.8 ms | 58.9 ms | 59.3 ms |
| range of shapes | CPU_AND_GPU | 6.7 ms | 59.4 ms | 59.4 ms |
| range of shapes | CPU_ONLY | 28.6 to 29.9 ms | 32.9 ms | 32.6 ms |
| enumerated shapes N = 1 to 64 | ALL | 6.7 ms | 59.6 ms | 59.4 ms |

Core ML prepares the GPU program again when the input shape changes. In the MVP the point count
changes with almost every interaction (a box has 2 points, each stroke adds 4). A fixed point count
needs padding points, and SAM 2.1 attends to every point token, so padding changes the mask.

## 11. UI on macOS

`ai_mask_app.py --provider coreml_native` stopped with SIGSEGV at the first encoder prediction, 3 of
3 starts (crash report: thread `QThread`, `take_gil` in `_PyThreadState_Attach` under
`libcoremlpython.so`). The same calls in a Python thread, in a QThread `run`, and in the app worker
pattern without the main window did not fail. The model jobs now run in one Python
`threading.Thread` with a queue (`ModelWorker`); the result goes to the UI thread with the same
signal. A probable cause: PySide gives a Qt thread a Python thread state only during a slot call,
and coremltools keeps the thread state that it saw. After the change the UI starts, loads the four
models (1,295, 183, 8, 64 ms), opens `DSC_1535`, and runs the encoder (248 ms) and the head
(30 ms). The user checks the interaction by eye.

## 12. Acceptance criteria (proposal of issue #8)

| Criterion | Result |
| --- | --- |
| Each graph creates a Core ML session with the stated settings, or the report records the real error | ONNX Runtime provider: sessions created; decoder in MLProgram stops the process (section 4). Native: all models load (sections 9, 10) |
| Same Sky and People decisions and People list as the CPU; SAM 2.1 masks IoU >= 0.999 | **met** with the native models against ONNX Runtime 1.31.0 (section 8); **not met** by ONNX Runtime 1.24.4 itself (section 7) |
| S3OD alpha IoU >= 0.998 against S3OD float32 on the CPU | **met**: 0.99907 (main native model, ALL), 0.99897 (accepted precision, CPU_AND_GPU); **not met** by the accepted precision with ALL (0.0) |
| No NaN or infinite output | **met** (sections 8, 10) |
| A table per graph: placement, load, first run, warm latency, peak memory, failures in 9 first runs | sections 5, 9, 10 (native); section 4 (provider placement) |
| MVP pipeline: stage times and peak memory against Windows and the 3 GB budget | section 9: 3,190 MB process footprint with the decoder on CPU_ONLY; **met** against the macOS value of U15 (3.5 GB) |
| No photo, crop, sheet, or overlay committed | met |

## 13. Recommendation

1. **Use native Core ML models on macOS**, loaded as `.mlmodelc` with `ComputeUnits::All` (the
   decoder with `ComputeUnits::CpuOnly`), as the SigLIP2 models are: SAM 2.1 tiny encoder,
   decoder, and head in float32, S3OD as `float16_embeddings_attention_float32`. They meet the
   output criteria of issue #8. The ONNX
   Runtime Core ML provider does not run S3OD (CPU only or the attention on the CPU) and stops the
   process with the decoder in the MLProgram format.
2. **Raise ONNX Runtime to 1.28.0 or later in Alcedo Studio** (the `ort` version with that API),
   on all platforms. ONNX Runtime 1.24 (the current `api-24`) folds the SAM 2.1 tiny position
   embedding into freed memory (microsoft/onnxruntime#29071); on this Mac the encoder output is
   wrong in every process, and on Windows the result depends on the memory reuse.
3. **Decisions (README U15, 2026-10-22).** The user selected CPU_ONLY for the decoder and gave
   the other decisions to the exploration:
   - S3OD on macOS is `float16_embeddings_attention_float32` (more float32 than the accepted
     graph, section 6); the Windows graph does not change;
   - the decoder runs with CPU_ONLY: about 30 ms for each call and no cost for a change of the
     point count (ALL: 6.8 ms with the same point count, 59 ms after a change); the other models
     run with ALL. A per-model setting exists in the native SigLIP2 code;
   - the macOS memory value is 3.5 GB peak physical footprint of the process with all models
     loaded (section 9: 3,190 MB).

## 14. Open items

- The user check of the UI interaction on macOS (section 11), together with the open check of
  Windows (U1).
- The single slow S3OD calls in the pipeline with the decoder on ALL (up to 3.0 s) and the time
  spread of S3OD between runs (531 to 709 ms) are not explained.
- The 3,190 MB footprint is not split into the models, the Python runtime, and the pipeline state.
- Not measured: the ONNX Runtime Core ML provider cost and output; the native models in the
  Alcedo Studio Rust sidecar (`coreml` crate); macOS on other machines (only the M4 with 16 GB).
- The SM1 record found a wrong EfficientSAM3 encoder output on macOS; whether it has `If` nodes
  that constant folding inlines (section 7) is not checked.

## 15. Reproduction

Machine: the Mac of section 3, repository at `~/Projects/alcedo-ai-mask-lab`, the graphs, photos,
and references under `build/tmp/explore/` (copied from the Windows machine; the SHA-256 values of
the four graphs are equal to the acceptance record). Environments and conversion:
[`scripts/ai_models/coreml/README.md`](../../scripts/ai_models/coreml/README.md).

```bash
E=build/tmp/explore; M=build/tmp/macos_coreml; R=results/macos_coreml
PY="$E/venv_ai_mask_mvp/bin/python -I"; CPY="$E/venv_coreml_convert/bin/python -I"  # onnxruntime 1.24.4 / 1.31.0
P=$E/e4/input_4096/photos; Q=build/tmp/sm1/quality/prompts.json; STROKES=$E/e3/runtime/tiny_directml.json
S3=$E/s3od/onnx_mha_chunk8/float16/subject_masks.onnx; S3F32=$E/s3od/onnx_mha_chunk8/float32/subject_masks.onnx
NE=$E/coreml/float16_embeddings_attention_float32/s3od.mlmodelc; NL=$E/coreml/float16_layer0_float32/s3od.mlmodelc
V=scripts/poc/ai_mask_mvp; X=scripts/ai_models/s3od; W=scripts/ai_models/sam2

# Section 4: ONNX Runtime Core ML provider (each graph and format).
$PY $X/profile_operator_placement.py --model s3od --graph $S3 --photo $P/DSC_1535.jpg --provider coreml \
  --coreml-model-format NeuralNetwork --profile-dir $M/profiles --out $R/s3od_mha_chunk8_float16_coreml_NeuralNetwork_placement.json
$PY $V/measure_graph_cost.py placement --graph-name point_encoder --provider coreml --coreml-model-format MLProgram \
  --photos $P --prompts $Q --features-dir $M/features_cpu --profile-dir $M/profiles --out $R/point_encoder_coreml_MLProgram_placement.json
$PY $V/measure_graph_cost.py first-runs --graph-name point_decoder --provider coreml --coreml-model-format MLProgram \
  --photos $P --prompts $Q --features-dir $M/features_cpu --count 3 --out $R/point_decoder_coreml_MLProgram_first_runs.json

# Section 7: the CPU encoder.
$PY $W/probe_cpu_encoder_settings.py --onnx $E/e4/onnx_float32_batch16/tiny/point_encoder.onnx \
  --photo $P/DSC_1535.jpg --out $R/cpu_encoder_settings_onnxruntime_1_24_4.json
$PY $W/probe_cpu_graph_optimizers.py --onnx $E/e4/onnx_float32_batch16/tiny/point_encoder.onnx \
  --photo $P/DSC_1535.jpg --out $R/cpu_encoder_optimizers_onnxruntime_1_24_4.json
# The version sweep: pip install onnxruntime==<version> in the conversion environment, then
#   $CPY $W/probe_cpu_graph_optimizers.py ... --optimizers ConstantFolding --out $R/cpu_encoder_optimizers_onnxruntime_<version>.json

# Section 8: outputs (features and the CPU reference with onnxruntime 1.31.0).
$CPY $V/measure_graph_cost.py features --photos $P --prompts $Q --features-dir $M/features_cpu
$CPY $V/compare_providers.py run --provider cpu --photos $P --strokes-report $STROKES --subject-onnx $S3F32 \
  --out-dir $M/outputs_cpu_onnxruntime_1_31_0
$PY $V/compare_providers.py run --provider coreml_native --coreml-compute-units ALL --photos $P \
  --strokes-report $STROKES --subject-onnx $S3 --subject-coreml $NE --out-dir $M/outputs_native_ALL
$PY $V/compare_providers.py compare --reference $M/outputs_cpu_onnxruntime_1_31_0 --candidate $M/outputs_native_ALL \
  --photos $P --out $R/native_ALL_outputs_against_cpu.json
#   CPU_AND_GPU: the same with --coreml-compute-units CPU_AND_GPU --subject-coreml $NL.
#   onnxruntime 1.24.4: run --provider cpu with $PY to $M/outputs_cpu, compare it as the candidate.
$PY $X/probe_subject_masks.py --photos $P --prompts $Q --subject-masks $E/e1/compare/base_plus/reference_masks \
  --alpha "s3od_float32_cpu=s3od:$E/s3od/compare_cpu/float32/alpha" \
  --alpha "s3od_native_float16_embeddings_attention_float32_ALL=s3od:$M/alpha/s3od_native_float16_embeddings_attention_float32_ALL" \
  --iou-reference s3od_float32_cpu --out $R/native_subject_quality.json

# Sections 9 and 10: cost, memory, first runs (repeat per model and compute units).
$PY $W/run_with_memory_sampling.py --out $R/native_point_encoder_ALL_cost_memory.json --log $M/encoder_cost.log -- \
  $PY $V/measure_graph_cost.py cost --graph-name point_encoder --provider coreml_native --coreml-compute-units ALL \
  --photos $P --prompts $Q --features-dir $M/features_cpu --out $R/native_point_encoder_ALL_cost.json
L=s3od_native_float16_embeddings_attention_float32_ALL
$PY $W/run_with_memory_sampling.py --out $R/${L}_cost_memory.json --log $M/${L}_cost.log -- \
  $PY $X/measure_directml_cost.py --model s3od --graph $NE --photos $P --prompts $Q --provider coreml_native \
  --coreml-compute-units ALL --alpha-dir $M/alpha/$L --out $R/${L}_cost.json
$PY $V/measure_graph_cost.py first-runs --graph-name point_encoder --provider coreml_native --coreml-compute-units ALL \
  --photos $P --prompts $Q --features-dir $M/features_cpu --out $R/native_point_encoder_ALL_first_runs.json
$PY $X/measure_first_runs.py --model s3od --graph $NE --photo $P/DSC_1535.jpg --provider coreml_native \
  --coreml-compute-units ALL --out $R/${L}_first_runs.json
$PY $W/run_with_memory_sampling.py --out $R/mvp_pipeline_native_ALL_cost_memory.json --log $M/mvp.log -- \
  $PY $V/measure_pipeline_cost.py --provider coreml_native --coreml-compute-units ALL --photos $P \
  --strokes-report $STROKES --subject-onnx $S3 --subject-coreml $NE --out $R/mvp_pipeline_native_ALL_cost.json
# U15 defaults (S3OD $NE, the decoder with CPU_ONLY, the other models with ALL):
$PY $V/compare_providers.py run --provider coreml_native --photos $P --strokes-report $STROKES \
  --subject-onnx $S3 --out-dir $M/outputs_native_ALL_decoder_CPU_ONLY
$PY $V/compare_providers.py compare --reference $M/outputs_cpu_onnxruntime_1_31_0 \
  --candidate $M/outputs_native_ALL_decoder_CPU_ONLY --photos $P --out $R/native_ALL_decoder_CPU_ONLY_outputs_against_cpu.json
$PY $W/run_with_memory_sampling.py --out $R/mvp_pipeline_native_ALL_decoder_CPU_ONLY_cost_memory.json --log $M/mvp_cpu_only.log -- \
  $PY $V/measure_pipeline_cost.py --provider coreml_native --photos $P --strokes-report $STROKES \
  --subject-onnx $S3 --out $R/mvp_pipeline_native_ALL_decoder_CPU_ONLY_cost.json
#   The decoder alone: measure_graph_cost.py cost and first-runs with --coreml-compute-units CPU_ONLY.
$PY scripts/ai_models/coreml/probe_decoder_point_counts.py --model $E/coreml/float32/point_decoder.mlmodelc \
  --features $M/features_cpu/DSC_1535.npz --compute-units ALL --out $R/native_point_decoder_ALL_point_counts.json

# Section 11: the UI.
$PY $V/ai_mask_app.py $P/DSC_1535.jpg --provider coreml_native --coreml-compute-units ALL \
  --subject-onnx $S3 --subject-coreml $NE
```

## 16. Changes

- `scripts/poc/ai_mask_mvp/mask_pipeline.py`: providers `coreml` (production Core ML provider
  settings, model format option) and `coreml_native` (`coreml_native.py`); shared provider
  arguments; profiling in `create_session`; SHA-256 and size of compiled model directories.
- New: `coreml_native.py`, `provider_placement.py`, `measure_graph_cost.py`,
  `compare_providers.py` (MVP); `scripts/ai_models/coreml/` (conversion, compute plan, S3OD,
  encoder, and decoder probes); `scripts/ai_models/sam2/macos_records.py`,
  `run_with_memory_sampling.py`, `probe_cpu_encoder_settings.py`, `probe_cpu_graph_optimizers.py`.
- `ai_mask_app.py`: the model jobs run in a Python thread; Core ML options; the Subject label names
  the Subject model.
- `measure_pipeline_cost.py`, `scripts/ai_models/s3od/` (`measure_directml_cost.py`,
  `measure_first_runs.py`, `profile_operator_placement.py`, `subject_runtime.py`): the Core ML
  providers, the macOS memory and crash records, NaN and infinite counts.
- `results/macos_coreml/` (new), this report, the acceptance record (section 6, item 1), the
  index, `results/README.md`, and README.
