# Results

Measurement results of the prototypes under `scripts/poc/`. Each step has one directory. The
report of the step explains the values; the script README states the commands that make them.

Only the text results (JSON, Markdown) are committed. Images that show the quality photos (sheets,
overlays) are written to the same directories but are not committed: `*.jpg` and `*.png` are in
`.gitignore`, because the photos must not be shared again. The binary arrays (`*.npz`) are also
local; run the commands of the script README to make them again.

| Directory | Step | Report |
| --- | --- | --- |
| [`p0_sam_feature_localization/`](p0_sam_feature_localization/) | P0, issue [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1) | [P0 report](../docs/exploration/p0_sam_feature_localization.md) |
| [`p1_people_sky_head/`](p1_people_sky_head/) | P1 (README section 5, step 1), issue [#1](https://github.com/zidage/alcedo-ai-mask-lab/issues/1) | [P1 report](../docs/exploration/p1_people_sky_head.md) |
| [`ai_mask_mvp/`](ai_mask_mvp/) | AI Mask MVP, issue [#5](https://github.com/zidage/alcedo-ai-mask-lab/issues/5) | [MVP report](../docs/exploration/mvp_ai_mask.md) |
| [`subject_alternatives/`](subject_alternatives/) | Subject models in place of BiRefNet lite, issue [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6) | [Subject alternatives report](../docs/exploration/subject_alternatives.md) |
| [`macos_coreml/`](macos_coreml/) | macOS Core ML verification of the accepted set, issue [#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8) | [macOS Core ML report](../docs/exploration/macos_coreml_verification.md) |

## `p0_sam_feature_localization/`

| File | Content | Script |
| --- | --- | --- |
| `labels.json` | Per photo: class status, reference files, cell counts | `make_label_grids.py` |
| `labels.npz` (local only) | Per photo: label grid, class fraction, positive, negative, and center cells (64 x 64) | `make_label_grids.py` |
| `score_maps.npz` (local only) | Score map per feature source, photo, and class (64 x 64, float32) | `score_prototypes.py` |
| `metrics.json` | AUC, peak precision, peak scores, separation, acceptance check, prototype cell counts | `score_prototypes.py` |
| `summary.md` | The tables of `metrics.json` for all feature sources | `score_prototypes.py` |
| `end_to_end.json` | Prompts, predicted IoU, and IoU with the reference of every decoder run | `decode_top_cells.py` |
| `sheets/` (local only) | Label grid contact sheet and score map sheets per photo | `make_label_grids.py`, `make_score_sheets.py` |

## `p1_people_sky_head/`

| File | Content | Script |
| --- | --- | --- |
| `colab_run/run.json`, `colab_run/metrics.jsonl` | Training arguments, per-epoch loss terms and validation measurements of the first Colab run | `train_head.py` |
| `colab_run/people_sky_head.json` | ONNX hash and size, the validation measurements of `best.pt`, the ONNX Runtime against PyTorch comparison | `export_head_onnx.py` |
| `colab_run/*_image_embed_progress.json` | Encoder source, checkpoint hash, device, time, CUDA memory peak per split | `encode_image_embed.py` |
| `colab_run/*_manifest_summary.json` | The split manifest without the photo list, with the SHA-256 of the full manifest and the selected photos per license | `prepare_coco_targets.py` |
| `quality_photos.json` | Per quality photo: presence, sky regions and IoU, People list with rank, score, box, IoU; timings | `evaluate_quality_photos.py` |
| `sheets/` (local only) | Sky and people overlays per photo | `make_quality_sheets.py` |
| `run2_coco_openimages/`, `run3_box_giou/` | The same files for runs 2 and 3 (`colab_run/` also has `best_val_measurements.json` of `evaluate_head.py` and the `oi_train` and `oi_val` records of `prepare_openimages_targets.py`) | as above |

## `ai_mask_mvp/`

| File | Content | Script |
| --- | --- | --- |
| `comparison_cpu.json` | The MVP (onnxruntime-directml 1.24.4, CPU provider) against `evaluate_quality_photos.py` run 2, the E3 stroke sequences, and the E2 BiRefNet alpha: per check and photo, the differences, IoU, and result | `compare_with_measurement_scripts.py` |
| `comparison_cpu_onnxruntime_1_31.json` | The same comparison in the reference environment (onnxruntime 1.31.0) | `compare_with_measurement_scripts.py` |
| `cost_directml.json` | Graph load times, stage times per photo and warm, decoder times per use, working set and private bytes, with all models loaded on DirectML | `measure_pipeline_cost.py` |
| `cost_directml_gpu_memory.json` | Peak dedicated and shared GPU memory and the samples of that run | `run_with_gpu_memory.ps1` |
| `birefnet_memory_phases.json` | GPU memory of the BiRefNet DirectML session per phase, and the refused arena shrinkage run options | `measure_birefnet_memory_phases.py` |
| `birefnet_load_time.json` | BiRefNet session creation time per graph optimization level and with a saved optimized graph | `measure_birefnet_load_time.py` |
| `gpu_reset_events.json` | Windows System events 4101 and `nvlddmkm` 153 (display driver resets) of 2026-10-21 | `Get-WinEvent` (MVP report, section 9) |

## `subject_alternatives/`

Scripts: [`scripts/ai_models/s3od/`](../scripts/ai_models/s3od/README.md),
[`scripts/ai_models/isnet/`](../scripts/ai_models/isnet/README.md), and
[`scripts/ai_models/universal_matting/`](../scripts/ai_models/universal_matting/README.md).
`<graph>` is one of `s3od_float32`, `s3od_float16` (the export of `export_onnx.py`),
`s3od_cpu_resize_float32`, `s3od_cpu_resize_float16` (`--keep-identity-resize`), `isnet_float32`,
`isnet_float16` (the IS-Net general-use export), `universal_matting_float32`,
`universal_matting_float16` (the universal matting conversion), and `birefnet_lite_float16` (the
MVP graph). The files with `_float16_overflow` (float16 attention scores in all layers: NaN output on
DirectML) and `_float16_all_attention_float32` (all attention scores in float32) record the
earlier float16 forms of report section 4.3; they are not candidates.

| File | Content | Script |
| --- | --- | --- |
| `s3od_comparison_cpu.json` | ONNX Runtime CPU against the PyTorch model of the source, per graph and photo: mask and IoU score differences, selected mask, photo-size IoU | `compare_onnx_outputs.py` |
| `s3od_float32_spread.json` | Float32 differences of PyTorch against itself and of ONNX Runtime with and without graph optimizations | `measure_float32_spread.py` |
| `<graph>_placement.json` | DirectML partitions and CPU nodes from an ONNX Runtime profile | `profile_operator_placement.py` |
| `<graph>_cost.json` | Session load, first run, warm and per-photo times, GPU memory and working set per phase, graph hash | `measure_directml_cost.py` |
| `<graph>_cost_gpu_memory.json` | Peak dedicated and shared GPU memory and the samples of that run | `run_with_gpu_memory.ps1` |
| `<graph>_first_runs.json` | First runs in new processes, with the System events 153 and 4101 of each run | `measure_first_runs.py` |
| `quality.json` | Per alpha and photo: foreground, soft share, edge width, parts, holes, islands, subject recall, outside box, IoU with the BiRefNet lite and the S3OD float32 masks; means | `probe_subject_masks.py` |
| `gpu_reset_events.json` | Windows System events 4101 and `nvlddmkm` 153 of 2026-10-21 17:50 to the end of the runs | `Get-WinEvent` (report section 5.4) |
| `isnet_comparison_cpu.json` | IS-Net ONNX Runtime CPU against the PyTorch model, per graph and photo: d1 and alpha differences, photo-size IoU, preprocessing difference to `Inference.py` | `isnet/compare_onnx_outputs.py` |
| `isnet_<precision>_export.json` | IS-Net export record: sources, hashes, operators; float16: activation maxima of the range check | `isnet/export_onnx.py` |
| `quality_isnet.json` | The metrics of `quality.json` for the IS-Net alphas (PyTorch, ONNX CPU, DirectML), with the IoU with the BiRefNet lite, S3OD float32, and IS-Net PyTorch masks | `probe_subject_masks.py` |
| `mvp_pipeline_isnet_float16_cost.json`, `..._gpu_memory.json` | The MVP pipeline with the IS-Net float16 graph as the Subject graph (report section 11.5) | `measure_pipeline_cost.py`, `run_with_gpu_memory.ps1` |
| `birefnet_lite_float16_isnet_step_failed_runs.json`, `birefnet_lite_float16_isnet_step_attempt{1,2}_cost_gpu_memory.json` | The two BiRefNet reference cost runs of the IS-Net step: phases until the error, the `887A0006` error, the System events 153 and 4101 (report section 11.4) | log of `measure_directml_cost.py`, `Get-WinEvent` |
| `universal_matting_comparison_cpu.json` | Universal matting ONNX Runtime CPU against TensorFlow, per graph and photo: alpha differences (also against TensorFlow with grappler off), IoU, branch tensors, the uint8 output check | `universal_matting/compare_onnx_outputs.py` |
| `universal_matting_tensorflow_spread.json` | TensorFlow with six CPU settings against each other and against the ONNX float32 alpha | `universal_matting/measure_tensorflow_spread.py` |
| `universal_matting_<precision>_export.json` | Universal matting conversion record: sources, hashes, graph rules, the rewrite check, operators, the float64 divisions; float16: activation maxima and float16 nodes per U-Net | `universal_matting/export_onnx.py` |
| `quality_universal_matting.json` | The metrics of `quality.json` for the universal matting alphas (TensorFlow, ONNX CPU, DirectML), with the IoU with the BiRefNet lite, S3OD float32, and TensorFlow masks | `probe_subject_masks.py` |
| `mvp_pipeline_universal_matting_float32_cost.json`, `..._gpu_memory.json` | The MVP pipeline with the universal matting float32 graph as the Subject graph (report section 12.5) | `measure_pipeline_cost.py`, `run_with_gpu_memory.ps1` |
| `*_float32_divisions.json` | Quality, float32 cost, and float32 first runs of the first universal matting graph (scalar divisions in float32; DirectML IoU 0.936 on `DSC00674`, report section 12.3) | as above, graph of `export_onnx.py --scalar-divisions float32` |
| `birefnet_lite_float16_universal_matting_step_failed_runs.json`, `..._cost_gpu_memory.json` | The BiRefNet reference cost run of the universal matting step: phases until the error, the `887A0006` error, the System events 153 and 4101 (report section 12.4) | log of `measure_directml_cost.py`, `Get-WinEvent` |

## `macos_coreml/`

Scripts: [`scripts/ai_models/coreml/`](../scripts/ai_models/coreml/README.md),
[`scripts/poc/ai_mask_mvp/`](../scripts/poc/ai_mask_mvp/README.md),
[`scripts/ai_models/s3od/`](../scripts/ai_models/s3od/README.md), and
[`scripts/ai_models/sam2/`](../scripts/ai_models/sam2/README.md). Machine: Apple M4, 16 GB,
macOS 27.0.1. `<model>` is `point_encoder`, `point_decoder`, `people_sky_head`, or `s3od`.

| File | Content | Script |
| --- | --- | --- |
| `<graph>_coreml_<format>_placement.json` | ONNX Runtime Core ML provider (NeuralNetwork or MLProgram): Core ML capability log, executed nodes and kernel time per provider, CPU nodes per operator | `profile_operator_placement.py`, `measure_graph_cost.py placement` |
| `point_decoder_coreml_MLProgram_first_runs.json` | The decoder in the MLProgram format: 3 first runs, the MPSGraph assertion, the crash reports | `measure_graph_cost.py first-runs` |
| `native_<model>_convert.json`, `native_s3od_<precision>_convert.json`, `native_point_decoder_enumerated_convert.json` | Native Core ML conversion records: precision, float32 ops, checkpoint and model SHA-256, difference against PyTorch | `convert_models.py` |
| `native_<model>_<units>_compute_plan.json`, `native_s3od_*_compute_plan.json` | Core ML compute plan: ops and estimated cost per preferred device | `profile_compute_plan.py` |
| `native_s3od_variants/<variant>_*.json` | S3OD precision variants: conversion record, compute plan, output per compute units on two photos against the CPU float32 alpha | `convert_models.py`, `profile_compute_plan.py`, `probe_s3od_compute_units.py` |
| `native_point_encoder_outputs_photos.json`, `native_point_encoder_outputs_random.json` | Encoder outputs of Core ML per compute units, ONNX Runtime 1.24.4 CPU, and PyTorch against each other | `probe_encoder_outputs.py` |
| `cpu_encoder_settings_onnxruntime_1_24_4.json` | The ONNX Runtime 1.24.4 CPU encoder against PyTorch per graph optimization level and thread count | `probe_cpu_encoder_settings.py` |
| `cpu_encoder_optimizers_onnxruntime_<version>.json` | The encoder with one graph optimizer disabled at a time (1.24.4: 49 optimizers; other versions and Windows 1.24.4: `ConstantFolding`) | `probe_cpu_graph_optimizers.py` |
| `cpu_onnxruntime_1_24_4_outputs_against_1_31_0.json` | The MVP outputs of ONNX Runtime 1.24.4 CPU against 1.31.0 CPU | `compare_providers.py` |
| `native_ALL_outputs_against_cpu.json`, `native_CPU_AND_GPU_outputs_against_cpu.json`, `native_ALL_decoder_CPU_ONLY_outputs_against_cpu.json` | The MVP outputs of the native models against ONNX Runtime 1.31.0 CPU: the checks of issue #8 per photo | `compare_providers.py` |
| `native_subject_quality.json` | Subject quality metrics of the native S3OD forms and S3OD float32 CPU | `probe_subject_masks.py` |
| `native_<model>_ALL_cost.json`, `native_point_decoder_CPU_ONLY_cost.json`, `s3od_native_<precision>_<units>_cost.json` (and `_cost_memory.json`) | Session load, first run, warm runs, one run per photo, physical footprint per phase; the sampled peak footprint, system memory, and crash reports | `measure_graph_cost.py cost`, `measure_directml_cost.py`, `run_with_memory_sampling.py` |
| `native_<model>_ALL_first_runs.json`, `native_point_decoder_CPU_ONLY_first_runs.json`, `s3od_native_float16_embeddings_attention_float32_ALL_first_runs.json` | Nine first runs in new processes: load, first run, exit code, crash reports | `measure_graph_cost.py first-runs`, `measure_first_runs.py` |
| `native_point_decoder*_point_counts.json` | Decoder call time with the same, alternating, and changing point counts (range and enumerated shapes) | `probe_decoder_point_counts.py` |
| `mvp_pipeline_native_ALL_decoder_CPU_ONLY_cost*.json` (U15 defaults), `mvp_pipeline_native_ALL_cost*.json` (all ALL) | The MVP pipeline with all native models: load times, stage times, warm runs, memory | `measure_pipeline_cost.py`, `run_with_memory_sampling.py` |
