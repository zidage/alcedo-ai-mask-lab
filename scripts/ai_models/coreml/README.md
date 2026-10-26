# Native Core ML models of the accepted set (issue #8)

Issue [#8](https://github.com/zidage/alcedo-ai-mask-lab/issues/8) verifies the accepted AI Mask
model set on macOS. The ONNX Runtime Core ML execution provider does not run the set as it is (the
report, section 4). On 2026-10-22 the user decided to try native Core ML models for macOS, as
Alcedo Studio does for its SigLIP2 models (`rust/puerh_mind/src/service/coreml_clip.rs`: a
precompiled `.mlmodelc`, loaded with the `coreml` crate and `ComputeUnits::All`). These scripts
convert the PyTorch models of the set with coremltools, compile them, and measure where Core ML
runs them and whether the output is correct. The report is
[macOS Core ML verification of the accepted model set](../../../docs/exploration/macos_coreml_verification.md).

Models, weights, source checkouts, photos, and alphas are not committed. Keep them under
`build/tmp/explore/` on the Mac.

## Files

| File | Purpose |
| --- | --- |
| `convert_models.py` | PyTorch to Core ML (MLProgram, macOS 15 deployment target), `.mlpackage` and compiled `.mlmodelc`, with a record (`<model>.json`): precision, the float32 ops, checkpoint SHA-256, model SHA-256, and the difference against PyTorch on one input. |
| `profile_compute_plan.py` | The Core ML compute plan (`MLComputePlan`) of a compiled model: ops and estimated cost per preferred device (CPU, GPU, Neural Engine). |
| `probe_s3od_compute_units.py` | S3OD output per compute units on photos against the ONNX Runtime CPU float32 alpha (IoU of alpha > 0.5, largest difference, repeat calls). |
| `probe_encoder_outputs.py`, `probe_decoder_point_counts.py` | See below. |
| `probe_sky_head.py` | Sky head (AlcedoStudio issue 538): the sky alpha of each Core ML conversion and compute units against ONNX Runtime CPU on photos, alone and after the Core ML encoder, and its run time. |
| `requirements.txt` | Packages of the conversion environment. |

The runtime of the native models is `scripts/poc/ai_mask_mvp/coreml_native.py` (provider
`coreml_native` of `mask_pipeline.py`). The measurement scripts of the MVP and of S3OD accept
`--provider coreml_native --coreml-compute-units <units>`; the MVP pipeline runs the decoder with
`--coreml-decoder-compute-units` (default CPU_ONLY, README U15).

## Models

| Model | PyTorch module | Inputs | Outputs | Precision |
| --- | --- | --- | --- | --- |
| `point_encoder.mlmodelc` | `sam2_graphs.PointEncoder` (SAM 2.1 tiny); the Hiera position embedding (bicubic resize of a weight to the 256 x 256 token grid) is a constant | `pixel_values` [1, 3, 1024, 1024] | `image_embed`, `high_res_feats_0`, `high_res_feats_1` | float32 |
| `point_decoder.mlmodelc` | `sam2_graphs.PointDecoder` | the encoder outputs, `point_coords` [1, N, 2], `point_labels` [1, N] (N = 1 to 64, flexible shape), `mask_input` [1, 1, 256, 256], `has_mask_input` [1] | `masks`, `iou_predictions`, `object_score_logits` | float32 |
| `people_sky_head.mlmodelc` | `head_model.PeopleSkyHead`, P1 run 2 `best.pt` (epoch 15) | `image_embed` | `head_model.OUTPUT_NAMES` | float32 |
| `sky_head.mlmodelc` | `sky_head_model.ExportedSkyHead`, sky head run 1 `best.pt` (epoch 15; the graph of `sky_head.onnx`) | `image_embed`, `high_res_feats_0`, `high_res_feats_1`, `pixel_values` [1, 3, 1024, 1024] | `sky_alpha` [1, 1, 1024, 1024] | float32 (`float16` measured and rejected, [report](../../../docs/exploration/sky_head_mass13k.md) section 11) |
| `s3od.mlmodelc` | `s3od_graphs.subject_masks_module` with the attention of the source (no query chunks) | `pixel_values` [1, 3, 1024, 1024] | `masks` [1, 3, 1024, 1024], `iou_scores` [1, 3] | `float16_embeddings_attention_float32` (the macOS model, README U15: patch embedding and encoder attention float32, other ops float16), `float16_encoder_float32` (DINOv3 encoder float32, DPT decoder float16), or `float16_layer0_float32` (float16, layer 0 attention float32, the precision of the accepted ONNX graph) |

The preprocessing and postprocessing are the rules of the ONNX graphs (`sam2_graphs.py`,
`s3od_graphs.py`, `head_model.py`).

## Environments (Mac)

The conversion needs PyTorch 2.7.0 and NumPy 2.2: coremltools 9.0 is tested with PyTorch up to
2.7.0, and its PyTorch frontend calls `int()` on one-element arrays, which NumPy 2.5 refuses
(`TypeError: only 0-dimensional arrays can be converted to Python scalars`). The measurements run
in the MVP environment (onnxruntime 1.24.4, coremltools 9.0 for `CompiledMLModel`).

```bash
E=build/tmp/explore
/opt/homebrew/bin/python3.13 -m venv $E/venv_coreml_convert
$E/venv_coreml_convert/bin/python -m pip install -r scripts/ai_models/coreml/requirements.txt
/opt/homebrew/bin/python3.13 -m venv $E/venv_ai_mask_mvp
$E/venv_ai_mask_mvp/bin/python -m pip install onnxruntime==1.24.4 coremltools==9.0 \
  PySide6==6.12.0 numpy==2.5.2 pillow==12.3.0 opencv-python==5.0.0.93 scipy==1.18.1 \
  psutil==7.2.2 torch==2.14.1 torchvision==0.29.1 onnx
```

## Commands (from the repository root on the Mac)

```bash
E=build/tmp/explore; M=build/tmp/macos_coreml; R=results/macos_coreml
CPY="$E/venv_coreml_convert/bin/python -I"; PY="$E/venv_ai_mask_mvp/bin/python -I"
P=$E/e4/input_4096/photos; Q=build/tmp/sm1/quality/prompts.json
# Sources: facebookresearch/sam2 at 2b90b9f5 ($E/src/sam2), sam2.1_hiera_tiny.pt,
# KupynOrest/s3od at 11385423 ($E/src/s3od/s3od), s3od.pt, P1 run 2 run/best.pt.
for G in point_encoder point_decoder people_sky_head; do
  $CPY scripts/ai_models/coreml/convert_models.py --model $G --precision float32 --out-dir $E/coreml/float32
done
$CPY scripts/ai_models/coreml/convert_models.py --model s3od --precision float16_embeddings_attention_float32 \
  --out-dir $E/coreml/float16_embeddings_attention_float32
$CPY scripts/ai_models/coreml/convert_models.py --model s3od --precision float16_encoder_float32 \
  --out-dir $E/coreml/float16_encoder_float32
$CPY scripts/ai_models/coreml/convert_models.py --model s3od --precision float16_layer0_float32 \
  --out-dir $E/coreml/float16_layer0_float32
# Sky head (AlcedoStudio issue 538; sky head run 1 best.pt): float32 is the macOS model.
for PR in float32 float16; do
  $CPY scripts/ai_models/coreml/convert_models.py --model sky_head --precision $PR --out-dir $E/coreml/$PR
done

# Placement (compute plan) and the S3OD output per compute units.
$PY scripts/ai_models/coreml/profile_compute_plan.py --model $E/coreml/float32/point_encoder.mlmodelc \
  --compute-units ALL --out $R/native_point_encoder_ALL_compute_plan.json
$PY scripts/ai_models/coreml/probe_s3od_compute_units.py \
  --model $E/coreml/float16_layer0_float32/s3od.mlmodelc --compute-units ALL CPU_AND_GPU CPU_AND_NE CPU_ONLY \
  --photos $P --photo DSC_1535 --photo P1024439 --reference-alpha $E/s3od/compare_cpu/float32/alpha \
  --out $R/native_s3od_variants/float16_layer0_float32_outputs.json
# Sky head against ONNX Runtime CPU: the conversion environment with onnxruntime==1.31.0 added;
# --onnx-dir holds point_encoder.onnx and sky_head.onnx; MaSS13K photos of the test split.
$CPY scripts/ai_models/coreml/probe_sky_head.py --onnx-dir <onnx dir> \
  --encoder $E/coreml/float32/point_encoder.mlmodelc \
  --sky-head float32=$E/coreml/float32/sky_head.mlmodelc --sky-head float16=$E/coreml/float16/sky_head.mlmodelc \
  --compute-units ALL CPU_AND_GPU CPU_AND_NE CPU_ONLY --photos $P --photos <MaSS13K photos> \
  --runs 3 --out $R/native_sky_head_probe.json
```

The S3OD Neural Engine investigation (`--float32-op-types`, `--float32-scopes`, section 6 of the
report) and all measurements of the set are in the report, section 15.

| File | Purpose |
| --- | --- |
| `probe_encoder_outputs.py` | SAM 2.1 encoder: Core ML per compute units, ONNX Runtime CPU, and PyTorch float32 against each other on photos or a random input. |
| `probe_decoder_point_counts.py` | Decoder call time with the same, alternating, and changing point counts. |
