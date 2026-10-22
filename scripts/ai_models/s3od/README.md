# S3OD export and measurement tools (issue #6)

Issue [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6) measures Subject models as
alternatives to BiRefNet lite (F1 Subject, F2 Background). These scripts export the S3OD
`s3od-synth` model to ONNX, compare the export with PyTorch, and measure the cost and the quality on
ONNX Runtime DirectML with the MVP settings. The same DirectML and quality scripts measure the
BiRefNet lite 1024 float16 graph of the MVP as the reference, the IS-Net general-use graph
(`--model isnet`, kind `isnet`; export and CPU comparison in
[`scripts/ai_models/isnet/`](../isnet/README.md)), and the Alibaba universal matting graph
(`--model universal_matting`, kind `universal_matting`; conversion and CPU comparison in
[`scripts/ai_models/universal_matting/`](../universal_matting/README.md)). The report is
[Subject model alternatives to BiRefNet lite](../../../docs/exploration/subject_alternatives.md) (section S3OD).

Weights, source checkouts, ONNX files, photos, alphas, and logs are not committed. Keep them under
`build/tmp/explore/`.

## Sources

| Item | Source | Revision | License |
| --- | --- | --- | --- |
| Weights `s3od.pt` (464,526,126 bytes, SHA-256 `c67f26415ad5493edaac48c8f2a08fecb1d9933aed2ed473e8ad713b93b8e5f3`) | Hugging Face `okupyn/s3od-synth` | `4b40ce663c31fdfbb446a347282960033e624b4d` | MIT (model card `license: mit`) |
| Model code (`s3od/s3od/model.py`, `predictor.py`, `utils.py`, `dinov3_config/`) | GitHub `KupynOrest/s3od` | `11385423e8d04d78b19e71cb166971b79c49f4fe` | MIT (`LICENSE`, SHA-256 `bcb7a447...`) |
| DINOv3 ViT-B/16 encoder (part of the weights) | `facebookresearch/dinov3` | — | DINOv3 License (commercial use permitted; the license text must go with the weights) |

`s3od.pt` is a `torch.save` dictionary with one `state_dict` (371 tensors, float32), not a
TorchScript file. The scripts load it with `torch.load(..., weights_only=True)`. The model has
116,097,990 parameters.

## Files

| File | Purpose |
| --- | --- |
| `s3od_graphs.py` | Model loading from the source checkout, the preprocessing and postprocessing of the source (letterbox to 1024, mask selection, unpadding), the export wrapper, the attention in query chunks (`use_query_chunked_attention`, the `com.microsoft.MultiHeadAttention` operator), the attention score maxima per encoder layer. |
| `export_onnx.py` | Float32 graph (dynamo exporter, opset 18, Reshape `allowzero` change, Resize to the same size removed, optional attention in query chunks) and float16 graph (attention of the layers with large scores in float32), with `export_report.json`. |
| `probe_attention_layer.py` | One encoder attention on DirectML in different graph forms (MHA, MatMul form, query chunks, chained query chunks): GPU memory, time, float16 output. |
| `compare_onnx_outputs.py` | `OnnxMasksMatchPytorchWithinTolerance`: ONNX Runtime CPU against the PyTorch model of the source on the quality photos; writes the reference and ONNX alphas. |
| `measure_float32_spread.py` | Float32 differences of PyTorch against itself (threads, attention kernel) and of ONNX Runtime with and without graph optimizations. |
| `subject_runtime.py` | Shared code of the DirectML scripts: preprocessing and stored alpha of S3OD, BiRefNet, IS-Net, and universal matting, MVP session settings, GPU memory and working set. |
| `profile_operator_placement.py` | Nodes per execution provider (DirectML partitions, CPU nodes) from an ONNX Runtime profile. |
| `measure_directml_cost.py` | Session load, first run, 20 warm runs, one run per photo, release, with the GPU memory after each phase; stores the DirectML alphas. |
| `measure_first_runs.py` | First runs in new processes and the display driver resets (System events 153 and 4101). |
| `probe_subject_masks.py` | Quality metrics of E2 on a common grid (long edge 1024) and the IoU with the BiRefNet lite mask. |
| `requirements.txt` | Packages of the export and comparison environment. |

`scripts/ai_models/sam2/run_with_gpu_memory.ps1` records the peak GPU memory of a run.

## Graph

| File | Inputs | Outputs |
| --- | --- | --- |
| `subject_masks.onnx` | `pixel_values` [1, 3, 1024, 1024] float32 | `masks` [1, 3, 1024, 1024], `iou_scores` [1, 3], float32, after the sigmoid |

Data rules (the rules of `s3od.predictor.BackgroundRemoval`):

- Scale the RGB photo so that the long edge is 1024 px (`round` of each side, OpenCV
  `INTER_LINEAR`). Put it in the middle of a black 1024 x 1024 image (padding `(1024 - side) // 2`
  at the top or the left). Pixel values are `(value / 255 - mean) / std` with the ImageNet mean
  `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- The alpha is the mask with the largest `iou_scores` value. Remove the padding, then resize it to
  the photo size (bilinear, `align_corners=False`).
- Float32 graph: dynamo exporter (`torch.onnx.export(..., dynamo=True)`), opset 18. The TorchScript
  exporter has no symbolic for `aten::_upsample_bilinear2d_aa` (the `antialias=True` resize of the
  mask head); the dynamo exporter writes `Resize` with `antialias=1`.
- Reshape `allowzero`: the exporter writes `allowzero=1` on 51 Reshape nodes. ONNX Runtime
  DirectML 1.24.4 refuses that graph at session creation with `80070057` (E_INVALIDARG,
  `MLOperatorAuthorImpl.cpp(2853)`). `export_onnx.py` sets `allowzero=0` on these nodes; their
  shapes are constants without a 0, so the output does not change. `--keep-reshape-allowzero`
  writes the graph without the change.
- Resize to the same size: the mask head resizes its 1024 x 1024 map to 1024 x 1024 with
  `antialias=True`, which gives a bitwise equal tensor. DirectML has no antialias `Resize`, so
  ONNX Runtime runs that node on the CPU. `export_onnx.py` removes it (one DirectML partition).
  `--keep-identity-resize` keeps it (the "CPU Resize" form: 4 DirectML partitions and one CPU
  node, less GPU memory and more time; see the report).
- Float16 graph: `convert_float_to_float16` with `keep_io_types=True` (float32 inputs and
  outputs), the default operator block list of the converter, and the 11 attention score MatMul
  nodes and 11 Softmax nodes in float32. The layer 0 attention scores reach 134,850, more than the
  float16 maximum; with these nodes in float16 the DirectML output is NaN on every photo. The
  script removes the float32 -> float16 -> float32 Cast pairs that the converter puts between the
  blocked nodes. ONNX Runtime CPU does not show the overflow (it runs most float16 operators in
  float32), so check a float16 graph on DirectML. The rule is per encoder layer: the attention
  nodes of a layer stay in float32 when the largest absolute score of the layer on the
  calibration photos (PyTorch) is above 65,504 / 16; only layer 0 is above it.
- Attention in query chunks (`--query-chunks N`, `--attention-kernel`): each encoder attention is
  computed in N query chunks; chunk k + 1 adds 0 x max(output of chunk k) to its query, so that the
  chunks are a sequence and DirectML uses the memory of one score tensor for all chunks. The
  output is the same attention. `--attention-kernel multi_head_attention` writes one
  `com.microsoft.MultiHeadAttention` node per chunk; `sdpa` writes MatMul, Softmax, MatMul. The
  selected form is 8 chunks with `multi_head_attention`, float16 (report section 5.5: 191 ms,
  1,358 MB on DirectML). The DirectML `MultiHeadAttention` alone (1 chunk) uses the same memory as
  the MatMul form. The float16 converter writes Cast nodes with the same name for the shared key
  and value of the chunks; the script merges them.

## Commands

```bash
# Sources and weights (Git Bash, Windows). The export environment is the E2 BiRefNet environment
# (scripts/ai_models/birefnet/README.md); requirements.txt lists its packages.
E=build/tmp/explore; W=$E/weights/s3od_synth
git clone https://github.com/KupynOrest/s3od.git $E/src/s3od
git -C $E/src/s3od checkout 11385423e8d04d78b19e71cb166971b79c49f4fe
mkdir -p $W
curl -sSL -o $W/s3od.pt \
  https://huggingface.co/okupyn/s3od-synth/resolve/4b40ce663c31fdfbb446a347282960033e624b4d/s3od.pt
# revisions.json: {"huggingface": {"repository": "okupyn/s3od-synth", "revision": "4b40ce66..."},
#                  "github": {"repository": "KupynOrest/s3od", "commit": "11385423..."}}

PY="$E/venv_birefnet/Scripts/python -I"; S=$E/s3od; X=scripts/ai_models/s3od
P=$E/e4/input_4096/photos; Q=build/tmp/sm1/quality/prompts.json; R=results/subject_alternatives
# Two graph forms: onnx/ (Resize to the same size removed) and onnx_cpu_resize/ (kept).
$PY $X/export_onnx.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --revisions $W/revisions.json --out-root $S/onnx
$PY $X/export_onnx.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --revisions $W/revisions.json --out-root $S/onnx_cpu_resize --keep-identity-resize
$PY $X/compare_onnx_outputs.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --graph float32=$S/onnx/float32/subject_masks.onnx \
  --graph float16=$S/onnx/float16/subject_masks.onnx \
  --graph cpu_resize_float32=$S/onnx_cpu_resize/float32/subject_masks.onnx \
  --graph cpu_resize_float16=$S/onnx_cpu_resize/float16/subject_masks.onnx \
  --required float32 --required cpu_resize_float32 \
  --photos $P --prompts $Q --out $S/compare_cpu/compare.json
cp $S/compare_cpu/compare.json $R/s3od_comparison_cpu.json
$PY $X/measure_float32_spread.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --graph $S/onnx/float32/subject_masks.onnx --photo $P/P1024439.jpg --photo $P/DSC00562.jpg \
  --out $R/s3od_float32_spread.json

# Attention in query chunks (report section 5.5). Graph directories onnx_mha_chunk<N> (N = 4, 8,
# 16, 32) and onnx_sdpa_chunk8 (--attention-kernel sdpa).
$PY $X/export_onnx.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --revisions $W/revisions.json --out-root $S/onnx_mha_chunk8 \
  --calibration-photos $P --calibration-prompts $Q \
  --attention-kernel multi_head_attention --query-chunks 8
cp $S/onnx_mha_chunk8/float32/export_report.json $R/s3od_mha_chunk8_float32_export.json
cp $S/onnx_mha_chunk8/float16/export_report.json $R/s3od_mha_chunk8_float16_export.json
$PY $X/compare_onnx_outputs.py --source $E/src/s3od/s3od --checkpoint $W/s3od.pt \
  --graph mha_chunk8_float32=$S/onnx_mha_chunk8/float32/subject_masks.onnx \
  --graph mha_chunk8_float16=$S/onnx_mha_chunk8/float16/subject_masks.onnx \
  --graph sdpa_chunk8_float32=$S/onnx_sdpa_chunk8/float32/subject_masks.onnx \
  --graph sdpa_chunk8_float16=$S/onnx_sdpa_chunk8/float16/subject_masks.onnx \
  --required mha_chunk8_float32 --required sdpa_chunk8_float32 \
  --photos $P --prompts $Q --out $S/compare_cpu_chunk8/compare.json
cp $S/compare_cpu_chunk8/compare.json $R/s3od_chunk8_comparison_cpu.json
# One encoder attention (capture and build in this environment, run in the MVP environment).
$PY $X/probe_attention_layer.py capture --work-dir $S/attention_probe --source $E/src/s3od/s3od \
  --checkpoint $W/s3od.pt --photo $P/DSC_1535.jpg
$PY $X/probe_attention_layer.py build --work-dir $S/attention_probe
$E/venv_ai_mask_mvp/Scripts/python -I $X/probe_attention_layer.py run \
  --work-dir $S/attention_probe --out $R/attention_layer0_probe.json
```

```bash
# DirectML (MVP environment). One model job at a time. Run each command for each graph:
#   label s3od_float32             --model s3od      $S/onnx/float32/subject_masks.onnx
#   label s3od_float16             --model s3od      $S/onnx/float16/subject_masks.onnx
#   label s3od_cpu_resize_float32  --model s3od      $S/onnx_cpu_resize/float32/subject_masks.onnx
#   label s3od_cpu_resize_float16  --model s3od      $S/onnx_cpu_resize/float16/subject_masks.onnx
#   label birefnet_lite_float16    --model birefnet  $E/e2/onnx_fp16/lite/subject_matte.onnx (MVP)
#   label s3od_mha_chunk<N>_<p>    --model s3od      $S/onnx_mha_chunk<N>/<p>/subject_masks.onnx
#   label s3od_sdpa_chunk8_<p>     --model s3od      $S/onnx_sdpa_chunk8/<p>/subject_masks.onnx
PY="$E/venv_ai_mask_mvp/Scripts/python -I"; L=s3od_float16; M=s3od
G=$S/onnx/float16/subject_masks.onnx
$PY $X/profile_operator_placement.py --model $M --graph $G --photo $P/DSC_1535.jpg \
  --profile-dir $S/placement/profiles_$L --out $R/${L}_placement.json
$PY $X/measure_first_runs.py --model $M --graph $G --photo $P/DSC_1535.jpg --count 9 \
  --out $R/${L}_first_runs.json
```

```powershell
# Cost with the peak GPU memory (PowerShell, from the repository root). Repeat for each graph.
$e = "build\tmp\explore"; $py = "$e\venv_ai_mask_mvp\Scripts\python.exe"; $r = "results\subject_alternatives"
$l = "s3od_float16"; $m = "s3od"; $g = "$e\s3od\onnx\float16\subject_masks.onnx"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
  -Arguments @('-I', 'scripts\ai_models\s3od\measure_directml_cost.py', '--model', $m,
               '--graph', $g, '--photos', "$e\e4\input_4096\photos",
               '--prompts', 'build\tmp\sm1\quality\prompts.json',
               '--alpha-dir', "$e\s3od\directml\$l\alpha", '--out', "$r\${l}_cost.json") `
  -Out "$r\${l}_cost_gpu_memory.json" -Log "$e\s3od\directml\${l}_cost.log"

# The MVP pipeline with S3OD as the Subject graph (report section 5.5).
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
  -Arguments @('-I', 'scripts\poc\ai_mask_mvp\measure_pipeline_cost.py', '--provider', 'directml',
               '--photos', "$e\e4\input_4096\photos",
               '--strokes-report', "$e\e3\runtime\tiny_directml.json",
               '--subject-onnx', "$e\s3od\onnx_mha_chunk8\float16\subject_masks.onnx",
               '--out', "$r\mvp_pipeline_s3od_mha_chunk8_float16_cost.json") `
  -Out "$r\mvp_pipeline_s3od_mha_chunk8_float16_cost_gpu_memory.json" `
  -Log "$e\s3od\directml\mvp_pipeline_s3od_mha_chunk8_float16.log"
```

```bash
# Quality metrics (export environment). The BiRefNet lite float16 CPU alpha is the MVP reference
# output (scripts/poc/ai_mask_mvp/README.md, "The comparison needs ...").
PY="$E/venv_birefnet/Scripts/python -I"; D=$S/directml
$PY $X/probe_subject_masks.py --photos $P --prompts $Q \
  --subject-masks $E/e1/compare/base_plus/reference_masks \
  --alpha "birefnet_lite_float16_cpu=birefnet:build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha" \
  --alpha "birefnet_lite_float16_directml=birefnet:$D/birefnet_lite_float16/alpha" \
  --alpha "s3od_float32_cpu=s3od:$S/compare_cpu/float32/alpha" \
  --alpha "s3od_float16_cpu=s3od:$S/compare_cpu/float16/alpha" \
  --alpha "s3od_float32_directml=s3od:$D/s3od_float32/alpha" \
  --alpha "s3od_float16_directml=s3od:$D/s3od_float16/alpha" \
  --alpha "s3od_cpu_resize_float32_directml=s3od:$D/s3od_cpu_resize_float32/alpha" \
  --alpha "s3od_cpu_resize_float16_directml=s3od:$D/s3od_cpu_resize_float16/alpha" \
  --iou-reference birefnet_lite_float16_cpu --iou-reference s3od_float32_cpu --out $R/quality.json \
  --region "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --region "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
  --region "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --region "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
  --region "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --region "DSC00936.jpg:person:0.47,0.63,0.68,0.93"
# Attention in query chunks: the same command with
#   --alpha "birefnet_lite_float16_cpu=..." --alpha "s3od_float32_cpu=..."
#   --alpha "s3od_mha_chunk8_float16_directml=s3od:$D/s3od_mha_chunk8_float16/alpha"
#   --alpha "s3od_mha_chunk8_float32_directml=s3od:$D/s3od_mha_chunk8_float32/alpha"
#   --out $R/quality_chunked_attention.json
```
