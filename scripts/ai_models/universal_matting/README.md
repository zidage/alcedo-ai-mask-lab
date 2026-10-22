# Alibaba universal matting export and comparison tools (issue #6)

Issue [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6) measures Subject models as
alternatives to BiRefNet lite (F1 Subject, F2 Background). These scripts convert the Alibaba
universal matting model (ModelScope `iic/cv_unet_universal-matting`, TensorFlow 1 frozen graph
`tf_graph.pb`) to ONNX and compare the ONNX graphs with TensorFlow on the CPU. The DirectML cost,
the first runs, the operator placement, and the quality metrics use the shared scripts of
[`scripts/ai_models/s3od/`](../s3od/README.md) with `--model universal_matting` (kind
`universal_matting`). The MVP pipeline (`scripts/poc/ai_mask_mvp/mask_pipeline.py`) accepts the
graph with `--subject-onnx` (graph output `rgba`). The report is
[Subject model alternatives to BiRefNet lite](../../../docs/exploration/subject_alternatives.md)
(section 12).

The graph, the source files, ONNX files, photos, alphas, and logs are not committed. Keep them
under `build/tmp/explore/`.

## Sources

| Item | Source | Revision | License |
| --- | --- | --- | --- |
| `tf_graph.pb` (204,740,209 bytes, SHA-256 `afcad611b6707f8a7e815f5702216af69fa77a72ef470d0b2c25a6afc340ea1f`) | ModelScope `iic/cv_unet_universal-matting` | master `2e7fe175a9a4a8ce234324b7cccd0a756c75c229` (tag v1.0.0 `de441e1b...`; last commit of the file `482493ba...`), downloaded 2026-10-21 | Apache License 2.0 (model card front matter) |
| Model card `README.md` (4,334 bytes, SHA-256 `d21254bf...`), `configuration.json` (`e7f58ee5...`) | same | same | — |
| Pipeline code (`modelscope/pipelines/cv/image_matting_pipeline.py` `b0333642...`, `modelscope/preprocessors/image.py` `ba44792d...`) | GitHub `modelscope/modelscope` | `958c3f7da68b700cc671134e267c356a2444398b` | Apache-2.0 (`LICENSE` `cfc7749b...`) |
| TensorFlow kernels (`resize_area_op.cc`, `resize_bilinear_op.cc`, `crop_and_resize_op.cc`, `image_resizer_state.h`) | GitHub `tensorflow/tensorflow` | v2.21.0 | Apache-2.0 (read only, for the rewrites) |

The model card states the training data: public data sets ("COCO etc.") and images from internet
searches, annotated, pasted on other backgrounds, about 200,000 images. It states that the model
gives the expected result on images below 2000 x 2000 and that larger images can have blurred
edges.

## Files

| File | Purpose |
| --- | --- |
| `universal_matting_graphs.py` | Photo reading of the source pipeline, TensorFlow sessions, the graph rules of `tf_graph.pb`, the graph rewrites before `tf2onnx`, the conversion. Also used by `subject_runtime.py` (kind `universal_matting`) and `mask_pipeline.py`. |
| `export_onnx.py` | Rewrite check `RewrittenGraphMatchesTensorflowGraph`, float32 graph (`tf2onnx` without grappler, float64 scalar divisions) and float16 graph (float16 only in the U-Nets), each with `export_report.json`. |
| `compare_onnx_outputs.py` | `OnnxAlphaMatchesTensorflowWithinTolerance`: ONNX Runtime CPU against TensorFlow CPU on the quality photos; writes the reference and ONNX alphas. |
| `measure_tensorflow_spread.py` | `TensorflowSpreadAgainstOnnx`: TensorFlow with six CPU settings (oneDNN, threads, grappler) against each other and against the ONNX alpha. |
| `requirements.txt` | Packages of the export and comparison environment (Python 3.12). |

## Graph

| File | Inputs | Outputs |
| --- | --- | --- |
| `universal_matting.onnx` | `input_image` [H, W, 3] float32, the photo in RGB 0 to 255, any size | `rgba` [H, W, 4] float32, values 0 to 255; alpha = channel 3 / 255 |

The graph does all resizing itself. The scripts feed the photo at its size (the quality photos
have a long edge of 4096 px); no input size is chosen. Rules of `tf_graph.pb`
(`universal_matting_graphs.GRAPH_RULES`):

- A person detector U-Net at 192 x 192 (`detect/`). If more than 50 rows and columns have person
  pixels, the human branch runs: a crop of the person box at 1920 x 1920 (AvgPool 5 x 5) or 640 x
  640, a coarse U-Net at 224 (`cond_1/coarse/`), and a refine U-Net at 480 (`cond_1/refine_human/`).
- The human result is used if its alpha covers more than 10 % of the photo. Else the object
  branch runs: a U-Net at 224 (`cond_3/object_stage0/`), a crop of its box with a margin of 1/4 at
  640, a U-Net at 224 (`cond_3/object_stage1/`), and a refine U-Net at 640
  (`cond_3/refine_object/`).
- The alpha is resized to the box in photo pixels and padded to the photo.

The six U-Nets use instance normalization; the refine U-Nets use LeakyRelu and ConvTranspose.

## Conversion rules

- **Rewrites before `tf2onnx` 1.17** (`rewrite_for_onnx`): the constant `cond_2` becomes a Cast;
  `ResizeArea` (converted by `tf2onnx` to a nearest-neighbor Resize) becomes two weight matrices
  and MatMul; `ResizeBilinear` and `CropAndResize` become Gather with the float32 expressions of
  the TensorFlow kernels (ONNX Runtime Resize gives other values: 1e-4 to 7e-4, and 0.996 in the
  last row of a full-box crop). With grappler off in both sessions, the rewritten graph gives the
  same RGBA as the original graph on the 14 photos (difference 0.0).
- **No grappler in the conversion** (`convert_to_onnx` calls `process_tf_graph` and
  `optimize_graph`). The grappler constant folding of `tf2onnx.from_graph_def` changes the order
  of `(x * y) / c` and moves a crop position by one ulp; on `DSC00674` that changed the alpha
  (IoU 0.934).
- **Float64 scalar divisions** (`float64_scalar_divisions`): each Div outside the U-Nets with one
  float32 output element is Cast to float64, Div, Cast to float32 (42 nodes). The CPU output is
  bit-identical. DirectML float32 Div is not correctly rounded: 4095 / 1919 gives 2.13392377
  (one ulp low), so the last row of the 1920 crop reads the photo instead of the extrapolation
  value 0 of the TensorFlow kernel, and the alpha of `DSC00674` changed (IoU 0.936 against
  TensorFlow). DirectML has no float64 Div; ONNX Runtime runs these one-element nodes on the CPU.
  `--scalar-divisions float32` makes the first graph again.
- **Float16** (`convert_regions_to_float16`): only nodes of the six U-Nets, and only when the
  output and each input of the node have a largest absolute value of at most 4,094 on the 14
  photos. The instance norms reach 5.2e8 (squared deviations), so most of the coarse U-Net nodes
  stay float32 (888 of 1,954 region nodes are float16). The ONNX Runtime float16 converter cannot
  convert this graph (If branch outputs).
- `tf2onnx` merges duplicate nodes in set order: two exports can differ by some Reshape and Sub
  nodes (3,272 to 3,298 nodes before the float64 division step) with bitwise equal outputs. The
  records name each graph by its SHA-256.

## Commands

```bash
# Sources (Git Bash, Windows). Download the ModelScope files with the URL of revisions.json.
E=build/tmp/explore; W=$E/weights/universal_matting; U=$E/universal_matting
mkdir -p $W
for f in tf_graph.pb README.md configuration.json; do
  curl -sSL -o $W/$f \
    "https://modelscope.cn/api/v1/models/iic/cv_unet_universal-matting/repo?Revision=master&FilePath=$f"
done
sha256sum $W/tf_graph.pb   # afcad611b6707f8a...
# revisions.json: {"modelscope": {"model": ..., "commit": ...}, "files": {...}, "pipeline_source": {...}}

py -3.12 -m venv $E/venv_universal_matting
$E/venv_universal_matting/Scripts/python -m pip install -r scripts/ai_models/universal_matting/requirements.txt
PY="$E/venv_universal_matting/Scripts/python -I"; X=scripts/ai_models/universal_matting
P=$E/e4/input_4096/photos; Q=build/tmp/sm1/quality/prompts.json; R=results/subject_alternatives
$PY $X/export_onnx.py --graph $W/tf_graph.pb --revisions $W/revisions.json --out-root $U/onnx \
  --calibration-photos $P --calibration-prompts $Q
cp $U/onnx/float32/export_report.json $R/universal_matting_float32_export.json
cp $U/onnx/float16/export_report.json $R/universal_matting_float16_export.json
$PY $X/compare_onnx_outputs.py --graph $W/tf_graph.pb \
  --graph-onnx float32=$U/onnx/float32/universal_matting.onnx \
  --graph-onnx float16=$U/onnx/float16/universal_matting.onnx \
  --required float32 --photos $P --prompts $Q --out $U/compare/onnx_comparison.json
cp $U/compare/onnx_comparison.json $R/universal_matting_comparison_cpu.json
$PY $X/measure_tensorflow_spread.py --graph $W/tf_graph.pb --photos $P --prompts $Q \
  --onnx-alpha $U/compare/float32/alpha --work-dir $U/tensorflow_spread \
  --out $R/universal_matting_tensorflow_spread.json
```

```bash
# DirectML (MVP environment, the shared scripts of scripts/ai_models/s3od). One model job at a time.
PY="$E/venv_ai_mask_mvp/Scripts/python -I"; S=scripts/ai_models/s3od
for p in float32 float16; do
  $PY $S/profile_operator_placement.py --model universal_matting \
    --graph $U/onnx/$p/universal_matting.onnx --photo $P/DSC_1535.jpg \
    --profile-dir $U/placement/profiles_$p --out $R/universal_matting_${p}_placement.json
  $PY $S/measure_first_runs.py --model universal_matting --graph $U/onnx/$p/universal_matting.onnx \
    --photo $P/DSC_1535.jpg --count 9 --out $R/universal_matting_${p}_first_runs.json
done
```

```powershell
# Cost with the peak GPU memory (PowerShell, from the repository root).
$e = "build\tmp\explore"; $py = "$e\venv_ai_mask_mvp\Scripts\python.exe"; $r = "results\subject_alternatives"
foreach ($p in @("float32", "float16")) {
  $l = "universal_matting_$p"; $g = "$e\universal_matting\onnx\$p\universal_matting.onnx"
  .\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
    -Arguments @('-I', 'scripts\ai_models\s3od\measure_directml_cost.py', '--model', 'universal_matting',
                 '--graph', $g, '--photos', "$e\e4\input_4096\photos",
                 '--prompts', 'build\tmp\sm1\quality\prompts.json',
                 '--alpha-dir', "$e\universal_matting\directml\$l\alpha", '--out', "$r\${l}_cost.json") `
    -Out "$r\${l}_cost_gpu_memory.json" -Log "$e\universal_matting\directml\${l}_cost.log"
}
# The MVP pipeline with the universal matting float32 graph as the Subject graph.
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
  -Arguments @('-I', 'scripts\poc\ai_mask_mvp\measure_pipeline_cost.py', '--provider', 'directml',
               '--photos', "$e\e4\input_4096\photos",
               '--strokes-report', "$e\e3\runtime\tiny_directml.json",
               '--subject-onnx', "$e\universal_matting\onnx\float32\universal_matting.onnx",
               '--out', "$r\mvp_pipeline_universal_matting_float32_cost.json") `
  -Out "$r\mvp_pipeline_universal_matting_float32_cost_gpu_memory.json" `
  -Log "$e\universal_matting\directml\mvp_pipeline_universal_matting_float32.log"
```

```bash
# Quality metrics and the local sheets (E2 BiRefNet environment, PyTorch).
PY="$E/venv_birefnet/Scripts/python -I"; S=scripts/ai_models/s3od
REGIONS=(--region "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --region "P1024439.jpg:hair:0.66,0.27,0.84,0.5"
         --region "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --region "DSC00326.jpg:hair:0.44,0.6,0.6,0.76"
         --region "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --region "DSC00936.jpg:person:0.47,0.63,0.68,0.93")
$PY $S/probe_subject_masks.py --photos $P --prompts $Q \
  --subject-masks $E/e1/compare/base_plus/reference_masks \
  --alpha "birefnet_lite_float16_cpu=birefnet:build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha" \
  --alpha "s3od_float32_cpu=s3od:$E/s3od/compare_cpu/float32/alpha" \
  --alpha "universal_matting_tensorflow=universal_matting:$U/compare/reference_alpha" \
  --alpha "universal_matting_float32_cpu=universal_matting:$U/compare/float32/alpha" \
  --alpha "universal_matting_float16_cpu=universal_matting:$U/compare/float16/alpha" \
  --alpha "universal_matting_float32_directml=universal_matting:$U/directml/universal_matting_float32/alpha" \
  --alpha "universal_matting_float16_directml=universal_matting:$U/directml/universal_matting_float16/alpha" \
  --iou-reference birefnet_lite_float16_cpu --iou-reference s3od_float32_cpu \
  --iou-reference universal_matting_tensorflow --out $R/quality_universal_matting.json "${REGIONS[@]}"
$PY $S/make_subject_sheets.py --photos $P --prompts $Q \
  --alpha "BiRefNet lite float16 (CPU)=birefnet:build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha" \
  --alpha "S3OD 8 chunk MHA float16 (DirectML)=s3od:$E/s3od/directml/s3od_mha_chunk8_float16/alpha" \
  --alpha "Universal matting float32 (DirectML)=universal_matting:$U/directml/universal_matting_float32/alpha" \
  --crop "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --crop "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
  --crop "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --crop "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
  --crop "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --crop "DSC00936.jpg:person:0.47,0.63,0.68,0.93" \
  --out-dir docs/exploration/images/subject_alternatives_universal_matting
```

The records `*_float32_divisions.json` (quality, float32 cost, float32 first runs) are from the
first float32 graph (`--scalar-divisions float32`, export to `$U/onnx_float32_divisions`); they
show the DirectML division defect of the conversion rules above.
