# IS-Net general-use export and comparison tools (issue #6)

Issue [#6](https://github.com/zidage/alcedo-ai-mask-lab/issues/6) measures Subject models as
alternatives to BiRefNet lite (F1 Subject, F2 Background). These scripts export the IS-Net
general-use model (`isnet-general-use.pth` of `xuebinqin/DIS`) to ONNX and compare the export with
PyTorch on the CPU. The DirectML cost, the first runs, the operator placement, and the quality
metrics use the shared scripts of [`scripts/ai_models/s3od/`](../s3od/README.md) with
`--model isnet` (kind `isnet`). The MVP pipeline (`scripts/poc/ai_mask_mvp/mask_pipeline.py`)
accepts the IS-Net graph with `--subject-onnx`. The report is
[Subject model alternatives to BiRefNet lite](../../../docs/exploration/subject_alternatives.md)
(section 11).

Weights, source checkouts, ONNX files, photos, alphas, and logs are not committed. Keep them under
`build/tmp/explore/`.

## Sources

| Item | Source | Revision | License |
| --- | --- | --- | --- |
| Weights `isnet-general-use.pth` (176,579,397 bytes, SHA-256 `9e1aafea58f0b55d0c35077e0ceade6ba1ba2bce372fd4f8f77215391f3fac13`) | Google Drive file `1XHIzgTzY5BQHw140EDIgwIb53K659ENH` (linked by the DIS README) | Downloaded 2026-10-21; the server file name is `isnet-general-use.pth` | **No license statement** (see below) |
| Model code (`IS-Net/models/isnet.py`, SHA-256 `83505426...`; `IS-Net/Inference.py`, `e4d4c3b7...`) | GitHub `xuebinqin/DIS` | `b6764e20381f6f42a70f83fa3324181529ed1403` | Apache-2.0 (`LICENSE.md` 11,357 bytes, SHA-256 `c71d239d...`) |
| Training data | Not stated for the general-use weights ("NOT DIS V2.0"). The academic `isnet.pth` uses DIS5K | — | DIS5K: `DIS5K-Dataset-Terms-of-Use.pdf` (SHA-256 `d509ad22...`), non-commercial |

- The DIS README states "Our code and evaluation metric use Apache License 2.0". It has no
  statement for the weights. [DIS#150](https://github.com/xuebinqin/DIS/issues/150) (2026-08-20,
  open) asks for the commercial use of `isnet-general-use.pth`; on 2026-10-21 it has one comment
  from a second requester and no answer from the author.
- The README has two Google Drive links that are not consistent. README line 145 links file
  `1XHIzgTz...` as `isnet.pth` and file `1klUkUnQ...` as `isnet-general-use.pth`. The server file
  names are the reverse: `1XHIzgTz...` sends `isnet-general-use.pth` (176,579,397 bytes) and
  `1klUkUnQ...` sends `isnet.pth` (176,597,693 bytes, SHA-256 `ea088974...`). README line 28 links
  `1XHIzgTz...` as `isnet-general-use.pth`. The scripts use the file that the server names
  `isnet-general-use.pth`.

`isnet-general-use.pth` is a `torch.save` `state_dict` (zip format). `torch.load(...,
weights_only=True)` reads it and `ISNetDIS().load_state_dict(strict=True)` accepts it. The model
has 44,046,790 parameters; the graph has 44,004,737 (the side outputs d2 to d6 are not in the
graph).

## Files

| File | Purpose |
| --- | --- |
| `isnet_graphs.py` | Model loading from the DIS checkout, the preprocessing and postprocessing of `Inference.py`, the export wrapper (output d1). Also used by `subject_runtime.py` (kind `isnet`) and `mask_pipeline.py`. |
| `export_onnx.py` | Float32 graph (TorchScript exporter, opset 17) and float16 graph (`convert_float_to_float16`, `keep_io_types=True`), with `export_report.json` and the activation maxima of the float16 range check. |
| `compare_onnx_outputs.py` | `OnnxD1MatchesPytorchWithinTolerance`: ONNX Runtime CPU against the PyTorch model on the quality photos; writes the reference and ONNX alphas. |
| `requirements.txt` | Packages of the export and comparison environment. |

## Graph

| File | Inputs | Outputs |
| --- | --- | --- |
| `subject_d1.onnx` | `pixel_values` [1, 3, 1024, 1024] float32 | `d1` [1, 1, 1024, 1024] float32, the sigmoid of side output d1 |

Data rules (the rules of `IS-Net/Inference.py`):

- Read the photo as RGB uint8. Stretch it to 1024 x 1024 with `F.interpolate(mode="bilinear",
  align_corners=False)` on the float32 tensor (the source calls the deprecated name
  `F.upsample`; no antialias), then `.type(torch.uint8)` (truncation). Pixel values are
  `value / 255 - 0.5` (mean 0.5, std 1.0).
- The model output is the first side output d1 after the sigmoid (`result[0][0]`).
- The source resizes d1 to the photo size (bilinear, `align_corners=False`) and scales it to
  [0, 1] with its minimum and maximum. The scripts store the alpha at 1024 x 1024 after the
  scaling (`normalize_range`) and resize it after that. On the 14 photos the two orders differ by
  at most 3.1e-2, on the photo where IS-Net finds no subject (d1 maximum 0.047); on the other 13
  photos by at most 3e-4.
- The scaling always gives a maximum of 1.0. When the model finds no subject, the scaling makes
  the weak output a mask (report section 11.6).
- Float16: the default block list of the converter, no node block list. The largest PyTorch
  module output on the 14 photos that is a graph tensor is the d1 logit (1,789). The convolutions
  of the REBNCONV blocks reach 29,832 before their BatchNorm, but the exporter folds the
  BatchNorm into the convolution weights, so that value is not a graph tensor. The DirectML
  float16 output is finite on all photos.

## Commands

```bash
# Sources and weights (Git Bash, Windows). The export environment is the E2 BiRefNet environment
# (scripts/ai_models/birefnet/README.md); requirements.txt lists its packages.
E=build/tmp/explore; W=$E/weights/isnet_general_use
git clone https://github.com/xuebinqin/DIS.git $E/src/DIS
git -C $E/src/DIS checkout b6764e20381f6f42a70f83fa3324181529ed1403
mkdir -p $W
curl -sSL -o $W/isnet-general-use.pth \
  "https://drive.usercontent.google.com/download?id=1XHIzgTzY5BQHw140EDIgwIb53K659ENH&export=download&confirm=t"
sha256sum $W/isnet-general-use.pth   # 9e1aafea58f0b55d...
# revisions.json: {"github": {"repository": "xuebinqin/DIS", "commit": "b6764e20..."},
#                  "weights": {"file": "isnet-general-use.pth", "url": "...", "sha256": "..."}}

PY="$E/venv_birefnet/Scripts/python -I"; I=$E/isnet; X=scripts/ai_models/isnet
P=$E/e4/input_4096/photos; Q=build/tmp/sm1/quality/prompts.json; R=results/subject_alternatives
$PY $X/export_onnx.py --source $E/src/DIS/IS-Net --checkpoint $W/isnet-general-use.pth \
  --revisions $W/revisions.json --out-root $I/onnx --calibration-photos $P --calibration-prompts $Q
cp $I/onnx/float32/export_report.json $R/isnet_float32_export.json
cp $I/onnx/float16/export_report.json $R/isnet_float16_export.json
$PY $X/compare_onnx_outputs.py --source $E/src/DIS/IS-Net --checkpoint $W/isnet-general-use.pth \
  --graph float32=$I/onnx/float32/subject_d1.onnx --graph float16=$I/onnx/float16/subject_d1.onnx \
  --required float32 --photos $P --prompts $Q --out $I/compare_cpu/compare.json
cp $I/compare_cpu/compare.json $R/isnet_comparison_cpu.json
```

```bash
# DirectML (MVP environment, the shared scripts of scripts/ai_models/s3od). One model job at a time.
PY="$E/venv_ai_mask_mvp/Scripts/python -I"; S=scripts/ai_models/s3od
for p in float32 float16; do
  $PY $S/profile_operator_placement.py --model isnet --graph $I/onnx/$p/subject_d1.onnx \
    --photo $P/DSC_1535.jpg --profile-dir $I/placement/profiles_isnet_$p \
    --out $R/isnet_${p}_placement.json
  $PY $S/measure_first_runs.py --model isnet --graph $I/onnx/$p/subject_d1.onnx \
    --photo $P/DSC_1535.jpg --count 9 --out $R/isnet_${p}_first_runs.json
done
```

```powershell
# Cost with the peak GPU memory (PowerShell, from the repository root).
$e = "build\tmp\explore"; $py = "$e\venv_ai_mask_mvp\Scripts\python.exe"; $r = "results\subject_alternatives"
foreach ($p in @("float32", "float16")) {
  $l = "isnet_$p"; $g = "$e\isnet\onnx\$p\subject_d1.onnx"
  .\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
    -Arguments @('-I', 'scripts\ai_models\s3od\measure_directml_cost.py', '--model', 'isnet',
                 '--graph', $g, '--photos', "$e\e4\input_4096\photos",
                 '--prompts', 'build\tmp\sm1\quality\prompts.json',
                 '--alpha-dir', "$e\isnet\directml\$l\alpha", '--out', "$r\${l}_cost.json") `
    -Out "$r\${l}_cost_gpu_memory.json" -Log "$e\isnet\directml\${l}_cost.log"
}
# The MVP pipeline with IS-Net float16 as the Subject graph.
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 -Exe $py `
  -Arguments @('-I', 'scripts\poc\ai_mask_mvp\measure_pipeline_cost.py', '--provider', 'directml',
               '--photos', "$e\e4\input_4096\photos",
               '--strokes-report', "$e\e3\runtime\tiny_directml.json",
               '--subject-onnx', "$e\isnet\onnx\float16\subject_d1.onnx",
               '--out', "$r\mvp_pipeline_isnet_float16_cost.json") `
  -Out "$r\mvp_pipeline_isnet_float16_cost_gpu_memory.json" `
  -Log "$e\isnet\directml\mvp_pipeline_isnet_float16.log"
```

```bash
# Quality metrics and the local sheets (export environment).
PY="$E/venv_birefnet/Scripts/python -I"; S=scripts/ai_models/s3od
REGIONS=(--region "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --region "P1024439.jpg:hair:0.66,0.27,0.84,0.5"
         --region "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --region "DSC00326.jpg:hair:0.44,0.6,0.6,0.76"
         --region "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --region "DSC00936.jpg:person:0.47,0.63,0.68,0.93")
$PY $S/probe_subject_masks.py --photos $P --prompts $Q \
  --subject-masks $E/e1/compare/base_plus/reference_masks \
  --alpha "birefnet_lite_float16_cpu=birefnet:build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha" \
  --alpha "s3od_float32_cpu=s3od:$E/s3od/compare_cpu/float32/alpha" \
  --alpha "isnet_pytorch=isnet:$I/compare_cpu/reference_d1" \
  --alpha "isnet_float32_cpu=isnet:$I/compare_cpu/float32/alpha" \
  --alpha "isnet_float16_cpu=isnet:$I/compare_cpu/float16/alpha" \
  --alpha "isnet_float32_directml=isnet:$I/directml/isnet_float32/alpha" \
  --alpha "isnet_float16_directml=isnet:$I/directml/isnet_float16/alpha" \
  --iou-reference birefnet_lite_float16_cpu --iou-reference s3od_float32_cpu \
  --iou-reference isnet_pytorch --out $R/quality_isnet.json "${REGIONS[@]}"
$PY $S/make_subject_sheets.py --photos $P --prompts $Q \
  --alpha "BiRefNet lite float16 (CPU)=birefnet:build/tmp/ai_mask_mvp/e2_reference/lite_fp16/alpha" \
  --alpha "S3OD 8 chunk MHA float16 (DirectML)=s3od:$E/s3od/directml/s3od_mha_chunk8_float16/alpha" \
  --alpha "IS-Net general-use float16 (DirectML)=isnet:$I/directml/isnet_float16/alpha" \
  --crop "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --crop "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
  --crop "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --crop "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
  --crop "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --crop "DSC00936.jpg:person:0.47,0.63,0.68,0.93" \
  --out-dir docs/exploration/images/subject_alternatives_isnet
```
