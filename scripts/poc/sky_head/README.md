# Fine sky head on SAM 2.1 tiny features (issue #10)

Issue: [#10](https://github.com/zidage/alcedo-ai-mask-lab/issues/10). Decisions: README U17 (own
sky head with a fine alpha output, SAM 2.1 stays the interactive model) and U18 (MaSS13K as
training data, with a notice to the user that the training data may not permit commercial use).

The head reads the outputs of the frozen SAM 2.1 tiny encoder (`image_embed`,
`high_res_feats_0`, `high_res_feats_1`) and the normalized encoder input, and gives a
1,024 x 1,024 sky alpha. The product runs no second image encoder. The edge comes from the model
output (U3).

| File | Purpose |
| --- | --- |
| `sky_head_data.py` | Prepared data layout, sample loading, augmentation, the 1,024 x 1,024 stretch of the product. |
| `prepare_mass13k.py` | MaSS13K archives -> prepared data (2,048 px photos, sky coverage, weight); `--inspect` describes the archives. Reads the split volumes `train.zip.001` to `.006` as one file. |
| `sky_head_model.py` | The head (`SkyAlphaHead`, 1.4 M parameters) and the frozen encoder (`sam2_graphs.PointEncoder`). |
| `sky_head_losses.py` | Alpha L1, gradient, Laplacian pyramid, and coarse BCE terms with the weight map. |
| `sky_head_metrics.py` | IoU, MAE, edge-band SAD, MSE, and gradient error, boundary F1. |
| `train_sky_head.py` | Training; `--pilot-steps` (time and memory) and `--overfit-images` (the head learns a few photos). |
| `evaluate_sky_head.py` | The measurements of a checkpoint on a split (encoder in float32). |
| `export_sky_head_onnx.py` | `sky_head.onnx` and the ONNX Runtime against PyTorch comparison (tolerance 1e-4). |
| `runpod_sky_head.sh` | All stages on a RunPod GPU pod. |
| `requirements_runpod.txt` | The packages that the RunPod PyTorch image does not have. |

## Data

MaSS13K (CVPR 2025, [repository](https://github.com/xiechenxi99/MaSS13K)): 4,032 x 3,024
smartphone photos; `masks.zip` has the labels of all splits (`annotations/<split>/<stem>.png`,
mode L, class index; 5 = sky): train 11,348, val 500, test 1,500 (plus 2 Jupyter checkpoint
copies in test, which the script skips). The photos are in `train.zip.001` to `.006` (a byte
split of one zip), `val.zip`, and `test.zip`. About 58 GB in total.

`masks.zip` SHA-256: `7629e90d50598bea49c8139befbbf48e4f839dad498e91815814e7a3d21d1d1d`.

## RunPod

A pod with the RunPod PyTorch image (CUDA, PyTorch 2.8 or later), a GPU with at least 16 GB, and
about 120 GB of disk for `/workspace`. In a tmux session:

```text
curl -sL https://raw.githubusercontent.com/zidage/alcedo-ai-mask-lab/feature/sky-mask-exploration/scripts/poc/sky_head/runpod_sky_head.sh -o run.sh
bash run.sh setup download inspect prepare pilot overfit   # data and checks
bash run.sh train evaluate export                          # the training run
```

Settings are environment variables (`RUN_NAME`, `EPOCHS`, `BATCH_SIZE`, `LR`, `WORKERS`; see the
script). The output is in `/workspace/sky_head/runs/<RUN_NAME>/`: `pilot/pilot.json`,
`overfit/overfit.json`, `train/` (`run.json`, `metrics.jsonl`, `last.pt`, `best.pt`, the
measurements), and `export/` (`sky_head.onnx`, `sky_head.json`).
