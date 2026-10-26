#!/usr/bin/env bash
# Fine sky head on a RunPod GPU pod (issue #10).
#
# Usage:  bash runpod_sky_head.sh <stage> [<stage> ...]      (run it in tmux: the stages are long)
# Stages: setup download inspect prepare pilot overfit train evaluate export
#         all = every stage in this order
#
# Every stage can run again: finished downloads, the prepared data, and the training checkpoint
# are kept, and `train` continues from last.pt. Settings come from environment variables (the
# defaults are below), for example:  RUN_NAME=run2 EPOCHS=30 bash runpod_sky_head.sh train
#
# Data: MaSS13K (README U18) from a Google Drive copy of the user (DRIVE_FILE_IDS). A Drive file
# that many clients downloaded recently returns "Too many users have viewed or downloaded this
# file recently" for up to 24 hours; the stage then stops with that error and can run again later.
set -euo pipefail

WORK=${WORK:-/workspace/sky_head}
REPO_URL=${REPO_URL:-https://github.com/zidage/alcedo-ai-mask-lab.git}
REPO_BRANCH=${REPO_BRANCH:-feature/sky-mask-exploration}
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_CHECKPOINT_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
SAM2_CHECKPOINT_SHA256=7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
RUN_NAME=${RUN_NAME:-run1}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-3}
WORKERS=${WORKERS:-16}
PREPARE_WORKERS=${PREPARE_WORKERS:-32}
PILOT_STEPS=${PILOT_STEPS:-60}
OVERFIT_IMAGES=${OVERFIT_IMAGES:-16}
OVERFIT_STEPS=${OVERFIT_STEPS:-300}

# Google Drive copy of MaSS13K (folder 1mQ0RsmfumfqR8V83eRYQLzaZ53x_YkdX).
declare -A DRIVE_FILE_IDS=(
  [masks.zip]=1sfBh23Ntq0paRyut6x0GG2ia1WMOUVba
  [val.zip]=1zyUZ_-MJm9RLtbGGhQ8VePosHPby_0vV
  [test.zip]=1krhp1KGjThvtpVlUWQPYlOBt4pcGkB8b
  [train.zip.001]=1GPe_Z8ZuhJDBcQruDrrK6OZEnd_tmpis
  [train.zip.002]=1LSulTnVtJ-oczbpTIQeyZaF8frR8Ka5n
  [train.zip.003]=1TXiXVhcd8rDi5zSYrWap0So2bOs2RofX
  [train.zip.004]=1-fYYvEw4993soDhh_re0yZkSl9tIfvuK
  [train.zip.005]=1Wl7QTHufgdpf3Td_udj9gLmMV-08gpg2
  [train.zip.006]=1Ep9BVsF39SKFnpVYe-ZS__E7MHXykCXF
)
DOWNLOAD_ORDER=(masks.zip val.zip test.zip train.zip.001 train.zip.002 train.zip.003
                train.zip.004 train.zip.005 train.zip.006)

REPO=$WORK/alcedo-ai-mask-lab
S=$REPO/scripts/poc/sky_head
SAM2=$WORK/sam2
CHECKPOINT=$WORK/sam2.1_hiera_tiny.pt
DOWNLOAD=$WORK/download
PREPARED=$WORK/prepared
RUN_DIR=$WORK/runs/$RUN_NAME
PY=$WORK/venv/bin/python
PHOTO_ARCHIVES="$DOWNLOAD/val.zip $DOWNLOAD/test.zip $DOWNLOAD/train.zip.001"
LABEL_ARCHIVES="$DOWNLOAD/masks.zip"
SAM2_ARGS="--sam2-repo $SAM2 --sam2-checkpoint $CHECKPOINT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

stage_setup() {
  mkdir -p "$WORK"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  df -h "$WORK"
  if [ -d "$REPO" ]; then
    git -C "$REPO" fetch -q origin "$REPO_BRANCH"
    git -C "$REPO" checkout -q -B "$REPO_BRANCH" "origin/$REPO_BRANCH"
  else
    git clone -q --branch "$REPO_BRANCH" "$REPO_URL" "$REPO"
  fi
  git -C "$REPO" log -1 --oneline
  [ -d "$SAM2" ] || git clone -q https://github.com/facebookresearch/sam2.git "$SAM2"
  git -C "$SAM2" checkout -q "$SAM2_COMMIT"
  # The venv uses the PyTorch of the RunPod image (system site packages).
  [ -x "$PY" ] || python3 -m venv --system-site-packages "$WORK/venv"
  "$PY" -m pip install -q -r "$S/requirements_runpod.txt"
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  [ -f "$CHECKPOINT" ] || wget -q -O "$CHECKPOINT" "$SAM2_CHECKPOINT_URL"
  echo "$SAM2_CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum -c -
}

stage_download() {
  # Tries every file; a file that fails (for example the Drive quota) does not stop the others.
  # The stage fails at the end when a file is missing.
  mkdir -p "$DOWNLOAD"
  local failed=()
  for name in "${DOWNLOAD_ORDER[@]}"; do
    if [ -f "$DOWNLOAD/$name.done" ]; then
      log "download: $name already done"
      continue
    fi
    log "download: $name"
    if "$WORK/venv/bin/gdown" -c "${DRIVE_FILE_IDS[$name]}" -O "$DOWNLOAD/$name"; then
      sha256sum "$DOWNLOAD/$name" | tee "$DOWNLOAD/$name.done"
    else
      failed+=("$name")
    fi
  done
  ls -la "$DOWNLOAD"
  if [ ${#failed[@]} -gt 0 ]; then
    echo "download failed: ${failed[*]} (run the download stage again later)" >&2
    return 1
  fi
  cat "$DOWNLOAD"/*.done > "$DOWNLOAD/sha256.txt"
}

stage_inspect() {
  # shellcheck disable=SC2086
  "$PY" "$S/prepare_mass13k.py" --photo-archive $PHOTO_ARCHIVES \
    --label-archive $LABEL_ARCHIVES --inspect --inspect-json "$WORK/inspect.json" > /dev/null
  "$PY" - "$WORK/inspect.json" <<'PYTHON'
import json, sys
report = json.load(open(sys.argv[1]))
for archive in report["archives"]:
    print(archive["archive"], archive["bytes"], archive["members"], archive["suffixes"])
    for sample in archive["samples"][:2]:
        print("   ", sample["kind"], sample["member"], sample["mode"], sample["size"])
print("pairs:", report["pairs"])
print("skipped checkpoint copies:", len(report["skipped_checkpoint_copies"]))
PYTHON
}

stage_prepare() {
  if [ -f "$PREPARED/train/manifest.json" ] && [ -f "$PREPARED/val/manifest.json" ]; then
    log "prepare: $PREPARED already has the manifests"
    return
  fi
  # shellcheck disable=SC2086
  "$PY" "$S/prepare_mass13k.py" --photo-archive $PHOTO_ARCHIVES \
    --label-archive $LABEL_ARCHIVES --out-dir "$PREPARED" --workers "$PREPARE_WORKERS"
  du -sh "$PREPARED"
}

train_args() {
  echo "--data-dir $PREPARED $SAM2_ARGS --batch-size $BATCH_SIZE --lr $LR --workers $WORKERS"
}

stage_pilot() {
  # shellcheck disable=SC2046
  "$PY" "$S/train_sky_head.py" $(train_args) --epochs "$EPOCHS" \
    --pilot-steps "$PILOT_STEPS" --out-dir "$RUN_DIR/pilot"
}

stage_overfit() {
  # shellcheck disable=SC2046
  "$PY" "$S/train_sky_head.py" $(train_args) --overfit-images "$OVERFIT_IMAGES" \
    --overfit-steps "$OVERFIT_STEPS" --out-dir "$RUN_DIR/overfit"
}

stage_train() {
  local resume=""
  [ -f "$RUN_DIR/train/last.pt" ] && resume="--resume"
  # shellcheck disable=SC2046
  "$PY" "$S/train_sky_head.py" $(train_args) --epochs "$EPOCHS" \
    --out-dir "$RUN_DIR/train" $resume
}

stage_evaluate() {
  for split in val test; do
    "$PY" "$S/evaluate_sky_head.py" --checkpoint "$RUN_DIR/train/best.pt" \
      --data-dir "$PREPARED" --split "$split" $SAM2_ARGS --workers "$WORKERS" \
      --out "$RUN_DIR/train/best_${split}_measurements.json"
  done
}

stage_export() {
  "$PY" "$S/export_sky_head_onnx.py" --checkpoint "$RUN_DIR/train/best.pt" \
    --data-dir "$PREPARED" --split val $SAM2_ARGS --out-dir "$RUN_DIR/export"
}

[ $# -gt 0 ] || { sed -n '2,15p' "$0"; exit 1; }
STAGES=("$@")
[ "${STAGES[0]}" = "all" ] && STAGES=(setup download inspect prepare pilot overfit train evaluate export)
for stage in "${STAGES[@]}"; do
  log "=== $stage ==="
  "stage_$stage"
done
log "=== done: ${STAGES[*]} ==="
