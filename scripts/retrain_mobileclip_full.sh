#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
ROOT=${SCRIPT_DIR:h}
PYTHON="$ROOT/.venv-train/bin/python"
OBJECTS="$ROOT/data/objects.en.json"
BASELINE="$ROOT/models/mobileclip-object-icon-index-v2.pt"
WEIGHTS="$ROOT/models/mobileclip-partial-v2.pt"
INDEX="$ROOT/models/mobileclip-object-partial-index-v2.pt"

cd "$ROOT"

echo "[1/3] Building the 1020-object baseline index"
"$PYTHON" scripts/mobileclip_item_search.py build-index \
  --objects-json "$OBJECTS" \
  --output "$BASELINE" \
  --batch-size 96

echo "[2/3] Fine-tuning MobileCLIP on all 1020 objects"
"$PYTHON" scripts/train_mobileclip_partial.py train \
  --objects-json "$OBJECTS" \
  --baseline-index "$BASELINE" \
  --output "$WEIGHTS" \
  --phase1-epochs 6 \
  --phase2-epochs 4 \
  --batch-size 48

echo "[3/3] Building the fine-tuned 1020-object index"
"$PYTHON" scripts/mobileclip_item_search.py build-index \
  --objects-json "$OBJECTS" \
  --weights "$WEIGHTS" \
  --output "$INDEX" \
  --batch-size 96

echo "Full retraining pipeline complete"
