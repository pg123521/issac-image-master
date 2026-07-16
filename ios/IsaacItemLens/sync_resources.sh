#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
ROOT=${SCRIPT_DIR:h:h}
RESOURCES="$SCRIPT_DIR/App/Resources"

echo "[1/4] Syncing encyclopedia"
cp "$ROOT/data/objects.en.json" "$RESOURCES/objects.json"

echo "[2/4] Syncing Float16 search index"
cp "$ROOT/ios/ModelConversion/output/item-vectors.f16" "$RESOURCES/item-vectors.f16"
cp "$ROOT/ios/ModelConversion/output/item-vectors.json" "$RESOURCES/item-vectors.json"

echo "[3/4] Syncing item icons"
mkdir -p "$RESOURCES/items" "$RESOURCES/objects"
cp -R "$ROOT/public/items/icons" "$RESOURCES/items/"

echo "[4/4] Syncing object icons"
cp -R "$ROOT/public/objects/icons" "$RESOURCES/objects/"
echo "Resource sync complete"
