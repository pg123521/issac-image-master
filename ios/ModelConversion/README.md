# iOS model conversion POC

This directory converts the current offline Python inference assets into files
that can be bundled in a native iOS app.

## Export

Run from the repository root:

```bash
.venv-train/bin/pip install -r requirements-ios.txt
.venv-train/bin/python ios/ModelConversion/export_ios_assets.py all
.venv-train/bin/python ios/ModelConversion/validate_ios_assets.py
```

The command prints progress and writes:

- `RoomCollectibleDetector.mlpackage`: 1024x1024 room collectible detector.
- `MobileCLIPImageEncoderRaw.mlpackage`: 256x256 RGB to raw 512-D embedding.
- `item-vectors.f16`: row-major little-endian Float16 reference vectors.
- `item-vectors.json`: vector dimensions and object metadata.

Pass `--overwrite` to replace previously generated output.

## iOS preprocessing contract

- The detector receives the same full-frame and overlapping square tiles used
  by `scripts/room_detector_service.py`.
- The encoder receives a square RGB crop resized to 256x256. Core ML applies
  the `1/255` input scale; no mean/std normalization is required.
- The app L2-normalizes the encoder output in Float32. Search then uses a dot
  product because both query and reference vectors are normalized.

Generated `.mlpackage` directories are POC artifacts. Source weights remain the
canonical inputs so the export can be reproduced after model updates.

The validation report is written to `output/parity-report.json`. Static model
and index validation works with Command Line Tools. Full Core ML prediction
validation requires a complete Xcode installation because it supplies Apple's
model compiler and runtime tooling.
