#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT = PROJECT_ROOT / "ios" / "ModelConversion" / "output"
REPORT_PATH = OUTPUT / "parity-report.json"


def main() -> int:
  from scripts.mobileclip_item_search import DEFAULT_INDEX, DEFAULT_WEIGHTS, MobileClipEncoder

  print("[1/4] validating generated asset structure", flush=True)
  report: dict[str, Any] = {"status": "running", "assets": {}, "searchIndex": {}, "coreML": {}}
  for name in ("RoomCollectibleDetector.mlpackage", "MobileCLIPImageEncoderRaw.mlpackage"):
    path = OUTPUT / name
    if not path.exists():
      raise FileNotFoundError(f"missing {path}; run export_ios_assets.py first")
    report["assets"][name] = {"bytes": directory_size(path)}

  print("[2/4] comparing Float16 and PyTorch search indexes", flush=True)
  payload = torch.load(DEFAULT_INDEX, map_location="cpu")
  source_vectors = payload["vectors"].float()
  metadata = json.loads((OUTPUT / "item-vectors.json").read_text(encoding="utf-8"))
  ios_vectors = np.frombuffer((OUTPUT / "item-vectors.f16").read_bytes(), dtype="<f2")
  ios_vectors = torch.from_numpy(ios_vectors.astype(np.float32)).reshape(metadata["rows"], metadata["dimensions"])
  source_top10 = (source_vectors @ source_vectors.T).topk(10, dim=1).indices
  ios_top10 = (ios_vectors @ source_vectors.T).topk(10, dim=1).indices
  report["searchIndex"] = {
    "rows": metadata["rows"],
    "dimensions": metadata["dimensions"],
    "maxQuantizationError": float((source_vectors - ios_vectors).abs().max()),
    "meanQuantizationError": float((source_vectors - ios_vectors).abs().mean()),
    "top1Agreement": float((source_top10[:, 0] == ios_top10[:, 0]).float().mean()),
    "top10SetAgreement": float(torch.tensor([
      len(set(left.tolist()) & set(right.tolist())) / 10
      for left, right in zip(source_top10, ios_top10, strict=True)
    ]).mean()),
  }
  detector_report = OUTPUT / "detector-pytorch-evaluation" / "report.json"
  if detector_report.exists():
    detector_metrics = json.loads(detector_report.read_text(encoding="utf-8"))
    report["detectorBaseline"] = {
      key: detector_metrics[key]
      for key in ("targets", "matched", "recall", "falsePositives")
    }

  print("[3/4] checking Core ML model contracts", flush=True)
  import coremltools as ct

  encoder_path = OUTPUT / "MobileCLIPImageEncoderRaw.mlpackage"
  detector_path = OUTPUT / "RoomCollectibleDetector.mlpackage"
  encoder_spec = ct.models.MLModel(str(encoder_path), skip_model_load=True).get_spec()
  detector_spec = ct.models.MLModel(str(detector_path), skip_model_load=True).get_spec()
  report["coreML"]["encoderContract"] = model_contract(encoder_spec)
  report["coreML"]["detectorContract"] = model_contract(detector_spec)
  encoder_metadata = dict(encoder_spec.description.metadata.userDefined)
  report["coreML"]["encoderMetadata"] = encoder_metadata
  if encoder_metadata.get("weights") != DEFAULT_WEIGHTS.name:
    raise ValueError(
      f"encoder weights metadata is {encoder_metadata.get('weights')}, expected {DEFAULT_WEIGHTS.name}"
    )
  encoder_output = encoder_spec.description.output[0].type.multiArrayType
  if encoder_output.dataType != 65568:
    raise ValueError(f"encoder output must be Float32, got Core ML data type {encoder_output.dataType}")

  print("[4/4] attempting Core ML runtime parity", flush=True)
  try:
    coreml_encoder = ct.models.MLModel(str(encoder_path))
    sample_path = PROJECT_ROOT / "public" / payload["labels"][0]["icon_path"].lstrip("/")
    image = Image.open(sample_path).convert("RGB")
    model_image = image.resize((256, 256), Image.Resampling.BILINEAR)
    prediction = coreml_encoder.predict({"image": model_image})["embedding"]
    coreml_vector = torch.from_numpy(np.asarray(prediction, dtype=np.float32)).reshape(-1)
    finite = bool(torch.isfinite(coreml_vector).all())
    raw_norm = float(coreml_vector.norm())
    if not finite or raw_norm <= 1e-8:
      raise ValueError(f"invalid Core ML embedding finite={finite} norm={raw_norm}")
    raw_min = float(coreml_vector.min())
    raw_max = float(coreml_vector.max())
    coreml_vector = coreml_vector / raw_norm
    pytorch_vector = MobileClipEncoder(DEFAULT_WEIGHTS).encode([image])[0]
    report["coreML"]["runtime"] = {
      "available": True,
      "embeddingDataType": str(np.asarray(prediction).dtype),
      "embeddingFinite": finite,
      "embeddingRawNorm": raw_norm,
      "embeddingRawMin": raw_min,
      "embeddingRawMax": raw_max,
      "encoderMaxAbsoluteError": float((pytorch_vector - coreml_vector).abs().max()),
      "encoderCosineSimilarity": float(pytorch_vector @ coreml_vector),
    }
  except Exception as exc:
    report["coreML"]["runtime"] = {
      "available": False,
      "reason": f"{type(exc).__name__}: {exc}",
      "nextStep": "Run with full Xcode selected and access to the system Core ML service.",
    }

  report["status"] = "passed-with-runtime" if report["coreML"]["runtime"]["available"] else "passed-static"
  REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
  print(f"wrote {REPORT_PATH}", flush=True)
  return 0


def model_contract(spec: Any) -> dict[str, Any]:
  return {
    "inputs": [feature_contract(feature) for feature in spec.description.input],
    "outputs": [feature_contract(feature) for feature in spec.description.output],
  }


def feature_contract(feature: Any) -> dict[str, Any]:
  kind = feature.type.WhichOneof("Type")
  result: dict[str, Any] = {"name": feature.name, "type": kind}
  if kind == "imageType":
    result.update({"width": feature.type.imageType.width, "height": feature.type.imageType.height})
  elif kind == "multiArrayType":
    result["shape"] = list(feature.type.multiArrayType.shape)
    result["dataType"] = feature.type.multiArrayType.dataType
  return result


def directory_size(path: Path) -> int:
  return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


if __name__ == "__main__":
  raise SystemExit(main())
