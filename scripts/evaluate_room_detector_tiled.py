#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from room_detector_service import DEFAULT_WEIGHTS, RoomCollectibleDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "sources" / "detection"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "detection" / "room-collectible-evaluation"


def main() -> int:
  parser = argparse.ArgumentParser(description="Evaluate the exact tiled detector path used by the app.")
  parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--image-size", type=int, default=1024)
  parser.add_argument("--confidence", type=float, default=0.18)
  parser.add_argument("--match-iou", type=float, default=0.30)
  args = parser.parse_args()

  detector = RoomCollectibleDetector(args.weights, args.image_size, args.confidence)
  metadata = json.loads((SOURCE_ROOT / "annotations.json").read_text(encoding="utf-8"))
  args.output.mkdir(parents=True, exist_ok=True)

  rows = []
  total_targets = 0
  total_matches = 0
  total_false_positives = 0
  for filename, entry in metadata.items():
    image = Image.open(SOURCE_ROOT / filename).convert("RGB")
    result = detector.detect(image)
    predictions = [to_xyxy(box) for box in result["boxes"]]
    targets = [tuple(float(value) for value in box) for box in entry.get("boxes", entry.get("excludeBoxes", []))]
    matches, false_positives = match_predictions(predictions, targets, args.match_iou)
    total_targets += len(targets)
    total_matches += len(matches)
    total_false_positives += len(false_positives)

    preview = draw_preview(image, predictions, targets, matches)
    preview.save(args.output / filename, quality=94, subsampling=0)
    row = {
      "image": filename,
      "targets": len(targets),
      "matched": len(matches),
      "falsePositives": len(false_positives),
      "predictions": len(predictions),
      "boxes": result["boxes"],
    }
    rows.append(row)
    print(
      f"{filename}: targets={len(targets)} matched={len(matches)} "
      f"false_positives={len(false_positives)} predictions={len(predictions)}",
      flush=True,
    )

  report: dict[str, Any] = {
    "weights": str(args.weights),
    "device": detector.device,
    "confidence": args.confidence,
    "matchIoU": args.match_iou,
    "targets": total_targets,
    "matched": total_matches,
    "recall": round(total_matches / max(1, total_targets), 6),
    "falsePositives": total_false_positives,
    "images": rows,
  }
  (args.output / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps({key: report[key] for key in report if key != "images"}, ensure_ascii=False, indent=2))
  return 0


def to_xyxy(box: dict[str, float]) -> tuple[float, float, float, float, float]:
  return (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"], box["score"])


def match_predictions(
  predictions: list[tuple[float, float, float, float, float]],
  targets: list[tuple[float, float, float, float]],
  threshold: float,
) -> tuple[list[tuple[int, int]], list[int]]:
  candidates = []
  for prediction_index, prediction in enumerate(predictions):
    for target_index, target in enumerate(targets):
      candidates.append((iou(prediction[:4], target), prediction_index, target_index))
  candidates.sort(reverse=True)
  matches: list[tuple[int, int]] = []
  used_predictions: set[int] = set()
  used_targets: set[int] = set()
  for score, prediction_index, target_index in candidates:
    if score < threshold:
      break
    if prediction_index in used_predictions or target_index in used_targets:
      continue
    matches.append((prediction_index, target_index))
    used_predictions.add(prediction_index)
    used_targets.add(target_index)
  false_positives = [index for index in range(len(predictions)) if index not in used_predictions]
  return matches, false_positives


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
  left = max(a[0], b[0])
  top = max(a[1], b[1])
  right = min(a[2], b[2])
  bottom = min(a[3], b[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  return intersection / max(1e-9, area_a + area_b - intersection)


def draw_preview(
  image: Image.Image,
  predictions: list[tuple[float, float, float, float, float]],
  targets: list[tuple[float, float, float, float]],
  matches: list[tuple[int, int]],
) -> Image.Image:
  preview = image.copy()
  draw = ImageDraw.Draw(preview)
  matched_predictions = {prediction for prediction, _ in matches}
  for target in targets:
    draw.rectangle(target, outline=(70, 210, 255), width=7)
  for index, prediction in enumerate(predictions):
    color = (70, 225, 120) if index in matched_predictions else (255, 80, 80)
    draw.rectangle(prediction[:4], outline=color, width=7)
    draw.text((prediction[0] + 4, max(0, prediction[1] - 24)), f"{prediction[4]:.2f}", fill=color)
  return preview


if __name__ == "__main__":
  raise SystemExit(main())
