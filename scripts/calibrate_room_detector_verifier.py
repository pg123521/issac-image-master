#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from mobileclip_item_search import DEFAULT_INDEX, DEFAULT_WEIGHTS, Searcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "sources" / "detection"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "detection" / "room-collectible-evaluation" / "report.json"


def main() -> int:
  parser = argparse.ArgumentParser(description="Calibrate MobileCLIP as an open-set detector verifier.")
  parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
  parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
  parser.add_argument("--match-iou", type=float, default=0.30)
  args = parser.parse_args()

  report = json.loads(args.report.read_text(encoding="utf-8"))
  annotations = json.loads((SOURCE_ROOT / "annotations.json").read_text(encoding="utf-8"))
  crops: list[Image.Image] = []
  records: list[dict[str, object]] = []
  for image_report in report["images"]:
    filename = image_report["image"]
    image = Image.open(SOURCE_ROOT / filename).convert("RGB")
    entry = annotations[filename]
    targets = entry.get("boxes", entry.get("excludeBoxes", []))
    for box in image_report["boxes"]:
      crop_box = square_crop_box(box, image.width, image.height)
      crops.append(image.crop(crop_box))
      prediction = (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])
      records.append({
        "image": filename,
        "detectorScore": box["score"],
        "isTarget": any(iou(prediction, target) >= args.match_iou for target in targets),
      })

  searcher = Searcher(args.index, args.weights)
  features = []
  batch_size = 64
  for start in range(0, len(crops), batch_size):
    features.append(searcher.encoder.encode(crops[start : start + batch_size]))
    print(f"encoded {min(start + batch_size, len(crops))}/{len(crops)}", flush=True)
  query_vectors = torch.cat(features)
  scores = query_vectors @ searcher.vectors.T
  values, indices = scores.topk(2, dim=1)
  for record, score_values, vector_indices in zip(records, values.tolist(), indices.tolist(), strict=True):
    item_index = int(searcher.index_to_item[vector_indices[0]])
    record["verifierScore"] = round(float(score_values[0]), 6)
    record["verifierMargin"] = round(float(score_values[0] - score_values[1]), 6)
    record["nearestItem"] = searcher.labels[item_index]["name_en"]

  positive_scores = sorted(float(record["verifierScore"]) for record in records if record["isTarget"])
  negative_scores = sorted(float(record["verifierScore"]) for record in records if not record["isTarget"])
  thresholds = sorted(set(positive_scores + [round(value / 100, 2) for value in range(20, 96)]))
  sweep = []
  for threshold in thresholds:
    true_positives = sum(score >= threshold for score in positive_scores)
    false_positives = sum(score >= threshold for score in negative_scores)
    sweep.append({
      "threshold": round(threshold, 6),
      "recall": round(true_positives / max(1, len(positive_scores)), 6),
      "falsePositives": false_positives,
    })
  useful = [row for row in sweep if row["recall"] >= 1.0]
  best_full_recall = min(useful, key=lambda row: row["falsePositives"]) if useful else None
  summary = {
    "targets": len(positive_scores),
    "negatives": len(negative_scores),
    "targetScores": positive_scores,
    "negativeScoreRange": [negative_scores[0], negative_scores[-1]],
    "bestFullRecall": best_full_recall,
  }
  output = args.report.with_name("verifier-calibration.json")
  output.write_text(
    json.dumps({"summary": summary, "records": records, "sweep": sweep}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))
  print(f"wrote {output}")
  return 0


def square_crop_box(box: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
  center_x = box["x"] + box["w"] / 2
  center_y = box["y"] + box["h"] / 2
  side = max(box["w"], box["h"])
  left = max(0, min(width - side, center_x - side / 2))
  top = max(0, min(height - side, center_y - side / 2))
  return (round(left), round(top), round(left + side), round(top + side))


def iou(a: tuple[float, ...], b: list[float]) -> float:
  left = max(a[0], b[0])
  top = max(a[1], b[1])
  right = min(a[2], b[2])
  bottom = min(a[3], b[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  return intersection / max(1e-9, area_a + area_b - intersection)


if __name__ == "__main__":
  raise SystemExit(main())
