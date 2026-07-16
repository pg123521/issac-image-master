#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mobileclip_item_search import DEFAULT_INDEX, DEFAULT_WEIGHTS, MobileClipEncoder
from train_mobileclip_partial import load_icon, render_query


def main() -> int:
  parser = argparse.ArgumentParser(description="Render and evaluate exact MobileCLIP training augmentations.")
  parser.add_argument("object_id")
  parser.add_argument("--count", type=int, default=120)
  parser.add_argument("--preview-count", type=int, default=12)
  parser.add_argument("--seed", type=int, default=20260716)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
  parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  args = parser.parse_args()

  objects = json.loads((PROJECT_ROOT / "data" / "objects.en.json").read_text(encoding="utf-8"))
  target = next((obj for obj in objects if obj["id"] == args.object_id), None)
  if target is None:
    raise ValueError(f"unknown object id: {args.object_id}")

  icon = load_icon(target)
  samples = [
    render_query(icon, random.Random(args.seed + sample * 10_007), (0.50, 1.00), training=True)
    for sample in range(args.count)
  ]
  write_preview(icon, samples[: args.preview_count], target, args.seed, args.output)

  payload = torch.load(args.index, map_location="cpu")
  vectors = payload["vectors"].float()
  labels = payload["labels"]
  index_to_item = payload["index_to_item"].tolist()
  vector_index = next(
    index
    for index, item_index in enumerate(index_to_item)
    if labels[int(item_index)]["item_id"] == args.object_id
  )

  encoder = MobileClipEncoder(args.weights)
  query_vectors = []
  for start in range(0, len(samples), 32):
    query_vectors.append(encoder.encode(samples[start : start + 32]))
  queries = torch.cat(query_vectors)
  scores = queries @ vectors.T
  target_scores = scores[:, vector_index]
  ranks = 1 + (scores > target_scores.unsqueeze(1)).sum(dim=1)
  top1 = scores.argmax(dim=1)
  errors = Counter(
    labels[int(index_to_item[index])]["item_id"]
    for index, rank in zip(top1.tolist(), ranks.tolist(), strict=True)
    if rank != 1
  )
  gallery_scores = vectors[vector_index] @ vectors.T
  neighbors = []
  for index in gallery_scores.topk(8).indices.tolist():
    item_index = int(index_to_item[index])
    label = labels[item_index]
    neighbors.append({
      "id": label["item_id"],
      "nameZh": label["name_zh"],
      "nameEn": label["name_en"],
      "score": round(float(gallery_scores[index]), 6),
    })

  report = {
    "object": {
      "id": target["id"],
      "nameZh": target["nameZh"],
      "nameEn": target["nameEn"],
      "iconPath": target["iconPath"],
    },
    "samples": args.count,
    "device": str(encoder.device),
    "recall@1": round(float((ranks <= 1).float().mean()), 6),
    "recall@5": round(float((ranks <= 5).float().mean()), 6),
    "recall@10": round(float((ranks <= 10).float().mean()), 6),
    "meanRank": round(float(ranks.float().mean()), 4),
    "worstRank": int(ranks.max()),
    "meanTargetScore": round(float(target_scores.mean()), 6),
    "mostCommonTop1Errors": errors.most_common(10),
    "galleryNeighbors": neighbors,
    "preview": str(args.output),
  }
  print(json.dumps(report, ensure_ascii=False, indent=2))
  return 0


def write_preview(
  icon: Image.Image,
  samples: list[Image.Image],
  target: dict,
  seed: int,
  output: Path,
) -> None:
  columns = 4
  cell_width = 220
  cell_height = 180
  header_height = 230
  rows = (len(samples) + columns - 1) // columns
  canvas = Image.new("RGB", (columns * cell_width, header_height + rows * cell_height), (28, 30, 34))
  draw = ImageDraw.Draw(canvas)
  original = icon.resize((160, 160), Image.Resampling.NEAREST)
  canvas.paste(original.convert("RGB"), (24, 42), original.getchannel("A").resize((160, 160), Image.Resampling.NEAREST))
  draw.text((205, 56), f"{target['nameEn']} / {target['id']}", fill=(245, 245, 245))
  draw.text((205, 88), "Exact training augmentation: scale 0.30-1.25", fill=(200, 205, 214))
  draw.text((205, 116), "visibility 0.50-1.00 + lighting/blur/JPEG", fill=(200, 205, 214))
  draw.text((205, 144), f"base seed {seed}", fill=(160, 168, 180))

  for index, sample in enumerate(samples):
    row, column = divmod(index, columns)
    left = column * cell_width
    top = header_height + row * cell_height
    preview = sample.resize((160, 160), Image.Resampling.NEAREST)
    canvas.paste(preview, (left + 30, top))
    draw.text((left + 8, top + 8), f"#{index + 1}", fill=(255, 196, 64))
  output.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output)


if __name__ == "__main__":
  raise SystemExit(main())
