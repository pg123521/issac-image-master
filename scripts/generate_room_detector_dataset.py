#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_JSON = PROJECT_ROOT / "data" / "objects.en.json"
ICON_ROOT = PROJECT_ROOT / "public"
SOURCE_ROOT = PROJECT_ROOT / "data" / "sources" / "detection"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "detection" / "room-collectible-v1"
CANVAS_SIZE = 1024


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate a one-class Isaac room collectible detector dataset.")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--train-count", type=int, default=600)
  parser.add_argument("--val-count", type=int, default=120)
  parser.add_argument("--seed", type=int, default=20260715)
  args = parser.parse_args()

  objects = json.loads(OBJECTS_JSON.read_text(encoding="utf-8"))
  icons = [Image.open(ICON_ROOT / item["iconPath"].lstrip("/")).convert("RGBA") for item in objects]
  metadata = json.loads((SOURCE_ROOT / "annotations.json").read_text(encoding="utf-8"))
  train_sources = load_sources(metadata, "train-background")
  validation_sources = load_sources(metadata, "real-validation")

  if args.output.exists():
    shutil.rmtree(args.output)
  for split in ("train", "val", "real"):
    (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
    (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

  generate_split(args.output, "train", args.train_count, train_sources, icons, args.seed)
  real_tile_count = generate_real_training_tiles(args.output, train_sources)
  generate_split(args.output, "val", args.val_count, validation_sources, icons, args.seed + 1_000_003)
  copy_real_validation(args.output, validation_sources)
  write_yaml(args.output)
  write_preview(args.output, "train", min(16, args.train_count))
  write_real_preview(args.output, len(validation_sources))
  print(f"wrote {args.output}")
  print(
    f"train={args.train_count} synthetic + {real_tile_count} real tiles "
    f"synthetic_val={args.val_count} real_val={len(validation_sources)}"
  )
  return 0


def load_sources(metadata: dict[str, Any], split: str) -> list[dict[str, Any]]:
  sources = []
  for filename, entry in metadata.items():
    if entry["split"] != split:
      continue
    sources.append({
      **entry,
      "filename": filename,
      "image": Image.open(SOURCE_ROOT / filename).convert("RGB"),
    })
  if not sources:
    raise ValueError(f"no detection sources for split {split}")
  return sources


def generate_split(
  output: Path,
  split: str,
  count: int,
  sources: list[dict[str, Any]],
  icons: list[Image.Image],
  seed: int,
) -> None:
  for index in range(count):
    rng = random.Random(seed + index * 1009)
    background = sample_background(rng.choice(sources), rng)
    labels: list[tuple[int, int, int, int]] = []
    object_count = 0 if rng.random() < 0.16 else rng.randint(1, 5)
    for _ in range(object_count):
      box = place_collectible(background, rng.choice(icons), rng, labels)
      if box:
        labels.append(box)

    image_path = output / "images" / split / f"synthetic-{index:05d}.jpg"
    label_path = output / "labels" / split / f"synthetic-{index:05d}.txt"
    quality = rng.randint(72, 94)
    background.convert("RGB").save(image_path, quality=quality, subsampling=0)
    write_labels(label_path, labels, CANVAS_SIZE, CANVAS_SIZE)
    if (index + 1) % 100 == 0 or index + 1 == count:
      print(f"{split}: {index + 1}/{count}", flush=True)


def sample_background(source: dict[str, Any], rng: random.Random) -> Image.Image:
  image: Image.Image = source["image"]
  left, top, right, bottom = source["roomRect"]
  excludes = source.get("excludeBoxes", source.get("boxes", []))
  crop = None
  for _ in range(60):
    side = rng.randint(520, min(880, right - left, bottom - top))
    x = rng.randint(left, max(left, right - side))
    y = rng.randint(top, max(top, bottom - side))
    candidate = (x, y, x + side, y + side)
    if not any(intersection_ratio(candidate, tuple(box)) > 0 for box in excludes):
      crop = image.crop(candidate)
      break
  if crop is None:
    side = min(right - left, bottom - top)
    crop = image.crop((left, top, left + side, top + side))

  crop = crop.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR)
  crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.82, 1.12))
  crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.88, 1.12))
  if rng.random() < 0.28:
    crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.65)))
  return crop.convert("RGBA")


def generate_real_training_tiles(output: Path, sources: list[dict[str, Any]]) -> int:
  written = 0
  for source_index, source in enumerate(sources):
    image: Image.Image = source["image"]
    targets = [tuple(int(value) for value in box) for box in source.get("excludeBoxes", [])]
    seen: set[tuple[int, int, int, int]] = set()
    for side in (640, 900, 1100):
      if side > min(image.width, image.height):
        continue
      x_count = max(2, math.ceil(image.width / side) + 1)
      y_count = max(2, math.ceil(image.height / side) + 1)
      for x_index in range(x_count):
        left = round((image.width - side) * x_index / max(1, x_count - 1))
        for y_index in range(y_count):
          top = round((image.height - side) * y_index / max(1, y_count - 1))
          tile_box = (left, top, left + side, top + side)
          if tile_box in seen:
            continue
          seen.add(tile_box)
          coverage = [intersection_over_target(tile_box, target) for target in targets]
          if any(0 < ratio < 0.70 for ratio in coverage):
            continue

          labels = []
          for target, ratio in zip(targets, coverage, strict=True):
            if ratio < 0.70:
              continue
            target_left = max(target[0], tile_box[0]) - tile_box[0]
            target_top = max(target[1], tile_box[1]) - tile_box[1]
            target_right = min(target[2], tile_box[2]) - tile_box[0]
            target_bottom = min(target[3], tile_box[3]) - tile_box[1]
            scale = CANVAS_SIZE / side
            labels.append((
              round(target_left * scale),
              round(target_top * scale),
              round(target_right * scale),
              round(target_bottom * scale),
            ))

          crop = image.crop(tile_box).resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR)
          stem = f"real-tile-{source_index + 1:02d}-{written:04d}"
          crop.save(output / "images" / "train" / f"{stem}.jpg", quality=92, subsampling=0)
          write_labels(output / "labels" / "train" / f"{stem}.txt", labels, CANVAS_SIZE, CANVAS_SIZE)
          written += 1
  print(f"real training tiles: {written}", flush=True)
  return written


def place_collectible(
  canvas: Image.Image,
  icon: Image.Image,
  rng: random.Random,
  existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
  bbox = icon.getchannel("A").getbbox()
  if not bbox:
    return None
  sprite = icon.crop(bbox)
  longest = math.exp(rng.uniform(math.log(24), math.log(112)))
  scale = longest / max(sprite.size)
  width = max(8, round(sprite.width * scale))
  height = max(8, round(sprite.height * scale))
  interpolation = rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR])
  sprite = sprite.resize((width, height), interpolation)
  sprite = sprite.rotate(rng.uniform(-4, 4), resample=Image.Resampling.BICUBIC, expand=True)

  alpha_box = sprite.getchannel("A").getbbox()
  if not alpha_box:
    return None
  sprite = sprite.crop(alpha_box)
  width, height = sprite.size
  for _ in range(30):
    x = rng.randint(30, max(30, CANVAS_SIZE - width - 30))
    y = rng.randint(55, max(55, CANVAS_SIZE - height - 70))
    box = (x, y, x + width, y + height)
    if all(intersection_ratio(box, other) < 0.08 for other in existing):
      break
  else:
    return None

  if rng.random() < 0.45:
    draw = ImageDraw.Draw(canvas, "RGBA")
    shadow_width = max(8, int(width * rng.uniform(0.45, 0.85)))
    shadow_height = max(3, int(height * 0.14))
    shadow_x = x + (width - shadow_width) // 2
    shadow_y = min(CANVAS_SIZE - shadow_height, y + height - shadow_height // 2)
    draw.ellipse(
      (shadow_x, shadow_y, shadow_x + shadow_width, shadow_y + shadow_height),
      fill=(20, 10, 10, rng.randint(55, 115)),
    )
  canvas.alpha_composite(sprite, (x, y))
  return box


def copy_real_validation(output: Path, sources: list[dict[str, Any]]) -> None:
  for index, source in enumerate(sources):
    image: Image.Image = source["image"]
    stem = f"real-{index + 1:02d}"
    image.save(output / "images" / "real" / f"{stem}.jpg", quality=96, subsampling=0)
    boxes = [tuple(int(value) for value in box) for box in source.get("boxes", [])]
    write_labels(output / "labels" / "real" / f"{stem}.txt", boxes, image.width, image.height)


def write_labels(path: Path, boxes: list[tuple[int, int, int, int]], width: int, height: int) -> None:
  lines = []
  for left, top, right, bottom in boxes:
    center_x = (left + right) / 2 / width
    center_y = (top + bottom) / 2 / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    lines.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
  path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(output: Path) -> None:
  root = output.resolve()
  (output / "dataset.yaml").write_text(
    f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: room_collectible\n",
    encoding="utf-8",
  )
  (output / "real-validation.yaml").write_text(
    f"path: {root}\ntrain: images/train\nval: images/real\nnames:\n  0: room_collectible\n",
    encoding="utf-8",
  )


def write_preview(output: Path, split: str, count: int) -> None:
  cell = 320
  columns = 4
  rows = math.ceil(count / columns)
  preview = Image.new("RGB", (columns * cell, rows * cell), (18, 18, 18))
  for index in range(count):
    image = Image.open(output / "images" / split / f"synthetic-{index:05d}.jpg").convert("RGB")
    labels = parse_labels(output / "labels" / split / f"synthetic-{index:05d}.txt")
    draw = ImageDraw.Draw(image)
    for center_x, center_y, width, height in labels:
      left = int((center_x - width / 2) * image.width)
      top = int((center_y - height / 2) * image.height)
      right = int((center_x + width / 2) * image.width)
      bottom = int((center_y + height / 2) * image.height)
      draw.rectangle((left, top, right, bottom), outline=(255, 194, 61), width=5)
    image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
    preview.paste(image, ((index % columns) * cell, (index // columns) * cell))
  preview.save(output / "preview.jpg", quality=92)


def write_real_preview(output: Path, count: int) -> None:
  cell_width = 900
  cell_height = 420
  preview = Image.new("RGB", (cell_width, cell_height * count), (18, 18, 18))
  for index in range(count):
    stem = f"real-{index + 1:02d}"
    image = Image.open(output / "images" / "real" / f"{stem}.jpg").convert("RGB")
    labels = parse_labels(output / "labels" / "real" / f"{stem}.txt")
    draw = ImageDraw.Draw(image)
    for center_x, center_y, width, height in labels:
      left = int((center_x - width / 2) * image.width)
      top = int((center_y - height / 2) * image.height)
      right = int((center_x + width / 2) * image.width)
      bottom = int((center_y + height / 2) * image.height)
      draw.rectangle((left, top, right, bottom), outline=(255, 194, 61), width=8)
    image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
    preview.paste(image, (0, index * cell_height))
  preview.save(output / "real-preview.jpg", quality=94)


def parse_labels(path: Path) -> list[tuple[float, float, float, float]]:
  labels = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      _, center_x, center_y, width, height = line.split()
      labels.append((float(center_x), float(center_y), float(width), float(height)))
  return labels


def intersection_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
  left = max(a[0], b[0])
  top = max(a[1], b[1])
  right = min(a[2], b[2])
  bottom = min(a[3], b[3])
  intersection = max(0, right - left) * max(0, bottom - top)
  return intersection / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def intersection_over_target(a: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> float:
  left = max(a[0], target[0])
  top = max(a[1], target[1])
  right = min(a[2], target[2])
  bottom = min(a[3], target[3])
  intersection = max(0, right - left) * max(0, bottom - top)
  target_area = max(1, (target[2] - target[0]) * (target[3] - target[1]))
  return intersection / target_area


if __name__ == "__main__":
  raise SystemExit(main())
