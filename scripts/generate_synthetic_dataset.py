#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITEMS_JSON = PROJECT_ROOT / "data" / "items.zh-CN.json"
ICON_ROOT = PROJECT_ROOT / "public"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "synthetic-v3"
DEFAULT_BACKGROUND = PROJECT_ROOT / "4081784094085_.pic.jpg"


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate synthetic Isaac item crops for classifier/embedding training.")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--per-item", type=int, default=100)
  parser.add_argument("--size", type=int, default=96)
  parser.add_argument("--seed", type=int, default=20260715)
  parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
  parser.add_argument("--limit-items", type=int, default=0)
  parser.add_argument("--local-patch-rate", type=float, default=0.5, help="Fraction of samples that keep only a local visible patch.")
  parser.add_argument("--visible-min", type=float, default=0.65, help="Minimum visible area ratio for truncated detector crops.")
  parser.add_argument("--visible-max", type=float, default=0.9, help="Maximum visible area ratio for truncated detector crops.")
  parser.add_argument("--closeup-rate", type=float, default=0.28, help="Fraction of samples rendered as close-up user crops.")
  parser.add_argument("--hard-style-rate", type=float, default=0.45, help="Fraction of samples with stronger real-screenshot styling.")
  args = parser.parse_args()

  random.seed(args.seed)
  items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
  if args.limit_items:
    items = items[: args.limit_items]

  image_dir = args.output / "images"
  image_dir.mkdir(parents=True, exist_ok=True)
  backgrounds = load_backgrounds(args.background)

  manifest_path = args.output / "manifest.csv"
  with manifest_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
      "path",
      "item_id",
      "game_id",
      "name_zh",
      "name_en",
      "scale",
      "rotation",
      "background",
      "style",
      "crop_mode",
      "visible_ratio",
      "crop_anchor",
    ])
    writer.writeheader()

    total = 0
    for item in items:
      icon_path = ICON_ROOT / item["iconPath"].lstrip("/")
      icon = Image.open(icon_path).convert("RGBA")
      for sample_index in range(args.per_item):
        sample, meta = synthesize_sample(
          icon,
          backgrounds,
          args.size,
          local_patch_rate=args.local_patch_rate,
          visible_min=args.visible_min,
          visible_max=args.visible_max,
          closeup_rate=args.closeup_rate,
          hard_style_rate=args.hard_style_rate,
        )
        filename = f"{item['id']}_{sample_index:04d}.png"
        rel_path = Path("images") / filename
        sample.save(args.output / rel_path)
        writer.writerow({
          "path": str(rel_path),
          "item_id": item["id"],
          "game_id": item["gameId"],
          "name_zh": item["nameZh"],
          "name_en": item["nameEn"],
          **meta,
        })
        total += 1

  labels = [{
    "item_id": item["id"],
    "game_id": item["gameId"],
    "name_zh": item["nameZh"],
    "name_en": item["nameEn"],
    "icon_path": item["iconPath"],
  } for item in items]
  (args.output / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  print(f"Generated {total} samples for {len(items)} items")
  print(f"Wrote {manifest_path}")
  return 0


def load_backgrounds(path: Path) -> list[Image.Image]:
  backgrounds: list[Image.Image] = []
  if path.exists():
    screenshot = Image.open(path).convert("RGB")
    width, height = screenshot.size
    for _ in range(80):
      crop_size = random.randint(96, 220)
      x = random.randint(max(0, int(width * 0.18)), max(0, min(width - crop_size, int(width * 0.82))))
      y = random.randint(max(0, int(height * 0.18)), max(0, min(height - crop_size, int(height * 0.82))))
      backgrounds.append(screenshot.crop((x, y, x + crop_size, y + crop_size)).resize((128, 128), Image.Resampling.BILINEAR))
  for _ in range(40):
    backgrounds.append(make_noise_background(128))
  return backgrounds


def make_noise_background(size: int) -> Image.Image:
  base = Image.new("RGB", (size, size), random.choice([(105, 84, 68), (88, 72, 60), (118, 92, 74), (76, 68, 64)]))
  draw = ImageDraw.Draw(base)
  for _ in range(160):
    x = random.randint(0, size - 1)
    y = random.randint(0, size - 1)
    radius = random.randint(1, 7)
    color = random.choice([(80, 68, 62), (129, 101, 82), (92, 88, 82), (60, 56, 54)])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
  return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))


def synthesize_sample(
  icon: Image.Image,
  backgrounds: list[Image.Image],
  size: int,
  *,
  local_patch_rate: float,
  visible_min: float,
  visible_max: float,
  closeup_rate: float,
  hard_style_rate: float,
) -> tuple[Image.Image, dict]:
  background_name = "screenshot" if random.random() < 0.7 else "noise"
  style = "hard" if random.random() < hard_style_rate else "normal"
  closeup = random.random() < closeup_rate
  render_size = round(size * 2.4)
  background = random.choice(backgrounds).resize((render_size, render_size), Image.Resampling.BILINEAR).convert("RGBA")
  background = ImageEnhance.Brightness(background).enhance(random.uniform(0.64, 1.26) if style == "hard" else random.uniform(0.72, 1.18))
  background = ImageEnhance.Contrast(background).enhance(random.uniform(0.72, 1.34) if style == "hard" else random.uniform(0.82, 1.18))

  if random.random() < 0.72:
    draw_pedestal(background)

  prepared_icon = prepare_icon(icon, hard_style=style == "hard")
  crop_meta = choose_detection_crop(local_patch_rate=local_patch_rate, visible_min=visible_min, visible_max=visible_max)
  scale = random.uniform(1.0, 1.62) if closeup else random.uniform(0.58, 1.18)
  target = max(18, round(size * 0.52 * scale))
  prepared_icon.thumbnail((target, target), Image.Resampling.NEAREST)

  rotation = random.uniform(-10, 10) if style == "hard" else random.uniform(-7, 7)
  if abs(rotation) > 1.5:
    prepared_icon = prepared_icon.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

  shadow = make_shadow(prepared_icon)
  cx = render_size // 2 + random.randint(-5, 5)
  cy = render_size // 2 + random.randint(-6, 6)
  paste_center(background, shadow, cx + random.randint(1, 3), cy + random.randint(3, 6))
  paste_center(background, prepared_icon, cx, cy)

  sample = crop_detector_view(background, size, cx, cy, prepared_icon.size, crop_meta).convert("RGB")
  sample = apply_capture_style(sample, hard_style=style == "hard")

  return sample, {
    "scale": f"{scale:.3f}",
    "rotation": f"{rotation:.2f}",
    "background": background_name,
    "style": f"{style}{'_closeup' if closeup else ''}",
    **crop_meta,
  }


def prepare_icon(icon: Image.Image, *, hard_style: bool) -> Image.Image:
  alpha = icon.getchannel("A")
  bbox = alpha.point(lambda value: 255 if value > 16 else 0).getbbox()
  cropped = icon.crop(bbox) if bbox else icon
  cropped = ImageEnhance.Brightness(cropped).enhance(random.uniform(0.72, 1.32) if hard_style else random.uniform(0.82, 1.22))
  cropped = ImageEnhance.Contrast(cropped).enhance(random.uniform(0.78, 1.46) if hard_style else random.uniform(0.86, 1.22))
  if hard_style and random.random() < 0.35:
    cropped = add_dark_outline(cropped)
  return cropped


def add_dark_outline(icon: Image.Image) -> Image.Image:
  alpha = icon.getchannel("A")
  outline_alpha = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(radius=0.4))
  outline = Image.new("RGBA", icon.size, (24, 7, 9, 0))
  outline.putalpha(outline_alpha.point(lambda value: round(value * 0.7)))
  outline.alpha_composite(icon, (0, 0))
  return outline


def apply_capture_style(sample: Image.Image, *, hard_style: bool) -> Image.Image:
  if hard_style:
    if random.random() < 0.45:
      sample = pixelate_roundtrip(sample, factor=random.choice([2, 3]))
    sample = ImageEnhance.Brightness(sample).enhance(random.uniform(0.82, 1.2))
    sample = ImageEnhance.Contrast(sample).enhance(random.uniform(0.86, 1.28))
    sample = ImageEnhance.Color(sample).enhance(random.uniform(0.78, 1.22))
    sample = sample.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.72)))
    if random.random() < 0.7:
      sample = jpeg_roundtrip(sample, quality=random.randint(42, 82))
  else:
    sample = sample.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.45)))
    if random.random() < 0.45:
      sample = jpeg_roundtrip(sample, quality=random.randint(55, 88))
  return sample


def pixelate_roundtrip(image: Image.Image, factor: int) -> Image.Image:
  small = image.resize((max(1, image.width // factor), max(1, image.height // factor)), Image.Resampling.BILINEAR)
  return small.resize(image.size, Image.Resampling.NEAREST)


def choose_detection_crop(*, local_patch_rate: float, visible_min: float, visible_max: float) -> dict:
  if random.random() >= local_patch_rate:
    return {
      "crop_mode": "full",
      "visible_ratio": "1.000",
      "crop_anchor": "center",
    }

  low = min(0.99, max(0.2, visible_min))
  high = min(0.99, max(low, visible_max))
  ratio = random.uniform(low, high)
  anchor = random.choice([
    "top_left",
    "top",
    "top_right",
    "left",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
  ])
  return {
    "crop_mode": "truncated",
    "visible_ratio": f"{ratio:.3f}",
    "crop_anchor": anchor,
  }


def crop_detector_view(canvas: Image.Image, size: int, icon_cx: int, icon_cy: int, icon_size: tuple[int, int], crop_meta: dict) -> Image.Image:
  if crop_meta["crop_mode"] == "full":
    crop_cx = icon_cx + random.randint(-4, 4)
    crop_cy = icon_cy + random.randint(-4, 4)
  else:
    ratio = float(crop_meta["visible_ratio"])
    icon_w, icon_h = icon_size
    anchor = crop_meta["crop_anchor"]
    cuts_x = "left" in anchor or "right" in anchor
    cuts_y = "top" in anchor or "bottom" in anchor
    axis_ratio = ratio ** 0.5 if cuts_x and cuts_y else ratio
    missing_w = round(icon_w * (1 - axis_ratio) * random.uniform(0.9, 1.08))
    missing_h = round(icon_h * (1 - axis_ratio) * random.uniform(0.9, 1.08))
    icon_left = icon_cx - icon_w / 2
    icon_right = icon_cx + icon_w / 2
    icon_top = icon_cy - icon_h / 2
    icon_bottom = icon_cy + icon_h / 2

    if "left" in anchor:
      crop_cx = icon_left + missing_w + size / 2
    elif "right" in anchor:
      crop_cx = icon_right - missing_w - size / 2
    else:
      crop_cx = icon_cx + random.randint(-3, 3)

    if "top" in anchor:
      crop_cy = icon_top + missing_h + size / 2
    elif "bottom" in anchor:
      crop_cy = icon_bottom - missing_h - size / 2
    else:
      crop_cy = icon_cy + random.randint(-3, 3)

  left = round(crop_cx - size / 2)
  top = round(crop_cy - size / 2)
  right = left + size
  bottom = top + size
  if left >= 0 and top >= 0 and right <= canvas.width and bottom <= canvas.height:
    return canvas.crop((left, top, right, bottom))

  padded = Image.new("RGBA", (canvas.width + size * 2, canvas.height + size * 2), (0, 0, 0, 255))
  padded.alpha_composite(canvas, (size, size))
  return padded.crop((left + size, top + size, right + size, bottom + size))


def draw_pedestal(canvas: Image.Image) -> None:
  draw = ImageDraw.Draw(canvas, "RGBA")
  width, height = canvas.size
  cx = width // 2 + random.randint(-4, 4)
  cy = int(height * random.uniform(0.63, 0.76))
  w = random.randint(round(width * 0.28), round(width * 0.46))
  h = random.randint(round(height * 0.09), round(height * 0.16))
  draw.ellipse((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), fill=(72, 66, 63, 125))
  draw.ellipse((cx - w // 3, cy - h // 3, cx + w // 3, cy + h // 4), fill=(132, 124, 116, 115))


def make_shadow(icon: Image.Image) -> Image.Image:
  shadow = Image.new("RGBA", icon.size, (0, 0, 0, 0))
  shadow.putalpha(icon.getchannel("A").filter(ImageFilter.GaussianBlur(radius=2)))
  return shadow


def paste_center(canvas: Image.Image, image: Image.Image, cx: int, cy: int) -> None:
  x = round(cx - image.width / 2)
  y = round(cy - image.height / 2)
  canvas.alpha_composite(image, (x, y))


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
  from io import BytesIO

  buffer = BytesIO()
  image.save(buffer, format="JPEG", quality=quality)
  buffer.seek(0)
  return Image.open(buffer).convert("RGB")


if __name__ == "__main__":
  raise SystemExit(main())
