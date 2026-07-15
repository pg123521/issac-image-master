#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from PIL import Image


SOURCE_URL = "https://issac-icecat.azurewebsites.net/"
CSS_PATH = "assets/main.css?v=7"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "sources" / "icecat"
OUTPUT_JSON = PROJECT_ROOT / "data" / "items.zh-CN.json"
ICON_DIR = PROJECT_ROOT / "public" / "items" / "icons"


@dataclass(frozen=True)
class CssIcon:
  sprite_url: str
  x: int
  y: int
  width: int
  height: int


BASE_SELECTORS = {
  "r-itm": ".rebirth-item",
  "a-itm": ".a-item",
  "ap-itm": ".ap-item",
}


def main() -> int:
  parser = argparse.ArgumentParser(description="Import IcaCat item text and icons.")
  parser.add_argument("--source-url", default=SOURCE_URL)
  parser.add_argument("--limit", type=int, default=0, help="Import only first N items for debugging.")
  parser.add_argument("--offline", action="store_true", help="Reuse previously downloaded IcaCat source files.")
  args = parser.parse_args()

  SOURCE_DIR.mkdir(parents=True, exist_ok=True)
  ICON_DIR.mkdir(parents=True, exist_ok=True)
  OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

  index_html = read_or_fetch_text(args.source_url, SOURCE_DIR / "index.html", args.offline)
  css_url = urljoin(args.source_url, CSS_PATH)
  css = read_or_fetch_text(css_url, SOURCE_DIR / "main.css", args.offline)

  css_icons = parse_css_icons(css, css_url)
  sprite_images = download_sprites(css_icons.values(), offline=args.offline)
  items = parse_items(index_html)
  if args.limit:
    items = items[: args.limit]

  imported = []
  missing_icons = []
  for item in items:
    icon_class = item["iconClass"]
    icon = css_icons.get(icon_class)
    if not icon:
      missing_icons.append(icon_class)
      continue

    game_id = item["gameId"]
    icon_name = f"item-{game_id:03d}.png"
    icon_path = ICON_DIR / icon_name
    item["iconFeature"] = crop_icon(sprite_images[icon.sprite_url], icon, icon_path)

    item["iconPath"] = f"/items/icons/{icon_name}"
    item["sourceUrl"] = args.source_url
    item["sourceName"] = "IcaCat 以撒图鉴"
    item.pop("iconClass", None)
    imported.append(item)

  imported.sort(key=lambda entry: entry["gameId"])
  output_json = SOURCE_DIR / "items.preview.zh-CN.json" if args.limit else OUTPUT_JSON
  output_json.write_text(json.dumps(imported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  metadata = {
    "sourceUrl": args.source_url,
    "cssUrl": css_url,
    "itemCount": len(imported),
    "missingIconCount": len(missing_icons),
    "missingIcons": sorted(set(missing_icons)),
  }
  (SOURCE_DIR / "import-report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  print(f"Imported {len(imported)} items")
  print(f"Wrote {output_json}")
  print(f"Wrote icons to {ICON_DIR}")
  if missing_icons:
    print(f"Missing {len(missing_icons)} icons; see {SOURCE_DIR / 'import-report.json'}", file=sys.stderr)
    return 1
  return 0


def fetch_text(url: str, dest: Path) -> str:
  with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read()
  dest.write_bytes(data)
  return data.decode("utf-8-sig", errors="replace")


def read_or_fetch_text(url: str, dest: Path, offline: bool) -> str:
  if offline and dest.exists():
    return dest.read_text(encoding="utf-8-sig", errors="replace")
  return fetch_text(url, dest)


def fetch_bytes(url: str, dest: Path) -> bytes:
  with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read()
  dest.write_bytes(data)
  return data


def parse_css_icons(css: str, css_url: str) -> dict[str, CssIcon]:
  rules = parse_css_rules(css)
  base_images: dict[str, str] = {}
  base_sizes: dict[str, tuple[int, int]] = {}

  for prefix, selector in BASE_SELECTORS.items():
    declaration = find_base_declaration(rules, selector)
    sprite = extract_url(declaration)
    if sprite:
      base_images[prefix] = urljoin(css_url, sprite)
    base_sizes[prefix] = (
      extract_px(declaration, "width") or 50,
      extract_px(declaration, "height") or 50,
    )

  icons: dict[str, CssIcon] = {}
  for selector, declaration in rules.items():
    for prefix, base_selector in BASE_SELECTORS.items():
      match = re.search(rf"(?:^|[\s>+~]){re.escape(base_selector)}\.({prefix}\d+)(?:$|[:.\s#])", selector)
      if not match:
        continue
      icon_class = match.group(1)
      x, y = extract_background_position(declaration)
      width = extract_px(declaration, "width") or base_sizes[prefix][0]
      height = extract_px(declaration, "height") or base_sizes[prefix][1]
      sprite_url = base_images[prefix]
      icons[icon_class] = CssIcon(sprite_url=sprite_url, x=x, y=y, width=width, height=height)

  return icons


def find_base_declaration(rules: dict[str, str], base_selector: str) -> str:
  exact = rules.get(base_selector, "")
  if extract_url(exact):
    return exact
  for selector, declaration in rules.items():
    if selector.endswith(base_selector) and extract_url(declaration):
      return declaration
  return exact


def parse_css_rules(css: str) -> dict[str, str]:
  rules: dict[str, str] = {}
  for selector_text, declaration in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
    selectors = [selector.strip() for selector in selector_text.split(",")]
    for selector in selectors:
      rules[selector] = declaration
  return rules


def extract_url(declaration: str) -> str | None:
  match = re.search(r"url\(([^)]+)\)", declaration)
  if not match:
    return None
  return match.group(1).strip("'\"")


def extract_px(declaration: str, prop: str) -> int | None:
  match = re.search(rf"(?:^|;){re.escape(prop)}:(-?\d+)px", declaration)
  if not match:
    return None
  return int(match.group(1))


def extract_background_position(declaration: str) -> tuple[int, int]:
  match = re.search(r"background-position:(-?\d+)px\s+(-?\d+)px", declaration)
  if match:
    return abs(int(match.group(1))), abs(int(match.group(2)))
  match = re.search(r"background-position:(-?\d+)px\s+0", declaration)
  if match:
    return abs(int(match.group(1))), 0
  return 0, 0


def download_sprites(icons: Iterable[CssIcon], offline: bool) -> dict[str, Image.Image]:
  sprites: dict[str, Image.Image] = {}
  for sprite_url in sorted({icon.sprite_url for icon in icons}):
    filename = Path(sprite_url.split("?")[0]).name
    dest = SOURCE_DIR / filename
    if not offline or not dest.exists():
      fetch_bytes(sprite_url, dest)
    sprites[sprite_url] = Image.open(dest).convert("RGBA")
  return sprites


def parse_items(index_html: str) -> list[dict]:
  items: list[dict] = []
  for block in re.findall(r"<li class=\"textbox[^>]*data-sid=\"([^\"]+)\"[^>]*>(.*?)</li>", index_html, flags=re.S):
    data_sid, body = block
    icon_match = re.search(r"<div[^>]+class=[\"']([^\"']*(?:rebirth-item|a-item|ap-item)[^\"']*)[\"']", body)
    if not icon_match:
      continue
    classes = icon_match.group(1).split()
    icon_class = next((class_name for class_name in classes if re.fullmatch(r"(?:r|a|ap)-itm\d+", class_name)), None)
    if not icon_class:
      continue

    p_entries = extract_paragraphs(body)
    name_zh = first_class_text(p_entries, "item-title")
    name_en = first_class_text(p_entries, "item-title2") or name_zh
    pickup = clean_pickup(first_class_text(p_entries, "pickup"))
    game_id = parse_item_id(first_class_text(p_entries, "r-itemid")) or int(float(data_sid))
    tags_text = first_class_text(p_entries, "tags")
    effects = [
      text.removeprefix("•").strip()
      for classes, text in p_entries
      if text
      and "item-title" not in classes
      and "item-title2" not in classes
      and "r-itemid" not in classes
      and "pickup" not in classes
      and "tags" not in classes
      and not text.startswith("类型：")
      and not text.startswith("道具池：")
    ]
    item_type = prefixed_text(p_entries, "类型：")
    pools = split_cn_list(prefixed_text(p_entries, "道具池："))

    items.append({
      "id": f"item-{game_id:03d}",
      "gameId": game_id,
      "kind": "item",
      "nameZh": name_zh,
      "nameEn": name_en,
      "pickup": pickup,
      "description": pickup,
      "effects": effects,
      "type": item_type,
      "pools": pools,
      "tags": split_tags(tags_text),
      "iconClass": icon_class,
    })
  return dedupe_items(items)


def extract_paragraphs(body: str) -> list[tuple[set[str], str]]:
  entries: list[tuple[set[str], str]] = []
  for attrs, content in re.findall(r"<p([^>]*)>(.*?)(?=</p>|<p|<ul|</span>)", body, flags=re.S):
    class_match = re.search(r"class=[\"']([^\"']+)[\"']", attrs)
    classes = set(class_match.group(1).split()) if class_match else set()
    text = normalize_text(strip_tags(content))
    if text:
      entries.append((classes, text))
  return entries


def strip_tags(value: str) -> str:
  return html.unescape(re.sub(r"<[^>]*>", "", value))


def normalize_text(value: str) -> str:
  return re.sub(r"\s+", " ", value).strip()


def first_class_text(entries: list[tuple[set[str], str]], class_name: str) -> str:
  return next((text for classes, text in entries if class_name in classes), "")


def prefixed_text(entries: list[tuple[set[str], str]], prefix: str) -> str:
  for _, text in entries:
    if text.startswith(prefix):
      return text[len(prefix):].strip()
  return ""


def parse_item_id(value: str) -> int | None:
  match = re.search(r"(\d+)", value)
  return int(match.group(1)) if match else None


def clean_pickup(value: str) -> str:
  return value.strip().strip("\"“”")


def split_cn_list(value: str) -> list[str]:
  if not value:
    return []
  return [part.strip() for part in re.split(r"[，,、/]", value) if part.strip()]


def split_tags(value: str) -> list[str]:
  value = value.strip()
  if value.startswith("*"):
    value = value[1:]
  return [part.strip() for part in value.split(",") if part.strip()]


def dedupe_items(items: list[dict]) -> list[dict]:
  seen = set()
  deduped = []
  for item in items:
    if item["gameId"] in seen:
      continue
    seen.add(item["gameId"])
    deduped.append(item)
  return deduped


def crop_icon(sprite: Image.Image, icon: CssIcon, dest: Path) -> dict[str, float]:
  crop = sprite.crop((icon.x, icon.y, icon.x + icon.width, icon.y + icon.height))
  square = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
  scale = min(56 / crop.width, 56 / crop.height, 1)
  resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.NEAREST)
  square.alpha_composite(resized, ((64 - resized.width) // 2, (64 - resized.height) // 2))
  square.save(dest)
  feature = image_feature(square)
  feature["aspect"] = round(icon.width / max(1, icon.height), 3)
  return feature


def image_feature(image: Image.Image) -> dict[str, float]:
  rgba = image.convert("RGBA")
  pixels = list(rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata())
  visible = [(r, g, b, a) for r, g, b, a in pixels if a > 24]
  if not visible:
    return {"r": 0, "g": 0, "b": 0, "aspect": image.width / max(1, image.height), "hash": perceptual_hash(rgba)}
  r = sum(pixel[0] for pixel in visible) / len(visible)
  g = sum(pixel[1] for pixel in visible) / len(visible)
  b = sum(pixel[2] for pixel in visible) / len(visible)
  alpha = rgba.getchannel("A")
  bbox = alpha.point(lambda value: 255 if value > 24 else 0).getbbox()
  if bbox:
    aspect = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
  else:
    aspect = image.width / max(1, image.height)
  return {
    "r": round(r, 3),
    "g": round(g, 3),
    "b": round(b, 3),
    "aspect": round(aspect, 3),
    "hash": perceptual_hash(rgba),
    "descriptor": visual_descriptor(rgba),
  }


def perceptual_hash(image: Image.Image) -> str:
  small = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
  values = [pixel / 255 for pixel in (small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())]
  coeffs: list[float] = []
  for v in range(8):
    for u in range(8):
      total = 0.0
      for y in range(32):
        for x in range(32):
          total += values[y * 32 + x] * math.cos(((2 * x + 1) * u * math.pi) / 64) * math.cos(((2 * y + 1) * v * math.pi) / 64)
      cu = 1 / math.sqrt(2) if u == 0 else 1
      cv = 1 / math.sqrt(2) if v == 0 else 1
      coeffs.append(0.25 * cu * cv * total)
  comparable = coeffs[1:]
  median = sorted(comparable)[len(comparable) // 2]
  bits = ["1" if value > median else "0" for value in comparable]
  return bits_to_hex(bits)


def visual_descriptor(image: Image.Image) -> list[int]:
  canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
  canvas.alpha_composite(image.convert("RGBA").resize((64, 64), Image.Resampling.NEAREST))
  gray = canvas.convert("L").resize((16, 16), Image.Resampling.LANCZOS)
  gray_values = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
  edges: list[int] = []
  for y in range(16):
    for x in range(16):
      current = gray_values[y * 16 + x]
      right = gray_values[y * 16 + min(15, x + 1)]
      down = gray_values[min(15, y + 1) * 16 + x]
      edges.append(min(255, abs(current - right) + abs(current - down)))
  return [round(value / 8) for value in gray_values] + [round(value / 8) for value in edges]


def bits_to_hex(bits: list[str]) -> str:
  padded = bits + ["0"] * ((4 - len(bits) % 4) % 4)
  return "".join(f"{int(''.join(padded[index:index + 4]), 2):x}" for index in range(0, len(padded), 4))


if __name__ == "__main__":
  raise SystemExit(main())
