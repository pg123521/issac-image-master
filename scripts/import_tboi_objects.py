#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from PIL import Image


SOURCE_URL = "https://tboi.com/all"
BASE_URL = "https://tboi.com/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "sources" / "tboi"
OUTPUT_JSON = PROJECT_ROOT / "data" / "objects.en.json"
ICON_DIR = PROJECT_ROOT / "public" / "objects" / "icons"
ICECAT_ITEMS_JSON = PROJECT_ROOT / "data" / "items.zh-CN.json"
GENERATED_TRANSLATIONS_JSON = PROJECT_ROOT / "data" / "translations" / "objects.qwen3.5.zh-CN.json"
MANUAL_TRANSLATIONS_JSON = PROJECT_ROOT / "data" / "translations" / "objects.manual.zh-CN.json"


SPRITES = {
  "rebirth": "images/repentance-rebirth-items.png",
  "ab": "images/repentance-ab-items.png",
  "ap": "images/repentance-ap-items.png",
  "rep": "images/repentance-items.png",
  "rep2": "images/repentance-items2.png",
  "rep_trinkets": "images/repentance-trinkets.png",
  "rep_cards": "images/repentance-cards.png",
  "icecat_trinkets": "https://issac-icecat.azurewebsites.net/images/rebirth-trinkets-final.png",
  "icecat_cards": "https://issac-icecat.azurewebsites.net/images/ab-cards4.png",
}


@dataclass(frozen=True)
class IconSpec:
  sprite: str
  index: int = 0
  width: int = 50
  height: int = 50
  x: int | None = None
  y: int = 0


def main() -> int:
  parser = argparse.ArgumentParser(description="Import Platinum God / tboi.com objects.")
  parser.add_argument("--offline", action="store_true")
  parser.add_argument("--limit", type=int, default=0)
  args = parser.parse_args()

  SOURCE_DIR.mkdir(parents=True, exist_ok=True)
  ICON_DIR.mkdir(parents=True, exist_ok=True)
  html_text = read_or_fetch_text(SOURCE_URL, SOURCE_DIR / "all.html", args.offline)
  sprites = load_sprites(args.offline)
  icecat_css = read_or_fetch_text(
    "https://issac-icecat.azurewebsites.net/assets/main.css?v=7",
    SOURCE_DIR / "icecat-main.css",
    args.offline,
  )
  icecat_icons = parse_icecat_icons(icecat_css)
  tboi_icons = parse_tboi_icons(
    read_or_fetch_text(
      urljoin(BASE_URL, "assets/main.css"),
      SOURCE_DIR / "main.css",
      args.offline,
    )
  )
  icecat_html = read_or_fetch_text(
    "https://issac-icecat.azurewebsites.net/",
    SOURCE_DIR / "icecat-index.html",
    args.offline,
  )
  icecat_name_icons = parse_icecat_name_icons(icecat_html)
  objects = parse_objects(html_text)
  if args.limit:
    objects = objects[: args.limit]

  imported = []
  missing: list[dict] = []
  for obj in objects:
    spec = icon_spec_for(obj, tboi_icons, icecat_icons, icecat_name_icons)
    if not spec:
      missing.append({"id": obj["id"], "kind": obj["kind"], "classes": obj["iconClasses"]})
      continue
    sprite = sprites[spec.sprite]
    x = spec.index * spec.width if spec.x is None else spec.x
    crop = sprite.crop((x, spec.y, x + spec.width, spec.y + spec.height))
    icon_name = f"{obj['kind']}-{obj['gameId']:03d}.png"
    icon_path = ICON_DIR / icon_name
    crop.save(icon_path)
    obj["iconPath"] = f"/objects/icons/{icon_name}"
    obj["sourceName"] = "Platinum God"
    obj["sourceUrl"] = BASE_URL
    obj.pop("iconClasses", None)
    imported.append(obj)

  imported = merge_icecat_items(dedupe_objects(imported))
  imported = merge_icecat_localizations(imported, parse_icecat_localizations(icecat_html))
  imported = merge_generated_translations(imported)
  imported = merge_translation_file(imported, MANUAL_TRANSLATIONS_JSON)
  imported.sort(key=lambda entry: (kind_order(entry["kind"]), entry["gameId"], entry["nameEn"]))
  remove_unreferenced_icons(imported)
  OUTPUT_JSON.write_text(json.dumps(imported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  (SOURCE_DIR / "objects-import-report.json").write_text(json.dumps({
    "sourceUrl": SOURCE_URL,
    "count": len(imported),
    "missingCount": len(missing),
    "missing": missing[:200],
    "kindCounts": kind_counts(imported),
  }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"Imported {len(imported)} objects")
  print(kind_counts(imported))
  if missing:
    print(f"Missing {len(missing)} icons; see report", file=sys.stderr)
  return 0 if not missing else 1


def read_or_fetch_text(url: str, dest: Path, offline: bool) -> str:
  if offline and dest.exists():
    return decode_text(dest.read_bytes())
  data = urllib.request.urlopen(url, timeout=60).read()
  dest.write_bytes(data)
  return decode_text(data)


def decode_text(data: bytes) -> str:
  if data.startswith(b"\x1f\x8b"):
    data = gzip.decompress(data)
  return data.decode("utf-8", errors="replace")


def load_sprites(offline: bool) -> dict[str, Image.Image]:
  sprites = {}
  for key, rel in SPRITES.items():
    dest = SOURCE_DIR / Path(rel.split("?")[0]).name
    if not offline or not dest.exists():
      url = rel if rel.startswith("http") else urljoin(BASE_URL, rel)
      dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    sprites[key] = Image.open(dest).convert("RGBA")
  return sprites


def parse_objects(text: str) -> list[dict]:
  out = []
  for attrs, body in re.findall(r"<li class=\"textbox\"([^>]*)>(.*?)</li>", text, flags=re.S):
    sid = attr_value(attrs, "data-sid")
    if not sid or not sid.isdigit():
      continue
    icon_match = re.search(r"<div[^>]+class=\"([^\"]*)\"", body)
    if not icon_match:
      continue
    classes = icon_match.group(1).split()
    kind = kind_from_classes(classes)
    if not kind:
      continue
    game_id = int(sid)
    name = first_p(body, "item-title") or first_p(body, "trinket-title") or first_p(body, "card-title")
    if not name:
      continue
    pickup = strip_tags(first_p(body, "pickup"))
    paragraphs = extract_plain_paragraphs(body)
    effects = [
      p for p in paragraphs
      if p
      and p != name
      and p != pickup
      and not p.startswith("ItemID:")
      and not p.startswith("Quality:")
      and not p.startswith("Type:")
      and not p.startswith("Item Pool:")
      and not p.startswith("Recharge Time:")
      and not p.startswith("Unlock this")
    ]
    out.append({
      "id": f"{kind}-{game_id:03d}",
      "kind": kind,
      "gameId": game_id,
      "nameZh": name,
      "nameEn": name,
      "pickup": pickup,
      "description": pickup,
      "effects": effects,
      "type": prefixed(paragraphs, "Type:"),
      "pools": split_list(prefixed(paragraphs, "Item Pool:")),
      "tags": split_tags(first_p(body, "tags")),
      "iconClasses": classes,
    })
  return out


def attr_value(attrs: str, name: str) -> str:
  match = re.search(rf"{re.escape(name)}=\"([^\"]*)\"", attrs)
  return match.group(1) if match else ""


def kind_from_classes(classes: list[str]) -> str | None:
  if "trinket" in classes or any(cls.startswith("trinket") or "trink" in cls for cls in classes):
    return "trinket"
  if any("card" in cls for cls in classes) or any("rune" in cls for cls in classes):
    return "card"
  if "item" in classes:
    return "item"
  return None


def icon_spec_for(
  obj: dict,
  tboi_icons: dict[str, IconSpec],
  icecat_icons: dict[str, IconSpec],
  icecat_name_icons: dict[str, str],
) -> IconSpec | None:
  classes = obj["iconClasses"]
  game_id = obj["gameId"]
  if obj["kind"] == "item":
    rep_class = next((cls for cls in classes if re.fullmatch(r"rep\d+", cls)), None)
    if rep_class and rep_class in tboi_icons:
      return tboi_icons[rep_class]
    old_class = next(
      (cls for cls in classes if re.fullmatch(r"(?:re-itm|abn-itm|apn-itm)\d+", cls)),
      None,
    )
    if old_class and old_class in tboi_icons:
      return tboi_icons[old_class]
  if obj["kind"] == "trinket":
    junxx = next((cls for cls in classes if re.fullmatch(r"rep-junxx\d+", cls)), None)
    if junxx and junxx in tboi_icons:
      return tboi_icons[junxx]
    icecat_class = icecat_name_icons.get(name_key(obj["nameEn"]))
    if icecat_class and icecat_class in icecat_icons:
      return icecat_icons[icecat_class]
  if obj["kind"] == "card":
    old_card = next((cls for cls in classes if re.fullmatch(r"r-card\d+", cls)), None)
    if old_card and old_card in icecat_icons:
      return icecat_icons[old_card]
    repc = next((cls for cls in classes if re.fullmatch(r"repc\d+", cls)), None)
    if repc and repc in tboi_icons:
      return tboi_icons[repc]
  return None


def parse_tboi_icons(css: str) -> dict[str, IconSpec]:
  icons: dict[str, IconSpec] = {}
  patterns = (
    (r"\.reb-itm-new\.(re-itm\d+)(?:$|[:.\s#])", "rebirth", 50, 50),
    (r"\.ab-itm-new\.(abn-itm\d+)(?:$|[:.\s#])", "ab", 50, 50),
    (r"\.ap-itm-new\.(apn-itm\d+)(?:$|[:.\s#])", "ap", 50, 50),
    (r"\.rep-item\.(rep\d+)(?:$|[:.\s#])", "rep2", 50, 50),
    (r"\.rep-trink\.(rep-junxx\d+)(?:$|[:.\s#])", "rep_trinkets", 50, 50),
    (r"\.rep-card\.(repc\d+)(?:$|[:.\s#])", "rep_cards", 43, 50),
  )
  for selector_text, declaration in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
    for selector in selector_text.split(","):
      selector = selector.strip()
      for pattern, sprite, default_width, default_height in patterns:
        match = re.search(pattern, selector)
        if not match:
          continue
        x, y = extract_background_position(declaration)
        icons[match.group(1)] = IconSpec(
          sprite,
          x=x,
          y=y,
          width=extract_px(declaration, "width") or default_width,
          height=extract_px(declaration, "height") or default_height,
        )
        break
  return icons


def remove_unreferenced_icons(objects: list[dict]) -> None:
  referenced = {
    Path(obj["iconPath"]).name
    for obj in objects
    if obj.get("iconPath", "").startswith("/objects/icons/")
  }
  for icon in ICON_DIR.glob("*.png"):
    if icon.name not in referenced:
      icon.unlink()


def parse_icecat_icons(css: str) -> dict[str, IconSpec]:
  icons: dict[str, IconSpec] = {}
  for selector_text, declaration in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
    for selector in selector_text.split(","):
      selector = selector.strip()
      trinket = re.search(r"\.rebirth-trinket\.(r-junxx\d+)(?:$|[:.\s#])", selector)
      card = re.search(r"\.rebirth-card\.(r-card\d+)(?:$|[:.\s#])", selector)
      match = trinket or card
      if not match:
        continue
      x, y = extract_background_position(declaration)
      width = extract_px(declaration, "width") or 50
      height = extract_px(declaration, "height") or 50
      icons[match.group(1)] = IconSpec(
        "icecat_trinkets" if trinket else "icecat_cards",
        x=x,
        y=y,
        width=width,
        height=height,
      )
  return icons


def parse_icecat_name_icons(text: str) -> dict[str, str]:
  out: dict[str, str] = {}
  for class_attr, body in re.findall(
    r'class="([^"]*(?:r-junxx\d+|r-card\d+)[^"]*)".*?<span>(.*?)</span>',
    text,
    flags=re.S,
  ):
    icon_class = next((part for part in class_attr.split() if re.fullmatch(r"(?:r-junxx|r-card)\d+", part)), "")
    if not icon_class:
      continue
    plain = normalize(strip_tags(body))
    for candidate in re.findall(r"([A-Z0-9?][A-Za-z0-9?'!:\- ]{1,60})", plain):
      key = name_key(candidate)
      if key and len(key) > 2:
        out.setdefault(key, icon_class)
  return out


def parse_icecat_localizations(text: str) -> dict[tuple[str, str], dict]:
  localizations: dict[tuple[str, str], dict] = {}
  for _, body in re.findall(r'<li class="textbox[^>]*"([^>]*)>(.*?)</li>', text, flags=re.S):
    icon_match = re.search(r'<div[^>]+class=["\']([^"\']+)["\']', body)
    if not icon_match:
      continue
    classes = icon_match.group(1).split()
    if any("trinket" in class_name for class_name in classes):
      kind = "trinket"
    elif any("card" in class_name for class_name in classes):
      kind = "card"
    elif any("item" in class_name for class_name in classes):
      kind = "item"
    else:
      continue

    titles_zh = p_texts(body, "item-title")
    titles_en = p_texts(body, "item-title2")
    if not titles_zh:
      continue
    name_zh = titles_zh[0]
    name_en = titles_en[0] if titles_en else (titles_zh[1] if len(titles_zh) > 1 else "")
    if not name_en or not re.search(r"[A-Za-z]", name_en):
      continue
    pickup = first_p(body, "pickup")
    effects = []
    for attrs, content in re.findall(r"<p([^>]*)>(.*?)</p>", body, flags=re.S):
      class_match = re.search(r'class=["\']([^"\']+)["\']', attrs)
      paragraph_classes = set(class_match.group(1).split()) if class_match else set()
      if paragraph_classes.intersection({"item-title", "item-title2", "pickup", "tags", "r-itemid"}):
        continue
      value = normalize(strip_tags(content)).removeprefix("•").strip()
      if value and re.search(r"[\u4e00-\u9fff]", value):
        effects.append(value)
    localizations[(kind, name_key(name_en))] = {
      "nameZh": name_zh,
      "pickup": pickup,
      "description": pickup,
      "effects": effects,
    }
  return localizations


def p_texts(body: str, class_name: str) -> list[str]:
  return [
    normalize(strip_tags(content))
    for content in re.findall(
      rf'<p class=["\']{re.escape(class_name)}["\']>(.*?)</p>',
      body,
      flags=re.S,
    )
  ]


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


def name_key(value: str) -> str:
  value = normalize(value).lower().replace("’", "'")
  value = re.sub(r"^rune of ", "", value)
  value = value.replace("monkey paw", "monkey's paw")
  return re.sub(r"[^a-z0-9?]+", "", value)


def rebirth_family(game_id: int) -> str:
  if game_id <= 346:
    return "rebirth"
  if game_id <= 438:
    return "ab"
  if game_id <= 552:
    return "ap"
  return "rebirth"


def rebirth_index(game_id: int) -> int:
  if game_id <= 346:
    return game_id - 1
  if game_id <= 438:
    return game_id - 347
  if game_id <= 552:
    return game_id - 439
  return game_id - 1


def rep_index(game_id: int) -> int:
  return game_id - 553


def first_p(body: str, class_name: str) -> str:
  match = re.search(rf"<p class=\"{re.escape(class_name)}\">(.*?)</p>", body, flags=re.S)
  return normalize(strip_tags(match.group(1))) if match else ""


def extract_plain_paragraphs(body: str) -> list[str]:
  return [normalize(strip_tags(match)) for match in re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S)]


def strip_tags(value: str) -> str:
  return html.unescape(re.sub(r"<[^>]*>", "", value or ""))


def normalize(value: str) -> str:
  return re.sub(r"\s+", " ", value).strip().strip('"')


def prefixed(values: list[str], prefix: str) -> str:
  for value in values:
    if value.startswith(prefix):
      return value[len(prefix):].strip()
  return ""


def split_list(value: str) -> list[str]:
  return [part.strip() for part in value.split(",") if part.strip()]


def split_tags(value: str) -> list[str]:
  value = normalize(strip_tags(value)).lstrip("*").strip()
  return split_list(value)


def dedupe_objects(objects: list[dict]) -> list[dict]:
  seen = set()
  out = []
  for obj in objects:
    key = (obj["kind"], obj["gameId"])
    if key in seen:
      continue
    seen.add(key)
    out.append(obj)
  return out


def merge_icecat_items(objects: list[dict]) -> list[dict]:
  if not ICECAT_ITEMS_JSON.exists():
    return objects
  icecat_items = {
    item["id"]: item
    for item in json.loads(ICECAT_ITEMS_JSON.read_text(encoding="utf-8"))
  }
  merged = []
  for obj in objects:
    icecat = icecat_items.get(obj["id"])
    if obj["kind"] == "item" and icecat:
      obj = {
        **obj,
        "nameZh": icecat.get("nameZh", obj["nameZh"]),
        "nameEn": icecat.get("nameEn", obj["nameEn"]),
        "pickup": icecat.get("pickup", obj["pickup"]),
        "description": icecat.get("description", obj["description"]),
        "effects": icecat.get("effects", obj["effects"]),
        "type": icecat.get("type", obj["type"]),
        "pools": icecat.get("pools", obj["pools"]),
        "tags": icecat.get("tags", obj["tags"]),
        "iconPath": icecat.get("iconPath", obj["iconPath"]),
        "sourceName": f"{icecat.get('sourceName', 'IcaCat 以撒图鉴')} + Platinum God",
      }
    merged.append(obj)
  return merged


def merge_icecat_localizations(
  objects: list[dict],
  localizations: dict[tuple[str, str], dict],
) -> list[dict]:
  merged = []
  for obj in objects:
    localization = localizations.get((obj["kind"], name_key(obj["nameEn"])))
    if localization:
      obj = {
        **obj,
        "nameZh": localization["nameZh"] or obj["nameZh"],
        "pickup": localization["pickup"] or obj["pickup"],
        "description": localization["description"] or obj["description"],
        "effects": localization["effects"] or obj["effects"],
        "sourceName": f"IcaCat 以撒图鉴 + {obj['sourceName']}",
      }
    merged.append(obj)
  return merged


def merge_generated_translations(objects: list[dict]) -> list[dict]:
  return merge_translation_file(objects, GENERATED_TRANSLATIONS_JSON)


def merge_translation_file(objects: list[dict], path: Path) -> list[dict]:
  if not path.exists():
    return objects
  translations = json.loads(path.read_text(encoding="utf-8"))
  merged = []
  for obj in objects:
    translated = translations.get(obj["id"])
    if translated:
      obj = {
        **obj,
        "nameZh": translated.get("nameZh") or obj["nameZh"],
        "pickup": translated.get("pickup") or obj["pickup"],
        "description": translated.get("description") or obj["description"],
        "effects": translated.get("effects") or obj["effects"],
      }
    merged.append(obj)
  return merged


def kind_counts(objects: list[dict]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for obj in objects:
    counts[obj["kind"]] = counts.get(obj["kind"], 0) + 1
  return counts


def kind_order(kind: str) -> int:
  return {"item": 0, "trinket": 1, "card": 2}.get(kind, 99)


if __name__ == "__main__":
  raise SystemExit(main())
