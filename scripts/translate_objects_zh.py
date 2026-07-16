#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PATH = ROOT / "data" / "objects.en.json"
CACHE_PATH = ROOT / "data" / "translations" / "objects.qwen3.5.zh-CN.json"
API_URL = "http://127.0.0.1:11434/api/chat"


def main() -> int:
  parser = argparse.ArgumentParser(description="Translate remaining Isaac encyclopedia text with local Ollama.")
  parser.add_argument("--model", default="qwen3.5:latest")
  parser.add_argument("--batch-size", type=int, default=12)
  args = parser.parse_args()

  objects = json.loads(OBJECTS_PATH.read_text(encoding="utf-8"))
  CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
  cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
  pending = [obj for obj in objects if obj["id"] not in cache and needs_translation(obj)]
  total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
  print(f"[翻译] objects={len(objects)} pending={len(pending)} cached={len(cache)} batches={total_batches}", flush=True)

  for batch_index, start in enumerate(range(0, len(pending), args.batch_size), 1):
    batch = pending[start:start + args.batch_size]
    print(f"[翻译 {batch_index}/{total_batches}] {batch[0]['id']} ... {batch[-1]['id']}", flush=True)
    translated = translate_batch(batch, args.model)
    by_id = {entry["id"]: entry for entry in translated}
    missing = [obj["id"] for obj in batch if obj["id"] not in by_id]
    if missing:
      raise RuntimeError(f"模型漏掉了条目：{', '.join(missing)}")
    for obj in batch:
      cache[obj["id"]] = normalize_translation(obj, by_id[obj["id"]])
    write_json(CACHE_PATH, cache)
    print(f"[翻译 {batch_index}/{total_batches}] saved={len(cache)}", flush=True)

  apply_cache(objects, cache)
  write_json(OBJECTS_PATH, objects)
  print(f"[完成] cache={CACHE_PATH} encyclopedia={OBJECTS_PATH}", flush=True)
  return 0


def needs_translation(obj: dict) -> bool:
  values = [obj.get("nameZh", ""), obj.get("pickup", ""), obj.get("description", "")]
  return any(value and has_latin(value) and not has_chinese(value) for value in values)


def translate_batch(batch: list[dict], model: str) -> list[dict]:
  payload_objects = []
  for obj in batch:
    payload_objects.append({
      "id": obj["id"],
      "kind": obj["kind"],
      "nameEn": obj["nameEn"],
      "nameZh": obj["nameZh"] if should_translate(obj["nameZh"]) else "",
      "pickup": obj["pickup"] if should_translate(obj["pickup"]) else "",
      "description": obj["description"] if should_translate(obj["description"]) else "",
      "effects": [],
    })
  prompt = f"""
你是《以撒的结合：重生/忏悔》的简体中文本地化编辑。把下面 JSON 中非空的英文文本翻译为准确、简洁、自然的简体中文。
要求：
1. 优先采用玩家常用译名和游戏术语，例如 tears=射速/眼泪（按语境）、damage=伤害、room=房间、floor=楼层、pickup=掉落物、trinket=饰品、card=卡牌。
2. 名称字段只返回中文名称；刻意使用符号或型号的名称（如 D6、1UP、20/20）可原样保留。
3. 数值、百分比、角色名、道具机制和效果条件不得遗漏或添加。
4. effects 保持空数组，不生成效果明细。
5. 只返回符合给定结构的 JSON，不解释。

输入：
{json.dumps(payload_objects, ensure_ascii=False)}
"""
  schema = {
    "type": "object",
    "properties": {
      "translations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "nameZh": {"type": "string"},
            "pickup": {"type": "string"},
            "description": {"type": "string"},
            "effects": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["index", "text"],
              },
            },
          },
          "required": ["id", "nameZh", "pickup", "description", "effects"],
        },
      },
    },
    "required": ["translations"],
  }
  request_data = json.dumps({
    "model": model,
    "stream": False,
    "think": False,
    "format": schema,
    "messages": [{"role": "user", "content": prompt}],
    "options": {"temperature": 0, "num_ctx": 16384},
  }).encode()
  request = urllib.request.Request(API_URL, data=request_data, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(request, timeout=900) as response:
    result = json.loads(response.read())
  content = json.loads(result["message"]["content"])
  if isinstance(content, list):
    return content
  return content["translations"]


def normalize_translation(source: dict, translated: dict) -> dict:
  effects = list(source["effects"])
  for entry in translated.get("effects", []):
    index = int(entry["index"])
    if 0 <= index < len(effects) and entry.get("text"):
      effects[index] = entry["text"].strip()
  return {
    "nameZh": translated.get("nameZh", "").strip() or source["nameZh"],
    "pickup": translated.get("pickup", "").strip() or source["pickup"],
    "description": translated.get("description", "").strip() or source["description"],
    "effects": effects,
  }


def apply_cache(objects: list[dict], cache: dict) -> None:
  for obj in objects:
    translated = cache.get(obj["id"])
    if not translated:
      continue
    for key in ("nameZh", "pickup", "description", "effects"):
      if translated.get(key):
        obj[key] = translated[key]


def should_translate(value: str) -> bool:
  return bool(value and has_latin(value) and not has_chinese(value))


def has_chinese(value: str) -> bool:
  return bool(re.search(r"[\u4e00-\u9fff]", value or ""))


def has_latin(value: str) -> bool:
  return bool(re.search(r"[A-Za-z]", value or ""))


def write_json(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  raise SystemExit(main())
