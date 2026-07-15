#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import open_clip
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_JSON = PROJECT_ROOT / "data" / "objects.en.json"
ICON_ROOT = PROJECT_ROOT / "public"
DEFAULT_INDEX = PROJECT_ROOT / "models" / "mobileclip-object-icon-index-v1.pt"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "training" / "synthetic-v3"
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"


def main() -> int:
  parser = argparse.ArgumentParser(description="MobileCLIP embedding retrieval for Isaac items.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  build = subparsers.add_parser("build-index")
  build.add_argument("--output", type=Path, default=DEFAULT_INDEX)
  build.add_argument("--objects-json", type=Path, default=OBJECTS_JSON)
  build.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
  build.add_argument("--synthetic-per-item", type=int, default=12)
  build.add_argument("--batch-size", type=int, default=96)

  query = subparsers.add_parser("query")
  query.add_argument("image", type=Path)
  query.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  query.add_argument("--top-k", type=int, default=10)

  serve_cmd = subparsers.add_parser("serve")
  serve_cmd.add_argument("--index", type=Path, default=DEFAULT_INDEX)
  serve_cmd.add_argument("--host", default="127.0.0.1")
  serve_cmd.add_argument("--port", type=int, default=8766)
  serve_cmd.add_argument("--top-k", type=int, default=10)

  args = parser.parse_args()
  if args.command == "build-index":
    encoder = MobileClipEncoder()
    build_index(encoder, args.objects_json, args.output, args.dataset, args.synthetic_per_item, args.batch_size)
  elif args.command == "query":
    searcher = Searcher(args.index)
    image = Image.open(args.image).convert("RGB")
    print(json.dumps(searcher.search(image, args.top_k), ensure_ascii=False, indent=2))
  elif args.command == "serve":
    searcher = Searcher(args.index)
    serve(searcher, args.host, args.port, args.top_k)
  return 0


class MobileClipEncoder:
  def __init__(self) -> None:
    self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    self.model = model.to(self.device).eval()
    self.preprocess = preprocess

  @torch.no_grad()
  def encode(self, images: list[Image.Image]) -> torch.Tensor:
    batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
    features = self.model.encode_image(batch)
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().float()


def build_index(
  encoder: MobileClipEncoder,
  objects_json: Path,
  output: Path,
  dataset: Path,
  synthetic_per_item: int,
  batch_size: int,
) -> None:
  items = json.loads(objects_json.read_text(encoding="utf-8"))
  labels = [{
    "item_id": item["id"],
    "kind": item.get("kind", "item"),
    "game_id": item["gameId"],
    "name_zh": item["nameZh"],
    "name_en": item["nameEn"],
    "icon_path": item["iconPath"],
  } for item in items]

  examples: list[tuple[int, Path, str]] = []
  for index, item in enumerate(items):
    examples.append((index, ICON_ROOT / item["iconPath"].lstrip("/"), "icon"))

  if dataset.exists() and synthetic_per_item > 0:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with (dataset / "manifest.csv").open(encoding="utf-8") as handle:
      for row in csv.DictReader(handle):
        grouped[row["item_id"]].append(row)
    item_to_index = {label["item_id"]: index for index, label in enumerate(labels)}
    for item_id, rows in grouped.items():
      if item_id not in item_to_index:
        continue
      rows = sorted(rows, key=lambda row: (row.get("style", ""), row["path"]))
      hard = [row for row in rows if "hard" in row.get("style", "")]
      closeup = [row for row in rows if "closeup" in row.get("style", "")]
      full = [row for row in rows if row.get("crop_mode") == "full"]
      picked = dedupe_rows([*hard[: synthetic_per_item // 2], *closeup[: synthetic_per_item // 3], *full[: synthetic_per_item], *rows])[:synthetic_per_item]
      for row in picked:
        examples.append((item_to_index[item_id], dataset / row["path"], row.get("style", "synthetic")))

  vectors: list[torch.Tensor] = []
  index_to_item: list[int] = []
  sources: list[str] = []
  for start in range(0, len(examples), batch_size):
    chunk = examples[start : start + batch_size]
    images = [Image.open(path).convert("RGB") for _, path, _ in chunk]
    vectors.append(encoder.encode(images))
    index_to_item.extend(item_index for item_index, _, _ in chunk)
    sources.extend(source for _, _, source in chunk)
    print(f"encoded {min(start + len(chunk), len(examples))}/{len(examples)}", flush=True)

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model_name": MODEL_NAME,
    "pretrained": PRETRAINED,
    "vectors": torch.cat(vectors, dim=0),
    "index_to_item": torch.tensor(index_to_item, dtype=torch.long),
    "sources": sources,
    "labels": labels,
  }, output)
  print(f"wrote {output}", flush=True)


def dedupe_rows(rows: list[dict]) -> list[dict]:
  seen: set[str] = set()
  out: list[dict] = []
  for row in rows:
    path = row["path"]
    if path in seen:
      continue
    seen.add(path)
    out.append(row)
  return out


class Searcher:
  def __init__(self, index_path: Path) -> None:
    self.encoder = MobileClipEncoder()
    payload = torch.load(index_path, map_location="cpu")
    self.vectors = payload["vectors"].float()
    self.index_to_item = payload["index_to_item"].long()
    self.labels = payload["labels"]
    self.sources = payload["sources"]

  def search(self, image: Image.Image, top_k: int) -> dict[str, Any]:
    query = self.encoder.encode([image])[0]
    scores = self.vectors @ query
    raw_k = min(max(top_k * 8, 40), scores.numel())
    values, indices = scores.topk(raw_k)
    best_by_item: dict[int, tuple[float, int]] = {}
    for value, vector_index in zip(values.tolist(), indices.tolist(), strict=True):
      item_index = int(self.index_to_item[vector_index])
      if item_index not in best_by_item or value > best_by_item[item_index][0]:
        best_by_item[item_index] = (float(value), vector_index)
      if len(best_by_item) >= top_k:
        break
    matches = []
    for item_index, (score, vector_index) in sorted(best_by_item.items(), key=lambda entry: entry[1][0], reverse=True)[:top_k]:
      label = self.labels[item_index]
      matches.append({
        "itemId": label["item_id"],
        "kind": label.get("kind", "item"),
        "gameId": label["game_id"],
        "nameZh": label["name_zh"],
        "nameEn": label["name_en"],
        "iconPath": label["icon_path"],
        "score": round(score, 6),
        "source": self.sources[vector_index],
      })
    return {
      "device": str(self.encoder.device),
      "model": MODEL_NAME,
      "topK": matches,
    }


def serve(searcher: Searcher, host: str, port: int, top_k: int) -> None:
  class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
      self.send_response(204)
      self.send_cors_headers()
      self.end_headers()

    def do_GET(self) -> None:
      if self.path != "/health":
        self.send_json({"error": "not found"}, status=404)
        return
      self.send_json({"ok": True, "device": str(searcher.encoder.device), "model": MODEL_NAME, "vectors": len(searcher.vectors)})

    def do_POST(self) -> None:
      if self.path != "/predict":
        self.send_json({"error": "not found"}, status=404)
        return
      try:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image = decode_image(payload["image"])
        self.send_json(searcher.search(image, int(payload.get("topK", top_k))))
      except Exception as exc:
        self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
      body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
      self.send_response(status)
      self.send_cors_headers()
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def send_cors_headers(self) -> None:
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
      self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
      print(f"{self.address_string()} - {format % args}")

  server = ThreadingHTTPServer((host, port), Handler)
  print(f"MobileCLIP search server on http://{host}:{port} ({searcher.encoder.device})", flush=True)
  server.serve_forever()


def decode_image(value: str) -> Image.Image:
  if value.startswith("data:"):
    value = value.split(",", 1)[1]
  return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


if __name__ == "__main__":
  raise SystemExit(main())
