#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "room-collectible-detector-v1.pt"


def main() -> int:
  parser = argparse.ArgumentParser(description="Local Isaac room collectible detection service.")
  parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8767)
  parser.add_argument("--image-size", type=int, default=1024)
  parser.add_argument("--confidence", type=float, default=0.18)
  args = parser.parse_args()
  detector = RoomCollectibleDetector(args.weights, args.image_size, args.confidence)
  serve(detector, args.host, args.port)
  return 0


class RoomCollectibleDetector:
  def __init__(self, weights: Path, image_size: int = 1024, confidence: float = 0.18) -> None:
    self.weights = weights
    self.image_size = image_size
    self.confidence = confidence
    self.device = "mps" if torch.backends.mps.is_available() else "0" if torch.cuda.is_available() else "cpu"
    self.model = YOLO(str(weights))

  def detect(self, image: Image.Image) -> dict[str, Any]:
    detections: list[dict[str, float]] = []
    tiles = detection_tiles(image.width, image.height)
    for left, top, right, bottom in tiles:
      tile = image.crop((left, top, right, bottom))
      result = self.model.predict(
        tile,
        imgsz=self.image_size,
        conf=self.confidence,
        iou=0.55,
        device=self.device,
        max_det=40,
        verbose=False,
      )[0]
      if result.boxes is None:
        continue
      for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True):
        x1, y1, x2, y2 = xyxy
        detections.append({
          "x1": max(0.0, left + x1),
          "y1": max(0.0, top + y1),
          "x2": min(float(image.width), left + x2),
          "y2": min(float(image.height), top + y2),
          "score": float(confidence),
        })

    kept = non_maximum_suppression(detections, 0.42)
    boxes = [{
      "x": round(box["x1"], 2),
      "y": round(box["y1"], 2),
      "w": round(box["x2"] - box["x1"], 2),
      "h": round(box["y2"] - box["y1"], 2),
      "score": round(box["score"], 6),
    } for box in kept]
    return {
      "model": "YOLO26n room_collectible",
      "device": self.device,
      "imageWidth": image.width,
      "imageHeight": image.height,
      "tiles": len(tiles),
      "boxes": boxes,
    }


def detection_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
  tiles = [(0, 0, width, height)]
  if width <= height * 1.35:
    return tiles
  side = height
  travel = width - side
  tile_count = max(2, min(4, round(width / height) + 1))
  for index in range(tile_count):
    left = round(travel * index / max(1, tile_count - 1))
    tile = (left, 0, left + side, height)
    if tile not in tiles:
      tiles.append(tile)
  return tiles


def non_maximum_suppression(boxes: list[dict[str, float]], threshold: float) -> list[dict[str, float]]:
  ordered = sorted(boxes, key=lambda box: box["score"], reverse=True)
  kept: list[dict[str, float]] = []
  for candidate in ordered:
    if all(intersection_over_union(candidate, existing) < threshold for existing in kept):
      kept.append(candidate)
  return kept[:20]


def intersection_over_union(a: dict[str, float], b: dict[str, float]) -> float:
  left = max(a["x1"], b["x1"])
  top = max(a["y1"], b["y1"])
  right = min(a["x2"], b["x2"])
  bottom = min(a["y2"], b["y2"])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
  area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
  return intersection / max(1e-9, area_a + area_b - intersection)


def serve(detector: RoomCollectibleDetector, host: str, port: int) -> None:
  class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
      self.send_response(204)
      self.send_cors_headers()
      self.end_headers()

    def do_GET(self) -> None:
      if self.path != "/health":
        self.send_json({"error": "not found"}, status=404)
        return
      self.send_json({
        "ok": True,
        "model": "YOLO26n room_collectible",
        "device": detector.device,
        "weights": detector.weights.name,
        "imageSize": detector.image_size,
      })

    def do_POST(self) -> None:
      if self.path != "/detect":
        self.send_json({"error": "not found"}, status=404)
        return
      try:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image = decode_image(payload["image"])
        self.send_json(detector.detect(image))
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
  print(f"Room detector on http://{host}:{port} ({detector.device})", flush=True)
  server.serve_forever()


def decode_image(value: str) -> Image.Image:
  if value.startswith("data:"):
    value = value.split(",", 1)[1]
  return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


if __name__ == "__main__":
  raise SystemExit(main())
