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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "item-classifier-v1"


def main() -> int:
  parser = argparse.ArgumentParser(description="Run Isaac item model inference on a crop image.")
  parser.add_argument("image", nargs="?", type=Path, help="Crop image path for one-shot CLI inference.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--top-k", type=int, default=10)
  parser.add_argument("--serve", action="store_true", help="Start local HTTP inference server.")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()

  predictor = Predictor(args.model_dir)
  if args.serve:
    serve(predictor, args.host, args.port)
    return 0

  if not args.image:
    parser.error("image path is required unless --serve is used")
  image = Image.open(args.image).convert("RGB")
  print(json.dumps(predictor.predict(image, args.top_k), ensure_ascii=False, indent=2))
  return 0


class Predictor:
  def __init__(self, model_dir: Path) -> None:
    self.model_dir = model_dir
    self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    self.labels = self.metadata["labels"]
    self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    self.model = torch.jit.load(str(model_dir / "model.torchscript.pt"), map_location=self.device)
    self.model.eval()

  @torch.no_grad()
  def predict(self, image: Image.Image, top_k: int) -> dict[str, Any]:
    tensor = image_to_tensor(image).unsqueeze(0).to(self.device)
    logits = self.model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = probs.topk(k=min(top_k, len(self.labels)))
    matches = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist(), strict=True):
      label = self.labels[index]
      matches.append({
        "itemId": label["item_id"],
        "gameId": label["game_id"],
        "nameZh": label["name_zh"],
        "nameEn": label["name_en"],
        "iconPath": label["icon_path"],
        "confidence": round(float(value), 6),
      })
    return {
      "device": str(self.device),
      "topK": matches,
    }


def image_to_tensor(image: Image.Image) -> torch.Tensor:
  if image.size != (96, 96):
    image = image.resize((96, 96), Image.Resampling.BILINEAR)
  data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
  data = data.view(96, 96, 3).permute(2, 0, 1).float().div(255.0)
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  return (data - mean) / std


def serve(predictor: Predictor, host: str, port: int) -> None:
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
        "device": str(predictor.device),
        "classes": len(predictor.labels),
      })

    def do_POST(self) -> None:
      if self.path != "/predict":
        self.send_json({"error": "not found"}, status=404)
        return
      try:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image = decode_image(payload["image"])
        top_k = int(payload.get("topK", 10))
        self.send_json(predictor.predict(image, top_k))
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
  print(f"Isaac item inference server on http://{host}:{port} ({predictor.device})", flush=True)
  server.serve_forever()


def decode_image(value: str) -> Image.Image:
  if value.startswith("data:"):
    value = value.split(",", 1)[1]
  return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


if __name__ == "__main__":
  raise SystemExit(main())
