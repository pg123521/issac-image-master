#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "detection" / "room-collectible-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "room-collectible-detector-v1.pt"
DEFAULT_RUNS = PROJECT_ROOT / "data" / "detection" / "runs"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolo26n.pt"


def main() -> int:
  parser = argparse.ArgumentParser(description="Train a one-class detector for collectible objects in Isaac rooms.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  train = subparsers.add_parser("train")
  train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
  train.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
  train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  train.add_argument("--epochs", type=int, default=24)
  train.add_argument("--batch-size", type=int, default=4)
  train.add_argument("--image-size", type=int, default=1024)
  train.add_argument("--device", default=best_device())
  train.add_argument("--run-name", default="room-collectible-v1")

  evaluate = subparsers.add_parser("evaluate")
  evaluate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
  evaluate.add_argument("--weights", type=Path, default=DEFAULT_OUTPUT)
  evaluate.add_argument("--image-size", type=int, default=1024)
  evaluate.add_argument("--batch-size", type=int, default=4)
  evaluate.add_argument("--device", default=best_device())

  args = parser.parse_args()
  if args.command == "train":
    train_detector(args)
  else:
    report = evaluate_detector(args.weights, args.dataset, args.image_size, args.batch_size, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))
  return 0


def best_device() -> str:
  if torch.backends.mps.is_available():
    return "mps"
  if torch.cuda.is_available():
    return "0"
  return "cpu"


def train_detector(args: argparse.Namespace) -> None:
  model = YOLO(str(args.base_model))
  results = model.train(
    data=str(args.dataset / "dataset.yaml"),
    epochs=args.epochs,
    imgsz=args.image_size,
    batch=args.batch_size,
    device=args.device,
    workers=0,
    project=str(DEFAULT_RUNS),
    name=args.run_name,
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=1e-3,
    weight_decay=5e-4,
    cos_lr=True,
    patience=max(8, args.epochs // 3),
    close_mosaic=min(5, max(1, args.epochs // 4)),
    mosaic=0.45,
    mixup=0.0,
    degrees=3.0,
    translate=0.08,
    scale=0.25,
    fliplr=0.0,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.20,
    plots=True,
    verbose=True,
  )
  best = Path(results.save_dir) / "weights" / "best.pt"
  if not best.exists():
    raise FileNotFoundError(f"training did not produce {best}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(best, args.output)
  print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
  report = evaluate_detector(args.output, args.dataset, args.image_size, args.batch_size, args.device)
  print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def evaluate_detector(
  weights: Path,
  dataset: Path,
  image_size: int,
  batch_size: int,
  device: str,
) -> dict[str, Any]:
  model = YOLO(str(weights))
  report = {
    "weights": str(weights),
    "device": device,
    "imageSize": image_size,
    "synthetic": validation_metrics(model, dataset / "dataset.yaml", image_size, batch_size, device),
    "real": validation_metrics(model, dataset / "real-validation.yaml", image_size, batch_size, device),
  }
  return report


def validation_metrics(
  model: YOLO,
  data: Path,
  image_size: int,
  batch_size: int,
  device: str,
) -> dict[str, float]:
  metrics = model.val(
    data=str(data),
    imgsz=image_size,
    batch=batch_size,
    device=device,
    workers=0,
    plots=False,
    verbose=False,
  )
  return {
    "precision": round(float(metrics.box.mp), 6),
    "recall": round(float(metrics.box.mr), 6),
    "mAP50": round(float(metrics.box.map50), 6),
    "mAP50-95": round(float(metrics.box.map), 6),
  }


if __name__ == "__main__":
  raise SystemExit(main())
