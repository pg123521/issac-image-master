#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "training" / "synthetic-v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "item-classifier-v1"


def main() -> int:
  parser = argparse.ArgumentParser(description="Train an offline Isaac item top-k classifier.")
  parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--epochs", type=int, default=18)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=2.5e-3)
  parser.add_argument("--weight-decay", type=float, default=2e-4)
  parser.add_argument("--val-ratio", type=float, default=0.16)
  parser.add_argument("--seed", type=int, default=20260715)
  parser.add_argument("--workers", type=int, default=0)
  parser.add_argument("--width", type=int, default=64)
  parser.add_argument("--limit-per-class", type=int, default=0, help="Debug only: cap samples per item.")
  parser.add_argument("--log-every", type=int, default=25, help="Print training progress every N batches; 0 disables batch logs.")
  parser.add_argument("--resume", action="store_true", help="Resume from output/last.pt when it exists.")
  args = parser.parse_args()

  set_seed(args.seed)
  args.output.mkdir(parents=True, exist_ok=True)

  records, labels = load_records(args.dataset, args.limit_per_class)
  train_records, val_records = stratified_split(records, args.val_ratio, args.seed)
  num_classes = len(labels)
  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

  train_ds = ItemDataset(args.dataset, train_records, train=True)
  val_ds = ItemDataset(args.dataset, val_records, train=False)
  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

  model = IsaacItemNet(num_classes=num_classes, width=args.width).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
  start_epoch = 1
  best_top5 = -1.0

  meta = {
    "dataset": str(args.dataset),
    "num_classes": num_classes,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "image_size": 96,
    "width": args.width,
    "labels": labels,
  }
  (args.output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  last_checkpoint = args.output / "last.pt"
  if args.resume and last_checkpoint.exists():
    checkpoint = torch.load(last_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_top5 = float(checkpoint.get("metrics", {}).get("top5", -1.0))
    for _ in range(start_epoch - 1):
      scheduler.step()

  print(f"classes={num_classes} train={len(train_ds)} val={len(val_ds)} device={device} params={count_params(model):,}", flush=True)
  if start_epoch > 1:
    print(f"resuming from epoch {start_epoch}", flush=True)

  for epoch in range(start_epoch, args.epochs + 1):
    started = time.time()
    train_loss, train_top1 = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, log_every=args.log_every)
    metrics = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    print(
      f"epoch {epoch:02d}/{args.epochs} "
      f"loss={train_loss:.4f} top1={train_top1:.3f} "
      f"val_loss={metrics.loss:.4f} val_top1={metrics.top1:.3f} "
      f"val_top5={metrics.top5:.3f} val_top10={metrics.top10:.3f} "
      f"time={time.time() - started:.1f}s"
    , flush=True)

    checkpoint = {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "epoch": epoch,
      "metrics": metrics.__dict__,
      "metadata": meta,
    }
    torch.save(checkpoint, args.output / "last.pt")
    if metrics.top5 > best_top5:
      best_top5 = metrics.top5
      torch.save(checkpoint, args.output / "best.pt")

  scripted = torch.jit.script(model.cpu().eval())
  scripted.save(str(args.output / "model.torchscript.pt"))
  print(f"wrote {args.output}", flush=True)
  return 0


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)


def load_records(dataset: Path, limit_per_class: int) -> tuple[list[dict], list[dict]]:
  labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
  id_to_index = {label["item_id"]: index for index, label in enumerate(labels)}
  rows = list(csv.DictReader((dataset / "manifest.csv").open(encoding="utf-8")))
  per_class_seen: dict[str, int] = defaultdict(int)
  records: list[dict] = []
  for row in rows:
    item_id = row["item_id"]
    if limit_per_class and per_class_seen[item_id] >= limit_per_class:
      continue
    per_class_seen[item_id] += 1
    row["label"] = id_to_index[item_id]
    records.append(row)
  return records, labels


def stratified_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
  rng = random.Random(seed)
  grouped: dict[int, list[dict]] = defaultdict(list)
  for record in records:
    grouped[record["label"]].append(record)

  train: list[dict] = []
  val: list[dict] = []
  for group in grouped.values():
    rng.shuffle(group)
    val_count = max(1, round(len(group) * val_ratio))
    val.extend(group[:val_count])
    train.extend(group[val_count:])
  rng.shuffle(train)
  rng.shuffle(val)
  return train, val


class ItemDataset(Dataset):
  def __init__(self, dataset: Path, records: list[dict], train: bool) -> None:
    self.dataset = dataset
    self.records = records
    self.train = train

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    record = self.records[index]
    image = Image.open(self.dataset / record["path"]).convert("RGB")
    if self.train:
      image = augment_image(image)
    tensor = image_to_tensor(image)
    return tensor, torch.tensor(record["label"], dtype=torch.long)


def augment_image(image: Image.Image) -> Image.Image:
  if random.random() < 0.45:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))
  if random.random() < 0.45:
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.88, 1.14))
  if random.random() < 0.2:
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.35)))
  if random.random() < 0.12:
    x_shift = random.randint(-2, 2)
    y_shift = random.randint(-2, 2)
    shifted = Image.new("RGB", image.size, (0, 0, 0))
    shifted.paste(image, (x_shift, y_shift))
    image = shifted
  return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
  if image.size != (96, 96):
    image = image.resize((96, 96), Image.Resampling.BILINEAR)
  data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
  data = data.view(96, 96, 3).permute(2, 0, 1).float().div(255.0)
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  return (data - mean) / std


class ConvBnAct(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.SiLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class ResidualBlock(nn.Module):
  def __init__(self, channels: int) -> None:
    super().__init__()
    self.block = nn.Sequential(
      ConvBnAct(channels, channels),
      nn.Conv2d(channels, channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(channels),
    )
    self.act = nn.SiLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.act(x + self.block(x))


class IsaacItemNet(nn.Module):
  def __init__(self, num_classes: int, width: int = 48) -> None:
    super().__init__()
    channels = [width, width * 2, width * 3, width * 5]
    self.features = nn.Sequential(
      ConvBnAct(3, channels[0]),
      ResidualBlock(channels[0]),
      ConvBnAct(channels[0], channels[1], stride=2),
      ResidualBlock(channels[1]),
      ResidualBlock(channels[1]),
      ConvBnAct(channels[1], channels[2], stride=2),
      ResidualBlock(channels[2]),
      ResidualBlock(channels[2]),
      ConvBnAct(channels[2], channels[3], stride=2),
      ResidualBlock(channels[3]),
      ResidualBlock(channels[3]),
    )
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.embedding = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.18),
      nn.Linear(channels[3], width * 6),
      nn.SiLU(inplace=True),
      nn.Dropout(0.12),
    )
    self.classifier = nn.Linear(width * 6, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.pool(x)
    x = self.embedding(x)
    return self.classifier(x)


@dataclass
class Metrics:
  loss: float
  top1: float
  top5: float
  top10: float


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  device: torch.device,
  *,
  epoch: int,
  log_every: int,
) -> tuple[float, float]:
  model.train()
  total_loss = 0.0
  total = 0
  correct = 0
  started = time.time()
  for batch_index, (images, labels) in enumerate(loader, start=1):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    batch = labels.numel()
    total += batch
    total_loss += loss.item() * batch
    correct += (logits.argmax(dim=1) == labels).sum().item()
    if log_every and (batch_index == 1 or batch_index % log_every == 0 or batch_index == len(loader)):
      elapsed = max(1e-6, time.time() - started)
      seen = min(total, len(loader.dataset))
      speed = seen / elapsed
      print(
        f"epoch {epoch:02d} batch {batch_index:03d}/{len(loader)} "
        f"loss={total_loss / max(1, total):.4f} top1={correct / max(1, total):.3f} "
        f"{speed:.1f} img/s",
        flush=True,
      )
  return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Metrics:
  model.eval()
  total_loss = 0.0
  total = 0
  top1 = 0
  top5 = 0
  top10 = 0
  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    loss = criterion(logits, labels)
    batch = labels.numel()
    total += batch
    total_loss += loss.item() * batch
    topk = logits.topk(k=min(10, logits.shape[1]), dim=1).indices
    matches = topk.eq(labels.view(-1, 1))
    top1 += matches[:, :1].any(dim=1).sum().item()
    top5 += matches[:, : min(5, topk.shape[1])].any(dim=1).sum().item()
    top10 += matches.any(dim=1).sum().item()
  return Metrics(
    loss=total_loss / max(1, total),
    top1=top1 / max(1, total),
    top5=top5 / max(1, total),
    top10=top10 / max(1, total),
  )


def count_params(model: nn.Module) -> int:
  return sum(math.prod(param.shape) for param in model.parameters())


if __name__ == "__main__":
  raise SystemExit(main())
