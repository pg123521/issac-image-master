#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from mobileclip_item_search import (
  BASELINE_INDEX,
  DEFAULT_WEIGHTS,
  ICON_ROOT,
  MODEL_NAME,
  OBJECTS_JSON,
  PRETRAINED,
  load_visual_weights,
)


VISIBLE_BUCKETS = {
  "90-100": (0.90, 1.00),
  "70-90": (0.70, 0.90),
  "50-70": (0.50, 0.70),
}


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Fine-tune MobileCLIP visual blocks for partial Isaac icon retrieval."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  evaluate = subparsers.add_parser("evaluate")
  add_shared_args(evaluate)
  evaluate.add_argument("--weights", type=Path)
  evaluate.add_argument("--samples-per-object", type=int, default=1)
  evaluate.add_argument("--batch-size", type=int, default=64)
  evaluate.add_argument("--seed", type=int, default=20260715)
  evaluate.add_argument("--limit", type=int)

  train = subparsers.add_parser("train")
  add_shared_args(train)
  train.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
  train.add_argument("--initial-weights", type=Path)
  train.add_argument("--phase1-epochs", type=int, default=6)
  train.add_argument("--phase2-epochs", type=int, default=4)
  train.add_argument("--batch-size", type=int, default=48)
  train.add_argument("--learning-rate", type=float, default=2e-5)
  train.add_argument("--weight-decay", type=float, default=0.02)
  train.add_argument("--temperature", type=float, default=0.07)
  train.add_argument("--margin", type=float, default=0.10)
  train.add_argument("--preserve-weight", type=float, default=0.20)
  train.add_argument("--seed", type=int, default=20260715)
  train.add_argument("--limit", type=int)

  args = parser.parse_args()
  seed_everything(args.seed)
  device = best_device()
  objects = load_objects(args.objects_json, args.limit)
  icons = [load_icon(item) for item in objects]
  model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  if getattr(args, "initial_weights", None):
    load_visual_weights(model, args.initial_weights)
  if getattr(args, "weights", None):
    load_visual_weights(model, args.weights)
  model = model.to(device)

  if args.command == "evaluate":
    report = evaluate_model(
      model,
      preprocess,
      icons,
      device,
      args.batch_size,
      args.samples_per_object,
      args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

  baseline_vectors = load_baseline_vectors(args.baseline_index, objects)
  hard_neighbors = mine_hard_neighbors(baseline_vectors, count=12)
  phases = [
    ("last-stage", args.phase1_epochs, 1, args.learning_rate),
    ("last-two-stages", args.phase2_epochs, 2, args.learning_rate * 0.35),
  ]
  history: list[dict[str, Any]] = []
  epoch_number = 0
  for phase_name, epochs, trainable_stages, learning_rate in phases:
    if epochs <= 0:
      continue
    trainable = configure_trainable_visual(model, trainable_stages)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=args.weight_decay)
    print(
      f"phase={phase_name} epochs={epochs} trainable={sum(p.numel() for p in trainable):,} "
      f"lr={learning_rate:g} device={device}",
      flush=True,
    )
    for _ in range(epochs):
      epoch_number += 1
      metrics = train_epoch(
        model=model,
        preprocess=preprocess,
        icons=icons,
        baseline_vectors=baseline_vectors,
        hard_neighbors=hard_neighbors,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
        margin=args.margin,
        preserve_weight=args.preserve_weight,
        seed=args.seed + epoch_number * 100_003,
        phase_name=phase_name,
        epoch_number=epoch_number,
      )
      metrics.update({"epoch": epoch_number, "phase": phase_name})
      history.append(metrics)
      print(json.dumps(metrics, ensure_ascii=False), flush=True)

  save_checkpoint(model, args.output, history, args)
  print(f"wrote {args.output}", flush=True)
  return 0


def add_shared_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--objects-json", type=Path, default=OBJECTS_JSON)
  parser.add_argument("--baseline-index", type=Path, default=BASELINE_INDEX)


def best_device() -> torch.device:
  if torch.backends.mps.is_available():
    return torch.device("mps")
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def seed_everything(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)


def load_objects(path: Path, limit: int | None) -> list[dict[str, Any]]:
  objects = json.loads(path.read_text(encoding="utf-8"))
  return objects[:limit] if limit else objects


def load_icon(item: dict[str, Any]) -> Image.Image:
  path = ICON_ROOT / item["iconPath"].lstrip("/")
  return Image.open(path).convert("RGBA")


def load_baseline_vectors(index_path: Path, objects: list[dict[str, Any]]) -> torch.Tensor:
  payload = torch.load(index_path, map_location="cpu")
  by_id = {
    payload["labels"][int(item_index)]["item_id"]: payload["vectors"][vector_index].float()
    for vector_index, item_index in enumerate(payload["index_to_item"].tolist())
  }
  missing = [item["id"] for item in objects if item["id"] not in by_id]
  if missing:
    raise ValueError(f"baseline index is missing {len(missing)} objects, first={missing[0]}")
  return F.normalize(torch.stack([by_id[item["id"]] for item in objects]), dim=-1)


def mine_hard_neighbors(vectors: torch.Tensor, count: int) -> torch.Tensor:
  similarities = vectors @ vectors.T
  similarities.fill_diagonal_(-float("inf"))
  return similarities.topk(min(count, len(vectors) - 1), dim=1).indices


def configure_trainable_visual(model: torch.nn.Module, trainable_stages: int) -> list[torch.nn.Parameter]:
  for parameter in model.parameters():
    parameter.requires_grad = False

  visual = model.visual
  modules = [visual.trunk.final_conv, visual.trunk.head, visual.head]
  modules.extend(list(visual.trunk.stages)[-trainable_stages:])
  for module in modules:
    for parameter in module.parameters():
      parameter.requires_grad = True

  trainable = [parameter for parameter in visual.parameters() if parameter.requires_grad]
  if not trainable:
    raise RuntimeError("no visual parameters were selected for training")
  return trainable


def set_training_mode(model: torch.nn.Module) -> None:
  model.eval()
  for module in model.visual.modules():
    if any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
      module.train()
  for module in model.visual.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.eval()


def hard_negative_batches(
  object_count: int,
  batch_size: int,
  neighbors: torch.Tensor,
  rng: random.Random,
) -> Iterable[list[int]]:
  seed_ids = list(range(object_count))
  rng.shuffle(seed_ids)
  seeds_per_batch = max(1, batch_size // 2)
  for start in range(0, object_count, seeds_per_batch):
    selected: list[int] = []
    for item_id in seed_ids[start : start + seeds_per_batch]:
      selected.append(item_id)
      candidates = neighbors[item_id].tolist()
      selected.append(rng.choice(candidates))
    selected = list(dict.fromkeys(selected))
    while len(selected) < min(batch_size, object_count):
      candidate = rng.randrange(object_count)
      if candidate not in selected:
        selected.append(candidate)
    yield selected[:batch_size]


def train_epoch(
  model: torch.nn.Module,
  preprocess: Any,
  icons: list[Image.Image],
  baseline_vectors: torch.Tensor,
  hard_neighbors: torch.Tensor,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  batch_size: int,
  temperature: float,
  margin: float,
  preserve_weight: float,
  seed: int,
  phase_name: str,
  epoch_number: int,
) -> dict[str, float]:
  set_training_mode(model)
  rng = random.Random(seed)
  totals = {"loss": 0.0, "contrastive": 0.0, "margin": 0.0, "preserve": 0.0, "top1": 0.0}
  steps = 0
  total_steps = math.ceil(len(icons) / max(1, batch_size // 2))
  for ids in hard_negative_batches(len(icons), batch_size, hard_neighbors, rng):
    clean_images = [canonical_icon(icons[item_id]) for item_id in ids]
    query_images = [render_query(icons[item_id], rng, (0.50, 1.00), training=True) for item_id in ids]
    clean_batch = torch.stack([preprocess(image) for image in clean_images]).to(device)
    query_batch = torch.stack([preprocess(image) for image in query_images]).to(device)
    baseline = baseline_vectors[ids].to(device)

    optimizer.zero_grad(set_to_none=True)
    clean_features = F.normalize(model.encode_image(clean_batch), dim=-1)
    query_features = F.normalize(model.encode_image(query_batch), dim=-1)
    logits = query_features @ clean_features.T / temperature
    labels = torch.arange(len(ids), device=device)
    contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    similarities = query_features @ clean_features.T
    positives = similarities.diagonal()
    negatives = similarities.masked_fill(torch.eye(len(ids), device=device, dtype=torch.bool), -1.0)
    hardest_negative = negatives.max(dim=1).values
    margin_loss = F.relu(margin + hardest_negative - positives).mean()
    preserve_loss = (1.0 - (clean_features * baseline).sum(dim=-1)).mean()
    loss = contrastive + 0.5 * margin_loss + preserve_weight * preserve_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.visual.parameters() if p.requires_grad], 1.0)
    optimizer.step()

    totals["loss"] += float(loss.detach().cpu())
    totals["contrastive"] += float(contrastive.detach().cpu())
    totals["margin"] += float(margin_loss.detach().cpu())
    totals["preserve"] += float(preserve_loss.detach().cpu())
    totals["top1"] += float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
    steps += 1
    if steps % 5 == 0 or steps == total_steps:
      print(
        f"progress phase={phase_name} epoch={epoch_number} batch={steps}/{total_steps} "
        f"queries={min(steps * batch_size, total_steps * batch_size)} "
        f"loss={float(loss.detach().cpu()):.4f} "
        f"batch_top1={float((logits.argmax(dim=1) == labels).float().mean().detach().cpu()):.3f}",
        flush=True,
      )

  return {key: round(value / max(steps, 1), 6) for key, value in totals.items()}


@torch.no_grad()
def encode_images(
  model: torch.nn.Module,
  preprocess: Any,
  images: list[Image.Image],
  device: torch.device,
  batch_size: int,
) -> torch.Tensor:
  model.eval()
  vectors = []
  for start in range(0, len(images), batch_size):
    batch = torch.stack([preprocess(image) for image in images[start : start + batch_size]]).to(device)
    vectors.append(F.normalize(model.encode_image(batch), dim=-1).cpu().float())
  return torch.cat(vectors)


@torch.no_grad()
def evaluate_model(
  model: torch.nn.Module,
  preprocess: Any,
  icons: list[Image.Image],
  device: torch.device,
  batch_size: int,
  samples_per_object: int,
  seed: int,
) -> dict[str, Any]:
  gallery = encode_images(model, preprocess, [canonical_icon(icon) for icon in icons], device, batch_size)
  report: dict[str, Any] = {"model": MODEL_NAME, "device": str(device), "objects": len(icons), "buckets": {}}
  for bucket_name, visible_range in VISIBLE_BUCKETS.items():
    queries: list[Image.Image] = []
    targets: list[int] = []
    for item_id, icon in enumerate(icons):
      for sample in range(samples_per_object):
        rng = random.Random(seed + item_id * 10_007 + sample * 97 + int(visible_range[0] * 1000))
        queries.append(render_query(icon, rng, visible_range, training=False))
        targets.append(item_id)
    query_vectors = encode_images(model, preprocess, queries, device, batch_size)
    scores = query_vectors @ gallery.T
    top10 = scores.topk(min(10, len(icons)), dim=1).indices
    target_tensor = torch.tensor(targets).unsqueeze(1)
    metrics = {}
    for k in (1, 5, 10):
      actual_k = min(k, top10.shape[1])
      recall = (top10[:, :actual_k] == target_tensor).any(dim=1).float().mean().item()
      metrics[f"recall@{k}"] = round(recall, 6)
    report["buckets"][bucket_name] = metrics
  return report


def canonical_icon(icon: Image.Image) -> Image.Image:
  background = Image.new("RGB", icon.size, (0, 0, 0))
  background.paste(icon.convert("RGB"), mask=icon.getchannel("A"))
  return background


def render_query(
  icon: Image.Image,
  rng: random.Random,
  visible_range: tuple[float, float],
  training: bool,
) -> Image.Image:
  alpha = icon.getchannel("A")
  bbox = alpha.getbbox()
  sprite = icon.crop(bbox) if bbox else icon.copy()
  rotation = rng.uniform(-4.0, 4.0) if training else rng.uniform(-2.5, 2.5)
  sprite = sprite.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

  canvas_size = rng.randint(88, 120)
  background = make_background(canvas_size, rng)
  scale = rng.uniform(0.55, 0.90)
  longest = max(sprite.size)
  target = max(18, int(canvas_size * scale))
  width = max(1, round(sprite.width * target / longest))
  height = max(1, round(sprite.height * target / longest))
  interpolation = rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR])
  sprite = sprite.resize((width, height), interpolation)

  visible = rng.uniform(*visible_range)
  sprite = apply_partial_mask(sprite, visible, rng)
  jitter = int(canvas_size * 0.10)
  x = (canvas_size - width) // 2 + rng.randint(-jitter, jitter)
  y = (canvas_size - height) // 2 + rng.randint(-jitter, jitter)
  background.alpha_composite(sprite, (x, y))
  result = background.convert("RGB")

  if training:
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.82, 1.18))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.85, 1.18))
  if rng.random() < (0.35 if training else 0.20):
    result = result.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
  if rng.random() < (0.30 if training else 0.15):
    buffer = BytesIO()
    result.save(buffer, format="JPEG", quality=rng.randint(55, 88))
    result = Image.open(BytesIO(buffer.getvalue())).convert("RGB")
  return result


def make_background(size: int, rng: random.Random) -> Image.Image:
  palettes = [
    ((91, 71, 56), (119, 91, 70)),
    ((111, 85, 69), (145, 109, 83)),
    ((71, 78, 73), (99, 104, 91)),
    ((91, 62, 63), (126, 78, 74)),
    ((55, 50, 45), (83, 74, 63)),
  ]
  base, accent = rng.choice(palettes)
  image = Image.new("RGBA", (size, size), (*base, 255))
  draw = ImageDraw.Draw(image, "RGBA")
  tile = rng.randint(10, 22)
  for y in range(-tile, size + tile, tile):
    offset = (y // tile % 2) * (tile // 2)
    for x in range(-tile, size + tile, tile):
      noise = rng.randint(-16, 16)
      color = tuple(max(0, min(255, channel + noise)) for channel in accent)
      draw.rounded_rectangle(
        (x + offset + 1, y + 1, x + offset + tile - 2, y + tile - 2),
        radius=2,
        fill=(*color, rng.randint(25, 70)),
      )
  for _ in range(size * size // 18):
    x = rng.randrange(size)
    y = rng.randrange(size)
    shade = rng.choice([-1, 1]) * rng.randint(5, 18)
    pixel = tuple(max(0, min(255, channel + shade)) for channel in base)
    draw.point((x, y), fill=(*pixel, rng.randint(20, 65)))
  return image


def apply_partial_mask(sprite: Image.Image, visible: float, rng: random.Random) -> Image.Image:
  width, height = sprite.size
  mode = rng.choice(["left", "right", "top", "bottom", "corner"])
  mask = Image.new("L", sprite.size, 0)
  draw = ImageDraw.Draw(mask)
  if mode == "left":
    keep = max(1, round(width * visible))
    draw.rectangle((0, 0, keep - 1, height - 1), fill=255)
  elif mode == "right":
    keep = max(1, round(width * visible))
    draw.rectangle((width - keep, 0, width - 1, height - 1), fill=255)
  elif mode == "top":
    keep = max(1, round(height * visible))
    draw.rectangle((0, 0, width - 1, keep - 1), fill=255)
  elif mode == "bottom":
    keep = max(1, round(height * visible))
    draw.rectangle((0, height - keep, width - 1, height - 1), fill=255)
  else:
    side_ratio = math.sqrt(visible)
    keep_w = max(1, round(width * side_ratio))
    keep_h = max(1, round(height * side_ratio))
    left = 0 if rng.random() < 0.5 else width - keep_w
    top = 0 if rng.random() < 0.5 else height - keep_h
    draw.rectangle((left, top, left + keep_w - 1, top + keep_h - 1), fill=255)
  original_alpha = sprite.getchannel("A")
  sprite.putalpha(Image.composite(original_alpha, Image.new("L", sprite.size, 0), mask))
  return sprite


def save_checkpoint(
  model: torch.nn.Module,
  output: Path,
  history: list[dict[str, Any]],
  args: argparse.Namespace,
) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  visual_state = {
    key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
    for key, value in model.visual.state_dict().items()
  }
  torch.save({
    "model_name": MODEL_NAME,
    "pretrained": PRETRAINED,
    "visual_state_dict": visual_state,
    "training": {
      "objective": "asymmetric partial-query to canonical-icon contrastive retrieval",
      "phase1_epochs": args.phase1_epochs,
      "phase2_epochs": args.phase2_epochs,
      "visible_range": [0.50, 1.00],
      "history": history,
    },
  }, output)


if __name__ == "__main__":
  raise SystemExit(main())
