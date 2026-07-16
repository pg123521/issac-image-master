#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "ios" / "ModelConversion" / "output"
DETECTOR_WEIGHTS = PROJECT_ROOT / "models" / "room-collectible-detector-v1.pt"
ENCODER_WEIGHTS = PROJECT_ROOT / "models" / "mobileclip-partial-v1.pt"
SEARCH_INDEX = PROJECT_ROOT / "models" / "mobileclip-object-partial-index-v1.pt"


class ImageEncoder(torch.nn.Module):
  def __init__(self, visual: torch.nn.Module) -> None:
    super().__init__()
    self.visual = visual

  def forward(self, image: torch.Tensor) -> torch.Tensor:
    return self.visual(image)


def main() -> int:
  parser = argparse.ArgumentParser(description="Export offline Isaac models and search assets for iOS.")
  parser.add_argument(
    "target",
    choices=("all", "detector", "encoder", "index"),
    nargs="?",
    default="all",
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)

  targets = ("detector", "encoder", "index") if args.target == "all" else (args.target,)
  print(f"[0/{len(targets)}] output={args.output}", flush=True)
  for step, target in enumerate(targets, 1):
    print(f"[{step}/{len(targets)}] starting {target}", flush=True)
    if target == "detector":
      export_detector(args.output, args.overwrite)
    elif target == "encoder":
      export_encoder(args.output, args.overwrite)
    else:
      export_search_index(args.output, args.overwrite)
  print("iOS asset export complete", flush=True)
  return 0


def export_detector(output: Path, overwrite: bool) -> None:
  from ultralytics import YOLO

  patch_coremltools_scalar_cast()
  destination = output / "RoomCollectibleDetector.mlpackage"
  prepare_destination(destination, overwrite)
  print(f"  loading {DETECTOR_WEIGHTS.name}", flush=True)
  exported = Path(YOLO(str(DETECTOR_WEIGHTS)).export(
    format="coreml",
    imgsz=1024,
    half=True,
    nms=False,
    batch=1,
    device="cpu",
  ))
  shutil.move(str(exported), str(destination))
  print(f"  wrote {destination} ({directory_size(destination) / 1024 / 1024:.1f} MiB)", flush=True)


def patch_coremltools_scalar_cast() -> None:
  """Work around coremltools 9 treating a one-element constant as a scalar."""
  from coremltools.converters.mil import Builder as mb
  from coremltools.converters.mil.frontend.torch import ops

  def scalar_cast(context: Any, node: Any, dtype: type, dtype_name: str) -> None:
    inputs = ops._get_inputs(context, node, expected=1)
    value = inputs[0]
    if not (len(value.shape) == 0 or np.all([dimension == 1 for dimension in value.shape])):
      raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if value.can_be_folded_to_const():
      scalar = np.asarray(value.val).reshape(-1)[0].item()
      result = mb.const(val=dtype(scalar), name=node.name)
    elif len(value.shape) > 0:
      value = mb.squeeze(x=value, name=node.name + "_item")
      result = mb.cast(x=value, dtype=dtype_name, name=node.name)
    else:
      result = mb.cast(x=value, dtype=dtype_name, name=node.name)
    context.add(result, node.name)

  ops._cast = scalar_cast


def export_encoder(output: Path, overwrite: bool) -> None:
  import coremltools as ct
  import open_clip

  from scripts.mobileclip_item_search import MODEL_NAME, PRETRAINED, load_visual_weights

  destination = output / "MobileCLIPImageEncoderRaw.mlpackage"
  prepare_destination(destination, overwrite)
  patch_coremltools_scalar_cast()
  print(f"  loading {MODEL_NAME} with {ENCODER_WEIGHTS.name}", flush=True)
  model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  load_visual_weights(model, ENCODER_WEIGHTS)
  wrapper = ImageEncoder(model.visual.eval()).eval()
  example = torch.rand(1, 3, 256, 256)
  with torch.no_grad():
    reference = wrapper(example)
    traced = torch.jit.trace(wrapper, example, strict=True)
    traced_result = traced(example)
  trace_error = float((reference - traced_result).abs().max())
  print(f"  traced encoder output={tuple(reference.shape)} max_error={trace_error:.3g}", flush=True)

  converted = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.ImageType(
      name="image",
      shape=example.shape,
      scale=1.0 / 255.0,
      color_layout=ct.colorlayout.RGB,
    )],
    outputs=[ct.TensorType(name="embedding")],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
  )
  converted.short_description = "Fine-tuned MobileCLIP2-S0 image encoder for Isaac object retrieval."
  converted.author = "Isaac Item Lens"
  converted.version = "1.0"
  converted.user_defined_metadata.update({
    "model": MODEL_NAME,
    "weights": ENCODER_WEIGHTS.name,
    "input_size": "256x256 RGB",
    "input_scale": "1/255",
    "output": "Raw 512-dimensional embedding; normalize in Float32 in the app",
  })
  converted.save(str(destination))
  print(f"  wrote {destination} ({directory_size(destination) / 1024 / 1024:.1f} MiB)", flush=True)


def export_search_index(output: Path, overwrite: bool) -> None:
  vector_path = output / "item-vectors.f16"
  metadata_path = output / "item-vectors.json"
  prepare_destination(vector_path, overwrite)
  prepare_destination(metadata_path, overwrite)
  payload = torch.load(SEARCH_INDEX, map_location="cpu")
  vectors = payload["vectors"].float().numpy().astype("<f2", copy=False)
  labels = []
  for vector_index, item_index in enumerate(payload["index_to_item"].tolist()):
    label = payload["labels"][int(item_index)]
    labels.append({
      "vectorIndex": vector_index,
      "id": label["item_id"],
      "kind": label.get("kind", "item"),
      "gameId": label["game_id"],
      "nameZh": label["name_zh"],
      "nameEn": label["name_en"],
      "iconPath": label["icon_path"],
    })
  vector_path.write_bytes(vectors.tobytes(order="C"))
  metadata = {
    "format": "row-major IEEE 754 little-endian Float16",
    "normalized": True,
    "rows": int(vectors.shape[0]),
    "dimensions": int(vectors.shape[1]),
    "model": payload["model_name"],
    "weights": payload.get("visual_weights"),
    "objects": labels,
  }
  metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
  print(
    f"  wrote {vectors.shape[0]}x{vectors.shape[1]} vectors "
    f"({vector_path.stat().st_size / 1024:.1f} KiB) and metadata",
    flush=True,
  )


def prepare_destination(path: Path, overwrite: bool) -> None:
  if not path.exists():
    return
  if not overwrite:
    raise FileExistsError(f"{path} already exists; pass --overwrite to replace generated output")
  if path.is_dir():
    shutil.rmtree(path)
  else:
    path.unlink()


def directory_size(path: Path) -> int:
  return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


if __name__ == "__main__":
  raise SystemExit(main())
