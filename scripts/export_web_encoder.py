#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mobileclip_item_search import MODEL_NAME, PRETRAINED, load_visual_weights


DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "mobileclip-partial-v2.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "public" / "models" / "mobileclip-image-encoder.onnx"
IOS_ASSETS = PROJECT_ROOT / "ios" / "ModelConversion" / "output"


class ImageEncoder(torch.nn.Module):
  def __init__(self, visual: torch.nn.Module) -> None:
    super().__init__()
    self.visual = visual

  def forward(self, image: torch.Tensor) -> torch.Tensor:
    return self.visual(image)


def main() -> int:
  parser = argparse.ArgumentParser(description="Export the fine-tuned MobileCLIP encoder for browser inference.")
  parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  load_visual_weights(model, args.weights)
  wrapper = ImageEncoder(model.visual.eval()).eval()

  generator = torch.Generator().manual_seed(20260717)
  example = torch.rand((1, 3, 256, 256), generator=generator)
  with torch.no_grad():
    torch_output = wrapper(example).cpu().numpy().astype(np.float32)

  torch.onnx.export(
    wrapper,
    (example,),
    str(args.output),
    input_names=["image"],
    output_names=["embedding"],
    opset_version=18,
    dynamo=False,
    do_constant_folding=True,
  )
  onnx.checker.check_model(onnx.load(str(args.output)))

  session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
  ort_output = session.run(["embedding"], {"image": example.numpy()})[0]
  max_error = float(np.max(np.abs(torch_output - ort_output)))
  cosine = float(
    np.dot(torch_output[0], ort_output[0])
    / max(1e-12, np.linalg.norm(torch_output[0]) * np.linalg.norm(ort_output[0]))
  )
  if max_error > 2e-4 or cosine < 0.99999:
    raise RuntimeError(f"ONNX parity failed: max_error={max_error:.6g} cosine={cosine:.9f}")

  shutil.copy2(IOS_ASSETS / "item-vectors.f16", args.output.parent / "item-vectors.f16")
  shutil.copy2(IOS_ASSETS / "item-vectors.json", args.output.parent / "item-vectors.json")
  report = {
    "model": MODEL_NAME,
    "weights": args.weights.name,
    "input": "1x3x256x256 RGB float32 in 0...1, NCHW",
    "output": "1x512 raw float32 embedding; L2-normalize in browser",
    "maxAbsoluteError": max_error,
    "cosineSimilarity": cosine,
    "sizeBytes": args.output.stat().st_size,
  }
  (args.output.parent / "web-model-report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(report, ensure_ascii=False, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
