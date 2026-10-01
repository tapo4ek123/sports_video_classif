"""Measure all supervised checkpoints with batch=1 and the same test set."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sports_video.data import VideoDataset
from sports_video.models import VideoClassifier
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Output directories produced by scripts/train.py")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in args.runs:
        run = Path(name)
        config = json.loads((run / "config.json").read_text())
        model_name = config["model"]
        frames = config["frames"]
        dataset = VideoDataset("data/splits/test.csv", frames,
                               temporal=model_name in {"x3d_xs", "r3d_18"},
                               size=160 if model_name == "x3d_xs" else 112 if model_name == "r3d_18" else 224,
                               normalization="kinetics_r3d" if model_name == "r3d_18" else "half"
                               if model_name == "vit_small_patch32_224" else "dinov2"
                               if model_name == "dinov2_small" else "imagenet")
        local_backbone = config.get("weights") if model_name in {"dinov2_small", "dinov3_vits16"} else None
        model = VideoClassifier(model_name, pretrained=False, weights=local_backbone).to(device)
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location=device, weights_only=True))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        metrics, _ = evaluate(model, loader, device)
        report = {"model": model_name, "frames": frames,
                  "resolution": 160 if model_name == "x3d_xs" else 112 if model_name == "r3d_18" else 224,
                  "batch_size": 1, "precision": "AMP fp16" if device.type == "cuda" else "fp32",
                  "hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                  **metrics["runtime"]}
        (run / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(name, json.dumps(report), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
