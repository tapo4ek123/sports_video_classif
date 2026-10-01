"""Evaluate existing checkpoints on a separate video manifest without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from sports_video.data import LABELS, VideoDataset
from sports_video.models import VideoClassifier
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest.empty or manifest["sport_label"].isna().any():
        raise ValueError("Manifest must have verified sport labels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"manifest": str(args.manifest), "n_videos": len(manifest),
              "n_source_groups": int(manifest["source_video_id"].nunique()),
              "sports": manifest["sport_label"].value_counts().to_dict(),
              "models": {}}
    sport_names = LABELS["sport_label"]
    observed = [sport_names.index(x) for x in sport_names
                if x in set(manifest["sport_label"])]
    for name in args.runs:
        run = Path(name)
        config = json.loads((run / "config.json").read_text())
        model_name = config["model"]
        temporal = model_name in {"x3d_xs", "r3d_18"}
        size = 160 if model_name == "x3d_xs" else 112 if model_name == "r3d_18" else 224
        normalization = "kinetics_r3d" if model_name == "r3d_18" else "dinov2" if model_name == "dinov2_small" else "imagenet"
        dataset = VideoDataset(args.manifest, config["frames"], temporal=temporal,
                               size=size, normalization=normalization)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        backbone = config.get("weights") if model_name in {"dinov2_small", "dinov3_vits16"} else None
        model = VideoClassifier(model_name, pretrained=False, weights=backbone).to(device)
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location=device,
                                         weights_only=True))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        metrics, rows = evaluate(model, loader, device)
        true = [r["sport_label"]["true"] for r in rows]
        pred = [r["sport_label"]["pred"] for r in rows]
        assert all(x is not None for x in true + pred)
        focused = {
            "n": len(rows),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1_observed_sports": float(f1_score(true, pred, labels=observed,
                                                        average="macro", zero_division=0)),
            "confusion_matrix_all_four": confusion_matrix(true, pred,
                                                             labels=list(range(len(sport_names)))).tolist(),
            "sport_order": sport_names,
        }
        report["models"][model_name] = {"sport": focused,
                                        "runtime": metrics["runtime"]}
        (args.output / f"{model_name}_predictions.json").write_text(
            json.dumps(rows, indent=2))
        print(model_name, json.dumps(report["models"][model_name]), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
