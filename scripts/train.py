from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sports_video.data import LABELS, VideoDataset
from sports_video.metrics import head_metrics
from sports_video.models import VideoClassifier
from sports_video.provenance import dataset_provenance, sha256_file


def evaluate(model, loader, device):
    model.eval()
    truth = {name: [] for name in LABELS}
    pred = {name: [] for name in LABELS}
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for x, y, ids in loader:
            x = x.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(x)
            for name in LABELS:
                p = output[name].argmax(1).cpu().tolist()
                t = y[name].tolist()
                truth[name].extend(t)
                pred[name].extend(p)
            for i, clip_id in enumerate(ids):
                rows.append({"clip_id": clip_id, **{
                    name: {"true": int(y[name][i]) if int(y[name][i]) >= 0 else None,
                           "pred": int(output[name][i].argmax()) if int(y[name][i]) >= 0 else None,
                           "confidence": float(output[name][i].softmax(0).max()) if int(y[name][i]) >= 0 else None}
                    for name in LABELS}})
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics = {name: head_metrics(truth[name], pred[name], name) for name in LABELS}
    metrics["runtime"] = {"elapsed_sec": elapsed, "videos_per_sec": len(rows) / elapsed,
                          "ms_per_video": elapsed * 1000 / max(1, len(rows)),
                          "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0}
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["efficientnet_b0", "dinov2_small", "dinov3_vits16", "x3d_xs", "r3d_18",
                                            "tf_efficientnet_b4", "vit_small_patch32_224"], default="efficientnet_b0")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--weights", help="Local safetensors file for timm backbones")
    parser.add_argument("--init-from", help="Existing classifier checkpoint for continued training")
    parser.add_argument("--train-manifest", default="data/splits/train.csv")
    parser.add_argument("--val-manifest", default="data/splits/val.csv")
    parser.add_argument("--test-manifest", default="data/splits/test.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.model == "x3d_xs" and args.frames != 4:
        parser.error("X3D-XS pretrained preprocessing uses 4 temporal frames")
    if args.model == "r3d_18" and args.frames != 16:
        parser.error("R3D-18 pretrained preprocessing uses 16 temporal frames")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifests = {split: Path(getattr(args, f"{split}_manifest"))
                 for split in ("train", "val", "test")}
    defaults = {split: Path(f"data/splits/{split}.csv") for split in manifests}
    provenance = dataset_provenance() if manifests == defaults else {
        "dataset": "custom manifests",
        "split_sha256": {split: sha256_file(path) for split, path in manifests.items()},
    }
    config = vars(args) | provenance | {"torch_version": torch.__version__}
    if args.weights:
        source = Path(args.weights)
        file = source / "model.safetensors" if source.is_dir() else source
        config["pretrained_checkpoint_sha256"] = sha256_file(file)
    if args.init_from:
        config["init_checkpoint_sha256"] = sha256_file(args.init_from)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = VideoDataset(manifests[split], args.frames,
                               train=split == "train", temporal=args.model in {"x3d_xs", "r3d_18"},
                               size=160 if args.model == "x3d_xs" else 112 if args.model == "r3d_18" else 224,
                               normalization="kinetics_r3d" if args.model == "r3d_18" else "half"
                               if args.model == "vit_small_patch32_224" else "dinov2"
                               if args.model == "dinov2_small" else "imagenet")
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=split == "train", num_workers=args.workers)
    model = VideoClassifier(args.model, freeze=args.freeze, weights=args.weights).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device,
                                         weights_only=True))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze:
            model.encoder.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        for x, labels, _ in loaders["train"]:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                losses = []
                for name in LABELS:
                    target = labels[name].to(device)
                    mask = target >= 0
                    if mask.any():
                        losses.append(nn.functional.cross_entropy(logits[name][mask], target[mask]))
                if not losses:
                    raise ValueError("Training split has no known labels")
                loss = sum(losses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        train_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val_metrics, _ = evaluate(model, loaders["val"], device)
        score = val_metrics["sport_label"].get("macro_f1", 0)
        row = {"epoch": epoch + 1, "train_loss_sum": total_loss,
               "val_sport_macro_f1": score, "seconds": time.perf_counter() - epoch_start,
               "peak_train_vram_mb": train_peak}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out / "checkpoint.pt", map_location=device, weights_only=True))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics, rows = evaluate(model, loaders["test"], device)
    metrics["efficiency"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_mb": (out / "checkpoint.pt").stat().st_size / 2**20,
        "peak_training_vram_mb": max(item["peak_train_vram_mb"] for item in history),
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        "frames": args.frames, "resolution": 160 if args.model == "x3d_xs" else 112 if args.model == "r3d_18" else 224,
        "batch_size": args.batch_size, "precision": "AMP fp16" if device.type == "cuda" else "fp32",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "predictions.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
