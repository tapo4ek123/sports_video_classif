"""Train video-only multi-task models on sports_training_bundle.

The JSONL splits are fixed by source_group_id. Titles/descriptions are never model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sports_video.data import image_transform
from sports_video.models import VideoClassifier
from sports_video.provenance import sha256_file


CLASSES = {
    "sport": ["basketball", "football", "handball", "table_tennis", "tennis", "volleyball"],
    "gender": ["men", "women"],
    "age": ["under_18", "adult_18_plus"],
}
HEADS = {"sport": "sport_label", "gender": "gender_category", "age": "age_category"}


def read_split(root: Path, split: str) -> list[dict]:
    path = root / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def check_splits(splits: dict[str, list[dict]], root: Path) -> dict:
    groups = {}
    label_by_group = {}
    hashes = {}
    for split, rows in splits.items():
        groups[split] = {row["source_group_id"] for row in rows}
        hashes[split] = hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
        for row in rows:
            labels = tuple(row[key] for key in CLASSES)
            group = row["source_group_id"]
            if group in label_by_group and label_by_group[group] != labels:
                raise ValueError(f"Conflicting labels within source group {group}")
            label_by_group[group] = labels
            if len(row["frames"]) < 4:
                raise ValueError(f"Too few frames in {row['clip_id']}")
    for a, ga in groups.items():
        for b, gb in groups.items():
            if a < b and ga & gb:
                raise ValueError(f"Source group overlap: {a} and {b}")
    return {"sha256": hashes,
            "clips": {name: len(rows) for name, rows in splits.items()},
            "groups": {name: len(groups[name]) for name in groups}}


class BundleDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], model: str, frames: int, train: bool,
                 x3d_preprocess: str = "legacy"):
        self.root, self.rows, self.model, self.frames, self.train = root, rows, model, frames, train
        self.temporal = model in {"x3d_xs", "r3d_18"}
        self.x3d_preprocess = x3d_preprocess
        if model == "x3d_xs" and x3d_preprocess == "shortside182":
            self.transform = v2.Compose([
                v2.ToImage(), v2.Resize(182, antialias=True), v2.CenterCrop(160),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
            ])
        else:
            self.transform = image_transform(
                size=160 if model == "x3d_xs" else 112 if model == "r3d_18" else 224,
                temporal=self.temporal,
                normalization="kinetics_r3d" if model == "r3d_18" else "dinov2"
                if model == "dinov2_small" else "imagenet",
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        paths = row["frames"]
        if self.model == "x3d_xs" and self.x3d_preprocess == "shortside182":
            # 5 fps extracted frames: step 2 approximates 12/30 seconds in the X3D-XS recipe.
            stride = 2
            first = max(0, (len(paths) - 1 - (self.frames - 1) * stride) // 2)
            indices = np.arange(self.frames) * stride + first
        else:
            indices = np.linspace(0, len(paths) - 1, self.frames + 2)[1:-1].round().astype(int)
        flip = self.train and random.random() < 0.5
        frames = []
        for i in indices:
            path = self.root / paths[int(i)]
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = self.transform(rgb)
            frames.append(frame.flip(-1) if flip else frame)
        x = torch.stack(frames)
        if self.temporal:
            x = x.permute(1, 0, 2, 3)
        labels = torch.tensor([CLASSES[key].index(row[key]) for key in CLASSES], dtype=torch.long)
        return x, labels, index


def metrics(true: list[int], pred: list[int], classes: list[str]) -> dict:
    ids = list(range(len(classes)))
    p, r, f, support = precision_recall_fscore_support(true, pred, labels=ids, zero_division=0)
    return {
        "n": len(true), "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, labels=ids, average="macro", zero_division=0),
        "weighted_f1": f1_score(true, pred, labels=ids, average="weighted", zero_division=0),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(support[i])}
                      for i, name in enumerate(classes)},
        "confusion_matrix": confusion_matrix(true, pred, labels=ids).tolist(),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, rows: list[dict], device: torch.device):
    model.eval()
    clip_logits = {key: {} for key in CLASSES}
    t0 = time.perf_counter()
    for x, _, indices in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        for key, head in HEADS.items():
            for i, logit in zip(indices.tolist(), out[head].float().cpu()):
                clip_logits[key][i] = logit.numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - t0
    group_indices = defaultdict(list)
    for i, row in enumerate(rows):
        group_indices[row["source_group_id"]].append(i)
    result = {"clip": {}, "group": {}, "elapsed_sec": duration,
              "clips_per_sec": len(rows) / duration}
    group_predictions = []
    for key, classes in CLASSES.items():
        true = [classes.index(row[key]) for row in rows]
        pred = [int(clip_logits[key][i].argmax()) for i in range(len(rows))]
        result["clip"][key] = metrics(true, pred, classes)
        gt, gp = [], []
        for group, indices in group_indices.items():
            gt.append(true[indices[0]])
            gp.append(int(np.mean([clip_logits[key][i] for i in indices], axis=0).argmax()))
            if key == "sport":
                group_predictions.append({"source_group_id": group, "n_clips": len(indices)})
        result["group"][key] = metrics(gt, gp, classes)
        for item, t, p in zip(group_predictions, gt, gp):
            item[key] = {"true": classes[t], "pred": classes[p]}
    result["group"]["all_three_accuracy"] = float(np.mean([
        all(item[key]["true"] == item[key]["pred"] for key in CLASSES)
        for item in group_predictions
    ]))
    return result, group_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("sports_training_bundle"))
    ap.add_argument("--split-dir", type=Path,
                    help="JSONL split directory; frame paths remain relative to --root")
    ap.add_argument("--corrections", type=Path,
                    default=Path("data/metadata/vk_visual_corrections.json"))
    ap.add_argument("--model", choices=["resnet18", "dinov2_small", "x3d_xs", "r3d_18"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--x3d-preprocess", choices=["legacy", "shortside182"], default="legacy")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--class-weighting", choices=["none", "inverse_group_frequency"],
                    default="none")
    ap.add_argument("--head-lr", type=float)
    ap.add_argument("--init-from", type=Path)
    ap.add_argument("--unfreeze-last-blocks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0, help="For a speed probe only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    frames = args.frames or {"resnet18": 1, "dinov2_small": 4, "x3d_xs": 4, "r3d_18": 16}[args.model]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = args.split_dir or args.root
    splits = {name: read_split(split_dir, name) for name in ("train", "validation", "test")}
    corrections = json.loads(args.corrections.read_text(encoding="utf-8"))["sport_by_video_id"]
    correction_counts = {}
    for name, rows in splits.items():
        correction_counts[name] = 0
        for row in rows:
            corrected = corrections.get(row["video_id"])
            if corrected is not None and corrected != row["sport"]:
                row["sport"] = corrected
                correction_counts[name] += 1
    audit = check_splits(splits, split_dir)
    audit["correction_sha256"] = hashlib.sha256(args.corrections.read_bytes()).hexdigest()
    audit["corrected_clips"] = correction_counts
    if args.max_train:
        splits["train"] = splits["train"][:args.max_train]
    loss_weights = {}
    train_groups = {row["source_group_id"]: row for row in splits["train"]}
    if args.class_weighting == "inverse_group_frequency":
        for key, classes in CLASSES.items():
            counts = np.array([sum(row[key] == label for row in train_groups.values())
                               for label in classes], dtype=np.float32)
            if (counts == 0).any():
                raise ValueError(f"Cannot weight absent training class for {key}: {counts}")
            values = np.minimum(3.0, len(train_groups) / (len(classes) * counts))
            values /= values.mean()
            loss_weights[key] = values.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"root": str(args.root), "output": str(args.output),
                           "split_dir": str(split_dir),
                           "corrections": str(args.corrections),
                           "init_from": str(args.init_from) if args.init_from else None,
                           "frames": frames, "device": str(device), "audit": audit,
                           "loss_class_weights": loss_weights,
                           "torch_version": torch.__version__}
    weight_path = Path(args.weights)
    if weight_path.is_dir():
        weight_path /= "model.safetensors"
    config["pretrained_checkpoint_sha256"] = sha256_file(weight_path)
    if args.init_from:
        config["init_checkpoint_sha256"] = sha256_file(args.init_from)
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    loaders = {
        name: DataLoader(BundleDataset(args.root, rows, args.model, frames, name == "train",
                                       args.x3d_preprocess),
                         batch_size=args.batch_size, shuffle=name == "train",
                         num_workers=args.workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.workers > 0)
        for name, rows in splits.items()
    }
    model = VideoClassifier(args.model, freeze=args.freeze, weights=args.weights)
    model.heads["sport_label"] = nn.Linear(model.width, len(CLASSES["sport"]))
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=True))
    if args.unfreeze_last_blocks:
        if args.model != "dinov2_small" or not args.freeze:
            raise ValueError("Partial tuning requires --model dinov2_small --freeze")
        layers = model.encoder.encoder.encoder.layer
        if args.unfreeze_last_blocks > len(layers):
            raise ValueError("Too many blocks requested")
        for layer in layers[-args.unfreeze_last_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in model.encoder.encoder.layernorm.parameters():
            parameter.requires_grad_(True)
    model.to(device)
    if args.head_lr is not None:
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr},
            {"params": model.heads.parameters(), "lr": args.head_lr},
        ])
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_weight_tensors = {key: torch.tensor(value, device=device) for key, value in loss_weights.items()}
    best_score = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        if args.freeze and not args.unfreeze_last_blocks:
            model.encoder.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss_sum = 0.0
        for x, labels, _ in loaders["train"]:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = sum(nn.functional.cross_entropy(
                    out[HEADS[key]], labels[:, j], weight=loss_weight_tensors.get(key))
                           for j, key in enumerate(CLASSES))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(x)
        val, _ = evaluate(model, loaders["validation"], splits["validation"], device)
        score = float(np.mean([val["group"][key]["macro_f1"] for key in CLASSES]))
        record = {"epoch": epoch + 1, "seconds": time.perf_counter() - t0,
                  "train_loss": loss_sum / len(splits["train"]),
                  "val_group_macro_f1": {key: val["group"][key]["macro_f1"] for key in CLASSES},
                  "val_mean_macro_f1": score,
                  "peak_train_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.output / "checkpoint.pt")
            (args.output / "validation_metrics.json").write_text(json.dumps(val, indent=2))
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(args.output / "checkpoint.pt", map_location=device, weights_only=True))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    test, predictions = evaluate(model, loaders["test"], splits["test"], device)
    test["efficiency"] = {"parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint_mb": (args.output / "checkpoint.pt").stat().st_size / 2**20,
                          "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                          "peak_training_vram_mb": max(row["peak_train_vram_mb"] for row in history),
                          "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                          "frames": frames, "batch_size": args.batch_size,
                          "precision": "AMP fp16" if device.type == "cuda" else "fp32"}
    (args.output / "test_metrics.json").write_text(json.dumps(test, indent=2))
    (args.output / "group_predictions.json").write_text(json.dumps(predictions, indent=2))
    print(json.dumps({"test_group_macro_f1": {key: test["group"][key]["macro_f1"] for key in CLASSES},
                      "test_all_three_accuracy": test["group"]["all_three_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
