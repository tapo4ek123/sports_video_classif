from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from sports_video.data import LABELS, decode_frames
from sports_video.metrics import head_metrics
from sports_video.provenance import dataset_provenance, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--output", default="outputs/E0_siglip")
    args = parser.parse_args()
    prompts = json.loads(Path("configs/siglip_prompts.json").read_text())
    classes = LABELS["sport_label"]
    assert list(prompts) == classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    texts = [prompt for name in classes for prompt in prompts[name]]
    text_inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        text_features = model.get_text_features(**text_inputs)
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    rows = pd.read_csv(f"data/splits/{args.split}.csv")
    if args.max_videos:
        rows = rows.iloc[:args.max_videos]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dataset_provenance() | {"prompts": prompts, "classes": classes,
                  "precision": "AMP fp16" if device.type == "cuda" else "fp32",
                  "torch_version": torch.__version__}
    checkpoint = Path(args.model) / "model.safetensors"
    if checkpoint.exists():
        config["pretrained_checkpoint_sha256"] = sha256_file(checkpoint)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    results = {}
    for count in (1, 4, 8):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        truth, pred, predictions = [], [], []
        for _, row in rows.iterrows():
            images = decode_frames(row.video_path, count)
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            similarity = features @ text_features.T
            class_scores = []
            offset = 0
            for name in classes:
                n = len(prompts[name])
                class_scores.append(similarity[:, offset:offset + n].mean().item())
                offset += n
            prediction = int(np.argmax(class_scores))
            truth.append(classes.index(row.sport_label))
            pred.append(prediction)
            predictions.append({"clip_id": row.clip_id, "true": row.sport_label,
                                "pred": classes[prediction], "scores": class_scores})
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results[str(count)] = {
            "sport_label": head_metrics(truth, pred, "sport_label"),
            "gender_category": {"n": 0, "status": "not evaluated"},
            "age_category": {"n": 0, "status": "not evaluated"},
            "runtime": {"elapsed_sec": elapsed, "ms_per_video": elapsed * 1000 / len(rows),
                        "videos_per_sec": len(rows) / elapsed,
                        "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0},
            "efficiency": {"parameters": sum(p.numel() for p in model.parameters()),
                           "checkpoint_mb": (Path(args.model) / "model.safetensors").stat().st_size / 2**20
                           if (Path(args.model) / "model.safetensors").exists() else None,
                           "hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                           "frames": count, "resolution": 224, "batch_size": 1,
                           "precision": config["precision"]},
        }
        (out / f"predictions_{count}f.json").write_text(json.dumps(predictions, indent=2))
        print(count, results[str(count)]["sport_label"]["macro_f1"], round(elapsed, 1), flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
