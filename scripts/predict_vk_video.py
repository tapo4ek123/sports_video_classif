"""Predict sport, event gender and event age from a local video using a VK checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from sports_video.models import VideoClassifier
from train_vk_bundle import BundleDataset, CLASSES, HEADS


def sample_times(frame_count: int, model: str, x3d_preprocess: str, start: float,
                 duration: float) -> np.ndarray:
    available = max(1, int(round(duration * 5)))
    if model == "x3d_xs" and x3d_preprocess == "shortside182":
        first = max(0, (available - 1 - (frame_count - 1) * 2) // 2)
        ids = first + np.arange(frame_count) * 2
        ids = np.minimum(ids, available - 1)
    else:
        ids = np.linspace(0, available - 1, frame_count + 2)[1:-1].round().astype(int)
    return start + ids / 5.0


def decode_window(cap: cv2.VideoCapture, times: np.ndarray, transform, temporal: bool):
    frames = []
    for second in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame at {second:.2f}s")
        frames.append(transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    x = torch.stack(frames)
    return x.permute(1, 0, 2, 3) if temporal else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/vk_owner_dinov2_1f"))
    ap.add_argument("--windows", type=int, default=3, help="At most this many 5-second windows")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.windows < 1:
        ap.error("--windows must be positive")
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    name = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(name, pretrained=False, freeze=config["freeze"],
                            weights=config["weights"] if name == "dinov2_small" else None)
    model.heads["sport_label"] = nn.Linear(model.width, len(CLASSES["sport"]))
    model.load_state_dict(torch.load(args.checkpoint_dir / "checkpoint.pt", map_location="cpu",
                                     weights_only=True))
    model.to(device).eval()
    sample = BundleDataset(Path("."), [], name, config["frames"], False,
                           config.get("x3d_preprocess", "legacy"))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or count <= 0:
        cap.release()
        raise ValueError("Video duration is unavailable")
    duration = count / fps
    window_s = min(5.0, duration)
    span = max(0.0, duration - window_s)
    starts = [span * (i + 1) / (args.windows + 1) for i in range(args.windows)]
    if span == 0:
        starts = [0.0]
    sums = {key: [] for key in CLASSES}
    tic = time.perf_counter()
    with torch.inference_mode():
        for start in starts:
            times = sample_times(config["frames"], name,
                                 config.get("x3d_preprocess", "legacy"), start, window_s)
            x = decode_window(cap, times, sample.transform, sample.temporal).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            for key, head in HEADS.items():
                sums[key].append(logits[head][0].float().cpu().numpy())
    cap.release()
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "video": str(args.video), "checkpoint_dir": str(args.checkpoint_dir),
        "video_duration_s": duration, "windows": len(starts),
        "window_starts_s": [round(value, 2) for value in starts],
        "elapsed_s": time.perf_counter() - tic, "predictions": {},
    }
    for key, classes in CLASSES.items():
        mean = np.mean(sums[key], axis=0)
        probs = np.exp(mean - mean.max())
        probs /= probs.sum()
        result["predictions"][key] = {
            "label": classes[int(probs.argmax())],
            "probabilities": {label: float(probs[i]) for i, label in enumerate(classes)},
        }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
