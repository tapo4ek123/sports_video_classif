from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

LABELS = {
    "sport_label": ["basketball", "football", "tennis", "volleyball"],
    "gender_category": ["male", "female"],
    "age_category": ["youth", "adult"],
}


def frame_indices(total: int, count: int, *, temporal: bool = False,
                  sample_rate: int | None = None) -> np.ndarray:
    if total < 1:
        raise ValueError("Video contains no frames")
    if count < 1:
        raise ValueError("count must be positive")
    if temporal:
        if sample_rate is not None and total >= (count - 1) * sample_rate + 1:
            first = (total - ((count - 1) * sample_rate + 1)) // 2
            return first + np.arange(count) * sample_rate
        span = min(total, max(count, 48))
        start = (total - span) // 2
        return np.linspace(start, start + span - 1, count).round().astype(int)
    return np.linspace(0, total - 1, count + 2)[1:-1].round().astype(int)


def decode_frames(path: str | Path, count: int, *, temporal: bool = False,
                  sample_rate: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise ValueError(f"Cannot count frames: {path}")
    frames = []
    for idx in frame_indices(total, count, temporal=temporal, sample_rate=sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Cannot decode frame {idx}: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def image_transform(size: int = 224, train: bool = False, temporal: bool = False,
                    normalization: str = "imagenet"):
    steps = [v2.ToImage()]
    if normalization == "kinetics_r3d":
        steps.extend([v2.Resize((128, 171), antialias=True), v2.CenterCrop(size)])
    elif normalization == "dinov2":
        steps.extend([v2.Resize(256, antialias=True), v2.CenterCrop(size)])
    else:
        steps.append(v2.Resize((size, size), antialias=True))
    steps.append(v2.ToDtype(torch.float32, scale=True))
    if train:
        steps.append(v2.RandomHorizontalFlip())
    if normalization == "kinetics_r3d":
        mean, std = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
    elif temporal:
        mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
    elif normalization == "half":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    steps.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(steps)


class VideoDataset(Dataset):
    def __init__(self, manifest: str | Path, num_frames: int, *, train: bool = False,
                 temporal: bool = False, size: int = 224, root: str | Path = ".",
                 normalization: str = "imagenet"):
        self.rows = pd.read_csv(manifest).to_dict("records")
        self.num_frames = num_frames
        self.temporal = temporal
        self.train = train
        self.root = Path(root)
        self.transform = image_transform(size, train and not temporal, temporal, normalization)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["video_path"]
        frames = decode_frames(path, self.num_frames, temporal=self.temporal,
                               sample_rate=12 if self.temporal and self.num_frames == 4 else None)
        x = torch.stack([self.transform(frame) for frame in frames])
        if self.temporal:
            if self.train and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-1,))
            x = x.permute(1, 0, 2, 3)
        labels = {}
        for key, values in LABELS.items():
            value = row.get(key)
            labels[key] = values.index(value) if isinstance(value, str) and value in values else -1
        return x, labels, str(row["clip_id"])
