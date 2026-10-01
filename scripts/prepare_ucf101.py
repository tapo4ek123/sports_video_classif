"""Extract a focused UCF101 sport subset and build leakage-safe fixed splits."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASS_MAP = {
    "Basketball": "basketball",
    "SoccerPenalty": "football",
    "TennisSwing": "tennis",
    "VolleyballSpiking": "volleyball",
}
NAME = re.compile(r"^v_([A-Za-z]+)_g(\d+)_c(\d+)\.(?:avi|mp4)$", re.I)
FIELDS = ["clip_id", "source_dataset", "source_video_id", "event_id", "video_path",
          "original_label", "sport_label", "gender_category", "age_category",
          "sport_label_source", "gender_label_source", "age_label_source",
          "duration_sec", "fps", "width", "height", "split"]


def inspect_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n - 1))
        last_ok, _ = cap.read()
        ok = ok and last_ok
    cap.release()
    if not ok or fps <= 0 or n <= 0 or width <= 0 or height <= 0:
        return None
    return {"duration_sec": n / fps, "fps": fps, "width": width, "height": height}


def extract_selected(archives: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"Scanning {archive}", flush=True)
        with tarfile.open(archive, mode="r|*") as src:
            for item in src:
                name = Path(item.name).name
                match = NAME.match(name)
                if not item.isfile() or not match or match.group(1) not in CLASS_MAP:
                    continue
                target = output / match.group(1) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == item.size:
                    continue
                stream = src.extractfile(item)
                if stream is None:
                    continue
                with stream, target.open("wb") as dst:
                    shutil.copyfileobj(stream, dst)


def assign_splits(frame: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    frame["split"] = ""
    for sport, part in frame.groupby("sport_label"):
        groups = sorted(part["source_video_id"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * .15))
        n_val = max(1, round(n * .15))
        if n - n_test - n_val < 1:
            raise ValueError(f"Too few source groups for {sport}: {n}")
        mapping = {g: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
                   for i, g in enumerate(groups)}
        frame.loc[part.index, "split"] = part["source_video_id"].map(mapping)
    assert frame.groupby("source_video_id")["split"].nunique().max() == 1
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", nargs="+", default=["data/raw/ucf101_archives/shard-00000.tar"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    archives = [Path(p) for p in args.archives]
    if any(not p.exists() for p in archives):
        raise FileNotFoundError("Download UCF101 archives first with scripts/download_ucf101.py")
    root = Path("data/processed/ucf101")
    extract_selected(archives, root)
    rows, invalid = [], []
    for path in sorted(root.glob("*/*")):
        match = NAME.match(path.name)
        if not match or match.group(1) not in CLASS_MAP:
            continue
        stats = inspect_video(path)
        if stats is None:
            invalid.append(str(path))
            continue
        original, group, _ = match.groups()
        rows.append({
            "clip_id": path.stem,
            "source_dataset": "UCF101",
            "source_video_id": f"UCF101:{original}:g{group}",
            "event_id": pd.NA,
            "video_path": path.as_posix(),
            "original_label": original,
            "sport_label": CLASS_MAP[original],
            "gender_category": pd.NA,
            "age_category": pd.NA,
            "sport_label_source": "UCF101 action label mapped to sport",
            "gender_label_source": pd.NA,
            "age_label_source": pd.NA,
            **stats,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No selected videos found in archives")
    if frame.clip_id.duplicated().any():
        raise ValueError("Duplicate clip IDs")
    frame = assign_splits(frame, args.seed)[FIELDS]
    Path("data/metadata").mkdir(parents=True, exist_ok=True)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    frame.to_csv("data/metadata/manifest.csv", index=False)
    for split in ("train", "val", "test"):
        frame.loc[frame.split == split].to_csv(f"data/splits/{split}.csv", index=False)
    Path("data/metadata/invalid_videos.json").write_text(json.dumps(invalid, indent=2))
    print(frame.groupby(["split", "sport_label"]).size().to_string())
    print(f"Valid clips: {len(frame)}, invalid: {len(invalid)}")


if __name__ == "__main__":
    main()
