"""Simple middle-frame near-duplicate audit across the fixed train/test split."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd


def middle_dhash(path: str) -> int:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot decode middle frame: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8))
    return sum(int(bit) << i for i, bit in enumerate((small[:, 1:] > small[:, :-1]).ravel()))


def main():
    manifest = pd.read_csv("data/metadata/manifest.csv")
    manifest["dhash"] = [middle_dhash(p) for p in manifest.video_path]
    train = manifest.loc[manifest.split == "train"]
    test = manifest.loc[manifest.split == "test"]
    rows = []
    for _, row in test.iterrows():
        nearest = min(((int(row.dhash) ^ int(item.dhash)).bit_count(), item.clip_id)
                      for _, item in train.iterrows())
        rows.append({"test_clip_id": row.clip_id, "nearest_train_clip_id": nearest[1],
                     "hamming_distance": nearest[0]})
    report = {"method": "64-bit dHash of the middle decoded frame",
              "test_clips": len(rows), "minimum_distance": min(r["hamming_distance"] for r in rows),
              "pairs_at_distance_5_or_less": sum(r["hamming_distance"] <= 5 for r in rows),
              "nearest_pairs": rows}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/duplicate_audit.json").write_text(json.dumps(report, indent=2))
    print(report["minimum_distance"], report["pairs_at_distance_5_or_less"])


if __name__ == "__main__":
    main()
