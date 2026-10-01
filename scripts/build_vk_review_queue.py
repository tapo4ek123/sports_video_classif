"""Rank channel-held-out groups for human review of weak metadata labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from train_vk_bundle import read_split


ROOT = Path("sports_training_bundle")
SPLIT = Path("data/splits/vk_owner_holdout")
OUTPUT = Path("data/metadata/vk_owner_test_review_queue.csv")
RUNS = {
    "dino_1f": Path("outputs/vk_owner_dinov2_1f/group_predictions.json"),
    "dino_4f": Path("outputs/vk_owner_dinov2_4f/group_predictions.json"),
    "x3d_4f": Path("outputs/vk_owner_x3d_shortside_4f/group_predictions.json"),
}


def main():
    labels = json.loads(Path("data/metadata/vk_visual_corrections.json").read_text())[
        "sport_by_video_id"]
    metadata = {}
    for row in read_split(SPLIT, "test"):
        metadata.setdefault(row["source_group_id"], row)
    predictions = {name: {row["source_group_id"]: row for row in json.loads(path.read_text())}
                   for name, path in RUNS.items()}
    rows = []
    for group, source in metadata.items():
        item = {
            "source_group_id": group, "video_id": source["video_id"],
            "title": source["title"], "description": source["description"],
            "frame_path": str(ROOT / source["frames"][len(source["frames"]) // 2]),
            "weak_sport": labels.get(source["video_id"], source["sport"]),
            "weak_gender": source["gender"], "weak_age": source["age"],
            "quality_review": "", "sport_review": "", "gender_review": "", "age_review": "",
        }
        disagreements = 0
        for task in ("sport", "gender", "age"):
            votes = []
            for name, output in predictions.items():
                vote = output[group][task]["pred"]
                item[f"{name}_{task}"] = vote
                votes.append(vote)
                disagreements += vote != item[f"weak_{task}"]
            disagreements += len(set(votes)) > 1
        item["priority"] = disagreements
        rows.append(item)
    rows.sort(key=lambda x: (-x["priority"], x["source_group_id"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} review groups to {OUTPUT}")
    print("Priority >= 5:", sum(row["priority"] >= 5 for row in rows))


if __name__ == "__main__":
    main()
