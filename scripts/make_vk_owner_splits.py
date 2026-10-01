"""Create a fixed VK-channel-disjoint robustness split from the bundle."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from train_vk_bundle import CLASSES, read_split


ROOT = Path("sports_training_bundle")
OUT = Path("data/splits/vk_owner_holdout")


def owner(row: dict) -> str:
    return row["video_id"].split("_", 1)[0]


def main():
    rows = [row for part in ("train", "validation", "test") for row in read_split(ROOT, part)]
    corrections = json.loads(Path("data/metadata/vk_visual_corrections.json").read_text())[
        "sport_by_video_id"]
    group_rows = {}
    for row in rows:
        group_rows.setdefault(row["source_group_id"], row)
    owners = sorted({owner(row) for row in rows})
    owner_index = {name: i for i, name in enumerate(owners)}
    feature_names = (["total"] + [f"sport:{x}" for x in CLASSES["sport"]]
                     + [f"gender:{x}" for x in CLASSES["gender"]]
                     + [f"age:{x}" for x in CLASSES["age"]])
    vectors = np.zeros((len(owners), len(feature_names)), dtype=np.int32)
    for row in group_rows.values():
        i = owner_index[owner(row)]
        sport = corrections.get(row["video_id"], row["sport"])
        values = ["total", f"sport:{sport}", f"gender:{row['gender']}", f"age:{row['age']}"]
        for value in values:
            vectors[i, feature_names.index(value)] += 1
    ratios = np.array([0.70, 0.15, 0.15])
    target = ratios[:, None] * vectors.sum(0)[None, :]
    rng = np.random.default_rng(20260929)
    best_score, best_assignment = float("inf"), None
    for _ in range(30000):
        assignment = rng.choice(3, size=len(owners), p=ratios)
        counts = np.stack([vectors[assignment == j].sum(0) for j in range(3)])
        if (counts[1:, 1:7] < 3).any() or (counts[0, 1:7] < 10).any():
            continue
        score = float(np.square((counts - target) / (target + 1)).sum())
        if score < best_score:
            best_score, best_assignment = score, assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Could not make a sport-covered channel split")
    assigned = {name: int(best_assignment[i]) for i, name in enumerate(owners)}
    names = ["train", "validation", "test"]
    parts = {name: [] for name in names}
    for row in rows:
        parts[names[assigned[owner(row)]]].append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"seed": 20260929, "trials": 30000, "score": best_score,
               "split_by": "VK owner ID in video_id; original source_group_id preserved",
               "splits": {}}
    for name, part in parts.items():
        group_part = {row["source_group_id"]: row for row in part}
        summary["splits"][name] = {
            "clips": len(part), "videos": len({row["video_id"] for row in part}),
            "groups": len(group_part), "owners": len({owner(row) for row in part}),
            "sport_groups": dict(Counter(corrections.get(row["video_id"], row["sport"])
                                          for row in group_part.values())),
            "gender_groups": dict(Counter(row["gender"] for row in group_part.values())),
            "age_groups": dict(Counter(row["age"] for row in group_part.values())),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
