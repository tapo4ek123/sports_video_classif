"""Create a small group-separated adaptation split with UCF101 tennis anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def one_clip_per_group(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    return rows.groupby("source_video_id", group_keys=False).sample(n=1, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multisports", default="data/metadata/multisports_train_sample.csv")
    parser.add_argument("--external-test", default="data/metadata/multisports_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    multi = pd.read_csv(args.multisports)
    external = pd.read_csv(args.external_test)
    train_parts = []
    val_parts = []
    for sport in ("basketball", "football", "volleyball"):
        rows = one_clip_per_group(multi[multi.sport_label == sport], args.seed)
        if len(rows) < 10:
            raise ValueError(f"Need 10 source groups for {sport}; found {len(rows)}")
        rows = rows.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        train_parts.append(rows.iloc[:8])
        val_parts.append(rows.iloc[8:10])
    for split, n, parts in (("train", 15, train_parts), ("val", 4, val_parts)):
        ucf = pd.read_csv(f"data/splits/{split}.csv")
        tennis = one_clip_per_group(ucf[ucf.sport_label == "tennis"], args.seed)
        if len(tennis) < n:
            raise ValueError(f"Need {n} UCF101 tennis groups in {split}")
        parts.append(tennis.sample(frac=1, random_state=args.seed).iloc[:n])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=args.seed)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=args.seed)
    train_groups = set(train.source_video_id)
    val_groups = set(val.source_video_id)
    test_groups = set(external.source_video_id)
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Source group overlap between train, validation and external test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "adapt_train.csv", index=False)
    val.to_csv(args.output_dir / "adapt_val.csv", index=False)
    summary = {
        "seed": args.seed,
        "train_clips": len(train), "val_clips": len(val),
        "train_sports": train.sport_label.value_counts().to_dict(),
        "val_sports": val.sport_label.value_counts().to_dict(),
        "external_test_groups": len(test_groups),
        "group_intersection": 0,
    }
    (args.output_dir / "adapt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
