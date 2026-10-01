"""Mix all UCF101 training clips with the small MultiSports adaptation sample.

MultiSports train clips are repeated to balance the two video domains. Their
augmentation is drawn independently at each pass through the training loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapt-train", default="data/splits/adapt_train.csv")
    parser.add_argument("--adapt-val", default="data/splits/adapt_val.csv")
    parser.add_argument("--ucf-train", default="data/splits/train.csv")
    parser.add_argument("--ucf-val", default="data/splits/val.csv")
    parser.add_argument("--external-test", default="data/metadata/multisports_sample.csv")
    parser.add_argument("--multi-repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.multi_repeat < 1:
        parser.error("multi-repeat must be positive")
    adapt_train = pd.read_csv(args.adapt_train)
    adapt_val = pd.read_csv(args.adapt_val)
    ucf_train = pd.read_csv(args.ucf_train)
    ucf_val = pd.read_csv(args.ucf_val)
    external = pd.read_csv(args.external_test)
    multi_train = adapt_train[adapt_train.source_dataset == "MultiSports"]
    multi_val = adapt_val[adapt_val.source_dataset == "MultiSports"]
    train = pd.concat([ucf_train] + [multi_train] * args.multi_repeat,
                      ignore_index=True).sample(frac=1, random_state=args.seed)
    val = pd.concat([ucf_val, multi_val], ignore_index=True)
    for left, right, name in ((train, val, "train/val"),
                              (train, external, "train/external"),
                              (val, external, "val/external")):
        if set(left.source_video_id) & set(right.source_video_id):
            raise ValueError(f"Source group overlap: {name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "mixed_train.csv", index=False)
    val.to_csv(args.output_dir / "mixed_val.csv", index=False)
    summary = {
        "seed": args.seed,
        "multi_repeat": args.multi_repeat,
        "train_rows": len(train), "val_rows": len(val),
        "train_unique_clips": train.clip_id.nunique(),
        "train_source_groups": train.source_video_id.nunique(),
        "val_source_groups": val.source_video_id.nunique(),
        "train_by_source": train.source_dataset.value_counts().to_dict(),
        "val_by_source": val.source_dataset.value_counts().to_dict(),
    }
    (args.output_dir / "mixed_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
