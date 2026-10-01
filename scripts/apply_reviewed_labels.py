"""Apply owner-reviewed source-video labels without converting hints into ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("data/metadata/event_review_queue.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data/metadata/user_reviewed_labels.csv"))
    args = parser.parse_args()
    queue = pd.read_csv(args.queue, dtype=str).fillna("")
    reviews = pd.read_csv(args.reviews, dtype=str).fillna("")
    if reviews.youtube_id.duplicated().any():
        raise ValueError("Duplicate YouTube ID in manual reviews")
    for name in ("event_gender_raw", "label_source", "reviewed_at", "video_evidence_url",
                 "official_event_status"):
        if name not in queue:
            queue[name] = ""
    for item in reviews.itertuples(index=False):
        if item.event_gender not in {"male", "female", "mixed"}:
            raise ValueError(f"Unsupported event gender: {item.event_gender}")
        if item.age_category not in {"youth", "adult"}:
            raise ValueError(f"Unsupported age category: {item.age_category}")
        mask = ((queue.source_dataset == "MultiSports") &
                (queue.source_video_id == f"MultiSports:{item.sport_label}:v_{item.youtube_id}"))
        if mask.sum() != 1:
            raise ValueError(f"Expected one source video for {item.youtube_id}, found {mask.sum()}")
        index = queue.index[mask][0]
        existing = queue.loc[index, "event_gender_raw"]
        if existing and existing != item.event_gender:
            raise ValueError(f"Conflicting gender review for {item.youtube_id}")
        if queue.loc[index, "age_category"] not in ("", item.age_category):
            raise ValueError(f"Conflicting age review for {item.youtube_id}")
        gender = item.event_gender if item.event_gender in {"male", "female"} else ""
        if queue.loc[index, "gender_category"] not in ("", gender):
            raise ValueError(f"Conflicting gender category for {item.youtube_id}")
        queue.loc[index, "event_gender_raw"] = item.event_gender
        queue.loc[index, "gender_category"] = gender
        queue.loc[index, "age_category"] = item.age_category
        queue.loc[index, "label_source"] = item.annotation_source
        queue.loc[index, "reviewed_at"] = item.reviewed_at
        queue.loc[index, "video_evidence_url"] = queue.loc[index, "source_url"]
        queue.loc[index, "video_match_checked"] = "True"
        if not queue.loc[index, "official_event_status"]:
            queue.loc[index, "official_event_status"] = "pending"
        if queue.loc[index, "review_status"] in {"", "unreviewed", "owner_video_reviewed"}:
            queue.loc[index, "review_status"] = "owner_video_reviewed"
    queue.to_csv(args.queue, index=False)
    print(f"applied={len(reviews)} reviewed_sources={(queue.review_status == 'owner_video_reviewed').sum()} "
          f"output={args.queue}")


if __name__ == "__main__":
    main()
