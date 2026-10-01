"""Build a source-video review queue; title hints never become training labels."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FEMALE = re.compile(r"\b(?:women'?s?|female|girls?|wnba)\b", re.IGNORECASE)
MALE = re.compile(r"\b(?:men'?s?|male|boys?)\b", re.IGNORECASE)
YOUTH = re.compile(r"\b(?:u(?:1[0-8]|[6-9])|under[- ]?(?:1[0-8]|[6-9])|youth|junior)\b", re.IGNORECASE)


def hints(title: str) -> tuple[str | None, str | None]:
    female, male = bool(FEMALE.search(title)), bool(MALE.search(title))
    gender = "female" if female and not male else "male" if male and not female else None
    age = "youth" if YOUTH.search(title) else None
    return gender, age


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multisports", default="data/metadata/multisports_sources.csv")
    parser.add_argument("--muvy", default="data/metadata/muvy_sample.csv")
    parser.add_argument("--output", type=Path, default=Path("data/metadata/event_review_queue.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; preserve manual reviews")
    source = pd.read_csv(args.multisports).fillna("")
    muvy = pd.read_csv(args.muvy).fillna("")
    rows = []
    for item in source.itertuples(index=False):
        gender, age = hints(item.title)
        rows.append({
            "source_dataset": "MultiSports",
            "source_video_id": f"MultiSports:{item.sport_label}:v_{item.youtube_id}",
            "source_split": item.source_split,
            "sport_label": item.sport_label,
            "title": item.title,
            "source_url": item.source_video_url,
            "candidate_gender": gender,
            "candidate_age": age,
            "event_id": None,
            "competition_id": None,
            "gender_category": None,
            "age_category": None,
            "evidence_url": None,
            "video_match_checked": False,
            "review_status": "unreviewed",
        })
    for item in muvy.drop_duplicates("event_id").itertuples(index=False):
        rows.append({
            "source_dataset": "MUVY",
            "source_video_id": item.source_video_id,
            "source_split": "external_test",
            "sport_label": item.sport_label,
            "title": item.event_id,
            "source_url": None,
            "candidate_gender": None,
            "candidate_age": None,
            "event_id": item.event_id,
            "competition_id": None,
            "gender_category": None,
            "age_category": None,
            "evidence_url": None,
            "video_match_checked": False,
            "review_status": "unreviewed",
        })
    queue = pd.DataFrame(rows)
    if queue.duplicated(["source_dataset", "source_video_id"]).any():
        raise ValueError("Duplicate source video in review queue")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    print(f"review_rows={len(queue)} gender_hints={queue.candidate_gender.notna().sum()} "
          f"youth_hints={queue.candidate_age.notna().sum()} output={args.output}")


if __name__ == "__main__":
    main()
