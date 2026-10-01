"""Apply explicit visual reviews and produce one clip per verified event.

This script fails if an accepted clip has no review, if two clips from one event
are accepted, or if a referenced file/evidence URL is missing.  It does not
silently create train/test splits from a tiny and imbalanced pilot.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("data/metadata/verified_youtube_sample.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data/metadata/verified_youtube_quality_reviews.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/verified_youtube_accepted.csv"))
    args = parser.parse_args()
    source = read_csv(args.sample)
    reviews = read_csv(args.reviews)
    ids = [r["clip_id"] for r in source]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id in source")
    decisions = {r["clip_id"]: r for r in reviews}
    if len(decisions) != len(reviews) or set(ids) != set(decisions):
        raise ValueError("every sampled clip must have exactly one review")
    accepted = []
    for item in source:
        review = decisions[item["clip_id"]]
        if review["decision"] not in {"accept", "reject"}:
            raise ValueError(f"invalid decision for {item['clip_id']}")
        if review["decision"] == "reject":
            continue
        if not Path(item["video_path"]).is_file():
            raise FileNotFoundError(item["video_path"])
        if not (item["event_evidence_url"].startswith("https://")
                and item["competition_url"].startswith("https://")):
            raise ValueError(f"missing event evidence: {item['clip_id']}")
        item["video_review_status"] = "accepted_visual_frame_review"
        item["video_review_note"] = review["reason"]
        accepted.append(item)
    groups = Counter(r["event_id"] for r in accepted)
    if any(count != 1 for count in groups.values()):
        raise ValueError("only one accepted clip per event is allowed in this pilot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(source[0]) + ["video_review_note"]
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    temporary.replace(args.output)
    cells = Counter((r["sport_label"], r["gender_category"], r["age_category"]) for r in accepted)
    print(f"accepted={len(accepted)} events={len(groups)} rejected={len(source)-len(accepted)}")
    for cell, count in sorted(cells.items()):
        print("/".join(cell), count)
    print(args.output)


if __name__ == "__main__":
    main()
