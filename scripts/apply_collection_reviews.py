"""Apply owner-reviewed event labels and clip rejections to collected data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


MANIFESTS = [
    Path("data/metadata/muvs_sample.csv"),
    Path("data/metadata/internet_archive_sample.csv"),
    Path("data/metadata/internet_archive_college_sample.csv"),
    Path("data/metadata/internet_archive_middle_school_sample.csv"),
    Path("data/metadata/wikimedia_u17_sample.csv"),
]
QUEUE = Path("data/metadata/collection_label_queue.csv")
ACCEPTED = Path("data/metadata/collection_accepted.csv")
REVIEWS = Path("data/metadata/collection_user_reviews.csv")
QUALITY = Path("data/metadata/collection_quality_reviews.csv")


def main() -> None:
    reviews = pd.read_csv(REVIEWS, dtype=str).fillna("")
    quality = pd.read_csv(QUALITY, dtype=str).fillna("")
    if reviews.event_group_id.duplicated().any() or quality.clip_id.duplicated().any():
        raise ValueError("Duplicate manual review ID")
    frames = {path: pd.read_csv(path, dtype=str).fillna("")
              for path in [*MANIFESTS, QUEUE, ACCEPTED]}
    for path, frame in frames.items():
        for column in ("age_detail_raw", "label_source", "reviewed_at",
                       "video_evidence_url", "official_event_status", "review_note",
                       "age_review_status", "age_rule_url"):
            if column not in frame:
                frame[column] = ""
    for review in reviews.itertuples(index=False):
        if review.event_gender not in {"male", "female", "mixed"}:
            raise ValueError(f"Unsupported gender: {review.event_gender}")
        if review.age_category not in {"", "youth", "adult"}:
            raise ValueError(f"Unsupported age: {review.age_category}")
        if review.age_candidate not in {"", "youth", "adult"}:
            raise ValueError(f"Unsupported age candidate: {review.age_candidate}")
        if review.age_review_status not in {"visual_confirmed", "visual_tentative",
                                            "visual_ambiguous_event"}:
            raise ValueError(f"Unsupported age review status: {review.age_review_status}")
        if (review.age_review_status == "visual_confirmed") != bool(review.age_category):
            raise ValueError(f"Age status/category mismatch: {review.event_group_id}")
        if review.age_review_status != "visual_confirmed" and not review.age_candidate:
            raise ValueError(f"Unresolved age requires a candidate: {review.event_group_id}")
        if review.evidence_url.startswith("data/") and not Path(review.evidence_url).exists():
            raise FileNotFoundError(review.evidence_url)
        observed = 0
        for path, frame in frames.items():
            id_column = ("event_group_id" if path in {QUEUE, ACCEPTED}
                         else "source_video_id")
            mask = frame[id_column] == review.event_group_id
            if not mask.any():
                continue
            if (frame.loc[mask, "sport_label"] != review.sport_label).any():
                raise ValueError(f"Sport conflict for {review.event_group_id} in {path}")
            gender = review.event_gender if review.event_gender in {"male", "female"} else ""
            for column, value in (("gender_category", gender),
                                  ("age_category", review.age_category)):
                if not value:
                    if column == "age_category":
                        frame.loc[mask, column] = ""
                    continue
                prior = set(frame.loc[mask, column]) - {"", value}
                if prior:
                    raise ValueError(f"{column} conflict for {review.event_group_id}: {prior}")
                frame.loc[mask, column] = value
            if review.age_candidate:
                frame.loc[mask, "candidate_age"] = review.age_candidate
            for column, value in (("age_detail_raw", review.age_detail_raw),
                                  ("label_source", review.annotation_source),
                                  ("reviewed_at", review.reviewed_at),
                                  ("video_evidence_url", review.evidence_url),
                                  ("official_event_status", review.official_event_status),
                                  ("review_note", review.review_note),
                                  ("age_review_status", review.age_review_status),
                                  ("age_rule_url", review.age_rule_url)):
                frame.loc[mask, column] = value
            if path == QUEUE:
                frame.loc[mask, "gender_evidence_url"] = review.evidence_url
                frame.loc[mask, "age_evidence_url"] = (review.evidence_url if review.age_category else "")
                frame.loc[mask, "review_status"] = (
                    "owner_video_reviewed" if review.age_category else
                    "owner_age_ambiguous" if review.age_review_status == "visual_ambiguous_event"
                    else "owner_age_tentative")
            observed += 1
        if observed < 3:
            raise ValueError(f"Event missing from queue, accepted set, or source manifest: {review.event_group_id}")
    for quality_review in quality.itertuples(index=False):
        if quality_review.decision != "reject_quality":
            raise ValueError(f"Unsupported quality decision: {quality_review.decision}")
        matches = 0
        for path in MANIFESTS:
            frame = frames[path]
            mask = frame.clip_id == quality_review.clip_id
            if not mask.any():
                continue
            frame.loc[mask, "split"] = "reject_quality"
            frame.loc[mask, "review_status"] = "rejected_owner_review"
            frame.loc[mask, "review_note"] = quality_review.review_note
            matches += int(mask.sum())
        if matches != 1:
            raise ValueError(f"Quality review matched {matches} clips: {quality_review.clip_id}")
        if quality_review.clip_id in set(frames[ACCEPTED].clip_id):
            raise ValueError(f"Rejected clip still in accepted set: {quality_review.clip_id}")
    for path, frame in frames.items():
        frame.to_csv(path, index=False)
    print(f"event_reviews={len(reviews)} quality_rejections={len(quality)}")


if __name__ == "__main__":
    main()
