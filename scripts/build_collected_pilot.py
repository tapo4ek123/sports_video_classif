"""Combine visually screened short clips without promoting title hints to labels."""
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
OUTPUT = Path("data/metadata/collection_accepted.csv")


def main() -> None:
    events = pd.read_csv("data/metadata/muvs_event_candidates.csv").fillna("")
    muvs = events.set_index("event_id")
    rows = []
    for path in MANIFESTS:
        frame = pd.read_csv(path).fillna("")
        for item in frame[frame.split == "unlabeled_review"].to_dict("records"):
            video = Path(item["video_path"])
            if not video.exists() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
            if float(item["duration_sec"]) < 4.5:
                raise ValueError(f"Too short: {video}")
            candidate_gender = item.get("candidate_gender", "")
            candidate_age = item.get("candidate_age", "")
            source_url = item.get("source_url", "")
            if item["source_dataset"] == "MUVS":
                event_id = item["source_video_id"].removeprefix("MUVS:")
                event = muvs.loc[event_id]
                candidate_gender = event.candidate_gender
                candidate_age = event.candidate_age
                source_url = source_url or event.sample_video_url
            if candidate_age == "junior":
                candidate_age = "youth"
            rows.append({
                "clip_id": item["clip_id"],
                "event_group_id": item["source_video_id"],
                "source_dataset": item["source_dataset"],
                "source_url": source_url,
                "video_path": video.as_posix(),
                "sport_label": item["sport_label"],
                "candidate_gender": candidate_gender,
                "candidate_age": candidate_age,
                "gender_category": item.get("gender_category", ""),
                "age_category": item.get("age_category", ""),
                "age_detail_raw": item.get("age_detail_raw", ""),
                "age_review_status": item.get("age_review_status", ""),
                "age_rule_url": item.get("age_rule_url", ""),
                "label_source": item.get("label_source", ""),
                "reviewed_at": item.get("reviewed_at", ""),
                "video_evidence_url": item.get("video_evidence_url", ""),
                "official_event_status": item.get("official_event_status", ""),
                "review_note": item.get("review_note", ""),
                "duration_sec": float(item["duration_sec"]),
                "bytes": video.stat().st_size,
                "quality_review_basis": item["quality_review_basis"],
                "split": "unlabeled_review",
            })
    result = pd.DataFrame(rows)
    if result.clip_id.duplicated().any():
        raise ValueError("Duplicate clip_id in accepted collection")
    result = result.sort_values(["sport_label", "source_dataset", "event_group_id", "clip_id"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"clips={len(result)} events={result.event_group_id.nunique()} "
          f"size_mib={result.bytes.sum()/2**20:.2f} output={OUTPUT}")
    print(result.groupby(["sport_label", "source_dataset"]).size().to_string())


if __name__ == "__main__":
    main()
