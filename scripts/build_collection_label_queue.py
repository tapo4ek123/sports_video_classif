"""Prepare event-level review rows from the collected video pilot."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT = Path("data/metadata/collection_label_queue.csv")
ARCHIVE_CATALOGS = [
    Path("data/metadata/internet_archive_candidates.csv"),
    Path("data/metadata/internet_archive_college_candidates.csv"),
    Path("data/metadata/internet_archive_middle_school_soccer.csv"),
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} exists; preserve any manual event reviews")
    clips = pd.read_csv("data/metadata/collection_accepted.csv").fillna("")
    archive = pd.concat([pd.read_csv(path).fillna("") for path in ARCHIVE_CATALOGS])
    archive = archive.drop_duplicates("source_video_id").set_index("source_video_id")
    commons = pd.read_csv("data/metadata/wikimedia_u17_football.csv").fillna("")
    commons = commons.set_index("source_video_id")
    rows = []
    for event_id, group in clips.groupby("event_group_id", sort=False):
        first = group.iloc[0]
        if first.source_dataset == "InternetArchive":
            title = archive.loc[event_id, "title"]
            description = str(archive.loc[event_id, "description"])
        elif first.source_dataset == "WikimediaCommons":
            title = commons.loc[event_id, "title"]
            description = ""
        else:
            title = event_id.removeprefix("MUVS:").replace("_", " ")
            description = ""
        rows.append({
            "event_group_id": event_id,
            "source_dataset": first.source_dataset,
            "source_url": first.source_url,
            "title": title,
            "description": description,
            "sport_label": first.sport_label,
            "candidate_gender": first.candidate_gender,
            "candidate_age": first.candidate_age,
            "clip_count": len(group),
            "sample_path": first.video_path,
            "gender_category": first.gender_category,
            "age_category": first.age_category,
            "age_detail_raw": getattr(first, "age_detail_raw", ""),
            "age_review_status": getattr(first, "age_review_status", ""),
            "age_rule_url": getattr(first, "age_rule_url", ""),
            "label_source": getattr(first, "label_source", ""),
            "reviewed_at": getattr(first, "reviewed_at", ""),
            "video_evidence_url": getattr(first, "video_evidence_url", ""),
            "official_event_status": getattr(first, "official_event_status", ""),
            "review_note": getattr(first, "review_note", ""),
            "gender_evidence_url": getattr(first, "video_evidence_url", ""),
            "age_evidence_url": getattr(first, "video_evidence_url", ""),
            "review_status": ("owner_age_ambiguous" if getattr(first, "age_review_status", "") == "visual_ambiguous_event"
                              else "owner_age_tentative" if getattr(first, "age_review_status", "") == "visual_tentative"
                              else "owner_video_reviewed" if first.gender_category or first.age_category
                              else "needs_event_evidence"),
        })
    result = pd.DataFrame(rows)
    result = result.sort_values(["candidate_age", "sport_label", "candidate_gender", "event_group_id"],
                                ascending=[False, True, True, True])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"events={len(result)} youth_hints={sum(result.candidate_age == 'youth')} output={OUTPUT}")


if __name__ == "__main__":
    main()
