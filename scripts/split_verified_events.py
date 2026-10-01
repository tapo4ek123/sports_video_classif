"""Create a group-safe split only after all requested event cells are covered.

Default policy requires at least five independent events in each of the 16
sport/gender/age cells.  A coverage report is always written; an incomplete
dataset does not produce a misleading three-head train/test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPORTS = ("basketball", "football", "volleyball", "tennis")
GENDERS = ("female", "male")
AGES = ("adult", "youth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/verified_youtube_accepted.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/verified_youtube_event_split.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/metadata/verified_youtube_coverage.json"))
    parser.add_argument("--min-events-per-cell", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_events_per_cell < 3:
        parser.error("at least 3 events per cell are needed for train/validation/test")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty manifest")
    by_event: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cell = (row["sport_label"], row["gender_category"], row["age_category"])
        if cell[0] not in SPORTS or cell[1] not in GENDERS or cell[2] not in AGES:
            raise ValueError(f"invalid or missing cell: {cell}")
        if not row.get("event_id") or not row.get("source_video_id"):
            raise ValueError("every row needs event_id and source_video_id")
        by_event[row["event_id"]].append(row)
        by_source[row["source_video_id"]].add(row["event_id"])
    if any(len(events) > 1 for events in by_source.values()):
        raise ValueError("one source video maps to multiple events; resolve before splitting")
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for event_id, group in by_event.items():
        labels = {(r["sport_label"], r["gender_category"], r["age_category"]) for r in group}
        if len(labels) != 1:
            raise ValueError(f"inconsistent labels within event {event_id}")
        cells[next(iter(labels))].append(event_id)
    coverage = {"/".join(cell): len(cells[cell]) for cell in
                ((s, g, a) for s in SPORTS for g in GENDERS for a in AGES)}
    missing = {key: args.min_events_per_cell - count for key, count in coverage.items()
               if count < args.min_events_per_cell}
    report = {"n_clips": len(rows), "n_events": len(by_event),
              "n_source_videos": len(by_source), "min_events_per_cell": args.min_events_per_cell,
              "coverage": coverage, "additional_events_needed_by_cell": missing,
              "ready_for_three_head_split": not missing}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if missing:
        print(f"NOT READY: {len(missing)}/16 cells below {args.min_events_per_cell} events; report={args.report}")
        print("missing_total=" + str(sum(missing.values())))
        return
    assignments = {}
    for cell, event_ids in cells.items():
        ordered = sorted(event_ids, key=lambda event_id:
                         hashlib.sha256(f"{args.seed}:{event_id}".encode()).hexdigest())
        test_count = max(1, round(len(ordered) * .2))
        val_count = max(1, round(len(ordered) * .2))
        if test_count + val_count >= len(ordered):
            raise ValueError(f"insufficient train events for {cell}")
        for index, event_id in enumerate(ordered):
            assignments[event_id] = ("test" if index < test_count else
                                     "validation" if index < test_count + val_count else "train")
    for row in rows:
        row["split"] = assignments[row["event_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print("READY", Counter(assignments.values()), args.output)


if __name__ == "__main__":
    main()
