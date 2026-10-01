"""Catalog original MultiSports source-video titles via YouTube oEmbed.

Titles are candidate evidence only. They do not establish rights to the video or
verified competition labels, and should not be used by the visual classifier.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/raw/multisports/data/trainval/multisports_GT.pkl")
    parser.add_argument("--output", type=Path, default=Path("data/metadata/multisports_sources.csv"))
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()
    # This pickle is downloaded from the dataset authors' gated repository.
    with open(args.annotation, "rb") as handle:
        annotation = pickle.load(handle)
    groups: dict[tuple[str, str], set[str]] = {}
    for split, key in (("train", "train_videos"), ("test", "test_videos")):
        for clip in annotation[key][0]:
            sport, name = clip.split("/", 1)
            source_id = name.removeprefix("v_").rsplit("_c", 1)[0]
            groups.setdefault((sport, source_id), set()).add(split)
    existing = {}
    if args.output.exists():
        old = pd.read_csv(args.output).fillna("")
        existing = {(r.sport_label, r.youtube_id): r._asdict()
                    for r in old.itertuples(index=False)}
    session = requests.Session()
    rows = []
    for index, ((sport, source_id), splits) in enumerate(sorted(groups.items()), 1):
        if (sport, source_id) in existing:
            rows.append(existing[(sport, source_id)])
            continue
        url = f"https://www.youtube.com/watch?v={source_id}"
        try:
            response = session.get("https://www.youtube.com/oembed",
                                   params={"url": url, "format": "json"},
                                   timeout=20)
            payload = response.json() if response.ok else {}
            status = response.status_code
        except requests.RequestException:
            payload, status = {}, 0
        rows.append({"sport_label": sport, "youtube_id": source_id,
                     "source_video_url": url, "source_split": ",".join(sorted(splits)),
                     "title": payload.get("title"), "channel": payload.get("author_name"),
                     "metadata_status": status,
                     "gender_category": None, "age_category": None,
                     "label_evidence_url": None, "review_status": "unreviewed"})
        if index % 20 == 0:
            print(f"checked {index}/{len(groups)}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
        time.sleep(args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"sources={len(rows)} titles={sum(bool(r.get('title')) for r in rows)} output={args.output}")


if __name__ == "__main__":
    main()
