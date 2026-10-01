"""Search YouTube metadata for tournament videos, with strict candidate filters.

Rows remain *candidates*: a tournament title and channel do not prove that a
particular match appears in the video.  No videos are downloaded here.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import yt_dlp
except ImportError as exc:
    raise SystemExit("Install optional dependency: pip install -r requirements-acquisition.txt") from exc


FIELDS = ["query_id", "source_dataset", "youtube_id", "source_url", "title", "channel",
          "duration_sec", "sport_label", "candidate_gender", "candidate_age", "competition_url",
          "event_id", "match_evidence_url", "video_review_status", "review_status"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("data/metadata/youtube_search_queries.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/youtube_tournament_candidates.csv"))
    parser.add_argument("--max-per-query", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.max_per_query <= 100:
        parser.error("max-per-query must be 1..100")
    with args.queries.open(newline="", encoding="utf-8") as stream:
        queries = list(csv.DictReader(stream))
    rows: dict[str, dict] = {}
    options = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
               "skip_download": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(options) as loader:
        for rule in queries:
            limit = min(int(rule.get("limit") or args.max_per_query), args.max_per_query)
            try:
                result = loader.extract_info(f"ytsearch{limit}:{rule['query']}", download=False)
            except yt_dlp.utils.DownloadError as error:
                print(f"FAIL {rule['query_id']}: {str(error)[-160:]}", flush=True)
                continue
            accepted = 0
            for item in (result or {}).get("entries") or []:
                if not item or not item.get("id"):
                    continue
                title = item.get("title") or ""
                channel = item.get("channel") or item.get("uploader") or ""
                if rule["required_title"].casefold() not in title.casefold():
                    continue
                if rule["forbidden_title"] and rule["forbidden_title"].casefold() in title.casefold():
                    continue
                if rule["required_channel"] and rule["required_channel"].casefold() not in channel.casefold():
                    continue
                video_id = item["id"]
                if not re.fullmatch(r"[\w-]{11}", video_id):
                    continue
                rows.setdefault(video_id, {
                    "query_id": rule["query_id"], "source_dataset": "YouTubeSearch",
                    "youtube_id": video_id,
                    "source_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title, "channel": channel,
                    "duration_sec": item.get("duration") or "",
                    "sport_label": rule["sport_label"],
                    "candidate_gender": rule["candidate_gender"],
                    "candidate_age": rule["candidate_age"],
                    "competition_url": rule["competition_url"],
                    "event_id": "", "match_evidence_url": "",
                    "video_review_status": "not_downloaded",
                    "review_status": "needs_match_evidence",
                })
                accepted += 1
            print(f"{rule['query_id']}: search={limit} accepted={accepted}", flush=True)
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as stream:
            prior = list(csv.DictReader(stream))
        for item in prior:
            if item["youtube_id"] in rows:
                rows[item["youtube_id"]].update({name: item[name] for name in
                                                  ("event_id", "match_evidence_url", "video_review_status", "review_status")
                                                  if item.get(name)})
            elif item.get("review_status") != "needs_match_evidence":
                rows[item["youtube_id"]] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (row["sport_label"], row["query_id"], row["title"])))
    temporary.replace(args.output)
    print(f"unique_candidates={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
