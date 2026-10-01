"""Catalog sports videos from Internet Archive without downloading video files."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


SEARCH_URL = "https://archive.org/advancedsearch.php"
SPORT_QUERIES = {
    "basketball": "Basketball",
    "football": "Soccer",
    "volleyball": "Volleyball",
    "tennis": "Tennis",
}
GENDER_QUERIES = {"female": "Girls", "male": "Boys"}


def search(sport: str, gender: str, rows: int, *, youth: bool,
           licensed_only: bool) -> list[dict]:
    phrase = f"{'8th Grade ' if youth else ''}{GENDER_QUERIES[gender]} {SPORT_QUERIES[sport]}"
    query = f'title:"{phrase}" AND mediatype:movies'
    if licensed_only:
        query += " AND licenseurl:*"
    response = requests.get(
        SEARCH_URL,
        params={"q": query, "fl[]": ["identifier", "title", "licenseurl", "date"],
                "rows": rows, "output": "json", "sort[]": "date desc"},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()["response"]
    print(f"{sport}/{gender}/{'8th-grade' if youth else 'all'}: "
          f"matches={payload['numFound']} requested={rows}", flush=True)
    return [dict(item, sport_label=sport, candidate_gender=gender,
                 candidate_age="youth" if youth else "", search_phrase=phrase)
            for item in payload["docs"]]


def fetch_metadata(item: dict) -> dict:
    identifier = item["identifier"]
    try:
        response = requests.get(f"https://archive.org/metadata/{quote(identifier)}", timeout=25)
        response.raise_for_status()
        payload = response.json()
        metadata = payload.get("metadata", {})
        videos = [file for file in payload.get("files", [])
                  if file.get("name", "").lower().endswith(".mp4")
                  and file.get("source") in {"original", "derivative"}]
        videos.sort(key=lambda file: (file.get("source") != "original",
                                      -int(file.get("size") or 0)))
        video = videos[0] if videos else {}
        filename = video.get("name", "")
        return {
            "source_dataset": "InternetArchive",
            "source_video_id": f"IA:{identifier}",
            "identifier": identifier,
            "source_url": f"https://archive.org/details/{quote(identifier)}",
            "video_url": f"https://archive.org/download/{quote(identifier)}/{quote(filename)}" if filename else "",
            "title": metadata.get("title", item.get("title", "")),
            "description": str(metadata.get("description", "")),
            "creator": str(metadata.get("creator", "")),
            "date": str(metadata.get("date", item.get("date", ""))),
            "license_url": str(metadata.get("licenseurl", item.get("licenseurl", ""))),
            "sport_label": item["sport_label"],
            "candidate_gender": item["candidate_gender"],
            "candidate_age": item["candidate_age"],
            "gender_category": "",
            "age_category": "",
            "search_phrase": item["search_phrase"],
            "video_filename": filename,
            "video_bytes": int(video.get("size") or 0),
            "video_duration_sec": float(video.get("length") or 0),
            "review_status": "unreviewed",
            "error": "",
        }
    except (requests.RequestException, ValueError, KeyError) as error:
        return {"identifier": identifier, "sport_label": item["sport_label"],
                "candidate_gender": item["candidate_gender"], "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-query", type=int, default=60)
    parser.add_argument("--youth-rows-per-query", type=int, default=40)
    parser.add_argument("--licensed-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/metadata/internet_archive_candidates.csv"))
    args = parser.parse_args()
    if (args.rows_per_query < 1 or args.rows_per_query > 200
            or args.youth_rows_per_query < 1 or args.youth_rows_per_query > 200
            or args.workers < 1 or args.workers > 8):
        parser.error("row limits must be 1..200 and workers 1..8")
    if args.output.exists():
        parser.error(f"{args.output} already exists; preserve manual reviews")
    candidates: dict[str, dict] = {}
    for sport in SPORT_QUERIES:
        for gender in GENDER_QUERIES:
            for youth, limit in ((False, args.rows_per_query),
                                 (True, args.youth_rows_per_query)):
                for item in search(sport, gender, limit, youth=youth,
                                   licensed_only=args.licensed_only):
                    prior = candidates.get(item["identifier"])
                    if prior is None or (item["candidate_age"] and not prior["candidate_age"]):
                        candidates[item["identifier"]] = item
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_metadata, item) for item in candidates.values()]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 50 == 0:
                print(f"metadata={index}/{len(futures)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["sport_label", "candidate_gender", "identifier"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    valid = frame[(frame.error == "") & (frame.video_filename != "")]
    print(f"candidates={len(frame)} mp4={len(valid)} output={args.output}")


if __name__ == "__main__":
    main()
