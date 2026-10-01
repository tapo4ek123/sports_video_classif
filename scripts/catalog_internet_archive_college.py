"""Catalog additional men's/women's match candidates on Internet Archive."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from catalog_internet_archive import SEARCH_URL, SPORT_QUERIES, fetch_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-query", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/metadata/internet_archive_college_candidates.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; preserve manual reviews")
    if not 1 <= args.rows_per_query <= 100 or not 1 <= args.workers <= 8:
        parser.error("rows must be 1..100 and workers 1..8")
    catalog: dict[str, dict] = {}
    for sport, sport_term in SPORT_QUERIES.items():
        for gender, gender_term in (("female", "Women's"), ("male", "Men's")):
            phrase = f"{gender_term} {sport_term}"
            response = requests.get(
                SEARCH_URL,
                params={"q": f'title:"{phrase}" AND title:vs AND mediatype:movies',
                        "fl[]": ["identifier", "title", "date"],
                        "rows": args.rows_per_query, "sort[]": "date desc", "output": "json"},
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()["response"]
            print(f"{sport}/{gender}: matches={payload['numFound']}", flush=True)
            for doc in payload["docs"]:
                catalog.setdefault(doc["identifier"], dict(doc, sport_label=sport,
                    candidate_gender=gender, candidate_age="", search_phrase=phrase))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_metadata, item) for item in catalog.values()]
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
