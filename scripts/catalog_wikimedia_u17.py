"""Catalog Wikimedia Commons U17 football videos without downloading media."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/metadata/wikimedia_u17_football.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; preserve manual reviews")
    response = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": "U17 football filetype:video", "gsrnamespace": 6,
            "gsrlimit": 50, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        },
        headers={"User-Agent": "MIPT-sports-video-research/0.1 (educational catalog)"},
        timeout=30,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    rows = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        rows.append({
            "source_dataset": "WikimediaCommons",
            "source_video_id": f"Commons:{page['title']}",
            "title": page["title"],
            "source_url": info.get("descriptionurl", ""),
            "video_url": info.get("url", ""),
            "mime": info.get("mime", ""),
            "video_bytes": info.get("size", 0),
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "sport_label": "football",
            "candidate_age": "junior",
            "gender_category": "",
            "age_category": "",
            "review_status": "unreviewed",
        })
    frame = pd.DataFrame(rows).sort_values("title")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"candidates={len(frame)} video_files={sum(frame.mime.str.startswith('video/'))} output={args.output}")


if __name__ == "__main__":
    main()
