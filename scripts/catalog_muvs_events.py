"""Summarize MUVS events and one candidate video URL per event.

Names supply review hints only; they never become verified age/gender labels.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


BASE = "https://storage.googleapis.com/dataset-ugv-sports/public/RAW_DATA"
METADATA_URL = "https://zenodo.org/api/records/20708683/files/data.zip/content"
METADATA_BYTES = 9_742_259


def ensure_metadata(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(METADATA_URL, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(2**20):
                handle.write(chunk)
    if temporary.stat().st_size != METADATA_BYTES:
        temporary.unlink(missing_ok=True)
        raise IOError("MUVS metadata download has an unexpected size")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/raw/muvs/data.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/muvs_event_candidates.csv"))
    parser.add_argument("--probe", action="store_true", help="HEAD one video per event")
    args = parser.parse_args()
    ensure_metadata(args.zip)
    rows = []
    session = requests.Session() if args.probe else None
    with zipfile.ZipFile(args.zip) as archive:
        for name in sorted(archive.namelist()):
            if "/fragments/" not in name or not name.endswith("_FRAGMENTS.csv"):
                continue
            data = pd.read_csv(io.BytesIO(archive.read(name)),
                               usecols=["sport_genre", "event_id", "camera_id", "period"])
            choices = data[["sport_genre", "event_id", "camera_id", "period"]]
            choices = choices.drop_duplicates().sort_values(["period", "camera_id"])
            first = choices.iloc[0]
            event = str(first.event_id)
            url = (f"{BASE}/{first.sport_genre}/{event}/{first.camera_id}"
                   f"/VIDEOS/P{int(first.period)}/video.mp4")
            status = size = None
            if session is not None:
                try:
                    response = session.head(url, timeout=15)
                    status = response.status_code
                    size = int(response.headers.get("Content-Length", 0)) or None
                except requests.RequestException:
                    status = 0
            rows.append({
                "source_dataset": "MUVS",
                "event_id": event,
                "source_split": name.split("/")[2],
                "sport_label": "football" if first.sport_genre == "soccer" else first.sport_genre,
                "n_period_camera_files": len(choices),
                "candidate_gender": "female" if "fem_youth" in event.lower() else None,
                "candidate_age": "youth" if any(word in event.lower()
                                                 for word in ("kids", "youth")) else None,
                "sample_video_url": url,
                "sample_video_http_status": status,
                "sample_video_bytes": size,
                "gender_category": None,
                "age_category": None,
                "review_status": "unreviewed",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"events={len(rows)} youth_hints={sum(bool(r['candidate_age']) for r in rows)} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
