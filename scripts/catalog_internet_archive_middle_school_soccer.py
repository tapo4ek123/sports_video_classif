"""Catalog junior soccer candidates missed by Girls/Boys Soccer title queries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from catalog_internet_archive import SEARCH_URL, fetch_metadata


def main() -> None:
    output = Path("data/metadata/internet_archive_middle_school_soccer.csv")
    if output.exists():
        raise FileExistsError(f"{output} exists; preserve manual reviews")
    docs = {}
    for gender, term in (("female", "Girls"), ("male", "Boys")):
        phrase = f"Middle School {term} Soccer"
        response = requests.get(SEARCH_URL, params={
            "q": f'title:"{phrase}" AND mediatype:movies',
            "fl[]": ["identifier", "title", "date"],
            "rows": 50, "output": "json"}, timeout=25)
        response.raise_for_status()
        payload = response.json()["response"]
        print(f"{gender}: matches={payload['numFound']}", flush=True)
        for doc in payload["docs"]:
            docs.setdefault(doc["identifier"], dict(doc, sport_label="football",
                candidate_gender=gender, candidate_age="junior",
                search_phrase=phrase))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_metadata, item) for item in docs.values()]
        rows = [future.result() for future in as_completed(futures)]
    frame = pd.DataFrame(rows).sort_values(["candidate_gender", "identifier"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"candidates={len(frame)} mp4={sum(frame.video_filename != '')} output={output}")


if __name__ == "__main__":
    main()
