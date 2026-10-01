"""Extract one short clip from each selected Commons U17 football match."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title-contains", nargs="+", required=True,
                        help="Unique substrings of catalog file titles")
    parser.add_argument("--start-sec", type=float, default=60)
    parser.add_argument("--duration-sec", type=float, default=5)
    parser.add_argument("--timeout-sec", type=int, default=100)
    parser.add_argument("--catalog", type=Path, default=Path("data/metadata/wikimedia_u17_football.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/wikimedia_u17_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/wikimedia_u17_sample.csv"))
    args = parser.parse_args()
    if args.start_sec < 0 or args.duration_sec <= 0:
        parser.error("start must be nonnegative and duration positive")
    catalog = pd.read_csv(args.catalog).fillna("")
    rows = pd.read_csv(args.manifest).fillna("").to_dict("records") if args.manifest.exists() else []
    seen = {row["clip_id"] for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in args.title_contains:
        matches = catalog[catalog.title.str.contains(pattern, case=False, regex=False)]
        if len(matches) != 1:
            print(f"SKIP {pattern}: {len(matches)} title matches", flush=True)
            continue
        item = matches.iloc[0]
        if item.review_status == "irrelevant_title":
            print(f"SKIP {pattern}: irrelevant title", flush=True)
            continue
        source_id = str(item.source_video_id)
        clip_id = f"{source_id}:{args.start_sec:g}-{args.start_sec + args.duration_sec:g}"
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", str(item.title))[:85]
        target = args.output / f"{name}_{args.start_sec:g}s_{args.duration_sec:g}s.mp4"
        if not target.exists():
            temporary = target.with_name(target.stem + ".part.mp4")
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(args.start_sec),
                       "-i", str(item.video_url), "-t", str(args.duration_sec),
                       "-vf", "scale=640:-2", "-an", "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", "28", "-y", str(temporary)]
            start = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=args.timeout_sec)
                if result.returncode != 0:
                    print(f"FAIL {pattern}: {result.stderr[-300:]!r}", flush=True)
                    continue
                temporary.replace(target)
            except subprocess.TimeoutExpired:
                print(f"FAIL {pattern}: timeout {args.timeout_sec}s", flush=True)
                continue
            finally:
                temporary.unlink(missing_ok=True)
            print(f"downloaded {pattern} in {time.monotonic()-start:.1f}s", flush=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                                "-of", "json", str(target)], capture_output=True, text=True,
                               timeout=15, check=True)
        info = json.loads(probe.stdout)["format"]
        if float(info["duration"]) < args.duration_sec * 0.9:
            print(f"FAIL {pattern}: short clip", flush=True)
            target.unlink(missing_ok=True)
            continue
        if clip_id not in seen:
            rows.append({"clip_id": clip_id, "source_dataset": "WikimediaCommons",
                         "source_video_id": source_id, "source_url": item.source_url,
                         "video_path": target.as_posix(), "sport_label": "football",
                         "candidate_age": "junior", "candidate_gender": "",
                         "gender_category": "", "age_category": "",
                         "license_url": item.license_url,
                         "duration_sec": float(info["duration"]), "bytes": int(info["size"]),
                         "split": "unlabeled_review", "review_status": "",
                         "quality_review_basis": ""})
            seen.add(clip_id)
            pd.DataFrame(rows).to_csv(args.manifest, index=False)
        print(f"OK {pattern}: {info['duration']}s {int(info['size'])/2**20:.2f} MiB", flush=True)
    print(f"manifest={args.manifest} clips={len(rows)}")


if __name__ == "__main__":
    main()
