"""Extract short Internet Archive clips from a reviewed list of item identifiers."""
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
    parser.add_argument("--catalog", type=Path, default=Path("data/metadata/internet_archive_candidates.csv"))
    parser.add_argument("--ids", nargs="*", default=[], help="Archive item identifiers, chosen after title review")
    parser.add_argument("--ids-file", type=Path, help="One Archive identifier per line; # comments allowed")
    parser.add_argument("--duration-sec", type=float, default=5)
    parser.add_argument("--start-sec", type=float, default=-1,
                        help="Fixed start, or -1 for a fraction of known video duration")
    parser.add_argument("--start-fraction", type=float, default=0.35)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("data/processed/internet_archive_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/internet_archive_sample.csv"))
    args = parser.parse_args()
    identifiers = list(args.ids)
    if args.ids_file:
        identifiers.extend(line.strip() for line in args.ids_file.read_text().splitlines()
                           if line.strip() and not line.lstrip().startswith("#"))
    if not identifiers:
        parser.error("provide --ids or --ids-file")
    if args.duration_sec <= 0 or args.start_sec < -1 or not 0 < args.start_fraction < 1:
        parser.error("duration must be positive and start must be -1 or nonnegative")
    catalog = pd.read_csv(args.catalog).set_index("identifier")
    rows = pd.read_csv(args.manifest).to_dict("records") if args.manifest.exists() else []
    known = {row["clip_id"] for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    for identifier in identifiers:
        if identifier not in catalog.index:
            print(f"SKIP {identifier}: absent from catalog", flush=True)
            continue
        item = catalog.loc[identifier]
        if not isinstance(item.video_url, str) or not item.video_url.startswith("https://"):
            print(f"SKIP {identifier}: no video URL", flush=True)
            continue
        source_duration = float(item.video_duration_sec) if pd.notna(item.video_duration_sec) else 0
        start = args.start_sec if args.start_sec >= 0 else (args.start_fraction * source_duration
                                                            if source_duration > 120 else 30)
        clip_id = f"IA:{identifier}:{start:g}-{start + args.duration_sec:g}"
        safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", identifier)[:120]
        target_dir = args.output / str(item.sport_label)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{safe_id}_{start:g}s_{args.duration_sec:g}s.mp4"
        if not target.exists():
            temporary = target.with_name(target.stem + ".part.mp4")
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                       "-i", item.video_url, "-t", str(args.duration_sec), "-vf", "scale=640:-2",
                       "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-y", str(temporary)]
            began = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=args.timeout_sec)
                if result.returncode != 0:
                    print(f"FAIL {identifier}: ffmpeg {result.stderr[-300:]!r}", flush=True)
                    continue
                temporary.replace(target)
            except subprocess.TimeoutExpired:
                print(f"FAIL {identifier}: timeout {args.timeout_sec}s", flush=True)
                continue
            finally:
                temporary.unlink(missing_ok=True)
            print(f"downloaded {identifier} in {time.monotonic()-began:.1f}s", flush=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                                "-of", "json", str(target)], capture_output=True, text=True,
                               timeout=15, check=True)
        info = json.loads(probe.stdout)["format"]
        if float(info["duration"]) < args.duration_sec * 0.9:
            print(f"FAIL {identifier}: short clip {info['duration']}s", flush=True)
            target.unlink(missing_ok=True)
            continue
        if clip_id not in known:
            rows.append({"clip_id": clip_id, "source_dataset": "InternetArchive",
                         "source_video_id": item.source_video_id,
                         "source_url": item.source_url,
                         "video_path": target.as_posix(),
                         "sport_label": item.sport_label,
                         "candidate_gender": item.candidate_gender,
                         "candidate_age": item.candidate_age,
                         "gender_category": None, "age_category": None,
                         "license_url": item.license_url,
                         "duration_sec": float(info["duration"]),
                         "bytes": int(info["size"]), "split": "unlabeled_review"})
            known.add(clip_id)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.manifest, index=False)
        print(f"OK {identifier}: {info['duration']}s {int(info['size'])/2**20:.2f} MiB", flush=True)
    print(f"manifest={args.manifest} clips={len(rows)}")


if __name__ == "__main__":
    main()
