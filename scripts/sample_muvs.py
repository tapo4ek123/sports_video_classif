"""Extract short MUVS MP4 samples via HTTP Range without full period downloads."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from catalog_muvs_events import BASE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/metadata/muvs_event_candidates.csv")
    parser.add_argument("--events", nargs="+", default=["naperville_youth_match_20240608"])
    parser.add_argument("--start-sec", type=float, default=30)
    parser.add_argument("--duration-sec", type=float, default=5)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--period", type=int, help="Use a specified MUVS period instead of catalog's first")
    parser.add_argument("--camera", help="Camera for --period, for example cam_1")
    parser.add_argument("--output", type=Path, default=Path("data/processed/muvs_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/muvs_sample.csv"))
    args = parser.parse_args()
    if args.start_sec < 0 or args.duration_sec <= 0:
        parser.error("start-sec must be nonnegative and duration-sec positive")
    if bool(args.period) != bool(args.camera):
        parser.error("--period and --camera must be provided together")
    catalog = pd.read_csv(args.catalog).set_index("event_id")
    if args.manifest.exists():
        rows = pd.read_csv(args.manifest).to_dict("records")
    else:
        rows = []
    seen = {item["clip_id"] for item in rows}
    for event in args.events:
        if event not in catalog.index:
            parser.error(f"Unknown MUVS event: {event}")
        item = catalog.loc[event]
        if not args.period and item.sample_video_http_status != 200:
            raise RuntimeError(f"Video not accessible for {event}")
        video_url = (f"{BASE}/{'soccer' if item.sport_label == 'football' else item.sport_label}"
                     f"/{event}/{args.camera}/VIDEOS/P{args.period}/video.mp4"
                     if args.period else str(item.sample_video_url))
        source_segment = f":P{args.period}:{args.camera}" if args.period else ""
        clip_id = f"MUVS:{event}{source_segment}:{args.start_sec:g}-{args.start_sec + args.duration_sec:g}"
        suffix = f"_P{args.period}_{args.camera}" if args.period else ""
        target = args.output / str(item.sport_label) / f"{event}{suffix}_{args.start_sec:g}s_{args.duration_sec:g}s.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.stem + ".part.mp4")
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(args.start_sec),
                       "-i", video_url, "-t", str(args.duration_sec),
                       "-vf", "scale=640:-2", "-an", "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", "28", "-y", str(temporary)]
            try:
                subprocess.run(command, check=True, timeout=args.timeout_sec)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                                "-of", "json", str(target)], capture_output=True, text=True,
                               check=True, timeout=15)
        info = json.loads(probe.stdout)["format"]
        if float(info["duration"]) < args.duration_sec * 0.9:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Short extracted clip: {target}")
        if clip_id not in seen:
            rows.append({"clip_id": clip_id, "source_dataset": "MUVS",
                         "source_video_id": f"MUVS:{event}", "event_id": f"MUVS:{event}",
                         "source_url": video_url,
                         "source_period": args.period, "source_camera": args.camera,
                         "video_path": target.as_posix(), "original_label": item.sport_label,
                         "sport_label": item.sport_label,
                         "gender_category": None, "age_category": None,
                         "source_split": item.source_split,
                         "duration_sec": float(info["duration"]), "bytes": int(info["size"]),
                         "split": "unlabeled_review"})
            seen.add(clip_id)
        print(f"{event}: {info['duration']} s, {int(info['size']) / 2**20:.2f} MiB", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(f"manifest={args.manifest} clips={len(rows)}")


if __name__ == "__main__":
    main()
