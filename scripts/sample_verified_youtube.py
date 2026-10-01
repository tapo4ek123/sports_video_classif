"""Download short, silent samples from manually linked competition videos.

The input CSV is a reviewed event catalog, not a search result.  The script checks
video identity and availability but cannot decide whether the selected seconds
show play; every output remains in needs_visual_review until inspected.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError as exc:
    raise SystemExit("Install optional dependency: pip install -r requirements-acquisition.txt") from exc


FIELDS = [
    "clip_id", "event_id", "source_dataset", "source_video_id", "source_url",
    "video_path", "title", "channel", "sport_label", "gender_category",
    "age_category", "competition_url", "event_evidence_url", "label_source",
    "label_review_status", "video_review_status", "start_sec", "duration_sec",
    "width", "height", "bytes", "split",
]


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_type,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    return {"duration_sec": float(info["format"]["duration"]),
            "bytes": int(info["format"]["size"]),
            "width": int(video["width"]), "height": int(video["height"])}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/metadata/verified_youtube_events.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/verified_youtube_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/verified_youtube_sample"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all catalog rows")
    parser.add_argument("--sport", choices=["basketball", "football", "volleyball", "tennis"])
    parser.add_argument("--event-id", action="append", help="Sample only the named event; repeatable")
    parser.add_argument("--timeout-sec", type=int, default=45)
    args = parser.parse_args()
    with args.catalog.open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    if args.sport:
        candidates = [r for r in candidates if r["sport_label"] == args.sport]
    if args.event_id:
        candidates = [r for r in candidates if r["event_id"] in set(args.event_id)]
    if args.limit > 0:
        candidates = candidates[:args.limit]
    if len({r["event_id"] for r in candidates}) != len(candidates):
        parser.error("event_id must be unique in the selected catalog")
    rows = []
    if args.manifest.exists():
        with args.manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    known = {r["clip_id"] for r in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    options = {"quiet": True, "no_warnings": True, "skip_download": True,
               "noplaylist": True, "extract_flat": False}
    for item in candidates:
        video_id = item["youtube_id"]
        start = float(item["start_sec"])
        duration = float(item.get("duration_sec") or 5)
        clip_id = f"YT:{video_id}:{start:g}-{start + duration:g}"
        if clip_id in known:
            print(f"SKIP existing {clip_id}", flush=True)
            continue
        try:
            with yt_dlp.YoutubeDL(options) as loader:
                info = loader.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if info["id"] != video_id or info.get("availability") not in (None, "public", "unlisted"):
                raise ValueError("video identity or availability mismatch")
            title = info.get("title") or ""
            channel = info.get("channel") or info.get("uploader") or ""
            for expected, actual, field in ((item["title_contains"], title, "title"),
                                            (item["channel_contains"], channel, "channel")):
                if expected.casefold() not in actual.casefold():
                    raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")
            if start < 0 or duration <= 0 or start + duration > float(info.get("duration") or 0):
                raise ValueError("sample range exceeds video duration")
            target = args.output / item["sport_label"] / f"{video_id}_{start:g}s_{duration:g}s.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_name(target.stem + ".part.mp4")
                command = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings",
                           "--no-progress", "--no-part", "--force-overwrites",
                           "-f", "134/133/160/bestvideo[ext=mp4][height<=360]",
                           "--download-sections", f"*{start:g}-{start + duration:g}",
                           "--force-keyframes-at-cuts", "-o", str(temporary),
                           f"https://www.youtube.com/watch?v={video_id}"]
                try:
                    result = subprocess.run(command, capture_output=True, text=True,
                                            timeout=args.timeout_sec)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-500:])
                    details = probe(temporary)
                    if details["duration_sec"] < duration * .8 or details["bytes"] < 10_000:
                        raise ValueError(f"invalid output: {details}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            details = probe(target)
            rows.append({
                "clip_id": clip_id, "event_id": item["event_id"],
                "source_dataset": item["source_dataset"], "source_video_id": f"YouTube:{video_id}",
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_path": target.as_posix(), "title": title, "channel": channel,
                "sport_label": item["sport_label"], "gender_category": item["gender_category"],
                "age_category": item["age_category"], "competition_url": item["competition_url"],
                "event_evidence_url": item["event_evidence_url"],
                "label_source": "organizer_event_page_plus_video_title",
                "label_review_status": "source_matched", "video_review_status": "needs_visual_review",
                "start_sec": f"{start:g}", **details, "split": "unassigned",
            })
            known.add(clip_id)
            write_rows(args.manifest, rows)
            print(f"OK {clip_id} {details['bytes']/2**20:.2f} MiB", flush=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError,
                RuntimeError, yt_dlp.utils.DownloadError) as error:
            print(f"FAIL {video_id}: {str(error)[-350:]}", flush=True)
    print(f"manifest={args.manifest} clips={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
