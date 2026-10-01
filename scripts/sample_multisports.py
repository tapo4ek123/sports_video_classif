"""Download a small, group-diverse MultiSports sample through HTTP byte ranges.

The official sport archives are uncompressed tar files. Reading their headers and
selected members by range avoids downloading tens of gigabytes for a smoke test.
Requires access to the gated, noncommercial dataset; the token is never saved.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import tarfile
from pathlib import Path

import cv2
import pandas as pd
import requests
from huggingface_hub import hf_hub_url


REPO = "MCG-NJU/MultiSports"
REVISION = "01600ce7eabbf42a5ee7c82b82f49a11597b3a5f"
SPORTS = ("basketball", "football", "volleyball")
GROUP_PATTERN = re.compile(r"^(v_.+)_c\d+\.mp4$")


def get_range(session: requests.Session, url: str, start: int, size: int,
              destination: Path | None = None) -> bytes | None:
    """Read exactly one byte interval, optionally to an atomic local file."""
    end = start + size - 1
    with session.get(url, headers={"Range": f"bytes={start}-{end}"},
                     stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Server ignored Range request: HTTP {response.status_code}")
        returned = response.headers.get("Content-Range", "")
        if not returned.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Unexpected Content-Range: {returned}")
        if destination is None:
            data = response.content
            if len(data) != size:
                raise IOError(f"Short range response: {len(data)} != {size}")
            return data
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        written = 0
        try:
            with temporary.open("wb") as handle:
                for block in response.iter_content(2**20):
                    handle.write(block)
                    written += len(block)
            if written != size:
                raise IOError(f"Short video download: {written} != {size}")
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return None


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or frames < 1:
        raise ValueError(f"No frames in {path}")
    return {"duration_sec": frames / fps if fps else None,
            "fps": fps, "width": width, "height": height}


def sample_sport(session: requests.Session, sport: str, subset: str, count: int,
                 output: Path, max_bytes: int) -> list[dict]:
    archive = hf_hub_url(REPO, f"data/{subset}/{sport}.tar", repo_type="dataset",
                         revision=REVISION)
    offset = 0
    rows: list[dict] = []
    groups: set[str] = set()
    bytes_used = 0
    while len(rows) < count:
        header = get_range(session, archive, offset, 512)
        assert header is not None
        if not any(header):
            break
        member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        data_start = offset + 512
        offset = data_start + ((member.size + 511) // 512) * 512
        filename = Path(member.name).name
        match = GROUP_PATTERN.match(filename)
        if not member.isfile() or not match or match.group(1) in groups:
            continue
        if bytes_used + member.size > max_bytes:
            print(f"{sport}: byte budget reached after {len(rows)} clips", flush=True)
            break
        group = match.group(1)
        target = output / sport / filename
        if not target.exists() or target.stat().st_size != member.size:
            get_range(session, archive, data_start, member.size, target)
        properties = probe_video(target)
        groups.add(group)
        bytes_used += member.size
        rows.append({
            "clip_id": f"multisports:{sport}:{target.stem}",
            "source_dataset": "MultiSports",
            "source_video_id": f"MultiSports:{sport}:{group}",
            "event_id": None,
            "video_path": str(target.as_posix()),
            "original_label": sport,
            "sport_label": sport,
            "gender_category": None,
            "age_category": None,
            "sport_label_source": "MultiSports archive directory",
            "gender_label_source": None,
            "age_label_source": None,
            **properties,
            "split": "external_test" if subset == "test" else "train_candidate",
        })
        print(f"{sport}: {len(rows)}/{count} {filename} {member.size / 2**20:.1f} MiB", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["trainval", "test"], default="test")
    parser.add_argument("--sports", nargs="+", choices=SPORTS, default=list(SPORTS))
    parser.add_argument("--per-sport", type=int, default=10)
    parser.add_argument("--max-mb-per-sport", type=int, default=400)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/multisports_sample"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/metadata/multisports_sample.csv"))
    args = parser.parse_args()
    if args.per_sport < 1 or args.max_mb_per_sport < 1:
        parser.error("per-sport and max-mb-per-sport must be positive")
    token = os.environ.get("HF_TOKEN") or getpass.getpass("HF token: ")
    if args.subset == "trainval" and args.output == Path("data/processed/multisports_sample"):
        args.output = Path("data/processed/multisports_train_sample")
    if args.subset == "trainval" and args.manifest == Path("data/metadata/multisports_sample.csv"):
        args.manifest = Path("data/metadata/multisports_train_sample.csv")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    rows = []
    for sport in args.sports:
        rows.extend(sample_sport(session, sport, args.subset, args.per_sport, args.output,
                                 args.max_mb_per_sport * 2**20))
    if not rows:
        raise RuntimeError("No videos downloaded")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(f"manifest={args.manifest} videos={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
