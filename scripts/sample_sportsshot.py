"""Extract one SportsShot member with HTTP Range, then keep only a short sample.

Requires HF_TOKEN or a local Hugging Face login. The source ZIP and full MP4
are never saved in the project. Only ZIP method 8 (standard deflate) is handled.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
import zlib
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import get_token, hf_hub_url

from catalog_sportsshot import RemoteZip


REPO_ID = "MCG-NJU/SportsShot"


def range_stream(session: requests.Session, url: str, start: int, length: int):
    end = start + length - 1
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"},
                           stream=True, timeout=(20, 90))
    response.raise_for_status()
    expected = f"bytes {start}-{end}/"
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
        response.close()
        raise IOError(f"HTTP Range unavailable for {start}-{end}: {response.status_code}")
    return response


def extract_member_deflate(session: requests.Session, url: str, row: pd.Series,
                           destination: Path) -> None:
    offset = int(row.header_offset)
    with range_stream(session, url, offset, 30) as response:
        header = response.content
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", header)
    if signature != 0x04034B50 or method != 8 or flags & 1:
        raise ValueError("Unsupported ZIP member header")
    start = offset + 30 + name_len + extra_len
    remaining = int(row.compressed_bytes)
    decompressor = zlib.decompressobj(-15)
    count = 0
    checksum = 0
    with range_stream(session, url, start, remaining) as response, destination.open("wb") as target:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            remaining -= len(chunk)
            data = decompressor.decompress(chunk)
            target.write(data)
            count += len(data)
            checksum = zlib.crc32(data, checksum)
        data = decompressor.flush()
        target.write(data)
        count += len(data)
        checksum = zlib.crc32(data, checksum)
    if remaining or not decompressor.eof or count != int(row.member_bytes):
        raise IOError(f"Incomplete member: remaining={remaining}, bytes={count}")
    # The central-directory CRC is not yet part of the CSV index. Print it for an audit trail.
    print(f"downloaded_bytes={int(row.compressed_bytes)} mp4_bytes={count} crc32={checksum:08x}")


def copy_range(session: requests.Session, url: str, start: int, length: int,
               target, chunk_size: int = 1024 * 1024) -> None:
    remaining = length
    with range_stream(session, url, start, length) as response:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                target.write(chunk)
                remaining -= len(chunk)
    if remaining:
        raise IOError(f"Incomplete HTTP Range: missing {remaining} bytes")


def extract_member_deflate64(session: requests.Session, url: str, token: str,
                             row: pd.Series, destination: Path, scratch: Path) -> None:
    """Rebuild a sparse ZIP containing one member; system unzip handles Deflate64."""
    remote = RemoteZip(url, token, int(row.archive_bytes))
    try:
        with zipfile.ZipFile(remote) as archive:
            member = archive.getinfo(str(row.member_path))
            tail_start = archive.start_dir
        if member.compress_type != 9:
            raise ValueError("Expected ZIP method 9")
    finally:
        remote.close()
    offset = int(row.header_offset)
    with range_stream(session, url, offset, 30) as response:
        header = response.content
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", header)
    if signature != 0x04034B50 or method != 9 or flags & 1:
        raise ValueError("Unsupported ZIP member header")
    # Add room for a possible data descriptor following the compressed stream.
    member_length = min(int(row.archive_bytes) - offset,
                        30 + name_len + extra_len + int(row.compressed_bytes) + 32)
    sparse_zip = scratch / "selected_member.zip"
    with sparse_zip.open("wb") as target:
        target.truncate(int(row.archive_bytes))
        target.seek(offset)
        copy_range(session, url, offset, member_length, target)
        target.seek(tail_start)
        copy_range(session, url, tail_start, int(row.archive_bytes) - tail_start, target)
    with destination.open("wb") as output:
        result = subprocess.run(["unzip", "-p", str(sparse_zip), str(row.member_path)],
                                stdout=output, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"unzip failed: {result.stderr.strip()}")
    if destination.stat().st_size != int(row.member_bytes):
        raise IOError("Deflate64 decoded member has unexpected size")
    print(f"downloaded_bytes={member_length + int(row.archive_bytes) - tail_start} "
          f"mp4_bytes={destination.stat().st_size} method=deflate64")


def choose_start(row: pd.Series, duration: float, sample_duration: float,
                 annotation_dir: Path, mode: str) -> tuple[float, str]:
    midpoint = max(0.0, (duration - sample_duration) / 2)
    if mode != "full_view" or row.source_split not in {"train", "validation"}:
        return midpoint, "midpoint"
    annotations = annotation_dir / f"{row.source_split}.zip"
    label_name = Path(row.member_path).with_suffix(".txt").name
    if not annotations.exists():
        return midpoint, "midpoint_no_annotations"
    with zipfile.ZipFile(annotations) as archive:
        if label_name not in archive.namelist():
            return midpoint, "midpoint_no_annotations"
        labels = archive.read(label_name).decode().splitlines()
    frames = round(sample_duration * 25)
    starts = [index for index in range(0, len(labels) - frames + 1, 25)
              if all(label == "full_view" for label in labels[index:index + frames])]
    if not starts:
        return midpoint, "midpoint_no_full_view"
    start = min(starts, key=lambda index: abs(index / 25 - midpoint)) / 25
    return min(start, max(0.0, duration - sample_duration)), "annotated_full_view"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/metadata/sportsshot_archive_index.csv"))
    parser.add_argument("--source-youtube-id", required=True)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--sampling-mode", choices=["full_view", "midpoint"], default="full_view")
    parser.add_argument("--annotation-dir", type=Path,
                        default=Path("data/raw/sportsshot/metadata/GT/shot segmentation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sportsshot_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/sportsshot_sample.csv"))
    args = parser.parse_args()
    token = get_token()
    if not token:
        parser.error("HF_TOKEN or hf auth login is required")
    matches = pd.read_csv(args.index)
    matches = matches[(matches.source_youtube_id == args.source_youtube_id) &
                      (matches.compression_type.isin([8, 9]))]
    if matches.empty:
        parser.error("No supported ZIP member for this YouTube ID")
    row = matches.nsmallest(1, "compressed_bytes").iloc[0]
    output = args.output_dir / str(row.sport_label) / (Path(row.member_path).stem +
                                                       f"_{args.sampling_mode}_{args.duration_sec:g}s.mp4")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    url = hf_hub_url(REPO_ID, str(row.archive_path), repo_type="dataset")
    with requests.Session() as session, tempfile.TemporaryDirectory(prefix="sportsshot-") as temp_dir:
        session.headers["Authorization"] = f"Bearer {token}"
        source = Path(temp_dir) / "member.mp4"
        if int(row.compression_type) == 8:
            extract_member_deflate(session, url, row, source)
        else:
            extract_member_deflate64(session, url, token, row, source, Path(temp_dir))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", str(source)],
                               capture_output=True, text=True, check=True)
        duration = float(probe.stdout.strip())
        start, selection = choose_start(row, duration, args.duration_sec,
                                        args.annotation_dir, args.sampling_mode)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(start), "-i", str(source), "-t", str(args.duration_sec),
                        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                        str(output)], check=True)
    print(f"archive={row.archive_path} member={row.member_path} duration={duration:.2f}s "
          f"sample_start={start:.2f}s selection={selection} output={output} "
          f"sample_bytes={output.stat().st_size}")
    record = pd.DataFrame([{
        "source_dataset": "SportsShot",
        "source_youtube_id": row.source_youtube_id,
        "source_video_url": f"https://www.youtube.com/watch?v={row.source_youtube_id}",
        "archive_path": row.archive_path,
        "member_path": row.member_path,
        "original_split": row.source_split,
        "sport_label": row.sport_label,
        "gender_category": "",
        "age_category": "",
        "sample_start_sec": round(start, 3),
        "sample_duration_sec": args.duration_sec,
        "sample_selection": selection,
        "video_path": str(output),
        "bytes": output.stat().st_size,
        "review_status": "needs_event_evidence_and_quality_review",
    }])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        previous = pd.read_csv(args.manifest)
        if (previous.video_path == str(output)).any():
            raise ValueError(f"Sample already in manifest: {output}")
        record = pd.concat([previous, record], ignore_index=True)
    record.to_csv(args.manifest, index=False)


if __name__ == "__main__":
    main()
