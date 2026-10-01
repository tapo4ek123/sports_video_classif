"""Extract one video per MUVY event through HTTP Range from the Zenodo ZIP."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import fsspec
import pandas as pd


URL = "https://zenodo.org/records/13883315/files/MUVY_v01.zip"
SPORT_MAP = {"basketball": "basketball", "soccer": "football", "tennis": "tennis"}


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frames < 1:
        raise ValueError(f"Cannot decode {path}")
    return {"duration_sec": frames / fps if fps else None,
            "fps": fps, "width": width, "height": height}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/processed/muvy_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/muvy_sample.csv"))
    args = parser.parse_args()
    chosen: dict[tuple[str, str], zipfile.ZipInfo] = {}
    rows = []
    with fsspec.open(URL, "rb", block_size=2**20) as remote:
        with zipfile.ZipFile(remote) as archive:
            candidates = defaultdict(list)
            for member in archive.infolist():
                if not member.filename.endswith(".mp4"):
                    continue
                pieces = member.filename.split("/")
                if len(pieces) < 5 or pieces[2] not in SPORT_MAP:
                    continue
                candidates[(pieces[2], pieces[3])].append(member)
            for key, members in sorted(candidates.items()):
                sport, event = key
                eligible = [x for x in members if x.file_size >= 2**20]
                member = min(eligible or members, key=lambda x: x.file_size)
                chosen[key] = member
                target = args.output / sport / event / Path(member.filename).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != member.file_size:
                    temporary = target.with_suffix(".part")
                    with archive.open(member) as source, temporary.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=2**20)
                    if temporary.stat().st_size != member.file_size:
                        temporary.unlink(missing_ok=True)
                        raise IOError(f"Short extraction: {member.filename}")
                    temporary.replace(target)
                info = video_info(target)
                rows.append({"clip_id": f"MUVY:{sport}:{event}:{target.stem}",
                             "source_dataset": "MUVY",
                             "source_video_id": f"MUVY:{sport}:{event}",
                             "event_id": f"MUVY:{sport}:{event}",
                             "video_path": target.as_posix(),
                             "original_label": sport,
                             "sport_label": SPORT_MAP[sport],
                             "gender_category": None, "age_category": None,
                             "sport_label_source": "MUVY event directory",
                             "gender_label_source": None, "age_label_source": None,
                             **info, "split": "external_test"})
                print(f"{sport}/{event}: {member.file_size / 2**20:.1f} MiB", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(json.dumps({"events": len(rows), "manifest": str(args.manifest),
                      "sport_counts": pd.DataFrame(rows).sport_label.value_counts().to_dict()}))


if __name__ == "__main__":
    main()
