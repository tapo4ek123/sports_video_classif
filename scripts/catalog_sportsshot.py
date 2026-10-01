"""Index SportsShot ZIP members using byte ranges without downloading video archives.

Requires an approved Hugging Face token in HF_TOKEN or the local HF login cache.
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, get_token, hf_hub_url


REPO_ID = "MCG-NJU/SportsShot"


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, token: str, size: int) -> None:
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (offset if whence == 0 else self.position + offset if whence == 1
                         else self.size + offset)
        if not 0 <= self.position <= self.size:
            raise ValueError("seek outside remote ZIP")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size > 4 * 1024 * 1024:
            raise ValueError("refusing a large remote ZIP read")
        size = min(size, self.size - self.position)
        if not size:
            return b""
        start = self.position
        end = start + size - 1
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                    timeout=(15, 45))
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != size:
            raise IOError(f"Range read failed: status={response.status_code}, bytes={len(response.content)}")
        self.position += size
        return response.content

    def close(self) -> None:
        self.session.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/metadata/sportsshot_archive_index.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; preserve its current contents")
    token = get_token()
    if not token:
        parser.error("HF_TOKEN or hf auth login is required")
    repo = HfApi().dataset_info(REPO_ID, token=token, files_metadata=True)
    archives = [file for file in repo.siblings or []
                if file.rfilename.startswith("videos/") and file.rfilename.endswith(".zip")]
    rows = []
    for archive in archives:
        archive_name = archive.rfilename
        archive_size = int(archive.size)
        remote = RemoteZip(hf_hub_url(REPO_ID, archive_name, repo_type="dataset"),
                           token, archive_size)
        try:
            with zipfile.ZipFile(remote) as zip_file:
                members = [item for item in zip_file.infolist() if not item.is_dir()]
            split, sport = archive_name.split("/")[1:3]
            sport = sport.removesuffix(".zip")
            for member in members:
                basename = Path(member.filename).name
                match = re.fullmatch(r"v_(.+)_\d+\.mp4", basename)
                rows.append({
                    "source_dataset": "SportsShot",
                    "archive_path": archive_name,
                    "archive_bytes": archive_size,
                    "source_split": split,
                    "sport_label": sport,
                    "member_path": member.filename,
                    "source_youtube_id": match.group(1) if match else "",
                    "member_bytes": member.file_size,
                    "compressed_bytes": member.compress_size,
                    "compression_type": member.compress_type,
                    "header_offset": member.header_offset,
                })
            print(f"{archive_name}: members={len(members)}", flush=True)
        finally:
            remote.close()
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"members={len(frame)} youtube_ids={frame.source_youtube_id.nunique()} output={args.output}")


if __name__ == "__main__":
    main()
