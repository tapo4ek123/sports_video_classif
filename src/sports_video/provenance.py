from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance() -> dict:
    return {
        "dataset": "UCF101 mirror first shard",
        "dataset_revision": "057753e5d0709d3f5b8104a803b91a420a069103",
        "manifest_sha256": sha256_file("data/metadata/manifest.csv"),
        "split_sha256": {name: sha256_file(f"data/splits/{name}.csv")
                         for name in ("train", "val", "test")},
    }
