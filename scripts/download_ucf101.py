"""Download the public UCF101 mirror archives without unpacking all 101 classes."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "guyuchao/UCF101"
REVISION = "057753e5d0709d3f5b8104a803b91a420a069103"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-shards", action="store_true", help="Download both archives; rerun preparation and all experiments")
    args = parser.parse_args()
    root = Path("data/raw/ucf101_archives")
    root.mkdir(parents=True, exist_ok=True)
    names = ("shard-00000.tar", "shard-00001.tar") if args.all_shards else ("shard-00000.tar",)
    for name in names:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=root)
        print(path, flush=True)


if __name__ == "__main__":
    main()
