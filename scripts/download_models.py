"""Fetch pinned public pretrained weights used by stage 1.

DINOv3 requires accepting Meta's license on Hugging Face before downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import urllib.request
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path("data/models")
SIGLIP_FILES = ["config.json", "model.safetensors", "preprocessor_config.json",
                "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]


def download_ranged(url: str, dest: Path, chunk_size: int = 4 * 2**20) -> None:
    """Resume a public checkpoint in small verified byte ranges."""
    session = requests.Session()
    head = session.head(url, timeout=(10, 30))
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    temporary = dest.with_suffix(dest.suffix + ".partial")
    current = temporary.stat().st_size if temporary.exists() else 0
    if current > expected:
        temporary.unlink()
        current = 0
    with temporary.open("ab") as handle:
        while current < expected:
            end = min(expected, current + chunk_size) - 1
            for attempt in range(3):
                try:
                    response = session.get(url, headers={"Range": f"bytes={current}-{end}"},
                                           timeout=(10, 60))
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {current}-{end}":
                        raise IOError("Server did not return the requested byte range")
                    if len(response.content) != end - current + 1:
                        raise IOError("Short checkpoint chunk")
                    handle.write(response.content)
                    current = end + 1
                    print(f"{current}/{expected} bytes", flush=True)
                    break
                except (requests.RequestException, IOError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
    if temporary.stat().st_size != expected:
        raise IOError(f"Incomplete checkpoint: {temporary.stat().st_size}/{expected}")
    os.replace(temporary, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["siglip", "efficientnet_b0", "timm_b4", "timm_vit_small", "r3d18", "resnet18", "x3d_xs", "dinov2", "dinov3"])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.model == "siglip":
        snapshot_download("google/siglip-base-patch16-224",
                          revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
                          allow_patterns=SIGLIP_FILES,
                          local_dir=ROOT / "siglip-base-patch16-224")
    elif args.model in {"timm_b4", "timm_vit_small"}:
        repo, revision, target = {
            "timm_b4": ("timm/tf_efficientnet_b4.ns_jft_in1k",
                        "40be4c2e3f4a91b2d9772137bfc2ea959e6e5260",
                        "tf_efficientnet_b4.ns_jft_in1k.safetensors"),
            "timm_vit_small": ("timm/vit_small_patch32_224.augreg_in21k_ft_in1k",
                               "f698858ece8c7f13429b66da3d768740bb9decda",
                               "vit_small_patch32_224.augreg_in21k_ft_in1k.safetensors"),
        }[args.model]
        source = hf_hub_download(repo, "model.safetensors", revision=revision)
        dest = ROOT / "timm" / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    elif args.model == "dinov2":
        snapshot_download("facebook/dinov2-small",
                          revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
                          allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
                          local_dir=ROOT / "dinov2-small")
    elif args.model == "dinov3":
        snapshot_download("facebook/dinov3-vits16-pretrain-lvd1689m",
                          allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
                          local_dir=ROOT / "dinov3-vits16")
    else:
        urls = {
            "efficientnet_b0": ("https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
                                ROOT / "efficientnet_b0" / "efficientnet_b0_rwightman-7f5810bc.pth"),
            "r3d18": ("https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
                      ROOT / "r3d18" / "r3d_18-b3b3357e.pth"),
            "resnet18": ("https://download.pytorch.org/models/resnet18-f37072fd.pth",
                         ROOT / "resnet18" / "resnet18-f37072fd.pth"),
            "x3d_xs": ("https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth",
                       ROOT / "x3d" / "X3D_XS.pyth"),
        }
        url, dest = urls[args.model]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.model == "x3d_xs":
            download_ranged(url, dest)
        else:
            temporary = dest.with_suffix(dest.suffix + ".partial")
            _, headers = urllib.request.urlretrieve(url, temporary)
            expected = headers.get("Content-Length")
            if expected and temporary.stat().st_size != int(expected):
                raise IOError(f"Incomplete download: {temporary}")
            if args.model == "resnet18":
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if digest != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
                    raise IOError(f"ResNet-18 checksum mismatch: {digest}")
            os.replace(temporary, dest)
    print(f"Ready: {args.model}")


if __name__ == "__main__":
    main()
