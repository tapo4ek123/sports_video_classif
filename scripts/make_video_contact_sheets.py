"""Create compact visual review sheets from downloaded clip manifests."""
from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def frame(path: str, size: tuple[int, int]) -> Image.Image:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "2", "-i", path,
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True, timeout=20, check=True,
    )
    return ImageOps.fit(Image.open(io.BytesIO(result.stdout)).convert("RGB"), size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/review_sheets"))
    parser.add_argument("--pending-only", action="store_true", help="Show rows without review_status")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    width, height, footer, columns = 320, 180, 36, 4
    for manifest in args.manifests:
        items = pd.read_csv(manifest)
        if args.pending_only and "review_status" in items:
            items = items[items.review_status.isna() | (items.review_status == "")]
        for sport, group in items.groupby("sport_label"):
            rows = (len(group) + columns - 1) // columns
            sheet = Image.new("RGB", (width * columns, (height + footer) * rows), "white")
            draw = ImageDraw.Draw(sheet)
            for index, item in enumerate(group.itertuples()):
                x, y = (index % columns) * width, (index // columns) * (height + footer)
                try:
                    sheet.paste(frame(item.video_path, (width, height)), (x, y))
                except (subprocess.SubprocessError, OSError, ValueError):
                    draw.rectangle((x, y, x + width, y + height), fill="red")
                label = str(item.source_video_id).split(":", 1)[-1][:43]
                draw.text((x + 3, y + height + 2), label, fill="black")
            suffix = "_pending" if args.pending_only else ""
            path = args.output / f"{manifest.stem}{suffix}_{sport}.jpg"
            sheet.save(path, quality=90)
            print(f"{path}: {len(group)} clips")


if __name__ == "__main__":
    main()
