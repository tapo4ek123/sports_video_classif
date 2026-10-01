"""Render selected owner-holdout groups as a reproducible qualitative audit sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, default=Path("outputs/vk_casebook/selection.json"))
    ap.add_argument("--split", type=Path, default=Path("data/splits/vk_owner_holdout/test.jsonl"))
    ap.add_argument("--bundle", type=Path, default=Path("sports_training_bundle"))
    ap.add_argument("--predictions", type=Path,
                    default=Path("outputs/vk_owner_dinov2_1f/group_predictions.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/vk_casebook/contact_sheet.jpg"))
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.split.open(encoding="utf-8")]
    by_group = {}
    for row in rows:
        by_group.setdefault(row["source_group_id"], row)
    predictions = {row["source_group_id"]: row
                   for row in json.loads(args.predictions.read_text(encoding="utf-8"))}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 12)
    tile_w, tile_h, columns = 450, 300, 4
    sheet = Image.new("RGB", (tile_w * columns,
                              tile_h * ((len(selection) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(selection):
        group = item["group"]
        row = by_group[group]
        pred = predictions[group]
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        for j, fraction in enumerate((0.15, 0.5, 0.85)):
            frame_path = args.bundle / row["frames"][round((len(row["frames"]) - 1) * fraction)]
            with Image.open(frame_path) as frame:
                sheet.paste(ImageOps.fit(frame.convert("RGB"), (146, 105)),
                            (x + j * 150, y + 25))
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#999999")
        draw.text((x + 4, y + 4), f"{i + 1}. {item['tag']}  {group[:8]}", fill="black", font=font)
        draw.text((x + 4, y + 137), "True: " + " / ".join(pred[k]["true"]
                  for k in ("sport", "gender", "age")), fill="#111111", font=small)
        draw.text((x + 4, y + 156), "Pred: " + " / ".join(pred[k]["pred"]
                  for k in ("sport", "gender", "age")), fill="#b22222", font=small)
        title = item["title"]
        for line, offset in enumerate(range(0, min(len(title), 120), 44)):
            draw.text((x + 4, y + 178 + 18 * line), title[offset:offset + 44],
                      fill="#333333", font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    print(f"{args.output}: {len(selection)} groups")


if __name__ == "__main__":
    main()
