from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

manifest = pd.read_csv("data/metadata/manifest.csv")
report = {
    "clips": len(manifest),
    "source_videos": int(manifest.source_video_id.nunique()),
    "events_known": int(manifest.event_id.nunique()),
    "sport_counts": manifest.sport_label.value_counts().to_dict(),
    "gender_counts": manifest.gender_category.value_counts().to_dict(),
    "age_counts": manifest.age_category.value_counts().to_dict(),
    "source_by_sport": pd.crosstab(manifest.source_dataset, manifest.sport_label).to_dict(),
    "split_by_sport": pd.crosstab(manifest.split, manifest.sport_label).to_dict(),
    "statistics": {},
}
for column in ("duration_sec", "fps", "width", "height"):
    report["statistics"][column] = manifest[column].describe().to_dict()
report["statistics"]["aspect_ratio"] = (manifest.width / manifest.height).describe().to_dict()
Path("outputs").mkdir(exist_ok=True)
Path("outputs/eda.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
