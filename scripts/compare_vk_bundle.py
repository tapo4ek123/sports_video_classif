"""Paired source-group bootstrap for the VK model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_vk_bundle import CLASSES


OUTPUTS = {
    "DINOv2-S 1f": Path("outputs/vk_dinov2_1f"),
    "DINOv2-S 4f": Path("outputs/vk_dinov2_4f"),
    "DINOv2-S 8f": Path("outputs/vk_dinov2_8f"),
    "DINOv2-S partial 4f": Path("outputs/vk_dinov2_partial_4f"),
    "X3D-XS 4f": Path("outputs/vk_x3d_xs_4f"),
    "X3D-XS shortside 4f": Path("outputs/vk_x3d_shortside_4f"),
    "R3D-18 16f": Path("outputs/vk_r3d18_16f"),
}
OWNER_OUTPUTS = {
    "ResNet-18 1f": Path("outputs/vk_owner_resnet18_1f"),
    "DINOv2-S 1f": Path("outputs/vk_owner_dinov2_1f"),
    "DINOv2-S 1f weighted": Path("outputs/vk_owner_dinov2_1f_weighted"),
    "DINOv2-S 4f": Path("outputs/vk_owner_dinov2_4f"),
    "DINOv2-S partial 4f": Path("outputs/vk_owner_dinov2_partial_4f"),
    "X3D-XS 4f": Path("outputs/vk_owner_x3d_xs_4f"),
    "X3D-XS shortside 4f": Path("outputs/vk_owner_x3d_shortside_4f"),
    "R3D-18 16f": Path("outputs/vk_owner_r3d18_16f"),
}


def macro_f1(true: np.ndarray, pred: np.ndarray, n: int) -> float:
    c = np.bincount(true * n + pred, minlength=n * n).reshape(n, n)
    denom = c.sum(0) + c.sum(1)
    return float(np.divide(2 * np.diag(c), denom, out=np.zeros(n), where=denom != 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", action="store_true")
    args = ap.parse_args()
    outputs = OWNER_OUTPUTS if args.owner else OUTPUTS
    records = {}
    summary = {}
    for name, path in outputs.items():
        rows = json.loads((path / "group_predictions.json").read_text())
        records[name] = {r["source_group_id"]: r for r in rows}
        m = json.loads((path / "test_metrics.json").read_text())
        b = json.loads((path / "benchmark.json").read_text())
        h = json.loads((path / "history.json").read_text())
        best = max(h, key=lambda x: x["val_mean_macro_f1"])
        summary[name] = {
            "sport_macro_f1": m["group"]["sport"]["macro_f1"],
            "gender_macro_f1": m["group"]["gender"]["macro_f1"],
            "age_macro_f1": m["group"]["age"]["macro_f1"],
            "all_three_accuracy": m["group"]["all_three_accuracy"],
            "best_epoch": best["epoch"],
            "training_seconds": sum(x["seconds"] for x in h),
            "checkpoint_mb": m["efficiency"]["checkpoint_mb"],
            "peak_training_vram_mb": m["efficiency"]["peak_training_vram_mb"],
            "end_to_end_ms_per_clip": b["end_to_end_ms_per_clip"],
            "model_ms_median": b["model_ms_median"],
        }
    groups = sorted(records["DINOv2-S 4f"])
    if any(set(groups) != set(rec) for rec in records.values()):
        raise ValueError("Model outputs do not share the same test groups")
    rng = np.random.default_rng(42)
    bootstrap_indices = rng.integers(0, len(groups), size=(5000, len(groups)))
    intervals = {}
    per_model_samples = {}
    for name, rec in records.items():
        intervals[name] = {}
        per_model_samples[name] = {}
        for task, classes in CLASSES.items():
            true = np.array([classes.index(rec[g][task]["true"]) for g in groups])
            pred = np.array([classes.index(rec[g][task]["pred"]) for g in groups])
            samples = np.array([macro_f1(true[ix], pred[ix], len(classes))
                                for ix in bootstrap_indices])
            per_model_samples[name][task] = samples
            intervals[name][task] = [float(v) for v in np.quantile(samples, [0.025, 0.975])]
    comparisons = {}
    for other in outputs:
        if other == "DINOv2-S 4f":
            continue
        comparisons[f"DINOv2-S 4f minus {other}"] = {
            task: {"difference": summary["DINOv2-S 4f"][f"{task}_macro_f1"] - summary[other][f"{task}_macro_f1"],
                   "paired_bootstrap_95pct": [float(v) for v in np.quantile(
                       per_model_samples["DINOv2-S 4f"][task] - per_model_samples[other][task],
                       [0.025, 0.975])]}
            for task in CLASSES
        }
    if args.owner:
        for first, second in (
            ("DINOv2-S 1f", "ResNet-18 1f"),
            ("DINOv2-S 1f weighted", "DINOv2-S 1f"),
            ("X3D-XS shortside 4f", "DINOv2-S 1f"),
        ):
            comparisons[f"{first} minus {second}"] = {
                task: {
                    "difference": summary[first][f"{task}_macro_f1"]
                    - summary[second][f"{task}_macro_f1"],
                    "paired_bootstrap_95pct": [float(v) for v in np.quantile(
                        per_model_samples[first][task] - per_model_samples[second][task],
                        [0.025, 0.975])],
                }
                for task in CLASSES
            }
    result = {"test_groups": len(groups), "bootstrap_replicates": len(bootstrap_indices),
              "summary": summary, "macro_f1_95pct_intervals": intervals,
              "paired_comparisons": comparisons}
    target = Path("outputs/vk_owner_comparison.json" if args.owner else "outputs/vk_comparison.json")
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
