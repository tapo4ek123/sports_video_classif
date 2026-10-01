"""Comparable batch-1 inference benchmark on the same VK test clips."""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sports_video.models import VideoClassifier
from train_vk_bundle import BundleDataset, CLASSES, read_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", type=Path, nargs="+")
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable")
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    workers = args.workers if args.workers is not None else (0 if device.type == "cpu" else 2)
    results = {}
    for output in args.outputs:
        config = json.loads((output / "config.json").read_text())
        name, root = config["model"], Path(config["root"])
        rows = read_split(Path(config.get("split_dir") or root), "test")[: args.warmup + args.clips]
        loader = DataLoader(BundleDataset(root, rows, name, config["frames"], False,
                                          config.get("x3d_preprocess", "legacy")),
                            batch_size=1, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        model = VideoClassifier(name, pretrained=False, freeze=config["freeze"],
                                weights=config["weights"] if name == "dinov2_small" else None)
        model.heads["sport_label"] = nn.Linear(model.width, len(CLASSES["sport"]))
        model.load_state_dict(torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()
        times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        e2e_start = None
        with torch.inference_mode():
            for i, (x, _, _) in enumerate(itertools.islice(loader, len(rows))):
                if i == args.warmup:
                    e2e_start = time.perf_counter()
                begin = time.perf_counter()
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= args.warmup:
                    times.append((time.perf_counter() - begin) * 1000)
        e2e_sec = time.perf_counter() - e2e_start
        result = {
            "clips": len(times), "batch_size": 1, "frames": config["frames"],
            "resolution": 160 if name == "x3d_xs" else 112 if name == "r3d_18" else 224,
            "model_ms_mean": statistics.mean(times), "model_ms_median": statistics.median(times),
            "model_ms_p90": sorted(times)[int(0.9 * (len(times) - 1))],
            "end_to_end_ms_per_clip": e2e_sec * 1000 / len(times),
            "end_to_end_clips_per_sec": len(times) / e2e_sec,
            "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else 0,
            "peak_cpu_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if device.type == "cpu" else None,
            "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
            "precision": "AMP fp16" if device.type == "cuda" else "fp32",
            "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        }
        target = "benchmark_cpu.json" if device.type == "cpu" else "benchmark.json"
        (output / target).write_text(json.dumps(result, indent=2))
        results[str(output)] = result
        print(json.dumps({"output": str(output), **result}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
