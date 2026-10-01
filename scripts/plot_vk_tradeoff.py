"""Plot owner-holdout quality versus measured inference time for the demo/report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "ResNet-18, 1f": Path("outputs/vk_owner_resnet18_1f"),
    "DINOv2-S, 1f": Path("outputs/vk_owner_dinov2_1f"),
    "X3D-XS, 4f": Path("outputs/vk_owner_x3d_shortside_4f"),
    "R3D-18, 16f": Path("outputs/vk_owner_r3d18_16f"),
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ["#277da1", "#f8961e", "#43aa8b", "#b56576"]
    for (name, path), color in zip(RUNS.items(), colors):
        metrics = json.loads((path / "test_metrics.json").read_text())["group"]
        for ax, hardware in zip(axes, ("gpu", "cpu")):
            speed = json.loads((path / ("benchmark.json" if hardware == "gpu"
                                        else "benchmark_cpu.json")).read_text())
            ax.scatter(speed["end_to_end_ms_per_clip"], metrics["sport"]["macro_f1"],
                       s=85, color=color, label=name, zorder=3)
            ax.annotate(name, (speed["end_to_end_ms_per_clip"],
                               metrics["sport"]["macro_f1"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    for ax, hardware in zip(axes, ("GPU, RTX 3050 Laptop", "CPU, Ryzen 5 5600U")):
        ax.set_title(hardware)
        ax.set_xlabel("Время на окно, мс (меньше лучше)")
        ax.set_ylabel("Macro-F1 по видам спорта (больше лучше)")
        ax.set_ylim(0.68, 0.93)
        ax.grid(alpha=0.25)
    fig.suptitle("Новые VK-каналы: качество и задержка, 20 одинаковых окон")
    output = Path("outputs/vk_quality_speed.png")
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
