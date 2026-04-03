from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.data_io import ensure_dir


def _box(axis, xy, width, height, text, facecolor):
    rect = Rectangle(xy, width, height, facecolor=facecolor, edgecolor="#263238", linewidth=1.2)
    axis.add_patch(rect)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="semibold",
    )


def _arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.2, color="#455a64")
    )


def create_pipeline_overview(output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")

    _box(axis, (0.4, 1.7), 1.6, 0.8, "Input Image", "#e3f2fd")
    _box(axis, (2.2, 1.7), 2.0, 0.8, "User Seeds\n(FG / BG)", "#ffebee")
    _box(axis, (4.5, 1.7), 2.1, 0.8, "Histogram-Based\nUnary Modeling", "#e8f5e9")
    _box(axis, (6.9, 1.7), 1.8, 0.8, "Graph\nConstruction", "#fff3e0")
    _box(axis, (9.0, 1.7), 1.8, 0.8, "Min-Cut /\nMax-Flow", "#ede7f6")
    _box(axis, (4.5, 0.4), 2.1, 0.8, "Iterative Model\nUpdate", "#f3e5f5")
    _box(axis, (9.0, 0.4), 1.8, 0.8, "Refinement +\nOverlay", "#e0f7fa")

    _arrow(axis, (2.0, 2.1), (2.2, 2.1))
    _arrow(axis, (4.2, 2.1), (4.5, 2.1))
    _arrow(axis, (6.6, 2.1), (6.9, 2.1))
    _arrow(axis, (8.7, 2.1), (9.0, 2.1))
    _arrow(axis, (10.8, 2.1), (10.8, 1.25))
    _arrow(axis, (9.0, 0.8), (6.6, 0.8))
    _arrow(axis, (5.6, 1.2), (5.6, 1.65))

    axis.text(6.0, 3.2, "Graph Cut Segmentation Pipeline", ha="center", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def create_energy_schematic(output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5)
    axis.axis("off")

    _box(axis, (0.6, 2.0), 1.2, 0.9, "Source\n(FG)", "#ffebee")
    _box(axis, (4.1, 2.0), 1.8, 0.9, "Pixel Node\n$p$", "#fffde7")
    _box(axis, (8.0, 2.0), 1.2, 0.9, "Sink\n(BG)", "#e3f2fd")
    _box(axis, (4.1, 0.6), 1.8, 0.9, "Neighbor\n$q$", "#e8f5e9")

    _arrow(axis, (1.8, 2.45), (4.1, 2.45))
    _arrow(axis, (5.9, 2.45), (8.0, 2.45))
    _arrow(axis, (5.0, 2.0), (5.0, 1.5))

    axis.text(3.0, 2.8, "$D_p(\\mathrm{BG})$", fontsize=12)
    axis.text(6.4, 2.8, "$D_p(\\mathrm{FG})$", fontsize=12)
    axis.text(5.2, 1.6, "$\\lambda w_{pq}[L_p \\neq L_q]$", fontsize=12)
    axis.text(
        5.0,
        4.1,
        r"$E(L)=\sum_p D_p(L_p)+\lambda\sum_{(p,q)\in\mathcal{N}}w_{pq}[L_p\neq L_q]$",
        ha="center",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def sync_report_figures(results_root: str | Path, report_root: str | Path) -> None:
    results_root = Path(results_root)
    report_figures = ensure_dir(Path(report_root) / "figures")
    for pattern in ["*_comparison.png", "*_boundary.png", "*_energy.png", "pipeline_overview.png", "energy_graph_schematic.png"]:
        for existing in report_figures.glob(pattern):
            existing.unlink()
    create_pipeline_overview(report_figures / "pipeline_overview.png")
    create_energy_schematic(report_figures / "energy_graph_schematic.png")

    for panel in results_root.glob("*/comparison_panel.png"):
        shutil.copy2(panel, report_figures / f"{panel.parent.name}_comparison.png")
    for boundary in results_root.glob("*/boundary_refinement.png"):
        shutil.copy2(boundary, report_figures / f"{boundary.parent.name}_boundary.png")
    for plot in results_root.glob("*/energy_iterations.png"):
        shutil.copy2(plot, report_figures / f"{plot.parent.name}_energy.png")
