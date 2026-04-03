from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import load_experiment_config
from src.data_io import (
    case_output_dir,
    ensure_dir,
    load_case,
    read_json,
    save_binary_mask,
    save_rgb_image,
    write_csv,
    write_json,
)
from src.evaluation import aggregate_case_summaries, summarize_mask
from src.optimization import run_segmentation
from src.prepare_data import bundle_sample_dataset
from src.report_assets import sync_report_figures
from src.visualization import (
    create_annotation_overlay,
    create_mask_overlay,
    save_boundary_comparison,
    save_comparison_panel,
    save_iteration_plot,
)


def run_pipeline(config_path: str | Path) -> list[dict[str, float | str]]:
    config = load_experiment_config(config_path)
    ensure_dir(config.output_dir)
    rows: list[dict[str, float | str]] = []

    for item in config.dataset_items:
        case = load_case(item, config.max_dim)
        output_dir = Path(config.output_dir) / case.name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir = case_output_dir(config.output_dir, case.name)
        result = run_segmentation(
            case.image_rgb, case.fg_seed, case.bg_seed, config, bbox=case.bbox
        )

        annotation_overlay = create_annotation_overlay(case.image_rgb, case.fg_seed, case.bg_seed)
        final_overlay = create_mask_overlay(
            case.image_rgb, result.refined_mask, alpha=config.visualization.overlay_alpha
        )

        save_rgb_image(output_dir / "original.png", case.image_rgb)
        save_rgb_image(output_dir / "annotation_overlay.png", annotation_overlay)
        save_binary_mask(output_dir / "naive_mask.png", result.baseline_mask)
        save_binary_mask(output_dir / "graph_cut_raw_mask.png", result.raw_mask)
        save_binary_mask(output_dir / "graph_cut_refined_mask.png", result.refined_mask)
        save_rgb_image(output_dir / "final_overlay.png", final_overlay)

        save_comparison_panel(
            output_dir / "comparison_panel.png",
            case.image_rgb,
            annotation_overlay,
            result.baseline_mask,
            result.raw_mask,
            result.refined_mask,
            final_overlay,
            dpi=config.visualization.figure_dpi,
        )
        save_boundary_comparison(
            output_dir / "boundary_refinement.png",
            case.image_rgb,
            result.raw_mask,
            result.refined_mask,
            dpi=config.visualization.figure_dpi,
        )
        save_iteration_plot(
            output_dir / "energy_iterations.png",
            [record.to_dict() for record in result.iteration_records],
            dpi=config.visualization.figure_dpi,
        )

        naive_metrics = summarize_mask(
            case.image_rgb,
            result.baseline_mask,
            result.baseline_fg_cost,
            result.baseline_bg_cost,
            result.pairwise,
            case.fg_seed,
            case.bg_seed,
            bbox=case.bbox,
        )
        raw_metrics = summarize_mask(
            case.image_rgb,
            result.raw_mask,
            result.final_fg_cost,
            result.final_bg_cost,
            result.pairwise,
            case.fg_seed,
            case.bg_seed,
            bbox=case.bbox,
        )
        refined_metrics = summarize_mask(
            case.image_rgb,
            result.refined_mask,
            result.final_fg_cost,
            result.final_bg_cost,
            result.pairwise,
            case.fg_seed,
            case.bg_seed,
            bbox=case.bbox,
        )

        row = {
            "case_name": case.name,
            "target_label": case.target_label,
            "beta": result.beta,
            "iterations": float(len(result.iteration_records)),
            "final_mask_change": float(result.iteration_records[-1].mask_change),
            "runtime_total_seconds": float(
                sum(record.runtime_seconds for record in result.iteration_records)
            ),
            "naive_total_energy": naive_metrics["total_energy"],
            "graph_cut_total_energy": raw_metrics["total_energy"],
            "refined_total_energy": refined_metrics["total_energy"],
            "naive_component_count": naive_metrics["component_count"],
            "graph_cut_component_count": raw_metrics["component_count"],
            "refined_component_count": refined_metrics["component_count"],
            "naive_boundary_length": naive_metrics["boundary_length"],
            "graph_cut_boundary_length": raw_metrics["boundary_length"],
            "refined_boundary_length": refined_metrics["boundary_length"],
            "naive_compactness": naive_metrics["compactness"],
            "graph_cut_compactness": raw_metrics["compactness"],
            "refined_compactness": refined_metrics["compactness"],
            "naive_edge_alignment_score": naive_metrics["edge_alignment_score"],
            "graph_cut_edge_alignment_score": raw_metrics["edge_alignment_score"],
            "refined_edge_alignment_score": refined_metrics["edge_alignment_score"],
            "naive_bbox_leakage_ratio": naive_metrics["bbox_leakage_ratio"],
            "graph_cut_bbox_leakage_ratio": raw_metrics["bbox_leakage_ratio"],
            "refined_bbox_leakage_ratio": refined_metrics["bbox_leakage_ratio"],
            "naive_bbox_fill_ratio": naive_metrics["bbox_fill_ratio"],
            "graph_cut_bbox_fill_ratio": raw_metrics["bbox_fill_ratio"],
            "refined_bbox_fill_ratio": refined_metrics["bbox_fill_ratio"],
            "naive_seed_consistency_rate": naive_metrics["seed_consistency_rate"],
            "graph_cut_seed_consistency_rate": raw_metrics["seed_consistency_rate"],
            "refined_seed_consistency_rate": refined_metrics["seed_consistency_rate"],
            "naive_foreground_fraction": naive_metrics["foreground_fraction"],
            "graph_cut_foreground_fraction": raw_metrics["foreground_fraction"],
            "refined_foreground_fraction": refined_metrics["foreground_fraction"],
        }
        rows.append(row)

        write_json(
            output_dir / "metrics.json",
            {
                "case_name": case.name,
                "target_label": case.target_label,
                "original_shape": case.original_shape,
                "resized_shape": case.resized_shape,
                "scale_factor": case.scale_factor,
                "bbox": list(case.bbox) if case.bbox else None,
                "beta": result.beta,
                "iteration_records": [record.to_dict() for record in result.iteration_records],
                "naive_metrics": naive_metrics,
                "graph_cut_metrics": raw_metrics,
                "refined_metrics": refined_metrics,
            },
        )

    summary_dir = Path(config.output_dir) / "summary"
    if summary_dir.exists():
        shutil.rmtree(summary_dir)
    summary_dir = ensure_dir(summary_dir)
    write_csv(summary_dir / "metrics.csv", rows)
    write_json(summary_dir / "aggregate.json", aggregate_case_summaries(rows))
    sync_report_figures(config.output_dir, Path(config_path).resolve().parent.parent / "report")
    return rows


def evaluate_results(results_dir: str | Path) -> list[dict[str, float | str]]:
    results_dir = Path(results_dir)
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        payload = read_json(metrics_path)
        naive = payload["naive_metrics"]
        graph_cut = payload["graph_cut_metrics"]
        refined = payload["refined_metrics"]
        rows.append(
            {
                "case_name": payload["case_name"],
                "target_label": payload["target_label"],
                "beta": payload["beta"],
                "iterations": float(len(payload["iteration_records"])),
                "final_mask_change": float(payload["iteration_records"][-1]["mask_change"]),
                "runtime_total_seconds": float(
                    sum(record["runtime_seconds"] for record in payload["iteration_records"])
                ),
                "naive_total_energy": naive["total_energy"],
                "graph_cut_total_energy": graph_cut["total_energy"],
                "refined_total_energy": refined["total_energy"],
                "naive_component_count": naive["component_count"],
                "graph_cut_component_count": graph_cut["component_count"],
                "refined_component_count": refined["component_count"],
                "naive_boundary_length": naive["boundary_length"],
                "graph_cut_boundary_length": graph_cut["boundary_length"],
                "refined_boundary_length": refined["boundary_length"],
                "naive_compactness": naive["compactness"],
                "graph_cut_compactness": graph_cut["compactness"],
                "refined_compactness": refined["compactness"],
                "naive_edge_alignment_score": naive["edge_alignment_score"],
                "graph_cut_edge_alignment_score": graph_cut["edge_alignment_score"],
                "refined_edge_alignment_score": refined["edge_alignment_score"],
                "naive_bbox_leakage_ratio": naive["bbox_leakage_ratio"],
                "graph_cut_bbox_leakage_ratio": graph_cut["bbox_leakage_ratio"],
                "refined_bbox_leakage_ratio": refined["bbox_leakage_ratio"],
                "naive_bbox_fill_ratio": naive["bbox_fill_ratio"],
                "graph_cut_bbox_fill_ratio": graph_cut["bbox_fill_ratio"],
                "refined_bbox_fill_ratio": refined["bbox_fill_ratio"],
                "naive_seed_consistency_rate": naive["seed_consistency_rate"],
                "graph_cut_seed_consistency_rate": graph_cut["seed_consistency_rate"],
                "refined_seed_consistency_rate": refined["seed_consistency_rate"],
                "naive_foreground_fraction": naive["foreground_fraction"],
                "graph_cut_foreground_fraction": graph_cut["foreground_fraction"],
                "refined_foreground_fraction": refined["foreground_fraction"],
            }
        )
    summary_dir = ensure_dir(results_dir / "summary")
    write_csv(summary_dir / "metrics.csv", rows)
    if rows:
        write_json(summary_dir / "aggregate.json", aggregate_case_summaries(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph cut segmentation assignment pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Bundle sample images and annotations.")
    prepare.add_argument("--project-root", default=Path(__file__).resolve().parent.parent)

    run = subparsers.add_parser("run", help="Execute the graph cut pipeline.")
    run.add_argument("--config", default="configs/experiment.yaml")

    evaluate = subparsers.add_parser("evaluate", help="Regenerate summary metrics from saved results.")
    evaluate.add_argument("--results-dir", default="results")

    all_cmd = subparsers.add_parser("all", help="Prepare data, run the pipeline, and refresh report assets.")
    all_cmd.add_argument("--config", default="configs/experiment.yaml")
    all_cmd.add_argument("--project-root", default=Path(__file__).resolve().parent.parent)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        bundle_sample_dataset(args.project_root)
    elif args.command == "run":
        run_pipeline(args.config)
    elif args.command == "evaluate":
        evaluate_results(args.results_dir)
    elif args.command == "all":
        bundle_sample_dataset(args.project_root)
        run_pipeline(args.config)


if __name__ == "__main__":
    main()
