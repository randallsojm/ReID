"""Orchestrate optional AirSim reports without coupling them to teste.py."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .angle_analysis import angle_by_map, plot_angle_heatmap
from .context import EvaluationContext
from .coverage import coverage_breakdown
from .retrieval_metrics import (
    per_vehicle_breakdown,
    retrieval_errors,
    retrieval_metrics,
)
from .visualizations import (
    best_pairs,
    plot_pair_grid,
    plot_vehicle_heatmap,
    vehicle_distance_rows,
)


@dataclass
class AnalysisOptions:
    results_dir: str = "results"
    angle_by_map: bool = False
    coverage_breakdown: bool = False
    best_pairs: int = 0
    best_pairs_img: bool = False
    fp_fn_imgs: bool = False
    vehicle_heatmap: bool = False
    heatmap: bool = False
    breakdown: bool = False
    fpfn_topk: int = 1
    image_limit: int = 50


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def run_analyses(ctx: EvaluationContext, options: AnalysisOptions) -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output = ctx.ensure_output_dir(Path(options.results_dir) / stamp)

    retrieval = retrieval_metrics(ctx)
    metrics = {**ctx.summary(), "retrieval": retrieval}
    # Keep detailed per-query rows in CSV, not metrics.json.
    _write_csv(output / "per_query.csv", retrieval.pop("per_query"))

    if options.breakdown:
        _write_csv(
            output / "vehicle_breakdown.csv",
            per_vehicle_breakdown(ctx),
        )

    angle_rows: list[dict] = []
    if options.angle_by_map or options.heatmap:
        angle_rows, gap_rows = angle_by_map(ctx)
        _write_csv(output / "angle_map.csv", angle_rows)
        _write_csv(output / "angle_gap_map.csv", gap_rows)
        if options.heatmap:
            plot_angle_heatmap(angle_rows, output / "angle_heatmap.png")

    if options.coverage_breakdown:
        _write_csv(
            output / "coverage.csv",
            coverage_breakdown(ctx),
        )

    selected_pairs = best_pairs(ctx, options.best_pairs or 10)
    if options.best_pairs:
        _write_csv(output / "best_pairs.csv", selected_pairs)
    if options.best_pairs_img:
        plot_pair_grid(
            selected_pairs,
            output / "best_pairs.png",
            title="Best correct retrieval pairs",
            limit=options.image_limit,
        )

    if options.fp_fn_imgs:
        retrieval_fp, retrieval_fn = retrieval_errors(ctx, options.fpfn_topk)
        _write_csv(output / "retrieval_false_positives.csv", retrieval_fp)
        _write_csv(output / "retrieval_false_negatives.csv", retrieval_fn)
        plot_pair_grid(
            retrieval_fp,
            output / "false_positives.png",
            title=f"Wrong gallery entries in Top-{options.fpfn_topk}",
            limit=options.image_limit,
        )
        plot_pair_grid(
            retrieval_fn,
            output / "false_negatives.png",
            title=f"Queries with no correct match in Top-{options.fpfn_topk}",
            limit=options.image_limit,
        )

    if options.vehicle_heatmap:
        vehicle_rows = vehicle_distance_rows(ctx)
        _write_csv(output / "vehicle_distances.csv", vehicle_rows)
        plot_vehicle_heatmap(vehicle_rows, output / "vehicle_heatmap.png")

    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2)
    return output
