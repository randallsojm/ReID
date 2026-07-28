"""Azimuth-stratified retrieval analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .context import EvaluationContext
from .retrieval_metrics import average_precision


def angle_by_map(ctx: EvaluationContext) -> tuple[list[dict], list[dict]]:
    """Calculate mAP for each query-azimuth/gallery-azimuth cell."""

    q_angles = sorted(set(ctx.q_azimuths.tolist()))
    g_angles = sorted(set(ctx.g_azimuths.tolist()))
    rows: list[dict] = []
    gap_accumulator: dict[float, list[tuple[float, int]]] = defaultdict(list)

    for q_angle in q_angles:
        query_indexes = np.flatnonzero(ctx.q_azimuths == q_angle)
        for g_angle in g_angles:
            gallery_indexes = np.flatnonzero(ctx.g_azimuths == g_angle)
            ap_values: list[float] = []
            for q_idx in query_indexes:
                query = ctx.query_records[int(q_idx)]
                # Keep standard ReID junk removal within the angle cell.
                valid = gallery_indexes[
                    ~(
                        (ctx.g_pids[gallery_indexes] == query.pid)
                        & (ctx.g_camids[gallery_indexes] == query.camid)
                    )
                ]
                if not len(valid):
                    continue
                order = valid[np.argsort(ctx.distance_matrix[q_idx, valid])]
                correct = ctx.g_pids[order] == query.pid
                if np.any(correct):
                    ap_values.append(average_precision(correct))
            mean_ap = float(np.mean(ap_values)) if ap_values else None
            gap = circular_angle_gap(q_angle, g_angle)
            rows.append(
                {
                    "query_azimuth": q_angle,
                    "gallery_azimuth": g_angle,
                    "azimuth_gap": gap,
                    "mAP": mean_ap,
                    "queries": len(ap_values),
                }
            )
            if mean_ap is not None:
                gap_accumulator[gap].append((mean_ap, len(ap_values)))

    gap_rows: list[dict] = []
    for gap, values in sorted(gap_accumulator.items()):
        samples = sum(count for _, count in values)
        weighted = sum(value * count for value, count in values)
        gap_rows.append(
            {
                "azimuth_gap": gap,
                "weighted_mAP": weighted / samples if samples else None,
                "queries": samples,
            }
        )
    return rows, gap_rows


def circular_angle_gap(first: float, second: float) -> float:
    gap = abs((float(first) - float(second)) % 360.0)
    return min(gap, 360.0 - gap)


def plot_angle_heatmap(rows: list[dict], save_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_angles = sorted({row["query_azimuth"] for row in rows})
    g_angles = sorted({row["gallery_azimuth"] for row in rows})
    matrix = np.full((len(q_angles), len(g_angles)), np.nan)
    q_index = {angle: index for index, angle in enumerate(q_angles)}
    g_index = {angle: index for index, angle in enumerate(g_angles)}
    for row in rows:
        if row["mAP"] is not None:
            matrix[q_index[row["query_azimuth"]], g_index[row["gallery_azimuth"]]] = (
                row["mAP"] * 100.0
            )

    fig, ax = plt.subplots(
        figsize=(max(6, len(g_angles) * 0.8), max(5, len(q_angles) * 0.7))
    )
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    for row in rows:
        value = row["mAP"]
        if value is None:
            continue
        i = q_index[row["query_azimuth"]]
        j = g_index[row["gallery_azimuth"]]
        percent = value * 100.0
        ax.text(
            j,
            i,
            f"{percent:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black" if 20 < percent < 80 else "white",
        )
    ax.set_xticks(range(len(g_angles)), [f"{value:g}°" for value in g_angles], rotation=45)
    ax.set_yticks(range(len(q_angles)), [f"{value:g}°" for value in q_angles])
    ax.set_xlabel("Gallery azimuth")
    ax.set_ylabel("Query azimuth")
    ax.set_title("mAP (%) by query and gallery azimuth")
    fig.colorbar(image, ax=ax, label="mAP (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
