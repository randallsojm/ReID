"""Retrieval pair selection and static visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .context import EvaluationContext
from .retrieval_metrics import pair_to_dict, valid_pairs


def best_pairs(ctx: EvaluationContext, count: int = 10) -> list[dict]:
    """Return lowest-distance, valid, same-vehicle query/gallery pairs."""

    pairs = [pair for pair in valid_pairs(ctx) if pair.same_vehicle]
    pairs.sort(key=lambda pair: pair.distance)
    return [
        pair_to_dict(ctx, pair.query_index, pair.gallery_index)
        for pair in pairs[: max(0, count)]
    ]


def plot_pair_grid(
    rows: list[dict],
    save_path: str | Path,
    *,
    title: str,
    limit: int = 50,
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = rows[:limit]
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, max(3, 3.2 * len(selected))))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for index, (row, axis_row) in enumerate(zip(selected, axes), start=1):
        for side, axis in zip(("query", "gallery"), axis_row):
            path = Path(row[f"{side}_path"])
            if path.is_file():
                with Image.open(path) as image:
                    axis.imshow(image.convert("RGB"))
            else:
                axis.text(0.5, 0.5, "Image not found", ha="center", va="center")
            axis.set_title(
                f"#{index} {side}: {row[f'{side}_vehicle']}\n"
                f"az={row[f'{side}_azimuth']:g}°  {path.name}",
                fontsize=8,
            )
            axis.axis("off")
        axis_row[0].text(
            0.02,
            0.02,
            f"distance={row['distance']:.5f}",
            transform=axis_row[0].transAxes,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 2},
        )
    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def vehicle_distance_rows(ctx: EvaluationContext) -> list[dict]:
    query_labels = sorted({record.vehicle_label for record in ctx.query_records})
    gallery_labels = sorted({record.vehicle_label for record in ctx.gallery_records})
    rows: list[dict] = []
    for q_label in query_labels:
        q_indexes = [i for i, record in enumerate(ctx.query_records) if record.vehicle_label == q_label]
        for g_label in gallery_labels:
            g_indexes = [i for i, record in enumerate(ctx.gallery_records) if record.vehicle_label == g_label]
            values = ctx.distance_matrix[np.ix_(q_indexes, g_indexes)].ravel()
            rows.append(
                {
                    "query_vehicle": q_label,
                    "gallery_vehicle": g_label,
                    "mean_distance": float(np.mean(values)) if len(values) else None,
                    "pairs": int(len(values)),
                }
            )
    return rows


def plot_vehicle_heatmap(rows: list[dict], save_path: str | Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_labels = sorted({row["query_vehicle"] for row in rows})
    g_labels = sorted({row["gallery_vehicle"] for row in rows})
    matrix = np.full((len(q_labels), len(g_labels)), np.nan)
    q_index = {label: index for index, label in enumerate(q_labels)}
    g_index = {label: index for index, label in enumerate(g_labels)}
    for row in rows:
        matrix[q_index[row["query_vehicle"]], g_index[row["gallery_vehicle"]]] = row[
            "mean_distance"
        ]

    fig, ax = plt.subplots(
        figsize=(max(6, len(g_labels)), max(5, len(q_labels) * 0.8))
    )
    image = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    midpoint = float(np.nanmedian(matrix))
    for i in range(len(q_labels)):
        for j in range(len(g_labels)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > midpoint else "black",
                )
    ax.set_xticks(range(len(g_labels)), g_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(q_labels)), q_labels)
    ax.set_xlabel("Gallery vehicle")
    ax.set_ylabel("Query vehicle")
    ax.set_title("Mean active distance by vehicle (lower is better)")
    fig.colorbar(image, ax=ax, label="Mean distance")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
