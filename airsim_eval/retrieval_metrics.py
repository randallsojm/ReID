"""Distance-based vehicle-ReID retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .context import EvaluationContext


def average_precision(correct: Iterable[bool]) -> float:
    flags = np.asarray(list(correct), dtype=bool)
    positives = int(flags.sum())
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(flags)
    ranks = np.arange(1, len(flags) + 1)
    return float((cumulative[flags] / ranks[flags]).sum() / positives)


def valid_gallery_mask(ctx: EvaluationContext, query_index: int) -> np.ndarray:
    """Standard ReID junk rule: same PID and same camera is excluded."""

    same_pid = ctx.g_pids == ctx.q_pids[query_index]
    same_cam = ctx.g_camids == ctx.q_camids[query_index]
    return ~(same_pid & same_cam)


def cross_camera_mask(ctx: EvaluationContext, query_index: int) -> np.ndarray:
    """Pair-analysis rule inherited from reid_crops.py."""

    return ctx.g_camids != ctx.q_camids[query_index]


def ranked_gallery(ctx: EvaluationContext, query_index: int) -> np.ndarray:
    valid = valid_gallery_mask(ctx, query_index)
    indexes = np.flatnonzero(valid)
    return indexes[np.argsort(ctx.distance_matrix[query_index, indexes])]


def retrieval_metrics(
    ctx: EvaluationContext,
    ranks=(1, 5, 10),
    query_indexes: Iterable[int] | None = None,
) -> dict:
    """Calculate mAP and CMC over all or a selected set of queries."""

    ap_values: list[float] = []
    hits = {int(rank): 0 for rank in ranks}
    eligible = 0
    no_match = 0
    per_query: list[dict] = []

    selected = (
        range(len(ctx.query_records))
        if query_indexes is None
        else [int(index) for index in query_indexes]
    )
    for q_idx in selected:
        query = ctx.query_records[q_idx]
        ranked = ranked_gallery(ctx, q_idx)
        correct = ctx.g_pids[ranked] == query.pid
        if not np.any(correct):
            no_match += 1
            per_query.append(
                {
                    "query": query.crop_filename,
                    "vehicle": query.vehicle_label,
                    "AP": None,
                    "first_correct_rank": None,
                    **{f"CMC@{rank}": None for rank in ranks},
                }
            )
            continue
        eligible += 1
        ap = average_precision(correct)
        ap_values.append(ap)
        first_correct_rank = int(np.flatnonzero(correct)[0]) + 1
        row = {
            "query": query.crop_filename,
            "vehicle": query.vehicle_label,
            "AP": ap,
            "first_correct_rank": first_correct_rank,
        }
        for rank in ranks:
            hit = bool(np.any(correct[: int(rank)]))
            hits[int(rank)] += int(hit)
            row[f"CMC@{rank}"] = hit
        per_query.append(row)

    return {
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "cmc": {
            str(rank): hits[int(rank)] / eligible if eligible else 0.0 for rank in ranks
        },
        "eligible_queries": eligible,
        "queries_without_match": no_match,
        "per_query": per_query,
    }


def per_vehicle_breakdown(ctx: EvaluationContext, ranks=(1, 5, 10)) -> list[dict]:
    """Return retrieval metrics independently for every query vehicle."""

    rows: list[dict] = []
    labels = sorted({record.vehicle_label for record in ctx.query_records})
    for label in labels:
        indexes = [
            index
            for index, record in enumerate(ctx.query_records)
            if record.vehicle_label == label
        ]
        result = retrieval_metrics(ctx, ranks=ranks, query_indexes=indexes)
        first_ranks = [
            row["first_correct_rank"]
            for row in result["per_query"]
            if row["first_correct_rank"] is not None
        ]
        rows.append(
            {
                "vehicle": label,
                "queries": len(indexes),
                "eligible_queries": result["eligible_queries"],
                "queries_without_match": result["queries_without_match"],
                "mAP": result["mAP"],
                **{f"CMC@{rank}": result["cmc"][str(rank)] for rank in ranks},
                "mean_first_correct_rank": (
                    float(np.mean(first_ranks)) if first_ranks else None
                ),
                "median_first_correct_rank": (
                    float(np.median(first_ranks)) if first_ranks else None
                ),
            }
        )
    return rows


@dataclass(frozen=True)
class Pair:
    query_index: int
    gallery_index: int
    distance: float
    same_vehicle: bool


def valid_pairs(ctx: EvaluationContext) -> list[Pair]:
    pairs: list[Pair] = []
    for q_idx, query in enumerate(ctx.query_records):
        valid = cross_camera_mask(ctx, q_idx)
        for g_idx in np.flatnonzero(valid):
            pairs.append(
                Pair(
                    query_index=q_idx,
                    gallery_index=int(g_idx),
                    distance=float(ctx.distance_matrix[q_idx, g_idx]),
                    same_vehicle=query.pid == ctx.gallery_records[g_idx].pid,
                )
            )
    return pairs


def retrieval_errors(ctx: EvaluationContext, topk: int = 1) -> tuple[list[dict], list[dict]]:
    if topk < 1:
        raise ValueError("topk must be at least 1")
    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    for q_idx, query in enumerate(ctx.query_records):
        ranked = ranked_gallery(ctx, q_idx)
        top = ranked[:topk]
        for rank, g_idx in enumerate(top, start=1):
            gallery = ctx.gallery_records[int(g_idx)]
            if gallery.pid != query.pid:
                false_positives.append(
                    pair_to_dict(ctx, q_idx, int(g_idx), rank=rank, kind="FP")
                )
        correct_all = ranked[ctx.g_pids[ranked] == query.pid]
        correct_top = top[ctx.g_pids[top] == query.pid]
        if len(correct_all) and not len(correct_top):
            false_negatives.append(
                pair_to_dict(ctx, q_idx, int(correct_all[0]), rank=None, kind="FN")
            )
    return false_positives, false_negatives


def pair_to_dict(
    ctx: EvaluationContext,
    q_idx: int,
    g_idx: int,
    *,
    rank: int | None = None,
    kind: str | None = None,
) -> dict:
    query = ctx.query_records[q_idx]
    gallery = ctx.gallery_records[g_idx]
    return {
        "kind": kind,
        "rank": rank,
        "distance": float(ctx.distance_matrix[q_idx, g_idx]),
        "query_path": str(query.path),
        "query_filename": query.crop_filename,
        "query_vehicle": query.vehicle_label,
        "query_camera": query.camera,
        "query_azimuth": query.azimuth,
        "gallery_path": str(gallery.path),
        "gallery_filename": gallery.crop_filename,
        "gallery_vehicle": gallery.vehicle_label,
        "gallery_camera": gallery.camera,
        "gallery_azimuth": gallery.azimuth,
    }
