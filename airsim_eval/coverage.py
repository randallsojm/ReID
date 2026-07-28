"""Crop-coverage breakdown matching the definition in reid_crops.py."""

from __future__ import annotations

from .context import EvaluationContext
from .retrieval_metrics import retrieval_metrics


DEFAULT_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0)


def coverage_breakdown(
    ctx: EvaluationContext,
    bins=DEFAULT_BINS,
    ranks=(1, 5, 10),
) -> list[dict]:
    """Group query mAP and CMC by crop-coverage bucket."""

    rows: list[dict] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        query_indexes = [
            index
            for index, record in enumerate(ctx.query_records)
            if record.crop_coverage is not None
            and lower <= float(record.crop_coverage) < upper
        ]
        result = retrieval_metrics(ctx, ranks=ranks, query_indexes=query_indexes)
        rows.append(
            {
                "coverage_min": lower,
                "coverage_max": upper,
                "queries": len(query_indexes),
                "eligible_queries": result["eligible_queries"],
                "queries_without_match": result["queries_without_match"],
                "mAP": result["mAP"],
                **{f"CMC@{rank}": result["cmc"][str(rank)] for rank in ranks},
            }
        )
    return rows
