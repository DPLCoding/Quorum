"""Out-of-fold prediction-quality metrics for Quorum experts and ensembles.

Evaluation is deliberately separate from P&L: it asks whether scores rank and
sign realized forward returns before any sizing, costs, or execution. Inputs
are already-realized OOF rows; this module fits nothing and reads no market data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

_COLUMNS = frozenset(
    {"predictor", "asset", "event_at", "fold_index", "role", "score", "forward_return"}
)
_SATURATION_TOLERANCE = 1e-12


def _number(value: float) -> float | None:
    """Return a JSON-safe float; undefined statistics become ``None``."""
    value = float(value)
    return value if math.isfinite(value) else None


def _rank_ic(score: pd.Series, forward_return: pd.Series) -> float | None:
    """Spearman rank correlation, computed as Pearson over average ranks."""
    if len(score) < 3 or score.nunique() < 2 or forward_return.nunique() < 2:
        return None
    return _number(score.rank().corr(forward_return.rank()))


def _predictor_metrics(rows: pd.DataFrame, slots: int) -> dict[str, Any]:
    fold_ics = [
        ic
        for _, fold in rows.groupby("fold_index", sort=True)
        if (ic := _rank_ic(fold["score"], fold["forward_return"])) is not None
    ]
    fold_series = pd.Series(fold_ics, dtype="float64")
    fold_std = fold_series.std(ddof=1) if len(fold_ics) > 1 else float("nan")
    signed = rows[(rows["score"] != 0.0) & (rows["forward_return"] != 0.0)]
    hits = (signed["score"] > 0.0) == (signed["forward_return"] > 0.0)
    return {
        "count": len(rows),
        "coverage": _number(len(rows) / slots) if slots else None,
        "ic": _rank_ic(rows["score"], rows["forward_return"]),
        "fold_count": len(fold_ics),
        "fold_ic_mean": _number(fold_series.mean()) if fold_ics else None,
        "fold_ic_std": _number(fold_std),
        "fold_ic_t": (
            _number(fold_series.mean() / fold_std * math.sqrt(len(fold_ics)))
            if len(fold_ics) > 1 and fold_std > 0.0
            else None
        ),
        "fold_ic_positive_fraction": (
            _number((fold_series > 0.0).mean()) if fold_ics else None
        ),
        "hit_count": len(signed),
        "hit_rate": _number(hits.mean()) if len(signed) else None,
        # Base rate for hit_rate: share of the same rows whose outcome rose.
        "up_rate": (
            _number((signed["forward_return"] > 0.0).mean()) if len(signed) else None
        ),
        "score_mean": _number(rows["score"].mean()),
        "score_std": _number(rows["score"].std(ddof=1)) if len(rows) > 1 else None,
        "saturation": _number(
            (rows["score"].abs() >= 1.0 - _SATURATION_TOLERANCE).mean()
        ),
    }


def _score_correlation(rows: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Rank correlation of predictor scores on rows where every predictor exists."""
    wide = rows.pivot_table(
        index=["asset", "event_at"], columns="predictor", values="score"
    ).dropna()
    matrix = wide.rank().corr()
    return {
        str(left): {str(right): _number(matrix.loc[left, right]) for right in matrix}
        for left in matrix.index
    }


def evaluate_predictions(
    rows: pd.DataFrame, *, slot_counts: Mapping[str, int]
) -> dict[str, Any]:
    """Return JSON-safe OOF metrics per role and predictor.

    ``rows`` holds one realized prediction per (predictor, asset, event_at) with
    columns ``predictor, asset, event_at, fold_index, role, score,
    forward_return``. ``slot_counts`` gives the authorized (asset, slot) count
    per role so missing predictions show up as coverage below one.

    Per-fold IC t-statistics treat folds as independent. Rows inside a fold
    with overlapping forward horizons are not independent, so pooled IC carries
    no significance claim.
    """
    missing = _COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"evaluation rows are missing columns: {sorted(missing)}")
    if rows.duplicated(["predictor", "asset", "event_at"]).any():
        raise ValueError("evaluation rows contain duplicate predictions")

    roles: dict[str, Any] = {}
    for role, role_rows in rows.groupby("role", sort=True):
        slots = int(slot_counts[role])
        roles[str(role)] = {
            "slot_count": slots,
            "predictors": {
                str(name): _predictor_metrics(group, slots)
                for name, group in role_rows.groupby("predictor", sort=True)
            },
            "score_correlation": _score_correlation(role_rows),
            "asset_ic": {
                str(name): {
                    str(asset): _rank_ic(g["score"], g["forward_return"])
                    for asset, g in group.groupby("asset", sort=True)
                }
                for name, group in role_rows.groupby("predictor", sort=True)
            },
        }
    return {"roles": roles}


__all__ = ["evaluate_predictions"]
