"""Out-of-fold prediction-quality metrics for Quorum experts and ensembles.

Evaluation is deliberately separate from P&L: it asks whether scores rank and
sign realized forward returns before any sizing, costs, or execution. Inputs
are already-realized OOF rows; this module fits nothing and reads no market data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
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


def _sign_agreement(rows: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Share of rows where two predictors point the same way (both non-zero)."""
    signs = np.sign(
        rows.pivot_table(
            index=["asset", "event_at"], columns="predictor", values="score"
        )
    )
    result: dict[str, dict[str, float | None]] = {}
    for left in signs.columns:
        result[str(left)] = {}
        for right in signs.columns:
            a, b = signs[left], signs[right]
            both = a.notna() & b.notna() & (a != 0) & (b != 0)
            result[str(left)][str(right)] = (
                _number((a[both] == b[both]).mean()) if both.any() else None
            )
    return result


def _grouped_ic(rows: pd.DataFrame, column: str) -> dict[str, dict[str, float | None]]:
    return {
        str(name): {
            str(key): _rank_ic(g["score"], g["forward_return"])
            for key, g in group.groupby(column, sort=True)
        }
        for name, group in rows.groupby("predictor", sort=True)
    }


def _fold_ics(rows: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            fold: ic
            for fold, group in rows.groupby("fold_index", sort=True)
            if (ic := _rank_ic(group["score"], group["forward_return"])) is not None
        },
        dtype="float64",
    )


def paired_fold_delta(
    rows: pd.DataFrame, *, base: str, other: str, role: str = "test"
) -> dict[str, Any]:
    """Per-fold IC of ``other`` minus ``base`` on the folds both can be scored.

    The t-statistic treats folds as independent, as the per-fold IC t does.
    """
    selected = rows[rows["role"] == role]
    deltas = (
        _fold_ics(selected[selected["predictor"] == other])
        - _fold_ics(selected[selected["predictor"] == base])
    ).dropna()
    std = deltas.std(ddof=1) if len(deltas) > 1 else float("nan")
    return {
        "fold_count": len(deltas),
        "mean_delta": _number(deltas.mean()) if len(deltas) else None,
        "t": (
            _number(deltas.mean() / std * math.sqrt(len(deltas)))
            if len(deltas) > 1 and std > 0.0
            else None
        ),
        "positive_fraction": _number((deltas > 0.0).mean()) if len(deltas) else None,
        "fold_deltas": [_number(value) for value in deltas],
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
            "sign_agreement": _sign_agreement(role_rows),
            "asset_ic": _grouped_ic(role_rows, "asset"),
        }
        if "regime" in role_rows.columns:
            roles[str(role)]["regime_ic"] = _grouped_ic(role_rows, "regime")
    return {"roles": roles}


def _moments(returns_a: np.ndarray, returns_b: np.ndarray) -> np.ndarray:
    return np.stack([returns_a, returns_b, returns_a**2, returns_b**2], axis=-1)


def _sharpe_gap(mu: np.ndarray) -> np.ndarray:
    """Daily Sharpe(a) - Sharpe(b) from first and second moments."""
    var_a = mu[..., 2] - mu[..., 0] ** 2
    var_b = mu[..., 3] - mu[..., 1] ** 2
    return mu[..., 0] / np.sqrt(var_a) - mu[..., 1] / np.sqrt(var_b)


def _sharpe_gap_gradient(mu: np.ndarray) -> np.ndarray:
    var_a = mu[..., 2] - mu[..., 0] ** 2
    var_b = mu[..., 3] - mu[..., 1] ** 2
    return np.stack(
        [
            mu[..., 2] / var_a**1.5,
            -mu[..., 3] / var_b**1.5,
            -mu[..., 0] / (2.0 * var_a**1.5),
            mu[..., 1] / (2.0 * var_b**1.5),
        ],
        axis=-1,
    )


def sharpe_difference_test(
    returns_a: Sequence[float],
    returns_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    block_length: int | None = None,
) -> dict[str, Any]:
    """One-sided test of H1: Sharpe(a) > Sharpe(b) on paired returns.

    Ledoit & Wolf (2008), "Robust performance hypothesis testing with the
    Sharpe ratio": delta-method standard error on the moment vector
    (a, b, a^2, b^2) with a Newey-West (Bartlett) HAC of lag equal to the block
    length, studentized by a circular block bootstrap whose resamples use the
    block-sum covariance. The block length defaults to round(n ** (1/3)); blocks
    preserve serial dependence and the pairing between the two series, so the
    test does not assume IID or normal returns (unlike Jobson-Korkie/Memmel).
    Deterministic for a given seed.
    """
    a = np.asarray(returns_a, dtype="float64")
    b = np.asarray(returns_b, dtype="float64")
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("returns must be paired one-dimensional series")
    n = len(a)
    block = block_length or max(1, round(n ** (1.0 / 3.0)))
    blocks = n // block
    if blocks < 2:
        raise ValueError("too few observations for the block bootstrap")
    y = _moments(a, b)
    mu = y.mean(axis=0)
    centered = y - mu
    psi = centered.T @ centered / n
    for lag in range(1, block + 1):
        gamma = centered[lag:].T @ centered[:-lag] / n
        psi += (1.0 - lag / (block + 1.0)) * (gamma + gamma.T)
    grad = _sharpe_gap_gradient(mu)
    gap = float(_sharpe_gap(mu))
    se = float(np.sqrt(grad @ psi @ grad / n))
    statistic = gap / se

    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    exceed = 0
    for start in range(0, resamples, 1_000):
        count = min(1_000, resamples - start)
        heads = rng.integers(0, n, size=(count, blocks))
        index = ((heads[:, :, None] + offsets) % n).reshape(count, blocks * block)
        sample = y[index]
        mu_star = sample.mean(axis=1)
        sums = (
            (sample - mu_star[:, None, :]).reshape(count, blocks, block, 4).sum(axis=2)
        )
        zeta = sums / np.sqrt(block)
        psi_star = np.einsum("mji,mjk->mik", zeta, zeta) / blocks
        g = _sharpe_gap_gradient(mu_star)
        se_star = np.sqrt(np.einsum("mi,mik,mk->m", g, psi_star, g) / (blocks * block))
        exceed += int(np.sum((_sharpe_gap(mu_star) - gap) / se_star >= statistic))
    return {
        "n": n,
        "block_length": block,
        "resamples": resamples,
        "seed": seed,
        "sharpe_a_annualized": float(
            mu[0] / np.sqrt(mu[2] - mu[0] ** 2) * math.sqrt(252)
        ),
        "sharpe_b_annualized": float(
            mu[1] / np.sqrt(mu[3] - mu[1] ** 2) * math.sqrt(252)
        ),
        "sharpe_difference_daily": gap,
        "sharpe_difference_annualized": gap * math.sqrt(252),
        "standard_error_daily": se,
        "studentized_statistic": statistic,
        "p_value_one_sided": (exceed + 1) / (resamples + 1),
    }


__all__ = ["evaluate_predictions", "paired_fold_delta", "sharpe_difference_test"]
