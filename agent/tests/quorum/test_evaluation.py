"""Known-answer tests for Quorum's out-of-fold prediction evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.quorum.evaluation import evaluate_predictions

UTC = timezone.utc
RETURNS = (0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.05, -0.03)


def _rows(predictor: str, scores: list[float], *, asset: str = "A") -> list[dict]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        {
            "predictor": predictor,
            "asset": asset,
            "event_at": t0 + timedelta(days=i),
            "fold_index": i // 4,
            "role": "test",
            "score": score,
            "forward_return": RETURNS[i],
        }
        for i, score in enumerate(scores)
    ]


def _evaluate(*groups: list[dict], slots: int = 8) -> dict:
    frame = pd.DataFrame([row for group in groups for row in group])
    return evaluate_predictions(frame, slot_counts={"test": slots})


def test_perfect_and_inverted_predictors_have_known_ic_and_hit_rate() -> None:
    report = _evaluate(
        _rows("perfect", [r * 10 for r in RETURNS]),
        _rows("inverted", [-r * 10 for r in RETURNS]),
    )
    perfect = report["roles"]["test"]["predictors"]["perfect"]
    inverted = report["roles"]["test"]["predictors"]["inverted"]

    assert perfect["ic"] == pytest.approx(1.0)
    assert perfect["hit_rate"] == 1.0
    # Hit rate needs the base rate of rising outcomes to be interpretable.
    assert perfect["up_rate"] == 0.5
    assert perfect["fold_count"] == 2
    assert perfect["fold_ic_mean"] == pytest.approx(1.0)
    assert inverted["ic"] == pytest.approx(-1.0)
    assert inverted["hit_rate"] == 0.0
    assert report["roles"]["test"]["score_correlation"]["perfect"][
        "inverted"
    ] == pytest.approx(-1.0)


def test_constant_scores_report_no_ic_or_hit_rate_instead_of_nan() -> None:
    report = _evaluate(_rows("flat", [0.0] * 8))
    flat = report["roles"]["test"]["predictors"]["flat"]

    assert flat["ic"] is None
    assert flat["hit_rate"] is None
    assert flat["hit_count"] == 0
    json.dumps(report, allow_nan=False)


def test_coverage_saturation_and_per_asset_ic() -> None:
    clipped = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5]
    report = _evaluate(
        _rows("partial", clipped, asset="A"),
        slots=8,
    )
    partial = report["roles"]["test"]["predictors"]["partial"]

    assert partial["count"] == 6
    assert partial["coverage"] == pytest.approx(0.75)
    assert partial["saturation"] == pytest.approx(4 / 6)
    assert report["roles"]["test"]["asset_ic"]["partial"]["A"] == pytest.approx(
        partial["ic"]
    )
