"""Formula and missing-evidence tests for the frozen V0 mean-reversion expert."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.quorum import MeanReversionExpert, PredictionContext

UTC = timezone.utc


def _case(closes: list[object]) -> tuple[dict[str, object], PredictionContext]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = [start + timedelta(days=index) for index in range(len(closes))]
    data: dict[str, object] = {
        "asset": "REVERSION",
        "close": closes,
        "event_at": events,
        "available_at": events.copy(),
    }
    context = PredictionContext(
        event_at=events[-1],
        available_at=events[-1],
        decision_at=events[-1],
        horizon_bars=2,
    )
    return data, context


def _score(closes: list[object]) -> float:
    data, context = _case(closes)
    return MeanReversionExpert().predict(data, context).predictions[0].score


def test_mean_reversion_exact_golden_sample_standard_deviation() -> None:
    closes = [float(value) for value in range(81, 101)]
    data, context = _case(closes)
    expected_mean = sum(closes) / 20.0
    sum_squared_deviations = sum((value - expected_mean) ** 2 for value in closes)
    expected_std = math.sqrt(sum_squared_deviations / 19.0)
    expected_zscore = (closes[-1] - expected_mean) / expected_std
    expected_score = -expected_zscore / 3.0

    prediction = MeanReversionExpert().predict(data, context).predictions[0]

    assert prediction.metadata["raw_zscore"] == pytest.approx(expected_zscore)
    assert prediction.score == pytest.approx(expected_score)
    assert prediction.metadata["window_bars"] == 20
    assert prediction.metadata["zscore_scale"] == 3.0


def test_mean_reversion_direction_neutrality_and_saturation() -> None:
    assert _score([100.0] * 19 + [95.0]) > 0.0
    assert _score([100.0] * 19 + [105.0]) < 0.0
    assert _score([100.0] * 19 + [1.0]) == 1.0
    assert _score([100.0] * 19 + [1000.0]) == -1.0
    centered = [100.0] * 17 + [90.0, 110.0, 100.0]
    assert _score(centered) == 0.0


def test_mean_reversion_warmup_and_zero_variance_produce_no_evidence() -> None:
    data, context = _case([100.0] * 19)
    assert MeanReversionExpert().predict(data, context).predictions == ()

    data, context = _case([100.0] * 20)
    assert MeanReversionExpert().predict(data, context).predictions == ()


def test_mean_reversion_interior_active_missing_value_is_not_dropped() -> None:
    closes: list[object] = [100.0 + index for index in range(20)]
    closes[5] = None
    data, context = _case(closes)
    assert MeanReversionExpert().predict(data, context).predictions == ()
