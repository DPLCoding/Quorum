"""Formula and missing-evidence tests for the frozen V0 trend expert."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.quorum import PredictionContext, TrendExpert

UTC = timezone.utc


def _case(closes: list[object]) -> tuple[dict[str, object], PredictionContext]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = [start + timedelta(days=index) for index in range(len(closes))]
    data: dict[str, object] = {
        "asset": "TREND",
        "close": closes,
        "event_at": events,
        "available_at": events.copy(),
    }
    context = PredictionContext(
        event_at=events[-1],
        available_at=events[-1],
        decision_at=events[-1],
        horizon_bars=4,
    )
    return data, context


def _score(closes: list[object]) -> float:
    data, context = _case(closes)
    return TrendExpert().predict(data, context).predictions[0].score


def test_trend_exact_golden_formula_and_metadata() -> None:
    closes = [100.0] * 40 + [102.0] * 10
    data, context = _case(closes)
    expected_fast = 102.0
    expected_slow = (40.0 * 100.0 + 10.0 * 102.0) / 50.0
    expected_raw = expected_fast / expected_slow - 1.0
    expected_score = expected_raw / 0.05

    prediction = TrendExpert().predict(data, context).predictions[0]

    assert prediction.metadata["raw_spread"] == pytest.approx(expected_raw)
    assert prediction.score == pytest.approx(expected_score)
    assert prediction.metadata["fast_window_bars"] == 10
    assert prediction.metadata["slow_window_bars"] == 50
    assert prediction.metadata["spread_scale"] == 0.05


def test_trend_direction_neutrality_and_saturation() -> None:
    assert _score([100.0 + index for index in range(50)]) > 0.0
    assert _score([150.0 - index for index in range(50)]) < 0.0
    assert _score([100.0] * 50) == 0.0
    assert _score([100.0] * 40 + [200.0] * 10) == 1.0
    assert _score([200.0] * 40 + [100.0] * 10) == -1.0


def test_trend_warmup_is_not_neutral_evidence() -> None:
    data, context = _case([100.0] * 49)
    assert TrendExpert().predict(data, context).predictions == ()

    data, context = _case([100.0] * 50)
    result = TrendExpert().predict(data, context)
    assert len(result.predictions) == 1
    assert result.predictions[0].score == 0.0


def test_trend_interior_active_missing_value_is_not_dropped() -> None:
    closes: list[object] = [100.0 + index for index in range(50)]
    closes[20] = float("nan")
    data, context = _case(closes)
    assert TrendExpert().predict(data, context).predictions == ()
