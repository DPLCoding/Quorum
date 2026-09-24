"""Formula and missing-evidence tests for the frozen V0 momentum expert."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.quorum import MomentumExpert, PredictionContext

UTC = timezone.utc


def _case(closes: list[object]) -> tuple[dict[str, object], PredictionContext]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = [start + timedelta(days=index) for index in range(len(closes))]
    data: dict[str, object] = {
        "asset": "MOM",
        "close": closes,
        "event_at": events,
        "available_at": events.copy(),
    }
    context = PredictionContext(
        event_at=events[-1],
        available_at=events[-1],
        decision_at=events[-1],
        horizon_bars=3,
    )
    return data, context


def _score(closes: list[object]) -> float:
    data, context = _case(closes)
    return MomentumExpert().predict(data, context).predictions[0].score


def test_momentum_exact_golden_formula_and_metadata() -> None:
    closes = [100.0] + [101.0] * 19 + [105.0]
    data, context = _case(closes)

    prediction = MomentumExpert().predict(data, context).predictions[0]

    assert prediction.metadata["raw_return"] == pytest.approx(0.05)
    assert prediction.score == pytest.approx(0.5)
    assert prediction.metadata == {
        "config_id": "quorum:momentum:v0",
        "lookback_bars": 20,
        "return_scale": 0.10,
        "raw_return": pytest.approx(0.05),
    }


def test_momentum_direction_neutrality_and_saturation() -> None:
    assert _score([100.0 + index for index in range(21)]) > 0.0
    assert _score([120.0 - index for index in range(21)]) < 0.0
    assert _score([100.0] + [110.0] * 19 + [100.0]) == 0.0
    assert _score([100.0] * 20 + [120.0]) == 1.0
    assert _score([100.0] * 20 + [80.0]) == -1.0


def test_momentum_warmup_is_not_neutral_evidence() -> None:
    data, context = _case([100.0] * 20)
    assert MomentumExpert().predict(data, context).predictions == ()

    data, context = _case([100.0] * 21)
    result = MomentumExpert().predict(data, context)
    assert len(result.predictions) == 1
    assert result.predictions[0].score == 0.0


def test_momentum_interior_active_missing_value_is_not_dropped() -> None:
    closes: list[object] = [100.0 + index for index in range(21)]
    closes[10] = None
    data, context = _case(closes)
    assert MomentumExpert().predict(data, context).predictions == ()
