"""Spec tests for the four frozen v1 experts (docs/quorum/EXPERTS_V1.md)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.quorum import PredictionContext
from src.quorum.experts import (
    AbnormalVolumeExpert,
    OvernightMomentumExpert,
    RangeReversalExpert,
    VolatilityRegimeExpert,
)

UTC = timezone.utc


def _case(
    n: int,
    *,
    close=None,  # noqa: ANN001
    open_=None,  # noqa: ANN001
    high=None,  # noqa: ANN001
    low=None,  # noqa: ANN001
    volume=None,  # noqa: ANN001
    seed: int = 0,
) -> tuple[dict[str, object], PredictionContext]:
    rng = np.random.default_rng(seed)
    close = (
        list(close)
        if close is not None
        else list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    )
    open_ = list(open_) if open_ is not None else [c * 0.999 for c in close]
    high = (
        list(high)
        if high is not None
        else [max(o, c) * 1.01 for o, c in zip(open_, close)]
    )
    low = (
        list(low)
        if low is not None
        else [min(o, c) * 0.99 for o, c in zip(open_, close)]
    )
    volume = list(volume) if volume is not None else list(rng.uniform(1e6, 2e6, n))
    start = datetime(2024, 1, 1, 21, tzinfo=UTC)
    events = [start + timedelta(days=i) for i in range(n)]
    data = {
        "asset": "X",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "event_at": events,
        "available_at": events.copy(),
    }
    context = PredictionContext(
        event_at=events[-1],
        available_at=events[-1],
        decision_at=events[-1],
        horizon_bars=5,
    )
    return data, context


def _scores(expert, data, context) -> list[float]:  # noqa: ANN001
    return [p.score for p in expert.predict(data, context).predictions]


# ---------------------------------------------------------------- identities


def test_identities_are_frozen_and_distinct() -> None:
    experts = (
        VolatilityRegimeExpert,
        OvernightMomentumExpert,
        AbnormalVolumeExpert,
        RangeReversalExpert,
    )
    assert [e.expert_id for e in experts] == [
        "quorum.volatility_regime",
        "quorum.overnight_momentum",
        "quorum.abnormal_volume",
        "quorum.range_reversal",
    ]
    assert {e.expert_version for e in experts} == {"v1.0.0"}
    for expert in experts:
        with pytest.raises(AttributeError):
            expert.expert_version = "v9"  # type: ignore[misc]


# ------------------------------------------------------- volatility regime


def test_volatility_regime_golden_formula() -> None:
    data, context = _case(300, seed=1)
    closes = np.array(data["close"][-253:])
    r = np.diff(np.log(closes))
    raw = math.log(math.sqrt(np.mean(r[-20:] ** 2)) / math.sqrt(np.mean(r**2)))
    (score,) = _scores(VolatilityRegimeExpert(), data, context)
    assert score == pytest.approx(max(-1.0, min(1.0, -raw / 0.5)))


def test_volatility_regime_is_direction_free_and_bearish_on_spikes() -> None:
    data, context = _case(300, seed=2)
    closes = np.array(data["close"])
    mirrored = closes[0] ** 2 / closes  # negates every log return
    flipped, _ = _case(300, close=mirrored, seed=2)
    assert _scores(VolatilityRegimeExpert(), flipped, context) == pytest.approx(
        _scores(VolatilityRegimeExpert(), data, context)
    )

    rng = np.random.default_rng(3)
    calm_then_wild = np.concatenate(
        [rng.normal(0, 0.005, 280), rng.normal(0, 0.04, 20)]
    )
    spiky, ctx = _case(300, close=100 * np.exp(np.cumsum(calm_then_wild)))
    assert _scores(VolatilityRegimeExpert(), spiky, ctx)[0] < -0.9


def test_volatility_regime_warmup_missing_and_zero_variance() -> None:
    data, context = _case(252)
    assert _scores(VolatilityRegimeExpert(), data, context) == []
    data, context = _case(300)
    data["close"][-100] = None
    assert _scores(VolatilityRegimeExpert(), data, context) == []
    flat, context = _case(300, close=[100.0] * 300)
    assert _scores(VolatilityRegimeExpert(), flat, context) == []


# ------------------------------------------------------ overnight momentum


def test_overnight_momentum_uses_only_close_to_open_gaps() -> None:
    data, context = _case(40, seed=4)
    o = np.array(data["open"][-20:])
    prev_c = np.array(data["close"][-21:-1])
    raw = float(np.sum(np.log(o / prev_c)))
    (score,) = _scores(OvernightMomentumExpert(), data, context)
    assert score == pytest.approx(max(-1.0, min(1.0, raw / 0.05)))

    # Intraday (open->close) moves do not matter; only the gaps do.
    moved = dict(data)
    moved["close"] = list(data["close"])
    moved["close"][-1] = data["close"][-1] * 1.5
    moved["high"] = list(data["high"])
    moved["high"][-1] = moved["close"][-1] * 1.01
    assert _scores(OvernightMomentumExpert(), moved, context) == pytest.approx([score])


def test_overnight_momentum_sign_and_warmup() -> None:
    closes = [100.0] * 30
    gap_up, context = _case(30, close=closes, open_=[101.0] * 30)
    assert _scores(OvernightMomentumExpert(), gap_up, context)[0] > 0
    short, context = _case(20)
    assert _scores(OvernightMomentumExpert(), short, context) == []
    data, context = _case(30)
    data["open"][-5] = float("nan")
    assert _scores(OvernightMomentumExpert(), data, context) == []


# --------------------------------------------------------- abnormal volume


def test_abnormal_volume_golden_formula_ignores_prices() -> None:
    data, context = _case(80, seed=5)
    v = np.log(np.array(data["volume"][-65:]))
    base = v[:60]
    z = (v[60:].mean() - base.mean()) / base.std(ddof=1)
    (score,) = _scores(AbnormalVolumeExpert(), data, context)
    assert score == pytest.approx(max(-1.0, min(1.0, z / 2)))

    other_prices, _ = _case(80, seed=99, volume=data["volume"])
    assert _scores(AbnormalVolumeExpert(), other_prices, context) == pytest.approx(
        [score]
    )


def test_abnormal_volume_sign_warmup_zero_and_degenerate() -> None:
    surge, context = _case(70, volume=[1e6 + i for i in range(65)] + [5e6] * 5)
    assert _scores(AbnormalVolumeExpert(), surge, context)[0] > 0.9
    short, context = _case(64)
    assert _scores(AbnormalVolumeExpert(), short, context) == []
    zero, context = _case(70)
    zero["volume"][-3] = 0.0
    assert _scores(AbnormalVolumeExpert(), zero, context) == []
    constant, context = _case(70, volume=[1e6] * 70)
    assert _scores(AbnormalVolumeExpert(), constant, context) == []


# ---------------------------------------------------------- range reversal


def test_range_reversal_golden_formula_and_zero_range() -> None:
    data, context = _case(10, seed=6)
    h = np.array(data["high"][-5:])
    lo = np.array(data["low"][-5:])
    c = np.array(data["close"][-5:])
    clv = (2 * c - h - lo) / (h - lo)
    (score,) = _scores(RangeReversalExpert(), data, context)
    assert score == pytest.approx(-clv.mean())

    flat, context = _case(
        6, close=[50.0] * 6, open_=[50.0] * 6, high=[50.0] * 6, low=[50.0] * 6
    )
    assert _scores(RangeReversalExpert(), flat, context) == [0.0]


def test_range_reversal_sign_and_missing() -> None:
    at_high, context = _case(
        6, close=[10.0] * 6, open_=[9.0] * 6, high=[10.0] * 6, low=[8.0] * 6
    )
    assert _scores(RangeReversalExpert(), at_high, context) == [-1.0]
    data, context = _case(6)
    data["low"][-2] = None
    assert _scores(RangeReversalExpert(), data, context) == []


# -------------------------------------------------------- information time


@pytest.mark.parametrize(
    "expert",
    [
        VolatilityRegimeExpert(),
        OvernightMomentumExpert(),
        AbnormalVolumeExpert(),
        RangeReversalExpert(),
    ],
)
def test_rows_available_after_the_decision_are_rejected(expert) -> None:  # noqa: ANN001
    data, context = _case(300)
    data["available_at"] = list(data["available_at"])
    data["available_at"][-1] = context.decision_at + timedelta(minutes=1)
    with pytest.raises(ValueError, match="available_at"):
        expert.predict(data, context)
