"""Frozen v1 range/intraday-reversal expert (docs/quorum/EXPERTS_V1.md, section 4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _clamp_score,
    _ImmutableExpertMeta,
    _normalize_prepared_ohlcv_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class RangeReversalExpert(metaclass=_ImmutableExpertMeta):
    """Bearish after closes near the day's high, bullish after closes near the low.

    Close-location value ``(2C - H - L) / (H - L)`` averaged over 5 bars and
    negated. A zero-range bar (``H == L``) has CLV 0 by definition.
    """

    expert_id: ClassVar[str] = "quorum.range_reversal"
    expert_version: ClassVar[str] = "v1.0.0"
    config_id: ClassVar[str] = "quorum:range-reversal:v1"
    BARS: ClassVar[int] = 5
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {"expert_id", "expert_version", "config_id", "BARS"}
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        series = _normalize_prepared_ohlcv_series(prepared_data, context)
        if len(series.close) < self.BARS:
            return ExpertResult(())
        rows = list(
            zip(
                series.high[-self.BARS :],
                series.low[-self.BARS :],
                series.close[-self.BARS :],
            )
        )
        if any(value is None for row in rows for value in row):
            return ExpertResult(())
        clv = [
            0.0 if high == low else (2.0 * close - high - low) / (high - low)
            for high, low, close in rows  # type: ignore[operator]
        ]
        mean_clv = fmean(clv)
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=_clamp_score(-mean_clv),
            metadata={"config_id": self.config_id, "mean_close_location": mean_clv},
        )
