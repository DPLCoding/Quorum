"""Frozen V0 single-asset simple-moving-average trend expert."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar, cast

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _ImmutableExpertMeta,
    _clamp_score,
    _normalize_prepared_close_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class TrendExpert(metaclass=_ImmutableExpertMeta):
    """Emit evidence from the frozen SMA10-to-SMA50 relative spread."""

    expert_id: ClassVar[str] = "quorum.trend"
    expert_version: ClassVar[str] = "v0.1.0"
    config_id: ClassVar[str] = "quorum:trend:v0"
    FAST_WINDOW: ClassVar[int] = 10
    SLOW_WINDOW: ClassVar[int] = 50
    SPREAD_SCALE: ClassVar[float] = 0.05
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {
            "expert_id",
            "expert_version",
            "config_id",
            "FAST_WINDOW",
            "SLOW_WINDOW",
            "SPREAD_SCALE",
        }
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        """Return one causal prediction, or no evidence during warmup/missingness."""
        series = _normalize_prepared_close_series(prepared_data, context)
        if len(series.close) < self.SLOW_WINDOW:
            return ExpertResult(())
        active = series.close[-self.SLOW_WINDOW :]
        if any(value is None for value in active):
            return ExpertResult(())

        values = cast(tuple[float, ...], active)
        fast_mean = fmean(values[-self.FAST_WINDOW :])
        slow_mean = fmean(values)
        raw_spread = fast_mean / slow_mean - 1.0
        score = _clamp_score(raw_spread / self.SPREAD_SCALE)
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=score,
            metadata={
                "config_id": self.config_id,
                "fast_window_bars": self.FAST_WINDOW,
                "slow_window_bars": self.SLOW_WINDOW,
                "spread_scale": self.SPREAD_SCALE,
                "raw_spread": raw_spread,
            },
        )
