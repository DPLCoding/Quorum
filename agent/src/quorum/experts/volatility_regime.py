"""Frozen v1 volatility-regime expert (docs/quorum/EXPERTS_V1.md, section 1)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _clamp_score,
    _ImmutableExpertMeta,
    _normalize_prepared_ohlcv_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class VolatilityRegimeExpert(metaclass=_ImmutableExpertMeta):
    """Bearish when 20-bar realized volatility is high versus its 252-bar level.

    Volatility is uncentered RMS of log returns, so return direction never
    enters: negating every return leaves the score unchanged.
    """

    expert_id: ClassVar[str] = "quorum.volatility_regime"
    expert_version: ClassVar[str] = "v1.0.0"
    config_id: ClassVar[str] = "quorum:volatility-regime:v1"
    SHORT_RETURNS: ClassVar[int] = 20
    LONG_RETURNS: ClassVar[int] = 252
    LOG_RATIO_SCALE: ClassVar[float] = 0.5
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {
            "expert_id",
            "expert_version",
            "config_id",
            "SHORT_RETURNS",
            "LONG_RETURNS",
            "LOG_RATIO_SCALE",
        }
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        series = _normalize_prepared_ohlcv_series(prepared_data, context)
        required = self.LONG_RETURNS + 1
        if len(series.close) < required:
            return ExpertResult(())
        active = series.close[-required:]
        if any(value is None for value in active):
            return ExpertResult(())
        closes = cast(tuple[float, ...], active)
        squared = [math.log(b / a) ** 2 for a, b in zip(closes, closes[1:])]
        short = math.sqrt(
            math.fsum(squared[-self.SHORT_RETURNS :]) / self.SHORT_RETURNS
        )
        long = math.sqrt(math.fsum(squared) / self.LONG_RETURNS)
        if short == 0.0 or long == 0.0:
            return ExpertResult(())
        raw = math.log(short / long)
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=_clamp_score(-raw / self.LOG_RATIO_SCALE),
            metadata={
                "config_id": self.config_id,
                "short_rms_vol": short,
                "long_rms_vol": long,
                "raw_log_vol_ratio": raw,
            },
        )
