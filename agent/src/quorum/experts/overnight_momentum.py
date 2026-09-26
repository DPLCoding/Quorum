"""Frozen v1 overnight-momentum expert (docs/quorum/EXPERTS_V1.md, section 2)."""

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
class OvernightMomentumExpert(metaclass=_ImmutableExpertMeta):
    """Bullish after a positive 20-gap sum of close-to-next-open log returns.

    Only overnight gaps ``ln(open_t / close_{t-1})`` enter; intraday
    ``ln(close_t / open_t)`` moves are deliberately ignored. The latest open
    used is the decision bar's own, available at that bar's close.
    """

    expert_id: ClassVar[str] = "quorum.overnight_momentum"
    expert_version: ClassVar[str] = "v1.0.0"
    config_id: ClassVar[str] = "quorum:overnight-momentum:v1"
    GAPS: ClassVar[int] = 20
    RETURN_SCALE: ClassVar[float] = 0.05
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {"expert_id", "expert_version", "config_id", "GAPS", "RETURN_SCALE"}
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        series = _normalize_prepared_ohlcv_series(prepared_data, context)
        required = self.GAPS + 1
        if len(series.close) < required:
            return ExpertResult(())
        closes = series.close[-required:-1]
        opens = series.open[-self.GAPS :]
        if any(value is None for value in closes + opens):
            return ExpertResult(())
        raw = math.fsum(
            math.log(o / c)
            for o, c in zip(
                cast(tuple[float, ...], opens), cast(tuple[float, ...], closes)
            )
        )
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=_clamp_score(raw / self.RETURN_SCALE),
            metadata={"config_id": self.config_id, "raw_overnight_log_sum": raw},
        )
