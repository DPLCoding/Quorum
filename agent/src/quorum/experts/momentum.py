"""Frozen V0 single-asset trailing-return expert."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _clamp_score,
    _normalize_prepared_close_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class MomentumExpert:
    """Emit bullish/bearish evidence from the frozen 20-bar simple return."""

    expert_id: ClassVar[str] = "quorum.momentum"
    expert_version: ClassVar[str] = "v0.1.0"
    config_id: ClassVar[str] = "quorum:momentum:v0"
    LOOKBACK: ClassVar[int] = 20
    RETURN_SCALE: ClassVar[float] = 0.10

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        """Return one causal prediction, or no evidence during warmup/missingness."""
        series = _normalize_prepared_close_series(prepared_data, context)
        required = self.LOOKBACK + 1
        if len(series.close) < required:
            return ExpertResult(())
        active = series.close[-required:]
        if any(value is None for value in active):
            return ExpertResult(())

        first = active[0]
        last = active[-1]
        assert first is not None and last is not None
        raw_return = last / first - 1.0
        score = _clamp_score(raw_return / self.RETURN_SCALE)
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=score,
            metadata={
                "config_id": self.config_id,
                "lookback_bars": self.LOOKBACK,
                "return_scale": self.RETURN_SCALE,
                "raw_return": raw_return,
            },
        )
