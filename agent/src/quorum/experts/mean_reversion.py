"""Frozen V0 single-asset price-z-score mean-reversion expert."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import ClassVar, cast

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _ImmutableExpertMeta,
    _clamp_score,
    _normalize_prepared_close_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class MeanReversionExpert(metaclass=_ImmutableExpertMeta):
    """Emit inverse evidence from the frozen 20-bar sample price z-score."""

    expert_id: ClassVar[str] = "quorum.mean_reversion"
    expert_version: ClassVar[str] = "v0.1.0"
    config_id: ClassVar[str] = "quorum:mean-reversion:v0"
    WINDOW: ClassVar[int] = 20
    ZSCORE_SCALE: ClassVar[float] = 3.0
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {"expert_id", "expert_version", "config_id", "WINDOW", "ZSCORE_SCALE"}
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        """Return one causal prediction, or no evidence without a valid z-score."""
        series = _normalize_prepared_close_series(prepared_data, context)
        if len(series.close) < self.WINDOW:
            return ExpertResult(())
        active = series.close[-self.WINDOW :]
        if any(value is None for value in active):
            return ExpertResult(())

        values = cast(tuple[float, ...], active)
        window_mean = fmean(values)
        window_std = stdev(values)
        if not math.isfinite(window_std) or window_std <= 0.0:
            return ExpertResult(())
        raw_zscore = (values[-1] - window_mean) / window_std
        score = _clamp_score(-raw_zscore / self.ZSCORE_SCALE)
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=score,
            metadata={
                "config_id": self.config_id,
                "window_bars": self.WINDOW,
                "zscore_scale": self.ZSCORE_SCALE,
                "raw_zscore": raw_zscore,
            },
        )
