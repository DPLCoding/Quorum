"""Frozen v1 abnormal-volume expert (docs/quorum/EXPERTS_V1.md, section 3)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import ClassVar, cast

from src.quorum.contracts import ExpertResult, PredictionContext
from src.quorum.experts.common import (
    _clamp_score,
    _ImmutableExpertMeta,
    _normalize_prepared_ohlcv_series,
    _prediction_result,
)


@dataclass(frozen=True, slots=True)
class AbnormalVolumeExpert(metaclass=_ImmutableExpertMeta):
    """High-volume return premium: bullish when recent volume is abnormally high.

    Reads volume only, never price, so it cannot reduce to momentum times
    volume: the z-score of the latest 5 bars' mean log volume against the
    preceding 60 bars.
    """

    expert_id: ClassVar[str] = "quorum.abnormal_volume"
    expert_version: ClassVar[str] = "v1.0.0"
    config_id: ClassVar[str] = "quorum:abnormal-volume:v1"
    RECENT_BARS: ClassVar[int] = 5
    BASELINE_BARS: ClassVar[int] = 60
    Z_SCALE: ClassVar[float] = 2.0
    _FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset(
        {
            "expert_id",
            "expert_version",
            "config_id",
            "RECENT_BARS",
            "BASELINE_BARS",
            "Z_SCALE",
        }
    )

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        series = _normalize_prepared_ohlcv_series(prepared_data, context)
        required = self.RECENT_BARS + self.BASELINE_BARS
        if len(series.volume) < required:
            return ExpertResult(())
        active = series.volume[-required:]
        if any(value is None or value <= 0.0 for value in active):
            return ExpertResult(())
        logs = [math.log(v) for v in cast(tuple[float, ...], active)]
        baseline, recent = logs[: self.BASELINE_BARS], logs[self.BASELINE_BARS :]
        spread = stdev(baseline)
        if not math.isfinite(spread) or spread <= 0.0:
            return ExpertResult(())
        z = (fmean(recent) - fmean(baseline)) / spread
        return _prediction_result(
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            series=series,
            context=context,
            score=_clamp_score(z / self.Z_SCALE),
            metadata={"config_id": self.config_id, "raw_log_volume_z": z},
        )
