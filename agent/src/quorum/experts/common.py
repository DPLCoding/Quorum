"""Shared validation and construction helpers for the frozen V0 experts.

The helpers in this module deliberately define only the narrow, single-asset
close-price input used by Task 4.  They do not load, align, sort, fill, or select
market data; callers must supply the already approved chronological slice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any

from src.quorum.contracts import (
    ExpertPrediction,
    ExpertResult,
    PredictionContext,
)

_PREPARED_FIELDS = frozenset({"asset", "close", "event_at", "available_at"})


class _ImmutableExpertMeta(type):
    """Prevent runtime changes to a V0 class's scientific identity or parameters."""

    def __new__(
        mcls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        if any(isinstance(base, _ImmutableExpertMeta) for base in bases):
            raise TypeError("frozen V0 expert classes cannot be subclassed")
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __setattr__(cls, name: str, value: object) -> None:
        frozen = cls.__dict__.get("_FROZEN_SCIENTIFIC_ATTRIBUTES", frozenset())
        if name == "_FROZEN_SCIENTIFIC_ATTRIBUTES" or name in frozen:
            raise AttributeError(f"{cls.__name__}.{name} is a frozen V0 attribute")
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        frozen = cls.__dict__.get("_FROZEN_SCIENTIFIC_ATTRIBUTES", frozenset())
        if name == "_FROZEN_SCIENTIFIC_ATTRIBUTES" or name in frozen:
            raise AttributeError(f"{cls.__name__}.{name} is a frozen V0 attribute")
        super().__delattr__(name)


@dataclass(frozen=True, slots=True)
class _PreparedCloseSeries:
    """Immutable normalized copy of one approved close-price history."""

    asset: str
    close: tuple[float | None, ...]
    event_at: tuple[datetime, ...]
    available_at: tuple[datetime, ...]


def _instant(value: datetime) -> datetime:
    """Return a transient UTC key without changing the stored timestamp."""
    return value.astimezone(timezone.utc)


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must contain only datetime values")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} values must be timezone-aware")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _close_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("close values must be real numbers, None, or NaN")
    result = float(value)
    if math.isnan(result):
        return None
    if not math.isfinite(result):
        raise ValueError("close values must be finite or explicitly missing")
    if result <= 0.0:
        raise ValueError("close values must be strictly positive")
    return result


def _normalize_prepared_close_series(
    prepared_data: Mapping[str, object],
    context: PredictionContext,
) -> _PreparedCloseSeries:
    """Validate and copy the exact V0 close-price prepared-data contract."""
    if not isinstance(prepared_data, Mapping):
        raise TypeError("prepared_data must be a mapping")
    if not isinstance(context, PredictionContext):
        raise TypeError("context must be a PredictionContext")

    actual_fields = set(prepared_data)
    if actual_fields != _PREPARED_FIELDS:
        missing = sorted(_PREPARED_FIELDS - actual_fields)
        unknown = sorted(repr(field) for field in actual_fields - _PREPARED_FIELDS)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"invalid prepared_data fields: {', '.join(details)}")

    asset = prepared_data["asset"]
    if not isinstance(asset, str):
        raise TypeError("asset must be a string")
    if not asset or asset != asset.strip():
        raise ValueError("asset must be non-empty without surrounding whitespace")

    close_input = _sequence("close", prepared_data["close"])
    event_input = _sequence("event_at", prepared_data["event_at"])
    available_input = _sequence("available_at", prepared_data["available_at"])
    if len(close_input) != len(event_input) or len(close_input) != len(available_input):
        raise ValueError("close, event_at, and available_at must have equal lengths")
    if not close_input:
        raise ValueError("prepared_data sequences must not be empty")

    close = tuple(_close_value(value) for value in close_input)
    event_at = tuple(_aware_datetime("event_at", value) for value in event_input)
    available_at = tuple(
        _aware_datetime("available_at", value) for value in available_input
    )

    event_instants = tuple(_instant(value) for value in event_at)
    if any(
        current >= following
        for current, following in zip(event_instants, event_instants[1:])
    ):
        raise ValueError("event_at must be strictly increasing by represented instant")

    decision_instant = _instant(context.decision_at)
    if any(_instant(value) > decision_instant for value in available_at):
        raise ValueError("available_at must not be later than context.decision_at")
    if event_instants[-1] != _instant(context.event_at):
        raise ValueError("final event_at must equal context.event_at")
    if max(_instant(value) for value in available_at) != _instant(context.available_at):
        raise ValueError("latest available_at must equal context.available_at")

    return _PreparedCloseSeries(
        asset=asset,
        close=close,
        event_at=event_at,
        available_at=available_at,
    )


def _clamp_score(value: float) -> float:
    """Clamp one finite raw score to the ExpertPrediction score range."""
    if not math.isfinite(value):
        raise ValueError("raw expert score must be finite")
    return min(1.0, max(-1.0, value))


def _prediction_result(
    *,
    expert_id: str,
    expert_version: str,
    series: _PreparedCloseSeries,
    context: PredictionContext,
    score: float,
    metadata: Mapping[str, Any],
) -> ExpertResult:
    """Construct the sole standardized prediction allowed per V0 call."""
    return ExpertResult(
        (
            ExpertPrediction(
                expert_id=expert_id,
                expert_version=expert_version,
                asset=series.asset,
                event_at=context.event_at,
                available_at=context.available_at,
                decision_at=context.decision_at,
                horizon_bars=context.horizon_bars,
                score=score,
                probability_up=None,
                confidence=None,
                metadata=metadata,
                experiment_id=context.experiment_id,
                split_id=context.split_id,
            ),
        )
    )
