"""Immutable scientific contracts for Quorum research.

This module is deliberately independent of market-data loaders, backtesting,
agents, APIs, and trading infrastructure.  It describes data and declared
research policy; it does not fetch, fit, vote, size positions, or execute.

All temporal intervals are half-open: ``[start, end)``.  Datetimes retain the
timezone supplied by the caller.  Comparisons use the represented instant, but
the contracts never silently convert a timestamp to another timezone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

_SCHEMA_VERSION = 1
# Prediction schema v2 adds the required ``decision_at`` cutoff. Version 1 is
# intentionally not inferred on load because treating ``event_at`` as that
# cutoff recreates the information-time ambiguity this version removes.
_PREDICTION_SCHEMA_VERSION = 2
_MAX_METADATA_BYTES = 4096
_MAX_METADATA_DEPTH = 4
_UINT32_MAX = (2**32) - 1


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Return compact, deterministic JSON for a contract payload."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_mapping(text: str, contract_name: str) -> Mapping[str, Any]:
    """Parse a JSON object, rejecting non-object top-level values."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{contract_name} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{contract_name} JSON must contain an object")
    return value


def _require_payload_keys(
    data: Mapping[str, Any],
    *,
    contract_name: str,
    fields: frozenset[str],
    schema_version: int = _SCHEMA_VERSION,
) -> None:
    """Reject missing, unknown, or mismatched contract payload fields."""
    expected = fields | {"contract", "schema_version"}
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"invalid {contract_name} payload: {', '.join(details)}")
    if data["contract"] != contract_name:
        raise ValueError(
            f"expected contract={contract_name!r}, got {data['contract']!r}"
        )
    if data["schema_version"] != schema_version:
        raise ValueError(
            f"unsupported {contract_name} schema_version " f"{data['schema_version']!r}"
        )


def _payload(
    contract_name: str,
    *,
    schema_version: int = _SCHEMA_VERSION,
    **fields: Any,
) -> dict[str, Any]:
    """Build a versioned, JSON-serializable contract payload."""
    return {
        "contract": contract_name,
        "schema_version": schema_version,
        **fields,
    }


def _required_text(name: str, value: Any) -> str:
    """Validate a non-empty identifier without silently normalizing it."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    return value


def _optional_text(name: str, value: Any) -> str | None:
    """Validate an optional identifier."""
    if value is None:
        return None
    return _required_text(name, value)


def _finite_float(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return a bounded finite real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not minimum <= converted <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return converted


def _integer(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Validate an integer, explicitly rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return converted


def _aware_datetime(name: str, value: Any) -> datetime:
    """Validate a timezone-aware datetime without converting its timezone."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _parse_datetime(name: str, value: Any) -> datetime:
    """Parse an ISO-8601 datetime and apply the awareness contract."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 datetime") from exc
    return _aware_datetime(name, parsed)


def _validate_information_time(
    event_at: Any, available_at: Any, decision_at: Any
) -> tuple[datetime, datetime, datetime]:
    """Validate the prediction information boundary.

    ``event_at`` is the timestamp associated with the underlying observation or
    event. ``available_at`` is the latest availability timestamp among the
    information used. ``decision_at`` is the prediction cutoff. The only
    universal ordering invariant is therefore ``available_at <= decision_at``
    when compared as actual instants.

    No order is imposed between ``event_at`` and either other timestamp. A news
    event, filing period, or economic observation can precede publication, while
    a scheduled target event can be associated with a prediction made earlier.

    This structural check does not prove that a feature or source publication
    time is semantically correct; later validation must audit that provenance.
    """
    event = _aware_datetime("event_at", event_at)
    available = _aware_datetime("available_at", available_at)
    decision = _aware_datetime("decision_at", decision_at)
    if available > decision:
        raise ValueError("available_at must not be later than decision_at")
    return event, available, decision


def _freeze_json_value(value: Any, *, depth: int) -> Any:
    """Copy a JSON-compatible value into a recursively immutable form."""
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError(f"metadata nesting exceeds {_MAX_METADATA_DEPTH} levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metadata keys must be non-empty strings")
            items.append((key, _freeze_json_value(item, depth=depth + 1)))
        return MappingProxyType(dict(sorted(items)))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, depth=depth + 1) for item in value)
    raise TypeError(
        "metadata values must be JSON-compatible scalars, mappings, or sequences"
    )


def _thaw_json_value(value: Any) -> Any:
    """Return a mutable JSON representation of a frozen metadata value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _immutable_metadata(metadata: Any) -> Mapping[str, Any]:
    """Validate, size-limit, copy, and recursively freeze metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen = _freeze_json_value(metadata, depth=0)
    serialized = _canonical_json(_thaw_json_value(frozen))
    if len(serialized.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError(f"metadata must not exceed {_MAX_METADATA_BYTES} UTF-8 bytes")
    return frozen


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Approved timing and provenance supplied to an expert.

    ``event_at`` identifies the underlying observation or event.
    ``available_at`` is the latest declared availability time of any supplied
    information. ``decision_at`` is the unambiguous prediction cutoff and must
    not precede ``available_at``. No universal ordering is imposed between
    ``event_at`` and the other timestamps. ``horizon_bars`` is a strictly
    positive number of bars; the bar interval is deliberately defined outside
    this contract.

    ``experiment_id`` and ``split_id`` are references only. A split cannot be
    named without its owning experiment.
    """

    event_at: datetime
    available_at: datetime
    decision_at: datetime
    horizon_bars: int
    experiment_id: str | None = None
    split_id: str | None = None

    def __post_init__(self) -> None:
        event, available, decision = _validate_information_time(
            self.event_at, self.available_at, self.decision_at
        )
        object.__setattr__(self, "event_at", event)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "decision_at", decision)
        object.__setattr__(
            self,
            "horizon_bars",
            _integer("horizon_bars", self.horizon_bars, minimum=1),
        )
        experiment_id = _optional_text("experiment_id", self.experiment_id)
        split_id = _optional_text("split_id", self.split_id)
        if split_id is not None and experiment_id is None:
            raise ValueError("split_id requires experiment_id")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "split_id", split_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "prediction_context",
            schema_version=_PREDICTION_SCHEMA_VERSION,
            event_at=self.event_at.isoformat(),
            available_at=self.available_at.isoformat(),
            decision_at=self.decision_at.isoformat(),
            horizon_bars=self.horizon_bars,
            experiment_id=self.experiment_id,
            split_id=self.split_id,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionContext":
        """Reconstruct a context from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="prediction_context",
            schema_version=_PREDICTION_SCHEMA_VERSION,
            fields=frozenset(
                {
                    "event_at",
                    "available_at",
                    "decision_at",
                    "horizon_bars",
                    "experiment_id",
                    "split_id",
                }
            ),
        )
        return cls(
            event_at=_parse_datetime("event_at", data["event_at"]),
            available_at=_parse_datetime("available_at", data["available_at"]),
            decision_at=_parse_datetime("decision_at", data["decision_at"]),
            horizon_bars=data["horizon_bars"],
            experiment_id=data["experiment_id"],
            split_id=data["split_id"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "PredictionContext":
        """Reconstruct a context from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "prediction_context"))


@dataclass(frozen=True, slots=True)
class ExpertPrediction:
    """One immutable item of expert evidence.

    ``score`` is finite evidence in ``[-1, 1]``: ``-1`` is strongly bearish,
    ``0`` neutral, and ``+1`` strongly bullish. It is not a position size,
    leverage instruction, or trade order.

    ``probability_up``, when present, is a genuine probability estimate for the
    expert's declared target over ``horizon_bars``. It is never derived here
    from score, rank, z-score, or confidence. ``confidence``, when present, is
    a separately defined finite quantity in ``[0, 1]``; it is not probability,
    absolute score, or expected return magnitude. Its definition belongs in
    the expert specification or small metadata.

    ``event_at``, ``available_at``, and ``decision_at`` obey the information-time
    model documented by :class:`PredictionContext`. The check captures a
    declared boundary but cannot by itself prove semantic absence of lookahead.
    """

    expert_id: str
    expert_version: str
    asset: str
    event_at: datetime
    available_at: datetime
    decision_at: datetime
    horizon_bars: int
    score: float
    probability_up: float | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    experiment_id: str | None = None
    split_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expert_id", _required_text("expert_id", self.expert_id)
        )
        object.__setattr__(
            self,
            "expert_version",
            _required_text("expert_version", self.expert_version),
        )
        object.__setattr__(self, "asset", _required_text("asset", self.asset))
        event, available, decision = _validate_information_time(
            self.event_at, self.available_at, self.decision_at
        )
        object.__setattr__(self, "event_at", event)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "decision_at", decision)
        object.__setattr__(
            self,
            "horizon_bars",
            _integer("horizon_bars", self.horizon_bars, minimum=1),
        )
        object.__setattr__(
            self,
            "score",
            _finite_float("score", self.score, minimum=-1.0, maximum=1.0),
        )
        if self.probability_up is not None:
            object.__setattr__(
                self,
                "probability_up",
                _finite_float(
                    "probability_up",
                    self.probability_up,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_float("confidence", self.confidence, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))
        experiment_id = _optional_text("experiment_id", self.experiment_id)
        split_id = _optional_text("split_id", self.split_id)
        if split_id is not None and experiment_id is None:
            raise ValueError("split_id requires experiment_id")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "split_id", split_id)

    @property
    def prediction_key(self) -> tuple[str, str, str, datetime, datetime, int]:
        """Return the logical row identity used for duplicate detection.

        Availability, score, confidence, probability, metadata, and provenance
        references are deliberately excluded. Two rows from the same expert
        version for the same asset, event instant, decision cutoff, and horizon
        are conflicting revisions of one prediction rather than independent
        observations. A later cutoff is a distinct prediction even when it is
        associated with the same underlying event.
        """
        return (
            self.expert_id,
            self.expert_version,
            self.asset,
            self.event_at,
            self.decision_at,
            self.horizon_bars,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "expert_prediction",
            schema_version=_PREDICTION_SCHEMA_VERSION,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            asset=self.asset,
            event_at=self.event_at.isoformat(),
            available_at=self.available_at.isoformat(),
            decision_at=self.decision_at.isoformat(),
            horizon_bars=self.horizon_bars,
            score=self.score,
            probability_up=self.probability_up,
            confidence=self.confidence,
            metadata=_thaw_json_value(self.metadata),
            experiment_id=self.experiment_id,
            split_id=self.split_id,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertPrediction":
        """Reconstruct a prediction from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="expert_prediction",
            schema_version=_PREDICTION_SCHEMA_VERSION,
            fields=frozenset(
                {
                    "expert_id",
                    "expert_version",
                    "asset",
                    "event_at",
                    "available_at",
                    "decision_at",
                    "horizon_bars",
                    "score",
                    "probability_up",
                    "confidence",
                    "metadata",
                    "experiment_id",
                    "split_id",
                }
            ),
        )
        return cls(
            expert_id=data["expert_id"],
            expert_version=data["expert_version"],
            asset=data["asset"],
            event_at=_parse_datetime("event_at", data["event_at"]),
            available_at=_parse_datetime("available_at", data["available_at"]),
            decision_at=_parse_datetime("decision_at", data["decision_at"]),
            horizon_bars=data["horizon_bars"],
            score=data["score"],
            probability_up=data["probability_up"],
            confidence=data["confidence"],
            metadata=data["metadata"],
            experiment_id=data["experiment_id"],
            split_id=data["split_id"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExpertPrediction":
        """Reconstruct a prediction from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "expert_prediction"))


def _prediction_sort_key(prediction: ExpertPrediction) -> tuple[Any, ...]:
    """Return the canonical batch ordering for predictions."""
    return prediction.prediction_key


@dataclass(frozen=True, slots=True)
class ExpertResult:
    """Immutable, canonically ordered batch of standardized predictions.

    A batch may contain multiple expert identities. Duplicate logical rows are
    rejected using :attr:`ExpertPrediction.prediction_key`. Input order is not
    part of scientific identity; rows are stored in canonical key order.
    """

    predictions: tuple[ExpertPrediction, ...] = ()

    def __post_init__(self) -> None:
        predictions = tuple(self.predictions)
        for prediction in predictions:
            if not isinstance(prediction, ExpertPrediction):
                raise TypeError("predictions must contain only ExpertPrediction rows")
        keys = [prediction.prediction_key for prediction in predictions]
        if len(keys) != len(set(keys)):
            raise ValueError("predictions contain duplicate logical prediction keys")
        object.__setattr__(
            self,
            "predictions",
            tuple(sorted(predictions, key=_prediction_sort_key)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "expert_result",
            predictions=[prediction.to_dict() for prediction in self.predictions],
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertResult":
        """Reconstruct a result from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="expert_result",
            fields=frozenset({"predictions"}),
        )
        rows = data["predictions"]
        if not isinstance(rows, list):
            raise TypeError("predictions must be a list")
        return cls(tuple(ExpertPrediction.from_dict(row) for row in rows))

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExpertResult":
        """Reconstruct a result from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "expert_result"))


@runtime_checkable
class ExpertProtocol(Protocol):
    """Structural interface for a Quorum expert.

    ``prepared_data`` is an approved, read-only mapping supplied by the future
    coordinator; an expert receives no loader or split-control authority.
    Implementations return standardized predictions and must not fetch market
    data, open holdouts, size positions, execute trades, or mutate experiment
    state. Those restrictions are architectural responsibilities and cannot be
    proven by a Python protocol alone.
    """

    @property
    def expert_id(self) -> str:
        """Stable logical expert identifier."""
        ...

    @property
    def expert_version(self) -> str:
        """Immutable expert implementation/parameter version."""
        ...

    def predict(
        self,
        prepared_data: Mapping[str, object],
        context: PredictionContext,
    ) -> ExpertResult:
        """Return standardized predictions for approved prepared data."""
        ...


@dataclass(frozen=True, slots=True)
class ExpertWeight:
    """One versioned expert and its fixed positive ensemble weight."""

    expert_id: str
    expert_version: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expert_id", _required_text("expert_id", self.expert_id)
        )
        object.__setattr__(
            self,
            "expert_version",
            _required_text("expert_version", self.expert_version),
        )
        weight = _finite_float("weight", self.weight, minimum=0.0, maximum=1.0)
        if weight <= 0.0:
            raise ValueError("weight must be strictly positive")
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "expert_weight",
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            weight=self.weight,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertWeight":
        """Reconstruct a weight from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="expert_weight",
            fields=frozenset({"expert_id", "expert_version", "weight"}),
        )
        return cls(
            expert_id=data["expert_id"],
            expert_version=data["expert_version"],
            weight=data["weight"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExpertWeight":
        """Reconstruct a weight from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "expert_weight"))


@dataclass(frozen=True, slots=True)
class StaticEnsembleConfig:
    """Frozen configuration for a future static weighted ensemble.

    Weights are fixed research parameters: every weight must be finite,
    strictly positive, and the weights must already sum to one within a
    ``1e-12`` absolute floating-point tolerance. The contract never normalizes
    caller input. A logical ``expert_id`` may appear only once, even under a
    different version.

    ``sell_threshold`` and ``buy_threshold`` define a future reporting-only
    neutral band. Both lie in ``[-1, 1]`` and ``sell_threshold`` must be strictly
    less than ``buy_threshold``. This class contains no voting implementation.
    """

    experts: tuple[ExpertWeight, ...]
    sell_threshold: float
    buy_threshold: float

    def __post_init__(self) -> None:
        experts = tuple(self.experts)
        if not experts:
            raise ValueError("an ensemble must contain at least one expert")
        for expert in experts:
            if not isinstance(expert, ExpertWeight):
                raise TypeError("experts must contain only ExpertWeight values")
        ids = [expert.expert_id for expert in experts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate logical expert_id values are not allowed")
        total = math.fsum(expert.weight for expert in experts)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "ensemble weights must sum to 1.0; no normalization is applied"
            )
        sell = _finite_float(
            "sell_threshold", self.sell_threshold, minimum=-1.0, maximum=1.0
        )
        buy = _finite_float(
            "buy_threshold", self.buy_threshold, minimum=-1.0, maximum=1.0
        )
        if sell >= buy:
            raise ValueError("sell_threshold must be strictly less than buy_threshold")
        object.__setattr__(
            self,
            "experts",
            tuple(
                sorted(experts, key=lambda item: (item.expert_id, item.expert_version))
            ),
        )
        object.__setattr__(self, "sell_threshold", sell)
        object.__setattr__(self, "buy_threshold", buy)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "static_ensemble_config",
            experts=[expert.to_dict() for expert in self.experts],
            sell_threshold=self.sell_threshold,
            buy_threshold=self.buy_threshold,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticEnsembleConfig":
        """Reconstruct a config from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="static_ensemble_config",
            fields=frozenset({"experts", "sell_threshold", "buy_threshold"}),
        )
        experts = data["experts"]
        if not isinstance(experts, list):
            raise TypeError("experts must be a list")
        return cls(
            experts=tuple(ExpertWeight.from_dict(item) for item in experts),
            sell_threshold=data["sell_threshold"],
            buy_threshold=data["buy_threshold"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StaticEnsembleConfig":
        """Reconstruct a config from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "static_ensemble_config"))


class EvaluationMode(str, Enum):
    """Supported chronological training-window policies."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


class FinalHoldoutState(str, Enum):
    """Unambiguous state of a final holdout.

    ``LOCKED`` means no final evaluation access has occurred. ``OPENED`` means
    the one declared final access has occurred and requires ``accessed_at``.
    This state records intent/history; filesystem enforcement belongs to later
    validation and experiment coordination.
    """

    LOCKED = "locked"
    OPENED = "opened"


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """A timezone-aware half-open temporal interval ``[start, end)``."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _aware_datetime("start", self.start)
        end = _aware_datetime("end", self.end)
        if start >= end:
            raise ValueError("interval start must be earlier than end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def overlaps(self, other: "TimeInterval") -> bool:
        """Return whether two half-open intervals overlap."""
        if not isinstance(other, TimeInterval):
            raise TypeError("other must be a TimeInterval")
        return self.start < other.end and other.start < self.end

    def contains(self, other: "TimeInterval") -> bool:
        """Return whether this interval fully contains ``other``."""
        if not isinstance(other, TimeInterval):
            raise TypeError("other must be a TimeInterval")
        return self.start <= other.start and other.end <= self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "time_interval",
            start=self.start.isoformat(),
            end=self.end.isoformat(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimeInterval":
        """Reconstruct an interval from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="time_interval",
            fields=frozenset({"start", "end"}),
        )
        return cls(
            start=_parse_datetime("start", data["start"]),
            end=_parse_datetime("end", data["end"]),
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "TimeInterval":
        """Reconstruct an interval from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "time_interval"))


@dataclass(frozen=True, slots=True)
class FinalHoldout:
    """Definition and access state of the final half-open holdout interval.

    A locked holdout has no access timestamp. An opened holdout requires a
    timezone-aware ``accessed_at`` no earlier than the holdout end, preventing
    the contract from representing an early peek as a valid final evaluation.
    The later coordinator must enforce the one-open transition operationally.
    """

    start: datetime
    end: datetime
    state: FinalHoldoutState
    accessed_at: datetime | None = None

    def __post_init__(self) -> None:
        interval = TimeInterval(self.start, self.end)
        object.__setattr__(self, "start", interval.start)
        object.__setattr__(self, "end", interval.end)
        if not isinstance(self.state, FinalHoldoutState):
            raise TypeError("state must be a FinalHoldoutState")
        if self.state is FinalHoldoutState.LOCKED:
            if self.accessed_at is not None:
                raise ValueError("a locked final holdout cannot have accessed_at")
            return
        if self.accessed_at is None:
            raise ValueError("an opened final holdout requires accessed_at")
        accessed_at = _aware_datetime("accessed_at", self.accessed_at)
        if accessed_at < self.end:
            raise ValueError("final holdout cannot be opened before its end")
        object.__setattr__(self, "accessed_at", accessed_at)

    @property
    def interval(self) -> TimeInterval:
        """Return the holdout's half-open interval."""
        return TimeInterval(self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "final_holdout",
            start=self.start.isoformat(),
            end=self.end.isoformat(),
            state=self.state.value,
            accessed_at=(
                self.accessed_at.isoformat() if self.accessed_at is not None else None
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalHoldout":
        """Reconstruct a holdout from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="final_holdout",
            fields=frozenset({"start", "end", "state", "accessed_at"}),
        )
        try:
            state = FinalHoldoutState(data["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid final holdout state {data['state']!r}") from exc
        accessed_at = data["accessed_at"]
        return cls(
            start=_parse_datetime("start", data["start"]),
            end=_parse_datetime("end", data["end"]),
            state=state,
            accessed_at=(
                _parse_datetime("accessed_at", accessed_at)
                if accessed_at is not None
                else None
            ),
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "FinalHoldout":
        """Reconstruct a holdout from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "final_holdout"))


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Frozen declaration of a chronological evaluation policy.

    ``EXPANDING`` uses all eligible history once ``minimum_train_bars`` is met
    and therefore requires ``train_window_bars=None``. ``ROLLING`` requires an
    explicit fixed train window at least as large as the minimum. Validation,
    test, step, purge, and embargo sizes are bar counts; no bar interval or
    asset is assumed.

    The protocol is a pre-results declaration, so its final holdout must be
    ``LOCKED``. Actual materialized boundaries and later access state belong in
    :class:`SplitManifest`; this class does not generate splits.
    """

    protocol_id: str
    mode: EvaluationMode
    minimum_train_bars: int
    train_window_bars: int | None
    validation_bars: int
    test_bars: int
    step_bars: int
    purge_bars: int
    embargo_bars: int
    final_holdout: FinalHoldout
    random_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", _required_text("protocol_id", self.protocol_id)
        )
        if not isinstance(self.mode, EvaluationMode):
            raise TypeError("mode must be an EvaluationMode")
        minimum = _integer("minimum_train_bars", self.minimum_train_bars, minimum=1)
        object.__setattr__(self, "minimum_train_bars", minimum)
        if self.mode is EvaluationMode.EXPANDING:
            if self.train_window_bars is not None:
                raise ValueError("expanding evaluation requires train_window_bars=None")
        else:
            if self.train_window_bars is None:
                raise ValueError(
                    "rolling evaluation requires explicit train_window_bars"
                )
            window = _integer("train_window_bars", self.train_window_bars, minimum=1)
            if window < minimum:
                raise ValueError("train_window_bars must be >= minimum_train_bars")
            object.__setattr__(self, "train_window_bars", window)
        for name in ("validation_bars", "test_bars", "step_bars"):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=1)
            )
        for name in ("purge_bars", "embargo_bars"):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=0)
            )
        if not isinstance(self.final_holdout, FinalHoldout):
            raise TypeError("final_holdout must be a FinalHoldout")
        if self.final_holdout.state is not FinalHoldoutState.LOCKED:
            raise ValueError("an EvaluationProtocol requires a locked final holdout")
        object.__setattr__(
            self,
            "random_seed",
            _integer("random_seed", self.random_seed, minimum=0, maximum=_UINT32_MAX),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "evaluation_protocol",
            protocol_id=self.protocol_id,
            mode=self.mode.value,
            minimum_train_bars=self.minimum_train_bars,
            train_window_bars=self.train_window_bars,
            validation_bars=self.validation_bars,
            test_bars=self.test_bars,
            step_bars=self.step_bars,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            final_holdout=self.final_holdout.to_dict(),
            random_seed=self.random_seed,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationProtocol":
        """Reconstruct a protocol from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="evaluation_protocol",
            fields=frozenset(
                {
                    "protocol_id",
                    "mode",
                    "minimum_train_bars",
                    "train_window_bars",
                    "validation_bars",
                    "test_bars",
                    "step_bars",
                    "purge_bars",
                    "embargo_bars",
                    "final_holdout",
                    "random_seed",
                }
            ),
        )
        try:
            mode = EvaluationMode(data["mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid evaluation mode {data['mode']!r}") from exc
        return cls(
            protocol_id=data["protocol_id"],
            mode=mode,
            minimum_train_bars=data["minimum_train_bars"],
            train_window_bars=data["train_window_bars"],
            validation_bars=data["validation_bars"],
            test_bars=data["test_bars"],
            step_bars=data["step_bars"],
            purge_bars=data["purge_bars"],
            embargo_bars=data["embargo_bars"],
            final_holdout=FinalHoldout.from_dict(data["final_holdout"]),
            random_seed=data["random_seed"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "EvaluationProtocol":
        """Reconstruct a protocol from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "evaluation_protocol"))


def _intervals(name: str, values: Any) -> tuple[TimeInterval, ...]:
    """Validate and canonically sort a sequence of non-overlapping intervals."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of TimeInterval values")
    intervals = tuple(values)
    for interval in intervals:
        if not isinstance(interval, TimeInterval):
            raise TypeError(f"{name} must contain only TimeInterval values")
    ordered = tuple(sorted(intervals, key=lambda item: (item.start, item.end)))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.overlaps(current):
            raise ValueError(f"{name} must not contain overlapping intervals")
    return ordered


def _reject_cross_overlap(
    left_name: str,
    left: Sequence[TimeInterval],
    right_name: str,
    right: Sequence[TimeInterval],
) -> None:
    """Reject overlap between two interval categories."""
    if any(a.overlaps(b) for a in left for b in right):
        raise ValueError(f"{left_name} must not overlap {right_name}")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Materialized temporal boundaries for one evaluation fold.

    This record is distinct from :class:`EvaluationProtocol`: a protocol says
    what was declared, while a manifest records the actual half-open intervals.
    Included train/validation/test intervals cannot overlap each other or the
    excluded purge/embargo intervals, and training must finish no later than the
    earliest validation/test boundary.

    ``uses_final_holdout=False`` requires a locked, untouched holdout and no
    included interval may overlap it. ``uses_final_holdout=True`` requires an
    opened holdout and every test interval to be fully contained by it; training
    and validation remain outside. This is logical validation only, not an
    access-control mechanism.
    """

    protocol_id: str
    split_id: str
    fold_index: int
    train_intervals: tuple[TimeInterval, ...]
    validation_intervals: tuple[TimeInterval, ...]
    test_intervals: tuple[TimeInterval, ...]
    purge_intervals: tuple[TimeInterval, ...]
    embargo_intervals: tuple[TimeInterval, ...]
    final_holdout: FinalHoldout
    uses_final_holdout: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", _required_text("protocol_id", self.protocol_id)
        )
        object.__setattr__(self, "split_id", _required_text("split_id", self.split_id))
        object.__setattr__(
            self, "fold_index", _integer("fold_index", self.fold_index, minimum=0)
        )
        for name in (
            "train_intervals",
            "validation_intervals",
            "test_intervals",
            "purge_intervals",
            "embargo_intervals",
        ):
            object.__setattr__(self, name, _intervals(name, getattr(self, name)))
        if not self.train_intervals:
            raise ValueError("train_intervals must not be empty")
        evaluation = self.validation_intervals + self.test_intervals
        if not evaluation:
            raise ValueError(
                "at least one validation_intervals or test_intervals value is required"
            )
        if not isinstance(self.uses_final_holdout, bool):
            raise TypeError("uses_final_holdout must be a bool")
        if not isinstance(self.final_holdout, FinalHoldout):
            raise TypeError("final_holdout must be a FinalHoldout")

        included = (
            ("train_intervals", self.train_intervals),
            ("validation_intervals", self.validation_intervals),
            ("test_intervals", self.test_intervals),
        )
        excluded = (
            ("purge_intervals", self.purge_intervals),
            ("embargo_intervals", self.embargo_intervals),
        )
        for index, (left_name, left) in enumerate(included):
            for right_name, right in included[index + 1 :]:
                _reject_cross_overlap(left_name, left, right_name, right)
            for right_name, right in excluded:
                _reject_cross_overlap(left_name, left, right_name, right)
        _reject_cross_overlap(
            "purge_intervals",
            self.purge_intervals,
            "embargo_intervals",
            self.embargo_intervals,
        )

        first_evaluation = min(interval.start for interval in evaluation)
        last_training = max(interval.end for interval in self.train_intervals)
        if last_training > first_evaluation:
            raise ValueError(
                "training must finish no later than the first evaluation interval"
            )

        holdout = self.final_holdout.interval
        if any(holdout.overlaps(interval) for interval in self.train_intervals):
            raise ValueError("train_intervals must not overlap the final holdout")
        if any(holdout.overlaps(interval) for interval in self.validation_intervals):
            raise ValueError("validation_intervals must not overlap the final holdout")
        if self.uses_final_holdout:
            if self.final_holdout.state is not FinalHoldoutState.OPENED:
                raise ValueError("using the final holdout requires state=OPENED")
            if not self.test_intervals:
                raise ValueError("using the final holdout requires test_intervals")
            if not all(holdout.contains(interval) for interval in self.test_intervals):
                raise ValueError(
                    "all final-holdout test_intervals must be contained by the holdout"
                )
        else:
            if self.final_holdout.state is not FinalHoldoutState.LOCKED:
                raise ValueError("an unused final holdout must remain LOCKED")
            if any(holdout.overlaps(interval) for interval in self.test_intervals):
                raise ValueError(
                    "test_intervals must not overlap a locked final holdout"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "split_manifest",
            protocol_id=self.protocol_id,
            split_id=self.split_id,
            fold_index=self.fold_index,
            train_intervals=[item.to_dict() for item in self.train_intervals],
            validation_intervals=[item.to_dict() for item in self.validation_intervals],
            test_intervals=[item.to_dict() for item in self.test_intervals],
            purge_intervals=[item.to_dict() for item in self.purge_intervals],
            embargo_intervals=[item.to_dict() for item in self.embargo_intervals],
            final_holdout=self.final_holdout.to_dict(),
            uses_final_holdout=self.uses_final_holdout,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        """Reconstruct a manifest from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="split_manifest",
            fields=frozenset(
                {
                    "protocol_id",
                    "split_id",
                    "fold_index",
                    "train_intervals",
                    "validation_intervals",
                    "test_intervals",
                    "purge_intervals",
                    "embargo_intervals",
                    "final_holdout",
                    "uses_final_holdout",
                }
            ),
        )

        def load_intervals(name: str) -> tuple[TimeInterval, ...]:
            values = data[name]
            if not isinstance(values, list):
                raise TypeError(f"{name} must be a list")
            return tuple(TimeInterval.from_dict(item) for item in values)

        return cls(
            protocol_id=data["protocol_id"],
            split_id=data["split_id"],
            fold_index=data["fold_index"],
            train_intervals=load_intervals("train_intervals"),
            validation_intervals=load_intervals("validation_intervals"),
            test_intervals=load_intervals("test_intervals"),
            purge_intervals=load_intervals("purge_intervals"),
            embargo_intervals=load_intervals("embargo_intervals"),
            final_holdout=FinalHoldout.from_dict(data["final_holdout"]),
            uses_final_holdout=data["uses_final_holdout"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "SplitManifest":
        """Reconstruct a manifest from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "split_manifest"))
