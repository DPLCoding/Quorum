"""Deterministic fixed-weight combination of standardized expert evidence.

This module groups already-created predictions by scientific decision identity.
It does not fetch data, fit weights, reinterpret missing experts, size positions,
or execute trades.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral, Real
from typing import Any

from src.quorum.contracts import (
    ExpertPrediction,
    ExpertResult,
    StaticEnsembleConfig,
)

_SCHEMA_VERSION = 1
_SCORE_BOUNDARY_TOLERANCE = 1e-12


def _instant(value: datetime) -> datetime:
    """Return a transient absolute-instant key without changing stored offsets."""
    return value.astimezone(timezone.utc)


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


def _finite_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _parse_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 datetime") from exc
    return _aware_datetime(name, result)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_mapping(text: str, contract_name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{contract_name} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{contract_name} JSON must contain an object")
    return value


def _require_payload(
    data: object, *, contract_name: str, fields: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{contract_name} payload must be a mapping")
    expected = fields | {"contract", "schema_version"}
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(repr(field) for field in actual - expected)
        raise ValueError(
            f"invalid {contract_name} payload: missing={missing}, unknown={unknown}"
        )
    if data["contract"] != contract_name:
        raise ValueError(f"expected contract={contract_name!r}")
    version = data["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {contract_name} schema_version {version!r}; "
            f"expected integer {_SCHEMA_VERSION}"
        )
    return data


def _payload(contract_name: str, **fields: Any) -> dict[str, Any]:
    return {
        "contract": contract_name,
        "schema_version": _SCHEMA_VERSION,
        **fields,
    }


def _bounded_combined_score(raw_score: float) -> float:
    """Snap only contract-tolerance boundary noise; reject material violations."""
    if not math.isfinite(raw_score):
        raise ValueError("combined score must be finite")
    if raw_score > 1.0:
        if raw_score - 1.0 <= _SCORE_BOUNDARY_TOLERANCE:
            return 1.0
        raise ValueError("combined score materially exceeds +1")
    if raw_score < -1.0:
        if -1.0 - raw_score <= _SCORE_BOUNDARY_TOLERANCE:
            return -1.0
        raise ValueError("combined score materially exceeds -1")
    return raw_score


class ReportingLabel(str, Enum):
    """Reporting-only description of combined evidence, never an order."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


def _reporting_label(score: float, config: StaticEnsembleConfig) -> ReportingLabel:
    if score <= config.sell_threshold:
        return ReportingLabel.SELL
    if score >= config.buy_threshold:
        return ReportingLabel.BUY
    return ReportingLabel.HOLD


@dataclass(frozen=True, slots=True)
class ExpertAttribution:
    """One configured expert's present or missing contribution to a row."""

    expert_id: str
    expert_version: str
    configured_weight: float
    present: bool
    expert_score: float | None
    weighted_contribution: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expert_id", _required_text("expert_id", self.expert_id)
        )
        object.__setattr__(
            self,
            "expert_version",
            _required_text("expert_version", self.expert_version),
        )
        weight = _finite_float(
            "configured_weight",
            self.configured_weight,
            minimum=0.0,
            maximum=1.0,
        )
        if weight <= 0.0:
            raise ValueError("configured_weight must be strictly positive")
        object.__setattr__(self, "configured_weight", weight)
        if type(self.present) is not bool:
            raise TypeError("present must be a boolean")

        if self.present:
            if self.expert_score is None or self.weighted_contribution is None:
                raise ValueError("present attribution requires score and contribution")
            score = _finite_float(
                "expert_score", self.expert_score, minimum=-1.0, maximum=1.0
            )
            contribution = _finite_float(
                "weighted_contribution",
                self.weighted_contribution,
                minimum=-1.0,
                maximum=1.0,
            )
            if contribution != weight * score:
                raise ValueError(
                    "weighted_contribution must equal configured_weight * expert_score"
                )
            object.__setattr__(self, "expert_score", score)
            object.__setattr__(self, "weighted_contribution", contribution)
        elif self.expert_score is not None or self.weighted_contribution is not None:
            raise ValueError("missing attribution cannot contain score or contribution")

    def to_dict(self) -> dict[str, Any]:
        return _payload(
            "expert_attribution",
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            configured_weight=self.configured_weight,
            present=self.present,
            expert_score=self.expert_score,
            weighted_contribution=self.weighted_contribution,
        )

    @classmethod
    def from_dict(cls, data: object) -> "ExpertAttribution":
        payload = _require_payload(
            data,
            contract_name="expert_attribution",
            fields=frozenset(
                {
                    "expert_id",
                    "expert_version",
                    "configured_weight",
                    "present",
                    "expert_score",
                    "weighted_contribution",
                }
            ),
        )
        return cls(
            expert_id=payload["expert_id"],
            expert_version=payload["expert_version"],
            configured_weight=payload["configured_weight"],
            present=payload["present"],
            expert_score=payload["expert_score"],
            weighted_contribution=payload["weighted_contribution"],
        )

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExpertAttribution":
        return cls.from_dict(_json_mapping(text, "expert_attribution"))


def _complete_decision_metrics(
    attributions: tuple[ExpertAttribution, ...],
) -> tuple[float, float]:
    """Recompute the only valid score and disagreement for complete evidence."""
    if not all(
        item.present
        and item.expert_score is not None
        and item.weighted_contribution is not None
        for item in attributions
    ):
        raise ValueError("complete decision metrics require complete attribution")

    combined_score = _bounded_combined_score(
        math.fsum(
            item.weighted_contribution
            for item in attributions
            if item.weighted_contribution is not None
        )
    )
    disagreement = math.sqrt(
        math.fsum(
            item.configured_weight * (item.expert_score - combined_score) ** 2
            for item in attributions
            if item.expert_score is not None
        )
    )
    return combined_score, disagreement


def _optional_key(value: str | None) -> tuple[bool, str]:
    return (value is not None, value or "")


@dataclass(frozen=True, slots=True, eq=False)
class StaticEnsembleDecision:
    """One aligned ensemble evidence row with complete attribution."""

    asset: str
    event_at: datetime
    available_at: datetime
    decision_at: datetime
    horizon_bars: int
    experiment_id: str | None
    split_id: str | None
    attributions: tuple[ExpertAttribution, ...]
    combined_score: float | None
    label: ReportingLabel | None
    disagreement: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _required_text("asset", self.asset))
        event_at = _aware_datetime("event_at", self.event_at)
        available_at = _aware_datetime("available_at", self.available_at)
        decision_at = _aware_datetime("decision_at", self.decision_at)
        if _instant(available_at) > _instant(decision_at):
            raise ValueError("available_at must not be later than decision_at")
        object.__setattr__(self, "event_at", event_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(
            self, "horizon_bars", _positive_integer("horizon_bars", self.horizon_bars)
        )
        experiment_id = _optional_text("experiment_id", self.experiment_id)
        split_id = _optional_text("split_id", self.split_id)
        if split_id is not None and experiment_id is None:
            raise ValueError("split_id requires experiment_id")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "split_id", split_id)

        attributions = tuple(self.attributions)
        if not attributions:
            raise ValueError("attributions must not be empty")
        if not all(isinstance(item, ExpertAttribution) for item in attributions):
            raise TypeError("attributions must contain only ExpertAttribution values")
        ids = [item.expert_id for item in attributions]
        if len(ids) != len(set(ids)):
            raise ValueError("attributions contain duplicate expert_id values")
        attributions = tuple(
            sorted(attributions, key=lambda item: (item.expert_id, item.expert_version))
        )
        if not any(item.present for item in attributions):
            raise ValueError(
                "an ensemble decision requires at least one present expert"
            )
        object.__setattr__(self, "attributions", attributions)

        complete = all(item.present for item in attributions)
        if complete:
            if (
                self.combined_score is None
                or self.label is None
                or self.disagreement is None
            ):
                raise ValueError(
                    "complete decision requires score, label, and disagreement"
                )
            combined_score = _finite_float(
                "combined_score",
                self.combined_score,
                minimum=-1.0,
                maximum=1.0,
            )
            if not isinstance(self.label, ReportingLabel):
                raise TypeError("label must be a ReportingLabel")
            disagreement = _finite_float(
                "disagreement",
                self.disagreement,
                minimum=0.0,
                maximum=math.inf,
            )
            expected_score, expected_disagreement = _complete_decision_metrics(
                attributions
            )
            if combined_score != expected_score:
                raise ValueError("combined_score does not match attribution")
            if disagreement != expected_disagreement:
                raise ValueError("disagreement does not match weighted dispersion")
            object.__setattr__(self, "combined_score", combined_score)
            object.__setattr__(self, "disagreement", disagreement)
        elif any(
            value is not None
            for value in (self.combined_score, self.label, self.disagreement)
        ):
            raise ValueError(
                "incomplete decision cannot contain score, label, or disagreement"
            )

    @property
    def alignment_key(
        self,
    ) -> tuple[str, datetime, datetime, int, str | None, str | None]:
        return (
            self.asset,
            _instant(self.event_at),
            _instant(self.decision_at),
            self.horizon_bars,
            self.experiment_id,
            self.split_id,
        )

    def _equality_key(self) -> tuple[Any, ...]:
        return (
            self.asset,
            _instant(self.event_at),
            _instant(self.available_at),
            _instant(self.decision_at),
            self.horizon_bars,
            self.experiment_id,
            self.split_id,
            self.attributions,
            self.combined_score,
            self.label,
            self.disagreement,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StaticEnsembleDecision):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def __hash__(self) -> int:
        return hash(self._equality_key())

    def to_dict(self) -> dict[str, Any]:
        return _payload(
            "static_ensemble_decision",
            asset=self.asset,
            event_at=self.event_at.isoformat(),
            available_at=self.available_at.isoformat(),
            decision_at=self.decision_at.isoformat(),
            horizon_bars=self.horizon_bars,
            experiment_id=self.experiment_id,
            split_id=self.split_id,
            attributions=[item.to_dict() for item in self.attributions],
            combined_score=self.combined_score,
            label=self.label.value if self.label is not None else None,
            disagreement=self.disagreement,
        )

    def to_json(self) -> str:
        """Serialize this embedded row; reconstruct through StaticEnsembleResult."""
        return _canonical_json(self.to_dict())


def _decision_from_dict(data: object) -> StaticEnsembleDecision:
    """Parse an embedded row whose label is later validated by its result config."""
    payload = _require_payload(
        data,
        contract_name="static_ensemble_decision",
        fields=frozenset(
            {
                "asset",
                "event_at",
                "available_at",
                "decision_at",
                "horizon_bars",
                "experiment_id",
                "split_id",
                "attributions",
                "combined_score",
                "label",
                "disagreement",
            }
        ),
    )
    attribution_data = payload["attributions"]
    if not isinstance(attribution_data, list):
        raise TypeError("attributions must be a list")
    label_data = payload["label"]
    if label_data is not None and not isinstance(label_data, str):
        raise TypeError("label must be a string or null")
    try:
        label = ReportingLabel(label_data) if label_data is not None else None
    except ValueError as exc:
        raise ValueError(f"unknown reporting label {label_data!r}") from exc
    return StaticEnsembleDecision(
        asset=payload["asset"],
        event_at=_parse_datetime("event_at", payload["event_at"]),
        available_at=_parse_datetime("available_at", payload["available_at"]),
        decision_at=_parse_datetime("decision_at", payload["decision_at"]),
        horizon_bars=payload["horizon_bars"],
        experiment_id=payload["experiment_id"],
        split_id=payload["split_id"],
        attributions=tuple(
            ExpertAttribution.from_dict(item) for item in attribution_data
        ),
        combined_score=payload["combined_score"],
        label=label,
        disagreement=payload["disagreement"],
    )


def _decision_sort_key(decision: StaticEnsembleDecision) -> tuple[Any, ...]:
    return (
        decision.asset,
        _instant(decision.event_at),
        _instant(decision.decision_at),
        decision.horizon_bars,
        _optional_key(decision.experiment_id),
        _optional_key(decision.split_id),
    )


def _validate_decision_against_config(
    decision: StaticEnsembleDecision, config: StaticEnsembleConfig
) -> None:
    expected_specs = tuple(
        (item.expert_id, item.expert_version, item.weight) for item in config.experts
    )
    actual_specs = tuple(
        (item.expert_id, item.expert_version, item.configured_weight)
        for item in decision.attributions
    )
    if actual_specs != expected_specs:
        raise ValueError("decision attributions do not match ensemble config")

    if all(item.present for item in decision.attributions):
        expected_score, expected_disagreement = _complete_decision_metrics(
            decision.attributions
        )
        if decision.combined_score != expected_score:
            raise ValueError("decision combined_score does not match attribution")
        if decision.disagreement != expected_disagreement:
            raise ValueError("decision disagreement does not match weighted dispersion")
        if decision.label is not _reporting_label(expected_score, config):
            raise ValueError("decision label does not match configured thresholds")


@dataclass(frozen=True, slots=True)
class StaticEnsembleResult:
    """Authoritative serialized decisions plus their validating configuration."""

    config: StaticEnsembleConfig
    decisions: tuple[StaticEnsembleDecision, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, StaticEnsembleConfig):
            raise TypeError("config must be a StaticEnsembleConfig")
        decisions = tuple(self.decisions)
        if not all(isinstance(item, StaticEnsembleDecision) for item in decisions):
            raise TypeError("decisions must contain only StaticEnsembleDecision values")
        keys = [item.alignment_key for item in decisions]
        if len(keys) != len(set(keys)):
            raise ValueError("decisions contain duplicate alignment keys")

        for decision in decisions:
            _validate_decision_against_config(decision, self.config)

        object.__setattr__(
            self, "decisions", tuple(sorted(decisions, key=_decision_sort_key))
        )

    def to_dict(self) -> dict[str, Any]:
        return _payload(
            "static_ensemble_result",
            config=self.config.to_dict(),
            decisions=[decision.to_dict() for decision in self.decisions],
        )

    @classmethod
    def from_dict(cls, data: object) -> "StaticEnsembleResult":
        payload = _require_payload(
            data,
            contract_name="static_ensemble_result",
            fields=frozenset({"config", "decisions"}),
        )
        config_data = payload["config"]
        if not isinstance(config_data, Mapping):
            raise TypeError("config must be a mapping")
        decision_data = payload["decisions"]
        if not isinstance(decision_data, list):
            raise TypeError("decisions must be a list")
        config = StaticEnsembleConfig.from_dict(config_data)
        return cls(
            config=config,
            decisions=tuple(_decision_from_dict(item) for item in decision_data),
        )

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StaticEnsembleResult":
        return cls.from_dict(_json_mapping(text, "static_ensemble_result"))


def _prediction_alignment_key(
    prediction: ExpertPrediction,
) -> tuple[str, datetime, datetime, int, str | None, str | None]:
    return (
        prediction.asset,
        _instant(prediction.event_at),
        _instant(prediction.decision_at),
        prediction.horizon_bars,
        prediction.experiment_id,
        prediction.split_id,
    )


@dataclass(frozen=True, slots=True)
class StaticEnsemble:
    """Pure fixed-weight ensemble over aligned standardized predictions."""

    config: StaticEnsembleConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, StaticEnsembleConfig):
            raise TypeError("config must be a StaticEnsembleConfig")

    def combine(self, expert_result: ExpertResult) -> StaticEnsembleResult:
        """Combine each alignment group without filling or reweighting missing experts."""
        if not isinstance(expert_result, ExpertResult):
            raise TypeError("expert_result must be an ExpertResult")

        configured = {item.expert_id: item for item in self.config.experts}
        groups: dict[
            tuple[str, datetime, datetime, int, str | None, str | None],
            dict[str, ExpertPrediction],
        ] = {}
        for prediction in expert_result.predictions:
            specification = configured.get(prediction.expert_id)
            if specification is None:
                raise ValueError(
                    f"unexpected expert_id {prediction.expert_id!r} for ensemble"
                )
            if prediction.expert_version != specification.expert_version:
                raise ValueError(
                    f"expert_version mismatch for {prediction.expert_id!r}: "
                    f"expected {specification.expert_version!r}, got "
                    f"{prediction.expert_version!r}"
                )
            key = _prediction_alignment_key(prediction)
            group = groups.setdefault(key, {})
            if prediction.expert_id in group:
                raise ValueError(
                    f"duplicate prediction for expert {prediction.expert_id!r} "
                    "in one alignment group"
                )
            group[prediction.expert_id] = prediction

        decisions: list[StaticEnsembleDecision] = []
        for key in sorted(
            groups,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                _optional_key(item[4]),
                _optional_key(item[5]),
            ),
        ):
            group = groups[key]
            ordered_predictions = tuple(
                group[item.expert_id]
                for item in self.config.experts
                if item.expert_id in group
            )
            reference = ordered_predictions[0]
            latest_availability = max(
                _instant(prediction.available_at) for prediction in ordered_predictions
            )
            available_at = next(
                prediction.available_at
                for prediction in ordered_predictions
                if _instant(prediction.available_at) == latest_availability
            )

            attributions = tuple(
                ExpertAttribution(
                    expert_id=specification.expert_id,
                    expert_version=specification.expert_version,
                    configured_weight=specification.weight,
                    present=specification.expert_id in group,
                    expert_score=(
                        group[specification.expert_id].score
                        if specification.expert_id in group
                        else None
                    ),
                    weighted_contribution=(
                        specification.weight * group[specification.expert_id].score
                        if specification.expert_id in group
                        else None
                    ),
                )
                for specification in self.config.experts
            )
            complete = len(group) == len(self.config.experts)
            combined_score: float | None = None
            label: ReportingLabel | None = None
            disagreement: float | None = None
            if complete:
                combined_score, disagreement = _complete_decision_metrics(attributions)
                label = _reporting_label(combined_score, self.config)

            decisions.append(
                StaticEnsembleDecision(
                    asset=key[0],
                    event_at=reference.event_at,
                    available_at=available_at,
                    decision_at=reference.decision_at,
                    horizon_bars=key[3],
                    experiment_id=key[4],
                    split_id=key[5],
                    attributions=attributions,
                    combined_score=combined_score,
                    label=label,
                    disagreement=disagreement,
                )
            )

        return StaticEnsembleResult(config=self.config, decisions=tuple(decisions))


__all__ = [
    "ExpertAttribution",
    "ReportingLabel",
    "StaticEnsemble",
    "StaticEnsembleDecision",
    "StaticEnsembleResult",
]
