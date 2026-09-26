"""Pure deterministic conversion of ensemble evidence into requested exposure.

The policy owns desired portfolio weights only. It does not inspect market data,
place orders, simulate fills, or infer what an execution engine can realize.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.quorum.ensemble import StaticEnsembleDecision, StaticEnsembleResult
from src.quorum.risk._codec import (
    aware_datetime,
    canonical_json,
    finite_float,
    json_mapping,
    optional_text,
    parse_datetime,
    payload,
    positive_integer,
    required_text,
    require_payload,
)

_INVARIANT_TOLERANCE = 1e-12


def _instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _optional_weight(name: str, value: object) -> float | None:
    if value is None:
        return None
    return finite_float(name, value, minimum=-1.0, maximum=1.0)


def _optional_metric(name: str, value: object, *, maximum: float) -> float | None:
    if value is None:
        return None
    return finite_float(name, value, minimum=0.0, maximum=maximum)


def _optional_key(value: str | None) -> tuple[bool, str]:
    return (value is not None, value or "")


@dataclass(frozen=True, slots=True)
class RiskPolicyConfig:
    """Caller-frozen V0 exposure limits; no parameter is selected here."""

    max_gross_exposure: float
    max_abs_weight_per_asset: float
    max_turnover: float

    def __post_init__(self) -> None:
        max_gross = finite_float(
            "max_gross_exposure",
            self.max_gross_exposure,
            minimum=0.0,
            maximum=1.0,
        )
        if max_gross <= 0.0:
            raise ValueError("max_gross_exposure must be strictly positive")
        max_name = finite_float(
            "max_abs_weight_per_asset",
            self.max_abs_weight_per_asset,
            minimum=0.0,
            maximum=1.0,
        )
        if max_name <= 0.0:
            raise ValueError("max_abs_weight_per_asset must be strictly positive")
        max_turnover = finite_float(
            "max_turnover", self.max_turnover, minimum=0.0, maximum=2.0
        )
        object.__setattr__(self, "max_gross_exposure", max_gross)
        object.__setattr__(self, "max_abs_weight_per_asset", max_name)
        object.__setattr__(self, "max_turnover", max_turnover)

    def to_dict(self) -> dict[str, Any]:
        return payload(
            "risk_policy_config",
            max_gross_exposure=self.max_gross_exposure,
            max_abs_weight_per_asset=self.max_abs_weight_per_asset,
            max_turnover=self.max_turnover,
        )

    @classmethod
    def from_dict(cls, data: object) -> RiskPolicyConfig:
        data = require_payload(
            data,
            contract_name="risk_policy_config",
            fields=frozenset(
                {
                    "max_gross_exposure",
                    "max_abs_weight_per_asset",
                    "max_turnover",
                }
            ),
        )
        return cls(
            max_gross_exposure=data["max_gross_exposure"],
            max_abs_weight_per_asset=data["max_abs_weight_per_asset"],
            max_turnover=data["max_turnover"],
        )

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> RiskPolicyConfig:
        return cls.from_dict(json_mapping(text, "risk_policy_config"))


@dataclass(frozen=True, slots=True, eq=False)
class RiskTarget:
    """One asset's auditable desired-exposure path, never a fill or position."""

    asset: str
    event_at: datetime
    available_at: datetime
    decision_at: datetime
    horizon_bars: int
    experiment_id: str | None
    split_id: str | None
    ensemble_score: float | None
    proposed_target_weight: float | None
    gross_constrained_target_weight: float | None
    final_target_weight: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", required_text("asset", self.asset))
        event_at = aware_datetime("event_at", self.event_at)
        available_at = aware_datetime("available_at", self.available_at)
        decision_at = aware_datetime("decision_at", self.decision_at)
        if _instant(available_at) > _instant(decision_at):
            raise ValueError("available_at must not be later than decision_at")
        object.__setattr__(self, "event_at", event_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(
            self, "horizon_bars", positive_integer("horizon_bars", self.horizon_bars)
        )
        experiment_id = optional_text("experiment_id", self.experiment_id)
        split_id = optional_text("split_id", self.split_id)
        if split_id is not None and experiment_id is None:
            raise ValueError("split_id requires experiment_id")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "split_id", split_id)

        score = _optional_weight("ensemble_score", self.ensemble_score)
        proposed = _optional_weight(
            "proposed_target_weight", self.proposed_target_weight
        )
        gross_constrained = _optional_weight(
            "gross_constrained_target_weight",
            self.gross_constrained_target_weight,
        )
        final = _optional_weight("final_target_weight", self.final_target_weight)
        weights = (proposed, gross_constrained, final)
        if score is None:
            if any(value is not None for value in weights):
                raise ValueError(
                    "missing ensemble evidence cannot contain target weights"
                )
        elif any(value is None for value in weights) and not all(
            value is None for value in weights
        ):
            raise ValueError("target weight stages must be all present or all absent")
        object.__setattr__(self, "ensemble_score", score)
        object.__setattr__(self, "proposed_target_weight", proposed)
        object.__setattr__(self, "gross_constrained_target_weight", gross_constrained)
        object.__setattr__(self, "final_target_weight", final)

    @property
    def stream_key(self) -> tuple[str | None, str | None, int]:
        return (self.experiment_id, self.split_id, self.horizon_bars)

    def _equality_key(self) -> tuple[Any, ...]:
        return (
            self.asset,
            _instant(self.event_at),
            _instant(self.available_at),
            _instant(self.decision_at),
            self.horizon_bars,
            self.experiment_id,
            self.split_id,
            self.ensemble_score,
            self.proposed_target_weight,
            self.gross_constrained_target_weight,
            self.final_target_weight,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiskTarget):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def __hash__(self) -> int:
        return hash(self._equality_key())

    def to_dict(self) -> dict[str, Any]:
        return payload(
            "risk_target",
            asset=self.asset,
            event_at=self.event_at.isoformat(),
            available_at=self.available_at.isoformat(),
            decision_at=self.decision_at.isoformat(),
            horizon_bars=self.horizon_bars,
            experiment_id=self.experiment_id,
            split_id=self.split_id,
            ensemble_score=self.ensemble_score,
            proposed_target_weight=self.proposed_target_weight,
            gross_constrained_target_weight=self.gross_constrained_target_weight,
            final_target_weight=self.final_target_weight,
        )


@dataclass(frozen=True, slots=True, eq=False)
class RiskRebalance:
    """One simultaneous portfolio request within an isolated risk stream."""

    experiment_id: str | None
    split_id: str | None
    horizon_bars: int
    decision_at: datetime
    targets: tuple[RiskTarget, ...]
    executable: bool
    proposed_gross: float | None
    gross_scale: float | None
    requested_turnover: float | None
    turnover_scale: float | None
    final_gross: float | None

    def __post_init__(self) -> None:
        experiment_id = optional_text("experiment_id", self.experiment_id)
        split_id = optional_text("split_id", self.split_id)
        if split_id is not None and experiment_id is None:
            raise ValueError("split_id requires experiment_id")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "split_id", split_id)
        horizon = positive_integer("horizon_bars", self.horizon_bars)
        decision_at = aware_datetime("decision_at", self.decision_at)
        object.__setattr__(self, "horizon_bars", horizon)
        object.__setattr__(self, "decision_at", decision_at)
        if type(self.executable) is not bool:
            raise TypeError("executable must be a boolean")

        targets = tuple(self.targets)
        if not targets:
            raise ValueError("risk rebalance targets must not be empty")
        if not all(isinstance(item, RiskTarget) for item in targets):
            raise TypeError("targets must contain only RiskTarget values")
        targets = tuple(sorted(targets, key=lambda item: item.asset))
        assets = tuple(item.asset for item in targets)
        if len(assets) != len(set(assets)):
            raise ValueError("risk rebalance contains duplicate assets")
        expected_stream = (experiment_id, split_id, horizon)
        for target in targets:
            if target.stream_key != expected_stream:
                raise ValueError("risk target does not match rebalance stream")
            if _instant(target.decision_at) != _instant(decision_at):
                raise ValueError(
                    "risk target does not match rebalance decision instant"
                )
        object.__setattr__(self, "targets", targets)

        metrics = (
            _optional_metric("proposed_gross", self.proposed_gross, maximum=math.inf),
            _optional_metric("gross_scale", self.gross_scale, maximum=1.0),
            _optional_metric(
                "requested_turnover", self.requested_turnover, maximum=2.0
            ),
            _optional_metric("turnover_scale", self.turnover_scale, maximum=1.0),
            _optional_metric("final_gross", self.final_gross, maximum=1.0),
        )
        if self.executable:
            if any(value is None for value in metrics):
                raise ValueError("executable rebalance requires complete audit metrics")
            if any(item.final_target_weight is None for item in targets):
                raise ValueError(
                    "executable rebalance requires complete target weights"
                )
        else:
            if any(value is not None for value in metrics):
                raise ValueError("incomplete rebalance cannot contain audit metrics")
            if any(
                value is not None
                for item in targets
                for value in (
                    item.proposed_target_weight,
                    item.gross_constrained_target_weight,
                    item.final_target_weight,
                )
            ):
                raise ValueError("incomplete rebalance cannot contain target weights")
        (
            proposed_gross,
            gross_scale,
            requested_turnover,
            turnover_scale,
            final_gross,
        ) = metrics
        object.__setattr__(self, "proposed_gross", proposed_gross)
        object.__setattr__(self, "gross_scale", gross_scale)
        object.__setattr__(self, "requested_turnover", requested_turnover)
        object.__setattr__(self, "turnover_scale", turnover_scale)
        object.__setattr__(self, "final_gross", final_gross)

    @property
    def stream_key(self) -> tuple[str | None, str | None, int]:
        return (self.experiment_id, self.split_id, self.horizon_bars)

    def _equality_key(self) -> tuple[Any, ...]:
        return (
            self.stream_key,
            _instant(self.decision_at),
            self.targets,
            self.executable,
            self.proposed_gross,
            self.gross_scale,
            self.requested_turnover,
            self.turnover_scale,
            self.final_gross,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiskRebalance):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def __hash__(self) -> int:
        return hash(self._equality_key())

    def to_dict(self) -> dict[str, Any]:
        return payload(
            "risk_rebalance",
            experiment_id=self.experiment_id,
            split_id=self.split_id,
            horizon_bars=self.horizon_bars,
            decision_at=self.decision_at.isoformat(),
            targets=[item.to_dict() for item in self.targets],
            executable=self.executable,
            proposed_gross=self.proposed_gross,
            gross_scale=self.gross_scale,
            requested_turnover=self.requested_turnover,
            turnover_scale=self.turnover_scale,
            final_gross=self.final_gross,
        )


@dataclass(frozen=True, slots=True)
class _PortfolioCalculation:
    executable: bool
    weights: tuple[tuple[str, float | None, float | None, float | None], ...]
    proposed_gross: float | None
    gross_scale: float | None
    requested_turnover: float | None
    turnover_scale: float | None
    final_gross: float | None


def _calculate_portfolio(
    scores: tuple[tuple[str, float | None], ...],
    previous: Mapping[str, float],
    config: RiskPolicyConfig,
) -> _PortfolioCalculation:
    if any(score is None for _, score in scores):
        return _PortfolioCalculation(
            executable=False,
            weights=tuple((asset, None, None, None) for asset, _ in scores),
            proposed_gross=None,
            gross_scale=None,
            requested_turnover=None,
            turnover_scale=None,
            final_gross=None,
        )

    proposed = tuple(
        (asset, score * config.max_abs_weight_per_asset)
        for asset, score in scores
        if score is not None
    )
    proposed_gross = math.fsum(abs(weight) for _, weight in proposed)
    gross_scale = (
        config.max_gross_exposure / proposed_gross
        if proposed_gross > config.max_gross_exposure
        else 1.0
    )
    # Dividing by the gross sum can overshoot the cap by an ulp; step the scale
    # down until the exact sum complies, so records never carry leverage.
    while gross_scale < 1.0 and (
        math.fsum(abs(weight * gross_scale) for _, weight in proposed)
        > config.max_gross_exposure
    ):
        gross_scale = math.nextafter(gross_scale, 0.0)
    gross_constrained = tuple(
        (asset, weight * gross_scale) for asset, weight in proposed
    )
    requested_turnover = math.fsum(
        abs(weight - previous[asset]) for asset, weight in gross_constrained
    )
    turnover_scale = (
        config.max_turnover / requested_turnover
        if requested_turnover > config.max_turnover
        else 1.0
    )
    final = tuple(
        (
            asset,
            previous[asset] + turnover_scale * (weight - previous[asset]),
        )
        for asset, weight in gross_constrained
    )
    # Interpolating between two compliant portfolios can also round an ulp
    # over; shrink only such rounding-level excess (larger excess still fails).
    shrink = 1.0
    while (
        config.max_gross_exposure
        < math.fsum(abs(weight * shrink) for _, weight in final)
        <= config.max_gross_exposure + _INVARIANT_TOLERANCE
    ):
        shrink = math.nextafter(shrink, 0.0)
    if shrink != 1.0:
        final = tuple((asset, weight * shrink) for asset, weight in final)
    final_gross = math.fsum(abs(weight) for _, weight in final)
    realized_turnover = math.fsum(
        abs(weight - previous[asset]) for asset, weight in final
    )

    if any(
        abs(weight) > config.max_abs_weight_per_asset + _INVARIANT_TOLERANCE
        for _, weight in final
    ):
        raise ValueError("final target violates the per-asset exposure cap")
    if final_gross > config.max_gross_exposure + _INVARIANT_TOLERANCE:
        raise ValueError("final portfolio violates the gross exposure cap")
    if realized_turnover > config.max_turnover + _INVARIANT_TOLERANCE:
        raise ValueError("final portfolio violates the turnover cap")

    proposed_by_asset = dict(proposed)
    gross_by_asset = dict(gross_constrained)
    final_by_asset = dict(final)
    return _PortfolioCalculation(
        executable=True,
        weights=tuple(
            (
                asset,
                proposed_by_asset[asset],
                gross_by_asset[asset],
                final_by_asset[asset],
            )
            for asset, _ in scores
        ),
        proposed_gross=proposed_gross,
        gross_scale=gross_scale,
        requested_turnover=requested_turnover,
        turnover_scale=turnover_scale,
        final_gross=final_gross,
    )


def _rebalance_sort_key(rebalance: RiskRebalance) -> tuple[Any, ...]:
    return (
        _optional_key(rebalance.experiment_id),
        _optional_key(rebalance.split_id),
        rebalance.horizon_bars,
        _instant(rebalance.decision_at),
    )


def _validate_rebalances(
    rebalances: tuple[RiskRebalance, ...], config: RiskPolicyConfig
) -> None:
    group_keys = tuple(
        (item.stream_key, _instant(item.decision_at)) for item in rebalances
    )
    if len(group_keys) != len(set(group_keys)):
        raise ValueError("risk result contains duplicate portfolio decision groups")

    universes: dict[tuple[str | None, str | None, int], tuple[str, ...]] = {}
    previous_by_stream: dict[tuple[str | None, str | None, int], dict[str, float]] = {}
    for rebalance in rebalances:
        stream = rebalance.stream_key
        assets = tuple(item.asset for item in rebalance.targets)
        expected_universe = universes.setdefault(stream, assets)
        if assets != expected_universe:
            raise ValueError("asset universe changed within a risk stream")
        previous = previous_by_stream.setdefault(
            stream, {asset: 0.0 for asset in expected_universe}
        )
        calculation = _calculate_portfolio(
            tuple((item.asset, item.ensemble_score) for item in rebalance.targets),
            previous,
            config,
        )
        if rebalance.executable != calculation.executable:
            raise ValueError("rebalance executable state does not match evidence")
        expected_metrics = (
            calculation.proposed_gross,
            calculation.gross_scale,
            calculation.requested_turnover,
            calculation.turnover_scale,
            calculation.final_gross,
        )
        actual_metrics = (
            rebalance.proposed_gross,
            rebalance.gross_scale,
            rebalance.requested_turnover,
            rebalance.turnover_scale,
            rebalance.final_gross,
        )
        if actual_metrics != expected_metrics:
            raise ValueError("rebalance audit metrics do not match risk policy")
        expected_weights = {
            asset: (proposed, gross, final)
            for asset, proposed, gross, final in calculation.weights
        }
        for target in rebalance.targets:
            actual = (
                target.proposed_target_weight,
                target.gross_constrained_target_weight,
                target.final_target_weight,
            )
            if actual != expected_weights[target.asset]:
                raise ValueError("risk target weights do not match risk policy")
        if calculation.executable:
            previous_by_stream[stream] = {
                item.asset: item.final_target_weight
                for item in rebalance.targets
                if item.final_target_weight is not None
            }


@dataclass(frozen=True, slots=True)
class RiskPolicyResult:
    """Authoritative serialized risk artifact with config and audited rebalances."""

    config: RiskPolicyConfig
    rebalances: tuple[RiskRebalance, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, RiskPolicyConfig):
            raise TypeError("config must be a RiskPolicyConfig")
        rebalances = tuple(self.rebalances)
        if not all(isinstance(item, RiskRebalance) for item in rebalances):
            raise TypeError("rebalances must contain only RiskRebalance values")
        rebalances = tuple(sorted(rebalances, key=_rebalance_sort_key))
        _validate_rebalances(rebalances, self.config)
        object.__setattr__(self, "rebalances", rebalances)

    @property
    def targets(self) -> tuple[RiskTarget, ...]:
        return tuple(
            target for rebalance in self.rebalances for target in rebalance.targets
        )

    def to_dict(self) -> dict[str, Any]:
        return payload(
            "risk_policy_result",
            config=self.config.to_dict(),
            rebalances=[item.to_dict() for item in self.rebalances],
        )

    @classmethod
    def from_dict(cls, data: object) -> RiskPolicyResult:
        data = require_payload(
            data,
            contract_name="risk_policy_result",
            fields=frozenset({"config", "rebalances"}),
        )
        config_data = data["config"]
        if not isinstance(config_data, Mapping):
            raise TypeError("config must be a mapping")
        rebalance_data = data["rebalances"]
        if not isinstance(rebalance_data, list):
            raise TypeError("rebalances must be a list")
        config = RiskPolicyConfig.from_dict(config_data)
        return cls(
            config=config,
            rebalances=tuple(_rebalance_from_dict(item) for item in rebalance_data),
        )

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> RiskPolicyResult:
        return cls.from_dict(json_mapping(text, "risk_policy_result"))


def _target_from_dict(data: object) -> RiskTarget:
    data = require_payload(
        data,
        contract_name="risk_target",
        fields=frozenset(
            {
                "asset",
                "event_at",
                "available_at",
                "decision_at",
                "horizon_bars",
                "experiment_id",
                "split_id",
                "ensemble_score",
                "proposed_target_weight",
                "gross_constrained_target_weight",
                "final_target_weight",
            }
        ),
    )
    return RiskTarget(
        asset=data["asset"],
        event_at=parse_datetime("event_at", data["event_at"]),
        available_at=parse_datetime("available_at", data["available_at"]),
        decision_at=parse_datetime("decision_at", data["decision_at"]),
        horizon_bars=data["horizon_bars"],
        experiment_id=data["experiment_id"],
        split_id=data["split_id"],
        ensemble_score=data["ensemble_score"],
        proposed_target_weight=data["proposed_target_weight"],
        gross_constrained_target_weight=data["gross_constrained_target_weight"],
        final_target_weight=data["final_target_weight"],
    )


def _rebalance_from_dict(data: object) -> RiskRebalance:
    data = require_payload(
        data,
        contract_name="risk_rebalance",
        fields=frozenset(
            {
                "experiment_id",
                "split_id",
                "horizon_bars",
                "decision_at",
                "targets",
                "executable",
                "proposed_gross",
                "gross_scale",
                "requested_turnover",
                "turnover_scale",
                "final_gross",
            }
        ),
    )
    target_data = data["targets"]
    if not isinstance(target_data, list):
        raise TypeError("targets must be a list")
    return RiskRebalance(
        experiment_id=data["experiment_id"],
        split_id=data["split_id"],
        horizon_bars=data["horizon_bars"],
        decision_at=parse_datetime("decision_at", data["decision_at"]),
        targets=tuple(_target_from_dict(item) for item in target_data),
        executable=data["executable"],
        proposed_gross=data["proposed_gross"],
        gross_scale=data["gross_scale"],
        requested_turnover=data["requested_turnover"],
        turnover_scale=data["turnover_scale"],
        final_gross=data["final_gross"],
    )


def _decision_group_key(
    decision: StaticEnsembleDecision,
) -> tuple[str | None, str | None, int, datetime]:
    return (
        decision.experiment_id,
        decision.split_id,
        decision.horizon_bars,
        _instant(decision.decision_at),
    )


@dataclass(frozen=True, slots=True)
class FixedRiskPolicy:
    """Apply the frozen V0 name, gross, and turnover rules to ensemble rows."""

    config: RiskPolicyConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, RiskPolicyConfig):
            raise TypeError("config must be a RiskPolicyConfig")

    def apply(self, ensemble_result: StaticEnsembleResult) -> RiskPolicyResult:
        if not isinstance(ensemble_result, StaticEnsembleResult):
            raise TypeError("ensemble_result must be a StaticEnsembleResult")

        groups: dict[
            tuple[str | None, str | None, int, datetime],
            dict[str, StaticEnsembleDecision],
        ] = {}
        for decision in ensemble_result.decisions:
            key = _decision_group_key(decision)
            group = groups.setdefault(key, {})
            if decision.asset in group:
                raise ValueError(
                    "duplicate ensemble decision for one asset and portfolio cutoff"
                )
            group[decision.asset] = decision

        universes: dict[tuple[str | None, str | None, int], tuple[str, ...]] = {}
        previous_by_stream: dict[
            tuple[str | None, str | None, int], dict[str, float]
        ] = {}
        rebalances: list[RiskRebalance] = []
        for key in sorted(
            groups,
            key=lambda item: (
                _optional_key(item[0]),
                _optional_key(item[1]),
                item[2],
                item[3],
            ),
        ):
            experiment_id, split_id, horizon_bars, _ = key
            stream = (experiment_id, split_id, horizon_bars)
            group = groups[key]
            assets = tuple(sorted(group))
            expected_universe = universes.setdefault(stream, assets)
            if assets != expected_universe:
                raise ValueError("asset universe changed within a risk stream")
            previous = previous_by_stream.setdefault(
                stream, {asset: 0.0 for asset in expected_universe}
            )
            calculation = _calculate_portfolio(
                tuple((asset, group[asset].combined_score) for asset in assets),
                previous,
                self.config,
            )
            weights = {
                asset: (proposed, gross, final)
                for asset, proposed, gross, final in calculation.weights
            }
            targets = tuple(
                RiskTarget(
                    asset=asset,
                    event_at=group[asset].event_at,
                    available_at=group[asset].available_at,
                    decision_at=group[asset].decision_at,
                    horizon_bars=horizon_bars,
                    experiment_id=experiment_id,
                    split_id=split_id,
                    ensemble_score=group[asset].combined_score,
                    proposed_target_weight=weights[asset][0],
                    gross_constrained_target_weight=weights[asset][1],
                    final_target_weight=weights[asset][2],
                )
                for asset in assets
            )
            rebalances.append(
                RiskRebalance(
                    experiment_id=experiment_id,
                    split_id=split_id,
                    horizon_bars=horizon_bars,
                    decision_at=targets[0].decision_at,
                    targets=targets,
                    executable=calculation.executable,
                    proposed_gross=calculation.proposed_gross,
                    gross_scale=calculation.gross_scale,
                    requested_turnover=calculation.requested_turnover,
                    turnover_scale=calculation.turnover_scale,
                    final_gross=calculation.final_gross,
                )
            )
            if calculation.executable:
                previous_by_stream[stream] = {
                    target.asset: target.final_target_weight
                    for target in targets
                    if target.final_target_weight is not None
                }

        return RiskPolicyResult(config=self.config, rebalances=tuple(rebalances))


__all__ = [
    "FixedRiskPolicy",
    "RiskPolicyConfig",
    "RiskPolicyResult",
    "RiskRebalance",
    "RiskTarget",
]
