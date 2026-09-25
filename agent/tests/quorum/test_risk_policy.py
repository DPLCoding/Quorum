"""Tests for Task 6's deterministic portfolio risk boundary."""

from __future__ import annotations

import ast
import inspect
import itertools
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

import src.quorum.risk as risk_package
import src.quorum.risk._codec as risk_codec_module
import src.quorum.risk.policy as risk_module
from src.quorum import (
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    StaticEnsemble,
    StaticEnsembleConfig,
    StaticEnsembleResult,
)
from src.quorum.risk import (
    FixedRiskPolicy,
    RiskPolicyConfig,
    RiskPolicyResult,
)

UTC = timezone.utc
BASE_EVENT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)


def _ensemble(
    rows: tuple[
        tuple[
            str,
            float,
            datetime,
            datetime,
            str | None,
            str | None,
            int,
            bool,
        ],
        ...,
    ],
) -> StaticEnsembleResult:
    config = StaticEnsembleConfig(
        (
            ExpertWeight("expert.a", "v1", 0.5),
            ExpertWeight("expert.b", "v1", 0.5),
        ),
        sell_threshold=-0.25,
        buy_threshold=0.25,
    )
    predictions: list[ExpertPrediction] = []
    for (
        asset,
        score,
        event_at,
        decision_at,
        experiment,
        split,
        horizon,
        complete,
    ) in rows:
        expert_ids = ("expert.a", "expert.b") if complete else ("expert.a",)
        predictions.extend(
            ExpertPrediction(
                expert_id=expert_id,
                expert_version="v1",
                asset=asset,
                event_at=event_at,
                available_at=event_at,
                decision_at=decision_at,
                horizon_bars=horizon,
                score=score,
                experiment_id=experiment,
                split_id=split,
            )
            for expert_id in expert_ids
        )
    return StaticEnsemble(config).combine(ExpertResult(tuple(predictions)))


def _row(
    asset: str,
    score: float,
    step: int = 0,
    *,
    experiment: str | None = "exp-1",
    split: str | None = "split-1",
    horizon: int = 5,
    complete: bool = True,
) -> tuple[str, float, datetime, datetime, str | None, str | None, int, bool]:
    event_at = BASE_EVENT + timedelta(hours=step)
    return (
        asset,
        score,
        event_at,
        event_at + timedelta(minutes=5),
        experiment,
        split,
        horizon,
        complete,
    )


def _policy(
    *, gross: float = 1.0, name: float = 0.4, turnover: float = 2.0
) -> FixedRiskPolicy:
    return FixedRiskPolicy(
        RiskPolicyConfig(
            max_gross_exposure=gross,
            max_abs_weight_per_asset=name,
            max_turnover=turnover,
        )
    )


def test_risk_package_exports_only_public_task6_types() -> None:
    assert risk_package.__all__ == [
        "FixedRiskPolicy",
        "RiskPolicyConfig",
        "RiskPolicyResult",
        "RiskRebalance",
        "RiskTarget",
    ]


def test_config_validation_immutability_and_deterministic_serialization() -> None:
    config = RiskPolicyConfig(0.8, 0.4, 0.0)

    assert RiskPolicyConfig.from_json(config.to_json()) == config
    assert config.to_json() == config.to_json()
    with pytest.raises(FrozenInstanceError):
        config.max_turnover = 1.0  # type: ignore[misc]

    for field in (
        "max_gross_exposure",
        "max_abs_weight_per_asset",
        "max_turnover",
    ):
        values: dict[str, object] = {
            "max_gross_exposure": config.max_gross_exposure,
            "max_abs_weight_per_asset": config.max_abs_weight_per_asset,
            "max_turnover": config.max_turnover,
        }
        values[field] = True
        with pytest.raises(TypeError):
            RiskPolicyConfig(**values)  # type: ignore[arg-type]

    for invalid_version in (True, 1.0):
        payload = config.to_dict()
        payload["schema_version"] = invalid_version
        with pytest.raises(ValueError, match="schema_version"):
            RiskPolicyConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ((0.0, 0.4, 1.0), ValueError),
        ((1.1, 0.4, 1.0), ValueError),
        ((0.8, 0.0, 1.0), ValueError),
        ((0.8, 1.1, 1.0), ValueError),
        ((0.8, 0.4, -0.1), ValueError),
        ((0.8, 0.4, 2.1), ValueError),
        ((math.nan, 0.4, 1.0), ValueError),
        ((0.8, math.inf, 1.0), ValueError),
        ((0.8, 0.4, math.inf), ValueError),
    ),
)
def test_config_rejects_nonfinite_and_out_of_range_values(
    values: tuple[float, float, float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        RiskPolicyConfig(*values)


def test_score_to_target_mapping_preserves_continuous_evidence() -> None:
    scores = (1.0, 0.5, 0.0, -0.5, -1.0)
    rows = tuple(_row("A", score, step) for step, score in enumerate(scores))

    result = _policy(name=0.4).apply(_ensemble(rows))

    assert [item.targets[0].proposed_target_weight for item in result.rebalances] == [
        0.4,
        0.2,
        0.0,
        -0.2,
        -0.4,
    ]
    assert [
        item.targets[0].gross_constrained_target_weight for item in result.rebalances
    ] == [0.4, 0.2, 0.0, -0.2, -0.4]


def test_gross_cap_scales_proportionally_and_preserves_sign() -> None:
    result = _policy(gross=0.5, name=0.4).apply(
        _ensemble((_row("A", 1.0), _row("B", -1.0)))
    )
    rebalance = result.rebalances[0]

    assert rebalance.proposed_gross == pytest.approx(0.8)
    assert rebalance.gross_scale == pytest.approx(0.625)
    assert [item.gross_constrained_target_weight for item in rebalance.targets] == [
        pytest.approx(0.25),
        pytest.approx(-0.25),
    ]
    assert rebalance.final_gross == pytest.approx(0.5)


def test_gross_under_cap_is_not_scaled_up_and_long_short_uses_absolute_gross() -> None:
    under = _policy(gross=1.0, name=0.4).apply(_ensemble((_row("A", 0.5),)))
    mixed = _policy(gross=1.0, name=0.4).apply(
        _ensemble((_row("A", 1.0), _row("B", -1.0)))
    )

    assert under.rebalances[0].gross_scale == 1.0
    assert under.rebalances[0].final_gross == pytest.approx(0.2)
    assert mixed.rebalances[0].proposed_gross == pytest.approx(0.8)
    assert mixed.rebalances[0].final_gross == pytest.approx(0.8)


def test_turnover_from_flat_scales_the_whole_delta_vector() -> None:
    result = _policy(name=0.4, turnover=0.4).apply(
        _ensemble((_row("A", 1.0), _row("B", 1.0)))
    )
    rebalance = result.rebalances[0]

    assert rebalance.requested_turnover == pytest.approx(0.8)
    assert rebalance.turnover_scale == pytest.approx(0.5)
    assert [item.final_target_weight for item in rebalance.targets] == [
        pytest.approx(0.2),
        pytest.approx(0.2),
    ]


def test_zero_turnover_is_a_valid_fully_frozen_policy() -> None:
    result = _policy(name=0.4, turnover=0.0).apply(_ensemble((_row("A", 1.0),)))
    rebalance = result.rebalances[0]

    assert rebalance.requested_turnover == pytest.approx(0.4)
    assert rebalance.turnover_scale == 0.0
    assert rebalance.targets[0].final_target_weight == 0.0


def test_sign_reversal_counts_both_sides_of_the_move() -> None:
    result = _policy(name=0.3, turnover=0.4).apply(
        _ensemble((_row("A", 1.0, 0), _row("A", -1.0, 1)))
    )
    first, second = result.rebalances

    assert first.targets[0].final_target_weight == pytest.approx(0.3)
    assert second.requested_turnover == pytest.approx(0.6)
    assert second.turnover_scale == pytest.approx(2.0 / 3.0)
    assert second.targets[0].final_target_weight == pytest.approx(-0.1)


def test_rotation_counts_close_and_open_symmetrically() -> None:
    result = _policy(name=0.4, turnover=2.0).apply(
        _ensemble(
            (
                _row("A", 1.0, 0),
                _row("B", 0.0, 0),
                _row("A", 0.0, 1),
                _row("B", 1.0, 1),
            )
        )
    )

    assert result.rebalances[1].requested_turnover == pytest.approx(0.8)


def test_sequential_turnover_uses_previous_final_constrained_portfolio() -> None:
    result = _policy(name=0.4, turnover=0.2).apply(
        _ensemble((_row("A", 1.0, 0), _row("A", -1.0, 1)))
    )
    first, second = result.rebalances

    assert first.targets[0].proposed_target_weight == pytest.approx(0.4)
    assert first.targets[0].final_target_weight == pytest.approx(0.2)
    assert second.requested_turnover == pytest.approx(0.6)
    assert second.targets[0].final_target_weight == pytest.approx(0.0)


def test_streams_are_isolated_by_experiment_split_and_horizon() -> None:
    rows = (
        _row("A", 1.0, 0, experiment="exp-1", split="split-1", horizon=5),
        _row("A", 1.0, 1, experiment="exp-1", split="split-2", horizon=5),
        _row("A", 1.0, 2, experiment="exp-1", split="split-1", horizon=10),
        _row("A", 1.0, 3, experiment="exp-2", split="split-1", horizon=5),
    )

    result = _policy(name=0.4, turnover=0.2).apply(_ensemble(rows))

    assert len(result.rebalances) == 4
    assert all(
        item.requested_turnover == pytest.approx(0.4) for item in result.rebalances
    )
    assert all(
        item.targets[0].final_target_weight == pytest.approx(0.2)
        for item in result.rebalances
    )


def test_portfolio_grouping_uses_decision_instant_not_event_time_or_offset() -> None:
    decision_utc = BASE_EVENT + timedelta(minutes=30)
    decision_ny = decision_utc.astimezone(timezone(timedelta(hours=-5)))
    rows = (
        (
            "A",
            0.5,
            BASE_EVENT,
            decision_utc,
            "exp-1",
            "split-1",
            5,
            True,
        ),
        (
            "B",
            -0.5,
            BASE_EVENT - timedelta(hours=1),
            decision_ny,
            "exp-1",
            "split-1",
            5,
            True,
        ),
    )

    result = _policy().apply(_ensemble(rows))

    assert len(result.rebalances) == 1
    assert [item.asset for item in result.rebalances[0].targets] == ["A", "B"]
    assert result.rebalances[0].targets[1].event_at == rows[1][2]


def test_stable_universe_and_duplicate_asset_rules_fail_closed() -> None:
    changed_universe = _ensemble(
        (
            _row("A", 0.5, 0),
            _row("A", 0.5, 1),
            _row("B", 0.5, 1),
        )
    )
    with pytest.raises(ValueError, match="universe changed"):
        _policy().apply(changed_universe)

    first = _row("A", 0.5, 0)
    duplicate = list(_row("A", 0.5, 0))
    duplicate[2] = BASE_EVENT - timedelta(minutes=1)
    with pytest.raises(ValueError, match="duplicate ensemble decision"):
        _policy().apply(_ensemble((first, tuple(duplicate))))  # type: ignore[arg-type]


def test_incomplete_group_abstains_and_does_not_update_stream_state() -> None:
    result = _policy(name=0.4, turnover=0.4).apply(
        _ensemble(
            (
                _row("A", 1.0, 0),
                _row("B", 1.0, 0),
                _row("A", -1.0, 1),
                _row("B", -1.0, 1, complete=False),
                _row("A", -1.0, 2),
                _row("B", -1.0, 2),
            )
        )
    )
    first, incomplete, third = result.rebalances

    assert first.executable
    assert not incomplete.executable
    assert incomplete.proposed_gross is None
    assert all(
        item.proposed_target_weight is None
        and item.gross_constrained_target_weight is None
        and item.final_target_weight is None
        for item in incomplete.targets
    )
    assert third.requested_turnover == pytest.approx(1.2)
    assert [item.final_target_weight for item in third.targets] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]


def test_permutation_invariance_and_authoritative_result_round_trip() -> None:
    ensemble = _ensemble(
        (
            _row("B", -0.5, 1),
            _row("A", 0.5, 0),
            _row("B", 1.0, 0),
            _row("A", -1.0, 1),
        )
    )
    expected = _policy(gross=0.7, name=0.4, turnover=0.5).apply(ensemble)

    for permutation in itertools.permutations(ensemble.decisions):
        reordered = StaticEnsembleResult(ensemble.config, permutation)
        actual = _policy(gross=0.7, name=0.4, turnover=0.5).apply(reordered)
        assert actual.to_json() == expected.to_json()

    restored = RiskPolicyResult.from_json(expected.to_json())
    assert restored == expected
    assert restored.to_json() == expected.to_json()


def test_result_deserialization_recomputes_policy_and_rejects_tampering() -> None:
    result = _policy(gross=0.6, name=0.4, turnover=0.5).apply(
        _ensemble((_row("A", 1.0), _row("B", 1.0)))
    )
    payload = result.to_dict()
    payload["rebalances"][0]["targets"][0]["final_target_weight"] = 0.0

    with pytest.raises(ValueError, match="weights do not match"):
        RiskPolicyResult.from_dict(payload)


def test_all_complete_rebalances_satisfy_feasibility_invariants() -> None:
    result = _policy(gross=0.65, name=0.35, turnover=0.3).apply(
        _ensemble(
            tuple(
                _row(asset, score, step)
                for step, scores in enumerate(((1.0, -1.0), (-1.0, 0.8), (0.2, 0.0)))
                for asset, score in zip(("A", "B"), scores, strict=True)
            )
        )
    )
    previous = {"A": 0.0, "B": 0.0}
    for rebalance in result.rebalances:
        final = {item.asset: item.final_target_weight for item in rebalance.targets}
        assert all(value is not None for value in final.values())
        final_values = {asset: float(value) for asset, value in final.items()}
        assert math.fsum(abs(value) for value in final_values.values()) <= 0.65 + 1e-12
        assert all(abs(value) <= 0.35 + 1e-12 for value in final_values.values())
        assert (
            math.fsum(
                abs(final_values[asset] - previous[asset]) for asset in final_values
            )
            <= 0.3 + 1e-12
        )
        previous = final_values


def test_risk_outputs_are_immutable_and_policy_requires_contract_types() -> None:
    result = _policy().apply(_ensemble((_row("A", 0.5),)))

    with pytest.raises(FrozenInstanceError):
        result.rebalances = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.rebalances[0].targets[0].final_target_weight = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="RiskPolicyConfig"):
        FixedRiskPolicy(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StaticEnsembleResult"):
        _policy().apply(object())  # type: ignore[arg-type]


def test_risk_policy_module_has_only_allowed_dependencies() -> None:
    allowed_standard_roots = {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "json",
        "math",
        "numbers",
        "typing",
    }
    for module in (risk_module, risk_codec_module):
        tree = ast.parse(inspect.getsource(module))
        dependencies = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        dependencies.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert all(
            dependency in {"src.quorum.ensemble", "src.quorum.risk._codec"}
            or dependency.split(".", 1)[0] in allowed_standard_roots
            for dependency in dependencies
        )
    source = inspect.getsource(risk_module)
    assert ".label" not in source
    assert ".disagreement" not in source
