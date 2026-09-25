"""Tests for Task 5's deterministic fixed-weight evidence ensemble."""

from __future__ import annotations

import ast
import inspect
import itertools
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import src.quorum.ensemble as ensemble_package
import src.quorum.ensemble.static as ensemble_module
from src.quorum import (
    ExpertAttribution,
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    ReportingLabel,
    StaticEnsemble,
    StaticEnsembleConfig,
    StaticEnsembleDecision,
    StaticEnsembleResult,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
BASE_EVENT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
BASE_DECISION = datetime(2026, 1, 2, 12, 5, tzinfo=UTC)


def _config(
    experts: tuple[tuple[str, str, float], ...] = (
        ("expert.a", "v1", 0.5),
        ("expert.b", "v1", 0.3),
        ("expert.c", "v1", 0.2),
    ),
    *,
    sell: float = -0.25,
    buy: float = 0.25,
) -> StaticEnsembleConfig:
    return StaticEnsembleConfig(
        tuple(
            ExpertWeight(expert_id, expert_version, weight)
            for expert_id, expert_version, weight in experts
        ),
        sell_threshold=sell,
        buy_threshold=buy,
    )


def _prediction(
    expert_id: str,
    score: float,
    *,
    expert_version: str = "v1",
    asset: str = "AAA",
    event_at: datetime = BASE_EVENT,
    available_at: datetime | None = None,
    decision_at: datetime = BASE_DECISION,
    horizon_bars: int = 5,
    experiment_id: str | None = "exp-1",
    split_id: str | None = "split-1",
    probability_up: float | None = None,
    confidence: float | None = None,
) -> ExpertPrediction:
    return ExpertPrediction(
        expert_id=expert_id,
        expert_version=expert_version,
        asset=asset,
        event_at=event_at,
        available_at=available_at or event_at,
        decision_at=decision_at,
        horizon_bars=horizon_bars,
        score=score,
        probability_up=probability_up,
        confidence=confidence,
        experiment_id=experiment_id,
        split_id=split_id,
    )


def _complete_predictions(
    scores: tuple[float, float, float] = (1.0, 0.0, -1.0),
) -> tuple[ExpertPrediction, ...]:
    return tuple(
        _prediction(expert_id, score)
        for expert_id, score in zip(
            ("expert.a", "expert.b", "expert.c"), scores, strict=True
        )
    )


def _decision_for_scores(
    scores: tuple[float, float, float],
    *,
    config: StaticEnsembleConfig | None = None,
) -> StaticEnsembleDecision:
    ensemble = StaticEnsemble(config or _config())
    return ensemble.combine(ExpertResult(_complete_predictions(scores))).decisions[0]


def test_ensemble_package_exports_only_public_task5_types() -> None:
    assert ensemble_package.__all__ == [
        "ExpertAttribution",
        "ReportingLabel",
        "StaticEnsemble",
        "StaticEnsembleDecision",
        "StaticEnsembleResult",
    ]


def test_exact_weighted_score_full_attribution_and_disagreement() -> None:
    config = _config(
        (
            ("expert.c", "v1", 0.2),
            ("expert.a", "v1", 0.5),
            ("expert.b", "v1", 0.3),
        )
    )
    available = (
        BASE_EVENT + timedelta(minutes=1),
        BASE_EVENT + timedelta(minutes=3),
        BASE_EVENT + timedelta(minutes=2),
    )
    predictions = tuple(
        _prediction(expert_id, score, available_at=available_at)
        for expert_id, score, available_at in zip(
            ("expert.a", "expert.b", "expert.c"),
            (1.0, 0.0, -1.0),
            available,
            strict=True,
        )
    )

    decision = StaticEnsemble(config).combine(ExpertResult(predictions)).decisions[0]

    assert decision.combined_score == pytest.approx(0.3)
    assert decision.label is ReportingLabel.BUY
    assert decision.available_at is predictions[1].available_at
    assert [item.expert_id for item in decision.attributions] == [
        "expert.a",
        "expert.b",
        "expert.c",
    ]
    assert [item.expert_version for item in decision.attributions] == [
        "v1",
        "v1",
        "v1",
    ]
    assert [item.configured_weight for item in decision.attributions] == [
        0.5,
        0.3,
        0.2,
    ]
    assert [item.present for item in decision.attributions] == [True, True, True]
    assert [item.expert_score for item in decision.attributions] == [1.0, 0.0, -1.0]
    assert [item.weighted_contribution for item in decision.attributions] == [
        0.5,
        0.0,
        -0.2,
    ]
    expected_disagreement = math.sqrt(
        0.5 * (1.0 - 0.3) ** 2 + 0.3 * (0.0 - 0.3) ** 2 + 0.2 * (-1.0 - 0.3) ** 2
    )
    assert decision.disagreement == pytest.approx(expected_disagreement)


def test_missing_expert_abstains_without_renormalization_or_omission() -> None:
    result = StaticEnsemble(_config()).combine(
        ExpertResult((_prediction("expert.a", 1.0), _prediction("expert.c", -1.0)))
    )
    decision = result.decisions[0]

    assert decision.combined_score is None
    assert decision.label is None
    assert decision.disagreement is None
    assert len(decision.attributions) == 3
    missing = decision.attributions[1]
    assert missing.expert_id == "expert.b"
    assert missing.configured_weight == 0.3
    assert not missing.present
    assert missing.expert_score is None
    assert missing.weighted_contribution is None
    assert decision.attributions[0].weighted_contribution == 0.5
    assert decision.attributions[2].weighted_contribution == -0.2


def test_missing_expert_is_distinct_from_genuine_neutral_evidence() -> None:
    incomplete = (
        StaticEnsemble(_config())
        .combine(
            ExpertResult((_prediction("expert.a", 1.0), _prediction("expert.c", -1.0)))
        )
        .decisions[0]
    )
    complete = _decision_for_scores((1.0, 0.0, -1.0))

    assert incomplete.combined_score is None
    assert incomplete.label is None
    assert complete.attributions[1].present
    assert complete.attributions[1].expert_score == 0.0
    assert complete.attributions[1].weighted_contribution == 0.0
    assert complete.combined_score == pytest.approx(0.3)


def test_wrong_version_and_unexpected_expert_fail_closed() -> None:
    ensemble = StaticEnsemble(_config())
    with pytest.raises(ValueError, match="version mismatch"):
        ensemble.combine(
            ExpertResult((_prediction("expert.a", 0.2, expert_version="v2"),))
        )
    with pytest.raises(ValueError, match="unexpected expert_id"):
        ensemble.combine(ExpertResult((_prediction("expert.unknown", 0.2),)))


@pytest.mark.parametrize(
    "changed",
    (
        {"asset": "BBB"},
        {"event_at": BASE_EVENT + timedelta(seconds=1)},
        {"decision_at": BASE_DECISION + timedelta(seconds=1)},
        {"horizon_bars": 6},
        {"experiment_id": "exp-2", "split_id": "split-1"},
        {"split_id": "split-2"},
    ),
)
def test_every_alignment_field_separates_rows(changed: dict[str, object]) -> None:
    first = _prediction("expert.a", 0.4)
    second_kwargs: dict[str, object] = {
        "asset": "AAA",
        "event_at": BASE_EVENT,
        "decision_at": BASE_DECISION,
        "horizon_bars": 5,
        "experiment_id": "exp-1",
        "split_id": "split-1",
    }
    second_kwargs.update(changed)
    second = _prediction("expert.b", 0.6, **second_kwargs)  # type: ignore[arg-type]

    decisions = (
        StaticEnsemble(_config()).combine(ExpertResult((first, second))).decisions
    )

    assert len(decisions) == 2
    assert all(decision.combined_score is None for decision in decisions)


def test_timezone_equivalent_instants_align_and_config_order_selects_offsets() -> None:
    event_ny = BASE_EVENT.astimezone(NEW_YORK)
    decision_ny = BASE_DECISION.astimezone(NEW_YORK)
    first = _prediction("expert.a", 0.2)
    second = _prediction(
        "expert.b",
        0.2,
        event_at=event_ny,
        available_at=BASE_EVENT.astimezone(NEW_YORK),
        decision_at=decision_ny,
    )
    config = _config(
        (("expert.b", "v1", 0.5), ("expert.a", "v1", 0.5)),
        sell=-0.1,
        buy=0.1,
    )

    forward = StaticEnsemble(config).combine(ExpertResult((first, second)))
    reverse = StaticEnsemble(config).combine(ExpertResult((second, first)))

    assert len(forward.decisions) == 1
    assert forward.to_json() == reverse.to_json()
    decision = forward.decisions[0]
    assert decision.event_at is first.event_at
    assert decision.decision_at is first.decision_at
    assert decision.available_at is first.available_at
    assert decision.event_at.utcoffset() == timedelta(0)


def test_latest_contributor_availability_is_propagated_with_original_offset() -> None:
    earlier = BASE_EVENT
    later = (BASE_EVENT + timedelta(minutes=4)).astimezone(NEW_YORK)
    predictions = (
        _prediction("expert.a", 0.1, available_at=earlier),
        _prediction("expert.b", 0.1, available_at=later),
        _prediction("expert.c", 0.1, available_at=BASE_EVENT + timedelta(minutes=2)),
    )

    decision = StaticEnsemble(_config()).combine(ExpertResult(predictions)).decisions[0]

    assert decision.available_at is predictions[1].available_at
    assert decision.available_at.utcoffset() == later.utcoffset()
    assert decision.available_at <= decision.decision_at


def test_permutation_invariance_with_interleaved_assets_and_rows() -> None:
    config = _config((("expert.b", "v1", 0.4), ("expert.a", "v1", 0.6)))
    predictions = (
        _prediction("expert.a", 0.8, asset="BBB"),
        _prediction("expert.b", -0.2, asset="AAA"),
        _prediction("expert.a", 0.4, asset="AAA"),
        _prediction("expert.b", 0.1, asset="BBB"),
    )
    expected = StaticEnsemble(config).combine(ExpertResult(predictions)).to_json()

    for permutation in itertools.permutations(predictions):
        actual = StaticEnsemble(config).combine(ExpertResult(permutation)).to_json()
        assert actual == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    (
        (-0.5, ReportingLabel.SELL),
        (-0.25, ReportingLabel.SELL),
        (-0.1, ReportingLabel.HOLD),
        (0.0, ReportingLabel.HOLD),
        (0.25, ReportingLabel.BUY),
        (0.5, ReportingLabel.BUY),
    ),
)
def test_reporting_threshold_edges_are_inclusive_and_zero_is_hold(
    score: float, expected: ReportingLabel
) -> None:
    config = _config((("expert.a", "v1", 1.0),))
    decision = (
        StaticEnsemble(config)
        .combine(ExpertResult((_prediction("expert.a", score),)))
        .decisions[0]
    )
    assert decision.combined_score == score
    assert decision.label is expected


def test_disagreement_uses_weighted_population_dispersion() -> None:
    identical = _decision_for_scores((0.4, 0.4, 0.4))
    mixed = _decision_for_scores((1.0, 0.0, -1.0))
    opposed = _decision_for_scores((1.0, -1.0, -1.0))
    opposed_score = 0.5 * 1.0 + 0.3 * -1.0 + 0.2 * -1.0
    expected_opposed = math.sqrt(
        0.5 * (1.0 - opposed_score) ** 2
        + 0.3 * (-1.0 - opposed_score) ** 2
        + 0.2 * (-1.0 - opposed_score) ** 2
    )

    assert identical.disagreement == 0.0
    assert mixed.disagreement is not None and mixed.disagreement > 0.0
    assert opposed.disagreement == pytest.approx(expected_opposed)
    assert opposed.disagreement > identical.disagreement


def test_score_bounds_and_tolerance_only_boundary_snap() -> None:
    config = _config((("expert.a", "v1", 0.5), ("expert.b", "v1", 0.5000000000005)))
    ensemble = StaticEnsemble(config)
    positive = ensemble.combine(
        ExpertResult((_prediction("expert.a", 1.0), _prediction("expert.b", 1.0)))
    ).decisions[0]
    negative = ensemble.combine(
        ExpertResult((_prediction("expert.a", -1.0), _prediction("expert.b", -1.0)))
    ).decisions[0]

    assert positive.combined_score == 1.0
    assert positive.disagreement == 0.0
    assert negative.combined_score == -1.0
    assert negative.disagreement == 0.0
    with pytest.raises(ValueError, match="materially exceeds"):
        ensemble_module._bounded_combined_score(1.0 + 2e-12)
    with pytest.raises(ValueError, match="materially exceeds"):
        ensemble_module._bounded_combined_score(-1.0 - 2e-12)


def test_multiple_rows_are_separate_and_canonically_sorted() -> None:
    config = _config((("expert.a", "v1", 0.6), ("expert.b", "v1", 0.4)))
    later_event = BASE_EVENT + timedelta(days=1)
    later_decision = BASE_DECISION + timedelta(days=1)
    predictions = (
        _prediction("expert.b", 0.5, asset="ZZZ"),
        _prediction("expert.a", -0.5, asset="ZZZ"),
        _prediction(
            "expert.a",
            1.0,
            asset="AAA",
            event_at=later_event,
            decision_at=later_decision,
        ),
        _prediction(
            "expert.b",
            0.0,
            asset="AAA",
            event_at=later_event,
            decision_at=later_decision,
        ),
    )

    result = StaticEnsemble(config).combine(ExpertResult(predictions))

    assert [decision.asset for decision in result.decisions] == ["AAA", "ZZZ"]
    assert result.decisions[0].combined_score == pytest.approx(0.6)
    assert result.decisions[1].combined_score == pytest.approx(-0.1)


def test_probability_and_confidence_are_not_used_or_manufactured() -> None:
    config = _config((("expert.a", "v1", 0.5), ("expert.b", "v1", 0.5)))
    plain = ExpertResult((_prediction("expert.a", 0.6), _prediction("expert.b", -0.2)))
    decorated = ExpertResult(
        (
            _prediction("expert.a", 0.6, probability_up=0.99, confidence=0.01),
            _prediction("expert.b", -0.2, probability_up=0.01, confidence=0.99),
        )
    )

    first = StaticEnsemble(config).combine(plain).decisions[0]
    second = StaticEnsemble(config).combine(decorated).decisions[0]

    assert first.combined_score == second.combined_score
    assert first.disagreement == second.disagreement
    assert first.attributions == second.attributions
    assert not hasattr(first, "probability_up")
    assert not hasattr(first, "confidence")


def test_empty_input_returns_empty_result_with_frozen_config() -> None:
    config = _config()
    result = StaticEnsemble(config).combine(ExpertResult(()))
    assert result.config is config
    assert result.decisions == ()


def test_result_serialization_round_trip_is_deterministic_and_instant_aware() -> None:
    predictions = _complete_predictions()
    result = StaticEnsemble(_config()).combine(ExpertResult(predictions))

    first_json = result.to_json()
    restored = StaticEnsembleResult.from_json(first_json)

    assert restored == result
    assert restored.to_json() == first_json
    assert result.to_json() == first_json
    assert (
        ExpertAttribution.from_json(result.decisions[0].attributions[0].to_json())
        == result.decisions[0].attributions[0]
    )
    assert (
        StaticEnsembleDecision.from_json(result.decisions[0].to_json())
        == result.decisions[0]
    )


def test_outputs_are_immutable_and_do_not_retain_caller_lists() -> None:
    result = StaticEnsemble(_config()).combine(ExpertResult(_complete_predictions()))
    decision = result.decisions[0]
    attribution = decision.attributions[0]

    with pytest.raises(FrozenInstanceError):
        attribution.present = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.combined_score = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.decisions = ()  # type: ignore[misc]
    assert isinstance(result.decisions, tuple)
    assert isinstance(decision.attributions, tuple)


def test_output_contracts_fail_closed_on_inconsistent_missingness_and_config() -> None:
    with pytest.raises(ValueError, match="cannot contain"):
        ExpertAttribution("expert.a", "v1", 1.0, False, 0.0, 0.0)
    with pytest.raises(ValueError, match="requires score"):
        ExpertAttribution("expert.a", "v1", 1.0, True, None, None)

    decision = _decision_for_scores((1.0, 0.0, -1.0))
    wrong_config = _config(
        (("expert.a", "v1", 0.4), ("expert.b", "v1", 0.4), ("expert.c", "v1", 0.2))
    )
    with pytest.raises(ValueError, match="do not match ensemble config"):
        StaticEnsembleResult(wrong_config, (decision,))


def test_static_ensemble_requires_contract_types() -> None:
    with pytest.raises(TypeError, match="StaticEnsembleConfig"):
        StaticEnsemble(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExpertResult"):
        StaticEnsemble(_config()).combine(())  # type: ignore[arg-type]


def test_static_ensemble_module_has_only_allowed_dependencies() -> None:
    tree = ast.parse(inspect.getsource(ensemble_module))
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
    allowed_standard_roots = {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "enum",
        "json",
        "math",
        "numbers",
        "typing",
    }
    assert all(
        dependency == "src.quorum.contracts"
        or dependency.split(".", 1)[0] in allowed_standard_roots
        for dependency in dependencies
    )
