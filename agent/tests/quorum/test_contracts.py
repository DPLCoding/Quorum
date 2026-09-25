"""Tests for the additive Quorum 0.1 scientific contract spine."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import src.quorum as quorum_package
import src.quorum.contracts as contract_module
from src.quorum import (
    EvaluationMode,
    EvaluationProtocol,
    ExpertPrediction,
    ExpertProtocol,
    ExpertResult,
    ExpertWeight,
    FinalHoldout,
    FinalHoldoutState,
    PredictionContext,
    SplitManifest,
    StaticEnsembleConfig,
    TimeInterval,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def test_package_exports_only_intended_quorum_contracts() -> None:
    assert set(quorum_package.__all__) == {
        "BoundaryLeakageError",
        "ChronologicalEvaluationPlan",
        "ChronologicalValidationError",
        "DatasetProvenance",
        "DatasetSnapshot",
        "DatasetSnapshotError",
        "DatasetSnapshotIntegrityError",
        "DatasetSnapshotManifest",
        "DatasetSnapshotNotFoundError",
        "DatasetSnapshotStore",
        "EvaluationMode",
        "EvaluationProtocol",
        "ExperimentAttempt",
        "ExperimentEvent",
        "ExperimentHistoryRecord",
        "ExperimentLedger",
        "ExperimentLedgerCorruptionError",
        "ExperimentOutcome",
        "ExperimentRecord",
        "ExperimentSpec",
        "ExperimentState",
        "ExpertAttribution",
        "ExpertPrediction",
        "ExpertProtocol",
        "ExpertResult",
        "ExpertWeight",
        "ExternalRecordRefs",
        "FinalHoldout",
        "FinalHoldoutState",
        "FixedRiskPolicy",
        "InvalidExperimentTransition",
        "InsufficientHistoryError",
        "LeakageAudit",
        "MaterializedFold",
        "MarketBar",
        "MeanReversionExpert",
        "MomentumExpert",
        "OOFRole",
        "OOFSlot",
        "PredictionContext",
        "ReportingLabel",
        "RiskPolicyConfig",
        "RiskPolicyResult",
        "RiskRebalance",
        "RiskTarget",
        "SplitManifest",
        "StaticEnsemble",
        "StaticEnsembleConfig",
        "StaticEnsembleDecision",
        "StaticEnsembleResult",
        "TimeInterval",
        "materialize_chronological_plan",
        "require_clean_boundary",
        "TrendExpert",
        "validate_v0_vibe_config",
        "VibeSignalAdapter",
    }


def test_contract_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(inspect.getsource(contract_module))
    roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert roots <= sys.stdlib_module_names | {"__future__"}


def _dt(
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    *,
    tz: timezone = UTC,
) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=tz)


def _prediction(**overrides: object) -> ExpertPrediction:
    values: dict[str, object] = {
        "expert_id": "momentum",
        "expert_version": "1.0.0",
        "asset": "AAPL",
        "event_at": _dt(1, 10, 16),
        "available_at": _dt(1, 10, 16),
        "decision_at": _dt(1, 10, 16),
        "horizon_bars": 5,
        "score": 0.25,
    }
    values.update(overrides)
    return ExpertPrediction(**values)  # type: ignore[arg-type]


def _repeated_hour(minute: int, *, fold: int) -> datetime:
    return datetime(2026, 11, 1, 1, minute, tzinfo=NEW_YORK, fold=fold)


def _locked_holdout() -> FinalHoldout:
    return FinalHoldout(
        start=_dt(3, 1),
        end=_dt(4, 1),
        state=FinalHoldoutState.LOCKED,
    )


def _opened_holdout() -> FinalHoldout:
    return FinalHoldout(
        start=_dt(3, 1),
        end=_dt(4, 1),
        state=FinalHoldoutState.OPENED,
        accessed_at=_dt(4, 2),
    )


def _protocol(**overrides: object) -> EvaluationProtocol:
    values: dict[str, object] = {
        "protocol_id": "v0-oof",
        "mode": EvaluationMode.EXPANDING,
        "minimum_train_bars": 252,
        "train_window_bars": None,
        "validation_bars": 63,
        "test_bars": 63,
        "step_bars": 21,
        "purge_bars": 5,
        "embargo_bars": 5,
        "final_holdout": _locked_holdout(),
        "random_seed": 42,
    }
    values.update(overrides)
    return EvaluationProtocol(**values)  # type: ignore[arg-type]


def _manifest(**overrides: object) -> SplitManifest:
    values: dict[str, object] = {
        "protocol_id": "v0-oof",
        "split_id": "fold-000",
        "fold_index": 0,
        "train_intervals": (TimeInterval(_dt(1, 1), _dt(1, 20)),),
        "purge_intervals": (TimeInterval(_dt(1, 20), _dt(1, 22)),),
        "validation_intervals": (TimeInterval(_dt(1, 22), _dt(2, 1)),),
        "embargo_intervals": (TimeInterval(_dt(2, 1), _dt(2, 3)),),
        "test_intervals": (TimeInterval(_dt(2, 3), _dt(2, 10)),),
        "final_holdout": _locked_holdout(),
        "uses_final_holdout": False,
    }
    values.update(overrides)
    return SplitManifest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
def test_prediction_accepts_score_boundaries(score: float) -> None:
    assert _prediction(score=score).score == score


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), -float("inf"), -1.01, 1.01]
)
def test_prediction_rejects_invalid_scores(score: float) -> None:
    with pytest.raises(ValueError):
        _prediction(score=score)


@pytest.mark.parametrize("score", [True, "0.5", None])
def test_prediction_rejects_non_numeric_scores(score: object) -> None:
    with pytest.raises(TypeError):
        _prediction(score=score)


@pytest.mark.parametrize("field", ["probability_up", "confidence"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_optional_probability_and_confidence_accept_boundaries(
    field: str, value: float
) -> None:
    prediction = _prediction(**{field: value})
    assert getattr(prediction, field) == value


def test_score_only_prediction_does_not_invent_probability_or_confidence() -> None:
    prediction = _prediction(score=-0.8)
    assert prediction.probability_up is None
    assert prediction.confidence is None


@pytest.mark.parametrize("field", ["probability_up", "confidence"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01]
)
def test_prediction_rejects_invalid_optional_metrics(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        _prediction(**{field: value})


@pytest.mark.parametrize("field", ["expert_id", "expert_version", "asset"])
@pytest.mark.parametrize("value", ["", "   ", " padded "])
def test_prediction_rejects_invalid_identity_fields(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        _prediction(**{field: value})


@pytest.mark.parametrize("horizon", [0, -1])
def test_prediction_rejects_non_positive_horizon(horizon: int) -> None:
    with pytest.raises(ValueError):
        _prediction(horizon_bars=horizon)


@pytest.mark.parametrize("horizon", [True, 1.5, "5"])
def test_prediction_rejects_non_integer_horizon(horizon: object) -> None:
    with pytest.raises(TypeError):
        _prediction(horizon_bars=horizon)


@pytest.mark.parametrize("field", ["event_at", "available_at", "decision_at"])
def test_prediction_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _prediction(**{field: datetime(2026, 1, 10, 16)})


def test_ordinary_ohlc_observation_can_share_all_three_timestamps() -> None:
    timestamp = _dt(1, 10, 16)
    prediction = _prediction(
        event_at=timestamp,
        available_at=timestamp,
        decision_at=timestamp,
    )
    assert prediction.event_at == timestamp
    assert prediction.available_at == timestamp
    assert prediction.decision_at == timestamp


def test_delayed_news_can_be_available_after_underlying_event() -> None:
    prediction = _prediction(
        event_at=_dt(1, 10, 9),
        available_at=_dt(1, 10, 9, 47, 3),
        decision_at=_dt(1, 10, 9, 47, 5),
    )
    assert prediction.event_at < prediction.available_at < prediction.decision_at


def test_prediction_rejects_information_available_after_decision_cutoff() -> None:
    with pytest.raises(ValueError, match="decision_at"):
        _prediction(
            event_at=_dt(1, 10, 9),
            available_at=_dt(1, 10, 9, 47, 6),
            decision_at=_dt(1, 10, 9, 47, 5),
        )


def test_dst_fold_rejects_information_later_than_cutoff_by_actual_instant() -> None:
    decision = _repeated_hour(30, fold=0)
    later_availability = _repeated_hour(30, fold=1)

    # Python compares these equal because they share one ZoneInfo object, even
    # though the folds represent distinct instants one hour apart.
    assert decision == later_availability
    assert decision.astimezone(UTC) < later_availability.astimezone(UTC)

    with pytest.raises(ValueError, match="decision_at"):
        _prediction(
            event_at=decision,
            available_at=later_availability,
            decision_at=decision,
        )


def test_equivalent_timezone_instants_are_valid_and_offsets_are_preserved() -> None:
    eastern = timezone(timedelta(hours=-5))
    decision = _dt(1, 10, 16)
    same_instant = _dt(1, 10, 11, tz=eastern)
    prediction = _prediction(
        event_at=_dt(1, 10, 14),
        available_at=same_instant,
        decision_at=decision,
    )

    assert prediction.available_at == prediction.decision_at
    assert prediction.to_dict()["available_at"].endswith("-05:00")
    assert ExpertPrediction.from_json(prediction.to_json()) == prediction


def test_split_reference_requires_experiment_reference() -> None:
    with pytest.raises(ValueError, match="split_id requires"):
        _prediction(split_id="fold-1")
    prediction = _prediction(experiment_id="exp-1", split_id="fold-1")
    assert prediction.experiment_id == "exp-1"


def test_prediction_is_frozen() -> None:
    prediction = _prediction()
    with pytest.raises(FrozenInstanceError):
        prediction.score = 0.9  # type: ignore[misc]


def test_metadata_is_copied_recursively_frozen_and_serialized() -> None:
    caller_metadata = {
        "definition": "calibrated validation reliability",
        "source": {"name": "fixture", "columns": ["close", "volume"]},
    }
    prediction = _prediction(metadata=caller_metadata)
    caller_metadata["source"]["name"] = "mutated"  # type: ignore[index]
    caller_metadata["source"]["columns"].append("future")  # type: ignore[index,union-attr]

    assert isinstance(prediction.metadata, MappingProxyType)
    assert prediction.metadata["source"]["name"] == "fixture"  # type: ignore[index]
    assert prediction.metadata["source"]["columns"] == ("close", "volume")  # type: ignore[index]
    with pytest.raises(TypeError):
        prediction.metadata["new"] = "value"  # type: ignore[index]
    assert ExpertPrediction.from_json(prediction.to_json()) == prediction


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": object()},
        {"": "empty-key"},
    ],
)
def test_prediction_rejects_invalid_metadata(metadata: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _prediction(metadata=metadata)


def test_prediction_rejects_large_or_deep_metadata() -> None:
    with pytest.raises(ValueError, match="4096"):
        _prediction(metadata={"payload": "x" * 5000})
    with pytest.raises(ValueError, match="nesting"):
        _prediction(metadata={"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}})


def test_prediction_serialization_is_deterministic_and_strict() -> None:
    prediction = _prediction(
        probability_up=0.6,
        confidence=0.7,
        metadata={"z": 2, "a": 1},
        experiment_id="exp-1",
        split_id="fold-1",
    )
    text = prediction.to_json()
    assert text == prediction.to_json()
    assert ExpertPrediction.from_json(text) == prediction
    assert json.loads(text)["contract"] == "expert_prediction"
    assert json.loads(text)["schema_version"] == 2
    assert json.loads(text)["decision_at"] == prediction.decision_at.isoformat()

    payload = prediction.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        ExpertPrediction.from_dict(payload)

    version_one = prediction.to_dict()
    version_one["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported"):
        ExpertPrediction.from_dict(version_one)


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_schema_version_one_requires_an_actual_integer(
    invalid_version: object,
) -> None:
    payload = TimeInterval(_dt(1, 1), _dt(1, 2)).to_dict()
    payload["schema_version"] = invalid_version

    with pytest.raises(ValueError, match="expected integer 1"):
        TimeInterval.from_dict(payload)


def test_prediction_schema_version_two_rejects_equal_float() -> None:
    payload = _prediction().to_dict()
    payload["schema_version"] = 2.0

    with pytest.raises(ValueError, match="expected integer 2"):
        ExpertPrediction.from_dict(payload)


def test_prediction_context_obeys_same_time_and_reference_contract() -> None:
    context = PredictionContext(
        event_at=_dt(1, 10, 9),
        available_at=_dt(1, 10, 9, 47, 3),
        decision_at=_dt(1, 10, 9, 47, 5),
        horizon_bars=5,
        experiment_id="exp-1",
        split_id="fold-1",
    )
    restored = PredictionContext.from_json(context.to_json())
    assert restored == context
    assert hash(restored) == hash(context)
    with pytest.raises(ValueError, match="available_at"):
        PredictionContext(
            event_at=_dt(1, 10, 9),
            available_at=_dt(1, 10, 9, 47, 6),
            decision_at=_dt(1, 10, 9, 47, 5),
            horizon_bars=5,
        )


def test_expert_result_rejects_duplicate_logical_keys() -> None:
    original = _prediction(score=0.2, available_at=_dt(1, 10, 14))
    conflicting = _prediction(score=0.9, available_at=_dt(1, 10, 15))
    with pytest.raises(ValueError, match="duplicate"):
        ExpertResult((original, conflicting))


def test_expert_result_order_is_not_part_of_identity() -> None:
    apple = _prediction(asset="AAPL")
    microsoft = _prediction(asset="MSFT")
    forward = ExpertResult((apple, microsoft))
    reverse = ExpertResult((microsoft, apple))
    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert ExpertResult.from_json(forward.to_json()) == forward
    with pytest.raises(FrozenInstanceError):
        forward.predictions = ()  # type: ignore[misc]


def test_equivalent_instant_is_a_duplicate_prediction_key() -> None:
    eastern = timezone(timedelta(hours=-5))
    first = _prediction(
        event_at=_dt(1, 10, 14),
        decision_at=_dt(1, 10, 16),
    )
    second = _prediction(
        event_at=_dt(1, 10, 9, tz=eastern),
        decision_at=_dt(1, 10, 11, tz=eastern),
    )
    with pytest.raises(ValueError, match="duplicate"):
        ExpertResult((first, second))


def test_dst_folds_are_distinct_prediction_keys() -> None:
    first_fold = _repeated_hour(30, fold=0)
    second_fold = _repeated_hour(30, fold=1)
    first = _prediction(
        event_at=first_fold,
        available_at=first_fold,
        decision_at=first_fold,
    )
    second = _prediction(
        event_at=second_fold,
        available_at=second_fold,
        decision_at=second_fold,
    )

    assert first.prediction_key != second.prediction_key
    assert len(ExpertResult((first, second)).predictions) == 2


def test_dst_instant_expressed_in_utc_is_still_a_duplicate_key() -> None:
    local = _repeated_hour(30, fold=1)
    utc = local.astimezone(UTC)
    local_prediction = _prediction(
        event_at=local,
        available_at=local,
        decision_at=local,
    )
    utc_prediction = _prediction(
        event_at=utc,
        available_at=utc,
        decision_at=utc,
    )

    with pytest.raises(ValueError, match="duplicate"):
        ExpertResult((local_prediction, utc_prediction))


def test_dst_prediction_round_trip_preserves_instant_offset_and_equality() -> None:
    repeated = _repeated_hour(30, fold=1)
    prediction = _prediction(
        event_at=repeated,
        available_at=repeated,
        decision_at=repeated,
    )
    restored = ExpertPrediction.from_json(prediction.to_json())

    assert restored == prediction
    assert restored.prediction_key == prediction.prediction_key
    assert restored.event_at.astimezone(UTC) == repeated.astimezone(UTC)
    assert restored.to_json() == prediction.to_json()
    assert restored.to_dict()["event_at"].endswith("-05:00")


def test_same_event_at_different_decision_cutoffs_is_not_a_duplicate() -> None:
    first = _prediction(decision_at=_dt(1, 10, 16))
    later = _prediction(decision_at=_dt(1, 10, 17))
    result = ExpertResult((later, first))
    assert result.predictions == (first, later)


def test_empty_expert_result_is_a_valid_no_prediction_batch() -> None:
    result = ExpertResult()
    assert result.predictions == ()
    assert ExpertResult.from_json(result.to_json()) == result


def test_expert_protocol_is_structural() -> None:
    class StubExpert:
        @property
        def expert_id(self) -> str:
            return "stub"

        @property
        def expert_version(self) -> str:
            return "1"

        def predict(
            self,
            prepared_data: dict[str, object],
            context: PredictionContext,
        ) -> ExpertResult:
            del prepared_data, context
            return ExpertResult()

    assert isinstance(StubExpert(), ExpertProtocol)


@pytest.mark.parametrize("weight", [0.0, -0.1])
def test_expert_weight_must_be_strictly_positive(weight: float) -> None:
    with pytest.raises(ValueError):
        ExpertWeight("momentum", "1", weight)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf"), 1.1])
def test_expert_weight_rejects_nonfinite_or_out_of_bounds(weight: float) -> None:
    with pytest.raises(ValueError):
        ExpertWeight("momentum", "1", weight)


def test_expert_weight_round_trip() -> None:
    weight = ExpertWeight("momentum", "1", 1.0)
    assert ExpertWeight.from_json(weight.to_json()) == weight


def test_static_ensemble_is_canonical_immutable_and_not_normalized() -> None:
    momentum = ExpertWeight("momentum", "1", 0.6)
    trend = ExpertWeight("trend", "2", 0.4)
    forward = StaticEnsembleConfig((momentum, trend), -0.1, 0.1)
    reverse = StaticEnsembleConfig((trend, momentum), -0.1, 0.1)

    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert StaticEnsembleConfig.from_json(forward.to_json()) == forward
    with pytest.raises(FrozenInstanceError):
        forward.buy_threshold = 0.2  # type: ignore[misc]


def test_static_ensemble_rejects_empty_or_duplicate_experts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StaticEnsembleConfig((), -0.1, 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        StaticEnsembleConfig(
            (
                ExpertWeight("momentum", "1", 0.5),
                ExpertWeight("momentum", "2", 0.5),
            ),
            -0.1,
            0.1,
        )


@pytest.mark.parametrize("weights", [(0.2, 0.2), (0.6, 0.5)])
def test_static_ensemble_rejects_weights_that_do_not_sum_to_one(
    weights: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        StaticEnsembleConfig(
            (
                ExpertWeight("momentum", "1", weights[0]),
                ExpertWeight("trend", "1", weights[1]),
            ),
            -0.1,
            0.1,
        )


@pytest.mark.parametrize(
    ("sell", "buy"),
    [
        (-1.1, 0.1),
        (-0.1, 1.1),
        (0.1, 0.1),
        (0.2, 0.1),
        (float("nan"), 0.1),
        (-0.1, float("inf")),
    ],
)
def test_static_ensemble_rejects_invalid_thresholds(sell: float, buy: float) -> None:
    with pytest.raises(ValueError):
        StaticEnsembleConfig((ExpertWeight("momentum", "1", 1.0),), sell, buy)


@pytest.mark.parametrize(
    ("sell", "buy"),
    [(0.1, 0.2), (-0.2, -0.1), (0.0, 0.1), (-0.1, 0.0)],
)
def test_static_ensemble_requires_zero_inside_the_neutral_band(
    sell: float, buy: float
) -> None:
    with pytest.raises(ValueError, match="sell_threshold < 0 < buy_threshold"):
        StaticEnsembleConfig((ExpertWeight("momentum", "1", 1.0),), sell, buy)


def test_static_ensemble_accepts_threshold_range_boundaries_around_zero() -> None:
    config = StaticEnsembleConfig(
        (ExpertWeight("momentum", "1", 1.0),),
        sell_threshold=-1.0,
        buy_threshold=1.0,
    )
    assert config.sell_threshold == -1.0
    assert config.buy_threshold == 1.0


def test_time_interval_is_half_open_timezone_aware_and_serializable() -> None:
    first = TimeInterval(_dt(1, 1), _dt(1, 2))
    adjacent = TimeInterval(_dt(1, 2), _dt(1, 3))
    assert not first.overlaps(adjacent)
    assert TimeInterval.from_json(first.to_json()) == first
    with pytest.raises(ValueError, match="timezone-aware"):
        TimeInterval(datetime(2026, 1, 1), _dt(1, 2))
    with pytest.raises(ValueError, match="earlier"):
        TimeInterval(_dt(1, 2), _dt(1, 2))


def test_dst_interval_rejects_absolute_reverse_despite_wall_clock_order() -> None:
    start = _repeated_hour(15, fold=1)
    end = _repeated_hour(45, fold=0)
    assert start.replace(tzinfo=None) < end.replace(tzinfo=None)
    assert start.astimezone(UTC) > end.astimezone(UTC)

    with pytest.raises(ValueError, match="earlier"):
        TimeInterval(start, end)


def test_dst_interval_spanning_repeated_hour_uses_absolute_instants() -> None:
    interval = TimeInterval(
        _repeated_hour(45, fold=0),
        _repeated_hour(15, fold=1),
    )
    inner = TimeInterval(
        _repeated_hour(55, fold=0),
        _repeated_hour(5, fold=1),
    )
    restored = TimeInterval.from_json(interval.to_json())

    assert interval.overlaps(inner)
    assert interval.contains(inner)
    assert restored == interval
    assert hash(restored) == hash(interval)
    assert restored.to_json() == interval.to_json()
    assert restored.start.astimezone(UTC) == interval.start.astimezone(UTC)
    assert restored.end.astimezone(UTC) == interval.end.astimezone(UTC)
    assert restored.to_dict()["start"].endswith("-04:00")
    assert restored.to_dict()["end"].endswith("-05:00")


def test_final_holdout_state_is_unambiguous_and_serializable() -> None:
    locked = _locked_holdout()
    opened = _opened_holdout()
    assert FinalHoldout.from_json(locked.to_json()) == locked
    restored_opened = FinalHoldout.from_json(opened.to_json())
    assert restored_opened == opened
    assert hash(restored_opened) == hash(opened)
    with pytest.raises(TypeError, match="FinalHoldoutState"):
        FinalHoldout(_dt(3, 1), _dt(4, 1), "maybe locked")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot have"):
        FinalHoldout(
            _dt(3, 1),
            _dt(4, 1),
            FinalHoldoutState.LOCKED,
            accessed_at=_dt(4, 2),
        )
    with pytest.raises(ValueError, match="requires"):
        FinalHoldout(_dt(3, 1), _dt(4, 1), FinalHoldoutState.OPENED)
    with pytest.raises(ValueError, match="before its end"):
        FinalHoldout(
            _dt(3, 1),
            _dt(4, 1),
            FinalHoldoutState.OPENED,
            accessed_at=_dt(3, 31),
        )


def test_final_holdout_access_uses_absolute_instant_during_dst_fold() -> None:
    with pytest.raises(ValueError, match="before its end"):
        FinalHoldout(
            datetime(2026, 11, 1, 0, 30, tzinfo=NEW_YORK),
            _repeated_hour(30, fold=1),
            FinalHoldoutState.OPENED,
            accessed_at=_repeated_hour(30, fold=0),
        )


def test_expanding_evaluation_protocol_round_trip_and_immutability() -> None:
    protocol = _protocol()
    assert EvaluationProtocol.from_json(protocol.to_json()) == protocol
    with pytest.raises(FrozenInstanceError):
        protocol.random_seed = 7  # type: ignore[misc]


def test_rolling_evaluation_protocol_requires_explicit_sufficient_window() -> None:
    protocol = _protocol(
        mode=EvaluationMode.ROLLING,
        minimum_train_bars=100,
        train_window_bars=252,
    )
    assert protocol.train_window_bars == 252
    with pytest.raises(ValueError, match="train_window_bars"):
        _protocol(mode=EvaluationMode.ROLLING, train_window_bars=None)
    with pytest.raises(ValueError, match="minimum_train_bars"):
        _protocol(
            mode=EvaluationMode.ROLLING,
            minimum_train_bars=252,
            train_window_bars=100,
        )


def test_evaluation_protocol_rejects_ambiguous_or_open_holdout_policy() -> None:
    with pytest.raises(TypeError, match="EvaluationMode"):
        _protocol(mode="random")
    with pytest.raises(ValueError, match="locked"):
        _protocol(final_holdout=_opened_holdout())
    with pytest.raises(ValueError, match="None"):
        _protocol(mode=EvaluationMode.EXPANDING, train_window_bars=252)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_train_bars", 0),
        ("validation_bars", 0),
        ("test_bars", 0),
        ("step_bars", 0),
        ("purge_bars", -1),
        ("embargo_bars", -1),
        ("random_seed", -1),
        ("random_seed", 2**32),
    ],
)
def test_evaluation_protocol_rejects_invalid_counts(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        _protocol(**{field: value})


def test_split_manifest_round_trip_and_canonical_interval_order() -> None:
    early = TimeInterval(_dt(1, 1), _dt(1, 5))
    late = TimeInterval(_dt(1, 5), _dt(1, 20))
    manifest = _manifest(train_intervals=[late, early])
    assert manifest.train_intervals == (early, late)
    assert SplitManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(FrozenInstanceError):
        manifest.fold_index = 2  # type: ignore[misc]


def test_split_manifest_rejects_missing_train_or_evaluation_intervals() -> None:
    with pytest.raises(ValueError, match="train_intervals"):
        _manifest(train_intervals=())
    with pytest.raises(ValueError, match="at least one"):
        _manifest(validation_intervals=(), test_intervals=())


def test_split_manifest_rejects_overlapping_or_nonchronological_boundaries() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _manifest(
            train_intervals=(TimeInterval(_dt(1, 1), _dt(1, 25)),),
        )
    with pytest.raises(ValueError, match="finish"):
        _manifest(
            train_intervals=(TimeInterval(_dt(2, 11), _dt(2, 20)),),
            purge_intervals=(),
        )
    with pytest.raises(ValueError, match="overlapping intervals"):
        _manifest(
            train_intervals=(
                TimeInterval(_dt(1, 1), _dt(1, 15)),
                TimeInterval(_dt(1, 10), _dt(1, 20)),
            )
        )
    with pytest.raises(ValueError, match="purge_intervals"):
        _manifest(
            purge_intervals=(TimeInterval(_dt(1, 19), _dt(1, 22)),),
        )


def test_split_manifest_requires_validation_to_finish_before_test() -> None:
    with pytest.raises(ValueError, match="validation intervals must finish"):
        _manifest(
            train_intervals=(TimeInterval(_dt(1, 1), _dt(2, 1)),),
            validation_intervals=(TimeInterval(_dt(2, 20), _dt(2, 25)),),
            test_intervals=(TimeInterval(_dt(2, 5), _dt(2, 10)),),
            purge_intervals=(),
            embargo_intervals=(),
        )


def test_split_manifest_keeps_validation_only_and_test_only_valid() -> None:
    assert _manifest(test_intervals=()).test_intervals == ()
    assert _manifest(validation_intervals=()).validation_intervals == ()


def test_split_manifest_orders_and_compares_dst_folds_by_instant() -> None:
    early = TimeInterval(
        _repeated_hour(10, fold=0),
        _repeated_hour(20, fold=0),
    )
    late = TimeInterval(
        _repeated_hour(10, fold=1),
        _repeated_hour(20, fold=1),
    )
    manifest = _manifest(
        train_intervals=(late, early),
        validation_intervals=(),
        test_intervals=(TimeInterval(_dt(11, 2), _dt(11, 3)),),
        purge_intervals=(),
        embargo_intervals=(),
    )
    assert manifest.train_intervals == (early, late)

    with pytest.raises(ValueError, match="training must finish"):
        _manifest(
            train_intervals=(late,),
            validation_intervals=(),
            test_intervals=(early,),
            purge_intervals=(),
            embargo_intervals=(),
        )


def test_split_manifest_rejects_inconsistent_final_holdout_state() -> None:
    with pytest.raises(ValueError, match="OPENED"):
        _manifest(
            final_holdout=_locked_holdout(),
            uses_final_holdout=True,
            validation_intervals=(),
            test_intervals=(TimeInterval(_dt(3, 1), _dt(4, 1)),),
            purge_intervals=(),
            embargo_intervals=(),
        )
    with pytest.raises(ValueError, match="remain LOCKED"):
        _manifest(final_holdout=_opened_holdout(), uses_final_holdout=False)
    with pytest.raises(ValueError, match="locked final holdout"):
        _manifest(test_intervals=(TimeInterval(_dt(3, 5), _dt(3, 10)),))


def test_final_holdout_manifest_rejects_test_before_validation() -> None:
    with pytest.raises(ValueError, match="validation intervals must finish"):
        _manifest(
            final_holdout=_opened_holdout(),
            uses_final_holdout=True,
            train_intervals=(TimeInterval(_dt(1, 1), _dt(2, 1)),),
            validation_intervals=(TimeInterval(_dt(4, 2), _dt(4, 5)),),
            test_intervals=(TimeInterval(_dt(3, 1), _dt(4, 1)),),
            purge_intervals=(),
            embargo_intervals=(),
        )


def test_final_holdout_manifest_requires_contained_test_and_no_train_or_validation() -> (
    None
):
    final_manifest = _manifest(
        final_holdout=_opened_holdout(),
        uses_final_holdout=True,
        train_intervals=(TimeInterval(_dt(1, 1), _dt(2, 1)),),
        validation_intervals=(TimeInterval(_dt(2, 3), _dt(2, 10)),),
        test_intervals=(TimeInterval(_dt(3, 1), _dt(4, 1)),),
        purge_intervals=(TimeInterval(_dt(2, 1), _dt(2, 3)),),
        embargo_intervals=(TimeInterval(_dt(2, 10), _dt(2, 12)),),
    )
    assert final_manifest.uses_final_holdout is True
    assert SplitManifest.from_json(final_manifest.to_json()) == final_manifest

    with pytest.raises(ValueError, match="contained"):
        _manifest(
            final_holdout=_opened_holdout(),
            uses_final_holdout=True,
            validation_intervals=(),
            test_intervals=(TimeInterval(_dt(2, 20), _dt(3, 10)),),
            purge_intervals=(),
            embargo_intervals=(),
        )
    with pytest.raises(ValueError, match="train_intervals"):
        _manifest(
            final_holdout=_opened_holdout(),
            uses_final_holdout=True,
            train_intervals=(TimeInterval(_dt(2, 20), _dt(3, 5)),),
            validation_intervals=(),
            test_intervals=(TimeInterval(_dt(3, 5), _dt(3, 10)),),
            purge_intervals=(),
            embargo_intervals=(),
        )
