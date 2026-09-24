"""Tests for Quorum's deterministic chronological validation coordinator."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import src.quorum.validation.protocol as validation_module
from src.quantlib.crossvalidation import Split, detect_boundary_leakage
from src.quorum import (
    EvaluationMode,
    EvaluationProtocol,
    ExperimentAttempt,
    ExperimentSpec,
    FinalHoldout,
    FinalHoldoutState,
    SplitManifest,
    TimeInterval,
)
from src.quorum.validation import (
    BoundaryLeakageError,
    ChronologicalEvaluationPlan,
    ChronologicalValidationError,
    InsufficientHistoryError,
    OOFRole,
    materialize_chronological_plan,
    require_clean_boundary,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
ATTEMPT_ID = "exp_" + ("a" * 32)


def _bars(
    count: int,
    *,
    start: datetime | None = None,
    duration: timedelta = timedelta(hours=1),
    gap_before: int | None = None,
) -> tuple[TimeInterval, ...]:
    cursor = start or datetime(2026, 1, 1, tzinfo=UTC)
    result: list[TimeInterval] = []
    for position in range(count):
        if gap_before == position:
            cursor += duration * 2
        end = cursor + duration
        result.append(TimeInterval(cursor, end))
        cursor = end
    return tuple(result)


def _holdout_for(bars: tuple[TimeInterval, ...], start_position: int) -> FinalHoldout:
    if start_position < len(bars):
        start = bars[start_position].start
        end = bars[-1].end
    else:
        start = bars[-1].end + timedelta(hours=1)
        end = start + timedelta(hours=4)
    return FinalHoldout(start, end, FinalHoldoutState.LOCKED)


def _protocol(
    bars: tuple[TimeInterval, ...],
    *,
    protocol_id: str = "protocol:chronological:v1",
    mode: EvaluationMode = EvaluationMode.EXPANDING,
    minimum_train_bars: int = 5,
    train_window_bars: int | None = None,
    validation_bars: int = 2,
    test_bars: int = 2,
    step_bars: int = 4,
    purge_bars: int = 1,
    embargo_bars: int = 2,
    holdout_start_position: int = 32,
) -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_id=protocol_id,
        mode=mode,
        minimum_train_bars=minimum_train_bars,
        train_window_bars=train_window_bars,
        validation_bars=validation_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        final_holdout=_holdout_for(bars, holdout_start_position),
        random_seed=42,
    )


def _attempt(protocol_id: str = "protocol:chronological:v1") -> ExperimentAttempt:
    spec = ExperimentSpec(
        evaluation_protocol_id=protocol_id,
        expert_config_ids=("expert:placeholder:v1",),
        ensemble_config_id=None,
        target_definition_id="target:forward-return:v1",
        horizon_bars=3,
        data_snapshot_id="dataset:synthetic:v1",
        data_cutoff_at=datetime(2026, 1, 1, tzinfo=UTC),
        universe_id="universe:synthetic:v1",
        cost_model_id="cost:none:v1",
        code_id="git:task3",
        config_id="config:task3",
        random_seed=42,
        trial_family_id="family:task3",
    )
    return ExperimentAttempt(ATTEMPT_ID, spec, datetime(2025, 12, 1, tzinfo=UTC))


def _same_bar_labels(count: int) -> tuple[int, ...]:
    return tuple(range(count))


def _default_plan() -> (
    tuple[tuple[TimeInterval, ...], EvaluationProtocol, ChronologicalEvaluationPlan]
):
    bars = _bars(40)
    protocol = _protocol(bars)
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, _same_bar_labels(len(bars))
    )
    return bars, protocol, plan


def test_validation_module_has_only_allowed_dependencies() -> None:
    tree = ast.parse(inspect.getsource(validation_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    nonstdlib = {
        module
        for module in imported_modules
        if module.split(".", 1)[0] not in sys.stdlib_module_names
        and module != "__future__"
    }
    assert nonstdlib == {
        "numpy",
        "src.quantlib.crossvalidation",
        "src.quorum.contracts",
        "src.quorum.experiments",
    }


def test_basic_chronological_materialization_and_manifest_round_trip() -> None:
    _, protocol, plan = _default_plan()
    first = plan.folds[0]

    assert first.train_positions == (0, 1, 2, 3, 4)
    assert first.purge_positions == (5,)
    assert first.validation_positions == (6, 7)
    assert first.test_positions == (8, 9)
    assert first.embargo_positions == (10, 11)
    assert first.leakage_audit.clean
    assert first.manifest.protocol_id == protocol.protocol_id
    assert not first.manifest.uses_final_holdout
    assert SplitManifest.from_json(first.manifest.to_json()) == first.manifest
    assert all(
        fold.train_positions[-1] < fold.validation_positions[0] < fold.test_positions[0]
        for fold in plan.folds
    )


def test_expanding_folds_retain_all_eligible_history_without_future_rows() -> None:
    _, _, plan = _default_plan()
    sizes = [len(fold.train_positions) for fold in plan.folds]

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert set(plan.folds[0].train_positions) <= set(plan.folds[1].train_positions)
    for fold in plan.folds:
        assert max(fold.train_positions) < min(fold.validation_positions)


def test_rolling_folds_use_only_the_most_recent_bounded_history() -> None:
    bars = _bars(40)
    protocol = _protocol(
        bars,
        mode=EvaluationMode.ROLLING,
        minimum_train_bars=4,
        train_window_bars=6,
    )
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, _same_bar_labels(len(bars))
    )

    assert plan.folds[0].train_positions == (0, 1, 2, 3, 4, 5)
    assert plan.folds[1].train_positions == (4, 5, 6, 7, 8, 9)
    assert all(len(fold.train_positions) <= 6 for fold in plan.folds)


def test_explicit_purge_is_absent_from_training_and_manifested() -> None:
    bars, _, plan = _default_plan()
    first = plan.folds[0]

    assert set(first.train_positions).isdisjoint(first.purge_positions)
    assert first.purge_positions == (5,)
    assert first.manifest.purge_intervals == (bars[5],)


def test_forward_label_overlap_is_purged_and_quantlib_audit_is_clean() -> None:
    bars = _bars(40)
    protocol = _protocol(bars, minimum_train_bars=4)
    labels = tuple(min(position + 3, len(bars) - 1) for position in range(len(bars)))
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, labels
    )
    first = plan.folds[0]
    evaluation_start = first.validation_positions[0]

    assert all(
        labels[position] < evaluation_start for position in first.train_positions
    )
    assert {evaluation_start - 3, evaluation_start - 2} <= set(first.purge_positions)
    split = Split(
        train=np.asarray(first.train_positions),
        test=np.asarray(first.validation_positions + first.test_positions),
        purged=len(first.purge_positions),
        embargoed=len(first.embargo_positions),
        test_bounds=(first.validation_positions[0], first.test_positions[-1]),
    )
    assert detect_boundary_leakage(split, labels, n_samples=len(bars)).clean


def test_closed_endpoint_label_touching_evaluation_is_purged() -> None:
    bars = _bars(40)
    protocol = _protocol(
        bars, minimum_train_bars=3, purge_bars=0, holdout_start_position=32
    )
    labels = tuple(min(position + 1, len(bars) - 1) for position in range(len(bars)))
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, labels
    )
    first = plan.folds[0]
    boundary = first.validation_positions[0]

    assert labels[boundary - 1] == boundary
    assert boundary - 1 in first.purge_positions
    assert boundary - 1 not in first.train_positions


def test_intentionally_dirty_split_is_rejected_not_warned() -> None:
    labels = (0, 2, 2, 3)
    dirty = Split(
        train=np.asarray((0, 1)),
        test=np.asarray((2, 3)),
        purged=0,
        embargoed=0,
        test_bounds=(2, 3),
    )

    with pytest.raises(BoundaryLeakageError, match="dirty chronological boundary"):
        require_clean_boundary(dirty, labels, n_samples=4)


def test_embargo_is_materialized_immediately_after_evaluation_where_available() -> None:
    _, protocol, plan = _default_plan()
    for fold in plan.folds:
        expected = tuple(
            range(
                fold.test_positions[-1] + 1,
                min(
                    fold.test_positions[-1] + 1 + protocol.embargo_bars,
                    32,
                ),
            )
        )
        assert fold.embargo_positions == expected


@pytest.mark.parametrize(
    "case",
    ["too_few_total", "label_purge", "rolling_window", "evaluation_room"],
)
def test_insufficient_history_cases_fail_closed(case: str) -> None:
    if case == "too_few_total":
        bars = _bars(8)
        protocol = _protocol(
            bars, minimum_train_bars=6, holdout_start_position=len(bars)
        )
        labels = _same_bar_labels(len(bars))
    elif case == "label_purge":
        bars = _bars(20)
        protocol = _protocol(bars, minimum_train_bars=4, holdout_start_position=16)
        labels = tuple(max(position, 15) for position in range(len(bars)))
    elif case == "rolling_window":
        bars = _bars(16)
        protocol = _protocol(
            bars,
            mode=EvaluationMode.ROLLING,
            minimum_train_bars=8,
            train_window_bars=8,
            holdout_start_position=12,
        )
        labels = _same_bar_labels(len(bars))
    else:
        bars = _bars(14)
        protocol = _protocol(bars, minimum_train_bars=5, holdout_start_position=9)
        labels = _same_bar_labels(len(bars))

    with pytest.raises(InsufficientHistoryError):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id), protocol, bars, labels
        )


def test_locked_final_holdout_is_physically_fenced() -> None:
    bars, protocol, plan = _default_plan()
    holdout = protocol.final_holdout.interval

    for fold in plan.folds:
        assert all(position < 32 for position in fold.train_positions)
        assert all(position < 32 for position in fold.validation_positions)
        assert all(position < 32 for position in fold.test_positions)
        assert all(
            not holdout.overlaps(interval)
            for interval in (
                fold.manifest.train_intervals
                + fold.manifest.validation_intervals
                + fold.manifest.test_intervals
            )
        )
    assert max(slot.sample_position for slot in plan.oof_slots) < 32
    assert bars[32].start == protocol.final_holdout.start


def test_holdout_straddling_bar_is_rejected_without_clipping() -> None:
    bars = _bars(16)
    boundary = bars[10].start + timedelta(minutes=30)
    protocol = EvaluationProtocol(
        protocol_id="protocol:chronological:v1",
        mode=EvaluationMode.EXPANDING,
        minimum_train_bars=3,
        train_window_bars=None,
        validation_bars=1,
        test_bars=1,
        step_bars=2,
        purge_bars=0,
        embargo_bars=0,
        final_holdout=FinalHoldout(
            boundary,
            bars[-1].end,
            FinalHoldoutState.LOCKED,
        ),
        random_seed=42,
    )

    with pytest.raises(ChronologicalValidationError, match="straddles"):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id),
            protocol,
            bars,
            _same_bar_labels(len(bars)),
        )


def test_every_oof_test_position_is_assigned_exactly_once() -> None:
    _, _, plan = _default_plan()
    expected = {position for fold in plan.folds for position in fold.test_positions}
    actual = [slot.sample_position for slot in plan.test_slots]

    assert set(actual) == expected
    assert len(actual) == len(set(actual))
    assert all(slot.role is OOFRole.TEST for slot in plan.test_slots)
    assert all(slot.role is OOFRole.VALIDATION for slot in plan.validation_slots)
    for fold in plan.folds:
        assert set(fold.train_positions).isdisjoint(fold.test_positions)


def test_duplicate_oof_assignment_fails_closed() -> None:
    _, _, plan = _default_plan()

    with pytest.raises(ValueError, match="cannot be assigned more than once"):
        replace(plan, oof_slots=plan.oof_slots + (plan.oof_slots[0],))


def test_missing_or_unknown_oof_assignment_fails_closed() -> None:
    _, _, plan = _default_plan()
    with pytest.raises(ValueError, match="exactly match"):
        replace(plan, oof_slots=plan.oof_slots[:-1])

    unknown = replace(
        plan.oof_slots[0],
        split_id="split_0000_" + ("f" * 64),
    )
    with pytest.raises(ValueError, match="exactly match"):
        replace(plan, oof_slots=(unknown, *plan.oof_slots[1:]))


def test_plan_and_nested_boundaries_are_immutable() -> None:
    _, _, plan = _default_plan()

    assert isinstance(plan.folds, tuple)
    assert isinstance(plan.folds[0].train_positions, tuple)
    assert isinstance(plan.oof_slots, tuple)
    with pytest.raises(FrozenInstanceError):
        plan.attempt_id = "exp_" + ("b" * 32)  # type: ignore[misc]


def test_overlapping_evaluation_configuration_is_rejected() -> None:
    bars = _bars(40)
    protocol = _protocol(bars, validation_bars=2, test_bars=2, step_bars=3)

    with pytest.raises(ChronologicalValidationError, match="do not overlap"):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id),
            protocol,
            bars,
            _same_bar_labels(len(bars)),
        )


def test_identical_inputs_produce_identical_plans_ids_and_manifests() -> None:
    bars = _bars(40)
    protocol = _protocol(bars)
    attempt = _attempt(protocol.protocol_id)
    labels = _same_bar_labels(len(bars))

    first = materialize_chronological_plan(attempt, protocol, bars, labels)
    second = materialize_chronological_plan(attempt, protocol, bars, labels)

    assert first == second
    assert tuple(fold.manifest.split_id for fold in first.folds) == tuple(
        fold.manifest.split_id for fold in second.folds
    )
    assert tuple(fold.manifest.to_json() for fold in first.folds) == tuple(
        fold.manifest.to_json() for fold in second.folds
    )


def test_split_ids_bind_science_not_repeated_attempt_identity() -> None:
    bars = _bars(40)
    protocol = _protocol(bars)
    first_attempt = _attempt(protocol.protocol_id)
    repeated_attempt = ExperimentAttempt(
        "exp_" + ("b" * 32),
        first_attempt.spec,
        first_attempt.registered_at + timedelta(days=1),
    )
    labels = _same_bar_labels(len(bars))

    first = materialize_chronological_plan(first_attempt, protocol, bars, labels)
    repeated = materialize_chronological_plan(repeated_attempt, protocol, bars, labels)
    changed_labels = (*labels[:-1], labels[-1] + 1)
    changed = materialize_chronological_plan(
        first_attempt, protocol, bars, changed_labels
    )

    assert first.attempt_id != repeated.attempt_id
    assert tuple(fold.manifest.split_id for fold in first.folds) == tuple(
        fold.manifest.split_id for fold in repeated.folds
    )
    assert tuple(fold.manifest.split_id for fold in first.folds) != tuple(
        fold.manifest.split_id for fold in changed.folds
    )


def test_experiment_attempt_must_reference_the_exact_protocol() -> None:
    bars = _bars(40)
    protocol = _protocol(bars)

    with pytest.raises(ChronologicalValidationError, match="does not match"):
        materialize_chronological_plan(
            _attempt("protocol:other"),
            protocol,
            bars,
            _same_bar_labels(len(bars)),
        )


@pytest.mark.parametrize("malformation", ["unsorted", "overlap", "duplicate"])
def test_bar_axis_rejects_non_chronological_or_overlapping_input(
    malformation: str,
) -> None:
    valid = _bars(20)
    if malformation == "unsorted":
        bars = (valid[1], valid[0], *valid[2:])
    elif malformation == "overlap":
        bars = (
            TimeInterval(valid[0].start, valid[1].end),
            valid[1],
            *valid[2:],
        )
    else:
        bars = (valid[0], valid[0], *valid[2:])
    protocol = _protocol(valid, minimum_train_bars=3, holdout_start_position=16)

    with pytest.raises(ValueError, match="ordered|overlap"):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id),
            protocol,
            bars,
            _same_bar_labels(len(bars)),
        )


def test_reversed_bar_is_rejected_by_the_accepted_interval_contract() -> None:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="start must be earlier"):
        TimeInterval(start, start - timedelta(hours=1))


def test_dst_fallback_bar_axis_uses_actual_instants() -> None:
    utc_start = datetime(2026, 11, 1, 4, tzinfo=UTC)
    utc_bars = _bars(24, start=utc_start, duration=timedelta(minutes=30))
    bars = tuple(
        TimeInterval(
            bar.start.astimezone(NEW_YORK),
            bar.end.astimezone(NEW_YORK),
        )
        for bar in utc_bars
    )
    assert any(
        bars[index].start.replace(tzinfo=None)
        > bars[index + 1].start.replace(tzinfo=None)
        for index in range(len(bars) - 1)
    )
    protocol = _protocol(
        bars,
        minimum_train_bars=4,
        validation_bars=2,
        test_bars=2,
        step_bars=4,
        purge_bars=0,
        embargo_bars=0,
        holdout_start_position=20,
    )

    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, _same_bar_labels(len(bars))
    )

    assert plan.folds
    assert all(
        fold.train_positions[-1] < fold.validation_positions[0] for fold in plan.folds
    )


def test_genuine_market_gap_is_not_coalesced_into_observed_time() -> None:
    bars = _bars(40, gap_before=3)
    protocol = _protocol(bars)
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, _same_bar_labels(len(bars))
    )

    first = plan.folds[0]
    assert first.train_positions == (0, 1, 2, 3, 4)
    assert len(first.manifest.train_intervals) == 2
    assert first.manifest.train_intervals[0].end == bars[2].end
    assert first.manifest.train_intervals[1].start == bars[3].start


def test_coordinator_requires_only_boundaries_and_label_metadata() -> None:
    bars, protocol, plan = _default_plan()

    assert plan.attempt_id == ATTEMPT_ID
    assert len(bars) == 40
    assert not hasattr(plan, "prices")
    assert not hasattr(plan, "features")
    assert protocol.final_holdout.state is FinalHoldoutState.LOCKED


@pytest.mark.parametrize(
    ("mode", "minimum", "window", "purge", "embargo", "label_horizon"),
    [
        (EvaluationMode.EXPANDING, 3, None, 0, 0, 0),
        (EvaluationMode.EXPANDING, 5, None, 2, 3, 2),
        (EvaluationMode.ROLLING, 4, 8, 1, 2, 0),
        (EvaluationMode.ROLLING, 5, 10, 2, 1, 3),
    ],
)
def test_materialized_fold_invariants_across_protocol_combinations(
    mode: EvaluationMode,
    minimum: int,
    window: int | None,
    purge: int,
    embargo: int,
    label_horizon: int,
) -> None:
    bars = _bars(64)
    protocol = _protocol(
        bars,
        mode=mode,
        minimum_train_bars=minimum,
        train_window_bars=window,
        validation_bars=2,
        test_bars=3,
        step_bars=5,
        purge_bars=purge,
        embargo_bars=embargo,
        holdout_start_position=52,
    )
    labels = tuple(
        min(position + label_horizon, len(bars) - 1) for position in range(len(bars))
    )
    plan = materialize_chronological_plan(
        _attempt(protocol.protocol_id), protocol, bars, labels
    )

    seen: set[int] = set()
    for fold in plan.folds:
        train = set(fold.train_positions)
        validation = set(fold.validation_positions)
        test = set(fold.test_positions)
        assert len(train) >= minimum
        assert max(train) < min(validation)
        assert train.isdisjoint(validation | test)
        assert validation.isdisjoint(test)
        assert train.isdisjoint(fold.purge_positions)
        assert train.isdisjoint(fold.embargo_positions)
        assert fold.leakage_audit.clean
        assert seen.isdisjoint(test)
        seen.update(test)
        if window is not None:
            assert len(train) <= window
    assert seen == {slot.sample_position for slot in plan.test_slots}


def test_label_span_shape_and_direction_are_strict() -> None:
    bars = _bars(20)
    protocol = _protocol(bars, minimum_train_bars=3, holdout_start_position=16)

    with pytest.raises(ValueError, match="entries"):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id), protocol, bars, tuple(range(19))
        )
    invalid = list(range(20))
    invalid[5] = 4
    with pytest.raises(ValueError, match="cannot end before"):
        materialize_chronological_plan(
            _attempt(protocol.protocol_id), protocol, bars, invalid
        )
