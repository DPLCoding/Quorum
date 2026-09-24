"""Tests for the Quorum append-only experiment and research ledger."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import src.quorum.experiments as experiment_module
from src.governance.ledger import append_record
from src.quorum import (
    ExperimentAttempt,
    ExperimentEvent,
    ExperimentLedger,
    ExperimentLedgerCorruptionError,
    ExperimentOutcome,
    ExperimentRecord,
    ExperimentSpec,
    ExperimentState,
    ExternalRecordRefs,
    InvalidExperimentTransition,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
ATTEMPT_A = "exp_" + ("a" * 32)
ATTEMPT_B = "exp_" + ("b" * 32)
EVENT_A = "evt_" + ("a" * 32)
EVENT_B = "evt_" + ("b" * 32)


def _dt(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, minute, tzinfo=UTC)


def _repeated_hour(*, fold: int) -> datetime:
    return datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=fold)


def _spec(**overrides: object) -> ExperimentSpec:
    values: dict[str, object] = {
        "evaluation_protocol_id": "protocol:v0-oof",
        "expert_config_ids": ("expert:momentum:v1", "expert:trend:v1"),
        "ensemble_config_id": "ensemble:static:v1",
        "target_definition_id": "target:forward-return:v1",
        "horizon_bars": 5,
        "data_snapshot_id": "dataset:ohlcv:sha256-abc",
        "data_cutoff_at": _dt(10, 16),
        "universe_id": "universe:sp500-pit:v1",
        "cost_model_id": "cost:us-equity:v1",
        "code_id": "git:0123456789abcdef",
        "config_id": "config:quorum-v0",
        "random_seed": 42,
        "trial_family_id": "family:momentum-vs-trend",
    }
    values.update(overrides)
    return ExperimentSpec(**values)  # type: ignore[arg-type]


def _outcome(**overrides: object) -> ExperimentOutcome:
    values: dict[str, object] = {
        "detail": "Evaluation finished.",
        "references": ExternalRecordRefs(
            hypothesis_id="hyp_0123456789ab",
            strategy_artifact_id="art_0123456789ab",
            run_card_ref="run-card:sha256-abc",
            quorum_artifact_ids=("quorum:predictions:abc",),
        ),
        "metadata": {"summary": {"usable_result": True}},
    }
    values.update(overrides)
    return ExperimentOutcome(**values)  # type: ignore[arg-type]


def _ledger(tmp_path: Path) -> ExperimentLedger:
    return ExperimentLedger(tmp_path / "experiments.jsonl")


def test_experiment_module_has_only_clean_core_dependencies() -> None:
    tree = ast.parse(inspect.getsource(experiment_module))
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
    nonstdlib_imports = {
        module
        for module in imported_modules
        if module.split(".", 1)[0] not in sys.stdlib_module_names
        and module != "__future__"
    }
    assert nonstdlib_imports == {
        "src.governance.ledger",
        "src.quorum.contracts",
    }


def test_identical_specs_are_canonical_and_have_same_full_sha256_identity() -> None:
    forward = _spec(expert_config_ids=("expert:momentum:v1", "expert:trend:v1"))
    reverse = _spec(expert_config_ids=("expert:trend:v1", "expert:momentum:v1"))
    reordered_payload = dict(reversed(list(forward.to_dict().items())))
    reconstructed = ExperimentSpec.from_dict(reordered_payload)

    assert forward == reverse == reconstructed
    assert forward.to_json() == reverse.to_json() == reconstructed.to_json()
    assert forward.fingerprint == reverse.fingerprint == reconstructed.fingerprint
    assert forward.fingerprint.startswith("sha256:")
    assert len(forward.fingerprint) == 71
    assert hash(forward) == hash(reverse)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_bars", 10),
        ("data_snapshot_id", "dataset:ohlcv:sha256-def"),
        ("cost_model_id", "cost:us-equity:v2"),
        ("random_seed", 43),
        ("trial_family_id", "family:alternative"),
    ],
)
def test_changed_scientific_fields_change_fingerprint(
    field: str, value: object
) -> None:
    original = _spec()
    changed = replace(original, **{field: value})
    assert changed.fingerprint != original.fingerprint


def test_spec_timestamp_identity_uses_actual_instant_and_distinguishes_dst_folds() -> (
    None
):
    eastern = timezone(timedelta(hours=-5))
    utc_spec = _spec(data_cutoff_at=_dt(10, 16))
    eastern_spec = _spec(data_cutoff_at=datetime(2026, 1, 10, 11, tzinfo=eastern))
    first_fold = _spec(data_cutoff_at=_repeated_hour(fold=0))
    second_fold = _spec(data_cutoff_at=_repeated_hour(fold=1))

    assert utc_spec == eastern_spec
    assert utc_spec.fingerprint == eastern_spec.fingerprint
    assert first_fold.fingerprint != second_fold.fingerprint


def test_spec_round_trip_preserves_dst_instant_and_serialized_offset() -> None:
    spec = _spec(data_cutoff_at=_repeated_hour(fold=1))
    restored = ExperimentSpec.from_json(spec.to_json())

    assert restored == spec
    assert restored.fingerprint == spec.fingerprint
    assert restored.to_json() == spec.to_json()
    assert restored.to_dict()["data_cutoff_at"].endswith("-05:00")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("evaluation_protocol_id", "", "non-empty"),
        ("code_id", "local path with spaces", "path-independent"),
        ("expert_config_ids", (), "must not be empty"),
        (
            "expert_config_ids",
            ("expert:one", "expert:one"),
            "duplicate",
        ),
        ("horizon_bars", 0, ">= 1"),
        ("random_seed", True, "integer"),
    ],
)
def test_spec_rejects_invalid_scientific_identity_fields(
    field: str, value: object, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _spec(**{field: value})


def test_spec_rejects_naive_data_cutoff() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _spec(data_cutoff_at=datetime(2026, 1, 10, 16))


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_spec_schema_version_is_type_strict(invalid_version: object) -> None:
    payload = _spec().to_dict()
    payload["schema_version"] = invalid_version
    with pytest.raises(ValueError, match="expected integer 1"):
        ExperimentSpec.from_dict(payload)


def test_spec_deserialization_rejects_unknown_keys_and_wrong_contract() -> None:
    unknown = _spec().to_dict()
    unknown["result_metric"] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSpec.from_dict(unknown)

    wrong_contract = _spec().to_dict()
    wrong_contract["contract"] = "completed_experiment"
    with pytest.raises(ValueError, match="expected contract"):
        ExperimentSpec.from_dict(wrong_contract)


def test_external_record_references_are_opaque_optional_and_canonical() -> None:
    empty = ExternalRecordRefs()
    refs = ExternalRecordRefs(
        hypothesis_id="hyp_0123456789ab",
        strategy_artifact_id="art_0123456789ab",
        run_card_ref="run-card:sha256-abc",
        quorum_artifact_ids=("quorum:z", "quorum:a"),
    )

    assert ExternalRecordRefs.from_json(empty.to_json()) == empty
    assert ExternalRecordRefs.from_json(refs.to_json()) == refs
    assert refs.quorum_artifact_ids == ("quorum:a", "quorum:z")
    assert set(refs.to_dict()) == {
        "contract",
        "schema_version",
        "hypothesis_id",
        "strategy_artifact_id",
        "run_card_ref",
        "quorum_artifact_ids",
    }

    for machine_path in ("C:\\research\\run-card.json", "/tmp/run-card.json"):
        with pytest.raises(ValueError, match="path-independent"):
            ExternalRecordRefs(run_card_ref=machine_path)


def test_outcome_metadata_is_copied_frozen_bounded_and_finite() -> None:
    caller = {"summary": {"usable": True, "labels": ["oof"]}}
    outcome = _outcome(metadata=caller)
    caller["summary"]["usable"] = False  # type: ignore[index]
    caller["summary"]["labels"].append("mutated")  # type: ignore[index,union-attr]

    assert isinstance(outcome.metadata, MappingProxyType)
    assert outcome.metadata["summary"]["usable"] is True  # type: ignore[index]
    assert outcome.metadata["summary"]["labels"] == ("oof",)  # type: ignore[index]
    assert ExperimentOutcome.from_json(outcome.to_json()) == outcome

    with pytest.raises(ValueError, match="finite"):
        _outcome(metadata={"bad": float("nan")})
    with pytest.raises(ValueError, match="4096"):
        _outcome(metadata={"too_large": "x" * 5000})
    with pytest.raises(ValueError, match="2048"):
        _outcome(detail="x" * 2049)


def test_attempt_and_event_serialization_are_strict_and_instant_aware() -> None:
    refs = ExternalRecordRefs(hypothesis_id="hyp_0123456789ab")
    attempt = ExperimentAttempt(
        ATTEMPT_A,
        _spec(),
        _repeated_hour(fold=1),
        references=refs,
    )
    running = ExperimentEvent(
        EVENT_A,
        ATTEMPT_A,
        ExperimentState.RUNNING,
        _repeated_hour(fold=1),
    )

    restored_attempt = ExperimentAttempt.from_json(attempt.to_json())
    restored_event = ExperimentEvent.from_json(running.to_json())
    assert restored_attempt == attempt
    assert hash(restored_attempt) == hash(attempt)
    assert restored_event == running
    assert restored_attempt.to_json() == attempt.to_json()
    assert restored_event.to_json() == running.to_json()

    invalid = running.to_dict()
    invalid["schema_version"] = 1.0
    with pytest.raises(ValueError, match="expected integer 1"):
        ExperimentEvent.from_dict(invalid)


def test_attempt_ids_are_immutable_and_invalid_ids_fail_closed() -> None:
    attempt = ExperimentAttempt(ATTEMPT_A, _spec(), _dt(1))
    with pytest.raises(FrozenInstanceError):
        attempt.attempt_id = ATTEMPT_B  # type: ignore[misc]
    with pytest.raises(ValueError, match="UUID-backed"):
        ExperimentAttempt("experiment-1", _spec(), _dt(1))
    with pytest.raises(ValueError, match="parent itself"):
        ExperimentAttempt(
            ATTEMPT_A,
            _spec(),
            _dt(1),
            parent_attempt_id=ATTEMPT_A,
        )


def test_registration_rejects_result_references() -> None:
    refs = ExternalRecordRefs(run_card_ref="run-card:already-known")
    with pytest.raises(ValueError, match="result artifact"):
        ExperimentAttempt(ATTEMPT_A, _spec(), _dt(1), references=refs)


def test_repeated_registration_has_distinct_attempt_identity_not_new_science(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    first = ledger.register(_spec(), registered_at=_dt(1))
    second = ledger.register(_spec(), registered_at=_dt(2))

    assert first.attempt_id != second.attempt_id
    assert first.spec_fingerprint == second.spec_fingerprint
    assert first.spec_fingerprint == _spec().fingerprint
    assert ledger.count_attempts(spec_fingerprint=first.spec_fingerprint) == 2


def test_registration_time_never_changes_spec_fingerprint(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    morning = ledger.register(_spec(), registered_at=_dt(1, 9))
    evening = ledger.register(_spec(), registered_at=_dt(20, 17))
    assert morning.spec_fingerprint == evening.spec_fingerprint


def test_child_attempt_links_to_parent_without_mutating_parent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    parent = ledger.register(_spec(), attempt_id=ATTEMPT_A, registered_at=_dt(1))
    parent_json = parent.to_json()
    changed_spec = replace(_spec(), horizon_bars=10)
    child = ledger.register(
        changed_spec,
        attempt_id=ATTEMPT_B,
        parent_attempt_id=parent.attempt_id,
        registered_at=_dt(2),
    )

    assert child.parent_attempt_id == parent.attempt_id
    assert child.spec_fingerprint != parent.spec_fingerprint
    assert ledger.get(parent.attempt_id).attempt.to_json() == parent_json
    assert ledger.get(parent.attempt_id).attempt.parent_attempt_id is None


def test_registration_rejects_self_parent_and_unknown_parent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="parent itself"):
        ledger.register(_spec(), attempt_id=ATTEMPT_A, parent_attempt_id=ATTEMPT_A)
    with pytest.raises(KeyError, match="parent experiment not found"):
        ledger.register(_spec(), attempt_id=ATTEMPT_A, parent_attempt_id=ATTEMPT_B)


def test_valid_lifecycle_is_registered_running_completed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(
        _spec(),
        references=ExternalRecordRefs(hypothesis_id="hyp_0123456789ab"),
        registered_at=_dt(1),
    )
    running = ledger.transition(
        attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2)
    )
    completed = ledger.transition(
        attempt.attempt_id,
        ExperimentState.COMPLETED,
        outcome=_outcome(),
        occurred_at=_dt(3),
    )
    record = ledger.get(attempt.attempt_id)

    assert record.state is ExperimentState.COMPLETED
    assert record.events == (running, completed)
    assert record.outcome == completed.outcome
    assert ledger.history(attempt.attempt_id) == (attempt, running, completed)


@pytest.mark.parametrize(
    "terminal",
    [
        ExperimentState.FAILED,
        ExperimentState.INTERRUPTED,
        ExperimentState.REJECTED,
    ],
)
def test_non_successful_attempts_remain_queryable_and_counted(
    tmp_path: Path, terminal: ExperimentState
) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    ledger.transition(
        attempt.attempt_id,
        terminal,
        outcome=ExperimentOutcome(detail=f"Ended as {terminal.value}."),
        occurred_at=_dt(2),
    )

    assert ledger.get(attempt.attempt_id).state is terminal
    assert ledger.count_attempts(trial_family_id=_spec().trial_family_id) == 1


def test_trial_family_accounting_includes_every_terminal_outcome(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    states = (
        ExperimentState.FAILED,
        ExperimentState.INTERRUPTED,
        ExperimentState.REJECTED,
    )
    for index, state in enumerate(states, start=1):
        attempt = ledger.register(_spec(), registered_at=_dt(index))
        ledger.transition(
            attempt.attempt_id,
            state,
            outcome=ExperimentOutcome(detail=state.value),
            occurred_at=_dt(index, 1),
        )
    other = ledger.register(
        _spec(trial_family_id="family:other"), registered_at=_dt(10)
    )

    assert ledger.count_attempts(trial_family_id="family:momentum-vs-trend") == 3
    assert ledger.count_attempts() == 4
    assert other in ledger.attempts(trial_family_id="family:other")


def test_result_or_status_requires_a_registered_attempt(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(KeyError, match="experiment not found"):
        ledger.transition(ATTEMPT_A, ExperimentState.RUNNING, occurred_at=_dt(1))


def test_invalid_transitions_and_terminal_rewrites_fail_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    with pytest.raises(InvalidExperimentTransition, match="registered to completed"):
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.COMPLETED,
            outcome=_outcome(),
            occurred_at=_dt(2),
        )

    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
    with pytest.raises(InvalidExperimentTransition, match="running to running"):
        ledger.transition(
            attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(3)
        )
    ledger.transition(
        attempt.attempt_id,
        ExperimentState.COMPLETED,
        outcome=_outcome(),
        occurred_at=_dt(3),
    )
    before = len(ledger.history())
    with pytest.raises(InvalidExperimentTransition, match="completed to failed"):
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.FAILED,
            outcome=ExperimentOutcome(detail="conflicting rewrite"),
            occurred_at=_dt(4),
        )
    with pytest.raises(InvalidExperimentTransition, match="completed to completed"):
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.COMPLETED,
            outcome=ExperimentOutcome(detail="conflicting second result"),
            occurred_at=_dt(4),
        )
    with pytest.raises(InvalidExperimentTransition, match="completed to running"):
        ledger.transition(
            attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(4)
        )
    assert len(ledger.history()) == before


def test_null_terminal_outcome_remains_an_explicit_counted_attempt(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
    ledger.transition(
        attempt.attempt_id,
        ExperimentState.COMPLETED,
        outcome=ExperimentOutcome(),
        occurred_at=_dt(3),
    )

    record = ledger.get(attempt.attempt_id)
    assert record.state is ExperimentState.COMPLETED
    assert record.outcome == ExperimentOutcome()
    assert ledger.count_attempts(trial_family_id=_spec().trial_family_id) == 1


def test_event_shape_and_chronology_fail_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(_spec(), registered_at=_repeated_hour(fold=0))
    ledger.transition(
        attempt.attempt_id,
        ExperimentState.RUNNING,
        occurred_at=_repeated_hour(fold=1),
    )

    later_registration = ledger.register(_spec(), registered_at=_repeated_hour(fold=1))
    with pytest.raises(ValueError, match="cannot precede"):
        ledger.transition(
            later_registration.attempt_id,
            ExperimentState.RUNNING,
            occurred_at=_repeated_hour(fold=0),
        )
    with pytest.raises(ValueError, match="terminal event requires"):
        ExperimentEvent(
            EVENT_A,
            ATTEMPT_A,
            ExperimentState.FAILED,
            _dt(2),
        )
    with pytest.raises(ValueError, match="running event cannot"):
        ExperimentEvent(
            EVENT_A,
            ATTEMPT_A,
            ExperimentState.RUNNING,
            _dt(2),
            outcome=_outcome(),
        )
    with pytest.raises(ValueError, match="recorded by ExperimentAttempt"):
        ExperimentEvent(
            EVENT_A,
            ATTEMPT_A,
            ExperimentState.REGISTERED,
            _dt(2),
        )


def test_naive_attempt_and_event_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ExperimentAttempt(ATTEMPT_A, _spec(), datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        ExperimentEvent(
            EVENT_A,
            ATTEMPT_A,
            ExperimentState.RUNNING,
            datetime(2026, 1, 2),
        )


def test_outcome_cannot_replace_registered_hypothesis_reference(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(
        _spec(),
        references=ExternalRecordRefs(hypothesis_id="hyp_original"),
        registered_at=_dt(1),
    )
    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
    conflicting = _outcome(
        references=ExternalRecordRefs(hypothesis_id="hyp_replacement")
    )
    with pytest.raises(ValueError, match="cannot replace"):
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.COMPLETED,
            outcome=conflicting,
            occurred_at=_dt(3),
        )


def test_history_is_append_only_and_reopens_identically(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    ledger = ExperimentLedger(path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    first_bytes = path.read_bytes()
    first_history = ledger.history()

    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
    assert path.read_bytes().startswith(first_bytes)
    assert ledger.history()[0] == first_history[0]

    reopened = ExperimentLedger(path)
    assert reopened.history() == ledger.history()
    assert reopened.get(attempt.attempt_id) == ledger.get(attempt.attempt_id)

    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["seq"] == 1
    assert raw["prev_record_hash"] == "sha256:genesis"
    assert raw["record_hash"].startswith("sha256:")


def test_callers_cannot_mutate_returned_ledger_history(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
    caller_metadata = {"nested": {"value": 1}}
    outcome = _outcome(metadata=caller_metadata)
    ledger.transition(
        attempt.attempt_id,
        ExperimentState.COMPLETED,
        outcome=outcome,
        occurred_at=_dt(3),
    )
    caller_metadata["nested"]["value"] = 999  # type: ignore[index]

    record = ledger.get(attempt.attempt_id)
    assert record.outcome is not None
    assert record.outcome.metadata["nested"]["value"] == 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.state = ExperimentState.FAILED  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.outcome.metadata["new"] = True  # type: ignore[index]


def test_derived_record_rejects_invented_or_inconsistent_history() -> None:
    attempt = ExperimentAttempt(ATTEMPT_A, _spec(), _dt(1))
    completed = ExperimentEvent(
        EVENT_A,
        ATTEMPT_A,
        ExperimentState.COMPLETED,
        _dt(2),
        outcome=ExperimentOutcome(),
    )

    with pytest.raises(InvalidExperimentTransition, match="registered to completed"):
        ExperimentRecord(attempt, ExperimentState.COMPLETED, (completed,))
    with pytest.raises(ValueError, match="state must equal"):
        ExperimentRecord(attempt, ExperimentState.RUNNING, ())


def test_corrupted_or_truncated_records_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    ledger = ExperimentLedger(path)
    ledger.register(_spec(), registered_at=_dt(1))
    with path.open("ab") as handle:
        handle.write(b'{"contract":"experiment_event"')

    with pytest.raises(ExperimentLedgerCorruptionError, match="corrupt"):
        ledger.history()
    with pytest.raises(ExperimentLedgerCorruptionError):
        ledger.register(_spec(), registered_at=_dt(2))


def test_valid_hash_chain_with_invalid_domain_history_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments.jsonl"
    ledger = ExperimentLedger(path)
    attempt = ledger.register(_spec(), registered_at=_dt(1))
    skipped_running = ExperimentEvent(
        EVENT_A,
        attempt.attempt_id,
        ExperimentState.COMPLETED,
        _dt(2),
        outcome=ExperimentOutcome(),
    )
    append_record(path, skipped_running.to_dict())

    with pytest.raises(
        ExperimentLedgerCorruptionError, match="invalid experiment history"
    ):
        ledger.history()


def test_duplicate_attempt_and_event_ids_never_overwrite(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    first = ledger.register(_spec(), attempt_id=ATTEMPT_A, registered_at=_dt(1))
    with pytest.raises(ValueError, match="duplicate attempt_id"):
        ledger.register(_spec(), attempt_id=ATTEMPT_A, registered_at=_dt(2))

    second = ledger.register(_spec(), attempt_id=ATTEMPT_B, registered_at=_dt(1))
    ledger.transition(
        first.attempt_id,
        ExperimentState.RUNNING,
        event_id=EVENT_A,
        occurred_at=_dt(2),
    )
    with pytest.raises(ValueError, match="duplicate event_id"):
        ledger.transition(
            second.attempt_id,
            ExperimentState.RUNNING,
            event_id=EVENT_A,
            occurred_at=_dt(2),
        )
    assert len(ledger.history()) == 3


def test_two_instances_concurrently_preserve_all_registration_and_event_appends(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments.jsonl"
    spec = _spec()

    def register_one(index: int) -> str:
        attempt = ExperimentLedger(path).register(
            spec, registered_at=_dt(1, minute=index)
        )
        return attempt.attempt_id

    with ThreadPoolExecutor(max_workers=6) as pool:
        attempt_ids = list(pool.map(register_one, range(12)))

    assert len(set(attempt_ids)) == 12
    assert ExperimentLedger(path).count_attempts() == 12

    def finish_one(attempt_id: str) -> None:
        instance = ExperimentLedger(path)
        instance.transition(attempt_id, ExperimentState.RUNNING, occurred_at=_dt(2))
        instance.transition(
            attempt_id,
            ExperimentState.COMPLETED,
            outcome=ExperimentOutcome(detail="done"),
            occurred_at=_dt(3),
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(finish_one, attempt_ids))

    reopened = ExperimentLedger(path)
    assert len(reopened.history()) == 36
    assert all(
        reopened.get(attempt_id).state is ExperimentState.COMPLETED
        for attempt_id in attempt_ids
    )


def test_concurrent_duplicate_attempt_id_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"

    def try_register() -> str:
        try:
            ExperimentLedger(path).register(
                _spec(), attempt_id=ATTEMPT_A, registered_at=_dt(1)
            )
        except ValueError:
            return "duplicate"
        return "registered"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: try_register(), range(2)))

    assert sorted(outcomes) == ["duplicate", "registered"]
    assert ExperimentLedger(path).count_attempts() == 1


def test_concurrent_state_transition_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    attempt = ExperimentLedger(path).register(_spec(), registered_at=_dt(1))

    def try_start() -> str:
        try:
            ExperimentLedger(path).transition(
                attempt.attempt_id,
                ExperimentState.RUNNING,
                occurred_at=_dt(2),
            )
        except InvalidExperimentTransition:
            return "conflict"
        return "running"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: try_start(), range(2)))

    assert sorted(outcomes) == ["conflict", "running"]
    assert len(ExperimentLedger(path).history(attempt.attempt_id)) == 2
