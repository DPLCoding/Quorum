"""Deterministic chronological fold materialization for Quorum.

This module consumes immutable experiment/protocol declarations plus an explicit
bar axis and positional label spans. It does not read market values, fit models,
produce predictions, or mutate experiment/holdout state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from typing import Any

import numpy as np

from src.quantlib.crossvalidation import Split, detect_boundary_leakage
from src.quorum.contracts import (
    EvaluationMode,
    EvaluationProtocol,
    FinalHoldout,
    FinalHoldoutState,
    SplitManifest,
    TimeInterval,
)
from src.quorum.experiments import ExperimentAttempt

_SPLIT_IDENTITY_NAMESPACE = "quorum-chronological-split-v1"
_SPLIT_ID_RE = re.compile(r"^split_[0-9]{4}_[0-9a-f]{64}$")
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^exp_[0-9a-f]{32}$")


def _instant(value: datetime) -> datetime:
    """Return a transient UTC comparison/identity key."""
    return value.astimezone(timezone.utc)


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    return value


def _position(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _position_tuple(
    name: str, values: Any, *, allow_empty: bool = True
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integer positions")
    result = tuple(_position(name, value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ChronologicalValidationError(ValueError):
    """Raised when an accepted chronological plan cannot be materialized."""


class InsufficientHistoryError(ChronologicalValidationError):
    """Raised when no fold satisfies the declared history requirements."""


class BoundaryLeakageError(ChronologicalValidationError):
    """Raised when the mandatory quantlib boundary audit is dirty."""


class OOFRole(str, Enum):
    """Role of one authorized held-out assignment."""

    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    """Immutable Quorum representation of a quantlib leakage report."""

    overlapping_positions: tuple[int, ...] = ()
    shared_positions: tuple[int, ...] = ()
    embargo_violation_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "overlapping_positions",
            "shared_positions",
            "embargo_violation_positions",
        ):
            object.__setattr__(self, name, _position_tuple(name, getattr(self, name)))

    @property
    def clean(self) -> bool:
        """Return whether no boundary contamination was detected."""
        return not (
            self.overlapping_positions
            or self.shared_positions
            or self.embargo_violation_positions
        )


def require_clean_boundary(
    split: Split,
    label_end_positions: Sequence[int],
    *,
    n_samples: int,
    embargo_size: int = 0,
) -> LeakageAudit:
    """Run the mandatory quantlib audit and reject a dirty boundary."""
    if not isinstance(split, Split):
        raise TypeError("split must be a quantlib Split")
    if split.test.size == 0:
        raise ValueError("split.test must not be empty")
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive")
    if isinstance(embargo_size, bool) or not isinstance(embargo_size, Integral):
        raise TypeError("embargo_size must be an integer")
    if int(embargo_size) < 0:
        raise ValueError("embargo_size must be non-negative")

    report = detect_boundary_leakage(
        split,
        label_end_positions,
        n_samples=int(n_samples),
        embargo_size=int(embargo_size),
    )
    audit = LeakageAudit(
        overlapping_positions=tuple(int(value) for value in report.overlapping),
        shared_positions=tuple(int(value) for value in report.shared),
        embargo_violation_positions=tuple(
            int(value) for value in report.embargo_violations
        ),
    )
    if not audit.clean:
        raise BoundaryLeakageError(
            "dirty chronological boundary: "
            f"overlapping={audit.overlapping_positions}, "
            f"shared={audit.shared_positions}, "
            f"embargo_violations={audit.embargo_violation_positions}"
        )
    return audit


@dataclass(frozen=True, slots=True)
class OOFSlot:
    """One authorized but unfilled out-of-fold evaluation assignment."""

    sample_position: int
    split_id: str
    fold_index: int
    role: OOFRole

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_position",
            _position("sample_position", self.sample_position),
        )
        split_id = _required_text("split_id", self.split_id)
        if not _SPLIT_ID_RE.fullmatch(split_id):
            raise ValueError("split_id must be a deterministic Quorum split identity")
        object.__setattr__(self, "split_id", split_id)
        object.__setattr__(self, "fold_index", _position("fold_index", self.fold_index))
        if not isinstance(self.role, OOFRole):
            raise TypeError("role must be an OOFRole")


@dataclass(frozen=True, slots=True)
class MaterializedFold:
    """Runtime positional view paired with its persisted ``SplitManifest``."""

    manifest: SplitManifest
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    purge_positions: tuple[int, ...]
    embargo_positions: tuple[int, ...]
    leakage_audit: LeakageAudit

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SplitManifest):
            raise TypeError("manifest must be a SplitManifest")
        for name in (
            "train_positions",
            "validation_positions",
            "test_positions",
            "purge_positions",
            "embargo_positions",
        ):
            object.__setattr__(
                self,
                name,
                _position_tuple(
                    name,
                    getattr(self, name),
                    allow_empty=name in {"purge_positions", "embargo_positions"},
                ),
            )
        if not isinstance(self.leakage_audit, LeakageAudit):
            raise TypeError("leakage_audit must be a LeakageAudit")
        if not self.leakage_audit.clean:
            raise BoundaryLeakageError("a dirty leakage audit cannot become a fold")

        train = set(self.train_positions)
        validation = set(self.validation_positions)
        test = set(self.test_positions)
        purge = set(self.purge_positions)
        embargo = set(self.embargo_positions)
        named = (
            ("train", train),
            ("validation", validation),
            ("test", test),
            ("purge", purge),
            ("embargo", embargo),
        )
        for index, (left_name, left) in enumerate(named):
            for right_name, right in named[index + 1 :]:
                if left & right:
                    raise ValueError(
                        f"{left_name} positions must not overlap {right_name}"
                    )
        first_evaluation = min(self.validation_positions[0], self.test_positions[0])
        if self.train_positions[-1] >= first_evaluation:
            raise ValueError("training positions must be strictly before evaluation")


@dataclass(frozen=True, slots=True)
class ChronologicalEvaluationPlan:
    """Immutable fold plan and authorized OOF slots for one registered attempt."""

    attempt_id: str
    spec_fingerprint: str
    protocol: EvaluationProtocol
    folds: tuple[MaterializedFold, ...]
    oof_slots: tuple[OOFSlot, ...]

    def __post_init__(self) -> None:
        attempt_id = _required_text("attempt_id", self.attempt_id)
        if not _ATTEMPT_ID_RE.fullmatch(attempt_id):
            raise ValueError("attempt_id must use the Task 2 UUID-backed format")
        object.__setattr__(self, "attempt_id", attempt_id)
        fingerprint = _required_text("spec_fingerprint", self.spec_fingerprint)
        if not _FINGERPRINT_RE.fullmatch(fingerprint):
            raise ValueError("spec_fingerprint must be a full SHA-256 identity")
        object.__setattr__(self, "spec_fingerprint", fingerprint)
        if not isinstance(self.protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        folds = tuple(self.folds)
        if not folds or any(not isinstance(fold, MaterializedFold) for fold in folds):
            raise ValueError("folds must contain at least one MaterializedFold")
        object.__setattr__(self, "folds", folds)
        slots = tuple(self.oof_slots)
        if any(not isinstance(slot, OOFSlot) for slot in slots):
            raise TypeError("oof_slots must contain only OOFSlot values")
        object.__setattr__(self, "oof_slots", slots)

        expected_indices = tuple(range(len(folds)))
        actual_indices = tuple(fold.manifest.fold_index for fold in folds)
        if actual_indices != expected_indices:
            raise ValueError("folds must use consecutive chronological fold indices")
        split_ids = tuple(fold.manifest.split_id for fold in folds)
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("fold split IDs must be unique")

        expected_slots: list[OOFSlot] = []
        for fold in folds:
            manifest = fold.manifest
            if manifest.protocol_id != self.protocol.protocol_id:
                raise ValueError("every fold must reference the plan protocol")
            if manifest.final_holdout != self.protocol.final_holdout:
                raise ValueError("every fold must retain the declared final holdout")
            if manifest.uses_final_holdout:
                raise ValueError(
                    "ordinary chronological folds cannot use final holdout"
                )
            if len(fold.train_positions) < self.protocol.minimum_train_bars:
                raise ValueError("fold training is below minimum_train_bars")
            if (
                self.protocol.mode is EvaluationMode.ROLLING
                and len(fold.train_positions) > self.protocol.train_window_bars
            ):
                raise ValueError("rolling fold exceeds train_window_bars")
            if len(fold.validation_positions) != self.protocol.validation_bars:
                raise ValueError("fold validation size does not match protocol")
            if len(fold.test_positions) != self.protocol.test_bars:
                raise ValueError("fold test size does not match protocol")
            expected_slots.extend(
                OOFSlot(
                    position, manifest.split_id, manifest.fold_index, OOFRole.VALIDATION
                )
                for position in fold.validation_positions
            )
            expected_slots.extend(
                OOFSlot(position, manifest.split_id, manifest.fold_index, OOFRole.TEST)
                for position in fold.test_positions
            )

        sample_positions = tuple(slot.sample_position for slot in slots)
        if len(sample_positions) != len(set(sample_positions)):
            raise ValueError("an OOF sample position cannot be assigned more than once")
        if slots != tuple(expected_slots):
            raise ValueError(
                "oof_slots must exactly match all fold evaluation positions"
            )

    @property
    def final_holdout(self) -> FinalHoldout:
        """Return the still-locked final holdout boundary."""
        return self.protocol.final_holdout

    @property
    def validation_slots(self) -> tuple[OOFSlot, ...]:
        """Return validation assignments without calling them test rows."""
        return tuple(slot for slot in self.oof_slots if slot.role is OOFRole.VALIDATION)

    @property
    def test_slots(self) -> tuple[OOFSlot, ...]:
        """Return ordinary OOF test assignments."""
        return tuple(slot for slot in self.oof_slots if slot.role is OOFRole.TEST)


def _bar_axis(values: Any) -> tuple[TimeInterval, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("bar_intervals must be a sequence of TimeInterval values")
    bars = tuple(values)
    if not bars:
        raise ValueError("bar_intervals must not be empty")
    if any(not isinstance(bar, TimeInterval) for bar in bars):
        raise TypeError("bar_intervals must contain only TimeInterval values")
    for previous, current in zip(bars, bars[1:]):
        if _instant(previous.start) >= _instant(current.start):
            raise ValueError("bar_intervals must be strictly ordered by actual instant")
        if _instant(previous.end) > _instant(current.start):
            raise ValueError("bar_intervals must not overlap")
    return bars


def _label_ends(values: Any, n_samples: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("label_end_positions must be a sequence of integers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("label_end_positions must be a sequence of integers") from exc
    if len(raw) != n_samples:
        raise ValueError(
            f"label_end_positions has {len(raw)} entries but timeline has {n_samples}"
        )
    ends = tuple(_position("label_end_positions", value) for value in raw)
    for start, end in enumerate(ends):
        if end < start:
            raise ValueError("a label cannot end before its observation position")
    return ends


def _reject_holdout_straddles(
    bars: Sequence[TimeInterval], final_holdout: FinalHoldout
) -> None:
    boundaries = (_instant(final_holdout.start), _instant(final_holdout.end))
    for position, bar in enumerate(bars):
        start, end = _instant(bar.start), _instant(bar.end)
        if any(start < boundary < end for boundary in boundaries):
            raise ChronologicalValidationError(
                f"bar position {position} straddles a final-holdout boundary"
            )


def _ordinary_stop(bars: Sequence[TimeInterval], final_holdout: FinalHoldout) -> int:
    holdout_start = _instant(final_holdout.start)
    stop = 0
    for position, bar in enumerate(bars):
        if _instant(bar.end) <= holdout_start:
            stop = position + 1
        else:
            break
    return stop


def _positions_to_intervals(
    positions: Sequence[int], bars: Sequence[TimeInterval]
) -> tuple[TimeInterval, ...]:
    if not positions:
        return ()
    result: list[TimeInterval] = []
    run_start = positions[0]
    run_end = positions[0]
    for position in positions[1:]:
        touches = _instant(bars[run_end].end) == _instant(bars[position].start)
        if position == run_end + 1 and touches:
            run_end = position
            continue
        result.append(TimeInterval(bars[run_start].start, bars[run_end].end))
        run_start = run_end = position
    result.append(TimeInterval(bars[run_start].start, bars[run_end].end))
    return tuple(result)


def _protocol_identity(protocol: EvaluationProtocol) -> dict[str, Any]:
    holdout = protocol.final_holdout
    return {
        "protocol_id": protocol.protocol_id,
        "mode": protocol.mode.value,
        "minimum_train_bars": protocol.minimum_train_bars,
        "train_window_bars": protocol.train_window_bars,
        "validation_bars": protocol.validation_bars,
        "test_bars": protocol.test_bars,
        "step_bars": protocol.step_bars,
        "purge_bars": protocol.purge_bars,
        "embargo_bars": protocol.embargo_bars,
        "final_holdout": {
            "start": _instant(holdout.start).isoformat(),
            "end": _instant(holdout.end).isoformat(),
            "state": holdout.state.value,
        },
        "random_seed": protocol.random_seed,
    }


def _split_id(
    *,
    fold_index: int,
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    bars: Sequence[TimeInterval],
    label_ends: Sequence[int],
    train: Sequence[int],
    validation: Sequence[int],
    test: Sequence[int],
    purge: Sequence[int],
    embargo: Sequence[int],
) -> str:
    payload = {
        "identity_namespace": _SPLIT_IDENTITY_NAMESPACE,
        "spec_fingerprint": attempt.spec_fingerprint,
        "protocol": _protocol_identity(protocol),
        "bar_axis": [
            [_instant(bar.start).isoformat(), _instant(bar.end).isoformat()]
            for bar in bars
        ],
        "label_end_positions": list(label_ends),
        "fold_index": fold_index,
        "train_positions": list(train),
        "validation_positions": list(validation),
        "test_positions": list(test),
        "purge_positions": list(purge),
        "embargo_positions": list(embargo),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"split_{fold_index:04d}_{digest}"


def materialize_chronological_plan(
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    bar_intervals: Sequence[TimeInterval],
    label_end_positions: Sequence[int],
) -> ChronologicalEvaluationPlan:
    """Materialize deterministic leakage-audited ordinary chronological folds.

    Early candidate origins that remain below ``minimum_train_bars`` after
    explicit and label-overlap purging are skipped. If no origin is valid, the
    function fails closed with :class:`InsufficientHistoryError`.
    """
    if not isinstance(attempt, ExperimentAttempt):
        raise TypeError("attempt must be an ExperimentAttempt")
    if not isinstance(protocol, EvaluationProtocol):
        raise TypeError("protocol must be an EvaluationProtocol")
    if attempt.spec.evaluation_protocol_id != protocol.protocol_id:
        raise ChronologicalValidationError(
            "attempt evaluation_protocol_id does not match protocol.protocol_id"
        )
    if protocol.final_holdout.state is not FinalHoldoutState.LOCKED:
        raise ChronologicalValidationError(
            "ordinary evaluation requires locked holdout"
        )

    bars = _bar_axis(bar_intervals)
    label_ends = _label_ends(label_end_positions, len(bars))
    _reject_holdout_straddles(bars, protocol.final_holdout)

    evaluation_width = protocol.validation_bars + protocol.test_bars
    if protocol.step_bars < evaluation_width:
        raise ChronologicalValidationError(
            "step_bars must be at least validation_bars + test_bars so ordinary "
            "evaluation assignments do not overlap"
        )

    stop = _ordinary_stop(bars, protocol.final_holdout)
    history_target = (
        protocol.minimum_train_bars
        if protocol.mode is EvaluationMode.EXPANDING
        else protocol.train_window_bars
    )
    assert history_target is not None
    first_evaluation = history_target + protocol.purge_bars
    last_evaluation = stop - evaluation_width
    if first_evaluation > last_evaluation:
        raise InsufficientHistoryError(
            "no room for declared training, purge, validation, and test bars "
            "before the locked final holdout"
        )

    folds: list[MaterializedFold] = []
    slots: list[OOFSlot] = []
    skipped_for_history = 0
    for evaluation_start in range(
        first_evaluation, last_evaluation + 1, protocol.step_bars
    ):
        validation = tuple(
            range(evaluation_start, evaluation_start + protocol.validation_bars)
        )
        test_start = evaluation_start + protocol.validation_bars
        test_end = test_start + protocol.test_bars
        test = tuple(range(test_start, test_end))
        held_out = validation + test

        explicit_purge_start = evaluation_start - protocol.purge_bars
        explicit_purge = tuple(range(explicit_purge_start, evaluation_start))
        historical = tuple(range(explicit_purge_start))
        label_purge = tuple(
            position
            for position in historical
            if label_ends[position] >= evaluation_start
        )
        eligible = tuple(
            position
            for position in historical
            if label_ends[position] < evaluation_start
        )
        if protocol.mode is EvaluationMode.ROLLING:
            assert protocol.train_window_bars is not None
            eligible = eligible[-protocol.train_window_bars :]
        train = eligible
        if len(train) < protocol.minimum_train_bars:
            skipped_for_history += 1
            continue

        purge = tuple(sorted((*explicit_purge, *label_purge)))
        embargo = tuple(range(test_end, min(test_end + protocol.embargo_bars, stop)))
        quantlib_split = Split(
            train=np.asarray(train, dtype=int),
            test=np.asarray(held_out, dtype=int),
            purged=len(purge),
            embargoed=len(embargo),
            test_bounds=(held_out[0], held_out[-1]),
        )
        audit = require_clean_boundary(
            quantlib_split,
            label_ends,
            n_samples=len(bars),
            embargo_size=protocol.embargo_bars,
        )

        fold_index = len(folds)
        split_id = _split_id(
            fold_index=fold_index,
            attempt=attempt,
            protocol=protocol,
            bars=bars,
            label_ends=label_ends,
            train=train,
            validation=validation,
            test=test,
            purge=purge,
            embargo=embargo,
        )
        manifest = SplitManifest(
            protocol_id=protocol.protocol_id,
            split_id=split_id,
            fold_index=fold_index,
            train_intervals=_positions_to_intervals(train, bars),
            validation_intervals=_positions_to_intervals(validation, bars),
            test_intervals=_positions_to_intervals(test, bars),
            purge_intervals=_positions_to_intervals(purge, bars),
            embargo_intervals=_positions_to_intervals(embargo, bars),
            final_holdout=protocol.final_holdout,
            uses_final_holdout=False,
        )
        fold = MaterializedFold(
            manifest=manifest,
            train_positions=train,
            validation_positions=validation,
            test_positions=test,
            purge_positions=purge,
            embargo_positions=embargo,
            leakage_audit=audit,
        )
        folds.append(fold)
        slots.extend(
            OOFSlot(position, split_id, fold_index, OOFRole.VALIDATION)
            for position in validation
        )
        slots.extend(
            OOFSlot(position, split_id, fold_index, OOFRole.TEST) for position in test
        )

    if not folds:
        raise InsufficientHistoryError(
            "no valid chronological folds remain after purge and minimum-history "
            f"checks; skipped_origins={skipped_for_history}"
        )
    return ChronologicalEvaluationPlan(
        attempt_id=attempt.attempt_id,
        spec_fingerprint=attempt.spec_fingerprint,
        protocol=protocol,
        folds=tuple(folds),
        oof_slots=tuple(slots),
    )


__all__ = [
    "BoundaryLeakageError",
    "ChronologicalEvaluationPlan",
    "ChronologicalValidationError",
    "InsufficientHistoryError",
    "LeakageAudit",
    "MaterializedFold",
    "OOFRole",
    "OOFSlot",
    "materialize_chronological_plan",
    "require_clean_boundary",
]
