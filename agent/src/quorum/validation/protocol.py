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
from dataclasses import dataclass, field
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
    bar_intervals: tuple[TimeInterval, ...] = field(repr=False)
    label_end_positions: tuple[int, ...] = field(repr=False)
    embargo_size: int
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    purge_positions: tuple[int, ...]
    embargo_positions: tuple[int, ...]
    leakage_audit: LeakageAudit = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SplitManifest):
            raise TypeError("manifest must be a SplitManifest")
        bars = _bar_axis(self.bar_intervals)
        _reject_holdout_straddles(bars, self.manifest.final_holdout)
        ordinary_stop = _ordinary_stop(bars, self.manifest.final_holdout)
        label_ends = _label_ends(
            self.label_end_positions,
            len(bars),
            expected_count=ordinary_stop,
        )
        embargo_size = _position("embargo_size", self.embargo_size)
        object.__setattr__(self, "bar_intervals", bars)
        object.__setattr__(self, "label_end_positions", label_ends)
        object.__setattr__(self, "embargo_size", embargo_size)

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

        for name in (
            "train_positions",
            "validation_positions",
            "test_positions",
            "purge_positions",
            "embargo_positions",
        ):
            positions = getattr(self, name)
            if positions and positions[-1] >= ordinary_stop:
                raise ValueError(f"{name} must remain before the locked final holdout")

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

        expected_validation = tuple(
            range(
                self.validation_positions[0],
                self.validation_positions[0] + len(self.validation_positions),
            )
        )
        expected_test = tuple(
            range(
                self.validation_positions[-1] + 1,
                self.validation_positions[-1] + 1 + len(self.test_positions),
            )
        )
        if self.validation_positions != expected_validation:
            raise ValueError("validation positions must be contiguous")
        if self.test_positions != expected_test:
            raise ValueError("test positions must be contiguous and follow validation")

        for position in self.validation_positions + self.test_positions:
            if label_ends[position] >= ordinary_stop:
                raise ChronologicalValidationError(
                    "ordinary evaluation labels must resolve before the locked "
                    "final holdout"
                )

        expected_embargo = tuple(
            range(
                self.test_positions[-1] + 1,
                min(
                    self.test_positions[-1] + 1 + embargo_size,
                    ordinary_stop,
                ),
            )
        )
        if self.embargo_positions != expected_embargo:
            raise ValueError(
                "embargo_positions must exactly match the declared post-test embargo"
            )

        position_fields = {
            "train_intervals": self.train_positions,
            "validation_intervals": self.validation_positions,
            "test_intervals": self.test_positions,
            "purge_intervals": self.purge_positions,
            "embargo_intervals": self.embargo_positions,
        }
        for manifest_name, positions in position_fields.items():
            expected_intervals = _positions_to_intervals(positions, bars)
            if getattr(self.manifest, manifest_name) != expected_intervals:
                raise ChronologicalValidationError(
                    f"{manifest_name} does not match its authoritative positions"
                )

        held_out = self.validation_positions + self.test_positions
        audit = require_clean_boundary(
            Split(
                train=np.asarray(self.train_positions, dtype=int),
                test=np.asarray(held_out, dtype=int),
                purged=len(self.purge_positions),
                embargoed=len(self.embargo_positions),
                test_bounds=(held_out[0], held_out[-1]),
            ),
            label_ends,
            n_samples=ordinary_stop,
            embargo_size=embargo_size,
        )
        object.__setattr__(self, "leakage_audit", audit)


@dataclass(frozen=True, slots=True)
class _CanonicalFold:
    """Internal deterministic fold blueprint derived from frozen science inputs."""

    fold_index: int
    split_id: str
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    purge_positions: tuple[int, ...]
    embargo_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalDerivation:
    """Normalized ordinary inputs and their only accepted fold sequence."""

    bar_intervals: tuple[TimeInterval, ...]
    label_end_positions: tuple[int, ...]
    ordinary_stop: int
    folds: tuple[_CanonicalFold, ...]


@dataclass(frozen=True, slots=True)
class ChronologicalEvaluationPlan:
    """Canonical fold realization for one immutable experiment registration."""

    attempt: ExperimentAttempt
    protocol: EvaluationProtocol
    bar_intervals: tuple[TimeInterval, ...] = field(repr=False)
    label_end_positions: tuple[int, ...] = field(repr=False)
    folds: tuple[MaterializedFold, ...]
    oof_slots: tuple[OOFSlot, ...]

    def __post_init__(self) -> None:
        derivation = _derive_canonical_folds(
            self.attempt,
            self.protocol,
            self.bar_intervals,
            self.label_end_positions,
        )
        object.__setattr__(self, "bar_intervals", derivation.bar_intervals)
        object.__setattr__(self, "label_end_positions", derivation.label_end_positions)

        folds = tuple(self.folds)
        if any(not isinstance(fold, MaterializedFold) for fold in folds):
            raise TypeError("folds must contain only MaterializedFold values")
        if len(folds) != len(derivation.folds):
            raise ChronologicalValidationError(
                "folds must exactly match the canonical accepted-fold sequence"
            )
        object.__setattr__(self, "folds", folds)

        slots = tuple(self.oof_slots)
        if any(not isinstance(slot, OOFSlot) for slot in slots):
            raise TypeError("oof_slots must contain only OOFSlot values")
        object.__setattr__(self, "oof_slots", slots)

        assigned_evaluation_positions: set[int] = set()
        previous_evaluation_start: int | None = None
        for fold, expected in zip(folds, derivation.folds):
            if fold.bar_intervals != derivation.bar_intervals:
                raise ChronologicalValidationError(
                    "every fold must reference the canonical plan bar axis"
                )
            if fold.label_end_positions != derivation.label_end_positions:
                raise ChronologicalValidationError(
                    "every fold must reference the canonical plan label metadata"
                )
            if fold.embargo_size != self.protocol.embargo_bars:
                raise ChronologicalValidationError(
                    "every fold must use the protocol embargo_bars"
                )

            for name in (
                "train_positions",
                "validation_positions",
                "test_positions",
                "purge_positions",
                "embargo_positions",
            ):
                if getattr(fold, name) != getattr(expected, name):
                    raise ChronologicalValidationError(
                        f"{name} must exactly match the canonical fold derivation"
                    )

            expected_manifest = _manifest_for_canonical_fold(
                expected,
                self.protocol,
                derivation.bar_intervals,
            )
            if fold.manifest != expected_manifest:
                raise ChronologicalValidationError(
                    "fold manifest must exactly match its canonical derivation"
                )

            evaluation_start = expected.validation_positions[0]
            if (
                previous_evaluation_start is not None
                and evaluation_start <= previous_evaluation_start
            ):
                raise ChronologicalValidationError(
                    "canonical folds must be in chronological evaluation order"
                )
            previous_evaluation_start = evaluation_start
            evaluation_positions = set(
                expected.validation_positions + expected.test_positions
            )
            if assigned_evaluation_positions & evaluation_positions:
                raise ChronologicalValidationError(
                    "canonical ordinary evaluation positions must not overlap"
                )
            assigned_evaluation_positions.update(evaluation_positions)

        sample_positions = tuple(slot.sample_position for slot in slots)
        if len(sample_positions) != len(set(sample_positions)):
            raise ValueError("an OOF sample position cannot be assigned more than once")
        expected_slots = _oof_slots_for_canonical_folds(derivation.folds)
        if slots != expected_slots:
            raise ChronologicalValidationError(
                "oof_slots must exactly match the canonical fold assignments"
            )

    @property
    def attempt_id(self) -> str:
        """Return the ID of the retained immutable experiment registration."""
        return self.attempt.attempt_id

    @property
    def spec_fingerprint(self) -> str:
        """Return the scientific identity derived from the retained attempt."""
        return self.attempt.spec_fingerprint

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


def _label_ends(
    values: Any,
    n_samples: int,
    *,
    expected_count: int | None = None,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("label_end_positions must be a sequence of integers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("label_end_positions must be a sequence of integers") from exc
    count = n_samples if expected_count is None else expected_count
    if len(raw) != count:
        raise ValueError(
            f"label_end_positions has {len(raw)} entries but expected {count}"
        )
    ends = tuple(_position("label_end_positions", value) for value in raw)
    for start, end in enumerate(ends):
        if end < start:
            raise ValueError("a label cannot end before its observation position")
        if end >= n_samples:
            raise ValueError(
                "label_end_positions must reference a bar inside the known timeline"
            )
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
        "ordinary_bar_axis": [
            [_instant(bar.start).isoformat(), _instant(bar.end).isoformat()]
            for bar in bars
        ],
        "ordinary_label_end_positions": list(label_ends),
        "fold_index": fold_index,
        "train_positions": list(train),
        "validation_positions": list(validation),
        "test_positions": list(test),
        "purge_positions": list(purge),
        "embargo_positions": list(embargo),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"split_{fold_index:04d}_{digest}"


def _derive_canonical_folds(
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    bar_intervals: Sequence[TimeInterval],
    ordinary_label_end_positions: Sequence[int],
) -> _CanonicalDerivation:
    """Derive the only accepted ordinary fold sequence for frozen inputs."""
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
    _reject_holdout_straddles(bars, protocol.final_holdout)
    ordinary_stop = _ordinary_stop(bars, protocol.final_holdout)
    label_ends = _label_ends(
        ordinary_label_end_positions,
        len(bars),
        expected_count=ordinary_stop,
    )

    evaluation_width = protocol.validation_bars + protocol.test_bars
    if protocol.step_bars < evaluation_width:
        raise ChronologicalValidationError(
            "step_bars must be at least validation_bars + test_bars so ordinary "
            "evaluation assignments do not overlap"
        )

    history_target = (
        protocol.minimum_train_bars
        if protocol.mode is EvaluationMode.EXPANDING
        else protocol.train_window_bars
    )
    assert history_target is not None
    first_evaluation = history_target + protocol.purge_bars
    last_evaluation = ordinary_stop - evaluation_width
    if first_evaluation > last_evaluation:
        raise InsufficientHistoryError(
            "no room for declared training, purge, validation, and test bars "
            "before the locked final holdout"
        )

    ordinary_bars = bars[:ordinary_stop]
    canonical_folds: list[_CanonicalFold] = []
    skipped_for_history = 0
    skipped_for_holdout_labels = 0
    for evaluation_start in range(
        first_evaluation,
        last_evaluation + 1,
        protocol.step_bars,
    ):
        validation = tuple(
            range(evaluation_start, evaluation_start + protocol.validation_bars)
        )
        test_start = evaluation_start + protocol.validation_bars
        test_end = test_start + protocol.test_bars
        test = tuple(range(test_start, test_end))
        held_out = validation + test
        if any(label_ends[position] >= ordinary_stop for position in held_out):
            skipped_for_holdout_labels += 1
            continue

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
        embargo = tuple(
            range(
                test_end,
                min(test_end + protocol.embargo_bars, ordinary_stop),
            )
        )
        fold_index = len(canonical_folds)
        split_id = _split_id(
            fold_index=fold_index,
            attempt=attempt,
            protocol=protocol,
            bars=ordinary_bars,
            label_ends=label_ends,
            train=train,
            validation=validation,
            test=test,
            purge=purge,
            embargo=embargo,
        )
        canonical_folds.append(
            _CanonicalFold(
                fold_index=fold_index,
                split_id=split_id,
                train_positions=train,
                validation_positions=validation,
                test_positions=test,
                purge_positions=purge,
                embargo_positions=embargo,
            )
        )

    if not canonical_folds:
        raise InsufficientHistoryError(
            "no valid chronological folds remain after purge and minimum-history "
            "or final-holdout label checks; "
            f"skipped_for_history={skipped_for_history}, "
            f"skipped_for_holdout_labels={skipped_for_holdout_labels}"
        )
    return _CanonicalDerivation(
        bar_intervals=bars,
        label_end_positions=label_ends,
        ordinary_stop=ordinary_stop,
        folds=tuple(canonical_folds),
    )


def _manifest_for_canonical_fold(
    fold: _CanonicalFold,
    protocol: EvaluationProtocol,
    bars: Sequence[TimeInterval],
) -> SplitManifest:
    """Materialize persisted temporal truth from a canonical positional fold."""
    return SplitManifest(
        protocol_id=protocol.protocol_id,
        split_id=fold.split_id,
        fold_index=fold.fold_index,
        train_intervals=_positions_to_intervals(fold.train_positions, bars),
        validation_intervals=_positions_to_intervals(fold.validation_positions, bars),
        test_intervals=_positions_to_intervals(fold.test_positions, bars),
        purge_intervals=_positions_to_intervals(fold.purge_positions, bars),
        embargo_intervals=_positions_to_intervals(fold.embargo_positions, bars),
        final_holdout=protocol.final_holdout,
        uses_final_holdout=False,
    )


def _materialize_canonical_fold(
    fold: _CanonicalFold,
    protocol: EvaluationProtocol,
    bars: tuple[TimeInterval, ...],
    label_ends: tuple[int, ...],
) -> MaterializedFold:
    """Create one audited public fold from its canonical blueprint."""
    return MaterializedFold(
        manifest=_manifest_for_canonical_fold(fold, protocol, bars),
        bar_intervals=bars,
        label_end_positions=label_ends,
        embargo_size=protocol.embargo_bars,
        train_positions=fold.train_positions,
        validation_positions=fold.validation_positions,
        test_positions=fold.test_positions,
        purge_positions=fold.purge_positions,
        embargo_positions=fold.embargo_positions,
    )


def _oof_slots_for_canonical_folds(
    folds: Sequence[_CanonicalFold],
) -> tuple[OOFSlot, ...]:
    """Derive the exact authorized validation/test slots from canonical folds."""
    slots: list[OOFSlot] = []
    for fold in folds:
        slots.extend(
            OOFSlot(position, fold.split_id, fold.fold_index, OOFRole.VALIDATION)
            for position in fold.validation_positions
        )
        slots.extend(
            OOFSlot(position, fold.split_id, fold.fold_index, OOFRole.TEST)
            for position in fold.test_positions
        )
    return tuple(slots)


def materialize_chronological_plan(
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    bar_intervals: Sequence[TimeInterval],
    label_end_positions: Sequence[int],
) -> ChronologicalEvaluationPlan:
    """Materialize deterministic leakage-audited ordinary chronological folds.

    ``bar_intervals`` is the full supplied timeline, while
    ``label_end_positions`` must contain exactly the pre-holdout ordinary prefix.
    Locked-holdout label metadata is neither accepted nor inspected.

    Early candidate origins that remain below ``minimum_train_bars`` after
    explicit and label-overlap purging are skipped. If no origin is valid, the
    function fails closed with :class:`InsufficientHistoryError`.
    """
    if not isinstance(protocol, EvaluationProtocol):
        raise TypeError("protocol must be an EvaluationProtocol")
    bars = _bar_axis(bar_intervals)
    _reject_holdout_straddles(bars, protocol.final_holdout)
    ordinary_stop = _ordinary_stop(bars, protocol.final_holdout)
    ordinary_label_ends = _label_ends(
        label_end_positions,
        len(bars),
        expected_count=ordinary_stop,
    )
    derivation = _derive_canonical_folds(
        attempt,
        protocol,
        bars,
        ordinary_label_ends,
    )
    folds = tuple(
        _materialize_canonical_fold(
            fold,
            protocol,
            derivation.bar_intervals,
            derivation.label_end_positions,
        )
        for fold in derivation.folds
    )
    return ChronologicalEvaluationPlan(
        attempt=attempt,
        protocol=protocol,
        bar_intervals=derivation.bar_intervals,
        label_end_positions=derivation.label_end_positions,
        folds=folds,
        oof_slots=_oof_slots_for_canonical_folds(derivation.folds),
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
