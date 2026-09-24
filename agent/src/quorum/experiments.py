"""Immutable experiment identity and append-only research ledger for Quorum.

The module records scientific specifications separately from individual attempts.
Specifications have deterministic SHA-256 identities; attempts and lifecycle
events have independently generated UUID-based identities. Durable storage reuses
the standard-library-only governance ledger for hash chaining, fsync, and atomic
appends, while a ledger-scoped lock keeps Quorum state validation and appending in
one cross-process critical section.

The application API is append-only. The local forward hash chain detects changes
within the retained history, including modification, interior deletion or
reordering, malformed or partial records, and hash/sequence discontinuities. It
cannot prove that complete trailing records were not cleanly removed without an
external trusted checkpoint or monotonic anchor.

This module does not run evaluations, load data, compute predictions, or import
backtest, agent, API, frontend, broker, expert, or ensemble implementations.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

try:  # POSIX advisory locking.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows advisory byte-range locking.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

from src.governance.ledger import (
    LedgerCorruptionError as GovernanceLedgerCorruptionError,
)
from src.governance.ledger import append_record, verify_chain
from src.quorum.contracts import (
    _aware_datetime,
    _canonical_json,
    _immutable_metadata,
    _instant,
    _integer,
    _json_mapping,
    _optional_text,
    _parse_datetime,
    _payload,
    _require_payload_keys,
    _required_text,
    _thaw_json_value,
)

_EXPERIMENT_SPEC_IDENTITY_NAMESPACE = "quorum-experiment-spec-v1"
_MAX_OUTCOME_DETAIL_BYTES = 2048
_ATTEMPT_ID_RE = re.compile(r"^exp_[0-9a-f]{32}$")
_EVENT_ID_RE = re.compile(r"^evt_[0-9a-f]{32}$")
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$")
_CHAIN_FIELDS = frozenset({"seq", "prev_record_hash", "record_hash"})


def _opaque_id(name: str, value: Any) -> str:
    """Validate a bounded opaque external or scientific identity."""
    validated = _required_text(name, value)
    if not _OPAQUE_ID_RE.fullmatch(validated):
        raise ValueError(f"{name} must be 1-256 path-independent identifier characters")
    return validated


def _optional_opaque_id(name: str, value: Any) -> str | None:
    """Validate an optional opaque identity."""
    if value is None:
        return None
    return _opaque_id(name, value)


def _outcome_detail(value: Any) -> str | None:
    """Validate a small human-readable outcome summary."""
    detail = _optional_text("detail", value)
    if detail is not None and len(detail.encode("utf-8")) > _MAX_OUTCOME_DETAIL_BYTES:
        raise ValueError(
            f"detail must not exceed {_MAX_OUTCOME_DETAIL_BYTES} UTF-8 bytes"
        )
    return detail


def _generated_id(prefix: str) -> str:
    """Return a UUID4-backed attempt or event identity."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _validated_generated_id(name: str, value: Any, pattern: re.Pattern[str]) -> str:
    """Validate one UUID-backed ledger identity."""
    validated = _required_text(name, value)
    if not pattern.fullmatch(validated):
        raise ValueError(f"{name} must use the expected UUID-backed format")
    return validated


def _fingerprint(value: Any) -> str:
    """Return a full SHA-256 identity over canonical JSON."""
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}"


def _canonical_ids(name: str, values: Any, *, allow_empty: bool) -> tuple[str, ...]:
    """Validate, de-duplicate, and sort an ID sequence."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers")
    identifiers = tuple(_opaque_id(name, value) for value in values)
    if not allow_empty and not identifiers:
        raise ValueError(f"{name} must not be empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} must not contain duplicate identifiers")
    return tuple(sorted(identifiers))


def _utc_now() -> datetime:
    """Return the current aware UTC time for non-scientific ledger metadata."""
    return datetime.now(timezone.utc)


class ExperimentState(str, Enum):
    """Explicit experiment lifecycle states.

    Registration is represented by :class:`ExperimentAttempt`. A running event
    must precede successful completion. Setup failures, interruptions, and
    scientific rejections may occur directly after registration. All terminal
    states are final.
    """

    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


_TERMINAL_STATES = frozenset(
    {
        ExperimentState.COMPLETED,
        ExperimentState.FAILED,
        ExperimentState.INTERRUPTED,
        ExperimentState.REJECTED,
    }
)
_ALLOWED_TRANSITIONS: Mapping[ExperimentState, frozenset[ExperimentState]] = {
    ExperimentState.REGISTERED: frozenset(
        {
            ExperimentState.RUNNING,
            ExperimentState.FAILED,
            ExperimentState.INTERRUPTED,
            ExperimentState.REJECTED,
        }
    ),
    ExperimentState.RUNNING: _TERMINAL_STATES,
}


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentSpec:
    """Frozen scientific choices that define one experiment specification.

    All fields are identities or small scalar declarations. Future components
    own their detailed payloads; this contract references those payloads rather
    than embedding mutable external records. ``data_cutoff_at`` is an optional
    experiment-level information boundary, not a prediction ``decision_at`` or
    a split definition.
    """

    evaluation_protocol_id: str
    expert_config_ids: tuple[str, ...]
    ensemble_config_id: str | None
    target_definition_id: str
    horizon_bars: int
    data_snapshot_id: str
    data_cutoff_at: datetime | None
    universe_id: str
    cost_model_id: str
    code_id: str
    config_id: str
    random_seed: int | None
    trial_family_id: str

    def __post_init__(self) -> None:
        for name in (
            "evaluation_protocol_id",
            "target_definition_id",
            "data_snapshot_id",
            "universe_id",
            "cost_model_id",
            "code_id",
            "config_id",
            "trial_family_id",
        ):
            object.__setattr__(self, name, _opaque_id(name, getattr(self, name)))
        object.__setattr__(
            self,
            "expert_config_ids",
            _canonical_ids(
                "expert_config_ids", self.expert_config_ids, allow_empty=False
            ),
        )
        object.__setattr__(
            self,
            "ensemble_config_id",
            _optional_opaque_id("ensemble_config_id", self.ensemble_config_id),
        )
        object.__setattr__(
            self,
            "horizon_bars",
            _integer("horizon_bars", self.horizon_bars, minimum=1),
        )
        if self.data_cutoff_at is not None:
            object.__setattr__(
                self,
                "data_cutoff_at",
                _aware_datetime("data_cutoff_at", self.data_cutoff_at),
            )
        if self.random_seed is not None:
            object.__setattr__(
                self,
                "random_seed",
                _integer(
                    "random_seed", self.random_seed, minimum=0, maximum=(2**32) - 1
                ),
            )

    def _scientific_identity_payload(self) -> dict[str, Any]:
        """Return the versioned scientific identity, not persistence metadata."""
        return {
            "identity_namespace": _EXPERIMENT_SPEC_IDENTITY_NAMESPACE,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "expert_config_ids": list(self.expert_config_ids),
            "ensemble_config_id": self.ensemble_config_id,
            "target_definition_id": self.target_definition_id,
            "horizon_bars": self.horizon_bars,
            "data_snapshot_id": self.data_snapshot_id,
            "data_cutoff_at": (
                _instant(self.data_cutoff_at).isoformat()
                if self.data_cutoff_at is not None
                else None
            ),
            "universe_id": self.universe_id,
            "cost_model_id": self.cost_model_id,
            "code_id": self.code_id,
            "config_id": self.config_id,
            "random_seed": self.random_seed,
            "trial_family_id": self.trial_family_id,
        }

    @property
    def fingerprint(self) -> str:
        """Full persisted scientific identity, independent of runtime metadata."""
        return _fingerprint(self._scientific_identity_payload())

    def __eq__(self, other: object) -> bool:
        """Compare scientific identity, normalizing timestamps by instant."""
        if not isinstance(other, ExperimentSpec):
            return NotImplemented
        return (
            self._scientific_identity_payload() == other._scientific_identity_payload()
        )

    def __hash__(self) -> int:
        """Hash consistently with scientific equality for in-memory use."""
        return hash(self.fingerprint)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "experiment_spec",
            evaluation_protocol_id=self.evaluation_protocol_id,
            expert_config_ids=list(self.expert_config_ids),
            ensemble_config_id=self.ensemble_config_id,
            target_definition_id=self.target_definition_id,
            horizon_bars=self.horizon_bars,
            data_snapshot_id=self.data_snapshot_id,
            data_cutoff_at=(
                self.data_cutoff_at.isoformat()
                if self.data_cutoff_at is not None
                else None
            ),
            universe_id=self.universe_id,
            cost_model_id=self.cost_model_id,
            code_id=self.code_id,
            config_id=self.config_id,
            random_seed=self.random_seed,
            trial_family_id=self.trial_family_id,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        """Reconstruct a specification from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="experiment_spec",
            fields=frozenset(
                {
                    "evaluation_protocol_id",
                    "expert_config_ids",
                    "ensemble_config_id",
                    "target_definition_id",
                    "horizon_bars",
                    "data_snapshot_id",
                    "data_cutoff_at",
                    "universe_id",
                    "cost_model_id",
                    "code_id",
                    "config_id",
                    "random_seed",
                    "trial_family_id",
                }
            ),
        )
        cutoff = data["data_cutoff_at"]
        return cls(
            evaluation_protocol_id=data["evaluation_protocol_id"],
            expert_config_ids=data["expert_config_ids"],
            ensemble_config_id=data["ensemble_config_id"],
            target_definition_id=data["target_definition_id"],
            horizon_bars=data["horizon_bars"],
            data_snapshot_id=data["data_snapshot_id"],
            data_cutoff_at=(
                _parse_datetime("data_cutoff_at", cutoff)
                if cutoff is not None
                else None
            ),
            universe_id=data["universe_id"],
            cost_model_id=data["cost_model_id"],
            code_id=data["code_id"],
            config_id=data["config_id"],
            random_seed=data["random_seed"],
            trial_family_id=data["trial_family_id"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExperimentSpec":
        """Reconstruct a specification from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "experiment_spec"))


@dataclass(frozen=True, slots=True)
class ExternalRecordRefs:
    """Opaque identities of existing records; no external payload is copied."""

    hypothesis_id: str | None = None
    strategy_artifact_id: str | None = None
    run_card_ref: str | None = None
    quorum_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "strategy_artifact_id", "run_card_ref"):
            object.__setattr__(
                self, name, _optional_opaque_id(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "quorum_artifact_ids",
            _canonical_ids(
                "quorum_artifact_ids", self.quorum_artifact_ids, allow_empty=True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "external_record_refs",
            hypothesis_id=self.hypothesis_id,
            strategy_artifact_id=self.strategy_artifact_id,
            run_card_ref=self.run_card_ref,
            quorum_artifact_ids=list(self.quorum_artifact_ids),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalRecordRefs":
        """Reconstruct references from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="external_record_refs",
            fields=frozenset(
                {
                    "hypothesis_id",
                    "strategy_artifact_id",
                    "run_card_ref",
                    "quorum_artifact_ids",
                }
            ),
        )
        return cls(
            hypothesis_id=data["hypothesis_id"],
            strategy_artifact_id=data["strategy_artifact_id"],
            run_card_ref=data["run_card_ref"],
            quorum_artifact_ids=data["quorum_artifact_ids"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExternalRecordRefs":
        """Reconstruct references from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "external_record_refs"))


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Small immutable terminal outcome and external artifact references."""

    detail: str | None = None
    references: ExternalRecordRefs = field(default_factory=ExternalRecordRefs)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", _outcome_detail(self.detail))
        if not isinstance(self.references, ExternalRecordRefs):
            raise TypeError("references must be ExternalRecordRefs")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible representation."""
        return _payload(
            "experiment_outcome",
            detail=self.detail,
            references=self.references.to_dict(),
            metadata=_thaw_json_value(self.metadata),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentOutcome":
        """Reconstruct an outcome from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="experiment_outcome",
            fields=frozenset({"detail", "references", "metadata"}),
        )
        return cls(
            detail=data["detail"],
            references=ExternalRecordRefs.from_dict(data["references"]),
            metadata=data["metadata"],
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExperimentOutcome":
        """Reconstruct an outcome from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "experiment_outcome"))


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentAttempt:
    """Immutable registration of one attempt of a frozen specification."""

    attempt_id: str
    spec: ExperimentSpec
    registered_at: datetime
    parent_attempt_id: str | None = None
    references: ExternalRecordRefs = field(default_factory=ExternalRecordRefs)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_id",
            _validated_generated_id("attempt_id", self.attempt_id, _ATTEMPT_ID_RE),
        )
        if not isinstance(self.spec, ExperimentSpec):
            raise TypeError("spec must be an ExperimentSpec")
        object.__setattr__(
            self,
            "registered_at",
            _aware_datetime("registered_at", self.registered_at),
        )
        parent = self.parent_attempt_id
        if parent is not None:
            parent = _validated_generated_id(
                "parent_attempt_id", parent, _ATTEMPT_ID_RE
            )
            if parent == self.attempt_id:
                raise ValueError("an experiment attempt cannot parent itself")
        object.__setattr__(self, "parent_attempt_id", parent)
        if not isinstance(self.references, ExternalRecordRefs):
            raise TypeError("references must be ExternalRecordRefs")
        if (
            self.references.run_card_ref is not None
            or self.references.quorum_artifact_ids
        ):
            raise ValueError(
                "registration references cannot include result artifact references"
            )

    @property
    def state(self) -> ExperimentState:
        """Registration establishes the initial lifecycle state."""
        return ExperimentState.REGISTERED

    @property
    def spec_fingerprint(self) -> str:
        """Return the deterministic identity of the referenced specification."""
        return self.spec.fingerprint

    def __eq__(self, other: object) -> bool:
        """Compare registration time by actual instant."""
        if not isinstance(other, ExperimentAttempt):
            return NotImplemented
        return (
            self.attempt_id,
            self.spec,
            _instant(self.registered_at),
            self.parent_attempt_id,
            self.references,
        ) == (
            other.attempt_id,
            other.spec,
            _instant(other.registered_at),
            other.parent_attempt_id,
            other.references,
        )

    def __hash__(self) -> int:
        """Hash consistently with instant-aware equality."""
        return hash(
            (
                self.attempt_id,
                self.spec,
                _instant(self.registered_at),
                self.parent_attempt_id,
                self.references,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible registration record."""
        return _payload(
            "experiment_attempt",
            attempt_id=self.attempt_id,
            spec_fingerprint=self.spec_fingerprint,
            spec=self.spec.to_dict(),
            state=self.state.value,
            registered_at=self.registered_at.isoformat(),
            parent_attempt_id=self.parent_attempt_id,
            references=self.references.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentAttempt":
        """Reconstruct a registration and verify its embedded fingerprint."""
        _require_payload_keys(
            data,
            contract_name="experiment_attempt",
            fields=frozenset(
                {
                    "attempt_id",
                    "spec_fingerprint",
                    "spec",
                    "state",
                    "registered_at",
                    "parent_attempt_id",
                    "references",
                }
            ),
        )
        try:
            state = ExperimentState(data["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid experiment state {data['state']!r}") from exc
        if state is not ExperimentState.REGISTERED:
            raise ValueError("an experiment attempt must have state=registered")
        claimed = data["spec_fingerprint"]
        if not isinstance(claimed, str) or not _FINGERPRINT_RE.fullmatch(claimed):
            raise ValueError("spec_fingerprint must be a full SHA-256 identity")
        spec = ExperimentSpec.from_dict(data["spec"])
        if claimed != spec.fingerprint:
            raise ValueError("spec_fingerprint does not match the embedded spec")
        return cls(
            attempt_id=data["attempt_id"],
            spec=spec,
            registered_at=_parse_datetime("registered_at", data["registered_at"]),
            parent_attempt_id=data["parent_attempt_id"],
            references=ExternalRecordRefs.from_dict(data["references"]),
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExperimentAttempt":
        """Reconstruct a registration from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "experiment_attempt"))


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentEvent:
    """One immutable lifecycle transition after experiment registration."""

    event_id: str
    attempt_id: str
    state: ExperimentState
    occurred_at: datetime
    outcome: ExperimentOutcome | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _validated_generated_id("event_id", self.event_id, _EVENT_ID_RE),
        )
        object.__setattr__(
            self,
            "attempt_id",
            _validated_generated_id("attempt_id", self.attempt_id, _ATTEMPT_ID_RE),
        )
        if not isinstance(self.state, ExperimentState):
            raise TypeError("state must be an ExperimentState")
        if self.state is ExperimentState.REGISTERED:
            raise ValueError("registration is recorded by ExperimentAttempt")
        object.__setattr__(
            self,
            "occurred_at",
            _aware_datetime("occurred_at", self.occurred_at),
        )
        if self.state is ExperimentState.RUNNING:
            if self.outcome is not None:
                raise ValueError("a running event cannot contain an outcome")
        elif self.state in _TERMINAL_STATES:
            if not isinstance(self.outcome, ExperimentOutcome):
                raise ValueError("a terminal event requires an ExperimentOutcome")
        else:  # pragma: no cover - the enum is exhaustively handled above.
            raise ValueError(f"unsupported experiment state {self.state.value!r}")

    def __eq__(self, other: object) -> bool:
        """Compare event time by actual instant."""
        if not isinstance(other, ExperimentEvent):
            return NotImplemented
        return (
            self.event_id,
            self.attempt_id,
            self.state,
            _instant(self.occurred_at),
            self.outcome,
        ) == (
            other.event_id,
            other.attempt_id,
            other.state,
            _instant(other.occurred_at),
            other.outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible event record."""
        return _payload(
            "experiment_event",
            event_id=self.event_id,
            attempt_id=self.attempt_id,
            state=self.state.value,
            occurred_at=self.occurred_at.isoformat(),
            outcome=self.outcome.to_dict() if self.outcome is not None else None,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentEvent":
        """Reconstruct an event from :meth:`to_dict` output."""
        _require_payload_keys(
            data,
            contract_name="experiment_event",
            fields=frozenset(
                {"event_id", "attempt_id", "state", "occurred_at", "outcome"}
            ),
        )
        try:
            state = ExperimentState(data["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid experiment state {data['state']!r}") from exc
        outcome = data["outcome"]
        return cls(
            event_id=data["event_id"],
            attempt_id=data["attempt_id"],
            state=state,
            occurred_at=_parse_datetime("occurred_at", data["occurred_at"]),
            outcome=(
                ExperimentOutcome.from_dict(outcome) if outcome is not None else None
            ),
        )

    def to_json(self) -> str:
        """Return deterministic JSON."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExperimentEvent":
        """Reconstruct an event from deterministic JSON."""
        return cls.from_dict(_json_mapping(text, "experiment_event"))


ExperimentHistoryRecord: TypeAlias = ExperimentAttempt | ExperimentEvent


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Derived current state over one immutable attempt and its event history."""

    attempt: ExperimentAttempt
    state: ExperimentState
    events: tuple[ExperimentEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, ExperimentAttempt):
            raise TypeError("attempt must be an ExperimentAttempt")
        if not isinstance(self.state, ExperimentState):
            raise TypeError("state must be an ExperimentState")
        events = tuple(self.events)
        if any(not isinstance(event, ExperimentEvent) for event in events):
            raise TypeError("events must contain only ExperimentEvent values")
        if any(event.attempt_id != self.attempt.attempt_id for event in events):
            raise ValueError("all events must reference the record's attempt")
        event_ids: set[str] = set()
        expected_state = ExperimentState.REGISTERED
        prior_time = self.attempt.registered_at
        for event in events:
            if event.event_id in event_ids:
                raise ValueError("events must not contain duplicate event IDs")
            _validate_transition(expected_state, event.state)
            if _instant(event.occurred_at) < _instant(prior_time):
                raise ValueError("experiment event time cannot precede prior history")
            _validate_reference_continuity(self.attempt, event)
            event_ids.add(event.event_id)
            expected_state = event.state
            prior_time = event.occurred_at
        if self.state is not expected_state:
            raise ValueError("state must equal the state derived from event history")
        object.__setattr__(self, "events", events)

    @property
    def outcome(self) -> ExperimentOutcome | None:
        """Return the terminal outcome, when one has been appended."""
        if not self.events:
            return None
        return self.events[-1].outcome


class InvalidExperimentTransition(ValueError):
    """Raised when a requested lifecycle transition is not legal."""


class ExperimentLedgerCorruptionError(RuntimeError):
    """Raised when durable history cannot be trusted or reconstructed."""


@dataclass(frozen=True, slots=True)
class _LedgerSnapshot:
    records: tuple[ExperimentHistoryRecord, ...]
    attempts: Mapping[str, ExperimentAttempt]
    events: Mapping[str, tuple[ExperimentEvent, ...]]
    states: Mapping[str, ExperimentState]
    event_ids: frozenset[str]


def _lock_exclusive(handle: BinaryIO) -> None:
    """Acquire the experiment ledger's cross-process transaction lock."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("no supported file-locking backend is available")


def _unlock(handle: BinaryIO) -> None:
    """Release the experiment ledger's transaction lock."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    """Lock validation and append as one transaction across ledger instances."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+b") as handle:
        _lock_exclusive(handle)
        try:
            yield
        finally:
            _unlock(handle)


def _validate_transition(current: ExperimentState, requested: ExperimentState) -> None:
    """Reject reopening, skipping required running state, or duplicate states."""
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if requested not in allowed:
        raise InvalidExperimentTransition(
            f"cannot transition experiment from {current.value} to {requested.value}"
        )


def _validate_reference_continuity(
    attempt: ExperimentAttempt, event: ExperimentEvent
) -> None:
    """Prevent an event from silently changing an established hypothesis link."""
    if event.outcome is None:
        return
    registered = attempt.references.hypothesis_id
    reported = event.outcome.references.hypothesis_id
    if registered is not None and reported is not None and registered != reported:
        raise ValueError("an outcome cannot replace the registered hypothesis_id")


def _record_from_payload(payload: Mapping[str, Any]) -> ExperimentHistoryRecord:
    """Parse one strict domain record from a verified chain payload."""
    contract = payload.get("contract")
    if contract == "experiment_attempt":
        return ExperimentAttempt.from_dict(payload)
    if contract == "experiment_event":
        return ExperimentEvent.from_dict(payload)
    raise ValueError(f"unsupported experiment ledger contract {contract!r}")


def _snapshot(records: Sequence[ExperimentHistoryRecord]) -> _LedgerSnapshot:
    """Validate all cross-record invariants and build the current read model."""
    attempts: dict[str, ExperimentAttempt] = {}
    events: dict[str, list[ExperimentEvent]] = {}
    states: dict[str, ExperimentState] = {}
    event_ids: set[str] = set()

    for record in records:
        if isinstance(record, ExperimentAttempt):
            if record.attempt_id in attempts:
                raise ValueError(f"duplicate attempt_id {record.attempt_id!r}")
            if record.parent_attempt_id is not None:
                parent = attempts.get(record.parent_attempt_id)
                if parent is None:
                    raise ValueError(
                        f"unknown parent attempt_id {record.parent_attempt_id!r}"
                    )
                if _instant(record.registered_at) < _instant(parent.registered_at):
                    raise ValueError("a child cannot be registered before its parent")
            attempts[record.attempt_id] = record
            events[record.attempt_id] = []
            states[record.attempt_id] = ExperimentState.REGISTERED
            continue

        attempt = attempts.get(record.attempt_id)
        if attempt is None:
            raise ValueError(
                f"event references unknown attempt_id {record.attempt_id!r}"
            )
        if record.event_id in event_ids:
            raise ValueError(f"duplicate event_id {record.event_id!r}")
        current = states[record.attempt_id]
        _validate_transition(current, record.state)
        prior_time = (
            events[record.attempt_id][-1].occurred_at
            if events[record.attempt_id]
            else attempt.registered_at
        )
        if _instant(record.occurred_at) < _instant(prior_time):
            raise ValueError("experiment event time cannot precede prior history")
        _validate_reference_continuity(attempt, record)
        events[record.attempt_id].append(record)
        states[record.attempt_id] = record.state
        event_ids.add(record.event_id)

    return _LedgerSnapshot(
        records=tuple(records),
        attempts=dict(attempts),
        events={key: tuple(value) for key, value in events.items()},
        states=dict(states),
        event_ids=frozenset(event_ids),
    )


class ExperimentLedger:
    """Durable append-only experiment ledger with derived current state.

    Cooperating writers using this class are serialized across the complete
    read/validate/append transaction, and no application operation rewrites an
    existing record. The underlying governance ledger hash-chains and fsyncs each
    line. Modification, interior deletion or reordering, malformed or partial
    records, and hash/sequence discontinuities in the retained chain fail closed.

    A clean rollback that removes only complete trailing records leaves a valid
    prefix and cannot be proven from this local forward chain alone. Detecting that
    case requires an external trusted checkpoint or monotonic anchor, which Task 2
    deliberately does not provide.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("experiment ledger path must be a file")

    def _load_locked(self) -> _LedgerSnapshot:
        """Load and validate the entire history while the transaction lock is held."""
        try:
            verification = verify_chain(self.path)
        except (AttributeError, TypeError, UnicodeError, ValueError) as exc:
            raise ExperimentLedgerCorruptionError(
                f"experiment ledger chain is malformed: {exc}"
            ) from exc
        if not verification.ok:
            broken = verification.first_break
            raise ExperimentLedgerCorruptionError(
                f"experiment ledger chain is corrupt: {broken}"
            )

        parsed: list[ExperimentHistoryRecord] = []
        if self.path.exists():
            for line_number, raw in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not raw.strip():
                    continue
                try:
                    chained = json.loads(raw)
                    if not isinstance(chained, Mapping):
                        raise TypeError("ledger line must be a JSON object")
                    payload = {
                        key: value
                        for key, value in chained.items()
                        if key not in _CHAIN_FIELDS
                    }
                    parsed.append(_record_from_payload(payload))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ExperimentLedgerCorruptionError(
                        f"invalid experiment record on line {line_number}: {exc}"
                    ) from exc
        try:
            return _snapshot(parsed)
        except (InvalidExperimentTransition, ValueError) as exc:
            raise ExperimentLedgerCorruptionError(
                f"invalid experiment history: {exc}"
            ) from exc

    def _append_locked(self, record: ExperimentHistoryRecord) -> None:
        """Append one already-validated record to the durable hash chain."""
        try:
            append_record(self.path, record.to_dict())
        except GovernanceLedgerCorruptionError as exc:
            raise ExperimentLedgerCorruptionError(str(exc)) from exc

    def register(
        self,
        spec: ExperimentSpec,
        *,
        parent_attempt_id: str | None = None,
        references: ExternalRecordRefs | None = None,
        attempt_id: str | None = None,
        registered_at: datetime | None = None,
    ) -> ExperimentAttempt:
        """Register an attempt before any evaluation result can be appended."""
        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec must be an ExperimentSpec")
        if parent_attempt_id is not None:
            parent_attempt_id = _validated_generated_id(
                "parent_attempt_id", parent_attempt_id, _ATTEMPT_ID_RE
            )
        if references is None:
            references = ExternalRecordRefs()
        if not isinstance(references, ExternalRecordRefs):
            raise TypeError("references must be ExternalRecordRefs")
        if attempt_id is not None:
            attempt_id = _validated_generated_id(
                "attempt_id", attempt_id, _ATTEMPT_ID_RE
            )
        if registered_at is not None:
            registered_at = _aware_datetime("registered_at", registered_at)

        with _ledger_lock(self.path):
            snapshot = self._load_locked()
            chosen_id = attempt_id or _generated_id("exp")
            while chosen_id in snapshot.attempts:
                if attempt_id is not None:
                    raise ValueError(f"duplicate attempt_id {chosen_id!r}")
                chosen_id = _generated_id("exp")
            if parent_attempt_id == chosen_id:
                raise ValueError("an experiment attempt cannot parent itself")
            if (
                parent_attempt_id is not None
                and parent_attempt_id not in snapshot.attempts
            ):
                raise KeyError(f"parent experiment not found: {parent_attempt_id}")
            attempt = ExperimentAttempt(
                attempt_id=chosen_id,
                spec=spec,
                registered_at=registered_at or _utc_now(),
                parent_attempt_id=parent_attempt_id,
                references=references,
            )
            if parent_attempt_id is not None:
                parent = snapshot.attempts[parent_attempt_id]
                if _instant(attempt.registered_at) < _instant(parent.registered_at):
                    raise ValueError("a child cannot be registered before its parent")
            self._append_locked(attempt)
            return attempt

    def transition(
        self,
        attempt_id: str,
        state: ExperimentState,
        *,
        outcome: ExperimentOutcome | None = None,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ExperimentEvent:
        """Validate and append one lifecycle transition for an existing attempt."""
        attempt_id = _validated_generated_id("attempt_id", attempt_id, _ATTEMPT_ID_RE)
        if not isinstance(state, ExperimentState):
            raise TypeError("state must be an ExperimentState")
        if event_id is not None:
            event_id = _validated_generated_id("event_id", event_id, _EVENT_ID_RE)
        if occurred_at is not None:
            occurred_at = _aware_datetime("occurred_at", occurred_at)

        with _ledger_lock(self.path):
            snapshot = self._load_locked()
            attempt = snapshot.attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(f"experiment not found: {attempt_id}")
            _validate_transition(snapshot.states[attempt_id], state)
            chosen_event_id = event_id or _generated_id("evt")
            while chosen_event_id in snapshot.event_ids:
                if event_id is not None:
                    raise ValueError(f"duplicate event_id {chosen_event_id!r}")
                chosen_event_id = _generated_id("evt")
            event = ExperimentEvent(
                event_id=chosen_event_id,
                attempt_id=attempt_id,
                state=state,
                occurred_at=occurred_at or _utc_now(),
                outcome=outcome,
            )
            prior_time = (
                snapshot.events[attempt_id][-1].occurred_at
                if snapshot.events[attempt_id]
                else attempt.registered_at
            )
            if _instant(event.occurred_at) < _instant(prior_time):
                raise ValueError("experiment event time cannot precede prior history")
            _validate_reference_continuity(attempt, event)
            self._append_locked(event)
            return event

    def history(
        self, attempt_id: str | None = None
    ) -> tuple[ExperimentHistoryRecord, ...]:
        """Return immutable history, optionally restricted to one attempt."""
        if attempt_id is not None:
            attempt_id = _validated_generated_id(
                "attempt_id", attempt_id, _ATTEMPT_ID_RE
            )
        with _ledger_lock(self.path):
            snapshot = self._load_locked()
        if attempt_id is None:
            return snapshot.records
        attempt = snapshot.attempts.get(attempt_id)
        if attempt is None:
            raise KeyError(f"experiment not found: {attempt_id}")
        return (attempt, *snapshot.events[attempt_id])

    def get(self, attempt_id: str) -> ExperimentRecord:
        """Return the derived current state for one attempt."""
        attempt_id = _validated_generated_id("attempt_id", attempt_id, _ATTEMPT_ID_RE)
        with _ledger_lock(self.path):
            snapshot = self._load_locked()
        attempt = snapshot.attempts.get(attempt_id)
        if attempt is None:
            raise KeyError(f"experiment not found: {attempt_id}")
        return ExperimentRecord(
            attempt=attempt,
            state=snapshot.states[attempt_id],
            events=snapshot.events[attempt_id],
        )

    def attempts(
        self,
        *,
        trial_family_id: str | None = None,
        spec_fingerprint: str | None = None,
    ) -> tuple[ExperimentAttempt, ...]:
        """List registered attempts for deterministic trial-family accounting."""
        if trial_family_id is not None:
            trial_family_id = _opaque_id("trial_family_id", trial_family_id)
        if spec_fingerprint is not None:
            if not isinstance(spec_fingerprint, str) or not _FINGERPRINT_RE.fullmatch(
                spec_fingerprint
            ):
                raise ValueError("spec_fingerprint must be a full SHA-256 identity")
        with _ledger_lock(self.path):
            snapshot = self._load_locked()
        return tuple(
            attempt
            for attempt in snapshot.attempts.values()
            if (
                trial_family_id is None
                or attempt.spec.trial_family_id == trial_family_id
            )
            and (
                spec_fingerprint is None or attempt.spec_fingerprint == spec_fingerprint
            )
        )

    def count_attempts(
        self,
        *,
        trial_family_id: str | None = None,
        spec_fingerprint: str | None = None,
    ) -> int:
        """Count all matching trials regardless of current or terminal state."""
        return len(
            self.attempts(
                trial_family_id=trial_family_id,
                spec_fingerprint=spec_fingerprint,
            )
        )


__all__ = [
    "ExperimentAttempt",
    "ExperimentEvent",
    "ExperimentHistoryRecord",
    "ExperimentLedger",
    "ExperimentLedgerCorruptionError",
    "ExperimentOutcome",
    "ExperimentRecord",
    "ExperimentSpec",
    "ExperimentState",
    "ExternalRecordRefs",
    "InvalidExperimentTransition",
]
