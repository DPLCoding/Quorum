"""Content-addressed market-bar snapshots and deterministic offline replay.

This module accepts normalized records only.  It deliberately has no loader,
provider, network, experiment-runner, or execution dependency.  A Vibe loader
cache is an operational optimization; this store is immutable scientific
evidence and treats a missing or damaged snapshot as an error.

Identity has two layers:

``content_sha256``
    SHA-256 of the canonical market-bar content payload.  The payload binds the
    content identity namespace, schema version, dataset kind, interval,
    timestamp convention, and canonically ordered bars.  Datetimes are rendered
    in UTC, so equivalent representations of one instant have one identity.

``snapshot_id``
    SHA-256 of the snapshot identity namespace, schema version,
    ``content_sha256``, semantic declarations, derived assets/row count, and
    canonical provenance.  Identical content with different provenance shares
    ``content_sha256`` but has a different ``snapshot_id``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Self, TypeAlias

from src.quorum.contracts import (
    _aware_datetime,
    _canonical_json,
    _immutable_metadata,
    _instant,
    _parse_datetime,
    _payload,
    _required_text,
    _require_payload_keys,
    _thaw_json_value,
)
from src.quorum.experiments import _ledger_lock

JSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JSONValue", ...]
    | Mapping[str, "JSONValue"]
)

_SCHEMA_VERSION = 1
_DATASET_KIND = "market_bar"
_CONTENT_IDENTITY_NAMESPACE = "quorum-market-bar-content-v1"
_SNAPSHOT_IDENTITY_NAMESPACE = "quorum-dataset-snapshot-v1"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_FILE = "manifest.json"
_BARS_FILE = "bars.jsonl"


class DatasetSnapshotError(RuntimeError):
    """Base error for durable dataset-snapshot operations."""


class DatasetSnapshotNotFoundError(DatasetSnapshotError):
    """Raised when a requested immutable snapshot does not exist."""


class DatasetSnapshotIntegrityError(DatasetSnapshotError):
    """Raised when persisted snapshot evidence cannot be trusted."""


def _utc_text(value: datetime) -> str:
    """Return one canonical ISO-8601 rendering for an absolute instant."""
    return _instant(value).isoformat()


def _finite_number(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    # Canonicalize signed zero so numerically identical records have one identity.
    return 0.0 if result == 0.0 else result


def _sha256(payload: Mapping[str, Any]) -> str:
    rendered = _canonical_json(payload).encode("utf-8")
    return f"sha256:{hashlib.sha256(rendered).hexdigest()}"


def _validated_hash(name: str, value: object) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex characters>")
    return value


def _strict_json_object(text: str, name: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} JSON contains non-finite number {value!r}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} JSON must contain an object")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class MarketBar:
    """One normalized half-open market-bar observation ``[start_at, event_at)``."""

    asset: str
    start_at: datetime
    event_at: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        asset = _required_text("asset", self.asset)
        start_at = _aware_datetime("start_at", self.start_at)
        event_at = _aware_datetime("event_at", self.event_at)
        available_at = _aware_datetime("available_at", self.available_at)
        if _instant(start_at) >= _instant(event_at):
            raise ValueError("start_at must be earlier than event_at")

        open_value = _finite_number("open", self.open)
        high = _finite_number("high", self.high)
        low = _finite_number("low", self.low)
        close = _finite_number("close", self.close)
        volume = _finite_number("volume", self.volume, minimum=0.0)
        if high < max(open_value, close):
            raise ValueError("high must be >= max(open, close)")
        if low > min(open_value, close):
            raise ValueError("low must be <= min(open, close)")
        if high < low:
            raise ValueError("high must be >= low")

        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "start_at", start_at)
        object.__setattr__(self, "event_at", event_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "open", open_value)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)

    @property
    def key(self) -> tuple[str, datetime]:
        """Return the logical bar identity using the event's actual instant."""
        return (self.asset, _instant(self.event_at))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarketBar):
            return NotImplemented
        return self._identity_tuple() == other._identity_tuple()

    def __hash__(self) -> int:
        return hash(self._identity_tuple())

    def _identity_tuple(self) -> tuple[Any, ...]:
        return (
            self.asset,
            _instant(self.start_at),
            _instant(self.event_at),
            _instant(self.available_at),
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the strict canonical persisted representation."""
        return _payload(
            "market_bar",
            schema_version=_SCHEMA_VERSION,
            asset=self.asset,
            start_at=_utc_text(self.start_at),
            event_at=_utc_text(self.event_at),
            available_at=_utc_text(self.available_at),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("market_bar payload must be a mapping")
        _require_payload_keys(
            value,
            contract_name="market_bar",
            schema_version=_SCHEMA_VERSION,
            fields=frozenset(
                {
                    "asset",
                    "start_at",
                    "event_at",
                    "available_at",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                }
            ),
        )
        return cls(
            asset=value["asset"],  # type: ignore[arg-type]
            start_at=_parse_datetime("start_at", value["start_at"]),
            event_at=_parse_datetime("event_at", value["event_at"]),
            available_at=_parse_datetime("available_at", value["available_at"]),
            open=value["open"],  # type: ignore[arg-type]
            high=value["high"],  # type: ignore[arg-type]
            low=value["low"],  # type: ignore[arg-type]
            close=value["close"],  # type: ignore[arg-type]
            volume=value["volume"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True, eq=False)
class DatasetProvenance:
    """Immutable acquisition and interpretation evidence for one snapshot."""

    source_id: str
    retrieved_at: datetime
    adjustment_id: str
    source_revision_id: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = _required_text("source_id", self.source_id)
        retrieved_at = _aware_datetime("retrieved_at", self.retrieved_at)
        adjustment_id = _required_text("adjustment_id", self.adjustment_id)
        source_revision_id = self.source_revision_id
        if source_revision_id is not None:
            source_revision_id = _required_text(
                "source_revision_id", source_revision_id
            )
        metadata = _immutable_metadata(self.metadata)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "adjustment_id", adjustment_id)
        object.__setattr__(self, "source_revision_id", source_revision_id)
        object.__setattr__(self, "metadata", metadata)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetProvenance):
            return NotImplemented
        return self._identity_payload() == other._identity_payload()

    def __hash__(self) -> int:
        return hash(_canonical_json(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "source_id": self.source_id,
            "retrieved_at": _utc_text(self.retrieved_at),
            "adjustment_id": self.adjustment_id,
            "source_revision_id": self.source_revision_id,
            "metadata": _thaw_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        return _payload(
            "dataset_provenance",
            schema_version=_SCHEMA_VERSION,
            source_id=self.source_id,
            retrieved_at=_utc_text(self.retrieved_at),
            adjustment_id=self.adjustment_id,
            source_revision_id=self.source_revision_id,
            metadata=_thaw_json_value(self.metadata),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("dataset_provenance payload must be a mapping")
        _require_payload_keys(
            value,
            contract_name="dataset_provenance",
            schema_version=_SCHEMA_VERSION,
            fields=frozenset(
                {
                    "source_id",
                    "retrieved_at",
                    "adjustment_id",
                    "source_revision_id",
                    "metadata",
                }
            ),
        )
        return cls(
            source_id=value["source_id"],  # type: ignore[arg-type]
            retrieved_at=_parse_datetime("retrieved_at", value["retrieved_at"]),
            adjustment_id=value["adjustment_id"],  # type: ignore[arg-type]
            source_revision_id=value["source_revision_id"],  # type: ignore[arg-type]
            metadata=value["metadata"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetSnapshotManifest:
    """Strict versioned manifest for one complete persisted snapshot."""

    snapshot_id: str
    content_sha256: str
    dataset_kind: str
    interval_id: str
    timestamp_convention: str
    assets: tuple[str, ...]
    row_count: int
    provenance: DatasetProvenance

    def __post_init__(self) -> None:
        snapshot_id = _validated_hash("snapshot_id", self.snapshot_id)
        content_sha256 = _validated_hash("content_sha256", self.content_sha256)
        if self.dataset_kind != _DATASET_KIND:
            raise ValueError(f"dataset_kind must be {_DATASET_KIND!r}")
        interval_id = _required_text("interval_id", self.interval_id)
        timestamp_convention = _required_text(
            "timestamp_convention", self.timestamp_convention
        )
        if isinstance(self.assets, (str, bytes)):
            raise TypeError("assets must be a sequence of strings")
        assets = tuple(_required_text("asset", asset) for asset in self.assets)
        if not assets:
            raise ValueError("assets must not be empty")
        if assets != tuple(sorted(set(assets))):
            raise ValueError("assets must be unique and canonically sorted")
        if type(self.row_count) is not int:
            raise TypeError("row_count must be an integer")
        if self.row_count < 1:
            raise ValueError("row_count must be positive")
        if not isinstance(self.provenance, DatasetProvenance):
            raise TypeError("provenance must be a DatasetProvenance")
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(self, "interval_id", interval_id)
        object.__setattr__(self, "timestamp_convention", timestamp_convention)
        object.__setattr__(self, "assets", assets)

    def to_dict(self) -> dict[str, JSONValue]:
        return _payload(
            "dataset_snapshot_manifest",
            schema_version=_SCHEMA_VERSION,
            content_identity_namespace=_CONTENT_IDENTITY_NAMESPACE,
            snapshot_identity_namespace=_SNAPSHOT_IDENTITY_NAMESPACE,
            snapshot_id=self.snapshot_id,
            content_sha256=self.content_sha256,
            dataset_kind=self.dataset_kind,
            interval_id=self.interval_id,
            timestamp_convention=self.timestamp_convention,
            assets=list(self.assets),
            row_count=self.row_count,
            provenance=self.provenance.to_dict(),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("dataset_snapshot_manifest payload must be a mapping")
        _require_payload_keys(
            value,
            contract_name="dataset_snapshot_manifest",
            schema_version=_SCHEMA_VERSION,
            fields=frozenset(
                {
                    "content_identity_namespace",
                    "snapshot_identity_namespace",
                    "snapshot_id",
                    "content_sha256",
                    "dataset_kind",
                    "interval_id",
                    "timestamp_convention",
                    "assets",
                    "row_count",
                    "provenance",
                }
            ),
        )
        if value["content_identity_namespace"] != _CONTENT_IDENTITY_NAMESPACE:
            raise ValueError("unsupported content identity namespace")
        if value["snapshot_identity_namespace"] != _SNAPSHOT_IDENTITY_NAMESPACE:
            raise ValueError("unsupported snapshot identity namespace")
        assets = value["assets"]
        if not isinstance(assets, list):
            raise TypeError("assets must be a list")
        provenance = value["provenance"]
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be an object")
        return cls(
            snapshot_id=value["snapshot_id"],  # type: ignore[arg-type]
            content_sha256=value["content_sha256"],  # type: ignore[arg-type]
            dataset_kind=value["dataset_kind"],  # type: ignore[arg-type]
            interval_id=value["interval_id"],  # type: ignore[arg-type]
            timestamp_convention=value["timestamp_convention"],  # type: ignore[arg-type]
            assets=tuple(assets),  # type: ignore[arg-type]
            row_count=value["row_count"],  # type: ignore[arg-type]
            provenance=DatasetProvenance.from_dict(provenance),
        )


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """Verified immutable manifest and canonical market-bar records."""

    manifest: DatasetSnapshotManifest
    bars: tuple[MarketBar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DatasetSnapshotManifest):
            raise TypeError("manifest must be a DatasetSnapshotManifest")
        canonical = _canonical_bars(self.bars)
        if tuple(self.bars) != canonical:
            raise ValueError("bars must already be in canonical order")
        _verify_manifest_against_bars(self.manifest, canonical)
        object.__setattr__(self, "bars", canonical)


def _canonical_bars(values: Iterable[MarketBar]) -> tuple[MarketBar, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("bars must be an iterable of MarketBar values")
    try:
        bars = tuple(values)
    except TypeError as exc:
        raise TypeError("bars must be an iterable of MarketBar values") from exc
    if not bars:
        raise ValueError("bars must not be empty")
    if any(not isinstance(bar, MarketBar) for bar in bars):
        raise TypeError("bars must contain only MarketBar values")
    ordered = tuple(sorted(bars, key=lambda bar: bar.key))
    keys = [bar.key for bar in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("bars contain duplicate (asset, event_at instant) keys")
    previous_by_asset: dict[str, MarketBar] = {}
    for bar in ordered:
        previous = previous_by_asset.get(bar.asset)
        if previous is not None and _instant(bar.start_at) < _instant(
            previous.event_at
        ):
            raise ValueError(
                f"bars for asset {bar.asset!r} contain overlapping intervals"
            )
        previous_by_asset[bar.asset] = bar
    return ordered


def _content_payload(
    bars: tuple[MarketBar, ...], interval_id: str, timestamp_convention: str
) -> dict[str, Any]:
    return {
        "identity_namespace": _CONTENT_IDENTITY_NAMESPACE,
        "schema_version": _SCHEMA_VERSION,
        "dataset_kind": _DATASET_KIND,
        "interval_id": interval_id,
        "timestamp_convention": timestamp_convention,
        "bars": [bar.to_dict() for bar in bars],
    }


def _content_hash(
    bars: tuple[MarketBar, ...], interval_id: str, timestamp_convention: str
) -> str:
    return _sha256(_content_payload(bars, interval_id, timestamp_convention))


def _snapshot_identity_payload(
    *,
    content_sha256: str,
    interval_id: str,
    timestamp_convention: str,
    assets: tuple[str, ...],
    row_count: int,
    provenance: DatasetProvenance,
) -> dict[str, Any]:
    return {
        "identity_namespace": _SNAPSHOT_IDENTITY_NAMESPACE,
        "schema_version": _SCHEMA_VERSION,
        "content_sha256": content_sha256,
        "dataset_kind": _DATASET_KIND,
        "interval_id": interval_id,
        "timestamp_convention": timestamp_convention,
        "assets": list(assets),
        "row_count": row_count,
        "provenance": provenance._identity_payload(),
    }


def _manifest_for(
    bars: tuple[MarketBar, ...],
    *,
    interval_id: str,
    timestamp_convention: str,
    provenance: DatasetProvenance,
) -> DatasetSnapshotManifest:
    interval_id = _required_text("interval_id", interval_id)
    timestamp_convention = _required_text("timestamp_convention", timestamp_convention)
    if not isinstance(provenance, DatasetProvenance):
        raise TypeError("provenance must be a DatasetProvenance")
    assets = tuple(sorted({bar.asset for bar in bars}))
    content_sha256 = _content_hash(bars, interval_id, timestamp_convention)
    snapshot_id = _sha256(
        _snapshot_identity_payload(
            content_sha256=content_sha256,
            interval_id=interval_id,
            timestamp_convention=timestamp_convention,
            assets=assets,
            row_count=len(bars),
            provenance=provenance,
        )
    )
    return DatasetSnapshotManifest(
        snapshot_id=snapshot_id,
        content_sha256=content_sha256,
        dataset_kind=_DATASET_KIND,
        interval_id=interval_id,
        timestamp_convention=timestamp_convention,
        assets=assets,
        row_count=len(bars),
        provenance=provenance,
    )


def _verify_manifest_against_bars(
    manifest: DatasetSnapshotManifest, bars: tuple[MarketBar, ...]
) -> None:
    assets = tuple(sorted({bar.asset for bar in bars}))
    if manifest.row_count != len(bars):
        raise ValueError("manifest row_count does not match bars payload")
    if manifest.assets != assets:
        raise ValueError("manifest assets do not match bars payload")
    content_sha256 = _content_hash(
        bars, manifest.interval_id, manifest.timestamp_convention
    )
    if manifest.content_sha256 != content_sha256:
        raise ValueError("manifest content_sha256 does not match bars payload")
    snapshot_id = _sha256(
        _snapshot_identity_payload(
            content_sha256=content_sha256,
            interval_id=manifest.interval_id,
            timestamp_convention=manifest.timestamp_convention,
            assets=assets,
            row_count=len(bars),
            provenance=manifest.provenance,
        )
    )
    if manifest.snapshot_id != snapshot_id:
        raise ValueError("manifest snapshot_id does not match snapshot payload")


def _manifest_bytes(manifest: DatasetSnapshotManifest) -> bytes:
    return (_canonical_json(manifest.to_dict()) + "\n").encode("utf-8")


def _bars_bytes(bars: tuple[MarketBar, ...]) -> bytes:
    return ("".join(_canonical_json(bar.to_dict()) + "\n" for bar in bars)).encode(
        "utf-8"
    )


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class DatasetSnapshotStore:
    """Append-only local store for verified market-bar snapshots."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a pathlib.Path")
        self.root = root

    def create(
        self,
        bars: Iterable[MarketBar],
        *,
        interval_id: str,
        timestamp_convention: str,
        provenance: DatasetProvenance,
    ) -> DatasetSnapshotManifest:
        """Create or idempotently reuse one immutable scientific snapshot."""
        canonical = _canonical_bars(bars)
        manifest = _manifest_for(
            canonical,
            interval_id=interval_id,
            timestamp_convention=timestamp_convention,
            provenance=provenance,
        )
        manifest_payload = _manifest_bytes(manifest)
        bars_payload = _bars_bytes(canonical)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir():
            raise DatasetSnapshotIntegrityError("snapshot root is not a directory")

        destination = self._snapshot_directory(manifest.snapshot_id)
        lock_target = self.root / ".dataset-snapshots"
        with _ledger_lock(lock_target):
            if destination.exists() or destination.is_symlink():
                existing = self._load_verified(manifest.snapshot_id)
                if (
                    existing.manifest != manifest
                    or _bars_bytes(existing.bars) != bars_payload
                ):
                    raise DatasetSnapshotIntegrityError(
                        "existing snapshot destination conflicts with derived identity"
                    )
                return existing.manifest

            staging = self.root / (
                f".tmp-{destination.name}-{os.getpid()}-{uuid.uuid4().hex}"
            )
            try:
                staging.mkdir(mode=0o700)
                _write_durable(staging / _MANIFEST_FILE, manifest_payload)
                _write_durable(staging / _BARS_FILE, bars_payload)
                _fsync_directory(staging)
                os.rename(staging, destination)
                _fsync_directory(self.root)
            except FileExistsError as exc:
                raise DatasetSnapshotIntegrityError(
                    "snapshot publication destination already exists"
                ) from exc
            finally:
                if staging.exists() and staging.parent == self.root:
                    shutil.rmtree(staging)
        return manifest

    def load(self, snapshot_id: str) -> DatasetSnapshot:
        """Verify and return a complete snapshot without any provider fallback."""
        return self._load_verified(snapshot_id)

    def verify(self, snapshot_id: str) -> DatasetSnapshotManifest:
        """Verify all persisted evidence and return its trusted manifest."""
        return self._load_verified(snapshot_id).manifest

    def _snapshot_directory(self, snapshot_id: str) -> Path:
        validated = _validated_hash("snapshot_id", snapshot_id)
        return self.root / f"sha256-{validated.removeprefix('sha256:')}"

    def _load_verified(self, snapshot_id: str) -> DatasetSnapshot:
        snapshot_id = _validated_hash("snapshot_id", snapshot_id)
        directory = self._snapshot_directory(snapshot_id)
        if not directory.exists():
            raise DatasetSnapshotNotFoundError(
                f"dataset snapshot {snapshot_id!r} was not found"
            )
        if directory.is_symlink() or not directory.is_dir():
            raise DatasetSnapshotIntegrityError(
                "snapshot destination must be a real directory"
            )
        manifest_path = directory / _MANIFEST_FILE
        bars_path = directory / _BARS_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DatasetSnapshotIntegrityError("snapshot manifest is missing")
        if bars_path.is_symlink() or not bars_path.is_file():
            raise DatasetSnapshotIntegrityError("snapshot bars payload is missing")

        try:
            manifest_raw = manifest_path.read_bytes()
            bars_raw = bars_path.read_bytes()
            manifest_text = manifest_raw.decode("utf-8")
            bars_text = bars_raw.decode("utf-8")
            manifest = DatasetSnapshotManifest.from_dict(
                _strict_json_object(manifest_text, "dataset_snapshot_manifest")
            )
            if manifest.snapshot_id != snapshot_id:
                raise ValueError("manifest snapshot_id does not match requested ID")
            bars = self._parse_bars(bars_text)
            canonical = _canonical_bars(bars)
            if bars != canonical:
                raise ValueError("bars payload is not in canonical order")
            _verify_manifest_against_bars(manifest, canonical)
            if manifest_raw != _manifest_bytes(manifest):
                raise ValueError("manifest bytes are not canonical")
            if bars_raw != _bars_bytes(canonical):
                raise ValueError("bars bytes are not canonical")
            return DatasetSnapshot(manifest=manifest, bars=canonical)
        except DatasetSnapshotIntegrityError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise DatasetSnapshotIntegrityError(
                f"dataset snapshot {snapshot_id!r} failed integrity verification"
            ) from exc

    @staticmethod
    def _parse_bars(text: str) -> tuple[MarketBar, ...]:
        if not text:
            raise ValueError("bars payload must not be empty")
        lines = text.splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("bars payload contains an empty record")
        return tuple(
            MarketBar.from_dict(_strict_json_object(line, "market_bar"))
            for line in lines
        )


__all__ = [
    "DatasetProvenance",
    "DatasetSnapshot",
    "DatasetSnapshotError",
    "DatasetSnapshotIntegrityError",
    "DatasetSnapshotManifest",
    "DatasetSnapshotNotFoundError",
    "DatasetSnapshotStore",
    "JSONValue",
    "MarketBar",
]
