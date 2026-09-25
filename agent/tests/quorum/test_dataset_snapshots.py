"""Tests for Task 8 content-addressed market-bar dataset snapshots."""

from __future__ import annotations

import ast
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import src.quorum.data.snapshots as snapshot_module
from src.quorum import (
    DatasetProvenance,
    DatasetSnapshotIntegrityError,
    DatasetSnapshotNotFoundError,
    DatasetSnapshotStore,
    EvaluationMode,
    EvaluationProtocol,
    ExperimentAttempt,
    ExperimentSpec,
    FinalHoldout,
    FinalHoldoutState,
    MarketBar,
    TimeInterval,
    materialize_chronological_plan,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
ATTEMPT_ID = "exp_" + ("8" * 32)


def _bar(
    position: int = 0,
    *,
    asset: str = "AAA.US",
    start_at: datetime | None = None,
    event_at: datetime | None = None,
    available_at: datetime | None = None,
    open: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: float = 1_000.0,
) -> MarketBar:
    start = start_at or datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=position)
    event = event_at or start + timedelta(hours=1)
    available = available_at or event + timedelta(minutes=5)
    return MarketBar(
        asset=asset,
        start_at=start,
        event_at=event,
        available_at=available,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _bars(count: int = 4) -> tuple[MarketBar, ...]:
    return tuple(_bar(position) for position in range(count))


def _provenance(**overrides: object) -> DatasetProvenance:
    values: dict[str, object] = {
        "source_id": "local:test-fixture:v1",
        "retrieved_at": datetime(2026, 2, 1, tzinfo=UTC),
        "adjustment_id": "raw",
        "source_revision_id": "revision-1",
        "metadata": {"query": {"venue": "TEST", "fields": ["ohlcv"]}},
    }
    values.update(overrides)
    return DatasetProvenance(**values)  # type: ignore[arg-type]


def _create(
    root: Path,
    bars: tuple[MarketBar, ...] | list[MarketBar] | None = None,
    *,
    provenance: DatasetProvenance | None = None,
    interval_id: str = "1h",
    timestamp_convention: str = "bar_close",
):
    return DatasetSnapshotStore(root).create(
        _bars() if bars is None else bars,
        interval_id=interval_id,
        timestamp_convention=timestamp_convention,
        provenance=provenance or _provenance(),
    )


def _directory(root: Path, snapshot_id: str) -> Path:
    return root / f"sha256-{snapshot_id.removeprefix('sha256:')}"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_manifest(root: Path, snapshot_id: str, mutate) -> None:
    path = _directory(root, snapshot_id) / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_canonical_bytes(value))


def _rewrite_first_bar(root: Path, snapshot_id: str, mutate) -> None:
    path = _directory(root, snapshot_id) / "bars.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))


def test_snapshot_module_has_only_clean_core_dependencies() -> None:
    tree = ast.parse(Path(snapshot_module.__file__).read_text(encoding="utf-8"))
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
        if module.split(".", 1)[0] not in {"src", "__future__"}
        and module.split(".", 1)[0] not in __import__("sys").stdlib_module_names
    }
    project_imports = {
        module for module in imported_modules if module.startswith("src.")
    }
    assert nonstdlib == set()
    assert project_imports == {
        "src.quorum.contracts",
        "src.quorum.experiments",
    }


def test_deterministic_identity_order_and_bytes(tmp_path: Path) -> None:
    bars = [
        _bar(1, asset="BBB.US"),
        _bar(1, asset="AAA.US"),
        _bar(0, asset="BBB.US"),
        _bar(0, asset="AAA.US"),
    ]
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _create(first_root, bars)
    second = _create(second_root, list(reversed(bars)))

    assert first.content_sha256 == second.content_sha256
    assert first.snapshot_id == second.snapshot_id
    assert first.assets == ("AAA.US", "BBB.US")
    first_dir = _directory(first_root, first.snapshot_id)
    second_dir = _directory(second_root, second.snapshot_id)
    assert (first_dir / "manifest.json").read_bytes() == (
        second_dir / "manifest.json"
    ).read_bytes()
    assert (first_dir / "bars.jsonl").read_bytes() == (
        second_dir / "bars.jsonl"
    ).read_bytes()
    loaded = DatasetSnapshotStore(first_root).load(first.snapshot_id)
    assert [bar.key for bar in loaded.bars] == sorted(bar.key for bar in bars)


def test_identity_definition_is_pinned(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    assert (
        manifest.content_sha256
        == "sha256:484b12cb1ffa8e9e3fa71e9f3c72b40ccc061102d0a247bb63774120b51b30a8"
    )
    assert (
        manifest.snapshot_id
        == "sha256:b433dd0195e68b22987cc025c1aa5fa2c6f6ad1506729f78d50bf831d0ffd7dc"
    )


def test_idempotent_create_load_and_serialization_are_stable(tmp_path: Path) -> None:
    store = DatasetSnapshotStore(tmp_path)
    first = store.create(
        _bars(),
        interval_id="1h",
        timestamp_convention="bar_close",
        provenance=_provenance(),
    )
    directory = _directory(tmp_path, first.snapshot_id)
    before = {
        name: (directory / name).read_bytes()
        for name in ("manifest.json", "bars.jsonl")
    }
    second = store.create(
        reversed(_bars()),
        interval_id="1h",
        timestamp_convention="bar_close",
        provenance=_provenance(),
    )
    loaded = store.load(first.snapshot_id)

    assert first == second == loaded.manifest == store.verify(first.snapshot_id)
    assert before == {
        name: (directory / name).read_bytes()
        for name in ("manifest.json", "bars.jsonl")
    }
    assert directory.joinpath("manifest.json").read_bytes() == _canonical_bytes(
        loaded.manifest.to_dict()
    )


def test_multi_asset_intervals_are_validated_per_asset(tmp_path: Path) -> None:
    bars = tuple(
        _bar(position, asset=asset)
        for position in range(3)
        for asset in ("BBB.US", "AAA.US")
    )
    manifest = _create(tmp_path, list(reversed(bars)))
    loaded = DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)
    assert manifest.assets == ("AAA.US", "BBB.US")
    assert manifest.row_count == 6
    assert tuple((bar.asset, bar.event_at) for bar in loaded.bars) == tuple(
        sorted((bar.asset, bar.event_at) for bar in bars)
    )


@pytest.mark.parametrize("field", ["start_at", "event_at", "available_at"])
def test_naive_timestamps_are_rejected(field: str) -> None:
    values = {
        "start_at": datetime(2026, 1, 1, 0, tzinfo=UTC),
        "event_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
        "available_at": datetime(2026, 1, 1, 1, 5, tzinfo=UTC),
    }
    values[field] = values[field].replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        _bar(**values)  # type: ignore[arg-type]


def test_provenance_requires_aware_retrieval_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _provenance(retrieved_at=datetime(2026, 1, 1))


def test_equivalent_timezone_representations_have_one_identity(tmp_path: Path) -> None:
    utc_bar = _bar(
        start_at=datetime(2026, 1, 1, 14, tzinfo=UTC),
        event_at=datetime(2026, 1, 1, 15, tzinfo=UTC),
        available_at=datetime(2026, 1, 1, 15, 5, tzinfo=UTC),
    )
    ny_bar = _bar(
        start_at=datetime(2026, 1, 1, 9, tzinfo=NEW_YORK),
        event_at=datetime(2026, 1, 1, 10, tzinfo=NEW_YORK),
        available_at=datetime(2026, 1, 1, 10, 5, tzinfo=NEW_YORK),
    )
    first = _create(tmp_path / "utc", [utc_bar])
    second = _create(
        tmp_path / "ny",
        [ny_bar],
        provenance=_provenance(retrieved_at=datetime(2026, 1, 31, 19, tzinfo=NEW_YORK)),
    )
    assert utc_bar == ny_bar
    assert first.content_sha256 == second.content_sha256
    assert first.snapshot_id == second.snapshot_id


def test_equivalent_event_instant_is_a_duplicate() -> None:
    utc_bar = _bar(
        start_at=datetime(2026, 1, 1, 14, tzinfo=UTC),
        event_at=datetime(2026, 1, 1, 15, tzinfo=UTC),
    )
    ny_bar = _bar(
        start_at=datetime(2026, 1, 1, 9, tzinfo=NEW_YORK),
        event_at=datetime(2026, 1, 1, 10, tzinfo=NEW_YORK),
    )
    with pytest.raises(ValueError, match="duplicate"):
        DatasetSnapshotStore(Path("unused")).create(
            (utc_bar, ny_bar),
            interval_id="1h",
            timestamp_convention="bar_close",
            provenance=_provenance(),
        )


def test_dst_fallback_repeated_hour_preserves_distinct_instants(tmp_path: Path) -> None:
    first = _bar(
        start_at=datetime(2026, 11, 1, 1, 0, tzinfo=NEW_YORK, fold=0),
        event_at=datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=0),
        available_at=datetime(2026, 11, 1, 1, 35, tzinfo=NEW_YORK, fold=0),
    )
    second = _bar(
        start_at=datetime(2026, 11, 1, 1, 0, tzinfo=NEW_YORK, fold=1),
        event_at=datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=1),
        available_at=datetime(2026, 11, 1, 1, 35, tzinfo=NEW_YORK, fold=1),
    )
    manifest = _create(tmp_path, [second, first])
    loaded = DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)
    assert len(loaded.bars) == 2
    assert loaded.bars[0].event_at == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    assert loaded.bars[1].event_at == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)


def test_delayed_availability_is_valid(tmp_path: Path) -> None:
    bar = _bar(available_at=datetime(2026, 1, 2, tzinfo=UTC))
    manifest = _create(tmp_path, [bar])
    loaded = DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)
    assert loaded.bars[0].available_at > loaded.bars[0].event_at


def test_availability_before_event_is_not_globally_rejected(tmp_path: Path) -> None:
    bar = _bar(available_at=datetime(2025, 12, 31, 23, 59, tzinfo=UTC))
    manifest = _create(tmp_path, [bar])
    assert DatasetSnapshotStore(tmp_path).verify(manifest.snapshot_id) == manifest


def test_start_at_must_precede_event_by_actual_instant() -> None:
    with pytest.raises(ValueError, match="start_at must be earlier"):
        _bar(
            start_at=datetime(2026, 11, 1, 1, 45, tzinfo=NEW_YORK, fold=1),
            event_at=datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=0),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"open": float("nan")}, "finite"),
        ({"high": float("inf")}, "finite"),
        ({"low": float("-inf")}, "finite"),
        ({"volume": -1.0}, ">= 0"),
        ({"high": 100.5, "open": 101.0}, "high must"),
        ({"low": 101.5, "close": 101.0}, "low must"),
        ({"high": 98.0, "low": 99.0}, "high must"),
    ],
)
def test_invalid_market_values_fail_closed(
    changes: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _bar(**changes)


def test_blank_asset_and_semantic_declarations_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="asset"):
        _bar(asset=" ")
    with pytest.raises(ValueError, match="interval_id"):
        _create(tmp_path / "interval", interval_id="")
    with pytest.raises(ValueError, match="timestamp_convention"):
        _create(tmp_path / "timestamp", timestamp_convention=" ")


def test_duplicate_event_and_overlapping_intervals_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _create(tmp_path / "duplicate", [_bar(), _bar(open=100.5)])
    with pytest.raises(ValueError, match="overlapping"):
        _create(
            tmp_path / "overlap",
            [
                _bar(0),
                _bar(
                    start_at=datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                    event_at=datetime(2026, 1, 1, 1, 30, tzinfo=UTC),
                ),
            ],
        )


@pytest.mark.parametrize(
    ("case", "content_changes"),
    [
        ("ohlcv", True),
        ("event", True),
        ("start", True),
        ("availability", True),
        ("interval", True),
        ("timestamp_convention", True),
        ("adjustment", False),
        ("source", False),
        ("revision", False),
        ("metadata", False),
    ],
)
def test_identity_sensitivity(tmp_path: Path, case: str, content_changes: bool) -> None:
    base_bar = _bar()
    bars = [base_bar]
    provenance = _provenance()
    interval = "1h"
    convention = "bar_close"
    first = _create(
        tmp_path / "base",
        bars,
        provenance=provenance,
        interval_id=interval,
        timestamp_convention=convention,
    )

    if case == "ohlcv":
        bars = [replace(base_bar, close=100.5)]
    elif case == "event":
        bars = [
            replace(
                base_bar,
                event_at=base_bar.event_at + timedelta(minutes=1),
            )
        ]
    elif case == "start":
        bars = [
            replace(
                base_bar,
                start_at=base_bar.start_at + timedelta(minutes=1),
            )
        ]
    elif case == "availability":
        bars = [
            replace(
                base_bar,
                available_at=base_bar.available_at + timedelta(minutes=1),
            )
        ]
    elif case == "interval":
        interval = "60m"
    elif case == "timestamp_convention":
        convention = "period_end"
    elif case == "adjustment":
        provenance = replace(provenance, adjustment_id="split_adjusted")
    elif case == "source":
        provenance = replace(provenance, source_id="local:other:v1")
    elif case == "revision":
        provenance = replace(provenance, source_revision_id="revision-2")
    elif case == "metadata":
        provenance = replace(provenance, metadata={"query": {"venue": "OTHER"}})

    second = _create(
        tmp_path / case,
        bars,
        provenance=provenance,
        interval_id=interval,
        timestamp_convention=convention,
    )
    assert (first.content_sha256 != second.content_sha256) is content_changes
    assert first.snapshot_id != second.snapshot_id


def test_retrieval_instant_changes_snapshot_not_content(tmp_path: Path) -> None:
    first = _create(tmp_path / "one")
    second = _create(
        tmp_path / "two",
        provenance=_provenance(retrieved_at=datetime(2026, 2, 2, tzinfo=UTC)),
    )
    assert first.content_sha256 == second.content_sha256
    assert first.snapshot_id != second.snapshot_id


def test_provenance_is_strict_bounded_and_defensively_frozen() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    provenance = _provenance(metadata=metadata)
    metadata["nested"]["values"].append(3)  # type: ignore[index,union-attr]
    assert provenance.to_dict()["metadata"] == {"nested": {"values": [1, 2]}}
    assert isinstance(provenance.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        provenance.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError, match="JSON-compatible"):
        _provenance(metadata={"bad": object()})
    with pytest.raises(ValueError, match="4096"):
        _provenance(metadata={"too_large": "x" * 5000})


@pytest.mark.parametrize(
    "field",
    ["source_id", "adjustment_id", "source_revision_id"],
)
def test_provenance_string_identities_are_nonblank(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _provenance(**{field: " "})


@pytest.mark.parametrize(
    "mutation",
    [
        "row_count",
        "content_hash",
        "snapshot_id",
        "assets",
        "manifest_value",
        "manifest_timestamp",
    ],
)
def test_manifest_tampering_fails_before_load(tmp_path: Path, mutation: str) -> None:
    manifest = _create(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        if mutation == "row_count":
            value["row_count"] = 999
        elif mutation == "content_hash":
            value["content_sha256"] = "sha256:" + ("0" * 64)
        elif mutation == "snapshot_id":
            value["snapshot_id"] = "sha256:" + ("1" * 64)
        elif mutation == "assets":
            value["assets"] = ["WRONG.US"]
        elif mutation == "manifest_value":
            value["interval_id"] = "tampered"
        else:
            provenance = value["provenance"]
            assert isinstance(provenance, dict)
            provenance["retrieved_at"] = "2027-01-01T00:00:00+00:00"

    _rewrite_manifest(tmp_path, manifest.snapshot_id, mutate)
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)


@pytest.mark.parametrize("mutation", ["ohlc", "timestamp", "order"])
def test_bars_tampering_fails_before_load(tmp_path: Path, mutation: str) -> None:
    manifest = _create(tmp_path)
    path = _directory(tmp_path, manifest.snapshot_id) / "bars.jsonl"
    if mutation == "order":
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    else:
        _rewrite_first_bar(
            tmp_path,
            manifest.snapshot_id,
            lambda row: row.__setitem__(
                "close" if mutation == "ohlc" else "available_at",
                100.5 if mutation == "ohlc" else "2026-01-01T10:00:00+00:00",
            ),
        )
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)


def test_noncanonical_bytes_are_rejected(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    path = _directory(tmp_path, manifest.snapshot_id) / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).verify(manifest.snapshot_id)


@pytest.mark.parametrize("missing", ["manifest.json", "bars.jsonl"])
def test_missing_snapshot_file_fails_closed(tmp_path: Path, missing: str) -> None:
    manifest = _create(tmp_path)
    (_directory(tmp_path, manifest.snapshot_id) / missing).unlink()
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)


@pytest.mark.parametrize("payload", [b"", b'{"contract":', b"not-json\n"])
def test_empty_truncated_or_invalid_bars_fail_closed(
    tmp_path: Path, payload: bytes
) -> None:
    manifest = _create(tmp_path)
    path = _directory(tmp_path, manifest.snapshot_id) / "bars.jsonl"
    path.write_bytes(payload)
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    path = _directory(tmp_path, manifest.snapshot_id) / "manifest.json"
    original = path.read_text(encoding="utf-8").strip()
    path.write_text(original[:-1] + ',"row_count":4}\n', encoding="utf-8")
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).verify(manifest.snapshot_id)


def test_unsupported_schema_version_and_unknown_field_fail_closed(
    tmp_path: Path,
) -> None:
    first = _create(tmp_path / "version")
    _rewrite_manifest(
        tmp_path / "version",
        first.snapshot_id,
        lambda value: value.__setitem__("schema_version", 2),
    )
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path / "version").load(first.snapshot_id)

    second = _create(tmp_path / "field")
    _rewrite_manifest(
        tmp_path / "field",
        second.snapshot_id,
        lambda value: value.__setitem__("unknown", True),
    )
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path / "field").load(second.snapshot_id)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_version_is_type_strict(tmp_path: Path, version: object) -> None:
    manifest = _create(tmp_path)
    _rewrite_manifest(
        tmp_path,
        manifest.snapshot_id,
        lambda value: value.__setitem__("schema_version", version),
    )
    with pytest.raises(DatasetSnapshotIntegrityError):
        DatasetSnapshotStore(tmp_path).verify(manifest.snapshot_id)


def test_unknown_snapshot_and_invalid_id_are_distinct_failures(tmp_path: Path) -> None:
    unknown = "sha256:" + ("a" * 64)
    with pytest.raises(DatasetSnapshotNotFoundError):
        DatasetSnapshotStore(tmp_path).load(unknown)
    with pytest.raises(ValueError, match="sha256"):
        DatasetSnapshotStore(tmp_path).load("../escape")


def test_conflicting_preexisting_destination_is_never_overwritten(
    tmp_path: Path,
) -> None:
    manifest = _create(tmp_path)
    bars_path = _directory(tmp_path, manifest.snapshot_id) / "bars.jsonl"
    bars_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(DatasetSnapshotIntegrityError):
        _create(tmp_path)
    assert bars_path.read_text(encoding="utf-8") == "corrupt"


def test_interrupted_creation_exposes_no_snapshot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _create(tmp_path / "identity")
    calls = 0
    original = snapshot_module._write_durable

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr(snapshot_module, "_write_durable", fail_second_write)
    root = tmp_path / "interrupted"
    with pytest.raises(OSError, match="simulated interruption"):
        _create(root)
    assert not _directory(root, expected.snapshot_id).exists()
    assert list(root.glob(".tmp-*")) == []


def test_concurrent_identical_creation_is_safe_and_deterministic(
    tmp_path: Path,
) -> None:
    store = DatasetSnapshotStore(tmp_path)

    def create_one(_: int) -> str:
        return store.create(
            tuple(reversed(_bars())),
            interval_id="1h",
            timestamp_convention="bar_close",
            provenance=_provenance(),
        ).snapshot_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        identities = list(executor.map(create_one, range(24)))
    assert len(set(identities)) == 1
    assert len(list(tmp_path.glob("sha256-*"))) == 1
    assert store.verify(identities[0]).snapshot_id == identities[0]


def test_original_inputs_and_loaded_evidence_are_isolated(tmp_path: Path) -> None:
    bars = list(_bars())
    metadata = {"query": {"fields": ["ohlcv"]}}
    provenance = _provenance(metadata=metadata)
    manifest = _create(tmp_path, bars, provenance=provenance)
    bars.clear()
    metadata["query"]["fields"].append("changed")  # type: ignore[index,union-attr]

    loaded = DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)
    assert len(loaded.bars) == 4
    assert loaded.manifest.provenance.to_dict()["metadata"] == {
        "query": {"fields": ["ohlcv"]}
    }
    with pytest.raises(FrozenInstanceError):
        loaded.bars[0].close = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loaded.manifest.row_count = 0  # type: ignore[misc]


def test_load_and_verify_are_fully_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create(tmp_path)

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden during snapshot replay")

    monkeypatch.setattr(socket, "create_connection", blocked)
    store = DatasetSnapshotStore(tmp_path)
    assert store.verify(manifest.snapshot_id) == manifest
    assert store.load(manifest.snapshot_id).manifest == manifest


def test_experiment_spec_preserves_generated_snapshot_id(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    spec = ExperimentSpec(
        evaluation_protocol_id="protocol:task8:v1",
        expert_config_ids=("expert:placeholder:v1",),
        ensemble_config_id=None,
        target_definition_id="target:forward-return:v1",
        horizon_bars=1,
        data_snapshot_id=manifest.snapshot_id,
        data_cutoff_at=datetime(2026, 2, 1, tzinfo=UTC),
        universe_id="universe:task8:v1",
        cost_model_id="cost:none:v1",
        code_id="git:task8",
        config_id="config:task8",
        random_seed=8,
        trial_family_id="family:task8",
    )
    rebuilt = ExperimentSpec.from_json(spec.to_json())
    assert rebuilt.data_snapshot_id == manifest.snapshot_id
    assert DatasetSnapshotStore(tmp_path).verify(rebuilt.data_snapshot_id) == manifest


def test_loaded_snapshot_feeds_existing_task3_chronology(tmp_path: Path) -> None:
    bars = _bars(40)
    manifest = _create(tmp_path, list(reversed(bars)))
    loaded = DatasetSnapshotStore(tmp_path).load(manifest.snapshot_id)
    intervals = tuple(TimeInterval(bar.start_at, bar.event_at) for bar in loaded.bars)
    protocol_id = "protocol:task8-compatibility:v1"
    protocol = EvaluationProtocol(
        protocol_id=protocol_id,
        mode=EvaluationMode.EXPANDING,
        minimum_train_bars=5,
        train_window_bars=None,
        validation_bars=2,
        test_bars=2,
        step_bars=4,
        purge_bars=1,
        embargo_bars=1,
        final_holdout=FinalHoldout(
            intervals[32].start,
            intervals[-1].end,
            FinalHoldoutState.LOCKED,
        ),
        random_seed=8,
    )
    spec = ExperimentSpec(
        evaluation_protocol_id=protocol_id,
        expert_config_ids=("expert:placeholder:v1",),
        ensemble_config_id=None,
        target_definition_id="target:same-bar:v1",
        horizon_bars=1,
        data_snapshot_id=manifest.snapshot_id,
        data_cutoff_at=loaded.bars[-1].available_at,
        universe_id="universe:task8:v1",
        cost_model_id="cost:none:v1",
        code_id="git:task8",
        config_id="config:task8",
        random_seed=8,
        trial_family_id="family:task8",
    )
    attempt = ExperimentAttempt(
        ATTEMPT_ID,
        spec,
        datetime(2026, 2, 2, tzinfo=UTC),
    )
    plan = materialize_chronological_plan(
        attempt,
        protocol,
        intervals,
        tuple(range(32)),
    )
    assert plan.attempt.spec.data_snapshot_id == manifest.snapshot_id
    assert plan.bar_intervals == intervals
    assert plan.folds


def test_store_requires_path_and_empty_input_fails_without_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        DatasetSnapshotStore(str(tmp_path))  # type: ignore[arg-type]
    empty_root = tmp_path / "empty"
    with pytest.raises(ValueError, match="must not be empty"):
        _create(empty_root, [])
    assert not empty_root.exists()
