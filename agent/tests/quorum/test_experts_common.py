"""Shared prepared-data, causality, and boundary tests for Task 4 experts."""

from __future__ import annotations

import ast
import inspect
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import src.quorum.experts.common as common_module
import src.quorum.experts.mean_reversion as mean_reversion_module
import src.quorum.experts.momentum as momentum_module
import src.quorum.experts.trend as trend_module
from src.quorum import (
    ExpertProtocol,
    MeanReversionExpert,
    MomentumExpert,
    PredictionContext,
    TrendExpert,
)

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")

EXPERT_CASES = (
    (MomentumExpert, 21, "quorum.momentum", "quorum:momentum:v0"),
    (TrendExpert, 50, "quorum.trend", "quorum:trend:v0"),
    (
        MeanReversionExpert,
        20,
        "quorum.mean_reversion",
        "quorum:mean-reversion:v0",
    ),
)

FROZEN_SPEC_CASES = (
    (
        MomentumExpert,
        21,
        {
            "expert_id": "quorum.momentum",
            "expert_version": "v0.1.0",
            "config_id": "quorum:momentum:v0",
            "LOOKBACK": 20,
            "RETURN_SCALE": 0.10,
        },
    ),
    (
        TrendExpert,
        50,
        {
            "expert_id": "quorum.trend",
            "expert_version": "v0.1.0",
            "config_id": "quorum:trend:v0",
            "FAST_WINDOW": 10,
            "SLOW_WINDOW": 50,
            "SPREAD_SCALE": 0.05,
        },
    ),
    (
        MeanReversionExpert,
        20,
        {
            "expert_id": "quorum.mean_reversion",
            "expert_version": "v0.1.0",
            "config_id": "quorum:mean-reversion:v0",
            "WINDOW": 20,
            "ZSCORE_SCALE": 3.0,
        },
    ),
)


def _prepared(
    closes: list[object],
    *,
    asset: object = "ASSET-A",
    events: list[object] | None = None,
    available: list[object] | None = None,
) -> dict[str, object]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    event_values = events or [
        base + timedelta(days=index) for index in range(len(closes))
    ]
    available_values = available or list(event_values)
    return {
        "asset": asset,
        "close": closes,
        "event_at": event_values,
        "available_at": available_values,
    }


def _context(data: dict[str, object], **changes: Any) -> PredictionContext:
    events = data["event_at"]
    available = data["available_at"]
    assert isinstance(events, list) and isinstance(available, list)
    values: dict[str, object] = {
        "event_at": events[-1],
        "available_at": max(
            available,
            key=lambda value: value.astimezone(UTC),  # type: ignore[union-attr]
        ),
        "decision_at": max(
            available,
            key=lambda value: value.astimezone(UTC),  # type: ignore[union-attr]
        ),
        "horizon_bars": 5,
        "experiment_id": "exp_task4",
        "split_id": "split_task4",
    }
    values.update(changes)
    return PredictionContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expert_type", "minimum", "expert_id", "config_id"), EXPERT_CASES
)
def test_expert_protocol_identity_and_frozen_constructor(
    expert_type: type[object], minimum: int, expert_id: str, config_id: str
) -> None:
    del minimum
    expert = expert_type()
    assert isinstance(expert, ExpertProtocol)
    assert expert.expert_id == expert_id  # type: ignore[attr-defined]
    assert expert.expert_version == "v0.1.0"  # type: ignore[attr-defined]
    assert expert.config_id == config_id  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        expert_type(window=7)
    with pytest.raises((AttributeError, TypeError)):
        expert.config_id = "changed"  # type: ignore[attr-defined,misc]


@pytest.mark.parametrize(
    ("expert_type", "kwargs"),
    (
        (MomentumExpert, {"window": 7}),
        (TrendExpert, {"fast_window": 5}),
        (MeanReversionExpert, {"window": 10}),
    ),
)
def test_experts_expose_no_constructor_tuning_knobs(
    expert_type: type[object], kwargs: dict[str, int]
) -> None:
    with pytest.raises(TypeError):
        expert_type(**kwargs)


@pytest.mark.parametrize(("expert_type", "minimum", "spec"), FROZEN_SPEC_CASES)
def test_every_scientific_identity_and_parameter_rejects_runtime_mutation(
    expert_type: type[object], minimum: int, spec: dict[str, object]
) -> None:
    data = _prepared([100.0 + index for index in range(minimum)])
    context = _context(data)
    baseline = expert_type().predict(data, context).to_json()  # type: ignore[attr-defined]

    for name, expected in spec.items():
        replacement: object = "changed" if isinstance(expected, str) else 999
        assert getattr(expert_type, name) == expected
        with pytest.raises(AttributeError, match="frozen V0 attribute"):
            setattr(expert_type, name, replacement)
        with pytest.raises(AttributeError, match="frozen V0 attribute"):
            delattr(expert_type, name)

        expert = expert_type()
        with pytest.raises((AttributeError, TypeError)):
            setattr(expert, name, replacement)
        assert getattr(expert, name) == expected

    with pytest.raises(AttributeError, match="frozen V0 attribute"):
        expert_type._FROZEN_SCIENTIFIC_ATTRIBUTES = frozenset()  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type(f"Changed{expert_type.__name__}", (expert_type,), {})
    assert expert_type().predict(data, context).to_json() == baseline  # type: ignore[attr-defined]


def test_experts_package_exports_only_three_public_experts() -> None:
    import src.quorum.experts as experts_package

    assert experts_package.__all__ == [
        "MeanReversionExpert",
        "MomentumExpert",
        "TrendExpert",
    ]


@pytest.mark.parametrize(
    ("expert_type", "minimum", "expert_id", "config_id"), EXPERT_CASES
)
def test_prediction_propagates_identity_context_and_no_fake_probability(
    expert_type: type[object], minimum: int, expert_id: str, config_id: str
) -> None:
    closes = [float(index + 100) for index in range(minimum)]
    data = _prepared(closes)
    event = datetime(2024, 4, 1, 9, 30, tzinfo=NEW_YORK)
    available = datetime(2024, 4, 1, 9, 31, tzinfo=NEW_YORK)
    decision = datetime(2024, 4, 1, 9, 32, tzinfo=NEW_YORK)
    events = data["event_at"]
    availability = data["available_at"]
    assert isinstance(events, list) and isinstance(availability, list)
    events[-1] = event
    availability[-1] = available
    context = _context(
        data,
        event_at=event,
        available_at=available,
        decision_at=decision,
    )

    result = expert_type().predict(data, context)  # type: ignore[attr-defined]

    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.expert_id == expert_id
    assert prediction.expert_version == "v0.1.0"
    assert prediction.asset == "ASSET-A"
    assert prediction.event_at is context.event_at
    assert prediction.available_at is context.available_at
    assert prediction.decision_at is context.decision_at
    assert prediction.horizon_bars == 5
    assert prediction.experiment_id == "exp_task4"
    assert prediction.split_id == "split_task4"
    assert prediction.probability_up is None
    assert prediction.confidence is None
    assert prediction.metadata["config_id"] == config_id
    assert prediction.event_at.utcoffset() == event.utcoffset()


@pytest.mark.parametrize(
    ("expert_type", "minimum", "expert_id", "config_id"), EXPERT_CASES
)
def test_asset_rename_and_positive_rescaling_do_not_change_score(
    expert_type: type[object], minimum: int, expert_id: str, config_id: str
) -> None:
    del expert_id, config_id
    closes = [100.0 + index for index in range(minimum)]
    original = _prepared(closes)
    renamed = _prepared(closes.copy(), asset="ASSET-B")
    scaled = _prepared([value * 17.0 for value in closes])

    first = expert_type().predict(original, _context(original)).predictions[0]  # type: ignore[attr-defined]
    second = expert_type().predict(renamed, _context(renamed)).predictions[0]  # type: ignore[attr-defined]
    third = expert_type().predict(scaled, _context(scaled)).predictions[0]  # type: ignore[attr-defined]

    assert first.asset == "ASSET-A"
    assert second.asset == "ASSET-B"
    assert second.score == pytest.approx(first.score)
    assert third.score == pytest.approx(first.score)


@pytest.mark.parametrize(
    ("expert_type", "minimum", "expert_id", "config_id"), EXPERT_CASES
)
def test_missing_close_outside_active_window_is_irrelevant(
    expert_type: type[object], minimum: int, expert_id: str, config_id: str
) -> None:
    del expert_id, config_id
    valid_closes: list[object] = [80.0, 90.0] + [
        100.0 + index for index in range(minimum)
    ]
    missing_closes = valid_closes.copy()
    missing_closes[0] = None
    valid = _prepared(valid_closes)
    missing = _prepared(missing_closes)

    valid_prediction = expert_type().predict(valid, _context(valid)).predictions[0]  # type: ignore[attr-defined]
    missing_prediction = expert_type().predict(missing, _context(missing)).predictions[0]  # type: ignore[attr-defined]

    assert missing_prediction.score == valid_prediction.score
    assert missing_prediction.metadata == valid_prediction.metadata


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_close_in_active_window_produces_no_evidence(missing: object) -> None:
    closes: list[object] = [100.0 + index for index in range(21)]
    closes[-5] = missing
    data = _prepared(closes)
    assert MomentumExpert().predict(data, _context(data)).predictions == ()


@pytest.mark.parametrize("value", [True, "100", object()])
def test_non_numeric_and_boolean_close_values_are_rejected(value: object) -> None:
    closes: list[object] = [100.0] * 21
    closes[3] = value
    data = _prepared(closes)
    with pytest.raises(TypeError, match="close values"):
        MomentumExpert().predict(data, _context(data))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_close_values_are_rejected(value: float) -> None:
    closes = [100.0] * 21
    closes[3] = value
    data = _prepared(closes)
    with pytest.raises(ValueError, match="close values"):
        MomentumExpert().predict(data, _context(data))


@pytest.mark.parametrize("asset", ["", " ", " ASSET", "ASSET "])
def test_asset_must_be_nonempty_and_trimmed(asset: str) -> None:
    data = _prepared([100.0] * 21, asset=asset)
    with pytest.raises(ValueError, match="asset"):
        MomentumExpert().predict(data, _context(data))


def test_asset_must_be_a_string() -> None:
    data = _prepared([100.0] * 21, asset=123)
    with pytest.raises(TypeError, match="asset"):
        MomentumExpert().predict(data, _context(data))


def test_prepared_data_requires_exact_fields_and_rejects_label_input() -> None:
    data = _prepared([100.0] * 21)
    context = _context(data)
    del data["close"]
    with pytest.raises(ValueError, match="missing"):
        MomentumExpert().predict(data, context)

    data = _prepared([100.0] * 21)
    data["future_target_returns"] = [0.1]
    with pytest.raises(ValueError, match="unknown"):
        MomentumExpert().predict(data, _context(data))


def test_prepared_data_and_context_types_fail_closed() -> None:
    data = _prepared([100.0] * 21)
    context = _context(data)
    with pytest.raises(TypeError, match="prepared_data must be a mapping"):
        MomentumExpert().predict([data], context)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="context must be a PredictionContext"):
        MomentumExpert().predict(data, object())  # type: ignore[arg-type]


def test_prepared_data_sequences_must_be_nonempty_sequences_of_equal_length() -> None:
    data = _prepared([100.0] * 21)
    context = _context(data)
    data["close"] = 100.0
    with pytest.raises(TypeError, match="sequence"):
        MomentumExpert().predict(data, context)

    data = _prepared([100.0] * 21)
    data["available_at"] = data["available_at"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="equal lengths"):
        MomentumExpert().predict(data, _context(_prepared([100.0] * 21)))

    empty = _prepared([])
    context = PredictionContext(
        event_at=datetime(2024, 1, 1, tzinfo=UTC),
        available_at=datetime(2024, 1, 1, tzinfo=UTC),
        decision_at=datetime(2024, 1, 1, tzinfo=UTC),
        horizon_bars=1,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        MomentumExpert().predict(empty, context)


def test_timestamp_rows_must_be_aware_datetimes() -> None:
    data = _prepared([100.0] * 21)
    events = data["event_at"]
    assert isinstance(events, list)
    events[0] = datetime(2024, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        MomentumExpert().predict(data, _context(data))

    data = _prepared([100.0] * 21)
    availability = data["available_at"]
    assert isinstance(availability, list)
    availability[0] = "2024-01-01"
    with pytest.raises(TypeError, match="datetime"):
        MomentumExpert().predict(data, _context(_prepared([100.0] * 21)))


def test_post_decision_availability_fails_closed() -> None:
    data = _prepared([100.0] * 21)
    context = _context(data)
    availability = data["available_at"]
    assert isinstance(availability, list)
    availability[0] = context.decision_at + timedelta(seconds=1)
    with pytest.raises(ValueError, match="later than context.decision_at"):
        MomentumExpert().predict(data, context)


def test_terminal_event_and_latest_availability_must_bind_to_context() -> None:
    data = _prepared([100.0] * 21)
    context = _context(
        data,
        event_at=datetime(2030, 1, 1, tzinfo=UTC),
        decision_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="final event_at"):
        MomentumExpert().predict(data, context)

    data = _prepared([100.0] * 21)
    context = _context(
        data,
        available_at=data["available_at"][-2],  # type: ignore[index]
        decision_at=data["available_at"][-1],  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="latest available_at"):
        MomentumExpert().predict(data, context)


def test_future_event_row_cannot_be_appended_to_an_earlier_prediction() -> None:
    data = _prepared([100.0] * 21)
    context = _context(data)
    events = data["event_at"]
    availability = data["available_at"]
    closes = data["close"]
    assert isinstance(events, list)
    assert isinstance(availability, list)
    assert isinstance(closes, list)
    events.append(events[-1] + timedelta(days=1))
    availability.append(availability[-1])
    closes.append(999.0)
    with pytest.raises(ValueError, match="final event_at"):
        MomentumExpert().predict(data, context)


def test_dst_fallback_chronology_uses_actual_instants() -> None:
    first = datetime(2024, 11, 3, 1, 50, tzinfo=NEW_YORK, fold=0)
    second = datetime(2024, 11, 3, 1, 10, tzinfo=NEW_YORK, fold=1)
    accepted = _prepared(
        [100.0, 101.0],
        events=[first, second],
        available=[first, second],
    )
    assert MomentumExpert().predict(accepted, _context(accepted)).predictions == ()

    later = datetime(2024, 11, 3, 1, 10, tzinfo=NEW_YORK, fold=1)
    earlier = datetime(2024, 11, 3, 1, 50, tzinfo=NEW_YORK, fold=0)
    availability = datetime(2024, 11, 3, 1, 0, tzinfo=NEW_YORK, fold=0)
    rejected = _prepared(
        [100.0, 101.0],
        events=[later, earlier],
        available=[availability, availability],
    )
    context = _context(
        rejected,
        available_at=availability,
        decision_at=datetime(2024, 11, 3, 1, 30, tzinfo=NEW_YORK, fold=1),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        MomentumExpert().predict(rejected, context)


def test_timezone_equivalent_terminal_bindings_are_accepted_and_output_context_wins() -> (
    None
):
    data = _prepared([100.0 + index for index in range(21)])
    events = data["event_at"]
    available = data["available_at"]
    assert isinstance(events, list) and isinstance(available, list)
    terminal_utc = events[-1]
    assert isinstance(terminal_utc, datetime)
    terminal_new_york = terminal_utc.astimezone(NEW_YORK)
    context = _context(
        data,
        event_at=terminal_new_york,
        available_at=terminal_new_york,
        decision_at=terminal_new_york,
    )
    prediction = MomentumExpert().predict(data, context).predictions[0]
    assert prediction.event_at is terminal_new_york
    assert prediction.available_at is terminal_new_york


def test_repeated_serialization_is_deterministic_and_input_is_not_mutated() -> None:
    closes = [100.0 + index for index in range(21)]
    data = _prepared(closes)
    events = list(data["event_at"])  # type: ignore[arg-type]
    available = list(data["available_at"])  # type: ignore[arg-type]
    context = _context(data)

    first = MomentumExpert().predict(data, context)
    second = MomentumExpert().predict(data, context)

    assert first.to_json() == second.to_json()
    assert data["close"] == closes
    assert data["event_at"] == events
    assert data["available_at"] == available
    assert context == _context(data)


def test_clamp_rejects_nonfinite_internal_scores() -> None:
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            common_module._clamp_score(value)


def test_expert_modules_have_only_allowed_dependencies() -> None:
    allowed_nonstdlib = {
        "pandas",
        "src.factors.zoo.qlib158.roc20",
        "src.quorum.contracts",
        "src.quorum.experts.common",
        "src.quorum.experts.mean_reversion",
        "src.quorum.experts.momentum",
        "src.quorum.experts.trend",
    }
    for module in (
        common_module,
        mean_reversion_module,
        momentum_module,
        trend_module,
    ):
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
            dependency.split(".", 1)[0]
            in {
                "__future__",
                "collections",
                "dataclasses",
                "datetime",
                "math",
                "numbers",
                "statistics",
                "typing",
            }
            or dependency in allowed_nonstdlib
            for dependency in dependencies
        )
