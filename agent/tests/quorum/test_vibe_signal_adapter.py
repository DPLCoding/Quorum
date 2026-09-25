"""Tests for Task 6's narrow Quorum-to-Vibe signal adapter."""

from __future__ import annotations

import ast
import inspect
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.quorum.adapters as adapter_package
import src.quorum.adapters.vibe_signal as adapter_module
from backtest.engines.base import BaseEngine, _align
from src.quorum import (
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    ReportingLabel,
    StaticEnsemble,
    StaticEnsembleConfig,
    StaticEnsembleResult,
)
from src.quorum.adapters import VibeSignalAdapter, validate_v0_vibe_config
from src.quorum.risk import FixedRiskPolicy, RiskPolicyConfig, RiskPolicyResult

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
INDEX = pd.date_range("2026-01-02T12:00:00Z", periods=6, freq="h")


def _frame(index: pd.DatetimeIndex = INDEX) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
        },
        index=index,
    )


def _ensemble_result(
    rows: tuple[
        tuple[
            str,
            float,
            datetime,
            datetime,
            str | None,
            str | None,
            int,
            bool,
        ],
        ...,
    ],
) -> StaticEnsembleResult:
    ensemble_config = StaticEnsembleConfig(
        (
            ExpertWeight("expert.a", "v1", 0.5),
            ExpertWeight("expert.b", "v1", 0.5),
        ),
        sell_threshold=-0.25,
        buy_threshold=0.25,
    )
    predictions: list[ExpertPrediction] = []
    for (
        asset,
        score,
        event_at,
        decision_at,
        experiment,
        split,
        horizon,
        complete,
    ) in rows:
        expert_ids = ("expert.a", "expert.b") if complete else ("expert.a",)
        predictions.extend(
            ExpertPrediction(
                expert_id=expert_id,
                expert_version="v1",
                asset=asset,
                event_at=event_at,
                available_at=event_at,
                decision_at=decision_at,
                horizon_bars=horizon,
                score=score,
                experiment_id=experiment,
                split_id=split,
            )
            for expert_id in expert_ids
        )
    return StaticEnsemble(ensemble_config).combine(ExpertResult(tuple(predictions)))


def _risk_result(
    rows: tuple[
        tuple[
            str,
            float,
            datetime,
            datetime,
            str | None,
            str | None,
            int,
            bool,
        ],
        ...,
    ],
    *,
    gross: float = 1.0,
    name: float = 0.4,
    turnover: float = 2.0,
) -> RiskPolicyResult:
    ensemble_result = _ensemble_result(rows)
    return FixedRiskPolicy(RiskPolicyConfig(gross, name, turnover)).apply(
        ensemble_result
    )


def _row(
    asset: str,
    score: float,
    event_position: int,
    *,
    decision_at: datetime | None = None,
    experiment: str | None = "exp-1",
    split: str | None = "split-1",
    horizon: int = 5,
    complete: bool = True,
) -> tuple[str, float, datetime, datetime, str | None, str | None, int, bool]:
    event_at = INDEX[event_position].to_pydatetime()
    return (
        asset,
        score,
        event_at,
        decision_at or event_at + timedelta(minutes=10),
        experiment,
        split,
        horizon,
        complete,
    )


def test_adapter_package_exports_only_public_task6_boundary() -> None:
    assert adapter_package.__all__ == [
        "VibeSignalAdapter",
        "validate_v0_vibe_config",
    ]


def test_generate_emits_exact_series_with_leading_zero_and_target_persistence() -> None:
    risk_result = _risk_result((_row("A", 0.5, 1), _row("A", -0.5, 3)))
    data_map = {"A": _frame(), "EXTRA": _frame()}
    original = data_map["A"].copy(deep=True)

    signals = VibeSignalAdapter(risk_result).generate(data_map)

    assert list(signals) == ["A"]
    assert type(signals["A"]) is pd.Series
    assert signals["A"].index.equals(data_map["A"].index)
    assert signals["A"].tolist() == [0.0, 0.2, 0.2, -0.2, -0.2, -0.2]
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in signals["A"])
    assert_frame_equal(data_map["A"], original)


def test_timezone_equivalent_event_instant_matches_existing_index_representation() -> (
    None
):
    event_ny = INDEX[1].to_pydatetime().astimezone(NEW_YORK)
    risk_result = _risk_result(
        (
            (
                "A",
                1.0,
                event_ny,
                event_ny + timedelta(minutes=10),
                "exp-1",
                "split-1",
                5,
                True,
            ),
        )
    )

    signal = VibeSignalAdapter(risk_result).generate({"A": _frame()})["A"]

    assert signal.index.equals(INDEX)
    assert signal.iloc[0] == 0.0
    assert signal.iloc[1] == pytest.approx(0.4)


def test_timestamp_matching_is_independent_of_datetime_index_storage_unit() -> None:
    microsecond_index = INDEX.as_unit("us")
    risk_result = _risk_result((_row("A", 0.5, 1),))

    signal = VibeSignalAdapter(risk_result).generate({"A": _frame(microsecond_index)})[
        "A"
    ]

    assert signal.index.equals(microsecond_index)
    assert signal.iloc[1] == pytest.approx(0.2)


def test_missing_asset_and_duplicate_signal_slot_fail_closed() -> None:
    risk_result = _risk_result((_row("A", 0.5, 1),))
    with pytest.raises(ValueError, match="missing required risk assets"):
        VibeSignalAdapter(risk_result).generate({"B": _frame()})

    duplicate_slot = _risk_result(
        (
            _row("A", 0.5, 1, experiment="exp-1", split="split-1"),
            _row(
                "A",
                -0.5,
                1,
                decision_at=INDEX[1].to_pydatetime() + timedelta(minutes=20),
                experiment="exp-2",
                split="split-1",
            ),
        )
    )
    with pytest.raises(ValueError, match="one signal slot"):
        VibeSignalAdapter(duplicate_slot).generate({"A": _frame()})


def test_incomplete_risk_group_is_never_converted_to_zero_or_partial_signals() -> None:
    incomplete = _risk_result((_row("A", 0.5, 1, complete=False),))

    assert not incomplete.rebalances[0].executable
    assert incomplete.targets[0].final_target_weight is None
    with pytest.raises(ValueError, match="incomplete group"):
        VibeSignalAdapter(incomplete).generate({"A": _frame()})


def test_event_calendar_and_next_bar_causality_failures() -> None:
    missing_event = datetime(2026, 1, 2, 12, 30, tzinfo=UTC)
    missing = _risk_result(
        (
            (
                "A",
                0.5,
                missing_event,
                missing_event + timedelta(minutes=5),
                "exp-1",
                "split-1",
                5,
                True,
            ),
        )
    )
    with pytest.raises(ValueError, match="does not map"):
        VibeSignalAdapter(missing).generate({"A": _frame()})

    no_next = _risk_result((_row("A", 0.5, len(INDEX) - 1),))
    with pytest.raises(ValueError, match="no next executable bar"):
        VibeSignalAdapter(no_next).generate({"A": _frame()})

    late = _risk_result((_row("A", 0.5, 1, decision_at=INDEX[2].to_pydatetime()),))
    with pytest.raises(ValueError, match="not before its next bar"):
        VibeSignalAdapter(late).generate({"A": _frame()})


def test_ambiguous_or_non_absolute_market_indexes_fail_closed() -> None:
    risk_result = _risk_result((_row("A", 0.5, 1),))
    duplicate_index = INDEX.insert(2, INDEX[1])
    naive_index = INDEX.tz_localize(None)

    with pytest.raises(ValueError, match="unique increasing instants"):
        VibeSignalAdapter(risk_result).generate({"A": _frame(duplicate_index)})
    with pytest.raises(ValueError, match="timezone-aware"):
        VibeSignalAdapter(risk_result).generate({"A": _frame(naive_index)})


def test_v0_execution_config_validator_accepts_only_unmodified_rebalance_path() -> None:
    valid = {
        "position_adjustment": "rebalance",
        "optimizer": None,
        "constraints": [],
        "rebalance_mask": None,
        "rebalance_tolerance": 0,
        "leverage": 1.0,
    }
    snapshot = dict(valid)

    assert validate_v0_vibe_config(valid) is None
    assert valid == snapshot

    invalid = (
        ({**valid, "position_adjustment": "hold"}, "position_adjustment"),
        ({**valid, "optimizer": "mean_variance"}, "optimizer"),
        ({**valid, "constraints": [{"max_weight": 0.2}]}, "constraints"),
        ({**valid, "rebalance_mask": "MS"}, "rebalance_mask"),
        ({**valid, "rebalance_tolerance": 0.01}, "rebalance_tolerance"),
        ({**valid, "leverage": 2.0}, "leverage"),
    )
    for config, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate_v0_vibe_config(config)
    with pytest.raises(TypeError, match="rebalance_tolerance"):
        validate_v0_vibe_config({**valid, "rebalance_tolerance": False})
    with pytest.raises(TypeError, match="leverage"):
        validate_v0_vibe_config({**valid, "leverage": True})


def test_reporting_hold_is_not_vibe_hold_mode_or_a_flat_target() -> None:
    rows = (_row("A", 0.1, 1),)
    ensemble_result = _ensemble_result(rows)
    risk_result = FixedRiskPolicy(RiskPolicyConfig(1.0, 0.4, 2.0)).apply(
        ensemble_result
    )
    target = risk_result.targets[0]

    assert ensemble_result.decisions[0].label is ReportingLabel.HOLD
    assert target.ensemble_score == pytest.approx(0.1)
    assert target.final_target_weight == pytest.approx(0.04)
    assert VibeSignalAdapter(risk_result).generate({"A": _frame()})["A"].iloc[
        1
    ] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="position_adjustment"):
        validate_v0_vibe_config({"position_adjustment": "hold"})


def test_base_align_preserves_policy_compliant_weights_and_shifts_on_own_calendar() -> (
    None
):
    risk_result = _risk_result(
        (_row("A", 1.0, 1), _row("B", -1.0, 1)), gross=0.8, name=0.4
    )
    data_map = {"A": _frame(), "B": _frame()}
    signals = VibeSignalAdapter(risk_result).generate(data_map)

    _, _, _, target_positions, _ = _align(data_map, signals, ["A", "B"])

    assert target_positions.loc[INDEX[1], "A"] == 0.0
    assert target_positions.loc[INDEX[1], "B"] == 0.0
    assert target_positions.loc[INDEX[2], "A"] == pytest.approx(0.4)
    assert target_positions.loc[INDEX[2], "B"] == pytest.approx(-0.4)
    assert target_positions.loc[INDEX[2]].abs().sum() == pytest.approx(0.8)


class _ExecutionEngine(BaseEngine):
    def can_execute(self, symbol, direction, bar):  # noqa: ANN001, ANN201
        return True

    def round_size(self, raw_size, price):  # noqa: ANN001, ANN201
        return float(raw_size)

    def calc_commission(self, size, price, direction, is_open):  # noqa: ANN001, ANN201
        return 0.0

    def apply_slippage(self, price, direction):  # noqa: ANN001, ANN201
        return float(price)


class _ZeroLotEngine(_ExecutionEngine):
    def round_size(self, raw_size, price):  # noqa: ANN001, ANN201
        return 0.0


def _execute_with_engine(
    engine: BaseEngine, risk_result: RiskPolicyResult
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    data_map = {"A": _frame()}
    signals = VibeSignalAdapter(risk_result).generate(data_map)
    dates, close, close_value, targets, _ = _align(data_map, signals, ["A"])
    engine._execute_bars(
        dates,
        data_map,
        close,
        targets,
        ["A"],
        close_val_df=close_value,
    )
    return signals, targets


def test_signal_to_fill_causality_executes_at_next_bar_not_event_bar() -> None:
    risk_result = _risk_result((_row("A", 1.0, 0),))
    engine = _ExecutionEngine(
        {"initial_cash": 1_000.0, "leverage": 1.0, "position_adjustment": "rebalance"}
    )

    signals, targets = _execute_with_engine(engine, risk_result)

    assert risk_result.targets[0].decision_at < INDEX[1].to_pydatetime()
    assert signals["A"].iloc[0] == pytest.approx(0.4)
    assert targets.iloc[0, 0] == 0.0
    assert targets.iloc[1, 0] == pytest.approx(0.4)
    assert engine.fill_records[0].timestamp == INDEX[1]
    assert engine.actual_position_snapshots[0][1]["A"] == 0.0
    assert engine.actual_position_snapshots[1][1]["A"] == pytest.approx(0.4)


def test_nonzero_requested_target_remains_distinct_from_unfillable_lot() -> None:
    risk_result = _risk_result((_row("A", 1.0, 0),))
    engine = _ZeroLotEngine(
        {"initial_cash": 1_000.0, "leverage": 1.0, "position_adjustment": "rebalance"}
    )

    signals, targets = _execute_with_engine(engine, risk_result)

    assert risk_result.targets[0].final_target_weight == pytest.approx(0.4)
    assert signals["A"].iloc[0] == pytest.approx(0.4)
    assert targets.iloc[1, 0] == pytest.approx(0.4)
    assert engine.fill_records == []
    assert engine.positions == {}
    assert all(snapshot[1]["A"] == 0.0 for snapshot in engine.actual_position_snapshots)
    assert engine.plan_rejections[("A", "zero_size")] > 0


def test_adapter_module_has_only_allowed_dependencies() -> None:
    tree = ast.parse(inspect.getsource(adapter_module))
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
        "bisect",
        "collections",
        "dataclasses",
        "math",
        "numbers",
        "typing",
    }
    assert all(
        dependency == "pandas"
        or dependency == "src.quorum.risk"
        or dependency.split(".", 1)[0] in allowed_standard_roots
        for dependency in dependencies
    )
