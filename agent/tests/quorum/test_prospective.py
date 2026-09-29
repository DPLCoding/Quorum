"""Tests for the broker-free prospective prediction ledger."""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import src.quorum.prospective as prospective
from src.quorum.prospective import (
    backup_study,
    declare_study,
    evaluate_study,
    preview_declaration,
    record_day,
    restore_study,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
DATES = pd.bdate_range("2025-01-02", periods=400)
SYMBOLS = ["AAA.US", "BBB.US"]


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(DATES))))
    open_ = close * np.exp(rng.normal(0.0, 0.004, len(DATES)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, len(DATES)).astype(float),
        },
        index=pd.DatetimeIndex(DATES, name="trade_date"),
    )


FRAMES = {"AAA.US": _frame(1), "BBB.US": _frame(2)}


class _Loader:
    """Serves bars dated up to ``end_date``, including today's unfinished bar."""

    name = "test-loader"

    def fetch(
        self, codes, start_date, end_date, *, interval="1D", fields=None
    ):  # noqa: ANN001, ANN201
        end = pd.Timestamp(end_date)
        return {c: FRAMES[c][FRAMES[c].index <= end].copy() for c in codes}


class _BrokenLoader(_Loader):
    def fetch(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise ConnectionError("provider unreachable")


def _at(position: int, hour: int, minute: int = 0) -> datetime:
    day = DATES[position]
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)


def _study(tmp_path: Path, study_id: str = "study-1") -> Path:
    workspace = tmp_path / "ws"
    declare_study(
        workspace, study_id, SYMBOLS, declared_at=_at(299, 18), allow_dirty_code=True
    )
    return workspace


def _record(workspace: Path, position: int, hour: int = 17, minute: int = 0) -> dict:
    return record_day(
        workspace, "study-1", loader=_Loader(), now=_at(position, hour, minute)
    )


# ------------------------------------------------------------ declaration


def test_preview_is_exactly_what_declaration_freezes(tmp_path: Path) -> None:
    preview = preview_declaration("study-1", SYMBOLS, declared_at=_at(299, 18))
    workspace = _study(tmp_path)
    frozen = json.loads(
        (workspace / "prospective" / "study-1" / "study.json").read_text("utf-8")
    )
    assert frozen == preview
    assert set(frozen) >= {
        "universe",
        "data",
        "information_cutoff",
        "experts",
        "candidates",
        "construction",
        "risk",
        "execution",
        "costs",
        "initial_state",
        "missing_data",
        "abstention",
        "corporate_actions",
        "non_trading_days",
        "missed_runs",
        "evaluation",
        "horizon",
        "code",
        "broker",
        "external_reference",
        "confirmatory_test",
    }
    assert frozen["broker"] is None
    assert frozen["risk"]["max_abs_weight_per_asset"] == pytest.approx(1.0)  # 2/N
    assert frozen["candidates"]["equal_weight_control"]["predictor"] == "constant:0"
    declared = datetime.fromisoformat(frozen["declared_at"]).astimezone(NY).date()
    assert (
        frozen["horizon"]["end_date"]
        == declared.replace(year=declared.year + 2).isoformat()
    )
    assert frozen["confirmatory_test"]["method"]["name"].startswith("Ledoit-Wolf")
    with pytest.raises(FileExistsError):
        declare_study(workspace, "study-1", SYMBOLS, allow_dirty_code=True)


def test_declaration_refuses_uncommitted_code(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        prospective, "_code_identity", lambda: {"commit": "abc", "clean": False}
    )
    with pytest.raises(ValueError, match="uncommitted"):
        declare_study(tmp_path / "ws", "study-x", SYMBOLS)


# --------------------------------------------------------------- recording


def test_record_day_uses_only_closed_bars_and_no_ops_without_a_new_session(
    tmp_path: Path,
) -> None:
    workspace = _study(tmp_path)

    midday = _record(workspace, 300, 12)
    assert midday["decision_date"] == str(DATES[299].date())
    assert midday["unclosed_bars_excluded"] == len(SYMBOLS)
    assert _record(workspace, 300, 12, 30) == {
        "status": "no_new_session",
        "latest_closed_session": str(DATES[299].date()),
    }

    evening = _record(workspace, 300)
    assert evening["decision_date"] == str(DATES[300].date())
    benchmark = evening["candidates"]["equal_weight_control"]["targets"]
    assert benchmark == {s: pytest.approx(1.0 / len(SYMBOLS)) for s in SYMBOLS}
    for candidate in evening["candidates"].values():
        assert all(w >= 0.0 for w in candidate["targets"].values())
        assert sum(candidate["targets"].values()) <= 1.0 + 1e-12


def test_provider_failure_writes_a_skip_record_instead_of_crashing(
    tmp_path: Path,
) -> None:
    workspace = _study(tmp_path)
    result = record_day(workspace, "study-1", loader=_BrokenLoader(), now=_at(300, 17))
    assert result["type"] == "skip"
    assert "ConnectionError" in result["reason"]


def test_recording_stops_after_the_declared_end_date(tmp_path: Path) -> None:
    workspace = _study(tmp_path)
    study = json.loads(
        (workspace / "prospective" / "study-1" / "study.json").read_text("utf-8")
    )
    end = datetime.fromisoformat(study["horizon"]["end_date"] + "T12:00:00-04:00")
    after_end = end + timedelta(days=1)
    ledger = workspace / "prospective" / "study-1" / "ledger.jsonl"
    before = ledger.read_text("utf-8")
    assert record_day(workspace, "study-1", loader=_Loader(), now=after_end) == {
        "status": "study_complete"
    }
    assert ledger.read_text("utf-8") == before


def test_an_abstaining_asset_yields_no_targets_and_does_not_break_later_days() -> None:
    study = preview_declaration("s", SYMBOLS, declared_at=_at(299, 18))
    days = [_at(300 + k, 16).astimezone(UTC) for k in range(3)]
    complete = {"AAA.US": 0.2, "BBB.US": -0.2}
    partial = {"AAA.US": 0.2, "BBB.US": None}

    assert prospective._tilt_targets(study, "x", [(days[0], partial)]) is None
    later = prospective._tilt_targets(
        study, "x", [(days[0], partial), (days[1], complete), (days[2], complete)]
    )
    assert set(later) == set(SYMBOLS)


# -------------------------------------------------------------- evaluation


def test_evaluation_replays_fills_and_reports_late_missed_and_pending(
    tmp_path: Path,
) -> None:
    workspace = _study(tmp_path)
    for position in range(300, 306):
        _record(workspace, position)
    _record(workspace, 307, 10)  # after the 09:30 open that executes it: late
    _record(workspace, 309)  # sessions 307 and 308 never got a decision: missed

    report = evaluate_study(workspace, "study-1")
    assert report["decisions"] == 8
    assert report["late_decisions"] == [str(DATES[306].date())]
    assert report["missed_sessions"] == [str(DATES[307].date()), str(DATES[308].date())]
    assert report["pending_decisions"] == [str(DATES[309].date())]
    assert report["status"] == "interim"

    ledger = workspace / "prospective" / "study-1" / "ledger.jsonl"
    targets = {
        r["decision_date"]: r["candidates"]["equal_weight_control"]["targets"]
        for r in map(json.loads, ledger.read_text("utf-8").splitlines())
        if r.get("type") == "decision"
    }
    # Independent daily marking from the first valid execution (open of 301)
    # to the last observed open (309). Executions at 301..306 only; 306's
    # decision is late, 307-308 missed, 309 pending: the portfolio just drifts.
    opens = {s: FRAMES[s]["open"].to_numpy() for s in SYMBOLS}
    values, cash = {s: 0.0 for s in SYMBOLS}, 1.0
    for day in range(301, 310):
        if day > 301:
            values = {
                s: v * opens[s][day] / opens[s][day - 1] for s, v in values.items()
            }
        equity = sum(values.values()) + cash
        if 301 <= day <= 306:
            tau = targets[str(DATES[day - 1].date())]
            trade = sum(abs(tau[s] - values[s] / equity) for s in SYMBOLS)
            equity *= 1.0 - 0.0005 * trade
            values = {s: tau[s] * equity for s in SYMBOLS}
            cash = equity - sum(values.values())
    control = report["candidates"]["equal_weight_control"]
    assert control["days"] == 9
    assert control["executions"] == 6
    assert control["total_return"] == pytest.approx(sum(values.values()) + cash - 1.0)
    assert control["avg_gross_exposure"] + control["avg_cash_weight"] == pytest.approx(
        1.0
    )
    vs = report["candidates"]["vol_regime_tilt"]["vs_control"]
    assert vs["days"] == 9
    assert report["confirmatory_test"]["result"] == "insufficient_observations"
    assert report["external_reference"] is None  # SPY.US is not in this universe


def test_tampered_ledger_fails_closed(tmp_path: Path) -> None:
    workspace = _study(tmp_path)
    _record(workspace, 300)
    ledger = workspace / "prospective" / "study-1" / "ledger.jsonl"
    lines = ledger.read_text("utf-8").splitlines()
    lines[-1] = lines[-1].replace('"equal_weight_control"', '"equal_weight_contrl"')
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chain"):
        evaluate_study(workspace, "study-1")


# ------------------------------------------------------------------ backup


def test_backup_mirrors_detects_rollback_and_restores(tmp_path: Path) -> None:
    workspace = _study(tmp_path)
    backup = tmp_path / "backup"
    _record(workspace, 300)
    backup_study(workspace, "study-1", backup)
    _record(workspace, 301)
    backup_study(workspace, "study-1", backup)

    restored = tmp_path / "restored"
    restore_study(backup, "study-1", restored)
    assert evaluate_study(restored, "study-1") == evaluate_study(workspace, "study-1")
    with pytest.raises(FileExistsError):
        restore_study(backup, "study-1", restored)

    # A primary that lost records the backup already holds is a rollback.
    ledger = workspace / "prospective" / "study-1" / "ledger.jsonl"
    lines = ledger.read_text("utf-8").splitlines()
    ledger.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rollback"):
        backup_study(workspace, "study-1", backup)


def test_module_cannot_reach_brokers_or_live_trading() -> None:
    tree = ast.parse(inspect.getsource(prospective))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden = ("src.trading", "src.live", "alpaca", "ib_insync", "ccxt", "futu")
    assert not [m for m in imported if m and m.startswith(forbidden)]


# ------------------------------------------------------------ weight drift


def test_weights_drift_and_the_next_rebalance_trades_from_drifted_weights() -> None:
    from src.quorum.prospective import _simulate_portfolio

    opens = {"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0]}
    half = {"A": 0.5, "B": 0.5}
    run = _simulate_portfolio(opens, {0: half, 2: half}, rate=0.0005)

    # Day 0: all cash -> 50/50, paying 5 bp on turnover 1.0.
    assert run["turnover"][0] == pytest.approx(1.0)
    assert run["equity"][0] == pytest.approx(1.0 - 0.0005)
    # Day 1: no execution; A rose 10%, so weights drift (no free rebalance).
    drifted = {"A": 0.55 / 1.05, "B": 0.50 / 1.05}
    assert run["weights"][1] == pytest.approx(drifted)
    # Day 2: trades from the drifted weights back to 50/50; cost on that trade.
    assert run["pre_trade_weights"][2] == pytest.approx(drifted)
    trade = abs(0.5 - drifted["A"]) + abs(0.5 - drifted["B"])
    assert run["turnover"][2] == pytest.approx(trade)
    pre = (1.0 - 0.0005) * 1.05
    assert run["equity"][2] == pytest.approx(pre * (1.0 - 0.0005 * trade))
    assert run["weights"][2] == pytest.approx(half)


def test_cash_and_equity_evolve_consistently() -> None:
    from src.quorum.prospective import _simulate_portfolio

    opens = {"A": [100.0, 120.0, 90.0], "B": [100.0, 80.0, 100.0]}
    run = _simulate_portfolio(opens, {0: {"A": 0.3, "B": 0.3}}, rate=0.0005)

    equity0 = 1.0 - 0.0005 * 0.6
    cash = 0.4 * equity0  # cash earns nothing and never drifts
    for day, (a, b) in enumerate(zip(opens["A"], opens["B"])):
        risky = 0.3 * equity0 * (a / 100.0) + 0.3 * equity0 * (b / 100.0)
        assert run["equity"][day] == pytest.approx(risky + cash)
        assert run["cash"][day] == pytest.approx(cash / (risky + cash))
        assert run["gross"][day] == pytest.approx(risky / (risky + cash))
    compounded = np.prod([1.0 + r for r in run["returns"]])
    assert compounded == pytest.approx(run["equity"][-1])


def test_missed_late_and_abstaining_days_hold_the_drifted_portfolio() -> None:
    from src.quorum.prospective import _simulate_portfolio

    opens = {"A": [100.0, 105.0, 115.0, 120.0], "B": [100.0, 95.0, 90.0, 99.0]}
    start = {"A": 0.5, "B": 0.5}
    held = _simulate_portfolio(opens, {0: start}, rate=0.0005)
    # An abstaining execution (None) trades nothing: identical to no execution.
    abstained = _simulate_portfolio(opens, {0: start, 2: None}, rate=0.0005)
    assert abstained["equity"] == pytest.approx(held["equity"])
    assert abstained["turnover"][2] == 0.0
    # Buy and hold: final weights follow relative prices exactly.
    a, b = 0.5 * 120.0 / 100.0, 0.5 * 99.0 / 100.0
    assert held["weights"][3] == pytest.approx({"A": a / (a + b), "B": b / (a + b)})


def test_control_is_one_eighth_each_and_fully_invested_after_the_ramp() -> None:
    eight = [f"S{i}.US" for i in range(8)]
    study = preview_declaration("s8", eight, declared_at=_at(299, 18))
    assert study["risk"]["max_abs_weight_per_asset"] == pytest.approx(0.25)  # 2/8
    days = [_at(300 + k, 16).astimezone(UTC) for k in range(3)]
    zero = {s: 0.0 for s in eight}

    first = prospective._tilt_targets(study, "c", [(days[0], zero)])
    assert first == {s: pytest.approx(0.0625) for s in eight}  # turnover cap 0.5
    second = prospective._tilt_targets(study, "c", [(days[0], zero), (days[1], zero)])
    assert second == {s: pytest.approx(0.125) for s in eight}
    assert sum(second.values()) == pytest.approx(1.0)
    third = prospective._tilt_targets(study, "c", [(d, zero) for d in days])
    assert third == second


ALWAYS_REPORTED = {
    "n",
    "sharpe_difference_annualized",
    "studentized_statistic",
    "p_value_one_sided",
    "result",
}


def test_confirmatory_outcome_mapping_is_explicit_and_exact() -> None:
    study = preview_declaration("s", SYMBOLS, declared_at=_at(299, 18))
    test = study["confirmatory_test"]
    assert test["outcome_mapping"] == [
        {"when": "n < 250", "outcome": "insufficient_observations"},
        {"when": "n >= 250 and p_value_one_sided <= 0.05", "outcome": "supported"},
        {"when": "n >= 250 and p_value_one_sided > 0.05", "outcome": "not_supported"},
    ]
    assert set(test["always_reported"]) == ALWAYS_REPORTED
    outcome = prospective._confirmatory_outcome
    assert outcome(test, 249, 0.0001) == "insufficient_observations"
    assert outcome(test, 250, 0.05) == "supported"
    assert outcome(test, 250, 0.0500001) == "not_supported"
    assert outcome(test, 500, 0.9) == "not_supported"


def test_confirmatory_results_are_reported_even_when_insufficient(
    tmp_path: Path,
) -> None:
    workspace = _study(tmp_path)
    for position in range(300, 311):
        _record(workspace, position)
    confirmatory = evaluate_study(workspace, "study-1")["confirmatory_test"]
    assert confirmatory["result"] == "insufficient_observations"
    assert ALWAYS_REPORTED <= set(confirmatory)
    assert confirmatory["n"] == 10
    assert all(confirmatory[key] is not None for key in ALWAYS_REPORTED)
