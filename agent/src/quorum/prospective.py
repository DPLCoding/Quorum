"""Broker-free prospective prediction ledger (Quorum roadmap 0.9).

A study's declaration freezes every analytical choice before its first
decision, and the code reads its parameters from that declaration. Each run
after the close scores only bars that have closed, sizes them through the
frozen risk policy, and appends one hash-chained decision stamped with its
wall-clock ``recorded_at``. Evaluation verifies the chain and replays
hypothetical fills at each decision's next open; a decision recorded after
that open is late and never counts. Nothing here can place an order.

Layout::

    <workspace>/prospective/<study_id>/study.json    frozen declaration
    <workspace>/prospective/<study_id>/ledger.jsonl  authoritative hash chain
    <workspace>/prospective/<study_id>/snapshots/    bars behind each decision
    <backup>/<study_id>/                              read-only mirror +
                                                      checkpoints.jsonl chain

Usage::

    python -m src.quorum.prospective preview --study S --symbols SPY.US,...
    python -m src.quorum.prospective record --workspace W --study S --backup-dir B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.governance.ledger import append_record, verify_chain
from src.quorum.contracts import (
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    StaticEnsembleConfig,
)
from src.quorum.data import DatasetProvenance, DatasetSnapshotStore
from src.quorum.ensemble import StaticEnsemble
from src.quorum.evaluation import sharpe_difference_test
from src.quorum.research import (
    TIMESTAMP_CONVENTION,
    ResearchConfig,
    _ensembles,
    _experts,
    _predict,
    daily_bars_from_frame,
)
from src.quorum.risk import FixedRiskPolicy, RiskPolicyConfig

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
SCHEMA_VERSION = 2
DURATION_YEARS = 2
CONTROL = "equal_weight_control"
HORIZON_BARS = 5
_STUDY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_CONFIG = ResearchConfig(expert_set="v1")
_REPOSITORY = Path(__file__).resolve().parents[3]
_CANDIDATES = {
    "equal_weight_control": {
        "predictor": "constant:0",
        "definition": (
            "equal-weight fully-invested control: score 0 for every symbol gives "
            "tilt 0.5, so proposed weight 0.5 * (2/N) = 1/N per symbol (0.125 for "
            "N = 8) and gross N * 1/N = 1.0, within the cap. From all cash the "
            "0.5 turnover cap halves the first decision (0.0625 each, gross 0.5); "
            "the second reaches 0.125 each, gross 1.0. It is rebalanced back to "
            "1/N at every valid execution from its drifted weights, paying costs "
            "on the actual trades, exactly like every other candidate"
        ),
    },
    "ensemble_tilt": {
        "predictor": "ensemble",
        "definition": (
            "static V0 ensemble: fsum(w_i * score_i) with w_i = 1/3 over "
            "quorum.momentum, quorum.trend, quorum.mean_reversion; abstains "
            "when any member abstains"
        ),
    },
    "trend_tilt": {
        "predictor": "quorum.trend",
        "definition": "quorum.trend score (SMA10 vs SMA50) used directly",
    },
    "vol_regime_tilt": {
        "predictor": "quorum.volatility_regime",
        "definition": (
            "quorum.volatility_regime score used directly; the one confirmatory "
            "hypothesis"
        ),
    },
}


def _dir(workspace: Path, study_id: str) -> Path:
    if not _STUDY_ID_RE.fullmatch(study_id):
        raise ValueError("study_id must be a short name of letters, digits, . _ -")
    return Path(workspace) / "prospective" / study_id


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _code_identity() -> dict[str, Any]:
    """The exact committed code a study runs; a dirty tree is not reproducible."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "clean": git("status", "--porcelain") == "",
    }


def preview_declaration(
    study_id: str,
    symbols: Sequence[str],
    *,
    declared_at: datetime | None = None,
) -> dict[str, Any]:
    """Return exactly the declaration ``declare_study`` would freeze."""
    _dir(Path("."), study_id)
    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError("a study needs at least one symbol")
    declared_at = declared_at or datetime.now(UTC)
    start = declared_at.astimezone(NEW_YORK).date()
    try:
        end_date = start.replace(year=start.year + DURATION_YEARS)
    except ValueError:  # declared on 29 February
        end_date = start.replace(year=start.year + DURATION_YEARS, day=28)
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "declared_at": _utc(declared_at),
        "universe": {
            "symbols": symbols,
            "membership": "fixed for the whole study; nothing added, removed, or replaced",
            "selection_note": "large US ETFs and stocks chosen in 2026: survivorship-selected",
        },
        "data": {
            "source": "yfinance via Vibe loader backtest.loaders.yfinance_loader.DataLoader",
            "interval": "1D",
            "price_adjustment": (
                "provider auto_adjust=True: OHLC adjusted for splits and dividends; "
                "volume split-adjusted"
            ),
            "session": {
                "timezone": "America/New_York",
                "open": "09:30",
                "close": "16:00",
            },
            "timestamp_convention": TIMESTAMP_CONVENTION,
            "lookback_calendar_days": 550,
            "loader_bar_validation": (
                "validate_ohlc: high/low excursions <= 1e-9 relative are snapped; "
                "genuine OHLC violations are dropped by the loader"
            ),
            "snapshot": "each decision's input bars are a content-addressed snapshot it references",
        },
        "information_cutoff": {
            "rule": "a bar is used only if its close (event_at) <= recorded_at, the run's wall clock",
            "decision_bar": "the latest bar closed for every symbol",
            "decision_at": "that bar's close",
            "recorded_at": "the actual run time; never backdated",
            "unclosed_bars": "excluded and counted in the decision record",
        },
        "experts": {
            e.expert_id: {"version": e.expert_version, "config_id": e.config_id}
            for e in _experts(_CONFIG)
        },
        "expert_specifications": [
            "docs/quorum/ARCHITECTURE.md (Task 4 deterministic V0 experts)",
            "docs/quorum/EXPERTS_V1.md",
        ],
        "candidates": _CANDIDATES,
        "construction": {
            "name": "long_only_tilt",
            "tilt": "(1 + score) / 2, in [0, 1]",
            "proposed_weight": "tilt * max_abs_weight_per_asset",
        },
        "risk": {
            "policy": "src.quorum.risk.FixedRiskPolicy",
            "max_gross_exposure": 1.0,
            "max_abs_weight_per_asset": min(1.0, 2.0 / len(symbols)),
            "max_turnover": 0.5,
            "rules": (
                "gross above the cap scales all weights proportionally; then L1 "
                "turnover above the cap interpolates proportionally from the "
                "previous executable targets; no leverage, no shorts; the "
                "remainder is cash"
            ),
            "targets_vs_trades": (
                "the policy is price-unaware, as in research: its turnover cap "
                "limits changes between successive targets, while actual trades "
                "run from drifted pre-trade weights to the targets and may differ"
            ),
            "state": "replayed each run over the candidate's full recorded score history",
        },
        "execution": {
            "fill": "open of the first bar after the decision bar (next session open, 09:30 New York)",
            "late_rule": (
                "a decision counts only if recorded_at < that bar's start; a late "
                "decision never counts and the previous weights are held"
            ),
            "holding": (
                "between valid executions each position drifts with its own "
                "open-to-open price and cash earns 0; at each valid execution the "
                "portfolio trades from its drifted pre-trade weights to the targets"
            ),
            "marking": "every candidate is marked to market at every session open",
            "orders": "hypothetical only; no broker and no order path",
        },
        "costs": {
            "model": "proportional slippage on traded weight",
            "rate": 0.0005,
            "application": (
                "each valid execution pays rate * sum_i |target_i - pre_trade_i| "
                "of pre-trade equity, where pre_trade_i are the drifted weights (L1 "
                "turnover of the actual trades); targets are fractions of post-cost "
                "equity"
            ),
            "commission": 0.0,
            "borrow": "not applicable (long-only)",
            "cash_return": 0.0,
        },
        "initial_state": {
            "weights": "all cash: 0 for every symbol",
            "equity": 1.0,
            "ramp": (
                "the turnover cap limits target gross to 0.5 after the first "
                "decision; the control reaches 1.0 at the second"
            ),
        },
        "missing_data": {
            "fetch_or_validation_failure": "the run appends a 'skip' record with the reason; no decision",
            "missing_values": "any NaN in a fetched bar fails the run as a 'skip'",
            "calendar_mismatch": "a 'skip' when the symbols' latest 253 closed sessions differ",
        },
        "abstention": (
            "if any symbol lacks a candidate score, that candidate records no "
            "targets for the day; its previous weights are held and the day is "
            "excluded from its risk-policy history"
        ),
        "corporate_actions": (
            "provider adjustment only; evaluation reads opens from the most recent "
            "decision's snapshot, so dividends count as return and splits are continuous"
        ),
        "non_trading_days": (
            "a run with no newly closed session appends nothing and returns "
            "'no_new_session' (weekends, holidays, repeated runs)"
        ),
        "missed_runs": {
            "rule": "a session with no decision has none; the previous weights are held",
            "catch_up": (
                "a later run decides only the latest closed session, never backdates, "
                "and is late if it runs after the next open"
            ),
            "reporting": "every session after the first decision without a decision is listed as missed",
        },
        "evaluation": {
            "series": (
                "daily open-to-open returns from the first valid execution's open "
                "to the last observed open, identical dates for every candidate"
            ),
            "metrics": [
                "days and executions",
                "total_return",
                "annualized_return = (1 + total_return) ** (252 / days) - 1",
                "annualized_volatility = sample std of daily returns * sqrt(252)",
                "downside_volatility = sqrt(mean(min(r, 0) ** 2)) * sqrt(252)",
                "sharpe = mean / sample std of daily returns * sqrt(252), zero risk-free",
                "max_drawdown of the daily equity curve",
                "avg_gross_exposure and avg_cash_weight, averaged over daily post-trade marks",
                "total_turnover, avg_turnover_per_execution, total_cost",
                "vs_control: daily excess mean (annualized), t, tracking error, information ratio",
            ],
            "study_counts": [
                "decisions",
                "late",
                "pending",
                "missed sessions",
                "skips",
            ],
            "status": (
                "interim until the New York date passes end_date; interim results "
                "are descriptive only and cannot alter this study"
            ),
        },
        "confirmatory_test": {
            "hypothesis": (
                "the preregistered volatility-regime tilt improves prospective "
                "risk-adjusted performance relative to the equal-weight "
                "fully-invested control"
            ),
            "candidate": "vol_regime_tilt",
            "against": CONTROL,
            "effect": "annualized Sharpe(vol_regime_tilt) - Sharpe(equal_weight_control)",
            "method": {
                "name": "Ledoit-Wolf (2008) studentized circular block bootstrap",
                "series": "paired daily open-to-open returns over the common span",
                "standard_error": (
                    "delta method on moments (a, b, a^2, b^2) with a Newey-West "
                    "(Bartlett) HAC, lag = block length"
                ),
                "block_length": "round(n ** (1/3))",
                "bootstrap_standard_error": "block-sum covariance of each resample",
                "resamples": 10000,
                "seed": 0,
                "p_value": "(1 + #{T* >= T}) / (resamples + 1), one-sided",
                "implementation": "src.quorum.evaluation.sharpe_difference_test",
            },
            "alpha": 0.05,
            "n_definition": (
                "number of paired daily open-to-open returns of vol_regime_tilt "
                "and equal_weight_control over their common evaluation span"
            ),
            "outcome_mapping": [
                {"when": "n < 250", "outcome": "insufficient_observations"},
                {
                    "when": "n >= 250 and p_value_one_sided <= 0.05",
                    "outcome": "supported",
                },
                {
                    "when": "n >= 250 and p_value_one_sided > 0.05",
                    "outcome": "not_supported",
                },
            ],
            "always_reported": {
                "n": "number of paired daily observations",
                "sharpe_difference_annualized": (
                    "observed Sharpe(vol_regime_tilt) - Sharpe(equal_weight_control) "
                    "from daily moments (mean / population std), times sqrt(252)"
                ),
                "studentized_statistic": (
                    "T = daily Sharpe difference / its Newey-West HAC "
                    "delta-method standard error"
                ),
                "p_value_one_sided": "bootstrap (1 + #{T* >= T}) / (resamples + 1)",
                "result": "the outcome given by outcome_mapping",
            },
            "reporting_rule": (
                "all always_reported fields appear in every evaluation, including "
                "interim and insufficient_observations ones; interim values are "
                "descriptive only; statistic fields are null only when the "
                "bootstrap cannot run (fewer than two blocks)"
            ),
            "uncertainty": (
                "the procedure produces no confidence interval; it also reports "
                "standard_error_daily, the HAC delta-method standard error of the "
                "daily Sharpe difference"
            ),
            "minimum_days": 250,
            "outcomes": ["supported", "not_supported", "insufficient_observations"],
            "power_note": (
                "simulated power at 2 years and 0.95 correlation is about 22% for "
                "a true +0.2 annual Sharpe gain and about 59% for +0.5, so "
                "not_supported is not evidence of no effect"
            ),
            "secondary_descriptive": [
                "mean excess return",
                "annualized return and volatility",
                "downside volatility",
                "max drawdown",
                "tracking error and information ratio",
                "turnover and costs",
                "gross exposure and cash weight",
            ],
            "exploratory_candidates": ["ensemble_tilt", "trend_tilt"],
        },
        "external_reference": {
            "name": "SPY.US buy and hold",
            "rule": (
                "descriptive only and never part of the confirmatory test: bought "
                "at the first valid execution's open paying the declared rate once, "
                "then held and marked daily; omitted if SPY.US is not in the universe"
            ),
        },
        "horizon": {
            "start": "the first decision recorded after declaration",
            "end_date": end_date.isoformat(),
            "rule": (
                "no run is accepted after end_date (New York date); no early "
                "stopping for performance; the final evaluation is the first after end_date"
            ),
            "concurrency": (
                "materially changed systems receive new study IDs and may run "
                "concurrently; this study is never modified"
            ),
        },
        "operations": {
            "schedule": "weekdays 17:00 America/Denver, and as soon as possible after a missed start",
            "note": "scheduling never affects validity; recorded_at and the late rule decide",
        },
        "code": _code_identity(),
        "broker": None,
    }


def declare_study(
    workspace: Path,
    study_id: str,
    symbols: Sequence[str],
    *,
    declared_at: datetime | None = None,
    allow_dirty_code: bool = False,
) -> dict[str, Any]:
    """Freeze a study before its first decision; a second declaration fails."""
    directory = _dir(workspace, study_id)
    if (directory / "study.json").exists():
        raise FileExistsError(f"study {study_id!r} is already declared")
    study = preview_declaration(study_id, symbols, declared_at=declared_at)
    if not study["code"]["clean"] and not allow_dirty_code:
        raise ValueError("uncommitted code: commit before declaring a study")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "study.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(study, indent=2, sort_keys=True) + "\n")
    append_record(
        directory / "ledger.jsonl",
        {
            "type": "declaration",
            "study_sha256": hashlib.sha256(_canonical(study).encode()).hexdigest(),
        },
    )
    return study


def _load_dir(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the chain and the declaration; return both."""
    study = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    ledger = directory / "ledger.jsonl"
    check = verify_chain(ledger)
    if not check.ok:
        raise ValueError(f"prospective ledger chain is broken: {check}")
    records = [json.loads(line) for line in ledger.read_text("utf-8").splitlines()]
    digest = hashlib.sha256(_canonical(study).encode()).hexdigest()
    if not records or records[0].get("study_sha256") != digest:
        raise ValueError("study.json does not match the ledger's declaration")
    return study, records


def _tilt_targets(
    study: Mapping[str, Any],
    name: str,
    history: Sequence[tuple[datetime, Mapping[str, float | None]]],
) -> dict[str, float] | None:
    """Replay the frozen risk policy over a candidate's recorded tilt scores.

    A day where any asset lacks a score abstains: it yields no targets (the
    evaluation holds the previous ones) and stays out of the policy history,
    so it never shrinks the universe or advances the turnover state.
    """
    if any(score is None for score in history[-1][1].values()):
        return None
    history = [
        (at, scores)
        for at, scores in history
        if all(score is not None for score in scores.values())
    ]
    predictions = tuple(
        ExpertPrediction(
            expert_id=name,
            expert_version="prospective",
            asset=asset,
            event_at=at,
            available_at=at,
            decision_at=at,
            horizon_bars=HORIZON_BARS,
            score=(1.0 + score) / 2.0,
            experiment_id=study["study_id"],
        )
        for at, scores in history
        for asset, score in scores.items()
    )
    decisions = StaticEnsemble(
        StaticEnsembleConfig(
            experts=(ExpertWeight(name, "prospective", 1.0),),
            sell_threshold=-0.2,
            buy_threshold=0.2,
        )
    ).combine(ExpertResult(predictions))
    risk = {
        key: study["risk"][key]
        for key in ("max_gross_exposure", "max_abs_weight_per_asset", "max_turnover")
    }
    result = FixedRiskPolicy(RiskPolicyConfig(**risk)).apply(decisions)
    return {t.asset: t.final_target_weight for t in result.rebalances[-1].targets}


def record_day(
    workspace: Path,
    study_id: str,
    *,
    loader: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the latest closed session's decision; never backdates."""
    directory = _dir(workspace, study_id)
    study, records = _load_dir(directory)
    ledger = directory / "ledger.jsonl"
    now = now or datetime.now(UTC)
    end_date = date.fromisoformat(study["horizon"]["end_date"])
    if now.astimezone(NEW_YORK).date() > end_date:
        return {"status": "study_complete"}
    symbols = study["universe"]["symbols"]
    try:
        if loader is None:
            from backtest.loaders.yfinance_loader import DataLoader as YFinanceLoader

            loader = YFinanceLoader()
        lookback = timedelta(days=study["data"]["lookback_calendar_days"])
        frames = loader.fetch(
            list(symbols),
            (now - lookback).date().isoformat(),
            now.date().isoformat(),
            interval="1D",
        )
        fetched = {s: daily_bars_from_frame(s, frames[s]) for s in symbols}
    except Exception as exc:  # provider or data-validation failure: a skip
        record = {
            "type": "skip",
            "recorded_at": _utc(now),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        append_record(ledger, record)
        return record

    bars = {s: tuple(b for b in fetched[s] if b.event_at <= now) for s in symbols}
    excluded = sum(len(fetched[s]) - len(bars[s]) for s in symbols)
    windows = {s: [b.event_at for b in bars[s][-253:]] for s in symbols}
    reference = windows[symbols[0]]
    if not reference or any(w != reference for w in windows.values()):
        record = {
            "type": "skip",
            "recorded_at": _utc(now),
            "reason": "symbols' latest 253 closed sessions differ",
        }
        append_record(ledger, record)
        return record
    decision_bar = bars[symbols[0]][-1]
    decision_date = str(decision_bar.event_at.date())
    decisions = [r for r in records if r.get("type") == "decision"]
    if any(r["decision_date"] == decision_date for r in decisions):
        return {"status": "no_new_session", "latest_closed_session": decision_date}

    snapshot = DatasetSnapshotStore(directory / "snapshots").create(
        [b for s in symbols for b in bars[s]],
        interval_id="1D",
        timestamp_convention=TIMESTAMP_CONVENTION,
        provenance=DatasetProvenance(
            source_id=getattr(loader, "name", type(loader).__name__),
            retrieved_at=now,
            adjustment_id="provider-default",
        ),
    )

    ensemble = _ensembles(_CONFIG)["ensemble"]
    members = {w.expert_id for w in ensemble.config.experts}
    expert_scores: dict[str, dict[str, float]] = {}
    ensemble_scores: dict[str, float | None] = {}
    for s in symbols:
        predictions = _predict(
            s,
            bars[s],
            len(bars[s]) - 1,
            _experts(_CONFIG),
            horizon_bars=HORIZON_BARS,
            experiment_id=study_id,
        )
        expert_scores[s] = {p.expert_id: p.score for p in predictions}
        combined = ensemble.combine(
            ExpertResult(tuple(p for p in predictions if p.expert_id in members))
        ).decisions
        ensemble_scores[s] = combined[0].combined_score if combined else None

    def score(predictor: str, s: str) -> float | None:
        if predictor == "constant:0":
            return 0.0
        if predictor == "ensemble":
            return ensemble_scores[s]
        return expert_scores[s].get(predictor)

    candidates: dict[str, Any] = {}
    for name, definition in sorted(study["candidates"].items()):
        scores = {s: score(definition["predictor"], s) for s in symbols}
        history = [
            (datetime.fromisoformat(r["decision_at"]), r["candidates"][name]["scores"])
            for r in decisions
        ] + [(decision_bar.event_at, scores)]
        candidates[name] = {
            "scores": scores,
            "targets": _tilt_targets(study, name, history),
        }

    record = {
        "type": "decision",
        "decision_date": decision_date,
        "decision_at": _utc(decision_bar.event_at),
        "recorded_at": _utc(now),
        "snapshot_id": snapshot.snapshot_id,
        "unclosed_bars_excluded": excluded,
        "expert_scores": expert_scores,
        "candidates": candidates,
    }
    append_record(ledger, record)
    return record


def _simulate_portfolio(
    opens: Mapping[str, Sequence[float]],
    executions: Mapping[int, Mapping[str, float] | None],
    *,
    rate: float,
) -> dict[str, Any]:
    """Mark a portfolio at every open, trading only on execution days.

    Positions drift with their own open-to-open prices and cash earns 0. On an
    execution day the portfolio trades from its drifted pre-trade weights to
    the targets and pays ``rate`` times that L1 turnover of pre-trade equity.
    ``None`` targets abstain: nothing trades. Starts all cash, equity 1.0.
    """
    symbols = sorted(opens)
    values = {s: 0.0 for s in symbols}
    cash, previous = 1.0, 1.0
    run: dict[str, Any] = {
        "equity": [],
        "returns": [],
        "weights": [],
        "gross": [],
        "cash": [],
        "pre_trade_weights": {},
        "turnover": {},
        "cost": {},
    }
    for day in range(len(opens[symbols[0]])):
        if day:
            values = {
                s: v * opens[s][day] / opens[s][day - 1] for s, v in values.items()
            }
        equity = sum(values.values()) + cash
        if day in executions:
            pre = {s: values[s] / equity for s in symbols}
            targets = executions[day] or pre
            trade = sum(abs(targets[s] - pre[s]) for s in symbols)
            equity *= 1.0 - rate * trade
            values = {s: targets[s] * equity for s in symbols}
            cash = equity - sum(values.values())
            run["pre_trade_weights"][day] = pre
            run["turnover"][day] = trade
            run["cost"][day] = rate * trade
        run["equity"].append(equity)
        run["returns"].append(equity / previous - 1.0)
        run["weights"].append({s: values[s] / equity for s in symbols})
        run["gross"].append(sum(values.values()) / equity)
        run["cash"].append(cash / equity)
        previous = equity
    return run


def _portfolio_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    returns = run["returns"]
    n = len(returns)
    mean = sum(returns) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1)) if n > 1 else 0.0
    peak, drawdown = 1.0, 0.0
    for equity in run["equity"]:
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    executions = len(run["turnover"])
    final = run["equity"][-1]
    return {
        "days": n,
        "executions": executions,
        "total_return": final - 1.0,
        "annualized_return": final ** (252 / n) - 1.0,
        "annualized_volatility": std * math.sqrt(252) if n > 1 else None,
        "downside_volatility": math.sqrt(sum(min(r, 0.0) ** 2 for r in returns) / n)
        * math.sqrt(252),
        "sharpe": mean / std * math.sqrt(252) if std else None,
        "max_drawdown": drawdown,
        "avg_gross_exposure": sum(run["gross"]) / n,
        "avg_cash_weight": sum(run["cash"]) / n,
        "total_turnover": sum(run["turnover"].values()),
        "avg_turnover_per_execution": (
            sum(run["turnover"].values()) / executions if executions else None
        ),
        "total_cost": sum(run["cost"].values()),
    }


def _confirmatory_outcome(test: Mapping[str, Any], n: int, p_value: float) -> str:
    """The declared outcome_mapping, applied exactly."""
    if n < test["minimum_days"]:
        return "insufficient_observations"
    return "supported" if p_value <= test["alpha"] else "not_supported"


def _confirmatory_report(
    test: Mapping[str, Any],
    candidate: Sequence[float],
    control: Sequence[float],
    final: bool,
) -> dict[str, Any]:
    """Every always_reported field, whatever the outcome."""
    n = len(candidate)
    method = test["method"]
    try:
        inference = sharpe_difference_test(
            candidate, control, resamples=method["resamples"], seed=method["seed"]
        )
    except ValueError:  # fewer than two bootstrap blocks
        inference = {
            "sharpe_difference_annualized": None,
            "studentized_statistic": None,
            "p_value_one_sided": None,
        }
    p_value = inference["p_value_one_sided"]
    return {
        **inference,
        "n": n,
        "result": (
            "insufficient_observations"
            if p_value is None
            else _confirmatory_outcome(test, n, p_value)
        ),
        "final": final,
    }


def evaluate_study(
    workspace: Path, study_id: str, *, now: datetime | None = None
) -> dict[str, Any]:
    """Mark every candidate daily and replay valid executions exactly as declared."""
    directory = _dir(workspace, study_id)
    study, records = _load_dir(directory)
    now = now or datetime.now(UTC)
    end_date = date.fromisoformat(study["horizon"]["end_date"])
    final = now.astimezone(NEW_YORK).date() > end_date
    decisions = sorted(
        (r for r in records if r.get("type") == "decision"),
        key=lambda r: r["decision_at"],
    )
    report: dict[str, Any] = {
        "study_id": study_id,
        "status": "final" if final else "interim",
        "decisions": len(decisions),
        "skips": sum(r.get("type") == "skip" for r in records),
        "late_decisions": [],
        "pending_decisions": [],
        "missed_sessions": [],
        "candidates": {},
        "external_reference": None,
        "confirmatory_test": _confirmatory_report(
            study["confirmatory_test"], [], [], final
        ),
    }
    if not decisions:
        return report
    symbols = study["universe"]["symbols"]
    latest = DatasetSnapshotStore(directory / "snapshots").load(
        decisions[-1]["snapshot_id"]
    )
    by_asset = {
        s: sorted((b for b in latest.bars if b.asset == s), key=lambda b: b.event_at)
        for s in symbols
    }
    reference = by_asset[symbols[0]]
    position = {_utc(b.event_at): i for i, b in enumerate(reference)}
    decided = {position[r["decision_at"]] for r in decisions}
    report["missed_sessions"] = [
        str(reference[i].event_at.date())
        for i in range(min(decided), len(reference))
        if i not in decided
    ]

    executions = []
    for record in decisions:
        index = position[record["decision_at"]] + 1
        if index >= len(reference):
            report["pending_decisions"].append(record["decision_date"])
        elif datetime.fromisoformat(record["recorded_at"]) >= reference[index].start_at:
            report["late_decisions"].append(record["decision_date"])
        else:
            executions.append((record, index))
    if not executions:
        return report

    first = executions[0][1]
    opens = {s: [b.open for b in by_asset[s][first:]] for s in symbols}
    rate = study["costs"]["rate"]
    returns: dict[str, list[float]] = {}
    for name in sorted(study["candidates"]):
        plan = {
            index - first: record["candidates"][name]["targets"]
            for record, index in executions
        }
        run = _simulate_portfolio(opens, plan, rate=rate)
        returns[name] = run["returns"]
        report["candidates"][name] = _portfolio_metrics(run)

    for name, series in returns.items():
        if name == CONTROL:
            continue
        excess = [r - c for r, c in zip(series, returns[CONTROL])]
        n = len(excess)
        mean = sum(excess) / n
        std = (
            math.sqrt(sum((x - mean) ** 2 for x in excess) / (n - 1)) if n > 1 else 0.0
        )
        tracking = std * math.sqrt(252) if std else None
        report["candidates"][name]["vs_control"] = {
            "days": n,
            "mean_excess_annualized": mean * 252,
            "excess_t": mean / std * math.sqrt(n) if std else None,
            "tracking_error": tracking,
            "information_ratio": mean * 252 / tracking if tracking else None,
        }

    if "SPY.US" in symbols:
        spy = _simulate_portfolio(
            {"SPY.US": opens["SPY.US"]}, {0: {"SPY.US": 1.0}}, rate=rate
        )
        report["external_reference"] = {
            "name": study["external_reference"]["name"],
            **_portfolio_metrics(spy),
        }

    test = study["confirmatory_test"]
    report["confirmatory_test"] = _confirmatory_report(
        test, returns[test["candidate"]], returns[test["against"]], final
    )
    (directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _writable(path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def backup_study(workspace: Path, study_id: str, backup_dir: Path) -> dict[str, Any]:
    """Mirror the study read-only and checkpoint its ledger head.

    The mirror is never written by the recorder and never becomes a second
    writable ledger. Its checkpoint chain is an external anchor: a primary
    that no longer contains history already backed up is a rollback.
    """
    directory = _dir(workspace, study_id)
    _, records = _load_dir(directory)
    target = Path(backup_dir) / study_id
    checkpoints = target / "checkpoints.jsonl"
    if checkpoints.exists():
        if not verify_chain(checkpoints).ok:
            raise ValueError("backup checkpoint chain is broken")
        last = json.loads(checkpoints.read_text("utf-8").splitlines()[-1])
        count = last["record_count"]
        if (
            len(records) < count
            or records[count - 1]["record_hash"] != last["head_record_hash"]
        ):
            raise ValueError(
                "rollback: primary ledger no longer contains history already backed up"
            )
    target.mkdir(parents=True, exist_ok=True)
    for name in ("study.json", "ledger.jsonl"):
        source, destination = directory / name, target / name
        if name == "study.json" and destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(
                    "backup study.json differs from the primary declaration"
                )
            continue
        staging = target / f".{name}.tmp"
        shutil.copyfile(source, staging)
        _writable(destination)
        os.replace(staging, destination)
        os.chmod(destination, stat.S_IREAD)
    snapshots = directory / "snapshots"
    for snapshot in snapshots.glob("sha256-*") if snapshots.exists() else ():
        if not (target / "snapshots" / snapshot.name).exists():
            shutil.copytree(snapshot, target / "snapshots" / snapshot.name)
    append_record(
        checkpoints,
        {
            "record_count": len(records),
            "head_record_hash": records[-1]["record_hash"],
            "backed_up_at": _utc(datetime.now(UTC)),
        },
    )
    return {"study_id": study_id, "record_count": len(records), "backup": str(target)}


def restore_study(backup_dir: Path, study_id: str, workspace: Path) -> Path:
    """Rebuild a lost primary from its mirror; refuses to overwrite anything."""
    source = Path(backup_dir) / study_id
    _load_dir(source)
    destination = _dir(workspace, study_id)
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; restore only into an empty place"
        )
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("checkpoints.jsonl*")
    )
    for path in destination.rglob("*"):
        if path.is_file():
            _writable(path)
    _load_dir(destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quorum prospective paper ledger")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "declare"):
        command = commands.add_parser(name)
        command.add_argument("--study", required=True)
        command.add_argument("--symbols", required=True, help="comma-separated")
        if name == "declare":
            command.add_argument("--workspace", required=True, type=Path)
    for name in ("record", "evaluate", "backup"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", required=True, type=Path)
        command.add_argument("--study", required=True)
        if name in ("record", "backup"):
            command.add_argument("--backup-dir", type=Path, required=name == "backup")
    args = parser.parse_args(argv)
    symbols = [s.strip() for s in getattr(args, "symbols", "").split(",") if s.strip()]
    if args.command == "preview":
        result: Any = preview_declaration(args.study, symbols)
    elif args.command == "declare":
        result = declare_study(args.workspace, args.study, symbols)
    elif args.command == "record":
        result = record_day(args.workspace, args.study)
        result = {k: v for k, v in result.items() if k != "expert_scores"}
        if args.backup_dir is not None:
            result["backup"] = backup_study(args.workspace, args.study, args.backup_dir)
    elif args.command == "backup":
        result = backup_study(args.workspace, args.study, args.backup_dir)
    else:
        result = evaluate_study(args.workspace, args.study)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
