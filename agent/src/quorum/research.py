"""Quorum real-data research runner: snapshot ingestion and OOF evaluation.

``ingest`` turns one loader fetch into an immutable, content-addressed daily-bar
snapshot. ``run`` registers an experiment attempt, materializes the locked-holdout
chronological plan, predicts every authorized OOF slot with the frozen V0
experts and static ensemble, realizes tradable forward returns, and writes a
prediction-quality report. It sizes nothing, applies no costs, and never opens
the final holdout.

Workspace layout::

    <workspace>/snapshots/               DatasetSnapshotStore root
    <workspace>/experiment_ledger.jsonl  append-only trial ledger
    <workspace>/runs/<attempt_id>/       report and OOF predictions

Usage::

    python -m src.quorum.research ingest --workspace W --symbols SPY.US,QQQ.US \\
        --start 2005-01-01 --end 2025-12-31
    python -m src.quorum.research run --workspace W --snapshot-id sha256:...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.quorum.contracts import (
    EvaluationMode,
    EvaluationProtocol,
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    FinalHoldout,
    FinalHoldoutState,
    PredictionContext,
    StaticEnsembleConfig,
    TimeInterval,
)
from src.quorum.data import (
    DatasetProvenance,
    DatasetSnapshotManifest,
    DatasetSnapshotStore,
    MarketBar,
)
from src.quorum.ensemble import StaticEnsemble
from src.quorum.evaluation import evaluate_predictions, paired_fold_delta
from src.quorum.experiments import (
    ExperimentLedger,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentState,
    ExternalRecordRefs,
)
from src.quorum.experts import (
    AbnormalVolumeExpert,
    MeanReversionExpert,
    MomentumExpert,
    OvernightMomentumExpert,
    RangeReversalExpert,
    TrendExpert,
    VolatilityRegimeExpert,
)
from src.quorum.portfolio import run_oos_portfolios
from src.quorum.risk import RiskPolicyConfig
from src.quorum.stacking import RidgeStacker
from src.quorum.validation import materialize_chronological_plan

UTC = timezone.utc
US_EQUITY_TIMEZONE = "America/New_York"
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
TIMESTAMP_CONVENTION = "bar:[regular-open,regular-close):America/New_York:v1"
ENSEMBLE_PREDICTOR = "ensemble"
STACKER_PREDICTOR = "stacker.ridge"
PORTFOLIO_CONSTRUCTIONS = ("long_short", "long_only_tilt")
_FAMILY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
TRIAL_FAMILY_ID = "quorum:research:v0-experts"
# Longest V0 lookback is SMA50; a trailing window keeps prediction O(n).
_V0_EXPERTS = (MomentumExpert(), TrendExpert(), MeanReversionExpert())
# docs/quorum/EXPERTS_V1.md: frozen before any v1 result was observed.
_V1_EXPERTS = (
    VolatilityRegimeExpert(),
    OvernightMomentumExpert(),
    AbnormalVolumeExpert(),
    RangeReversalExpert(),
)
# Trailing bars each expert needs, and whether it reads full OHLCV bars.
_EXPERT_INPUTS = {
    "quorum.momentum": (21, False),
    "quorum.trend": (50, False),
    "quorum.mean_reversion": (20, False),
    "quorum.volatility_regime": (253, True),
    "quorum.overnight_momentum": (21, True),
    "quorum.abnormal_volume": (65, True),
    "quorum.range_reversal": (5, True),
}
EXPERT_SETS = ("v0", "v1")
INCREMENTAL_CRITERIA = {"orthogonal_max_abs_corr": 0.5, "incremental_min_t": 2.5}
_REGIME_VOL_BARS = 20
_REGIME_MEDIAN_BARS = 252


class _Loader(Protocol):
    def fetch(
        self, codes: list[str], start_date: str, end_date: str, *, interval: str
    ) -> Mapping[str, pd.DataFrame]: ...


def daily_bars_from_frame(
    asset: str,
    frame: pd.DataFrame,
    *,
    market_timezone: str = US_EQUITY_TIMEZONE,
    session_open: time = REGULAR_OPEN,
    session_close: time = REGULAR_CLOSE,
) -> tuple[MarketBar, ...]:
    """Stamp loader trade-date rows as regular-session bars.

    Each row becomes ``[date session_open, date session_close)`` in the market
    timezone, available at its close. Missing values fail closed: bars are
    never filled or dropped here.
    """
    # ponytail: half-days (13:00 close) are stamped 16:00, which only delays
    # availability; use an exchange calendar if intraday timing ever matters.
    columns = ["open", "high", "low", "close", "volume"]
    absent = [name for name in columns if name not in frame.columns]
    if absent:
        raise ValueError(f"{asset} frame lacks columns {absent}")
    values = frame[columns]
    if values.isna().to_numpy().any():
        raise ValueError(f"{asset} has missing values; refusing to fill or drop bars")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None or (index != index.normalize()).any():
        raise ValueError(f"{asset} index must hold naive trade dates")
    zone = ZoneInfo(market_timezone)
    bars = []
    for day, row in zip(index, values.itertuples(index=False)):
        start = datetime.combine(day.date(), session_open, tzinfo=zone)
        end = datetime.combine(day.date(), session_close, tzinfo=zone)
        bars.append(MarketBar(asset, start, end, end, *map(float, row)))
    return tuple(bars)


def ingest_daily_bars(
    store: DatasetSnapshotStore,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    loader: _Loader | None = None,
    source_id: str = "yfinance",
    adjustment_id: str = "yfinance:auto_adjust",
    retrieved_at: datetime | None = None,
) -> DatasetSnapshotManifest:
    """Fetch daily bars once and publish them as an immutable snapshot."""
    if loader is None:
        from backtest.loaders.yfinance_loader import DataLoader as YFinanceLoader

        loader = YFinanceLoader()
    retrieved_at = retrieved_at or datetime.now(UTC)
    symbols = sorted(set(symbols))
    frames = loader.fetch(list(symbols), start_date, end_date, interval="1D")
    empty = [s for s in symbols if s not in frames or frames[s].empty]
    if empty:
        raise ValueError(f"loader returned no bars for {empty}")
    bars = [bar for s in symbols for bar in daily_bars_from_frame(s, frames[s])]
    unclosed = [bar for bar in bars if bar.event_at > retrieved_at]
    if unclosed:
        raise ValueError(
            f"{len(unclosed)} bars were not closed at retrieval "
            f"(first: {unclosed[0].asset} {unclosed[0].event_at.isoformat()})"
        )
    provenance = DatasetProvenance(
        source_id=source_id,
        retrieved_at=retrieved_at,
        adjustment_id=adjustment_id,
        metadata={
            "symbols": list(symbols),
            "start_date": start_date,
            "end_date": end_date,
            "loader": type(loader).__name__,
        },
    )
    return store.create(
        bars,
        interval_id="1D",
        timestamp_convention=TIMESTAMP_CONVENTION,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Frozen evaluation settings; any change is a new scientific experiment."""

    horizon_bars: int = 5
    minimum_train_bars: int = 252
    validation_bars: int = 21
    test_bars: int = 63
    holdout_bars: int = 252
    sell_threshold: float = -0.2
    buy_threshold: float = 0.2
    stacker_alpha: float = 0.1
    max_gross_exposure: float = 1.0
    max_weight_per_asset: float = 0.25
    max_turnover: float = 0.5
    portfolio_construction: str = "long_short"
    rebalance_every_bars: int = 1
    stacker_window_bars: int | None = None
    expert_set: str = "v0"

    def __post_init__(self) -> None:
        if self.expert_set not in EXPERT_SETS:
            raise ValueError(f"expert_set must be one of {EXPERT_SETS}")
        if self.portfolio_construction not in PORTFOLIO_CONSTRUCTIONS:
            raise ValueError(
                f"portfolio_construction must be one of {PORTFOLIO_CONSTRUCTIONS}"
            )
        if self.rebalance_every_bars < 1:
            raise ValueError("rebalance_every_bars must be positive")
        if self.stacker_window_bars is not None and self.stacker_window_bars < 20:
            raise ValueError("stacker_window_bars must be None or at least 20")
        if self.validation_bars <= self.horizon_bars:
            raise ValueError("validation_bars must exceed horizon_bars")
        if self.holdout_bars <= self.horizon_bars:
            raise ValueError("holdout_bars must exceed horizon_bars")

    @property
    def config_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(asdict(self)).encode()).hexdigest()
        return f"quorum:research:config:{digest[:16]}"


@dataclass(frozen=True, slots=True)
class ResearchResult:
    attempt_id: str
    run_dir: Path
    report_path: Path
    predictions_path: Path
    metrics_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _universe_id(assets: Sequence[str]) -> str:
    joined = "+".join(assets)
    if len(joined) <= 200:
        return f"quorum:universe:{joined}"
    return "quorum:universe:sha256:" + hashlib.sha256(joined.encode()).hexdigest()


def _shared_calendar(
    bars_by_asset: Mapping[str, tuple[MarketBar, ...]],
) -> tuple[MarketBar, ...]:
    """Return one asset's bars once every asset shares its exact calendar."""
    # ponytail: missing-asset/dynamic-universe semantics are undecided, so any
    # calendar difference fails closed rather than choosing a fill policy.
    reference_asset, *others = sorted(bars_by_asset)
    reference = bars_by_asset[reference_asset]
    axis = [(bar.start_at, bar.event_at) for bar in reference]
    for asset in others:
        if [(bar.start_at, bar.event_at) for bar in bars_by_asset[asset]] != axis:
            raise ValueError(
                f"{asset} does not share the {reference_asset} calendar; "
                "missing-asset semantics are not defined yet"
            )
    return reference


def _protocol(config: ResearchConfig, bars: Sequence[MarketBar]) -> EvaluationProtocol:
    first_holdout = bars[-config.holdout_bars]
    return EvaluationProtocol(
        protocol_id=(
            f"quorum:research:daily:h{config.horizon_bars}"
            f":mt{config.minimum_train_bars}:v{config.validation_bars}"
            f":t{config.test_bars}:ho{config.holdout_bars}"
        ),
        mode=EvaluationMode.EXPANDING,
        minimum_train_bars=config.minimum_train_bars,
        train_window_bars=None,
        validation_bars=config.validation_bars,
        test_bars=config.test_bars,
        step_bars=config.validation_bars + config.test_bars,
        purge_bars=config.horizon_bars,
        embargo_bars=config.horizon_bars,
        final_holdout=FinalHoldout(
            first_holdout.start_at, bars[-1].event_at, FinalHoldoutState.LOCKED
        ),
        random_seed=0,
    )


def _experts(config: ResearchConfig) -> tuple[Any, ...]:
    return _V0_EXPERTS + (_V1_EXPERTS if config.expert_set == "v1" else ())


def _ensembles(config: ResearchConfig) -> dict[str, StaticEnsemble]:
    """Equal-weight static ensembles: the V0 three, plus all experts for v1."""
    members = {ENSEMBLE_PREDICTOR: _V0_EXPERTS}
    if config.expert_set == "v1":
        members["ensemble.all"] = _experts(config)
    return {
        name: StaticEnsemble(
            StaticEnsembleConfig(
                experts=tuple(
                    ExpertWeight(e.expert_id, e.expert_version, 1.0 / len(experts))
                    for e in experts
                ),
                sell_threshold=config.sell_threshold,
                buy_threshold=config.buy_threshold,
            )
        )
        for name, experts in members.items()
    }


def _stacker_features(config: ResearchConfig) -> dict[str, tuple[str, ...]]:
    """Feature sets: V0 alone, V0 plus each v1 expert, and every expert."""
    v0 = tuple(e.expert_id for e in _V0_EXPERTS)
    stackers = {STACKER_PREDICTOR: v0}
    if config.expert_set == "v1":
        for expert in _V1_EXPERTS:
            stackers[f"stacker.v0+{expert.expert_id}"] = v0 + (expert.expert_id,)
        stackers["stacker.all"] = tuple(e.expert_id for e in _experts(config))
    return stackers


def _predict(
    asset: str,
    bars: Sequence[MarketBar],
    position: int,
    experts: Sequence[Any],
    *,
    horizon_bars: int,
    experiment_id: str,
) -> list[ExpertPrediction]:
    """Run each expert on exactly its trailing window ending at ``position``."""
    predictions: list[ExpertPrediction] = []
    for expert in experts:
        need, ohlcv = _EXPERT_INPUTS[expert.expert_id]
        window = bars[max(0, position + 1 - need) : position + 1]
        latest_available = max((bar.available_at for bar in window), key=_utc)
        prepared: dict[str, object] = {
            "asset": asset,
            "close": tuple(b.close for b in window),
            "event_at": tuple(b.event_at for b in window),
            "available_at": tuple(b.available_at for b in window),
        }
        if ohlcv:
            prepared |= {
                "open": tuple(b.open for b in window),
                "high": tuple(b.high for b in window),
                "low": tuple(b.low for b in window),
                "volume": tuple(b.volume for b in window),
            }
        context = PredictionContext(
            event_at=bars[position].event_at,
            available_at=latest_available,
            decision_at=max(bars[position].event_at, latest_available, key=_utc),
            horizon_bars=horizon_bars,
            experiment_id=experiment_id,
        )
        predictions.extend(expert.predict(prepared, context).predictions)
    return predictions


def _volatility_regimes(bars: Sequence[MarketBar]) -> list[str]:
    """Causal per-bar regime: 20-bar RMS vol above its prior 252-bar median."""
    closes = pd.Series([bar.close for bar in bars], dtype="float64")
    vol = np.sqrt(
        (np.log(closes / closes.shift(1)) ** 2).rolling(_REGIME_VOL_BARS).mean()
    )
    median = vol.shift(1).rolling(_REGIME_MEDIAN_BARS).median()
    return [
        "unknown" if pd.isna(m) or pd.isna(v) else ("high" if v > m else "low")
        for v, m in zip(vol, median)
    ]


def _oof_rows(
    plan: Any,
    bars_by_asset: Mapping[str, tuple[MarketBar, ...]],
    config: ResearchConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict every authorized slot and attach its tradable forward return.

    The target enters at the next bar's open (the engine's fill) and exits at
    the close ``horizon_bars`` bars after the decision bar. Expert scores are
    computed once for every pre-holdout position; each per-fold ridge stacker
    fits only on that fold's purged training positions.
    """
    h = config.horizon_bars
    ordinary_stop = len(plan.label_end_positions)
    experts = _experts(config)
    ensembles = _ensembles(config)
    slots = {slot.sample_position: slot for slot in plan.oof_slots}
    rows: list[dict[str, Any]] = []
    regimes: dict[str, list[str]] = {}

    def row(asset: str, p: int, predictor: str, score: float) -> dict[str, Any]:
        bars = bars_by_asset[asset]
        slot, entry, exit_ = slots[p], bars[p + 1], bars[p + h]
        return {
            "predictor": predictor,
            "asset": asset,
            "event_at": _utc(bars[p].event_at),
            "decision_at": _utc(bars[p].event_at),
            "label_start_at": _utc(entry.start_at),
            "label_end_at": _utc(exit_.event_at),
            "fold_index": slot.fold_index,
            "split_id": slot.split_id,
            "role": slot.role.value,
            "regime": regimes[asset][p],
            "score": score,
            "forward_return": exit_.close / entry.open - 1.0,
        }

    # Scores and regimes never touch holdout bars: both stop at the boundary.
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for asset, bars in sorted(bars_by_asset.items()):
        regimes[asset] = _volatility_regimes(bars[:ordinary_stop])
        predictions = {
            p: _predict(
                asset,
                bars,
                p,
                experts,
                horizon_bars=h,
                experiment_id=plan.attempt_id,
            )
            for p in range(ordinary_stop)
        }
        scores[asset] = {
            p: {pred.expert_id: pred.score for pred in preds}
            for p, preds in predictions.items()
        }
        slot_predictions = [pred for p in slots for pred in predictions[p]]
        position_of = {_utc(bar.event_at): p for p, bar in enumerate(bars)}
        for pred in slot_predictions:
            rows.append(
                row(asset, position_of[_utc(pred.event_at)], pred.expert_id, pred.score)
            )
        for name, ensemble in ensembles.items():
            members = {w.expert_id for w in ensemble.config.experts}
            combined = ensemble.combine(
                ExpertResult(
                    tuple(q for q in slot_predictions if q.expert_id in members)
                )
            )
            for decision in combined.decisions:
                if decision.combined_score is not None:
                    p = position_of[_utc(decision.event_at)]
                    rows.append(row(asset, p, name, decision.combined_score))

    def target(asset: str, p: int) -> float:
        return (
            bars_by_asset[asset][p + h].close / bars_by_asset[asset][p + 1].open - 1.0
        )

    stacker = RidgeStacker(config.stacker_alpha)
    reference = bars_by_asset[min(bars_by_asset)]
    stacker_reports: dict[str, Any] = {}
    for predictor, names in _stacker_features(config).items():

        def complete(asset: str, p: int, names: tuple[str, ...] = names) -> bool:
            return all(name in scores[asset][p] for name in names)

        def features(asset: str, p: int, names: tuple[str, ...] = names) -> list[float]:
            return [scores[asset][p][name] for name in names]

        fold_cards = []
        for fold in plan.folds:
            train = [
                (asset, p)
                for asset in sorted(bars_by_asset)
                for p in fold.train_positions[-(config.stacker_window_bars or 0) :]
                if complete(asset, p)
            ]
            card: dict[str, Any] = {
                "fold_index": fold.manifest.fold_index,
                "split_id": fold.manifest.split_id,
                "evaluation_start": _utc(
                    reference[fold.validation_positions[0]].event_at
                ),
            }
            if len(train) < 2 * (len(names) + 1):
                # e.g. before a long warm-up: no fit, no predictions, reported.
                fold_cards.append(
                    card | {"skipped": "insufficient complete training rows"}
                )
                continue
            fitted = stacker.fit(
                np.array([features(a, p) for a, p in train]),
                np.array([target(a, p) for a, p in train]),
            )
            held_out = [
                (asset, p)
                for asset in sorted(bars_by_asset)
                for p in fold.validation_positions + fold.test_positions
                if complete(asset, p)
            ]
            if held_out:
                predicted = fitted.predict(
                    np.array([features(a, p) for a, p in held_out])
                )
                rows += [
                    row(a, p, predictor, float(score))
                    for (a, p), score in zip(held_out, predicted)
                ]
            fold_cards.append(
                card
                | {
                    "train_label_end_max": _utc(
                        max((reference[p + h].event_at for _, p in train), key=_utc)
                    ),
                    "model": fitted.to_dict(names),
                }
            )

        fitted_cards = [c for c in fold_cards if "model" in c]
        coef = pd.DataFrame([c["model"]["standardized_coef"] for c in fitted_cards])
        stacker_reports[predictor] = {
            "predictor": predictor,
            "alpha": config.stacker_alpha,
            "features": list(names),
            "folds": fold_cards,
            "fitted_folds": len(fitted_cards),
            "coef_summary": (
                {
                    name: {
                        "mean": float(coef[name].mean()),
                        "positive_fraction": float((coef[name] > 0.0).mean()),
                    }
                    for name in names
                }
                if fitted_cards
                else {}
            ),
            "note": (
                "Trained on expert scores at training positions, which is valid "
                "only because these experts have no fitted parameters; learned "
                "experts must feed a stacker their out-of-fold rows instead."
            ),
        }
    frame = pd.DataFrame(rows).sort_values(["predictor", "asset", "event_at"])
    return frame, stacker_reports


def _incremental(rows: pd.DataFrame, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered v1 criteria: orthogonality and paired per-fold IC gain."""
    correlation = evaluation["roles"]["test"]["score_correlation"]
    v0 = [e.expert_id for e in _V0_EXPERTS]
    experts: dict[str, Any] = {}
    for expert in _V1_EXPERTS:
        pairs = [
            abs(value)
            for name in v0
            if (value := correlation.get(expert.expert_id, {}).get(name)) is not None
        ]
        max_corr = max(pairs) if pairs else None
        delta = paired_fold_delta(
            rows, base=STACKER_PREDICTOR, other=f"stacker.v0+{expert.expert_id}"
        )
        experts[expert.expert_id] = {
            "max_abs_corr_with_v0": max_corr,
            "orthogonal": max_corr is not None
            and max_corr <= INCREMENTAL_CRITERIA["orthogonal_max_abs_corr"],
            "delta_vs_v0_stacker": delta,
            "adds_incremental_information": bool(
                delta["t"] is not None
                and delta["mean_delta"] > 0.0
                and delta["t"] >= INCREMENTAL_CRITERIA["incremental_min_t"]
            ),
        }
    experts["all"] = {
        "delta_vs_v0_stacker": paired_fold_delta(
            rows, base=STACKER_PREDICTOR, other="stacker.all"
        )
    }
    return {"criteria": dict(INCREMENTAL_CRITERIA), "experts": experts}


def _markdown(report: Mapping[str, Any]) -> str:
    def cell(value: object, fmt: str = "{:+.3f}") -> str:
        return "—" if value is None else fmt.format(value)

    lines = [
        f"# Quorum research report `{report['attempt_id']}`",
        "",
        f"- Snapshot: `{report['snapshot']['snapshot_id']}`",
        f"- Assets: {', '.join(report['snapshot']['assets'])}",
        f"- Target: {report['target']['definition']}",
        f"- Folds: {report['chronology']['fold_count']}; final holdout "
        f"{report['chronology']['final_holdout']['state']} from "
        f"{report['chronology']['final_holdout']['start']}",
        f"- Trial family `{report['trial_family']['id']}`: attempt "
        f"{report['trial_family']['attempt_count']}",
        "",
    ]
    for role, evaluation in report["evaluation"]["roles"].items():
        lines += [
            f"## {role.capitalize()} OOF predictions",
            "",
            "| Predictor | Coverage | IC | Fold IC mean | Fold IC t | Folds IC>0 "
            "| Hit rate | Up rate | Saturation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in evaluation["predictors"].items():
            lines.append(
                f"| {name} | {cell(m['coverage'], '{:.0%}')} | {cell(m['ic'])} "
                f"| {cell(m['fold_ic_mean'])} | {cell(m['fold_ic_t'], '{:+.2f}')} "
                f"| {cell(m['fold_ic_positive_fraction'], '{:.0%}')} "
                f"| {cell(m['hit_rate'], '{:.1%}')} "
                f"| {cell(m['up_rate'], '{:.1%}')} "
                f"| {cell(m['saturation'], '{:.0%}')} |"
            )
        corr = evaluation["score_correlation"]
        names = list(corr)
        lines += ["", "Score rank correlation:", "", "| | " + " | ".join(names) + " |"]
        lines.append("| --- " * (len(names) + 1) + "|")
        for left in names:
            lines.append(
                f"| {left} | "
                + " | ".join(cell(corr[left][right], "{:+.2f}") for right in names)
                + " |"
            )
        lines.append("")
    portfolio = report["portfolio"]
    span = portfolio["span"]
    lines += [
        f"## Stitched OOS portfolio ({span['bars']} bars, "
        f"{span['first_decision_at'][:10]} to {span['last_execution_at'][:10]})",
        "",
        "| Predictor | Costs | Total return | Annual | Sharpe | Max DD "
        "| Avg turnover |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, modes in portfolio["predictors"].items():
        for mode, m in modes.items():
            lines.append(
                f"| {name} | {mode} | {cell(m['total_return'], '{:+.1%}')} "
                f"| {cell(m['annual_return'], '{:+.1%}')} "
                f"| {cell(m['sharpe'], '{:+.2f}')} "
                f"| {cell(m['max_drawdown'], '{:.1%}')} "
                f"| {cell(m['avg_turnover'], '{:.3f}')} |"
            )
    b = portfolio["equal_weight_benchmark"]
    lines += [
        f"| equal-weight benchmark | none | {cell(b['total_return'], '{:+.1%}')} "
        f"| {cell(b['annual_return'], '{:+.1%}')} | {cell(b['sharpe'], '{:+.2f}')} "
        f"| {cell(b['max_drawdown'], '{:.1%}')} | — |",
        "",
    ]
    incremental = report.get("incremental")
    if incremental:
        criteria = incremental["criteria"]
        lines += [
            "## Incremental information (pre-registered criteria)",
            "",
            f"Orthogonal: max |rank corr| with any V0 expert <= "
            f"{criteria['orthogonal_max_abs_corr']}. Adds information: paired "
            f"per-fold test IC gain of `stacker.v0+X` over `stacker.ridge` > 0 "
            f"with t >= {criteria['incremental_min_t']}.",
            "",
            "| Expert | Max abs corr with V0 | Orthogonal | Mean fold IC gain "
            "| Gain t | Folds improved | Adds information |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, entry in incremental["experts"].items():
            delta = entry["delta_vs_v0_stacker"]
            lines.append(
                f"| {name} | {cell(entry.get('max_abs_corr_with_v0'), '{:.2f}')} "
                f"| {entry.get('orthogonal', '—')} "
                f"| {cell(delta['mean_delta'], '{:+.4f}')} "
                f"| {cell(delta['t'], '{:+.2f}')} "
                f"| {cell(delta['positive_fraction'], '{:.0%}')} "
                f"| {entry.get('adds_incremental_information', '—')} |"
            )
        lines.append("")
    stacker = report["stacker"]
    lines += [
        f"## Ridge stacker (alpha {stacker['alpha']}, "
        f"{len(stacker['folds'])} per-fold fits)",
        "",
        "| Expert | Mean standardized coef | Folds with coef > 0 |",
        "| --- | --- | --- |",
    ]
    for name, summary in stacker["coef_summary"].items():
        lines.append(
            f"| {name} | {cell(summary['mean'], '{:+.5f}')} "
            f"| {cell(summary['positive_fraction'], '{:.0%}')} |"
        )
    lines += ["", stacker["note"], ""]
    lines += ["## Caveats", ""] + [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines) + "\n"


_CAVEATS = (
    "IC and hit rate measure prediction quality only; the portfolio section "
    "adds sizing, engine fills, and slippage.",
    "The stitched portfolio uses validation and test slots, valid only while "
    "nothing is fitted on validation labels. Shorts carry no borrow cost.",
    "Forward horizons overlap within a fold, so pooled IC carries no "
    "significance claim; fold IC t treats folds as independent.",
    "Universe is the caller's symbol list, not point-in-time membership "
    "(survivorship bias is possible).",
    "Prices are provider-adjusted as retrieved and frozen by the snapshot.",
    "The final holdout is locked and was not read for prediction or evaluation.",
)


def _load_snapshot(
    workspace: Path, snapshot_id: str
) -> tuple[Any, dict[str, tuple[MarketBar, ...]]]:
    snapshot = DatasetSnapshotStore(workspace / "snapshots").load(snapshot_id)
    bars_by_asset = {
        asset: tuple(bar for bar in snapshot.bars if bar.asset == asset)
        for asset in snapshot.manifest.assets
    }
    return snapshot, bars_by_asset


def _spec(
    snapshot: Any,
    snapshot_id: str,
    bars_by_asset: Mapping[str, tuple[MarketBar, ...]],
    config: ResearchConfig,
) -> ExperimentSpec:
    assets = snapshot.manifest.assets
    return ExperimentSpec(
        evaluation_protocol_id=_protocol(config, bars_by_asset[assets[0]]).protocol_id,
        expert_config_ids=tuple(e.config_id for e in _experts(config)),
        ensemble_config_id="quorum:research:ensemble:equal-v1",
        target_definition_id=(
            f"quorum:target:next-open-to-close:h{config.horizon_bars}:v1"
        ),
        horizon_bars=config.horizon_bars,
        data_snapshot_id=snapshot_id,
        data_cutoff_at=max((bar.event_at for bar in snapshot.bars), key=_utc),
        universe_id=_universe_id(assets),
        cost_model_id="quorum:cost:engine-us-slippage-grid",
        code_id="quorum:research:v3",
        config_id=config.config_id,
        random_seed=0,
        trial_family_id=TRIAL_FAMILY_ID,
    )


def _preregistration_path(workspace: Path, family: str) -> Path:
    if not _FAMILY_NAME_RE.fullmatch(family):
        raise ValueError("family must be a short name of letters, digits, . _ -")
    return Path(workspace) / "preregistrations" / f"{family}.json"


def preregister(
    workspace: Path,
    snapshot_id: str,
    family: str,
    configs: Mapping[str, ResearchConfig],
) -> dict[str, str]:
    """Register every variant of a family before any of them produces results.

    The ledger's order is the proof: all registrations precede every run. The
    declaration file is written once and never overwritten.
    """
    path = _preregistration_path(workspace, family)
    if path.exists():
        raise FileExistsError(f"preregistration {family!r} already exists")
    if not configs:
        raise ValueError("a preregistered family needs at least one variant")
    snapshot, bars_by_asset = _load_snapshot(Path(workspace), snapshot_id)
    ledger = ExperimentLedger(Path(workspace) / "experiment_ledger.jsonl")
    variants = {}
    for name in sorted(configs):
        attempt = ledger.register(
            _spec(snapshot, snapshot_id, bars_by_asset, configs[name])
        )
        variants[name] = {
            "attempt_id": attempt.attempt_id,
            "spec_fingerprint": attempt.spec_fingerprint,
            "config": asdict(configs[name]),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    declaration = {
        "family": family,
        "snapshot_id": snapshot_id,
        "registered_at": datetime.now(UTC).isoformat(),
        "variants": variants,
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return {name: v["attempt_id"] for name, v in variants.items()}


def run_preregistered(workspace: Path, family: str) -> dict[str, "ResearchResult"]:
    """Run every still-registered variant of a preregistered family."""
    declaration = json.loads(
        _preregistration_path(workspace, family).read_text(encoding="utf-8")
    )
    ledger = ExperimentLedger(Path(workspace) / "experiment_ledger.jsonl")
    results = {}
    for name, variant in sorted(declaration["variants"].items()):
        if ledger.get(variant["attempt_id"]).state is not ExperimentState.REGISTERED:
            continue
        results[name] = run_research(
            workspace,
            declaration["snapshot_id"],
            ResearchConfig(**variant["config"]),
            attempt_id=variant["attempt_id"],
        )
    return results


def run_research(
    workspace: Path,
    snapshot_id: str,
    config: ResearchConfig | None = None,
    *,
    attempt_id: str | None = None,
) -> ResearchResult:
    """Evaluate and report one OOF research attempt on a snapshot.

    Without ``attempt_id`` the attempt is registered now. With it, a previously
    preregistered attempt runs, but only if it is still REGISTERED and its
    registered spec matches this snapshot and config exactly.
    """
    config = config or ResearchConfig()
    workspace = Path(workspace)
    snapshot, bars_by_asset = _load_snapshot(workspace, snapshot_id)
    assets = snapshot.manifest.assets
    ledger = ExperimentLedger(workspace / "experiment_ledger.jsonl")
    spec = _spec(snapshot, snapshot_id, bars_by_asset, config)
    if attempt_id is not None:
        record = ledger.get(attempt_id)
        if record.state is not ExperimentState.REGISTERED:
            raise ValueError(
                f"attempt {attempt_id} is {record.state.value}, not REGISTERED"
            )
        if record.attempt.spec_fingerprint != spec.fingerprint:
            raise ValueError("config does not match the registered spec")
    # Registration precedes every result-producing step.
    attempt = record.attempt if attempt_id is not None else ledger.register(spec)
    ledger.transition(attempt.attempt_id, ExperimentState.RUNNING)
    try:
        calendar = _shared_calendar(bars_by_asset)
        protocol = _protocol(config, calendar)
        ordinary_stop = len(calendar) - config.holdout_bars
        plan = materialize_chronological_plan(
            attempt,
            protocol,
            tuple(TimeInterval(bar.start_at, bar.event_at) for bar in calendar),
            tuple(p + config.horizon_bars for p in range(ordinary_stop)),
        )
        rows, stacker_reports = _oof_rows(plan, bars_by_asset, config)
        slot_counts = {
            role.value: sum(slot.role is role for slot in plan.oof_slots) * len(assets)
            for role in {slot.role for slot in plan.oof_slots}
        }
        evaluation = evaluate_predictions(rows, slot_counts=slot_counts)
        incremental = (
            _incremental(rows, evaluation) if config.expert_set == "v1" else None
        )
        run_dir = workspace / "runs" / attempt.attempt_id
        run_dir.mkdir(parents=True)
        portfolio = run_oos_portfolios(
            rows,
            bars_by_asset,
            holdout_start=protocol.final_holdout.start,
            horizon_bars=config.horizon_bars,
            experiment_id=attempt.attempt_id,
            risk_config=RiskPolicyConfig(
                max_gross_exposure=config.max_gross_exposure,
                max_abs_weight_per_asset=config.max_weight_per_asset,
                max_turnover=config.max_turnover,
            ),
            output_dir=run_dir / "portfolio",
            construction=config.portfolio_construction,
            rebalance_every_bars=config.rebalance_every_bars,
        )
        results = {
            "evaluation": evaluation,
            "portfolio": portfolio,
            "incremental": incremental,
        }
        metrics_sha256 = (
            "sha256:" + hashlib.sha256(_canonical_json(results).encode()).hexdigest()
        )
        holdout = protocol.final_holdout
        report = {
            "attempt_id": attempt.attempt_id,
            "spec_fingerprint": attempt.spec_fingerprint,
            "snapshot": snapshot.manifest.to_dict(),
            "config": asdict(config) | {"config_id": config.config_id},
            "target": {
                "definition_id": spec.target_definition_id,
                "definition": (
                    f"next bar open to close {config.horizon_bars} bars after the "
                    "decision bar"
                ),
            },
            "chronology": {
                "total_bars": len(calendar),
                "ordinary_bars": ordinary_stop,
                "fold_count": len(plan.folds),
                "final_holdout": {
                    "start": _utc(holdout.start),
                    "end": _utc(holdout.end),
                    "state": holdout.state.value,
                },
            },
            "trial_family": {
                "id": TRIAL_FAMILY_ID,
                "attempt_count": ledger.count_attempts(trial_family_id=TRIAL_FAMILY_ID),
            },
            "evaluation": evaluation,
            "incremental": incremental,
            "stacker": stacker_reports[STACKER_PREDICTOR],
            "stackers": stacker_reports,
            "portfolio": portfolio,
            "metrics_sha256": metrics_sha256,
            "caveats": list(_CAVEATS),
        }
        report_path = run_dir / "research_report.json"
        predictions_path = run_dir / "oof_predictions.csv"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "research_report.md").write_text(_markdown(report), encoding="utf-8")
        rows.to_csv(predictions_path, index=False)
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.COMPLETED,
            outcome=ExperimentOutcome(
                detail="OOF prediction evaluation completed",
                references=ExternalRecordRefs(
                    quorum_artifact_ids=(
                        "quorum:research:report:" + metrics_sha256.split(":", 1)[1],
                    )
                ),
                metadata={"metrics_sha256": metrics_sha256},
            ),
        )
    except Exception as exc:
        try:
            ledger.transition(
                attempt.attempt_id,
                ExperimentState.FAILED,
                outcome=ExperimentOutcome(
                    detail=f"research run failed: {type(exc).__name__}"
                ),
            )
        except Exception:
            pass  # keep the original failure; the attempt stays counted either way
        raise
    return ResearchResult(
        attempt.attempt_id, run_dir, report_path, predictions_path, metrics_sha256
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quorum real-data research runner")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="fetch daily bars into a snapshot")
    ingest.add_argument("--workspace", required=True, type=Path)
    ingest.add_argument("--symbols", required=True, help="comma-separated, e.g. SPY.US")
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end", required=True)
    run = commands.add_parser("run", help="evaluate V0 experts on a snapshot")
    run.add_argument("--workspace", required=True, type=Path)
    run.add_argument("--snapshot-id", required=True)
    run.add_argument("--horizon-bars", type=int, default=ResearchConfig().horizon_bars)
    prereg = commands.add_parser(
        "preregister", help="register a family of variants before running any"
    )
    prereg.add_argument("--workspace", required=True, type=Path)
    prereg.add_argument("--snapshot-id", required=True)
    prereg.add_argument("--family", required=True)
    prereg.add_argument(
        "--variants-json",
        required=True,
        type=Path,
        help="JSON object mapping variant name to ResearchConfig overrides",
    )
    run_family = commands.add_parser(
        "run-preregistered", help="run a preregistered family's variants"
    )
    run_family.add_argument("--workspace", required=True, type=Path)
    run_family.add_argument("--family", required=True)
    args = parser.parse_args(argv)

    if args.command == "preregister":
        overrides = json.loads(args.variants_json.read_text(encoding="utf-8"))
        declared = preregister(
            args.workspace,
            args.snapshot_id,
            args.family,
            {name: ResearchConfig(**o) for name, o in overrides.items()},
        )
        print(_canonical_json(declared))
        return 0
    if args.command == "run-preregistered":
        results = run_preregistered(args.workspace, args.family)
        reports = {name: str(r.report_path) for name, r in results.items()}
        print(_canonical_json(reports))
        return 0

    if args.command == "ingest":
        manifest = ingest_daily_bars(
            DatasetSnapshotStore(args.workspace / "snapshots"),
            [s.strip() for s in args.symbols.split(",") if s.strip()],
            args.start,
            args.end,
        )
        print(
            _canonical_json(
                {
                    "snapshot_id": manifest.snapshot_id,
                    "assets": list(manifest.assets),
                    "row_count": manifest.row_count,
                }
            )
        )
        return 0
    result = run_research(
        args.workspace, args.snapshot_id, ResearchConfig(horizon_bars=args.horizon_bars)
    )
    print(
        _canonical_json(
            {
                "attempt_id": result.attempt_id,
                "report": str(result.report_path),
                "metrics_sha256": result.metrics_sha256,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
