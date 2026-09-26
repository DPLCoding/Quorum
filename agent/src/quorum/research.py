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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

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
from src.quorum.evaluation import evaluate_predictions
from src.quorum.experiments import (
    ExperimentLedger,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentState,
    ExternalRecordRefs,
)
from src.quorum.experts import MeanReversionExpert, MomentumExpert, TrendExpert
from src.quorum.validation import materialize_chronological_plan

UTC = timezone.utc
US_EQUITY_TIMEZONE = "America/New_York"
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
TIMESTAMP_CONVENTION = "bar:[regular-open,regular-close):America/New_York:v1"
ENSEMBLE_PREDICTOR = "ensemble"
TRIAL_FAMILY_ID = "quorum:research:v0-experts"
# Longest V0 lookback is SMA50; a trailing window keeps prediction O(n).
_EXPERT_WINDOW_BARS = 60
_EXPERTS = (MomentumExpert(), TrendExpert(), MeanReversionExpert())


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

    def __post_init__(self) -> None:
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


def _ensemble_config(config: ResearchConfig) -> StaticEnsembleConfig:
    weight = 1.0 / len(_EXPERTS)
    return StaticEnsembleConfig(
        experts=tuple(
            ExpertWeight(e.expert_id, e.expert_version, weight) for e in _EXPERTS
        ),
        sell_threshold=config.sell_threshold,
        buy_threshold=config.buy_threshold,
    )


def _predict(
    asset: str,
    bars: Sequence[MarketBar],
    position: int,
    *,
    horizon_bars: int,
    experiment_id: str,
    split_id: str,
) -> list[ExpertPrediction]:
    """Run every expert on the trailing window ending at ``position`` only."""
    window = bars[max(0, position + 1 - _EXPERT_WINDOW_BARS) : position + 1]
    latest_available = max((bar.available_at for bar in window), key=_utc)
    bar = bars[position]
    prepared = {
        "asset": asset,
        "close": tuple(b.close for b in window),
        "event_at": tuple(b.event_at for b in window),
        "available_at": tuple(b.available_at for b in window),
    }
    context = PredictionContext(
        event_at=bar.event_at,
        available_at=latest_available,
        decision_at=max(bar.event_at, latest_available, key=_utc),
        horizon_bars=horizon_bars,
        experiment_id=experiment_id,
        split_id=split_id,
    )
    return [p for e in _EXPERTS for p in e.predict(prepared, context).predictions]


def _oof_rows(
    plan: Any,
    bars_by_asset: Mapping[str, tuple[MarketBar, ...]],
    config: ResearchConfig,
) -> pd.DataFrame:
    """Predict every authorized slot and attach its tradable forward return.

    The target enters at the next bar's open (the engine's fill) and exits at
    the close ``horizon_bars`` bars after the decision bar.
    """
    h = config.horizon_bars
    ensemble = StaticEnsemble(_ensemble_config(config))
    rows: list[dict[str, Any]] = []
    for asset, bars in sorted(bars_by_asset.items()):
        slot_by_event: dict[str, Any] = {}
        predictions: list[ExpertPrediction] = []
        for slot in plan.oof_slots:
            p = slot.sample_position
            slot_by_event[_utc(bars[p].event_at)] = (slot, p)
            predictions.extend(
                _predict(
                    asset,
                    bars,
                    p,
                    horizon_bars=h,
                    experiment_id=plan.attempt_id,
                    split_id=slot.split_id,
                )
            )
        combined = ensemble.combine(ExpertResult(tuple(predictions)))
        scored = [
            (p.expert_id, p.event_at, p.decision_at, p.score) for p in predictions
        ]
        scored += [
            (ENSEMBLE_PREDICTOR, d.event_at, d.decision_at, d.combined_score)
            for d in combined.decisions
            if d.combined_score is not None
        ]
        for predictor, event_at, decision_at, score in scored:
            slot, p = slot_by_event[_utc(event_at)]
            entry, exit_ = bars[p + 1], bars[p + h]
            rows.append(
                {
                    "predictor": predictor,
                    "asset": asset,
                    "event_at": _utc(event_at),
                    "decision_at": _utc(decision_at),
                    "label_start_at": _utc(entry.start_at),
                    "label_end_at": _utc(exit_.event_at),
                    "fold_index": slot.fold_index,
                    "split_id": slot.split_id,
                    "role": slot.role.value,
                    "score": score,
                    "forward_return": exit_.close / entry.open - 1.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["predictor", "asset", "event_at"])


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
    lines += ["## Caveats", ""] + [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines) + "\n"


_CAVEATS = (
    "Prediction quality only: no sizing, costs, or P&L.",
    "Forward horizons overlap within a fold, so pooled IC carries no "
    "significance claim; fold IC t treats folds as independent.",
    "Universe is the caller's symbol list, not point-in-time membership "
    "(survivorship bias is possible).",
    "Prices are provider-adjusted as retrieved and frozen by the snapshot.",
    "The final holdout is locked and was not read for prediction or evaluation.",
)


def run_research(
    workspace: Path, snapshot_id: str, config: ResearchConfig | None = None
) -> ResearchResult:
    """Register, evaluate, and report one OOF research attempt on a snapshot."""
    config = config or ResearchConfig()
    workspace = Path(workspace)
    snapshot = DatasetSnapshotStore(workspace / "snapshots").load(snapshot_id)
    assets = snapshot.manifest.assets
    bars_by_asset = {
        asset: tuple(bar for bar in snapshot.bars if bar.asset == asset)
        for asset in assets
    }
    ledger = ExperimentLedger(workspace / "experiment_ledger.jsonl")
    spec = ExperimentSpec(
        evaluation_protocol_id=_protocol(config, bars_by_asset[assets[0]]).protocol_id,
        expert_config_ids=tuple(e.config_id for e in _EXPERTS),
        ensemble_config_id="quorum:research:ensemble:equal-v1",
        target_definition_id=(
            f"quorum:target:next-open-to-close:h{config.horizon_bars}:v1"
        ),
        horizon_bars=config.horizon_bars,
        data_snapshot_id=snapshot_id,
        data_cutoff_at=max((bar.event_at for bar in snapshot.bars), key=_utc),
        universe_id=_universe_id(assets),
        cost_model_id="quorum:cost:none:prediction-evaluation",
        code_id="quorum:research:v1",
        config_id=config.config_id,
        random_seed=0,
        trial_family_id=TRIAL_FAMILY_ID,
    )
    # Registration precedes every result-producing step.
    attempt = ledger.register(spec)
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
        rows = _oof_rows(plan, bars_by_asset, config)
        slot_counts = {
            role.value: sum(slot.role is role for slot in plan.oof_slots) * len(assets)
            for role in {slot.role for slot in plan.oof_slots}
        }
        evaluation = evaluate_predictions(rows, slot_counts=slot_counts)
        metrics_sha256 = (
            "sha256:" + hashlib.sha256(_canonical_json(evaluation).encode()).hexdigest()
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
            "metrics_sha256": metrics_sha256,
            "caveats": list(_CAVEATS),
        }
        run_dir = workspace / "runs" / attempt.attempt_id
        run_dir.mkdir(parents=True)
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
    args = parser.parse_args(argv)

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
