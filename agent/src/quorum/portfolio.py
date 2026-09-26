"""Stitched out-of-sample portfolio evaluation through the unchanged Vibe engine.

Each predictor's OOF scores run the designed V0 path as one continuous stream:
single-predictor static ensemble -> frozen risk policy -> Vibe adapter (with
observed bar starts) -> ``GlobalEquityEngine`` fills, costs, and accounting.
No execution logic is reimplemented here; only frames and configs are built.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engines.global_equity import GlobalEquityEngine
from src.quorum.adapters import VibeSignalAdapter, validate_v0_vibe_config
from src.quorum.contracts import (
    ExpertPrediction,
    ExpertResult,
    ExpertWeight,
    StaticEnsembleConfig,
)
from src.quorum.data import MarketBar
from src.quorum.ensemble import StaticEnsemble
from src.quorum.risk import FixedRiskPolicy, RiskPolicyConfig

COST_MODES: tuple[tuple[str, float], ...] = (
    ("frictionless", 0.0),
    ("slippage-5bp", 0.0005),
    ("slippage-15bp", 0.0015),
)
INITIAL_CASH = 1_000_000.0
BARS_PER_YEAR = 252
_METRICS = (
    "total_return",
    "annual_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "trade_count",
    "benchmark_return",
    "excess_return",
)


class _FramesLoader:
    """In-memory loader over snapshot-derived frames; never fetches."""

    name = "quorum-snapshot"

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(
        self, codes, start_date, end_date, fields=None, interval="1D"
    ):  # noqa: ANN001, ANN201
        return {code: self.frames[code].copy(deep=True) for code in codes}


def _frames(
    bars_by_asset: Mapping[str, Sequence[MarketBar]], start: int, stop: int
) -> dict[str, pd.DataFrame]:
    """End-labelled (UTC ``event_at``) OHLCV frames for positions [start, stop)."""
    return {
        asset: pd.DataFrame(
            [(b.open, b.high, b.low, b.close, b.volume) for b in bars[start:stop]],
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex(
                [pd.Timestamp(b.event_at).tz_convert("UTC") for b in bars[start:stop]],
                name="timestamp",
            ),
        )
        for asset, bars in bars_by_asset.items()
    }


def _series_stats(returns: pd.Series) -> dict[str, float | None]:
    equity = (1.0 + returns).cumprod()
    years = len(returns) / BARS_PER_YEAR
    std = returns.std(ddof=1)
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": (
            float(returns.mean() / std * math.sqrt(BARS_PER_YEAR))
            if std > 0.0
            else None
        ),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }


def run_oos_portfolios(
    rows: pd.DataFrame,
    bars_by_asset: Mapping[str, Sequence[MarketBar]],
    *,
    holdout_start: datetime,
    horizon_bars: int,
    experiment_id: str,
    risk_config: RiskPolicyConfig,
    output_dir: Path,
    construction: str = "long_short",
    rebalance_every_bars: int = 1,
) -> dict[str, Any]:
    """Run every predictor's stitched OOF targets through the engine per cost mode.

    ``rows`` are the runner's realized OOF prediction rows (validation and test
    slots). Using validation slots is valid only while nothing is fitted on
    validation labels; they make the out-of-sample span contiguous.

    ``long_only_tilt`` maps a score to ``(1 + score) / 2`` before the risk
    policy, so a neutral score holds the equal-weight name cap, ``-1`` drops
    the name, and ``+1`` doubles it; the gross cap rescales the basket.
    ``rebalance_every_bars`` keeps only decisions on every n-th bar of the
    span; the adapter holds each target in between.
    """
    reference = bars_by_asset[min(bars_by_asset)]
    position_of = {
        pd.Timestamp(b.event_at).tz_convert("UTC"): p for p, b in enumerate(reference)
    }
    event_times = pd.to_datetime(rows["event_at"], utc=True)
    first = min(position_of[t] for t in event_times)
    # One bar past the last decision so its target can execute.
    stop = max(position_of[t] for t in event_times) + 2
    if reference[stop - 1].event_at >= holdout_start:
        raise ValueError("stitched portfolio execution would reach the final holdout")
    offsets = [position_of[t] - first for t in event_times]
    rows = rows[[offset % rebalance_every_bars == 0 for offset in offsets]]
    if construction == "long_only_tilt":
        rows = rows.assign(score=(1.0 + rows["score"]) / 2.0)
    elif construction != "long_short":
        raise ValueError(f"unknown portfolio construction {construction!r}")
    frames = _frames(bars_by_asset, first, stop)
    bar_starts = {
        asset: [b.start_at for b in bars[first:stop]]
        for asset, bars in bars_by_asset.items()
    }

    predictors: dict[str, Any] = {}
    benchmark: dict[str, float | None] = {}
    for name, group in rows.groupby("predictor", sort=True):
        predictions = tuple(
            ExpertPrediction(
                expert_id=name,
                expert_version="oof",
                asset=row.asset,
                event_at=at,
                available_at=at,
                decision_at=at,
                horizon_bars=horizon_bars,
                score=row.score,
                experiment_id=experiment_id,
            )
            for row, at in zip(
                group.itertuples(index=False),
                [
                    t.to_pydatetime()
                    for t in pd.to_datetime(group["event_at"], utc=True)
                ],
            )
        )
        single = StaticEnsembleConfig(
            experts=(ExpertWeight(name, "oof", 1.0),),
            sell_threshold=-0.2,
            buy_threshold=0.2,
        )
        decisions = StaticEnsemble(single).combine(ExpertResult(predictions))
        adapter = VibeSignalAdapter(
            FixedRiskPolicy(risk_config).apply(decisions), bar_starts=bar_starts
        )
        predictor_dir = output_dir / name
        predictor_dir.mkdir(parents=True)
        signals = adapter.generate(frames)
        pd.DataFrame(signals).to_csv(predictor_dir / "signals.csv")

        modes: dict[str, Any] = {}
        for mode, slippage in COST_MODES:
            run_dir = predictor_dir / mode
            config = {
                "codes": sorted(frames),
                "start_date": frames[min(frames)].index[0].isoformat(),
                "end_date": frames[min(frames)].index[-1].isoformat(),
                "interval": "1D",
                "engine": "global_equity",
                "source": "quorum-snapshot",
                "initial_cash": INITIAL_CASH,
                "position_adjustment": "rebalance",
                "optimizer": None,
                "constraints": [],
                "rebalance_mask": None,
                "rebalance_tolerance": 0.0,
                "leverage": 1.0,
                "slippage_us": slippage,
            }
            validate_v0_vibe_config(config)
            with contextlib.redirect_stdout(io.StringIO()):
                GlobalEquityEngine(config, market="us").run_backtest(
                    config,
                    _FramesLoader(frames),
                    adapter,
                    run_dir,
                    bars_per_year=BARS_PER_YEAR,
                )
            metrics = pd.read_csv(run_dir / "artifacts" / "metrics.csv").iloc[0]
            modes[mode] = {key: float(metrics[key]) for key in _METRICS}
            if not benchmark:
                equity = pd.read_csv(run_dir / "artifacts" / "equity.csv")
                benchmark = _series_stats(
                    equity["benchmark_equity"].pct_change().dropna()
                )
        predictors[str(name)] = modes

    return {
        "span": {
            "first_decision_at": reference[first].event_at.isoformat(),
            "last_execution_at": reference[stop - 1].event_at.isoformat(),
            "bars": stop - first,
        },
        "risk_config": risk_config.to_dict(),
        "construction": construction,
        "rebalance_every_bars": rebalance_every_bars,
        "cost_modes": dict(COST_MODES),
        "predictors": predictors,
        "equal_weight_benchmark": benchmark,
    }


__all__ = ["COST_MODES", "run_oos_portfolios"]
