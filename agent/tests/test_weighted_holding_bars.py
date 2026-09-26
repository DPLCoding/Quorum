"""Weighted holding bars stay exact while scaling to long-held rebalanced positions.

A position rebalanced every bar but never fully closed (e.g. a long-only tilt)
has an unbounded active lifecycle, so replaying every fill and rescaling every
lot on each reduction made one call quadratic and a run roughly cubic.
"""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.engines.base import BaseEngine
from backtest.models import FillRecord


class _Engine(BaseEngine):
    def can_execute(self, symbol, direction, bar):  # noqa: ANN001, ANN201
        return True

    def round_size(self, raw_size, price):  # noqa: ANN001, ANN201
        return float(raw_size)

    def calc_commission(self, size, price, direction, is_open):  # noqa: ANN001, ANN201
        return 0.0

    def apply_slippage(self, price, direction):  # noqa: ANN001, ANN201
        return float(price)


def _reference(fills: list[FillRecord], symbol: str, now: int, entry: int) -> float:
    """The pre-fix algorithm, verbatim: replay the active lifecycle each call."""
    active = []
    for fill in reversed(fills):
        if fill.symbol != symbol:
            continue
        if fill.action == "close":
            break
        active.append(fill)
    lots: list[list[float]] = []
    for fill in reversed(active):
        quantity = abs(fill.signed_quantity)
        if fill.action in {"open", "increase"}:
            lots.append([quantity, float(fill.bar_idx)])
            continue
        if fill.action not in {"reduce", "close"}:
            continue
        total = sum(lot[0] for lot in lots)
        if total <= 1e-12 or quantity >= total - 1e-12:
            lots.clear()
            continue
        for lot in lots:
            lot[0] *= (total - quantity) / total
    total = sum(lot[0] for lot in lots)
    if total <= 1e-12:
        return float(max(now - entry, 0))
    return sum(q * max(now - int(b), 0) for q, b in lots) / total


def _fill(symbol: str, bar: int, action: str, quantity: float) -> FillRecord:
    return FillRecord(
        symbol=symbol,
        timestamp=pd.Timestamp("2024-01-01"),
        bar_idx=bar,
        action=action,
        signed_quantity=quantity,
        notional=abs(quantity),
        execution_price=1.0,
        fee=0.0,
        margin=abs(quantity),
        reason="target_rebalance",
    )


def test_matches_reference_replay_on_random_multi_symbol_lifecycles() -> None:
    rng = random.Random(7)
    engine = _Engine({"initial_cash": 1_000.0})
    held = {"A": 0.0, "B": 0.0}
    for bar in range(600):
        engine._bar_idx = bar
        for symbol in ("A", "B"):
            if held[symbol] <= 0.0:
                action, quantity = "open", rng.uniform(1.0, 10.0)
            else:
                action = rng.choices(
                    ["increase", "reduce", "close", "funding"], [45, 45, 5, 5]
                )[0]
                quantity = {
                    "increase": rng.uniform(0.1, 5.0),
                    "reduce": held[symbol] * rng.uniform(0.01, 0.99),
                    "close": held[symbol],
                    "funding": 0.0,
                }[action]
            entry = max(bar - 3, 0)
            position = SimpleNamespace(symbol=symbol, entry_bar_idx=entry)
            expected = _reference(engine.fill_records, symbol, bar, entry)
            assert engine._weighted_holding_bars(position) == pytest.approx(
                expected, rel=1e-9, abs=1e-9
            )
            engine.fill_records.append(_fill(symbol, bar, action, quantity))
            held[symbol] += {"open": 1, "increase": 1, "reduce": -1}.get(
                action, 0
            ) * quantity
            if action == "close":
                held[symbol] = 0.0

    # A replaced ledger (e.g. a reset engine) is recomputed, not mixed with cache.
    engine.fill_records = [_fill("A", 0, "open", 2.0), _fill("A", 1, "increase", 2.0)]
    engine._bar_idx = 3
    position = SimpleNamespace(symbol="A", entry_bar_idx=0)
    assert engine._weighted_holding_bars(position) == pytest.approx(2.5)


def test_long_held_position_rebalanced_every_bar_stays_fast() -> None:
    engine = _Engine({"initial_cash": 1_000.0})
    position = SimpleNamespace(symbol="A", entry_bar_idx=0)
    engine.fill_records.append(_fill("A", 0, "open", 100.0))
    started = time.perf_counter()
    for bar in range(1, 1_000):
        engine._bar_idx = bar
        engine._weighted_holding_bars(position)
        action = "increase" if bar % 2 else "reduce"
        engine.fill_records.append(_fill("A", bar, action, 1.0))
    assert time.perf_counter() - started < 1.0
