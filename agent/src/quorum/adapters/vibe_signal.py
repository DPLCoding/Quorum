"""Narrow adapter from final Quorum risk targets to Vibe signal series."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd

from src.quorum.risk import RiskPolicyResult


def _exact_number(name: str, value: object, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number != expected:
        raise ValueError(f"{name} must be exactly {expected}")


def validate_v0_vibe_config(config: Mapping[str, Any]) -> None:
    """Reject engine options that would rewrite or suppress Quorum V0 targets."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if config.get("position_adjustment") != "rebalance":
        raise ValueError("position_adjustment must be 'rebalance' for Quorum V0")
    if config.get("optimizer") is not None:
        raise ValueError("optimizer must be absent or None for Quorum V0")
    constraints = config.get("constraints")
    if constraints is not None and not (
        isinstance(constraints, list) and not constraints
    ):
        raise ValueError("constraints must be absent, None, or [] for Quorum V0")
    if config.get("rebalance_mask") is not None:
        raise ValueError("rebalance_mask must be absent or None for Quorum V0")
    if "rebalance_tolerance" in config:
        _exact_number("rebalance_tolerance", config["rebalance_tolerance"], 0.0)
    if "leverage" in config:
        _exact_number("leverage", config["leverage"], 1.0)


def _index_instants(asset: str, frame: pd.DataFrame) -> tuple[int, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"data_map[{asset!r}] must be a pandas DataFrame")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"data_map[{asset!r}] index must be a DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"data_map[{asset!r}] index must be timezone-aware")
    instants = tuple(_timestamp_ns(value) for value in index)
    if any(right <= left for left, right in zip(instants, instants[1:])):
        raise ValueError(
            f"data_map[{asset!r}] index must contain unique increasing instants"
        )
    return instants


def _timestamp_ns(value: object) -> int:
    return int(pd.Timestamp(value).value)


@dataclass(frozen=True, slots=True)
class VibeSignalAdapter:
    """Expose final constrained targets through Vibe's generate(data_map) shape."""

    risk_result: RiskPolicyResult

    def __post_init__(self) -> None:
        if not isinstance(self.risk_result, RiskPolicyResult):
            raise TypeError("risk_result must be a RiskPolicyResult")

    def generate(self, data_map: Mapping[str, pd.DataFrame]) -> dict[str, pd.Series]:
        if not isinstance(data_map, Mapping):
            raise TypeError("data_map must be a mapping")
        if any(not rebalance.executable for rebalance in self.risk_result.rebalances):
            raise ValueError(
                "cannot adapt a risk result containing an incomplete group"
            )

        targets_by_asset: dict[str, list[tuple[int, float]]] = {}
        seen_slots: set[tuple[str, int]] = set()
        required_assets = {target.asset for target in self.risk_result.targets}
        missing_assets = sorted(required_assets - set(data_map))
        if missing_assets:
            raise ValueError(
                f"data_map is missing required risk assets: {missing_assets}"
            )

        index_instants: dict[str, tuple[int, ...]] = {}
        for asset in sorted(required_assets):
            index_instants[asset] = _index_instants(asset, data_map[asset])

        for target in self.risk_result.targets:
            final_weight = target.final_target_weight
            if final_weight is None:
                raise ValueError(
                    "executable risk target is missing final_target_weight"
                )
            instants = index_instants[target.asset]
            event_instant = _timestamp_ns(target.event_at)
            position = bisect_left(instants, event_instant)
            if position >= len(instants) or instants[position] != event_instant:
                raise ValueError(
                    f"event_at for {target.asset!r} does not map to its market calendar"
                )
            if position + 1 >= len(instants):
                raise ValueError(
                    f"event_at for {target.asset!r} has no next executable bar"
                )
            if _timestamp_ns(target.decision_at) >= instants[position + 1]:
                raise ValueError(
                    f"decision_at for {target.asset!r} is not before its next bar"
                )
            slot = (target.asset, position)
            if slot in seen_slots:
                raise ValueError(
                    f"multiple risk decisions map to one signal slot for {target.asset!r}"
                )
            seen_slots.add(slot)
            targets_by_asset.setdefault(target.asset, []).append(
                (position, final_weight)
            )

        signals: dict[str, pd.Series] = {}
        for asset in sorted(targets_by_asset):
            frame = data_map[asset]
            signal = pd.Series(float("nan"), index=frame.index, dtype="float64")
            for position, final_weight in sorted(targets_by_asset[asset]):
                signal.iloc[position] = final_weight
            signal = signal.ffill().fillna(0.0)
            if not signal.index.equals(frame.index):
                raise ValueError("adapter changed the market-data index")
            values = tuple(float(value) for value in signal)
            if any(
                not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values
            ):
                raise ValueError("adapter generated an invalid Vibe target weight")
            signals[asset] = signal
        return signals


__all__ = ["VibeSignalAdapter", "validate_v0_vibe_config"]
