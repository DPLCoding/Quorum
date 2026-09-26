"""Narrow adapter from final Quorum risk targets to Vibe signal series."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from src.quorum.risk import RiskPolicyResult, RiskRebalance

RiskStreamKey = tuple[str | None, str | None, int]


def _exact_number(name: str, value: object, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number != expected:
        raise ValueError(f"{name} must be exactly {expected}")


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    return value


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


def _timestamp_ns(value: object) -> int:
    return int(pd.Timestamp(value).value)


def _naive_wall_clock_instant(asset: str, value: object, zone: ZoneInfo) -> int:
    timestamp = pd.Timestamp(value)
    wall = timestamp.to_pydatetime(warn=False)
    candidates: list[datetime] = []
    for fold in (0, 1):
        candidate = wall.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
        if round_trip.replace(tzinfo=None) == wall:
            candidates.append(candidate)

    offsets = {candidate.utcoffset() for candidate in candidates}
    if not candidates:
        raise ValueError(
            f"nonexistent local market time for asset {asset!r}: {timestamp}"
        )
    if len(offsets) != 1:
        raise ValueError(
            f"ambiguous local market time for asset {asset!r}: {timestamp}"
        )
    offset = candidates[0].utcoffset()
    if offset is None:
        raise ValueError(f"timezone offset is unavailable for asset {asset!r}")
    return _timestamp_ns(timestamp) - int(pd.Timedelta(offset).value)


def _index_instants(
    asset: str,
    frame: pd.DataFrame,
    timezone_name: str | None,
) -> tuple[int, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"data_map[{asset!r}] must be a pandas DataFrame")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"data_map[{asset!r}] index must be a DatetimeIndex")
    if index.tz is not None:
        if timezone_name is not None:
            raise ValueError(
                f"timezone declaration for aware market index {asset!r} is redundant"
            )
        instants = tuple(_timestamp_ns(value) for value in index)
    else:
        if timezone_name is None:
            raise ValueError(
                f"naive market index requires explicit timezone for asset {asset!r}"
            )
        zone = ZoneInfo(timezone_name)
        instants = tuple(
            _naive_wall_clock_instant(asset, value, zone) for value in index
        )
    if any(right <= left for left, right in zip(instants, instants[1:])):
        raise ValueError(
            f"data_map[{asset!r}] index must contain unique increasing instants"
        )
    return instants


def _canonical_stream_key(value: object) -> RiskStreamKey:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("stream_key must be a three-item tuple")
    experiment_id, split_id, horizon_bars = value
    for name, identifier in (
        ("experiment_id", experiment_id),
        ("split_id", split_id),
    ):
        if identifier is not None:
            _required_text(name, identifier)
    if split_id is not None and experiment_id is None:
        raise ValueError("stream_key split_id requires experiment_id")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int):
        raise TypeError("stream_key horizon_bars must be an integer")
    if horizon_bars < 1:
        raise ValueError("stream_key horizon_bars must be positive")
    return (experiment_id, split_id, horizon_bars)


def _stream_sort_key(value: RiskStreamKey) -> tuple[Any, ...]:
    return (
        (value[0] is not None, value[0] or ""),
        (value[1] is not None, value[1] or ""),
        value[2],
    )


def _canonical_timezones(
    value: Mapping[str, str] | None,
) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("market_timezones must be a mapping")
    normalized: dict[str, str] = {}
    for asset, timezone_name in value.items():
        asset = _required_text("market timezone asset", asset)
        timezone_name = _required_text("market timezone name", timezone_name)
        try:
            ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"unknown market timezone {timezone_name!r} for asset {asset!r}"
            ) from exc
        normalized[asset] = timezone_name
    return MappingProxyType(dict(sorted(normalized.items())))


def _canonical_starts(
    value: Mapping[str, Sequence[datetime]] | None,
) -> Mapping[str, tuple[datetime, ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("bar_starts must be a mapping")
    normalized: dict[str, tuple[datetime, ...]] = {}
    for asset, starts in value.items():
        asset = _required_text("bar start asset", asset)
        if isinstance(starts, (str, bytes)) or not isinstance(starts, Sequence):
            raise TypeError(f"bar starts for {asset!r} must be a sequence")
        for start in starts:
            if not isinstance(start, datetime):
                raise TypeError(f"bar starts for {asset!r} must be datetimes")
            if start.tzinfo is None or start.utcoffset() is None:
                raise ValueError(f"bar starts for {asset!r} must be timezone-aware")
        normalized[asset] = tuple(starts)
    return MappingProxyType(dict(sorted(normalized.items())))


def _open_instants(
    asset: str, starts: Sequence[datetime], row_ends: Sequence[int]
) -> tuple[int, ...]:
    """Validate observed bar starts against the end-labelled market index."""
    if len(starts) != len(row_ends):
        raise ValueError(f"bar_starts for {asset!r} must hold one start per row")
    opens = tuple(_timestamp_ns(start) for start in starts)
    for position, (open_ns, end_ns) in enumerate(zip(opens, row_ends)):
        if open_ns >= end_ns:
            raise ValueError(
                f"bar start for {asset!r} at row {position} must precede its row"
            )
        if position and open_ns < row_ends[position - 1]:
            raise ValueError(
                f"bar start for {asset!r} at row {position} overlaps the previous bar"
            )
    return opens


@dataclass(frozen=True, slots=True)
class VibeSignalAdapter:
    """Expose one selected risk stream through Vibe's generate(data_map) shape.

    Index rows label bar ends, and the engine fills at the next row's open, so
    ``decision_at`` must not be later than that open. Equality is accepted: a
    decision at the close fills at a next open occurring at the same instant,
    i.e. zero decision-to-order latency. ``bar_starts`` gives each asset's
    observed bar start per index row, making the next open authoritative.
    Without it the next bar may open at this row's close, so ``decision_at``
    may not be later than ``event_at``.
    """

    risk_result: RiskPolicyResult
    stream_key: RiskStreamKey | None = None
    market_timezones: Mapping[str, str] | None = None
    bar_starts: Mapping[str, Sequence[datetime]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.risk_result, RiskPolicyResult):
            raise TypeError("risk_result must be a RiskPolicyResult")
        streams = tuple(
            sorted(
                {rebalance.stream_key for rebalance in self.risk_result.rebalances},
                key=_stream_sort_key,
            )
        )
        if self.stream_key is None:
            if not streams:
                raise ValueError("risk result contains no risk streams")
            if len(streams) != 1:
                raise ValueError(
                    "risk result contains multiple streams; explicit stream_key required"
                )
            stream_key = streams[0]
        else:
            stream_key = _canonical_stream_key(self.stream_key)
            if stream_key not in streams:
                raise ValueError(f"unknown risk stream selector {stream_key!r}")
        object.__setattr__(self, "stream_key", stream_key)
        object.__setattr__(
            self, "market_timezones", _canonical_timezones(self.market_timezones)
        )
        object.__setattr__(self, "bar_starts", _canonical_starts(self.bar_starts))

    def _selected_rebalances(self) -> tuple[RiskRebalance, ...]:
        return tuple(
            rebalance
            for rebalance in self.risk_result.rebalances
            if rebalance.stream_key == self.stream_key
        )

    def generate(self, data_map: Mapping[str, pd.DataFrame]) -> dict[str, pd.Series]:
        if not isinstance(data_map, Mapping):
            raise TypeError("data_map must be a mapping")
        rebalances = self._selected_rebalances()
        if any(not rebalance.executable for rebalance in rebalances):
            raise ValueError(
                "cannot adapt a risk result containing an incomplete group"
            )

        targets_by_asset: dict[str, list[tuple[int, float]]] = {}
        seen_slots: set[tuple[str, int]] = set()
        required_assets = {
            target.asset for rebalance in rebalances for target in rebalance.targets
        }
        missing_assets = sorted(required_assets - set(data_map))
        if missing_assets:
            raise ValueError(
                f"data_map is missing required risk assets: {missing_assets}"
            )

        index_instants: dict[str, tuple[int, ...]] = {}
        for asset in sorted(required_assets):
            index_instants[asset] = _index_instants(
                asset,
                data_map[asset],
                self.market_timezones.get(asset),
            )
        open_instants = {
            asset: _open_instants(asset, self.bar_starts[asset], index_instants[asset])
            for asset in sorted(required_assets)
            if asset in self.bar_starts
        }

        for rebalance in rebalances:
            resolved_targets: list[tuple[str, int, float]] = []
            execution_instants: set[int] = set()
            for target in rebalance.targets:
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
                execution_instant = instants[position + 1]
                slot = (target.asset, position)
                if slot in seen_slots:
                    raise ValueError(
                        "multiple risk decisions map to one signal slot "
                        f"for {target.asset!r}"
                    )
                seen_slots.add(slot)
                decision_instant = _timestamp_ns(target.decision_at)
                opens = open_instants.get(target.asset)
                if opens is None:
                    if decision_instant > event_instant:
                        raise ValueError(
                            f"decision_at for {target.asset!r} is later than its "
                            "bar close; the next bar may open at that instant"
                        )
                elif decision_instant > opens[position + 1]:
                    raise ValueError(
                        f"decision_at for {target.asset!r} is later than its "
                        "next bar open"
                    )
                execution_instants.add(execution_instant)
                resolved_targets.append((target.asset, position, final_weight))

            if len(execution_instants) != 1:
                raise ValueError(
                    "risk rebalance does not map to one coherent Vibe "
                    "execution instant"
                )
            for asset, position, final_weight in resolved_targets:
                targets_by_asset.setdefault(asset, []).append((position, final_weight))

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
