"""Offline tests for the Quorum real-data research runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.quorum import DatasetSnapshotStore, ExperimentLedger, ExperimentState
from src.quorum.research import (
    ResearchConfig,
    daily_bars_from_frame,
    ingest_daily_bars,
    run_research,
)

UTC = timezone.utc
RETRIEVED_AT = datetime(2030, 1, 1, tzinfo=UTC)
SMALL = ResearchConfig(
    horizon_bars=3,
    minimum_train_bars=60,
    validation_bars=10,
    test_bars=20,
    holdout_bars=40,
)


def _frame(seed: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
    open_ = close * np.exp(rng.normal(0.0, 0.004, len(dates)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, len(dates)).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="trade_date"),
    )


class _FakeLoader:
    """Loader-shaped test double: naive trade dates, like Vibe's loaders."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(
        self, codes, start_date, end_date, *, interval="1D", fields=None
    ):  # noqa: ANN001, ANN201
        return {code: self.frames[code].copy() for code in codes}


DATES = pd.bdate_range("2023-01-02", periods=300)
FRAMES = {"AAA.US": _frame(1, DATES), "BBB.US": _frame(2, DATES)}


def _ingest(store: DatasetSnapshotStore, frames=FRAMES) -> str:  # noqa: ANN001
    manifest = ingest_daily_bars(
        store,
        sorted(frames),
        "2023-01-02",
        "2024-02-23",
        loader=_FakeLoader(frames),
        source_id="test-fixture",
        adjustment_id="synthetic:none",
        retrieved_at=RETRIEVED_AT,
    )
    return manifest.snapshot_id


def test_daily_bars_use_the_new_york_regular_session_across_dst() -> None:
    frame = _frame(3, pd.DatetimeIndex(["2024-01-02", "2024-07-01"]))
    winter, summer = daily_bars_from_frame("AAA.US", frame)

    assert winter.start_at.astimezone(UTC) == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert winter.event_at.astimezone(UTC) == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert summer.start_at.astimezone(UTC) == datetime(2024, 7, 1, 13, 30, tzinfo=UTC)
    assert summer.event_at.astimezone(UTC) == datetime(2024, 7, 1, 20, 0, tzinfo=UTC)
    assert winter.available_at == winter.event_at


def test_ingest_refuses_unclosed_bars_and_missing_values(tmp_path: Path) -> None:
    store = DatasetSnapshotStore(tmp_path / "snapshots")
    with pytest.raises(ValueError, match="not closed"):
        ingest_daily_bars(
            store,
            ["AAA.US"],
            "2023-01-02",
            "2024-02-23",
            loader=_FakeLoader(FRAMES),
            source_id="test-fixture",
            adjustment_id="synthetic:none",
            retrieved_at=datetime(2023, 6, 1, tzinfo=UTC),
        )

    holed = {"AAA.US": FRAMES["AAA.US"].copy()}
    holed["AAA.US"].iloc[5, holed["AAA.US"].columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        _ingest(store, holed)


def test_research_run_registers_first_keeps_holdout_locked_and_is_deterministic(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    snapshot_id = _ingest(DatasetSnapshotStore(workspace / "snapshots"))

    first = run_research(workspace, snapshot_id, SMALL)
    second = run_research(workspace, snapshot_id, SMALL)

    ledger = ExperimentLedger(workspace / "experiment_ledger.jsonl")
    for result in (first, second):
        history = [event.state for event in ledger.get(result.attempt_id).events]
        assert history == [ExperimentState.RUNNING, ExperimentState.COMPLETED]
    assert first.attempt_id != second.attempt_id
    assert first.metrics_sha256 == second.metrics_sha256

    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    holdout_start = pd.Timestamp(report["chronology"]["final_holdout"]["start"])
    assert report["chronology"]["final_holdout"]["state"] == "locked"

    predictions = pd.read_csv(first.predictions_path)
    assert set(predictions["predictor"]) == {
        "quorum.momentum",
        "quorum.trend",
        "quorum.mean_reversion",
        "ensemble",
    }
    assert (pd.to_datetime(predictions["event_at"]) < holdout_start).all()
    assert (pd.to_datetime(predictions["label_end_at"]) < holdout_start).all()
    assert (
        pd.to_datetime(predictions["label_start_at"])
        > pd.to_datetime(predictions["decision_at"])
    ).all()

    test = report["evaluation"]["roles"]["test"]["predictors"]
    assert all(metrics["coverage"] == 1.0 for metrics in test.values())
    # Every registered attempt counts toward the trial family, in order.
    later = json.loads(second.report_path.read_text(encoding="utf-8"))
    assert report["trial_family"]["attempt_count"] == 1
    assert later["trial_family"]["attempt_count"] == 2


def test_future_bars_cannot_change_earlier_predictions(tmp_path: Path) -> None:
    cutoff = 150
    shocked = {name: frame.copy() for name, frame in FRAMES.items()}
    for frame in shocked.values():
        frame.iloc[cutoff + 1 :, :4] *= 3.0

    results = []
    for name, frames in (("base", FRAMES), ("shocked", shocked)):
        workspace = tmp_path / name
        snapshot_id = _ingest(DatasetSnapshotStore(workspace / "snapshots"), frames)
        results.append(run_research(workspace, snapshot_id, SMALL))

    base, changed = (pd.read_csv(r.predictions_path) for r in results)
    # The cutoff bar's own close (16:00 New York) is the last unshocked value.
    last_visible = DATES[cutoff].replace(hour=16).tz_localize("America/New_York")
    keys = ["predictor", "asset", "event_at"]

    def early(frame: pd.DataFrame) -> pd.DataFrame:
        at = pd.to_datetime(frame["event_at"], utc=True)
        return frame[at <= last_visible].set_index(keys)["score"].sort_index()

    assert len(early(base)) > 0
    pd.testing.assert_series_equal(early(base), early(changed))


def test_mismatched_calendars_fail_closed_and_record_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    gapped = dict(FRAMES)
    gapped["BBB.US"] = FRAMES["BBB.US"].drop(FRAMES["BBB.US"].index[100])
    snapshot_id = _ingest(DatasetSnapshotStore(workspace / "snapshots"), gapped)

    with pytest.raises(ValueError, match="calendar"):
        run_research(workspace, snapshot_id, SMALL)

    ledger = ExperimentLedger(workspace / "experiment_ledger.jsonl")
    (attempt,) = ledger.attempts()
    assert ledger.get(attempt.attempt_id).state is ExperimentState.FAILED


def test_cli_run_uses_default_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.quorum.research import main

    workspace = tmp_path / "workspace"
    frames = {
        name: _frame(seed, pd.bdate_range("2021-01-04", periods=700))
        for seed, name in enumerate(("AAA.US", "BBB.US"))
    }
    snapshot_id = _ingest(DatasetSnapshotStore(workspace / "snapshots"), frames)

    assert (
        main(["run", "--workspace", str(workspace), "--snapshot-id", snapshot_id]) == 0
    )
    printed = json.loads(capsys.readouterr().out)
    report = json.loads(Path(printed["report"]).read_text(encoding="utf-8"))
    assert report["config"]["horizon_bars"] == 5
