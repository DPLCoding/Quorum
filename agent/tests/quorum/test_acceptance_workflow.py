"""End-to-end controls for the offline Quorum V0 acceptance workflow."""

from __future__ import annotations

import ast
import inspect
import json
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.quorum.acceptance as acceptance
from src.quantlib.crossvalidation import Split
from src.quorum import ExpertResult, StaticEnsembleResult
from src.quorum.experiments import ExperimentLedger, ExperimentState
from src.quorum.risk import RiskPolicyResult
from src.quorum.validation import BoundaryLeakageError, require_clean_boundary


def _strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-strict JSON token: {token}")
        ),
    )
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def completed_acceptance(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("quorum-acceptance-primary") / "run"
    result = acceptance.run_v0_acceptance(output_dir)
    return result, _strict_json(result.report_path)


def test_end_to_end_acceptance_proves_oof_execution_and_artifacts(
    completed_acceptance,
) -> None:
    result, report = completed_acceptance

    assert result.status == report["status"] == "PASS"
    assert result.fold_count == report["chronology"]["fold_count"] == 5
    assert result.validation_oof_slots == report["oof"]["validation_slot_count"] == 25
    assert result.test_oof_slots == report["oof"]["test_slot_count"] == 50
    assert (
        result.scientific_fingerprint
        == report["reproducibility"]["scientific_fingerprint"]
    )
    assert result.report_path.is_file()
    assert result.markdown_path.is_file()

    history = report["experiment"]["lifecycle_history"]
    assert [row["state"] for row in history] == ["registered", "running", "completed"]
    assert report["experiment"]["lifecycle_state"] == "completed"
    assert report["experiment"]["registration_preceded_materialization"] is True
    assert report["experiment"]["lifecycle_timestamps_are_fixture_metadata"] is True

    chronology = report["chronology"]
    assert chronology["final_holdout_state"] == "locked"
    assert chronology["final_holdout_accessed"] is False
    assert chronology["final_holdout_positions"] == [140, 159]
    assert all(audit["clean"] for audit in chronology["leakage_audits"])

    assignments = report["oof"]["assignments"]
    positions = [assignment["sample_position"] for assignment in assignments]
    assert len(positions) == len(set(positions)) == 75
    assert all(position < 140 for position in positions)
    assert sum(row["role"] == "validation" for row in assignments) == 25
    assert sum(row["role"] == "test" for row in assignments) == 50

    audits = report["oof"]["prepared_input_audit"]
    assert len(audits) == 75
    assert all(row["prepared_end_position"] == row["sample_position"] for row in audits)
    assert all(
        row["prepared_row_count"] == row["sample_position"] + 1 for row in audits
    )
    assert all(row["prepared_end_position"] < 140 for row in audits)
    assert all(len(row["expert_ids"]) == 3 for row in audits)

    counts = report["oof"]["expert_prediction_counts"]
    assert set(counts) == {
        "quorum.momentum",
        "quorum.trend",
        "quorum.mean_reversion",
    }
    assert all(
        count == {"validation": 25, "test": 50, "total": 75}
        for count in counts.values()
    )
    assert report["oof"]["prediction_count"] == 225
    assert report["oof"]["duplicate_prediction_count"] == 0
    assert report["oof"]["missing_test_prediction_count"] == 0
    ExpertResult.from_dict(report["oof"]["expert_result"])

    expected_artifacts = {
        "config.json",
        "artifacts/quorum_protocol.json",
        "artifacts/equity.csv",
        "artifacts/positions.csv",
        "artifacts/target_positions.csv",
        "artifacts/trades.csv",
        "artifacts/fills.jsonl",
        "artifacts/metrics.csv",
        "artifacts/validation.json",
    }
    total_fills = 0
    for fold in report["folds"]:
        expert_result = ExpertResult.from_dict(fold["expert_result"])
        ensemble_result = StaticEnsembleResult.from_dict(fold["ensemble_result"])
        risk_result = RiskPolicyResult.from_dict(fold["risk_result"])
        assert len(expert_result.predictions) == 30
        assert len(ensemble_result.decisions) == 10
        assert len(risk_result.rebalances) == 10
        assert {rebalance.stream_key for rebalance in risk_result.rebalances} == {
            tuple(fold["risk_stream_key"])
        }
        assert (
            acceptance._sha256_json(
                {
                    "expert_result": fold["expert_result"],
                    "ensemble_result": fold["ensemble_result"],
                    "risk_result": fold["risk_result"],
                    "vibe_targets": fold["vibe_targets"],
                }
            )
            == fold["scientific_inputs_hash"]
        )
        assert fold["execution_window_positions"] == [
            fold["test_positions"][0],
            fold["test_positions"][-1] + 1,
        ]
        assert fold["execution_window_positions"][1] < 140
        assert fold["cost_effect"] > 0.0

        for row in fold["theoretical_target_causality"]:
            assert row["causal"] is True
            assert row["execution_position"] == row["event_position"] + 1
            assert (
                pd.Timestamp(row["available_at"])
                <= pd.Timestamp(row["decision_at"])
                < pd.Timestamp(row["execution_at"])
            )

        cost_on = fold["runs"]["cost-on"]
        cost_off = fold["runs"]["cost-off"]
        assert cost_on["scientific_inputs_hash"] == cost_off["scientific_inputs_hash"]
        assert cost_on["observed_slippage"] is True
        assert cost_off["observed_slippage"] is False
        assert cost_on["terminal_equity"] < cost_off["terminal_equity"]
        assert (
            cost_on["stable_artifact_hashes"]["artifacts/target_positions.csv"]
            == cost_off["stable_artifact_hashes"]["artifacts/target_positions.csv"]
        )

        for mode, run in fold["runs"].items():
            total_fills += run["fill_count"]
            actual_audit = run["actual_execution_audit"]
            assert actual_audit["passed"] is True
            assert actual_audit["authorized_execution_count"] == 10
            assert actual_audit["decision_fill_count"] > 0
            assert actual_audit["signal_fill_count"] > 0
            assert actual_audit["unauthorized_decision_fill_count"] == 0
            assert actual_audit["terminal_liquidation_fill_count"] == 1
            assert actual_audit["terminal_liquidation_exempt"] is True
            assert set(actual_audit["decision_fill_timestamps"]) <= set(
                actual_audit["authorized_execution_at"]
            )
            assert expected_artifacts <= set(run["stable_artifact_hashes"])
            assert (
                run["stable_artifact_hashes"]["artifacts/positions.csv"]
                != run["stable_artifact_hashes"]["artifacts/target_positions.csv"]
            )
            artifacts = {
                artifact["path"] for artifact in run["normalized_run_card"]["artifacts"]
            }
            assert "artifacts/quorum_protocol.json" in artifacts
            run_dir = (
                result.output_dir / "folds" / f"fold-{fold['fold_index']:03d}" / mode
            )
            _strict_json(run_dir / "run_card.json")
            _strict_json(run_dir / "artifacts" / "validation.json")
            sidecar = _strict_json(run_dir / "artifacts" / "quorum_protocol.json")
            assert sidecar["risk_stream_key"] == fold["risk_stream_key"]
            assert sidecar["oof_role"] == "test"
            assert sidecar["final_holdout_state"] == "locked"
            assert sidecar["final_holdout_accessed"] is False
            assert sidecar["cost_mode"] == mode
            assert (
                sidecar["theoretical_target_causality"]
                == fold["theoretical_target_causality"]
            )
            assert all(expert["fit_required"] is False for expert in sidecar["experts"])
            pd.read_csv(run_dir / "artifacts" / "equity.csv")
            pd.read_csv(run_dir / "artifacts" / "positions.csv")
            pd.read_csv(run_dir / "artifacts" / "target_positions.csv")
            pd.read_csv(run_dir / "artifacts" / "trades.csv")
            pd.read_csv(run_dir / "artifacts" / "metrics.csv")
            assert (run_dir / "artifacts" / "fills.jsonl").is_file()
            assert (run_dir / "run_card.md").is_file()

    assert total_fills > 0
    assert report["cost_comparison"]["identical_scientific_inputs"] is True
    assert report["cost_comparison"]["strict_cost_effect_fold_count"] == 5
    assert report["cost_comparison"]["total_terminal_equity_cost_effect"] > 0.0
    assert report["checks"] == {
        "actual_fill_execution_authorized": True,
        "causal_prepared_data_prefixes": True,
        "chronological_boundaries_clean": True,
        "costs_applied": True,
        "execution_after_availability": True,
        "final_holdout_accessed": False,
        "final_holdout_locked": True,
        "independent_fold_execution": True,
        "network_required": False,
        "requested_and_realized_exposure_separate": True,
        "test_predictions_are_oof": True,
        "theoretical_target_causality": True,
    }


def test_existing_vibe_run_consumer_reads_fold_artifacts(
    completed_acceptance,
) -> None:
    import api_server

    result, report = completed_acceptance
    fold = report["folds"][0]
    run_dir = result.output_dir / "folds" / "fold-000" / "cost-on"

    assert not (run_dir / "state.json").exists()
    response = api_server._build_response_from_run_dir(run_dir, elapsed=0.0)

    assert response.status == "unknown"
    assert response.metrics is not None
    assert response.metrics.final_value == pytest.approx(
        fold["runs"]["cost-on"]["terminal_equity"],
        abs=1e-10,
    )
    assert response.equity_curve
    assert response.trade_log
    assert response.artifacts_positions_csv
    assert response.artifacts_target_positions_csv
    assert response.validation == fold["runs"]["cost-on"]["validation"]
    assert response.run_card == _strict_json(run_dir / "run_card.json")


def test_unauthorized_later_signal_fill_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_read_fills = acceptance._read_fills
    moved_signal_fill = False

    def read_fills_with_late_signal(path: Path):
        nonlocal moved_signal_fill
        fills = original_read_fills(path)
        if moved_signal_fill:
            return fills
        for fill in fills:
            if fill["reason"] == "signal":
                fill["timestamp"] = (
                    datetime.fromisoformat(fill["timestamp"]) + timedelta(days=30)
                ).isoformat()
                moved_signal_fill = True
                break
        return fills

    monkeypatch.setattr(acceptance, "_read_fills", read_fills_with_late_signal)
    output_dir = tmp_path / "unauthorized-fill"

    with pytest.raises(
        ValueError,
        match="actual signal fill occurred at an unauthorized causal execution bar",
    ):
        acceptance.run_v0_acceptance(output_dir)

    assert moved_signal_fill is True
    assert (
        ExperimentLedger(output_dir / "experiment_ledger.jsonl")
        .get(acceptance.ATTEMPT_ID)
        .state
        is ExperimentState.FAILED
    )


def test_repeated_runs_have_identical_scientific_evidence(
    completed_acceptance,
    tmp_path: Path,
) -> None:
    first_result, first_report = completed_acceptance
    second_result = acceptance.run_v0_acceptance(tmp_path / "second")
    second_report = _strict_json(second_result.report_path)

    assert second_result.scientific_fingerprint == first_result.scientific_fingerprint
    assert second_result.dataset_snapshot_id == first_result.dataset_snapshot_id
    assert (
        second_result.experiment_spec_fingerprint
        == first_result.experiment_spec_fingerprint
    )
    assert second_report == first_report
    assert (
        second_result.report_path.read_bytes() == first_result.report_path.read_bytes()
    )
    assert (second_result.output_dir / "experiment_ledger.jsonl").read_bytes() == (
        first_result.output_dir / "experiment_ledger.jsonl"
    ).read_bytes()


def test_registration_precedes_materialization_and_network_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    original_register = acceptance.ExperimentLedger.register
    original_materialize = acceptance.materialize_chronological_plan

    def register_spy(self, *args, **kwargs):
        attempt = original_register(self, *args, **kwargs)
        order.append(f"registered:{attempt.attempt_id}")
        return attempt

    def materialize_spy(attempt, *args, **kwargs):
        assert order == [f"registered:{attempt.attempt_id}"]
        order.append(f"materialized:{attempt.attempt_id}")
        return original_materialize(attempt, *args, **kwargs)

    def blocked_network(*args, **kwargs):
        raise AssertionError("Task 7 attempted outbound networking")

    monkeypatch.setattr(acceptance.ExperimentLedger, "register", register_spy)
    monkeypatch.setattr(acceptance, "materialize_chronological_plan", materialize_spy)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket, "getaddrinfo", blocked_network)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)

    result = acceptance.run_v0_acceptance(tmp_path / "network-blocked")

    assert result.status == "PASS"
    assert order == [
        f"registered:{acceptance.ATTEMPT_ID}",
        f"materialized:{acceptance.ATTEMPT_ID}",
    ]


def test_locked_holdout_poisoning_cannot_change_ordinary_oof_science(
    tmp_path: Path,
) -> None:
    frame = acceptance._synthetic_frame()
    snapshot_id = acceptance._dataset_snapshot_id(frame)
    protocol = acceptance._evaluation_protocol(frame)
    ensemble_config = acceptance._ensemble_config()
    risk_config = acceptance._risk_config()
    ledger = ExperimentLedger(tmp_path / "poison-ledger.jsonl")
    attempt = ledger.register(
        acceptance._experiment_spec(frame, snapshot_id),
        attempt_id=acceptance.ATTEMPT_ID,
        registered_at=acceptance.REGISTERED_AT,
    )
    plan = acceptance.materialize_chronological_plan(
        attempt,
        protocol,
        acceptance._bar_intervals(frame),
        acceptance._ordinary_label_ends(frame),
    )
    ordinary = acceptance._generate_oof_evidence(
        frame,
        plan,
        ensemble_config,
        risk_config,
    )

    poisoned = frame.copy(deep=True)
    price_columns = ["open", "high", "low", "close"]
    poisoned.loc[poisoned.index[-acceptance.HOLDOUT_BARS :], price_columns] *= 50.0
    poisoned.loc[poisoned.index[-acceptance.HOLDOUT_BARS :], "volume"] *= 100
    poisoned_evidence = acceptance._generate_oof_evidence(
        poisoned,
        plan,
        ensemble_config,
        risk_config,
    )

    assert acceptance._dataset_snapshot_id(poisoned) != snapshot_id
    assert poisoned_evidence.expert_result.to_json() == ordinary.expert_result.to_json()
    assert [fold.ensemble_result.to_json() for fold in poisoned_evidence.folds] == [
        fold.ensemble_result.to_json() for fold in ordinary.folds
    ]
    assert [fold.risk_result.to_json() for fold in poisoned_evidence.folds] == [
        fold.risk_result.to_json() for fold in ordinary.folds
    ]
    assert acceptance._sha256_json(
        acceptance._ordinary_oof_payload(plan, poisoned_evidence)
    ) == acceptance._sha256_json(acceptance._ordinary_oof_payload(plan, ordinary))


def test_dirty_boundary_fails_before_predictions_or_engine_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downstream_calls: list[str] = []

    def dirty_materializer(*args, **kwargs):
        del args, kwargs
        dirty = Split(
            train=np.asarray([0, 1], dtype=int),
            test=np.asarray([2], dtype=int),
            purged=0,
            embargoed=0,
            test_bounds=(2, 2),
        )
        return require_clean_boundary(
            dirty,
            (0, 2, 2),
            n_samples=3,
            embargo_size=0,
        )

    def forbidden_predictions(*args, **kwargs):
        del args, kwargs
        downstream_calls.append("predictions")
        raise AssertionError("predictions ran after a dirty boundary")

    def forbidden_engine(*args, **kwargs):
        del args, kwargs
        downstream_calls.append("engine")
        raise AssertionError("engine ran after a dirty boundary")

    monkeypatch.setattr(
        acceptance, "materialize_chronological_plan", dirty_materializer
    )
    monkeypatch.setattr(acceptance, "_generate_oof_evidence", forbidden_predictions)
    monkeypatch.setattr(acceptance, "_execute_fold", forbidden_engine)
    output_dir = tmp_path / "dirty"

    with pytest.raises(BoundaryLeakageError, match="dirty chronological boundary"):
        acceptance.run_v0_acceptance(output_dir)

    assert downstream_calls == []
    assert (
        ExperimentLedger(output_dir / "experiment_ledger.jsonl")
        .get(acceptance.ATTEMPT_ID)
        .state
        is ExperimentState.FAILED
    )


def test_output_protection_and_dependency_boundary(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "existing.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        acceptance.run_v0_acceptance(occupied)
    assert marker.read_text(encoding="utf-8") == "do not overwrite"
    assert set(occupied.iterdir()) == {marker}

    tree = ast.parse(inspect.getsource(acceptance))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    banned = (
        "src.agent",
        "src.live",
        "src.trading",
        "src.api",
        "langchain",
        "openai",
        "anthropic",
        "google.generativeai",
        "ollama",
        "requests",
        "httpx",
        "fastapi",
    )
    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(root + ".") for root in banned)
    }


def test_module_cli_completes_offline(tmp_path: Path) -> None:
    agent_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "cli-run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.quorum.acceptance",
            "--output-dir",
            str(output_dir),
        ],
        cwd=agent_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "PASS"
    assert stdout["scientific_fingerprint"].startswith("sha256:")
    report = _strict_json(output_dir / "quorum_acceptance.json")
    assert report["status"] == "PASS"
    assert Path(stdout["report"]) == output_dir / "quorum_acceptance.json"
