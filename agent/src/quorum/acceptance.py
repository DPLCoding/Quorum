"""Fully offline, deterministic Quorum V0 acceptance workflow.

The constants in this module define an acceptance fixture, not product defaults.
The workflow deliberately uses the frozen Task 1--6 APIs and Vibe's existing
``GlobalEquityEngine``.  It does not fetch data, fit a model, contact a provider,
or join independent out-of-fold risk streams into one portfolio history.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engines.global_equity import GlobalEquityEngine
from src.quorum.adapters import VibeSignalAdapter, validate_v0_vibe_config
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
from src.quorum.ensemble import StaticEnsemble, StaticEnsembleResult
from src.quorum.experiments import (
    ExperimentAttempt,
    ExperimentLedger,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentState,
    ExternalRecordRefs,
)
from src.quorum.experts import MeanReversionExpert, MomentumExpert, TrendExpert
from src.quorum.risk import FixedRiskPolicy, RiskPolicyConfig, RiskPolicyResult
from src.quorum.validation import (
    ChronologicalEvaluationPlan,
    OOFRole,
    materialize_chronological_plan,
)

UTC = timezone.utc
ACCEPTANCE_FIXTURE_VERSION = "quorum-v0-acceptance-v1"
ACCEPTANCE_SCHEMA_VERSION = 1
ASSET = "QRM.US"
BAR_COUNT = 160
BAR_DURATION = timedelta(hours=1)
HOLDOUT_BARS = 20
HORIZON_BARS = 1
DECISION_DELAY = timedelta(minutes=1)
INITIAL_CASH = 100_000.0
BARS_PER_YEAR = 252 * 6
RANDOM_SEED = 1729
PROTOCOL_ID = "quorum:acceptance:evaluation:v1"
ENSEMBLE_CONFIG_ID = "quorum:acceptance:ensemble:equal-v1"
COST_MODEL_ID = "quorum:acceptance:cost:us-slippage-20bp-v1"
ATTEMPT_ID = "exp_00000000000000000000000000000007"
REGISTERED_AT = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
RUNNING_AT = REGISTERED_AT + timedelta(minutes=1)
COMPLETED_AT = REGISTERED_AT + timedelta(minutes=2)
FAILED_AT = REGISTERED_AT + timedelta(minutes=3)
RUNNING_EVENT_ID = "evt_00000000000000000000000000000001"
COMPLETED_EVENT_ID = "evt_00000000000000000000000000000002"
FAILED_EVENT_ID = "evt_00000000000000000000000000000003"

_COST_MODES: tuple[tuple[str, float], ...] = (
    ("cost-on", 0.002),
    ("cost-off", 0.0),
)
_DECISION_FILL_REASONS = frozenset({"signal", "target_rebalance"})
_TERMINAL_FILL_REASON = "end_of_backtest"
_REQUIRED_ARTIFACTS = frozenset(
    {
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
)


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """Small immutable handle to one completed acceptance run."""

    status: str
    output_dir: Path
    report_path: Path
    markdown_path: Path
    scientific_fingerprint: str
    ordinary_oof_fingerprint: str
    dataset_snapshot_id: str
    experiment_spec_fingerprint: str
    fold_count: int
    validation_oof_slots: int
    test_oof_slots: int


@dataclass(frozen=True, slots=True)
class _FoldScience:
    fold_index: int
    split_id: str
    test_positions: tuple[int, ...]
    expert_result: ExpertResult
    ensemble_result: StaticEnsembleResult
    risk_result: RiskPolicyResult


@dataclass(frozen=True, slots=True)
class _OOFEvidence:
    expert_result: ExpertResult
    prepared_input_audit: tuple[Mapping[str, Any], ...]
    folds: tuple[_FoldScience, ...]


@dataclass(frozen=True, slots=True)
class _FrozenFrameLoader:
    """Private in-memory loader used only by the acceptance fixture."""

    frame: pd.DataFrame
    asset: str = ASSET
    name: str = "quorum-synthetic"

    def fetch(
        self,
        codes: Sequence[str],
        start_date: str,
        end_date: str,
        fields: Sequence[str] | None = None,
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        del start_date, end_date, fields, interval
        if tuple(codes) != (self.asset,):
            raise ValueError("acceptance loader permits only its frozen asset")
        return {self.asset: self.frame.copy(deep=True)}


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    separators = (",", ":") if indent is None else None
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=separators,
        indent=indent,
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _prepare_output_directory(output_dir: Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        if not path.is_dir():
            raise ValueError("acceptance output path must be a directory")
        if any(path.iterdir()):
            raise ValueError("acceptance output directory must be empty")
    else:
        path.mkdir(parents=True)
    return path


def _synthetic_frame() -> pd.DataFrame:
    """Return the frozen, mathematical, provider-free acceptance dataset."""
    index = pd.date_range(
        "2026-01-05T14:00:00Z",
        periods=BAR_COUNT,
        freq="h",
    )
    closes: list[float] = []
    price = 100.0
    regime_drifts = (0.45, -0.38, 0.28, -0.50)
    for position in range(BAR_COUNT):
        drift = regime_drifts[(position // 20) % len(regime_drifts)]
        oscillation = 0.70 * math.sin(position * 0.80)
        slow_wave = 0.25 * math.cos(position * 0.23)
        price = max(25.0, price + drift + oscillation + slow_wave)
        closes.append(round(price, 8))

    opens = [
        round(
            (closes[position - 1] if position else closes[0] - 0.15)
            + 0.12 * math.sin(position * 0.37),
            8,
        )
        for position in range(BAR_COUNT)
    ]
    highs = [
        round(max(open_price, close_price) + 0.35 + 0.02 * (position % 5), 8)
        for position, (open_price, close_price) in enumerate(zip(opens, closes))
    ]
    lows = [
        round(min(open_price, close_price) - 0.35 - 0.02 * (position % 3), 8)
        for position, (open_price, close_price) in enumerate(zip(opens, closes))
    ]
    volumes = [100_000 + 1_000 * (position % 17) for position in range(BAR_COUNT)]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def _dataset_snapshot_payload(frame: pd.DataFrame) -> dict[str, Any]:
    required = ("open", "high", "low", "close", "volume")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("acceptance dataset requires a timezone-aware DatetimeIndex")
    if tuple(frame.columns) != required:
        raise ValueError("acceptance dataset has unexpected columns")
    rows = []
    for timestamp, row in frame.iterrows():
        rows.append(
            {
                "timestamp_ns": int(timestamp.value),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )
    return {
        "fixture_generator_version": ACCEPTANCE_FIXTURE_VERSION,
        "asset": ASSET,
        "rows": rows,
    }


def _dataset_snapshot_id(frame: pd.DataFrame) -> str:
    return _sha256_json(_dataset_snapshot_payload(frame))


def _bar_intervals(frame: pd.DataFrame) -> tuple[TimeInterval, ...]:
    return tuple(
        TimeInterval(
            start=(timestamp - BAR_DURATION).to_pydatetime(),
            end=timestamp.to_pydatetime(),
        )
        for timestamp in frame.index
    )


def _evaluation_protocol(frame: pd.DataFrame) -> EvaluationProtocol:
    bars = _bar_intervals(frame)
    holdout_start = bars[-HOLDOUT_BARS].start
    holdout_end = bars[-1].end
    return EvaluationProtocol(
        protocol_id=PROTOCOL_ID,
        mode=EvaluationMode.EXPANDING,
        minimum_train_bars=60,
        train_window_bars=None,
        validation_bars=5,
        test_bars=10,
        step_bars=15,
        purge_bars=1,
        embargo_bars=1,
        final_holdout=FinalHoldout(
            start=holdout_start,
            end=holdout_end,
            state=FinalHoldoutState.LOCKED,
        ),
        random_seed=RANDOM_SEED,
    )


def _ensemble_config() -> StaticEnsembleConfig:
    return StaticEnsembleConfig(
        experts=(
            ExpertWeight(
                MomentumExpert.expert_id,
                MomentumExpert.expert_version,
                1.0 / 3.0,
            ),
            ExpertWeight(
                TrendExpert.expert_id,
                TrendExpert.expert_version,
                1.0 / 3.0,
            ),
            ExpertWeight(
                MeanReversionExpert.expert_id,
                MeanReversionExpert.expert_version,
                1.0 / 3.0,
            ),
        ),
        sell_threshold=-0.20,
        buy_threshold=0.20,
    )


def _risk_config() -> RiskPolicyConfig:
    return RiskPolicyConfig(
        max_gross_exposure=0.60,
        max_abs_weight_per_asset=0.40,
        max_turnover=0.50,
    )


def _experiment_spec(frame: pd.DataFrame, snapshot_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        evaluation_protocol_id=PROTOCOL_ID,
        expert_config_ids=(
            MomentumExpert.config_id,
            TrendExpert.config_id,
            MeanReversionExpert.config_id,
        ),
        ensemble_config_id=ENSEMBLE_CONFIG_ID,
        target_definition_id="quorum:acceptance:target:one-bar-forward-v1",
        horizon_bars=HORIZON_BARS,
        data_snapshot_id=snapshot_id,
        data_cutoff_at=frame.index[-1].to_pydatetime(),
        universe_id="quorum:acceptance:universe:qrm-us-v1",
        cost_model_id=COST_MODEL_ID,
        code_id="quorum:acceptance:workflow:v1",
        config_id="quorum:acceptance:fixture:v1",
        random_seed=RANDOM_SEED,
        trial_family_id="quorum:acceptance:v0",
    )


def _ordinary_label_ends(frame: pd.DataFrame) -> tuple[int, ...]:
    ordinary_stop = len(frame) - HOLDOUT_BARS
    return tuple(position + HORIZON_BARS for position in range(ordinary_stop))


def _prepared_data(frame: pd.DataFrame, sample_position: int) -> dict[str, object]:
    prefix = frame.iloc[: sample_position + 1]
    instants = tuple(timestamp.to_pydatetime() for timestamp in prefix.index)
    return {
        "asset": ASSET,
        "close": tuple(float(value) for value in prefix["close"]),
        "event_at": instants,
        "available_at": instants,
    }


def _prediction_context(
    frame: pd.DataFrame,
    slot_position: int,
    split_id: str,
    attempt_id: str,
) -> PredictionContext:
    event_at = frame.index[slot_position].to_pydatetime()
    return PredictionContext(
        event_at=event_at,
        available_at=event_at,
        decision_at=event_at + DECISION_DELAY,
        horizon_bars=HORIZON_BARS,
        experiment_id=attempt_id,
        split_id=split_id,
    )


def _generate_oof_evidence(
    frame: pd.DataFrame,
    plan: ChronologicalEvaluationPlan,
    ensemble_config: StaticEnsembleConfig,
    risk_config: RiskPolicyConfig,
) -> _OOFEvidence:
    """Run the actual experts on prefixes ending exactly at authorized slots."""
    experts = (MomentumExpert(), TrendExpert(), MeanReversionExpert())
    predictions: list[ExpertPrediction] = []
    prepared_audit: list[Mapping[str, Any]] = []
    role_by_split_and_position: dict[tuple[str, int], OOFRole] = {}

    for slot in plan.oof_slots:
        role_by_split_and_position[(slot.split_id, slot.sample_position)] = slot.role
        prepared = _prepared_data(frame, slot.sample_position)
        context = _prediction_context(
            frame,
            slot.sample_position,
            slot.split_id,
            plan.attempt_id,
        )
        emitted_ids: list[str] = []
        for expert in experts:
            result = expert.predict(prepared, context)
            if len(result.predictions) != 1:
                raise ValueError(
                    "acceptance OOF slot did not receive exactly one prediction "
                    f"from {expert.expert_id!r}"
                )
            prediction = result.predictions[0]
            if prediction.expert_id != expert.expert_id:
                raise ValueError("expert emitted an unexpected identity")
            predictions.append(prediction)
            emitted_ids.append(expert.expert_id)
        prepared_audit.append(
            {
                "sample_position": slot.sample_position,
                "split_id": slot.split_id,
                "fold_index": slot.fold_index,
                "role": slot.role.value,
                "prepared_start_position": 0,
                "prepared_end_position": slot.sample_position,
                "prepared_row_count": slot.sample_position + 1,
                "expert_ids": sorted(emitted_ids),
            }
        )

    all_result = ExpertResult(tuple(predictions))
    expected_count = len(plan.oof_slots) * len(experts)
    if len(all_result.predictions) != expected_count:
        raise ValueError("acceptance expert prediction count is incomplete")

    timestamp_to_position = {
        int(timestamp.value): position for position, timestamp in enumerate(frame.index)
    }
    folds: list[_FoldScience] = []
    for fold in plan.folds:
        test_predictions = tuple(
            prediction
            for prediction in all_result.predictions
            if prediction.split_id == fold.manifest.split_id
            and role_by_split_and_position[
                (
                    fold.manifest.split_id,
                    timestamp_to_position[int(pd.Timestamp(prediction.event_at).value)],
                )
            ]
            is OOFRole.TEST
        )
        expert_result = ExpertResult(test_predictions)
        expected_fold_predictions = len(fold.test_positions) * len(experts)
        if len(expert_result.predictions) != expected_fold_predictions:
            raise ValueError("acceptance fold has incomplete test expert evidence")

        ensemble_result = StaticEnsemble(ensemble_config).combine(expert_result)
        if len(ensemble_result.decisions) != len(fold.test_positions):
            raise ValueError("acceptance fold ensemble output is incomplete")
        risk_result = FixedRiskPolicy(risk_config).apply(ensemble_result)
        if len(risk_result.rebalances) != len(fold.test_positions):
            raise ValueError("acceptance fold risk output is incomplete")
        if any(not rebalance.executable for rebalance in risk_result.rebalances):
            raise ValueError("acceptance fold contains an incomplete risk group")
        streams = {rebalance.stream_key for rebalance in risk_result.rebalances}
        expected_stream = {(plan.attempt_id, fold.manifest.split_id, HORIZON_BARS)}
        if streams != expected_stream:
            raise ValueError("acceptance fold risk stream is not isolated")

        folds.append(
            _FoldScience(
                fold_index=fold.manifest.fold_index,
                split_id=fold.manifest.split_id,
                test_positions=fold.test_positions,
                expert_result=expert_result,
                ensemble_result=ensemble_result,
                risk_result=risk_result,
            )
        )

    return _OOFEvidence(
        expert_result=all_result,
        prepared_input_audit=tuple(prepared_audit),
        folds=tuple(folds),
    )


def _serialize_signal(signal: pd.Series) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_ns": int(timestamp.value),
            "target_weight": float(value),
        }
        for timestamp, value in signal.items()
    ]


def _theoretical_target_causality_rows(
    frame: pd.DataFrame,
    fold_science: _FoldScience,
) -> list[dict[str, Any]]:
    position_by_instant = {
        int(timestamp.value): position for position, timestamp in enumerate(frame.index)
    }
    rows: list[dict[str, Any]] = []
    for rebalance in fold_science.risk_result.rebalances:
        for target in rebalance.targets:
            event_ns = int(pd.Timestamp(target.event_at).value)
            event_position = position_by_instant[event_ns]
            execution_position = event_position + 1
            if execution_position >= len(frame):
                raise ValueError("acceptance target has no next execution bar")
            execution_at = frame.index[execution_position].to_pydatetime()
            if not target.available_at <= target.decision_at < execution_at:
                raise ValueError(
                    "acceptance target violates information-time causality"
                )
            rows.append(
                {
                    "asset": target.asset,
                    "event_position": event_position,
                    "execution_position": execution_position,
                    "event_at": target.event_at.isoformat(),
                    "available_at": target.available_at.isoformat(),
                    "decision_at": target.decision_at.isoformat(),
                    "execution_at": execution_at.isoformat(),
                    "final_target_weight": target.final_target_weight,
                    "causal": True,
                }
            )
    return rows


def _engine_config(
    frame: pd.DataFrame, cost_mode: str, slippage: float
) -> dict[str, Any]:
    return {
        "codes": [ASSET],
        "start_date": frame.index[0].isoformat(),
        "end_date": frame.index[-1].isoformat(),
        "interval": "1h",
        "engine": "global_equity",
        "source": "quorum-synthetic",
        "initial_cash": INITIAL_CASH,
        "position_adjustment": "rebalance",
        "optimizer": None,
        "constraints": [],
        "rebalance_mask": None,
        "rebalance_tolerance": 0.0,
        "leverage": 1.0,
        "slippage_us": slippage,
        "acceptance_cost_mode": cost_mode,
        "validation": {
            "bootstrap": {
                "n_bootstrap": 64,
                "confidence": 0.95,
                "seed": RANDOM_SEED,
            }
        },
    }


def _protocol_sidecar(
    *,
    snapshot_id: str,
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    fold_science: _FoldScience,
    manifest: Mapping[str, Any],
    ensemble_config: StaticEnsembleConfig,
    risk_config: RiskPolicyConfig,
    theoretical_target_causality: list[dict[str, Any]],
    cost_mode: str,
    scientific_inputs_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "artifact": "quorum_v0_acceptance_protocol",
        "acceptance_fixture_version": ACCEPTANCE_FIXTURE_VERSION,
        "dataset_snapshot_id": snapshot_id,
        "experiment_attempt_id": attempt.attempt_id,
        "experiment_spec_fingerprint": attempt.spec_fingerprint,
        "evaluation_protocol": protocol.to_dict(),
        "split_manifest": dict(manifest),
        "fold_index": fold_science.fold_index,
        "oof_role": OOFRole.TEST.value,
        "test_sample_positions": list(fold_science.test_positions),
        "experts": [
            {
                "expert_id": expert.expert_id,
                "expert_version": expert.expert_version,
                "config_id": expert.config_id,
                "fit_required": False,
            }
            for expert in (MomentumExpert, TrendExpert, MeanReversionExpert)
        ],
        "ensemble_config_id": ENSEMBLE_CONFIG_ID,
        "ensemble_config": ensemble_config.to_dict(),
        "risk_config": risk_config.to_dict(),
        "risk_stream_key": [
            attempt.attempt_id,
            fold_science.split_id,
            HORIZON_BARS,
        ],
        "theoretical_target_causality": theoretical_target_causality,
        "cost_mode": cost_mode,
        "scientific_inputs_hash": scientific_inputs_hash,
        "final_holdout_state": protocol.final_holdout.state.value,
        "final_holdout_accessed": protocol.final_holdout.accessed_at is not None,
    }


def _normalized_run_card(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in card.items()
        if key not in {"generated_at", "run_dir"}
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_fills(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("fill artifact contains a non-object row")
            rows.append(value)
    return rows


def _audit_actual_fill_execution(
    fills: Sequence[Mapping[str, Any]],
    theoretical_target_causality: Sequence[Mapping[str, Any]],
    cost_mode: str,
) -> dict[str, Any]:
    """Fail closed unless every decision-driven fill uses an authorized bar.

    ``end_of_backtest`` is deliberately excluded from Quorum decision-fill
    causality: it is the engine's terminal accounting liquidation.  The V0
    rebalance engine can use both ``signal`` (open/flip) and
    ``target_rebalance`` (delta adjustment), so both reasons are audited.
    """
    authorized_by_ns: dict[int, str] = {}
    for row in theoretical_target_causality:
        if row.get("causal") is not True:
            raise ValueError("theoretical target causality is not established")
        timestamp = pd.Timestamp(row["execution_at"])
        if timestamp.tzinfo is None:
            raise ValueError("authorized execution timestamp must be timezone-aware")
        authorized_by_ns[int(timestamp.value)] = timestamp.isoformat()
    if not authorized_by_ns:
        raise ValueError("actual fill audit has no authorized execution bars")

    reason_counts = {reason: 0 for reason in sorted(_DECISION_FILL_REASONS)}
    decision_fill_timestamps: list[str] = []
    decision_fill_timestamp_ns: list[int] = []
    terminal_liquidation_count = 0
    for fill in fills:
        reason = fill.get("reason")
        if reason == _TERMINAL_FILL_REASON:
            terminal_liquidation_count += 1
            continue
        if reason not in _DECISION_FILL_REASONS:
            raise ValueError(
                f"{cost_mode} contains unexpected non-terminal fill reason {reason!r}"
            )
        if fill.get("symbol") != ASSET:
            raise ValueError(f"{cost_mode} decision fill has an unexpected asset")
        timestamp = pd.Timestamp(fill.get("timestamp"))
        if timestamp.tzinfo is None:
            raise ValueError(
                f"{cost_mode} decision fill timestamp must be timezone-aware"
            )
        timestamp_ns = int(timestamp.value)
        if timestamp_ns not in authorized_by_ns:
            raise ValueError(
                f"{cost_mode} actual {reason} fill occurred at an unauthorized "
                f"causal execution bar: {timestamp.isoformat()}"
            )
        reason_counts[str(reason)] += 1
        decision_fill_timestamps.append(timestamp.isoformat())
        decision_fill_timestamp_ns.append(timestamp_ns)

    return {
        "passed": True,
        "authorized_execution_at": [
            authorized_by_ns[key] for key in sorted(authorized_by_ns)
        ],
        "authorized_execution_count": len(authorized_by_ns),
        "decision_fill_reasons": sorted(_DECISION_FILL_REASONS),
        "decision_fill_count": len(decision_fill_timestamps),
        "signal_fill_count": reason_counts["signal"],
        "target_rebalance_fill_count": reason_counts["target_rebalance"],
        "decision_fill_timestamps": decision_fill_timestamps,
        "decision_fill_timestamp_ns": decision_fill_timestamp_ns,
        "unauthorized_decision_fill_count": 0,
        "terminal_liquidation_fill_count": terminal_liquidation_count,
        "terminal_liquidation_exempt": True,
    }


def _stable_artifact_hashes(run_dir: Path, card: Mapping[str, Any]) -> dict[str, str]:
    artifacts = card.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("run card artifact list is missing")
    hashes: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("run card contains an invalid artifact entry")
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("run card artifact identity is invalid")
        path = run_dir / relative
        if not path.is_file() or _file_sha256(path) != digest:
            raise ValueError(f"run card artifact hash mismatch for {relative!r}")
        hashes[relative] = digest
    missing = sorted(_REQUIRED_ARTIFACTS - set(hashes))
    if missing:
        raise ValueError(f"acceptance engine run is missing artifacts: {missing}")
    return dict(sorted(hashes.items()))


def _validate_engine_artifacts(
    run_dir: Path,
    execution_frame: pd.DataFrame,
    expected_shifted_targets: pd.Series,
    cost_mode: str,
    theoretical_target_causality: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for name in ("run_card.json", "run_card.md"):
        if not (run_dir / name).is_file():
            raise ValueError(f"acceptance engine run is missing {name}")
    card = _read_json(run_dir / "run_card.json")
    if not isinstance(card, Mapping):
        raise ValueError("run_card.json must contain an object")
    hashes = _stable_artifact_hashes(run_dir, card)
    if "artifacts/quorum_protocol.json" not in hashes:
        raise ValueError("run card did not discover the Quorum protocol sidecar")
    if card.get("data_sources") != ["quorum-synthetic"]:
        raise ValueError("run card did not preserve the synthetic data-source identity")

    _read_json(run_dir / "artifacts" / "validation.json")
    _read_json(run_dir / "artifacts" / "quorum_protocol.json")
    equity = pd.read_csv(run_dir / "artifacts" / "equity.csv", index_col=0)
    positions = pd.read_csv(run_dir / "artifacts" / "positions.csv", index_col=0)
    targets = pd.read_csv(
        run_dir / "artifacts" / "target_positions.csv",
        index_col=0,
    )
    pd.read_csv(run_dir / "artifacts" / "trades.csv")
    pd.read_csv(run_dir / "artifacts" / "metrics.csv")
    fills = _read_fills(run_dir / "artifacts" / "fills.jsonl")
    actual_execution_audit = _audit_actual_fill_execution(
        fills,
        theoretical_target_causality,
        cost_mode,
    )
    if actual_execution_audit["signal_fill_count"] == 0:
        raise ValueError(f"{cost_mode} acceptance run produced no signal fill evidence")

    observed_targets = tuple(float(value) for value in targets[ASSET])
    expected_targets = tuple(float(value) for value in expected_shifted_targets)
    if len(observed_targets) != len(expected_targets) or any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
        for observed, expected in zip(observed_targets, expected_targets)
    ):
        raise ValueError(
            "BaseEngine target artifact does not match shifted Vibe targets"
        )
    if len(equity) != len(execution_frame) or len(positions) != len(execution_frame):
        raise ValueError("engine artifacts do not match the fold execution window")
    if hashes["artifacts/positions.csv"] == hashes["artifacts/target_positions.csv"]:
        raise ValueError("requested targets and realized positions were conflated")

    base_prices: dict[int, tuple[float, float]] = {
        int(timestamp.value): (float(row["open"]), float(row["close"]))
        for timestamp, row in execution_frame.iterrows()
    }
    observed_slippage = False
    configured_slippage = dict(_COST_MODES)[cost_mode]
    for fill in fills:
        timestamp = pd.Timestamp(fill["timestamp"])
        open_price, close_price = base_prices[int(timestamp.value)]
        base_price = close_price if fill["reason"] == "end_of_backtest" else open_price
        execution_price = float(fill["execution_price"])
        trade_direction = 1 if float(fill["signed_quantity"]) > 0.0 else -1
        expected_execution_price = base_price * (
            1.0 + trade_direction * configured_slippage
        )
        if not math.isclose(
            execution_price,
            expected_execution_price,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{cost_mode} fill does not match configured US slippage")
        if not math.isclose(execution_price, base_price, rel_tol=0.0, abs_tol=1e-10):
            observed_slippage = True

    return {
        "metrics": card.get("metrics", {}),
        "validation": card.get("validation", {}),
        "normalized_run_card": _normalized_run_card(card),
        "stable_artifact_hashes": hashes,
        "fills": fills,
        "actual_execution_audit": actual_execution_audit,
        "fill_timestamp_ns": [
            int(pd.Timestamp(fill["timestamp"]).value) for fill in fills
        ],
        "fill_count": len(fills),
        "observed_slippage": observed_slippage,
        "terminal_equity": float(equity["equity"].iloc[-1]),
        "requested_target_artifact": "artifacts/target_positions.csv",
        "realized_position_artifact": "artifacts/positions.csv",
    }


def _execute_fold(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    snapshot_id: str,
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    plan: ChronologicalEvaluationPlan,
    fold_science: _FoldScience,
    ensemble_config: StaticEnsembleConfig,
    risk_config: RiskPolicyConfig,
) -> dict[str, Any]:
    fold = plan.folds[fold_science.fold_index]
    first_test = fold_science.test_positions[0]
    last_test = fold_science.test_positions[-1]
    execution_frame = frame.iloc[first_test : last_test + 2].copy(deep=True)
    if len(execution_frame) != len(fold_science.test_positions) + 1:
        raise ValueError("fold execution window lacks the final next bar")
    if any(
        protocol.final_holdout.interval.overlaps(plan.bar_intervals[position])
        for position in range(first_test, last_test + 2)
    ):
        raise ValueError("fold execution window touched the locked final holdout")

    stream_key = (attempt.attempt_id, fold_science.split_id, HORIZON_BARS)
    adapter = VibeSignalAdapter(
        fold_science.risk_result,
        stream_key=stream_key,
    )
    signal = adapter.generate({ASSET: execution_frame})[ASSET]
    expected_shifted = signal.shift(1).fillna(0.0)
    serialized_signal = _serialize_signal(signal)
    theoretical_target_causality = _theoretical_target_causality_rows(
        frame,
        fold_science,
    )
    scientific_inputs = {
        "expert_result": fold_science.expert_result.to_dict(),
        "ensemble_result": fold_science.ensemble_result.to_dict(),
        "risk_result": fold_science.risk_result.to_dict(),
        "vibe_targets": serialized_signal,
    }
    scientific_inputs_hash = _sha256_json(scientific_inputs)

    modes: dict[str, Any] = {}
    for cost_mode, slippage in _COST_MODES:
        run_dir = (
            output_dir / "folds" / f"fold-{fold_science.fold_index:03d}" / cost_mode
        )
        run_dir.mkdir(parents=True)
        config = _engine_config(execution_frame, cost_mode, slippage)
        validate_v0_vibe_config(config)
        _write_json(run_dir / "config.json", config)
        sidecar = _protocol_sidecar(
            snapshot_id=snapshot_id,
            attempt=attempt,
            protocol=protocol,
            fold_science=fold_science,
            manifest=fold.manifest.to_dict(),
            ensemble_config=ensemble_config,
            risk_config=risk_config,
            theoretical_target_causality=theoretical_target_causality,
            cost_mode=cost_mode,
            scientific_inputs_hash=scientific_inputs_hash,
        )
        _write_json(run_dir / "artifacts" / "quorum_protocol.json", sidecar)

        engine = GlobalEquityEngine(config, market="us")
        with redirect_stdout(io.StringIO()):
            engine.run_backtest(
                config,
                _FrozenFrameLoader(execution_frame),
                adapter,
                run_dir,
                bars_per_year=BARS_PER_YEAR,
            )
        mode_evidence = _validate_engine_artifacts(
            run_dir,
            execution_frame,
            expected_shifted,
            cost_mode,
            theoretical_target_causality,
        )
        mode_evidence["config"] = config
        mode_evidence["scientific_inputs_hash"] = scientific_inputs_hash
        modes[cost_mode] = mode_evidence

    if (
        modes["cost-on"]["scientific_inputs_hash"]
        != modes["cost-off"]["scientific_inputs_hash"]
    ):
        raise ValueError("cost comparison changed Quorum scientific inputs")
    if (
        modes["cost-on"]["stable_artifact_hashes"]["artifacts/target_positions.csv"]
        != modes["cost-off"]["stable_artifact_hashes"]["artifacts/target_positions.csv"]
    ):
        raise ValueError("cost comparison changed Vibe target positions")
    if (
        modes["cost-on"]["terminal_equity"]
        > modes["cost-off"]["terminal_equity"] + 1e-8
    ):
        raise ValueError("cost-on terminal equity is spuriously better than cost-off")

    return {
        "fold_index": fold_science.fold_index,
        "split_id": fold_science.split_id,
        "test_positions": list(fold_science.test_positions),
        "execution_window_positions": [first_test, last_test + 1],
        "risk_stream_key": list(stream_key),
        "expert_result": fold_science.expert_result.to_dict(),
        "ensemble_result": fold_science.ensemble_result.to_dict(),
        "risk_result": fold_science.risk_result.to_dict(),
        "vibe_targets": serialized_signal,
        "scientific_inputs_hash": scientific_inputs_hash,
        "theoretical_target_causality": theoretical_target_causality,
        "runs": modes,
        "cost_effect": modes["cost-off"]["terminal_equity"]
        - modes["cost-on"]["terminal_equity"],
    }


def _ordinary_oof_payload(
    plan: ChronologicalEvaluationPlan,
    evidence: _OOFEvidence,
) -> dict[str, Any]:
    return {
        "acceptance_fixture_version": ACCEPTANCE_FIXTURE_VERSION,
        "split_manifests": [fold.manifest.to_dict() for fold in plan.folds],
        "oof_slots": [
            {
                "sample_position": slot.sample_position,
                "split_id": slot.split_id,
                "fold_index": slot.fold_index,
                "role": slot.role.value,
            }
            for slot in plan.oof_slots
        ],
        "prepared_input_audit": list(evidence.prepared_input_audit),
        "expert_predictions": evidence.expert_result.to_dict(),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "split_id": fold.split_id,
                "test_positions": list(fold.test_positions),
                "expert_result": fold.expert_result.to_dict(),
                "ensemble_result": fold.ensemble_result.to_dict(),
                "risk_result": fold.risk_result.to_dict(),
            }
            for fold in evidence.folds
        ],
    }


def _scientific_fingerprint_payload(
    *,
    snapshot_id: str,
    attempt: ExperimentAttempt,
    protocol: EvaluationProtocol,
    plan: ChronologicalEvaluationPlan,
    evidence: _OOFEvidence,
    ensemble_config: StaticEnsembleConfig,
    risk_config: RiskPolicyConfig,
    fold_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable_folds = []
    for fold in fold_reports:
        stable_runs = {}
        runs = fold["runs"]
        for cost_mode, _ in _COST_MODES:
            run = runs[cost_mode]
            stable_runs[cost_mode] = {
                "config": run["config"],
                "metrics": run["metrics"],
                "validation": run["validation"],
                "normalized_run_card": run["normalized_run_card"],
                "stable_artifact_hashes": run["stable_artifact_hashes"],
                "fills": run["fills"],
                "actual_execution_audit": run["actual_execution_audit"],
                "terminal_equity": run["terminal_equity"],
            }
        stable_folds.append(
            {
                "fold_index": fold["fold_index"],
                "split_id": fold["split_id"],
                "test_positions": fold["test_positions"],
                "execution_window_positions": fold["execution_window_positions"],
                "risk_stream_key": fold["risk_stream_key"],
                "scientific_inputs_hash": fold["scientific_inputs_hash"],
                "theoretical_target_causality": fold["theoretical_target_causality"],
                "runs": stable_runs,
            }
        )
    return {
        "acceptance_fixture_version": ACCEPTANCE_FIXTURE_VERSION,
        "dataset_snapshot_id": snapshot_id,
        "experiment_spec": attempt.spec.to_dict(),
        "experiment_spec_fingerprint": attempt.spec_fingerprint,
        "evaluation_protocol": protocol.to_dict(),
        "ensemble_config": ensemble_config.to_dict(),
        "risk_config": risk_config.to_dict(),
        "ordinary_oof": _ordinary_oof_payload(plan, evidence),
        "fold_execution": stable_folds,
    }


def _expert_counts(
    plan: ChronologicalEvaluationPlan,
    predictions: Sequence[ExpertPrediction],
) -> dict[str, dict[str, int]]:
    role_by_key = {
        (
            slot.split_id,
            int(plan.bar_intervals[slot.sample_position].end.timestamp() * 1_000_000),
        ): slot.role
        for slot in plan.oof_slots
    }
    counts = {
        expert_id: {"validation": 0, "test": 0, "total": 0}
        for expert_id in (
            MomentumExpert.expert_id,
            TrendExpert.expert_id,
            MeanReversionExpert.expert_id,
        )
    }
    for prediction in predictions:
        instant_us = int(prediction.event_at.timestamp() * 1_000_000)
        role = role_by_key[(prediction.split_id or "", instant_us)]
        counts[prediction.expert_id][role.value] += 1
        counts[prediction.expert_id]["total"] += 1
    return counts


def _render_markdown(report: Mapping[str, Any], output_dir: Path) -> str:
    lines = [
        "# Quorum V0 Offline Acceptance",
        "",
        f"Status: **{report['status']}**",
        "",
        "PASS means the declared scientific and execution controls succeeded; it is not a performance grade.",
        "",
        f"Output directory: `{output_dir}`",
        f"Fixture: `{report['acceptance_fixture_version']}`",
        f"Scientific fingerprint: `{report['reproducibility']['scientific_fingerprint']}`",
        "",
        "## Dataset and chronology",
        "",
        f"- Asset: {report['dataset']['asset']}",
        f"- Bars: {report['dataset']['bars']}",
        f"- Dataset snapshot: `{report['dataset']['snapshot_id']}`",
        f"- Ordinary folds: {report['chronology']['fold_count']}",
        f"- Validation OOF slots: {report['oof']['validation_slot_count']}",
        f"- Test OOF slots: {report['oof']['test_slot_count']}",
        f"- Final holdout: {report['chronology']['final_holdout_state']} (accessed: {report['chronology']['final_holdout_accessed']})",
        "",
        "## Controls",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in report["checks"].items())
    lines.extend(
        [
            "",
            "## Fold executions",
            "",
            "Each fold is an independent risk stream and BaseEngine run; equity curves are not stitched.",
            "",
        ]
    )
    for fold in report["folds"]:
        lines.append(
            f"- fold-{fold['fold_index']:03d}: split `{fold['split_id']}`, "
            f"cost effect {fold['cost_effect']:.10f}"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "Raw run-card bytes are not the identity because `generated_at` and `run_dir` are runtime metadata. The fingerprint retains normalized scientific run-card content and stable artifact hashes.",
            f"- External repeat comparison: {report['reproducibility']['repeatability_status']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_v0_acceptance(output_dir: Path) -> AcceptanceResult:
    """Run one complete offline Task 7 acceptance workflow.

    ``output_dir`` must not exist or must be empty.  A successful call writes
    authoritative ``quorum_acceptance.json`` plus a human-readable Markdown
    report and returns their stable identities.
    """
    output_dir = _prepare_output_directory(Path(output_dir))
    ledger = ExperimentLedger(output_dir / "experiment_ledger.jsonl")
    attempt: ExperimentAttempt | None = None
    terminal = False
    try:
        frame = _synthetic_frame()
        snapshot_id = _dataset_snapshot_id(frame)
        protocol = _evaluation_protocol(frame)
        ensemble_config = _ensemble_config()
        risk_config = _risk_config()
        spec = _experiment_spec(frame, snapshot_id)

        # Registration precedes fold materialization and every result-producing step.
        attempt = ledger.register(
            spec,
            attempt_id=ATTEMPT_ID,
            registered_at=REGISTERED_AT,
        )
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.RUNNING,
            event_id=RUNNING_EVENT_ID,
            occurred_at=RUNNING_AT,
        )

        plan = materialize_chronological_plan(
            attempt,
            protocol,
            _bar_intervals(frame),
            _ordinary_label_ends(frame),
        )
        if len(plan.folds) < 2:
            raise ValueError("acceptance protocol requires at least two ordinary folds")
        if not all(fold.leakage_audit.clean for fold in plan.folds):
            raise ValueError("acceptance chronological boundary is dirty")
        if plan.final_holdout.state is not FinalHoldoutState.LOCKED:
            raise ValueError("acceptance final holdout is not locked")

        evidence = _generate_oof_evidence(
            frame,
            plan,
            ensemble_config,
            risk_config,
        )
        fold_reports = tuple(
            _execute_fold(
                output_dir=output_dir,
                frame=frame,
                snapshot_id=snapshot_id,
                attempt=attempt,
                protocol=protocol,
                plan=plan,
                fold_science=fold_science,
                ensemble_config=ensemble_config,
                risk_config=risk_config,
            )
            for fold_science in evidence.folds
        )

        if (
            sum(
                run["fill_count"]
                for fold in fold_reports
                for run in fold["runs"].values()
            )
            == 0
        ):
            raise ValueError("acceptance fixture produced no actual fills")
        if not any(
            fold["runs"]["cost-on"]["observed_slippage"] for fold in fold_reports
        ):
            raise ValueError("cost-on acceptance runs contain no slippage evidence")
        if not any(fold["cost_effect"] > 1e-8 for fold in fold_reports):
            raise ValueError("acceptance fixture produced no strict cost effect")

        validation_slots = len(plan.validation_slots)
        test_slots = len(plan.test_slots)
        prediction_keys = [
            (
                prediction.expert_id,
                prediction.split_id,
                prediction.event_at.isoformat(),
            )
            for prediction in evidence.expert_result.predictions
        ]
        expert_prediction_counts = _expert_counts(
            plan,
            evidence.expert_result.predictions,
        )
        expected_expert_counts = {
            "validation": validation_slots,
            "test": test_slots,
            "total": validation_slots + test_slots,
        }
        if any(
            counts != expected_expert_counts
            for counts in expert_prediction_counts.values()
        ):
            raise ValueError(
                "acceptance requires one prediction from every expert for "
                "every ordinary OOF slot"
            )
        duplicate_prediction_count = len(prediction_keys) - len(set(prediction_keys))
        if duplicate_prediction_count:
            raise ValueError("acceptance expert predictions contain duplicate keys")
        missing_test_prediction_count = test_slots * len(
            expert_prediction_counts
        ) - sum(counts["test"] for counts in expert_prediction_counts.values())
        if missing_test_prediction_count:
            raise ValueError("acceptance expert predictions are missing test slots")
        expected_test_positions = {
            (slot.split_id, slot.sample_position) for slot in plan.test_slots
        }
        emitted_test_positions = {
            (
                prediction.split_id,
                frame.index.get_loc(pd.Timestamp(prediction.event_at)),
            )
            for prediction in evidence.expert_result.predictions
            if any(
                slot.split_id == prediction.split_id
                and slot.role is OOFRole.TEST
                and frame.index[slot.sample_position]
                == pd.Timestamp(prediction.event_at)
                for slot in plan.test_slots
            )
        }
        if emitted_test_positions != expected_test_positions:
            raise ValueError(
                "acceptance emitted unauthorized or missing test positions"
            )

        ordinary_oof_payload = _ordinary_oof_payload(plan, evidence)
        ordinary_oof_fingerprint = _sha256_json(ordinary_oof_payload)
        fingerprint_payload = _scientific_fingerprint_payload(
            snapshot_id=snapshot_id,
            attempt=attempt,
            protocol=protocol,
            plan=plan,
            evidence=evidence,
            ensemble_config=ensemble_config,
            risk_config=risk_config,
            fold_reports=fold_reports,
        )
        scientific_fingerprint = _sha256_json(fingerprint_payload)

        outcome = ExperimentOutcome(
            detail="Quorum V0 offline acceptance controls passed",
            references=ExternalRecordRefs(
                run_card_ref="quorum:acceptance:fold-run-cards:v1",
                quorum_artifact_ids=(
                    "quorum:acceptance:report:"
                    + scientific_fingerprint.removeprefix("sha256:"),
                ),
            ),
            metadata={
                "status": "PASS",
                "scientific_fingerprint": scientific_fingerprint,
            },
        )
        ledger.transition(
            attempt.attempt_id,
            ExperimentState.COMPLETED,
            outcome=outcome,
            event_id=COMPLETED_EVENT_ID,
            occurred_at=COMPLETED_AT,
        )
        terminal = True

        history = ledger.history(attempt.attempt_id)
        report: dict[str, Any] = {
            "schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "status": "PASS",
            "acceptance_fixture_version": ACCEPTANCE_FIXTURE_VERSION,
            "acceptance_meaning": (
                "Scientific and execution controls passed; this is not a "
                "profitability or recommendation grade."
            ),
            "dataset": {
                "asset": ASSET,
                "bars": len(frame),
                "frequency": "1h",
                "timezone": "UTC",
                "timestamp_convention": "bar close/observation time",
                "snapshot_id": snapshot_id,
                "columns": list(frame.columns),
            },
            "experiment": {
                "attempt_id": attempt.attempt_id,
                "spec": spec.to_dict(),
                "spec_fingerprint": attempt.spec_fingerprint,
                "lifecycle_state": ledger.get(attempt.attempt_id).state.value,
                "lifecycle_history": [record.to_dict() for record in history],
                "registration_preceded_materialization": True,
                "lifecycle_timestamps_are_fixture_metadata": True,
            },
            "chronology": {
                "protocol_id": protocol.protocol_id,
                "protocol": protocol.to_dict(),
                "fold_count": len(plan.folds),
                "split_manifests": [fold.manifest.to_dict() for fold in plan.folds],
                "leakage_audits": [
                    {
                        "fold_index": fold.manifest.fold_index,
                        "clean": fold.leakage_audit.clean,
                        "overlapping_positions": list(
                            fold.leakage_audit.overlapping_positions
                        ),
                        "shared_positions": list(fold.leakage_audit.shared_positions),
                        "embargo_violation_positions": list(
                            fold.leakage_audit.embargo_violation_positions
                        ),
                    }
                    for fold in plan.folds
                ],
                "final_holdout_state": plan.final_holdout.state.value,
                "final_holdout_accessed": plan.final_holdout.accessed_at is not None,
                "final_holdout_positions": [len(frame) - HOLDOUT_BARS, len(frame) - 1],
            },
            "oof": {
                "validation_slot_count": validation_slots,
                "test_slot_count": test_slots,
                "assignments": ordinary_oof_payload["oof_slots"],
                "prepared_input_audit": list(evidence.prepared_input_audit),
                "expert_result": evidence.expert_result.to_dict(),
                "expert_prediction_counts": expert_prediction_counts,
                "prediction_count": len(evidence.expert_result.predictions),
                "duplicate_prediction_count": duplicate_prediction_count,
                "missing_test_prediction_count": missing_test_prediction_count,
            },
            "experts": [
                {
                    "expert_id": expert.expert_id,
                    "expert_version": expert.expert_version,
                    "config_id": expert.config_id,
                    "fit_required": False,
                }
                for expert in (MomentumExpert, TrendExpert, MeanReversionExpert)
            ],
            "ensemble": {
                "config_id": ENSEMBLE_CONFIG_ID,
                "config": ensemble_config.to_dict(),
            },
            "risk": {"config": risk_config.to_dict()},
            "folds": list(fold_reports),
            "cost_comparison": {
                "primary_mode": "cost-on",
                "counterfactual_mode": "cost-off",
                "cost_on_slippage_us": dict(_COST_MODES)["cost-on"],
                "cost_off_slippage_us": dict(_COST_MODES)["cost-off"],
                "identical_scientific_inputs": all(
                    fold["runs"]["cost-on"]["scientific_inputs_hash"]
                    == fold["runs"]["cost-off"]["scientific_inputs_hash"]
                    for fold in fold_reports
                ),
                "strict_cost_effect_fold_count": sum(
                    fold["cost_effect"] > 1e-8 for fold in fold_reports
                ),
                "total_terminal_equity_cost_effect": math.fsum(
                    fold["cost_effect"] for fold in fold_reports
                ),
            },
            "reproducibility": {
                "scientific_fingerprint": scientific_fingerprint,
                "ordinary_oof_fingerprint": ordinary_oof_fingerprint,
                "fingerprint_definition": [
                    "acceptance fixture version",
                    "dataset snapshot identity",
                    "ExperimentSpec and fingerprint",
                    "EvaluationProtocol and SplitManifest values",
                    "OOF slot assignments and causal prepared-input audit",
                    "expert, ensemble, and risk result payloads",
                    "cost configurations",
                    "stable per-fold metrics and execution evidence",
                    "stable artifact hashes",
                    "run-card content excluding generated_at and run_dir",
                ],
                "repeatability_status": "not-yet-compared",
                "raw_run_card_bytes_are_identity": False,
                "normalized_run_card_omissions": ["generated_at", "run_dir"],
            },
            "checks": {
                "chronological_boundaries_clean": True,
                "test_predictions_are_oof": True,
                "causal_prepared_data_prefixes": all(
                    audit["prepared_end_position"] == audit["sample_position"]
                    for audit in evidence.prepared_input_audit
                ),
                "final_holdout_locked": True,
                "final_holdout_accessed": False,
                "independent_fold_execution": True,
                "theoretical_target_causality": all(
                    row["causal"]
                    for fold in fold_reports
                    for row in fold["theoretical_target_causality"]
                ),
                "actual_fill_execution_authorized": all(
                    run["actual_execution_audit"]["passed"]
                    for fold in fold_reports
                    for run in fold["runs"].values()
                ),
                "execution_after_availability": all(
                    run["actual_execution_audit"]["passed"]
                    for fold in fold_reports
                    for run in fold["runs"].values()
                ),
                "costs_applied": any(
                    fold["cost_effect"] > 1e-8 for fold in fold_reports
                ),
                "requested_and_realized_exposure_separate": True,
                "network_required": False,
            },
        }
        report_path = output_dir / "quorum_acceptance.json"
        markdown_path = output_dir / "quorum_acceptance.md"
        _write_json(report_path, report)
        markdown_path.write_text(
            _render_markdown(report, output_dir),
            encoding="utf-8",
        )
        return AcceptanceResult(
            status="PASS",
            output_dir=output_dir,
            report_path=report_path,
            markdown_path=markdown_path,
            scientific_fingerprint=scientific_fingerprint,
            ordinary_oof_fingerprint=ordinary_oof_fingerprint,
            dataset_snapshot_id=snapshot_id,
            experiment_spec_fingerprint=attempt.spec_fingerprint,
            fold_count=len(plan.folds),
            validation_oof_slots=validation_slots,
            test_oof_slots=test_slots,
        )
    except Exception as exc:
        if attempt is not None and not terminal:
            try:
                ledger.transition(
                    attempt.attempt_id,
                    ExperimentState.FAILED,
                    outcome=ExperimentOutcome(
                        detail=f"Task 7 acceptance failed: {type(exc).__name__}",
                        metadata={"status": "FAIL"},
                    ),
                    event_id=FAILED_EVENT_ID,
                    occurred_at=FAILED_AT,
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m src.quorum.acceptance``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_v0_acceptance(args.output_dir)
    except Exception as exc:
        print(
            _canonical_json(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        _canonical_json(
            {
                "status": result.status,
                "scientific_fingerprint": result.scientific_fingerprint,
                "report": str(result.report_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AcceptanceResult", "run_v0_acceptance"]
