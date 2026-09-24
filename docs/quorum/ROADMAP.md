# Quorum roadmap

Status: proposed after the Phase 0 repository audit (2026-09-23). No item in
this document grants live-trading authority.

## Sequencing changes from the initial proposal

The requested sequence puts rigorous walk-forward/out-of-fold evaluation at 0.3.
That is too late: once 0.1 and 0.2 results are visible, the architecture and
thresholds will already have been influenced by the same history. Quorum should
define the experiment and chronological split contracts in 0.1 and make them
mandatory in 0.2. Calibration also cannot be evaluated honestly without held-out
predictions, so it follows that foundation.

The existing Vibe-Trading implementation supplies good split primitives in
[`src.quantlib.crossvalidation`](../../agent/src/quantlib/crossvalidation.py), but
[`backtest.validation.walk_forward_analysis`](../../agent/backtest/validation.py#L209)
only divides an already-realized equity curve into reporting windows. It does not
refit an expert or create out-of-fold predictions. The roadmap therefore treats a
Quorum evaluation coordinator as new integration work, not as a thin rename.

## Milestones

### Quorum 0.1 — contracts and experiment spine

- Immutable expert-prediction and batch-result contracts.
- Explicit `available_at`, prediction horizon, asset, signed score, optional
  probability/confidence, expert identity, and version semantics.
- Frozen evaluation protocol and split-manifest contracts, including final-holdout
  state.
- Append-only Quorum experiment record that references existing Hypothesis,
  Strategy Store artifact, and run-card IDs instead of copying their databases.
- Static ensemble configuration contract and a Vibe `SignalEngine` adapter
  boundary, without expert formulas or optimization.

Exit: contract/property tests reject non-finite values, invalid timestamps,
ambiguous probability semantics, duplicate prediction keys, versionless experts,
mutable configurations, and attempts to consume a locked holdout.

### Quorum 0.2 — deterministic V0 and enforced chronological evaluation

- Three deliberately simple, parameter-frozen experts: momentum, trend, and mean
  reversion.
- Static weighted voting and explicit reporting labels (`BUY`, `HOLD`, `SELL`).
  The backtester still receives continuous target scores/weights, because that is
  its native contract.
- A fixed, conservative risk policy after the ensemble: max gross exposure,
  per-name cap, zero leverage, and turnover cap.
- Expanding/rolling chronological evaluation using the existing purged split
  primitives where labels overlap.
- Existing Vibe backtest engines, fills, costs, metrics, run cards, and frontend
  artifacts.

Exit: a fully offline synthetic-data acceptance test proves that every fold trains
only on available history, all test predictions are out of fold, execution occurs
after prediction availability, costs are applied, and repeated runs are identical.

### Quorum 0.3 — confidence and calibration

- Define confidence separately from probability and signal magnitude.
- Fit calibration only on prior out-of-fold predictions; compare reliability
  diagrams, Brier/log loss, and uncalibrated baselines.
- Refuse probability claims from experts that emit only rankings or scores.

Exit: calibration improves or is rejected on a reserved validation interval, with
no final-holdout access and with calibration artifacts versioned.

### Quorum 0.4 — adaptive expert weighting

- Predeclared rolling reliability estimators and bounded, regularized weight
  updates.
- Weight latency, decay, minimum observations, missing-expert behavior, and maximum
  concentration are explicit.
- Static/equal-weight baselines remain mandatory.

Exit: adaptation is evaluated only from lagged out-of-fold evidence and beats or
is rejected against static baselines under identical costs.

### Quorum 0.5 — regime-conditioned analysis and weighting

- Begin with deterministic, causal regime descriptors; avoid an unconstrained
  regime classifier.
- Report expert reliability by predeclared regime before permitting regime weights.
- Treat the existing correlation edge-density regime helper as a separate
  descriptive feature, not as trend/bull/bear truth.

Exit: regime definitions use only information available at the decision time,
rare-regime uncertainty is reported, and a non-regime baseline remains competitive
evidence.

### Quorum 0.6 — robustness suite

- Dependence-aware/block bootstrap, cost and slippage grids, parameter perturbation,
  start-date sensitivity, missing-data stress, and universe perturbation.
- Integrate strict split-boundary leakage checks and produce a machine-readable
  protocol-deviation report.
- Quarantine or rename legacy diagnostics whose labels overstate their guarantees.

Exit: a run cannot receive a validation-complete status when required robustness
checks are missing, failed, or non-evaluable.

### Quorum 0.7 — interpretable meta-models

- First establish a regularized linear stacker over out-of-fold expert outputs and
  small causal state variables.
- Add symbolic regression only with an allow-listed grammar, hard depth/node limits,
  explicit complexity penalty, stability selection, and nested chronological
  evaluation.
- Never expose unrestricted raw market data to symbolic search by default.

Exit: every meta-model has a readable expression/model card, complexity accounting,
and an honest simpler baseline.

### Quorum 0.8 — multiple-testing and research-quality gates

- Make trial-family accounting, Deflated Sharpe Ratio, FDR, and CSCV/PBO first-class
  reports using [`src.quantlib.multipletesting`](../../agent/src/quantlib/multipletesting.py).
- Add a dependence-aware strategy-family reality-check procedure when justified by
  the selection process.
- Introduce explicit research-passport states and guarded transitions.

Exit: omitted/failed trials cannot disappear from the denominator and performance
cannot be published without its evaluation protocol.

### Quorum 0.9 — prospective paper evaluation

- Broker-write-free prospective prediction ledger first: timestamped signals,
  decisions, hypothetical orders, market observations, and immutable evaluation.
- Only later adapt approved paper-broker connectors behind an explicit capability
  gate; do not reuse live mandates as an implicit research simulator.
- Monitor drift, missing data, latency, cost error, and expert disagreement.

Exit: prospective results are reproducible from the recorded event stream and no
code path can submit a live order.

### Quorum 1.0 — frozen prospective research system

- Freeze protocol, code, data contracts, cost assumptions, expert versions, and
  decision policy before the prospective period.
- Permit changes only through new, linked experiment versions.
- Publish a complete research passport with negative results and limitations.

Exit: an independent replay reproduces decisions and the final report clearly
separates discovery, validation, holdout, and prospective evidence.

## V0 implementation plan

These are small, independently reviewable tasks. File names are proposed; they do
not exist yet unless marked existing.

| Task | Goal | Likely files | Required tests | Completion criterion |
| --- | --- | --- | --- | --- |
| 1. Contract spine | Define immutable prediction, expert protocol, static ensemble config, evaluation protocol, and split manifest. No formulas. | New `agent/src/quorum/contracts.py`, `agent/src/quorum/__init__.py`; new `agent/tests/quorum/test_contracts.py` | Bounds/finite/property tests; timezone and availability rules; serialization round trip; immutability | Invalid or ambiguous records fail closed; the module has no loader, backtest, agent, API, or live dependency. |
| 2. Experiment record | Add append-only experiment IDs and references to existing Hypothesis, Artifact, and RunCard records. | New `agent/src/quorum/experiments.py`; existing [`hypotheses/registry.py`](../../agent/src/hypotheses/registry.py), [`strategy_store/models.py`](../../agent/src/strategy_store/models.py), [`backtest/run_card.py`](../../agent/backtest/run_card.py) through adapters | Append-only behavior; same-spec identity; changed spec creates child; failed trials retained; concurrent/atomic write | A trial is registered before evaluation and cannot be rewritten after results exist. |
| 3. Chronological coordinator | Materialize train/test/purge/embargo/final-holdout boundaries and OOF prediction rows. | New `agent/src/quorum/validation/protocol.py`; reuse [`quantlib/crossvalidation.py`](../../agent/src/quantlib/crossvalidation.py) | Synthetic leakage traps; label overlap; embargo; insufficient history; locked holdout; every OOF row predicted exactly once | Boundary-leakage detector is clean and no fit call sees test/future rows. |
| 4. Deterministic experts | Implement momentum, trend, and mean-reversion experts over injected OHLCV data with frozen V0 parameters. | New `agent/src/quorum/experts/*.py`; reuse factor registry where definitions match | Causality tests; missing bars; warmup; scale/asset isolation; golden synthetic series | Each expert emits only standardized predictions and never fetches, sizes, ensembles, or writes. |
| 5. Static ensemble | Combine aligned expert scores with fixed weights; preserve missingness and report disagreement. | New `agent/src/quorum/ensemble/static.py` | Weight validation; missing experts; alignment; permutation invariance; no lookahead; BUY/HOLD/SELL threshold edges | Same input yields same combined score and a full attribution record. |
| 6. Risk boundary and adapter | Map ensemble score to constrained target weights, then adapt them to Vibe's `SignalEngine.generate` result. | New `agent/src/quorum/risk/policy.py`, `agent/src/quorum/adapters/vibe_signal.py`; existing [`backtest/engines/base.py`](../../agent/backtest/engines/base.py) | Gross/name/leverage/turnover caps; signal-to-fill causality; hold versus rebalance semantics; zero-capital/lot behavior | Existing BaseEngine runs Quorum signals with no engine fork and every requested/realized exposure difference is auditable. |
| 7. V0 acceptance workflow | Run frozen synthetic/local data end to end and emit existing metrics, validation, artifacts, and run card plus Quorum protocol metadata. | Existing [`tools/backtest_tool.py`](../../agent/src/tools/backtest_tool.py) and run artifacts; optional thin new Quorum tool only after the deterministic path works | Offline E2E, reproducibility hashes, costs on/off comparison, intentional leakage rejection, frontend artifact compatibility | One command produces a reproducible, cost-aware OOF V0 report; no provider, broker, LLM, or network is required. |

## Single best next Codex task

**Implement Task 1 only: the Quorum 0.1 contract spine and its tests.** Define
immutable `ExpertPrediction`, `ExpertProtocol`, `StaticEnsembleConfig`,
`EvaluationProtocol`, and `SplitManifest` types under `agent/src/quorum/`, with
strict timestamp/availability, finite-value, range, identity/version, and
serialization validation. Do not add expert formulas, voting execution, database
storage, backtest integration, API routes, UI, or live/paper trading in that task.
