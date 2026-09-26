# Quorum Phase 0: repository baseline and architecture

Audit snapshot: 2026-09-23. Scope: inspect and verify the existing
Vibe-Trading fork; design Quorum boundaries; do not implement Quorum trading
logic.

## Executive conclusions

1. The fork is a substantial, runnable research/trading application, not a blank
   strategy repository. Quorum should reuse its loaders, Alpha Zoo, target-weight
   signal adapter, market engines, accounting, artifacts, API, and frontend.
2. The scientific validation surface is uneven. Vibe includes strong low-level
   purged split and multiple-testing utilities, but the generic backtest's
   `walk_forward_analysis` does not train, refit, or produce out-of-fold (OOF)
   predictions. Its Monte Carlo test permutes the order of the same realized trade
   P&Ls, and its bootstrap samples bar returns IID. Those are useful diagnostics
   with narrower claims than their labels suggest.
3. The ML implementation is a documented code template, not a reusable model
   lifecycle. It correctly purges future labels in its example but has no persisted
   fitted model, OOF prediction store, final-holdout gate, or calibration framework.
4. Experiment information is split across Hypothesis Registry, Strategy Store,
   backtest run cards, and an agent governance manifest. Quorum should reference
   and extend these records with an append-only experiment/split ledger, not create
   a fourth competing strategy database.
5. Expert output, ensemble combination, risk sizing, and execution must be separate.
   Vibe's numeric signals are target weights in `[-1, 1]`; `HOLD` is currently an
   execution-adjustment mode, not a universal signal enum. Quorum may report
   `BUY/HOLD/SELL`, but the engine adapter should preserve continuous scores.
6. Live execution is mature enough to be dangerous: connectors, order service,
   mandates, and kill switches exist. None belongs in V0. Prospective paper
   evaluation should begin as a broker-write-free event ledger.

## A. Repository baseline report

### Git state at audit start

| Item | Observed state |
| --- | --- |
| Branch | `feat/quorum-core` |
| HEAD | `a4821b1e0775302d15afb99f71ca3a44391f4495` — `test(entities): the spine guards pin the interval set with 1W and 1M` |
| Working tree | Clean before this audit; the only intentional changes after it are these `docs/quorum/` files |
| `origin` | `https://github.com/DPLCoding/Quorum.git` for fetch and push — correct fork |
| `upstream` | `https://github.com/HKUDS/Vibe-Trading.git` for fetch and push — correctly configured |
| Upstream divergence | Not asserted: `refs/remotes/upstream/main` is not present locally. Fetching was not needed for Phase 0 and was deliberately not performed. |

No Git history, remote, credentials, or branches were modified. To refresh the
remote-tracking reference later, the non-destructive command is
`git fetch upstream main`; then inspect with
`git rev-list --left-right --count upstream/main...HEAD`.

### Runtime and dependency model

- Python requires 3.11 or newer in [`pyproject.toml`](../../pyproject.toml#L5).
  The audited host uses Python 3.13.3 and pip 25.0.1.
- Python packaging is setuptools/PEP 621. Local development uses editable
  `pip install -e ".[dev]"`; the `dev` extra supplies pytest, coverage,
  pytest-socket, Black, and Ruff. Console entries are `vibe-trading` and
  `vibe-trading-mcp` in [`pyproject.toml`](../../pyproject.toml#L84).
- `requirements-lock.txt` and `requirements-channels-lock.txt` are uv-generated,
  hashed universal lock files used by container/release paths. The Docker build
  installs them with `--require-hashes`; editable development remains governed by
  `pyproject.toml`.
- The frontend is React 19, TypeScript, Vite 8, and Vitest, with scripts defined in
  [`frontend/package.json`](../../frontend/package.json#L9). It declares Node
  `>=22.22.0`. The host has Node 24.13.1/npm 11.8.0; `jsdom@30.0.1` warns that this
  particular Node 24 release is below its `^24.15.0` support floor. Use Node
  24.15+ or 22.22.2+ for a warning-free supported toolchain. On this Windows host,
  use `npm.cmd` if PowerShell blocks the unsigned `npm.ps1` shim.

### Verified local setup

The documented contributor checks are in
[`AGENT_CONTRIBUTOR_GUIDE.md`](../../AGENT_CONTRIBUTOR_GUIDE.md#L28) and formatting
guidance is in [`CONTRIBUTING.md`](../../CONTRIBUTING.md#L142). The audit safely ran:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\vibe-trading.exe --help

cd frontend
npm.cmd ci --no-audit --no-fund
npm.cmd run build
npm.cmd run test:run
```

Results:

- Editable Python base/dev installation: passed.
- CLI import and command discovery: passed; no credentials needed.
- Focused offline execution/robustness/rebalance/validation/run-card tests:
  **135 passed, 1 skipped**. This exercises the research/backtest core without an
  LLM, market-data network call, or broker.
- Frontend TypeScript/production build: passed. Vite reported only large-chunk
  optimization warnings.
- Frontend tests: **68 files, 652 tests passed**.
- Python contributor suite: **14,691 passed, 166 skipped, 16 failed, 11 setup
  errors** in 10m38s. See “Verification status” for the classified failures. It
  was run without the two credential/network-oriented E2E groups.

No broker command, order path, live server, credential initializer, or remote market
data request was executed. `.venv`, `node_modules`, and generated frontend output are
ignored local build products, not repository changes.

### Recommended developer workflow

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# Copy only when ready to configure a provider; never commit this file.
Copy-Item agent\.env.example agent\.env

# Terminal/API backend (default documented API port is 8899)
vibe-trading serve --port 8899

# Separate terminal: development frontend (documented port 5899)
cd frontend
npm.cmd ci
npm.cmd run dev
```

For a production-like local UI, run `npm.cmd run build` and then
`vibe-trading serve --port 8899`; the backend serves the built frontend. The
natural-language example in [`README.md`](../../README.md#L1160) is:

```powershell
vibe-trading run -p "Backtest a 20/50-day moving average crossover on AAPL for the past year, show Sharpe ratio and max drawdown"
```

That workflow needs an LLM provider and may fetch market data. The deterministic
Quorum acceptance workflow bypasses both by using its frozen in-process fixture
and direct research/backtest APIs.

### Environment boundary

The agent requires `LANGCHAIN_PROVIDER`, `LANGCHAIN_MODEL_NAME`, and the chosen
provider's base URL/key as applicable; Ollama can be local/keyless and some provider
paths use OAuth. [`agent/.env.example`](../../agent/.env.example) is the template,
and the concise variable table is in [`README.md`](../../README.md#L980).

Market-data credentials are optional for many routes because loaders have free
fallbacks. Examples include `TUSHARE_TOKEN`, `FMP_API_KEY`, and `FRED_API_KEY`.
`API_AUTH_KEY` is recommended—and required by the application for sensitive access
from non-loopback clients. Connector/broker credentials are unrelated to research
setup and must not be configured for V0.

### Test, lint, and type-check commands

```powershell
# Maintainer's offline-safe Python suite
.\.venv\Scripts\python.exe -m pytest `
  --ignore=agent/tests/e2e_backtest `
  --ignore=agent/tests/test_e2e_harness_v2.py `
  --tb=short -q

# Targeted formatting/lint for touched Python files
.\.venv\Scripts\black.exe --check <files>
.\.venv\Scripts\ruff.exe check <files>

# Syntax smoke check used by contributor guidance
.\.venv\Scripts\python.exe -m compileall -q agent/src agent/backtest agent/cli

# Frontend type-check + build and tests
cd frontend
npm.cmd run build
npm.cmd run test:run
```

There is no configured repository-wide mypy/pyright gate. `tsc -b` inside the
frontend build is the effective TypeScript type check. Contributor guidance prefers
targeted Black/Ruff checks because the whole historical tree is not promised clean.

## B. Existing architecture map

### Strategies and signals

The generated-strategy contract is a run directory containing root `config.json`
and `code/signal_engine.py`. `SignalEngine.generate(data_map)` returns a mapping of
symbol to `pandas.Series`; values are numeric signed target weights/strengths in
`[-1, 1]`. See the concrete contract in
[`strategy-generate/SKILL.md`](../../agent/src/skills/strategy-generate/SKILL.md#L49),
the AST source gate
[`_validate_signal_engine_source`](../../agent/backtest/runner.py#L785), and the
runtime interface gate
[`_validate_signal_engine_class`](../../agent/backtest/runner.py#L819).

[`BacktestConfigSchema`](../../agent/backtest/runner.py#L77) owns symbols, dates,
source, interval, engine, initial cash, warmup/evaluation boundary, fundamentals,
event feeds, and execution mode. It permits extra fields for engine-specific
configuration, which aids extensibility but weakens global schema guarantees.

[`BaseEngine._align`](../../agent/backtest/engines/base.py#L238) shifts each signal
one bar on that instrument's own calendar, clips it to `[-1, 1]`, bounds forward
fill, and normalizes cross-sectional gross target weight to at most one.
[`BaseEngine.run_backtest`](../../agent/backtest/engines/base.py#L869) then fetches
data, enriches it, invokes the signal engine, optionally optimizes, excludes warmup,
executes bar by bar, and writes metrics/artifacts.

Important semantics:

- There is no universal `BUY/SELL/HOLD` domain object in this path. Negative,
  zero, and positive scores are the executable representation.
- `position_adjustment="hold"` means ignore same-direction resizing once a
  position exists; `"rebalance"` follows target changes, with optional tolerance
  and mask. It is not the same as a neutral signal.
- Signal magnitude currently encodes desired exposure, so strategy evidence and
  sizing are partly mixed. Quorum experts should emit evidence; a separate risk
  policy should create target weights.
- Immutable accounting records are [`Position`, `FillRecord`, `TradeRecord`, and
  `EquitySnapshot`](../../agent/backtest/models.py#L14).

Persisted research strategies already have a lifecycle. The
[`Artifact`](../../agent/src/strategy_store/models.py#L81) record stores factor or
strategy type, signal/entry/exit/sizing descriptions, source/run paths, parent and
hypothesis IDs, version/governance fields, and validation status. `ArtifactStatus`
covers `CREATED`, `BENCHING`, `ACTIVE`, `MONITORING`, `DECAYED`, and `DISABLED`.
Quorum should extend this through references and guarded status mapping.

### Alpha and factor system

The Alpha Zoo is one of the strongest reusable parts. Strict
[`AlphaMeta`](../../agent/src/factors/registry.py#L87) metadata and lazy
[`Registry`](../../agent/src/factors/registry.py#L201) loading govern factor
definitions. The registry checks required inputs/sector data, output shape,
infinities, severe missingness, and missing-input propagation. The shipped catalog
contains 462 factors across Qlib 158, Alpha101, GTJA191, academic, and fundamental
families.

Factor computation follows `compute(panel) -> DataFrame` with the same panel shape.
[`compute_ic_series`](../../agent/src/factors/factor_analysis_core.py#L8) calculates
cross-sectional daily Spearman IC and
[`compute_group_equity`](../../agent/src/factors/factor_analysis_core.py#L50)
evaluates quantile portfolios. [`FactorAnalysisTool`](../../agent/src/tools/factor_analysis_tool.py#L108)
writes IC series/summary and grouped equity artifacts.

The standard bench reports IC mean/std/IR, positive ratio, t-statistic, and
alive/reversed/dead categories. It applies Deflated Sharpe machinery to IC-IR; that
is a pragmatic ranking diagnostic, not literally a return Sharpe. The opt-in
[`run_bench_strict`](../../agent/src/factors/bench_runner_strict.py#L317) adds
same-universe random controls and an optional chronological OOS split, with explicit
confirmed/train-only/reversed/noise categories. Because strict mode is optional, it
is not a system-wide scientific gate.

[`ZooSignalEngine`](../../agent/src/skills/multi-factor/zoo_signal_engine.py#L60)
already combines zoo factors with weights/z-scores/ranks and adapts them to a Vibe
signal engine. Quorum factor experts should wrap this capability rather than clone
factor formulas. [`factor_costs.py`](../../agent/backtest/factor_costs.py) provides
weight-space ADV capacity, fixed/linear/square-root impact, borrow cost, and unfilled
turnover, but it is not automatically applied to every IC bench.

### Backtesting and accounting

[`_create_market_engine`](../../agent/backtest/runner.py#L1478) routes to engines for
China A-shares, global equities, crypto, China/global futures, forex, India, Korea,
Vietnam, composite markets, and a separate options portfolio path. The generic
fallback for an unhandled market deserves hardening because it ultimately selects
crypto behavior rather than failing on ambiguity.

The engines model market-specific fees, taxes, slippage, lot sizes, margin,
funding/borrow where applicable, fills, positions, trades, cash, equity, and
benchmarks. For example, the China engine models commission/stamp/transfer costs;
global equities distinguish US/HK/UK costs; crypto models maker/taker, funding, and
liquidation; and futures use product/default commissions and margin. These defaults
are useful but remain assumptions that each experiment must freeze and stress.

Orders are simulated from target weights against next-bar execution data. Actual
fills may differ because of cash, margin, rounding, market rules, halts, and
liquidity. [`resolve_benchmark`](../../agent/backtest/benchmark.py#L67) loads an
explicit benchmark; absent one, base behavior can use the equal-weight mean of the
tested instruments. Metrics include returns, drawdown, Sharpe/Sortino/Calmar,
win/loss statistics, benchmark excess/information ratio/tracking error/beta, and
actual-fill turnover.

Optional portfolio optimizers implement equal volatility, risk parity,
mean-variance, maximum diversification, and turnover-aware optimization under
[`BaseOptimizer`](../../agent/backtest/optimizers/base.py#L14). `MaxWeight`,
`MinWeight`, and `GroupExposure` constraints live in
[`constraints.py`](../../agent/backtest/constraints.py#L48).

### Validation: implemented capability versus scientific claim

| Capability | Implementation | Audit assessment |
| --- | --- | --- |
| Chronological/purged splits | `purged_kfold_splits`, `group_purged_kfold_splits`, `purged_walk_forward_splits`, and `combinatorial_purged_splits` in [`crossvalidation.py`](../../agent/src/quantlib/crossvalidation.py#L215) | Good reusable primitives, including purge/embargo concepts, but not enforced by generic strategies or the ML template. |
| Boundary audit | [`detect_boundary_leakage`](../../agent/src/quantlib/crossvalidation.py#L512) | Reuse as a mandatory Quorum split assertion. It cannot detect semantic leakage inside arbitrary features. |
| Generic “walk-forward” | [`walk_forward_analysis`](../../agent/backtest/validation.py#L209) | Misleading for model validation: it slices a completed equity/trade history into non-overlapping windows. It does not refit, tune, or make OOF predictions. Rename or clearly label it subperiod consistency. |
| Rolling training | ML skill's [`walk_forward_predict`](../../agent/src/skills/ml-strategy/SKILL.md#L100) | Example-level expanding/sliding refit, not a shared framework or experiment gate. |
| Bootstrap | [`bootstrap_sharpe_ci`](../../agent/backtest/validation.py#L137) | IID bar-return resampling; useful only with that dependence assumption. No block/stationary bootstrap in the generic path. |
| Monte Carlo | [`monte_carlo_test`](../../agent/backtest/validation.py#L30) | Permutes the order of the same trade P&Ls. Total P&L is fixed; it measures path/drawdown ordering, not whether signals beat random signals. |
| Multiple testing | PSR, DSR, Benjamini-Hochberg FDR, and CSCV PBO in [`multipletesting.py`](../../agent/src/quantlib/multipletesting.py#L180) | Substantive, tested library utilities. DSR is used in factor benching; generic strategy research does not centrally track the full tried family or enforce these. |
| Parameter/start-date sensitivity | No general framework found | New Quorum validation orchestration is required. |
| White's Reality Check | No implementation found | Later addition if selection spans a large, dependent strategy family; do not add merely as a checkbox. |
| Final holdout | Stages/fields appear in some discovery concepts | No central lock preventing reads/tuning and no one-open policy. New contract and gate required. |

The optional validation hook in
[`BaseEngine.run_backtest`](../../agent/backtest/engines/base.py#L1117) writes these
diagnostics after the backtest. It does not make an invalid strategy-generation
process valid.

### Machine learning

ML is currently a skill/document template in
[`ml-strategy/SKILL.md`](../../agent/src/skills/ml-strategy/SKILL.md), not a product
subsystem. Its embedded code builds OHLCV-derived features, creates a forward-return
binary label, fits a training-only `StandardScaler` plus random forest, gradient
boosting, or logistic regression, and maps `predict_proba` to `[-1, 1]`.

The template's `train_stop = i - prediction_horizon + 1` purges labels that would
not be observable at the prediction time. Tests extract and exercise this embedded
code, including horizon boundaries, single-class windows, skipped retrains, and
trailing unlabeled rows. That is a valuable example but not equivalent to a reusable
ML architecture. Missing pieces include a model interface, multi-asset split policy,
OOF prediction artifact, calibration, fitted-model persistence/versioning,
retraining scheduler, final-holdout gate, and immutable feature/label schema. The
template also warns that same-close feature availability must be handled; the
engine's next-bar shift alone does not prove every upstream field was published in
time.

### Data

[`DataLoaderProtocol`](../../agent/backtest/loaders/base.py#L795) defines loader
name/markets/auth/availability and `fetch(codes, start, end, interval, fields)`.
[`loaders/registry.py`](../../agent/backtest/loaders/registry.py#L36) declares 28
sources plus `auto`, market fallback chains, and sources prohibited from network
fallback. [`fetch_market_data`](../../agent/src/market_data.py#L177) normalizes
multi-symbol results and records requested/used source, fallback, currency/volume,
and adjustment provenance.

The opt-in loader cache (`VIBE_TRADING_DATA_CACHE`) in
[`loaders/base.py`](../../agent/backtest/loaders/base.py#L400) caches settled
historical ranges by source/symbol/timeframe/range/fields. Its key does not include
provider revision, retrieval timestamp, or a declared dataset-version hash, so two
machines can legitimately obtain different data under the same experiment config.
Quorum needs content hashes/snapshots in serious research passports.

OHLC validation and adjustment-caliber metadata are meaningful safeguards. The
registry distinguishes raw, ratio-adjusted, additive-adjusted, not-applicable, and
unknown data and warns about mixing incompatible calibers. Tushare fundamentals use
announcement/final-announcement dates, latest restatement per reporting period,
and a no-regression rule in
[`TushareFundamentalProvider`](../../agent/backtest/loaders/tushare_fundamentals.py#L118)
and [`enrich_price_frames_with_fundamentals`](../../agent/backtest/loaders/tushare_fundamentals.py#L311).
Subdaily enrichment is rejected by default because publication is only day-granular.
These protections are provider-specific, not a global proof of point-in-time safety.

Timestamp semantics are mostly DatetimeIndex/loader conventions rather than one
typed global event-time/availability-time contract. Signal alignment uses each
instrument's own calendar, which is sensible for mixed calendars but needs explicit
Quorum metadata. Generic backtests take explicit codes; factor benches can use
named universe membership and strict bench filters by date where membership is
available. There is no mandatory point-in-time universe/delisting contract across
all runs, leaving survivorship bias as a caller/data responsibility.

### Experiment tracking and reproducibility

The existing pieces are complementary:

- [`Hypothesis`](../../agent/src/hypotheses/registry.py#L83) records title, thesis,
  status, universe, signal definition, data sources, skills, linked run cards, and
  invalidation notes. [`HypothesisRegistry`](../../agent/src/hypotheses/registry.py#L145)
  stores file-backed JSON under the Vibe home. Records can be updated in place,
  which is convenient for workflow but insufficient as an immutable trial ledger.
- [`Artifact`](../../agent/src/strategy_store/models.py#L81) and
  [`BenchResult`](../../agent/src/strategy_store/models.py#L138) provide SQLite-backed
  strategy/factor identity, derivation, lifecycle, model governance, and train/test
  windows.
- [`write_run_card`](../../agent/backtest/run_card.py#L44) records config and
  strategy hashes, source names, metrics, validation, and content-hashed artifact
  files. Its summary does not capture Git commit, environment lock, dataset hashes,
  split manifest, random seeds, or total attempted trial family; the full config is
  hashed rather than embedded.
- [`RunManifest`](../../agent/src/governance/manifest.py#L219) hashes system prompt,
  skills, tools, packages, and extra agent context. That is an agent-governance
  manifest, not a market-data/model experiment passport.

Quorum should introduce one append-only `ExperimentRecord` that references the
existing hypothesis/artifact/run IDs and adds the missing scientific protocol. It
should not migrate or replace the three mature stores during V0.

#### Task 2 experiment ledger

The experiment spine makes scientific identity distinct from execution history:

- `ExperimentSpec` freezes protocol, expert/model configuration, optional ensemble,
  target and horizon, data snapshot/cutoff, universe, costs, code/config, seed, and
  trial-family identities. Its full `sha256:` fingerprint is computed from the
  `quorum-experiment-spec-v1` scientific-identity namespace plus a canonical
  representation of those scientific fields. That namespace changes only if the
  meaning of “same scientific experiment” changes; the persistence contract name
  and schema version are deliberately excluded. Timestamp identity uses the
  represented instant; registration wall time is also excluded.
  Each persisted `ExperimentSpec` records both its serialization schema version and
  its scientific identity namespace. The schema version describes how the record is
  encoded; the identity namespace defines the semantics of the scientific
  fingerprint. Historical specs retain their original identity namespace and are
  never reinterpreted using the current default.
  The current `ExperimentSpec` persistence contract is schema version 2, while its
  scientific identity namespace remains `quorum-experiment-spec-v1`. Persistence
  schemas may evolve independently of scientific equivalence. Development-time
  schema-v1 `ExperimentSpec` payloads are not implicitly migrated because their
  historical fingerprint semantics are ambiguous.
- `ExperimentAttempt` is a separately identified registration (`exp_` plus UUID4),
  so repeated attempts of the same specification remain separately countable.
  Optional immutable parent-attempt lineage records changed specifications without
  rewriting their parents.
- `ExperimentEvent` appends lifecycle evidence. The legal paths are
  `REGISTERED -> RUNNING -> {COMPLETED, FAILED, INTERRUPTED, REJECTED}` and direct
  `REGISTERED -> {FAILED, INTERRUPTED, REJECTED}` for setup-time outcomes. Terminal
  states cannot reopen or receive a replacement result.
- `ExperimentLedger` stores registration and event contracts in a hash-chained,
  fsynced JSONL file. The existing standard-library-only governance ledger provides
  retained-chain tamper evidence and atomic append locking; a Quorum ledger-scoped
  lock covers the complete read/validate/append transaction for cooperating
  processes and ledger instances. Waiting for that lock raises
  `ExperimentLedgerLockTimeout` after 120 seconds on every platform rather than
  failing silently or hanging. The application API only appends. Within the
  retained chain, modification, interior deletion/reordering/insertion, malformed
  records, partial writes, hash/sequence discontinuities, and chronologically
  inconsistent history fail closed. A clean rollback of complete trailing records
  leaves a valid prefix and cannot be proven from the local forward chain alone;
  that requires an external trusted checkpoint or monotonic anchor, which Task 2
  does not provide.
- `ExternalRecordRefs` stores only opaque Hypothesis Registry IDs, Strategy Store
  artifact IDs, stable run-card references, and Quorum artifact IDs. It never copies
  those external records or makes the Quorum core load their databases. Small frozen
  terminal metadata is bounded; large metrics and artifacts remain externally
  referenced.

Trial-family queries count every registered attempt regardless of whether it later
completed, failed, was interrupted, was rejected, or produced no usable result.
Split materialization and out-of-fold enforcement are owned by the Task 3
coordinator below.

#### Task 3 chronological validation coordinator

Task 3 consumes a registered `ExperimentAttempt`, its authoritative
`EvaluationProtocol`, an explicitly ordered sequence of `TimeInterval` bars, and
positional forward-label end metadata. The bar axis must already be strictly
chronological and non-overlapping by represented instant. The coordinator does not
sort it, assume equal duration, fill market gaps, or read OHLCV values. Consecutive
positions are coalesced in a manifest only when their temporal boundaries genuinely
touch, so weekends and other gaps remain visible.

Each ordinary fold orders validation immediately before test and advances the next
evaluation origin by `step_bars`. Evaluation blocks may not overlap. Expanding mode
contains every eligible historical row; rolling mode contains exactly the most
recent eligible window capped by `train_window_bars`. Accepted origins follow the
declared step schedule. A candidate is skipped only when deterministic purge rules
leave it below `minimum_train_bars` or its evaluation label reaches the final
holdout, and materialization fails closed if no valid fold remains.

The `purge_bars` positions immediately before evaluation are explicitly recorded.
Training labels use closed endpoints: any historical row whose label ends at or
after the first held-out validation position is additionally purged. The complete
validation-plus-test block is then audited through
`src.quantlib.crossvalidation.detect_boundary_leakage`; a dirty report raises and
cannot become a `MaterializedFold`. Validation is reserved for later fitting such
as calibration, so it must not see test outcomes: validation ends at the first
declared validation row whose label reaches the test block, and that row and the
rest of the declared validation span are purged. A fold whose validation span is
purged entirely fails closed. Declared embargo positions are recorded
immediately after test wherever they exist before the final-holdout boundary; they
are not future training rows for that fold.

Ordinary validation and test labels must also resolve entirely before the first
final-holdout bar. A candidate fold containing an evaluation label whose positional
end reaches the locked holdout is skipped without clipping or shortening the
evaluation block. Materialization fails closed if no candidate survives. Positional
label ends must identify actual bars in the supplied timeline; out-of-range ends are
rejected rather than clamped.

Ordinary chronological planning receives the full bar axis but only pre-holdout
label metadata. Label metadata for the locked final holdout is neither required nor
accepted by the ordinary coordinator. The required label sequence therefore has
exactly `ordinary_stop` entries, which physically prevents ordinary fold derivation
and split identity from depending on holdout outcomes.

A `ChronologicalEvaluationPlan` is not merely an internally consistent collection
of folds. It is the canonical deterministic realization of a registered
`ExperimentAttempt`, its `EvaluationProtocol`, the ordinary chronological bar axis,
and declared label spans. The plan retains the immutable attempt and derives its
attempt ID and scientific fingerprint from that registration rather than accepting
independent provenance strings.

One internal derivation produces the complete accepted fold sequence, including
canonical expanding/rolling training positions, scheduled origins, deterministic
skips, purge/embargo positions, and split IDs. Public plan construction reruns that
same derivation and requires every supplied fold to match exactly. Arbitrary fold
omission, reordering, shifted origins, or clean training-row cherry-picking cannot
form an accepted plan.

Accepted folds retain positional tuples for runtime slicing and use the existing
`SplitManifest` as persisted temporal truth. Split IDs are full deterministic
SHA-256 identities over the experiment specification fingerprint, protocol, actual
pre-holdout timeline instants, pre-holdout label spans, and fold positions. They are
recomputed during plan validation rather than trusted from a supplied manifest.
Metadata belonging only to the locked holdout is excluded from ordinary split
identity. `MaterializedFold` retains the immutable bar axis and ordinary label
metadata needed to prove that every positional set maps exactly to its corresponding
manifest intervals. Its leakage audit is recomputed internally and cannot be
supplied by a caller. Canonical folds mechanically produce immutable `OOFSlot`
values with distinct validation/test roles; the plan rejects missing, duplicate,
unknown, or otherwise noncanonical assignments.

Normal generation stops before the locked final holdout. Bars that straddle either
holdout boundary are rejected rather than clipped, and Task 3 neither opens nor
evaluates the holdout. It creates authorized OOF slots, not expert scores or fitted
predictions. The existing `backtest.validation.walk_forward_analysis` remains
retrospective equity-curve subperiod reporting and is not used as Quorum's
fit/predict coordinator.

#### Task 4 deterministic V0 experts

Task 4 adds three close-only, parameter-frozen, single-asset experts. Each call
receives exactly one coordinator-approved historical slice containing `asset`,
`close`, `event_at`, and `available_at`. The shared boundary requires strictly
increasing event instants, rejects every row available after `decision_at`, binds
the terminal event and latest availability to `PredictionContext`, and never sorts,
drops, fills, or interpolates observations. Task 3 remains responsible for deciding
which rows the expert may see; experts neither fetch data nor choose an evaluation
period.

The V0 formulas and identities are frozen:

- Momentum (`quorum.momentum`, `v0.1.0`, `quorum:momentum:v0`) uses the latest
  21 closes for the 20-bar simple return and emits
  `clamp((close[t] / close[t-20] - 1) / 0.10, -1, 1)`. The existing
  `qlib158_roc20` factor was audited but not reused: its `safe_div` epsilon changes
  the numerical definition for sufficiently small valid positive prices. Keeping
  exact ordinary division locally is an intentional exception where scientific
  semantics take priority over superficial formula similarity.
- Trend (`quorum.trend`, `v0.1.0`, `quorum:trend:v0`) uses arithmetic SMA10 and
  SMA50 and emits `clamp((SMA10 / SMA50 - 1) / 0.05, -1, 1)`.
- Mean Reversion (`quorum.mean_reversion`, `v0.1.0`,
  `quorum:mean-reversion:v0`) uses the latest 20 prices, their sample standard
  deviation (`ddof=1`), and emits `clamp(-price_zscore / 3, -1, 1)`.

These are asset-isolated time-series formulas: there is no training, optimization,
cross-sectional normalization, or dependency on another asset. Insufficient
warmup, a missing close in the active window, or an undefined mean-reversion
standard deviation produces no prediction rather than neutral evidence. Missing
closes outside the active window do not affect the formula. V0 scores make no
probability or confidence claim. Every prediction copies the authoritative
`event_at`, `available_at`, `decision_at`, horizon, and experiment/split references
from its context; experts do not write the experiment ledger, ensemble evidence,
size positions, or execute trades.
Their public identities and formula parameters are class-readable but guarded
against runtime reassignment or deletion, and instances have no mutable state or
constructor tuning knobs.

#### Task 5 static ensemble

Task 5 combines standardized predictions using an immutable, caller-supplied
`StaticEnsembleConfig`; it defines no default weights or thresholds. Predictions
align only when `asset`, event instant, decision instant, `horizon_bars`,
`experiment_id`, and `split_id` match. For a complete row, the combined evidence is
the numerically stable fixed sum `fsum(weight_i * score_i)`. The ensemble
availability is the latest contributing availability instant, with canonical
configuration order breaking equivalent-instant offset ties.

Every decision retains one attribution entry per configured expert in canonical
configuration order. If any configured expert is absent, the row explicitly has no
combined score, reporting label, or disagreement; remaining weights are neither
renormalized nor treated as zero evidence. Complete rows report weighted population
dispersion `sqrt(fsum(weight_i * (score_i - combined_score) ** 2))` and descriptive
threshold labels with `score <= sell` as `SELL`, `score >= buy` as `BUY`, and the
interior as `HOLD`. These labels are not orders or position instructions.

Grouping, timestamp selection, attribution, output ordering, and versioned JSON are
deterministic and invariant to prediction input order. Unexpected expert IDs and
configured-version mismatches fail closed. The ensemble reads no data or evaluation
state and produces evidence only; conversion to constrained exposure belongs
exclusively to the future Task 6 risk layer.

`StaticEnsembleResult` is the authoritative serialized Task 5 artifact: its included
configuration supplies the thresholds and expert specification needed to validate
every embedded decision during reconstruction. Decision construction recomputes the
combined score and disagreement from attribution rather than trusting supplied
values, while result validation additionally derives the reporting label from the
configured thresholds. Standalone decisions remain immutable public row values and
may be serialized for embedding, but cannot be independently deserialized because a
row alone cannot establish label correctness without its configuration.

#### Task 6 risk boundary and Vibe adapter

Task 6 maps each complete continuous ensemble score to a proposed target with
`score * max_abs_weight_per_asset`. Simultaneous decisions are grouped by decision
instant within isolated `(experiment_id, split_id, horizon_bars)` streams. The pure
risk policy proportionally scales the whole proposed vector when absolute gross
exposure exceeds its zero-leverage cap, then proportionally interpolates from the
stream's previous final target when requested L1 turnover exceeds its cap. It never
optimizes, redistributes clipped exposure, or forces full investment. A stream's
asset universe is fixed by its first group; changed membership or duplicate asset
rows fail closed.

If any asset has incomplete ensemble evidence, the entire portfolio group is marked
non-executable, every target stage is `None`, and turnover state is not advanced.
Immutable audit records keep ensemble score, proposed target, gross-constrained
target, final requested target, both scale factors, requested turnover, and final
gross distinct. `RiskPolicyResult` is the authoritative versioned serialization
boundary and replays the policy against its included configuration when rebuilt.
These remain requested exposures: Vibe's unchanged `BaseEngine` still determines
orders, fills, and actual positions under capital, lot, fee, and market rules.

`VibeSignalAdapter` is the only Quorum core layer aware of pandas/Vibe signal shape.
One adapter instance executes exactly one risk stream. A sole stream is inferred;
multiple `(experiment_id, split_id, horizon_bars)` streams require an explicit
selector and are never stitched into one execution history. Task 7 orchestrates
those independent OOF streams as separate engine runs.

Vibe loaders may provide timezone-naive market indexes while Quorum timestamps
remain absolute and aware. Naive asset calendars therefore require an explicit IANA
timezone declaration; the adapter interprets their wall times transiently with
`zoneinfo`, rejects ambiguous or nonexistent DST times, and leaves the original
DataFrame index unchanged. Aware indexes already define their instants and reject a
redundant timezone declaration.

The adapter validates each `RiskRebalance` before flattening its targets. Every
`event_at` must match one asset-calendar row, and all targets in the rebalance must
resolve to the same next execution row. Rows label bar ends and the engine fills
at the next row's open, which is that bar's start, so `decision_at` must not be
later than the next open. Equality is accepted and means zero decision-to-order
latency: a decision at the close fills at a next open occurring at the same
instant. The optional per-asset `bar_starts` supplies each row's observed bar
start (for example `MarketBar.start_at` from a snapshot), which makes the next
open authoritative rather than inferred. Without it, the next bar may open at
this row's close, so `decision_at` may not be later than `event_at`. Asynchronous portfolio transitions fail closed in V0 rather than
creating an unaudited intermediate portfolio. Accepted final targets are written on
their event rows, carried forward, and left at zero before the first target;
incomplete selected-stream groups and duplicate slots are rejected. V0 execution
requires `position_adjustment="rebalance"`, leverage one, no optimizer or
constraints, no rebalance mask, and zero tolerance. Reporting label `HOLD` remains
unrelated to the engine's rejected legacy `"hold"` mode.

#### Task 7 offline acceptance workflow

Task 7 exposes `run_v0_acceptance(output_dir)` and the thin
`python -m src.quorum.acceptance --output-dir <path>` command. Its versioned,
deterministic in-process OHLCV fixture requires no provider, configuration file,
credentials, broker, LLM, or network. The real experiment ledger registers the
attempt before evaluation and records `REGISTERED -> RUNNING -> COMPLETED` with
fixed fixture lifecycle metadata. The real Task 3 plan remains authoritative:
the final holdout stays locked, every validation/test assignment comes from its
OOF slots, and the no-fit V0 experts receive only a prepared-data prefix ending
at the assigned observation.

Each fold's test predictions pass through the frozen experts, static ensemble,
risk policy, and Vibe adapter, then execute in an independent
`GlobalEquityEngine` run from the first test event through the one required next
bar. Folds and their risk state are never stitched. The same requested targets
run once with frozen US slippage and once without it, proving the existing cost
path while retaining requested targets, actual positions, and fills as distinct
evidence. Existing engine metrics, seeded bootstrap validation, artifacts, and
run cards are reused unchanged; `artifacts/quorum_protocol.json` adds the split,
expert, risk-stream, holdout, and theoretical target-causality declarations and
is discovered by the normal run-card artifact scan. Acceptance separately audits
immutable `fills.jsonl` evidence against the shifted `target_positions.csv`
request stream and `positions.csv` execution truth. At every timestamp it
classifies the exact requested transition (open, flip, close, resize, unchanged,
or flat), verifies compatible `signal`/`target_rebalance` fills, and reconciles
the cumulative signed-fill ledger to the persisted actual position. A fill moved
to another otherwise-authorized bar therefore fails. The engine's
`end_of_backtest` liquidation is exempt from Quorum decision attribution only
when it occurs on the final execution-frame timestamp, uses a close fill, and
closes the remaining position. A theoretical causal row alone cannot make the
actual-execution control pass.

The fixture does not require one fill for every changed target row. The existing
rebalance engine can legitimately make no trade when lot rounding or the
cost-dependent execution-price band leaves the requested size unchanged. The
fail-closed invariant is therefore authorization of every actual decision fill,
not a fabricated one-to-one target/fill correspondence.

Task 7 fold directories are scientific subrun evidence nested beneath the
authoritative acceptance run, not top-level Vibe jobs. They intentionally omit
`state.json`, so the existing Vibe run consumer reports their status as
`unknown` while still loading their metrics, equity, trades, actual and target
positions, validation, and run card. The experiment ledger and top-level
acceptance report remain the lifecycle authorities; duplicating a second subrun
lifecycle would create conflicting status semantics.

`quorum_acceptance.json` is the authoritative control report, accompanied by a
human-readable Markdown report. Its canonical scientific fingerprint covers the
fixture and dataset snapshot, experiment specification, protocol/manifests, OOF
assignments and predictions, ensemble/risk results, cost configurations, stable
metrics, execution evidence, artifact hashes, and normalized run-card content.
Raw run-card bytes are not the identity because `generated_at` and `run_dir` are
legitimate runtime fields; only those fields are removed from the normalized
card. Acceptance `PASS` means the declared causal, reproducibility, artifact, and
cost controls passed, not that the strategy is profitable or recommended.

#### Task 8 content-addressed market-bar snapshots

Task 8 adds immutable scientific persistence for normalized market bars under
`src.quorum.data`. It accepts already-loaded `MarketBar` records and deliberately
does not fetch, discover, or fall back between providers. Each record carries an
asset, a half-open `[start_at, event_at)` bar interval, an explicit
`available_at`, and finite OHLCV values. All timestamps are aware, temporal
ordering uses actual instants, and canonical row ordering is `(asset, event_at
instant)`. No universal order is imposed between `event_at` and `available_at`;
prediction consumers remain responsible for enforcing `available_at <=
decision_at`.

Identity has two explicit layers. `content_sha256` hashes canonical JSON covering
the `quorum-market-bar-content-v1` namespace, schema version, dataset kind,
interval declaration, timestamp convention, and every canonical market-bar row.
Timestamp identity is normalized to UTC. `snapshot_id` separately hashes the
`quorum-dataset-snapshot-v1` namespace, schema version, `content_sha256`, the same
semantic declarations, derived asset/count summaries, and canonical
`DatasetProvenance`. Consequently, identical scientific bars acquired under
different source, revision, adjustment, retrieval, or metadata provenance share
content identity but receive distinct complete-snapshot identities. Both external
identities use `sha256:<64 lowercase hex>` and are always derived by the store.

`DatasetSnapshotStore` publishes `<root>/sha256-<digest>/manifest.json` and
`bars.jsonl`. Both files are compact, sorted-key UTF-8 JSON with a terminal
newline; bar records are one JSON object per line. Creation writes and fsyncs a
same-filesystem staging directory, then atomically renames the complete directory
under a cross-process lock. Existing evidence is never overwritten, exact repeated
creation is idempotent, and an interrupted staging directory is not a valid
snapshot path.

Every `load()` and `verify()` operation checks the requested ID, strict manifest
and row schemas, canonical bytes and row order, row count, asset declaration,
market-bar invariants, `content_sha256`, and recomputed `snapshot_id` before
returning trusted objects. Missing, partial, noncanonical, corrupt, or tampered
evidence fails closed; replay never invokes a provider. The resulting
`snapshot_id` is used directly as `ExperimentSpec.data_snapshot_id`, while later
coordination remains responsible for transforming verified bars into Task 3
intervals and prediction inputs.

This is intentionally distinct from the Vibe loader cache. The loader cache is a
best-effort performance facility keyed by request/loading identity and may refetch
on a miss or corrupt entry. A Quorum snapshot is append-only, content-addressed,
self-verifying scientific evidence; a missing snapshot is an error. Task 8 adds no
CSV/Parquet/DuckDB ingestion orchestration, provider integration, experiment
runner, expert registry, fitting, portfolio logic, API/UI surface, or execution
capability.

#### Task 9 real-data research runner and prediction evaluation

`src.quorum.research` is the first path from real market data to out-of-fold
evidence. `ingest` fetches daily bars once through a Vibe loader (yfinance by
default) and publishes them as a Task 8 snapshot. Each trade-date row becomes a
regular-session bar `[09:30, 16:00) America/New_York`, available at its close.
Half-days are stamped 16:00, which only delays availability. Missing values fail
closed; bars that had not closed at retrieval time are refused.

`run` loads and verifies a snapshot and registers an `ExperimentAttempt` in the
workspace ledger before any result exists. It then materializes an expanding
locked-holdout plan: `horizon_bars` purge and embargo, with validation labels
purged away from test. Every authorized validation and test slot is predicted by
the frozen V0 experts on a trailing 60-bar window ending at that slot, and those
predictions are combined by the equal-weight static ensemble. The target is
tradable: it enters at the next bar's open (the engine's fill) and exits at the
close `horizon_bars` bars after the decision bar. All assets must share one exact
calendar; any difference fails closed because missing-asset and dynamic-universe
semantics are not yet decided.

`src.quorum.evaluation` scores predictions separately from P&L. For each role and
predictor it reports:

- coverage against authorized slots;
- pooled Spearman IC;
- per-fold IC mean, standard deviation, t-statistic, and positive fraction;
- directional hit rate;
- score spread and saturation at ±1;
- pairwise score rank correlation and per-asset IC.

Pooled IC makes no significance claim because forward horizons overlap within a
fold. Reports, OOF prediction rows, and the ledger outcome live under
`<workspace>/runs/<attempt_id>/`. The metrics fingerprint is identical across
repeated attempts on the same snapshot and configuration, while every attempt
still counts toward the trial family.

Ingesting real data exposed an upstream defect: `validate_ohlc` dropped real
trading days whose adjusted close sat one rounding error outside the high or low
(about 1e-16 relative, from yfinance `auto_adjust`). Those excursions within
1e-9 relative are now snapped back onto the high or low, and genuine violations
are still dropped.

#### Task 10 per-fold ridge stacker

`src.quorum.stacking.RidgeStacker` is the first model with fitted parameters. It
regresses the tradable forward return on the three standardized expert scores.
The penalty `alpha * n` is declared as `ResearchConfig.stacker_alpha`, so it is
never tuned on evaluation rows, and the intercept is unpenalized. Scores are
`tanh` of the fitted deviation from the training-mean return, so zero means no
view beyond drift, and the transform is monotone.

The research runner refits the stacker for every fold on that fold's purged
training positions, pooled across assets. It then predicts the fold's validation
and test slots as predictor `stacker.ridge`. Each fold's model card records its
coefficients and proves `train_label_end_max < evaluation_start`. Training on
expert scores at training positions is valid only because the V0 experts have
no fitted parameters; experts that learn must feed the stacker their
out-of-fold rows instead. Expert scores are computed only for pre-holdout
positions.

### Portfolio and risk

Base execution normalizes requested gross weight, enforces cash/margin/lot/market
rules, and supports optional optimizers/constraints. Individual engines add
leverage, liquidation, funding, or product margin as appropriate. The resulting
book is real accounting, not merely vectorized returns.

[`risk_xray.py`](../../agent/backtest/risk_xray.py) computes trailing concentration,
volatility, drawdown, historical VaR/ES, and correlation/diversification for a
long-only portfolio. It is descriptive; it does not govern a backtest. The live
[`HardCaps`](../../agent/src/live/mandate/model.py#L47),
[`Mandate`](../../agent/src/live/mandate/model.py#L123), and
[`check_mandate`](../../agent/src/live/enforcement.py#L458) are strong execution
safety concepts but belong to the live runtime and should not be imported into
Quorum research core.

There is no single general research risk policy for max gross/net exposure,
per-name exposure, correlated sleeves, drawdown actions, and confidence-aware
sizing. V0 should add a small deterministic risk layer after ensemble voting and
before the existing SignalEngine adapter. It should use zero leverage and fixed
caps; drawdown-dependent or confidence-dependent sizing should wait until its own
OOF evaluation.

### Agents and skills

[`AgentLoop`](../../agent/src/agent/loop.py#L1071) is the ReAct-style orchestration
loop. [`SkillsLoader`](../../agent/src/agent/skills.py#L100) progressively exposes
bundled Markdown skills and user overrides. [`ToolRegistry`](../../agent/src/agent/tools.py#L68)
and [`build_registry`](../../agent/src/tools/__init__.py#L76) expose deterministic
Python tools such as [`BacktestTool`](../../agent/src/tools/backtest_tool.py#L115)
and `FactorAnalysisTool` to the LLM. The MCP server mirrors many of the same
capabilities.

Quorum's scientific invariants belong in `agent/src/quorum/` as ordinary tested
Python. After the deterministic API works, a thin tool and a Quorum skill may
orchestrate it. An LLM should propose a hypothesis/configuration and explain results;
it should not be the component that enforces chronology, locks holdouts, counts
trials, calculates weights, or decides whether a run is valid.

### Execution, paper, and live trading

[`trading/service.py`](../../agent/src/trading/service.py#L685) fronts connector
profiles and order placement. `agent/src/live/` adds mandate checks, hard caps,
order gates, audit, scheduler/runtime, and kill-switch behavior. Several connector
profiles are explicitly paper-only/read-only, but they still represent external
systems and operational state. The strategy-discovery “shadow” path converts
records into historical evidence; it is not a general prospective simulator.

Therefore:

- V0 must have no dependency on `src.trading`, `src.live`, broker SDKs, accounts, or
  secrets.
- Quorum 0.9 should first record prospective predictions and hypothetical fills
  against observed data in an immutable local ledger.
- Any later paper-broker adapter should be a capability-gated outer integration.
  Live-money support remains a separate project decision and review.

### API, CLI, and frontend

[`api_server.py`](../../agent/api_server.py#L163) constructs the FastAPI application
and includes route modules under `agent/src/api/`. [`cli/main.py`](../../agent/cli/main.py#L1455)
implements the `vibe-trading` entry, and [`mcp_server.py`](../../agent/mcp_server.py#L3036)
provides the MCP entry point.

The frontend is React/TypeScript with Vite, React Router, Zustand-style stores, and
a typed API client. [`router.tsx`](../../frontend/src/router.tsx#L54) exposes Agent,
Run Detail, Compare, Settings, Runtime, Scheduled, Reports, Portfolio, Correlation,
Alpha Zoo, and Options Lab views. Run Detail already renders equity/metrics,
trades/positions, attribution, factor research, validation, run card, and generated
code from artifacts. V0 should emit compatible run artifacts and add no new page;
Quorum-specific UI is justified only when its protocol/attribution data cannot be
expressed in the existing view.

### Test inventory and gaps

Relevant existing suites include:

| Area | Representative tests |
| --- | --- |
| Lookahead/leakage/data gaps | `agent/tests/factors/test_lookahead.py`, `test_execution_causality.py`, `test_price_limit_lookahead.py`, `test_rsshub_events_lookahead.py`, `test_get_market_data_provenance.py`, `test_ohlc_validation.py`, factor missing-bar and `pct_change(fill_method=None)` tests |
| Validation/statistics | `test_validation.py`, `test_validation_cli.py`, `quantlib/test_crossvalidation.py`, `quantlib/test_multipletesting.py`, `test_engine_robustness.py`, finite-metric tests |
| ML | `test_ml_strategy_skill.py`, which extracts and tests the Markdown template |
| Strategies/experiments | `test_strategy_store*.py`, `strategy_discovery/*`, `test_multi_factor_strategy_template.py`, strategy development manager and run-card tests |
| Factors | registry, purity/AST, golden, strict bench, core analysis, factor-cost, and API tests under `agent/tests/factors/` |
| Backtest/accounting | execution causality, engine robustness, rebalance evidence, options correctness, cross-market annualization, and market smoke tests |
| Costs | `test_factor_costs.py` plus market-engine-specific fee/slippage tests |
| Portfolio/risk | `quantlib/test_portfolio.py`, portfolio service/config/FX/routes/compatibility, and risk-X-ray tests |
| Execution safety | SDK order gate, mandate enforcement, kill switch, read-only defaults, capped-paper connector, and live API tests |
| Reproducibility | run-card hashing/artifacts, governance manifest/ledger tests, seeded validation tests |

The critical missing tests for Quorum are end-to-end proof that a final holdout
cannot influence tuning, that every meta-model training row is genuinely OOF, that
failed/discarded trials remain counted, that universe membership is point-in-time,
that a dataset snapshot is replayable, and that prospective paper evaluation cannot
reach a broker write.

## C. Reuse matrix

| Quorum requirement | Existing Vibe-Trading component | Decision | Reason |
| --- | --- | --- | --- |
| Strategy interface | `SignalEngine.generate` contract and BacktestTool validation | **Extend via adapter** | Preserve engine compatibility; keep Quorum experts independent from target-weight sizing. |
| Expert interface | No stable expert protocol | **New, narrow** | Define evidence/time/version semantics without replacing SignalEngine. |
| Market data | `DataLoaderProtocol`, registry/fallbacks, `fetch_market_data`, adjustment and PIT helpers | **Reuse + harden** | Broad provider coverage is mature; add availability-time normalization and dataset snapshot hashes. |
| Backtester | BaseEngine and market-specific engines | **Reuse** | Mature fills/accounting/cost/benchmark behavior; forking it would create divergent bugs. |
| Experiment tracking | Hypothesis Registry, Strategy Store, RunCard, governance manifest | **Extend/compose** | Reference existing identities; add append-only attempts, split/data/environment protocol. Do not replace stores. |
| Walk-forward validation | Generic subperiod analysis plus ML example | **New coordinator, reuse primitives** | Existing generic helper is not retraining/OOF walk-forward. |
| Purged CV/embargo | `src.quantlib.crossvalidation` | **Reuse + enforce** | Algorithms exist and are tested; Quorum must integrate and make boundary checks mandatory. |
| Transaction costs | Engine fees/slippage/funding and `factor_costs` | **Reuse + freeze/stress** | Good market-specific base; passports must lock assumptions and run plausible stress cases. |
| Portfolio management | Target weights, optimizers, constraints, accounting | **Reuse later** | Do not introduce optimization in V0; a static risk policy can feed existing targets. |
| Risk management | Engine feasibility, constraints, risk X-ray; live mandates | **Extend research side** | Add a pure research risk policy. Reuse concepts, not live-runtime coupling. |
| ML infrastructure | ML skill/template and sklearn dependencies | **Extend into framework later** | The example is useful but lacks lifecycle/OOF/persistence/calibration contracts. |
| Factor library | Alpha Zoo, registry, analysis, strict bench, ZooSignalEngine | **Reuse** | Mature, broad, and heavily tested. Wrap factors as experts; do not duplicate formulas. |
| Frontend | Run Detail/artifact rendering and API client | **Reuse** | Emit existing artifacts first; defer Quorum UI until a demonstrated gap. |
| Paper trading | External paper connector profiles and live gates | **New local ledger first; adapt later** | V0 needs no broker. A prospective research simulator requires deterministic, immutable evaluation semantics. |
| Regime layer | Correlation-regime helper and retrospective evidence labels | **New causal contract later** | Existing pieces do not represent a general trend/range/bull/bear model and must not be relabeled as one. |
| Multiple testing | DSR, FDR, CSCV PBO utilities | **Reuse + integrate** | Strong library code exists; missing piece is complete trial-family accounting and mandatory reporting. |

## D. Proposed Quorum architecture

### Domain contract

The native backtester contract argues against finalizing Quorum around only a
three-valued enum. Recommended logical prediction fields are:

- `expert_id` and immutable `expert_version`;
- `asset`;
- `event_at` (timestamp associated with the underlying observation/event),
  `available_at` (latest declared availability time among information used),
  `decision_at` (the prediction cutoff), and `horizon_bars`; the universal
  causality invariant is `available_at <= decision_at`, while no universal order
  is assumed between event and availability time; temporal ordering and identity
  compare actual instants, while serialization retains the supplied UTC offset;
- finite `score` in `[-1, 1]`, where sign is direction and magnitude is evidence;
- optional `probability_up` in `[0, 1]`, only when the value is a calibrated
  probability;
- optional `confidence` in `[0, 1]` with a declared definition, never an alias for
  absolute score;
- small immutable metadata/provenance and an experiment/split reference.

Use a batch `ExpertResult`/frame for performance while preserving the same validated
row semantics. Reporting may map the ensemble score through frozen thresholds to
`BUY/HOLD/SELL`; reporting thresholds must straddle zero so neutral evidence remains
inside the HOLD band. Execution receives the risk layer's numeric target weights.

An `ExpertProtocol` should accept prepared, read-only market inputs plus a prediction
context and return standardized predictions. It must not fetch data, choose folds,
size positions, access the final holdout, or write experiment state.

### Layers and dependency direction

```text
existing loaders / frozen local dataset
                  |
                  v
        validation coordinator --------> append-only experiment record
                  |                                 |
           fit/predict folds                        v
                  v                    existing Hypothesis / Artifact / RunCard
          independent experts
                  |
            OOF predictions
                  v
          static ensemble  ---> attribution/disagreement artifacts
                  |
                  v
        pure research risk policy
                  |
                  v
       Vibe SignalEngine adapter
                  |
                  v
        existing BaseEngine + market engine
                  |
                  v
      existing metrics/artifacts/API/frontend
```

Dependencies point inward toward contracts. Experts know contracts, not ensemble or
execution. Ensemble knows predictions, not market loaders. Risk knows combined
scores, not expert internals. The adapter is the only Quorum core component that
knows the Vibe SignalEngine shape. Validation orchestrates experts and stores OOF
results; the LLM/tool/API layers are outside and depend on the deterministic core.

### Integration decisions

- Use BaseEngine unchanged for V0. If a behavior is wrong, fix it upstream-style
  with a focused engine test rather than add a Quorum fork.
- Build split manifests before feature fitting. `detect_boundary_leakage` is a
  mandatory assertion, not an optional report.
- Freeze V0 expert parameters and ensemble thresholds. No grid search is required
  to prove the architecture.
- Convert ensemble score to exposure only in the risk layer. Record proposed target,
  constrained target, order, fill, and realized position as distinct artifacts.
- Extend run cards with a Quorum artifact reference or sidecar; avoid making generic
  Vibe consumers depend on Quorum.
- Keep the Research Passport as an aggregate/read model over existing records plus
  immutable Quorum experiments. Do not make it another editable source of truth.

## E. Proposed directory structure

Only the first two contract files and their tests should be created in the next
task. The rest show intended boundaries, not a request to scaffold empty modules.

```text
agent/
  src/
    quorum/                              # NEW, created incrementally
      __init__.py                        # NEW in 0.1: public contract exports only
      contracts.py                      # NEW in 0.1
      experiments.py                    # NEW after contract review
      experts/                          # NEW in V0: three deterministic experts
        momentum.py
        trend.py
        mean_reversion.py
      ensemble/
        static.py                       # IMPLEMENTED in Task 5
      validation/
        protocol.py                     # NEW coordinator; wraps quantlib splits
      risk/
        policy.py                       # IMPLEMENTED in Task 6: pure sizing/constraints
      adapters/
        vibe_signal.py                  # IMPLEMENTED in Task 6: SignalEngine boundary
    quantlib/crossvalidation.py         # EXISTING, REUSE; modify only for proven bug
    factors/                            # EXISTING, REUSE
    hypotheses/registry.py              # EXISTING, adapter/reference only
    strategy_store/                     # EXISTING, adapter/reference only
    tools/quorum_research_tool.py       # POSSIBLE LATER, thin orchestration only
  backtest/                             # EXISTING, REUSE; no Quorum engine fork
  tests/
    quorum/                             # NEW tests beside each introduced module
docs/quorum/                            # NEW Phase 0 design/audit documents
```

No new frontend route, API route, CLI command, MCP surface, skill, database, or live
module is needed to define and test the V0 core.

## F. Risk register

| Risk | Current evidence | Impact | Required control |
| --- | --- | --- | --- |
| “Walk-forward” overclaim | Generic helper slices realized equity; no refit/OOF | False assurance and optimized-history reporting | Name it subperiod analysis; add an actual fold coordinator and OOF ledger. |
| Feature/label leakage | Only template-level ML purge; provider availability varies | Inflated expert/meta results | Typed availability time, purged/embargoed splits, synthetic leakage traps, causal feature review. |
| Final-holdout contamination | No central read lock | Tuning against purported final evidence | Pre-register and physically gate holdout; record the one allowed opening. |
| Survivorship bias | Generic runs accept current explicit symbols | Inflated historical portfolios | Point-in-time membership/delistings artifact and fail/qualify reports when absent. |
| Data revision/reproducibility | Cache key lacks snapshot/version hash | Same config can produce different results | Content-addressed snapshots, retrieval/provider metadata, hashes in passport. |
| Multiple-testing undercount | Utilities exist; no complete attempt ledger | Selection bias/false discoveries | Register every trial before results; immutable family ID; DSR/FDR/PBO reporting. |
| Misinterpreted robustness | IID bootstrap and P&L-order permutation have narrow nulls | Stronger claims than tests support | Label assumptions, add block bootstrap and strategy-family controls later. |
| Cost optimization/optimism | Rich defaults but mutable config and mixed factor path | Fragile apparent alpha | Freeze cost ID, stress worse cases, report turnover/capacity/unfilled orders. |
| Signal/sizing coupling | Signal magnitude is target weight | Expert quality confounded with risk policy | Expert score → ensemble → pure risk → engine adapter boundaries. |
| Duplicate frameworks | Existing factors/backtests/stores/UI are extensive | Maintenance divergence | Adapters and references; require written justification before a replacement. |
| Tight LLM coupling | Strategy code/config can be generated through skills | Scientific rules become prompt conventions | Deterministic core + fail-closed tests; LLM only orchestrates/explains. |
| Nonstationarity/regime mining | Many possible regimes/weights | Small-sample overfit | Static baseline, predeclared causal regimes, minimum observations, shrinkage, OOF evaluation. |
| Expert dependence | “Independent” formulas may share price trend inputs | Diversification overstated | OOF residual/prediction correlation and contribution attribution; cap concentration. |
| Meta-model contamination | No shared OOF store | Stacker learns fitted predictions | Meta training accepts only ledger-marked OOF rows. |
| Performance bottleneck | Per-fold × expert × asset refits/fetches | Slow research encourages shortcuts | Fetch once, immutable cached features, batch result contract, profile before optimizing. |
| Mutable experiment records | Hypotheses update in place | Silent history rewrite | Append-only Quorum attempt/event ledger referencing existing records. |
| Engine ambiguity | Unknown market may fall back to crypto engine | Wrong rules/costs | V0 pins engine explicitly; later fail closed on ambiguous routing. |
| Live-path reachability | Broker/order stack is present | Accidental external side effect | No core dependency on trading/live; network-off acceptance tests; explicit capability gates. |
| Node compatibility drift | jsdom warning on Node 24.13.1 | CI/local inconsistency | Standardize Node 24.15+ or 22.22.2+ and pin CI. |

## Verification status

The exact contributor-recommended command shown above completed in 10m38s:

```text
14,691 passed, 166 skipped, 16 failed, 11 errors, 1,603 warnings
```

The result is **not a clean baseline**, although the very large passing surface and
the successful CLI/frontend checks establish that the project installs and runs.
The failures were rerun individually and classify as follows:

- 20 symlink-security/registry cases (nine failures plus eleven setup errors) could
  not create their fixture links because this Windows account lacks symlink
  privilege (`WinError 1314`). These tests need Developer Mode/elevated symlink
  permission or platform-aware fixture skipping; their protected behavior was not
  exercised here.
- Two background-task cancellation cases remained `cancelling` beyond the test's
  five-second Windows deadline. This is an unresolved process-tree cancellation
  baseline issue, not a Quorum change.
- One grounding artifact test writes UTF-8 but calls `Path.read_text()` without an
  encoding; Windows selected cp1252 and raised `UnicodeDecodeError`. This is a test
  portability defect.
- One OpenRouter header-isolation test lost an explicitly supplied `X-Case` header
  when an ambient header used different casing. Inspection shows explicit/ambient
  name exclusion is case-sensitive before case-insensitive HTTP header handling;
  treat this as a genuine existing provider-header regression.
- One KIS cache test asserts POSIX mode `0600`, which Windows `stat` does not expose
  with equivalent semantics.
- One concurrent `TaskStore.save_task` test hit Windows `PermissionError(13)` during
  atomic replacement from two store instances. Treat this as a genuine Windows
  concurrency gap.
- One WebSocket-channel test requests a Unix-domain socket and raises
  `NotImplementedError` on Windows; the test lacks a platform skip.

`python -m compileall -q agent/src agent/backtest agent/cli` passed. Frontend build
and all 652 frontend tests passed. A focused offline run of execution causality,
engine robustness, rebalance evidence, validation, and run-card tests passed 135
tests with one skip. Documentation link-target validation and `git diff --check`
passed. Credential/network E2E groups remain intentionally excluded; no live
market or brokerage behavior is claimed by this audit.
