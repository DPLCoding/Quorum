# Quorum research principles

Status: proposed project invariants from the Phase 0 audit (2026-09-23).

Quorum is a research system first. Its primary output is defensible evidence about
whether a signal survives out of sample, not a profitable-looking equity curve.
These rules apply to deterministic experts, machine-learning experts, ensembles,
and any future symbolic-regression layer.

## 1. Time and information

1. Every prediction must distinguish the underlying observation/event time
   (`event_at`), the latest declared availability time among information used
   (`available_at`), and the prediction cutoff (`decision_at`). Event time alone is
   not an information-time boundary, and event time need not equal or follow
   availability time for every source.
2. A prediction may use only information satisfying
   `available_at <= decision_at`. Execution occurs no earlier than the next
   executable market event specified by the experiment. This timestamp contract
   records the declared boundary; it does not by itself prove that a provider's
   publication or revision metadata is correct.
3. Time-series evaluation is chronological. Random splits are forbidden unless a
   documented scientific question genuinely makes samples exchangeable.
4. Forward-looking labels must be purged from adjacent training folds. An embargo
   is required when overlapping samples or delayed information can leak across the
   boundary.
5. Higher-level models train only on out-of-fold lower-level predictions. Fitted
   training predictions are never acceptable substitutes.
6. The final holdout is locked before model selection. It is opened once for the
   declared final evaluation; looking at it makes it part of model selection and
   requires a new holdout.

## 2. Experiments and versions

1. Every attempted experiment receives an immutable ID before its results are
   known. Failed, null, interrupted, and rejected trials remain in the trial count.
2. Changes to features, labels, parameters, thresholds, horizons, training windows,
   ensemble rules, costs, universe rules, or data snapshots create a new experiment
   version. Historical records are appended, not overwritten.
3. A run must record, at minimum: hypothesis and parent IDs; code and configuration
   hashes; data identifiers and hashes; universe construction; information-time
   rules; split manifest; random seeds; cost model; attempted-trial family; software
   environment; metrics; artifacts; and outcome status.
4. Re-running an identical frozen experiment should either reproduce the same
   artifacts or explain the nondeterminism in machine-readable metadata.
5. Research status and deployment status are separate. A strong backtest never
   implies permission to trade.

## 3. Selection and statistical claims

1. Selection criteria are declared before the selection run. In-sample return alone
   is never a sufficient criterion.
2. Report the number and family of tried alternatives. Apply an appropriate
   multiple-testing correction when making claims across many factors, experts, or
   parameterizations.
3. Report effect size and uncertainty, not only a p-value or Sharpe ratio. Include
   sample count, observation frequency, evaluation period, and annualization rule.
4. Deflated Sharpe Ratio and Probability of Backtest Overfitting are useful
   diagnostics, not certificates of validity. Their assumptions and inputs must be
   stated. White's Reality Check or a comparable dependence-aware procedure is a
   future requirement when selection is made over a large strategy family.
5. Bootstrap and Monte Carlo results must name the resampling null. IID resampling
   is not presented as dependence-robust; shuffling the same trade P&Ls is a path
   test, not evidence that the strategy beats random signals.
6. Parameter perturbation, start-date sensitivity, subperiod/regime performance,
   and cost stress are part of the evidence package, never after-the-fact decoration.

## 4. Market, universe, and costs

1. Universe membership must be point-in-time. Current constituents must not be
   projected backward. Delisted and unavailable instruments are represented rather
   than silently removed.
2. Corporate actions, price adjustment, currency, calendar, timezone, timestamp
   convention, missing-bar policy, and fundamental publication lag are explicit.
3. A data provider name is not a data version. Serious experiments must reference
   a reproducible snapshot or content hashes and retrieval metadata.
4. Transaction costs, spreads, slippage, borrow/funding, market impact, lot sizes,
   liquidity limits, and rejected/unfilled orders are declared before evaluation.
   Cost assumptions cannot be relaxed after seeing results without creating a new
   experiment.
5. Results are stress-tested under worse, plausible costs. Capacity and turnover are
   reported beside returns.

## 5. Architecture and responsibilities

1. Experts produce evidence, not positions. An expert emits a signed score and,
   only when justified, separately defined calibrated probability and confidence.
2. Ensemble combination is separate from risk and sizing. Risk policy maps a
   combined score to constrained target exposure; execution maps that target to
   feasible orders and fills.
3. An expert receives approved data through an interface; it does not fetch data,
   select its own validation period, or mutate the experiment record.
4. Scientific invariants live in deterministic Python code and tests. LLM prompts
   and skills may orchestrate those components but cannot be the sole enforcement
   mechanism.
5. Quorum extends Vibe-Trading through narrow adapters. It does not duplicate the
   Alpha Zoo, loaders, backtest engines, optimizers, run artifacts, or frontend
   unless an audited limitation requires an extension.
6. Live-money execution is outside V0. Prospective paper evaluation must remain
   broker-write-free by default and require explicit, separately reviewed authority
   before any future live integration.

## 6. Reporting

Every performance report must disclose:

- hypothesis and selection procedure;
- train, validation, and untouched holdout intervals;
- split/purge/embargo methodology;
- universe and data provenance;
- signal-to-fill timing;
- transaction-cost and execution assumptions;
- number of experiments and correction method;
- uncertainty and robustness checks;
- known limitations and any protocol deviations.

A missing required disclosure makes the run incomplete, not silently valid.
