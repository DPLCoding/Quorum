# Quorum expert round v1: frozen specifications

Status: **frozen 2026-09-26, before any v1 expert was implemented or evaluated.**
No v1 expert's performance had been observed when this document was written.
Any change after evaluation is a new expert version and a new registered
experiment. It is never an edit to this round.

## Purpose

The V0 experts (momentum, trend, mean reversion) all read only daily closes and
largely restate one another: pairwise score rank correlations reach ±0.7. This
round tests whether information sources those experts ignore carry predictive
information that is **orthogonal and incremental**. Beating buy-and-hold on its
own is not the goal.

The round has exactly four experts. None of them will be added to, removed,
re-parameterized, or re-signed during this round.

## Shared input and information-time contract

- Universe, data, and protocol are unchanged: the 8-asset US daily snapshot
  `sha256:67eee713…c2b8`, the V0 research protocol (5-bar target horizon, locked
  2025 holdout), and bars that are regular sessions `[09:30, 16:00)
  America/New_York`, available at their close.
- An expert receives only the trailing window ending at the decision bar. Every
  value in it, including that bar's open, high, low, and volume, is available by
  the decision bar's `available_at`. The next bar's open is never in the window.
  The shared normalizer rejects any row with `available_at > decision_at`.
- Prices must be finite and strictly positive and volume finite and
  non-negative. `None` or NaN marks a missing value. Inputs are never filled,
  dropped, or sorted.
- Scores lie in `[-1, 1]`: positive is bullish and negative bearish. No
  probability or confidence is claimed. When the expert declines to predict, it
  emits no prediction; it never emits a neutral 0 in that case.
- Prices are provider-adjusted and volume is split-adjusted. Volume continuity
  was checked on AAPL across its 2:1 (2005), 7:1 (2014), and 4:1 (2020) splits;
  the 20-day median ratio stayed between 0.53 and 1.01, with no split-sized
  jumps.

## 1. Volatility regime: `quorum.volatility_regime` v1.0.0

- **Hypothesis.** When short-term realized volatility spikes above its own
  long-run level, subsequent risk-adjusted returns are poor (volatility-timing
  literature). The signal is volatility *level* only, never return direction.
- **Inputs.** Closes only. Log returns are `r_t = ln(close_t / close_{t-1})`.
- **Definition.**
  - Short-run volatility: `σ_s = sqrt(mean(r²))` over the latest 20 returns.
  - Long-run volatility: `σ_l = sqrt(mean(r²))` over the latest 252 returns.
  - Both use uncentered root-mean-square, so the mean return, and therefore the
    direction, never enters.
  - `raw = ln(σ_s / σ_l)`.
- **Mapping.** `score = clamp(-raw / 0.5, -1, 1)`: bearish when volatility is
  elevated, bullish when it is calm.
- **Direction-freedom.** Negating every log return leaves the score unchanged.
  The tests enforce this.
- **Warm-up.** 253 closes. A missing close among them means no prediction.
- **Degenerate case.** If `σ_s = 0` or `σ_l = 0`, the logarithm is undefined,
  so there is no prediction.

## 2. Overnight momentum: `quorum.overnight_momentum` v1.0.0

- **Hypothesis.** Returns earned overnight persist: close-to-open flows are
  autocorrelated, as in the overnight-return literature.
- **Inputs.** Opens and closes. For each bar `t` in the window:
  - overnight return `o_t = ln(open_t / close_{t-1})`, from the previous close
    to this bar's open;
  - intraday return `i_t = ln(close_t / open_t)`, from open to close. This is
    **defined only to state the separation; it is not used.**
- **Definition.** `raw = Σ o_t` over the latest 20 bars, which spans 20
  overnight gaps.
- **Mapping.** `score = clamp(raw / 0.05, -1, 1)`.
- **Information time.** The latest open used is the decision bar's own open,
  available at that bar's close. No later open can enter.
- **Warm-up.** 21 bars: 20 opens and 21 closes. A missing value means no
  prediction.

## 3. Volume confirmation, as abnormal volume: `quorum.abnormal_volume` v1.0.0

- **Hypothesis.** The high-volume return premium: a stock whose trading
  activity is abnormally high relative to its own recent norm attracts
  attention and subsequently outperforms, and abnormally low activity
  underperforms. The expert reads **volume only and no price direction**, so it
  cannot collapse into momentum multiplied by volume.
- **Inputs.** Volume only.
- **Definition.**
  - `recent = mean(ln volume)` over the latest 5 bars.
  - `base_mean` and `base_sd` are the mean and sample standard deviation
    (ddof = 1) of `ln volume` over the 60 bars immediately before those 5.
  - `z = (recent - base_mean) / base_sd`.
- **Mapping.** `score = clamp(z / 2, -1, 1)`.
- **Warm-up.** 65 bars.
- **Missing or degenerate.** Any missing or zero volume in the 65 bars (the log
  is undefined) means no prediction. `base_sd = 0` also means no prediction.
- **Pre-declared check.** The score's rank correlation with `quorum.trend` and
  `quorum.momentum` is reported. By construction, its inputs contain no price.

## 4. Range/intraday reversal: `quorum.range_reversal` v1.0.0

- **Hypothesis.** Closes that finish near the day's high over-extend and
  partially reverse over the following days, and closes near the low
  under-extend: short-horizon intraday overreaction.
- **Inputs.** High, low, and close.
- **Definition.**
  - The close-location value is `CLV_t = (2·close_t - high_t - low_t) / (high_t
    - low_t)`, which lies in `[-1, 1]`: 1 means the close is at the high and -1
    at the low.
  - **Zero-range day** (`high = low`): the close is both the high and the low,
    so `CLV_t = 0` by definition.
  - `m = mean(CLV_t)` over the latest 5 bars.
- **Mapping.** `score = clamp(-m, -1, 1)`, so a reversal from closes near the
  high is bearish.
- **Warm-up.** 5 bars. A missing high, low, or close means no prediction.

## Pre-registered evaluation (one experiment)

The whole round is **one attempt**: `expert_set = "v1"` with
`portfolio_construction = "long_only_tilt"` and daily rebalancing, run on the
snapshot above. It is registered through `preregister` before it runs.

**Predictors.**

- The seven experts individually.
- Two equal-weight static ensembles: `ensemble` (the V0 three) and
  `ensemble.all` (all seven).
- Ridge stackers:
  - `stacker.ridge` on the V0 three;
  - `stacker.v0+<expert>`, the V0 three plus one new expert, for each new
    expert;
  - `stacker.all` on all seven.

**Reported regardless of outcome.**

- Standalone metrics for every predictor, test and validation: coverage, pooled
  IC, per-fold IC mean/t/positive fraction, hit rate with the up-rate base, and
  saturation.
- The score rank-correlation matrix and the pairwise sign-agreement matrix.
- IC by asset, and IC by volatility regime. The regime is causal: an asset is
  in "high" volatility when its 20-day uncentered realized volatility at the
  decision bar exceeds the median of that series over the previous 252 bars.
- Incremental information: the per-fold test IC of `stacker.v0+X` minus that of
  `stacker.ridge`, with the paired mean, t-statistic, and positive fraction.
  Also `stacker.all` against `stacker.ridge`.
- The stitched long-only tilt portfolio for every predictor at the three cost
  levels, against the equal-weight benchmark.

**Declared criteria, fixed now.**

- *Orthogonal:* the expert's maximum absolute test-score rank correlation with
  any V0 expert is ≤ 0.5.
- *Adds incremental information:* the paired per-fold test ΔIC of
  `stacker.v0+X` over `stacker.ridge` is positive with t ≥ 2.5. That is roughly
  Bonferroni for four comparisons at 5%, treating folds as independent.
- The portfolio Sharpe difference is reported and not used as a criterion,
  because it is too noisy over this span.

An expert that fails these criteria is reported as failing. This round tunes
nothing.

---

## Results (appended after evaluation; the specification above is unchanged)

Attempt `exp_1f50dc53002540eb911e9d7db238c351`, run 2026-09-26. It was the
single registered attempt of family `experts-v1`, and the trial family then
held 12 attempts. The V0 predictors reproduced earlier runs exactly.

**Pre-registered criteria.** No v1 expert met the incremental-information
criterion.

| Expert | Max abs corr with V0 | Orthogonal | Stacker IC gain (paired t) | Adds information |
| --- | --- | --- | --- | --- |
| volatility regime | 0.41 | yes | −0.013 (−1.05) | no |
| overnight momentum | 0.49 | yes | −0.005 (−0.95) | no |
| abnormal volume | 0.26 | yes | −0.018 (−1.59) | no |
| range reversal | 0.58 (mean reversion) | no | +0.001 (+0.09) | no |
| all seven | | | −0.025 (−1.66) | |

The stacker can learn negative weights, so the incremental test ignores sign. A
hypothesis with the wrong sign cannot explain these null gains.

**Standalone test IC** (per-fold mean, t, share of folds with positive IC):

| Expert | IC mean | t | Folds positive | Note |
| --- | --- | --- | --- | --- |
| range reversal | +0.062 | +4.6 | 75% | overlaps mean reversion; 72% sign agreement |
| abnormal volume | +0.039 | +2.0 | 61% | |
| overnight momentum | −0.033 | −2.4 | | opposite to its hypothesis |
| volatility regime | −0.054 | −3.1 | | opposite to its hypothesis |

**Regimes.** The reversal-type signals (range reversal, mean reversion, and
abnormal volume) are stronger in the high-volatility regime. This is
descriptive only.

**Portfolio.** Long-only tilt at 5 bp, against a benchmark Sharpe of 0.77:

- volatility regime: Sharpe 0.97, max drawdown −38%, against −48% for the
  benchmark;
- `ensemble.all`: 0.81;
- range reversal: 0.79.

The volatility-regime improvement comes from *risk timing*: it lowers exposure
in turbulent periods. Its directional IC is negative. The effect was noticed
after the fact, among about 15 predictors, and a Sharpe difference of 0.20 is
within one standard error (about 0.23) over this span. It is a hypothesis for a
future pre-registered or prospective test, not a finding.
