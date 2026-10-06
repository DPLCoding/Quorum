# Quorum V1.1: diversified research family `v1.1-diversified`

Status: **approved 2026-10-06, including the late-joining revision. Not
implemented or registered yet.** This specification is frozen before any variant
runs. Results are appended at the end and never edit the sections above them,
as in [EXPERTS_V1.md](EXPERTS_V1.md).

**Naming.** V1.1 means V1's frozen experts, protocol, and portfolio rules
applied to a broader universe whose funds can join after the start. If this
research passes, its prospective study is `quorum-prospective-v1.1`, which
runs alongside V1. The name V2 is reserved for a major change, such as new
kinds of experts or a survivorship-aware stock universe.

## Purpose

Collaborators asked for three things:

1. Quorum V1 on GitHub. Done: the core is merged into `main` and the live
   results are published in [v1-live/](v1-live/README.md).
2. A more diversified universe: Nasdaq exposure, more asset classes, and more
   sectors.
3. Training data from 2005 to 2025, skipping 2020–2022 inclusive.

This family answers one question: **on a diversified, multi-asset universe, do
Quorum's existing predictors beat an equal-weight portfolio out of sample, and
does leaving 2020–2022 out of training help?** It is research only. A V1.1
prospective study is proposed only if the primary hypothesis passes. V1 is not
touched.

## What stays the same

- **Experts:** the frozen V0 and v1 expert sets, the static ensemble, and the
  per-fold ridge stacker. Nothing is re-parameterized.
- **Protocol:** daily regular-session bars, 5-bar target horizon, expanding
  folds (252 minimum training bars, 21 validation, 63 test), with purge and
  embargo equal to the horizon.
- **Holdout:** the last 252 bars (calendar 2025) stay a locked final holdout.
- **Portfolio path:** the long-only tilt through the unchanged Vibe engine,
  with the three existing cost modes (frictionless, 5 bp, 15 bp).
- **No filling:** a missing bar is never filled. Funds that trade from the
  start must share one calendar exactly, as before. Funds that join later
  follow the late-joining rules below.

### What the stacker is

Each expert (momentum, trend, mean reversion, and so on) is a fixed formula
that scores every fund every day from −1 (expect a fall over the next 5 days)
to +1 (expect a rise). The static ensemble averages the scores with fixed
weights.

The stacker instead *learns* how much to trust each expert. It fits a ridge
regression of the realized 5-day forward return on the experts' scores. Ridge
is a linear regression with a fixed penalty that shrinks the weights, so noise
does not look like a pattern. Its forecast is turned into a score from −1 to
+1. A score of 0 holds the fund at its equal-weight share, +1 doubles that
share, and −1 drops the fund.

It is refitted at every fold, using only earlier, purged data. It is the only
component in Quorum that learns from data.

## Late-joining funds

The snapshot starts on 2005-01-01, but a fund does not have to exist then. It
joins the universe once it has enough of its own history. This is the
behavior the universe should always have had: a fund's absence before its
launch says nothing about it.

- **Master calendar.** This is the calendar shared by the funds that trade
  from the start of the snapshot. As before, those funds must match exactly.
  The master calendar defines the folds, the purge and embargo, and the
  holdout.
- **Contiguous history.** From its first bar, a late fund's bars must equal
  the master calendar exactly, as an unbroken suffix. A gap after launch
  still fails closed, because there is no fill policy.
- **Entry bar.** A fund enters on its **253rd own trading day**, the first bar
  on which every expert in the v1 set can score it. The longest requirement is
  the volatility-regime expert's 253 bars; the others need 5 to 65. A fund
  that trades from the start enters on the master calendar's 253rd bar, which
  matches the existing 252-bar minimum training.
- **Before entry the fund does not exist.** It has no scores, no rows, no
  stacker training data, no portfolio weight, and no benchmark weight. Its
  pre-entry bars serve only as expert inputs.
- **From entry it is a normal member.**
  - It is scored, enters the stacker's training rows (subject to the purge and
    the 2020–2022 rule), gets a portfolio weight, and enters the equal-weight
    benchmark. All of this starts on the same bar.
  - The number of funds `N_t` is the count that has entered by decision bar
    `t`.
  - The per-fund cap is `2 / N_t`, so a neutral score holds exactly `1 / N_t`,
    the benchmark weight, at every date.
  - The gross cap stays at 1.0 and the turnover cap at 0.5. When a fund
    enters, the move toward its weight is subject to the same turnover cap as
    any other trade.
- **Benchmark.** Equal weight across the funds entered at each date,
  rebalanced daily. The engine computes this already: its benchmark averages
  the returns of the funds present, and each fund's price frame starts at its
  entry bar.
- **No exits.** Every fund in this universe still trades today. Funds that
  closed between 2005 and 2025 are not considered; see the selection caveat.
  Exits and delistings belong to the survivorship-aware universe work
  (STATUS.md step 2).

## Universe (30 ETFs, fixed for the family)

The universe uses ETFs only. Hand-picking individual stocks today would
reintroduce survivorship bias. The table gives each fund's launch year.
Entry is about one trading year later.

| Sleeve | Symbols (launch year if after 2005) | Count |
| --- | --- | --- |
| US broad equity and Nasdaq | SPY (S&P 500), QQQ (Nasdaq-100), IWM (Russell 2000), IBB (Nasdaq Biotechnology) | 4 |
| US sectors (Select Sector SPDRs) | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLC (2018) | 10 |
| International equity | EFA (developed ex-US), EEM (emerging) | 2 |
| US Treasuries and credit | SHY (1–3y), IEF (7–10y), TLT (20y+), TIP (inflation-linked), LQD (investment-grade corporate), HYG (high yield, 2007) | 6 |
| International bonds | EMB (emerging-market USD, 2007), BWX (developed ex-US Treasuries, 2007) | 2 |
| Real estate | VNQ (US REITs) | 1 |
| Commodities | GLD (gold), SLV (silver, 2006), DBC (broad commodities, 2006), DBA (agriculture, 2007) | 4 |
| Currency | UUP (US dollar index, 2007) | 1 |

That is 22 funds from the start and 8 that join later. Symbols are ingested
with the `.US` suffix, as in V1. Launch dates are checked at ingest; a fund
with a gap after launch blocks ingestion.

**Composition when all 30 have entered:**

| Sleeve | Funds | Share |
| --- | --- | --- |
| Equity-like (including VNQ, IBB, and XLC) | 17 | 57% |
| Bonds | 8 | 27% |
| Commodities | 4 | 13% |
| Dollar | 1 | 3% |

From 2006 to 2007 the universe holds the original 22, which are 73%
equity-like.

**Excluded, with the reason for each:**

- **ONEQ (Nasdaq Composite):** about 0.99 correlated with QQQ. It would double
  big-tech exposure; IBB covers Nasdaq with a different return profile.
- **XLRE (2015):** close to a duplicate of VNQ.
- **USO (oil):** overlaps DBC, which is energy-heavy, and its 2020
  restructuring changed what it holds.
- **Crypto ETFs (2024):** they would enter in 2025, inside the locked holdout,
  so they contribute no out-of-sample data.
- **Leveraged, inverse, and volatility products:** they are path-dependent
  instruments, not asset classes.
- **More single-country equity funds:** they add equity weight to a benchmark
  that is already equity-heavy.
- **Individual stocks:** they bring survivorship bias until the universe study
  is done.

**Selection caveat.** These ETFs were chosen in 2026, and all of them survived
to 2026. They track broad asset classes rather than past winners, so the
hindsight is much smaller than for single stocks. It is still recorded as a
caveat in the report.

**Control: equal weight per entered fund.**

- **Same starting point.** A neutral score holds exactly `1 / N_t` per fund,
  so the control is every predictor's own starting point. Any difference from
  it comes from the predictions alone.
- **No judgment calls.** Equal weight per sleeve would mix an allocation
  choice into the comparison and add choices that could be tuned, such as
  sleeve boundaries or whether REITs count as equity.
- **Consistent with V1.**

## Data window and the 2020–2022 exclusion

- **Snapshot:** 2005-01-01 to 2025-12-31 from yfinance with auto-adjustment,
  ingested once and content-addressed. Each late fund's bars start at its
  launch.
- **Final holdout:** 2025 stays locked. "Training through 2025" means 2025 is
  in the snapshot but reserved. It is opened once, later, and only for the
  pre-named primary candidate if it passes. Opening it is a separate decision
  that this family does not make.

The exclusion applies to **training only**, which affects only the stacker.

- **Rule.** A stacker training position is dropped when its label interval
  (entry at the next bar's open through exit at the close `h` bars later)
  overlaps 2020-01-01 to 2022-12-31 in America/New_York dates. A decision in
  late December 2019 whose label ends in January 2020 is therefore dropped too.
- **Still in the test.** 2020–2022 remains in the validation and test slots.
  It is predicted, scored, and traded like any other period, because testing
  on the COVID crash and the 2022 bear market is the point. Excluding it from
  testing would flatter every result.
- **Expert inputs are not training.** An expert scoring a 2023 decision reads
  its trailing window, which can include 2022 bars. That is an input, not a
  fitted parameter, and it is allowed.
- **Effect by fold.** Folds whose training ends before 2020 are unaffected.
  Later folds train on 2005–2019 plus any post-2022 positions before the fold.
  XLC enters in mid-2019, so almost all of its training rows come from 2023
  onward.

## Variants (preregistered together)

Both variants use the 30-ETF snapshot and `expert_set = "v1"`, the V0 and v1
experts together. Each attempt reports every predictor: each expert, the
ensemble, and every stacker.

| Variant | `train_exclude` | Role |
| --- | --- | --- |
| `skip2020_22` | 2020-01-01 to 2022-12-31 | the collaborators' specification; holds the primary hypothesis |
| `full` | none | sensitivity: does the exclusion change anything? |

All other settings are the earlier families' long-only settings: horizon 5,
minimum training 252, validation 21, test 63, holdout 252, thresholds ±0.2,
stacker alpha 0.1, gross 1.0, turnover 0.5, and daily rebalance. The
per-fund cap is `2 / N_t` instead of a fixed value.

These two attempts join the existing trial family, which already holds 12
attempts. Every attempt counts, including failures.

## Hypotheses and success criteria

**Primary (confirmatory):** in `skip2020_22`, the `stacker.ridge` long-only
tilt has a higher out-of-sample Sharpe than the equal-weight benchmark at the
15 bp cost mode. The test is one-sided, using the same Ledoit–Wolf
Sharpe-difference test that V1 declares, at α = 0.05.

- **What `stacker.ridge` is.** The ridge stacker over the three V0 experts:
  momentum, trend, and mean reversion.
- **Why the stacker.** It is the only predictor the training-data rule can
  affect.
- **Why this version.** The earlier `experts-v1` round found that none of the
  four v1 experts adds incremental information.
- **Why not the volatility-regime tilt.** It is already under prospective test
  in V1. Re-testing it on the history where it was first noticed would count
  the same evidence twice.

**Exploratory, reported descriptively and never used to claim success:**

- every other predictor in both variants against equal weight, including
  `stacker.all`, the `stacker.v0+…` variants, each expert, and the ensemble;
- the difference between the two variants' `stacker.ridge`, which is the effect
  of the exclusion;
- every predictor and the benchmark split into three periods: 2006–2019,
  2020–2022, and 2023–2024, with the number of funds entered in each;
- the out-of-sample IC split between funds present from the start and funds
  that joined later.

**Outcomes:**

- **Pass:** the primary test rejects. The next steps are a separate decision
  to open the 2025 holdout for `stacker.ridge` only. If it holds there, a V1.1
  prospective declaration follows, previewed and reviewed before it is frozen.
- **Fail:** the result is reported as a negative and appended here. No
  variant is added after the results are seen, and no V1.1 prospective study
  starts.
- Every report states the size of the trial family. Deflated Sharpe and FDR
  corrections (STATUS.md step 3) are not built yet, so no pass is described as
  a discovery without that caveat.

**Expectation, stated in advance.** On the 8-asset universe the stacker roughly
matched equal weight without beating it. A negative result here is plausible
and is still informative.

## Changes to the code

The universe is the snapshot's symbols. Late joining is the main new
mechanism.

1. **Move the research workspace somewhere durable. Done 2026-10-05.** The
   trial ledger (12 attempts), 4 preregistrations, snapshots, and runs were
   copied from an old session's temp scratchpad to `C:\quorum-ws\research\`,
   with a OneDrive copy. All 220 file hashes were verified, and the ledger
   loads with 12 attempts.
2. **Master calendar and entry offsets.** Replace `_shared_calendar` in
   `research.py` with a function that returns three things: the master
   calendar, each fund's offset into it, and each fund's entry position. It
   enforces both calendar rules (an exact match for funds that trade from the
   start, an unbroken suffix for late funds) and fails closed otherwise. When
   every fund trades from the start, it returns exactly what
   `_shared_calendar` returned.
3. **Index by master position in `_oof_rows`.** Expert scores, labels,
   regimes, and stacker rows are computed from each fund's own bars and keyed
   by master position. A fund contributes nothing before its entry position.
   The folds and the plan stay on the master calendar.
4. **Add `ResearchConfig.train_exclude`**, a tuple of ISO date ranges that
   defaults to empty.
   - The field enters `config_id` and the spec fingerprint only when it is
     non-empty, so every existing configuration keeps its identifier.
   - Ranges from JSON (lists) are normalized to tuples, so a preregistered
     variant reloads to the same fingerprint.
5. **Filter stacker training.** Positions whose label interval overlaps an
   excluded range are dropped. Each fold's stacker report records how many
   training positions were excluded.
6. **Per-fund cap `2 / N_t`.** The cap becomes a declared rule in the config,
   while a fixed cap remains the default so existing configs keep their
   identifiers. The risk policy applies the rule at each decision timestamp,
   with `N_t` taken from the entry positions, not from how many decisions
   happen to exist.
7. **Portfolio frames start at entry.** In `portfolio.py`, each fund's engine
   frame begins at its entry bar, and the master calendar replaces "the
   alphabetically first fund" as the reference. The engine's benchmark then
   averages only the entered funds.
8. **Report.**
   - A per-period table, with fund counts.
   - The excluded ranges and each fund's launch and entry dates in the
     chronology section.
   - The primary test: the existing `evaluation.sharpe_difference_test` on
     `stacker.ridge`'s daily returns against the benchmark at 15 bp.
9. **Ingest.** Confirm that the loader returns each late fund from its launch
   without padding rows. `daily_bars_from_frame` still rejects any missing
   value.
10. **CLI:** `preregister` accepts the new config fields through the existing
    variants JSON file. No new command is added.

## Tests

**Regression (the most important):** on the existing 8-asset snapshot, where
every fund trades from the start, a run reproduces the earlier results
exactly. This covers the same rows, scores, stacker coefficients, and
portfolio metrics, including the same `config_id` and spec fingerprint.

**Late joining (synthetic):**

- A fund launched mid-sample has no rows before its 253rd bar and normal rows
  after it.
- A gap after launch fails closed.
- A late fund whose dates are not a suffix of the master calendar fails
  closed.
- The cap equals `2 / N_t` at each timestamp, and a neutral score gives
  exactly `1 / N_t`.
- The benchmark weight count steps up on the entry bar, not on the launch bar.

**Training exclusion:**

- The filter drops exactly the training positions whose labels overlap the
  range, including the December boundary case, and no validation or test
  slot.
- A preregistered variant with the new fields survives the JSON round trip
  and runs.

**Report:** the per-period table sums back to the full-span counts.

## Run order

1. ~~Move and verify the research workspace.~~ Done.
2. Implement changes 2 to 10 with their tests, regression first. Run the full
   suite against the known baseline (14 failed, 11 errors, all
   platform-related).
3. Ingest the 30-ETF snapshot into `C:\quorum-ws\research`. If any calendar
   rule fails, stop and report which funds and dates. Choosing a fill policy
   would be a new design decision.
4. Commit the preregistration file (variants JSON) to the repository, then
   `preregister` both variants as family `v1.1-diversified`. Registration
   precedes any run.
5. `run-family`, then append the results below, unedited above this line.

## Risks

- **More mechanics, more ways to be subtly wrong.** The regression test on the
  8-asset snapshot is the guard: without late funds, nothing may change.
- **Entry-day alignment.** The engine has no return for a fund's first frame
  bar, so the fund adds to the benchmark from its second bar. The strategy's
  first trade in it fills at the next open. The off-by-one is the same for
  every fund and is documented in the report.
- **Calendar mismatch:** with 30 Yahoo series, one missing day blocks
  ingestion. That is intended: it fails closed and gets reported.
- **Benchmark mix changes over time,** from 73% equity-like in 2006 to 57% once
  all funds have entered. Per-period results and exposure metrics show this.
- **Runtime:** 30 funds instead of 8 is roughly 3.5 times more expert scoring
  per run.
- **Vendor adjustments:** Yahoo's adjusted prices drift by about 1e-7 between
  requests. Every conclusion ties to the snapshot hash, not to a re-download.

## Out of scope

- The V1.1 prospective study and opening the 2025 holdout.
- Exits, delistings, and individual stocks (STATUS.md step 2).
- Deflated Sharpe, FDR, and PBO (STATUS.md step 3).
- New experts or re-tuned thresholds.

## Decisions

From the first review, 2026-10-06:

1. **Nasdaq:** IBB replaces ONEQ; QQQ stays.
2. **Primary candidate:** `stacker.ridge` (the V0-expert stacker). Everything
   else is exploratory.
3. **Control:** equal weight per fund.
4. **Naming:** this work is V1.1; V2 is reserved for a major change.

From the second revision, approved 2026-10-06:

5. **Late joining:** funds enter on their 253rd trading day. The cap is
   `2 / N_t` and the benchmark is equal weight over entered funds.
6. **Universe:** 30 ETFs, adding XLC, HYG, EMB, BWX, SLV, DBC, DBA, and UUP.
