# Quorum V1.1: diversified research family `v1.1-diversified`

Status: **design approved 2026-10-06; not implemented or registered yet.** This
specification is frozen before any variant runs. Results are appended at the
end and never edit the sections above them, as in
[EXPERTS_V1.md](EXPERTS_V1.md).

**Naming.** V1.1 means V1's frozen experts, protocol, and portfolio rules
applied to a broader universe. If this research passes, its prospective study
is `quorum-prospective-v1.1`, which runs alongside V1. The name V2 is reserved
for a major change, such as new kinds of experts or a survivorship-aware stock
universe.

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
- **Calendar:** every asset must share the exact same bar calendar. A mismatch
  still fails closed; this family adds no fill policy.

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

## Universe (22 ETFs, fixed for the family)

Every fund existed before 2005-01-01, so each has full history over the
snapshot. The universe uses ETFs only. Hand-picking individual stocks today
would reintroduce the survivorship bias that STATUS.md step 2 is meant to fix.

| Sleeve | Symbols | Count |
| --- | --- | --- |
| US broad equity and Nasdaq | SPY (S&P 500), QQQ (Nasdaq-100), IWM (Russell 2000), IBB (Nasdaq Biotechnology) | 4 |
| US sectors (Select Sector SPDRs) | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY | 9 |
| International equity | EFA (developed ex-US), EEM (emerging) | 2 |
| US Treasuries and credit | SHY (1–3y), IEF (7–10y), TLT (20y+), TIP (inflation-linked), LQD (investment-grade corporate) | 5 |
| Real estate | VNQ (US REITs) | 1 |
| Gold | GLD | 1 |

Symbols are ingested with the `.US` suffix, as in V1.

**Excluded, with the reason for each:**

- ONEQ (Nasdaq Composite) is about 0.99 correlated with QQQ. At an equal
  weight per fund it would double big-tech exposure and add no
  diversification. IBB covers Nasdaq exposure with a different return profile.
- XLRE (2015) and XLC (2018) start too late.
- Broad commodity funds (DBC 2006, GSG 2006) start after 2005-01-01.
- Individual stocks bring survivorship bias until the universe study is done.
- Crypto has no history before 2005.

**Selection caveat.** These ETFs were chosen in 2026, but they track broad
asset classes rather than past winners. The residual hindsight is much smaller
than for single stocks. It is still recorded as a caveat in the report.

**Weight cap.** The long-only tilt maps a neutral score to half the per-asset
cap. Full investment at neutral therefore needs `max_weight_per_asset = 2/22`
(0.0909…), just as 0.25 = 2/8 did for the 8-asset universe. The gross cap stays
at 1.0 and the turnover cap at 0.5.

**Control: equal weight per fund.** At 1/22 each, the benchmark holds 16
equity-like funds (73%, including VNQ and IBB), 5 bond funds (23%), and gold
(5%).

- **Same starting point.** A neutral score holds exactly 1/22 per fund, so the
  control is every predictor's own starting point. Any difference from it
  comes from the predictions alone.
- **No judgment calls.** Equal weight per sleeve would mix an allocation
  choice into the comparison and add choices that could be tuned, such as
  sleeve boundaries or whether REITs count as equity.
- **Consistent with V1.**

## Data window and the 2020–2022 exclusion

- **Snapshot:** 2005-01-01 to 2025-12-31 from yfinance with auto-adjustment,
  ingested once and content-addressed.
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

## Variants (preregistered together)

Both variants use the 22-ETF snapshot and `expert_set = "v1"`, the V0 and v1
experts together. Each attempt reports every predictor: each expert, the
ensemble, and every stacker.

| Variant | `train_exclude` | Role |
| --- | --- | --- |
| `skip2020_22` | 2020-01-01 to 2022-12-31 | the collaborators' specification; holds the primary hypothesis |
| `full` | none | sensitivity: does the exclusion change anything? |

All other settings are the earlier families' long-only settings: horizon 5,
minimum training 252, validation 21, test 63, holdout 252, thresholds ±0.2,
stacker alpha 0.1, gross 1.0, turnover 0.5, and daily rebalance. The one
change is `max_weight_per_asset = 2/22`.

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
  2020–2022, and 2023–2024.

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

This is the smallest change that runs the family. The universe itself needs no
code: it is just the snapshot's symbols.

1. **Move the research workspace somewhere durable. Done 2026-10-05.** The
   trial ledger (12 attempts), 4 preregistrations, snapshots, and runs were
   copied from an old session's temp scratchpad to `C:\quorum-ws\research\`,
   with a OneDrive copy. All 220 file hashes were verified, and the ledger
   loads with 12 attempts.
2. **Add `ResearchConfig.train_exclude`**, a tuple of ISO date ranges that
   defaults to empty.
   - The field enters `config_id` and the spec fingerprint only when it is
     non-empty, so every existing configuration keeps its identifier.
   - Ranges from JSON (lists) are normalized to tuples, so a preregistered
     variant reloads to the same fingerprint.
3. **Filter stacker training in `_oof_rows`.** Positions whose label interval
   overlaps an excluded range are dropped. Each fold's stacker report records
   how many training positions were excluded.
4. **Add a per-period table to the report**, using a fixed period list, plus
   the excluded ranges in the chronology section.
5. **Primary test in the report.** Apply the existing
   `evaluation.sharpe_difference_test` (the Ledoit–Wolf test V1 uses) to
   `stacker.ridge`'s daily returns against the equal-weight benchmark at
   15 bp.
6. **CLI:** `preregister` accepts `train_exclude` through the existing variants
   JSON file. No new command is added.

## Tests

- Every configuration in the existing preregistrations reproduces its old
  `config_id` and spec fingerprint.
- On a synthetic calendar, the filter drops exactly the training positions
  whose labels overlap the range, including the December boundary case, and
  no validation or test slot.
- With an empty `train_exclude`, the stacker's fitted coefficients match the
  current code exactly.
- A preregistered variant with `train_exclude` survives the JSON round trip
  and runs.
- The per-period table sums back to the full-span counts.

## Run order

1. ~~Move and verify the research workspace.~~ Done.
2. Implement changes 2 to 6 with their tests. Run the full suite against the
   known baseline (14 failed, 11 errors, all platform-related).
3. Ingest the 22-ETF snapshot into `C:\quorum-ws\research`. If the calendars
   differ, stop and report which assets and dates. Choosing a fill policy
   would be a new design decision.
4. Commit the preregistration file (variants JSON) to the repository, then
   `preregister` both variants as family `v1.1-diversified`. Registration
   precedes any run.
5. `run-family`, then append the results below, unedited above this line.

## Risks

- **Calendar mismatch:** with 22 Yahoo series, one missing day on one ETF
  blocks ingestion. That is intended: it fails closed and gets reported.
- **Equity-heavy control:** at 73% equity-like, equal weight is not a
  risk-balanced portfolio. A predictor that merely de-risks toward bonds could
  look good. The per-period table and the exposure metrics are there to show
  this.
- **Runtime:** 22 assets instead of 8 is roughly 2.75 times more expert
  scoring per run. This is acceptable, but the run takes longer.
- **Vendor adjustments:** Yahoo's adjusted prices drift by about 1e-7 between
  requests. Every conclusion ties to the snapshot hash, not to a re-download.

## Out of scope

- The V1.1 prospective study and opening the 2025 holdout.
- Individual stocks and dynamic universe membership (STATUS.md step 2).
- Deflated Sharpe, FDR, and PBO (STATUS.md step 3).
- New experts or re-tuned thresholds.

## Decisions (2026-10-06)

The review questions were resolved as recommended:

1. **Nasdaq:** IBB replaces ONEQ; QQQ stays.
2. **Primary candidate:** `stacker.ridge` (the V0-expert stacker). Everything
   else is exploratory.
3. **Control:** equal weight per fund.
4. **Naming:** this work is V1.1; V2 is reserved for a major change.
