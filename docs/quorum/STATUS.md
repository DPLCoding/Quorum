# Quorum status and next steps

Last updated: 2026-10-05. Read this first when resuming work. Details live in
[ARCHITECTURE.md](ARCHITECTURE.md) (Tasks 1–15), [ROADMAP.md](ROADMAP.md),
[RESEARCH_PRINCIPLES.md](RESEARCH_PRINCIPLES.md), and
[EXPERTS_V1.md](EXPERTS_V1.md).

## Where things stand

The research spine is complete and exercised on real data:

- Contracts, append-only experiment ledger, chronological fold planner with a
  locked holdout, content-addressed data snapshots (Tasks 1–8).
- Real-data runner and prediction evaluation (Task 9), per-fold ridge stacker
  (Task 10), stitched out-of-sample portfolio through the unchanged Vibe engine
  (Task 11), and pre-registered variant families (Task 12).
- Expert round v1 (Task 13): four orthogonal-by-design experts, pre-registered
  before evaluation.
- Broker-free prospective ledger (Tasks 14–15).

### Research results so far (8 US ETFs and large caps, daily, 2006–2024 OOS)

The results are honest negatives:

- The V0 experts (momentum, trend, mean reversion) have small out-of-sample IC,
  with mean reversion best at about +0.08 per fold. That IC does not survive
  costs in long/short construction. As long-only tilts, the best candidates
  roughly match the equal-weight benchmark (Sharpe about 0.79 against 0.77)
  without beating it.
- None of the v1 experts (volatility regime, overnight momentum, abnormal
  volume, range reversal) met the pre-registered incremental-information
  criterion.
- The volatility-regime tilt showed a higher long-only Sharpe (0.97) through
  risk timing. That was noticed after the fact, so it is being tested
  prospectively in V1.
- The trial family holds 12 research attempts, all recorded, including 2
  failed and 2 interrupted.

Upstream fixes made along the way:

- `validate_ohlc` snaps adjustment rounding instead of dropping real trading
  days.
- `BaseEngine._weighted_holding_bars` went from O(n³) to amortized O(1).
- The risk policy no longer overshoots its gross cap by one rounding step.

## Prospective study `quorum-prospective-v1` (live)

Frozen 2026-09-26 at commit `3e89f8bc7162767afba47a3fc11b96e92c4b5e75` and runs
until 2028-09-26. It compares four long-only candidates:

- `equal_weight_control` (the confirmatory control);
- `ensemble_tilt`;
- `trend_tilt`;
- `vol_regime_tilt`, the one confirmatory hypothesis, tested with a
  Ledoit–Wolf Sharpe-difference test.

- **It is immutable.** Never modify its declaration, ledger, snapshots, pinned
  code export, or isolated runtime. Improvements go into a new study ID, which
  can run concurrently.
- It records a decision every weekday at 17:00 America/Denver from a pinned
  code export and a frozen Python runtime, both outside this repository.
  Operations, verification commands, and restore steps are in the local runbook
  in the V1 workspace (`C:\quorum-ws\README.md`).
- It has no broker and places no orders. Evidence accumulates slowly: expect
  months before any read, and even two years has limited power.

## Next steps, in priority order

1. **V1 first unattended run: done** (Monday 2026-09-28 17:00). Friday's
   decision executed at Monday's open, Monday's decision was recorded, and the
   backup mirror updated. Live results are published to
   [v1-live/](v1-live/README.md) with `agent/scripts/publish_quorum_v1.py`;
   rerun it and commit to refresh.
   - **Missed sessions:** 2026-09-29, 09-30, and 10-01 have no decision. On
     09-29 the laptop was asleep at 17:00 after an unclean power loss, and
     catch-up did not fire. On 09-30 and 10-01 the cause is unknown: Task
     Scheduler history is disabled. Missed sessions are permanent and are
     reported as such.
   - **Research workspace** (trial ledger with 12 attempts, preregistrations,
     snapshots, runs) moved out of a temp folder on 2026-10-05 to the local V1
     machine's `C:\quorum-wsesearch`, plus a OneDrive copy. All 220 file
     hashes were verified.
   - **Next:** a diversified research family: a broader universe (more asset
     classes, sectors, and Nasdaq exposure) with training data from 2005–2025
     that skips 2020–2022. Preregister it before running; it is research
     first, and a V2 prospective study follows only if it beats the control
     out of sample.
     Design draft: [DIVERSIFIED_DESIGN.md](DIVERSIFIED_DESIGN.md) (under
     review; three open questions at the end).
2. **Survivorship-aware universe study (design document only).** This gates all
   cross-sectional research. Investigate sources of point-in-time index
   membership and delisted or acquired companies, their cost and licensing, how
   entries, exits, and suspensions appear in snapshots, and missing-asset rules.
   First check for WRDS/CRSP access through the university. Do not treat
   today's large caps with full history as a canonical universe (survivorship
   bias).
3. **Research-quality gates (roadmap 0.8).** Turn the trial family into
   corrected statistics (Deflated Sharpe, FDR, PBO) using
   `src.quorum.experiments` and `src.quantlib.multipletesting`, so every report
   discounts for the number of attempts.
4. **Optional:** offer the two upstream engine and loader fixes to
   HKUDS/Vibe-Trading as pull requests.
5. **After the universe decision:**
   - missing-asset and dynamic-universe semantics;
   - a cross-sectional expert round (Alpha Zoo factors), pre-registered;
   - a V2 prospective study only if something beats the control out of sample.

## Open decisions

- **V1 retry run.** A transient Yahoo failure, or a machine that is off at
  17:00, becomes a missed session (three already). Extra triggers are safe:
  `record` is idempotent (`no_new_session`) and the declaration says
  scheduling never affects validity. The schedule text is part of the frozen
  operations section, though, so a change needs a written addendum like the
  runtime one.
- **Commit email.** Commits use `DPLCoding` with a public university email.
  Optionally switch future commits to the GitHub noreply address.

## Working conventions learned

- Pre-register every experiment family before running any of it. Every attempt
  counts, including failed and interrupted ones. Specifications are frozen
  before results are seen.
- Freeze prospective studies only after a machine-generated declaration
  preview has been reviewed.
- On this Windows machine, run pytest with `--basetemp=<writable dir>` if the
  default temp directory is inaccessible. The full suite has a known baseline
  of 14 failed and 11 errors, all symlink or platform issues unrelated to
  Quorum.
