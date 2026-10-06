# V1 Freeze Receipt, Addendum 2: run time moved to 22:00

Operational change at **2026-10-05T06:49Z**. This is an operational change only,
not an experimental or methodological one. The original freeze receipts and
Addendum 1 are unchanged.

| Item | Value |
| --- | --- |
| Scheduled task | Quorum prospective V1 recorder |
| Old schedule | Mon–Fri 17:00 America/Denver |
| New schedule | Mon–Fri 22:00 America/Denver (00:00 America/New_York) |
| Task definition diff | the trigger's `StartBoundary` only (`2026-09-26T17:00:00-06:00` → `2026-10-05T22:00:00-06:00`) |
| First run at the new time | 2026-10-05 22:00 |

## Reason

The sessions of 2026-09-29, 09-30, and 10-01 were missed with no run at all. On
09-29 the laptop lost power, rebooted at 16:55, and was in standby from 16:57 to
17:29, and the catch-up did not fire. On 09-30 and 10-01 the cause is unknown,
because Task Scheduler history was disabled. The owner chose 22:00, when the
machine is more reliably on.

## Why validity is unaffected

- The declaration already states that "scheduling never affects validity;
  recorded_at and the late rule decide." A decision counts only if it is
  recorded before the next bar's start (09:30 America/New_York). 22:00 Denver
  leaves about 9.5 hours of margin, against 14.5 hours at 17:00.
- `record` always uses the latest closed session (bars with
  `event_at <= recorded_at`), so the run time selects no different data.
- The fetch window ends at `now.date()` in UTC. At 22:00 Denver that is the next
  UTC day, which only widens a window that is already filtered to closed bars.

## Verification

- **1_equivalence_replay: pass.** On a scratch copy of the study with the real
  2026-10-02 decision removed, `record_day` ran with the pinned export, the
  frozen runtime (`-E -s`), and `now = 2026-10-03T04:00Z` (Friday 22:00 MDT).
  It produced a decision for 2026-10-02 with the same `decision_at`
  (2026-10-02T20:00Z). The targets equal the real 17:00 decision exactly for
  `ensemble_tilt`, `equal_weight_control`, and `trend_tilt`. For
  `vol_regime_tilt` they differ by at most 1.4e-7, which is the known Yahoo
  adjusted-price nondeterminism recorded in Addendum 1.
- **2_idempotence: pass.** On a scratch copy with the full ledger, a 22:00 run
  returned `no_new_session` for 2026-10-02 and appended nothing.
- **3_real_ledger_untouched: pass.** The real ledger's sha256 was identical
  before and after the tests. No `record` ran against the real study.
- **4_post_switch: pass.** The exported task XML before and after differs only
  in `StartBoundary`. The weekdays, catch-up (StartWhenAvailable), command and
  interpreter flags, working directory, workspace, backup destination, log
  file, logon type, and time limit are all unchanged.

## Unchanged

- `study.json`, `ledger.jsonl`, the snapshots, the pinned code export, and the
  isolated runtime.
- The declaration's schedule text still reads 17:00. The declaration is frozen
  and is not edited; this addendum supersedes that operational line.

## Residual risks

- The laptop can still be asleep or off at 22:00. Task Scheduler history should
  be enabled so that the next miss can be diagnosed.
- A provider failure still produces a `skip` with no retry the same night.
