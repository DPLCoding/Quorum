# V1 Freeze Receipt, Addendum 3: wake from sleep to run

Operational change at **2026-10-06T20:34Z**. This is an operational change only,
not an experimental or methodological one. The original freeze receipts and
Addendums 1 and 2 are unchanged.

| Item | Value |
| --- | --- |
| Scheduled task | Quorum prospective V1 recorder |
| Change | `WakeToRun` false → true: the task may wake the computer from sleep at its start time |
| Task definition diff | `<WakeToRun>true</WakeToRun>` added; nothing else |
| Schedule | unchanged: Mon–Fri 22:00 America/Denver (Addendum 2) |
| Power plan | wake timers are allowed on AC power and disabled on battery. The power plan was not changed. |

## Reason

On 2026-10-05 the laptop entered standby at 21:08 and missed the 22:00 start.
The catch-up (StartWhenAvailable) launched the run at 00:29:51 during a brief
wake, and it recorded the 2026-10-05 decision at 00:30:06 MDT, which is valid
(before 09:30 New York). It then stalled in the backup step while the machine
slept, and the 1-hour execution limit terminated it at 01:30:50. The task
reported `0x41306`, but the decision and the backup mirror were complete: the
mirror ledger was byte-identical to the primary and the snapshot counts
matched. A later manual backup reported 6 records. Had the laptop stayed
asleep until after 07:30 MDT, the decision would have been late. Waking the
machine at 22:00 removes the dependence on a lucky wake.

## Why validity is unaffected

The declaration states that "scheduling never affects validity; recorded_at and
the late rule decide." Waking the machine changes only when the process starts;
the recorder, its inputs, and its rules are identical.

## Verification

- **1_post_switch: pass.** The exported task XML before and after differs only
  in `WakeToRun`. The trigger (22:00 Mon–Fri), catch-up, command and
  interpreter flags, working directory, workspace, backup destination, log
  file, logon type, and time limit are all unchanged.
- **2_ledger_and_backup: pass.** 6 records (declaration, 4 decisions, 1 skip);
  the backup mirror's ledger sha256 equals the primary.
- **3_no_record_run: pass.** No `record` was run against the real study.

## Unchanged

`study.json`, `ledger.jsonl`, the snapshots, the pinned code export, the
isolated runtime, and the schedule.

## Residual risks

- On battery the wake timer does not fire (the power plan's DC setting). Keep
  the laptop plugged in overnight on weekdays, or enable wake timers on battery
  in the power plan.
- Wake timers do not work while the machine is shut down or hibernated
  without wake support.
- The 1-hour execution limit can terminate a run that stalls during the backup
  step. The decision is appended before the backup, and the next run's backup
  re-mirrors.
