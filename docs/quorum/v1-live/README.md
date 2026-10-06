# quorum-prospective-v1: live results

A published copy of the live prospective study described in
[STATUS.md](../STATUS.md). The study runs on the maintainer's machine; these
files are copied here periodically with `agent/scripts/publish_quorum_v1.py`.

| File | What it is |
| --- | --- |
| `study.json` | the frozen declaration (universe, candidates, costs, confirmatory test) |
| `ledger.jsonl` | the hash-chained ledger: one declaration, then one decision or skip per trading day |
| `evaluation.json` | the latest interim evaluation, computed from the ledger and the price snapshots |
| `freeze_receipt_addendum_*.md` | operational changes since the freeze (run time, wake from sleep). None changes the study. Addendum 1 (runtime isolation) stays local because it contains machine paths. |

Each decision is recorded after the close and executes at the next open. The
ledger is append-only, so every older line stays byte-identical between
publishes. Any change to an earlier line means something is wrong.

Not published: the price snapshots (vendor data) and the freeze receipts
(local machine details). The ledger records each snapshot's content hash.

The study runs until 2028-09-26. Early numbers are noise: with a handful of
days, returns mostly reflect trading costs and the first rebalance.
