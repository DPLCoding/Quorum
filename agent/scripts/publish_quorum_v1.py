"""Copy the live quorum-prospective-v1 results into docs/quorum/v1-live/.

Read-only against V1 apart from ``evaluation.json``, which ``evaluate`` rewrites.
Evaluation runs from the pinned export with the frozen runtime, exactly as the
local runbook does. Freeze receipts and snapshots stay local: the receipts hold
machine paths and the snapshots hold vendor price data.

    python agent/scripts/publish_quorum_v1.py [--workspace C:\\quorum-ws]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

STUDY = "quorum-prospective-v1"
PINNED = "3e89f8bc7162767afba47a3fc11b96e92c4b5e75"
FILES = ("study.json", "ledger.jsonl", "evaluation.json")
DEST = Path(__file__).resolve().parents[2] / "docs" / "quorum" / "v1-live"
# Anything that looks like a machine path or a home directory must not go public.
LEAK = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users)/|OneDrive", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, default=Path(r"C:\quorum-ws"))
    workspace = parser.parse_args().workspace

    subprocess.run(
        [
            str(workspace / "runtime" / "v1-py3.13.3" / "python.exe"),
            "-E", "-s", "-m", "src.quorum.prospective", "evaluate",
            "--workspace", str(workspace), "--study", STUDY,
        ],
        cwd=workspace / "code" / PINNED / "agent",
        check=True,
        stdout=subprocess.DEVNULL,
    )

    source = workspace / "prospective" / STUDY
    for name in FILES:
        leak = LEAK.search((source / name).read_text(encoding="utf-8"))
        if leak:
            raise SystemExit(f"refusing to publish {name}: contains {leak.group()!r}")
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, DEST / name)
    print(f"published {', '.join(FILES)} to {DEST}")


if __name__ == "__main__":
    main()
