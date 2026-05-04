"""
EDA orchestrator — runs all 8 analysis scripts then assembles the master report.

Usage:
    python scripts/eda/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_SCRIPTS = sorted((PROJECT_ROOT / "scripts" / "eda").glob("[0-9][0-9]_*.py"))
REPORT_BUILDER = PROJECT_ROOT / "scripts" / "eda" / "99_build_report.py"


def main():
    py = sys.executable
    print("EDA orchestrator")
    print("=" * 60)
    t0 = time.time()
    failed = []
    for script in EDA_SCRIPTS:
        print(f"\n>>> {script.name}")
        res = subprocess.run([py, str(script)], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            failed.append(script.name)
            print(f"<<< FAILED ({res.returncode})")
        else:
            print(f"<<< OK")
    print(f"\n>>> {REPORT_BUILDER.name}")
    res = subprocess.run([py, str(REPORT_BUILDER)], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        failed.append(REPORT_BUILDER.name)

    total = time.time() - t0
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ Failed scripts: {failed}")
        sys.exit(1)
    else:
        print(f"✅ All EDA scripts succeeded ({total/60:.1f} min total)")
        print(f"   master report → data/processed/eda/EDA_REPORT.md")


if __name__ == "__main__":
    main()
