"""
Shared helpers for the EDA suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EDA_DIR = PROJECT_ROOT / "data" / "processed" / "eda"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_findings(name: str, lines: list[str]) -> Path:
    """Write a markdown fragment for one EDA section."""
    out = EDA_DIR / f"findings_{name}.md"
    out.write_text("\n".join(lines).rstrip() + "\n")
    return out


def fig_path(section: str, name: str) -> Path:
    """Return a figure path under results/figures/eda/<section>/<name>.png."""
    out = FIG_DIR / section / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a markdown table from headers + rows of strings."""
    line1 = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line1, sep] + body)
