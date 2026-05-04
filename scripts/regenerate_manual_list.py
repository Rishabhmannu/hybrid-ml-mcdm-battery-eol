"""Regenerate MANUAL_DOWNLOAD_LIST.md based on current state of research-papers/."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"
MIN_BYTES = 50_000

spec = importlib.util.spec_from_file_location("v3mod", ROOT / "scripts" / "download_papers_v3.py")
v3mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3mod)
MANIFEST = v3mod.MANIFEST


def is_pdf_file(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < MIN_BYTES:
        return False
    with p.open("rb") as f:
        return f.read(5) == b"%PDF-"


on_disk = []
still_missing = []
for entry in MANIFEST:
    if is_pdf_file(PAPERS_DIR / entry["filename"]):
        on_disk.append(entry)
    else:
        still_missing.append(entry)

print(f"On disk: {len(on_disk)} / {len(MANIFEST)}")
print(f"Still missing: {len(still_missing)}\n")
for e in still_missing:
    kind = "paywalled" if e.get("paywalled") else "blocked"
    print(f"  [{e['theme']}] [{kind}] {e['filename']}")

# Write manual list
titles = {"T1": "Hybrid ML for SoH/RUL", "T2": "MCDM for circular economy", "T3": "Battery DPP",
          "T4": "Anomaly detection", "T5": "India regulatory / market", "T6": "Physics simulation / synthetic data"}
by_theme: dict[str, list[dict]] = {}
for e in still_missing:
    by_theme.setdefault(e["theme"], []).append(e)

lines = [
    "# Manual Download List",
    "",
    f"**{len(still_missing)} papers** still need manual download (out of {len(MANIFEST)} candidates). {len(on_disk)} are already in `research-papers/`.",
    "",
    "## Why each is still missing",
    "- `paywalled`: behind Elsevier flagship Energy / ACS / Wiley / Springer / Nature flagship paywall — needs institutional access OR Sci-Hub.",
    "- `blocked`: open-access paper blocked by anti-bot (Cloudflare on Wiley/OUP, Akamai on ScienceDirect). **Open the DOI in browser and the PDF will download cleanly.**",
    "",
    "## How to download",
    "Cmd-click each DOI link below to open in a new tab → click the publisher's PDF button → save into `research-papers/` with the suggested filename.",
    "",
]
for theme in sorted(by_theme):
    lines += [f"## {theme} — {titles.get(theme, theme)}", "",
              "| # | Suggested Filename | Title | DOI / Link | Reason |",
              "|---|--------------------|-------|------------|--------|"]
    for i, e in enumerate(by_theme[theme], 1):
        kind = "paywalled" if e.get("paywalled") else "blocked"
        doi = e.get("doi", "")
        doi_link = f"[{doi}](https://doi.org/{doi})" if doi else "(no DOI)"
        lines.append(f"| {i} | `{e['filename']}` | {e['title']} | {doi_link} | {kind} |")
    lines.append("")

MANUAL_LIST.write_text("\n".join(lines))
print(f"\nWrote {MANUAL_LIST}")
