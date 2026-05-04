"""Build a per-paper structured working file for the MCDM weight extraction.

Produces `data/processed/mcdm_weights/literature_weights_WORKING.csv` with one
section per T2 paper, pre-populated with:
  - paper metadata (lit_seq, year, doi, paper title)
  - the lit-review-derived MCDM criteria flags (your authoritative guide for
    which canonical criteria each paper engages with)
  - any rows already auto-extracted by pdfplumber (paper #22 mostly)
  - one BLANK row per canonical criterion the lit review flagged for that
    paper, so you just fill in the weight number

The reviewer's job is to:
  1) Open each PDF (research-papers/T2_*.pdf)
  2) Find the criterion-weight table in the paper
  3) For each canonical-criterion row pre-populated, fill in `weight` and
     `paper_criterion_name (raw)` with the source paper's terminology
  4) Add extra rows if the paper covers criteria beyond what the lit review
     flagged (rare but possible)
  5) Delete rows for criteria the paper doesn't actually cover (also rare)
  6) Save as `literature_weights.csv` (drop _WORKING suffix) when done
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "data" / "processed" / "mcdm_weights"
PAPER_INDEX = WEIGHTS_DIR / "paper_index.csv"
TEMPLATE = WEIGHTS_DIR / "literature_weights_TEMPLATE.csv"
WORKING = WEIGHTS_DIR / "literature_weights_WORKING.csv"

CANON_CRITERIA = ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"]


def load_paper_index() -> list[dict]:
    rows = []
    with open(PAPER_INDEX) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_auto_extracted() -> dict:
    """Group already-extracted rows by lit_seq."""
    grouped: dict[str, list[dict]] = {}
    if not TEMPLATE.exists():
        return grouped
    with open(TEMPLATE) as f:
        reader = csv.DictReader(f)
        for r in reader:
            seq = r.get("paper_id (lit_seq)", "").strip()
            if not seq:
                continue
            grouped.setdefault(seq, []).append(r)
    return grouped


def parse_criteria(s: str) -> list[str]:
    if not s or s.strip() == "—":
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def main() -> int:
    papers = load_paper_index()
    auto = load_auto_extracted()

    # Sort papers by score desc, then year desc — score-5 papers come first
    def sort_key(p):
        try:
            score = int(p.get("lit_score", "0") or 0)
        except ValueError:
            score = 0
        try:
            year = int(p.get("year", "0") or 0)
        except ValueError:
            year = 0
        return (-score, -year)

    papers.sort(key=sort_key)

    fields = [
        "paper_id (lit_seq)", "pdf", "year", "doi",
        "paper_criterion_name (raw)",
        "canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)",
        "weight (0-1)", "source_table_csv", "notes",
    ]

    with open(WORKING, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)

        for p in papers:
            seq = p["lit_seq"]
            pdf = p["pdf"]
            year = p["year"]
            doi = p["doi"]
            score = p.get("lit_score", "")
            mcdm_criteria_str = p.get("mcdm_criteria", "")
            criteria_list = parse_criteria(mcdm_criteria_str)

            # --- Section header (commented row pattern; readable in Excel/CSV)
            w.writerow([
                f"# === PAPER #{seq} (score {score}) — {pdf} ===",
                "", "", "", "", "", "", "", "",
            ])
            authors = p.get("authors", "")[:60]
            title = p.get("title", "")[:80]
            w.writerow([
                f"# {authors} ({year})",
                f"# {title}",
                f"# Lit MCDM criteria: {mcdm_criteria_str or '—'}",
                f"# Auto-extracted rows: {len(auto.get(seq, []))}",
                "",
                "",
                "",
                "",
                "",
            ])

            # --- Auto-extracted rows (with canonical_criterion still blank for review)
            for r in auto.get(seq, []):
                w.writerow([
                    seq, pdf, year, doi,
                    r.get("paper_criterion_name (raw)", ""),
                    r.get("canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)", ""),
                    r.get("weight (0-1)", ""),
                    r.get("source_table_csv", ""),
                    r.get("notes", "[auto-extracted, REVIEW + map to canonical]"),
                ])

            # --- One BLANK row per canonical criterion the lit review flagged
            if not auto.get(seq):
                if criteria_list:
                    for c in criteria_list:
                        w.writerow([
                            seq, pdf, year, doi,
                            "",   # raw paper criterion name (you fill in)
                            c,    # pre-populated canonical criterion
                            "",   # weight (you fill in)
                            "",
                            f"[manual: read PDF, find weight for '{c}']",
                        ])
                else:
                    # Lit review marked criteria as "—" — paper doesn't cover any of our 6
                    w.writerow([
                        seq, pdf, year, doi,
                        "", "", "", "",
                        "[skip: lit review flagged no MCDM criteria for this paper]",
                    ])

            # blank separator row
            w.writerow(["", "", "", "", "", "", "", "", ""])

    n_lines = sum(1 for _ in open(WORKING))
    print(f"Wrote {WORKING.relative_to(PROJECT_ROOT)} ({n_lines} lines)")
    print(f"\nPapers ordered by relevance score (5 → 3):")
    for p in papers:
        seq = p["lit_seq"]
        n_auto = len(auto.get(seq, []))
        print(f"  #{seq:>2} (score {p.get('lit_score','?')}) {p['pdf'][:55]:55s}  auto-rows={n_auto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
