"""Extract EU Regulation 2023/1542 Annex XIII (Battery Passport fields).

Annex XIII spans pp. 108-109 of the regulation. It enumerates the data fields
that must appear in a battery's Digital Product Passport, organized by access
level (public / restricted / Commission+EU national authorities) and category.

Output: data/processed/dpp/eu_annex_xiii_fields.csv
        data/processed/dpp/eu_annex_xiii_raw.txt  (verbatim extracted text)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF = PROJECT_ROOT / "data/regulatory/eu/EU_Reg_2023_1542_full.pdf"
OUT_DIR = PROJECT_ROOT / "data/processed/dpp"
OUT_RAW = OUT_DIR / "eu_annex_xiii_raw.txt"
OUT_CSV = OUT_DIR / "eu_annex_xiii_fields.csv"

# Annex XIII spans pp. 108-109 of the 117-page regulation
ANNEX_FIRST_PAGE = 108
ANNEX_LAST_PAGE = 109


def extract_text() -> str:
    r = PdfReader(str(PDF))
    parts = []
    for p in range(ANNEX_FIRST_PAGE - 1, ANNEX_LAST_PAGE):
        parts.append(r.pages[p].extract_text() or "")
    return "\n".join(parts)


def parse_fields(text: str) -> list[dict]:
    """Parse the Annex XIII text into structured field rows.

    Annex XIII has 3 top-level sections:
      1. PUBLICLY ACCESSIBLE INFORMATION RELATING TO THE BATTERY MODEL
      2. INFORMATION ACCESSIBLE ONLY TO PERSONS WITH A LEGITIMATE INTEREST
         AND THE COMMISSION
      3. INFORMATION ACCESSIBLE ONLY TO NOTIFIED BODIES, MARKET SURVEILLANCE
         AUTHORITIES AND THE COMMISSION

    Each section contains lettered field entries (a)–(z, aa, bb, ...). We parse
    them by detecting the section header markers and the lettered prefixes.
    """
    # Drop page-break artifacts and excessive whitespace
    cleaned = re.sub(r"\n{2,}", "\n\n", text)

    # Detect 3 sections by their markers
    SECTION_MARKERS = [
        ("public",
         r"1\.\s+PUBLICLY ACCESSIBLE INFORMATION",
         r"2\.\s+INFORMATION ACCESSIBLE ONLY TO PERSONS WITH"),
        ("restricted",
         r"2\.\s+INFORMATION ACCESSIBLE ONLY TO PERSONS WITH",
         r"3\.\s+INFORMATION ACCESSIBLE ONLY TO NOTIFIED BODIES"),
        ("authorities_only",
         r"3\.\s+INFORMATION ACCESSIBLE ONLY TO NOTIFIED BODIES",
         r"$"),
    ]

    rows = []
    for access_level, start_re, end_re in SECTION_MARKERS:
        m_start = re.search(start_re, cleaned)
        if not m_start:
            continue
        m_end = re.search(end_re, cleaned[m_start.end():])
        section_text = cleaned[m_start.end(): m_start.end() + (m_end.start() if m_end else len(cleaned))]

        # Lettered entries: "(a)", "(b)", ..., "(aa)", "(bb)" — match each segment
        # Find positions of "(letter)" markers at line starts (allowing leading space)
        marker_re = re.compile(r"\n\s*\(([a-z]{1,3})\)\s+", re.IGNORECASE)
        positions = [(m.start(), m.group(1)) for m in marker_re.finditer(section_text)]
        for idx, (pos, letter) in enumerate(positions):
            next_pos = positions[idx + 1][0] if idx + 1 < len(positions) else len(section_text)
            chunk = section_text[pos:next_pos]
            # Strip the leading "\n  (letter)  " marker
            body = re.sub(r"^\s*\(" + re.escape(letter) + r"\)\s+", "", chunk.strip(), count=1)
            body = re.sub(r"\s+", " ", body).strip()
            rows.append({
                "access_level": access_level,
                "letter_id": letter.lower(),
                "field_text": body[:600],  # cap to keep CSV row sane
                "char_count": len(body),
            })
    return rows


def main() -> int:
    if not PDF.exists():
        print(f"FATAL: {PDF.relative_to(PROJECT_ROOT)} not found")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = extract_text()
    OUT_RAW.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_RAW.relative_to(PROJECT_ROOT)} ({len(text)} chars)")

    fields = parse_fields(text)
    print(f"Parsed {len(fields)} field entries across {len({r['access_level'] for r in fields})} access levels")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["access_level", "letter_id", "field_text", "char_count"])
        w.writeheader()
        for r in fields:
            w.writerow(r)
    print(f"Wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")

    # Summary by access level
    print("\n=== ANNEX XIII SUMMARY ===")
    from collections import Counter
    cnt = Counter(r["access_level"] for r in fields)
    for level in ("public", "restricted", "authorities_only"):
        print(f"  {level}: {cnt.get(level, 0)} fields")
    print()
    print("First 6 fields preview:")
    for r in fields[:6]:
        print(f"  [{r['access_level']:16s}] ({r['letter_id']}) {r['field_text'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
