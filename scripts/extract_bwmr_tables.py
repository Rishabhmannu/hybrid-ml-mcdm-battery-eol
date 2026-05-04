"""
Extract Schedule I/II tables from BWMR PDFs into structured CSVs.

What this gets us for the project:
- Schedule II Table 1: Per-battery-type EPR collection/recycling targets per FY
- Schedule II Table 2: Material-wise minimum recovery percentages
- Schedule II Table 3: Minimum recycled-content percentages by FY (2027-28 → 2030-31+)
- Plus any other tabular data (heavy-metal limits, etc.)

Output:
- data/regulatory/bwmr/extracted_tables/<pdf_stem>__page<NN>__table<K>.csv
- data/regulatory/bwmr/extracted_tables/index.json (manifest)

Usage:
    conda activate Eco-Research
    python scripts/extract_bwmr_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BWMR_DIR = PROJECT_ROOT / "data" / "regulatory" / "bwmr"
OUT_DIR = BWMR_DIR / "extracted_tables"


def has_devanagari(s: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in s or "")


def is_english_page(text: str) -> bool:
    """Skip Hindi-dominant pages."""
    if not text:
        return False
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin < 30:
        return False
    return devanagari < 0.4 * (devanagari + latin)


def clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    s = str(cell).replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def label_table(rows: list[list[str]]) -> str:
    """Heuristic label so the CSV filename is meaningful."""
    blob = " ".join(clean_cell(c) for r in rows for c in r).lower()
    if "minimum use of the recycled materials" in blob or "recycled material" in blob:
        return "recycled_content_targets"
    if "recovery" in blob and ("percentage" in blob or "minimum" in blob):
        return "recovery_targets"
    if "extended producer responsibility" in blob and ("collection" in blob or "target" in blob):
        return "epr_collection_targets"
    if "form 1" in blob or "form 2" in blob or "form 3" in blob or "form 4" in blob:
        m = re.search(r"form\s*(\d+[a-z]?)", blob)
        if m:
            return f"form_{m.group(1).lower()}"
    if "extended producer responsibility certificate" in blob:
        return "epr_certificates"
    if "heavy metal" in blob or "mercury" in blob or "cadmium" in blob:
        return "heavy_metal_limits"
    return "table"


def write_csv(rows: list[list[str]], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([clean_cell(c) for c in r])


def extract_pdf(pdf_path: Path) -> list[dict]:
    manifest = []
    stem = pdf_path.stem
    print(f"\n=== {stem} ===")
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not is_english_page(text):
                continue
            tables = page.extract_tables() or []
            if not tables:
                continue
            for k, raw in enumerate(tables, start=1):
                # Skip near-empty tables
                if not raw or sum(len([c for c in row if clean_cell(c)]) for row in raw) < 3:
                    continue
                # Skip tables that are mostly Hindi (sometimes a Hindi caption sneaks onto an English page)
                blob = " ".join(clean_cell(c) for row in raw for c in row)
                if has_devanagari(blob):
                    devanagari = sum(1 for c in blob if "ऀ" <= c <= "ॿ")
                    latin = sum(1 for c in blob if c.isascii() and c.isalpha())
                    if devanagari > 0.4 * max(devanagari + latin, 1):
                        continue
                label = label_table(raw)
                fname = f"{stem}__p{page_num:02d}__t{k}__{label}.csv"
                out = OUT_DIR / fname
                write_csv(raw, out)
                preview = " | ".join(clean_cell(c) for c in raw[0])[:120]
                print(f"  p{page_num:>2}  table {k}  -> {fname}")
                print(f"        header: {preview}")
                manifest.append({
                    "pdf": pdf_path.name,
                    "page": page_num,
                    "table_index": k,
                    "label": label,
                    "csv": str(out.relative_to(PROJECT_ROOT)),
                    "n_rows": len(raw),
                    "n_cols": max(len(r) for r in raw),
                })
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Discover all BWMR PDFs in the directory automatically
    pdfs = sorted(p.name for p in BWMR_DIR.glob("*.pdf"))
    full_manifest = []
    for fname in pdfs:
        pdf = BWMR_DIR / fname
        if not pdf.exists():
            print(f"SKIP missing {pdf}")
            continue
        full_manifest.extend(extract_pdf(pdf))

    (OUT_DIR / "index.json").write_text(json.dumps(full_manifest, indent=2))

    # Aggregate counts by label
    from collections import Counter
    by_label = Counter(item["label"] for item in full_manifest)
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total tables extracted: {len(full_manifest)}")
    print(f"By label:")
    for label, n in by_label.most_common():
        print(f"  {label}: {n}")
    print(f"\nManifest -> {OUT_DIR/'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
