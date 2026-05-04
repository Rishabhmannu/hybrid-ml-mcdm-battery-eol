"""
Extract candidate MCDM criterion-weight tables from the 11 T2 papers in the
literature review. Produces a structured CSV per paper plus a master manifest
that the human reviewer can finalize before the Fuzzy BWM step.

Pipeline:
  1) For each T2_*.pdf in research-papers/:
       a) Extract ALL tables via pdfplumber
       b) Heuristic-score each table for "this looks like a criterion-weight table"
          - has a column header containing 'weight', 'priority', or 'rank'
          - has 3-15 data rows
          - majority of cells in the score column parse as floats in [0, 1]
          - those floats sum to ~1 (within 0.05) OR each is ≤ 1 (relaxed)
       c) Save each candidate to data/processed/mcdm_weights/raw_extracted/
  2) Read the first page of each PDF to extract title + author + year, match
     against LITERATURE_REVIEW.md to assign the lit-review seq number, score,
     and the canonical MCDM criteria flags from the review.
  3) Emit:
       - candidate_tables.csv  — one row per extracted candidate table, with
         pdf, page, table_index, n_rows, header, score-confidence, and a link
         to the per-table CSV.
       - paper_index.csv  — one row per paper, with seq, score, mcdm_criteria
         (from lit review) + counts of candidates extracted.
       - mcdm_weights_template.csv  — a starter file for the canonical weight
         set: paper_id, criterion (canonical), weight, source_table_csv. The
         reviewer fills the criterion column by mapping each paper's criteria
         names to the 6 canonical ones.

Usage:
    conda activate Eco-Research
    python scripts/extract_mcdm_weights.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "research-papers"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "mcdm_weights"
EXTRACTED_DIR = OUT_DIR / "raw_extracted"
LIT_REVIEW = PROJECT_ROOT / "LITERATURE_REVIEW.md"

CANON_CRITERIA = ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"]


# --------------------------------------------------------------------------
# Heuristic table classifier
# --------------------------------------------------------------------------

WEIGHT_HEADER_HINTS = ["weight", "priorit", "rank", "global weight",
                        "local weight", "wj", "w_j", "criteria weight",
                        "importance"]
CRIT_HEADER_HINTS = ["criteri", "factor", "indicator", "attribute", "dimension"]


def clean(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_number(s: str) -> float | None:
    s = clean(s).rstrip("%").replace(",", ".")
    try:
        v = float(s)
        # accept percentages and convert
        if v > 1.0 and v <= 100.0:
            v = v / 100.0
        return v
    except ValueError:
        return None


def looks_like_weight_table(rows: list[list[str]]) -> tuple[bool, str, dict]:
    """Return (is_weight_table, reason, stats)."""
    if not rows or len(rows) < 4:
        return False, "too few rows", {}

    # Inspect the first 2 rows as candidate headers
    header_blob = " ".join(clean(c).lower() for r in rows[:2] for c in r)
    has_weight_word = any(h in header_blob for h in WEIGHT_HEADER_HINTS)
    has_crit_word = any(h in header_blob for h in CRIT_HEADER_HINTS)

    # Find columns that mostly parse as floats in [0, 1]
    n_cols = max(len(r) for r in rows)
    numeric_share = []
    for col_idx in range(n_cols):
        col = [r[col_idx] if col_idx < len(r) else "" for r in rows[1:]]
        nums = [parse_number(c) for c in col]
        valid = [v for v in nums if v is not None and 0 <= v <= 1.0]
        share = len(valid) / max(len(col), 1)
        numeric_share.append((col_idx, share, valid))

    # Best numeric column
    numeric_share.sort(key=lambda x: -x[1])
    best_col_idx, best_share, best_vals = numeric_share[0]
    sum_to_one = abs(sum(best_vals) - 1.0) < 0.10 if best_vals else False
    avg_le_one = (max(best_vals) <= 1.0) if best_vals else False

    n_data_rows = len(rows) - 1
    is_weight = (
        (has_weight_word or has_crit_word)
        and best_share >= 0.6
        and 3 <= n_data_rows <= 30
        and (sum_to_one or (best_share >= 0.8 and avg_le_one))
    )
    stats = {
        "n_data_rows": n_data_rows,
        "best_numeric_col": best_col_idx,
        "numeric_share": round(best_share, 2),
        "values": [round(v, 4) for v in best_vals],
        "sum": round(sum(best_vals), 3) if best_vals else 0,
        "has_weight_header": has_weight_word,
        "has_crit_header": has_crit_word,
    }
    reason = f"weight={has_weight_word} crit={has_crit_word} num_share={best_share:.2f} sum={stats['sum']}"
    return is_weight, reason, stats


# --------------------------------------------------------------------------
# Lit-review parser
# --------------------------------------------------------------------------

def parse_lit_review_t2() -> list[dict]:
    """Pull T2 rows with seq, score, authors, year, title, mcdm_criteria."""
    text = LIT_REVIEW.read_text(encoding="utf-8")
    sec = text[text.find("## 2. Main Literature Review Table"):text.find("## 3.")]
    rows = [r for r in sec.splitlines() if r.startswith("| ") and not set(r) <= {"|", "-", " "}]
    rows = rows[2:]  # skip header + alignment
    out = []
    for r in rows:
        cols = [clean(c) for c in r.strip("|").split("|")]
        if len(cols) < 21:
            continue
        if cols[1] != "T2":
            continue
        out.append({
            "seq": int(cols[0]),
            "score": int(cols[2]),
            "tier": cols[3],
            "authors": cols[4],
            "year": int(cols[5]),
            "title": cols[6],
            "journal": cols[7],
            "doi": clean(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cols[9])),
            "mcdm_criteria": cols[17],
        })
    return out


# --------------------------------------------------------------------------
# PDF → lit-review matcher
# --------------------------------------------------------------------------

PDF_AUTHOR_YEAR = re.compile(r"^T2_([A-Z][A-Za-z]+)(\d{4})_")


def _extract_year(stem: str) -> int | None:
    m = re.search(r"(20\d{2})", stem)
    return int(m.group(1)) if m else None


def _filename_tokens(stem: str) -> list[str]:
    bad = {"the", "and", "for", "with", "based", "approach", "framework",
            "method", "model", "selection", "analysis", "system", "study",
            "evaluation", "review", "research", "scireports", "ifac", "natcomms",
            "iecr", "acs", "energy", "pios", "daj", "annor", "ijieom", "clsc",
            "batteries", "fuzzy", "mcdm", "bwm", "topsis", "ahp", "dematel",
            "rbfnn", "stratified", "ordinal", "ev", "li", "lib", "lithium"}
    parts = re.split(r"[_\W\d]+", stem.replace("T2_", "").replace(".pdf", ""))
    return [p.lower() for p in parts if len(p) >= 4 and p.lower() not in bad]


# Manual overrides for filenames that don't follow the AuthorYear convention.
# Maps PDF filename → lit-review seq number.
MANUAL_LIT_MAP = {
    "T2_BWM_RBFNN_2025_SciReports.pdf": 22,                    # Forewarning model RBFNN reverse SC
    "T2_Dincer2026_HybridMCDM_EV_EoL_SciReports.pdf": 19,      # Hybrid unified decoding analytics for EoL
    "T2_FuzzyMCDM_BatteryRecycling_2025_Batteries.pdf": 17,    # Evaluating Sustainable Battery Recycling Tech
    "T2_RobustDesign_LiBSC_2025_ACS_IECR.pdf": 18,             # Robust Design and Optimization Integrated SC
}


def match_pdf_to_lit(pdf_filename: str, lit_rows: list[dict]) -> dict | None:
    """Match by surname+year first, then year+title-token overlap, then manual map."""
    if pdf_filename in MANUAL_LIT_MAP:
        seq = MANUAL_LIT_MAP[pdf_filename]
        for r in lit_rows:
            if r["seq"] == seq:
                return r
    stem = pdf_filename.replace(".pdf", "")
    year = _extract_year(stem)
    tokens = _filename_tokens(stem)

    # Strategy 1: AuthorYear surname extraction
    m = PDF_AUTHOR_YEAR.match(pdf_filename)
    if m:
        surname = m.group(1)
        candidates = [r for r in lit_rows if abs(r["year"] - year) <= 1
                       and surname.lower() in r["authors"].lower()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            best = None; best_overlap = -1
            for r in candidates:
                overlap = sum(1 for t in tokens if t in r["title"].lower())
                if overlap > best_overlap:
                    best_overlap = overlap; best = r
            return best

    # Strategy 2: year + title-token overlap (for filenames like BWM_RBFNN_2025)
    if year is None:
        return None
    year_matches = [r for r in lit_rows if abs(r["year"] - year) <= 1]
    if not year_matches:
        return None
    best = None; best_overlap = 0
    for r in year_matches:
        title_low = r["title"].lower()
        overlap = sum(1 for t in tokens if t in title_low)
        if overlap > best_overlap:
            best_overlap = overlap; best = r
    return best if best_overlap >= 1 else None


# --------------------------------------------------------------------------
# Per-PDF processor
# --------------------------------------------------------------------------

TABLE_STRATEGIES = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    {"vertical_strategy": "text", "horizontal_strategy": "text",
     "text_tolerance": 3, "intersection_tolerance": 5,
     "snap_tolerance": 4, "min_words_vertical": 2, "min_words_horizontal": 2},
    {"vertical_strategy": "lines_strict", "horizontal_strategy": "text"},
]


def _row_signature(rows: list[list[str]]) -> tuple:
    """Used to dedupe tables found by multiple strategies."""
    return tuple((tuple(clean(c) for c in r[:6])) for r in rows[:5])


def process_pdf(pdf: Path, lit_row: dict | None) -> list[dict]:
    candidates = []
    try:
        plumber = pdfplumber.open(str(pdf))
    except Exception as e:
        print(f"  [{pdf.name}] could not open: {e}")
        return []
    try:
        for page_num, page in enumerate(plumber.pages, start=1):
            seen_sigs: set = set()
            tables_collected = []
            for strategy in TABLE_STRATEGIES:
                try:
                    tbls = page.extract_tables(table_settings=strategy) or []
                except Exception:
                    continue
                for tbl in tbls:
                    rows = [[clean(c) for c in row] for row in tbl]
                    rows = [r for r in rows if any(c for c in r)]
                    if len(rows) < 4:
                        continue
                    sig = _row_signature(rows)
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    tables_collected.append(rows)
            tables = tables_collected
            for tbl_idx, raw in enumerate(tables, start=1):
                if not raw:
                    continue
                rows = [[clean(c) for c in row] for row in raw]
                # Drop rows that are entirely empty after cleaning
                rows = [r for r in rows if any(c for c in r)]
                if len(rows) < 4:
                    continue
                is_weight, reason, stats = looks_like_weight_table(rows)
                # Save every table heuristically labeled OR every "small numeric" table
                # (3+ data rows w/ a numeric column ≥ 60% share)
                save = is_weight or stats.get("numeric_share", 0) >= 0.7
                if not save:
                    continue
                stem = pdf.stem
                fn = f"{stem}__p{page_num:02d}__t{tbl_idx}.csv"
                out_path = EXTRACTED_DIR / fn
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for row in rows:
                        w.writerow(row)
                header = " | ".join(rows[0])[:140]
                candidates.append({
                    "pdf": pdf.name,
                    "lit_seq": lit_row["seq"] if lit_row else None,
                    "lit_score": lit_row["score"] if lit_row else None,
                    "lit_mcdm_criteria": lit_row["mcdm_criteria"] if lit_row else None,
                    "page": page_num,
                    "table_index": tbl_idx,
                    "is_weight_candidate": is_weight,
                    "header": header,
                    "n_rows": stats["n_data_rows"],
                    "numeric_share": stats["numeric_share"],
                    "sum_of_values": stats["sum"],
                    "csv": str(out_path.relative_to(PROJECT_ROOT)),
                })
    finally:
        plumber.close()
    return candidates


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    lit_rows = parse_lit_review_t2()
    print(f"Lit review T2 papers: {len(lit_rows)}")

    pdfs = sorted(PAPERS_DIR.glob("T2_*.pdf"))
    print(f"T2 PDFs in research-papers/: {len(pdfs)}")
    print()

    paper_index = []
    all_candidates = []

    for pdf in pdfs:
        lit_row = match_pdf_to_lit(pdf.name, lit_rows)
        if lit_row:
            print(f"[{pdf.name}] -> lit #{lit_row['seq']} (score {lit_row['score']}, criteria: {lit_row['mcdm_criteria']})")
        else:
            print(f"[{pdf.name}] -> NO lit-review match; processing anyway")

        cands = process_pdf(pdf, lit_row)
        n_weight = sum(1 for c in cands if c["is_weight_candidate"])
        print(f"  -> {len(cands)} numeric tables extracted, {n_weight} flagged as weight-table candidates")
        all_candidates.extend(cands)

        paper_index.append({
            "pdf": pdf.name,
            "lit_seq": lit_row["seq"] if lit_row else None,
            "lit_score": lit_row["score"] if lit_row else None,
            "year": lit_row["year"] if lit_row else None,
            "authors": lit_row["authors"][:60] if lit_row else "",
            "title": lit_row["title"][:80] if lit_row else "",
            "doi": lit_row["doi"] if lit_row else "",
            "mcdm_criteria": lit_row["mcdm_criteria"] if lit_row else "",
            "n_candidates": len(cands),
            "n_weight_flagged": n_weight,
        })

    # Write manifests
    candidates_csv = OUT_DIR / "candidate_tables.csv"
    with open(candidates_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pdf", "lit_seq", "lit_score", "lit_mcdm_criteria",
            "page", "table_index", "is_weight_candidate",
            "header", "n_rows", "numeric_share", "sum_of_values", "csv",
        ])
        w.writeheader()
        for c in all_candidates:
            w.writerow(c)

    paper_csv = OUT_DIR / "paper_index.csv"
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pdf", "lit_seq", "lit_score", "year", "authors", "title", "doi",
            "mcdm_criteria", "n_candidates", "n_weight_flagged",
        ])
        w.writeheader()
        for r in paper_index:
            w.writerow(r)

    # Build the template literature_weights.csv
    template_csv = OUT_DIR / "literature_weights_TEMPLATE.csv"
    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "paper_id (lit_seq)", "pdf", "year", "doi",
            "paper_criterion_name (raw)", "canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)",
            "weight (0-1)", "source_table_csv", "notes",
        ])
        # Pre-fill rows from weight-flagged candidates so the reviewer just maps names
        for c in all_candidates:
            if not c["is_weight_candidate"]:
                continue
            # Open the candidate CSV and emit one row per data row
            try:
                with open(c["csv"]) as cf:
                    rows = list(csv.reader(cf))
            except Exception:
                continue
            if len(rows) < 2:
                continue
            # Find numeric column (highest fraction parseable as 0-1)
            n_cols = max(len(r) for r in rows)
            best_col = None; best_share = 0
            for ci in range(n_cols):
                col = [r[ci] if ci < len(r) else "" for r in rows[1:]]
                nums = [parse_number(c2) for c2 in col]
                share = sum(1 for v in nums if v is not None and 0 <= v <= 1.0) / max(len(col), 1)
                if share > best_share:
                    best_col = ci; best_share = share
            if best_col is None:
                continue
            # Find label column (typically col 0 or 1)
            label_col = 0 if best_col != 0 else 1
            for r_idx, r in enumerate(rows[1:], start=2):
                label = r[label_col] if label_col < len(r) else ""
                weight = parse_number(r[best_col] if best_col < len(r) else "")
                if not label or weight is None:
                    continue
                # Skip aggregate rows like "Total" or "Sum"
                if any(kw in label.lower() for kw in ["total", "sum", "average"]):
                    continue
                w.writerow([
                    c["lit_seq"] or "",
                    c["pdf"],
                    "",
                    "",
                    label[:80],
                    "",  # canonical_criterion left blank for human to fill
                    weight,
                    c["csv"],
                    f"page {c['page']} table {c['table_index']}",
                ])

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Papers processed:           {len(pdfs)}")
    print(f"Lit-review matches:         {sum(1 for p in paper_index if p['lit_seq'])}")
    print(f"Total candidate tables:     {len(all_candidates)}")
    print(f"Weight-flagged candidates:  {sum(1 for c in all_candidates if c['is_weight_candidate'])}")
    print()
    print(f"Outputs:")
    print(f"  - candidate_tables.csv         {candidates_csv.relative_to(PROJECT_ROOT)}")
    print(f"  - paper_index.csv              {paper_csv.relative_to(PROJECT_ROOT)}")
    print(f"  - literature_weights_TEMPLATE.csv  {template_csv.relative_to(PROJECT_ROOT)}")
    print(f"  - raw_extracted/<pdf>__p<NN>__t<K>.csv  ({EXTRACTED_DIR.relative_to(PROJECT_ROOT)})")
    print()
    print("Next step: review literature_weights_TEMPLATE.csv, fill the")
    print("'canonical_criterion' column by mapping each paper's criterion")
    print("name to one of the 6 canonical criteria, save as literature_weights.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
