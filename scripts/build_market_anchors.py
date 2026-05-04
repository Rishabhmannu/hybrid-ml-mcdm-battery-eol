"""Build MCDM market anchors from L4 sources:
  1) CPCB EPR national dashboard scrape (already in data/raw/cpcb_epr/)
  2) ICEA-Accenture "Charging Ahead" 2025 report — pull projections via pdfplumber
  3) Wire to MCDM C2 (Value) and C6 (EPR Return) criteria

Outputs:
  data/processed/market/cpcb_metal_epr_table.csv
  data/processed/market/icea_projections.csv
  data/processed/market/mcdm_anchors_summary.md (human-readable)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data/processed/market"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CPCB_TEXT = PROJECT_ROOT / "data/raw/cpcb_epr/dashboard_text.txt"
ICEA_PDF = PROJECT_ROOT / "data/regulatory/india_market/ICEA_Accenture_Charging_Ahead_2025.pdf"


# ---------------- CPCB metal-wise EPR table ----------------

METALS = ["lead", "lithium", "nickel", "cadmium", "cobalt", "manganese",
          "iron", "aluminium", "copper"]


def parse_cpcb_metals() -> tuple[list[dict], list[dict]]:
    """Two tables on the CPCB dashboard:
      Table 1: Metal-wise EPR Targets (Target, Procured) per metal
      Table 2: Metal-wise EPR Credits Available (Target, Procured, Available)
    """
    if not CPCB_TEXT.exists():
        return [], []
    text = CPCB_TEXT.read_text(encoding="utf-8")

    # Find the two table headers
    table1_idx = text.find("EPR Target (in Tonnes)")
    table2_idx = text.find("EPR Credits Available (in tonnes)")

    table1_text = text[table1_idx:table2_idx] if table1_idx != -1 else ""
    table2_text = text[table2_idx:] if table2_idx != -1 else ""

    def extract_metal_rows(blob: str, n_cols: int) -> list[dict]:
        """Each row: <metal> <num> <num> [<num>]. Find them all."""
        out = []
        # Capture metal + n_cols numbers
        num = r"([\d,]+\.?\d*)"
        for metal in METALS:
            # Match metal name (case-insensitive) followed by numbers
            pattern = re.compile(
                rf"\b{metal}\b\s+" + r"\s+".join([num] * n_cols),
                re.IGNORECASE,
            )
            m = pattern.search(blob)
            if m:
                vals = [float(g.replace(",", "")) for g in m.groups()]
                row = {"metal": metal.capitalize()}
                if n_cols == 2:
                    row["epr_target_tonnes"] = vals[0]
                    row["epr_credits_procured_tonnes"] = vals[1]
                elif n_cols == 3:
                    row["epr_target_tonnes"] = vals[0]
                    row["epr_credits_procured_tonnes"] = vals[1]
                    row["epr_credits_available_tonnes"] = vals[2]
                out.append(row)
        return out

    t1 = extract_metal_rows(table1_text, 2)
    t2 = extract_metal_rows(table2_text, 3)
    return t1, t2


def write_cpcb_anchors():
    t1, t2 = parse_cpcb_metals()
    out_csv = OUT_DIR / "cpcb_metal_epr_table.csv"
    if not t1 and not t2:
        print("[CPCB] no rows parsed")
        return None
    # Merge tables on metal (table 2 supersedes since it has the available column)
    combined: dict[str, dict] = {}
    for r in t1:
        combined[r["metal"]] = dict(r)
    for r in t2:
        if r["metal"] in combined:
            combined[r["metal"]].update(r)
        else:
            combined[r["metal"]] = dict(r)
    rows = list(combined.values())
    fields = ["metal", "epr_target_tonnes", "epr_credits_procured_tonnes",
              "epr_credits_available_tonnes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[CPCB] wrote {out_csv.relative_to(PROJECT_ROOT)} ({len(rows)} metals)")
    return rows


# ---------------- ICEA projections ----------------

def extract_icea_projections() -> list[dict]:
    """Mine the ICEA-Accenture PDF for headline projections.

    Looks for: 'XX kt' / 'XX tonnes' / 'XX million' patterns near 2030.
    Falls back to extracting text and saving the first 3 pages verbatim if no
    specific patterns hit — the human reviewer can then read the report for
    detailed numbers.
    """
    try:
        import pdfplumber
    except ImportError:
        print("[ICEA] pdfplumber not available")
        return []

    if not ICEA_PDF.exists():
        print(f"[ICEA] {ICEA_PDF.relative_to(PROJECT_ROOT)} not found")
        return []

    out = []
    full_text_pages = []
    with pdfplumber.open(str(ICEA_PDF)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            full_text_pages.append(txt)

    full_text = "\n".join(full_text_pages)

    # Pattern 1: "X tonnes by 20YY" / "X kt by 20YY" / "X million tonnes"
    patterns = [
        (r"(\d[\d,\.]*\s*(?:million|kt|tonnes?|MT|t/year|tonnes per year))[^.\n]{0,80}?\bby\s+(20\d{2})",
         "growth_target"),
        (r"(\d[\d,\.]*\s*(?:million|kt|tonnes?|MT|t/year|tonnes per year))[^.\n]{0,40}?\bin\s+(20\d{2})",
         "growth_in_year"),
        (r"(\d{1,3}(?:,\d{3})*)\s*(?:tonnes?|t)\s+(?:to|→)\s*(\d{1,3}(?:[,\.]?\d+)*)\s*(?:million|MT)",
         "growth_range"),
    ]

    for regex, kind in patterns:
        for m in re.finditer(regex, full_text, flags=re.IGNORECASE):
            snippet = full_text[max(0, m.start() - 40): min(len(full_text), m.end() + 40)]
            snippet = re.sub(r"\s+", " ", snippet).strip()
            out.append({
                "kind": kind,
                "match": m.group(0).strip(),
                "context_snippet": snippet[:240],
                "groups": " | ".join(g for g in m.groups() if g),
            })

    # Dedupe by match string
    seen = set()
    unique = []
    for r in out:
        key = (r["kind"], r["match"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def write_icea_projections():
    rows = extract_icea_projections()
    out_csv = OUT_DIR / "icea_projections.csv"
    if not rows:
        print("[ICEA] no projection patterns hit; saving raw text page extract instead")
        # Fallback: save the raw text for manual reading
        try:
            import pdfplumber
            with pdfplumber.open(str(ICEA_PDF)) as pdf:
                txt = "\n\n".join((p.extract_text() or "") for p in pdf.pages[:6])
            (OUT_DIR / "icea_first6_pages.txt").write_text(txt, encoding="utf-8")
            print(f"[ICEA] wrote first 6 pages to icea_first6_pages.txt for manual review")
        except Exception as exc:
            print(f"[ICEA] fallback extraction failed: {exc}")
        return []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "match", "groups", "context_snippet"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ICEA] wrote {out_csv.relative_to(PROJECT_ROOT)} ({len(rows)} projection candidates)")
    return rows


# ---------------- MCDM mapping summary ----------------

def write_summary(cpcb_rows, icea_rows):
    out_md = OUT_DIR / "mcdm_anchors_summary.md"
    lines = ["# MCDM Market Anchors Summary\n",
             "_Generated by `scripts/build_market_anchors.py`. "
             "Companion to `data/processed/mcdm_weights/literature_weights.csv` "
             "for the Fuzzy BWM step._\n",
             "## CPCB Metal-wise EPR Targets (India)\n"]
    if cpcb_rows:
        lines.append("| Metal | EPR Target (t) | Credits Procured (t) | Credits Available (t) |")
        lines.append("|---|---:|---:|---:|")
        for r in cpcb_rows:
            tgt = r.get("epr_target_tonnes", "—")
            proc = r.get("epr_credits_procured_tonnes", "—")
            avail = r.get("epr_credits_available_tonnes", "—")
            lines.append(f"| {r.get('metal','—')} | {tgt} | {proc} | {avail} |")
        lines.append("")
        lines.append("**Maps to MCDM:** C6 (EPR Return) — recovery-volume anchors per metal. ")
        lines.append("Lithium/Cobalt/Nickel/Manganese matter most for EV battery routing decisions.\n")
    else:
        lines.append("_No CPCB rows parsed._\n")

    lines.append("## ICEA-Accenture Projections (2025 report)\n")
    if icea_rows:
        lines.append("| Kind | Match | Snippet |")
        lines.append("|---|---|---|")
        for r in icea_rows[:25]:
            lines.append(f"| {r['kind']} | `{r['match'][:50]}` | {r['context_snippet'][:100]} |")
        lines.append("")
        lines.append("**Maps to MCDM:** C2 (Value) — battery-waste market growth informs the value-stream sizing for refurbishment vs recycling routes.\n")
    else:
        lines.append("_No projections auto-extracted; see `icea_first6_pages.txt` for manual review._\n")

    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md.relative_to(PROJECT_ROOT)}")


def main() -> int:
    cpcb_rows = write_cpcb_anchors() or []
    icea_rows = write_icea_projections() or []
    write_summary(cpcb_rows, icea_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
