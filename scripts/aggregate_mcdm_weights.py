"""Validate and aggregate MCDM literature weights.

Reads `data/processed/mcdm_weights/literature_weights.csv` (the human-edited
final file — drop the _WORKING / _TEMPLATE suffix when ready) and:

  1) VALIDATES per-paper integrity:
       - canonical_criterion is one of the 6 canonical names
       - weight is in [0, 1]
       - per-paper weights approximately sum to 1.0 (tolerance ±0.10)

  2) AGGREGATES across papers per canonical criterion:
       - count, mean, std, median, min, max
       - lists which papers contributed
       - flags criteria with < 3 papers ("low-evidence" warning)

  3) BUILDS the Fuzzy BWM input:
       - Triangular Fuzzy Number (TFN) per criterion: (mean-std, mean, mean+std)
       - normalized so middle values sum to 1.0
       - written to `data/processed/mcdm_weights/fuzzy_bwm_input.csv`

Usage (during extraction, run as a self-check):
    conda activate Eco-Research
    python scripts/aggregate_mcdm_weights.py --check       # validate only

Usage (final aggregation):
    python scripts/aggregate_mcdm_weights.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "data" / "processed" / "mcdm_weights"

# Try the final file first, fall back to working file
INPUT_CANDIDATES = [
    WEIGHTS_DIR / "literature_weights.csv",
    WEIGHTS_DIR / "literature_weights_WORKING.csv",
]
SUMMARY_CSV = WEIGHTS_DIR / "canonical_weights_summary.csv"
FUZZY_BWM_CSV = WEIGHTS_DIR / "fuzzy_bwm_input.csv"
REPORT_MD = WEIGHTS_DIR / "AGGREGATION_REPORT.md"

CANON_CRITERIA = ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"]
PER_PAPER_SUM_TOLERANCE = 0.10
WEIGHT_SUM_TARGET = 1.0
LOW_EVIDENCE_THRESHOLD = 3  # papers


def find_input() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        f"No weights file found. Expected one of:\n  " +
        "\n  ".join(str(p.relative_to(PROJECT_ROOT)) for p in INPUT_CANDIDATES)
    )


def parse_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            paper_id = (r.get("paper_id (lit_seq)") or "").strip()
            # Skip section-header rows (start with "#") and blank rows
            if not paper_id or paper_id.startswith("#"):
                continue
            canonical = (r.get("canonical_criterion (one of: SoH/Value/Carbon/Compliance/Safety/EPR Return)") or "").strip()
            weight_raw = (r.get("weight (0-1)") or "").strip()
            if not canonical and not weight_raw:
                continue  # incomplete row
            try:
                weight = float(weight_raw) if weight_raw else None
            except ValueError:
                weight = None
            rows.append({
                "paper_id": paper_id,
                "pdf": (r.get("pdf") or "").strip(),
                "year": (r.get("year") or "").strip(),
                "doi": (r.get("doi") or "").strip(),
                "raw_criterion": (r.get("paper_criterion_name (raw)") or "").strip(),
                "canonical": canonical,
                "weight": weight,
                "notes": (r.get("notes") or "").strip(),
            })
    return rows


def validate(rows: list[dict]) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    # Validate per-row canonical + weight bounds
    for r in rows:
        if r["canonical"] and r["canonical"] not in CANON_CRITERIA:
            errors.append(f"paper #{r['paper_id']}: invalid canonical_criterion '{r['canonical']}'. "
                          f"Must be one of: {', '.join(CANON_CRITERIA)}")
        if r["weight"] is not None:
            if not (0.0 <= r["weight"] <= 1.0):
                errors.append(f"paper #{r['paper_id']}: weight {r['weight']} outside [0,1] for '{r['canonical']}'")
        if r["canonical"] and r["weight"] is None:
            warnings.append(f"paper #{r['paper_id']}: canonical='{r['canonical']}' set but weight is empty")

    # Per-paper sum check
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["weight"] is not None and r["canonical"]:
            by_paper[r["paper_id"]].append(r)
    for paper_id, prs in by_paper.items():
        s = sum(r["weight"] for r in prs)
        # Many papers report sub-criterion weights that don't sum to 1 across our 6 canonical
        # criteria (because their criterion list is broader). We warn instead of error.
        if not (1.0 - PER_PAPER_SUM_TOLERANCE <= s <= 1.0 + PER_PAPER_SUM_TOLERANCE):
            warnings.append(f"paper #{paper_id}: weights sum to {s:.3f} (expected ~1.00 ± {PER_PAPER_SUM_TOLERANCE}). "
                            f"OK if paper covers only a subset of our 6 canonical criteria.")
    return errors, warnings


def aggregate(rows: list[dict]) -> dict:
    """Per-criterion aggregation across papers."""
    by_crit: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        if r["canonical"] and r["weight"] is not None:
            by_crit[r["canonical"]].append((r["paper_id"], r["weight"]))

    summary = {}
    for crit in CANON_CRITERIA:
        contribs = by_crit.get(crit, [])
        if not contribs:
            summary[crit] = {
                "n_papers": 0, "mean": None, "std": None,
                "median": None, "min": None, "max": None,
                "papers": [],
            }
            continue
        weights = [w for _, w in contribs]
        s = {
            "n_papers": len(contribs),
            "mean": round(mean(weights), 5),
            "std": round(stdev(weights) if len(weights) >= 2 else 0.0, 5),
            "median": round(median(weights), 5),
            "min": round(min(weights), 5),
            "max": round(max(weights), 5),
            "papers": sorted({p for p, _ in contribs}),
        }
        summary[crit] = s
    return summary


def build_fuzzy_input(summary: dict) -> list[dict]:
    """TFN = (mean-std, mean, mean+std), then normalize middle to sum=1.0."""
    rows = []
    middles = []
    for crit in CANON_CRITERIA:
        s = summary[crit]
        if s["n_papers"] == 0:
            rows.append({
                "criterion": crit, "n_papers": 0,
                "tfn_lower": None, "tfn_middle": None, "tfn_upper": None,
                "tfn_lower_normalized": None, "tfn_middle_normalized": None, "tfn_upper_normalized": None,
            })
            middles.append(0.0)
            continue
        mu, sd = s["mean"], s["std"]
        lo = max(0.0, mu - sd)
        hi = min(1.0, mu + sd)
        rows.append({
            "criterion": crit, "n_papers": s["n_papers"],
            "tfn_lower": round(lo, 5), "tfn_middle": round(mu, 5), "tfn_upper": round(hi, 5),
        })
        middles.append(mu)

    total_mid = sum(middles) or 1.0
    for i, r in enumerate(rows):
        if r["tfn_middle"] is None:
            r["tfn_lower_normalized"] = None
            r["tfn_middle_normalized"] = None
            r["tfn_upper_normalized"] = None
            continue
        scale = 1.0 / total_mid
        r["tfn_lower_normalized"] = round(r["tfn_lower"] * scale, 5)
        r["tfn_middle_normalized"] = round(r["tfn_middle"] * scale, 5)
        r["tfn_upper_normalized"] = round(r["tfn_upper"] * scale, 5)
    return rows


def write_summary_csv(summary: dict) -> None:
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_criterion", "n_papers", "mean", "std", "median",
                    "min", "max", "papers_listed"])
        for crit in CANON_CRITERIA:
            s = summary[crit]
            w.writerow([
                crit, s["n_papers"], s["mean"], s["std"],
                s["median"], s["min"], s["max"],
                "; ".join(f"#{p}" for p in s["papers"]),
            ])
    print(f"  wrote {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")


def write_fuzzy_csv(fuzzy_rows: list[dict]) -> None:
    fields = ["criterion", "n_papers",
              "tfn_lower", "tfn_middle", "tfn_upper",
              "tfn_lower_normalized", "tfn_middle_normalized", "tfn_upper_normalized"]
    with open(FUZZY_BWM_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fuzzy_rows:
            w.writerow(r)
    print(f"  wrote {FUZZY_BWM_CSV.relative_to(PROJECT_ROOT)}")


def write_report_md(input_path: Path, summary: dict, errors: list[str],
                    warnings: list[str], n_rows: int, fuzzy_rows: list[dict]) -> None:
    n_papers = len({r["papers"][i] for r in summary.values() if r["papers"] for i in range(len(r["papers"]))})
    contrib_papers = set()
    for crit_summary in summary.values():
        contrib_papers.update(crit_summary["papers"])
    n_papers = len(contrib_papers)

    low_evidence = [c for c, s in summary.items() if 0 < s["n_papers"] < LOW_EVIDENCE_THRESHOLD]
    no_evidence = [c for c, s in summary.items() if s["n_papers"] == 0]

    lines = [
        "# MCDM Literature Weights — Aggregation Report",
        "",
        f"_Generated by `scripts/aggregate_mcdm_weights.py` from `{input_path.relative_to(PROJECT_ROOT)}`._",
        "",
        "## Summary",
        "",
        f"- Input rows parsed: **{n_rows}**",
        f"- Papers contributing weights: **{n_papers}** of 11 T2 papers",
        f"- Errors: **{len(errors)}**",
        f"- Warnings: **{len(warnings)}**",
        f"- Criteria with `< {LOW_EVIDENCE_THRESHOLD}` papers (low-evidence): {', '.join(low_evidence) or '—'}",
        f"- Criteria with **no** evidence: {', '.join(no_evidence) or '—'}",
        "",
        "## Per-criterion aggregation",
        "",
        "| Criterion | n papers | mean | std | min | max | papers |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for crit in CANON_CRITERIA:
        s = summary[crit]
        if s["n_papers"] == 0:
            lines.append(f"| {crit} | 0 | — | — | — | — | — |")
        else:
            papers_str = ", ".join(f"#{p}" for p in s["papers"][:8])
            if len(s["papers"]) > 8:
                papers_str += f" (+{len(s['papers']) - 8} more)"
            lines.append(f"| {crit} | {s['n_papers']} | {s['mean']:.3f} | {s['std']:.3f} | "
                         f"{s['min']:.3f} | {s['max']:.3f} | {papers_str} |")

    lines.append("")
    lines.append("## Triangular Fuzzy Numbers (Fuzzy BWM input)")
    lines.append("")
    lines.append("| Criterion | n | TFN raw (lower, middle, upper) | TFN normalized (sum middles = 1) |")
    lines.append("|---|---:|---|---|")
    for r in fuzzy_rows:
        if r["tfn_middle"] is None:
            lines.append(f"| {r['criterion']} | 0 | — | — |")
        else:
            raw = f"({r['tfn_lower']:.3f}, {r['tfn_middle']:.3f}, {r['tfn_upper']:.3f})"
            norm = f"({r['tfn_lower_normalized']:.3f}, {r['tfn_middle_normalized']:.3f}, {r['tfn_upper_normalized']:.3f})"
            lines.append(f"| {r['criterion']} | {r['n_papers']} | {raw} | {norm} |")

    if errors:
        lines.append("")
        lines.append("## ❌ Errors (must fix)")
        for e in errors:
            lines.append(f"- {e}")
    if warnings:
        lines.append("")
        lines.append("## ⚠️ Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Acceptance gate (per DATASET_IMPLEMENTATION_PLAN §7.4)")
    lines.append("")
    gate_pass = (
        len(errors) == 0
        and not no_evidence
        and not low_evidence
    )
    lines.append(f"- ≥ 3 papers per criterion: **{'PASS' if not low_evidence and not no_evidence else 'FAIL'}**")
    lines.append(f"- All canonical names valid: **{'PASS' if len([e for e in errors if 'invalid canonical' in e]) == 0 else 'FAIL'}**")
    lines.append(f"- Per-paper weight bounds [0,1]: **{'PASS' if len([e for e in errors if 'outside [0,1]' in e]) == 0 else 'FAIL'}**")
    lines.append(f"- TFN ordering (lower ≤ middle ≤ upper): **{'PASS' if all(r['tfn_lower'] is None or r['tfn_lower'] <= r['tfn_middle'] <= r['tfn_upper'] for r in fuzzy_rows) else 'FAIL'}**")
    lines.append(f"- Normalized middles sum ≈ 1.0: **{'PASS' if abs(sum(r['tfn_middle_normalized'] or 0 for r in fuzzy_rows) - 1.0) < 0.001 else 'FAIL'}**")
    lines.append("")
    lines.append(f"**OVERALL: {'✅ READY FOR FUZZY BWM' if gate_pass else '❌ FIX ITEMS ABOVE BEFORE PROCEEDING'}**")

    REPORT_MD.write_text("\n".join(lines))
    print(f"  wrote {REPORT_MD.relative_to(PROJECT_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Validate only (no aggregation outputs)")
    args = ap.parse_args()

    src = find_input()
    print(f"Reading {src.relative_to(PROJECT_ROOT)}")
    rows = parse_rows(src)
    print(f"  {len(rows)} weight rows parsed")

    errors, warnings = validate(rows)
    print(f"  validation: {len(errors)} errors, {len(warnings)} warnings")
    for e in errors[:10]:
        print(f"    ❌ {e}")
    for w in warnings[:6]:
        print(f"    ⚠️  {w}")
    if len(errors) > 10:
        print(f"    ... +{len(errors) - 10} more errors")

    summary = aggregate(rows)
    fuzzy_rows = build_fuzzy_input(summary)

    if args.check:
        # Print summary only; no file outputs
        print("\nPer-criterion coverage so far:")
        for crit in CANON_CRITERIA:
            s = summary[crit]
            if s["n_papers"] == 0:
                print(f"  {crit:12s}  ❌ no evidence")
            elif s["n_papers"] < LOW_EVIDENCE_THRESHOLD:
                print(f"  {crit:12s}  ⚠️  {s['n_papers']} papers (low evidence; need ≥ {LOW_EVIDENCE_THRESHOLD})")
            else:
                print(f"  {crit:12s}  ✅ {s['n_papers']} papers, mean={s['mean']:.3f} ± {s['std']:.3f}")
        return 0 if not errors else 1

    # Full aggregation outputs
    write_summary_csv(summary)
    write_fuzzy_csv(fuzzy_rows)
    write_report_md(src, summary, errors, warnings, len(rows), fuzzy_rows)

    print(f"\n{'✅ Aggregation complete.' if not errors else '❌ Aggregation produced errors. Fix and re-run.'}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
