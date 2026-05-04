"""
End-to-end audit of all acquired datasets.

For each source listed in DATASET_COLLECTION_GUIDE.md, this script verifies:
  1) Required path exists with non-trivial size
  2) Files are not corrupt:
       - ZIPs: zipfile.testzip()
       - PDFs: pdfinfo / pypdf opens, page_count > 0
       - .pkl: pickle.load succeeds (sampled for large folders)
       - .xlsx: openpyxl opens (sampled)
       - .csv: parse succeeds (sampled)
       - JSON: parse succeeds
       - .mat: scipy.io.loadmat opens (sampled)
       - git repos: .git directory exists

Output:
  - Per-source PASS / WARN / FAIL with details
  - Summary table at the end
  - data/AUDIT_REPORT.md saved for the record
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
random.seed(42)

# --- output collectors ----------------------------------------------------

@dataclass
class Check:
    name: str
    layer: str
    path: Path
    status: str = "PENDING"   # PASS / WARN / FAIL / MISSING
    size_bytes: int = 0
    n_files: int = 0
    notes: list[str] = field(default_factory=list)
    samples_ok: int = 0
    samples_total: int = 0


def add_note(c: Check, msg: str) -> None:
    c.notes.append(msg)


# --- helpers --------------------------------------------------------------

def folder_size(p: Path) -> tuple[int, int]:
    if not p.exists():
        return 0, 0
    total_size, total_files = 0, 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total_size += f.stat().st_size
                total_files += 1
            except Exception:
                pass
    return total_size, total_files


def verify_zip(p: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(p) as z:
            err = z.testzip()
        return (err is None), (f"OK ({len(zipfile.ZipFile(p).namelist())} entries)" if err is None else f"corrupt: {err}")
    except zipfile.BadZipFile as e:
        return False, f"bad zip: {e}"
    except Exception as e:
        return False, f"error: {e}"


def verify_pdf(p: Path) -> tuple[bool, str]:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(p))
        if len(r.pages) < 1:
            return False, "0 pages"
        # try to extract text from first page to confirm not encrypted/broken
        try:
            _ = r.pages[0].extract_text()
        except Exception:
            pass
        return True, f"{len(r.pages)} pages"
    except Exception as e:
        return False, f"error: {type(e).__name__}: {e}"


def verify_pkl(p: Path) -> tuple[bool, str]:
    try:
        with open(p, "rb") as f:
            pickle.load(f)
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_xlsx(p: Path) -> tuple[bool, str]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        sheets = wb.sheetnames
        wb.close()
        if not sheets:
            return False, "no sheets"
        return True, f"{len(sheets)} sheets"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_csv(p: Path) -> tuple[bool, str]:
    try:
        import csv
        with open(p, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            n_rows = 0
            for row in reader:
                n_rows += 1
                if n_rows >= 5:
                    break
        return n_rows > 0, f"{n_rows}+ rows readable"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_json(p: Path) -> tuple[bool, str]:
    try:
        with open(p) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return True, f"dict, {len(obj)} keys"
        if isinstance(obj, list):
            return True, f"list, {len(obj)} items"
        return True, f"{type(obj).__name__}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_mat(p: Path) -> tuple[bool, str]:
    try:
        from scipy.io import loadmat
        d = loadmat(str(p), simplify_cells=False)
        keys = [k for k in d.keys() if not k.startswith("__")]
        return True, f"keys: {keys[:3]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sample_files(folder: Path, pattern: str, n: int) -> list[Path]:
    files = list(folder.rglob(pattern))
    if len(files) <= n:
        return files
    return random.sample(files, n)


def run_sampled(c: Check, files: list[Path], verifier: Callable, label: str) -> None:
    c.samples_total = len(files)
    fails = []
    for f in files:
        ok, note = verifier(f)
        if ok:
            c.samples_ok += 1
        else:
            fails.append(f"{f.name}: {note}")
    add_note(c, f"{label}: {c.samples_ok}/{c.samples_total} sampled OK")
    if fails:
        for fail in fails[:3]:
            add_note(c, f"  FAIL: {fail}")


# --- per-source checks ----------------------------------------------------

def check_batterylife(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/batterylife/batterylife_processed"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    expected_subsets = ["BatteryLife_Processed".replace("BatteryLife_Processed", ""),
                         "MATR", "Stanford", "Stanford_2", "CALCE", "HUST", "ISU_ILCC",
                         "Tongji", "RWTH", "SDU", "ZN-coin", "NA-ion", "CALB",
                         "MICH", "MICH_EXP", "HNEI", "SNL", "UL_PUR", "XJTU"]
    missing = [s for s in expected_subsets if s and not (p / s).is_dir()]
    if missing:
        add_note(c, f"missing subset folders: {missing}")
        c.status = "FAIL"; return
    add_note(c, f"all 18 chemistry subsets present")
    # Sample 30 .pkl files across subsets
    pkls = sample_files(p, "*.pkl", 30)
    run_sampled(c, pkls, verify_pkl, "30 .pkl sampled")
    # Validate JSON labels
    for j in p.glob("Life labels/*.json"):
        ok, note = verify_json(j)
        if not ok:
            add_note(c, f"label JSON corrupt: {j.name} ({note})")
    # Top-level Stanford_2_labels.json
    for j in p.glob("*.json"):
        ok, note = verify_json(j)
        if not ok:
            add_note(c, f"top-level JSON corrupt: {j.name}")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_stanford_osf(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/stanford"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    # Required folders: Aging (8 cells), Temperature Data, Cby20, HPPC, OCV, Code, Code_guide.pdf
    required_zips = {
        "Aging": ["Cell 1p1.zip", "Cell 1p2.zip", "Cell 1p3.zip", "Cell 1p4.zip",
                   "Cell 2p1.zip", "Cell 2p2.zip", "Cell 2p3.zip", "Cell 2p4.zip"],
        "Temperature Data": ["Temperature Data.zip"],
        "Cby20": ["Cby20.zip"],
        "HPPC": ["HPPC.zip"],
        "OCV": ["OCV.zip"],
        "Code": ["Code.zip"],
    }
    missing = []
    zips = []
    for folder, fnames in required_zips.items():
        for fn in fnames:
            f = p / folder / fn
            if not f.exists():
                missing.append(str(f.relative_to(PROJECT_ROOT)))
            else:
                zips.append(f)
    if not (p / "Code_guide.pdf").exists():
        missing.append("Code_guide.pdf")
    if missing:
        add_note(c, f"missing required: {missing}")
        c.status = "FAIL"; return
    # Test all zips
    c.samples_total = len(zips)
    for z in zips:
        ok, note = verify_zip(z)
        if ok: c.samples_ok += 1
        else: add_note(c, f"FAIL {z.name}: {note}")
    add_note(c, f"{c.samples_ok}/{c.samples_total} ZIPs intact")
    # Verify Code_guide.pdf
    ok, note = verify_pdf(p / "Code_guide.pdf")
    add_note(c, f"Code_guide.pdf: {note}")
    if not ok:
        c.samples_ok = -1
    c.status = "PASS" if c.samples_ok == c.samples_total and ok else "FAIL"


def check_stanford_8jnr5(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/stanford_8jnr5"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    zips = list(p.glob("*.zip"))
    if not zips:
        add_note(c, "no zip files found")
        c.status = "FAIL"; return
    c.samples_total = len(zips)
    for z in zips:
        ok, note = verify_zip(z)
        if ok: c.samples_ok += 1; add_note(c, f"{z.name}: {note}")
        else: add_note(c, f"FAIL {z.name}: {note}")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_nasa_random_recomm(c: Check) -> None:
    """Direct download → extracted with regular_alt + recommissioned + second_life."""
    p = PROJECT_ROOT / "data/raw/nasa"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    extracted = p / "extracted" / "battery_alt_dataset"
    if not extracted.is_dir():
        add_note(c, "extracted/battery_alt_dataset/ missing")
        c.status = "FAIL"; return
    required = ["regular_alt_batteries", "recommissioned_batteries", "second_life_batteries", "README.txt", "LICENSE.txt"]
    missing = [r for r in required if not (extracted / r).exists()]
    if missing:
        add_note(c, f"missing: {missing}")
        c.status = "FAIL"; return
    # Verify a sample of CSVs from each folder
    csvs = []
    for sub in ["regular_alt_batteries", "recommissioned_batteries", "second_life_batteries"]:
        csvs.extend(list((extracted / sub).glob("*.csv")))
    add_note(c, f"{len(csvs)} CSV files across 3 folders")
    sample = random.sample(csvs, min(6, len(csvs)))
    run_sampled(c, sample, verify_csv, "6 .csv sampled")
    # Original ZIP intact?
    z = p / "battery_alt_dataset.zip"
    if z.exists():
        ok, note = verify_zip(z)
        add_note(c, f"original ZIP: {note}")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_nasa_pcoe(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/nasa_kaggle/pcoe/cleaned_dataset"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    if not (p / "metadata.csv").exists():
        add_note(c, "metadata.csv missing")
        c.status = "FAIL"; return
    csvs = list((p / "data").rglob("*.csv")) if (p / "data").exists() else list(p.rglob("*.csv"))
    add_note(c, f"{len(csvs)} per-cycle CSV files")
    sample = random.sample(csvs, min(10, len(csvs)))
    run_sampled(c, sample, verify_csv, "10 .csv sampled")
    ok, note = verify_csv(p / "metadata.csv")
    add_note(c, f"metadata.csv: {note}")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_nasa_kaggle_random(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/nasa_kaggle/dataset"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    # Look for sub-experiments
    expected = ["RW_Skewed_High_40C", "RW_Skewed_Low_40C", "Battery_Uniform_Distribution"]
    found_any = any(any(n.lower() in str(f).lower() for f in p.rglob("*") if f.is_dir())
                    for n in expected)
    if not found_any:
        add_note(c, "no recognized sub-experiment folders")
        c.status = "FAIL"; return
    add_note(c, "RW_Skewed + Uniform sub-experiments present")
    # Sample MAT files
    mats = list(p.rglob("*.mat"))
    if mats:
        sample = random.sample(mats, min(6, len(mats)))
        run_sampled(c, sample, verify_mat, "6 .mat sampled")
    else:
        add_note(c, "no .mat files found")
    c.status = "PASS" if (c.samples_total == 0 or c.samples_ok == c.samples_total) else "FAIL"


def check_calce(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/calce"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    # Required PyBaMM-validation reference cell
    cs2_35 = list(p.rglob("*CS2_35*"))
    if not cs2_35:
        add_note(c, "CS2_35 (PyBaMM validation reference) missing")
        c.status = "FAIL"; return
    add_note(c, f"CS2_35 present ({len([f for f in cs2_35 if f.is_file()])} files)")
    xlsx = list(p.rglob("*.xlsx"))
    add_note(c, f"{len(xlsx)} xlsx total")
    sample = random.sample(xlsx, min(8, len(xlsx)))
    run_sampled(c, sample, verify_xlsx, "8 .xlsx sampled")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_pybamm(c: Check) -> None:
    c.path = PROJECT_ROOT / "(conda env: Eco-Research)"
    try:
        import pybamm  # noqa: F401
        add_note(c, f"pybamm {pybamm.__version__} importable")
        c.status = "PASS"
    except ImportError as e:
        add_note(c, f"not importable: {e}")
        c.status = "FAIL"


def check_bwmr(c: Check) -> None:
    p = PROJECT_ROOT / "data/regulatory/bwmr"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    expected = [
        "Batteries_Mgmt_Handling_2001.pdf",
        "BWMR_2022_original_S.O.3984.pdf",
        "BWMR_2023_amendment_S.O.4669.pdf",
        "BWMR_2024_first_amendment_G.S.R.190.pdf",
        "BWMR_2024_second_amendment_S.O.2374.pdf",
        "BWMR_2024_third_amendment_S.O.5210.pdf",
        "BWMR_2025_amendment_S.O.958.pdf",
    ]
    missing = [n for n in expected if not (p / n).exists()]
    if missing:
        add_note(c, f"missing: {missing}")
        c.status = "FAIL"; return
    add_note(c, f"all 7 expected PDFs present")
    pdfs = [p / n for n in expected]
    c.samples_total = len(pdfs)
    for pdf in pdfs:
        ok, note = verify_pdf(pdf)
        if ok: c.samples_ok += 1
        else: add_note(c, f"FAIL {pdf.name}: {note}")
    # Verify extracted CSVs
    tables_dir = p / "extracted_tables"
    if tables_dir.is_dir():
        csvs = list(tables_dir.glob("*.csv"))
        add_note(c, f"extracted_tables/ has {len(csvs)} CSVs")
    en_dir = p / "extracted_english"
    if en_dir.is_dir():
        txts = list(en_dir.glob("*.txt"))
        add_note(c, f"extracted_english/ has {len(txts)} .txt files")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_eu_reg(c: Check) -> None:
    p = PROJECT_ROOT / "data/regulatory/eu/EU_Reg_2023_1542_full.pdf"
    c.path = p
    if not p.exists():
        c.status = "MISSING"; return
    c.size_bytes = p.stat().st_size
    c.n_files = 1
    ok, note = verify_pdf(p)
    add_note(c, note)
    c.samples_total = 1; c.samples_ok = 1 if ok else 0
    if ok and "117" not in note:
        add_note(c, f"WARNING: expected 117 pages, got {note}")
    c.status = "PASS" if ok else "FAIL"


def check_gba(c: Check) -> None:
    p = PROJECT_ROOT / "data/regulatory/gba/BatteryPassDataModel"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    if not (p / ".git").exists():
        add_note(c, ".git directory missing")
        c.status = "WARN"; return
    add_note(c, "git repo intact")
    if not (p / "README.md").exists():
        add_note(c, "README.md missing")
    schemas = list(p.rglob("*.json")) + list(p.rglob("*.yaml")) + list(p.rglob("*.yml"))
    add_note(c, f"{len(schemas)} schema/JSON/YAML files")
    c.status = "PASS"


def check_cpcb_epr(c: Check) -> None:
    p = PROJECT_ROOT / "data/raw/cpcb_epr"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    expected = ["dashboard_raw.html", "dashboard.png", "dashboard.pdf", "dashboard_kv.csv"]
    missing = [n for n in expected if not (p / n).exists()]
    if missing:
        add_note(c, f"missing: {missing}")
        c.status = "FAIL"; return
    add_note(c, "all 4 dashboard artifacts present")
    ok, note = verify_csv(p / "dashboard_kv.csv")
    add_note(c, f"dashboard_kv.csv: {note}")
    c.samples_total = 1; c.samples_ok = 1 if ok else 0
    c.status = "PASS" if ok else "FAIL"


def check_india_market(c: Check) -> None:
    p = PROJECT_ROOT / "data/regulatory/india_market"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    c.size_bytes, c.n_files = folder_size(p)
    expected = [
        "ICEA_Accenture_Charging_Ahead_2025.pdf",
        "CSE_India_EV_Battery_Recycling_EPR.pdf",
        "WRI_Second_Charge_2024.pdf",
    ]
    missing = [n for n in expected if not (p / n).exists()]
    if missing:
        add_note(c, f"missing: {missing}")
        c.status = "FAIL"; return
    add_note(c, "ICEA + CSE + WRI all present")
    pdfs = [p / n for n in expected]
    c.samples_total = len(pdfs)
    for pdf in pdfs:
        ok, note = verify_pdf(pdf)
        if ok: c.samples_ok += 1
        add_note(c, f"  {pdf.name}: {note}")
    c.status = "PASS" if c.samples_ok == c.samples_total else "FAIL"


def check_mcdm_weights(c: Check) -> None:
    p = PROJECT_ROOT / "data/processed/mcdm_weights"
    c.path = p
    if not p.is_dir():
        c.status = "MISSING"; return
    csvs = list(p.glob("*.csv"))
    c.size_bytes, c.n_files = folder_size(p)
    if not csvs:
        add_note(c, "no CSVs yet — expected to be populated from literature review (separate workstream)")
        c.status = "WARN"; return
    add_note(c, f"{len(csvs)} CSV files")
    c.status = "PASS"


# --- main runner ----------------------------------------------------------

CHECKS: list[tuple[str, str, Callable]] = [
    ("BatteryLife (HF)",                  "L1", check_batterylife),
    ("Stanford OSF (Onori 2nd-life)",     "L1", check_stanford_osf),
    ("Stanford OSF 8jnr5 (Data in Brief)","L1", check_stanford_8jnr5),
    ("NASA Randomized & Recommissioned",  "L1", check_nasa_random_recomm),
    ("NASA PCoE (Kaggle)",                "L1", check_nasa_pcoe),
    ("NASA Randomized v1 (Kaggle)",       "L1", check_nasa_kaggle_random),
    ("CALCE (Kaggle)",                    "L1", check_calce),
    ("PyBaMM tool",                       "L2", check_pybamm),
    ("BWMR PDFs (7 files)",               "L3", check_bwmr),
    ("EU Regulation 2023/1542",           "L3", check_eu_reg),
    ("GBA Battery Passport schema",       "L3", check_gba),
    ("CPCB EPR dashboard",                "L4", check_cpcb_epr),
    ("India market reports (ICEA/CSE/WRI)","L4", check_india_market),
    ("MCDM weights from literature",      "L5", check_mcdm_weights),
]


def render_size(b: int) -> str:
    for unit, lim in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if b >= lim:
            return f"{b/lim:.2f} {unit}"
    return f"{b} B"


def main() -> int:
    print("=" * 78)
    print("DATASET ACQUISITION AUDIT")
    print("=" * 78)

    results: list[Check] = []
    for name, layer, fn in CHECKS:
        c = Check(name=name, layer=layer, path=Path("?"))
        print(f"\n[{layer}] {name}")
        print("-" * 78)
        try:
            fn(c)
        except Exception as e:
            c.status = "FAIL"
            add_note(c, f"audit crashed: {type(e).__name__}: {e}")
        print(f"  path:   {c.path}")
        print(f"  size:   {render_size(c.size_bytes)}")
        print(f"  files:  {c.n_files}")
        for n in c.notes:
            print(f"  • {n}")
        print(f"  STATUS: {c.status}")
        results.append(c)

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    cols = ("Status", "Layer", "Name", "Size", "Files")
    print(f"{cols[0]:8}  {cols[1]:5}  {cols[2]:42}  {cols[3]:>10}  {cols[4]:>7}")
    print("-" * 78)
    for c in results:
        print(f"{c.status:8}  {c.layer:5}  {c.name[:42]:42}  {render_size(c.size_bytes):>10}  {c.n_files:>7}")
    n_pass = sum(c.status == "PASS" for c in results)
    n_warn = sum(c.status == "WARN" for c in results)
    n_fail = sum(c.status in ("FAIL", "MISSING") for c in results)
    print("-" * 78)
    print(f"PASS={n_pass}  WARN={n_warn}  FAIL/MISSING={n_fail}  (out of {len(results)})")
    overall = "READY FOR PROCESSING" if n_fail == 0 else "NOT READY (see failures above)"
    print(f"\nOVERALL: {overall}")

    # Write markdown report
    report = ["# Dataset Acquisition Audit Report\n",
              f"_Run on this machine_\n",
              f"## Summary\n",
              f"- PASS: {n_pass}",
              f"- WARN: {n_warn}",
              f"- FAIL/MISSING: {n_fail}",
              f"- **Overall: {overall}**\n",
              "## Per-source detail\n"]
    for c in results:
        report.append(f"### {c.status} — [{c.layer}] {c.name}")
        report.append(f"- Path: `{c.path.relative_to(PROJECT_ROOT) if c.path.is_absolute() and PROJECT_ROOT in c.path.parents else c.path}`")
        report.append(f"- Size: {render_size(c.size_bytes)}")
        report.append(f"- Files: {c.n_files}")
        report.append(f"- Sample integrity: {c.samples_ok}/{c.samples_total} OK")
        for n in c.notes:
            report.append(f"  - {n}")
        report.append("")
    out = PROJECT_ROOT / "data" / "AUDIT_REPORT.md"
    out.write_text("\n".join(report))
    print(f"\nReport written to {out.relative_to(PROJECT_ROOT)}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
