"""
Validate freshly-downloaded PDFs:
  1. Check magic bytes + size
  2. Extract title from PDF metadata + first-page text
  3. Match to expected paper from manifest (by DOI / known-PII / title fuzzy-match)
  4. Print proposed rename + move plan (DRY-RUN by default; pass --apply to execute)
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "freshly-downloaded-papers"
DST = ROOT / "research-papers"
APPLY = "--apply" in sys.argv

import importlib.util
spec = importlib.util.spec_from_file_location("v3mod", ROOT / "scripts" / "download_papers_v3.py")
v3mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3mod)
MANIFEST = v3mod.MANIFEST

# Build manifest indices
by_doi = {e["doi"]: e for e in MANIFEST if e.get("doi")}
by_filename = {e["filename"]: e for e in MANIFEST}

# Hard-coded source-name -> manifest-filename mapping derived from PII / DOI inspection.
# Anything not in this map will be auto-matched via title fuzzy-matching below.
KNOWN_MAPPING = {
    # Elsevier S2.0 PIIs verified against DOI registry
    "1-s2.0-S0360544224040076-main.pdf": "T1_Jin2025_LSTM_XGBoost_BinaryFirefly_Energy.pdf",   # Energy 314, 134229
    "1-s2.0-S0378775324001356-main.pdf": "T6_Li2024_HighThroughputPyBaMM_JPS.pdf",              # JPS 599, 234184
    "1-s2.0-S0951832020308838-main.pdf": "T4_Xu2021_StackedDenoisingAE_RESS.pdf",               # RESS 208, 107396
    "1-s2.0-S2405896324016872-main.pdf": "T2_Neri2024_BWM-TOPSIS_DLT_DPP_IFAC.pdf",             # IFAC PapersOnLine
    "1-s2.0-S2666791623000052-main.pdf": "T3_Berger2023_DPP_DataReqs_CPL.pdf",                  # Cleaner Production Letters
    "1-s2.0-S277239092500071X-main.pdf": "T2_Aishwarya2025_FuzzyAHPTOPSIS_CLSC.pdf",            # Cleaner Logistics & SC 2025
    "1-s2.0-S2772662224000924-main.pdf": "T2_Varchandi2024_BWM-FuzzyTOPSIS_DAJ.pdf",            # Decision Analytics J
    # Verified manually via filename
    "309-1-5018-1-10-20210608.pdf": "T6_Sulzer2021_PyBaMM_JORS.pdf",
    "A method for battery fault diagnosis and early warning combining isolated forest algorithm and sliding window.pdf": "T4_Cheng2023_iForest_SlidingWindow_ESE.pdf",
    "An Interpretable TCN– Transformer Framework for Lithium‐Ion Battery State of Health Estimation Using SHAP Analysis.pdf": "T1_Guo2026_TCN-Transformer-SHAP_QRE.pdf",
    "ijieom-07-2025-0144en.pdf": "T2_Lahri2026_FuzzyOrdinalMCDM_EVSupplier_IJIEOM.pdf",
    "s41598-026-44597-z_reference.pdf": "T2_Dincer2026_HybridMCDM_EV_EoL_SciReports.pdf",
    "zkag004.pdf": "T1_Mchara2026_Wavelet-Transformer-XGBoost_CleanEnergy.pdf",
    # Verified via title-extraction in dry-run
    "1-s2.0-S0959652622011131-main.pdf": "T3_Berger2022_DPP_Conceptualization_JCP.pdf",   # JCP — Berger 2022 DPP
    "1-s2.0-S2352152X24038532-main.pdf": "T4_Chan2024_VAE_SVDD_JEnergyStorage.pdf",       # J Energy Storage — Chan VAE-SVDD
    "1-s2.0-S2950264025000541-main.pdf": "T1_Varghese2025_DynamicAttentionTransformer_FutureBatt.pdf",  # Future Batteries — Varghese
    # New: Cui Stanford Second-Life
    "PIIS2666386424001905.pdf": "T5_Cui2024_StanfordSecondLife_CRPS.pdf",
    # Wrong / unrelated — DO NOT MOVE
    "1-s2.0-S0360544225021619-main.pdf": "__UNRELATED__",   # VIVACE ocean energy paper — wrong download
    "1-s2.0-S2772390924000167-main.pdf": "__UNRELATED__",   # CLSC 2024 SC Digital Twin — NOT Kisters DPP (my DOI was wrong in manual list)
    # Even newer batch (Apr 27 21:23+) — corrected Pohlmann find
    "1-s2.0-S2666791624000368-main.pdf": "T3_Pohlmann2024_DPP_EVbattery_CPL.pdf",
    # Final batch (Apr 27 21:28) — Asadabadi
    "s10479-022-04878-y (1).pdf": "T2_Asadabadi2022_StratifiedBWM-TOPSIS_AnnOR.pdf",
    # The very last one (Apr 27 21:35) — Ye ProGAN
    "1-s2.0-S0360544225015646-main.pdf": "T6_Ye2025_ProGAN_BatterySoH_Energy.pdf",
    # Newest batch (Apr 27 21:16-21:18)
    "Energy Storage - 2025 - Hemavathi - India s Lithium‐Ion Battery Landscape Strategic Opportunities  Market Dynamics  and.pdf": "T5_Hemavathi2025_IndiaLiBLandscape_EnergyStorage.pdf",
    "Expert Systems - 2023 - Tavana - A sustainable circular supply chain network design model for electric vehicle battery.pdf": "T5_Tavana2024_CircularSC_IoT_ExpSys.pdf",
    "Fault detection method for electric vehicle battery pack based on improved kurtosis and isolation forest.pdf": "T4_Wu2024_Kurtosis_iForest_IJGE.pdf",
    "Fault detection method for electric vehicle battery pack based on improved kurtosis and isolation forest (1).pdf": "__DUPLICATE__",
    "Intl J of Energy Research - 2021 - Naaz - A generative adversarial network‐based synthetic data augmentation technique for.pdf": "T6_Naaz2021_GAN_BatteryData_IJER.pdf",
    "Intl J of Energy Research - 2021 - Naaz - A generative adversarial network‐based synthetic data augmentation technique for (1).pdf": "__DUPLICATE__",
    "robust-design-and-optimization-of-integrated-sustainable-lithium-ion-battery-supply-chain-network-under-demand.pdf": "T2_RobustDesign_LiBSC_2025_ACS_IECR.pdf",
    "s10668-024-05055-w.pdf": "T5_Suman2024_CE_Barriers_IndianEV_EDS.pdf",
    "s41586-025-09617-4.pdf": "T5_Zhai2025_LiB_CircularEconomy_Nature.pdf",
    "s41660-025-00652-2.pdf": "T2_Soriano2025_ZBWM-DEMATEL-TOPSIS_Energy_PIOS.pdf",
}


def is_valid_pdf(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < 50_000:
        return False, f"too small ({size} bytes)"
    with path.open("rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        return False, f"bad magic {head!r}"
    return True, f"{size//1024} KB"


def extract_title_and_first_text(path: Path) -> tuple[str, str]:
    """Return (metadata_title, first-page-text-snippet)."""
    try:
        reader = pypdf.PdfReader(str(path))
        meta_title = (reader.metadata.title if reader.metadata else None) or ""
        first_text = ""
        if reader.pages:
            try:
                first_text = (reader.pages[0].extract_text() or "")[:400]
            except Exception as e:
                first_text = f"(extract err: {e})"
        return meta_title.strip(), first_text.strip()
    except Exception as e:
        return f"(err: {e})", ""


def auto_match(title: str, first_text: str) -> tuple[str | None, str]:
    """Fuzzy-match against manifest titles. Returns (target_filename, reason)."""
    haystack = (title + " " + first_text).lower()
    candidates = []
    for entry in MANIFEST:
        manifest_title = entry.get("title", "").lower()
        # Score = number of meaningful words from manifest title found in haystack
        words = [w for w in re.split(r"\W+", manifest_title) if len(w) >= 4]
        if not words:
            continue
        hits = sum(1 for w in words if w in haystack)
        if hits >= max(2, len(words) // 2):
            candidates.append((hits / len(words), entry["filename"], hits, len(words)))
    candidates.sort(reverse=True)
    if not candidates:
        return None, "no match"
    best_score, best_fname, hits, total = candidates[0]
    if best_score >= 0.5:
        return best_fname, f"matched ({hits}/{total} words, score {best_score:.2f})"
    return None, f"weak: best={best_fname} ({hits}/{total} words, score {best_score:.2f})"


def main() -> None:
    if not SRC.exists():
        print(f"Source folder missing: {SRC}")
        sys.exit(1)

    files = sorted(SRC.glob("*.pdf"))
    print(f"Found {len(files)} PDF(s) in {SRC.name}/\n")

    plan: list[tuple[Path, Path | None, str]] = []
    for src in files:
        ok, size_info = is_valid_pdf(src)
        if not ok:
            print(f"❌ {src.name}\n     INVALID: {size_info}")
            plan.append((src, None, f"invalid: {size_info}"))
            continue

        title, snippet = extract_title_and_first_text(src)

        target_fname = KNOWN_MAPPING.get(src.name)
        if target_fname == "__UNRELATED__":
            print(f"⛔ {src.name}  ({size_info})")
            print(f"     metadata title: {title[:120]!r}")
            print(f"     -> SKIPPED (not in our manifest — wrong download / unrelated paper)")
            plan.append((src, None, "unrelated to project"))
            continue
        if target_fname == "__DUPLICATE__":
            print(f"♻️  {src.name}  ({size_info})")
            print(f"     -> SKIPPED (duplicate of non-(1) version — leaving in place; safe to delete)")
            plan.append((src, None, "duplicate"))
            continue
        if target_fname is None:
            target_fname, reason = auto_match(title, snippet)
        else:
            reason = "known mapping"

        if target_fname is None:
            print(f"⚠️  {src.name}  ({size_info})")
            print(f"     metadata title: {title[:120]!r}")
            print(f"     first text:     {snippet[:150]!r}")
            print(f"     -> NO MATCH ({reason})")
            plan.append((src, None, f"unmatched ({reason})"))
            continue

        target_path = DST / target_fname
        already_exists = target_path.exists() and is_valid_pdf(target_path)[0]
        marker = "🔁 (overwrite)" if already_exists else "✅"
        print(f"{marker} {src.name}  ({size_info})")
        print(f"     metadata title: {title[:120]!r}")
        print(f"     -> {target_fname}  [{reason}]")
        plan.append((src, target_path, reason))

    print("\n" + "=" * 60)
    print(f"Plan: {sum(1 for _,t,_ in plan if t)} to move, {sum(1 for _,t,_ in plan if not t)} unmatched/invalid")
    if not APPLY:
        print("\n(DRY RUN — pass --apply to execute moves)")
        return

    print("\nExecuting moves...")
    for src, target, _ in plan:
        if target is None:
            continue
        if target.exists():
            target.unlink()  # overwrite (we validated source is good)
        shutil.move(str(src), str(target))
        print(f"  moved: {src.name} -> {target.name}")
    print("Done.")


if __name__ == "__main__":
    main()
