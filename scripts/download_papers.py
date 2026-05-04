"""
Bulk-download research papers for the literature review.

Each entry in MANIFEST is a dict with:
  filename       : final on-disk filename in research-papers/
  theme          : T1..T6
  title          : short paper title (for logs / manual list)
  doi            : canonical DOI (no URL prefix)
  urls           : ordered list of candidate PDF URLs to try
  notes          : freeform; appears in MANUAL_DOWNLOAD_LIST.md if all URLs fail
  paywalled      : if True, skip auto-download attempts and go straight to manual list

Usage:
  python scripts/download_papers.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
PAPERS_DIR.mkdir(exist_ok=True)

MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"
RESULTS_LOG = PAPERS_DIR / "_download_results.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 60
MIN_BYTES = 50_000  # 50 KB lower bound; smaller is almost certainly not a real paper


def is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def try_download(url: str) -> tuple[bool, bytes | str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        ctype = r.headers.get("Content-Type", "").lower()
        if "html" in ctype and not is_pdf(r.content):
            return False, f"got HTML (Content-Type={ctype})"
        if not is_pdf(r.content):
            return False, f"not a PDF (first bytes: {r.content[:8]!r})"
        if len(r.content) < MIN_BYTES:
            return False, f"too small ({len(r.content)} bytes)"
        return True, r.content
    except requests.RequestException as e:
        return False, f"network error: {e}"


def download_paper(entry: dict) -> dict:
    target = PAPERS_DIR / entry["filename"]
    if target.exists() and target.stat().st_size >= MIN_BYTES:
        return {"filename": entry["filename"], "status": "already_present", "size": target.stat().st_size}

    if entry.get("paywalled"):
        return {"filename": entry["filename"], "status": "skipped_paywalled", "reason": "marked paywalled"}

    attempts = []
    for url in entry.get("urls", []):
        ok, payload = try_download(url)
        attempts.append({"url": url, "ok": ok, "detail": (payload if isinstance(payload, str) else f"{len(payload)} bytes")})
        if ok:
            target.write_bytes(payload)
            return {
                "filename": entry["filename"],
                "status": "downloaded",
                "url": url,
                "size": len(payload),
                "attempts": attempts,
            }
        time.sleep(0.5)

    return {"filename": entry["filename"], "status": "failed", "attempts": attempts}


# ---- MANIFEST -------------------------------------------------------------

# Naming: T<theme>_<FirstAuthor><Year>_<short>_<journal>.pdf
MANIFEST: list[dict] = [
    # ===================== T1 — Hybrid ML for SoH/RUL =====================
    {
        "filename": "T1_Park2020_LSTM_RUL_IEEEAccess.pdf",
        "theme": "T1",
        "title": "LSTM-Based Battery RUL Prediction With Multi-Channel Charging Profiles",
        "doi": "10.1109/ACCESS.2020.2968939",
        "urls": [
            "https://ieeexplore.ieee.org/ielx7/6287639/8948470/08967059.pdf",
            "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967059",
        ],
        "notes": "IEEE Access — open access. If auto fails: https://doi.org/10.1109/ACCESS.2020.2968939",
    },
    {
        "filename": "T1_Song2020_XGBoost_SOH_Energies.pdf",
        "theme": "T1",
        "title": "Lithium-Ion Battery SOH Estimation Based on XGBoost Algorithm with Accuracy Correction",
        "doi": "10.3390/en13040812",
        "urls": [
            "https://www.mdpi.com/1996-1073/13/4/812/pdf",
            "https://www.mdpi.com/1996-1073/13/4/812/pdf?version=1582012896",
        ],
        "notes": "MDPI Energies open access",
    },
    {
        "filename": "T1_Jin2025_LSTM_XGBoost_BinaryFirefly_Energy.pdf",
        "theme": "T1",
        "title": "LSTM and XGBoost with feature selection via Binary Firefly Algorithm",
        "doi": "10.1016/j.energy.2024.134229",
        "urls": [],
        "paywalled": True,
        "notes": "Elsevier Energy — paywalled. https://doi.org/10.1016/j.energy.2024.134229",
    },
    {
        "filename": "T1_Zhao2024_AT-CNN-BiLSTM_SciReports.pdf",
        "theme": "T1",
        "title": "Application of SoH estimation and RUL prediction for Li-ion based on AT-CNN-BiLSTM",
        "doi": "10.1038/s41598-024-80421-2",
        "urls": [
            "https://www.nature.com/articles/s41598-024-80421-2.pdf",
        ],
        "notes": "Nature Sci Reports — open access",
    },
    {
        "filename": "T1_BOA-XGBoost-TreeSHAP_2024_Batteries.pdf",
        "theme": "T1",
        "title": "SoH Estimation for Li-Ion Using an Explainable XGBoost Model with Parameter Optimization",
        "doi": "10.3390/batteries10110394",
        "urls": [
            "https://www.mdpi.com/2313-0105/10/11/394/pdf",
        ],
        "notes": "MDPI Batteries open access",
    },
    {
        "filename": "T1_Rout2025_SoH_ML_Comparison_SciReports.pdf",
        "theme": "T1",
        "title": "Estimation of state of health for Li-ion batteries using advanced data-driven techniques",
        "doi": "10.1038/s41598-025-93775-y",
        "urls": [
            "https://www.nature.com/articles/s41598-025-93775-y.pdf",
        ],
        "notes": "Nature Sci Reports — open access",
    },

    # ===================== T2 — MCDM for battery / circular econ =====================
    {
        "filename": "T2_Aishwarya2025_FuzzyAHPTOPSIS_CLSC.pdf",
        "theme": "T2",
        "title": "A stakeholder-centric fuzzy AHP-TOPSIS framework for circular SC of EV batteries",
        "doi": "10.1016/j.clscn.2025.100272",
        "urls": [
            "https://www.sciencedirect.com/science/article/pii/S2772390925000162/pdfft",
            "https://www.sciencedirect.com/science/article/pii/S2772390925000162",
        ],
        "notes": "Cleaner Logistics & SC — Elsevier OA gold journal. May still need login.",
    },
    {
        "filename": "T2_BWM_RBFNN_2025_SciReports.pdf",
        "theme": "T2",
        "title": "Forewarning model for reverse SC of EoL power batteries: BWM + RBFNN",
        "doi": "10.1038/s41598-025-13573-4",
        "urls": [
            "https://www.nature.com/articles/s41598-025-13573-4.pdf",
        ],
        "notes": "Nature Sci Reports — open access",
    },
    {
        "filename": "T2_RobustDesign_LiBSC_2025_ACS_IECR.pdf",
        "theme": "T2",
        "title": "Robust Design of Integrated Sustainable Li-Ion Battery SC under Demand Uncertainty",
        "doi": "10.1021/acs.iecr.5c01990",
        "urls": [],
        "paywalled": True,
        "notes": "ACS IECR — paywalled. https://doi.org/10.1021/acs.iecr.5c01990",
    },
    {
        "filename": "T2_FuzzyMCDM_BatteryRecycling_2025_Batteries.pdf",
        "theme": "T2",
        "title": "Evaluating Sustainable Battery Recycling Technologies via Fuzzy MCDM",
        "doi": "10.3390/batteries11080294",
        "urls": [
            "https://www.mdpi.com/2313-0105/11/8/294/pdf",
        ],
        "notes": "MDPI Batteries open access",
    },
    {
        "filename": "T2_Ma2024_PathwayDecisions_NatComms.pdf",
        "theme": "T2",
        "title": "Pathway decisions for reuse and recycling of retired Li-ion batteries (LCA)",
        "doi": "10.1038/s41467-024-52030-0",
        "urls": [
            "https://www.nature.com/articles/s41467-024-52030-0.pdf",
        ],
        "notes": "Nature Communications open access",
    },

    # ===================== T3 — Battery DPP =====================
    {
        "filename": "T3_Kisters2024_DPP_EVbattery_CLSC.pdf",
        "theme": "T3",
        "title": "Digital product passports for EV batteries: stakeholder requirements",
        "doi": "10.1016/j.clscn.2024.100154",
        "urls": [
            "https://www.sciencedirect.com/science/article/pii/S2666791624000368/pdfft",
            "https://www.sciencedirect.com/science/article/pii/S2666791624000368",
        ],
        "notes": "Cleaner Logistics & SC — Elsevier OA. Verify exact DOI/page once downloaded.",
    },
    {
        "filename": "T3_Kumar2023_DigitalFramework_npjMS.pdf",
        "theme": "T3",
        "title": "A digital solution framework for enabling EV battery circularity",
        "doi": "10.1038/s44296-023-00001-9",
        "urls": [
            "https://www.nature.com/articles/s44296-023-00001-9.pdf",
        ],
        "notes": "npj Materials Sustainability open access",
    },

    # ===================== T4 — Anomaly Detection =====================
    {
        "filename": "T4_Chan2024_VAE_SVDD_JEnergyStorage.pdf",
        "theme": "T4",
        "title": "VAE-driven adversarial SVDD for power battery anomaly detection",
        "doi": "10.1016/j.est.2024.114267",
        "urls": [
            "https://www.sciencedirect.com/science/article/pii/S2352152X24039574/pdfft",
            "https://www.sciencedirect.com/science/article/pii/S2352152X24039574",
        ],
        "notes": "J. Energy Storage — Elsevier; usually paywalled but worth trying OA endpoints.",
    },
    {
        "filename": "T4_Lee2025_DNN_AnomalyDetection_Batteries.pdf",
        "theme": "T4",
        "title": "Deep Neural Network with Anomaly Detection for Single-Cycle Battery Lifetime",
        "doi": "10.3390/batteries11080288",
        "urls": [
            "https://www.mdpi.com/2313-0105/11/8/288/pdf",
        ],
        "notes": "MDPI Batteries open access",
    },
    {
        "filename": "T4_Xu2021_StackedDenoisingAE_RESS.pdf",
        "theme": "T4",
        "title": "Life prediction of Li-ion batteries based on stacked denoising autoencoders",
        "doi": "10.1016/j.ress.2020.107396",
        "urls": [],
        "paywalled": True,
        "notes": "Elsevier RESS — paywalled. https://doi.org/10.1016/j.ress.2020.107396",
    },

    # ===================== T5 — India / regulatory / market =====================
    {
        "filename": "T5_Tavana2024_CircularSC_IoT_ExpSys.pdf",
        "theme": "T5",
        "title": "Sustainable circular SC network design for EV battery production using IoT+BD",
        "doi": "10.1111/exsy.13395",
        "urls": [],
        "paywalled": True,
        "notes": "Wiley Expert Systems — paywalled. https://doi.org/10.1111/exsy.13395",
    },
    {
        "filename": "T5_Zhai2025_LiB_CircularEconomy_Nature.pdf",
        "theme": "T5",
        "title": "A circular economy approach for the global lithium-ion battery supply chain",
        "doi": "10.1038/s41586-025-09617-4",
        "urls": [
            "https://www.nature.com/articles/s41586-025-09617-4.pdf",
        ],
        "notes": "Nature flagship — typically paywalled, attempting OA endpoint anyway",
    },
    {
        "filename": "T5_Patel2024_SecondLife_Pathways_FrontChem.pdf",
        "theme": "T5",
        "title": "Lithium-ion battery second life: pathways, challenges and outlook",
        "doi": "10.3389/fchem.2024.1358417",
        "urls": [
            "https://www.frontiersin.org/articles/10.3389/fchem.2024.1358417/pdf",
            "https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2024.1358417/pdf",
        ],
        "notes": "Frontiers in Chemistry open access",
    },
    {
        "filename": "T5_Cui2024_StanfordSecondLife_CRPS.pdf",
        "theme": "T5",
        "title": "Taking second-life batteries from exhausted to empowered (Stanford dataset)",
        "doi": "10.1016/j.xcrp.2024.101941",
        "urls": [
            "https://www.cell.com/cell-reports-physical-science/pdf/S2666-3864(24)00203-X.pdf",
            "https://www.sciencedirect.com/science/article/pii/S266638642400203X/pdfft",
        ],
        "notes": "Cell Reports Physical Science — usually open access",
    },

    # ===================== T6 — Synthetic data / simulation =====================
    {
        "filename": "T6_Sulzer2021_PyBaMM_JORS.pdf",
        "theme": "T6",
        "title": "Python Battery Mathematical Modelling (PyBaMM)",
        "doi": "10.5334/jors.309",
        "urls": [
            "https://openresearchsoftware.metajnl.com/articles/10.5334/jors.309/galley/506/download/",
        ],
        "notes": "Journal of Open Research Software — open access",
    },
    {
        "filename": "T6_Naaz2021_GAN_BatteryData_IJER.pdf",
        "theme": "T6",
        "title": "GAN-based synthetic data augmentation for battery condition evaluation",
        "doi": "10.1002/er.7013",
        "urls": [],
        "paywalled": True,
        "notes": "Wiley Int J Energy Research — paywalled. https://doi.org/10.1002/er.7013",
    },
    {
        "filename": "T6_BatteryLife_Tan2025_KDD.pdf",
        "theme": "T6",
        "title": "BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction",
        "doi": "10.48550/arXiv.2502.18218",
        "urls": [
            "https://arxiv.org/pdf/2502.18218.pdf",
            "https://arxiv.org/pdf/2502.18218v2.pdf",
            "https://arxiv.org/pdf/2502.18218v1.pdf",
        ],
        "notes": "arXiv preprint of KDD 2025 paper",
    },
]


def main() -> None:
    print(f"Manifest: {len(MANIFEST)} papers; downloading to {PAPERS_DIR}\n")
    results = []
    for i, entry in enumerate(MANIFEST, 1):
        print(f"[{i:02d}/{len(MANIFEST)}] {entry['filename']}")
        res = download_paper(entry)
        res["theme"] = entry["theme"]
        res["title"] = entry["title"]
        res["doi"] = entry["doi"]
        results.append(res)
        status = res["status"]
        marker = {"downloaded": "OK", "already_present": "OK (cached)", "skipped_paywalled": "SKIP (paywall)", "failed": "FAIL"}.get(status, "?")
        size = res.get("size")
        sz = f" {size//1024} KB" if size else ""
        print(f"   -> {marker}{sz}")
        time.sleep(0.5)

    RESULTS_LOG.write_text(json.dumps(results, indent=2))
    write_manual_list(results)
    print("\n=== Summary ===")
    print(f"Downloaded:  {sum(1 for r in results if r['status'] == 'downloaded')}")
    print(f"Already had: {sum(1 for r in results if r['status'] == 'already_present')}")
    print(f"Paywalled:   {sum(1 for r in results if r['status'] == 'skipped_paywalled')}")
    print(f"Failed:      {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Manual list: {MANUAL_LIST}")


def write_manual_list(results: list[dict]) -> None:
    needs_manual = [r for r in results if r["status"] in ("skipped_paywalled", "failed")]
    if not needs_manual:
        MANUAL_LIST.write_text("# Manual Download List\n\nAll papers downloaded automatically — nothing to do here.\n")
        return
    lines = [
        "# Manual Download List",
        "",
        "These papers could not be auto-downloaded (paywalled, or all candidate URLs failed).",
        "Please download manually and save with the suggested filename in `research-papers/`.",
        "",
        "| # | Theme | Suggested Filename | Title | DOI / Link | Reason |",
        "|---|-------|--------------------|-------|------------|--------|",
    ]
    for i, r in enumerate(needs_manual, 1):
        doi = r.get("doi", "")
        doi_link = f"https://doi.org/{doi}" if doi else ""
        if r["status"] == "skipped_paywalled":
            reason = "Paywalled (Elsevier/ACS/Wiley/Nature flagship)"
        else:
            attempts = r.get("attempts", [])
            reason = f"All {len(attempts)} URLs failed"
        lines.append(f"| {i} | {r['theme']} | `{r['filename']}` | {r['title']} | {doi_link} | {reason} |")
    if any(r["status"] == "failed" for r in needs_manual):
        lines += ["", "## Failure detail (for debugging)", ""]
        for r in needs_manual:
            if r["status"] != "failed":
                continue
            lines.append(f"### {r['filename']}")
            for a in r.get("attempts", []):
                lines.append(f"- `{a['url']}` -> {a['detail']}")
            lines.append("")
    MANUAL_LIST.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
