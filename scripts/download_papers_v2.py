"""
v2 downloader — combines cloudscraper (Cloudflare bypass for MDPI/ScienceDirect),
unpaywall.org API (canonical OA URL discovery), and direct URL attempts.

Each entry in MANIFEST is a dict with:
  filename       : final on-disk filename in research-papers/
  theme          : T1..T6
  title          : short paper title
  doi            : canonical DOI
  urls           : ordered list of direct PDF URL candidates (tried before unpaywall)
  paywalled      : if True, skip auto-download attempts and go straight to manual list

Strategy per paper:
  1. If file already on disk -> skip.
  2. If paywalled flag -> add to manual list.
  3. Try each URL in `urls` via cloudscraper.
  4. If all fail, query unpaywall API for canonical OA PDF URL and try that.
  5. Validate response is a real PDF (magic bytes, size).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cloudscraper
import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
PAPERS_DIR.mkdir(exist_ok=True)

MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"
RESULTS_LOG = PAPERS_DIR / "_download_results_v2.json"

UNPAYWALL_EMAIL = "rishabh.research@example.com"
TIMEOUT = 90
MIN_BYTES = 50_000

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "darwin", "mobile": False}
)
scraper.headers.update(
    {
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
)


def is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def fetch(url: str, referer: str | None = None) -> tuple[bool, bytes | str]:
    headers = {}
    if referer:
        headers["Referer"] = referer
    try:
        r = scraper.get(url, timeout=TIMEOUT, allow_redirects=True, headers=headers)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        if not is_pdf(r.content):
            ctype = r.headers.get("Content-Type", "").lower()
            preview = r.content[:60]
            return False, f"not a PDF (CT={ctype}; head={preview!r})"
        if len(r.content) < MIN_BYTES:
            return False, f"too small ({len(r.content)} bytes)"
        return True, r.content
    except Exception as e:
        return False, f"error: {type(e).__name__}: {e}"


def unpaywall_pdf_url(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location")
        if not loc:
            return None
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def download_paper(entry: dict) -> dict:
    target = PAPERS_DIR / entry["filename"]
    if target.exists() and target.stat().st_size >= MIN_BYTES:
        return {"filename": entry["filename"], "status": "already_present", "size": target.stat().st_size}

    if entry.get("paywalled"):
        return {"filename": entry["filename"], "status": "skipped_paywalled", "reason": "marked paywalled"}

    attempts = []
    for url in entry.get("urls", []):
        ok, payload = fetch(url, referer="https://scholar.google.com/")
        attempts.append({"url": url, "ok": ok, "detail": (payload if isinstance(payload, str) else f"{len(payload)} bytes")})
        if ok:
            target.write_bytes(payload)
            return {"filename": entry["filename"], "status": "downloaded", "url": url, "size": len(payload), "attempts": attempts}
        time.sleep(0.4)

    # unpaywall fallback
    up_url = unpaywall_pdf_url(entry.get("doi", ""))
    if up_url:
        ok, payload = fetch(up_url, referer="https://unpaywall.org/")
        attempts.append({"url": f"[unpaywall] {up_url}", "ok": ok, "detail": (payload if isinstance(payload, str) else f"{len(payload)} bytes")})
        if ok:
            target.write_bytes(payload)
            return {"filename": entry["filename"], "status": "downloaded", "url": up_url, "size": len(payload), "attempts": attempts}

    return {"filename": entry["filename"], "status": "failed", "attempts": attempts}


# ---- MANIFEST -------------------------------------------------------------
# Combined: (a) seed papers (with corrected URLs), (b) new gap-fill papers from agents.

MANIFEST: list[dict] = [
    # ===================== T1 — Hybrid ML SoH/RUL =====================
    # SEED (already downloaded — keep entries idempotent so script is rerunnable)
    {"filename": "T1_Park2020_LSTM_RUL_IEEEAccess.pdf", "theme": "T1", "title": "LSTM RUL multi-channel charging", "doi": "10.1109/ACCESS.2020.2968939",
     "urls": ["https://ieeexplore.ieee.org/ielx7/6287639/8948470/08967059.pdf"]},
    {"filename": "T1_Song2020_XGBoost_SOH_Energies.pdf", "theme": "T1", "title": "XGBoost SoH accuracy correction", "doi": "10.3390/en13040812",
     "urls": ["https://www.mdpi.com/1996-1073/13/4/812/pdf"]},
    {"filename": "T1_Jin2025_LSTM_XGBoost_BinaryFirefly_Energy.pdf", "theme": "T1", "title": "LSTM+XGBoost+Binary Firefly", "doi": "10.1016/j.energy.2024.134229",
     "urls": [], "paywalled": True},
    {"filename": "T1_Zhao2024_AT-CNN-BiLSTM_SciReports.pdf", "theme": "T1", "title": "AT-CNN-BiLSTM SoH+RUL", "doi": "10.1038/s41598-024-80421-2",
     "urls": ["https://www.nature.com/articles/s41598-024-80421-2.pdf"]},
    {"filename": "T1_BOA-XGBoost-TreeSHAP_2024_Batteries.pdf", "theme": "T1", "title": "BOA-XGBoost + TreeSHAP", "doi": "10.3390/batteries10110394",
     "urls": ["https://www.mdpi.com/2313-0105/10/11/394/pdf"]},
    {"filename": "T1_Rout2025_SoH_ML_Comparison_SciReports.pdf", "theme": "T1", "title": "SoH ML comparison", "doi": "10.1038/s41598-025-93775-y",
     "urls": ["https://www.nature.com/articles/s41598-025-93775-y.pdf"]},
    # NEW (from agent)
    {"filename": "T1_Guo2026_TCN-Transformer-SHAP_QRE.pdf", "theme": "T1", "title": "TCN-Transformer + SHAP for SoH", "doi": "10.1002/qre.70155",
     "urls": ["https://onlinelibrary.wiley.com/doi/pdf/10.1002/qre.70155", "https://onlinelibrary.wiley.com/doi/epdf/10.1002/qre.70155"]},
    {"filename": "T1_Chen2025_TLSTM-DilatedCNN_SciReports.pdf", "theme": "T1", "title": "Spatial attention TLSTM + dilated CNN", "doi": "10.1038/s41598-025-17610-0",
     "urls": ["https://www.nature.com/articles/s41598-025-17610-0.pdf"]},
    {"filename": "T1_Varghese2025_DynamicAttentionTransformer_FutureBatt.pdf", "theme": "T1", "title": "Dynamic attention transformer RUL", "doi": "10.1016/j.fub.2025.100075",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2773294725000040/pdfft"]},
    {"filename": "T1_Yenioglu2026_CNN-ML-RUL_Batteries.pdf", "theme": "T1", "title": "CNN + ML for RUL framework", "doi": "10.3390/batteries12040135",
     "urls": ["https://www.mdpi.com/2313-0105/12/4/135/pdf"]},
    {"filename": "T1_Cai2025_Transformer-LSTM-SoH_AppSci.pdf", "theme": "T1", "title": "Transformer-LSTM fusion SoH", "doi": "10.3390/app15073747",
     "urls": ["https://www.mdpi.com/2076-3417/15/7/3747/pdf"]},
    {"filename": "T1_Mchara2026_Wavelet-Transformer-XGBoost_CleanEnergy.pdf", "theme": "T1", "title": "Wavelet+Transformer+XGBoost EV RUL", "doi": "10.1093/ce/zkag004",
     "urls": ["https://academic.oup.com/ce/article-pdf/10/2/119/65876323/zkag004.pdf"]},

    # ===================== T2 — MCDM =====================
    # SEED
    {"filename": "T2_Aishwarya2025_FuzzyAHPTOPSIS_CLSC.pdf", "theme": "T2", "title": "Fuzzy AHP-TOPSIS EV battery circular SC", "doi": "10.1016/j.clscn.2025.100272",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2772390925000162/pdfft"]},
    {"filename": "T2_BWM_RBFNN_2025_SciReports.pdf", "theme": "T2", "title": "BWM+RBFNN reverse SC forewarning", "doi": "10.1038/s41598-025-13573-4",
     "urls": ["https://www.nature.com/articles/s41598-025-13573-4.pdf"]},
    {"filename": "T2_RobustDesign_LiBSC_2025_ACS_IECR.pdf", "theme": "T2", "title": "Robust LiB SC under uncertainty", "doi": "10.1021/acs.iecr.5c01990",
     "urls": [], "paywalled": True},
    {"filename": "T2_FuzzyMCDM_BatteryRecycling_2025_Batteries.pdf", "theme": "T2", "title": "T-spherical fuzzy DEMATEL-CoCoSo recycling", "doi": "10.3390/batteries11080294",
     "urls": ["https://www.mdpi.com/2313-0105/11/8/294/pdf"]},
    {"filename": "T2_Ma2024_PathwayDecisions_NatComms.pdf", "theme": "T2", "title": "EoL pathway decisions", "doi": "10.1038/s41467-024-52030-0",
     "urls": ["https://www.nature.com/articles/s41467-024-52030-0.pdf"]},
    # NEW
    {"filename": "T2_Neri2024_BWM-TOPSIS_DLT_DPP_IFAC.pdf", "theme": "T2", "title": "BWM-TOPSIS for DLT selection in DPP", "doi": "10.1016/j.ifacol.2024.09.258",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2405896324016872/pdfft", "https://www.sciencedirect.com/science/article/pii/S2405896324016872/pdf"]},
    {"filename": "T2_Asadabadi2022_StratifiedBWM-TOPSIS_AnnOR.pdf", "theme": "T2", "title": "Stratified BWM TOPSIS supplier selection", "doi": "10.1007/s10479-022-04878-y",
     "urls": ["https://link.springer.com/content/pdf/10.1007/s10479-022-04878-y.pdf"]},
    {"filename": "T2_Varchandi2024_BWM-FuzzyTOPSIS_DAJ.pdf", "theme": "T2", "title": "BWM + Fuzzy TOPSIS resilient supplier", "doi": "10.1016/j.dajour.2024.100488",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2772662224000857/pdfft"]},
    {"filename": "T2_Dincer2026_HybridMCDM_EV_EoL_SciReports.pdf", "theme": "T2", "title": "Hybrid unified decoding analytics for EV battery EoL", "doi": "10.1038/s41598-026-44597-z",
     "urls": ["https://www.nature.com/articles/s41598-026-44597-z.pdf"]},
    {"filename": "T2_Lahri2026_FuzzyOrdinalMCDM_EVSupplier_IJIEOM.pdf", "theme": "T2", "title": "Fuzzy ordinal priority for EV battery supplier (India)", "doi": "10.1108/IJIEOM-07-2025-0144",
     "urls": ["https://www.emerald.com/ijieom/article-media/1311433/pdfviewer/10575135", "https://www.emerald.com/insight/content/doi/10.1108/IJIEOM-07-2025-0144"]},
    {"filename": "T2_Soriano2025_ZBWM-DEMATEL-TOPSIS_Energy_PIOS.pdf", "theme": "T2", "title": "Z-numbers BWM-DEMATEL-TOPSIS for energy storage", "doi": "10.1007/s41660-025-00652-2",
     "urls": ["https://link.springer.com/content/pdf/10.1007/s41660-025-00652-2.pdf"]},

    # ===================== T3 — Battery DPP =====================
    # SEED
    {"filename": "T3_Kisters2024_DPP_EVbattery_CLSC.pdf", "theme": "T3", "title": "DPP for EV batteries: stakeholder requirements", "doi": "10.1016/j.clscn.2024.100154",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2666791624000368/pdfft"]},
    {"filename": "T3_Kumar2023_DigitalFramework_npjMS.pdf", "theme": "T3", "title": "Digital framework EV battery circularity", "doi": "10.1038/s44296-023-00001-9",
     "urls": ["https://www.nature.com/articles/s44296-023-00001-9.pdf"]},
    # NEW
    {"filename": "T3_Berger2022_DPP_Conceptualization_JCP.pdf", "theme": "T3", "title": "Digital battery passports: conceptualization & use cases", "doi": "10.1016/j.jclepro.2022.131492",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S0959652622012471/pdfft"]},
    {"filename": "T3_Berger2023_DPP_DataReqs_CPL.pdf", "theme": "T3", "title": "DPP data requirements value chain", "doi": "10.1016/j.clpl.2023.100032",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2666791623000052/pdfft"]},
    {"filename": "T3_RufinoJunior2024_DPP_Review_Batteries.pdf", "theme": "T3", "title": "Towards Battery DPP regulations & standards 2nd-life", "doi": "10.3390/batteries10040115",
     "urls": ["https://www.mdpi.com/2313-0105/10/4/115/pdf"]},
    {"filename": "T3_Soavi2026_EU_BatteryReg_DPP_Batteries.pdf", "theme": "T3", "title": "EU Battery Reg & DPP prospects/challenges", "doi": "10.3390/batteries12030097",
     "urls": ["https://www.mdpi.com/2313-0105/12/3/97/pdf"]},
    {"filename": "T3_Tahir2025_DPP_OnlineEIS_Batteries.pdf", "theme": "T3", "title": "Battery passport + online diagnostics review", "doi": "10.3390/batteries11120442",
     "urls": ["https://www.mdpi.com/2313-0105/11/12/442/pdf"]},
    {"filename": "T3_Shen2024_Blockchain_Tracing_Energies.pdf", "theme": "T3", "title": "Blockchain tracing power batteries — game theory", "doi": "10.3390/en17122868",
     "urls": ["https://www.mdpi.com/1996-1073/17/12/2868/pdf"]},
    {"filename": "T3_DimicMisic2025_EV_Blockchain_Recycling.pdf", "theme": "T3", "title": "EV recycling sustainability via blockchain", "doi": "10.3390/recycling10020048",
     "urls": ["https://www.mdpi.com/2313-4321/10/2/48/pdf"]},

    # ===================== T4 — Anomaly Detection =====================
    # SEED
    {"filename": "T4_Chan2024_VAE_SVDD_JEnergyStorage.pdf", "theme": "T4", "title": "VAE-driven adversarial SVDD anomaly", "doi": "10.1016/j.est.2024.114267",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S2352152X24039574/pdfft"]},
    {"filename": "T4_Lee2025_DNN_AnomalyDetection_Batteries.pdf", "theme": "T4", "title": "DNN+anomaly for single-cycle lifetime", "doi": "10.3390/batteries11080288",
     "urls": ["https://www.mdpi.com/2313-0105/11/8/288/pdf"]},
    {"filename": "T4_Xu2021_StackedDenoisingAE_RESS.pdf", "theme": "T4", "title": "Stacked denoising AE life prediction", "doi": "10.1016/j.ress.2020.107396",
     "urls": [], "paywalled": True},
    # NEW (the IsolationForest gap-fillers — critical)
    {"filename": "T4_Zhang2024_IsolationForest_DT_EV_Processes.pdf", "theme": "T4", "title": "EV battery fault prediction with iForest + DT", "doi": "10.3390/pr12010136",
     "urls": ["https://www.mdpi.com/2227-9717/12/1/136/pdf"]},
    {"filename": "T4_Wu2024_Kurtosis_iForest_IJGE.pdf", "theme": "T4", "title": "Kurtosis + iForest for EV battery pack fault", "doi": "10.1080/15435075.2024.2422463",
     "urls": [], "paywalled": True},
    {"filename": "T4_Cheng2023_iForest_SlidingWindow_ESE.pdf", "theme": "T4", "title": "iForest + sliding window battery fault", "doi": "10.1002/ese3.1593",
     "urls": ["https://onlinelibrary.wiley.com/doi/pdf/10.1002/ese3.1593", "https://onlinelibrary.wiley.com/doi/epdf/10.1002/ese3.1593"]},
    {"filename": "T4_Kumar2025_HybridML_RF_Anomaly_SciReports.pdf", "theme": "T4", "title": "Hybrid ML for predictive maintenance + anomaly Li-ion", "doi": "10.1038/s41598-025-90810-w",
     "urls": ["https://www.nature.com/articles/s41598-025-90810-w.pdf"]},

    # ===================== T5 — India / regulatory =====================
    # SEED
    {"filename": "T5_Tavana2024_CircularSC_IoT_ExpSys.pdf", "theme": "T5", "title": "Circular SC + IoT EV battery", "doi": "10.1111/exsy.13395",
     "urls": [], "paywalled": True},
    {"filename": "T5_Zhai2025_LiB_CircularEconomy_Nature.pdf", "theme": "T5", "title": "Global LiB circular economy", "doi": "10.1038/s41586-025-09617-4",
     "urls": [], "paywalled": True},
    {"filename": "T5_Patel2024_SecondLife_Pathways_FrontChem.pdf", "theme": "T5", "title": "Li-ion second life pathways", "doi": "10.3389/fchem.2024.1358417",
     "urls": ["https://www.frontiersin.org/articles/10.3389/fchem.2024.1358417/pdf"]},
    {"filename": "T5_Cui2024_StanfordSecondLife_CRPS.pdf", "theme": "T5", "title": "Stanford second-life dataset", "doi": "10.1016/j.xcrp.2024.101941",
     "urls": ["https://www.sciencedirect.com/science/article/pii/S266638642400203X/pdfft"]},
    # NEW (Indian-author gap-fillers — critical)
    {"filename": "T5_Hemavathi2025_IndiaLiBLandscape_EnergyStorage.pdf", "theme": "T5", "title": "India's Li-Ion battery landscape (CSIR-CECRI)", "doi": "10.1002/est2.70244",
     "urls": ["https://onlinelibrary.wiley.com/doi/pdf/10.1002/est2.70244", "https://onlinelibrary.wiley.com/doi/epdf/10.1002/est2.70244"]},
    {"filename": "T5_Mulpuri2025_EV_DriveCycles_RSCAdv.pdf", "theme": "T5", "title": "EV battery health diverse environments (IIT-G)", "doi": "10.1039/D5RA04379D",
     "urls": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC12395550/pdf/RA-015-D5RA04379D.pdf", "https://pubs.rsc.org/en/content/articlepdf/2025/ra/d5ra04379d"]},
    {"filename": "T5_Kondru2025_IndianDriveCycles_SciReports.pdf", "theme": "T5", "title": "EV multi-mode Indian drive cycles (VIT)", "doi": "10.1038/s41598-025-02238-x",
     "urls": ["https://www.nature.com/articles/s41598-025-02238-x.pdf", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12095522/pdf/41598_2025_Article_2238.pdf"]},
    {"filename": "T5_Lakshmanan2024_2W_EV_DriveCycles_WEVJ.pdf", "theme": "T5", "title": "2W EV battery multi-mode cycles (CSIR-CEERI)", "doi": "10.3390/wevj15040145",
     "urls": ["https://www.mdpi.com/2032-6653/15/4/145/pdf"]},
    {"filename": "T5_Suman2024_CE_Barriers_IndianEV_EDS.pdf", "theme": "T5", "title": "Indian EV battery CE barriers TISM/MICMAC", "doi": "10.1007/s10668-024-05055-w",
     "urls": [], "paywalled": True},
    {"filename": "T5_NITIAayog2022_ACC_BatteryReuseRecycling_India.pdf", "theme": "T5", "title": "NITI Aayog ACC battery reuse & recycling India", "doi": "",
     "urls": ["https://www.niti.gov.in/sites/default/files/2022-07/ACC-battery-reuse-and-recycling-market-in-India_Niti-Aayog_UK.pdf"]},
    {"filename": "T5_CSE2025_EV_Battery_EPR_PolicyBrief.pdf", "theme": "T5", "title": "CSE India EV battery EPR policy brief", "doi": "",
     "urls": ["https://www.cseindia.org/content/downloadreports/12987"]},

    # ===================== T6 — Synthetic data / simulation =====================
    # SEED
    {"filename": "T6_Sulzer2021_PyBaMM_JORS.pdf", "theme": "T6", "title": "PyBaMM foundational", "doi": "10.5334/jors.309",
     "urls": ["https://openresearchsoftware.metajnl.com/articles/10.5334/jors.309"]},  # let unpaywall figure out the PDF
    {"filename": "T6_Naaz2021_GAN_BatteryData_IJER.pdf", "theme": "T6", "title": "GAN synthetic battery data", "doi": "10.1002/er.7013",
     "urls": [], "paywalled": True},
    {"filename": "T6_BatteryLife_Tan2025_KDD.pdf", "theme": "T6", "title": "BatteryLife dataset/benchmark", "doi": "10.48550/arXiv.2502.18807",
     "urls": ["https://arxiv.org/pdf/2502.18807.pdf", "https://arxiv.org/pdf/2502.18218.pdf"]},
    # NEW
    {"filename": "T6_Li2024_HighThroughputPyBaMM_JPS.pdf", "theme": "T6", "title": "Million cycles/day high-throughput PyBaMM (corrects Brosa Planella)", "doi": "10.1016/j.jpowsour.2024.234184",
     "urls": [], "paywalled": True},
    {"filename": "T6_Wang2024_PINN_LiBattery_NatComms.pdf", "theme": "T6", "title": "PINN for Li-ion degradation prognosis", "doi": "10.1038/s41467-024-48779-z",
     "urls": ["https://www.nature.com/articles/s41467-024-48779-z.pdf", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11109204/pdf/41467_2024_Article_48779.pdf"]},
    {"filename": "T6_Chowdhury2025_RCGAN_BatteryRUL_arXiv.pdf", "theme": "T6", "title": "RCGAN battery capacity time-series", "doi": "10.48550/arXiv.2503.12258",
     "urls": ["https://arxiv.org/pdf/2503.12258.pdf", "https://arxiv.org/pdf/2503.12258v1.pdf"]},
    {"filename": "T6_Sun2025_ProGAN_BatterySoH_Energy.pdf", "theme": "T6", "title": "ProGAN SoH limited samples", "doi": "10.1016/j.energy.2025.136519",
     "urls": [], "paywalled": True},
    {"filename": "T6_Jiang2024_RCVAE_Battery_arXiv.pdf", "theme": "T6", "title": "Refined Conditional VAE battery charging data", "doi": "10.48550/arXiv.2404.07577",
     "urls": ["https://arxiv.org/pdf/2404.07577.pdf", "https://arxiv.org/pdf/2404.07577v1.pdf"]},
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
        sz = f" {res.get('size', 0)//1024} KB" if res.get("size") else ""
        print(f"   -> {marker}{sz}")
        time.sleep(0.4)

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
    by_theme: dict[str, list[dict]] = {}
    for r in needs_manual:
        by_theme.setdefault(r["theme"], []).append(r)

    lines = [
        "# Manual Download List",
        "",
        "These papers could not be auto-downloaded (paywalled, or all candidate URLs blocked by Cloudflare/anti-bot).",
        "Please download manually using the links below and save with the suggested filename inside `research-papers/`.",
        "",
        "**Tip:** for paywalled papers, try:",
        "1. Your institutional access (IIIT/IIT VPN if available)",
        "2. ResearchGate (search by title — many authors post free copies)",
        "3. Sci-Hub or other shadow libraries (use at your own risk)",
        "4. Email the corresponding author directly",
        "",
    ]
    for theme in sorted(by_theme.keys()):
        lines.append(f"## {theme}")
        lines.append("")
        lines.append("| # | Suggested Filename | Title | DOI / Direct Link | Reason |")
        lines.append("|---|--------------------|-------|-------------------|--------|")
        for i, r in enumerate(by_theme[theme], 1):
            doi = r.get("doi", "")
            doi_link = f"https://doi.org/{doi}" if doi else "(no DOI; see attempt URLs below)"
            reason = "Paywalled" if r["status"] == "skipped_paywalled" else f"All {len(r.get('attempts', []))} URL attempts failed"
            lines.append(f"| {i} | `{r['filename']}` | {r['title']} | {doi_link} | {reason} |")
        lines.append("")

    # Attempt detail for failures (helps user find alt mirror)
    failures = [r for r in needs_manual if r["status"] == "failed"]
    if failures:
        lines += ["", "## Attempt detail (debug — for failed downloads only)", ""]
        for r in failures:
            lines.append(f"### {r['filename']}")
            for a in r.get("attempts", []):
                lines.append(f"- `{a['url']}` -> {a['detail'][:120]}")
            lines.append("")

    MANUAL_LIST.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
