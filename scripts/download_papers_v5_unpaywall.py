"""
v5 — last-pass downloader. Uses unpaywall.org API to find OA mirror URLs
(PMC, institutional repos, ResearchGate-fronted, etc.), then downloads via
the working /opt/anaconda3/bin/curl.

Targets only papers still missing after v3+v4.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
RESULTS_V5 = PAPERS_DIR / "_download_results_v5.json"
MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 90
MIN_BYTES = 50_000
CURL_BIN = "/opt/anaconda3/bin/curl"
UNPAYWALL_EMAIL = "rishabh.research@example.com"


def is_pdf_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def curl_fetch(url: str, target: Path, referer: str = "https://www.google.com/") -> tuple[bool, str]:
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        r = subprocess.run(
            [CURL_BIN, "-L", "-s", "--compressed", "--max-time", str(TIMEOUT),
             "-A", USER_AGENT,
             "-H", "Accept: application/pdf,*/*;q=0.8",
             "-H", "Accept-Language: en-US,en;q=0.9",
             "-H", f"Referer: {referer}",
             "-w", "%{http_code} %{content_type} %{size_download}",
             "-o", str(tmp),
             url],
            capture_output=True, text=True, timeout=TIMEOUT + 15,
        )
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "curl timeout"
    info = r.stdout.strip()
    if not tmp.exists():
        return False, f"no output ({info})"
    if is_pdf_file(tmp):
        tmp.rename(target)
        return True, f"OK {info}"
    size = tmp.stat().st_size
    head = tmp.open("rb").read(60)
    tmp.unlink()
    return False, f"not a PDF ({info}; size={size}; head={head[:40]!r})"


def unpaywall_locations(doi: str) -> list[dict]:
    """Return ALL OA locations (sorted with best first)."""
    if not doi:
        return []
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": UNPAYWALL_EMAIL}, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        locs = []
        if data.get("best_oa_location"):
            locs.append(data["best_oa_location"])
        for loc in data.get("oa_locations", []):
            if loc not in locs:
                locs.append(loc)
        return locs
    except Exception:
        return []


# Manifest of papers to retry — read from v3/v4 to find still-missing
import importlib.util
spec = importlib.util.spec_from_file_location("v3mod", ROOT / "scripts" / "download_papers_v3.py")
v3mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3mod)


def main() -> None:
    by_filename = {e["filename"]: e for e in v3mod.MANIFEST}
    # Identify still-missing: not on disk AND not paywalled
    targets = []
    for entry in v3mod.MANIFEST:
        target = PAPERS_DIR / entry["filename"]
        if is_pdf_file(target):
            continue
        if entry.get("paywalled"):
            continue
        targets.append(entry)

    print(f"v5 unpaywall pass: {len(targets)} papers still missing.\n")
    results = []
    last_mdpi = 0.0
    for i, entry in enumerate(targets, 1):
        target = PAPERS_DIR / entry["filename"]
        print(f"[{i:02d}/{len(targets)}] {entry['filename']}")
        attempts = []

        # Special-case: PyBaMM JORS — direct download URL
        if entry["doi"] == "10.5334/jors.309":
            for u in [
                "https://openresearchsoftware.metajnl.com/articles/10.5334/jors.309/galley/506/download/",
                "https://storage.googleapis.com/jnl-up-j-jors-files/journals/1/articles/309/submission/proof/309-1-1448-1-10-20210608.pdf",
                "https://openresearchsoftware.metajnl.com/article/10.5334/jors.309",
            ]:
                ok, det = curl_fetch(u, target)
                attempts.append({"url": u, "ok": ok, "detail": det[:160]})
                if ok:
                    break

        # Try unpaywall locations
        if not is_pdf_file(target):
            locs = unpaywall_locations(entry["doi"])
            tried_urls = {a["url"] for a in attempts}
            for loc in locs:
                pdf_url = loc.get("url_for_pdf") or loc.get("url")
                if not pdf_url or pdf_url in tried_urls:
                    continue
                tried_urls.add(pdf_url)
                # MDPI cooldown
                if "mdpi.com" in pdf_url:
                    wait = 10.0 - (time.time() - last_mdpi)
                    if wait > 0:
                        time.sleep(wait)
                ok, det = curl_fetch(pdf_url, target)
                if "mdpi.com" in pdf_url:
                    last_mdpi = time.time()
                host = loc.get("host_type", "?") + "/" + (loc.get("repository_institution") or loc.get("evidence", "?"))[:40]
                attempts.append({"url": f"[unpaywall:{host}] {pdf_url}", "ok": ok, "detail": det[:160]})
                if ok:
                    break

        status = "downloaded" if is_pdf_file(target) else "failed"
        size = target.stat().st_size if is_pdf_file(target) else None
        results.append({"filename": entry["filename"], "theme": entry["theme"], "title": entry["title"], "doi": entry["doi"], "status": status, "size": size, "attempts": attempts})
        marker = "OK" if status == "downloaded" else "FAIL"
        sz = f" {size//1024} KB" if size else ""
        print(f"   -> {marker}{sz}")

    RESULTS_V5.write_text(json.dumps(results, indent=2))

    # Build final manual list: paywalled + still-failed
    paywalled_entries = [e for e in v3mod.MANIFEST if e.get("paywalled")]
    still_failed_filenames = {r["filename"] for r in results if r["status"] == "failed"}
    still_failed_entries = [e for e in v3mod.MANIFEST if e["filename"] in still_failed_filenames]
    write_manual_list(paywalled_entries, still_failed_entries)

    print(f"\n=== v5 summary ===")
    print(f"Recovered:    {sum(1 for r in results if r['status'] == 'downloaded')}")
    print(f"Still failed: {sum(1 for r in results if r['status'] == 'failed')}")
    total_pdfs = sum(1 for e in v3mod.MANIFEST if is_pdf_file(PAPERS_DIR / e["filename"]))
    print(f"Total PDFs on disk: {total_pdfs} / {len(v3mod.MANIFEST)}")
    print(f"Manual list updated: {MANUAL_LIST}")


def write_manual_list(paywalled: list[dict], still_failed: list[dict]) -> None:
    needs_manual = [(e, "paywalled") for e in paywalled] + [(e, "blocked") for e in still_failed]
    if not needs_manual:
        MANUAL_LIST.write_text("# Manual Download List\n\nAll papers downloaded automatically.\n")
        return
    by_theme: dict[str, list[tuple[dict, str]]] = {}
    for entry, kind in needs_manual:
        by_theme.setdefault(entry["theme"], []).append((entry, kind))

    lines = [
        "# Manual Download List",
        "",
        f"**{len(needs_manual)} papers** still need manual download (out of 58 in the candidate pool).",
        "",
        "## Why",
        "- `paywalled`: behind Elsevier/ACS/Wiley/Springer/Nature-flagship paywall — needs institutional access or Sci-Hub.",
        "- `blocked`: open-access papers blocked by anti-bot (Cloudflare on Wiley/OUP/Emerald, Akamai on ScienceDirect, etc.). **Open the DOI link in your browser and the PDF will download cleanly** — programmatic access is being blocked, not paid access.",
        "",
        "## How to download",
        "1. **Cmd-click each DOI link below** to open in a new tab.",
        "2. Click the publisher's PDF download button.",
        "3. Save with the suggested filename into `research-papers/`.",
        "",
        "Faster alt: use a browser extension like [Open Access Helper](https://www.oahelper.org/) which auto-opens OA PDF link.",
        "",
    ]
    theme_titles = {
        "T1": "Theme 1 — Hybrid ML for SoH/RUL",
        "T2": "Theme 2 — MCDM for circular economy",
        "T3": "Theme 3 — Battery Digital Product Passport",
        "T4": "Theme 4 — Anomaly detection",
        "T5": "Theme 5 — India regulatory / market",
        "T6": "Theme 6 — Physics simulation / synthetic data",
    }
    for theme in sorted(by_theme.keys()):
        lines.append(f"## {theme_titles.get(theme, theme)}")
        lines.append("")
        lines.append("| # | Suggested Filename | Title | DOI / Link | Reason |")
        lines.append("|---|--------------------|-------|------------|--------|")
        for i, (e, kind) in enumerate(by_theme[theme], 1):
            doi = e.get("doi", "")
            doi_link = f"[{doi}](https://doi.org/{doi})" if doi else "(no DOI)"
            lines.append(f"| {i} | `{e['filename']}` | {e['title']} | {doi_link} | {kind} |")
        lines.append("")

    MANUAL_LIST.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
