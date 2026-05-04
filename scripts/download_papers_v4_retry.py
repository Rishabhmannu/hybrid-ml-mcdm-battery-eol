"""
v4 — focused retry of v3 failures.

Strategy by host:
  - MDPI (bulk of failures): plain curl with 10s cooldown between requests (rate-limit friendly)
  - ScienceDirect: scrape article landing page for the real /pdfft URL with token
  - Wiley / OUP / Springer: 1 attempt with curl, otherwise punt to manual

Reads v3 results JSON to know which papers still need attention; only retries those.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
RESULTS_V3 = PAPERS_DIR / "_download_results_v3.json"
RESULTS_V4 = PAPERS_DIR / "_download_results_v4.json"
MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT = 90
MIN_BYTES = 50_000

# CRITICAL: inside Eco-Research conda env, `curl` resolves to /opt/local/bin/curl
# (MacPorts) which has a TLS fingerprint MDPI's Cloudflare WAF rejects. The
# anaconda curl works fine, so we hardcode it here.
import shutil as _sh
CURL_BIN = "/opt/anaconda3/bin/curl" if Path("/opt/anaconda3/bin/curl").exists() else (_sh.which("curl") or "curl")


def is_pdf_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def curl_fetch(url: str, target: Path, referer: str = "https://www.google.com/") -> tuple[bool, str]:
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        r = subprocess.run(
            [
                CURL_BIN, "-L", "-s", "--compressed", "--max-time", str(TIMEOUT),
                "-A", USER_AGENT,
                "-H", "Accept: application/pdf,*/*;q=0.8",
                "-H", "Accept-Language: en-US,en;q=0.9",
                "-H", f"Referer: {referer}",
                "-w", "%{http_code} %{content_type} %{size_download}",
                "-o", str(tmp),
                url,
            ],
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
    return False, f"not a PDF ({info}; size={size}; head={head!r})"


def scrape_sciencedirect_pdf_url(article_url: str) -> str | None:
    """ScienceDirect OA articles embed the real signed PDF URL inside the landing-page
    HTML (look for `pdf-download-btn-link` href and similar)."""
    try:
        r = requests.get(article_url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://scholar.google.com/",
        }, timeout=30, allow_redirects=True)
        if r.status_code != 200:
            return None
        # Try several patterns Elsevier uses on landing pages
        patterns = [
            r'"pdfDownload":"(https?://[^"]+)"',
            r'"pdfUrl":"(https?://[^"]+)"',
            r'"linkType":"DOWNLOAD"[^}]*"url":"(https?://[^"]+)"',
            r'href="([^"]+pii[^"]+/pdfft[^"]*)"',
        ]
        for pat in patterns:
            m = re.search(pat, r.text)
            if m:
                url = m.group(1).encode().decode("unicode_escape")
                return url
        return None
    except Exception:
        return None


def main() -> None:
    with RESULTS_V3.open() as f:
        v3 = json.load(f)

    # Load manifest so we can re-issue the original URL list per paper
    # (just import from v3.py for simplicity)
    import importlib.util
    spec = importlib.util.spec_from_file_location("v3mod", ROOT / "scripts" / "download_papers_v3.py")
    v3mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3mod)
    by_filename = {e["filename"]: e for e in v3mod.MANIFEST}

    failed = [r for r in v3 if r["status"] == "failed"]
    print(f"v4 retrying {len(failed)} v3 failures.\n")
    results = []
    last_mdpi = 0.0

    for i, r in enumerate(failed, 1):
        entry = by_filename[r["filename"]]
        target = PAPERS_DIR / entry["filename"]

        if is_pdf_file(target):  # got it via another run
            results.append({"filename": entry["filename"], "theme": entry["theme"], "title": entry["title"], "doi": entry["doi"], "status": "already_present"})
            continue

        print(f"[{i:02d}/{len(failed)}] {entry['filename']}")
        attempts = []

        for url in entry.get("urls", []):
            host_is_mdpi = "mdpi.com" in url
            host_is_sd = "sciencedirect.com" in url

            if host_is_mdpi:
                # MDPI: enforce >= 10s gap from last MDPI hit
                wait = 10.0 - (time.time() - last_mdpi)
                if wait > 0:
                    print(f"   ...mdpi cooldown {wait:.1f}s")
                    time.sleep(wait)
                ok, detail = curl_fetch(url, target)
                last_mdpi = time.time()
                attempts.append({"url": url, "ok": ok, "detail": detail[:200]})
                if ok:
                    break
                continue

            if host_is_sd and "/pdfft" in url:
                landing = url.split("/pdfft")[0]
                real_pdf = scrape_sciencedirect_pdf_url(landing)
                attempts.append({"url": f"[scrape] {landing}", "ok": bool(real_pdf), "detail": f"resolved to {real_pdf}" if real_pdf else "no link found"})
                if real_pdf:
                    ok, detail = curl_fetch(real_pdf, target, referer=landing)
                    attempts.append({"url": real_pdf, "ok": ok, "detail": detail[:200]})
                    if ok:
                        break
                continue

            # Generic fallback
            ok, detail = curl_fetch(url, target)
            attempts.append({"url": url, "ok": ok, "detail": detail[:200]})
            if ok:
                break

        status = "downloaded" if is_pdf_file(target) else "failed"
        size = target.stat().st_size if is_pdf_file(target) else None
        results.append({"filename": entry["filename"], "theme": entry["theme"], "title": entry["title"], "doi": entry["doi"], "status": status, "size": size, "attempts": attempts})
        marker = "OK" if status == "downloaded" else "FAIL"
        sz = f" {size//1024} KB" if size else ""
        print(f"   -> {marker}{sz}")

    RESULTS_V4.write_text(json.dumps(results, indent=2))

    # Combined manual list (v3 paywalled + v4 still-failed)
    paywalled = [r for r in v3 if r["status"] == "skipped_paywalled"]
    still_failed = [r for r in results if r["status"] == "failed"]
    write_manual_list(paywalled, still_failed, by_filename)

    print(f"\n=== v4 retry summary ===")
    print(f"Recovered:   {sum(1 for r in results if r['status'] == 'downloaded')}")
    print(f"Still failed:{sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Already present (race): {sum(1 for r in results if r['status'] == 'already_present')}")
    print(f"Manual list: {MANUAL_LIST}")


def write_manual_list(paywalled: list[dict], still_failed: list[dict], by_filename: dict) -> None:
    needs_manual = paywalled + still_failed
    if not needs_manual:
        MANUAL_LIST.write_text("# Manual Download List\n\nAll papers downloaded automatically.\n")
        return
    # Use entry from manifest to populate links; fall back to result fields
    by_theme: dict[str, list[tuple[str, dict]]] = {}
    for r in needs_manual:
        entry = by_filename.get(r["filename"], {})
        theme = r.get("theme") or entry.get("theme", "?")
        kind = "paywalled" if r in paywalled else "blocked"
        by_theme.setdefault(theme, []).append((kind, {**entry, **r}))

    lines = [
        "# Manual Download List",
        "",
        f"**{len(needs_manual)} papers** could not be auto-downloaded — please grab manually.",
        "",
        "## Why each kind failed",
        "- `paywalled`: Elsevier (Energy, J Power Sources, RESS), ACS, Wiley (Expert Systems, IJER, IJGE), Springer (Env Dev Sustain, Process Integration), Nature flagship — need institutional access OR Sci-Hub.",
        "- `blocked`: open-access papers blocked by anti-bot (Cloudflare on MDPI/Wiley/OUP, Akamai on ScienceDirect). **Open the DOI link in your browser and the PDF will download cleanly** — we just can't do it programmatically without browser automation.",
        "",
        "## Tip for batch download",
        "Open all the DOI links below in browser tabs (Cmd-click), then save each PDF using the suggested filename.",
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
        for i, (kind, r) in enumerate(by_theme[theme], 1):
            doi = r.get("doi", "")
            if doi:
                doi_link = f"[{doi}](https://doi.org/{doi})"
            else:
                # NITI Aayog & CSE shouldn't appear here; if they do, fall back to first url
                urls = r.get("urls") or []
                doi_link = f"[direct]({urls[0]})" if urls else "(no link)"
            lines.append(f"| {i} | `{r['filename']}` | {r.get('title', '')} | {doi_link} | {kind} |")
        lines.append("")

    MANUAL_LIST.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
