"""
v6 — Playwright headless-browser pass.

For each remaining paper, navigate via doi.org to the publisher landing page in
a real Chrome instance. Cloudflare/Akamai JS challenges are auto-solved by the
browser. We then either:
  (a) intercept a download triggered by the publisher's PDF button, or
  (b) request the embedded PDF URL through the now-authenticated browser context
      (so cookies/session-tokens are attached).

Reads what's still missing on disk after v5.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "research-papers"
RESULTS_V6 = PAPERS_DIR / "_download_results_v6.json"
MANUAL_LIST = PAPERS_DIR / "MANUAL_DOWNLOAD_LIST.md"
MIN_BYTES = 50_000

import importlib.util
spec = importlib.util.spec_from_file_location("v3mod", ROOT / "scripts" / "download_papers_v3.py")
v3mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3mod)


def is_pdf_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


# Per-publisher PDF link selectors (best-guess; fall back to regex over HTML)
PDF_LINK_SELECTORS = [
    'a[href*=".pdf"]',
    'a[href*="/pdf/"]',
    'a[href*="/pdfdirect/"]',
    'a[href*="/pdfft"]',
    'a[href*="/epdf/"]',
    'a:has-text("Download PDF")',
    'a:has-text("PDF")',
    'a[data-track-action="download pdf"]',
    'a[data-test="article-download-pdf"]',
    'a[title*="PDF"]',
]


async def fetch_via_browser(context, doi: str, target: Path, log: list) -> bool:
    """Open doi.org/{doi}, wait for redirect, find PDF download link, save it."""
    page = await context.new_page()
    try:
        # Step 1: follow DOI to publisher landing page
        landing = f"https://doi.org/{doi}"
        try:
            await page.goto(landing, wait_until="domcontentloaded", timeout=45000)
            # Wait extra to let Cloudflare JS challenge finish
            await page.wait_for_load_state("networkidle", timeout=30000)
        except Exception as e:
            log.append(f"goto landing fail: {type(e).__name__}: {str(e)[:120]}")

        landing_url = page.url
        log.append(f"landed at: {landing_url}")

        # Step 2: Try to find a PDF download link in the DOM
        pdf_url = None
        for sel in PDF_LINK_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    href = await el.get_attribute("href")
                    if href and ("pdf" in href.lower()):
                        if href.startswith("/"):
                            from urllib.parse import urljoin
                            href = urljoin(landing_url, href)
                        pdf_url = href
                        log.append(f"selector {sel} -> {pdf_url[:120]}")
                        break
            except Exception:
                continue

        # Step 3: Fallback — regex over page HTML for any pdf-looking link
        if not pdf_url:
            html = await page.content()
            patterns = [
                r'href="(https?://[^"]+\.pdf[^"]*)"',
                r'href="(https?://[^"]+/pdfft[^"]*)"',
                r'href="(https?://[^"]+/pdfdirect/[^"]*)"',
                r'href="(https?://[^"]+/epdf/[^"]*)"',
                r'"pdfDownload"\s*:\s*"(https?://[^"]+)"',
            ]
            for pat in patterns:
                m = re.search(pat, html)
                if m:
                    pdf_url = m.group(1).encode().decode("unicode_escape")
                    log.append(f"regex {pat[:30]} -> {pdf_url[:100]}")
                    break

        if not pdf_url:
            log.append("no PDF URL found in landing page")
            return False

        # Step 4: Fetch the PDF using the browser's now-authenticated context
        try:
            resp = await context.request.get(pdf_url, timeout=60000, headers={"Referer": landing_url})
            if resp.status != 200:
                log.append(f"PDF fetch HTTP {resp.status}")
                return False
            body = await resp.body()
            if body[:5] != b"%PDF-":
                log.append(f"not a PDF (head={body[:40]!r})")
                return False
            if len(body) < MIN_BYTES:
                log.append(f"too small ({len(body)} bytes)")
                return False
            target.write_bytes(body)
            log.append(f"OK {len(body)} bytes")
            return True
        except Exception as e:
            log.append(f"context.request fail: {type(e).__name__}: {str(e)[:120]}")
            return False
    finally:
        await page.close()


async def main():
    targets = []
    for entry in v3mod.MANIFEST:
        target = PAPERS_DIR / entry["filename"]
        if is_pdf_file(target):
            continue
        if entry.get("paywalled"):
            continue
        if not entry.get("doi"):
            continue
        targets.append(entry)

    print(f"v6 Playwright pass: {len(targets)} papers still missing.\n")

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Asia/Kolkata",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        for i, entry in enumerate(targets, 1):
            target = PAPERS_DIR / entry["filename"]
            log: list[str] = []
            print(f"[{i:02d}/{len(targets)}] {entry['filename']}")
            ok = False
            try:
                ok = await fetch_via_browser(context, entry["doi"], target, log)
            except Exception as e:
                log.append(f"top-level error: {type(e).__name__}: {e}")
            status = "downloaded" if ok and is_pdf_file(target) else "failed"
            size = target.stat().st_size if status == "downloaded" else None
            results.append({
                "filename": entry["filename"], "theme": entry["theme"],
                "title": entry["title"], "doi": entry["doi"],
                "status": status, "size": size, "log": log,
            })
            sz = f" {size//1024} KB" if size else ""
            print(f"   -> {'OK' if status == 'downloaded' else 'FAIL'}{sz}")
            for ln in log[-3:]:
                print(f"      {ln[:140]}")

        await browser.close()

    RESULTS_V6.write_text(json.dumps(results, indent=2))

    # Final manual list — paywalled + still-failed
    paywalled = [e for e in v3mod.MANIFEST if e.get("paywalled")]
    still_failed_files = {r["filename"] for r in results if r["status"] == "failed"}
    still_failed = [e for e in v3mod.MANIFEST if e["filename"] in still_failed_files]
    write_manual_list(paywalled, still_failed)

    print(f"\n=== v6 summary ===")
    print(f"Recovered:    {sum(1 for r in results if r['status'] == 'downloaded')}")
    print(f"Still failed: {sum(1 for r in results if r['status'] == 'failed')}")
    total = sum(1 for e in v3mod.MANIFEST if is_pdf_file(PAPERS_DIR / e["filename"]))
    print(f"Total PDFs on disk: {total} / {len(v3mod.MANIFEST)}")
    print(f"Manual list: {MANUAL_LIST}")


def write_manual_list(paywalled: list[dict], still_failed: list[dict]) -> None:
    needs_manual = [(e, "paywalled") for e in paywalled] + [(e, "blocked") for e in still_failed]
    if not needs_manual:
        MANUAL_LIST.write_text("# Manual Download List\n\nAll papers downloaded automatically.\n")
        return
    by_theme: dict[str, list] = {}
    for entry, kind in needs_manual:
        by_theme.setdefault(entry["theme"], []).append((entry, kind))
    lines = [
        "# Manual Download List",
        "",
        f"**{len(needs_manual)} papers** still need manual download (out of 58 in candidate pool).",
        "",
        "## Why",
        "- `paywalled`: behind Elsevier/ACS/Wiley/Springer/Nature-flagship paywall — needs institutional access or Sci-Hub.",
        "- `blocked`: open-access papers blocked by anti-bot. **Open the DOI link in browser and the PDF will download cleanly.**",
        "",
        "## How to download",
        "Cmd-click each DOI link to open in a new tab → click the publisher's PDF button → save with the suggested filename into `research-papers/`.",
        "",
    ]
    titles = {"T1": "Hybrid ML for SoH/RUL", "T2": "MCDM for circular economy", "T3": "Battery DPP",
              "T4": "Anomaly detection", "T5": "India regulatory / market", "T6": "Physics simulation / synthetic data"}
    for theme in sorted(by_theme.keys()):
        lines += [f"## {theme} — {titles.get(theme, theme)}", "",
                  "| # | Suggested Filename | Title | DOI / Link | Reason |",
                  "|---|--------------------|-------|------------|--------|"]
        for i, (e, kind) in enumerate(by_theme[theme], 1):
            doi = e.get("doi", "")
            doi_link = f"[{doi}](https://doi.org/{doi})" if doi else "(no DOI)"
            lines.append(f"| {i} | `{e['filename']}` | {e['title']} | {doi_link} | {kind} |")
        lines.append("")
    MANUAL_LIST.write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
