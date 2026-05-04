"""
Selenium fallback for sources that block headless requests:
- EUR-Lex Regulation 2023/1542 (returns 202 to non-browser clients)
- WRI India Second Charge article (Cloudflare 403)
- CPCB EPR national dashboard (Angular SPA)

Usage:
    conda activate Eco-Research
    python scripts/download_via_selenium.py --target eu | wri | cpcb | all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "data" / "regulatory"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def make_driver(download_dir: Path | None = None) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1280,1800")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    )
    if download_dir:
        download_dir.mkdir(parents=True, exist_ok=True)
        prefs = {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
        }
        opts.add_experimental_option("prefs", prefs)
    # Force webdriver-manager to fetch a chromedriver matching the installed Chrome
    # (homebrew's /opt/homebrew/bin/chromedriver is stale and Selenium picks it up otherwise)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


# --------------------------------------------------------------------------
# EU Regulation 2023/1542
# --------------------------------------------------------------------------

def download_eu_via_selenium() -> None:
    """Open EUR-Lex in a real browser; transfer cookies to requests for the PDF."""
    out = REG_DIR / "eu" / "EU_Reg_2023_1542_full.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    driver = make_driver()
    try:
        # 1) Visit the regulation landing page so EUR-Lex sets its cookies
        landing = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1542"
        print(f"[EU] visiting landing {landing}")
        driver.get(landing)
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)

        # 2) Pull cookies from the browser
        sel_cookies = driver.get_cookies()
        ua = driver.execute_script("return navigator.userAgent;")
    finally:
        driver.quit()

    cookies = {c["name"]: c["value"] for c in sel_cookies}
    print(f"[EU] {len(cookies)} cookies acquired; downloading PDF via requests")

    pdf_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32023R1542"
    headers = {
        "User-Agent": ua,
        "Referer": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1542",
        "Accept": "application/pdf,application/xhtml+xml,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for attempt in range(10):
        r = requests.get(pdf_url, headers=headers, cookies=cookies, stream=True, timeout=120, allow_redirects=True)
        ct = r.headers.get("content-type", "").lower()
        if r.status_code == 200 and "pdf" in ct:
            total = 0
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            print(f"[EU] saved {total/1024:.1f} KB -> {out}")
            return
        print(f"[EU] attempt {attempt+1}: status={r.status_code} ctype={ct}")
        r.close()
        time.sleep(4)
    print("[EU] PDF still not ready; falling back to print-page-as-PDF")
    print_page_as_pdf("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1542", out)


def print_page_as_pdf(url: str, out: Path) -> None:
    """Use Chrome DevTools Protocol to print a rendered page as a PDF."""
    import base64
    driver = make_driver()
    try:
        print(f"[print] {url}")
        driver.get(url)
        time.sleep(8)  # let everything render
        result = driver.execute_cdp_cmd("Page.printToPDF", {
            "landscape": False,
            "printBackground": True,
            "preferCSSPageSize": True,
        })
        data = base64.b64decode(result["data"])
        out.write_bytes(data)
        print(f"[print] saved {len(data)/1024:.1f} KB -> {out}")
    finally:
        driver.quit()


# --------------------------------------------------------------------------
# WRI Second Charge
# --------------------------------------------------------------------------

def download_wri_via_selenium() -> None:
    url = "https://wri-india.org/perspectives/second-charge-unlocking-second-life-potential-ev-batteries"
    html_out = REG_DIR / "india_market" / "WRI_Second_Charge_2024.html"
    pdf_out = REG_DIR / "india_market" / "WRI_Second_Charge_2024.pdf"
    html_out.parent.mkdir(parents=True, exist_ok=True)

    driver = make_driver()
    try:
        print(f"[WRI] visiting {url}")
        driver.get(url)
        # Cloudflare interstitial — wait until the challenge clears
        for attempt in range(12):
            time.sleep(5)
            title = (driver.title or "").lower()
            if "just a moment" not in title and "checking your browser" not in title:
                break
            print(f"[WRI]   waiting for Cloudflare clearance ({attempt+1}/12), title='{title}'")
        else:
            print("[WRI] Cloudflare challenge did not clear in 60s")
        time.sleep(3)
        html = driver.page_source
        html_out.write_text(html, encoding="utf-8")
        print(f"[WRI] HTML saved ({len(html)/1024:.1f} KB) -> {html_out}; title='{driver.title}'")

        # Also save a print-to-PDF rendering
        import base64
        result = driver.execute_cdp_cmd("Page.printToPDF", {
            "landscape": False,
            "printBackground": True,
            "preferCSSPageSize": True,
        })
        data = base64.b64decode(result["data"])
        pdf_out.write_bytes(data)
        print(f"[WRI] PDF saved ({len(data)/1024:.1f} KB) -> {pdf_out}")
    finally:
        driver.quit()


# --------------------------------------------------------------------------
# CPCB EPR National Dashboard
# --------------------------------------------------------------------------

def scrape_cpcb_via_selenium() -> None:
    """Scrape the CPCB EPR national dashboard widgets after Angular renders."""
    url = "https://eprbattery.cpcb.gov.in/user/nationaldashboard"
    out_dir = RAW_DIR / "cpcb_epr"
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = make_driver()
    try:
        print(f"[CPCB] visiting {url}")
        driver.get(url)
        # Angular needs time to fetch widget data
        WebDriverWait(driver, 45).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(15)

        html = driver.page_source
        (out_dir / "dashboard_raw.html").write_text(html, encoding="utf-8")
        print(f"[CPCB] HTML saved ({len(html)/1024:.1f} KB)")

        # Also save a screenshot for human inspection
        driver.set_window_size(1400, 3500)
        time.sleep(2)
        screenshot = out_dir / "dashboard.png"
        driver.save_screenshot(str(screenshot))
        print(f"[CPCB] screenshot saved -> {screenshot}")

        # Save print-to-PDF for offline review
        import base64
        result = driver.execute_cdp_cmd("Page.printToPDF", {
            "landscape": False,
            "printBackground": True,
        })
        pdf_path = out_dir / "dashboard.pdf"
        pdf_path.write_bytes(base64.b64decode(result["data"]))
        print(f"[CPCB] PDF saved -> {pdf_path}")

        # Try to extract any visible numbers (heuristic)
        import re
        text = driver.find_element(By.TAG_NAME, "body").text
        (out_dir / "dashboard_text.txt").write_text(text, encoding="utf-8")
        nums = re.findall(r"([A-Za-z][^\n]{2,40})\s*\n\s*(\d[\d,]*)", text)
        print(f"[CPCB] extracted {len(nums)} candidate (label, value) pairs")
        if nums:
            import csv
            with open(out_dir / "dashboard_kv.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["label", "value"])
                for label, val in nums:
                    w.writerow([label.strip(), val.replace(",", "")])
            print(f"[CPCB] candidate KV saved -> {out_dir/'dashboard_kv.csv'}")
    finally:
        driver.quit()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

TARGETS = {
    "eu": download_eu_via_selenium,
    "wri": download_wri_via_selenium,
    "cpcb": scrape_cpcb_via_selenium,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="all", choices=list(TARGETS.keys()) + ["all"])
    args = p.parse_args()
    targets = list(TARGETS.keys()) if args.target == "all" else [args.target]
    for t in targets:
        print(f"\n=== {t.upper()} ===")
        try:
            TARGETS[t]()
        except Exception as exc:
            print(f"[{t}] FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
