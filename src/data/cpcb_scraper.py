"""
CPCB EPR Portal scraper for India-specific battery regulatory data.

Scrapes the public CPCB EPR Battery national dashboard to extract:
- Registered producers, recyclers, refurbishers (count + state-wise)
- Quarterly recovery volumes (Form 4 aggregates)
- EPR certificate transaction data (where public)

Source: https://eprbattery.cpcb.gov.in/user/nationaldashboard
Note: Respect robots.txt and rate limits. Data is public; scrape gently.

Usage:
    conda run -n Eco-Research python src/data/cpcb_scraper.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DIR


CPCB_DASHBOARD_URL = "https://eprbattery.cpcb.gov.in/user/nationaldashboard"
RATE_LIMIT_SECONDS = 2.0


def fetch_dashboard_html(url: str = CPCB_DASHBOARD_URL) -> Optional[str]:
    """Fetch the CPCB national dashboard HTML.

    CPCB dashboard is a JS-rendered page (Angular). For production scraping
    consider Selenium/Playwright. This stub uses requests for initial exploration.
    """
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Academic research; Digital Economics coursework; public data scraper)",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_SECONDS)
        return resp.text
    except Exception as exc:
        print(f"Failed to fetch dashboard: {exc}")
        return None


def parse_dashboard(html: str) -> pd.DataFrame:
    """Parse dashboard HTML into structured records.

    The CPCB dashboard uses JS for data rendering. For serious scraping,
    instrument Selenium/Playwright and wait for Angular to populate the DOM.
    For this project stub, we document the structure and return placeholders.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    # The dashboard exposes aggregated counts in widgets. These selectors
    # need to be confirmed against the live DOM (likely .widget, .card, etc.)
    records = []
    # Placeholder: widgets = soup.select(".dashboard-widget")
    # for w in widgets:
    #     records.append({
    #         "widget": w.select_one(".title").text.strip(),
    #         "value": w.select_one(".count").text.strip(),
    #     })

    return pd.DataFrame(records)


def scrape_cpcb_dashboard(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Full scrape pipeline: fetch + parse + save."""
    output_dir = Path(output_dir) if output_dir else RAW_DIR / "cpcb_epr"
    output_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_dashboard_html()
    if not html:
        print("No HTML fetched. Is the CPCB portal reachable?")
        return pd.DataFrame()

    # Save raw HTML for inspection / later re-parsing
    (output_dir / "dashboard_raw.html").write_text(html)

    df = parse_dashboard(html)

    if not df.empty:
        df.to_csv(output_dir / "cpcb_dashboard.csv", index=False)
        print(f"Saved {len(df)} records to {output_dir / 'cpcb_dashboard.csv'}")
    else:
        print("Dashboard fetched but no widgets parsed.")
        print("Note: CPCB dashboard uses JS rendering; use Selenium/Playwright for full scraping.")
        print(f"Raw HTML saved to {output_dir / 'dashboard_raw.html'} for manual inspection.")

    return df


def main():
    print("CPCB EPR Battery Portal -- India-specific regulatory data")
    print("=" * 60)
    print(f"URL: {CPCB_DASHBOARD_URL}")
    print("Public national dashboard (registered producers, recyclers, recovery volumes)")
    print()
    scrape_cpcb_dashboard()


if __name__ == "__main__":
    main()
