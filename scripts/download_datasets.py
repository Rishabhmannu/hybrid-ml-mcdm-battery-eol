"""
Automated downloader for Field2Fork-EoL project datasets.

Handles the awkward downloads:
- EUR-Lex CELEX (needs proper headers)
- WRI India page (blocks default UAs)
- Stanford OSF (walks the public API tree)
- BWMR PDFs (CPCB / MoEFCC gazette URLs)
- NASA Randomized & Recommissioned (multiple mirror attempts)

Usage:
    conda activate Eco-Research
    python scripts/download_datasets.py --target eu | wri | osf | bwmr | nasa | all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "data" / "regulatory"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def http_get(url: str, headers: dict | None = None, stream: bool = False, timeout: int = 60):
    h = {"User-Agent": BROWSER_UA, "Accept": "*/*"}
    if headers:
        h.update(headers)
    return requests.get(url, headers=h, stream=stream, timeout=timeout, allow_redirects=True)


def save_stream(resp: requests.Response, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    return total


# --------------------------------------------------------------------------
# EU Regulation 2023/1542 — EUR-Lex
# --------------------------------------------------------------------------

def download_eu_reg() -> None:
    """EUR-Lex returns HTTP 202 to indicate the PDF is being generated; poll the same URL with a session until 200."""
    out = REG_DIR / "eu" / "EU_Reg_2023_1542_full.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": BROWSER_UA,
        "Referer": "https://eur-lex.europa.eu/",
        "Accept": "application/pdf,application/xhtml+xml,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })

    # Warm up session (set EUR-Lex cookies)
    session.get("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1542", timeout=30)

    pdf_urls = [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32023R1542",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02023R1542-20240718",
    ]
    for url in pdf_urls:
        print(f"[EU] {url[:90]}...")
        for attempt in range(8):
            r = session.get(url, stream=True, timeout=90, allow_redirects=True)
            ct = r.headers.get("content-type", "").lower()
            if r.status_code == 200 and "pdf" in ct:
                size = save_stream(r, out)
                print(f"[EU] saved {size/1024:.1f} KB -> {out}")
                return
            retry_after = r.headers.get("Retry-After", "5")
            wait = max(2, min(int(retry_after) if retry_after.isdigit() else 5, 15))
            print(f"[EU]   attempt {attempt+1}: status={r.status_code} ctype={ct} -> wait {wait}s")
            r.close()
            time.sleep(wait)

    # Fallback: HTML version
    html_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R1542"
    print("[EU] PDF still pending; downloading HTML fallback")
    r = session.get(html_url, timeout=60)
    r.raise_for_status()
    html_out = REG_DIR / "eu" / "EU_Reg_2023_1542_full.html"
    html_out.write_bytes(r.content)
    print(f"[EU] HTML fallback saved -> {html_out} ({len(r.content)/1024:.1f} KB)")


# --------------------------------------------------------------------------
# WRI India
# --------------------------------------------------------------------------

def download_wri() -> None:
    url = "https://wri-india.org/perspectives/second-charge-unlocking-second-life-potential-ev-batteries"
    out = REG_DIR / "india_market" / "WRI_Second_Charge_2024.html"
    headers = {
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    print(f"[WRI] {url}")
    r = http_get(url, headers=headers, timeout=60)
    print(f"[WRI] status={r.status_code} bytes={len(r.content)}")
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f"[WRI] saved -> {out}")


# --------------------------------------------------------------------------
# Stanford Second-Life — OSF (public, no auth needed)
# --------------------------------------------------------------------------

OSF_PROJECT = "fns57"
OSF_API = "https://api.osf.io/v2"


def osf_walk(folder_url: str, local_root: Path, indent: int = 0) -> int:
    """Recursively walk an OSF folder URL, downloading every file."""
    n_files = 0
    next_url = folder_url
    while next_url:
        r = http_get(next_url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("data", []):
            name = item["attributes"]["name"]
            kind = item["attributes"]["kind"]
            if kind == "folder":
                sub_url = item["relationships"]["files"]["links"]["related"]["href"]
                sub_local = local_root / name
                sub_local.mkdir(parents=True, exist_ok=True)
                print("  " * indent + f"[osf] folder {name}/")
                n_files += osf_walk(sub_url, sub_local, indent + 1)
            else:
                dl = item["links"].get("download")
                if not dl:
                    continue
                target = local_root / name
                if target.exists() and target.stat().st_size > 0:
                    print("  " * indent + f"[osf] skip {name} (exists)")
                    n_files += 1
                    continue
                print("  " * indent + f"[osf] file   {name}")
                rr = http_get(dl, stream=True, timeout=300)
                rr.raise_for_status()
                size = save_stream(rr, target)
                print("  " * indent + f"          {size/1024:.1f} KB")
                n_files += 1
                time.sleep(0.3)  # be polite
        next_url = payload.get("links", {}).get("next")
    return n_files


def download_stanford() -> None:
    out = RAW_DIR / "stanford"
    out.mkdir(parents=True, exist_ok=True)
    root = f"{OSF_API}/nodes/{OSF_PROJECT}/files/osfstorage/"
    print(f"[stanford] walking OSF project {OSF_PROJECT}")
    n = osf_walk(root, out)
    print(f"[stanford] downloaded {n} files into {out}")


# --------------------------------------------------------------------------
# BWMR (India) — direct PDF URLs from CPCB / MoEFCC
# --------------------------------------------------------------------------

def download_bwmr() -> None:
    out_dir = REG_DIR / "bwmr"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "BWMR_2022_original.pdf": [
            "https://cpcb.nic.in/uploads/Projects/Bio-Medical-Waste/BWMR_2022.pdf",
            "https://greentribunal.gov.in/sites/default/files/news_updates/BATTERY%20WASTE%20MANAGEMENT%20RULES%2C%202022%20%5BMOEFCC%5D.pdf",
            "https://moef.gov.in/uploads/2022/08/Battery-Waste-Management-Rules-2022.pdf",
        ],
        "BWMR_2024_amendment.pdf": [
            "https://moef.gov.in/uploads/2024/06/267884.pdf",
            "https://cpcb.nic.in/displaypdf.php?id=Qmlvcm1lZGljYWwvQldNUl8yMDI0X0FtZW5kbWVudC5wZGY=",
        ],
        "BWMR_2025_amendment.pdf": [
            "https://moef.gov.in/uploads/2025/02/BWMR-Amendment-Feb-2025.pdf",
        ],
    }
    headers = {"Accept": "application/pdf,*/*"}
    for fname, urls in candidates.items():
        target = out_dir / fname
        if target.exists() and target.stat().st_size > 1024:
            print(f"[bwmr] skip {fname} (exists)")
            continue
        for url in urls:
            print(f"[bwmr] {fname} <- {url}")
            try:
                r = http_get(url, headers=headers, stream=True, timeout=90)
                if r.status_code == 200 and len(r.content if not r.raw else b"x") >= 0:
                    ctype = r.headers.get("content-type", "").lower()
                    if "pdf" in ctype or url.endswith(".pdf"):
                        size = save_stream(r, target)
                        if size > 1024:
                            print(f"[bwmr]   saved {size/1024:.1f} KB")
                            break
                        else:
                            target.unlink(missing_ok=True)
                            print(f"[bwmr]   too small ({size}B), trying next")
                            continue
                print(f"[bwmr]   status={r.status_code} ctype={r.headers.get('content-type')}")
            except Exception as exc:
                print(f"[bwmr]   error: {exc}")
        else:
            print(f"[bwmr] FAILED all sources for {fname}")


# --------------------------------------------------------------------------
# NASA Randomized & Recommissioned
# --------------------------------------------------------------------------

def download_nasa() -> None:
    """data.nasa.gov serves a presigned S3 URL (1-hr expiry). Resume + retry handles flakiness."""
    out_dir = RAW_DIR / "nasa"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "battery_alt_dataset.zip"
    redirector = "https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip"

    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        existing = target.stat().st_size if target.exists() else 0
        headers = {"User-Agent": BROWSER_UA}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            print(f"[nasa] resuming from byte {existing} (attempt {attempt}/{max_attempts})")
        else:
            print(f"[nasa] fresh download (attempt {attempt}/{max_attempts})")
        try:
            r = requests.get(redirector, headers=headers, stream=True, timeout=300, allow_redirects=True)
            if r.status_code in (200, 206):
                expected = int(r.headers.get("Content-Length", 0)) + existing
                mode = "ab" if existing and r.status_code == 206 else "wb"
                if mode == "wb":
                    existing = 0
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                size = target.stat().st_size
                print(f"[nasa]   bytes on disk: {size/(1024*1024):.1f} MB (expected ~{expected/(1024*1024):.1f} MB)")
                # Validate ZIP
                try:
                    import zipfile
                    with zipfile.ZipFile(target) as z:
                        n = len(z.namelist())
                    print(f"[nasa] ZIP OK ({n} entries) -> {target}")
                    return
                except zipfile.BadZipFile:
                    print(f"[nasa]   ZIP truncated; will retry/resume")
            else:
                print(f"[nasa]   status={r.status_code}")
        except Exception as exc:
            print(f"[nasa]   error: {exc}")
        time.sleep(3)
    print(f"[nasa] FAILED after {max_attempts} attempts; manual download required")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

TARGETS = {
    "eu": download_eu_reg,
    "wri": download_wri,
    "osf": download_stanford,
    "bwmr": download_bwmr,
    "nasa": download_nasa,
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
            print(f"[{t}] FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
