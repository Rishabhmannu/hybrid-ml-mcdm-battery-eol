"""
Resume Stanford OSF dataset by re-fetching only the missing Cell 2p3.zip.

OSF API has been intermittently 502 — this script polls until it recovers,
then walks the Aging folder to find Cell 2p3.zip's download URL and fetches it.
Optionally grabs the sister dataset osf.io/8jnr5/ (Data in Brief 2024).

Usage:
    conda activate Eco-Research
    python scripts/refetch_stanford.py
    python scripts/refetch_stanford.py --include-sister
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "stanford" / "Aging"
SISTER_DIR = PROJECT_ROOT / "data" / "raw" / "stanford_8jnr5"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def get_with_retry(url: str, max_minutes: int = 60) -> requests.Response:
    """Poll OSF API until it returns 2xx, capped at max_minutes."""
    deadline = time.time() + max_minutes * 60
    backoff = 5
    while True:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 200:
                return r
            print(f"  [OSF] status={r.status_code}, retrying in {backoff}s")
        except Exception as exc:
            print(f"  [OSF] {exc.__class__.__name__}: {exc}, retrying in {backoff}s")
        if time.time() > deadline:
            raise RuntimeError(f"OSF API never recovered within {max_minutes} min")
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 60)


def find_aging_cell(target_name: str) -> tuple[str, int]:
    """Walk OSF fns57 → Aging → return download URL + size for the target cell."""
    print(f"[OSF] looking up {target_name}...")
    root = get_with_retry("https://api.osf.io/v2/nodes/fns57/files/osfstorage/").json()
    aging_url = None
    for x in root["data"]:
        if x["attributes"]["name"] == "Aging":
            aging_url = x["relationships"]["files"]["links"]["related"]["href"]
            break
    if not aging_url:
        raise RuntimeError("Aging folder not found in OSF fns57")
    aging = get_with_retry(aging_url).json()
    for y in aging["data"]:
        if y["attributes"]["name"] == target_name:
            dl = y["links"]["download"]
            sz = y["attributes"]["size"]
            return dl, sz
    raise RuntimeError(f"{target_name} not found in Aging folder")


def download_with_resume(url: str, out: Path, expected_size: int | None = None,
                         max_attempts: int = 30) -> None:
    """Download with chunk-level resume. Each network failure resumes from current byte offset."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        existing = out.stat().st_size if out.exists() else 0
        if expected_size and existing >= expected_size:
            print(f"  already complete ({existing/1e6:.1f} MB)")
            return

        headers = {"User-Agent": UA}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            print(f"  attempt {attempt}: resuming from {existing/1e6:.1f} MB")
        else:
            print(f"  attempt {attempt}: fresh download")

        try:
            r = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
            if r.status_code not in (200, 206):
                print(f"    status={r.status_code}, retrying")
                time.sleep(min(5 * attempt, 60))
                continue
            mode = "ab" if (existing and r.status_code == 206) else "wb"
            if mode == "wb":
                existing = 0

            written_this_attempt = 0
            last_log_total = existing
            with open(out, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written_this_attempt += len(chunk)
                    total = existing + written_this_attempt
                    if total - last_log_total >= 50 * 1024 * 1024:
                        pct = (total / expected_size * 100) if expected_size else 0
                        print(f"    ... {total/1e6:.0f} MB ({pct:.0f}%)")
                        last_log_total = total

            final = out.stat().st_size
            if expected_size and final < expected_size:
                print(f"    short read: {final/1e6:.1f}/{expected_size/1e6:.1f} MB, will resume")
                time.sleep(min(5 * attempt, 60))
                continue
            print(f"  done: {final/1e6:.1f} MB -> {out}")
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            written_this_attempt = (out.stat().st_size if out.exists() else 0) - existing
            print(f"    {type(exc).__name__} after {written_this_attempt/1e6:.1f} MB this attempt; will resume")
            time.sleep(min(5 * attempt, 60))
            continue

    raise RuntimeError(f"download failed after {max_attempts} attempts")


def fetch_cell_2p3() -> None:
    target = TARGET_DIR / "Cell 2p3.zip"
    dl_url, expected = find_aging_cell("Cell 2p3.zip")
    print(f"  download URL: {dl_url}")
    print(f"  expected size: {expected/1e6:.1f} MB")
    download_with_resume(dl_url, target, expected)
    # Validate ZIP
    import zipfile
    try:
        with zipfile.ZipFile(target) as z:
            n = len(z.namelist())
            err = z.testzip()
        if err is None:
            print(f"  ZIP OK: {n} entries")
        else:
            print(f"  ZIP CORRUPT: testzip returned {err}")
    except zipfile.BadZipFile as e:
        print(f"  ZIP INVALID: {e}")


def fetch_sister_dataset() -> None:
    """Walk osf.io/8jnr5/ — Data in Brief 2024 paper, 6 cells."""
    print("\n[sister] walking osf.io/8jnr5/")
    SISTER_DIR.mkdir(parents=True, exist_ok=True)

    def walk(folder_url: str, local: Path, indent: int = 0):
        next_url = folder_url
        while next_url:
            r = get_with_retry(next_url).json()
            for item in r.get("data", []):
                nm = item["attributes"]["name"]
                kind = item["attributes"]["kind"]
                if kind == "folder":
                    sub = item["relationships"]["files"]["links"]["related"]["href"]
                    sub_local = local / nm
                    sub_local.mkdir(parents=True, exist_ok=True)
                    print("  " * indent + f"folder {nm}/")
                    walk(sub, sub_local, indent + 1)
                else:
                    dl = item["links"].get("download")
                    if not dl:
                        continue
                    out = local / nm
                    sz = item["attributes"].get("size") or 0
                    if out.exists() and out.stat().st_size >= sz:
                        print("  " * indent + f"skip {nm}")
                        continue
                    print("  " * indent + f"file {nm} ({sz/1e6:.1f} MB)")
                    download_with_resume(dl, out, sz)
                    time.sleep(0.5)
            next_url = r.get("links", {}).get("next")

    walk("https://api.osf.io/v2/nodes/8jnr5/files/osfstorage/", SISTER_DIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-sister", action="store_true", help="Also fetch osf.io/8jnr5/ (Data in Brief 2024 dataset)")
    args = ap.parse_args()

    fetch_cell_2p3()
    if args.include_sister:
        fetch_sister_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
