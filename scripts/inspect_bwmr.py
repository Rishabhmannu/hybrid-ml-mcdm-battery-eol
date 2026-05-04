"""
Inspect Hindi/English page distribution in BWMR PDFs and extract the English regulatory text.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def script_breakdown(text: str) -> tuple[int, int]:
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return devanagari, latin


def classify_page(devanagari: int, latin: int) -> str:
    total = devanagari + latin
    if total < 50:
        return "blank/image"
    hindi_ratio = devanagari / total
    if hindi_ratio > 0.6:
        return "hindi"
    if hindi_ratio < 0.2:
        return "english"
    return "mixed"


def inspect(pdf_path: Path) -> dict:
    reader = PdfReader(str(pdf_path))
    rows = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
        d, l = script_breakdown(text)
        rows.append({
            "page": i,
            "devanagari": d,
            "latin": l,
            "kind": classify_page(d, l),
            "text_preview": text[:120].replace("\n", " ").strip(),
        })
    return {"path": str(pdf_path), "n_pages": len(rows), "pages": rows}


def extract_english(pdf_path: Path, out_path: Path) -> tuple[int, int]:
    """Extract pages classified as english/mixed into a single .txt for downstream parsing."""
    reader = PdfReader(str(pdf_path))
    chunks = []
    n_kept = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        d, l = script_breakdown(text)
        if classify_page(d, l) in ("english", "mixed") and (d + l) > 50:
            chunks.append(f"\n\n=== PAGE {i} ===\n{text.strip()}")
            n_kept += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    return n_kept, len(reader.pages)


def main() -> int:
    bwmr_dir = Path("data/regulatory/bwmr")
    out_dir = bwmr_dir / "extracted_english"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p.name for p in bwmr_dir.glob("*.pdf"))

    for fname in pdfs:
        pdf = bwmr_dir / fname
        if not pdf.exists():
            print(f"SKIP missing {pdf}")
            continue
        info = inspect(pdf)
        kinds = [p["kind"] for p in info["pages"]]
        from collections import Counter
        c = Counter(kinds)
        print(f"\n=== {fname} ({info['n_pages']} pp) ===")
        for kind in ("english", "mixed", "hindi", "blank/image"):
            pages = [p["page"] for p in info["pages"] if p["kind"] == kind]
            print(f"  {kind:13s} {c[kind]}/{info['n_pages']}  pages={pages}")
        # Extract English+mixed pages into a clean .txt
        out_txt = out_dir / (fname.rsplit(".", 1)[0] + ".en.txt")
        kept, total = extract_english(pdf, out_txt)
        print(f"  -> wrote {kept}/{total} pages to {out_txt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
