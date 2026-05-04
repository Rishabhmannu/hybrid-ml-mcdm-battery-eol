"""Build battery-level train/val/test splits with anti-leakage guarantee.

The split is **at the battery level**, never the cycle level — a battery is
either entirely in train, val, or test, never mixed. Within each split,
stratification preserves the joint distribution of:
  - source (e.g. BL_MATR vs BL_HUST)
  - chemistry (NMC, LFP, ...)
  - second_life flag

Anti-leakage check: train ∩ val == ∅, train ∩ test == ∅, val ∩ test == ∅
on battery_id, asserted before persistence.

Output: data/processed/cycling/splits.json with structure
  {"seed": 42, "ratios": [0.7,0.15,0.15],
   "stats": {...}, "train": [...], "val": [...], "test": [...]}

Usage:
    conda activate Eco-Research
    python src/data/splits.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

UNIFIED_PARQUET = PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
OUT_JSON = PROJECT_ROOT / "data" / "processed" / "cycling" / "splits.json"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum stratum size below which we drop into a fallback bucket
MIN_STRATUM = 6


def stratify_key(row) -> str:
    sl = "SL" if row.second_life else "NL"
    return f"{row.source}|{row.chemistry}|{sl}"


def split_one_stratum(battery_ids: list[str], rng) -> tuple[list[str], list[str], list[str]]:
    """Split a single stratum into train/val/test."""
    n = len(battery_ids)
    if n == 0:
        return [], [], []
    bids = list(battery_ids)
    rng.shuffle(bids)

    if n == 1:
        return bids, [], []
    if n == 2:
        return [bids[0]], [], [bids[1]]
    if n < MIN_STRATUM:
        # Tiny stratum: spread evenly using interleave to keep at least one in test
        n_test = max(1, int(round(n * TEST_RATIO)))
        n_val = max(1, int(round(n * VAL_RATIO))) if n >= 4 else 0
        n_train = n - n_test - n_val
        return bids[:n_train], bids[n_train:n_train + n_val], bids[n_train + n_val:]

    n_test = max(1, int(round(n * TEST_RATIO)))
    n_val = max(1, int(round(n * VAL_RATIO)))
    n_train = n - n_test - n_val
    return bids[:n_train], bids[n_train:n_train + n_val], bids[n_train + n_val:]


def main() -> int:
    if not UNIFIED_PARQUET.exists():
        print(f"FATAL: {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} not found. Run src/data/unify.py first.")
        return 1

    print(f"Loading {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET, columns=["battery_id", "source", "chemistry", "second_life"])
    bdf_all = df.drop_duplicates("battery_id").reset_index(drop=True)
    bdf = bdf_all  # default; overridden below if synthetic cells are present
    print(f"  {len(bdf_all):,} unique batteries · {bdf_all.source.nunique()} sources · {bdf_all.chemistry.nunique()} chemistries")

    # Anti-leakage rule (DATASET_IMPLEMENTATION_PLAN §5.2 #3): synthetic cells
    # only appear in train. Pull them aside before stratification so they all
    # land in the train bucket regardless of stratum.
    is_syn = bdf_all["source"].str.startswith("SYN_IN_")
    syn_train_ids = bdf_all.loc[is_syn, "battery_id"].tolist()
    real = bdf_all.loc[~is_syn].reset_index(drop=True)
    if syn_train_ids:
        print(f"  {len(syn_train_ids)} synthetic batteries pinned to TRAIN (anti-leakage)")

    real["stratum"] = real.apply(stratify_key, axis=1)
    n_strata = real.stratum.nunique()
    bdf = real  # downstream loop only iterates real cells
    print(f"  {n_strata} strata (source × chemistry × second_life)")

    import random
    rng = random.Random(SEED)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    per_stratum: list[dict] = []

    for stratum, sub in bdf.groupby("stratum", sort=True):
        bids = sub.battery_id.tolist()
        tr, va, te = split_one_stratum(bids, rng)
        train_ids.extend(tr); val_ids.extend(va); test_ids.extend(te)
        per_stratum.append({
            "stratum": stratum,
            "n": len(bids),
            "train": len(tr), "val": len(va), "test": len(te),
        })

    # Append synthetic cells to train (pinned per §5.2 anti-leakage rule)
    train_ids.extend(syn_train_ids)

    # Anti-leakage assertions
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert s_train.isdisjoint(s_val),  "LEAK: train ∩ val"
    assert s_train.isdisjoint(s_test), "LEAK: train ∩ test"
    assert s_val.isdisjoint(s_test),   "LEAK: val ∩ test"
    assert len(s_train) + len(s_val) + len(s_test) == len(bdf_all), "Battery count mismatch"
    print("  anti-leakage check: PASSED")

    # Cycle-level row counts (informational)
    full = pd.read_parquet(UNIFIED_PARQUET, columns=["battery_id"])
    counts = full.battery_id.value_counts().to_dict()
    rows_train = sum(counts.get(b, 0) for b in s_train)
    rows_val = sum(counts.get(b, 0) for b in s_val)
    rows_test = sum(counts.get(b, 0) for b in s_test)

    out = {
        "seed": SEED,
        "ratios": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO],
        "anti_leakage_check": "PASSED",
        "stats": {
            "n_batteries_total": len(bdf),
            "n_train_batteries": len(s_train),
            "n_val_batteries": len(s_val),
            "n_test_batteries": len(s_test),
            "n_train_rows": rows_train,
            "n_val_rows": rows_val,
            "n_test_rows": rows_test,
            "n_strata": n_strata,
        },
        "per_stratum_breakdown": per_stratum,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    print("\n=== SPLIT SUMMARY ===")
    print(f"  Train: {len(s_train):,} batteries  ({rows_train:,} rows)")
    print(f"  Val:   {len(s_val):,} batteries  ({rows_val:,} rows)")
    print(f"  Test:  {len(s_test):,} batteries  ({rows_test:,} rows)")
    print(f"  Total: {len(bdf_all):,} batteries ({rows_train + rows_val + rows_test:,} rows)")

    print("\n=== PER-CHEMISTRY DISTRIBUTION ===")
    chem_dist = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for _, r in bdf_all.iterrows():
        bucket = "train" if r.battery_id in s_train else "val" if r.battery_id in s_val else "test"
        chem_dist[r.chemistry][bucket] += 1
    for chem, d in sorted(chem_dist.items()):
        total = d["train"] + d["val"] + d["test"]
        print(f"  {chem:8s}  train={d['train']:>4}  val={d['val']:>3}  test={d['test']:>3}  total={total:>4}")

    print("\n=== SECOND-LIFE CELL ASSIGNMENT ===")
    for _, r in bdf_all[bdf_all.second_life].iterrows():
        bucket = "train" if r.battery_id in s_train else "val" if r.battery_id in s_val else "test"
        print(f"  {r.battery_id} ({r.source} / {r.chemistry}) -> {bucket}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
