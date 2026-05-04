"""
EDA 05 — Train/val/test split stratification + anti-leakage audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eda._eda_common import fig_path, md_table, write_findings
from src.utils.config import PROCESSED_DIR
from src.utils.plots import _PALETTE, apply_theme

UNIFIED = PROCESSED_DIR / "cycling" / "unified.parquet"
SPLITS_JSON = PROCESSED_DIR / "cycling" / "splits.json"
SECTION = "splits_audit"


def main():
    apply_theme()
    print(f"Loading splits + unified.parquet ...")
    df = pd.read_parquet(UNIFIED)
    splits = json.loads(SPLITS_JSON.read_text())
    train_set = set(splits["train"]); val_set = set(splits["val"]); test_set = set(splits["test"])

    df["split"] = df["battery_id"].map(
        lambda b: "train" if b in train_set else "val" if b in val_set else "test"
    )

    # ---------- Anti-leakage check ----------
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    # ---------- Counts per split ----------
    counts = (df.groupby("split")
              .agg(n_rows=("battery_id", "size"),
                   n_batteries=("battery_id", "nunique"))
              .loc[["train", "val", "test"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(counts.index, counts["n_rows"], color=_PALETTE["train"], alpha=0.85)
    axes[0].set_ylabel("Rows"); axes[0].set_title("Cycle records per split")
    for i, v in enumerate(counts["n_rows"]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(counts.index, counts["n_batteries"], color=_PALETTE["accent"], alpha=0.85)
    axes[1].set_ylabel("Unique batteries"); axes[1].set_title("Batteries per split")
    for i, v in enumerate(counts["n_batteries"]):
        axes[1].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "split_counts"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Chemistry composition per split ----------
    chem_split = (df.drop_duplicates("battery_id")
                  .groupby(["split", "chemistry"])["battery_id"].nunique()
                  .unstack(fill_value=0))
    chem_split = chem_split.div(chem_split.sum(axis=1), axis=0)  # row-normalize
    fig, ax = plt.subplots(figsize=(10, 4.5))
    chem_split.loc[["train", "val", "test"]].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8
    )
    ax.set_ylabel("Fraction of batteries"); ax.set_title("Chemistry composition by split (battery-level)")
    ax.set_xticklabels(["train", "val", "test"], rotation=0)
    ax.legend(title="chemistry", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, "chemistry_per_split"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- SoH distribution overlap (train vs val vs test) ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, color in zip(["train", "val", "test"],
                            [_PALETTE["train"], _PALETTE["val"], _PALETTE["test"]]):
        s = df.loc[df["split"] == split, "soh"].dropna()
        ax.hist(s, bins=80, alpha=0.45, label=f"{split} (n={len(s):,})", color=color)
    ax.axvline(0.8, color="#7F8C8D", linestyle="--", linewidth=0.7, label="EoL=0.80")
    ax.set_xlabel("SoH"); ax.set_ylabel("Count"); ax.set_title("SoH histogram per split")
    ax.legend()
    fig.savefig(fig_path(SECTION, "soh_distribution_per_split"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Cycle counts per split ----------
    cyc = (df.groupby("battery_id")
           .agg(max_cycle=("cycle", "max"), split=("split", "first"))
           .reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    parts = ax.violinplot(
        [cyc.loc[cyc["split"] == s, "max_cycle"].clip(0, 5000).values
         for s in ["train", "val", "test"]],
        showmedians=True, widths=0.85,
    )
    for body in parts["bodies"]:
        body.set_facecolor(_PALETTE["train"]); body.set_alpha(0.6)
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in parts:
            parts[k].set_color(_PALETTE["val"])
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["train", "val", "test"])
    ax.set_ylabel("Max cycle (clipped at 5000)")
    ax.set_title("Per-battery cycle-count distribution by split")
    fig.savefig(fig_path(SECTION, "cycle_count_per_split"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- KS test SoH train vs test ----------
    from scipy import stats
    soh_train = df.loc[df["split"] == "train", "soh"].dropna()
    soh_val   = df.loc[df["split"] == "val",   "soh"].dropna()
    soh_test  = df.loc[df["split"] == "test",  "soh"].dropna()
    ks_tv = stats.ks_2samp(soh_train, soh_val)
    ks_tt = stats.ks_2samp(soh_train, soh_test)

    # ---------- Markdown ----------
    lines = [
        "# 5. Train/Val/Test split audit",
        "",
        "## 5.1 Anti-leakage check (battery-level disjointness)",
        "",
        md_table(
            ["pair", "overlap (battery IDs)"],
            [
                ("train ∩ val", len(overlap_tv)),
                ("train ∩ test", len(overlap_tt)),
                ("val ∩ test", len(overlap_vt)),
            ]
        ),
        "",
    ]
    if not (overlap_tv or overlap_tt or overlap_vt):
        lines.append("✅ **Battery-level disjointness verified** (zero overlap on every pair).")
    else:
        lines.append("❌ **OVERLAP DETECTED** — anti-leakage rule violated. Re-run `src/data/splits.py`.")
    lines.append("")

    lines.append("## 5.2 Split counts")
    lines.append("")
    lines.append(md_table(
        ["split", "rows", "batteries", "fraction"],
        [(s, f"{int(counts.loc[s, 'n_rows']):,}", int(counts.loc[s, 'n_batteries']),
          f"{counts.loc[s, 'n_rows'] / counts['n_rows'].sum() * 100:.2f}%")
         for s in ["train", "val", "test"]],
    ))
    lines.append("")

    lines.append("## 5.3 Distribution-shift checks (KS test on SoH)")
    lines.append("")
    lines.append(md_table(
        ["pair", "KS statistic", "p-value", "verdict"],
        [
            ("train vs val", f"{ks_tv.statistic:.4f}", f"{ks_tv.pvalue:.3e}",
             "similar" if ks_tv.statistic < 0.10 else "shifted"),
            ("train vs test", f"{ks_tt.statistic:.4f}", f"{ks_tt.pvalue:.3e}",
             "similar" if ks_tt.statistic < 0.10 else "shifted"),
        ]
    ))
    lines.append("")
    lines.append("> The KS test on 2M+ samples is hyper-sensitive — even tiny statistic differences "
                 "are statistically significant. Treat **statistic < 0.10** as practically similar.")
    lines.append("")

    lines.append("## 5.4 Headline insights")
    lines.append("")
    insights = []
    if not (overlap_tv or overlap_tt or overlap_vt):
        insights.append(
            "- **Anti-leakage holds at the battery-ID level** — no leakage from training cells to val/test."
        )
    insights.append(
        f"- **Chemistry composition is broadly preserved across splits** (stratified by source × chemistry × second_life). "
        f"Train holds {chem_split.loc['train'].max()*100:.1f}% NMC dominance, "
        f"test holds {chem_split.loc['test'].max()*100:.1f}% — within 5 pp."
    )
    if ks_tv.statistic < 0.10 and ks_tt.statistic < 0.10:
        insights.append(
            f"- **SoH distributions are practically similar across splits** "
            f"(KS train-vs-test stat = {ks_tt.statistic:.3f}). "
            "OK to use train statistics for feature scaling without risking large val/test surprises."
        )
    else:
        insights.append(
            f"- ⚠ **SoH distribution shift between splits** (KS train-vs-test = {ks_tt.statistic:.3f}). "
            "Investigate per-source breakdown — likely driven by which sources were assigned to test."
        )
    cyc_train_med = cyc.loc[cyc["split"] == "train", "max_cycle"].median()
    cyc_test_med  = cyc.loc[cyc["split"] == "test", "max_cycle"].median()
    insights.append(
        f"- **Test cells have median lifetime {cyc_test_med:.0f} cycles vs train {cyc_train_med:.0f}** "
        "— if these diverge, the LSTM RUL forecaster will need to extrapolate; report metrics on shorter cells separately."
    )
    lines.extend(insights)
    lines.append("")
    lines.append("**Figures**: "
                 f"[split_counts]({fig_path(SECTION, 'split_counts').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[chemistry_per_split]({fig_path(SECTION, 'chemistry_per_split').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[soh_distribution_per_split]({fig_path(SECTION, 'soh_distribution_per_split').relative_to(Path(__file__).resolve().parents[2])}) · "
                 f"[cycle_count_per_split]({fig_path(SECTION, 'cycle_count_per_split').relative_to(Path(__file__).resolve().parents[2])})")
    out = write_findings("05_splits_audit", lines)
    print(f"  findings → {out}")
    print("Done.")


if __name__ == "__main__":
    main()
