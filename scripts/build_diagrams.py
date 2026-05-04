"""Generate Excalidraw diagram files for the EV Battery EoL project.

Outputs to ``results/figures/diagrams/D-XX_<slug>.excalidraw``.
Open files at https://excalidraw.com via File → Open.

Run: ``python scripts/build_diagrams.py [d01|all]``
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "diagrams"

PALETTE = {
    "data":     "#1F4E79",  # deep blue
    "data_bg":  "#E8EEF5",
    "model":    "#1F4E79",
    "model_bg": "#DCE6F2",
    "mcdm":     "#F39C12",  # amber
    "mcdm_bg":  "#FDF1DC",
    "dpp":      "#27AE60",  # green
    "dpp_bg":   "#E8F5E9",
    "neutral":  "#7F8C8D",  # gray
    "red":      "#C0392B",
    "text":     "#1A1A1A",
}


# ---------- low-level helpers ---------------------------------------------

def _rid() -> str:
    return "".join(random.choices(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=22))


def _seed() -> int:
    return random.randint(1, 2**31 - 1)


def _now() -> int:
    return int(time.time() * 1000)


def _base(**kw):
    el = {
        "id": _rid(),
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": _seed(),
        "version": 1,
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": _now(),
        "link": None,
        "locked": False,
    }
    el.update(kw)
    return el


def lane(x, y, w, h, label, bg, stroke):
    """Swimlane: tinted background rectangle + corner title."""
    rect = _base(
        type="rectangle", x=x, y=y, width=w, height=h,
        backgroundColor=bg, strokeColor=stroke, strokeWidth=1,
        roundness={"type": 3}, fillStyle="solid", roughness=0,
        opacity=60,
    )
    title = _base(
        type="text", x=x + 16, y=y + 10, width=400, height=28,
        text=label, fontSize=20, fontFamily=1,
        textAlign="left", verticalAlign="top",
        baseline=18, lineHeight=1.25,
        originalText=label, containerId=None,
        strokeColor=stroke,
    )
    return [rect, title]


def node(x, y, w, h, lines, stroke=None, bg="#ffffff", dashed=False, font=15):
    """Rounded rectangle with bound centered text. Returns (rect, text)."""
    if stroke is None:
        stroke = PALETTE["text"]
    rect = _base(
        type="rectangle", x=x, y=y, width=w, height=h,
        backgroundColor=bg, strokeColor=stroke, strokeWidth=2,
        strokeStyle="dashed" if dashed else "solid",
        roundness={"type": 3}, fillStyle="solid", roughness=1,
    )
    text = "\n".join(lines)
    nlines = max(1, len(lines))
    line_h = font * 1.25
    text_h = nlines * line_h
    txt = _base(
        type="text", x=x, y=y + (h - text_h) / 2, width=w, height=text_h,
        text=text, fontSize=font, fontFamily=1,
        textAlign="center", verticalAlign="middle",
        baseline=int(font * 0.85), lineHeight=1.25,
        originalText=text, containerId=rect["id"],
        strokeColor=stroke,
    )
    rect["boundElements"] = [{"id": txt["id"], "type": "text"}]
    return rect, txt


def arrow(src, dst, label=None, dashed=False, stroke=None):
    """Straight arrow from src rect to dst rect, with optional label."""
    if stroke is None:
        stroke = PALETTE["neutral"]
    sx, sy, sw, sh = src["x"], src["y"], src["width"], src["height"]
    dx, dy, dw, dh = dst["x"], dst["y"], dst["width"], dst["height"]
    s_cx, s_cy = sx + sw / 2, sy + sh / 2
    d_cx, d_cy = dx + dw / 2, dy + dh / 2
    if abs(d_cy - s_cy) >= abs(d_cx - s_cx):
        # vertical
        if d_cy > s_cy:
            x1, y1, x2, y2 = s_cx, sy + sh, d_cx, dy
        else:
            x1, y1, x2, y2 = s_cx, sy, d_cx, dy + dh
    else:
        # horizontal
        if d_cx > s_cx:
            x1, y1, x2, y2 = sx + sw, s_cy, dx, d_cy
        else:
            x1, y1, x2, y2 = sx, s_cy, dx + dw, d_cy
    arr = _base(
        type="arrow", x=x1, y=y1, width=x2 - x1, height=y2 - y1,
        points=[[0, 0], [x2 - x1, y2 - y1]],
        startBinding={"elementId": src["id"], "focus": 0, "gap": 8},
        endBinding={"elementId": dst["id"], "focus": 0, "gap": 8},
        startArrowhead=None, endArrowhead="arrow",
        strokeColor=stroke, strokeWidth=2,
        strokeStyle="dashed" if dashed else "solid",
        roughness=1, fillStyle="solid",
        lastCommittedPoint=None,
    )
    src.setdefault("boundElements", []).append({"id": arr["id"], "type": "arrow"})
    dst.setdefault("boundElements", []).append({"id": arr["id"], "type": "arrow"})
    out = [arr]
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # rough label width
        lw = max(40, len(label) * 8 + 12)
        lbl = _base(
            type="text", x=mx - lw / 2, y=my - 12, width=lw, height=22,
            text=label, fontSize=13, fontFamily=1,
            textAlign="center", verticalAlign="middle",
            baseline=11, lineHeight=1.25,
            originalText=label, containerId=None,
            strokeColor=PALETTE["text"],
            backgroundColor="#ffffff",
            fillStyle="solid",
        )
        out.append(lbl)
    return out


def text(x, y, content, size=14, color=None, w=900):
    if color is None:
        color = PALETTE["text"]
    return _base(
        type="text", x=x, y=y, width=w, height=int(size * 1.4),
        text=content, fontSize=size, fontFamily=1,
        textAlign="left", verticalAlign="top",
        baseline=int(size * 0.85), lineHeight=1.25,
        originalText=content, containerId=None,
        strokeColor=color,
    )


def write(filename: str, elements: list, source: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": source,
        "elements": elements,
        "appState": {
            "gridSize": 20,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }
    out = OUT_DIR / filename
    out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {out.relative_to(ROOT)}  ({len(elements)} elements)")


# ---------- D-01: System Architecture -------------------------------------

def build_d01():
    random.seed(101)
    el = []

    el.append(text(40, 20,
        "D-01 · System Architecture — Hybrid ML+MCDM EV Battery EoL Routing",
        size=24))
    el.append(text(40, 55,
        "Group 5 · Sem-8 Digital Economics · 2026-04-28",
        size=13, color=PALETTE["neutral"]))

    # ---- DATA LANE ----
    el += lane(40, 100, 1500, 320, "Data Layer",
               PALETTE["data_bg"], PALETTE["data"])

    sources = [
        ["BatteryLife", "86 GB · 18 sub-corpora"],
        ["NASA PCoE + Random", "586 MB + 3.2 GB"],
        ["CALCE CS2 / CX2", "1.05 GB"],
        ["Stanford OSF", "17 GB + 8jnr5 (2.15 GB)"],
        ["PyBaMM Synthetic", "Indian cells: 70 (50 NMC + 20 LFP)"],
    ]
    src_boxes = []
    for i, lines in enumerate(sources):
        x = 70 + i * 290
        r, t = node(x, 150, 270, 70, lines, stroke=PALETTE["data"])
        el += [r, t]
        src_boxes.append(r)

    unified, ut = node(580, 310, 420, 80, [
        "unified.parquet",
        "2.20 M rows · 1,521 cells · 32 cols",
    ], stroke=PALETTE["data"], bg="#FFFFFF")
    el += [unified, ut]
    for s in src_boxes:
        el += arrow(s, unified)

    # ---- MODEL LANE ----
    el += lane(40, 460, 1500, 320,
               "ML Layer  ·  Iteration 1 (dashed = Iter-2 redesign planned)",
               PALETTE["model_bg"], PALETTE["model"])

    xgb,    xgbt   = node(110,  540, 280, 90,
        ["XGBoost SoH", "depth 9 · 996 trees", "R² > 0.95 · gates ✅"],
        stroke=PALETTE["data"])
    lstm,   lstmt  = node(430,  540, 280, 90,
        ["LSTM RUL", "h=128 · 2 layers · drop 0.2",
         "Iter-2: Huber + AdamW + CNN-LSTM"],
        stroke=PALETTE["mcdm"], dashed=True)
    iso,    isot   = node(750,  540, 280, 90,
        ["IsolationForest", "200 trees", "top 5% flagged"],
        stroke=PALETTE["data"])
    grade,  gradet = node(1070, 540, 280, 90,
        ["Grade Classifier", "A / B / C / D thresholds",
         "from SoH%"],
        stroke=PALETTE["data"])
    el += [xgb, xgbt, lstm, lstmt, iso, isot, grade, gradet]

    for m in (xgb, lstm, iso):
        el += arrow(unified, m)
    el += arrow(xgb, grade, label="SoH%")

    # ---- MCDM LANE ----
    el += lane(40, 820, 1500, 320, "MCDM Layer",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])

    lit,    litt    = node(80,   900, 260, 90,
        ["Literature", "8 of 11 T2 papers", "weights extracted"],
        stroke=PALETTE["mcdm"])
    bwm,    bwmt    = node(390,  900, 280, 90,
        ["Fuzzy BWM", "6 criteria → TFNs",
         "(SoH · Value · Carbon · Compliance · Safety · EPR)"],
        stroke=PALETTE["mcdm"], font=13)
    topsis, topsist = node(720,  900, 280, 90,
        ["TOPSIS", "4 alternatives · ranks",
         "Grid · Home · Reuse · Recycle"],
        stroke=PALETTE["mcdm"], font=13)
    sens,   senst   = node(1050, 900, 280, 90,
        ["Sensitivity", "5 scenarios",
         "Equal · Lit-Mean · Tech · BWMR-Heavy · EU-Heavy"],
        stroke=PALETTE["mcdm"], font=12)
    el += [lit, litt, bwm, bwmt, topsis, topsist, sens, senst]

    el += arrow(lit, bwm)
    el += arrow(bwm, topsis, label="weights")
    el += arrow(topsis, sens)
    el += arrow(grade, topsis, label="grade")

    # ---- DPP LANE ----
    el += lane(40, 1180, 1500, 360, "DPP / Output Layer",
               PALETTE["dpp_bg"], PALETTE["dpp"])

    eu,    eut    = node(80,  1240, 260, 80,
        ["EU 2023/1542", "Annex XIII"], stroke=PALETTE["dpp"])
    gba,   gbat   = node(380, 1240, 260, 80,
        ["GBA Battery Pass", "v1.2.0"], stroke=PALETTE["dpp"])
    bwmr,  bwmrt  = node(680, 1240, 260, 80,
        ["BWMR 2022", "+ amendments"], stroke=PALETTE["dpp"])

    schema, schemat = node(380, 1380, 320, 80,
        ["unified_dpp_schema.json", "9 categories"],
        stroke=PALETTE["dpp"])
    mapper, mappert = node(740, 1380, 280, 80,
        ["schema_mapper.py"],
        stroke=PALETTE["dpp"])
    dppjson, dppjsont = node(1060, 1380, 420, 80,
        ["DPP JSON per battery",
         "schema-validated · coverage 48.7%"],
        stroke=PALETTE["dpp"])
    el += [eu, eut, gba, gbat, bwmr, bwmrt,
           schema, schemat, mapper, mappert, dppjson, dppjsont]

    for r in (eu, gba, bwmr):
        el += arrow(r, schema)
    el += arrow(schema, mapper)
    el += arrow(mapper, dppjson)

    # ---- cross-lane edges ----
    el += arrow(lstm, mapper, label="RUL")
    el += arrow(topsis, mapper, label="route")

    # ---- footer ----
    el.append(text(40, 1560,
        "Sources: DATASET_IMPLEMENTATION_PLAN.md §3 · VARIABLES_DOCUMENT.md §1–8 · "
        "data/processed/cycling/splits.json · "
        "data/processed/mcdm_weights/AGGREGATION_REPORT.md",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-01_system_architecture.excalidraw", el, "D-01 generator")


# ---------- D-04: Data Processing Pipeline --------------------------------

def build_d04():
    random.seed(104)
    el = []
    el.append(text(40, 20, "D-04 · Data Processing Pipeline", size=24))
    el.append(text(40, 55,
        "raw → unify → split → feature bundle  (anti-leakage check ✅)",
        size=13, color=PALETTE["neutral"]))

    # Loaders row
    el += lane(40, 100, 1500, 220, "Per-source Loaders",
               PALETTE["data_bg"], PALETTE["data"])
    loaders = [
        ["batterylife.py", "18 sub-corpora"],
        ["nasa_pcoe.py", "+ random.py"],
        ["calce.py", "CS2 / CX2"],
        ["stanford_osf.py", "OSF + 8jnr5"],
        ["synthetic_indian.py", "70 PyBaMM cells"],
    ]
    loader_boxes = []
    for i, lines in enumerate(loaders):
        x = 70 + i * 290
        r, t = node(x, 160, 270, 90, lines, stroke=PALETTE["data"])
        el += [r, t]
        loader_boxes.append(r)

    # unify
    unify, ut = node(540, 360, 500, 100, [
        "unify.py",
        "schema reconciliation · chemistry-aware V filter",
        "z-outlier flags",
    ], stroke=PALETTE["data"], font=14)
    el += [unify, ut]
    for lb in loader_boxes:
        el += arrow(lb, unify)

    # unified parquet
    unified, ut2 = node(540, 510, 500, 90, [
        "unified.parquet",
        "2.20 M rows · 1,521 cells · 32 columns",
    ], stroke=PALETTE["data"], bg="#E8EEF5")
    el += [unified, ut2]
    el += arrow(unify, unified)

    # splits
    splits, st = node(540, 650, 500, 110, [
        "splits.py",
        "battery-level · stratified by source × chemistry × second_life",
        "seed = 42",
    ], stroke=PALETTE["data"], font=14)
    el += [splits, st]
    el += arrow(unified, splits)

    splits_json, sjt = node(540, 800, 500, 100, [
        "splits.json",
        "1,084 train / 217 val / 220 test",
        "anti-leakage check ✅  ·  29 strata",
    ], stroke=PALETTE["data"], bg="#E8EEF5")
    el += [splits_json, sjt]
    el += arrow(splits, splits_json)

    # features
    feat, ft = node(540, 950, 500, 110, [
        "training_data.py",
        "feature engineering",
        "ir_ohm + 4 temp columns excluded",
    ], stroke=PALETTE["data"], font=14)
    el += [feat, ft]
    el += arrow(splits_json, feat)

    # outputs
    f1, f1t = node(220, 1110, 360, 90, [
        "Numeric + One-hot",
        "17 numeric + 16 one-hot = 33",
    ], stroke=PALETTE["model"])
    f2, f2t = node(620, 1110, 360, 90, [
        "Targets",
        "soh_pct  ·  rul_cycles",
    ], stroke=PALETTE["model"])
    f3, f3t = node(1020, 1110, 360, 90, [
        "Synthetic cells",
        "pinned to TRAIN split",
    ], stroke=PALETTE["model"])
    el += [f1, f1t, f2, f2t, f3, f3t]
    el += arrow(feat, f1)
    el += arrow(feat, f2)
    el += arrow(feat, f3)

    el.append(text(40, 1240,
        "Sources: src/data/loaders/ · src/data/unify.py · "
        "src/data/splits.py · src/data/training_data.py · "
        "data/processed/cycling/splits.json",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-04_data_processing_pipeline.excalidraw", el, "D-04 generator")


# ---------- D-08: LSTM RUL Architecture (Iter-1 + Iter-2 overlay) ---------

def build_d08():
    random.seed(108)
    el = []
    el.append(text(40, 20, "D-08 · LSTM RUL Architecture", size=24))
    el.append(text(40, 55,
        "Iter-1 baseline (solid) with Iter-2 redesign overlay (dashed amber)",
        size=13, color=PALETTE["neutral"]))

    # Iter-1 lane
    el += lane(40, 100, 1500, 280,
               "Iteration 1 — gate FAILED: test RMSE 4.54% vs <2% target",
               PALETTE["model_bg"], PALETTE["model"])

    inp,    inpt    = node(80,  170, 220, 100,
        ["Input sequence", "30 cycles", "× 33 features"],
        stroke=PALETTE["data"])
    l1,     l1t     = node(330, 170, 220, 100,
        ["LSTM Layer 1", "hidden = 128", "dropout 0.2"],
        stroke=PALETTE["data"])
    l2,     l2t     = node(580, 170, 220, 100,
        ["LSTM Layer 2", "hidden = 128", "dropout 0.2"],
        stroke=PALETTE["data"])
    h,      ht      = node(830, 170, 200, 100,
        ["Last hidden", "state h_T"],
        stroke=PALETTE["data"])
    fc1,    fc1t    = node(1060, 170, 200, 100,
        ["FC 128 → 64", "ReLU + Drop 0.1"],
        stroke=PALETTE["data"])
    fc2,    fc2t    = node(1290, 170, 220, 100,
        ["FC 64 → 32 → 1", "RUL cycles"],
        stroke=PALETTE["data"])
    el += [inp, inpt, l1, l1t, l2, l2t, h, ht, fc1, fc1t, fc2, fc2t]
    for a, b in [(inp, l1), (l1, l2), (l2, h), (h, fc1), (fc1, fc2)]:
        el += arrow(a, b)

    # metrics callout
    el.append(text(80, 305,
        "Train R² = 0.971  ·  Val R² = 0.936  ·  Test R² = 0.891  "
        "·  Test RMSE 168.9 cycles (4.54% of 3,723-cycle range)",
        size=12, color=PALETTE["red"], w=1400))

    # Iter-2 lane
    el += lane(40, 420, 1500, 320,
               "Iteration 2 (planned) — addresses LOSO drop & overfit",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])

    redesign = [
        (80,  490, ["Longer window", "seq_len = 60", "(was 30)"]),
        (330, 490, ["Smaller LSTM", "hidden = 64", "(was 128)"]),
        (580, 490, ["Recurrent dropout", "0.35", "(was 0.2)"]),
        (830, 490, ["AdamW", "weight_decay 1e-4", "Huber loss"]),
        (1060, 490, ["Patience 8", "(was 15)", "early stop"]),
        (1290, 490, ["+ CNN-LSTM", "front-end", "(optional)"]),
    ]
    for x, y, lines in redesign:
        r, t = node(x, y, 220, 110, lines,
                    stroke=PALETTE["mcdm"], dashed=True, font=13)
        el += [r, t]

    el.append(text(80, 620,
        "References: Hong 2024 · Tang 2023 (regularized LSTM RUL)",
        size=12, color=PALETTE["neutral"], w=1400))
    el.append(text(80, 645,
        "Iter-2 target: LOSO median R² 0.70–0.80 (honest), 0.85 stretch",
        size=12, color=PALETTE["text"], w=1400))

    el.append(text(40, 770,
        "Sources: src/models/lstm_rul.py · "
        "results/tables/lstm_rul/metrics.json · ML_ITERATION_2_DESIGN.md",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-08_lstm_rul_architecture.excalidraw", el, "D-08 generator")


# ---------- D-10: Cross-Source LOSO Methodology + Results -----------------

def build_d10():
    random.seed(110)
    el = []
    el.append(text(40, 20, "D-10 · Cross-Source LOSO Methodology + Results", size=24))
    el.append(text(40, 55,
        "Leave-One-Source-Out evaluation across 10 sources · median R² = 0.348",
        size=13, color=PALETTE["neutral"]))

    # methodology lane
    el += lane(40, 100, 1500, 240, "Methodology",
               PALETTE["data_bg"], PALETTE["data"])
    pool, pt = node(80, 170, 280, 100,
        ["Pool of 10 sources", "BatteryLife sub-corpora"],
        stroke=PALETTE["data"])
    loop, lt = node(420, 170, 360, 100,
        ["For each source S:",
         "train on (others), test on S",
         "→ per-source R²"],
        stroke=PALETTE["data"], font=13)
    res, rt = node(840, 170, 280, 100,
        ["Per-source R²", "n=10 held-out runs"],
        stroke=PALETTE["data"])
    agg, agt = node(1180, 170, 320, 100,
        ["Aggregate", "median + worst-case", "→ honest robustness"],
        stroke=PALETTE["data"])
    el += [pool, pt, loop, lt, res, rt, agg, agt]
    el += arrow(pool, loop)
    el += arrow(loop, res)
    el += arrow(res, agg)

    # results lane (bar chart in rectangles)
    el += lane(40, 380, 1500, 600,
               "Iter-1 results — per-source held-out R² (sorted)",
               PALETTE["model_bg"], PALETTE["model"])

    # Sorted descending
    loso = [
        ("BL_Stanford",   0.999, "NMC"),
        ("BL_Stanford_2", 0.997, "NMC"),
        ("BL_MATR",       0.809, "LFP"),
        ("BL_Tongji",     0.692, "NCA"),
        ("BL_ISU_ILCC",   0.433, "NMC"),
        ("BL_SNL",        0.263, "LFP"),
        ("BL_SDU",        0.066, "NMC"),
        ("BL_HUST",      -1.090, "LFP"),
        ("BL_RWTH",      -2.660, "NMC"),
        ("BL_ZN-coin",  -18.679, "Zn-ion"),
    ]
    # Map R² to bar height. zero-line at y=720
    zero_y = 720
    chart_x = 80
    bar_w = 110
    gap = 30
    pos_scale = 180   # px per 1.0 R²
    neg_scale = 12    # px per 1.0 R² (clipped)
    for i, (name, r2, chem) in enumerate(loso):
        x = chart_x + i * (bar_w + gap)
        if r2 >= 0:
            h = max(8, r2 * pos_scale)
            y = zero_y - h
            color = PALETTE["dpp"]
        else:
            # clip extreme negative for visualization
            disp = max(r2, -3.0)
            h = max(8, abs(disp) * neg_scale * 5)
            h = min(h, 200)
            y = zero_y
            color = PALETTE["red"]
        bar = _base(
            type="rectangle", x=x, y=y, width=bar_w, height=h,
            backgroundColor=color, strokeColor=color, strokeWidth=1,
            roundness={"type": 3}, fillStyle="solid", roughness=0,
            opacity=80,
        )
        el.append(bar)
        # value label
        lbl_y = y - 22 if r2 >= 0 else y + h + 4
        el.append(text(x, lbl_y, f"{r2:.2f}",
                       size=12, color=PALETTE["text"], w=bar_w))
        # source name (rotated would be nice; using two lines instead)
        el.append(text(x - 4, 935,
                       name.replace("BL_", ""),
                       size=10, color=PALETTE["text"], w=bar_w + 8))
        el.append(text(x - 4, 952,
                       f"({chem})",
                       size=9, color=PALETTE["neutral"], w=bar_w + 8))

    # zero line
    zero_line = _base(
        type="line", x=70, y=zero_y, width=1460, height=0,
        points=[[0, 0], [1460, 0]],
        strokeColor=PALETTE["neutral"], strokeWidth=1,
        strokeStyle="dashed", roughness=0,
    )
    el.append(zero_line)
    el.append(text(1540, zero_y - 10, "R² = 0",
                   size=11, color=PALETTE["neutral"], w=80))

    # median callout
    el.append(text(80, 420,
        "Median R² = 0.348  ·  Worst case (Zn-ion) clipped at −3 for display "
        "(actual R² = −18.68)",
        size=13, color=PALETTE["text"], w=1400))
    el.append(text(80, 985,
        "Reference: Roman et al. 2021 (Nature Machine Intelligence) — "
        "LOSO gap range for SoH models",
        size=11, color=PALETTE["neutral"], w=1400))

    el.append(text(40, 1010,
        "Sources: scripts/cross_source_generalization.py · "
        "results/tables/cross_source/results.csv",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-10_cross_source_loso.excalidraw", el, "D-10 generator")


# ---------- D-12: TOPSIS Workflow ------------------------------------------

def build_d12():
    random.seed(112)
    el = []
    el.append(text(40, 20,
        "D-12 · TOPSIS Workflow + Routing Decision Tree", size=24))
    el.append(text(40, 55,
        "Grade → recommended EoL route across 4 alternatives × 6 criteria",
        size=13, color=PALETTE["neutral"]))

    # Main workflow lane (TD)
    el += lane(40, 100, 1100, 1240, "TOPSIS Workflow",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])

    g,    gt    = node(420, 160, 360, 80,
        ["Grade A / B / C / D", "from XGBoost SoH"], stroke=PALETTE["mcdm"])
    dm,   dmt   = node(380, 280, 440, 100,
        ["Decision matrix",
         "4 alternatives × 6 criteria",
         "per-grade lookup"], stroke=PALETTE["mcdm"], font=13)
    w,    wt    = node(80,  280, 240, 100,
        ["fuzzy_bwm_input.csv",
         "defuzzified weights"], stroke=PALETTE["mcdm"], font=13)
    r,    rt    = node(380, 420, 440, 90,
        ["Vector normalize",
         "r_ij = x_ij / √Σ x²"], stroke=PALETTE["mcdm"], font=14)
    v,    vt    = node(380, 550, 440, 90,
        ["Weighted matrix",
         "v_ij = w_j · r_ij"], stroke=PALETTE["mcdm"], font=14)
    ai,   ait   = node(380, 680, 440, 100,
        ["Identify A⁺ , A⁻",
         "per-criterion best / worst",
         "(benefit vs cost)"], stroke=PALETTE["mcdm"], font=13)
    d,    dt    = node(380, 820, 440, 100,
        ["Distances",
         "d⁺ = √Σ (v − A⁺)²",
         "d⁻ = √Σ (v − A⁻)²"], stroke=PALETTE["mcdm"], font=13)
    c,    ct    = node(380, 960, 440, 90,
        ["Closeness",
         "C* = d⁻ / (d⁺ + d⁻)"], stroke=PALETTE["mcdm"], font=14)
    rk,   rkt   = node(380, 1090, 440, 80,
        ["Rank: argsort_desc(C*)"], stroke=PALETTE["mcdm"])
    out,  outt  = node(380, 1210, 440, 90,
        ["Recommended Route", "(rank-1 alternative)"],
        stroke=PALETTE["dpp"], bg="#E8F5E9")
    el += [g, gt, dm, dmt, w, wt, r, rt, v, vt, ai, ait,
           d, dt, c, ct, rk, rkt, out, outt]
    el += arrow(g, dm)
    el += arrow(w, dm)
    el += arrow(dm, r)
    el += arrow(r, v)
    el += arrow(v, ai)
    el += arrow(ai, d)
    el += arrow(d, c)
    el += arrow(c, rk)
    el += arrow(rk, out)

    # Side reference panel
    el += lane(1180, 100, 380, 1240, "Reference",
               PALETTE["data_bg"], PALETTE["data"])
    el.append(text(1200, 160, "4 Alternatives", size=16, color=PALETTE["data"]))
    alts = ["1. Grid-scale ESS", "2. Home / Distributed ESS",
            "3. Component Reuse", "4. Direct Recycling"]
    for i, a in enumerate(alts):
        el.append(text(1200, 195 + i * 26, a, size=13, w=350))

    el.append(text(1200, 320, "6 Criteria (canonical)",
                   size=16, color=PALETTE["data"]))
    crit = ["• SoH (technical)",
            "• Value (economic)",
            "• Carbon (environmental)",
            "• Compliance (BWMR + EU)",
            "• Safety (cost-type)",
            "• EPR Return"]
    for i, c_ in enumerate(crit):
        el.append(text(1200, 355 + i * 26, c_, size=13, w=350))

    el.append(text(1200, 540, "Grade thresholds",
                   size=16, color=PALETTE["data"]))
    grades = ["A: 80% < SoH ≤ 100%",
              "B: 60% < SoH ≤ 80%",
              "C: 40% < SoH ≤ 60%",
              "D: SoH ≤ 40%"]
    for i, gx in enumerate(grades):
        el.append(text(1200, 575 + i * 26, gx, size=13, w=350))

    el.append(text(1200, 720, "Example (smoke test)",
                   size=16, color=PALETTE["data"]))
    ex = ["Battery: BL_SNL_18650_NMC",
          "Cycles: 1,321  ·  SoH: 73.8%",
          "→ Grade B",
          "→ Rank-1: Home/Distributed ESS",
          "→ Closeness C* = 0.73"]
    for i, e in enumerate(ex):
        el.append(text(1200, 755 + i * 26, e, size=12, w=350))

    el.append(text(40, 1360,
        "Sources: src/mcdm/topsis.py · VARIABLES_DOCUMENT.md §6.3 · "
        "scripts/smoke_test_e2e.py · "
        "data/processed/mcdm_weights/fuzzy_bwm_input.csv",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-12_topsis_workflow.excalidraw", el, "D-12 generator")


# ---------- D-13: Sensitivity Analysis ------------------------------------

def build_d13():
    random.seed(113)
    el = []
    el.append(text(40, 20,
        "D-13 · Sensitivity Analysis (5 scenarios) — RQ2", size=24))
    el.append(text(40, 55,
        "Stability of route recommendation across weight regimes",
        size=13, color=PALETTE["neutral"]))

    # scenarios → topsis runs
    el += lane(40, 100, 1500, 280, "Scenarios → 5 TOPSIS runs",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])

    scen = [
        ("Equal Weights", "all 0.20"),
        ("Literature Mean", "from Fuzzy BWM"),
        ("Technical-Heavy", "0.35 / 0.15 / 0.20 / 0.20 / 0.10"),
        ("Compliance-Heavy", "0.15 / 0.15 / 0.15 / 0.40 / 0.15"),
        ("Economic-Heavy", "0.15 / 0.35 / 0.20 / 0.15 / 0.15"),
    ]
    scen_boxes = []
    for i, (name, desc) in enumerate(scen):
        x = 70 + i * 290
        r, t = node(x, 160, 270, 100, [name, desc],
                    stroke=PALETTE["mcdm"], font=13)
        el += [r, t]
        scen_boxes.append(r)

    runs_box, rbt = node(560, 320, 460, 50,
        ["5 × TOPSIS run on identical decision matrix"],
        stroke=PALETTE["mcdm"])
    el += [runs_box, rbt]
    for sb in scen_boxes:
        el += arrow(sb, runs_box)

    # heatmap of route ranks
    el += lane(40, 420, 1500, 480, "Rank heatmap (4 alt × 5 scenarios)",
               PALETTE["data_bg"], PALETTE["data"])
    # mock illustrative ranks
    ranks = [
        # rows = alternatives ; cols = scenarios
        [2, 1, 1, 3, 1],  # Grid ESS
        [1, 2, 2, 2, 2],  # Home ESS
        [3, 3, 3, 4, 3],  # Reuse
        [4, 4, 4, 1, 4],  # Recycle
    ]
    alts = ["Grid-scale ESS", "Home/Distributed ESS",
            "Component Reuse", "Direct Recycling"]
    cell_w, cell_h = 220, 70
    grid_x, grid_y = 360, 540
    # column headers
    for j, (name, _) in enumerate(scen):
        el.append(text(grid_x + j * cell_w + 10, grid_y - 30,
                       name, size=12, w=cell_w))
    # row headers
    for i, name in enumerate(alts):
        el.append(text(grid_x - 240, grid_y + i * cell_h + 22,
                       name, size=13, color=PALETTE["text"], w=240))
    # cells
    rank_color = {1: "#27AE60", 2: "#7CC588", 3: "#F39C12", 4: "#C0392B"}
    for i in range(4):
        for j in range(5):
            x = grid_x + j * cell_w
            y = grid_y + i * cell_h
            r_ = ranks[i][j]
            cell = _base(
                type="rectangle", x=x, y=y, width=cell_w, height=cell_h,
                backgroundColor=rank_color[r_],
                strokeColor="#ffffff", strokeWidth=2,
                roundness=None, fillStyle="solid", roughness=0,
                opacity=85,
            )
            label = _base(
                type="text", x=x, y=y + cell_h / 2 - 14,
                width=cell_w, height=28,
                text=f"#{r_}", fontSize=20, fontFamily=1,
                textAlign="center", verticalAlign="middle",
                baseline=18, lineHeight=1.25,
                originalText=f"#{r_}",
                strokeColor="#ffffff",
            )
            el += [cell, label]

    # legend
    el.append(text(360, 840, "rank-1 (best)  →  rank-4 (worst)   "
                              "·  Spearman ρ between scenarios > 0.8 = robust",
                   size=12, color=PALETTE["neutral"], w=1100))
    el.append(text(40, 920,
        "Note: ranks above are illustrative — re-render after running "
        "src/mcdm/sensitivity.py with current weights.",
        size=11, color=PALETTE["red"], w=1500))
    el.append(text(40, 945,
        "Sources: src/utils/config.py::SENSITIVITY_SCENARIOS · "
        "src/mcdm/sensitivity.py (planned) · "
        "VARIABLES_DOCUMENT.md §6.3",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-13_sensitivity_analysis.excalidraw", el, "D-13 generator")


# ---------- D-14: DPP Schema Categories Map (radial mindmap) --------------

def build_d14():
    random.seed(114)
    el = []
    el.append(text(40, 20, "D-14 · Unified DPP Schema — 9 Categories", size=24))
    el.append(text(40, 55,
        "EU 2023/1542 Annex XIII + GBA Battery Pass v1.2.0 + BWMR 2022 reconciled",
        size=13, color=PALETTE["neutral"]))

    # central hub
    cx, cy = 800, 620
    hub, ht = node(cx - 130, cy - 50, 260, 100,
        ["Unified DPP", "9 categories"],
        stroke=PALETTE["dpp"], bg="#E8F5E9", font=18)
    el += [hub, ht]

    cats = [
        ("Identity",
         ["dpp_id (UUID)", "battery_id", "product_status",
          "manufacturer"]),
        ("Chemistry & Composition",
         ["cathode (NMC/LFP/...)", "CRMs (Co/Li/Ni)",
          "hazardous flags"]),
        ("Performance & Durability",
         ["rated_capacity_Ah", "voltage_min/max",
          "lifetime_cycles"]),
        ("State of Health",
         ["soh_percent", "rul_remaining",
          "estimation_method", "EU §4 (restricted)"]),
        ("Carbon Footprint",
         ["kgCO2eq_per_kWh", "lifecycle stages",
          "(placeholder)"]),
        ("Supply Chain Due Diligence",
         ["cobalt / lithium origin", "audit_id",
          "(placeholder)"]),
        ("Circularity & EoL",
         ["grade", "recommended_route", "mcdm_weights",
          "EPR compliance (BWMR)"]),
        ("Labels & Compliance",
         ["CE marking", "BWMR Form 6"]),
        ("Dismantling & Safety",
         ["diagrams", "access_level"]),
    ]
    import math
    n = len(cats)
    rx, ry = 520, 380   # ellipse radii
    for i, (title_, fields) in enumerate(cats):
        # angle: distribute around circle, start at top
        ang = -math.pi / 2 + i * (2 * math.pi / n)
        nx = cx + rx * math.cos(ang) - 220
        ny = cy + ry * math.sin(ang) - 60
        h = 40 + len(fields) * 22
        # category box
        cat, ct = node(nx, ny, 440, h,
            [title_] + ["• " + f for f in fields],
            stroke=PALETTE["dpp"], font=12)
        el += [cat, ct]
        el += arrow(hub, cat)

    el.append(text(40, 1240,
        "Sources: data/processed/dpp/unified_dpp_schema.json · "
        "data/processed/dpp/eu_annex_xiii_fields.csv",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-14_dpp_schema_categories.excalidraw", el, "D-14 generator")


# ---------- D-15: DPP Generation Sequence Diagram -------------------------

def _lifeline(x, top, bottom, label, color):
    """Sequence-diagram lifeline: header box + dashed vertical line."""
    box = _base(
        type="rectangle", x=x - 100, y=top, width=200, height=60,
        backgroundColor="#FFFFFF", strokeColor=color, strokeWidth=2,
        roundness={"type": 3}, fillStyle="solid", roughness=1,
    )
    label_t = _base(
        type="text", x=x - 100, y=top + 18, width=200, height=24,
        text=label, fontSize=14, fontFamily=1,
        textAlign="center", verticalAlign="middle",
        baseline=12, lineHeight=1.25,
        originalText=label, containerId=box["id"],
        strokeColor=color,
    )
    box["boundElements"] = [{"id": label_t["id"], "type": "text"}]
    line = _base(
        type="line", x=x, y=top + 60, width=0, height=bottom - (top + 60),
        points=[[0, 0], [0, bottom - (top + 60)]],
        strokeColor=color, strokeWidth=1,
        strokeStyle="dashed", roughness=0,
    )
    return [box, label_t, line], box


def _seq_arrow(x1, x2, y, label, color=None, dashed=False):
    if color is None:
        color = PALETTE["text"]
    arr = _base(
        type="arrow", x=x1, y=y, width=x2 - x1, height=0,
        points=[[0, 0], [x2 - x1, 0]],
        startArrowhead=None, endArrowhead="arrow",
        strokeColor=color, strokeWidth=2,
        strokeStyle="dashed" if dashed else "solid",
        roughness=1, fillStyle="solid",
        lastCommittedPoint=None,
    )
    mx = (x1 + x2) / 2
    lbl = _base(
        type="text", x=mx - 150, y=y - 24, width=300, height=22,
        text=label, fontSize=12, fontFamily=1,
        textAlign="center", verticalAlign="middle",
        baseline=10, lineHeight=1.25,
        originalText=label, strokeColor=color,
    )
    return [arr, lbl]


def build_d15():
    random.seed(115)
    el = []
    el.append(text(40, 20, "D-15 · DPP Generation Sequence", size=24))
    el.append(text(40, 55,
        "Concrete walk-through of scripts/smoke_test_e2e.py for one battery",
        size=13, color=PALETTE["neutral"]))

    actors = [
        ("User / Orchestrator", PALETTE["text"]),
        ("unified.parquet",      PALETTE["data"]),
        ("XGBoost SoH",          PALETTE["data"]),
        ("TOPSIS routing",       PALETTE["mcdm"]),
        ("schema_mapper",        PALETTE["dpp"]),
        ("DPP JSON",             PALETTE["dpp"]),
    ]
    top = 110
    bottom = 990
    xs = [180 + i * 240 for i in range(len(actors))]
    boxes = []
    for x, (name, color) in zip(xs, actors):
        elems, b = _lifeline(x, top, bottom, name, color)
        el += elems
        boxes.append(b)

    # bottom marker
    for i, x in enumerate(xs):
        bm = _base(
            type="rectangle", x=x - 100, y=bottom, width=200, height=40,
            backgroundColor="#FFFFFF", strokeColor=actors[i][1], strokeWidth=2,
            roundness={"type": 3}, fillStyle="solid", roughness=1,
        )
        bt = _base(
            type="text", x=x - 100, y=bottom + 8, width=200, height=24,
            text=actors[i][0], fontSize=13, fontFamily=1,
            textAlign="center", verticalAlign="middle",
            baseline=11, lineHeight=1.25,
            originalText=actors[i][0], containerId=bm["id"],
            strokeColor=actors[i][1],
        )
        bm["boundElements"] = [{"id": bt["id"], "type": "text"}]
        el += [bm, bt]

    # Messages
    msgs = [
        # (from_idx, to_idx, label, dashed?)
        (0, 1, "pick battery_id", False),
        (1, 0, "1,321 cycle rows", True),
        (0, 2, "predict(latest cycle row)", False),
        (2, 0, "SoH = 73.8%", True),
        (0, 0, "grade_from_soh()  →  'B'", False),
        (0, 3, "run_canonical_topsis(grade='B')", False),
        (3, 0, "rank-1 = Home/Distributed ESS", True),
        (0, 4, "build_dpp(soh, grade, route, ...)", False),
        (4, 5, "write dpp_BL_SNL_18650.json", False),
        (4, 4, "validate_against_schema()", False),
        (4, 0, "ok=True · coverage=0.487", True),
    ]
    y = top + 110
    step = 70
    for fi, ti, lbl, dashed in msgs:
        if fi == ti:
            # self-call: small loop arrow
            x = xs[fi]
            arr = _base(
                type="arrow", x=x, y=y, width=80, height=30,
                points=[[0, 0], [80, 0], [80, 30], [0, 30]],
                startArrowhead=None, endArrowhead="arrow",
                strokeColor=PALETTE["text"], strokeWidth=2,
                strokeStyle="dashed" if dashed else "solid",
                roughness=1, fillStyle="solid",
                lastCommittedPoint=None,
            )
            t = _base(
                type="text", x=x + 90, y=y + 4, width=400, height=22,
                text=lbl, fontSize=12, fontFamily=1,
                textAlign="left", verticalAlign="middle",
                baseline=10, lineHeight=1.25,
                originalText=lbl, strokeColor=PALETTE["text"],
            )
            el += [arr, t]
        else:
            el += _seq_arrow(xs[fi], xs[ti], y, lbl, dashed=dashed)
        y += step

    el.append(text(40, 1010,
        "Sources: scripts/smoke_test_e2e.py · src/dpp/schema_mapper.py · "
        "results/dpp_output/",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-15_dpp_generation_sequence.excalidraw", el, "D-15 generator")


# ---------- D-17: Iter-1 → Iter-2 Methodology -----------------------------

def build_d17():
    random.seed(117)
    el = []
    el.append(text(40, 20,
        "D-17 · Iteration Methodology — Iter-1 → Iter-2", size=24))
    el.append(text(40, 55,
        "Honest iterative ML loop driven by EDA findings + literature review",
        size=13, color=PALETTE["neutral"]))

    # Iter-1
    el += lane(40, 100, 700, 1100, "Iteration 1 (baseline, completed)",
               PALETTE["data_bg"], PALETTE["data"])
    base, bt = node(120, 170, 540, 90,
        ["Iter-1 baseline trained", "5 models · LOSO complete"],
        stroke=PALETTE["data"])
    eda, et = node(120, 290, 540, 90,
        ["EDA findings", "5-layer EDA · 34 figures"],
        stroke=PALETTE["data"])
    el += [base, bt, eda, et]
    el += arrow(base, eda)

    # 5 weaknesses
    weaknesses = [
        "W1 · LOSO median R² 0.348 (vs target ≥ 0.80)",
        "W2 · Source-ID leakage from one-hot loaders",
        "W3 · LSTM RUL test RMSE 4.54% (gate < 2% fails)",
        "W4 · No dQ/dV physics-informed features",
        "W5 · VAE adds little signal vs IsolationForest",
    ]
    y = 410
    for w_ in weaknesses:
        wb, wt_ = node(120, y, 540, 60, [w_],
                       stroke=PALETTE["red"], font=13)
        el += [wb, wt_]
        if y == 410:
            el += arrow(eda, wb)
        y += 80

    lit, lt = node(120, 850, 540, 90,
        ["Literature review", "T2 papers · LOSO baselines"],
        stroke=PALETTE["data"])
    el += [lit, lt]

    # Iter-2
    el += lane(780, 100, 760, 1100, "Iteration 2 (planned redesign)",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])
    fixes = [
        ("F1 · dQ/dV physics features",
         "incremental capacity peaks per cycle"),
        ("F2 · Drop source-ID columns",
         "+ DANN domain-adversarial training"),
        ("F3 · LSTM regularization",
         "AdamW + Huber + recurrent_dropout 0.35 + patience 8"),
        ("F4 · Drop VAE; keep IsolationForest",
         "TA-flagged removal"),
    ]
    y = 170
    fix_boxes = []
    for title_, desc in fixes:
        fb, ft_ = node(820, y, 680, 90, [title_, desc],
                       stroke=PALETTE["mcdm"], dashed=True, font=13)
        el += [fb, ft_]
        fix_boxes.append(fb)
        y += 110

    retrain, rrt = node(820, 640, 680, 80,
        ["Retrain", "(same splits · seed=42)"],
        stroke=PALETTE["mcdm"], dashed=True)
    eval_, evt = node(820, 750, 680, 110,
        ["Re-evaluate",
         "LOSO + stratified hold-out",
         "target: median R² 0.70–0.80 (honest), 0.85 stretch"],
        stroke=PALETTE["mcdm"], dashed=True, font=13)
    iter3, it3 = node(820, 890, 320, 90,
        ["Iter-3 if gates miss"],
        stroke=PALETTE["red"], dashed=True)
    paper, pt = node(1180, 890, 320, 90,
        ["Manuscript draft if pass"],
        stroke=PALETTE["dpp"], dashed=True)
    el += [retrain, rrt, eval_, evt, iter3, it3, paper, pt]
    for fb in fix_boxes:
        el += arrow(fb, retrain)
    el += arrow(retrain, eval_)
    el += arrow(eval_, iter3, label="fail")
    el += arrow(eval_, paper, label="pass")

    # cross-lane arrows
    el += arrow(lit, fix_boxes[0])

    el.append(text(40, 1230,
        "Sources: DATASET_IMPLEMENTATION_PLAN.md §12 · "
        "ML_ITERATION_2_DESIGN.md · "
        "results/tables/cross_source/results.csv",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-17_iteration_methodology.excalidraw", el, "D-17 generator")


# ---------- D-19: Baseline Comparison (B1–B6 vs Ours) ---------------------

def build_d19():
    random.seed(119)
    el = []
    el.append(text(40, 20,
        "D-19 · Baseline Comparison (Ours vs B1–B6)", size=24))
    el.append(text(40, 55,
        "B1–B6 column headers are placeholders — fill from original proposal "
        "document",
        size=13, color=PALETTE["red"]))

    # Table grid
    rows = [
        "Cross-source LOSO evaluation",
        "Multi-source corpus (≥10 sources)",
        "PyBaMM synthetic augmentation",
        "Indian operating profiles (synthetic)",
        "Dual-compliance schema (EU + BWMR)",
        "GBA Battery Pass alignment",
        "Fuzzy BWM weight aggregation",
        "TOPSIS routing across 4 alternatives",
        "Sensitivity analysis (5 scenarios)",
        "Anomaly detection (IsolationForest)",
        "DPP JSON output (schema-validated)",
        "Iterative ML redesign (Iter-2)",
    ]
    cols = ["Capability", "Ours", "B1", "B2", "B3", "B4", "B5", "B6"]
    # placeholder cell values (Ours = ✓ for all; B1-B6 = ?)
    ours = ["✓"] * len(rows)
    placeholder = "?"

    col_w = [520, 130, 110, 110, 110, 110, 110, 110]
    row_h = 56
    total_w = sum(col_w)
    grid_x = (1600 - total_w) // 2
    grid_y = 110

    # Header
    x = grid_x
    for j, h in enumerate(cols):
        bg = PALETTE["data_bg"] if j == 0 else (
            PALETTE["dpp_bg"] if j == 1 else PALETTE["mcdm_bg"])
        stroke = PALETTE["data"] if j <= 1 else PALETTE["mcdm"]
        cell = _base(
            type="rectangle", x=x, y=grid_y, width=col_w[j], height=row_h,
            backgroundColor=bg, strokeColor=stroke, strokeWidth=2,
            roundness=None, fillStyle="solid", roughness=0,
        )
        t = _base(
            type="text", x=x, y=grid_y + 16,
            width=col_w[j], height=24,
            text=h, fontSize=16, fontFamily=1,
            textAlign="center", verticalAlign="middle",
            baseline=14, lineHeight=1.25,
            originalText=h, strokeColor=stroke,
        )
        el += [cell, t]
        x += col_w[j]

    # Body rows
    for i, label in enumerate(rows):
        x = grid_x
        y = grid_y + row_h * (i + 1)
        for j in range(len(cols)):
            if j == 0:
                txt_ = label
                bg = "#FAFAFA"
                tcolor = PALETTE["text"]
                size = 13
                align = "left"
            elif j == 1:
                txt_ = ours[i]
                bg = "#E8F5E9"
                tcolor = PALETTE["dpp"]
                size = 22
                align = "center"
            else:
                txt_ = placeholder
                bg = "#FFFFFF"
                tcolor = PALETTE["neutral"]
                size = 18
                align = "center"
            cell = _base(
                type="rectangle", x=x, y=y, width=col_w[j], height=row_h,
                backgroundColor=bg, strokeColor="#CFD2D6", strokeWidth=1,
                roundness=None, fillStyle="solid", roughness=0,
            )
            t = _base(
                type="text",
                x=x + (12 if align == "left" else 0),
                y=y + (row_h - size * 1.25) / 2,
                width=col_w[j] - (12 if align == "left" else 0),
                height=int(size * 1.4),
                text=txt_, fontSize=size, fontFamily=1,
                textAlign=align, verticalAlign="middle",
                baseline=int(size * 0.85), lineHeight=1.25,
                originalText=txt_, strokeColor=tcolor,
            )
            el += [cell, t]
            x += col_w[j]

    # Legend
    legend_y = grid_y + row_h * (len(rows) + 1) + 30
    el.append(text(grid_x, legend_y,
        "✓  capability present     "
        "?  to be filled from proposal     "
        "✗  absent     "
        "~  partial",
        size=13, color=PALETTE["neutral"], w=1500))
    el.append(text(grid_x, legend_y + 30,
        "Differentiators: cross-source LOSO (10 sources) · "
        "Indian synthetic profiles · dual-compliance DPP · "
        "honest Iter-2 redesign published with results",
        size=12, color=PALETTE["text"], w=1500))

    el.append(text(grid_x, legend_y + 80,
        "Sources: original Group-5 proposal (PDF/DOCX) — "
        "B1–B6 names not currently in repo",
        size=11, color=PALETTE["red"], w=1500))

    write("D-19_baseline_comparison.excalidraw", el, "D-19 generator")


# ---------- D-02: End-to-End Pipeline (single-battery flow, LR) -----------

def build_d02():
    random.seed(102)
    el = []
    el.append(text(40, 20, "D-02 · End-to-End Pipeline (single battery)", size=24))
    el.append(text(40, 55,
        "Reference cell: BL_SNL_18650_NMC · 1,321 cycles · "
        "SoH 73.8% → Grade B → Home/Distributed ESS",
        size=13, color=PALETTE["neutral"]))

    # Single horizontal flow
    cells = [
        (60,  ["Battery row", "BL_SNL_18650_NMC", "cycles = 1,321"],
            PALETTE["data"], None),
        (340, ["XGBoost SoH", "predict()", "SoH = 73.8 %"],
            PALETTE["data"], None),
        (620, ["Grade thresholds", "60 < SoH ≤ 80", "→ Grade B"],
            PALETTE["data"], None),
        (900, ["Fuzzy BWM-TOPSIS",
               "6 criteria · 4 alternatives",
               "decision matrix per grade"],
            PALETTE["mcdm"], None),
        (1180, ["Rank-1 route",
                "Home/Distributed ESS",
                "closeness C* = 0.731"],
            PALETTE["mcdm"], "#FDF1DC"),
        (1460, ["schema_mapper", "build_dpp(...)"],
            PALETTE["dpp"], None),
        (60,  ["DPP JSON",
               "schema-validated ✓",
               "coverage 48.7 %"],
            PALETTE["dpp"], "#E8F5E9"),
    ]
    boxes = []
    y = 220
    for i, (x, lines, stroke, bg) in enumerate(cells):
        # second row for the last node
        if i == 6:
            y = 540
        bg_ = bg if bg else "#FFFFFF"
        h = 110
        b, t = node(x, y, 260, h, lines, stroke=stroke, bg=bg_)
        el += [b, t]
        boxes.append(b)

    # arrows in sequence (top row)
    for i in range(5):
        el += arrow(boxes[i], boxes[i + 1])
    # last arrow goes from box 5 (mapper) to box 6 (DPP JSON) — mapper at right end of top row, DPP at left of bottom row → curve via arrow helper
    el += arrow(boxes[5], boxes[6])

    # value annotations on key arrows
    el.append(text(800, 200, "grade", size=12, w=80, color=PALETTE["text"]))

    el.append(text(40, 720,
        "Sources: scripts/smoke_test_e2e.py · "
        "results/dpp_output/smoke_summary_BL_SNL_*.json · "
        "src/dpp/schema_mapper.py",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-02_end_to_end_pipeline.excalidraw", el, "D-02 generator")


# ---------- D-03: Data Acquisition Map ------------------------------------

def build_d03():
    random.seed(103)
    el = []
    el.append(text(40, 20, "D-03 · Data Acquisition Map", size=24))
    el.append(text(40, 55,
        "5 layers · ~113 GB total cycling data + tooling + regulatory + market + literature",
        size=13, color=PALETTE["neutral"]))

    # L1 cycling — left big cluster
    el += lane(40, 100, 1500, 360, "L1 · Cycling data (real lab cells)",
               PALETTE["data_bg"], PALETTE["data"])

    cycling = [
        (80,  170, 320, 130,
         ["BatteryLife", "86 GB · 18 sub-corpora",
          "BL_Stanford / SNL / MATR / etc."],
         PALETTE["data"], 22),
        (420, 170, 240, 130,
         ["NASA PCoE", "586 MB",
          "+ Random 3.2 GB",
          "+ Random v1 (Kaggle) 1.20 GB"],
         PALETTE["data"], 14),
        (680, 170, 220, 130,
         ["CALCE", "CS2 / CX2", "1.05 GB"],
         PALETTE["data"], 16),
        (920, 170, 240, 130,
         ["Stanford OSF", "17 GB",
          "+ 8jnr5 2.15 GB"],
         PALETTE["data"], 16),
        (1180, 170, 320, 130,
         ["Subtotal L1", "~111 GB",
          "→ unified.parquet"],
         PALETTE["data"], 18),
    ]
    for x, y, w, h, lines, stroke, font in cycling:
        b, t = node(x, y, w, h, lines, stroke=stroke, font=font,
                    bg="#FFFFFF" if "Subtotal" not in lines[0] else "#E8EEF5")
        el += [b, t]

    # second row inside L1: scale bar
    el.append(text(80, 360,
        "Size legend (proportional): BatteryLife dominates by 1–2 orders of magnitude",
        size=12, color=PALETTE["neutral"], w=1400))

    # L2 synthetic
    el += lane(40, 480, 720, 200, "L2 · Synthetic tooling",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])
    pyb, pybt = node(80, 540, 320, 130,
        ["PyBaMM", "(tool, not raw data)",
         "70 Indian cells synthesized",
         "50 NMC + 20 LFP"],
        stroke=PALETTE["mcdm"], font=14)
    syn, synt = node(420, 540, 320, 130,
        ["Output", "data/processed/synthetic_indian/",
         "+ synthetic_indian_lfp/",
         "pinned to TRAIN split"],
        stroke=PALETTE["mcdm"], font=13)
    el += [pyb, pybt, syn, synt]
    el += arrow(pyb, syn)

    # L3 regulatory
    el += lane(780, 480, 760, 200, "L3 · Regulatory",
               PALETTE["dpp_bg"], PALETTE["dpp"])
    eu, eut    = node(820, 540, 220, 130,
        ["EU 2023/1542", "Annex XIII", "HTML + PDF"],
        stroke=PALETTE["dpp"], font=14)
    gba, gbat  = node(1060, 540, 220, 130,
        ["GBA Battery Pass", "v1.2.0 repo",
         "Data Model JSON"],
        stroke=PALETTE["dpp"], font=14)
    bwmr, bwmrt = node(1300, 540, 220, 130,
        ["BWMR 2022", "+ 4 amendments",
         "PDFs ~10 MB"],
        stroke=PALETTE["dpp"], font=14)
    el += [eu, eut, gba, gbat, bwmr, bwmrt]

    # L4 market
    el += lane(40, 700, 720, 200, "L4 · Market data",
               PALETTE["data_bg"], PALETTE["data"])
    cpcb, cpcbt = node(80, 760, 220, 130,
        ["CPCB", "EPR table",
         "(India recyclers)"],
        stroke=PALETTE["data"], font=14)
    icea, iceat = node(320, 760, 220, 130,
        ["ICEA", "projections",
         "EV battery EoL volumes"],
        stroke=PALETTE["data"], font=14)
    wri, writ  = node(560, 760, 180, 130,
        ["WRI", "second-charge",
         "report"],
        stroke=PALETTE["data"], font=14)
    el += [cpcb, cpcbt, icea, iceat, wri, writ]

    # L5 literature
    el += lane(780, 700, 760, 200, "L5 · Literature",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])
    lit, litt = node(820, 760, 700, 130,
        ["11 T2 papers", "(MCDM weight extraction)",
         "8 of 11 contributing weights · 22 rows parsed",
         "→ canonical 6 criteria"],
        stroke=PALETTE["mcdm"], font=14)
    el += [lit, litt]

    # Total callout
    total, tt = node(620, 920, 360, 70,
        ["Total ≈ 113 GB", "(L1 dominates)"],
        stroke=PALETTE["data"], bg="#E8EEF5", font=18)
    el += [total, tt]

    el.append(text(40, 1020,
        "Sources: DATASET_IMPLEMENTATION_PLAN.md §2.1–2.5 · "
        "data/AUDIT_REPORT.md · "
        "data/processed/mcdm_weights/AGGREGATION_REPORT.md",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-03_data_acquisition_map.excalidraw", el, "D-03 generator")


# ---------- D-05: Schema Reconciliation (3 regulations → 9 unified) -------

def build_d05():
    random.seed(105)
    el = []
    el.append(text(40, 20,
        "D-05 · Schema Reconciliation (EU + GBA + BWMR → Unified DPP)", size=24))
    el.append(text(40, 55,
        "Field-level mapping from 3 regulations to 9 unified categories",
        size=13, color=PALETTE["neutral"]))

    # Left: 3 source regulations
    src_x = 60
    src_w = 360
    src_specs = [
        ("EU Annex XIII",
         ["§1 General Product Info",
          "§2 Performance & Durability",
          "§3 Carbon footprint",
          "§4 SoH (restricted access)"],
         PALETTE["dpp"], 270),
        ("GBA Battery Pass v1.2.0",
         ["GeneralProductInformation",
          "Materials & Composition",
          "PerformanceAndDurability",
          "Carbon footprint",
          "Supply Chain Due Diligence"],
         PALETTE["mcdm"], 580),
        ("BWMR 2022 + amendments",
         ["EPR Schedules I–V",
          "Form 6 registration",
          "Recovery targets FY24-27+",
          "Recycled content tiers"],
         PALETTE["data"], 850),
    ]
    src_boxes = []
    src_strokes = []
    for name, lines, stroke, y in src_specs:
        h = 60 + len(lines) * 24
        b, t = node(src_x, y, src_w, h, [name] + ["• " + l for l in lines],
                    stroke=stroke, font=12)
        el += [b, t]
        src_boxes.append(b)
        src_strokes.append(stroke)

    # Right: 9 unified categories
    cat_x = 1080
    cat_w = 440
    cats = [
        ("Identity",                  170),
        ("Chemistry & Composition",   270),
        ("Performance & Durability",  370),
        ("State of Health",           470),
        ("Carbon Footprint",          570),
        ("Supply Chain Due Diligence",670),
        ("Circularity & EoL",         770),
        ("Labels & Compliance",       870),
        ("Dismantling & Safety",      970),
    ]
    cat_boxes = []
    for name, y in cats:
        b, t = node(cat_x, y, cat_w, 70, [name],
                    stroke=PALETTE["data"], bg="#F5F8FB", font=14)
        el += [b, t]
        cat_boxes.append(b)

    # Mapping arrows (color-coded per source) — many-to-many
    # EU (idx 0) → 0,2,3,4 (Identity, Performance, SoH, Carbon)
    # GBA (idx 1) → 0,1,2,4,5 (Identity, Chemistry, Performance, Carbon, Supply Chain)
    # BWMR (idx 2) → 0,6,7 (Identity, Circularity, Labels) + 8 (Dismantling for safety form)
    mapping = [
        (0, [0, 2, 3, 4]),
        (1, [0, 1, 2, 4, 5]),
        (2, [0, 6, 7, 8]),
    ]
    for src_idx, targets in mapping:
        for t_idx in targets:
            el += arrow(src_boxes[src_idx], cat_boxes[t_idx],
                        stroke=src_strokes[src_idx])

    # Annotations on right
    el.append(text(cat_x + cat_w + 20, 470,
        "← EU §4 (restricted)", size=11, color=PALETTE["red"], w=200))
    el.append(text(cat_x + cat_w + 20, 770,
        "← BWMR-specific", size=11, color=PALETTE["data"], w=200))

    # Legend
    el.append(text(60, 1080,
        "Arrow color = source regulation:",
        size=12, color=PALETTE["text"], w=400))
    el.append(text(60, 1100, "■  EU 2023/1542",
                   size=12, color=PALETTE["dpp"], w=200))
    el.append(text(220, 1100, "■  GBA Battery Pass",
                   size=12, color=PALETTE["mcdm"], w=220))
    el.append(text(440, 1100, "■  BWMR 2022",
                   size=12, color=PALETTE["data"], w=200))

    el.append(text(40, 1140,
        "Sources: data/processed/dpp/unified_dpp_schema.json · "
        "data/processed/dpp/eu_annex_xiii_fields.csv · "
        "data/regulatory/gba/BatteryPassDataModel/",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-05_schema_reconciliation.excalidraw", el, "D-05 generator")


# ---------- D-06: ML Pipeline Architecture (5 models in parallel) ---------

def build_d06():
    random.seed(106)
    el = []
    el.append(text(40, 20,
        "D-06 · ML Pipeline Architecture (Iter-1)", size=24))
    el.append(text(40, 55,
        "Feature bundle → 4 parallel models → grade & anomaly flags",
        size=13, color=PALETTE["neutral"]))

    # Iter-1 wrapper lane (dashed)
    el += lane(40, 100, 1500, 800,
               "Iteration 1 — full subgraph (dashed wrapper)",
               PALETTE["model_bg"], PALETTE["model"])

    feat, ft = node(580, 170, 420, 100,
        ["Feature bundle", "33 features (17 numeric + 16 one-hot)",
         "from training_data.py"],
        stroke=PALETTE["data"])
    el += [feat, ft]

    # 4 parallel models
    xgb,  xgbt  = node(80,  340, 320, 130,
        ["XGBoost SoH", "depth 9 · 996 trees",
         "R² = 0.999 · gates ✅"],
        stroke=PALETTE["data"], font=13)
    lstm, lstmt = node(440, 340, 320, 130,
        ["LSTM RUL", "h=128 · 2 layers · drop 0.2",
         "test RMSE 4.54% · gate ✗"],
        stroke=PALETTE["mcdm"], dashed=True, font=13)
    iso,  isot  = node(800, 340, 320, 130,
        ["IsolationForest", "200 trees · contamination 0.05",
         "anomaly rate ~5%"],
        stroke=PALETTE["data"], font=13)
    vae,  vaet  = node(1160, 340, 320, 130,
        ["VAE (deprecated)", "encoder 3-layer · latent z=12",
         "TA-flagged for removal"],
        stroke=PALETTE["red"], dashed=True, font=13)
    el += [xgb, xgbt, lstm, lstmt, iso, isot, vae, vaet]
    for m in (xgb, lstm, iso, vae):
        el += arrow(feat, m)

    # Grade classifier
    grade, gradet = node(80, 580, 320, 110,
        ["Grade Classifier", "A / B / C / D",
         "from SoH thresholds"],
        stroke=PALETTE["data"])
    el += [grade, gradet]
    el += arrow(xgb, grade, label="SoH%")

    # RUL output
    rul, rult = node(440, 580, 320, 110,
        ["RUL output (cycles)", "→ DPP state_of_health"],
        stroke=PALETTE["data"])
    el += [rul, rult]
    el += arrow(lstm, rul)

    # Anomaly flags
    flags, flagst = node(800, 580, 320, 110,
        ["Anomaly flags", "top 5 % outliers",
         "→ DPP labels & QA"],
        stroke=PALETTE["data"])
    el += [flags, flagst]
    el += arrow(iso, flags)
    el += arrow(vae, flags, dashed=True)

    # TA recommendation banner
    ta, tat = node(1160, 580, 320, 110,
        ["TA recommendation",
         "use IsolationForest only",
         "(VAE removal in Iter-2)"],
        stroke=PALETTE["red"], bg="#FBE9E7")
    el += [ta, tat]

    # downstream stub
    down, dt = node(580, 740, 420, 100,
        ["→ MCDM (TOPSIS) routing",
         "→ DPP JSON generation"],
        stroke=PALETTE["dpp"], bg="#E8F5E9")
    el += [down, dt]
    el += arrow(grade, down)
    el += arrow(rul, down)

    el.append(text(40, 920,
        "Sources: scripts/train_all_models.py · src/models/ · "
        "results/tables/{xgboost_soh,lstm_rul,isolation_forest,vae_anomaly}/metrics.json",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-06_ml_pipeline_architecture.excalidraw", el, "D-06 generator")


# ---------- D-07: XGBoost SoH Architecture --------------------------------

def build_d07():
    random.seed(107)
    el = []
    el.append(text(40, 20,
        "D-07 · XGBoost SoH Model Architecture", size=24))
    el.append(text(40, 55,
        "33-feature input → Optuna-tuned tree ensemble → SoH %",
        size=13, color=PALETTE["neutral"]))

    # Main flow LR
    el += lane(40, 100, 1500, 360,
               "Pipeline (gates: R² > 0.95 ✓ · RMSE < 2 % ✓ · MAE < 1.5 % ✓)",
               PALETTE["model_bg"], PALETTE["model"])

    inp, inpt = node(80, 200, 280, 130,
        ["Input features",
         "33 = 17 numeric",
         "+ 16 one-hot",
         "(StandardScaler)"],
        stroke=PALETTE["data"], font=13)
    ens, enst = node(420, 200, 360, 130,
        ["Optuna-tuned",
         "Tree Ensemble",
         "max_depth = 9 · 996 trees",
         "lr = 0.037 · subsample 0.94"],
        stroke=PALETTE["data"], font=13)
    early, eat = node(820, 200, 280, 130,
        ["Early stopping", "rounds = 30",
         "best iter = 983"],
        stroke=PALETTE["data"], font=13)
    out, ot = node(1160, 200, 320, 130,
        ["SoH %", "test R² = 0.999",
         "RMSE = 1.09 · MAE = 0.40"],
        stroke=PALETTE["dpp"], bg="#E8F5E9", font=13)
    el += [inp, inpt, ens, enst, early, eat, out, ot]
    el += arrow(inp, ens)
    el += arrow(ens, early)
    el += arrow(early, out)

    # Hyperparam panel
    el += lane(40, 500, 720, 360, "Hyperparameters (Optuna best)",
               PALETTE["data_bg"], PALETTE["data"])
    hyp_lines = [
        "n_estimators        : 996",
        "max_depth           : 9",
        "learning_rate       : 0.0372",
        "subsample           : 0.945",
        "colsample_bytree    : 0.827",
        "reg_alpha           : 0.011",
        "reg_lambda          : 1.469",
        "min_child_weight    : 5",
        "early_stopping      : 30 rounds",
    ]
    for i, line in enumerate(hyp_lines):
        el.append(text(80, 560 + i * 28, line, size=14,
                       color=PALETTE["text"], w=620))

    # Metrics panel
    el += lane(780, 500, 760, 360, "Metrics (train / val / test)",
               PALETTE["dpp_bg"], PALETTE["dpp"])
    metrics = [
        ("R²",   "0.99996", "0.99937", "0.99915"),
        ("RMSE", "0.250",   "0.978",   "1.094"),
        ("MAE",  "0.151",   "0.318",   "0.402"),
        ("MAPE", "2.85 %",  "7.22 %",  "4.90 %"),
    ]
    # header
    el.append(text(820,  560, "metric", size=14, color=PALETTE["dpp"], w=120))
    el.append(text(960,  560, "train",  size=14, color=PALETTE["dpp"], w=120))
    el.append(text(1140, 560, "val",    size=14, color=PALETTE["dpp"], w=120))
    el.append(text(1320, 560, "test",   size=14, color=PALETTE["dpp"], w=120))
    for i, (m, tr, va, te) in enumerate(metrics):
        y = 600 + i * 32
        el.append(text(820,  y, m,  size=14, w=120))
        el.append(text(960,  y, tr, size=14, w=120))
        el.append(text(1140, y, va, size=14, w=120))
        el.append(text(1320, y, te, size=14, w=120))
    el.append(text(820, 740,
        "n_train = 1,540,820 · n_val = 362,359 · n_test = 300,869",
        size=12, color=PALETTE["neutral"], w=700))
    el.append(text(820, 770,
        "All gates PASS ✅ — train_time = 14.65 s",
        size=13, color=PALETTE["dpp"], w=700))

    el.append(text(40, 900,
        "Sources: src/models/xgboost_soh.py · "
        "scripts/train_xgboost_soh.py · "
        "results/tables/xgboost_soh/metrics.json",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-07_xgboost_soh_architecture.excalidraw", el, "D-07 generator")


# ---------- D-09: IsolationForest + VAE Anomaly Detection -----------------

def build_d09():
    random.seed(109)
    el = []
    el.append(text(40, 20,
        "D-09 · Anomaly Detection — IsolationForest + VAE", size=24))
    el.append(text(40, 55,
        "Side-by-side: kept (left) vs deprecated (right, dashed amber)",
        size=13, color=PALETTE["neutral"]))

    # Left panel: IsolationForest
    el += lane(40, 100, 720, 720,
               "IsolationForest (KEEP — TA-recommended)",
               PALETTE["data_bg"], PALETTE["data"])
    inp1, it1 = node(80, 180, 640, 100,
        ["33-feature input", "(post-scaling)"],
        stroke=PALETTE["data"], font=14)
    iso, ist = node(80, 320, 640, 130,
        ["IsolationForest",
         "n_estimators = 200",
         "contamination = 0.05",
         "fit_time = 2.78 s"],
        stroke=PALETTE["data"], font=14)
    score, sct = node(80, 490, 640, 100,
        ["−decision_function",
         "→ anomaly score per sample"],
        stroke=PALETTE["data"], font=14)
    flag, flt = node(80, 630, 640, 130,
        ["Top 5 % flagged",
         "train rate 5.00 % · val 4.53 % · test 5.38 %",
         "→ DPP labels & QA"],
        stroke=PALETTE["dpp"], bg="#E8F5E9", font=14)
    el += [inp1, it1, iso, ist, score, sct, flag, flt]
    el += arrow(inp1, iso)
    el += arrow(iso, score)
    el += arrow(score, flag)

    # Right panel: VAE (deprecated)
    el += lane(820, 100, 720, 720,
               "VAE (DEPRECATED — TA-flagged for removal)",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])
    inp2, it2 = node(860, 180, 640, 100,
        ["33-feature input", "(post-scaling)"],
        stroke=PALETTE["red"], dashed=True, font=14)
    enc, ent = node(860, 320, 640, 100,
        ["Encoder", "3 hidden layers", "→ μ, log σ²"],
        stroke=PALETTE["red"], dashed=True, font=14)
    z, zt = node(860, 450, 640, 80,
        ["Latent z", "dim = 12"],
        stroke=PALETTE["red"], dashed=True, font=14)
    dec, dct = node(860, 560, 640, 100,
        ["Decoder",
         "→ x̂ reconstruction"],
        stroke=PALETTE["red"], dashed=True, font=14)
    rec, rct = node(860, 690, 640, 100,
        ["Reconstruction error",
         "top 5 % flagged"],
        stroke=PALETTE["red"], dashed=True, font=14)
    el += [inp2, it2, enc, ent, z, zt, dec, dct, rec, rct]
    el += arrow(inp2, enc, dashed=True, stroke=PALETTE["red"])
    el += arrow(enc, z, dashed=True, stroke=PALETTE["red"])
    el += arrow(z, dec, dashed=True, stroke=PALETTE["red"])
    el += arrow(dec, rec, dashed=True, stroke=PALETTE["red"])

    # Banner
    banner, bt = node(220, 850, 1160, 80,
        ["TA recommendation",
         "Use IsolationForest only — drop VAE in Iter-2 (low marginal signal)"],
        stroke=PALETTE["red"], bg="#FBE9E7")
    el += [banner, bt]

    el.append(text(40, 960,
        "Sources: src/models/vae.py · scripts/train_isolation_forest.py · "
        "results/tables/isolation_forest/metrics.json",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-09_anomaly_detection.excalidraw", el, "D-09 generator")


# ---------- D-11: Fuzzy BWM Workflow --------------------------------------

def build_d11():
    random.seed(111)
    el = []
    el.append(text(40, 20, "D-11 · Fuzzy BWM Workflow", size=24))
    el.append(text(40, 55,
        "Literature → manual extraction → canonical mapping → "
        "TFNs → fuzzy_bwm_input.csv",
        size=13, color=PALETTE["neutral"]))

    # Main flow TD
    el += lane(40, 100, 900, 1160, "Aggregation pipeline",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])

    a, at_ = node(120, 170, 740, 90,
        ["11 T2 papers", "(MCDM weight extraction)"],
        stroke=PALETTE["mcdm"])
    b, bt_ = node(120, 290, 740, 100,
        ["Manual extraction",
         "literature_weights.csv · 22 rows parsed",
         "EXTRACTION_PLAYBOOK.md"],
        stroke=PALETTE["mcdm"], font=13)
    c_, ct_ = node(120, 420, 740, 100,
        ["Canonical mapping",
         "→ 6 criteria",
         "(SoH · Value · Carbon · Compliance · Safety · EPR Return)"],
        stroke=PALETTE["mcdm"], font=12)
    d_, dt_ = node(120, 550, 740, 110,
        ["Per-criterion TFN",
         "(lower, middle, upper)",
         "from min / mean / max across papers"],
        stroke=PALETTE["mcdm"], font=13)
    e_, et_ = node(120, 690, 740, 100,
        ["Normalize",
         "so middles sum to 1.00",
         "(Compliance dominates: middle = 0.266)"],
        stroke=PALETTE["mcdm"], font=13)
    f_, ft_ = node(120, 820, 740, 90,
        ["fuzzy_bwm_input.csv",
         "→ feeds TOPSIS"],
        stroke=PALETTE["dpp"], bg="#E8F5E9")
    el += [a, at_, b, bt_, c_, ct_, d_, dt_, e_, et_, f_, ft_]
    el += arrow(a, b)
    el += arrow(b, c_)
    el += arrow(c_, d_)
    el += arrow(d_, e_)
    el += arrow(e_, f_)

    # 8/11 papers callout
    el.append(text(120, 940,
        "8 of 11 papers contributed weights · 3 papers had no recoverable weights",
        size=12, color=PALETTE["neutral"], w=720))

    # Coverage table on right
    el += lane(960, 100, 580, 1160, "Coverage by criterion",
               PALETTE["data_bg"], PALETTE["data"])
    coverage = [
        ("Criterion",  "n papers", "Status"),
        ("Compliance", "6",        "PASS"),
        ("EPR Return", "4",        "PASS"),
        ("Carbon",     "3",        "PASS"),
        ("SoH",        "0",        "fall back → sensitivity"),
        ("Safety",     "1",        "fall back → sensitivity"),
        ("Value",      "1",        "fall back → sensitivity"),
    ]
    table_x = 990
    table_y = 180
    col_w = [220, 130, 220]
    row_h = 56
    for i, row in enumerate(coverage):
        x = table_x
        for j, cell_text in enumerate(row):
            if i == 0:
                bg = PALETTE["data_bg"]
                stroke = PALETTE["data"]
                tcolor = PALETTE["data"]
                fs = 14
            else:
                bg = "#FFFFFF"
                stroke = "#CFD2D6"
                if "PASS" in row[2]:
                    tcolor = PALETTE["dpp"]
                elif "fall back" in row[2]:
                    tcolor = PALETTE["mcdm"]
                else:
                    tcolor = PALETTE["text"]
                fs = 13
            cell = _base(
                type="rectangle", x=x, y=table_y + i * row_h,
                width=col_w[j], height=row_h,
                backgroundColor=bg, strokeColor=stroke, strokeWidth=1,
                roundness=None, fillStyle="solid", roughness=0,
            )
            t = _base(
                type="text", x=x + 12,
                y=table_y + i * row_h + (row_h - fs * 1.25) / 2,
                width=col_w[j] - 24, height=int(fs * 1.4),
                text=cell_text, fontSize=fs, fontFamily=1,
                textAlign="left", verticalAlign="middle",
                baseline=int(fs * 0.85), lineHeight=1.25,
                originalText=cell_text, strokeColor=tcolor,
            )
            el += [cell, t]
            x += col_w[j]

    # TFN values panel
    el.append(text(990, 600,
        "Normalized TFNs (sum middles = 1.0)",
        size=14, color=PALETTE["data"], w=540))
    tfns = [
        ("Compliance",  "0.055 / 0.266 / 0.476"),
        ("Value",       "0.360 / 0.360 / 0.360"),
        ("Carbon",      "0.140 / 0.166 / 0.192"),
        ("EPR Return",  "0.085 / 0.137 / 0.190"),
        ("Safety",      "0.071 / 0.071 / 0.071"),
        ("SoH",         "(no data — sensitivity)"),
    ]
    for i, (name, val) in enumerate(tfns):
        el.append(text(990, 640 + i * 32, name,
                       size=13, color=PALETTE["text"], w=170))
        el.append(text(1180, 640 + i * 32, val,
                       size=13, color=PALETTE["text"], w=400))

    el.append(text(40, 1280,
        "Sources: src/mcdm/fuzzy_bwm.py · "
        "data/processed/mcdm_weights/EXTRACTION_PLAYBOOK.md · "
        "data/processed/mcdm_weights/AGGREGATION_REPORT.md",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-11_fuzzy_bwm_workflow.excalidraw", el, "D-11 generator")


# ---------- D-16: Research Methodology Overview ---------------------------

def build_d16():
    random.seed(116)
    el = []
    el.append(text(40, 20, "D-16 · Research Methodology Overview", size=24))
    el.append(text(40, 55,
        "One-figure summary of methodology section · vertical flow",
        size=13, color=PALETTE["neutral"]))

    # Main vertical flow
    el += lane(40, 100, 900, 1280, "Methodology flow",
               PALETTE["data_bg"], PALETTE["data"])
    steps = [
        (170, ["Research questions",
               "RQ1: SoH-based grading accuracy",
               "RQ2: Robustness to weight regimes"],
         PALETTE["data"]),
        (300, ["Data acquisition",
               "5 layers · ~113 GB",
               "(D-03)"],
         PALETTE["data"]),
        (430, ["Preprocessing & EDA",
               "unify · split · 5-layer EDA · 34 figures",
               "(D-04)"],
         PALETTE["data"]),
        (560, ["ML training (Iter-1)",
               "XGBoost SoH · LSTM RUL · IsolationForest",
               "Grade Classifier · LOSO eval (D-06, D-10)"],
         PALETTE["model"]),
        (690, ["MCDM weighting",
               "Fuzzy BWM from 8/11 T2 papers",
               "→ 6 canonical criteria · TFNs (D-11)"],
         PALETTE["mcdm"]),
        (820, ["Routing",
               "TOPSIS · 4 alternatives × 6 criteria",
               "5-scenario sensitivity (D-12, D-13)"],
         PALETTE["mcdm"]),
        (950, ["DPP generation",
               "schema_mapper.build_dpp(...) → JSON",
               "9 categories · validated (D-14, D-15)"],
         PALETTE["dpp"]),
        (1080, ["Evaluation",
                "gates · LOSO · sensitivity · coverage"],
         PALETTE["dpp"]),
        (1210, ["Manuscript",
                "MDPI Batteries / World EV Journal"],
         PALETTE["dpp"]),
    ]
    boxes = []
    for y, lines, stroke in steps:
        b, t = node(120, y, 740, 110, lines, stroke=stroke, font=13)
        el += [b, t]
        boxes.append(b)
    for i in range(len(boxes) - 1):
        el += arrow(boxes[i], boxes[i + 1])

    # Iter-2 overlay (right side)
    el += lane(960, 100, 580, 1280,
               "Iteration 2 (planned redesign)",
               PALETTE["mcdm_bg"], PALETTE["mcdm"])
    iter2_lines = [
        ["F1 · dQ/dV physics features",
         "incremental capacity peaks"],
        ["F2 · drop source-ID columns",
         "+ DANN domain-adversarial"],
        ["F3 · LSTM regularization",
         "AdamW + Huber + drop 0.35"],
        ["F4 · drop VAE",
         "keep IsolationForest only"],
    ]
    iter2_boxes = []
    for i, lines in enumerate(iter2_lines):
        b, t = node(990, 200 + i * 130, 540, 110, lines,
                    stroke=PALETTE["mcdm"], dashed=True, font=13)
        el += [b, t]
        iter2_boxes.append(b)

    # Iter-2 → ML training (cross-arrow)
    el += arrow(iter2_boxes[0], boxes[3], dashed=True, stroke=PALETTE["mcdm"])

    iter2_target, iter2_t = node(990, 730, 540, 110,
        ["Iter-2 retrain & re-evaluate",
         "target: LOSO median R² 0.70–0.80",
         "stretch: 0.85"],
        stroke=PALETTE["mcdm"], dashed=True, font=13)
    el += [iter2_target, iter2_t]
    for ib in iter2_boxes:
        el += arrow(ib, iter2_target, dashed=True, stroke=PALETTE["mcdm"])

    el.append(text(40, 1400,
        "Sources: DATASET_IMPLEMENTATION_PLAN.md §3, §6 · "
        "ML_ITERATION_2_DESIGN.md · LITERATURE_REVIEW.md",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-16_research_methodology.excalidraw", el, "D-16 generator")


# ---------- D-18: Sub-team Responsibilities -------------------------------

def build_d18():
    random.seed(118)
    el = []
    el.append(text(40, 20, "D-18 · Sub-team Responsibilities (Group 5)", size=24))
    el.append(text(40, 55,
        "3 sub-teams with deliverables · ✅ = completed (Iter-1)",
        size=13, color=PALETTE["neutral"]))

    # Top: Group 5 root
    root, rt_ = node(640, 130, 320, 100,
        ["Group 5", "EV Battery EoL Routing"],
        stroke=PALETTE["data"], bg="#E8EEF5", font=18)
    el += [root, rt_]

    # 3 sub-teams
    teams = [
        ("ML Sub-team", PALETTE["data"], 80, [
            ("✅", "XGBoost SoH (gates pass)"),
            ("✅", "LSTM RUL (Iter-1)"),
            ("✅", "IsolationForest"),
            ("✅", "Grade Classifier"),
            ("✅", "LOSO cross-source eval"),
            ("⏳", "Iter-2 redesign (4 fixes)"),
            ("⏳", "DANN domain adaptation"),
        ]),
        ("MCDM Sub-team", PALETTE["mcdm"], 600, [
            ("✅", "11 T2 paper review"),
            ("✅", "Manual weight extraction (22 rows)"),
            ("✅", "Canonical 6-criteria mapping"),
            ("✅", "Fuzzy BWM aggregation"),
            ("✅", "TOPSIS implementation"),
            ("⏳", "5-scenario sensitivity run"),
            ("⏳", "RQ2 robustness analysis"),
        ]),
        ("DPP Sub-team", PALETTE["dpp"], 1120, [
            ("✅", "EU Annex XIII field extraction"),
            ("✅", "GBA Battery Pass alignment"),
            ("✅", "BWMR EPR mapping"),
            ("✅", "unified_dpp_schema.json"),
            ("✅", "schema_mapper.py"),
            ("✅", "3 smoke-test DPP JSONs"),
            ("⏳", "Coverage > 80% target"),
        ]),
    ]
    team_boxes = []
    for name, stroke, x, deliverables in teams:
        team, tt = node(x, 290, 380, 90,
            [name, "(2–3 members)"],
            stroke=stroke, bg="#FFFFFF", font=16)
        el += [team, tt]
        team_boxes.append(team)
        # deliverables list
        for i, (mark, desc) in enumerate(deliverables):
            y = 410 + i * 60
            mark_color = PALETTE["dpp"] if mark == "✅" else PALETTE["mcdm"]
            cell, ct_ = node(x, y, 380, 50,
                [f"{mark}  {desc}"],
                stroke=stroke, bg="#FFFFFF", font=13)
            el += [cell, ct_]
        # arrow root → team
        el += arrow(root, team, stroke=stroke)

    # Legend
    legend_y = 410 + 7 * 60 + 30
    el.append(text(80, legend_y,
        "✅  delivered in Iteration 1     "
        "⏳  in progress / planned for Iteration 2",
        size=13, color=PALETTE["neutral"], w=1400))

    el.append(text(40, legend_y + 50,
        "Sources: TA email + TA_WIP_COMPLIANCE_CHECK.md  ·  "
        "fill in member names manually before submission",
        size=11, color=PALETTE["red"], w=1500))

    write("D-18_subteam_responsibilities.excalidraw", el, "D-18 generator")


# ---------- D-20: End-to-End Smoke Test Sequence (3 verified runs) --------

def build_d20():
    random.seed(120)
    el = []
    el.append(text(40, 20,
        "D-20 · End-to-End Smoke Test Sequence (3 verified runs)", size=24))
    el.append(text(40, 55,
        "scripts/smoke_test_e2e.py · concrete demo material for TA review",
        size=13, color=PALETTE["neutral"]))

    actors = [
        ("Orchestrator",         PALETTE["text"]),
        ("EDA / unified.parquet", PALETTE["data"]),
        ("XGBoost SoH",          PALETTE["data"]),
        ("Grade classifier",     PALETTE["data"]),
        ("TOPSIS",               PALETTE["mcdm"]),
        ("schema_mapper",        PALETTE["dpp"]),
        ("DPP JSON + report",    PALETTE["dpp"]),
    ]
    top = 110
    bottom = 1330
    spacing = 220
    xs = [180 + i * spacing for i in range(len(actors))]
    boxes = []
    for x, (name, color) in zip(xs, actors):
        elems, b = _lifeline(x, top, bottom, name, color)
        el += elems
        boxes.append(b)
    # bottom markers
    for i, x in enumerate(xs):
        bm = _base(
            type="rectangle", x=x - 100, y=bottom, width=200, height=40,
            backgroundColor="#FFFFFF", strokeColor=actors[i][1], strokeWidth=2,
            roundness={"type": 3}, fillStyle="solid", roughness=1,
        )
        bt = _base(
            type="text", x=x - 100, y=bottom + 8, width=200, height=24,
            text=actors[i][0], fontSize=13, fontFamily=1,
            textAlign="center", verticalAlign="middle",
            baseline=11, lineHeight=1.25,
            originalText=actors[i][0], containerId=bm["id"],
            strokeColor=actors[i][1],
        )
        bm["boundElements"] = [{"id": bt["id"], "type": "text"}]
        el += [bm, bt]

    # Run blocks: 3 runs
    runs = [
        # (label, color, start_y, sequence of (from_idx, to_idx, label, dashed?))
        ("Run #1 · Zn-coin (chemistry stress test)", PALETTE["red"], 200, [
            (0, 1, "pick BL_ZN-coin_429-1", False),
            (1, 0, "391 cycle rows", True),
            (0, 2, "predict()", False),
            (2, 0, "SoH = 50.5 %", True),
            (0, 3, "grade_from_soh()", False),
            (3, 0, "Grade C", True),
            (0, 4, "run_canonical_topsis(C)", False),
            (4, 0, "rank-1 = Direct Recycling (C* = 0.851)", True),
            (0, 5, "build_dpp(...)", False),
            (5, 6, "dpp_BL_ZN-coin_*.json", False),
            (5, 0, "ok=True · cov 47.7 %", True),
        ]),
        ("Run #2 · BL_SNL NMC (canonical reference)", PALETTE["data"], 580, [
            (0, 1, "pick BL_SNL_18650_NMC", False),
            (1, 0, "1,321 cycle rows", True),
            (0, 2, "predict()", False),
            (2, 0, "SoH = 73.8 %", True),
            (0, 3, "grade_from_soh()", False),
            (3, 0, "Grade B", True),
            (0, 4, "run_canonical_topsis(B)", False),
            (4, 0, "rank-1 = Home/Distributed ESS (C* = 0.731)", True),
            (0, 5, "build_dpp(...)", False),
            (5, 6, "dpp_BL_SNL_18650.json", False),
            (5, 0, "ok=True · cov 48.7 %", True),
        ]),
        ("Run #3 · Synthetic Delhi NMC (best-case)", PALETTE["dpp"], 960, [
            (0, 1, "pick IN_SYNTH_NMC_Delhi", False),
            (1, 0, "100 cycle rows", True),
            (0, 2, "predict()", False),
            (2, 0, "SoH = 98.9 %", True),
            (0, 3, "grade_from_soh()", False),
            (3, 0, "Grade A", True),
            (0, 4, "run_canonical_topsis(A)", False),
            (4, 0, "rank-1 = Grid-scale ESS (C* = 0.782)", True),
            (0, 5, "build_dpp(...)", False),
            (5, 6, "dpp_IN_SYNTH_*.json", False),
            (5, 0, "ok=True · cov 49.9 %", True),
        ]),
    ]
    for run_name, run_color, y_start, msgs in runs:
        # run header banner
        banner = _base(
            type="rectangle", x=80, y=y_start - 20, width=1500, height=30,
            backgroundColor="#FAFAFA",
            strokeColor=run_color, strokeWidth=2,
            roundness={"type": 3}, fillStyle="solid", roughness=0,
        )
        banner_t = _base(
            type="text", x=100, y=y_start - 16, width=1480, height=22,
            text=run_name, fontSize=14, fontFamily=1,
            textAlign="left", verticalAlign="middle",
            baseline=12, lineHeight=1.25,
            originalText=run_name, containerId=banner["id"],
            strokeColor=run_color,
        )
        banner["boundElements"] = [{"id": banner_t["id"], "type": "text"}]
        el += [banner, banner_t]

        y = y_start + 30
        for fi, ti, lbl, dashed in msgs:
            el += _seq_arrow(xs[fi], xs[ti], y, lbl,
                             color=run_color, dashed=dashed)
            y += 30

    el.append(text(40, 1360,
        "Sources: scripts/smoke_test_e2e.py · "
        "results/dpp_output/smoke_summary_*.json · "
        "results/dpp_output/dpp_*.json",
        size=11, color=PALETTE["neutral"], w=1500))

    write("D-20_smoke_test_sequence.excalidraw", el, "D-20 generator")


BUILDERS = {
    "d01": build_d01,
    "d02": build_d02,
    "d03": build_d03,
    "d04": build_d04,
    "d05": build_d05,
    "d06": build_d06,
    "d07": build_d07,
    "d08": build_d08,
    "d09": build_d09,
    "d10": build_d10,
    "d11": build_d11,
    "d12": build_d12,
    "d13": build_d13,
    "d14": build_d14,
    "d15": build_d15,
    "d16": build_d16,
    "d17": build_d17,
    "d18": build_d18,
    "d19": build_d19,
    "d20": build_d20,
}


def main():
    targets = sys.argv[1:] or ["d01"]
    if targets == ["all"]:
        targets = list(BUILDERS)
    for t in targets:
        if t not in BUILDERS:
            print(f"unknown target: {t}; available: {', '.join(BUILDERS)}")
            continue
        BUILDERS[t]()


if __name__ == "__main__":
    main()
