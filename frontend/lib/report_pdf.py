"""Cell report — print-ready PDF in a certificate-style layout.

One entry point — `build_cell_report(...) -> bytes` — returns an A4 portrait
PDF mirroring the live demo's six sections.

Design language (inspired by independent battery certification reports):
  · Vertical accent band on the left of every page with a rotated section
    label, providing strong visual through-line.
  · Bold uppercase title + metadata strip on the cover.
  · Hero block: grade circle + large coloured SoH percentage + sparkline.
  · "Battery checks" pass/fail checklist on the cover.
  · Status callout box (green/amber/red) carrying the routing recommendation.
  · Charts rendered at their target aspect ratio so they never get squashed
    when ReportLab places them in fixed-size slots.
"""
from __future__ import annotations

import datetime as _dt
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from frontend.components.dpp import _category_coverage, _coverage_bar
from frontend.components.mcdm import (
    _decision_matrix_table,
    _ranking_bar,
    _weights_bar,
)
from frontend.components.rul import _trajectory_with_projection
from frontend.components.soh_grade import _bar_comparison, _grade_ladder, _shap_bar
from frontend.components.trajectory import _dqdv_plot, _soh_plot, _vit_plot

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
BAND_W = 15 * mm           # left accent band width
CONTENT_X = BAND_W + 8 * mm  # left edge of content frame
CONTENT_W = PAGE_W - CONTENT_X - 14 * mm
TOP_MARGIN = 18 * mm
BOT_MARGIN = 16 * mm

# Palette — teal accent + clear pass/warn/fail signal colours
TEAL = colors.HexColor("#0E7C7B")
TEAL_DARK = colors.HexColor("#0A5251")
TEAL_LIGHT = colors.HexColor("#E0F2F1")
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#666666")
PASS_GREEN = colors.HexColor("#2E8B57")
PASS_BG = colors.HexColor("#E8F5E9")
WARN_AMBER = colors.HexColor("#C67100")
WARN_BG = colors.HexColor("#FFF4E0")
FAIL_RED = colors.HexColor("#B33B3B")
FAIL_BG = colors.HexColor("#FBE9E9")
HAIRLINE = colors.HexColor("#DDDDDD")
BAND_BG = colors.HexColor("#F5F5F5")

_LINE = 1.2

# Grade colours
GRADE_COLOUR = {
    "A": PASS_GREEN,
    "B": colors.HexColor("#F0AD4E"),
    "C": colors.HexColor("#E07A5F"),
    "D": FAIL_RED,
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=base["BodyText"],
        fontName="Helvetica", fontSize=9, leading=9 * _LINE,
        textColor=INK, alignment=TA_LEFT, spaceAfter=4)
    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=body, fontName="Helvetica-Bold",
            fontSize=22, leading=22, spaceAfter=2, textColor=INK),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle", parent=body, fontName="Helvetica",
            fontSize=10, leading=11, textColor=MUTED, spaceAfter=12),
        "section": ParagraphStyle(
            "section", parent=body, fontName="Helvetica-Bold",
            fontSize=15, leading=15 * _LINE, textColor=INK,
            spaceBefore=4, spaceAfter=6),
        "subsection": ParagraphStyle(
            "subsection", parent=body, fontName="Helvetica-Bold",
            fontSize=10.5, leading=10.5 * _LINE,
            textColor=TEAL_DARK, spaceBefore=6, spaceAfter=3),
        "body": body,
        "muted": ParagraphStyle(
            "muted", parent=body, textColor=MUTED, fontSize=8.5,
            leading=8.5 * _LINE),
        "hero_grade_letter": ParagraphStyle(
            "hero_grade_letter", parent=body, fontName="Helvetica-Bold",
            fontSize=56, leading=56, alignment=TA_CENTER, textColor=colors.white),
        "hero_soh_value": ParagraphStyle(
            "hero_soh_value", parent=body, fontName="Helvetica-Bold",
            fontSize=44, leading=44, alignment=TA_CENTER),
        "hero_soh_label": ParagraphStyle(
            "hero_soh_label", parent=body, fontName="Helvetica",
            fontSize=8, leading=10, alignment=TA_CENTER,
            textColor=MUTED, spaceAfter=2),
        "callout_good": ParagraphStyle(
            "callout_good", parent=body, fontName="Helvetica-Bold",
            fontSize=10.5, alignment=TA_CENTER,
            textColor=PASS_GREEN, leading=12,
            backColor=PASS_BG, borderPadding=10, spaceBefore=4, spaceAfter=4),
        "callout_warn": ParagraphStyle(
            "callout_warn", parent=body, fontName="Helvetica-Bold",
            fontSize=10.5, alignment=TA_CENTER,
            textColor=WARN_AMBER, leading=12,
            backColor=WARN_BG, borderPadding=10, spaceBefore=4, spaceAfter=4),
        "callout_bad": ParagraphStyle(
            "callout_bad", parent=body, fontName="Helvetica-Bold",
            fontSize=10.5, alignment=TA_CENTER,
            textColor=FAIL_RED, leading=12,
            backColor=FAIL_BG, borderPadding=10, spaceBefore=4, spaceAfter=4),
        "story_callout": ParagraphStyle(
            "story_callout", parent=body, fontSize=8.5, leading=8.5 * _LINE,
            backColor=colors.HexColor("#FFFBE6"),
            borderPadding=8, spaceBefore=4, spaceAfter=6),
    }


# ---------------------------------------------------------------------------
# Chart → PNG with aspect-matched rendering
# ---------------------------------------------------------------------------
def _fig_to_image(fig, *, width_mm: float, height_mm: float,
                  px_width: int = 1100) -> Image:
    """Render Plotly figure at the placement's exact aspect ratio — no squashing."""
    aspect = height_mm / width_mm
    px_height = max(int(px_width * aspect), 200)
    png = fig.to_image(format="png", width=px_width, height=px_height, scale=1.5)
    img = Image(BytesIO(png), width=width_mm * mm, height=height_mm * mm)
    img.hAlign = "CENTER"
    return img


def _sparkline_png(cell_df: pd.DataFrame, cycle_n: int) -> Image:
    """Tiny SoH-vs-cycle sparkline for the cover hero block."""
    soh_pct = cell_df["soh"].clip(lower=0, upper=1.5) * 100
    fig = go.Figure(go.Scatter(
        x=cell_df["cycle"], y=soh_pct,
        mode="lines", line=dict(color="#0A5251", width=2),
        hoverinfo="skip",
    ))
    fig.add_hline(y=80, line=dict(color="#888888", dash="dot", width=1))
    fig.add_vline(x=cycle_n, line=dict(color="#0E7C7B", width=1.5))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False, height=200, width=600,
    )
    return _fig_to_image(fig, width_mm=58, height_mm=20, px_width=600)


# ---------------------------------------------------------------------------
# Per-page background callbacks (vertical band + section label)
# ---------------------------------------------------------------------------
def _draw_band(canvas, doc, section_label: str, cell_id: str,
               generation_date: str):
    """Draw the teal left band, rotated section label, header/footer chrome."""
    canvas.saveState()
    # Band
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, BAND_W, PAGE_H, fill=1, stroke=0)
    # Rotated section label
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.translate(BAND_W / 2, PAGE_H / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, section_label.upper())
    canvas.restoreState()
    # Header chrome (top-right)
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(PAGE_W - 14 * mm, PAGE_H - 10 * mm, cell_id)
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.3)
    canvas.line(CONTENT_X, PAGE_H - 12 * mm, PAGE_W - 14 * mm, PAGE_H - 12 * mm)
    # Footer chrome
    canvas.drawString(CONTENT_X, 10 * mm, generation_date)
    canvas.drawRightString(PAGE_W - 14 * mm, 10 * mm, f"Page {doc.page}")
    canvas.line(CONTENT_X, 12 * mm, PAGE_W - 14 * mm, 12 * mm)
    canvas.restoreState()


def _make_band_callback(section_label: str, cell_id: str, generation_date: str):
    def _cb(canvas, doc):
        _draw_band(canvas, doc, section_label, cell_id, generation_date)
    return _cb


# ---------------------------------------------------------------------------
# Cover-page hero blocks
# ---------------------------------------------------------------------------
def _grade_disc(grade: str) -> Table:
    """A round-ish coloured block displaying the grade letter."""
    colour = GRADE_COLOUR.get(grade, MUTED)
    cell = Paragraph(f"<b>{grade}</b>",
                     ParagraphStyle("g", fontName="Helvetica-Bold",
                                    fontSize=56, leading=56,
                                    alignment=TA_CENTER, textColor=colors.white))
    t = Table([[cell]], colWidths=[40 * mm], rowHeights=[40 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colour),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return t


def _soh_hero(soh_pct: float, observed_pct: float | None) -> Table:
    """Centre block of the hero — big SoH percent + tiny 'observed' line."""
    grade = _grade_from_soh(soh_pct)
    fg = GRADE_COLOUR.get(grade, MUTED)
    big = Paragraph(f"<b>{soh_pct:.1f}%</b>",
                    ParagraphStyle("h_pct", fontName="Helvetica-Bold",
                                   fontSize=42, leading=42,
                                   alignment=TA_CENTER, textColor=fg))
    label = Paragraph("STATE OF HEALTH",
                      ParagraphStyle("h_lbl", fontName="Helvetica",
                                     fontSize=8, leading=10,
                                     alignment=TA_CENTER, textColor=MUTED))
    sub = ""
    if observed_pct is not None and not np.isnan(observed_pct):
        sub = (f"observed {observed_pct:.2f}% · "
               f"prediction error {abs(soh_pct - observed_pct):.2f} pp")
    sub_p = Paragraph(sub,
                      ParagraphStyle("h_sub", fontName="Helvetica",
                                     fontSize=7.5, leading=9,
                                     alignment=TA_CENTER, textColor=MUTED))
    t = Table([[label], [big], [sub_p]], colWidths=[60 * mm])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t


def _checks_table(anomaly_out, rul_out, dpp_out, mcdm_out, soh_out) -> Table:
    """Sample-1-style 'Battery checks' pass/fail list."""
    iso_ok = not anomaly_out["iso"]["flagged"]
    vae_ok = not anomaly_out["vae"]["flagged"]
    rul_delta = abs(rul_out["rul_xgb"] - rul_out["rul_tcn"])
    rul_ok = rul_delta < 200  # good / modest threshold
    spec_delta = abs(soh_out["global_pred"] - soh_out["specialist_pred"])
    health_ok = spec_delta < 3.0
    dpp_ok = dpp_out["validated"]
    mcdm_ok = mcdm_out["closeness"] >= 0.50

    def _row(label, ok, detail):
        mark = "✓" if ok else "✗"
        mark_colour = PASS_GREEN if ok else FAIL_RED
        return [
            Paragraph(label,
                      ParagraphStyle("c_lbl", fontName="Helvetica",
                                     fontSize=9, textColor=INK, leading=11)),
            Paragraph(detail,
                      ParagraphStyle("c_det", fontName="Helvetica",
                                     fontSize=8, textColor=MUTED, leading=10)),
            Paragraph(f"<font color='{mark_colour.hexval()}'><b>{mark}</b></font>",
                      ParagraphStyle("c_mk", fontName="Helvetica-Bold",
                                     fontSize=13, alignment=TA_CENTER,
                                     leading=14)),
        ]

    rows = [
        _row("Data-quality gate · tree-based check", iso_ok,
             "Outlier detection over the cell's features"),
        _row("Data-quality gate · neural-network check", vae_ok,
             "Reconstruction-error check vs healthy training cells"),
        _row("Health model agreement", health_ok,
             f"Two health models spread {spec_delta:.2f} pp"),
        _row("Remaining-life model agreement", rul_ok,
             f"Two remaining-life models spread {rul_delta:,.0f} cycles"),
        _row("Routing decision confidence", mcdm_ok,
             f"Closeness coefficient {mcdm_out['closeness']:.3f}"),
        _row("Digital Product Passport validation", dpp_ok,
             "EU 2023/1542 · GBA v1.2 · BWMR 2022/24/25"),
    ]
    t = Table(rows, colWidths=[60 * mm, 70 * mm, 10 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HAIRLINE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _status_callout(*, style, text: str) -> Paragraph:
    return Paragraph(text, style)


def _grade_from_soh(soh_pct: float) -> str:
    if soh_pct > 80: return "A"
    if soh_pct > 60: return "B"
    if soh_pct > 40: return "C"
    return "D"


def _verdict_style(st, anomaly_out):
    iso_flag = anomaly_out["iso"]["flagged"]
    vae_flag = anomaly_out["vae"]["flagged"]
    if iso_flag and vae_flag: return st["callout_bad"]
    if iso_flag or vae_flag:  return st["callout_warn"]
    return st["callout_good"]


# ---------------------------------------------------------------------------
# Section helpers reused on every page
# ---------------------------------------------------------------------------
def _kv_table(rows: list[tuple[str, str]], col_widths=(48 * mm, 95 * mm)) -> Table:
    sty = _styles()["body"]
    data = [[Paragraph(f"<b>{k}</b>", sty), Paragraph(v, sty)] for k, v in rows]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, HAIRLINE),
    ]))
    return t


def _shaded_decision_matrix(grade: str) -> Table:
    from src.mcdm.topsis import (CANONICAL_ALTERNATIVES, CANONICAL_CRITERIA,
                                 CANONICAL_TYPES)
    df = _decision_matrix_table(grade)
    header = ["Route"] + CANONICAL_CRITERIA
    rows = [header]
    for alt in CANONICAL_ALTERNATIVES:
        rows.append([alt] + [f"{df.loc[alt, c]:.1f}" for c in CANONICAL_CRITERIA])
    t = Table(rows,
              colWidths=[35 * mm] + [(CONTENT_W / mm - 35) / 6 * mm] * 6)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BAND_BG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MUTED),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, HAIRLINE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]
    for c_idx, ctype in enumerate(CANONICAL_TYPES, start=1):
        col_vals = [df.iloc[r, c_idx - 1] for r in range(len(CANONICAL_ALTERNATIVES))]
        lo, hi = min(col_vals), max(col_vals)
        for r_idx, v in enumerate(col_vals, start=1):
            frac = (v - lo) / (hi - lo) if hi > lo else 0.0
            if ctype == "benefit":
                shade = colors.Color(0.92 - 0.30 * frac, 0.97 - 0.05 * frac,
                                     0.92 - 0.30 * frac)
            else:
                shade = colors.Color(0.97 - 0.05 * frac, 0.92 - 0.30 * frac,
                                     0.92 - 0.30 * frac)
            style_cmds.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), shade))
    t.setStyle(TableStyle(style_cmds))
    return t


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _build_cover(story, st, *, cell_meta, cycle_n, regime, soh_out, rul_out,
                 mcdm_out, anomaly_out, dpp_out, cell_df, cell_story,
                 report_id, generation_date, generator_name):

    story.append(Paragraph(
        "INDEPENDENT EV BATTERY CELL CERTIFICATE",
        ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
                       leading=20, textColor=TEAL_DARK, spaceAfter=3)))
    story.append(Paragraph(
        "ML-MCDM assessment · Hybrid framework for End-of-Life routing",
        st["cover_subtitle"]))

    # Metadata strip
    meta_rows = [
        [Paragraph(f"<b>Report ID</b><br/>{report_id}", st["body"]),
         Paragraph(f"<b>Generated</b><br/>{generation_date}", st["body"]),
         Paragraph(f"<b>Regulatory regime</b><br/>{regime}", st["body"])],
    ]
    meta_t = Table(meta_rows,
                   colWidths=[CONTENT_W / 3] * 3)
    meta_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND_BG),
        ("BOX", (0, 0), (-1, -1), 0.3, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 4 * mm))

    # Hero block — Grade | SoH | Sparkline (3 columns)
    grade = soh_out["grade"]
    soh_pct = soh_out["specialist_pred"]
    observed = soh_out.get("observed")
    grade_disc = _grade_disc(grade)
    soh_centre = _soh_hero(soh_pct, observed)
    spark_label = Paragraph("SoH TRAJECTORY",
                            ParagraphStyle("sl", fontName="Helvetica",
                                           fontSize=8, leading=10,
                                           alignment=TA_CENTER, textColor=MUTED))
    sparkline = _sparkline_png(cell_df, cycle_n)
    spark_block = Table([[spark_label], [sparkline]], colWidths=[60 * mm])
    spark_block.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    hero = Table([[grade_disc, soh_centre, spark_block]],
                 colWidths=[42 * mm, (CONTENT_W - 42 * mm - 60 * mm),
                            60 * mm])
    hero.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(hero)
    story.append(Spacer(1, 4 * mm))

    # Cell identification strip
    cell_strip_rows = [
        [Paragraph("<b>Cell ID</b>", st["body"]),
         Paragraph(cell_meta["battery_id"], st["body"]),
         Paragraph("<b>Chemistry</b>", st["body"]),
         Paragraph(cell_meta["chemistry"], st["body"])],
        [Paragraph("<b>Form factor</b>", st["body"]),
         Paragraph(str(cell_meta.get("form_factor") or "n/a"), st["body"]),
         Paragraph("<b>Nominal capacity</b>", st["body"]),
         Paragraph(f"{cell_meta['nominal_Ah']} Ah", st["body"])],
        [Paragraph("<b>Source</b>", st["body"]),
         Paragraph(cell_meta["source"], st["body"]),
         Paragraph("<b>Predicted at cycle</b>", st["body"]),
         Paragraph(f"{cycle_n:,} of {cell_meta['cycle_max']:,}", st["body"])],
    ]
    strip = Table(cell_strip_rows,
                  colWidths=[28 * mm, (CONTENT_W / 2 - 28 * mm),
                             34 * mm, (CONTENT_W / 2 - 34 * mm)])
    strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.3, HAIRLINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(strip)
    story.append(Spacer(1, 4 * mm))

    # Battery checks
    story.append(Paragraph("Battery checks", st["subsection"]))
    story.append(_checks_table(anomaly_out, rul_out, dpp_out, mcdm_out, soh_out))
    story.append(Spacer(1, 4 * mm))

    # Status callout
    verdict_style = _verdict_style(st, anomaly_out)
    top_route = mcdm_out["top_alternative"]
    closeness = mcdm_out["closeness"]
    iso_flag = anomaly_out["iso"]["flagged"]
    vae_flag = anomaly_out["vae"]["flagged"]
    if iso_flag and vae_flag:
        prefix = ("STATUS: ANOMALY FLAGS — RECOMMENDATION WITH CAUTION<br/>")
    elif iso_flag or vae_flag:
        prefix = "STATUS: PARTIAL ANOMALY FLAG — BORDERLINE CASE<br/>"
    else:
        prefix = "STATUS: ALL CHECKS PASS — RECOMMENDATION CONFIDENT<br/>"
    callout_text = (
        f"{prefix}"
        f"<font size='9'>This grade-<b>{grade}</b> {cell_meta['chemistry']} cell "
        f"is recommended for <b>{top_route}</b> under the <i>{regime}</i> regime "
        f"(TOPSIS closeness {closeness:.3f}).</font>"
    )
    story.append(_status_callout(style=verdict_style, text=callout_text))

    if cell_story:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            f"<b>What to look for —</b> {cell_story['headline']}<br/>"
            f"<font color='#444444'>{cell_story['walkthrough']}</font>",
            st["story_callout"]))


def _build_section1_trajectory(story, st, *, cell_df, cycle_n):
    story.append(NextPageTemplate("trajectory"))
    story.append(PageBreak())
    story.append(Paragraph("1 · Cell trajectory", st["section"]))
    story.append(Paragraph(
        "How this cell has aged across its observed cycles. The vertical "
        "dotted marker on each chart shows the cycle the predictions below "
        "were taken at.",
        st["muted"]))
    story.append(_fig_to_image(_soh_plot(cell_df, cycle_n),
                               width_mm=CONTENT_W / mm, height_mm=65))
    vit = _vit_plot(cell_df, cycle_n)
    if vit is not None:
        story.append(_fig_to_image(vit, width_mm=CONTENT_W / mm, height_mm=95))
    else:
        story.append(Paragraph(
            "<i>Voltage / current / temperature aggregates show no informative "
            "variation for this cell — the source dataset only logs "
            "protocol-fixed values per cycle.</i>",
            st["muted"]))
    dqdv = _dqdv_plot(cell_df, cycle_n)
    if dqdv is not None:
        # Override the chart title to fit better in print context
        dqdv.update_layout(title="dQ/dV peak voltage vs cycle")
        story.append(_fig_to_image(dqdv, width_mm=CONTENT_W / mm, height_mm=60))


def _build_section2_anomaly(story, st, *, anomaly_out):
    story.append(NextPageTemplate("anomaly"))
    story.append(PageBreak())
    story.append(Paragraph("2 · Anomaly gate", st["section"]))
    story.append(Paragraph(
        "Two unsupervised models quality-check the cell before any "
        "predictions are made. Each votes independently on whether this cell "
        "looks like the ones the framework was trained on.",
        st["muted"]))
    story.append(Spacer(1, 4 * mm))

    iso = anomaly_out["iso"]
    vae = anomaly_out["vae"]
    def _badge(flagged):
        col = FAIL_RED if flagged else PASS_GREEN
        word = "FLAGGED" if flagged else "PASS"
        return Paragraph(
            f"<font color='{col.hexval()}'><b>{word}</b></font>",
            ParagraphStyle("b", fontName="Helvetica-Bold", fontSize=10,
                           alignment=TA_CENTER, leading=12))
    rows = [
        [Paragraph("<b>Check</b>", st["body"]),
         Paragraph("<b>Score</b>", st["body"]),
         Paragraph("<b>Threshold</b>", st["body"]),
         Paragraph("<b>Verdict</b>", st["body"])],
        [Paragraph("Tree-based outlier detection", st["body"]),
         Paragraph(f"{iso['score']:.4f}", st["body"]),
         Paragraph(f"{iso['threshold']:.4f}", st["body"]),
         _badge(iso["flagged"])],
        [Paragraph("Neural-network reconstruction error", st["body"]),
         Paragraph(f"{vae['score']:.4f}", st["body"]),
         Paragraph(f"{vae['threshold']:.4f}", st["body"]),
         _badge(vae["flagged"])],
    ]
    t = Table(rows, colWidths=[(CONTENT_W / mm - 28 - 28 - 28) * mm,
                               28 * mm, 28 * mm, 28 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MUTED),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Spacer(1, 5 * mm))
    style = _verdict_style(st, anomaly_out)
    iso_flag = anomaly_out["iso"]["flagged"]
    vae_flag = anomaly_out["vae"]["flagged"]
    if iso_flag and vae_flag:
        msg = ("Both models flag this cell. Predictions on subsequent pages "
               "should be interpreted with caution — the cell is unusual "
               "relative to the framework's training distribution.")
    elif iso_flag or vae_flag:
        which = "Isolation Forest" if iso_flag else "VAE"
        msg = (f"Only {which} flags this cell. This is a borderline case — "
               "downstream predictions may still be usable with caveats.")
    else:
        msg = ("Both models pass. Subsequent predictions carry normal "
               "confidence.")
    story.append(Paragraph(msg, style))


def _build_section3_health(story, st, *, soh_out, chem_stats):
    story.append(NextPageTemplate("health"))
    story.append(PageBreak())
    story.append(Paragraph("3 · Health assessment", st["section"]))
    story.append(Paragraph(
        "Two parallel predictions of this cell's State of Health — a "
        "general-purpose model trained on every chemistry, and a specialist "
        "trained only on cells of this cell's chemistry.",
        st["muted"]))
    story.append(Spacer(1, 3 * mm))

    story.append(_fig_to_image(
        _bar_comparison(soh_out.get("observed"),
                        soh_out["global_pred"],
                        soh_out["specialist_pred"]),
        width_mm=CONTENT_W / mm, height_mm=55))

    chemistry = soh_out["chemistry"]
    stats = chem_stats.get(chemistry)
    if stats:
        story.append(Paragraph(
            f"Cell tagged as <b>{chemistry}</b>. A specialist model trained "
            f"only on {chemistry} cells ({stats['n_train']:,} training cycles, "
            f"test grade-accuracy {stats['test_grade_acc']:.2%}) predicts "
            f"SoH = <b>{soh_out['specialist_pred']:.2f}%</b>, "
            f"grade <b>{soh_out['grade']}</b>.",
            st["body"]))

    story.append(Spacer(1, 3 * mm))
    story.append(_fig_to_image(
        _grade_ladder(soh_out["specialist_pred"], soh_out["grade"]),
        width_mm=CONTENT_W / mm, height_mm=30))

    shap_vals = soh_out.get("shap_values")
    feature_names = soh_out.get("feature_names")
    if shap_vals is not None and feature_names is not None:
        story.append(Spacer(1, 3 * mm))
        story.append(_fig_to_image(
            _shap_bar(shap_vals, feature_names),
            width_mm=CONTENT_W / mm, height_mm=75))


def _build_section4_rul(story, st, *, cell_df, cycle_n, rul_out):
    story.append(NextPageTemplate("rul"))
    story.append(PageBreak())
    story.append(Paragraph("4 · Remaining useful life", st["section"]))
    story.append(Paragraph(
        "How many cycles until End-of-Life (SoH = 80 %). Two models — one "
        "classical machine-learning, one deep learning — predict in parallel.",
        st["muted"]))
    story.append(Spacer(1, 4 * mm))

    rows = [
        [Paragraph("<b>Model</b>", st["body"]),
         Paragraph("<b>Predicted remaining cycles</b>", st["body"])],
        [Paragraph("Classical machine-learning model", st["body"]),
         Paragraph(f"<b>{rul_out['rul_xgb']:,.0f} cycles</b>", st["body"])],
        [Paragraph("Deep-learning model", st["body"]),
         Paragraph(f"<b>{rul_out['rul_tcn']:,.0f} cycles</b>", st["body"])],
    ]
    t = Table(rows, colWidths=[(CONTENT_W / mm - 50) * mm, 50 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MUTED),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    fig = _trajectory_with_projection(
        cell_df, cycle_n,
        rul_out.get("current_soh_pct"),
        rul_out["rul_xgb"], rul_out["rul_tcn"],
    )
    story.append(_fig_to_image(fig, width_mm=CONTENT_W / mm, height_mm=85))

    story.append(Spacer(1, 3 * mm))
    delta = rul_out["rul_xgb"] - rul_out["rul_tcn"]
    agree = ("Good" if abs(delta) < 50 else
             "Modest" if abs(delta) < 200 else "Poor")
    story.append(Paragraph(
        f"<b>Model agreement: {agree}.</b> Distance between predictions "
        f"{abs(delta):,.0f} cycles. Smaller gaps mean both model families "
        "converge on the same answer.",
        st["body"]))


def _build_section5_mcdm(story, st, *, grade, mcdm_out, scenarios):
    from src.mcdm.topsis import (CANONICAL_ALTERNATIVES, CANONICAL_CRITERIA,
                                 CANONICAL_TYPES, build_canonical_decision_matrix,
                                 topsis_rank)
    story.append(NextPageTemplate("routing"))
    story.append(PageBreak())
    story.append(Paragraph("5 · MCDM routing decision", st["section"]))
    story.append(Paragraph(
        "TOPSIS ranks the four end-of-life routes using six weighted "
        "criteria. Weights come from the regulatory regime selected; the "
        "closer to 1.0, the more strongly that route is recommended.",
        st["muted"]))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"<b>Decision matrix for grade {grade}</b> "
        "(green = better on benefit criteria, red = worse on cost criteria)",
        st["body"]))
    story.append(_shaded_decision_matrix(grade))
    story.append(Spacer(1, 4 * mm))

    weights_arr = np.asarray([mcdm_out["weights"][c] for c in CANONICAL_CRITERIA])
    story.append(_fig_to_image(
        _weights_bar(weights_arr, mcdm_out["regime"]),
        width_mm=CONTENT_W / mm, height_mm=60))
    story.append(_fig_to_image(
        _ranking_bar(pd.DataFrame(mcdm_out["ranked"])),
        width_mm=CONTENT_W / mm, height_mm=55))

    story.append(Paragraph(
        f"<b>Recommendation.</b> Under <b>{mcdm_out['regime']}</b> weights "
        f"for a grade-<b>{grade}</b> cell, TOPSIS picks "
        f"<b>{mcdm_out['top_alternative']}</b> with closeness coefficient "
        f"{mcdm_out['closeness']:.4f}.",
        st["body"]))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "<b>RQ2 — recommendation under each regulatory regime</b>",
        st["body"]))
    rq2_rows = [[Paragraph(h, st["body"]) for h in
                 ("Regime", "Top recommendation", "Closeness", "Second choice")]]
    for r_name, r_weights in scenarios.items():
        dm = build_canonical_decision_matrix(grade)
        res = topsis_rank(dm, r_weights, CANONICAL_TYPES)
        order = np.argsort(res["ranking"])
        rq2_rows.append([
            Paragraph(r_name, st["body"]),
            Paragraph(CANONICAL_ALTERNATIVES[order[0]], st["body"]),
            Paragraph(f"{res['closeness'][order[0]]:.4f}", st["body"]),
            Paragraph(CANONICAL_ALTERNATIVES[order[1]], st["body"]),
        ])
    rq2_table = Table(rq2_rows,
                      colWidths=[35 * mm, 50 * mm, 22 * mm,
                                 (CONTENT_W / mm - 35 - 50 - 22) * mm])
    rq2_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MUTED),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rq2_table)


def _build_appendix_methodology(story, st):
    """Page 8 — single-page methodology reference for the technically-minded reader."""
    story.append(NextPageTemplate("methodology"))
    story.append(PageBreak())
    story.append(Paragraph("Appendix · Methodology", st["section"]))
    story.append(Paragraph(
        "The predictions, scores, and routing recommendation in this report "
        "are produced by an ensemble of seven machine-learning and "
        "decision-theory components. Each is summarised below for the reader "
        "who wants to verify the framework's machinery.",
        st["muted"]))
    story.append(Spacer(1, 4 * mm))

    items = [
        ("Tree-based anomaly check",
         "Isolation Forest (Liu et al. 2008). 200 random tree splits over the "
         "cell's feature vector; cells that get isolated in few splits are "
         "flagged as outliers. Threshold tuned to flag ~5 % of training cells."),
        ("Neural-network anomaly check",
         "Variational Autoencoder — a small neural network that compresses "
         "the cell's features to a latent representation and reconstructs them. "
         "Cells whose reconstruction error exceeds the training-time threshold "
         "are flagged as out-of-distribution."),
        ("Global health predictor (SoH)",
         "XGBoost gradient-boosted regressor over 32 audited features "
         "(capacity-derived columns deliberately excluded to prevent label "
         "leakage). Predicts State of Health to within ~2.4 percentage points "
         "on held-out test cells."),
        ("Chemistry-specialist health predictor (SoH)",
         "Seven separate XGBoost models, one per chemistry (NMC · LFP · NCA · "
         "LCO · Zn-ion · Na-ion · other). At inference time each cell is "
         "dispatched to its specialist. Lifts grade-accuracy on under-represented "
         "chemistries by up to 5 percentage points versus the global model alone."),
        ("Classical remaining-life predictor (RUL)",
         "XGBoost regressor predicting cycles until the cell drops below the "
         "End-of-Life threshold (SoH = 80 %). Trained on cells that actually "
         "reached EoL during observation. Accurate to within ~2 % of the "
         "prediction range on held-out test cells."),
        ("Deep-learning remaining-life predictor (RUL)",
         "Temporal Convolutional Network (Bai et al. 2018) reading the cell's "
         "last 60 cycles as a sequence rather than a single-cycle snapshot. "
         "Slightly less accurate than the classical model on average but a "
         "useful independent second opinion."),
        ("Multi-criteria routing engine",
         "Fuzzy Best-Worst Method aggregates criterion weights across 12+ "
         "published Multi-Criteria Decision-Making studies; TOPSIS "
         "(Technique for Order Preference by Similarity to Ideal Solution) "
         "then ranks the four End-of-Life routes against six criteria — "
         "State of Health, residual Value, Carbon footprint, Compliance fit, "
         "Safety risk, and Extended Producer Responsibility return. Five "
         "regulatory regimes modulate the weight vector to operationalise the "
         "regulatory-sensitivity research question."),
        ("Digital Product Passport schema",
         "JSON-schema reconciliation of EU Regulation 2023/1542 Annex XIII, "
         "the Global Battery Alliance Battery Pass Data Model v1.2.0, and "
         "India BWMR 2022 (with 2024 and 2025 amendments). Each cell's "
         "predictions are written into the unified schema and validated "
         "before download."),
    ]
    rows = []
    for name, desc in items:
        rows.append([
            Paragraph(f"<b>{name}</b>", st["body"]),
            Paragraph(desc, st["body"]),
        ])
    t = Table(rows, colWidths=[55 * mm, (CONTENT_W / mm - 55) * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, HAIRLINE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)


def _build_section6_dpp(story, st, *, dpp):
    story.append(NextPageTemplate("passport"))
    story.append(PageBreak())
    story.append(Paragraph("6 · Digital Product Passport", st["section"]))
    story.append(Paragraph(
        "All upstream predictions assembled into a structured passport, "
        "schema-aligned with EU Regulation 2023/1542 Annex XIII, the GBA "
        "Battery Pass data model, and India BWMR 2022 (with 2024–2025 "
        "amendments).",
        st["muted"]))
    story.append(Spacer(1, 3 * mm))

    iden = dpp["identity"]
    chem = dpp["chemistry_and_composition"]
    perf = dpp["performance_and_durability"]
    soh = dpp["state_of_health"]
    circ = dpp["circularity_and_eol"]
    epr = circ.get("epr_compliance", {})
    carbon = dpp["carbon_footprint"]

    story.append(Paragraph("Identity & origin", st["subsection"]))
    story.append(_kv_table([
        ("DPP ID", iden["dpp_id"]),
        ("Battery ID", iden["battery_id"]),
        ("Product status", iden["product_status"]),
        ("Form factor", iden["form_factor"] or "n/a"),
        ("Cathode chemistry", chem["cathode_chemistry"]),
        ("Critical raw materials",
         ", ".join(chem["critical_raw_materials_present"]) or "none"),
    ]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Performance & health", st["subsection"]))
    rul_str = (f"{int(soh['rul_remaining_cycles']):,} cycles"
               if soh["rul_remaining_cycles"] is not None else "n/a")
    story.append(_kv_table([
        ("Rated capacity", f"{perf['rated_capacity_Ah']} Ah"),
        ("Voltage window",
         f"{perf['voltage_min_V']:.2f} – {perf['voltage_max_V']:.2f} V"),
        ("Cycles completed", f"{soh['cycles_completed']:,}"),
        ("State of health", f"{soh['soh_percent']:.2f} %"),
        ("Remaining useful life", rul_str),
        ("Expected lifetime", f"{perf['expected_lifetime_cycles']:,} cycles"),
    ]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Sustainability", st["subsection"]))
    cf = carbon["carbon_footprint_kgCO2eq_per_kWh"]
    perf_class = carbon["performance_class"] or "n/a"
    status = carbon["calculation_status"]
    cf_str = f"{cf:.1f} kg CO2eq / kWh" if cf is not None else "n/a"
    story.append(_kv_table([
        ("Carbon footprint", cf_str),
        ("Performance class", perf_class),
        ("Provenance", status),
    ]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Circularity, EoL route & supply chain",
                           st["subsection"]))
    story.append(_kv_table([
        ("Grade", circ["grade"]),
        ("Recommended route", circ["recommended_route"]),
        ("Route score (closeness)",
         f"{circ['route_score']:.4f}" if circ["route_score"] is not None else "n/a"),
        ("Ranking method", circ["route_ranking_method"]),
        ("Take-back route", epr.get("take_back_route", "n/a")),
        ("BWMR recovery target",
         f"{epr.get('recovery_target_pct', 'n/a')} % "
         f"({epr.get('recovery_target_fy', '')})"),
    ]))

    story.append(Spacer(1, 3 * mm))
    cov_df = _category_coverage(dpp)
    story.append(_fig_to_image(_coverage_bar(cov_df),
                               width_mm=CONTENT_W / mm, height_mm=55))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<i>The full machine-readable DPP is available as a separate JSON "
        "download from the demo.</i>",
        st["muted"]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
SECTION_LABELS = [
    ("cover", "CELL REPORT"),
    ("trajectory", "TRAJECTORY"),
    ("anomaly", "ANOMALY"),
    ("health", "HEALTH"),
    ("rul", "REMAINING LIFE"),
    ("routing", "ROUTING"),
    ("passport", "PASSPORT"),
    ("methodology", "METHODOLOGY"),
]


def build_cell_report(
    *,
    selection,
    cell_df: pd.DataFrame,
    anomaly_out: dict,
    soh_out: dict,
    rul_out: dict,
    mcdm_out: dict,
    dpp_out: dict,
    cell_story: dict | None,
    chem_stats: dict,
    scenarios: dict,
    generator_name: str = "Hybrid ML-MCDM Demo Framework (live)",
) -> bytes:
    """Build and return a full PDF cell report as bytes."""
    buf = BytesIO()
    cell_id = selection.cell_meta["battery_id"]
    cycle_n = selection.cycle_n
    regime = selection.regime
    cell_meta = selection.cell_meta

    now = _dt.datetime.now()
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}-{now.strftime('%Y%m%d')}"
    generation_date = now.strftime("%Y-%m-%d %H:%M")

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=CONTENT_X, rightMargin=14 * mm,
        topMargin=TOP_MARGIN, bottomMargin=BOT_MARGIN,
        title=f"Cell Report — {cell_id}",
        author=generator_name,
    )
    frame = Frame(CONTENT_X, BOT_MARGIN, CONTENT_W,
                  PAGE_H - TOP_MARGIN - BOT_MARGIN, showBoundary=0,
                  leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    templates = [
        PageTemplate(id=tpl_id, frames=frame,
                     onPage=_make_band_callback(label, cell_id, generation_date))
        for tpl_id, label in SECTION_LABELS
    ]
    doc.addPageTemplates(templates)

    st = _styles()
    story: list = []

    _build_cover(story, st,
                 cell_meta=cell_meta, cycle_n=cycle_n, regime=regime,
                 soh_out=soh_out, rul_out=rul_out, mcdm_out=mcdm_out,
                 anomaly_out=anomaly_out, dpp_out=dpp_out,
                 cell_df=cell_df, cell_story=cell_story,
                 report_id=report_id, generation_date=generation_date,
                 generator_name=generator_name)
    _build_section1_trajectory(story, st, cell_df=cell_df, cycle_n=cycle_n)
    _build_section2_anomaly(story, st, anomaly_out=anomaly_out)
    _build_section3_health(story, st, soh_out=soh_out, chem_stats=chem_stats)
    _build_section4_rul(story, st, cell_df=cell_df, cycle_n=cycle_n, rul_out=rul_out)
    _build_section5_mcdm(story, st, grade=soh_out["grade"],
                         mcdm_out=mcdm_out, scenarios=scenarios)
    _build_section6_dpp(story, st, dpp=dpp_out["dpp"])
    _build_appendix_methodology(story, st)

    doc.build(story)
    return buf.getvalue()
