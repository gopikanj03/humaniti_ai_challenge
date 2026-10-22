"""Downloadable PDF report built from a pipeline result, written for readers with no technical background.

It mirrors the results page: what the result means, how sure it is, a short summary, why the class got this
result, what went well, what to improve, and each area rated in words with its examples. No metrics or
technical details (those stay in the app). Never the transcript; children appear only as pseudonyms. The source
file name is not included (it may contain names).
"""
from __future__ import annotations

import re
from datetime import date

from . import explain
from .pipeline import PipelineResult, summarize
from .scoring import load_rubric


def report_data(res: PipelineResult) -> dict:
    r, a, s = res.result, res.assessment, summarize(res)
    rubric = load_rubric()
    specs = {d["id"]: d for d in rubric["dimensions"]}
    sure = explain.certainty(r.confidence)
    return {
        "title": "ClassCritique Report",
        "generated": date.today().strftime("%d/%m/%Y"),  # Indian style: day/month/year
        "recording": {"duration_min": s["duration_min"], "language": s["language"] or "unknown"},
        "result": {"label": r.label, "score": r.score, "score_label": r.score_label,
                   "meaning": explain.label_meaning(r), "quality_issue": r.quality_issue,
                   "not_a_class": r.not_a_class,
                   "certainty": sure, "certainty_meaning": explain.CERTAINTY_MEANING[sure],
                   "certainty_reasons": [explain.plain_reason(x) for x in r.confidence_reasons],
                   "why": explain.main_reasons(r, a, rubric), "score_line": explain.score_line(r),
                   "marker": explain.bar_position(r),
                   "scale": [{"label": n, "from": lo, "to": hi} for n, lo, hi in explain.scale(r)]},
        "summary": a.summary,
        "strengths": [e.model_dump() for e in a.strengths],
        "improvements": [e.model_dump() for e in a.improvements],
        "dimensions": [{"id": d.dimension, "name": specs.get(d.dimension, {}).get("name", d.dimension),
                        "question": explain.AREA_QUESTION.get(d.dimension, ""), "rating": explain.rating(d.score),
                        "score": d.score, "rationale": d.rationale, "evidence": [e.model_dump() for e in d.evidence]}
                       for d in a.dimensions],
        "privacy": "No transcript is included.",
    }


# ---------------------------------------------------------------- PDF

_ASCII = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
          "‑": "-", "…": "...", " ": " ", "•": "-", "≥": ">=", "≤": "<=",
          "×": "x", "→": "->", "≈": "~"}


def _latin1(text: str) -> str:
    """The PDF uses a built-in font (latin-1 only), so non-Latin script (e.g. Malayalam quotes) is replaced by
    a marker. Evidence descriptions are written in English, so the point of each item is kept."""
    # Unusual spaces (e.g. the narrow no-break space in "Student 4") become plain spaces first.
    text = "".join(" " if c.isspace() else _ASCII.get(c, c) for c in str(text))
    text = re.sub(r"[^\x00-\xff]+(?:\s+[^\x00-\xff]+)*", "[quote in original language]", text)
    return text


def to_pdf(data: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(16, 16, 16)
    accent, text_col, muted = (110, 92, 220), (42, 40, 64), (110, 106, 136)

    def heading(t: str, size: int = 14) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(*accent)
        pdf.multi_cell(0, 7, _latin1(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*text_col)

    def para(t: str, style: str = "", size: int = 10, color=None) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*(color or text_col))
        pdf.multi_cell(0, 5.4, _latin1(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*text_col)

    def item(ts: str, t: str) -> None:
        ts = explain.display_time(ts)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*accent)
        width = max(16, pdf.get_string_width(_latin1(ts)) + 3) if ts else 5
        pdf.cell(width, 5.4, _latin1(ts) if ts else "-")
        pdf.set_text_color(*text_col)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5.4, _latin1(t), new_x="LMARGIN", new_y="NEXT")

    def scale_bar(r: dict) -> None:
        """Draw the four label bands with a marker at the score."""
        x0, y, width, h = pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.l_margin - pdf.r_margin, 7
        for band in r["scale"]:
            bx = x0 + band["from"] / 5 * width
            bw = (band["to"] - band["from"]) / 5 * width - 0.8
            on = band["label"] == r["label"]  # always the FINAL result
            pdf.set_fill_color(*((206, 196, 250) if on else (238, 235, 250)))
            pdf.rect(bx, y, bw, h, style="F")
            pdf.set_font("Helvetica", "B" if on else "", 8.5)
            pdf.set_text_color(*(text_col if on else muted))
            pdf.set_xy(bx, y)
            pdf.cell(bw, h, band["label"], align="C")
        mx = x0 + min(max(r["marker"] / 5, 0), 1) * width
        pdf.set_fill_color(*text_col)
        pdf.rect(mx - 0.7, y - 1.5, 1.4, h + 3, style="F")
        pdf.set_text_color(*text_col)
        pdf.set_xy(x0, y + h + 3)

    r = data["result"]
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*accent)
    pdf.cell(0, 12, data["title"], new_x="LMARGIN", new_y="NEXT")
    para(f"Generated {data['generated']}  |  {data['recording']['duration_min']} min class  |  "
         f"{data['recording']['language']}", size=9, color=muted)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*text_col)
    pdf.cell(0, 11, _latin1(f"Overall Result: {r['label']}"), new_x="LMARGIN", new_y="NEXT")
    if r["not_a_class"]:
        para("Not a class recording", "B", 11)
    elif r["quality_issue"]:
        para("Recording quality issue", "B", 11)
    para(r["meaning"], size=11)

    heading("Where This Class Falls", 12)
    scale_bar(r)
    para(r["score_line"], size=9.5, color=muted)

    heading("How Sure Is This Result?", 12)
    para(f"{r['certainty']}. {r['certainty_meaning']}")
    for x in r["certainty_reasons"]:
        item("", x)

    heading("In Short")
    para(data["summary"])
    if r["why"]:
        heading("Why This Result")
        if r["score_label"] and r["score_label"] != r["label"]:
            para("Required standards that were not met are listed first.", "I", 9.5, muted)
        for x in r["why"]:
            item("", x)

    para("Times like 12:05 mean minutes:seconds into the recording.", "I", 9, muted)
    for title, items in (("What Went Well", data["strengths"]), ("What To Improve", data["improvements"])):
        heading(title)
        for e in items or [{"timestamp": "", "observation": "Nothing specific was found."}]:
            item(e["timestamp"], e["observation"])

    heading("Area By Area")
    for d in data["dimensions"]:
        heading(f"{d['name']}: {d['rating']}", 11.5)
        para(d["question"] + ("" if d["score"] is None else f"  (score {d['score']:.1f} out of 5)"), "I", 9.5, muted)
        para(d["rationale"] or "This area could not be checked from this recording.", size=10)
        for e in d["evidence"]:
            item(e["timestamp"], e["observation"])

    pdf.ln(3)
    para(data["privacy"], "I", 9, muted)
    return bytes(pdf.output())
