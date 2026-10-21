"""Downloadable PDF report built from a pipeline result.

Reports carry the essentials only: overall result (with the reason if confidence is low), summary,
strengths, areas to improve, and each dimension's score, reasoning and timestamped evidence. Signals,
limitations and metrics stay in the app. Never the transcript;
children appear only as pseudonyms. The source file name is not included (it may contain names).
"""
from __future__ import annotations

import re
from datetime import date

from .pipeline import PipelineResult, summarize
from .scoring import load_rubric

SOURCE_LABEL = {"transcript": "Transcript", "metric": "Metric", "visual": "Video"}


def _confidence_word(c: float) -> str:
    return "High" if c >= 0.7 else "Medium" if c >= 0.5 else "Low"


def report_data(res: PipelineResult) -> dict:
    r, a, s = res.result, res.assessment, summarize(res)
    rubric = {d["id"]: d for d in load_rubric()["dimensions"]}
    return {
        "title": "ClassCritique Report",
        "generated": date.today().isoformat(),
        "recording": {"duration_min": s["duration_min"], "language": s["language"] or "unknown"},
        "result": {"label": r.label, "score": r.score, "confidence": r.confidence,
                   "confidence_level": _confidence_word(r.confidence), "low_confidence": r.low_confidence,
                   "confidence_reasons": r.confidence_reasons, "rubric_scored": r.weight_coverage},
        "summary": a.summary,
        "strengths": [e.model_dump() for e in a.strengths],
        "improvements": [e.model_dump() for e in a.improvements],
        "dimensions": [{"id": d.dimension, "name": rubric.get(d.dimension, {}).get("name", d.dimension),
                        "weight": rubric.get(d.dimension, {}).get("weight"), "score": d.score,
                        "rationale": d.rationale, "evidence": [e.model_dump() for e in d.evidence]}
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
    text = "".join(_ASCII.get(c, c) for c in str(text))
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
        pdf.multi_cell(0, 5.2, _latin1(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*text_col)

    def item(ts: str, t: str) -> None:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*accent)
        pdf.cell(22, 5.2, _latin1(f"[{ts}]") if ts else "-")
        pdf.set_text_color(*text_col)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5.2, _latin1(t), new_x="LMARGIN", new_y="NEXT")

    r = data["result"]
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*accent)
    pdf.cell(0, 12, data["title"], new_x="LMARGIN", new_y="NEXT")
    para(f"Generated {data['generated']}  |  {data['recording']['duration_min']} min  |  {data['recording']['language']}",
         size=9, color=muted)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*text_col)
    pdf.cell(0, 10, _latin1(f"Overall Label: {r['label']}"), new_x="LMARGIN", new_y="NEXT")
    para(f"Score {r['score']:.2f} / 5    Confidence {r['confidence']:.0%} ({r['confidence_level']})", "B", 11)
    if r["low_confidence"]:
        para("Low confidence: treat this label as provisional.", "I", 10, muted)
    for x in r["confidence_reasons"]:
        item("", x)

    heading("Summary")
    para(data["summary"])
    for title, items in (("Strengths", data["strengths"]), ("Areas To Improve", data["improvements"])):
        heading(title)
        for e in items or [{"timestamp": "", "observation": "None evidenced."}]:
            item(e["timestamp"], e["observation"])

    heading("How The Class Scored")
    for d in data["dimensions"]:
        score = "Not scored" if d["score"] is None else f"{d['score']:.1f} / 5"
        heading(f"{d['name']}: {score}", 11.5)
        para(f"Weight {d['weight']:.0%}. {d['rationale']}", size=9.5, color=muted)
        for e in d["evidence"]:
            item(e["timestamp"], f"({SOURCE_LABEL.get(e['source'], e['source'])}) {e['observation']}")

    pdf.ln(3)
    para(data["privacy"], "I", 9, muted)
    return bytes(pdf.output())
