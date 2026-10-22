"""Report downloads (PDF / Word / text / JSON) and plain-language output, on a synthetic result."""
import json
import re

from src import report
from src.metrics import humanize
from src.pipeline import PipelineResult, Signal
from src.schemas import (Assessment, DimensionScore, EvidenceItem, MediaInfo, Metric, ScoreResult, Transcript)

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿⏩-⏿]")


def _result() -> PipelineResult:
    ev = [EvidenceItem(timestamp="12:05", observation='Asked Student 2 "നല്ല നീക്കം ഏതാണ്?" and waited.')]
    a = Assessment(summary="Clear lesson with a recap.", confidence=0.7,
                   dimensions=[DimensionScore(dimension="teaching_accuracy", score=4.5, rationale="No errors found.",
                                              evidence=ev),
                               DimensionScore(dimension="professional_setup", score=None)],
                   strengths=ev, improvements=[EvidenceItem(timestamp="40:00", observation="Should give homework.")])
    return PipelineResult(
        file_hash="abc", media=MediaInfo(duration_s=2700, has_video=True, has_audio=True, size_mb=100, container="mp4"),
        audio_levels={}, transcript=Transcript(duration_s=2700, provider="groq", model="whisper", language="ml"),
        assessment=a, result=ScoreResult(label="Good", score=3.4, confidence=0.66, low_confidence=False,
                                         weight_coverage=0.9),
        metrics=[Metric(name="homework_given", value=False, source="events"),
                 Metric(name="speech_ratio", value=0.66, source="audio")],
        signals=[Signal(name="Transcript", status="used", detail="500 segments")])


def test_pdf_report_has_the_essentials_only():
    from fpdf import FPDF  # noqa: F401  (dependency present)
    data = report.report_data(_result())
    pdf = report.to_pdf(data)
    assert pdf[:4] == b"%PDF"
    assert not any(hasattr(report, f) for f in ("to_json", "to_docx", "to_text"))   # PDF is the only format
    assert set(data) >= {"result", "summary", "strengths", "improvements", "dimensions"}
    assert not {"metrics", "signals", "limitations"} & set(data)                     # these stay in the app


def test_pdf_marks_non_latin_text_instead_of_failing():
    assert report._latin1("Asked “നല്ല നീക്കം” – good") == 'Asked "[quote in original language]" - good'


def test_reports_contain_no_emojis_and_no_transcript():
    data = report.report_data(_result())
    assert not EMOJI.search(str(data))
    assert "segments" not in json.dumps(data["result"])  # no transcript content in the report body


def test_humanize_replaces_metric_names():
    assert humanize("recap_or_conclusion_in_last_10min = true") == "recap / closing at the end: yes"
    assert humanize("2+2=4 and Student 3") == "2+2=4 and Student 3"


def test_plain_language_helpers():
    from src import explain
    from src.schemas import ScoreResult
    assert [explain.rating(x) for x in (4.5, 3.2, 2.1, 1.0, None)] == ["Strong", "Fine", "Needs work", "Weak", "Not checked"]
    assert explain.certainty(0.8) == "Fairly sure" and explain.certainty(0.3) == "Not very sure"
    assert "voice" in explain.plain_reason("Teacher/student voice separation is uncertain (confidence 0.37).")
    assert "Malayalam" in explain.plain_reason("Class language is Malayalam; transcription is less accurate.")
    qi = ScoreResult(label="Poor", score=3.5, confidence=0.6, low_confidence=False, weight_coverage=1, quality_issue=True)
    assert "recording" in explain.label_meaning(qi).lower()


def test_main_reasons_lead_with_rules_then_weak_and_strong_areas():
    from src import explain
    from src.scoring import load_rubric
    from src.schemas import ScoreResult
    a = Assessment(summary="", confidence=0.7, dimensions=[
        DimensionScore(dimension="warmth_energy", score=1.5, rationale="Little praise. Mostly instructions."),
        DimensionScore(dimension="teaching_accuracy", score=4.5, rationale="All explanations were correct.")])
    r = ScoreResult(label="Average", score=3.0, confidence=0.7, low_confidence=False, weight_coverage=1,
                    rules_applied=["No homework and no recap or closing."])
    reasons = explain.main_reasons(r, a, load_rubric())
    assert reasons[0].startswith("No homework")
    assert any("Warmth And Energy needs work: Little praise." == x for x in reasons)
    assert reasons[-1].startswith("Teaching Accuracy was a strength")


def test_bar_always_shows_the_final_result():
    from src import explain
    from src.schemas import ScoreResult
    lowered = ScoreResult(label="Average", score=3.5, score_label="Good", confidence=0.8, low_confidence=False,
                          weight_coverage=1, rules_applied=["No homework and no recap or closing."])
    band = next(b for b in explain.scale(lowered) if b[0] == "Average")
    assert band[1] <= explain.bar_position(lowered) < band[2]          # marker inside the Average band
    assert "would be Good" in explain.score_line(lowered) and "Average" in explain.score_line(lowered)
    normal = ScoreResult(label="Good", score=3.5, score_label="Good", confidence=0.8, low_confidence=False, weight_coverage=1)
    assert explain.bar_position(normal) == 3.5


def test_report_date_is_day_month_year():
    from datetime import date
    data = report.report_data(_result())
    assert data["generated"] == date.today().strftime("%d/%m/%Y")      # e.g. 22/10/2026, not 2026-10-22
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", data["generated"])
