"""ClassCritique: Streamlit UI. Upload a recording or paste a link, then see the label and evidence."""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src import pipeline
from src.config import ConfigError, get_config, setup_logging
from src.llm import LLMOutputError
from src.media import MediaError
from src.net import ProviderError
from src import report
from src import ui_results as ui
from src.ui_theme import PALETTES, background_js, css

setup_logging()
log = logging.getLogger("classcritique.app")
pipeline.sweep_stale_temp()  # remove audio left by any run that was killed mid-way
cfg = get_config()
TITLE = cfg["app"]["title"]
ASSETS = Path(__file__).parent / "assets"
LOGO_B64 = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()

st.set_page_config(page_title=TITLE, page_icon=str(ASSETS / "logo.png"), layout="centered", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- theme
if "dark" not in st.session_state:
    st.session_state.dark = False
_, toggle_col = st.columns([5, 1])
with toggle_col:
    st.toggle("Dark", key="dark")
theme = "dark" if st.session_state.dark else "light"
st.markdown(css(theme), unsafe_allow_html=True)
components.html(background_js(theme), height=0)

# ---------------------------------------------------------------- header
st.markdown(f'<div class="app-brand"><img class="app-logo" alt="" src="data:image/png;base64,{LOGO_B64}">'
            f'<h1 class="app-title">{TITLE}</h1></div>', unsafe_allow_html=True)
st.markdown(f'<p class="app-subtitle">{cfg["app"]["subtitle"]}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- input
upload_tab, link_tab = st.tabs(["Upload Video", "Paste Link"])
with upload_tab:
    upload = st.file_uploader("Class recording", type=[e.lstrip(".") for e in cfg["media"]["allowed_extensions"]],
                              accept_multiple_files=False,  # one recording per analysis
                              label_visibility="collapsed")
    analyze_file = st.button("Analyze recording", key="analyze_file", disabled=upload is None)
with link_tab:
    link = st.text_input("Link to the recording", label_visibility="collapsed",
                         placeholder="Paste a YouTube, Google Drive, Zoom cloud recording or direct video link")
    st.caption("Private links must be shared as “Anyone with the link can view”.")
    analyze_link = st.button("Analyze link", key="analyze_link", disabled=not link.strip())

st.markdown('<p class="app-note">Recordings are processed only by the approved provider and deleted '
            'after analysis.</p>', unsafe_allow_html=True)


def analyze(tmp_path: str | None, url: str | None) -> None:
    tracker = st.empty()
    clock = st.empty()   # typical duration + rotating tips, animated in the browser while the server works
    with clock.container():
        components.html(ui.live_tips_html(PALETTES[theme]), height=52)
    is_link = tmp_path is None

    def on_stage(stage: str, frac: float, detail: str = "") -> None:
        # Streamlit can only draw from the script thread; ignore progress reported by worker threads.
        if stage == "Done" or get_script_run_ctx() is None:
            return
        tracker.markdown(ui.progress_html(stage, frac, is_link, detail=detail), unsafe_allow_html=True)

    on_stage("Downloading" if is_link else "Validating file", 0.0, "Starting…")
    try:
        result = pipeline.run(tmp_path, on_stage=on_stage) if tmp_path else pipeline.run(url=url, on_stage=on_stage)
    finally:
        clock.empty()
    tracker.markdown(ui.progress_html("", 1.0, is_link, done=True), unsafe_allow_html=True)
    st.session_state["result"] = result


if (analyze_file and upload is not None) or (analyze_link and link.strip()):
    st.session_state.pop("result", None)
    tmp_path = None
    try:
        if analyze_file:
            # Write the upload to a private temp file; removed in `finally` even if the pipeline fails.
            fd, tmp_path = tempfile.mkstemp(prefix="upload_", suffix=Path(upload.name).suffix.lower())
            with os.fdopen(fd, "wb") as f:
                f.write(upload.getbuffer())
        analyze(tmp_path, None if analyze_file else link)
    # User-facing messages stay plain; technical details (codes, limits, models) go to the server log only.
    except MediaError as e:
        st.error(f"**There is a problem with this recording.** {e}")
    except ConfigError as e:
        log.error("configuration problem: %s", e)
        st.error("**The app is not set up correctly.** Please contact the administrator.")
    except ProviderError as e:
        log.warning("analysis service error: %s (code=%s)", e, e.code)
        if e.code == "daily_limit":
            minutes = max(1, round((e.retry_after or 600) / 60))
            st.warning(f"**The analysis service has reached its usage limit for now.** Please try again in about "
                       f"{minutes} minutes; steps already finished will be reused.")
        else:
            st.warning("**The analysis service is busy right now.** Please click Analyze again in a minute; "
                       "steps already finished will be reused.")
    except LLMOutputError:
        log.warning("assessment returned unusable output twice")
        st.warning("**The analysis could not be completed this time.** Please click Analyze again.")
    except Exception:  # last resort: never show internals on screen
        log.exception("unexpected error during analysis")
        st.error("**Something went wrong while analysing this recording.** Please try again.")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result is not None and result.result is not None:
    st.markdown(ui.hero_html(result), unsafe_allow_html=True)
    st.markdown(ui.strengths_html(result), unsafe_allow_html=True)
    st.markdown(ui.dimensions_html(result), unsafe_allow_html=True)
    st.markdown('<h2 class="cc-h2">How This Was Assessed</h2>', unsafe_allow_html=True)
    st.markdown(ui.signals_html(result), unsafe_allow_html=True)
    with st.expander("Computed Metrics"):
        st.markdown(ui.metrics_html(result), unsafe_allow_html=True)
    data = report.report_data(result)
    st.markdown('<h2 class="cc-h2">Download Report</h2>', unsafe_allow_html=True)
    st.download_button("Download PDF", report.to_pdf(data), file_name="classcritique_report.pdf",
                       mime="application/pdf")
    st.caption("The PDF contains the result, strengths, areas to improve and the evidence for each score.")
