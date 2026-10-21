"""ClassCritique: Streamlit UI. Upload a recording or paste a link, then see the label and evidence."""
from __future__ import annotations

import base64
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
from src import ui_results as ui
from src.ui_theme import background_js, css

setup_logging()
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
upload_tab, link_tab = st.tabs(["⬆  Upload video", "🔗  Paste link"])
with upload_tab:
    upload = st.file_uploader("Class recording", type=[e.lstrip(".") for e in cfg["media"]["allowed_extensions"]],
                              label_visibility="collapsed")
    analyze_file = st.button("Analyze recording", key="analyze_file", disabled=upload is None)
with link_tab:
    link = st.text_input("Link to the recording", label_visibility="collapsed",
                         placeholder="Paste a YouTube, Google Drive, Zoom cloud recording or direct video link")
    st.caption("Private links must be shared as “Anyone with the link can view”.")
    analyze_link = st.button("Analyze link", key="analyze_link", disabled=not link.strip())

st.markdown('<p class="app-note">Recordings are processed only by the approved provider and deleted '
            'after analysis. Children are referred to as Student 1, Student 2, …</p>', unsafe_allow_html=True)


def analyze(tmp_path: str | None, url: str | None) -> None:
    tracker = st.empty()
    is_link = tmp_path is None
    state = {"stage": "Downloading" if is_link else "Validating file"}

    def on_stage(stage: str, frac: float) -> None:
        # Streamlit can only draw from the script thread; ignore progress reported by worker threads.
        if stage == "Done" or get_script_run_ctx() is None:
            return
        state["stage"] = stage
        tracker.markdown(ui.progress_html(stage, frac, is_link), unsafe_allow_html=True)

    on_stage(state["stage"], 0.0)
    result = pipeline.run(tmp_path, on_stage=on_stage) if tmp_path else pipeline.run(url=url, on_stage=on_stage)
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
    except MediaError as e:
        st.error(f"**Recording problem.** {e}")
    except ConfigError as e:
        st.error(f"**Configuration problem.** {e}")
    except ProviderError as e:
        if "429" in str(e) or "413" in str(e):
            st.warning("**The free-tier rate limit was reached.** Work done so far is saved, so clicking "
                       "Analyze again in a few minutes continues where it stopped.")
        else:
            st.error(f"**The AI provider failed** ({e}). Please try again in a minute; finished steps are saved.")
    except LLMOutputError:
        st.error("**The assessment model returned an unusable answer twice.** Please try again.")
    except Exception as e:  # last resort: show type only, never content
        st.error(f"**Unexpected error** ({type(e).__name__}). Check the server log.")
        raise
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result is not None and result.result is not None:
    st.markdown(ui.hero_html(result), unsafe_allow_html=True)
    st.markdown(ui.strengths_html(result), unsafe_allow_html=True)
    st.markdown(ui.dimensions_html(result), unsafe_allow_html=True)
    st.markdown('<h2 class="cc-h2">How this was assessed</h2>', unsafe_allow_html=True)
    st.markdown(ui.signals_html(result), unsafe_allow_html=True)
    with st.expander("Computed metrics"):
        st.markdown(ui.metrics_html(result), unsafe_allow_html=True)
    st.download_button("⬇  Download report (JSON)", ui.report_json(result), file_name="classcritique_report.json",
                       mime="application/json", help="Assessment and metrics only; no transcript.")
