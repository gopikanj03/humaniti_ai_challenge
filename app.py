"""Streamlit UI. Stage 1: upload -> validate -> transcribe, with progress by stage."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from src import pipeline
from src.config import ConfigError, get_config, setup_logging
from src.media import MediaError
from src.net import ProviderError
from src.transcribe import model_name

setup_logging()
cfg = get_config()

st.set_page_config(page_title="Class Recording Assessment", layout="wide")
st.title("Class Recording Assessment")
st.caption("Upload an online class recording to get a quality label with timestamped evidence.")

with st.sidebar:
    st.subheader("Processing")
    st.write(f"**Provider:** `{cfg['provider']}`")
    st.write(f"**Transcription model:** `{model_name(cfg['provider'])}`")
    if cfg["provider"] == "local":
        st.info("Local mode: audio never leaves this machine.")
    else:
        st.warning("Hosted mode: audio is sent to the configured provider.")

source = st.radio("Recording source", ["Upload file", "Paste link"], horizontal=True, label_visibility="collapsed")
upload, link = None, ""
if source == "Upload file":
    upload = st.file_uploader("Class recording", type=[e.lstrip(".") for e in cfg["media"]["allowed_extensions"]])
else:
    link = st.text_input("Link to the recording", placeholder="YouTube, Google Drive, Zoom cloud recording, or a direct video URL")
    st.caption("Private links must be shared as 'Anyone with the link can view'.")
ready = upload is not None or bool(link.strip())
analyze = st.button("Analyze", type="primary", disabled=not ready)

if analyze and ready:
    tmp_path = None
    try:
        if upload is not None:
            # Write the upload to a private temp file; removed in `finally` even if the pipeline fails.
            fd, tmp_path = tempfile.mkstemp(prefix="upload_", suffix=Path(upload.name).suffix.lower())
            with os.fdopen(fd, "wb") as f:
                f.write(upload.getbuffer())

        with st.status("Analyzing recording...", expanded=True) as status:
            bar = st.progress(0.0)
            last = {"stage": None}

            def on_stage(stage: str, frac: float) -> None:
                if stage != last["stage"]:
                    st.write(f"• {stage}")
                    last["stage"] = stage
                bar.progress(min(max(frac, 0.0), 1.0), text=stage)

            if tmp_path:
                result = pipeline.run(tmp_path, on_stage=on_stage)
            else:
                result = pipeline.run(url=link, on_stage=on_stage)
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["result"] = result
    except MediaError as e:
        st.error(f"Recording problem: {e}")
    except ConfigError as e:
        st.error(f"Configuration problem: {e}")
    except ProviderError as e:
        st.error(f"The transcription provider failed: {e}. Try again later or switch PROVIDER=local.")
    except Exception as e:  # last resort: show type only, never content
        st.error(f"Unexpected error ({type(e).__name__}). Check the server log.")
        raise
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

result = st.session_state.get("result")
if result is not None:
    summary = pipeline.summarize(result)
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("Language", summary["language"] or "unknown")
    c1.metric("Duration", f"{summary['duration_min']} min")
    c2.metric("Segments", f"{summary['usable_segments']} / {summary['segments']}", help="usable / total")
    c3.metric("Speech coverage", f"{summary['speech_coverage']:.0%}")
    c4.metric("Peak level", f"{summary['peak_dbfs']:.0f} dBFS")

    for w in result.warnings:
        st.warning(w)

    with st.expander("Transcript preview (shown here only, never logged)"):
        for s in result.transcript.segments:
            mm, ss = divmod(int(s.start), 60)
            marker = " ⚠️ low confidence" if s.flagged else ""
            st.markdown(f"`{mm:02d}:{ss:02d}` {s.text}{marker}")
