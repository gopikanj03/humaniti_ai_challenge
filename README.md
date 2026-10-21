# Class Recording Assessment (prototype)

Streamlit app that assesses an online children's class recording and returns one label
(Poor / Average / Good / Excellent) with timestamped, evidence-based observations.

> **Confidential data.** Recordings contain identifiable children. Media, frames, audio,
> transcripts, labels and feedback must never be committed. Everything under `data/` is
> gitignored; temp files are deleted after every run.

## Setup

1. Python 3.11, and **ffmpeg** (with ffprobe) on PATH:
   - Windows: `winget install Gyan.FFmpeg` (then open a new terminal)
   - macOS: `brew install ffmpeg` · Debian/Ubuntu: `apt install ffmpeg`
   - Streamlit Cloud: `packages.txt` installs it.
2. Install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and adjust if needed.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROVIDER` | `groq` | `groq` = hosted (organiser-approved). `local` = faster-whisper on CPU, nothing leaves the machine. |
| `GROQ_API_KEY` | — | Required for `PROVIDER=groq`. |
| `HOSTED_PROVIDER_APPROVED` | — | Must be `true` for `groq` to run; a guard against sending recordings anywhere by accident. |
| `LOG_LEVEL` | `INFO` | Logs never contain transcript text. |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe` | Paths, if they aren't on PATH. |

Model names, limits and thresholds live in `config/config.yaml`.

## Run

```
streamlit run app.py                         # UI
python -m src.pipeline path\to\recording.mp4 # CLI; prints a text-free summary
python -m pytest                             # tests
```
