# ClassCritique

Assesses a recorded online chess class for children (8x8) and returns **one label** (Poor / Average /
Good / Excellent) with **timestamped, evidence-based observations**. Input is a video file or a link
(YouTube, Google Drive, Zoom, direct URL).

> **Confidential data.** Recordings show identifiable children. Media, transcripts, labels and feedback are never
> committed (`data/` is gitignored), frames never leave the machine, logs never contain transcript text, uploads
> leave nothing on disk, and children appear only as *Student 1, Student 2, …*.

## How it works

1. **Check and prepare** the video (ffmpeg); links via yt-dlp.
2. **Transcribe** with Groq `whisper-large-v3`; separate tutor and children by voice (local).
3. **Check sampled frames** locally: tutor's camera, 8x8 background, black or frozen video.
4. **Summarise** each 10-minute section into timestamped moments (`gpt-oss-20b`); every quote must exist in the
   transcript.
5. **Judge** seven rubric areas from those moments, citing them; each reported teaching mistake is double-checked.
6. **Score in Python**: weighted score, then strict rules that can only lower the label (recording-quality
   problems or not a class → Poor; teaching mistakes, no homework and no recap, no 8x8 background, too few
   questions or praise → at most Average; two of these → Poor). Thresholds are in `config/config.yaml`.

Details and results: [docs/approach.md](docs/approach.md) · Decisions and failures:
[docs/ai_interaction_log.md](docs/ai_interaction_log.md)

## Setup

1. Python 3.11 and ffmpeg (Windows: `winget install Gyan.FFmpeg`; macOS: `brew install ffmpeg`).
2. Install and configure:
   ```
   python -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   copy .env.example .env            # then add your GROQ_API_KEY
   ```

Main settings (in `.env` or hosting secrets): `GROQ_API_KEY`, `HOSTED_PROVIDER_APPROVED=true`, and the models:
`gpt-oss-20b` for summaries and judging, `gpt-oss-120b` only when 20b's free daily limit is used up. Everything
else is in `config/config.yaml` and `config/rubric.json`.

## Run

```
streamlit run app.py                         # the app
python -m src.pipeline path\to\class.mp4     # command line (file or link)
python -m pytest                             # tests, no network needed
```

A 45–60 min class takes about **5–7 minutes** on the Groq free tier. Offline dataset scripts (rubric from
feedback, batch runs, calibration) are in `scripts/`; each explains its use at the top of the file.

## Hosting (Streamlit Community Cloud)

Create an app at https://share.streamlit.io from this repo (`main`, `app.py`, Python 3.11) and add these secrets:

```toml
GROQ_API_KEY = "your-groq-key"
HOSTED_PROVIDER_APPROVED = "true"
PROVIDER = "groq"
CONDENSE_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-20b"
CONDENSE_PARALLEL_MODELS = ""
CONDENSE_FALLBACK_MODELS = "openai/gpt-oss-120b"
JUDGE_FALLBACK_MODELS = "openai/gpt-oss-120b"
```

Uploads are capped at 500 MB. After pushing changes to `src/`, reboot the app.

## Limitations

- **Only 10 labelled classes**: 4 of 8 analysed classes matched 8x8's label, most misses one level off; the full
  run with the latest version is still to be done.
- **Voice separation is approximate** (0.70 balanced accuracy), so participation is reported cautiously.
- **Video is sampled** every ~45 s; the 8x8 background is only checked when the tutor's camera is full-size.
- **Not measured on purpose**: tone of voice, pitch, loudness, accent, speaking speed, lighting.
- **Non-English classes** work with lower confidence; history such as prior warnings is not observable.
