# ClassCritique

Assesses a recorded online chess class for children (8x8) and returns **one label** (Poor / Average /
Good / Excellent) with a score, a confidence, and **timestamped, evidence-based observations** for each
rubric dimension.

> **Confidential data.** Recordings show identifiable children. Media, frames, audio, transcripts, labels and
> feedback are never committed (`data/` and media types are gitignored), frames never leave the machine,
> logs never contain transcript text, and children appear only as *Student 1, Student 2, …* in results.

## How it works

```
video / link ─► validate (ffmpeg) ─┬─► audio ─► Whisper (Groq) ─┐
                                    ├─► voice separation (local) ─┼─► per-chunk event digests (LLM, quotes verified)
                                    └─► sampled frames (local) ───┘        │
                                                                           ▼
                                         metrics + digests + visual notes ─► judge (LLM, rubric) ─► evidence check
                                                                                                    │
                                                         weighted score ─► label + confidence (Python) ◄┘
```

| Stage | Module | What it does |
|---|---|---|
| Input | `src/media.py`, `src/download.py` | File type, size, duration and decode checks; links via yt-dlp (YouTube, Google Drive, Zoom, direct); 16 kHz mono audio in ~10 min chunks |
| Speech | `src/transcribe.py` | Groq `whisper-large-v3` with segment timestamps and confidence; language auto-detect plus a language lock; low-confidence segments are flagged and never used as evidence |
| Speakers | `src/speakers.py`, `src/audio.py` | Silero VAD, WeSpeaker embeddings (ONNX), clustering; the largest voice is taken as the teacher; stability-based confidence |
| Video | `src/visual.py` | One frame per ~45 s in memory: faces (YuNet), tutor camera full-size, 8x8 logo (template match), black frames |
| Digests | `src/condense.py` | Small model turns each chunk into timestamped English events; quotes must exist in the transcript; names replaced by pseudonyms |
| Metrics | `src/metrics.py` | Talk shares, student turns, questions (open/closed), wait time, children addressed individually, praise, homework, silences |
| Judge | `src/evaluator.py` | One call with rubric, metrics, digests, visual notes, reviewer style and anchors; Pydantic-validated JSON; evidence re-checked against real moments |
| Score | `src/scoring.py` | Python computes the weighted score and label; missing dimensions are re-weighted and lower confidence |
| UI | `app.py`, `src/ui_*.py` | Streamlit: upload or link, progress by stage, result card, evidence per dimension, signals used/missing |

### Strict rules (applied after the score; they can only lower the label)

| Rule | Effect |
|---|---|
| Recording quality: resolution < 360p, blurry video, frozen video (~7.5 min), > 30% black frames, unclear audio, < 15% audible speech | **Poor**, marked "Recording quality issue" |
| 3 or more teaching mistakes | Poor |
| 1–2 teaching mistakes | At most Average |
| Tutor's face visible in < 25% of sampled frames | Poor |
| No homework and no recap/closing | At most Average |
| No 8x8 background in any full-size camera frame (3 or more checked) | At most Average |
| < 2 questions per 10 min, or < 2 moments of praise | At most Average |
| 3 or more "at most Average" rules broken | Poor |

Every rule that fires is listed on the results page and in the PDF. Thresholds are in `config/config.yaml` under `rules`.

See [docs/approach.md](docs/approach.md) for the design and results, and
[docs/ai_interaction_log.md](docs/ai_interaction_log.md) for decisions and what failed along the way.

## Setup

1. Python 3.11 and **ffmpeg** (with ffprobe):
   - Windows: `winget install Gyan.FFmpeg`, then open a new terminal (or set `FFMPEG_BIN` / `FFPROBE_BIN`)
   - macOS: `brew install ffmpeg` · Debian/Ubuntu: `apt install ffmpeg` · Streamlit Cloud: `packages.txt`
2. Install:
   ```
   python -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Groq key. The speaker-embedding model (~26 MB) downloads
   on first run and is checked by SHA-256.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROVIDER` | `groq` | `groq` (hosted, organiser-approved) or `local` (faster-whisper on CPU; LLM stages still need Groq) |
| `GROQ_API_KEY` | — | Required for `groq` |
| `HOSTED_PROVIDER_APPROVED` | — | Must be `true` before any recording is sent to the hosted provider |
| `TRANSCRIBE_MODEL` | `whisper-large-v3` | Speech-to-text model |
| `CONDENSE_MODEL` | `openai/gpt-oss-20b` | Summarises each ~10 min section into timestamped events |
| `JUDGE_MODEL` | `openai/gpt-oss-120b` | Scores the rubric from the evidence |
| `CONDENSE_PARALLEL_MODELS` | `openai/gpt-oss-120b` | Extra model summarising sections in parallel (own per-minute allowance) |
| `CONDENSE_FALLBACK_MODELS` / `JUDGE_FALLBACK_MODELS` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Tried when a model's free-tier daily limit is reached |
| `LOG_LEVEL` | `INFO` | Logs never include transcript text |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe` | Full paths if they are not on PATH |

Model names, limits, thresholds and token budgets are in `config/config.yaml`; the rubric is in
`config/rubric.json`. Nothing model-specific is hard-coded.

## Run

```
streamlit run app.py                              # the app
python -m src.pipeline path\to\class.mp4          # CLI: text-free JSON summary
python -m src.pipeline https://youtu.be/...       # CLI with a link
python -m pytest                                  # 99 tests, no network needed
```

A 45–60 min class takes about **5–7 minutes** on the Groq free tier (8,000 tokens/min per model); most of
that is waiting on rate limits.

### Dataset scripts (offline; need `data/` with the labelled recordings and `feedbacks.pdf`)

| Script | Purpose |
|---|---|
| `scripts/derive_rubric.py` | Cluster feedback phrases into themes; draft rubric (`config/rubric.draft.json`) |
| `scripts/triage_themes.py` | MEASURABLE / JUDGEABLE / NOT OBSERVABLE per theme |
| `scripts/check_consistency.py` | Flag notes whose tone contradicts their label (owner decides exclusions) |
| `scripts/extract_style.py` | Reviewer style → `config/style.json` (gitignored) |
| `scripts/run_samples.py` | Run every recording (resumable; `--only key` for quick checks) |
| `scripts/analyze_signals.py` | Spearman correlation of each signal with the label |
| `scripts/calibrate.py` | Weights and thresholds with leave-one-out; 2 held-out recordings; `--apply` to write |
| `scripts/build_anchors.py` | One paraphrased reference per label → `config/anchors.json` (gitignored) |
| `scripts/feedback_coverage.py` | Do outputs cover the human feedback; is cited evidence real |
| `scripts/check_speakers.py` | Voice separation accuracy against the Zoom VTTs |
| `scripts/make_logo.py`, `make_logo_template.py` | App logo; 8x8 logo template for the visual check |

## Hosting (Streamlit Community Cloud, free)

1. Push this repo to GitHub (secrets are never committed: `.env` and `data/` are gitignored).
2. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
3. Repository: this repo · Branch: `main` · Main file: `app.py`.
4. **Advanced settings:** Python **3.11**, and paste the secrets below (they become environment variables):
   ```toml
   GROQ_API_KEY = "your-groq-key"
   HOSTED_PROVIDER_APPROVED = "true"
   PROVIDER = "groq"
   TRANSCRIBE_MODEL = "whisper-large-v3"
   CONDENSE_MODEL = "openai/gpt-oss-20b"
   CONDENSE_PARALLEL_MODELS = "openai/gpt-oss-120b"
   CONDENSE_FALLBACK_MODELS = "openai/gpt-oss-120b"
   JUDGE_MODEL = "openai/gpt-oss-120b"
   JUDGE_FALLBACK_MODELS = "openai/gpt-oss-20b"
   ```
5. **Deploy.** `packages.txt` installs ffmpeg; the speaker model downloads on first use.

Uploads are capped at 500 MB to fit the free host's memory. The hosted app shares the Groq free-tier allowance of
the configured key.

## Privacy and data handling

- **Uploads and links leave nothing on disk.** Each run uses a private temp folder deleted in `finally`; folders
  left by killed runs are swept on the next start. The app never writes results to the cache (it may reuse
  entries created by the offline scripts).
- **Frames are analysed locally only**: decoded into memory, never saved, never sent to a model.
- Hosted processing needs both `PROVIDER=groq` and `HOSTED_PROVIDER_APPROVED=true`.
- Children's names in the transcript are replaced by stable pseudonyms before the judge sees anything.

## Limitations

- **Small calibration set**: 10 labelled classes (2 held out). Weights and thresholds are sanity-checked, not
  validated; a few more labels could change them.
- **Teacher/student separation is approximate**: 0.70 mean balanced accuracy against Zoom VTTs (0.94 down to
  chance on individual classes). Participation metrics carry this reliability, and the judge is told.
- **Video is sampled**, every ~45 s: no claims of continuous observation. The 8x8 logo can only be checked when
  the tutor's camera is full-size, and one tutor's logo version matches only weakly.
- **Not measured on purpose**: lighting (face brightness depends on skin tone), tone of voice, pitch, loudness,
  accent and speaking speed. Warmth is judged only from what is said.
- **Non-English classes** (e.g. Malayalam) work but with less accurate transcription and extraction; confidence
  is lowered automatically.
- **Not observable from a recording**: history with the tutor (e.g. prior warnings), preparation, punctuality.
- Language-model outputs vary slightly between runs; the score is computed deterministically from their output.
