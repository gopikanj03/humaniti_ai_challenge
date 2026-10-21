# AI interaction log

Key decisions made while building this with an AI coding assistant, and the evidence behind them.
No transcript text, names or frames are recorded here.

## Stage 1: ingestion and transcription

**Dataset shape.** Label folders (excellent / good / average / weak / quality issue / demo), each with an
MP4 and a Zoom VTT; one `feedbacks.pdf` with a single feedback line per class. Mapping agreed:
weak and quality issue → Poor; demo classes have no feedback and are used as unlabelled test runs.

**VTTs are answer keys, not inputs.** Users will only provide a video file or a link, so the app never
accepts a transcript. The VTTs have per-line speaker names (the tutor was the top speaker in 10/10 labelled
classes), so they are used offline to measure how well our own transcription and speaker separation work.

**Input by link.** yt-dlp covers YouTube, Google Drive, Zoom cloud recordings and direct URLs. Downloads go
to the per-run temp dir; only public http(s) hosts are allowed; logs record the host name only.

**Whisper model: large-v3, not large-v3-turbo.** Tested on one ~8.6 min chunk of the Malayalam demo:

| Setting | Detected | Median avg_logprob | Note |
|---|---|---|---|
| turbo, auto | English | -0.71 | English-like output: effectively translating |
| large-v3, auto | Malayalam | -0.45 | Mixed Malayalam + English (tutor code-switches) |
| turbo, forced ml | Malayalam | -1.01 | Fewer segments: drops speech |
| large-v3, forced ml | Malayalam | -0.46 | Same as auto |

**Language lock.** With large-v3 on the full demo, 5/6 chunks were Malayalam; the first (mostly pre-class
waiting) came back "English". Chunks that disagree with the recording's main language are re-transcribed
with that language forced, and the forced version is kept only if Whisper is at least as confident, so a
genuinely English stretch survives. Result: Malayalam demo → Malayalam, 81% speech coverage; English class
B552 → English, 773/777 usable segments, about 33 s for 46 min of audio.

**Visual finding (for stage 5).** "8x8 background" means the company's branded virtual background: a large
white "88" logo with "Eight…Eight" text behind the tutor. These are chess classes: most of the time is
screen-shared lichess.org boards, with the tutor in a small corner tile; the logo is clearly visible only
when the tutor's camera is full-size. Frames also show children's names on tiles, so frame checks run
locally (faces blurred before any frame would leave the machine).

## Stage 2: speakers, metrics, digests, judge, scoring

**Speaker separation is only moderately reliable, and we measured it.** WeSpeaker ResNet34 (ONNX via
sherpa-onnx, no PyTorch) embeddings on 1.5 s windows, agglomerative clustering, largest cluster = teacher.
Scored against the Zoom VTTs (which know who spoke) on all 12 recordings:

| Threshold | Mean balanced accuracy | Mean teacher-share error |
|---|---|---|
| 0.30 | 0.68 | 0.21 |
| **0.35** | **0.70** | **0.16** |
| 0.40 | 0.69 | 0.18 |

Per-recording accuracy ranged from 0.94 down to chance (~0.5). Children's voices are outside the
embedding model's training data and Zoom mixes everyone into one track.

**The first confidence heuristic was wrong, so it was replaced.** Cluster margin/tightness correlated
*negatively* with accuracy (r = -0.25). Stability, i.e. how much the teacher share moves when re-clustering
at threshold ±0.05, correlated with accuracy (r = -0.48 between spread and balanced accuracy) and is now the
confidence, capped at 0.75. Participation metrics carry this reliability into the judge prompt.

**Free-tier limits shaped the LLM stages.** Groq free tier allows 8,000 tokens/min per model, and counts
`max_completion_tokens` against it, so the first runs failed with HTTP 413. Requests are now budgeted
(prompt + reply allowance <= 7,500 tokens): long transcript chunks are split, and the judge's event list
is trimmed by priority.

**A trimming bug made the judge criticise the teacher for our own data gap.** Trimming by priority-then-time
let 46 early "student_response" events push out all praise events, and the judge wrote "provide more
explicit praise... data lack specific praise moments". Fixes: event types take turns when trimming, and the
judge is told that omitted events are not evidence of absence. Warmth went from unscored to 4.0 on the same
class.

**Small-model JSON drift.** gpt-oss-20b occasionally invented event types, put the speaker in the type field,
or was cut off at the token limit. The parser now normalises synonyms per event, drops only invalid events,
and repairs truncated JSON to the last complete event, instead of failing the whole chunk.

**Fabrication checks work.** On B552, 21 of 154 extracted events had quotes not found in the transcript and
were dropped; the judge's evidence is re-checked against event timestamps (2 items dropped).
First end-to-end result: B552 (human label Good) -> Good, 3.41/5, confidence 0.72.
