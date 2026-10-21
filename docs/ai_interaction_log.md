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

## Stage 5: visual signals (local only)

**Frames never leave the machine.** Sampled every ~45 s, decoded by ffmpeg straight into memory, analysed
with OpenCV (YuNet faces, template matching). They contain children's faces and names, so no vision model
is used; the Groq account offers none anyway.

**8x8 logo detection by template matching.** Template = the right-hand "8" of the branded background, cut
from one B552 frame (logo only, checked for face overlap). Checked on three tutors:

| Recording | Feedback | Camera-view frames | Match scores | Logo detected |
|---|---|---|---|---|
| good/B552 | "fixed 8x8 background" | 7 | 0.35-0.47 / 1.00 | 2 of 7 (from ~6 min on) |
| excellent/B491 | "background was there" | 17 | 0.34-0.49 / 0.68 | 6 of 17 |
| quality issue/B392 | "no background" | 16 | 0.39-0.43 | 0 of 16 |

Threshold 0.6 separates all three, but the margin is narrow for B491, whose logo variant matches at only 0.68
(larger template sizes did not help, so the difference is in the artwork, not the scale). B552's 1.00 matches
are partly in-sample. The logo is only checkable when the tutor camera is full-size, which is a small share
of a chess class (most time is screen-shared boards).

**Wording bug that made the judge unfair.** The first observation text ("tutor camera shown large in 7/61")
led the judge to write "the tutor's face is visible in only 11% of frames". Faces were in fact visible in 84%:
the tutor is a small tile while screen sharing. Observations now state both numbers and say screen sharing
is normal; the judge has a rule not to treat it as a fault. B552 professional_setup 2.0 -> 3.0.

**Lighting is deliberately not measured.** Face brightness depends on skin tone and frame brightness on the
(dark) 8x8 background, so neither is a fair lighting signal; only fully black frames are reported.

## Owner feedback round: teaching accuracy, weights, reports, and a serious prompt bug

**Teaching Accuracy added (owner request), weighted highest (25%).** The judge checks every explanation and
correction against chess knowledge (rules, piece movement, check/mate/stalemate, castling, en passant,
promotion, notation, principles), must quote and timestamp any mistake and state the correct fact, ignores
apparent errors caused by speech recognition, and does not score "best move" claims it cannot verify
without the board. Opening/rapport weight cut from 15% to 5% at the owner's request.

**The judge was seeing zero events.** After the reviewer-style file and anchors were added, the fixed parts of the
judge prompt (rubric 968 + metrics 922 + style 357 + anchors 521 + visual 248 tokens) exceeded the 2,970-token
budget left on the free tier, so all 133 events of a class were trimmed. The judge then cited section ranges
("09:10-18:20") or "whole class" and labels drifted (good/B435 came out Poor at 0.60). Fixes: compact rubric and
metric lines, a 56-word style text, one-line anchors, a guaranteed 1,800-token floor for evidence (optional
context is dropped first, with a warning), transcript evidence spanning more than 2 minutes is rejected, and a
regression test asserts the judge sees 30+ events under the real budget. The token estimate was also relaxed
from 3.5 to 4.0 characters per token for Latin text. Result on B552: 50 events shown, style and anchors kept,
29 specific timestamps, label Good (human Good).

**No raw metric names in output.** A post-processing step rewrites names like
`recap_or_conclusion_in_last_10min = true` into plain words; the judge is also told not to use them.
