# ClassCritique: approach

## Approach note

**Approach.** ClassCritique turns a class recording into evidence before judging it. Groq Whisper transcribes;
local voice clustering separates the teacher from the children; sampled frames are checked locally for the
tutor's camera and the 8x8 branded background. A small model condenses each 10-minute chunk into timestamped
events, and every quote must exist in the transcript. A larger model scores six rubric dimensions derived from
8x8's own feedback, citing those moments; Python computes the weighted score and label.

**What worked.** Evidence checks: unverifiable events and citations are dropped (21 of 154 events on one class),
so observations point to real moments. Rubric themes from clustering reviewer feedback match what reviewers
care about: rapport, individual attention, homework, warmth, the 8x8 background. whisper-large-v3 with a
language lock handled a Malayalam class that the turbo model mistook for English. Early checks agreed with
human labels (Good→Good, Poor→Poor; Excellent→Good).

**What failed or was weak.** Voice-based teacher/student separation averaged only 0.70 balanced accuracy, and our
first confidence measure pointed the wrong way, so it was replaced by a stability check and participation is
reported cautiously. Two bugs made the judge unfair: token-budget trimming hid praise, then blamed the teacher
for it, and "camera full-size in 11% of frames" read as "face visible 11%". Both are fixed with rules and
tests. The free tier (8k tokens/min) forced strict per-request budgets.

**Limitations.** Ten labelled classes cannot validate weights or thresholds. Video is sampled every 45 s. Lighting
and tone of voice are not measured, to avoid skin-tone and voice biases. Prior warnings are not observable.
Non-English classes get lower confidence.

---

## 1. Goal and constraints

Given one recording, output exactly one of Poor / Average / Good / Excellent plus specific, timestamped
observations. Constraints shaped most decisions:

- **Confidential children's data**: nothing private is committed or logged; frames stay local; children are
  pseudonymised; uploads leave nothing on disk.
- **Groq free tier only, open-source models** (whisper-large-v3, gpt-oss-20b, gpt-oss-120b): 8,000 tokens/min
  per model, counted as prompt + `max_completion_tokens`.
- **No fabricated evidence; no proxy biases** (pitch, loudness, accent, gender, speaking speed); silence can be
  productive and a quiet child is not necessarily disengaged.

## 2. Data

| Folder | Label used | Recordings |
|---|---|---|
| excellent / good / average | same | 2 each |
| weak, quality issue | Poor | 2 + 2 |
| demo (English, Malayalam) | unlabelled | 2 |

Each recording comes with a Zoom VTT (speaker names per line) and a one-line reviewer note in `feedbacks.pdf`.
Users never provide a transcript, so VTTs are used only offline as an answer key for speaker separation.
Two recordings (good/B435, weak/B558) were held out before any scores were seen.

## 3. Rubric from the feedback

`scripts/derive_rubric.py` split the 10 notes into 41 phrases and clustered them (TF-IDF, agglomerative) into
13 clusters, all of which matched a named theme. `scripts/triage_themes.py` classified them:

| Theme | Category | Used as |
|---|---|---|
| 8x8 background, camera/face | MEASURABLE (visual) | `professional_setup` via frame checks |
| Homework; asks children individually; considers everyone | MEASURABLE | metrics + judge |
| Rapport/intro; new joiners' introductions; warmth/energy; conclusion; interaction | JUDGEABLE | rubric dimensions with evidence requirements |
| Prior warnings about quality | NOT OBSERVABLE | limitations |

Two observations shaped weighting: "asks children individually" is praised in every note, including Poor ones,
so it barely separates labels; "more pleasant / energetic interaction" is only ever requested, never praised, and
only below Excellent. Warmth is judged only from what is said, never from voice.

`scripts/check_consistency.py` flagged quality issue/B392 (its label partly rests on prior warnings); the owner
chose to keep it. average/B516 reads like a Good note.

## 4. Pipeline decisions

- **Transcription**: large-v3 instead of turbo, because turbo detected English on a Malayalam class and produced
  English-like text. Chunks whose detected language disagrees with the class's main language are re-transcribed
  in that language, and the result is kept only if Whisper is more confident.
- **Speakers**: separation against VTTs (12 recordings) gave 0.70 mean balanced accuracy at the best threshold
  (0.35). Cluster-size confidence correlated negatively with accuracy (r = −0.25) and was replaced by stability
  across nearby thresholds (r = −0.48 for spread vs accuracy), capped at 0.75.
- **Digests**: events must quote the transcript; quotes are fuzzy-matched (works for Malayalam script); timestamps
  snap to the quoted line; unknown event types are dropped individually; truncated JSON is repaired.
- **Judge**: sees digests, not the transcript. Event lists are trimmed to the token budget by priority, with types
  taking turns, and the judge is told omitted events are not evidence of absence. Evidence must point within 20 s
  of a real event (30 s of a sampled frame for visual evidence) or the item is removed; a score with no
  surviving evidence becomes "missing".
- **Scoring**: weighted mean over scored dimensions, with missing weights redistributed; confidence is multiplied
  by weight coverage and by quality factors (speaker stability, language, transcription clarity, dropped events).
  If nothing can be scored, the label falls back to Poor with confidence 0 (an unusable recording is itself a
  quality issue in 8x8's labelling).
- **Visual**: 1 frame per 45 s in memory; 8x8 logo by template matching (no-logo frames ≤ 0.49, logo frames
  0.68–1.00 across three tutors; threshold 0.6). Lighting is not measured.

## 5. Results so far

Checked one or two recordings per stage to save time; the full run (all recordings → calibration) comes last.

| Recording | Human | ClassCritique | Notes |
|---|---|---|---|
| good/B552 | Good | Good (3.60, conf 0.85) | all 6 dimensions scored |
| weak/B482 | Poor | Poor (1.29) | opening/rapport 0/5, as the reviewer noted |
| excellent/B472 | Excellent | Good (3.41) | one level low; thresholds not yet calibrated |
| demo Malayalam | – | Good (3.06), low confidence | non-English, unstable speaker separation |

Feedback coverage on two recordings: 5/7 human themes covered; 29/29 evidence items grounded in the transcript;
3/3 quotes verified.

*Calibration (leave-one-out accuracy, confusion matrix, held-out check) will be added after the final full run.*

## 6. Limitations and next steps

- Ten labels: thresholds will move with more data; prefer collecting labels over tuning.
- Speaker separation: a child-voice-aware embedding model, or Zoom's per-speaker audio where available, would
  make participation metrics more trustworthy.
- Visual: a second logo template (B491's variant) would widen the detection margin.
- Free tier: one class takes about 5–7 minutes (summaries are rate-limited); a paid tier would cut this to about a minute.
