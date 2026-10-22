# ClassCritique: approach

## Approach note

**Approach.** ClassCritique turns a recording into evidence before judging it. Groq Whisper transcribes; local
voice clustering separates the tutor from the children; sampled frames are checked locally for the tutor's camera
and the 8x8 background. gpt-oss-20b condenses each 10-minute section into timestamped events whose quotes must
exist in the transcript, then scores seven rubric areas derived from 8x8's feedback, citing those moments.
Python computes the weighted score, applies strict rules (recording-quality problems give Poor; teaching
mistakes, no homework or recap, too few questions or too little praise cap the label) and picks the label.

**What worked.** Evidence checks: unverifiable events and citations are dropped, so every observation points to a
real moment. Rubric themes from clustering reviewer feedback match what reviewers care about (rapport, individual
attention, homework, warmth, the 8x8 background). whisper-large-v3 with a language lock handled a Malayalam class.
Counting names and praise over the whole transcript matched Zoom's own transcripts closely.

**What failed or was weak.** Label agreement is modest: 4 of 8 analysed classes matched 8x8 at the last full
re-scoring, and Excellent classes tend to come out Good. Voice separation averaged 0.70 balanced accuracy, so
participation is reported cautiously. The judge, reading summaries without the board, once took a puzzle remark
("with the knight we cannot do a checkmate") for a wrong rule; a focused second check now withdraws such cases.
The free tier (8,000 tokens/min) forced strict budgets and slow runs.

**Limitations.** Ten labels cannot validate weights or thresholds, and the full run with the latest changes is
still to be done. Video is sampled every 45 s. Tone of voice and lighting are not measured, to avoid bias. Prior
warnings are not observable. Non-English classes get lower confidence.

---

## 1. Goal and constraints

Given one recording, output exactly one of Poor / Average / Good / Excellent plus specific, timestamped
observations. Constraints shaped most decisions:

- **Confidential children's data**: nothing private is committed or logged; frames stay local; children are
  pseudonymised; uploads leave nothing on disk.
- **Groq free tier only, open-source models** (whisper-large-v3 and gpt-oss-20b; gpt-oss-120b only as a backup when 20b's daily limit is used up): 8,000 tokens/min
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
  surviving evidence becomes "missing". Each reported teaching mistake gets a short second check that sees what
  was said around it (not the judge's claim) and must classify it as a general rule or a remark about the
  position; only a false general rule counts, and if one is withdrawn the judge runs once more with that note.
  Whole-class counts of children called by name and of praise are computed in Python from the transcript, because
  the summaries keep only about 20 moments per section. Temperature is 0 for repeatable results.
- **Scoring**: weighted mean over scored dimensions, with missing weights redistributed; confidence is multiplied
  by weight coverage and by quality factors (speaker stability, language, transcription clarity, dropped events).
  If nothing can be scored, the label falls back to Poor with confidence 0 (an unusable recording is itself a
  quality issue in 8x8's labelling).
- **Visual**: 1 frame per 45 s in memory; 8x8 logo by template matching (no-logo frames ≤ 0.49, logo frames
  0.68–1.00 across three tutors; threshold 0.6). Lighting is not measured.



## 5. Limitations and next steps

- Ten labels: thresholds will move with more data; prefer collecting labels over tuning.
- Speaker separation: a child-voice-aware embedding model, or Zoom's per-speaker audio where available, would
  make participation metrics more trustworthy.
- Visual: tutors use slightly different 8x8 backgrounds; three logo templates cover the ones seen so far, and a
  new variant would need its own template.
- Teaching accuracy: the judge reads summaries, not the board, so claims about a position cannot be checked;
  each reported mistake gets a second check, but a real mistake about a position would be missed.
- Free tier: one class takes about 5–7 minutes (summaries are rate-limited); a paid tier would cut this to about a minute.
