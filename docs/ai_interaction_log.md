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
