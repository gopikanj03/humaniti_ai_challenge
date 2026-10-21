"""Look and feel for the Streamlit app: fonts, light/dark palettes, and a mouse-reactive 3D background.

Everything visual lives here so app.py stays about behaviour. Colours are CSS variables, switched by
`data-theme` on the page root, so the background canvas and all widgets follow the same toggle.
"""
from __future__ import annotations

import json

PALETTES = {
    # Pastel light theme
    "light": {
        "bg": "#F6F4FF", "bg2": "#EEF8F5", "surface": "rgba(255,255,255,0.72)", "surface-strong": "#FFFFFF",
        "text": "#2A2840", "muted": "#6E6A88", "border": "rgba(139,124,246,0.20)",
        "accent": "#8B7CF6", "accent-2": "#F4A6C8", "accent-3": "#8ED6C4", "accent-4": "#FFD39B",
        "accent-text": "#FFFFFF", "shadow": "0 10px 40px rgba(110,96,200,0.12)",
        "input": "rgba(255,255,255,0.9)",
    },
    # Dark theme
    "dark": {
        "bg": "#0D0E1A", "bg2": "#141629", "surface": "rgba(26,28,48,0.72)", "surface-strong": "#1B1D33",
        "text": "#ECEAF8", "muted": "#A4A0BE", "border": "rgba(169,155,255,0.22)",
        "accent": "#A99BFF", "accent-2": "#F08DBA", "accent-3": "#67CDB3", "accent-4": "#F2C27E",
        "accent-text": "#14152A", "shadow": "0 10px 40px rgba(0,0,0,0.45)",
        "input": "rgba(13,14,26,0.75)",
    },
}

FONTS_URL = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800"
             "&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap")


def css(theme: str) -> str:
    # Streamlit's markdown ends an HTML block at the first blank line, so the stylesheet must have none.
    return "\n".join(line for line in _css(theme).splitlines() if line.strip())


def _css(theme: str) -> str:
    p = PALETTES[theme]
    variables = "\n".join(f"  --{k}: {v};" for k, v in p.items())
    return f"""
<link rel="stylesheet" href="{FONTS_URL}">
<style>
:root {{
{variables}
  --font-body: 'Plus Jakarta Sans', system-ui, sans-serif;
  --font-display: 'Bricolage Grotesque', 'Plus Jakarta Sans', sans-serif;
  --radius: 18px;
}}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important; color: var(--text); font-family: var(--font-body);
}}
/* Let the 3D canvas show through the app layers */
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main, [data-testid="stHeader"] {{
  background: transparent !important;
}}
[data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; }}
.stApp {{ background: transparent !important; }}
/* Streamlit gives <html> zero height, so the colour (which always fills the viewport) must be set
   explicitly, and the gradient attached to the viewport rather than to the html box. */
html {{
  background-color: var(--bg) !important; min-height: 100%;
  background-image: radial-gradient(1200px 800px at 15% 10%, var(--bg2), var(--bg)) !important;
  background-attachment: fixed !important;
}}

/* No sidebar, no Streamlit chrome */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{
  display: none !important;
}}
.block-container {{ max-width: 920px; padding-top: 2.2rem; padding-bottom: 4rem; }}

/* Typography */
/* Override Streamlit's font everywhere except its icon font (icons are ligatures: overriding the
   font prints the icon's name, e.g. "upload") and code. */
.stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre):not(.material-symbols-rounded) {{
  font-family: var(--font-body);
}}
.stApp p, .stApp li, .stApp label, .stMarkdown {{ color: var(--text); }}
h1.app-title, h1.app-title * {{
  font-family: var(--font-display) !important; font-weight: 800 !important; letter-spacing: -0.02em;
  -webkit-text-fill-color: transparent !important; color: transparent !important;
}}
h1.app-title {{
  text-align: center; font-size: clamp(2.4rem, 5.5vw, 3.8rem) !important; line-height: 1.05;
  margin: 0.6rem 0 0.4rem; padding: 0 !important;
  background: linear-gradient(120deg, var(--accent) 0%, var(--accent-2) 50%, var(--accent-3) 100%);
  -webkit-background-clip: text; background-clip: text;
}}
[data-testid="stHeaderActionElements"], h1.app-title a {{ display: none !important; }}
.app-brand {{ display: flex; align-items: center; justify-content: center; gap: 0.9rem; margin-top: 0.6rem; }}
.app-brand h1.app-title {{ margin: 0; }}
.app-logo {{
  width: clamp(3.6rem, 8vw, 5.2rem); height: auto; flex-shrink: 0;
  filter: drop-shadow(0 6px 14px color-mix(in srgb, var(--accent) 30%, transparent));
}}
.stApp p.app-subtitle {{ text-align: center; color: var(--muted); font-size: 1.05rem; margin-bottom: 1.8rem; }}
.stApp p.app-note {{ text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 0.9rem; }}

/* Glass cards: tabs panel, results, status */
.stTabs, [data-testid="stExpander"], [data-testid="stStatusWidget"], div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .result-card) {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}}
.stTabs {{ padding: 0.6rem 1.4rem 1.4rem; }}

/* Settings-style tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem; justify-content: center; background: transparent; border-bottom: 1px solid var(--border);
  padding-bottom: 0.2rem;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: var(--font-body); font-weight: 600; font-size: 0.98rem; color: var(--muted);
  padding: 0.7rem 1.4rem; border-radius: 12px 12px 0 0; background: transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); background: color-mix(in srgb, var(--accent) 8%, transparent); }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent); height: 3px; border-radius: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.2rem; }}

/* Inputs */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--input); border: 1.5px dashed var(--accent); border-radius: 14px; color: var(--text);
}}
[data-testid="stFileUploaderDropzone"] * {{ color: var(--text) !important; }}
[data-testid="stFileUploaderDropzone"] button {{
  background: transparent; border: 1px solid var(--border); color: var(--text);
}}
.stTextInput input {{
  background: var(--input) !important; color: var(--text) !important; border-radius: 12px !important;
  border: 1px solid var(--border) !important; padding: 0.7rem 0.9rem !important;
}}
.stTextInput [data-baseweb="input"] {{ background: transparent !important; border: none !important; }}
.stTextInput input::placeholder {{ color: var(--muted) !important; }}
/* One recording at a time: hide Streamlit's "Add files" (+) button; remove the file (x) to choose another. */
[data-testid="stFileUploader"] button[aria-label="Add files"] {{ display: none !important; }}
[data-testid="stUploadedFile"], [data-testid="stFileUploaderFile"] {{
  color: var(--text); background: var(--surface-strong) !important; border: 1px solid var(--border); border-radius: 10px;
}}
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] {{ color: var(--text) !important; }}

/* Buttons */
.stButton > button {{
  width: 100%; border: none; border-radius: 14px; padding: 0.75rem 1rem; font-weight: 700; font-size: 1rem;
  color: var(--accent-text); background: linear-gradient(120deg, var(--accent), var(--accent-2));
  box-shadow: 0 8px 24px color-mix(in srgb, var(--accent) 35%, transparent); transition: transform .15s, filter .15s;
}}
.stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.05); color: var(--accent-text); }}
.stButton > button:disabled {{ opacity: 0.45; box-shadow: none; transform: none; }}

/* Theme toggle */
[data-testid="stToggle"] label, [data-testid="stToggle"] p {{ color: var(--muted) !important; font-size: 0.85rem; }}

/* Metrics, alerts, expanders, captions */
[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 0.9rem 1rem;
  backdrop-filter: blur(10px);
}}
[data-testid="stMetricLabel"] * {{ color: var(--muted) !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; font-family: var(--font-display); }}
[data-testid="stAlert"] {{ border-radius: 14px; }}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
  background: transparent !important; border: none !important;
}}
[data-testid="stExpander"] summary:hover {{ background: color-mix(in srgb, var(--accent) 8%, transparent) !important; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--text) !important; font-weight: 600; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; }}
[data-testid="stProgress"] > div > div > div {{ background: linear-gradient(90deg, var(--accent), var(--accent-3)); }}
code {{ background: color-mix(in srgb, var(--accent) 14%, transparent) !important; color: var(--text) !important; }}
/* ---------- results page ---------- */
.cc-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: var(--text);
}}
.cc-muted {{ color: var(--muted) !important; }}
.cc-small {{ font-size: 0.8rem; }}
.stApp h2.cc-h2 {{ font-family: var(--font-display) !important; font-size: 1.4rem; margin: 1.6rem 0 0.8rem; color: var(--text); }}
.stApp h3.cc-h3 {{ font-size: 1.05rem; font-weight: 700; margin: 0 0 0.6rem; color: var(--text); padding: 0; }}
.cc-grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0; }}
@media (max-width: 760px) {{ .cc-grid2, .cc-dims {{ grid-template-columns: 1fr !important; }} .cc-hero-main {{ flex-direction: column; }} }}
/* progress tracker */
.cc-progress-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.8rem; }}
.cc-progress-title {{ font-weight: 700; }}
.cc-pct {{ font-family: var(--font-display) !important; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }}
.cc-live {{ display: flex; align-items: center; gap: 0.65rem; margin-top: 1rem; font-weight: 600; color: var(--text); }}
.cc-live.done {{ color: var(--accent-3); }}
.cc-live-icon {{ font-weight: 800; }}
.cc-spinner {{
  width: 1.05rem; height: 1.05rem; border-radius: 50%; flex-shrink: 0;
  border: 2.5px solid color-mix(in srgb, var(--accent) 22%, transparent); border-top-color: var(--accent);
  animation: cc-spin 0.9s linear infinite;
}}
@keyframes cc-spin {{ to {{ transform: rotate(360deg); }} }}
.cc-rotator {{ position: relative; display: inline-block; height: 1.5em; flex: 1; overflow: hidden; }}
.cc-rotator span {{
  position: absolute; left: 0; top: 0; white-space: nowrap; opacity: 0;
  animation-iteration-count: infinite; animation-fill-mode: both; animation-timing-function: ease-in-out;
}}
@keyframes cc-cycle1 {{ 0%, 100% {{ opacity: 1; transform: none; }} }}
@keyframes cc-cycle2 {{ 0% {{ opacity: 0; transform: translateY(6px); }} 6%, 44% {{ opacity: 1; transform: none; }}
                       50%, 100% {{ opacity: 0; transform: translateY(-6px); }} }}
@keyframes cc-cycle3 {{ 0% {{ opacity: 0; transform: translateY(6px); }} 4%, 29% {{ opacity: 1; transform: none; }}
                       33.3%, 100% {{ opacity: 0; transform: translateY(-6px); }} }}
@keyframes cc-cycle4 {{ 0% {{ opacity: 0; transform: translateY(6px); }} 3%, 22% {{ opacity: 1; transform: none; }}
                       25%, 100% {{ opacity: 0; transform: translateY(-6px); }} }}
.cc-detail {{ margin: 0.35rem 0 0 1.7rem; font-size: 0.85rem; color: var(--muted); }}
@media (prefers-reduced-motion: reduce) {{ .cc-spinner, .cc-rotator span {{ animation: none !important; }}
  .cc-rotator span:first-child {{ opacity: 1; }} }}
.cc-steps {{ list-style: none; display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem; padding: 0; margin: 0 0 1rem; }}
.cc-step {{ display: flex; align-items: center; gap: 0.45rem; font-size: 0.88rem; color: var(--muted); }}
.cc-dot {{
  width: 1.55rem; height: 1.55rem; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 700; border: 1.5px solid var(--border); background: var(--input);
}}
.cc-step.done {{ color: var(--text); }}
.cc-step.done .cc-dot {{ background: var(--accent-3); border-color: var(--accent-3); color: var(--accent-text); }}
.cc-step.active {{ color: var(--accent); font-weight: 700; }}
.cc-step.active .cc-dot {{ border-color: var(--accent); color: var(--accent); animation: cc-pulse 1.4s ease-in-out infinite; }}
@keyframes cc-pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 45%, transparent); }}
                      50% {{ box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent) 0%, transparent); }} }}
.cc-bar, .cc-scorebar {{ height: 8px; border-radius: 99px; background: color-mix(in srgb, var(--accent) 14%, transparent); overflow: hidden; }}
.cc-bar-fill, .cc-scorebar > div {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transition: width .4s; }}
.cc-scorebar.conf > div {{ background: linear-gradient(90deg, var(--accent-3), var(--accent)); }}
/* hero */
.cc-hero {{ border-top: 5px solid var(--label); padding: 1.5rem 1.6rem; }}
.cc-hero-main {{ display: flex; gap: 2rem; align-items: center; }}
.cc-kicker {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }}
.cc-label {{
  font-family: var(--font-display) !important; font-weight: 800; font-size: clamp(2.3rem, 6vw, 3.4rem); line-height: 1;
  color: var(--label); margin-top: 0.2rem; filter: saturate(1.2) brightness(0.92);
}}
.cc-hero-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; flex: 1; }}
.cc-stat-value {{ font-family: var(--font-display) !important; font-size: 1.8rem; font-weight: 700; margin: 0.15rem 0 0.4rem; }}
.cc-stat-value span {{ font-size: 0.95rem; color: var(--muted); font-weight: 600; }}
.cc-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.1rem 0 0.4rem; }}
.cc-meta span {{ font-size: 0.8rem; padding: 0.25rem 0.7rem; border-radius: 99px; background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--text); }}
.stApp p.cc-summary {{ font-size: 1.02rem; line-height: 1.6; margin: 0.8rem 0 0.2rem; }}
.cc-banner {{
  margin-top: 0.9rem; padding: 0.75rem 1rem; border-radius: 12px; font-size: 0.9rem;
  background: color-mix(in srgb, var(--accent-4) 28%, transparent); border: 1px solid color-mix(in srgb, var(--accent-4) 60%, transparent);
}}
.cc-rules {{
  margin-top: 0.9rem; padding: 0.75rem 1rem; border-radius: 12px; font-size: 0.9rem;
  background: color-mix(in srgb, var(--accent-2) 16%, transparent); border: 1px solid color-mix(in srgb, var(--accent-2) 45%, transparent);
}}
.cc-rules-head {{ font-weight: 700; margin-bottom: 0.3rem; }}
.cc-rules ul {{ margin: 0; padding-left: 1.1rem; }}
.cc-qi {{
  display: inline-block; margin-top: 0.45rem; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem;
  border-radius: 99px; background: color-mix(in srgb, var(--accent-2) 30%, transparent); color: var(--text);
}}
.cc-reasons {{ margin-top: 0.7rem; font-size: 0.88rem; color: var(--muted); }}
.cc-reasons summary {{ cursor: pointer; font-weight: 600; }}
/* evidence */
.cc-ev {{ list-style: none; padding: 0; margin: 0.4rem 0 0; }}
.cc-ev li {{ display: flex; gap: 0.55rem; align-items: baseline; padding: 0.45rem 0; border-bottom: 1px dashed var(--border); font-size: 0.93rem; line-height: 1.5; }}
.cc-ev li:last-child {{ border-bottom: none; }}
.cc-chip {{
  flex-shrink: 0; font-size: 0.76rem; font-weight: 700; padding: 0.12rem 0.55rem; border-radius: 99px;
  background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); font-variant-numeric: tabular-nums;
}}
.cc-src {{
  flex-shrink: 0; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
  padding: 0.14rem 0.45rem; border-radius: 6px; color: var(--text);
  background: color-mix(in srgb, var(--accent-3) 26%, transparent);
}}
.cc-src.metric {{ background: color-mix(in srgb, var(--accent-4) 34%, transparent); }}
.cc-src.visual {{ background: color-mix(in srgb, var(--accent-2) 26%, transparent); }}
/* dimensions */
.cc-dims {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.cc-dims .cc-card {{ margin-bottom: 0; }}
.cc-dim-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.5rem; }}
.cc-dim-name {{ font-weight: 700; font-size: 1rem; }}
.cc-dim-score {{ font-family: var(--font-display) !important; font-size: 1.5rem; font-weight: 700; white-space: nowrap; }}
.cc-dim-score span {{ font-size: 0.85rem; color: var(--muted); }}
.stApp p.cc-rationale {{ font-size: 0.92rem; color: var(--muted); margin: 0.6rem 0 0.2rem; line-height: 1.5; }}
/* pills, signals, limitations, metrics */
.cc-pill {{ font-size: 0.72rem; font-weight: 700; padding: 0.18rem 0.6rem; border-radius: 99px; white-space: nowrap; }}
.cc-pill.used {{ background: color-mix(in srgb, var(--accent-3) 30%, transparent); color: var(--text); }}
.cc-pill.partial {{ background: color-mix(in srgb, var(--accent-4) 40%, transparent); color: var(--text); }}
.cc-pill.missing {{ background: color-mix(in srgb, var(--accent-2) 28%, transparent); color: var(--text); }}
.cc-signals {{ list-style: none; padding: 0; margin: 0; }}
.cc-signals li {{ display: flex; gap: 0.7rem; align-items: flex-start; padding: 0.45rem 0; }}
.cc-sig-name {{ font-weight: 600; font-size: 0.92rem; }}
.cc-lims {{ margin: 0; padding-left: 1.1rem; font-size: 0.9rem; line-height: 1.55; }}
.cc-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; color: var(--text); }}
.cc-table th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; padding: 0.5rem; border-bottom: 1px solid var(--border); }}
.cc-table td {{ padding: 0.5rem; border-bottom: 1px dashed var(--border); vertical-align: top; }}
.cc-rel {{ font-size: 0.75rem; font-weight: 700; }}
.cc-rel.high {{ color: var(--accent-3); }} .cc-rel.medium {{ color: var(--accent-4); }} .cc-rel.low {{ color: var(--accent-2); }}
[data-testid="stDownloadButton"] button {{
  border-radius: 14px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-weight: 600;
}}
</style>
"""


def background_js(theme: str) -> str:
    """Injects (once) a full-screen canvas into the parent page and keeps its theme in sync.

    Streamlit renders components in an iframe that is re-created on every rerun, so the animation code is
    added to the PARENT document as a <script>; it then survives reruns. Pointer events pass through.
    """
    colours = {name: {"accents": [p["accent"], p["accent-2"], p["accent-3"], p["accent-4"]], "line": p["accent"]}
               for name, p in PALETTES.items()}
    return f"""
<script>
(function () {{
  const doc = window.parent.document;
  doc.documentElement.setAttribute('data-theme', '{theme}');
  if (doc.getElementById('bg3d-script')) return;          // already running in the parent page
  const s = doc.createElement('script');
  s.id = 'bg3d-script';
  s.textContent = `(${{bg3d.toString()}})(${{JSON.stringify({json.dumps(colours)})}});`;
  doc.body.appendChild(s);

  function bg3d(COLOURS) {{
    const cv = document.createElement('canvas');
    cv.id = 'bg3d';
    Object.assign(cv.style, {{position: 'fixed', inset: '0', width: '100vw', height: '100vh',
                             zIndex: '0', pointerEvents: 'none'}});
    document.body.prepend(cv);
    const ctx = cv.getContext('2d');
    let W = 0, H = 0, DPR = Math.min(window.devicePixelRatio || 1, 2);
    function resize() {{ W = innerWidth; H = innerHeight; cv.width = W * DPR; cv.height = H * DPR;
                        ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }}
    addEventListener('resize', resize); resize();

    // Mouse -> target rotation; eased every frame so the scene glides rather than snaps.
    let tx = 0, ty = 0, rx = 0, ry = 0;
    addEventListener('mousemove', e => {{ tx = (e.clientX / W - 0.5) * 0.9; ty = (e.clientY / H - 0.5) * 0.6; }});

    const rnd = (a, b) => a + Math.random() * (b - a);
    const nodes = Array.from({{length: 64}}, () => ({{x: rnd(-520, 520), y: rnd(-320, 320), z: rnd(-360, 360),
                                                   c: Math.floor(Math.random() * 4), p: rnd(0, 6.28)}}));
    const orbs = Array.from({{length: 5}}, (_, i) => ({{x: rnd(-600, 600), y: rnd(-360, 360), z: rnd(150, 500),
                                                     r: rnd(160, 300), c: i % 4, p: rnd(0, 6.28)}}));

    function rotate(x, y, z, a, b) {{               // yaw a (around Y), pitch b (around X)
      const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
      const x1 = x * ca + z * sa, z1 = -x * sa + z * ca;
      return [x1, y * cb - z1 * sb, y * sb + z1 * cb];
    }}
    function project(x, y, z) {{ const f = 900 / (900 + z); return [W / 2 + x * f, H / 2 + y * f, f]; }}
    function hexA(hex, a) {{ const n = parseInt(hex.slice(1), 16);
      return `rgba(${{n >> 16 & 255}},${{n >> 8 & 255}},${{n & 255}},${{a}})`; }}

    let t = 0;
    function frame() {{
      t += 0.004;
      rx += (tx - rx) * 0.05; ry += (ty - ry) * 0.05;
      const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const C = COLOURS[theme], dark = theme === 'dark';
      ctx.clearRect(0, 0, W, H);
      const yaw = rx + Math.sin(t) * 0.12, pitch = ry + 0.55;

      // Soft colour orbs far back (parallax: they move less than near objects)
      for (const o of orbs) {{
        const [x, y, z] = rotate(o.x, o.y + Math.sin(t * 2 + o.p) * 20, o.z, yaw * 0.4, (pitch - 0.55) * 0.4);
        const [px, py, f] = project(x, y, z);
        const g = ctx.createRadialGradient(px, py, 0, px, py, o.r * f);
        g.addColorStop(0, hexA(C.accents[o.c], dark ? 0.16 : 0.30)); g.addColorStop(1, hexA(C.accents[o.c], 0));
        ctx.fillStyle = g; ctx.fillRect(px - o.r, py - o.r, o.r * 2, o.r * 2);
      }}

      // Floating 8x8 board
      const N = 8, S = 70, bob = Math.sin(t * 1.5) * 14;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {{
        const corners = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) => {{
          const [x, y, z] = rotate((a - N / 2) * S, 120 + bob, (b - N / 2) * S, yaw, pitch);
          return project(x, y, z);
        }});
        ctx.beginPath(); corners.forEach(([px, py], k) => k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)); ctx.closePath();
        if ((i + j) % 2 === 0) {{ ctx.fillStyle = hexA(C.accents[(i + j) % 4], dark ? 0.07 : 0.13); ctx.fill(); }}
        ctx.strokeStyle = hexA(C.line, dark ? 0.16 : 0.18); ctx.lineWidth = 1; ctx.stroke();
      }}

      // Drifting nodes with depth-faded links
      const pts = nodes.map(n => {{
        const [x, y, z] = rotate(n.x, n.y + Math.sin(t * 3 + n.p) * 18, n.z, yaw, pitch - 0.55);
        return [...project(x, y, z), n.c];
      }});
      for (let a = 0; a < pts.length; a++) for (let b = a + 1; b < pts.length; b++) {{
        const dx = pts[a][0] - pts[b][0], dy = pts[a][1] - pts[b][1], d = Math.hypot(dx, dy);
        if (d < 130) {{
          ctx.strokeStyle = hexA(C.line, (1 - d / 130) * (dark ? 0.22 : 0.20) * Math.min(pts[a][2], 1));
          ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
        }}
      }}
      for (const [px, py, f, c] of pts) {{
        ctx.fillStyle = hexA(C.accents[c], dark ? 0.75 : 0.85);
        ctx.beginPath(); ctx.arc(px, py, 2.6 * f, 0, 6.283); ctx.fill();
      }}
      if (!still) requestAnimationFrame(frame);
    }}
    // Respect users who ask for less motion: keep redrawing only on mouse moves, no idle animation.
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (still) addEventListener('mousemove', () => requestAnimationFrame(frame));
    requestAnimationFrame(frame);
  }}
}})();
</script>
"""
