"""
LDCAP  —  AI Image Captioning
Streamlit inference interface
"""

import os
import time
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "LDCAP",
    page_icon   = "✦",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  —  Luxury dark aesthetic
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Root variables ───────────────────────────────────────────────────────── */
:root {
    --gold:       #f0a500;
    --gold-light: #f5c842;
    --gold-dim:   #a87200;
    --bg:         #0a0a0f;
    --bg2:        #13131f;
    --bg3:        #1c1c2e;
    --border:     rgba(240,165,0,0.18);
    --text:       #e8e8f0;
    --text-dim:   #888899;
    --white:      #ffffff;
    --radius:     16px;
    --glow:       0 0 40px rgba(240,165,0,0.12);
}

/* ── Base reset ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
.stDecoration { display: none; }

/* ── App container ────────────────────────────────────────────────────────── */
.main .block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1200px !important;
}

/* ── Hero header ──────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 4rem 0 2rem 0;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center,
        rgba(240,165,0,0.08) 0%,
        transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid var(--border);
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
    background: rgba(240,165,0,0.05);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: clamp(2.8rem, 6vw, 5rem) !important;
    font-weight: 300 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
    color: var(--white) !important;
    margin: 0 0 0.5rem 0 !important;
}
.hero-title span {
    color: var(--gold);
    font-style: italic;
}
.hero-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    margin-top: 0.75rem;
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
.gold-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 2rem auto;
}

/* ── Upload area ──────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold-dim) !important;
    background: var(--bg3) !important;
    box-shadow: var(--glow) !important;
}
[data-testid="stFileUploaderDropzone"] label {
    color: var(--text-dim) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.75rem;
    box-shadow: var(--glow);
    transition: box-shadow 0.3s ease;
}
.card:hover {
    box-shadow: 0 0 60px rgba(240,165,0,0.16);
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-label::before {
    content: '';
    display: inline-block;
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--gold);
}

/* ── Caption output ───────────────────────────────────────────────────────── */
.caption-box {
    background: linear-gradient(135deg, var(--bg3) 0%, rgba(240,165,0,0.04) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: var(--radius);
    padding: 2rem 2rem 2rem 2.25rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.caption-box::after {
    content: '✦';
    position: absolute;
    top: 1.25rem; right: 1.5rem;
    color: rgba(240,165,0,0.2);
    font-size: 1.2rem;
}
.caption-text {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    line-height: 1.55 !important;
    color: var(--white) !important;
    letter-spacing: 0.01em !important;
    margin: 0 !important;
}
.caption-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 1rem;
    display: flex;
    gap: 1.5rem;
}
.caption-meta span { color: var(--gold); }

/* ── Settings sliders ─────────────────────────────────────────────────────── */
.stSlider [data-testid="stSlider"] > div > div {
    background: var(--border) !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: var(--gold) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stSlider > label {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
}
div[data-testid="stSlider"] > div > div > div {
    background-color: var(--gold) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--gold) 0%, #c87f00 100%) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(240,165,0,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(240,165,0,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Radio buttons (decode mode) ─────────────────────────────────────────── */
.stRadio > label {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Image display ────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox [data-testid="stWidgetLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-family: 'Outfit', sans-serif !important;
}
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── History cards ────────────────────────────────────────────────────────── */
.history-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    transition: all 0.2s ease;
}
.history-item:hover {
    border-color: var(--gold-dim);
    background: var(--bg3);
}
.history-caption {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-style: italic;
    color: var(--text);
    line-height: 1.4;
}
.history-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
}

/* ── Stats row ────────────────────────────────────────────────────────────── */
.stat-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 0.4rem;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }

/* ── Alert / info ─────────────────────────────────────────────────────────── */
.stAlert {
    background: rgba(240,165,0,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state init
# ═══════════════════════════════════════════════════════════════════════════════
if 'history'    not in st.session_state: st.session_state.history    = []
if 'generator'  not in st.session_state: st.session_state.generator  = None
if 'total_runs' not in st.session_state: st.session_state.total_runs = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Model loader (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_generator(checkpoint_path, vocab_path):
    from inference import CaptionGenerator
    return CaptionGenerator(checkpoint_path, vocab_path)


def get_checkpoint():
    """Auto-detect a checkpoint in the project or nearby parent folders."""
    base_dir = Path(__file__).parent.resolve()
    search_dirs = [
        base_dir / "checkpoints",
        base_dir,
        base_dir.parent,
        base_dir.parent.parent,
    ]
    preferred_names = [
        "xe_best_model.pt",
        "scst_best_model.pt",
        "rin_ascap_pipeline.pth",
    ]

    for name in preferred_names:
        for directory in search_dirs:
            p = directory / name
            if p.exists():
                return str(p), name

    candidates = []
    for directory in search_dirs:
        for pattern in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
            candidates.extend(directory.glob(pattern))

    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        return str(newest), newest.name

    return None, None


def get_vocab():
    """Auto-detect the vocabulary JSON used by the caption model."""
    base_dir = Path(__file__).parent
    search_dirs = [
        base_dir,
        base_dir / "models",
        base_dir / "checkpoints",
    ]

    preferred_names = ["vocab.json", "vocabulary.json"]

    for directory in search_dirs:
        for name in preferred_names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate), candidate.name

    for directory in search_dirs:
        for candidate in directory.glob("*vocab*.json"):
            return str(candidate), candidate.name

    for directory in search_dirs:
        for candidate in directory.glob("*.json"):
            return str(candidate), candidate.name

    return None, None


def pil_to_b64(img, max_size=400):
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img_copy.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Adaptive Sparse Caption Prediction</div>
    <h1 class="hero-title">LDCAP</h1>
    <p class="hero-sub">Transformer-based image captioning · Bottom-up attention · SCST fine-tuned</p>
</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
ckpt_path, ckpt_name = get_checkpoint()
vocab_path, vocab_name = get_vocab()
model_ready = False

if ckpt_path and vocab_path and Path(vocab_path).exists():
    with st.spinner("Initialising model…"):
        try:
            st.session_state.generator = load_generator(ckpt_path, vocab_path)
            model_ready = True
        except Exception as e:
            st.error(f"Model load failed: {e}")
else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2.5rem;">
        <div class="card-label" style="justify-content:center">Setup required</div>
        <p style="color:var(--text-dim);font-size:0.95rem;margin:0.5rem 0 1.5rem 0;">
            Place your trained weights in the <code style="color:var(--gold)">checkpoints/</code> folder
            and your vocabulary JSON in the project root or <code style="color:var(--gold)">models/</code>.
        </p>
        <code style="font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--text-dim);">
            LDCAP/<br>
            ├── checkpoints/your_model_file.pth<br>
            ├── vocab.json<br>
            └── models/vocab.json
        </code>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="card-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label     = "Drop an image here or click to browse",
        type      = ["jpg", "jpeg", "png", "webp"],
        key       = "uploader",
        label_visibility = "collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Generation Settings</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            decode_mode = st.selectbox(
                "Decode mode",
                ["Greedy", "Beam Search (×3)", "Beam Search (×5)"],
                index=2,
            )
        with c2:
            max_tokens = st.slider("Max tokens", 10, 50, 20, step=1)

        beam_map = {"Greedy": 1, "Beam Search (×3)": 3, "Beam Search (×5)": 5}
        beam_size = beam_map[decode_mode]

        # ── Generate button ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✦  Generate Caption", disabled=not model_ready, key="gen_btn"):
            with st.spinner("Extracting features and generating caption…"):
                t0 = time.time()
                try:
                    caption, tokens = st.session_state.generator.generate(
                        img,
                        max_new_tokens = max_tokens,
                        beam_size      = beam_size,
                    )
                    elapsed = time.time() - t0
                    st.session_state.total_runs += 1

                    # Store in history
                    st.session_state.history.insert(0, {
                        "caption": caption,
                        "tokens":  tokens,
                        "time":    elapsed,
                        "mode":    decode_mode,
                        "thumb":   pil_to_b64(img, max_size=80),
                        "ts":      time.strftime("%H:%M:%S"),
                    })
                    if len(st.session_state.history) > 10:
                        st.session_state.history.pop()

                except Exception as e:
                    st.error(f"Generation error: {e}")


with right_col:
    # ── Caption result ────────────────────────────────────────────────────────
    if st.session_state.history:
        latest = st.session_state.history[0]

        st.markdown('<div class="card-label">Generated Caption</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="caption-box">
            <p class="caption-text">"{latest['caption']}"</p>
            <div class="caption-meta">
                <div>Tokens <span>{len(latest['tokens'])}</span></div>
                <div>Time <span>{latest['time']:.2f}s</span></div>
                <div>Mode <span>{latest['mode']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Token breakdown ───────────────────────────────────────────────────
        with st.expander("Token breakdown", expanded=False):
            token_html = " ".join(
                f'<span style="background:rgba(240,165,0,0.1);'
                f'border:1px solid rgba(240,165,0,0.25);'
                f'border-radius:6px;padding:2px 8px;'
                f'font-family:DM Mono,monospace;font-size:0.78rem;'
                f'color:#e8e8f0;display:inline-block;margin:2px;">{w}</span>'
                for w in latest['tokens']
                if w not in ('<PAD>','<BOS>','<EOS>','<UNK>','<unk>')
            )
            st.markdown(token_html, unsafe_allow_html=True)

        # ── Stats row ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{st.session_state.total_runs}</div>
                <div class="stat-label">Captions</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            avg_t = sum(h['time'] for h in st.session_state.history) / len(st.session_state.history)
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{avg_t:.1f}s</div>
                <div class="stat-label">Avg Time</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            avg_tok = sum(len(h['tokens']) for h in st.session_state.history) / len(st.session_state.history)
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{avg_tok:.0f}</div>
                <div class="stat-label">Avg Tokens</div>
            </div>""", unsafe_allow_html=True)

    else:
        # Placeholder when no caption yet
        st.markdown("""
        <div class="card" style="height:320px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;text-align:center;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.25;">✦</div>
            <div class="card-label" style="justify-content:center;">Awaiting image</div>
            <p style="color:var(--text-dim);font-size:0.875rem;max-width:260px;
               margin-top:0.5rem;line-height:1.6;">
                Upload an image on the left and click Generate Caption
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
if len(st.session_state.history) > 1:
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Caption History</div>', unsafe_allow_html=True)

    for item in st.session_state.history[1:]:   # skip latest (already shown)
        st.markdown(f"""
        <div class="history-item">
            <img src="data:image/jpeg;base64,{item['thumb']}"
                 style="width:56px;height:56px;object-fit:cover;
                        border-radius:8px;flex-shrink:0;
                        border:1px solid rgba(240,165,0,0.2);">
            <div>
                <div class="history-caption">"{item['caption']}"</div>
                <div class="history-time">{item['ts']} · {item['mode']} · {item['time']:.2f}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="gold-divider"></div>
<div style="text-align:center;padding:1rem 0 2rem 0;">
    <p style="font-family:'DM Mono',monospace;font-size:0.65rem;
       letter-spacing:0.15em;text-transform:uppercase;color:var(--text-dim);">
        LDCAP · Adaptive Sparse Caption Prediction · Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
