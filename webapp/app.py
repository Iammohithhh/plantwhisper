"""
PlantWhisper — Web Interface
Beautiful Gradio UI. All AI/ML logic lives in backend.py.

Deploy on HuggingFace Spaces or run locally:
    cd webapp && python app.py
"""

import gradio as gr
from pathlib import Path
from backend import analyze_plant, DIFFUSION_AVAILABLE

# ============================================
# THEME & STYLING
# ============================================

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #2d6a4f 0%, #52b788 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    block_label_text_size="sm",
    block_label_text_weight="500",
)

CSS = """
/* ── Layout ──────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
    background: #f4faf6 !important;
}
footer { display: none !important; }

/* ── Hero ─────────────────────────────────────────── */
.pw-hero {
    background: linear-gradient(135deg, #0d2818 0%, #1b4332 40%, #2d6a4f 75%, #52b788 100%);
    border-radius: 18px;
    padding: 44px 40px 36px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 12px 40px rgba(13, 40, 24, 0.35);
    position: relative;
    overflow: hidden;
}
.pw-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 30%, rgba(82,183,136,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.pw-hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.pw-hero .tagline {
    font-size: 1.1rem;
    opacity: 0.85;
    margin: 0 0 24px;
    font-weight: 300;
    font-style: italic;
}
.pw-badges {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
}
.pw-badge {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Upload panel ──────────────────────────────────── */
.upload-panel {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    height: 100%;
}

/* ── Analyze button ────────────────────────────────── */
#analyze-btn button {
    width: 100% !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 14px !important;
    border-radius: 10px !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 16px rgba(27,67,50,0.3) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
#analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(27,67,50,0.4) !important;
}
#analyze-btn button:active { transform: translateY(0) !important; }

/* ── Tips box ──────────────────────────────────────── */
.tips-box {
    background: linear-gradient(135deg, #f0faf4, #e8f5e9);
    border: 1px solid #c8e6c9;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.84rem;
    color: #2d6a4f;
    line-height: 1.6;
    margin-top: 12px;
}
.tips-box strong { color: #1b4332; }

/* ── Results panel ─────────────────────────────────── */
.results-panel {
    background: white;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Section divider ───────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 12px;
}
.section-header .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #c8e6c9, transparent);
}
.section-header span {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #52b788;
    white-space: nowrap;
}

/* ── Voice + recs cards ────────────────────────────── */
.voice-card, .recs-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Audio row ─────────────────────────────────────── */
.audio-row {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.audio-note {
    font-size: 0.78rem;
    color: #888;
    text-align: center;
    margin-top: 8px;
}
.audio-note a { color: #52b788; text-decoration: none; }
.audio-note a:hover { text-decoration: underline; }

/* ── About section ─────────────────────────────────── */
.about-card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-top: 8px;
}
.about-card h3 {
    margin: 0 0 16px;
    color: #1b4332;
    font-size: 1.1rem;
    font-weight: 700;
}
.about-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}
@media (max-width: 768px) { .about-grid { grid-template-columns: 1fr; } }
.about-item strong { color: #2d6a4f; font-size: 0.92rem; }
.about-item p {
    font-size: 0.82rem;
    color: #666;
    margin: 4px 0 0;
    line-height: 1.5;
}
.about-footer {
    font-size: 0.76rem;
    color: #999;
    margin: 16px 0 0;
    text-align: right;
}
.about-footer a { color: #52b788; text-decoration: none; }

/* ── Checkbox ──────────────────────────────────────── */
input[type="checkbox"]:checked {
    background-color: #2d6a4f !important;
    border-color: #2d6a4f !important;
}

/* ── Gradio image block ────────────────────────────── */
.image-container img { border-radius: 10px !important; }
"""

HERO_HTML = """
<div class="pw-hero">
    <h1>🌱 PlantWhisper</h1>
    <p class="tagline">"Listen to what your plants are trying to tell you"</p>
    <div class="pw-badges">
        <span class="pw-badge">🔬 SAM Segmentation</span>
        <span class="pw-badge">🧠 MobileNetV2 Classifier</span>
        <span class="pw-badge">🔥 Grad-CAM</span>
        <span class="pw-badge">🎵 Diffusion Audio</span>
        <span class="pw-badge">🗣️ LLM + TTS</span>
    </div>
</div>
"""

TIPS_HTML = """
<div class="tips-box">
    <strong>Tips for best results</strong><br>
    • Clear, well-lit photos work best<br>
    • Focus on a single leaf or plant<br>
    • Include visible damage / discoloration<br>
    • JPEG &amp; PNG both supported
</div>
"""

def _section(icon: str, label: str) -> str:
    return f"""
    <div class="section-header">
        <div class="line"></div>
        <span>{icon} {label}</span>
        <div class="line" style="background:linear-gradient(to left,#c8e6c9,transparent)"></div>
    </div>"""

VOICE_HEADER   = _section("🗣️", "Plant Voice")
RECS_HEADER    = _section("💊", "Care Recommendations")
AUDIO_HEADER   = _section("🔊", "Ultrasonic Emissions")
ABOUT_HEADER   = _section("🔬", "How It Works")

ABOUT_HTML = """
<div class="about-card">
    <h3>How PlantWhisper Works</h3>
    <div class="about-grid">
        <div class="about-item">
            <strong>1. Vision Pipeline</strong>
            <p>FastSAM/SAM isolates the plant from the background. MobileNetV2 classifies health across 38 disease classes. Grad-CAM highlights the areas the model focuses on.</p>
        </div>
        <div class="about-item">
            <strong>2. Stress → Sound</strong>
            <p>Based on Tel Aviv University research: stressed plants emit 20–150 kHz ultrasonic clicks via xylem cavitation. Drought = ~35 clicks/hour. Healthy = &lt;1 click/hour.</p>
        </div>
        <div class="about-item">
            <strong>3. Plant Voice &amp; Advice</strong>
            <p>Groq/Llama 3.3 70B generates emotionally appropriate inner speech. Edge-TTS converts it to audio with stress-modulated pitch and rate. Care advice is tailored to severity.</p>
        </div>
    </div>
    <p class="about-footer">
        Created by <strong>Mohith</strong> · IIT Bombay · 2026 ·&nbsp;
        <a href="https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3" target="_blank">
            Based on Khait et al. 2023 (Cell)
        </a>
    </p>
</div>
"""

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(css=CSS, title="🌱 PlantWhisper", theme=_THEME) as demo:

    # ── Hero ──────────────────────────────────────────
    gr.HTML(HERO_HTML)

    # ── Row 1: Upload + Results ───────────────────────
    with gr.Row(equal_height=False):

        # LEFT: Upload controls
        with gr.Column(scale=1, min_width=300):
            input_image = gr.Image(
                label="Upload Plant Photo",
                type="numpy",
                height=310,
                sources=["upload", "clipboard"],
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Plant",
                variant="primary",
                size="lg",
                elem_id="analyze-btn",
            )
            use_diffusion = gr.Checkbox(
                label="Use Diffusion Model for audio",
                value=DIFFUSION_AVAILABLE,
                interactive=True,
                info="Requires diffusion_model.pt in webapp/",
            )
            gr.HTML(TIPS_HTML)

        # RIGHT: Markdown results + image pair
        with gr.Column(scale=2, min_width=460):
            output_status = gr.Markdown(
                value="*Upload a plant photo and click **Analyze Plant** to begin.*"
            )
            with gr.Row():
                output_segmented = gr.Image(
                    label="Segmented Plant",
                    height=230,
                    show_download_button=True,
                )
                output_gradcam = gr.Image(
                    label="Attention Map (Grad-CAM)",
                    height=230,
                    show_download_button=True,
                )

    # ── Row 2: Plant Voice + Care Recommendations ─────
    gr.HTML(VOICE_HEADER)
    with gr.Row():
        with gr.Column(scale=1):
            output_speech = gr.Textbox(
                label="What your plant is saying",
                lines=4,
                placeholder="The plant's inner voice will appear here...",
                show_copy_button=True,
            )
            output_voice = gr.Audio(
                label="Plant Voice (TTS)",
                type="filepath",
            )

        with gr.Column(scale=1):
            gr.HTML(RECS_HEADER)
            output_recommendations = gr.Textbox(
                label="Actionable Care Instructions",
                lines=9,
                show_copy_button=True,
                elem_id="recommendations",
            )

    # ── Row 3: Ultrasonic Audio ───────────────────────
    gr.HTML(AUDIO_HEADER)
    with gr.Row():
        output_ultrasonic = gr.Audio(
            label="Synthetic Pops",
            type="filepath",
        )
        output_diffusion = gr.Audio(
            label="Diffusion-Generated Pops",
            type="filepath",
        )
    gr.HTML("""
    <p class="audio-note">
        Clicks are pitch-shifted representations of the ultrasonic sounds stressed plants actually make ·
        <a href="https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3" target="_blank">Khait et al. 2023 (Cell)</a>
    </p>
    """)

    # ── About ──────────────────────────────────────────
    gr.HTML(ABOUT_HEADER)
    gr.HTML(ABOUT_HTML)

    # ── Wire up ────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_plant,
        inputs=[input_image, use_diffusion],
        outputs=[
            output_segmented,
            output_gradcam,
            output_status,
            output_recommendations,
            output_speech,
            output_voice,
            output_ultrasonic,
            output_diffusion,
        ],
    )

    # ── Example images ─────────────────────────────────
    _assets = Path(__file__).parent.parent / "assets"
    _examples = [str(p) for p in sorted(_assets.glob("*.jfif")) if p.is_file()] if _assets.exists() else []
    if _examples:
        gr.Examples(
            examples=[[f] for f in _examples],
            inputs=[input_image],
        )


# ============================================
# LAUNCH
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
