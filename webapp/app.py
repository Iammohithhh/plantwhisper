"""
🌱 PlantWhisper Web App
"Listen to what your plants are trying to tell you"

Deploy on HuggingFace Spaces or run locally with: python app.py
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import asyncio
import nest_asyncio
import random
import tempfile
import os
import warnings

warnings.filterwarnings('ignore')
nest_asyncio.apply()

# ============================================
# CONFIGURATION
# ============================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
TTS_VOICE = "en-US-AriaNeural"
TARGET_SR = 22050
PEAK_FREQ_HZ = 53000
TARGET_FREQ_HZ = 1000
HEALTHY_POPS_PER_HOUR = 1.0
DROUGHT_PEAK_POPS_PER_HOUR = 35.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🌱 PlantWhisper starting on {device}...")

# ============================================
# LOAD MODELS
# ============================================

# --- Plant Classifier ---
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification

print("Loading plant classifier...")
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_NAME)
classifier = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
classifier.to(device)
classifier.eval()
print("✓ Classifier loaded")

# --- SAM Segmentation ---
print("Loading SAM segmentation...")
try:
    from segment_anything import sam_model_registry, SamPredictor
    
    SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
    if not os.path.exists(SAM_CHECKPOINT):
        print("Downloading SAM checkpoint...")
        os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything/{SAM_CHECKPOINT}")
    
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    SAM_AVAILABLE = True
    print("✓ SAM loaded")
except Exception as e:
    print(f"⚠ SAM not available: {e}")
    SAM_AVAILABLE = False

# --- Grad-CAM ---
print("Loading Grad-CAM...")
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    class HFModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x).logits
    
    wrapped_model = HFModelWrapper(classifier)
    wrapped_model.eval()
    # HF MobileNetV2Model stores blocks in .mobilenet_v2.layer (ModuleList)
    # Fall back to last conv block if attribute path differs between versions
    try:
        target_layer = classifier.mobilenet_v2.layer[-1]
    except (AttributeError, IndexError):
        # torchvision-style path
        target_layer = classifier.mobilenet_v2.features[-1]
    grad_cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    GRADCAM_AVAILABLE = True
    print("✓ Grad-CAM loaded")
except Exception as e:
    print(f"⚠ Grad-CAM not available: {e}")
    GRADCAM_AVAILABLE = False

# --- Groq LLM ---
print("Loading Groq client...")
try:
    import groq
    groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    GROQ_AVAILABLE = bool(GROQ_API_KEY)
    print("✓ Groq client ready" if GROQ_AVAILABLE else "⚠ Groq API key not set")
except Exception as e:
    print(f"⚠ Groq not available: {e}")
    GROQ_AVAILABLE = False
    groq_client = None

# --- Edge TTS ---
print("Loading TTS...")
try:
    import edge_tts
    TTS_AVAILABLE = True
    print("✓ Edge TTS loaded")
except:
    TTS_AVAILABLE = False
    print("⚠ Edge TTS not available")

# --- Audio libraries ---
try:
    from scipy import signal
    import soundfile as sf
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False

print("\n🌱 PlantWhisper ready!\n")

# ============================================
# CORE FUNCTIONS
# ============================================

def segment_plant(image: np.ndarray) -> tuple:
    """Segment plant using SAM or fallback to simple threshold."""
    if SAM_AVAILABLE:
        try:
            sam_predictor.set_image(image)
            h, w = image.shape[:2]
            center = np.array([[w // 2, h // 2]])
            masks, scores, _ = sam_predictor.predict(
                point_coords=center,
                point_labels=np.array([1]),
                multimask_output=True
            )
            mask = masks[np.argmax(scores)]
            segmented = image.copy()
            segmented[~mask] = [240, 240, 240]
            return segmented, mask
        except:
            pass
    
    # Fallback: simple green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (85, 255, 255))
    mask = mask > 0
    segmented = image.copy()
    segmented[~mask] = [240, 240, 240]
    return segmented, mask


def classify_plant(image: np.ndarray) -> dict:
    """Classify plant health."""
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = classifier(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    top_prob, top_idx = probs[0].topk(1)
    label = classifier.config.id2label[top_idx.item()]
    confidence = top_prob.item()
    is_healthy = "healthy" in label.lower()

    return {
        'is_healthy': is_healthy,
        'confidence': confidence,
        'label': label,
    }


def generate_gradcam(image: np.ndarray) -> np.ndarray:
    """Generate attention heatmap."""
    if not GRADCAM_AVAILABLE:
        return image
    
    try:
        pil_image = Image.fromarray(image)
        inputs = processor(images=pil_image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(device)
        
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
        
        image_float = image.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
        return cam_image
    except:
        return image


def estimate_stress(classification: dict) -> float:
    """Estimate stress level 0-1."""
    if classification['is_healthy']:
        stress = 0.1 * (1 - classification['confidence'])
    else:
        stress = 0.4 + (0.5 * classification['confidence'])
    return np.clip(stress, 0.0, 1.0)


def stress_to_pop_rate(stress: float) -> float:
    """Convert stress to pops/hour (hump-shaped curve)."""
    if stress < 0.1:
        return HEALTHY_POPS_PER_HOUR + stress * 30
    else:
        hump = 4 * stress * (1 - stress)
        return HEALTHY_POPS_PER_HOUR + (DROUGHT_PEAK_POPS_PER_HOUR - HEALTHY_POPS_PER_HOUR) * hump


def get_stress_category(stress: float) -> tuple:
    """Get stress category and color."""
    if stress < 0.15:
        return "Healthy", "#2ecc71"
    elif stress < 0.35:
        return "Mild Stress", "#f1c40f"
    elif stress < 0.55:
        return "Moderate Stress", "#e67e22"
    elif stress < 0.75:
        return "Severe Stress", "#e74c3c"
    else:
        return "Critical", "#8e44ad"


# ============================================
# CARE RECOMMENDATIONS
# ============================================

def get_care_recommendations(stress_level: float, is_healthy: bool) -> str:
    """Generate care recommendations using LLM or fallback."""
    
    category, _ = get_stress_category(stress_level)
    
    if GROQ_AVAILABLE and groq_client:
        try:
            prompt = f"""You are a plant care expert. Based on the following plant analysis, provide specific, actionable care recommendations.

Plant Status:
- Stress Level: {stress_level:.0%} ({category})
- Health Status: {"Healthy" if is_healthy else "Signs of disease/stress detected"}

Provide 3-5 specific care recommendations. Be practical and actionable.
Format as a numbered list. Keep each recommendation to 1-2 sentences.
Do NOT mention specific disease names - focus on general care actions.
Include immediate actions and preventive measures."""

            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
    
    # Fallback recommendations
    if stress_level < 0.15:
        return """✅ Your plant is thriving! Here's how to maintain it:

1. Continue your current watering schedule - it's working well
2. Ensure consistent light exposure (avoid sudden changes)
3. Monitor weekly for any early signs of stress
4. Consider light fertilizing during growing season
5. Keep doing what you're doing!"""
    
    elif stress_level < 0.35:
        return """⚠️ Mild stress detected. Recommended actions:

1. Check soil moisture - water if top inch is dry
2. Ensure adequate drainage (no waterlogging)
3. Verify light conditions match plant needs
4. Look for early signs of pests on leaf undersides
5. Monitor closely over the next few days"""
    
    elif stress_level < 0.55:
        return """🟠 Moderate stress detected. Take action:

1. Adjust watering immediately - check if over or under-watered
2. Move to appropriate light (not too harsh, not too dim)
3. Check roots for rot or binding
4. Remove any severely damaged leaves
5. Consider repotting if root-bound
6. Isolate from other plants if disease suspected"""
    
    elif stress_level < 0.75:
        return """🔴 Severe stress detected. Urgent care needed:

1. IMMEDIATE: Check and adjust watering
2. Remove all dead/dying foliage to prevent spread
3. Isolate plant from others immediately
4. Check for pest infestation (spider mites, aphids)
5. Ensure proper ventilation around plant
6. Consider fungicide if fungal infection suspected
7. Reduce fertilizer - stressed plants can't absorb it"""
    
    else:
        return """🚨 Critical condition. Emergency care:

1. ISOLATE immediately from other plants
2. Remove all affected leaves/stems (sterilize tools)
3. Check roots - trim any rotten portions
4. Repot in fresh, sterile soil if root issues found
5. Place in stable environment (no direct sun)
6. Water sparingly - only when soil is dry
7. Consider propagating healthy portions as backup
8. Monitor daily for improvement/decline"""


# ============================================
# PLANT VOICE
# ============================================

def get_plant_speech(stress_level: float, is_healthy: bool) -> str:
    """Generate plant's inner voice."""
    
    if GROQ_AVAILABLE and groq_client:
        try:
            if stress_level < 0.15:
                mood = "content, peaceful, thriving"
            elif stress_level < 0.35:
                mood = "slightly concerned but okay"
            elif stress_level < 0.55:
                mood = "worried, uncomfortable"
            elif stress_level < 0.75:
                mood = "distressed, in pain"
            else:
                mood = "desperate, fading"
            
            prompt = f"""You are a plant that has gained consciousness.

Your state:
- Stress level: {stress_level:.0%}
- Health: {"healthy" if is_healthy else "fighting an infection"}
- Mood: {mood}
- You emit ultrasonic clicks from your xylem when stressed

Speak in first person about how you feel. Be emotional and expressive.
Reference biology naturally (water transport, leaves, roots, sunlight).
Match tone to stress level. 2-3 sentences. No quotes. No emojis.
Do NOT mention specific disease names."""

            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.85
            )
            return response.choices[0].message.content.strip()
        except:
            pass
    
    # Fallback
    if stress_level < 0.3:
        return "I feel wonderful today. My leaves are soaking up the warm sunlight and my roots are happily drinking fresh water. Life is good."
    elif stress_level < 0.6:
        return "I'm feeling a bit stressed. My water transport isn't flowing as smoothly as I'd like, and my leaves feel tense. I could use some care."
    else:
        return "I'm struggling right now. My vessels are clicking in distress and I desperately need help. Please take care of me soon."


async def text_to_speech_async(text: str, output_path: str, stress_level: float):
    """Convert text to speech."""
    if stress_level < 0.3:
        rate, pitch = "-5%", "-2Hz"
    elif stress_level < 0.6:
        rate, pitch = "+0%", "+0Hz"
    else:
        rate, pitch = "+10%", "+3Hz"
    
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def generate_voice_audio(text: str, stress_level: float) -> str:
    """Generate TTS audio file."""
    if not TTS_AVAILABLE:
        return None

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        output_path = tmp.name
        tmp.close()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(text_to_speech_async(text, output_path, stress_level))
        return output_path
    except:
        return None


# ============================================
# ULTRASONIC AUDIO
# ============================================

def generate_ultrasonic_audio(stress_level: float, duration: int = 10) -> str:
    """Generate synthetic ultrasonic pops audio."""
    if not AUDIO_AVAILABLE:
        return None
    
    try:
        pops_per_hour = stress_to_pop_rate(stress_level)
        # Demo mode: compress 1 hour into duration
        pops_in_clip = int(pops_per_hour * (duration / 3600) * 60)
        
        total_samples = duration * TARGET_SR
        audio = np.random.randn(total_samples) * 0.01  # Background noise
        
        if pops_in_clip > 0:
            pop_times = np.random.uniform(0.2, duration - 0.2, pops_in_clip)
            
            for t in pop_times:
                # Create synthetic pop (short burst)
                pop_duration = 0.002  # 2ms
                pop_samples = int(pop_duration * TARGET_SR)
                pop = np.random.randn(pop_samples) * 0.5
                
                # Envelope
                envelope = np.hanning(pop_samples)
                pop = pop * envelope
                
                # Place in audio
                start = int(t * TARGET_SR)
                end = min(start + pop_samples, total_samples)
                length = end - start
                if length > 0:
                    audio[start:end] += pop[:length]
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()
        sf.write(output_path, audio.astype(np.float32), TARGET_SR)
        return output_path
    except:
        return None


# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_plant(image: np.ndarray):
    """Main analysis pipeline."""
    
    if image is None:
        return None, None, "Please upload an image", "", "", None, None
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 1. Segment
    segmented, mask = segment_plant(image)
    
    # 2. Classify
    classification = classify_plant(segmented)
    
    # 3. Grad-CAM
    gradcam_img = generate_gradcam(segmented)
    
    # 4. Stress estimation
    stress_level = estimate_stress(classification)
    pops_per_hour = stress_to_pop_rate(stress_level)
    category, color = get_stress_category(stress_level)
    
    # 5. Care recommendations
    recommendations = get_care_recommendations(stress_level, classification['is_healthy'])
    
    # 6. Plant speech
    speech_text = get_plant_speech(stress_level, classification['is_healthy'])
    
    # 7. Generate audio
    voice_audio = generate_voice_audio(speech_text, stress_level)
    ultrasonic_audio = generate_ultrasonic_audio(stress_level)
    
    # Format results
    status_text = f"""## 🌱 Plant Analysis Results

### Stress Level: {stress_level:.0%} ({category})

| Metric | Value |
|--------|-------|
| Health Status | {"✅ Healthy" if classification['is_healthy'] else "⚠️ Stress Detected"} |
| Classification | {classification['label']} |
| Confidence | {classification['confidence']:.1%} |
| Ultrasonic Clicks | {pops_per_hour:.1f} per hour |

---

### 🗣️ Plant Says:
> *"{speech_text}"*
"""
    
    return (
        segmented,
        gradcam_img,
        status_text,
        recommendations,
        f'"{speech_text}"',
        voice_audio,
        ultrasonic_audio
    )


# ============================================
# GRADIO INTERFACE
# ============================================

# Custom CSS
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
    border: none !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #27ae60, #1e8449) !important;
}
footer {display: none !important;}
"""

# Build interface
with gr.Blocks(css=css, title="🌱 PlantWhisper") as demo:
    
    gr.Markdown("""
    # 🌱 PlantWhisper
    ### *Listen to what your plants are trying to tell you*
    
    Upload a photo of your plant leaf to get:
    - **Health Analysis** with AI-powered stress detection
    - **Visual Attention Map** showing areas of concern
    - **Care Recommendations** tailored to your plant's condition
    - **Plant's Voice** - hear what your plant is "feeling"
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Plant Image",
                type="numpy",
                height=350
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Plant",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ---
            **Tips for best results:**
            - Use clear, well-lit photos
            - Focus on individual leaves
            - Include any visible damage/discoloration
            """)
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🖼️ Segmented"):
                    output_segmented = gr.Image(label="Segmented Plant", height=300)
                with gr.TabItem("🔥 Attention Map"):
                    output_gradcam = gr.Image(label="Areas of Concern", height=300)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            output_status = gr.Markdown(label="Analysis Results")
        
        with gr.Column(scale=1):
            output_recommendations = gr.Textbox(
                label="💊 Care Recommendations",
                lines=10,
                show_copy_button=True
            )
    
    gr.Markdown("---")
    
    gr.Markdown("### 🔊 Listen to Your Plant")
    
    with gr.Row():
        with gr.Column(scale=1):
            output_speech = gr.Textbox(
                label="🗣️ Plant Says",
                lines=3
            )
            output_voice = gr.Audio(
                label="🎤 Plant Voice",
                type="filepath"
            )
        
        with gr.Column(scale=1):
            output_ultrasonic = gr.Audio(
                label="🔊 Ultrasonic Emissions (Pitch-shifted)",
                type="filepath"
            )
            gr.Markdown("""
            *The clicks you hear are pitch-shifted representations of 
            ultrasonic emissions plants make when stressed (based on 
            [Tel Aviv University research](https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3)).*
            """)
    
    gr.Markdown("""
    ---
    
    ### About PlantWhisper
    
    PlantWhisper uses computer vision and AI to analyze plant health:
    
    1. **SAM Segmentation** - Isolates the plant from background
    2. **MobileNetV2 Classifier** - Detects health status
    3. **Grad-CAM** - Shows where the AI is looking
    4. **Stress Modeling** - Based on plant acoustics research
    5. **LLM + TTS** - Gives your plant a voice
    
    *Created by Mohith | IIT Bombay | 2026*
    """)
    
    # Connect button
    analyze_btn.click(
        fn=analyze_plant,
        inputs=[input_image],
        outputs=[
            output_segmented,
            output_gradcam,
            output_status,
            output_recommendations,
            output_speech,
            output_voice,
            output_ultrasonic
        ]
    )
    
    # Examples — use assets from repo (parent dir on HF Spaces, sibling locally)
    _examples_dir = Path(__file__).parent.parent / "assets"
    _example_files = [
        str(p) for p in sorted(_examples_dir.glob("*.jfif"))
        if p.is_file()
    ] if _examples_dir.exists() else []
    if _example_files:
        gr.Examples(
            examples=[[f] for f in _example_files],
            inputs=[input_image],
            outputs=[
                output_segmented,
                output_gradcam,
                output_status,
                output_recommendations,
                output_speech,
                output_voice,
                output_ultrasonic
            ],
            fn=analyze_plant,
            cache_examples=False
        )


# ============================================
# LAUNCH
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
