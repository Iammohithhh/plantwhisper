"""
PlantWhisper Backend
All AI/ML model loading and inference logic.
Imported by app.py (UI layer).
"""

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
print(f"🌱 PlantWhisper backend starting on {device}...")

# ============================================
# MODEL LOADING
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

# --- Segmentation: FastSAM → SAM → green threshold fallback ---
FASTSAM_AVAILABLE = False
SAM_AVAILABLE = False
fastsam_model = None
sam_predictor = None

print("Loading segmentation...")
try:
    from ultralytics import YOLO as FastSAM_YOLO
    FASTSAM_CHECKPOINT = "FastSAM-s.pt"
    fastsam_model = FastSAM_YOLO(FASTSAM_CHECKPOINT)
    FASTSAM_AVAILABLE = True
    print("✓ FastSAM loaded (lightweight)")
except Exception as e:
    print(f"⚠ FastSAM not available: {e}")

if not FASTSAM_AVAILABLE:
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
        print("✓ SAM loaded (full)")
    except Exception as e:
        print(f"⚠ SAM not available: {e}")

# --- Grad-CAM ---
GRADCAM_AVAILABLE = False
grad_cam = None

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
    try:
        target_layer = classifier.mobilenet_v2.layer[-1]
    except (AttributeError, IndexError):
        target_layer = classifier.mobilenet_v2.features[-1]
    grad_cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    GRADCAM_AVAILABLE = True
    print("✓ Grad-CAM loaded")
except Exception as e:
    print(f"⚠ Grad-CAM not available: {e}")

# --- Groq LLM ---
GROQ_AVAILABLE = False
groq_client = None

print("Loading Groq client...")
try:
    import groq
    groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    GROQ_AVAILABLE = bool(GROQ_API_KEY)
    print("✓ Groq client ready" if GROQ_AVAILABLE else "⚠ Groq API key not set")
except Exception as e:
    print(f"⚠ Groq not available: {e}")

# --- Edge TTS ---
TTS_AVAILABLE = False
print("Loading TTS...")
try:
    import edge_tts
    TTS_AVAILABLE = True
    print("✓ Edge TTS loaded")
except:
    print("⚠ Edge TTS not available")

# --- Audio libraries ---
AUDIO_AVAILABLE = False
try:
    from scipy import signal
    import soundfile as sf
    AUDIO_AVAILABLE = True
except:
    pass

# --- Diffusion Model ---
class ConditionalUNet(nn.Module):
    """UNet for spectrogram diffusion, conditioned on stress level."""

    def __init__(self, in_channels=1, out_channels=1, stress_embed_dim=64):
        super().__init__()
        self.stress_embed = nn.Sequential(
            nn.Linear(1, stress_embed_dim), nn.SiLU(),
            nn.Linear(stress_embed_dim, stress_embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(1, stress_embed_dim), nn.SiLU(),
            nn.Linear(stress_embed_dim, stress_embed_dim))
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.cond_proj = nn.Linear(stress_embed_dim, 128)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU())
        self.dec3 = self._conv_block(256, 64)
        self.dec2 = self._conv_block(128, 32)
        self.dec1 = self._conv_block(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU())

    def forward(self, x, t, stress_level):
        stress_emb = self.stress_embed(stress_level.unsqueeze(-1))
        time_emb = self.time_embed(t.unsqueeze(-1))
        cond = stress_emb + time_emb
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e3_cond = e3 + self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.bottleneck(e3_cond)
        b_up = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([b_up, e3], dim=1))
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        return self.out(d1)


class SimpleDiffusion:
    """DDPM diffusion process for spectrogram generation."""

    def __init__(self, n_steps=500, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def sample(self, model, shape, stress_level):
        model.eval()
        x = torch.randn(shape, device=self.device)
        stress_t = torch.tensor([stress_level], device=self.device).float()
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i / self.n_steps], device=self.device).float()
            pred_noise = model(x, t, stress_t)
            alpha = self.alphas[i]
            alpha_cum = self.alpha_cumprod[i]
            x = (x - (1 - alpha) / (1 - alpha_cum).sqrt() * pred_noise) / alpha.sqrt()
            if i > 0:
                x += self.betas[i].sqrt() * torch.randn_like(x)
        return x


DIFFUSION_AVAILABLE = False
diffusion_model = None
diffusion_process = None

DIFFUSION_CHECKPOINT = os.environ.get("DIFFUSION_CHECKPOINT", "diffusion_model.pt")
print("Loading diffusion model...")
try:
    if os.path.exists(DIFFUSION_CHECKPOINT):
        diffusion_model = ConditionalUNet().to(device)
        diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=device))
        diffusion_model.eval()
        diffusion_process = SimpleDiffusion(n_steps=500, device=device)
        DIFFUSION_AVAILABLE = True
        print("✓ Diffusion model loaded")
    else:
        print(f"⚠ Diffusion checkpoint not found ({DIFFUSION_CHECKPOINT}) - using synthetic pops")
except Exception as e:
    print(f"⚠ Diffusion model failed to load: {e}")

print("\n🌱 PlantWhisper backend ready!\n")

# ============================================
# INFERENCE FUNCTIONS
# ============================================

def segment_plant(image: np.ndarray) -> tuple:
    """Segment plant using FastSAM → SAM → green threshold fallback."""
    h, w = image.shape[:2]

    if FASTSAM_AVAILABLE and fastsam_model is not None:
        try:
            results = fastsam_model(image, retina_masks=True, conf=0.4, iou=0.9, verbose=False)
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                center_y, center_x = h // 2, w // 2
                best_idx, best_dist = 0, float('inf')
                for i, m in enumerate(masks):
                    m_resized = cv2.resize(m.astype(np.uint8), (w, h)) > 0
                    ys, xs = np.where(m_resized)
                    if len(ys) == 0:
                        continue
                    cy, cx = ys.mean(), xs.mean()
                    dist = (cy - center_y) ** 2 + (cx - center_x) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                mask = cv2.resize(masks[best_idx].astype(np.uint8), (w, h)) > 0
                segmented = image.copy()
                segmented[~mask] = [240, 240, 240]
                return segmented, mask
        except Exception:
            pass

    if SAM_AVAILABLE and sam_predictor is not None:
        try:
            sam_predictor.set_image(image)
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
        except Exception:
            pass

    # Fallback: green detection
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
    if not GRADCAM_AVAILABLE or grad_cam is None:
        return image

    try:
        pil_image = Image.fromarray(image)
        inputs = processor(images=pil_image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(device)
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
        image_float = image.astype(np.float32) / 255.0
        from pytorch_grad_cam.utils.image import show_cam_on_image
        return show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
    except:
        return image


def estimate_stress(classification: dict) -> float:
    """Estimate stress level 0–1."""
    if classification['is_healthy']:
        stress = 0.1 * (1 - classification['confidence'])
    else:
        stress = 0.4 + (0.5 * classification['confidence'])
    return float(np.clip(stress, 0.0, 1.0))


def stress_to_pop_rate(stress: float) -> float:
    """Convert stress to pops/hour (hump-shaped curve)."""
    if stress < 0.1:
        return HEALTHY_POPS_PER_HOUR + stress * 30
    hump = 4 * stress * (1 - stress)
    return HEALTHY_POPS_PER_HOUR + (DROUGHT_PEAK_POPS_PER_HOUR - HEALTHY_POPS_PER_HOUR) * hump


def get_stress_category(stress: float) -> tuple:
    """Return (label, hex_color) for a given stress 0–1."""
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


def get_care_recommendations(stress_level: float, is_healthy: bool) -> str:
    """Generate care recommendations using Groq LLM or built-in fallback."""
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

    if stress_level < 0.15:
        return "✅ Your plant is thriving! Here's how to maintain it:\n\n1. Continue your current watering schedule - it's working well\n2. Ensure consistent light exposure (avoid sudden changes)\n3. Monitor weekly for any early signs of stress\n4. Consider light fertilizing during growing season\n5. Keep doing what you're doing!"
    elif stress_level < 0.35:
        return "⚠️ Mild stress detected. Recommended actions:\n\n1. Check soil moisture - water if top inch is dry\n2. Ensure adequate drainage (no waterlogging)\n3. Verify light conditions match plant needs\n4. Look for early signs of pests on leaf undersides\n5. Monitor closely over the next few days"
    elif stress_level < 0.55:
        return "🟠 Moderate stress detected. Take action:\n\n1. Adjust watering immediately - check if over or under-watered\n2. Move to appropriate light (not too harsh, not too dim)\n3. Check roots for rot or binding\n4. Remove any severely damaged leaves\n5. Consider repotting if root-bound\n6. Isolate from other plants if disease suspected"
    elif stress_level < 0.75:
        return "🔴 Severe stress detected. Urgent care needed:\n\n1. IMMEDIATE: Check and adjust watering\n2. Remove all dead/dying foliage to prevent spread\n3. Isolate plant from others immediately\n4. Check for pest infestation (spider mites, aphids)\n5. Ensure proper ventilation around plant\n6. Consider fungicide if fungal infection suspected\n7. Reduce fertilizer - stressed plants can't absorb it"
    else:
        return "🚨 Critical condition. Emergency care:\n\n1. ISOLATE immediately from other plants\n2. Remove all affected leaves/stems (sterilize tools)\n3. Check roots - trim any rotten portions\n4. Repot in fresh, sterile soil if root issues found\n5. Place in stable environment (no direct sun)\n6. Water sparingly - only when soil is dry\n7. Consider propagating healthy portions as backup\n8. Monitor daily for improvement/decline"


def get_plant_speech(stress_level: float, is_healthy: bool) -> str:
    """Generate plant's inner voice using Groq LLM or built-in fallback."""
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

    if stress_level < 0.3:
        return "I feel wonderful today. My leaves are soaking up the warm sunlight and my roots are happily drinking fresh water. Life is good."
    elif stress_level < 0.6:
        return "I'm feeling a bit stressed. My water transport isn't flowing as smoothly as I'd like, and my leaves feel tense. I could use some care."
    else:
        return "I'm struggling right now. My vessels are clicking in distress and I desperately need help. Please take care of me soon."


async def _tts_async(text: str, output_path: str, stress_level: float):
    import edge_tts
    if stress_level < 0.3:
        rate, pitch = "-5%", "-2Hz"
    elif stress_level < 0.6:
        rate, pitch = "+0%", "+0Hz"
    else:
        rate, pitch = "+10%", "+3Hz"
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def generate_voice_audio(text: str, stress_level: float) -> str:
    """Generate TTS audio file. Returns filepath or None."""
    if not TTS_AVAILABLE:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        output_path = tmp.name
        tmp.close()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_tts_async(text, output_path, stress_level))
        return output_path
    except:
        return None


def generate_ultrasonic_audio(stress_level: float, duration: int = 10) -> str:
    """Generate synthetic ultrasonic pops audio. Returns filepath or None."""
    if not AUDIO_AVAILABLE:
        return None
    try:
        import soundfile as sf
        pops_per_hour = stress_to_pop_rate(stress_level)
        pops_in_clip = int(pops_per_hour * (duration / 3600) * 60)
        total_samples = duration * TARGET_SR
        audio = np.random.randn(total_samples) * 0.01

        if pops_in_clip > 0:
            pop_times = np.random.uniform(0.2, duration - 0.2, pops_in_clip)
            for t in pop_times:
                pop_duration = 0.002
                pop_samples = int(pop_duration * TARGET_SR)
                pop = np.random.randn(pop_samples) * 0.5
                pop = pop * np.hanning(pop_samples)
                start = int(t * TARGET_SR)
                end = min(start + pop_samples, total_samples)
                length = end - start
                if length > 0:
                    audio[start:end] += pop[:length]

        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()
        sf.write(output_path, audio.astype(np.float32), TARGET_SR)
        return output_path
    except:
        return None


def generate_diffusion_audio(stress_level: float, duration: int = 10) -> str:
    """Generate ultrasonic audio using the diffusion model. Returns filepath or None."""
    if not DIFFUSION_AVAILABLE or diffusion_model is None:
        return None
    try:
        import librosa
        import soundfile as sf
        spec_tensor = diffusion_process.sample(
            diffusion_model, shape=(1, 1, 64, 32), stress_level=stress_level)
        spec = spec_tensor.squeeze().cpu().numpy()
        spec = spec * 3
        spec = np.exp(spec)
        pop_audio = librosa.feature.inverse.mel_to_audio(
            spec, sr=TARGET_SR, n_fft=256, hop_length=64, n_iter=32)

        pops_per_hour = stress_to_pop_rate(stress_level)
        n_pops = max(1, int(pops_per_hour * (duration / 3600) * 60))
        total_samples = duration * TARGET_SR
        full_audio = np.random.randn(total_samples) * 0.001

        pop_times = np.sort(np.random.uniform(0.2, duration - 0.2, n_pops))
        for t in pop_times:
            start = int(t * TARGET_SR)
            end = min(start + len(pop_audio), total_samples)
            length = end - start
            if length > 0:
                full_audio[start:end] += pop_audio[:length] * np.random.uniform(0.7, 1.0)

        peak = np.abs(full_audio).max()
        if peak > 0:
            full_audio = full_audio * 0.8 / peak

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()
        sf.write(output_path, full_audio.astype(np.float32), TARGET_SR)
        return output_path
    except Exception as e:
        print(f"Diffusion audio error: {e}")
        return None


# ============================================
# MAIN ANALYSIS PIPELINE
# ============================================

def analyze_plant(image: np.ndarray, use_diffusion: bool = True):
    """
    Full analysis pipeline.

    Returns:
        segmented_img, gradcam_img, status_md, recommendations,
        speech_text, voice_audio, ultrasonic_audio, diffusion_audio
    """
    if image is None:
        return None, None, "Please upload an image", "", "", None, None, None

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    segmented, mask = segment_plant(image)
    classification = classify_plant(segmented)
    gradcam_img = generate_gradcam(segmented)

    stress_level = estimate_stress(classification)
    pops_per_hour = stress_to_pop_rate(stress_level)
    category, color = get_stress_category(stress_level)

    recommendations = get_care_recommendations(stress_level, classification['is_healthy'])
    speech_text = get_plant_speech(stress_level, classification['is_healthy'])

    voice_audio = generate_voice_audio(speech_text, stress_level)
    ultrasonic_audio = generate_ultrasonic_audio(stress_level)

    diffusion_audio = None
    if use_diffusion and DIFFUSION_AVAILABLE:
        diffusion_audio = generate_diffusion_audio(stress_level)

    audio_method = "Diffusion" if (diffusion_audio and use_diffusion) else "Synthetic"

    # Build stress bar (filled blocks)
    filled = int(stress_level * 20)
    bar = "█" * filled + "░" * (20 - filled)

    status_md = f"""## Plant Analysis Results

<span style="font-size:2rem; font-weight:700; color:{color}">{stress_level:.0%}</span>
&nbsp; **{category}**

`{bar}`

| Metric | Value |
|--------|-------|
| Health Status | {"✅ Healthy" if classification['is_healthy'] else "⚠️ Stress Detected"} |
| Classification | `{classification['label']}` |
| Confidence | {classification['confidence']:.1%} |
| Ultrasonic Clicks | **{pops_per_hour:.1f}** / hour |
| Audio Method | {audio_method} |

---
> *"{speech_text}"*
"""

    return (
        segmented,
        gradcam_img,
        status_md,
        recommendations,
        f'"{speech_text}"',
        voice_audio,
        ultrasonic_audio,
        diffusion_audio,
    )
