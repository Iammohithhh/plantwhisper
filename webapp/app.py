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

# --- Segmentation: FastSAM (preferred on CPU) → SAM → green threshold fallback ---
FASTSAM_AVAILABLE = False
SAM_AVAILABLE = False
fastsam_model = None
sam_predictor = None

# Try FastSAM first (lightweight, ~23MB, great for CPU)
print("Loading segmentation...")
try:
    from ultralytics import YOLO as FastSAM_YOLO
    FASTSAM_CHECKPOINT = "FastSAM-s.pt"
    fastsam_model = FastSAM_YOLO(FASTSAM_CHECKPOINT)
    FASTSAM_AVAILABLE = True
    print("✓ FastSAM loaded (lightweight)")
except Exception as e:
    print(f"⚠ FastSAM not available: {e}")

# Try full SAM as fallback (heavier, ~375MB, better on GPU)
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

# --- Diffusion Model for Acoustic Synthesis ---
# Generates spectrograms conditioned on stress level → Griffin-Lim → audio
# Falls back to synthetic pops if checkpoint not available

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

print("\n🌱 PlantWhisper ready!\n")

# ============================================
# CORE FUNCTIONS
# ============================================

def segment_plant(image: np.ndarray) -> tuple:
    """Segment plant using FastSAM → SAM → green threshold fallback."""
    h, w = image.shape[:2]

    # Try FastSAM first
    if FASTSAM_AVAILABLE and fastsam_model is not None:
        try:
            results = fastsam_model(image, retina_masks=True, conf=0.4, iou=0.9, verbose=False)
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                # Pick the mask closest to center
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

    # Try full SAM
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


def generate_diffusion_audio(stress_level: float, duration: int = 10) -> str:
    """Generate ultrasonic audio using the diffusion model.

    Generates a spectrogram conditioned on stress_level, converts it to a
    single pop via Griffin-Lim, then tiles pops across the clip duration.
    Returns path to wav file, or None on failure.
    """
    if not DIFFUSION_AVAILABLE or diffusion_model is None:
        return None

    try:
        import librosa

        # Generate spectrogram (64 mel bins x 32 time frames)
        spec_tensor = diffusion_process.sample(
            diffusion_model, shape=(1, 1, 64, 32), stress_level=stress_level)
        spec = spec_tensor.squeeze().cpu().numpy()

        # Denormalize (reverse training normalization)
        spec = spec * 3
        spec = np.exp(spec)

        # Griffin-Lim → single pop waveform
        pop_audio = librosa.feature.inverse.mel_to_audio(
            spec, sr=TARGET_SR, n_fft=256, hop_length=64, n_iter=32)

        # Tile pops across full duration based on stress-derived rate
        pops_per_hour = stress_to_pop_rate(stress_level)
        n_pops = max(1, int(pops_per_hour * (duration / 3600) * 60))
        total_samples = duration * TARGET_SR
        full_audio = np.random.randn(total_samples) * 0.001  # subtle background

        pop_times = np.sort(np.random.uniform(0.2, duration - 0.2, n_pops))
        for t in pop_times:
            start = int(t * TARGET_SR)
            end = min(start + len(pop_audio), total_samples)
            length = end - start
            if length > 0:
                full_audio[start:end] += pop_audio[:length] * np.random.uniform(0.7, 1.0)

        # Normalize
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
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_plant(image: np.ndarray, use_diffusion: bool = True):
    """Main analysis pipeline."""

    if image is None:
        return None, None, "Please upload an image", "", "", None, None, None
    
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

    # 8. Diffusion-generated audio (if available and requested)
    diffusion_audio = None
    diffusion_label = "Not available"
    if use_diffusion and DIFFUSION_AVAILABLE:
        diffusion_audio = generate_diffusion_audio(stress_level)
        diffusion_label = "Diffusion model" if diffusion_audio else "Generation failed"
    elif not DIFFUSION_AVAILABLE:
        diffusion_label = "No checkpoint loaded"

    # Format results
    audio_method = "Diffusion" if (diffusion_audio and use_diffusion) else "Synthetic"
    status_text = f"""## 🌱 Plant Analysis Results

### Stress Level: {stress_level:.0%} ({category})

| Metric | Value |
|--------|-------|
| Health Status | {"✅ Healthy" if classification['is_healthy'] else "⚠️ Stress Detected"} |
| Classification | {classification['label']} |
| Confidence | {classification['confidence']:.1%} |
| Ultrasonic Clicks | {pops_per_hour:.1f} per hour |
| Audio Method | {audio_method} |

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
        ultrasonic_audio,
        diffusion_audio,
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
            use_diffusion = gr.Checkbox(
                label="Use Diffusion Model for audio",
                value=DIFFUSION_AVAILABLE,
                interactive=True,
                info="Generate ultrasonic pops via trained diffusion model (requires checkpoint)"
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
                label="🔊 Ultrasonic Emissions (Synthetic)",
                type="filepath"
            )
            output_diffusion = gr.Audio(
                label="🧬 Ultrasonic Emissions (Diffusion-Generated)",
                type="filepath"
            )
            gr.Markdown("""
            *The clicks you hear are pitch-shifted representations of
            ultrasonic emissions plants make when stressed (based on
            [Tel Aviv University research](https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3)).
            When a diffusion checkpoint is loaded, the second player uses a
            trained model to generate realistic spectrogram-based pops.*
            """)
    
    gr.Markdown("""
    ---
    
    ### About PlantWhisper
    
    PlantWhisper uses computer vision and AI to analyze plant health:

    1. **SAM / FastSAM Segmentation** - Isolates the plant from background
    2. **MobileNetV2 Classifier** - Detects health status
    3. **Grad-CAM** - Shows where the AI is looking
    4. **Stress Modeling** - Based on plant acoustics research
    5. **Conditional Diffusion** - Generates realistic ultrasonic pop spectrograms
    6. **LLM + TTS** - Gives your plant a voice
    
    *Created by Mohith | IIT Bombay | 2026*
    """)
    
    # Connect button
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
                output_ultrasonic,
                output_diffusion,
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
