"""
PlantWhisper API — Headless FastAPI backend for HuggingFace Spaces.
Serves the full analysis pipeline (diffusion-only, no synthetic fallback) as a REST API.
"""

import os
import io
import tempfile
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backend as pw

app = FastAPI(title="PlantWhisper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "PlantWhisper API",
        "status": "ready",
        "diffusion_available": pw.DIFFUSION_AVAILABLE,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full plant analysis pipeline. Returns JSON with all results."""
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(pil_image)

    (
        segmented,
        gradcam_img,
        status_md,
        recommendations,
        speech_text,
        voice_audio,
        ultrasonic_audio,
        diffusion_audio,
    ) = pw.analyze_plant(image_np, use_diffusion=True)

    # Compute values for the frontend
    classification = pw.classify_plant(
        segmented if segmented is not None else image_np
    )
    stress_level = pw.estimate_stress(classification)
    pops_per_hour = pw.stress_to_pop_rate(stress_level)
    category, color = pw.get_stress_category(stress_level)

    # Encode audio files as base64 if available
    def encode_audio(path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(path)[1].lstrip(".")
            return f"data:audio/{ext};base64,{data}"
        return None

    # Encode images as base64 PNG data URIs
    def encode_image(img_array):
        if img_array is None:
            return None
        try:
            # Convert RGB to BGR for cv2 encoding
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = base64.b64encode(buf).decode("utf-8")
            return f"data:image/jpeg;base64,{data}"
        except Exception:
            return None

    # Generate waveform + spectrogram visualizations from the pop audio
    def render_waveform_spectrogram(audio_path, pops_per_hour_val, duration=10):
        """Render waveform and spectrogram as base64 JPEG images."""
        if not audio_path or not os.path.exists(audio_path):
            return None, None
        try:
            import soundfile as sf
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_data))

            # Detect pop positions (amplitude spikes)
            threshold = np.std(audio_data) * 4
            pop_mask = np.abs(audio_data) > threshold

            # --- Waveform ---
            fig_w, ax_w = plt.subplots(figsize=(6, 2), dpi=100)
            fig_w.patch.set_facecolor("#0a1a0f")
            ax_w.set_facecolor("#0a1a0f")
            ax_w.plot(time_axis, audio_data, color="#34d399", linewidth=0.4, alpha=0.9)
            # Highlight pop regions
            pop_indices = np.where(pop_mask)[0]
            if len(pop_indices) > 0:
                pop_times_arr = time_axis[pop_indices]
                pop_vals = audio_data[pop_indices]
                ax_w.scatter(pop_times_arr, pop_vals, color="#f87171", s=1, alpha=0.6, zorder=3)
            ax_w.set_xlabel("Time (s)", fontsize=7, color="#6ee7b7")
            ax_w.set_ylabel("Amplitude", fontsize=7, color="#6ee7b7")
            ax_w.set_title(
                f"Waveform — {pops_per_hour_val:.1f} pops/hour",
                fontsize=8, color="#a7f3d0", pad=4,
            )
            ax_w.tick_params(colors="#6ee7b7", labelsize=6)
            for spine in ax_w.spines.values():
                spine.set_color("#065f46")
            ax_w.set_xlim(0, time_axis[-1])
            fig_w.tight_layout(pad=0.5)
            buf_w = io.BytesIO()
            fig_w.savefig(buf_w, format="jpeg", bbox_inches="tight", pil_kwargs={"quality": 85})
            plt.close(fig_w)
            buf_w.seek(0)
            waveform_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_w.read()).decode()}"

            # --- Spectrogram ---
            fig_s, ax_s = plt.subplots(figsize=(6, 2), dpi=100)
            fig_s.patch.set_facecolor("#0a1a0f")
            ax_s.set_facecolor("#0a1a0f")
            ax_s.specgram(
                audio_data, NFFT=256, Fs=sr, noverlap=200,
                cmap="magma", scale="dB",
            )
            ax_s.set_xlabel("Time (s)", fontsize=7, color="#6ee7b7")
            ax_s.set_ylabel("Hz", fontsize=7, color="#6ee7b7")
            ax_s.set_title(
                f"Spectrogram — {pops_per_hour_val:.1f} pops/hour",
                fontsize=8, color="#a7f3d0", pad=4,
            )
            ax_s.tick_params(colors="#6ee7b7", labelsize=6)
            ax_s.set_ylim(0, 3000)
            for spine in ax_s.spines.values():
                spine.set_color("#065f46")
            fig_s.tight_layout(pad=0.5)
            buf_s = io.BytesIO()
            fig_s.savefig(buf_s, format="jpeg", bbox_inches="tight", pil_kwargs={"quality": 85})
            plt.close(fig_s)
            buf_s.seek(0)
            spectrogram_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_s.read()).decode()}"

            return waveform_b64, spectrogram_b64
        except Exception as e:
            print(f"Waveform/spectrogram render error: {e}")
            return None, None

    # Pick the best available audio file for visualization
    viz_audio = diffusion_audio or ultrasonic_audio
    waveform_img, spectrogram_img = render_waveform_spectrogram(viz_audio, pops_per_hour)

    # Generate TTS inline (edge_tts is async, avoid nested event loop)
    voice_audio_b64 = encode_audio(voice_audio)
    if voice_audio_b64 is None and pw.TTS_AVAILABLE:
        try:
            import edge_tts
            sl = stress_level
            if sl < 0.3:
                rate, pitch = "-5%", "-2Hz"
            elif sl < 0.6:
                rate, pitch = "+0%", "+0Hz"
            else:
                rate, pitch = "+10%", "+3Hz"
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_path = tmp.name
            tmp.close()
            communicate = edge_tts.Communicate(
                speech_text.strip('"'), pw.TTS_VOICE, rate=rate, pitch=pitch
            )
            await communicate.save(tmp_path)
            voice_audio_b64 = encode_audio(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"TTS fallback error: {e}")

    return JSONResponse(
        {
            "stress": stress_level,
            "category": category,
            "color": color,
            "label": classification["label"],
            "confidence": classification["confidence"],
            "popsPerHour": pops_per_hour,
            "speech": speech_text.strip('"'),
            "recommendations": recommendations,
            "voiceAudio": voice_audio_b64,
            "ultrasonicAudio": encode_audio(ultrasonic_audio),
            "diffusionAudio": encode_audio(diffusion_audio),
            "audioMethod": "Diffusion" if diffusion_audio else "Synthetic",
            "segmentedImage": encode_image(segmented),
            "gradcamImage": encode_image(gradcam_img),
            "waveformImage": waveform_img,
            "spectrogramImage": spectrogram_img,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
