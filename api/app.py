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

import backend as pw

app = FastAPI(title="PlantWhisper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            "voiceAudio": encode_audio(voice_audio),
            "ultrasonicAudio": encode_audio(ultrasonic_audio),
            "diffusionAudio": encode_audio(diffusion_audio),
            "audioMethod": "Diffusion" if diffusion_audio else "Synthetic",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
