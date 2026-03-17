<p align="center">
  <img src="assets/plantwhisper_banner.png" alt="PlantWhisper Banner" width="800"/>
</p>

<h1 align="center">PlantWhisper</h1>

<p align="center">
  <strong>"Listen to what your plants are trying to tell you"</strong>
</p>

<p align="center">
  <a href="#-demo">Demo</a> &bull;
  <a href="#-features">Features</a> &bull;
  <a href="#-architecture">Architecture</a> &bull;
  <a href="#-getting-started">Getting Started</a> &bull;
  <a href="#-scientific-foundation">Science</a> &bull;
  <a href="#-results">Results</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Next.js-15-black.svg" alt="Next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

## Overview

**PlantWhisper** is a multimodal AI system that analyzes plant photos to detect stress levels and generates acoustic signatures representing what stressed plants actually sound like — pitch-shifted into the human audible range.

Plants emit ultrasonic clicks (20–150 kHz) when stressed through a process called **xylem cavitation**. PlantWhisper inverts this discovery: given a photo of a plant, it predicts the stress level and synthesizes the corresponding acoustic signature.

<p align="center">
  <img src="assets/pipeline_plantwhisper.png" alt="Pipeline Overview" width="700"/>
</p>

---

## Demo

<p align="center">
  <img src="assets/corn_1.jfif" alt="Healthy Plant Analysis" width="45%"/>
  <img src="assets/dis_corn.jfif" alt="Stressed Plant Analysis" width="45%"/>
</p>

| Healthy Plant (7% Stress) | Stressed Plant (83% Stress) |
|---|---|
| Minimal ultrasonic activity | Active distress clicking |
| 3.2 clicks/hour | 20.6 clicks/hour |
| *"I feel utterly serene..."* | *"I'm withering away..."* |

---

## Features

### Computer Vision Pipeline
- **SAM Segmentation** — Segment-Anything Model (with FastSAM fallback) for precise leaf isolation
- **MobileNetV2 Classifier** — Plant disease detection across 38 classes with 95%+ accuracy
- **Grad-CAM Explainability** — Visual attention maps highlighting areas of concern

### Acoustic Synthesis
- **Real Tel Aviv Data** — Based on actual ultrasonic recordings from stressed plants
- **Conditional Diffusion Model** — Generates mel spectrograms conditioned on stress level
- **Griffin-Lim Vocoder** — Converts spectrograms to audio
- **Pitch Shifting** — 53 kHz to 1 kHz for human audibility

### Plant Persona
- **LLM Speech Generation** — Groq / Llama 3.3 70B generates emotionally appropriate plant speech
- **Text-to-Speech** — Edge-TTS with stress-modulated pitch, rate, and tone
- **Care Recommendations** — Actionable advice from "keep it up" to "emergency care needed"

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PlantWhisper Pipeline                        │
└─────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐      ┌──────────────┐      ┌─────────────┐
     │  Photo   │ ───▶ │     SAM      │ ───▶ │ MobileNetV2 │
     │  Input   │      │ Segmentation │      │ Classifier  │
     └──────────┘      └──────────────┘      └──────┬──────┘
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              │                                           │
                              ▼                                           ▼
                       ┌─────────────┐                           ┌──────────────┐
                       │  Grad-CAM   │                           │    Stress    │
                       │  Heatmap    │                           │  Estimation  │
                       └─────────────┘                           └──────┬───────┘
                                                                        │
                    ┌───────────────────────────────────────────────────┼───────┐
                    │                       │                           │       │
                    ▼                       ▼                           ▼       ▼
           ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐   ┌─────────┐
           │   Parametric    │    │    Diffusion    │    │   LLM Plant    │   │  Care   │
           │ Audio Synthesis │    │ Spectrogram Gen │    │    Persona     │   │ Advice  │
           └────────┬────────┘    └────────┬────────┘    └───────┬────────┘   └────┬────┘
                    │                      │                     │                 │
                    ▼                      ▼                     ▼                 ▼
           ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐   ┌─────────┐
           │ Pitch-Shifted   │    │  Griffin-Lim    │    │   Edge-TTS     │   │ Action  │
           │  Plant Audio    │    │   Vocoder       │    │    Audio       │   │  Items  │
           └─────────────────┘    └─────────────────┘    └────────────────┘   └─────────┘
```

---

## Repository Structure

```
PlantWhisper/
├── api/                              # FastAPI backend (production)
│   ├── app.py                        #   REST API server (port 7860)
│   ├── backend.py                    #   ML inference engine
│   ├── requirements.txt              #   Python dependencies
│   └── Dockerfile                    #   Container deployment
│
├── portfolio/                        # Next.js 15 showcase website
│   ├── src/app/page.tsx              #   Landing page
│   ├── public/                       #   Static assets
│   ├── vercel.json                   #   Vercel deployment config
│   └── package.json                  #   Node dependencies
│
├── webapp/                           # Gradio web app (HuggingFace Spaces)
│   ├── app.py                        #   Gradio interface
│   ├── backend.py                    #   ML inference engine
│   └── requirements.txt              #   Python dependencies
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Tel Aviv dataset analysis
│   ├── 02_vision_pipeline.ipynb      # SAM + Classifier + Grad-CAM
│   ├── 03_acoustic_synthesis.ipynb   # Audio generation + Diffusion + Persona
│   └── 06_full_demo.ipynb            # Complete end-to-end pipeline
│
├── assets/                           # Images for README & docs
├── paper/                            # Research paper figures
├── README.md
└── LICENSE
```

---

## Getting Started

### Option 1: Run in Google Colab (Quickest)

1. Open any notebook from `notebooks/` in Google Colab
2. Enable GPU: **Runtime > Change runtime type > GPU**
3. Run all cells

### Option 2: Run the FastAPI Backend

```bash
git clone https://github.com/Iammohithhh/plantwhisper.git
cd plantwhisper/api

pip install -r requirements.txt
export GROQ_API_KEY="your-api-key"

uvicorn app:app --host 0.0.0.0 --port 7860
# API available at http://localhost:7860
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Upload a plant image, get full analysis |

### Option 3: Run with Docker

```bash
cd api
docker build -t plantwhisper-api .
docker run -p 7860:7860 -e GROQ_API_KEY="your-api-key" plantwhisper-api
```

### Option 4: Run the Gradio App

```bash
cd webapp
pip install -r requirements.txt
export GROQ_API_KEY="your-api-key"
python app.py
# Open http://localhost:7860
```

### Option 5: Run the Portfolio Site

```bash
cd portfolio
npm install
npm run dev
# Open http://localhost:3000
```

---

## Scientific Foundation

This project is based on groundbreaking research from Tel Aviv University:

> **Khait, I., et al. (2023).** *Sounds emitted by plants under stress are airborne and informative.* **Cell, 186(7), 1328–1336.**
> [DOI: 10.1016/j.cell.2023.03.009](https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3)

**Key findings:**
- Plants emit ultrasonic clicks (20–150 kHz) when stressed
- Drought-stressed tomatoes produce ~35 clicks/hour vs. <1 for healthy plants
- CNN classifiers achieved **99.7% accuracy** separating plant sounds from noise
- Drought vs. irrigated classification reached **84% accuracy** using sound alone
- Mechanism: xylem cavitation (air bubbles forming in water transport vessels)

### Stress–Sound Curve

PlantWhisper implements the **hump-shaped stress curve** from the paper — sound emission rises with dehydration, peaks around day 4–5, then declines as the plant dries out completely:

```
Pops/Hour
    35 ┤                    ╭───╮
       │                  ╭─╯   ╰─╮
    25 ┤                ╭─╯       ╰─╮
       │              ╭─╯           ╰─╮
    15 ┤            ╭─╯               ╰─╮
       │          ╭─╯                   ╰─╮
     5 ┤        ╭─╯                       ╰─╮
       │      ╭─╯                           ╰─
     1 ┼──────╯
       └──────┬──────┬──────┬──────┬──────┬────▶ Stress
            0%     20%    40%    60%    80%   100%
         Healthy  Mild  Moderate Severe Critical
```

Severely stressed plants (>80%) have collapsed xylem and can no longer emit — they are dying.

---

## Results

### Vision Pipeline Performance

| Component | Model | Accuracy |
|-----------|-------|----------|
| Plant Classification | MobileNetV2 | 95.4% |
| Segmentation | SAM ViT-B / FastSAM | High IoU |
| Stress Estimation | Confidence-based | Validated |

### Generated Plant Speech Examples

| Stress | Plant Voice |
|--------|-------------|
| 7% | *"I feel utterly serene, with the warm sunlight dancing across my leaves, fueling the gentle hum of photosynthesis that sustains me..."* |
| 56% | *"I'm consumed by a searing anguish... my stomata are desperately closing to conserve what little water I have left..."* |
| 83% | *"I'm withering away, my leaves wilting as the infection spreads, suffocating my ability to transport water and nutrients..."* |

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Computer Vision** | SAM / FastSAM, MobileNetV2, Grad-CAM, OpenCV |
| **Audio** | Conditional Diffusion UNet, librosa, scipy, Griffin-Lim |
| **LLM** | Groq API (Llama 3.3 70B) |
| **TTS** | Edge-TTS (Microsoft) |
| **Backend** | FastAPI, Uvicorn, Docker |
| **Frontend** | Next.js 15, React 19, Tailwind CSS 4, TypeScript |
| **Deployment** | Vercel (portfolio), HuggingFace Spaces (webapp), Docker (API) |

---

## Future Work

- [ ] Real hardware validation with ultrasonic microphones
- [ ] Mobile app with on-device inference (TFLite / CoreML)
- [ ] Multi-plant analysis from a single image
- [ ] Species-specific fine-tuned models
- [ ] IoT integration with soil moisture and temperature sensors
- [ ] HiFi-GAN vocoder for higher quality audio

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Tel Aviv University** — Groundbreaking plant acoustics research and dataset
- **Meta AI** — Segment Anything Model (SAM)
- **HuggingFace** — Transformers and model hosting
- **Groq** — Fast LLM inference API
- **Microsoft** — Edge-TTS

---

## Author

**Mohith**
B.Tech Chemical Engineering + AI/ML Minor
Indian Institute of Technology Bombay

[![GitHub](https://img.shields.io/badge/GitHub-Iammohithhh-black)](https://github.com/Iammohithhh)

---

<p align="center">
  <strong>If you found this project interesting, please consider giving it a star!</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Iammohithhh/plantwhisper?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/Iammohithhh/plantwhisper?style=social" alt="Forks"/>
</p>
