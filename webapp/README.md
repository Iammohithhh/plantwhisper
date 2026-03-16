---
title: PlantWhisper
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌱 PlantWhisper

**"Listen to what your plants are trying to tell you"**

PlantWhisper is a multimodal AI system that analyzes plant photos to:
- Detect plant health and stress levels
- Generate visual attention maps showing areas of concern
- Provide personalized care recommendations
- Give your plant a "voice" using LLM + TTS

## Features

- **SAM Segmentation** - Isolates plant from background
- **MobileNetV2 Classifier** - Detects health status with 95%+ accuracy
- **Grad-CAM Explainability** - Shows where the AI is focusing
- **Stress Modeling** - Based on Tel Aviv University plant acoustics research
- **Care Recommendations** - AI-generated actionable advice
- **Plant Persona** - LLM-generated speech + text-to-speech

## Scientific Foundation

Based on [Khait et al. 2023 (Cell)](https://www.cell.com/cell/fulltext/S0092-8674(23)00262-3) - 
Plants emit ultrasonic clicks (20-150 kHz) when stressed through xylem cavitation.

## Usage

1. Upload a photo of your plant leaf
2. Click "Analyze Plant"
3. View results: segmentation, attention map, stress level
4. Read care recommendations
5. Listen to your plant's "voice"

## Environment Variables

Set `GROQ_API_KEY` in HuggingFace Spaces secrets for LLM features.

## Created By

**Mohith** | IIT Bombay | 2026

Chemical Engineering + AI/ML Minor
