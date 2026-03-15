# PlantWhisper — Implementation Plan

## What We're Building

A **multimodal plant stress predictor** where you photograph/describe your plant, provide sensor-like data, and get a first-person "plant voice" response that tells you how it's feeling — grounded in real science, not just vibes.

**First prototype goal:** Text-in, text-out. You describe your plant's situation → the system analyzes it → you get a scientifically-grounded plant personality response.

---

## Architecture Overview

```
User Input (text/image description)
        │
        ▼
┌──────────────────┐
│  Input Parser     │  ← Extracts structured plant data from natural language
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Stress Analyzer  │  ← Rule-based + ML scoring engine
│  (Core Brain)     │     Temporal tracking, multi-signal fusion
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Plant Persona    │  ← Generates first-person plant dialogue
│  Generator        │     grounded in stress analysis results
└──────────────────┘
       │
       ▼
  Text Output (plant "speaks")
```

---

## Phase 1: Foundation (What we build NOW)

### 1.1 — Project Structure & Data Models
- Set up Python package structure
- Define core data models:
  - `PlantProfile` — species, age, location, history
  - `PlantObservation` — snapshot of current state (leaf color, soil moisture, light, etc.)
  - `StressAssessment` — output of analysis (stress type, severity, confidence, prediction)

### 1.2 — Stress Analysis Engine (Rule-Based First)
- Encode real botanical science into rules:
  - **Water stress**: soil moisture trends, leaf turgor, wilting signs
  - **Light stress**: etiolation, leaf burn, phototropism signals
  - **Nutrient deficiency**: chlorosis patterns, leaf discoloration mapping
  - **Temperature stress**: leaf curling, cold damage, heat wilt
- Each stress type gets a severity score (0-1) and a temporal prediction
- Based on the 2023 Tel Aviv research: model the "ultrasonic emission rate" as a composite stress indicator
  - Simulated acoustic signal: stressed plants → 30-50 pops/hr, unstressed → near zero
  - This becomes our unified stress metric

### 1.3 — Temporal Tracking
- Store observation history per plant (simple JSON/SQLite)
- Trend analysis: "soil moisture dropping 5%/day → predict dehydration in X hours"
- This is the **key differentiator** — not just "what's wrong now" but "what will be wrong soon"

### 1.4 — Plant Persona Generator
- Template + logic system (no LLM dependency for v1)
- Maps stress assessments to first-person dialogue
- Personality varies by plant species (Monstera = dramatic, Cactus = stoic, Fern = anxious)
- Output includes:
  - How the plant "feels" (grounded in real signals)
  - What it "heard" (simulated ultrasonic reference)
  - What it needs (actionable advice)
  - Temporal warning if applicable

### 1.5 — CLI Interface
- Simple command-line "conversation" with your plant
- Commands:
  - `register` — add a new plant (species, name, location)
  - `observe` — log an observation (answer questions about leaf color, soil, light, etc.)
  - `talk` — hear what your plant has to say
  - `history` — see stress trends over time

---

## Phase 2: Vision Integration (Next)

### 2.1 — Image Analysis
- Accept plant photos
- Use a fine-tuned vision model (ViT on PlantVillage dataset) to detect:
  - Leaf health regions
  - Early chlorophyll degradation
  - Disease classification
- Generate attention/Grad-CAM maps showing "early warning regions"

### 2.2 — Multimodal Fusion
- Combine vision output + sensor data + temporal history
- Weighted fusion model that learns which signals matter most per species

---

## Phase 3: Full Intelligence (Later)

### 3.1 — Acoustic Simulation Model
- Train on the Tel Aviv dataset patterns
- Given stress indicators, simulate what the plant's ultrasonic emissions would sound like
- Generate audio visualizations (spectrograms)

### 3.2 — LLM-Powered Persona (upgrade from templates)
- Use an LLM to generate richer, more natural plant dialogue
- Grounded in structured stress data (no hallucination — LLM gets facts as context)

### 3.3 — Predictive Model
- LSTM/Transformer on temporal observation sequences
- Train to predict stress 24-72 hours ahead

---

## Tech Stack (Phase 1)

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11+ | ML ecosystem, rapid prototyping |
| Data models | Pydantic | Validation, serialization |
| Storage | SQLite via sqlite3 | Zero setup, good enough for v1 |
| CLI | Click | Clean CLI framework |
| Testing | pytest | Standard |
| Stress engine | Pure Python + numpy | No heavy deps for v1 |

---

## File Structure

```
plantwhisper/
├── pyproject.toml
├── README.md
├── src/
│   └── plantwhisper/
│       ├── __init__.py
│       ├── cli.py              # Click CLI interface
│       ├── models.py           # Pydantic data models
│       ├── storage.py          # SQLite persistence
│       ├── stress/
│       │   ├── __init__.py
│       │   ├── analyzer.py     # Main stress analysis engine
│       │   ├── water.py        # Water stress rules
│       │   ├── light.py        # Light stress rules
│       │   ├── nutrient.py     # Nutrient stress rules
│       │   └── temporal.py     # Trend analysis & prediction
│       ├── acoustic/
│       │   ├── __init__.py
│       │   └── simulator.py    # Simulated ultrasonic emission model
│       └── persona/
│           ├── __init__.py
│           ├── generator.py    # Plant dialogue generation
│           └── species.py      # Species-specific personality configs
└── tests/
    ├── test_models.py
    ├── test_analyzer.py
    ├── test_temporal.py
    └── test_persona.py
```

---

## Example Interaction (What Phase 1 looks like)

```
$ plantwhisper register
Plant name: Monty
Species: monstera deliciosa
Location: living room, east window

✓ Monty the Monstera registered!

$ plantwhisper observe --plant monty
Soil moisture (dry/moist/wet): moist
Leaf color (green/yellow-green/yellow/brown): yellow-green
Any wilting? (none/slight/moderate/severe): slight
Light exposure (low/medium/high): low
Days since last watering: 4

✓ Observation logged for Monty

$ plantwhisper talk --plant monty
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌿 Monty the Monstera says:

"Ugh, I've been trying to get your attention for days. My lower
leaves are starting to turn — see that yellow-green creeping in?
That's me waving a flag. My soil still has some moisture, but at
this rate I'm looking at real dehydration stress in about 48 hours.

And don't get me started on the light situation. I'm a tropical
understory plant, sure, but this corner is getting gloomy. I've
been reaching toward the window so hard my stems hurt.

If I could scream ultrasonically right now, I'd be popping about
35 times per hour. That's me stressed, friend.

What I need: Water me tomorrow morning, and scoot me 2 feet
closer to that window. I'll feel better by Thursday."

Stress Level: ██████░░░░ 62% (moderate)
Predicted: dehydration in ~48h without intervention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Build Order (Step by Step)

1. **Project setup** — pyproject.toml, package structure, dependencies
2. **Data models** — PlantProfile, PlantObservation, StressAssessment
3. **Storage layer** — SQLite CRUD for plants and observations
4. **Stress analyzers** — Water, light, nutrient, temperature rules
5. **Temporal engine** — Trend detection and future stress prediction
6. **Acoustic simulator** — Simulated ultrasonic pop rate from stress scores
7. **Persona generator** — Species personalities + dialogue templates
8. **CLI** — Wire it all together with Click
9. **Tests** — Unit tests for each component
10. **Polish** — README, example data, demo script
