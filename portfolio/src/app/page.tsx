"use client";

import { useState, useRef } from "react";
import Image from "next/image";

/* ─────────────── HERO ─────────────── */
function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-950 via-emerald-900 to-emerald-950" />
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-20 left-10 w-72 h-72 bg-emerald-500 rounded-full blur-[120px] animate-float" />
        <div
          className="absolute bottom-20 right-10 w-96 h-96 bg-emerald-400 rounded-full blur-[150px] animate-float"
          style={{ animationDelay: "3s" }}
        />
        <div
          className="absolute top-1/2 left-1/2 w-64 h-64 bg-emerald-300 rounded-full blur-[100px] animate-float"
          style={{ animationDelay: "1.5s" }}
        />
      </div>

      <div className="relative z-10 text-center px-6 max-w-5xl mx-auto">
        <div className="animate-slide-up opacity-0">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8 text-emerald-300 text-sm font-medium">
            <span className="sound-bar" style={{ animationDelay: "0s" }} />
            <span
              className="sound-bar"
              style={{ animationDelay: "0.2s" }}
            />
            <span
              className="sound-bar"
              style={{ animationDelay: "0.4s" }}
            />
            <span
              className="sound-bar"
              style={{ animationDelay: "0.1s" }}
            />
            <span
              className="sound-bar"
              style={{ animationDelay: "0.3s" }}
            />
            <span className="ml-2">Powered by Diffusion Models</span>
          </div>
        </div>

        <h1 className="text-6xl md:text-8xl font-black tracking-tight mb-6 animate-slide-up opacity-0 stagger-1">
          Plant<span className="gradient-text">Whisper</span>
        </h1>

        <p className="text-xl md:text-2xl text-emerald-200/80 max-w-2xl mx-auto mb-4 font-light animate-slide-up opacity-0 stagger-2">
          Give your plants a voice.
        </p>
        <p className="text-base md:text-lg text-emerald-300/60 max-w-xl mx-auto mb-12 animate-slide-up opacity-0 stagger-3">
          AI-powered health analysis with ultrasonic sound synthesis — hear the
          clicks your stressed plants actually emit.
        </p>

        <div className="flex flex-wrap justify-center gap-4 animate-slide-up opacity-0 stagger-4">
          <a
            href="#demo"
            className="px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold rounded-xl transition-all hover:scale-105 hover:shadow-lg hover:shadow-emerald-500/25"
          >
            Try Live Demo
          </a>
          <a
            href="#how-it-works"
            className="px-8 py-4 glass hover:bg-white/10 text-white font-semibold rounded-xl transition-all"
          >
            How It Works
          </a>
        </div>

        <div className="mt-16 animate-slide-up opacity-0 stagger-5">
          <p className="text-emerald-400/50 text-xs uppercase tracking-widest mb-4">
            Built with
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              "SAM",
              "MobileNetV2",
              "Grad-CAM",
              "Diffusion UNet",
              "Groq LLM",
              "Edge TTS",
            ].map((tech) => (
              <span
                key={tech}
                className="px-3 py-1.5 text-xs font-mono text-emerald-300 glass rounded-lg"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <svg
          className="w-6 h-6 text-emerald-400/50"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 14l-7 7m0 0l-7-7m7 7V3"
          />
        </svg>
      </div>
    </section>
  );
}

/* ─────────────── FEATURES ─────────────── */
const features = [
  {
    icon: "🔬",
    title: "Plant Segmentation",
    desc: "FastSAM isolates the plant from any background with pixel-perfect precision.",
    color: "from-emerald-500 to-teal-500",
  },
  {
    icon: "🧬",
    title: "Disease Detection",
    desc: "MobileNetV2 classifier identifies across 38 plant disease categories with 95%+ accuracy.",
    color: "from-teal-500 to-cyan-500",
  },
  {
    icon: "🔥",
    title: "Attention Heatmaps",
    desc: "Grad-CAM visualizes exactly where the model sees problems on your plant.",
    color: "from-cyan-500 to-blue-500",
  },
  {
    icon: "🎵",
    title: "Ultrasonic Synthesis",
    desc: "Diffusion model generates the actual ultrasonic clicks stressed plants emit via xylem cavitation.",
    color: "from-violet-500 to-purple-500",
  },
  {
    icon: "🗣️",
    title: "Plant Persona",
    desc: "LLM generates emotional first-person speech matching your plant's stress level.",
    color: "from-purple-500 to-pink-500",
  },
  {
    icon: "💊",
    title: "Care Recommendations",
    desc: "AI-generated, severity-appropriate care instructions to nurse your plant back to health.",
    color: "from-pink-500 to-rose-500",
  },
];

function Features() {
  return (
    <section id="features" className="py-32 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-20">
          <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
            Capabilities
          </p>
          <h2 className="text-4xl md:text-5xl font-bold">
            Six AI Models.{" "}
            <span className="text-emerald-400">One Pipeline.</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <div
              key={f.title}
              className={`group relative p-8 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:border-emerald-500/30 transition-all duration-500 hover:-translate-y-1 opacity-0 animate-slide-up stagger-${i + 1}`}
            >
              <div
                className={`inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br ${f.color} text-2xl mb-5`}
              >
                {f.icon}
              </div>
              <h3 className="text-xl font-bold mb-3 group-hover:text-emerald-400 transition-colors">
                {f.title}
              </h3>
              <p className="text-emerald-200/60 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ─────────────── HOW IT WORKS ─────────────── */
const steps = [
  {
    num: "01",
    title: "Upload",
    desc: "Drop a photo of your plant — any angle, any background.",
  },
  {
    num: "02",
    title: "Segment & Classify",
    desc: "FastSAM isolates the plant. MobileNetV2 detects disease across 38 classes.",
  },
  {
    num: "03",
    title: "Stress Analysis",
    desc: "The model estimates a 0–100% stress score with Grad-CAM attention maps.",
  },
  {
    num: "04",
    title: "Sound Synthesis",
    desc: "A conditional diffusion UNet generates ultrasonic spectrograms, converted to audio via Griffin-Lim.",
  },
  {
    num: "05",
    title: "Voice & Care",
    desc: "LLM crafts an emotional plant monologue. Edge TTS voices it. You get care instructions.",
  },
];

function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="py-32 px-6 bg-gradient-to-b from-emerald-950 via-emerald-900/50 to-emerald-950"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-20">
          <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
            Pipeline
          </p>
          <h2 className="text-4xl md:text-5xl font-bold">
            From Photo to <span className="gradient-text">Plant Voice</span>
          </h2>
        </div>

        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-emerald-500/0 via-emerald-500/50 to-emerald-500/0 hidden md:block" />

          <div className="space-y-12">
            {steps.map((s) => (
              <div key={s.num} className="flex gap-8 items-start group">
                <div className="hidden md:flex shrink-0 w-16 h-16 items-center justify-center rounded-2xl bg-emerald-500/10 border border-emerald-500/20 group-hover:bg-emerald-500/20 group-hover:border-emerald-500/40 transition-all">
                  <span className="text-emerald-400 font-black text-lg">
                    {s.num}
                  </span>
                </div>
                <div className="pt-2">
                  <h3 className="text-2xl font-bold mb-2 group-hover:text-emerald-400 transition-colors">
                    <span className="md:hidden text-emerald-500 mr-2">
                      {s.num}.
                    </span>
                    {s.title}
                  </h3>
                  <p className="text-emerald-200/60 text-lg leading-relaxed">
                    {s.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-20 rounded-2xl overflow-hidden border border-white/10">
          <Image
            src="/pipeline.png"
            alt="PlantWhisper Architecture Pipeline"
            width={736}
            height={634}
            className="w-full"
          />
        </div>
      </div>
    </section>
  );
}

/* ─────────────── SCIENCE ─────────────── */
function Science() {
  return (
    <section className="py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
            Scientific Foundation
          </p>
          <h2 className="text-4xl md:text-5xl font-bold">
            Plants Actually <span className="gradient-text">Make Sounds</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="p-8 rounded-2xl glass">
            <h3 className="text-xl font-bold text-emerald-300 mb-4">
              The Discovery
            </h3>
            <p className="text-emerald-200/70 leading-relaxed mb-4">
              In 2023, researchers at Tel Aviv University published a
              groundbreaking study in <em>Cell</em> showing that stressed plants
              emit ultrasonic clicks (20–150 kHz) through xylem cavitation — air
              bubbles forming in their water transport vessels.
            </p>
            <p className="text-emerald-200/70 leading-relaxed">
              Drought-stressed plants peak at ~35 clicks per hour. These sounds
              are inaudible to humans but carry real information about plant
              health.
            </p>
          </div>

          <div className="p-8 rounded-2xl glass">
            <h3 className="text-xl font-bold text-emerald-300 mb-4">
              Our Approach
            </h3>
            <p className="text-emerald-200/70 leading-relaxed mb-4">
              PlantWhisper uses a <strong>conditional diffusion model</strong>{" "}
              trained to generate mel spectrograms of these ultrasonic pops,
              conditioned on the plant&apos;s detected stress level.
            </p>
            <p className="text-emerald-200/70 leading-relaxed">
              The generated spectrograms are converted to audible audio via
              Griffin-Lim reconstruction, pitch-shifted into human hearing range
              so you can literally hear your plant&apos;s distress.
            </p>
          </div>
        </div>

        <div className="mt-10 p-6 rounded-2xl bg-emerald-900/30 border border-emerald-800/50 text-center">
          <p className="text-emerald-300/80 text-sm">
            <strong>Reference:</strong> Khait, I., et al. (2023).
            &ldquo;Sounds emitted by plants under stress are airborne and
            informative.&rdquo; <em>Cell</em>, 186(7), 1328–1336.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ─────────────── TECH STACK ─────────────── */
const techStack = [
  {
    category: "Vision",
    items: [
      { name: "FastSAM", role: "Plant segmentation" },
      { name: "MobileNetV2", role: "38-class disease detection" },
      { name: "Grad-CAM", role: "Attention visualization" },
    ],
  },
  {
    category: "Audio",
    items: [
      { name: "Diffusion UNet", role: "Spectrogram generation" },
      { name: "Griffin-Lim", role: "Audio reconstruction" },
      { name: "Edge TTS", role: "Plant voice synthesis" },
    ],
  },
  {
    category: "Intelligence",
    items: [
      { name: "Groq + Llama 3.3 70B", role: "Plant persona & care advice" },
      { name: "PyTorch", role: "Deep learning framework" },
      { name: "Next.js + Vercel", role: "Portfolio & deployment" },
    ],
  },
];

function TechStack() {
  return (
    <section
      id="tech"
      className="py-32 px-6 bg-gradient-to-b from-emerald-950 via-emerald-900/30 to-emerald-950"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
            Technology
          </p>
          <h2 className="text-4xl md:text-5xl font-bold">
            Under the <span className="gradient-text">Hood</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {techStack.map((group) => (
            <div key={group.category} className="space-y-4">
              <h3 className="text-lg font-bold text-emerald-400 uppercase tracking-wider">
                {group.category}
              </h3>
              {group.items.map((item) => (
                <div
                  key={item.name}
                  className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:border-emerald-500/30 transition-all"
                >
                  <p className="font-semibold text-white">{item.name}</p>
                  <p className="text-sm text-emerald-200/50">{item.role}</p>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ─────────────── DEMO ─────────────── */
function Demo() {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    stress: number;
    category: string;
    color: string;
    label: string;
    confidence: number;
    popsPerHour: number;
    speech: string;
    recommendations: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const lastFileRef = useRef<File | null>(null);

  async function waitForBackend(): Promise<boolean> {
    for (let i = 0; i < 8; i++) {
      try {
        setStatusMsg(
          i === 0
            ? "Checking backend..."
            : `Waking up HF Space (${i}/7)...`
        );
        const res = await fetch("/api/health");
        if (res.ok) {
          setStatusMsg(null);
          return true;
        }
      } catch {
        // Space still waking, wait and retry
      }
      await new Promise((r) => setTimeout(r, 5000));
    }
    return false;
  }

  async function analyzeImage(file: File): Promise<void> {
    setLoading(true);
    setError(null);
    setResult(null);
    setStatusMsg(null);
    lastFileRef.current = file;

    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target?.result as string);
    reader.readAsDataURL(file);

    // Step 1: Make sure backend is awake
    const isUp = await waitForBackend();
    if (!isUp) {
      setLoading(false);
      setError(
        "HuggingFace Space is not responding. Please visit https://huggingface.co/spaces/Iammohithhh/plantwhisper to check if it's running, then retry."
      );
      return;
    }

    // Step 2: Call analysis through Vercel serverless proxy (300s timeout, no CORS issues)
    setStatusMsg("Analyzing your plant — this takes ~60s on CPU...");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Analysis failed (HTTP ${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Analysis request failed — try again.";
      setError(msg);
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) analyzeImage(file);
  }

  const stressPercent = result ? Math.round(result.stress * 100) : 0;

  return (
    <section id="demo" className="py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
            Live Demo
          </p>
          <h2 className="text-4xl md:text-5xl font-bold">
            Try <span className="gradient-text">PlantWhisper</span>
          </h2>
          <p className="text-emerald-200/50 mt-4">
            Upload a plant photo and watch the AI analyze it in real time.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload */}
          <div>
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileRef.current?.click()}
              className="relative h-80 rounded-2xl border-2 border-dashed border-emerald-500/30 hover:border-emerald-400/60 bg-emerald-900/20 flex flex-col items-center justify-center cursor-pointer transition-all group"
            >
              {image ? (
                <img
                  src={image}
                  alt="Upload"
                  className="h-full w-full object-contain rounded-2xl p-2"
                />
              ) : (
                <>
                  <svg
                    className="w-16 h-16 text-emerald-500/40 group-hover:text-emerald-400/60 transition-colors mb-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <p className="text-emerald-300/60 font-medium">
                    Drop a plant photo or click to upload
                  </p>
                  <p className="text-emerald-400/30 text-sm mt-1">
                    JPG, PNG — any resolution
                  </p>
                </>
              )}
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) analyzeImage(file);
                }}
              />
            </div>

            {/* Sample images */}
            <div className="mt-4 flex gap-3">
              <p className="text-emerald-400/40 text-sm self-center">
                Or try:
              </p>
              {[
                { src: "/corn_healthy.jpg", label: "Healthy" },
                { src: "/corn_diseased.jpg", label: "Diseased" },
              ].map((sample) => (
                <button
                  key={sample.label}
                  onClick={async () => {
                    const res = await fetch(sample.src);
                    const blob = await res.blob();
                    const file = new File([blob], "sample.jpg", {
                      type: "image/jpeg",
                    });
                    analyzeImage(file);
                  }}
                  className="px-4 py-2 text-sm glass rounded-lg hover:bg-white/10 text-emerald-300 transition-all"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          {/* Results */}
          <div className="space-y-4">
            {loading && (
              <div className="h-80 rounded-2xl glass flex items-center justify-center">
                <div className="text-center">
                  <div className="flex justify-center gap-1 mb-4">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <span
                        key={i}
                        className="sound-bar h-8"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </div>
                  <p className="text-emerald-300/80">
                    {statusMsg || "Analyzing your plant..."}
                  </p>
                  <p className="text-emerald-400/40 text-sm mt-1">
                    Running 6 AI models on CPU — please be patient
                  </p>
                </div>
              </div>
            )}

            {error && (
              <div className="h-80 rounded-2xl glass flex items-center justify-center p-8">
                <div className="text-center">
                  <p className="text-amber-400 font-semibold mb-2">
                    Something went wrong
                  </p>
                  <p className="text-emerald-200/50 text-sm">{error}</p>
                  <button
                    onClick={() => {
                      if (lastFileRef.current) {
                        analyzeImage(lastFileRef.current);
                      } else {
                        fileRef.current?.click();
                      }
                    }}
                    className="mt-4 px-6 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-medium transition-colors"
                  >
                    Retry
                  </button>
                </div>
              </div>
            )}

            {result && (
              <div className="space-y-4 animate-fade-in">
                {/* Stress gauge */}
                <div className="p-6 rounded-2xl glass">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-emerald-200/60 text-sm font-medium">
                      Stress Level
                    </span>
                    <span
                      className="text-3xl font-black"
                      style={{ color: result.color }}
                    >
                      {stressPercent}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-emerald-900/50 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-1000"
                      style={{
                        width: `${stressPercent}%`,
                        background: `linear-gradient(90deg, #10b981, ${result.color})`,
                      }}
                    />
                  </div>
                  <p
                    className="text-sm font-semibold mt-2"
                    style={{ color: result.color }}
                  >
                    {result.category}
                  </p>
                </div>

                {/* Details */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <p className="text-emerald-400/50 text-xs uppercase">
                      Classification
                    </p>
                    <p className="text-white font-semibold text-sm mt-1">
                      {result.label}
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <p className="text-emerald-400/50 text-xs uppercase">
                      Confidence
                    </p>
                    <p className="text-white font-semibold text-sm mt-1">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <p className="text-emerald-400/50 text-xs uppercase">
                      Ultrasonic Clicks
                    </p>
                    <p className="text-white font-semibold text-sm mt-1">
                      {result.popsPerHour.toFixed(1)} / hour
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <p className="text-emerald-400/50 text-xs uppercase">
                      Audio Method
                    </p>
                    <p className="text-white font-semibold text-sm mt-1">
                      Diffusion
                    </p>
                  </div>
                </div>

                {/* Plant voice */}
                <div className="p-5 rounded-xl bg-emerald-900/30 border border-emerald-800/40">
                  <p className="text-emerald-400/50 text-xs uppercase mb-2">
                    Your Plant Says
                  </p>
                  <p className="text-emerald-100/80 italic leading-relaxed">
                    &ldquo;{result.speech}&rdquo;
                  </p>
                </div>

                {/* Care */}
                <div className="p-5 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                  <p className="text-emerald-400/50 text-xs uppercase mb-2">
                    Care Recommendations
                  </p>
                  <p className="text-emerald-200/70 text-sm whitespace-pre-line leading-relaxed">
                    {result.recommendations}
                  </p>
                </div>
              </div>
            )}

            {!loading && !error && !result && (
              <div className="h-80 rounded-2xl glass flex items-center justify-center">
                <p className="text-emerald-300/40">
                  Results will appear here
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─────────────── ABOUT ─────────────── */
function About() {
  return (
    <section
      id="about"
      className="py-32 px-6 bg-gradient-to-b from-emerald-950 via-emerald-900/30 to-emerald-950"
    >
      <div className="max-w-3xl mx-auto text-center">
        <p className="text-emerald-400 font-semibold text-sm uppercase tracking-widest mb-3">
          About
        </p>
        <h2 className="text-4xl md:text-5xl font-bold mb-8">
          Built at <span className="gradient-text">IIT Bombay</span>
        </h2>
        <p className="text-emerald-200/70 text-lg leading-relaxed mb-6">
          PlantWhisper was created by <strong>Mohith</strong> — B.Tech Chemical
          Engineering with AI/ML Minor at the Indian Institute of Technology
          Bombay (Class of 2027).
        </p>
        <p className="text-emerald-200/50 leading-relaxed">
          This project explores the intersection of computer vision, generative
          audio, and plant biology — turning cutting-edge acoustic research into
          an accessible, multimodal AI experience.
        </p>
      </div>
    </section>
  );
}

/* ─────────────── FOOTER ─────────────── */
function Footer() {
  return (
    <footer className="py-12 px-6 border-t border-white/5">
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <p className="text-emerald-400/40 text-sm">
          PlantWhisper &copy; {new Date().getFullYear()}
        </p>
        <div className="flex gap-6">
          {["Features", "How It Works", "Tech", "Demo", "About"].map(
            (item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(/ /g, "-")}`}
                className="text-emerald-300/40 hover:text-emerald-300 text-sm transition-colors"
              >
                {item}
              </a>
            )
          )}
        </div>
      </div>
    </footer>
  );
}

/* ─────────────── PAGE ─────────────── */
export default function Home() {
  return (
    <main>
      <Hero />
      <Features />
      <HowItWorks />
      <Science />
      <TechStack />
      <Demo />
      <About />
      <Footer />
    </main>
  );
}
