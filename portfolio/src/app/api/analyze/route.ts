import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 300; // 5 min — Vercel Hobby Fluid Compute limit
export const dynamic = "force-dynamic";

const HF_BACKEND =
  process.env.HF_API_URL || "https://mohithmaddur-plantwhisper-api.hf.space";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const res = await fetch(`${HF_BACKEND}/api/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "Unknown error");
      return NextResponse.json(
        { error: `Backend returned ${res.status}: ${text}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to reach backend: ${message}` },
      { status: 502 }
    );
  }
}
