import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const HF_BACKEND =
  process.env.HF_API_URL || "https://mohithmaddur-plantwhisper-api.hf.space";

export async function GET() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const res = await fetch(`${HF_BACKEND}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.ok) {
      return NextResponse.json({ status: "ok" });
    }
    return NextResponse.json({ status: "unhealthy" }, { status: 502 });
  } catch {
    return NextResponse.json({ status: "unreachable" }, { status: 503 });
  }
}
