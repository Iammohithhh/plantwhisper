import type { NextConfig } from "next";

const HF_BACKEND =
  process.env.HF_API_URL || "https://Iammohithhh-plantwhisper.hf.space";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/hf-api/:path*",
        destination: `${HF_BACKEND}/:path*`,
      },
    ];
  },
};

export default nextConfig;
