import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PlantWhisper — Give Your Plants a Voice",
  description:
    "AI-powered plant health analysis with ultrasonic sound synthesis using diffusion models.",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/favicon.png", type: "image/png" },
    ],
  },
  openGraph: {
    title: "PlantWhisper",
    description: "Hear what your plants are trying to tell you.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-emerald-950 text-white antialiased">{children}</body>
    </html>
  );
}
