import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = incoming.get("x-forwarded-host") ?? incoming.get("host") ?? "localhost:3000";
  const protocol = incoming.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const image = `${protocol}://${host}/og.png`;
  const title = "Fluid Lab — WebGPU CFD Workbench";
  const description = "A validation-first interactive Eulerian free-surface fluid laboratory powered by WebGPU.";
  return {
    title,
    description,
    openGraph: { title, description, type: "website", images: [{ url: image, width: 1536, height: 1024, alt: "Fluid Lab Eulerian free-surface simulation" }] },
    twitter: { card: "summary_large_image", title, description, images: [image] }
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
