import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = incoming.get("x-forwarded-host") ?? incoming.get("host") ?? "localhost:3000";
  const protocol = incoming.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const image = `${protocol}://${host}/og-frame-graph.png`;
  const title = "Fluid Lab — WebGPU CFD Workbench";
  const description = "A validation-first interactive Eulerian free-surface fluid laboratory powered by WebGPU.";
  return {
    title,
    description,
    openGraph: { title, description, type: "website", images: [{ url: image, width: 1536, height: 1024, alt: "Fluid Lab CPU, GPU, and async frame graph" }] },
    twitter: { card: "summary_large_image", title, description, images: [image] }
  };
}

/**
 * The stored theme, applied before the first paint.
 *
 * `system` is the default and is the absence of the attribute, so this only has
 * anything to do when the reader has chosen otherwise — which is exactly the
 * case that would otherwise flash the wrong palette. Kept deliberately tiny and
 * in sync with `lib/stores/theme-store.ts`.
 */
const THEME_BOOT = `try{var t=localStorage.getItem("fluid-lab-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en">
    <head><script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} /></head>
    <body>{children}</body>
  </html>;
}
