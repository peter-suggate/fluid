"use client";

// Composition root for the browser. The server components under `app/` are a
// different module graph from the one the browser and its SSR pass evaluate,
// so wiring the root layout wires neither: this is the outermost `use client`
// module the router reaches, and installing the catalog here is what makes
// every method resolvable by id in the browser.
import "../lib/methods";
import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useShellStore } from "../lib/core/stores/shell-store";
import { FluidLab } from "./FluidLab";

/**
 * Route ownership with a retained studio runtime.
 *
 * The library and a viewed scene are distinct pages, but an already-opened
 * studio is kept as a sibling of the active route content. Hiding it leaves the
 * exact canvas mounted, which is the identity the OffscreenCanvas worker and
 * its WebGPU device are attached to. The studio is lazy so a first visit to the
 * library still does no GPU work.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const sceneVisible = pathname === "/scene" || pathname.startsWith("/scene/");
  const studioEntered = useShellStore((state) => state.studioEntered);
  const studioMounted = studioEntered || sceneVisible;
  return <>
    {studioMounted && (
      <div className="persistent-studio-host" hidden={!sceneVisible} aria-hidden={!sceneVisible || undefined}>
        <FluidLab />
      </div>
    )}
    <div className="route-page-host" hidden={sceneVisible} aria-hidden={sceneVisible || undefined}>
      {children}
    </div>
  </>;
}
