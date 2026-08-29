"use client";

import { useEffect, useLayoutEffect } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { startQueryStateSync } from "../lib/core/url-state";
import { startSceneAutosave } from "../lib/core/scene-autosave";
import { browserSceneLibraryStorage } from "../lib/core/scene-library";
import { defaultSession } from "../lib/core/session/session";
import { SessionProvider, useSession } from "../lib/core/session/session-context";
import { useShellStore } from "../lib/core/stores/shell-store";
import { CompareHost } from "./CompareHost";

/**
 * The studio's root: the page-level wiring, and the shell that draws the scene.
 *
 * Everything visible moved into `CompareHost`, which draws one pane or two.
 * What is left here is what is true of the *page* however many panes there are:
 * the address bar is mirrored from pane A, autosave saves pane A (pane B is a
 * diff and is never a library entry unless it is promoted), and one animation
 * frame drives the one host clock both panes advance against.
 */
export function FluidLab() {
  const session = useSession();

  useLayoutEffect(() => {
    // A scene has its own route. During client navigation (and Fast Refresh)
    // the opened document is already in the stores, so keep it rather than
    // rebuilding an incomplete URL projection over it. A direct page load has
    // fresh stores and hydrates from the address as before.
    const hydrateFromUrl = !useShellStore.getState().studioEntered;
    useShellStore.getState().enterStudio();
    return startQueryStateSync(() => {
      // /scene is authoritative even for an old link carrying view=library.
      // Enter before the canonical URL write so that legacy flag is removed.
      useShellStore.getState().enterStudio();
      simulation.reset(undefined, undefined, session.id);
    }, { hydrateFromUrl, session });
  }, [session]);
  // After the URL has hydrated the document, so the first autosave records the
  // scene the reader actually arrived on rather than the default preset.
  useEffect(() => startSceneAutosave({ storage: browserSceneLibraryStorage() }, session), [session]);

  useEffect(() => {
    let frame = 0;
    const tick = (now: number) => { simulation.tick(now); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  // Every pane of chrome authors and reads one realm. This is pane A's; compare
  // mode mounts a second provider around a second `.viewport-shell` inside the
  // host, and nothing below there learns a new code path.
  return (
    <SessionProvider value={defaultSession}>
      <CompareHost />
    </SessionProvider>
  );
}
