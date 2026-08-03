"use client";

import { useEffect } from "react";
import { useShellStore } from "@/lib/stores/shell-store";
import { replaceQueryStateUrl, shellViewFromQuery } from "@/lib/url-state";

/**
 * The shell view's half of the query state.
 *
 * It lives apart from `startQueryStateSync` because the shell is not a source
 * of truth the simulation hydrates from: scene, method and UI are read once and
 * pushed into their stores, whereas the library layer is already open on a cold
 * load and only a link that says so may take it back. So this reads the URL
 * one way and writes it the other, through the same single writer.
 */
export function UrlStateSync() {
  useEffect(() => {
    if (shellViewFromQuery(window.location.search) === "library") useShellStore.getState().openLibrary();
    // Only the view is a URL state. The search box and the disclosure of the
    // research shelf change per keystroke and per click, and neither is worth
    // rewriting the address bar — or sharing — for.
    return useShellStore.subscribe((state, previous) => {
      if (state.view !== previous.view) replaceQueryStateUrl();
    });
  }, []);

  return null;
}
