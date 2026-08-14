"use client";

import { useSyncExternalStore } from "react";
import { safeBrowserGPUBringupEnabled } from "./gpu-startup";

const subscribeToStaticURL = () => () => {};
const serverSnapshot = (): null => null;
const browserSnapshot = (): boolean => safeBrowserGPUBringupEnabled(window.location.search);

/**
 * URL policy cannot be read during SSR. Keep the server and first client
 * render on the same neutral, locked state, then resolve the immutable
 * browser query after hydration. `null` deliberately means controls must
 * remain locked; it is not equivalent to normal mode.
 */
export function useSafeBrowserGPUBringup(): boolean | null {
  return useSyncExternalStore(subscribeToStaticURL, browserSnapshot, serverSnapshot);
}
