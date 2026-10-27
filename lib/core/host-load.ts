import { useEffect } from "react";
import { create } from "zustand";
import {
  LOAD_HISTORY_LENGTH,
  LOAD_SAMPLE_INTERVAL_MS,
  startThreadStallMeter,
  type CPUPressureState,
  type ThreadLoadSample,
} from "./system-load";

/**
 * The window's half of the load readings: its own thread, and the machine.
 *
 * The render worker measures itself (see `system-load.ts`); this measures the
 * thread React and the input handlers run on, which is the one a reader feels
 * as a frozen page, and asks the platform how pressed the whole CPU is. It runs
 * only while something on screen reads it, so an idle studio pays for nothing.
 */

interface HostLoadStore {
  readonly uiThread: readonly (ThreadLoadSample & { readonly at_ms: number })[];
  /** Absent where the browser has no Compute Pressure API, or refused it. */
  readonly cpuPressure?: CPUPressureState;
}

export const useHostLoadStore = create<HostLoadStore>(() => ({ uiThread: [] }));

interface PressureRecord { readonly state: CPUPressureState }
interface PressureObserverLike {
  observe(source: "cpu", options?: { sampleInterval?: number }): Promise<void>;
  disconnect(): void;
}
type PressureObserverConstructor = new (callback: (records: readonly PressureRecord[]) => void) => PressureObserverLike;

let readers = 0;
let stop: (() => void) | undefined;

function start(): () => void {
  const thread = startThreadStallMeter();
  const timer = setInterval(() => {
    // A hidden tab's timers are throttled to seconds; that lateness is the
    // browser's choice, not work, and would read as a frozen thread.
    if (document.hidden) { thread.meter.sample(performance.now()); return; }
    const now_ms = performance.now();
    const sample = { ...thread.meter.sample(now_ms), at_ms: now_ms };
    useHostLoadStore.setState((state) => ({ uiThread: [...state.uiThread.slice(1 - LOAD_HISTORY_LENGTH), sample] }));
  }, LOAD_SAMPLE_INTERVAL_MS);
  let observer: PressureObserverLike | undefined;
  const Pressure = (globalThis as { PressureObserver?: PressureObserverConstructor }).PressureObserver;
  if (Pressure) {
    observer = new Pressure((records) => {
      const latest = records.at(-1);
      if (latest) useHostLoadStore.setState({ cpuPressure: latest.state });
    });
    // Permission policy or an insecure context rejects; the reading is then simply absent.
    observer.observe("cpu", { sampleInterval: 1000 }).catch(() => { observer = undefined; });
  }
  return () => {
    thread.stop();
    clearInterval(timer);
    observer?.disconnect();
    useHostLoadStore.setState({ uiThread: [], cpuPressure: undefined });
  };
}

/** Keeps the window's meters running while the calling component is mounted and `active`. */
export function useHostLoadMeters(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    if (readers++ === 0) stop = start();
    return () => {
      if (--readers === 0) { stop?.(); stop = undefined; }
    };
  }, [active]);
}
