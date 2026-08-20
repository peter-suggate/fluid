import { create } from "zustand";
import type { RunState } from "../model";

export type NoticeTone = "info" | "warn";

/** Transport state: whether the clock advances, where it is, and the status line. */
interface RuntimeStore {
  runState: RunState;
  simulationTime: number;
  /** Monotonic identity for the t=0 simulation timeline. */
  simulationEpoch: number;
  notice: string;
  noticeTone: NoticeTone;
  /**
   * How many times something has been said, rather than what was said.
   *
   * The transport fades a notice out once it has been read, and the same
   * sentence twice — two resets in a row — is two notices, not a stale one. The
   * text alone cannot tell those apart.
   */
  noticeSaid: number;
  /** Simulated seconds per wall-clock second over the recent window; null while paused. */
  simRate: number | null;
  setRunState: (runState: RunState) => void;
  setSimulationTime: (simulationTime: number) => void;
  resetSimulationTime: () => void;
  setNotice: (notice: string, tone?: NoticeTone) => void;
  setSimRate: (simRate: number | null) => void;
}

export const useRuntimeStore = create<RuntimeStore>((set) => ({
  runState: "running",
  simulationTime: 0,
  simulationEpoch: 0,
  notice: "Dam-break initialized · Eulerian projection active",
  noticeTone: "info",
  noticeSaid: 0,
  simRate: null,
  setRunState: (runState) => set({ runState }),
  setSimulationTime: (simulationTime) => set({ simulationTime }),
  resetSimulationTime: () => set((state) => ({ simulationTime: 0, simulationEpoch: state.simulationEpoch + 1 })),
  setNotice: (notice, tone = "info") => set((state) => ({ notice, noticeTone: tone, noticeSaid: state.noticeSaid + 1 })),
  setSimRate: (simRate) => set({ simRate })
}));
