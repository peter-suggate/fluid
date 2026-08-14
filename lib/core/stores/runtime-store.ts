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
  simRate: null,
  setRunState: (runState) => set({ runState }),
  setSimulationTime: (simulationTime) => set({ simulationTime }),
  resetSimulationTime: () => set((state) => ({ simulationTime: 0, simulationEpoch: state.simulationEpoch + 1 })),
  setNotice: (notice, tone = "info") => set({ notice, noticeTone: tone }),
  setSimRate: (simRate) => set({ simRate })
}));
