import { create } from "zustand";
import type { RunState } from "../model";

export type NoticeTone = "info" | "warn";

export const MIN_TARGET_SIM_RATE = 0.1;
export const MAX_TARGET_SIM_RATE = 4;

export function clampTargetSimRate(rate: number): number {
  if (!Number.isFinite(rate)) return 1;
  return Math.min(MAX_TARGET_SIM_RATE, Math.max(MIN_TARGET_SIM_RATE, rate));
}

/** Transport state: whether the clock advances, where it is, and the status line. */
interface RuntimeStore {
  runState: RunState;
  simulationTime: number;
  /** Monotonic identity for the t=0 simulation timeline. */
  simulationEpoch: number;
  notice: string;
  noticeTone: NoticeTone;
  /** Desired simulated seconds per wall-clock second. */
  targetSimRate: number;
  /** Simulated seconds per wall-clock second over the recent window; null while paused. */
  simRate: number | null;
  setRunState: (runState: RunState) => void;
  setSimulationTime: (simulationTime: number) => void;
  resetSimulationTime: () => void;
  setNotice: (notice: string, tone?: NoticeTone) => void;
  setTargetSimRate: (targetSimRate: number) => void;
  setSimRate: (simRate: number | null) => void;
}

export const useRuntimeStore = create<RuntimeStore>((set) => ({
  runState: "running",
  simulationTime: 0,
  simulationEpoch: 0,
  notice: "Dam-break initialized · Eulerian projection active",
  noticeTone: "info",
  targetSimRate: 1,
  simRate: null,
  setRunState: (runState) => set({ runState }),
  setSimulationTime: (simulationTime) => set({ simulationTime }),
  resetSimulationTime: () => set((state) => ({ simulationTime: 0, simulationEpoch: state.simulationEpoch + 1 })),
  setNotice: (notice, tone = "info") => set({ notice, noticeTone: tone }),
  setTargetSimRate: (targetSimRate) => set({ targetSimRate: clampTargetSimRate(targetSimRate) }),
  setSimRate: (simRate) => set({ simRate })
}));
