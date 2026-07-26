import { create } from "zustand";

interface PerformanceInstrumentationStore {
  enabled: boolean;
  /** Old samples captured before this instant must not reappear after re-enabling. */
  enabledAt_ms: number;
  setEnabled: (enabled: boolean) => void;
}

/**
 * Controls measurement-only work. Correctness synchronization, diagnostics,
 * and presentation completion tracking are intentionally outside this switch.
 */
export const usePerformanceInstrumentationStore = create<PerformanceInstrumentationStore>((set) => ({
  // Match the Dawn production lane until the user explicitly asks to measure:
  // no timestamp queries, trace readbacks, forced encoder splits, or
  // measurement-only queue fences on the default simulation path.
  enabled: false,
  enabledAt_ms: -Infinity,
  setEnabled: (enabled) => set((state) => {
    if (enabled === state.enabled) return state;
    return {
      enabled,
      enabledAt_ms: enabled ? performance.now() : Infinity,
    };
  }),
}));
