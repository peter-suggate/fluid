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
  // Hardware timestamp support is adapter-dependent, and the portable
  // segmented fallback deliberately serializes semantic phases. Keep that
  // measurement load opt-in so normal simulation throughput is never changed
  // merely by opening the application.
  enabled: false,
  enabledAt_ms: Infinity,
  setEnabled: (enabled) => set((state) => {
    if (enabled === state.enabled) return state;
    return {
      enabled,
      enabledAt_ms: enabled ? performance.now() : Infinity,
    };
  }),
}));
