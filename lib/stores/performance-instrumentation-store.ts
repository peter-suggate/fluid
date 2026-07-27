import { create } from "zustand";

export type PerformanceInstrumentationMode = "off" | "timeline" | "activity";

interface PerformanceInstrumentationStore {
  mode: PerformanceInstrumentationMode;
  enabled: boolean;
  /** True only for the shader-recompiled heartbeat variant. */
  shaderActivityEnabled: boolean;
  /** Pipeline caches must include this value in keys for instrumented WGSL. */
  shaderGeneration: number;
  /** Old samples captured before this instant must not reappear after re-enabling. */
  enabledAt_ms: number;
  setMode: (mode: PerformanceInstrumentationMode) => void;
  setEnabled: (enabled: boolean) => void;
}

/**
 * Controls measurement-only work. Correctness synchronization, diagnostics,
 * and presentation completion tracking are intentionally outside this switch.
 */
export const usePerformanceInstrumentationStore = create<PerformanceInstrumentationStore>((set) => ({
  // Detailed profiling is the product default. Users can explicitly select
  // timeline-only or off when they need an uninstrumented throughput run.
  mode: "activity",
  enabled: true,
  shaderActivityEnabled: true,
  shaderGeneration: 1,
  enabledAt_ms: performance.now(),
  setMode: (mode) => set((state) => {
    if (mode === state.mode) return state;
    const enabled = mode !== "off";
    const shaderActivityEnabled = mode === "activity";
    return {
      mode,
      enabled,
      shaderActivityEnabled,
      shaderGeneration: shaderActivityEnabled === state.shaderActivityEnabled
        ? state.shaderGeneration
        : state.shaderGeneration + 1,
      enabledAt_ms: enabled ? performance.now() : Infinity,
    };
  }),
  // Backward-compatible control for callers that only understand on/off.
  // The full profiler is the default enabled mode; timeline-only remains
  // available through setMode without compiling heartbeat declarations.
  setEnabled: (enabled) => set((state) => {
    const mode: PerformanceInstrumentationMode = enabled ? "activity" : "off";
    if (mode === state.mode) return state;
    const shaderActivityEnabled = mode === "activity";
    return {
      mode,
      enabled,
      shaderActivityEnabled,
      shaderGeneration: shaderActivityEnabled === state.shaderActivityEnabled
        ? state.shaderGeneration
        : state.shaderGeneration + 1,
      enabledAt_ms: enabled ? performance.now() : Infinity,
    };
  }),
}));

/** Read once while building a pipeline; never branch on this inside WGSL. */
export function performanceShaderVariant() {
  const { shaderActivityEnabled: enabled, shaderGeneration: generation } =
    usePerformanceInstrumentationStore.getState();
  return { enabled, generation, cacheKey: enabled ? `activity-${generation}` : "production" } as const;
}
