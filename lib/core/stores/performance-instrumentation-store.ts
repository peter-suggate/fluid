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
  // Uninstrumented is the product default (POWER_LIQUIDS_ULTIMATE_M1MAX P0.1).
  // "activity" compiles per-workgroup atomics into every MGPCG entry point and
  // rewrites each entry point's returns, so leaving it on by default charged
  // the browser product for measurement it had not asked for. Users opt in
  // through the performance panel; the Dawn harness opts in through
  // FLUID_PERFORMANCE_TRACES.
  mode: "off",
  enabled: false,
  shaderActivityEnabled: false,
  shaderGeneration: 1,
  enabledAt_ms: Infinity,
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
  // Enabling measurement must preserve production WGSL: detailed shader
  // heartbeats are an explicit opt-in through setMode("activity").
  setEnabled: (enabled) => set((state) => {
    const mode: PerformanceInstrumentationMode = enabled ? "timeline" : "off";
    if (mode === state.mode) return state;
    const shaderActivityEnabled = false;
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
