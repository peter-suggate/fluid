import type { PerformanceSnapshot } from "./stores/diagnostics-store";

const physicsTimingFields = new Set<keyof PerformanceSnapshot>([
  "gpuPreparation_ms", "gpuLayerConstruction_ms", "gpuAdvection_ms", "gpuConditioning_ms",
  "gpuRemeshing_ms", "gpuPressure_ms", "gpuProjection_ms", "gpuSurfaceUpdate_ms",
  "gpuRigid_ms", "gpuDiagnostics_ms", "gpuOverhead_ms"
]);

const renderTimingFields = new Set<keyof PerformanceSnapshot>([
  "gpuRender_ms", "gpuSurfaceExtraction_ms", "gpuDryScene_ms", "gpuInterfaces_ms",
  "gpuOpticalComposite_ms", "gpuUpscale_ms"
]);

/** Average a frame window without treating missing GPU timestamps as zero-valued samples. */
export function averagePerformanceSnapshots(samples: PerformanceSnapshot[], fallback: PerformanceSnapshot) {
  if (!samples.length) return fallback;
  if (samples.length === 1) return samples[0];
  const latest = samples[samples.length - 1];
  const averaged: PerformanceSnapshot = { ...latest };
  const writable = averaged as unknown as Record<string, number>;

  for (const [rawKey, value] of Object.entries(latest)) {
    if (typeof value !== "number") continue;
    const key = rawKey as keyof PerformanceSnapshot;
    const eligible = physicsTimingFields.has(key)
      ? samples.filter((sample) => sample.gpuPhysicsTimingAvailable)
      : renderTimingFields.has(key)
        ? samples.filter((sample) => sample.gpuRenderTimingAvailable)
        : samples;
    if (!eligible.length) continue;
    writable[rawKey] = eligible.reduce((sum, sample) => sum + ((sample as unknown as Record<string, number>)[rawKey] ?? 0), 0) / eligible.length;
  }

  averaged.gpuPhysicsTimingAvailable = samples.some((sample) => sample.gpuPhysicsTimingAvailable);
  averaged.gpuRenderTimingAvailable = samples.some((sample) => sample.gpuRenderTimingAvailable);
  averaged.gpuRenderTimestampSupported = samples.some((sample) => sample.gpuRenderTimestampSupported);
  averaged.gpuActiveStages = [...new Set(samples.filter((sample) => sample.gpuPhysicsTimingAvailable).flatMap((sample) => sample.gpuActiveStages))];
  averaged.adaptiveRebuildBlockedFrames = latest.adaptiveRebuildBlockedFrames;
  averaged.adaptiveRebuildCompletedCount = latest.adaptiveRebuildCompletedCount;
  averaged.adaptiveRebuildPending = latest.adaptiveRebuildPending;
  return averaged;
}

/** Produce a trailing average at every history point for the selected frame window. */
export function rollingPerformanceSnapshots(samples: PerformanceSnapshot[], windowSize: number) {
  const size = Math.max(1, Math.round(windowSize));
  return samples.map((sample, index) => averagePerformanceSnapshots(samples.slice(Math.max(0, index - size + 1), index + 1), sample));
}
