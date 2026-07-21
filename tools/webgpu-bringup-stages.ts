import type { GPUInitializationProgress } from "../lib/methods/types";

/**
 * Ordered, independently runnable checkpoints for native Dawn bring-up.
 *
 * Keep these names stable: operators use them to advance one bounded process
 * at a time after a driver or WindowServer failure.
 */
export const webGPUBringupStages = [
  "adapter-device",
  "compute-sentinel",
  "solver-resources",
  "sparse-t0",
  "one-step",
] as const;

export type WebGPUBringupStage = typeof webGPUBringupStages[number];

export const DEFAULT_WEBGPU_BRINGUP_TIMEOUT_MS = 120_000;
export const MINIMUM_WEBGPU_BRINGUP_TIMEOUT_MS = 1_000;
export const MAXIMUM_WEBGPU_BRINGUP_TIMEOUT_MS = 10 * 60_000;

export function parseWebGPUBringupStage(value: string | undefined): WebGPUBringupStage {
  const selected = value ?? "adapter-device";
  if (!webGPUBringupStages.includes(selected as WebGPUBringupStage)) {
    throw new Error(`Unknown FLUID_BRINGUP_STAGE=${selected}; expected ${webGPUBringupStages.join(", ")}`);
  }
  return selected as WebGPUBringupStage;
}

export function parseWebGPUBringupTimeout(value: string | undefined): number {
  if (value === undefined) return DEFAULT_WEBGPU_BRINGUP_TIMEOUT_MS;
  const timeout = Number(value);
  if (!Number.isInteger(timeout) || timeout < MINIMUM_WEBGPU_BRINGUP_TIMEOUT_MS || timeout > MAXIMUM_WEBGPU_BRINGUP_TIMEOUT_MS) {
    throw new Error(`FLUID_BRINGUP_TIMEOUT_MS must be an integer from ${MINIMUM_WEBGPU_BRINGUP_TIMEOUT_MS} to ${MAXIMUM_WEBGPU_BRINGUP_TIMEOUT_MS}`);
  }
  return timeout;
}

/** The warmup task is the first task that submits the complete sparse t=0 publication. */
export function reachedSolverResourceBoundary(progress: GPUInitializationProgress): boolean {
  return progress.taskId === "solver.warmup" && progress.phase === "warmup";
}

export function stageIncludesComputeSentinel(stage: WebGPUBringupStage): boolean {
  return webGPUBringupStages.indexOf(stage) >= webGPUBringupStages.indexOf("compute-sentinel");
}

export function stageIncludesSparseT0(stage: WebGPUBringupStage): boolean {
  return webGPUBringupStages.indexOf(stage) >= webGPUBringupStages.indexOf("sparse-t0");
}

