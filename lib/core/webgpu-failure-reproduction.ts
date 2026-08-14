import type { GPUFailureReproduction } from "./gpu-status";
import type { MethodParamValues, MethodProfile } from "./method-contract";
import {
  ADAPTIVE_LOSASSO_UI_METHOD_PROFILE,
  COARSE_ONLY_POWER_DAM_METHOD_PROFILE,
} from "./scenes";

/** Live browser inputs that make a Dawn command an exact reproduction. */
export interface GPUFailureReproductionConfiguration {
  readonly sceneId: string;
  readonly methodId: string;
  readonly quality: string;
  readonly methodOverrides: Readonly<MethodParamValues>;
  readonly grid: readonly [number, number, number];
  readonly fixedDt_s: number;
  readonly maxDt_s: number;
}

/** One serialized Dawn boundary for browser construction and fenced t=0 failures. */
export const DAM_UI_T0_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "dam-ui-t0-two-step",
  command: "npm run test:webgpu:dam-ui-two-step",
  scene: "dam-break-ui",
  grid: [24, 18, 16] as const,
  steps: 2,
  targetTime_s: 0.016,
  maxDt_s: 0.008,
  validated: true,
  isolated: true,
});

/** The ordinary authored UI runtime run with Dawn validation left enabled. */
export const DAM_UI_RUNTIME_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "dam-ui-runtime-190-step",
  command: "npm run test:webgpu:dam-ui-runtime",
  scene: "dam-break-ui",
  grid: [24, 18, 16] as const,
  steps: 190,
  targetTime_s: 1.52,
  maxDt_s: 0.008,
  validated: true,
  isolated: true,
});

/** Fenced construction boundary for the authored coarse-only mini32 scene. */
export const MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "minimal-power-dam-32-t0-two-step",
  command: "npm run test:webgpu:minimal-power-dam-32-two-step",
  scene: "minimal-power-dam-break-32",
  grid: [32, 32, 32] as const,
  steps: 2,
  targetTime_s: 0.008,
  maxDt_s: 0.004,
  validated: true,
  isolated: true,
});

/** Current mini32 runtime/front reproduction: 69 authored 4 ms advances. */
export const MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "minimal-power-dam-32-runtime-69-step",
  command: "npm run test:webgpu:minimal-power-dam-32-runtime",
  scene: "minimal-power-dam-break-32",
  grid: [32, 32, 32] as const,
  steps: 69,
  targetTime_s: 0.276,
  maxDt_s: 0.004,
  validated: true,
  isolated: true,
});

function exactProfile(
  configuration: GPUFailureReproductionConfiguration,
  sceneId: string,
  grid: readonly [number, number, number],
  dt_s: number,
  profile: MethodProfile,
): boolean {
  if (configuration.sceneId !== sceneId
    || configuration.methodId !== profile.methodId
    || configuration.quality !== profile.quality
    || configuration.fixedDt_s !== dt_s
    || configuration.maxDt_s !== dt_s
    || configuration.grid.some((value, axis) => value !== grid[axis])) return false;
  const actualKeys = Object.keys(configuration.methodOverrides).sort();
  const expectedKeys = Object.keys(profile.overrides).sort();
  return actualKeys.length === expectedKeys.length
    && actualKeys.every((key, index) => key === expectedKeys[index]
      && configuration.methodOverrides[key] === profile.overrides[key]);
}

/**
 * Map every unexpected browser GPU stop to the smallest exact Dawn boundary
 * that exercises it. Intentional lifecycle stops remain unlabelled.
 */
export function dawnReproductionForGPUFailure(
  message: string,
  configuration?: GPUFailureReproductionConfiguration,
): GPUFailureReproduction | undefined {
  const mini32 = configuration !== undefined && exactProfile(configuration,
    "minimal-power-dam-break-32", [32, 32, 32], 0.004,
    COARSE_ONLY_POWER_DAM_METHOD_PROFILE);
  const damUI = configuration === undefined || exactProfile(configuration,
    "water-box-dam-break", [24, 18, 16], 0.008,
    ADAPTIVE_LOSASSO_UI_METHOD_PROFILE);
  if (/GPU initialization failed|Initial sparse authority|Paused t=0 authority rejected|before fenced sparse t=0 authority/.test(message)) {
    if (mini32) return MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION;
    if (damUI) return DAM_UI_T0_DAWN_REPRODUCTION;
  }
  if (/GPU runtime stopped|GPU device lost|device lost mid-simulation/.test(message)) {
    if (mini32) return MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION;
    if (damUI) return DAM_UI_RUNTIME_DAWN_REPRODUCTION;
  }
  return undefined;
}

type SmokeEnvironment = Readonly<Record<string, string | undefined>>;

/** Identify the exact authored UI reproducer from its existing smoke contract. */
export function dawnReproductionForSmokeEnvironment(
  environment: SmokeEnvironment,
): GPUFailureReproduction | undefined {
  const shared = environment.FLUID_SCENE === "dam-break-ui"
    && environment.FLUID_METHOD === "losasso"
    && environment.FLUID_MAX_DT === "0.008"
    && environment.FLUID_EXPECT_GRID === "24,18,16"
    && environment.FLUID_WEBGPU_DAWN_FEATURES === undefined;
  if (shared
    && environment.FLUID_TARGET_S === "0.016"
    && environment.FLUID_ORACLE_STEPS === "2"
    && environment.FLUID_EXPECT_EXACT_STEPS === "2"
  ) return DAM_UI_T0_DAWN_REPRODUCTION;
  if (shared
    && environment.FLUID_TARGET_S === "1.52"
    && environment.FLUID_ORACLE_STEPS === "190"
    && environment.FLUID_EXPECT_EXACT_STEPS === "190"
  ) return DAM_UI_RUNTIME_DAWN_REPRODUCTION;
  const mini32 = environment.FLUID_SCENE === "minimal-power-dam-break-32"
    && environment.FLUID_METHOD === "losasso"
    && environment.FLUID_QUALITY === "balanced"
    && environment.FLUID_MAX_DT === "0.004"
    && environment.FLUID_EXPECT_GRID === "32,32,32"
    && environment.FLUID_MAXIMUM_LEAF_SIZE === "32"
    && environment.FLUID_OCTREE_INTERFACE_BAND === "3"
    && environment.FLUID_OCTREE_SURFACE_GRADING === "3"
    && environment.FLUID_OCTREE_GLOBAL_FINE_FACTOR === "1"
    && environment.FLUID_WEBGPU_DAWN_FEATURES === undefined;
  if (mini32
    && environment.FLUID_TARGET_S === "0.008"
    && environment.FLUID_ORACLE_STEPS === "2"
    && environment.FLUID_EXPECT_EXACT_STEPS === "2"
  ) return MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION;
  if (mini32
    && environment.FLUID_TARGET_S === "0.276"
    && environment.FLUID_ORACLE_STEPS === "69"
    && environment.FLUID_EXPECT_EXACT_STEPS === "69"
  ) return MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION;
  return undefined;
}
