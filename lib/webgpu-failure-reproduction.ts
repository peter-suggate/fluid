export interface GPUFailureReproduction {
  readonly caseId: "dam-ui-t0-two-step" | "dam-ui-runtime-generation-281";
  readonly command:
    | "npm run test:webgpu:dam-ui-two-step"
    | "npm run test:webgpu:dam-ui-runtime";
  readonly scene: "dam-break-ui";
  readonly grid: readonly [24, 18, 16];
  readonly steps: 2 | 279;
  readonly targetTime_s: 0.016 | 2.232;
  readonly validated: true;
  readonly isolated: true;
}

/** One serialized Dawn boundary for browser construction and fenced t=0 failures. */
export const DAM_UI_T0_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "dam-ui-t0-two-step",
  command: "npm run test:webgpu:dam-ui-two-step",
  scene: "dam-break-ui",
  grid: [24, 18, 16] as const,
  steps: 2,
  targetTime_s: 0.016,
  validated: true,
  isolated: true,
});

/** The first mini dam-break boundary that previously rejected power publication. */
export const DAM_UI_RUNTIME_DAWN_REPRODUCTION: GPUFailureReproduction = Object.freeze({
  caseId: "dam-ui-runtime-generation-281",
  command: "npm run test:webgpu:dam-ui-runtime",
  scene: "dam-break-ui",
  grid: [24, 18, 16] as const,
  steps: 279,
  targetTime_s: 2.232,
  validated: true,
  isolated: true,
});

/**
 * Map every unexpected browser GPU stop to the smallest exact Dawn boundary
 * that exercises it. Intentional lifecycle stops remain unlabelled.
 */
export function dawnReproductionForGPUFailure(message: string): GPUFailureReproduction | undefined {
  if (/GPU initialization failed|Initial sparse authority|Paused t=0 authority rejected|before fenced sparse t=0 authority/.test(message)) {
    return DAM_UI_T0_DAWN_REPRODUCTION;
  }
  if (/GPU runtime stopped|GPU device lost|device lost mid-simulation/.test(message)) {
    return DAM_UI_RUNTIME_DAWN_REPRODUCTION;
  }
  return undefined;
}

type SmokeEnvironment = Readonly<Record<string, string | undefined>>;

/** Identify the exact authored UI reproducer from its existing smoke contract. */
export function dawnReproductionForSmokeEnvironment(
  environment: SmokeEnvironment,
): GPUFailureReproduction | undefined {
  const shared = environment.FLUID_SCENE === "dam-break-ui"
    && environment.FLUID_METHOD === "octree"
    && environment.FLUID_MAX_DT === "0.008"
    && environment.FLUID_EXPECT_GRID === "24,18,16"
    && environment.FLUID_WEBGPU_DAWN_FEATURES === undefined;
  if (shared
    && environment.FLUID_TARGET_S === "0.016"
    && environment.FLUID_ORACLE_STEPS === "2"
    && environment.FLUID_EXPECT_EXACT_STEPS === "2"
  ) return DAM_UI_T0_DAWN_REPRODUCTION;
  if (shared
    && environment.FLUID_TARGET_S === "2.232"
    && environment.FLUID_ORACLE_STEPS === "279"
    && environment.FLUID_EXPECT_EXACT_STEPS === "279"
  ) return DAM_UI_RUNTIME_DAWN_REPRODUCTION;
  return undefined;
}
