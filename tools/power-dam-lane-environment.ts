/**
 * The scene definition for each power-dam benchmark lane.
 *
 * This is deliberately the *only* copy. The throughput benchmark
 * (`benchmark-power-dam.ts`) and the xctrace frame profiler
 * (`profile-mini-dam-xctrace.ts`) both read it, so a profile can never
 * describe a different scene, step count, or refinement setting than the
 * number the benchmark gates on.
 */

export type PowerDamRuntimeLane = "mini" | "ui" | "moving-interface";

export const POWER_DAM_LANE_ENVIRONMENT: Record<PowerDamRuntimeLane, Record<string, string>> = {
  mini: {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "2",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "2",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "moving-interface": {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "2",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  ui: {
    FLUID_SCENE: "dam-break-ui", FLUID_TARGET_S: "0.496",
    FLUID_MAX_DT: "0.008", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "24,18,16",
  },
};

/**
 * Rewrite a lane's step count, keeping `FLUID_TARGET_S` consistent with
 * `FLUID_MAX_DT`. The smoke harness asserts the exact step count, so the three
 * fields must move together.
 */
export const powerDamLaneWithSteps = (
  lane: PowerDamRuntimeLane,
  steps: number,
): Record<string, string> => {
  const base = POWER_DAM_LANE_ENVIRONMENT[lane];
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error(`step count must be a positive integer; received ${steps}`);
  }
  const dt = Number(base.FLUID_MAX_DT);
  return {
    ...base,
    FLUID_TARGET_S: String(steps * dt),
    FLUID_ORACLE_STEPS: String(steps),
    FLUID_EXPECT_EXACT_STEPS: String(steps),
  };
};
