/**
 * The scene definition for each power-dam benchmark lane.
 *
 * This is deliberately the *only* copy. The throughput benchmark
 * (`benchmark-power-dam.ts`) and the xctrace frame profiler
 * (`profile-mini-dam-xctrace.ts`) both read it, so a profile can never
 * describe a different scene, step count, or refinement setting than the
 * number the benchmark gates on.
 */

export type PowerDamRuntimeLane = "mini" | "large" | "hydrostatic" | "ui" | "moving-interface" | "ocean" | "ceiling-drop" | "symmetric-expansion";

export const POWER_DAM_LANE_ENVIRONMENT: Record<PowerDamRuntimeLane, Record<string, string>> = {
  mini: {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "2",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  large: {
    FLUID_SCENE: "large-power-dam-break", FLUID_TARGET_S: "2",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
    FLUID_EXPECT_GRID: "64,20,64", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  hydrostatic: {
    // Match the large churn lane's timestep, fine factor and authored band.
    // The scene itself is the Section 4/5 still-water correctness oracle. Its
    // intentional quarter-cell cut prevents a grid-aligned surface from hiding
    // interface work or hydrostatic imbalance.
    FLUID_SCENE: "hydrostatic-power-large-offset", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "32,24,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "moving-interface": {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  ui: {
    FLUID_SCENE: "dam-break-ui", FLUID_TARGET_S: "0.496",
    FLUID_MAX_DT: "0.008", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "24,18,16",
  },
  ocean: {
    // Two submissions are intentional: the second semantic frame-start is the
    // exact GPU end boundary for the literal first advance selected by xctrace.
    FLUID_SCENE: "ocean-seiche", FLUID_TARGET_S: "0.01",
    FLUID_MAX_DT: "0.005", FLUID_ORACLE_STEPS: "2", FLUID_EXPECT_EXACT_STEPS: "2",
    FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES: "2147483648",
    FLUID_POWER_GENERATION_AUDIT: "1", FLUID_POWER_GENERATION_AUDIT_LOG: "1",
    // The unpublished-next-candidate forensic is intentionally not a capture
    // gate: frame 1/2 can be valid while generation 3 is rejected, and that
    // post-frame diagnosis must not discard an otherwise complete xctrace.
    FLUID_POWER_STAGE_AUDIT: "1",
    FLUID_POWER_AUDIT_EVERY_STEPS: "1", FLUID_STABILITY_ENVELOPE: "1",
  },
  "symmetric-expansion": {
    // The D4 symmetry oracle scene, at the fine factor its `fine-factor-4`
    // correctness lane uses. `performance` is the same solver configuration
    // without the evidence collectors, so a change scored here can be
    // re-gated on symmetry without changing the scene.
    FLUID_SCENE: "symmetric-expansion", FLUID_LANE: "performance",
    FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "32,16,32", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "ceiling-drop": {
    FLUID_SCENE: "ceiling-slab-drop", FLUID_TARGET_S: "0.024",
    FLUID_LANE: "performance",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "6", FLUID_EXPECT_EXACT_STEPS: "6",
    FLUID_EXPECT_GRID: "24,16,24", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
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

/** Rewrite a lane timestep at fixed simulated time. This is the authored
 * X-7 dt-elasticity contract: halving dt doubles advances without changing the
 * physical interval being measured. */
export const powerDamLaneWithDt = (
  lane: PowerDamRuntimeLane,
  dt: number,
  target_s = Number(POWER_DAM_LANE_ENVIRONMENT[lane].FLUID_TARGET_S),
): Record<string, string> => {
  if (!Number.isFinite(dt) || dt <= 0 || !Number.isFinite(target_s) || target_s <= 0) {
    throw new Error(`dt and target time must be positive and finite; received dt=${dt}, target=${target_s}`);
  }
  const steps = Math.round(target_s / dt);
  if (steps < 1 || Math.abs(steps * dt - target_s) > 1e-9 * Math.max(1, target_s)) {
    throw new Error(`target time ${target_s} must be an integer multiple of dt ${dt}`);
  }
  return {
    ...POWER_DAM_LANE_ENVIRONMENT[lane],
    FLUID_TARGET_S: String(target_s),
    FLUID_MAX_DT: String(dt),
    FLUID_ORACLE_STEPS: String(steps),
    FLUID_EXPECT_EXACT_STEPS: String(steps),
  };
};
