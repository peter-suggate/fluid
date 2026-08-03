/**
 * The scene definition for each power-dam benchmark lane.
 *
 * This is deliberately the *only* copy. The throughput benchmark
 * (`benchmark-power-dam.ts`) and the xctrace frame profiler
 * (`profile-mini-dam-xctrace.ts`) both read it, so a profile can never
 * describe a different scene, step count, or refinement setting than the
 * number the benchmark gates on.
 */

export type PowerDamRuntimeLane = "mini" | "large" | "hydrostatic-tiny" | "large-hydrostatic" | "deep-hydrostatic" | "hydrostatic" | "ui" | "moving-interface" | "ocean" | "ceiling-drop" | "symmetric-expansion";

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
    // A lane table that omits a scene's authored capacities does not run that
    // scene. `largePowerDamOverrides` authors 8,192 rows against a 1,185-row
    // cold publication; without this the lane died at t=0 with "Initial sparse
    // authority cold-topology published no liquid-row frontier", while the
    // identical scene through its smoke lane -- which does read the authored
    // profile -- was green for 150 steps.
    FLUID_PRESSURE_ROW_CAPACITY: "8192",
    // KNOWN HOLE, not an omission: this lane also needs
    // `globalFineLevelSetMaximumBricks: 32_768`, and that parameter has no
    // environment override at all (`lib/methods/octree.ts` reads it only from
    // authored method values). The footprint-shaped default for this scene is
    // 6,600 bricks -- 5x BELOW the authored reserve, and below the 28,672 that
    // the settled floor-spanning sheet needs -- so a benchmark run of this lane
    // is still not the scene its smoke lane runs, and its late-lane fine-band
    // behaviour cannot be compared to the 150-step smoke. Closing this needs an
    // override in `lib/methods/octree.ts` + `tools/webgpu-smoke-executor.ts`.
  },
  "hydrostatic-tiny": {
    FLUID_SCENE: "hydrostatic-power-two-level", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "large-hydrostatic": {
    FLUID_SCENE: "large-power-hydrostatic", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "64,20,64", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
    // Same defect class as the `large` lane above: the scene authors 4,096 rows
    // (`LARGE_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY`) against a planner
    // default of 2,048, which is below its air-support reach. This lane must
    // say so itself. It authors no fine-brick reserve, so it has no second hole.
    FLUID_PRESSURE_ROW_CAPACITY: "4096",
  },
  "deep-hydrostatic": {
    // The Bet-4.2 lane: the 20x still tank filled 40 cells deep, so that 67.8%
    // of its liquid is interior and interior coarsening has something to
    // coarsen. Every discretization knob is the `large-hydrostatic` lane's, so
    // the pair isolates depth: same 0.05 m lattice, same 3.2 x 3.2 m footprint,
    // same band 1 / fine factor 4 / leaf 32, same 240 still steps at 0.004 s.
    //
    // `FLUID_POWER_HYBRID_CENSUS` is the measurement, not a diagnostic load:
    // the census is one terminal readback after the measured window, so it
    // cannot contaminate the wall, and without it a run reports no
    // regular/power row split at all. No minimum reduction is pinned here —
    // the first capture establishes it (the CPU leaf model predicts >= 1.8x).
    //
    // Cost expectation, stated so the first capture can falsify it. The
    // 2026-08-03 four-cell matrix (still/small 361.6, still/large 143.6,
    // churn/small 245.3, churn/large 306.9 ms/adv at 4, 1, 4, 2 MGPCG
    // iterations) is ANTI-correlated with domain and tracks iteration count,
    // because all four lanes live within ~1-8k rows and the single-workgroup
    // solve dominates. This lane is the first with ~58k live rows, an order of
    // magnitude outside that range, so it is precisely the experiment that
    // separates "wall tracks iterations" from "wall tracks live rows". Predict
    // 2-4 iterations; if the wall still lands near 150-350 ms the iteration
    // model holds, and if it scales with rows it does not.
    //
    // 240 steps must also fit the isolated runner's 240 s ceiling (~1 s/adv).
    // Take the first capture with `--steps=20` before committing to the full lane.
    FLUID_SCENE: "deep-power-hydrostatic", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "64,48,64", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
    FLUID_POWER_HYBRID_CENSUS: "1",
    // The one capacity this scene genuinely needs, stated where the benchmark
    // can read it. The planner's own scene-shaped request is 59,392, which is
    // below the settling headroom a still lane needs over its 57,776-row
    // predicted publication; 65,536 is the authored reserve.
    FLUID_PRESSURE_ROW_CAPACITY: "65536",
    // This lane deliberately does NOT depend on the `globalFineLevelSetMaximum-
    // Bricks` hole described on the `large` lane. Its authored 67,584 is a
    // *reduction* below the 152,064-brick footprint-shaped default (the tank's
    // fluid footprint is the tank, so the default reserves all three faces of a
    // box with one free surface). A benchmark run that cannot express it is
    // still correct — it just carries ~108 MiB more fine-brick arena than the
    // smoke lane. Nothing load-bearing differs between the two paths.
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
