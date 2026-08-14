/**
 * The scene definition for each power-dam benchmark lane.
 *
 * This is deliberately the *only* copy. The throughput benchmark
 * (`benchmark-power-dam.ts`) and the xctrace frame profiler
 * (`profile-mini-dam-xctrace.ts`) both read it, so a profile can never
 * describe a different scene, step count, or refinement setting than the
 * number the benchmark gates on.
 *
 * Each lane names its own `FLUID_METHOD`. The coarse backend used to be a
 * parameter of one octree method, so a lane could inherit it from the scene's
 * authored profile; it is now the method id itself. A lane that left the id to
 * its caller would silently benchmark Losasso on a frozen Power reference
 * scene, and the wall would look like a speedup.
 */

export type PowerDamRuntimeLane = "mini" | "large" | "high-resolution-dam-break" | "hydrostatic-tiny" | "large-hydrostatic" | "deep-hydrostatic" | "hydrostatic" | "ui" | "moving-interface" | "ocean" | "ceiling-drop" | "symmetric-expansion" | "droplet-64" | "droplet-128" | "droplet-240" | "droplet-256" | "fill-100" | "fill-800" | "fill-6400";

/**
 * The droplet sweep, as lane records.
 *
 * Four lanes that differ in exactly two fields — the scene id and the expected
 * grid. Everything else is written identically on purpose: same 0.05 m
 * lattice, same leaf 32 / band 1 / fine factor 4, same 0.004 s step, same
 * 100-cell corner reservoir, same authored row reserve. A wall difference
 * between two of these lanes therefore has one possible cause, which is the
 * entire point of the instrument.
 *
 * `FLUID_PRESSURE_ROW_CAPACITY` is restated here because the benchmark path
 * resolves method values from the environment and never reads the scene's
 * authored profile ([[capacity-is-not-inert]]); the planner's own default for
 * a 100-cell footprint is 256 rows, which dies at the t=0 cold-topology gate.
 * The fine-brick reserve has no environment path at all, so a benchmark run
 * carries the footprint-shaped default of 1,073 resident bricks instead of the
 * authored 4,096. That is a *reduction* in reserve, not an increase, and it
 * does not change any launch shape — the recurring ladder is sized by the
 * logical lattice, not by residency — so the benchmark and smoke paths still
 * measure the same scene. Closing the hole properly needs an override in
 * `lib/methods/octree.ts` plus `tools/webgpu-smoke-executor.ts`.
 *
 * Step counts start at 20. The full 240-step lane is affordable only after the
 * slope falls; 240 cubed also pays two O(domain) CPU loops before the first
 * advance (the footprint budget triple loop and the tall-cell column walk),
 * which is seconds of cold start that no GPU change removes.
 */
const dropletLane = (edgeCells: number, steps = 20): Record<string, string> => ({
  FLUID_METHOD: "power-liquids", FLUID_SCENE: `power-droplet-${edgeCells}`,
  FLUID_TARGET_S: String(steps * 0.004),
  FLUID_MAX_DT: "0.004",
  FLUID_ORACLE_STEPS: String(steps),
  FLUID_EXPECT_EXACT_STEPS: String(steps),
  FLUID_EXPECT_GRID: `${edgeCells},${edgeCells},${edgeCells}`,
  FLUID_MAXIMUM_LEAF_SIZE: "32",
  FLUID_OCTREE_INTERFACE_BAND: "1",
  FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  FLUID_PRESSURE_ROW_CAPACITY: "4096",
  ...(edgeCells >= 256
    // At 256 cubed the fine level-set's two dense logical directories are
    // ~134 MB each, over Dawn's 128 MiB default storage-binding limit. The
    // ocean lane raises the same knob for the same reason. Removing the
    // allocation, rather than the limit, is the memory milestone of the
    // program; until then a lane that cannot bind cannot report anything.
    ? { FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES: "2147483648" }
    : {}),
});

/**
 * The live-occupancy sweep, as lane records: the droplet sweep's dual.
 *
 * Three lanes that differ in exactly ONE field — the scene id. Same 256-cubed
 * grid, same 0.05 m lattice, same leaf 32 / band 1 / fine factor 4, same
 * 0.004 s step, same authored row reserve, same storage-binding raise. The only
 * thing that moves between them is how much live water the scene contains, so a
 * per-label GPU cost read across the three is a direct measurement of how each
 * pass scales in live occupancy and in nothing else.
 *
 * `FLUID_PRESSURE_ROW_CAPACITY` is restated here for the same reason the
 * droplet lanes restate theirs: the harness's environment override is applied
 * AFTER the authored profile (`tools/webgpu-smoke-executor.ts`), so a lane that
 * omits it inherits whatever the authored lane says and cannot be A/B'd from
 * the command line. Stating it makes the capacity an explicit property of the
 * lane rather than an inherited one, which is what the capacity control needs.
 *
 * Steps start at 80, not the droplet family's 20: droplet-family divergence
 * lives past step 30, and the default per-label capture window (skip 40,
 * capture 25) needs at least 65 advances to be the window it claims to be.
 * Compare only runs at the same step count and the same capture window.
 */
const fillLane = (liquidCells: number, steps = 80): Record<string, string> => ({
  FLUID_METHOD: "power-liquids", FLUID_SCENE: `power-fill-256-${liquidCells}`,
  FLUID_TARGET_S: String(steps * 0.004),
  FLUID_MAX_DT: "0.004",
  FLUID_ORACLE_STEPS: String(steps),
  FLUID_EXPECT_EXACT_STEPS: String(steps),
  FLUID_EXPECT_GRID: "256,256,256",
  FLUID_MAXIMUM_LEAF_SIZE: "32",
  FLUID_OCTREE_INTERFACE_BAND: "1",
  FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  // The family's shared reserve, sized for the 6,400-cell member and carried
  // unchanged by the 100-cell one. A capacity that moved with the member would
  // make a capacity-shaped pass read as live-shaped, which is the one error
  // this instrument cannot afford.
  FLUID_PRESSURE_ROW_CAPACITY: "65536",
  // At 256 cubed the fine level-set's two dense logical directories are ~134 MB
  // each, over Dawn's 128 MiB default storage-binding limit. Same knob, same
  // reason, as `droplet-256` and the ocean lane.
  FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES: "2147483648",
});

export const POWER_DAM_LANE_ENVIRONMENT: Record<PowerDamRuntimeLane, Record<string, string>> = {
  mini: {
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "2",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  large: {
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "large-power-dam-break", FLUID_TARGET_S: "2",
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
  "high-resolution-dam-break": {
    FLUID_METHOD: "losasso", FLUID_SCENE: "high-resolution-dam-break", FLUID_LANE: "performance",
    FLUID_TARGET_S: "0.004", FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "1",
    FLUID_EXPECT_EXACT_STEPS: "1", FLUID_EXPECT_GRID: "128,128,128",
    FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_SURFACE_GRADING: "3",
    FLUID_OCTREE_GLOBAL_FINE_FACTOR: "1",
  },
  "hydrostatic-tiny": {
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "hydrostatic-power-two-level", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "large-hydrostatic": {
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "large-power-hydrostatic", FLUID_TARGET_S: "0.96",
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
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "deep-power-hydrostatic", FLUID_TARGET_S: "0.96",
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
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "hydrostatic-power-large-offset", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "32,24,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "moving-interface": {
    FLUID_METHOD: "power-liquids", FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  ui: {
    FLUID_METHOD: "losasso", FLUID_SCENE: "dam-break-ui", FLUID_TARGET_S: "0.496",
    FLUID_MAX_DT: "0.008", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "24,18,16",
  },
  ocean: {
    // Two submissions are intentional: the second semantic frame-start is the
    // exact GPU end boundary for the literal first advance selected by xctrace.
    FLUID_METHOD: "losasso", FLUID_SCENE: "ocean-seiche", FLUID_TARGET_S: "0.01",
    FLUID_MAX_DT: "0.005", FLUID_ORACLE_STEPS: "2", FLUID_EXPECT_EXACT_STEPS: "2",
    FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES: "2147483648",
    // Ocean is a product-default Losasso lane. The benchmark-level audit
    // suppression stays authoritative; Power generation/stage controls belong
    // only to the explicitly frozen reference lanes above.
    FLUID_STABILITY_ENVELOPE: "1",
  },
  "symmetric-expansion": {
    // The D4 symmetry oracle scene, at the fine factor its `fine-factor-4`
    // correctness lane uses. `performance` is the same solver configuration
    // without the evidence collectors, so a change scored here can be
    // re-gated on symmetry without changing the scene.
    FLUID_METHOD: "losasso", FLUID_SCENE: "symmetric-expansion", FLUID_LANE: "performance",
    FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "32,16,32", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "4", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "droplet-64": dropletLane(64),
  "droplet-128": dropletLane(128),
  "droplet-240": dropletLane(240),
  "droplet-256": dropletLane(256),
  "fill-100": fillLane(100),
  "fill-800": fillLane(800),
  "fill-6400": fillLane(6400),
  "ceiling-drop": {
    FLUID_METHOD: "losasso", FLUID_SCENE: "ceiling-slab-drop", FLUID_TARGET_S: "0.024",
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
