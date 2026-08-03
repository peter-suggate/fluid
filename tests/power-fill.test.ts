import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import {
  POWER_DROPLET_CELL_SIZE_M,
  POWER_DROPLET_FINE_BRICK_CAPACITY,
  POWER_DROPLET_PRESSURE_ROW_CAPACITY,
  POWER_DROPLET_RESERVOIR_M,
  POWER_FILL_EDGE_CELLS,
  POWER_FILL_FINE_BRICK_CAPACITY,
  POWER_FILL_LIQUID_CELLS,
  POWER_FILL_METHOD_PROFILE,
  POWER_FILL_PRESSURE_ROW_CAPACITY,
  POWER_FILL_RESERVOIR_CELLS,
  createPowerDropletScene,
  createPowerFillScene,
  getScenePreset,
  powerFillReservoirCells,
} from "../lib/scenes";
import {
  planFluidFootprintFineNarrowBandBrickCapacity,
  planOctreeFluidFootprintBudget,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";
import { POWER_DAM_LANE_ENVIRONMENT } from "../tools/power-dam-lane-environment";

const N = POWER_FILL_EDGE_CELLS;
const dims = { nx: N, ny: N, nz: N };
const extent_m = N * POWER_DROPLET_CELL_SIZE_M;

test("the fill family varies its live water and nothing else", () => {
  for (const liquidCells of POWER_FILL_LIQUID_CELLS) {
    const scene = createPowerFillScene(liquidCells);
    assert.deepEqual(validateScene(scene), [], `power-fill-${N}-${liquidCells} must be a valid document`);
    assert.equal(scene.sceneId, `power-fill-${N}-${liquidCells}`);
    // The container is the invariant of THIS family, the way the fluid is the
    // invariant of the droplet family. If it moves, the sweep is measuring two
    // things again.
    assert.deepEqual([scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      [extent_m, extent_m, extent_m]);
    assert.equal(scene.container.top, "closed");
    assert.equal(scene.container.fluidWallMode, "free-slip");
    assert.equal(scene.voxelDomain?.finestCellSize_m, POWER_DROPLET_CELL_SIZE_M);
    assert.equal(scene.numerics.maxDt_s, 0.004);
    assert.equal(scene.numerics.fixedDt_s, 0.004);
    assert.equal(scene.fluid.surfaceTension_N_m, 0);
    assert.deepEqual(scene.rigidBodies, []);

    // The four preconditions of `analyticSparseBootstrap`. Any one of them
    // flips the bootstrap dense — a 16.8M-cell Float32Array, a full-domain
    // signed-distance transform and a ~134 MB phi texture — which is
    // domain-shaped setup cost landing inside the measurement.
    assert.equal(scene.fluid.initialCondition, "dam-break");
    assert.equal(scene.fluid.initialDamBreakOrigin_m, undefined);
    assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
    assert.equal(scene.fluid.initialBrickSeedsAdditive, undefined);
    assert.equal(scene.fluid.inflow, undefined);

    const cells = powerFillReservoirCells(liquidCells);
    const footprint = planOctreeFluidFootprintBudget(scene, dims);
    assert.equal(footprint.initialLiquidCells, liquidCells,
      `power-fill-${N}-${liquidCells} must actually contain ${liquidCells} liquid cells`);
    assert.deepEqual([...footprint.minimumCell], [0, 0, 0]);
    assert.deepEqual([...footprint.maximumCell], [cells.x, cells.y, cells.z]);
  }
});

test("the sweep is an exact 8x geometric ladder at a fixed aspect ratio", () => {
  // The whole classification rests on this. Over one step a flat pass reads 1x,
  // a linear pass reads 8x and an area-shaped pass reads 4x; a ragged ratio or
  // a reshaped reservoir turns that arithmetic back into a judgement call.
  assert.deepEqual([...POWER_FILL_LIQUID_CELLS], [100, 800, 6_400]);
  for (const [index, cells] of POWER_FILL_RESERVOIR_CELLS.entries()) {
    assert.equal(cells.x * cells.y * cells.z, POWER_FILL_LIQUID_CELLS[index]);
    // 5:4:5 at every rung, so the surface/volume law is fixed too.
    assert.equal(cells.x, cells.z);
    assert.equal(cells.x * 4, cells.y * 5);
    if (index === 0) continue;
    const previous = POWER_FILL_RESERVOIR_CELLS[index - 1]!;
    assert.deepEqual([cells.x, cells.y, cells.z],
      [previous.x * 2, previous.y * 2, previous.z * 2]);
    assert.equal(POWER_FILL_LIQUID_CELLS[index], POWER_FILL_LIQUID_CELLS[index - 1]! * 8);
  }
  // The low rung is the droplet family's own reservoir, reproduced rather than
  // approximated, so the fill sweep's floor is literally the droplet sweep's
  // ceiling and the two instruments meet at one shared scene.
  const smallest = createPowerFillScene(100);
  assert.deepEqual(smallest.fluid.initialDamBreakDimensions_m, { ...POWER_DROPLET_RESERVOIR_M });
  assert.deepEqual(smallest.fluid.initialDamBreakDimensions_m,
    createPowerDropletScene(256).fluid.initialDamBreakDimensions_m);
});

test("both reserves are constants of the family, above the largest member's ask", () => {
  const asks = POWER_FILL_LIQUID_CELLS.map((liquidCells) => {
    const scene = createPowerFillScene(liquidCells);
    const cells = powerFillReservoirCells(liquidCells);
    const footprint = planOctreeFluidFootprintBudget(scene, dims);
    const rows = planOctreePressureCapacity(dims, 32, 1, undefined, true,
      scene.container.fillFraction, Number.MAX_SAFE_INTEGER, footprint).rowCapacity;
    const band = planFluidFootprintFineNarrowBandBrickCapacity(
      [N, N, N], [cells.x, cells.y, cells.z], 5);
    return { rows, bricks: band.maximumResidentBricks, logical: band.logicalBrickCount };
  });
  // Recorded rather than merely bounded: these are the numbers the shared
  // reserve was authored against, and a planner change that moves them should
  // fail here rather than silently re-baseline the instrument.
  assert.deepEqual(asks.map(({ rows }) => rows), [256, 1_792, 12_800]);
  assert.deepEqual(asks.map(({ bricks }) => bricks), [1_073, 4_290, 17_160]);
  for (const { rows, bricks, logical } of asks) {
    assert.ok(POWER_FILL_PRESSURE_ROW_CAPACITY > rows);
    assert.ok(POWER_FILL_FINE_BRICK_CAPACITY > bricks);
    // At fine factor 4 the logical lattice is one brick per finest cell, which
    // is what makes a domain-shaped ladder domain-shaped. The reserve must stay
    // orders of magnitude below it or the family stops being sparse.
    assert.equal(logical, N ** 3);
    assert.ok(POWER_FILL_FINE_BRICK_CAPACITY < logical / 100);
  }
  // The capacity control: same 100 cells, same 256-cubed container as
  // `power-droplet-256`, and exactly 16x both reserves. Diffing the two
  // per-label captures measures capacity-shaped GPU cost at identical live
  // occupancy, so the factor must stay exact and stay on both knobs.
  assert.equal(POWER_FILL_PRESSURE_ROW_CAPACITY, 16 * POWER_DROPLET_PRESSURE_ROW_CAPACITY);
  assert.equal(POWER_FILL_FINE_BRICK_CAPACITY, 16 * POWER_DROPLET_FINE_BRICK_CAPACITY);

  assert.equal(POWER_FILL_METHOD_PROFILE.methodId, "octree");
  assert.deepEqual(POWER_FILL_METHOD_PROFILE.overrides, {
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 1,
    globalFineLevelSetFactor: "4",
    pressureRowCapacity: POWER_FILL_PRESSURE_ROW_CAPACITY,
    globalFineLevelSetMaximumBricks: POWER_FILL_FINE_BRICK_CAPACITY,
  });
});

test("every fill lane carries the same capacity and the same solver knobs", () => {
  for (const liquidCells of POWER_FILL_LIQUID_CELLS) {
    const preset = getScenePreset(`power-fill-${N}-${liquidCells}`);
    assert.equal(preset.create().sceneId, `power-fill-${N}-${liquidCells}`);
    assert.equal(preset.methodProfile, POWER_FILL_METHOD_PROFILE);

    for (const laneId of ["default", "runtime-240"] as const) {
      const lane = getSceneWebGPUSmokeLane(`power-fill-${N}-${liquidCells}`, laneId);
      // The confound this family exists to avoid: one capacity for every
      // member, expressed where both the smoke and benchmark paths read it.
      assert.equal(lane.methods[0]?.overrides.pressureRowCapacity, POWER_FILL_PRESSURE_ROW_CAPACITY);
      assert.equal(lane.methods[0]?.overrides.globalFineLevelSetMaximumBricks,
        POWER_FILL_FINE_BRICK_CAPACITY);
      assert.equal(lane.methods[0]?.overrides.maximumLeafSize, "32");
      assert.equal(lane.methods[0]?.overrides.interfaceRefinementBandCells, 1);
      assert.equal(lane.methods[0]?.overrides.globalFineLevelSetFactor, "4");
      assert.equal(lane.stop.maxDt_s, 0.004);

      const rule = (id: string) => lane.acceptance.find((entry) => entry.id === id);
      assert.deepEqual(rule("expected-grid")?.expected, [N, N, N]);
      // Ceilings and overflow gates, never flat pins on the live counters: an
      // overflowing fine band degrades its resident count to the 0xFFFFFFFF
      // sentinel and leaves the pressure solve executing zero iterations while
      // the run still reports PASS, and a silently no-op solver would read as a
      // beautifully flat pass.
      assert.equal(rule("power-fill-row-arena-fits")?.expected, false);
      assert.equal(rule("power-fill-rows-within-reserve")?.expected, POWER_FILL_PRESSURE_ROW_CAPACITY);
      assert.equal(rule("power-fill-fine-residency-within-reserve")?.expected,
        POWER_FILL_FINE_BRICK_CAPACITY);
      assert.equal(rule("power-droplet-pinned-fine-residency"), undefined,
        "the droplet family's flat live pins would gate away this sweep's signal");
    }
    assert.equal(getSceneWebGPUSmokeLane(`power-fill-${N}-${liquidCells}`, "default").stop.exactSteps, 1);
    assert.equal(getSceneWebGPUSmokeLane(`power-fill-${N}-${liquidCells}`, "runtime-240").stop.exactSteps, 240);
  }
});

test("the fill benchmark lanes differ in exactly one field", () => {
  const lanes = POWER_FILL_LIQUID_CELLS.map((liquidCells) =>
    POWER_DAM_LANE_ENVIRONMENT[`fill-${liquidCells}`]);
  for (const [index, lane] of lanes.entries()) {
    assert.equal(lane.FLUID_SCENE, `power-fill-${N}-${POWER_FILL_LIQUID_CELLS[index]}`);
    for (const key of Object.keys(lanes[0]!)) {
      if (key === "FLUID_SCENE") continue;
      assert.equal(lane[key], lanes[0]![key],
        `fill-${POWER_FILL_LIQUID_CELLS[index]} must match fill-100 on ${key}`);
    }
    assert.equal(lane.FLUID_EXPECT_GRID, "256,256,256");
    assert.equal(lane.FLUID_PRESSURE_ROW_CAPACITY, String(POWER_FILL_PRESSURE_ROW_CAPACITY));
    // 256 cubed binds two ~134 MB dense fine directories, over Dawn's 128 MiB
    // default; a lane that cannot bind reports nothing.
    assert.equal(lane.FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES, "2147483648");
    // The default per-label capture window is skip 40 / capture 25, so a lane
    // shorter than 65 advances silently captures a different window than it
    // claims — and the window must match across all three members or the
    // comparison is meaningless.
    assert.ok(Number(lane.FLUID_ORACLE_STEPS) >= 65);
    assert.equal(lane.FLUID_ORACLE_STEPS, lane.FLUID_EXPECT_EXACT_STEPS);
    assert.equal(Number(lane.FLUID_TARGET_S),
      Number(lane.FLUID_ORACLE_STEPS) * Number(lane.FLUID_MAX_DT));
  }

  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  for (const liquidCells of POWER_FILL_LIQUID_CELLS) {
    assert.equal(packageJson.scripts[`benchmark:power-fill-${N}-${liquidCells}`],
      `node --import tsx tools/benchmark-power-dam.ts --lane=fill-${liquidCells}`);
    for (const [script, laneId] of [
      [`test:webgpu:power-fill-${N}-${liquidCells}`, "default"],
      [`test:webgpu:power-fill-${N}-${liquidCells}-240`, "runtime-240"],
    ] as const) {
      const command = packageJson.scripts[script];
      assert.ok(command, `${script} must exist`);
      assert.match(command, new RegExp(`FLUID_SCENE=power-fill-${N}-${liquidCells} `));
      assert.match(command, new RegExp(`FLUID_LANE=${laneId} `));
      assert.match(command, /FLUID_EXPECT_GRID=256,256,256 /);
      assert.match(command, /FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES=2147483648/);
      assert.match(command, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000/);
      assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
    }
  }
});
