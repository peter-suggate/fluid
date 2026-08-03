import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import {
  POWER_DROPLET_CELL_SIZE_M,
  POWER_DROPLET_EDGE_CELLS,
  POWER_DROPLET_FINE_BRICK_CAPACITY,
  POWER_DROPLET_LIQUID_CELLS,
  POWER_DROPLET_METHOD_PROFILE,
  POWER_DROPLET_PRESSURE_ROW_CAPACITY,
  POWER_DROPLET_RESERVOIR_M,
  createPowerDropletScene,
  getScenePreset,
} from "../lib/scenes";
import {
  planFluidFootprintFineNarrowBandBrickCapacity,
  planOctreeFluidFootprintBudget,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";
import { POWER_DAM_LANE_ENVIRONMENT } from "../tools/power-dam-lane-environment";

const dims = (edgeCells: number) => ({ nx: edgeCells, ny: edgeCells, nz: edgeCells });

test("the droplet family varies its container and nothing else", () => {
  const documents = POWER_DROPLET_EDGE_CELLS.map((edgeCells) => createPowerDropletScene(edgeCells));
  for (const [index, scene] of documents.entries()) {
    const edgeCells = POWER_DROPLET_EDGE_CELLS[index];
    const extent_m = edgeCells * POWER_DROPLET_CELL_SIZE_M;
    assert.deepEqual(validateScene(scene), [], `power-droplet-${edgeCells} must be a valid document`);
    assert.equal(scene.sceneId, `power-droplet-${edgeCells}`);
    // A cube whose extents are the authored spacing times one integer: the
    // power catalog rejects anisotropic finest cells outright.
    assert.deepEqual([scene.container.width_m, scene.container.height_m, scene.container.depth_m],
      [extent_m, extent_m, extent_m]);
    assert.deepEqual([
      Math.round(scene.container.width_m / POWER_DROPLET_CELL_SIZE_M),
      Math.round(scene.container.height_m / POWER_DROPLET_CELL_SIZE_M),
      Math.round(scene.container.depth_m / POWER_DROPLET_CELL_SIZE_M),
    ], [edgeCells, edgeCells, edgeCells]);
    assert.equal(scene.container.top, "closed");
    assert.equal(scene.container.fluidWallMode, "free-slip");
    assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
    assert.deepEqual(scene.rigidBodies, []);
    assert.equal(scene.fluid.surfaceTension_N_m, 0);
    assert.equal(scene.fluid.inflow, undefined);
    assert.equal(scene.numerics.fixedDt_s, 0.004);
    assert.equal(scene.numerics.maxDt_s, 0.004);

    // The analytic-sparse-bootstrap contract. A brick seed, an offset origin,
    // terrain or a body each flip the whole t=0 publication onto the dense
    // path: a full-domain signed-distance transform and an nx*ny*nz phi
    // texture, which at 240 cubed is the very cost this family measures.
    assert.equal(scene.fluid.initialCondition, "dam-break");
    assert.deepEqual(scene.fluid.initialDamBreakDimensions_m, { ...POWER_DROPLET_RESERVOIR_M });
    assert.equal(scene.fluid.initialDamBreakOrigin_m, undefined);
    assert.equal(scene.fluid.initialBrickSeeds_m, undefined);

    // The invariant the whole program rests on: the same one hundred cells at
    // every N. If this moves, the sweep is measuring two things again.
    const footprint = planOctreeFluidFootprintBudget(scene, dims(edgeCells));
    assert.equal(footprint.initialLiquidCells, POWER_DROPLET_LIQUID_CELLS);
    assert.deepEqual([...footprint.minimumCell], [0, 0, 0]);
    assert.deepEqual([...footprint.maximumCell], [5, 4, 5]);
  }
});

test("both authored reserves are constants of the family, above the planner's own asks", () => {
  for (const edgeCells of POWER_DROPLET_EDGE_CELLS) {
    const scene = createPowerDropletScene(edgeCells);
    const footprint = planOctreeFluidFootprintBudget(scene, dims(edgeCells));
    // 100 liquid cells with the 2x footprint headroom, 256-aligned, is 256
    // rows — too tight for the closed-wall strip and 2:1 balance around a
    // corner reservoir, and an under-authored value dies at the t=0
    // cold-topology gate rather than merely reserving less.
    const planned = planOctreePressureCapacity(dims(edgeCells), 32, 1, undefined,
      true, scene.container.fillFraction, Number.MAX_SAFE_INTEGER, footprint);
    assert.equal(planned.rowCapacity, 256);
    assert.ok(POWER_DROPLET_PRESSURE_ROW_CAPACITY >= 16 * planned.rowCapacity);

    // The footprint-shaped fine-band ask is the three exposed faces of a
    // 5x4x5 reservoir through the eleven-layer capacity band; 4,096 is 3.8x
    // of it and four orders of magnitude below the logical lattice.
    const band = planFluidFootprintFineNarrowBandBrickCapacity(
      [edgeCells, edgeCells, edgeCells], [5, 4, 5], 5);
    assert.equal(band.maximumResidentBricks, 1_073);
    assert.ok(POWER_DROPLET_FINE_BRICK_CAPACITY > band.maximumResidentBricks);
    // At fine factor 4 the logical lattice is one brick per finest cell, which
    // is what makes the recurring ladder domain-shaped in the first place.
    assert.equal(band.logicalBrickCount, edgeCells ** 3);
    assert.ok(POWER_DROPLET_FINE_BRICK_CAPACITY < band.logicalBrickCount);
  }

  assert.equal(POWER_DROPLET_METHOD_PROFILE.methodId, "octree");
  assert.deepEqual(POWER_DROPLET_METHOD_PROFILE.overrides, {
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 1,
    globalFineLevelSetFactor: "4",
    pressureRowCapacity: POWER_DROPLET_PRESSURE_ROW_CAPACITY,
    globalFineLevelSetMaximumBricks: POWER_DROPLET_FINE_BRICK_CAPACITY,
  });
});

test("every droplet lane pins the same t=0 counters and the same solver knobs", () => {
  for (const edgeCells of POWER_DROPLET_EDGE_CELLS) {
    const preset = getScenePreset(`power-droplet-${edgeCells}`);
    assert.equal(preset.create().sceneId, `power-droplet-${edgeCells}`);
    assert.equal(preset.methodProfile, POWER_DROPLET_METHOD_PROFILE);

    for (const laneId of ["default", "runtime-240"] as const) {
      const lane = getSceneWebGPUSmokeLane(`power-droplet-${edgeCells}`, laneId);
      assert.equal(lane.methods[0]?.overrides.pressureRowCapacity,
        POWER_DROPLET_PRESSURE_ROW_CAPACITY);
      assert.equal(lane.methods[0]?.overrides.globalFineLevelSetMaximumBricks,
        POWER_DROPLET_FINE_BRICK_CAPACITY);
      assert.equal(lane.stop.maxDt_s, 0.004);
      const rule = (id: string) => lane.acceptance.find((entry) => entry.id === id);
      assert.deepEqual(rule("expected-grid")?.expected, [edgeCells, edgeCells, edgeCells]);
      // Not per-scene baselines: the same numbers at 64 cubed and at 256
      // cubed. A lane that needs a larger one is publishing domain-shaped
      // work, which is the defect the sweep exists to find. Measured at 64
      // cubed: 100 air-support rows and 624 resident fine bricks.
      assert.equal(rule("power-droplet-pinned-air-support-rows")?.expected, 1_024);
      assert.equal(rule("power-droplet-pinned-fine-residency")?.expected, 2_048);
      assert.equal(rule("power-droplet-pinned-rows")?.expected, 1_024);
      assert.equal(rule("power-droplet-row-arena-fits")?.expected, false);
    }
    assert.equal(getSceneWebGPUSmokeLane(`power-droplet-${edgeCells}`, "default").stop.exactSteps, 1);
    assert.equal(getSceneWebGPUSmokeLane(`power-droplet-${edgeCells}`, "runtime-240").stop.exactSteps, 240);
  }
});

test("the droplet benchmark lanes are self-describing and differ only in scene and grid", () => {
  const lanes = POWER_DROPLET_EDGE_CELLS.map((edgeCells) =>
    POWER_DAM_LANE_ENVIRONMENT[`droplet-${edgeCells}`]);
  for (const [index, lane] of lanes.entries()) {
    const edgeCells = POWER_DROPLET_EDGE_CELLS[index];
    assert.equal(lane.FLUID_SCENE, `power-droplet-${edgeCells}`);
    assert.equal(lane.FLUID_EXPECT_GRID, `${edgeCells},${edgeCells},${edgeCells}`);
    // The benchmark path resolves method values from the environment and never
    // reads the scene's authored profile, so a lane that omits this capacity
    // does not run this scene — it dies at t=0 on the planner's 256 rows.
    assert.equal(lane.FLUID_PRESSURE_ROW_CAPACITY, String(POWER_DROPLET_PRESSURE_ROW_CAPACITY));
    for (const key of ["FLUID_MAX_DT", "FLUID_MAXIMUM_LEAF_SIZE", "FLUID_OCTREE_INTERFACE_BAND",
      "FLUID_OCTREE_GLOBAL_FINE_FACTOR", "FLUID_ORACLE_STEPS"] as const) {
      assert.equal(lane[key], lanes[0][key], `droplet-${edgeCells} must match droplet-64 on ${key}`);
    }
    // 256 cubed binds two ~134 MB dense fine directories, over Dawn's 128 MiB
    // default. Removing the allocation is the memory milestone; until then the
    // lane must at least be able to bind.
    assert.equal(Object.hasOwn(lane, "FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES"), edgeCells >= 256);
  }

  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  for (const edgeCells of POWER_DROPLET_EDGE_CELLS) {
    assert.equal(packageJson.scripts[`benchmark:power-droplet-${edgeCells}`],
      `node --import tsx tools/benchmark-power-dam.ts --lane=droplet-${edgeCells}`);
    for (const [script, laneId] of [
      [`test:webgpu:power-droplet-${edgeCells}`, "default"],
      [`test:webgpu:power-droplet-${edgeCells}-240`, "runtime-240"],
    ] as const) {
      const command = packageJson.scripts[script];
      assert.match(command, new RegExp(`FLUID_SCENE=power-droplet-${edgeCells} `));
      assert.match(command, new RegExp(`FLUID_LANE=${laneId} `));
      assert.match(command, new RegExp(`FLUID_EXPECT_GRID=${edgeCells},${edgeCells},${edgeCells} `));
      assert.match(command, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000/);
      assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
    }
  }
});
