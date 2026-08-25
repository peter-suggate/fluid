import assert from "node:assert/strict";
import test from "node:test";

import { validateScene } from "../lib/core/model";
import {
  createTallCellsHillsideDamBreakScene,
  findSceneDefinition,
  TALL_CELLS_FLOOD_CELL_SIZE_M,
  TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M,
  TALL_CELLS_FLOOD_GRID,
  TALL_CELLS_FLOOD_RESERVOIR_M,
  TALL_CELLS_FLOOD_UPHILL_HEIGHT_M,
} from "../lib/core/scenes";
import { sampleSolidWorld, solidWorldForScene,
  SOLID_WORLD_TERRAIN_MATERIAL_ID } from "../lib/core/solid-world";
import { terrainHeightAt } from "../lib/core/terrain";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  buildSvoPrimitiveCandidates,
  packSvoPrimitiveCandidateArena,
  querySvoPrimitiveCandidates,
} from "../lib/svo/svo-primitive-candidates";
import { buildSvoScenePrimitives } from "../lib/svo/svo-scene-primitives";

const [NX, NY, NZ] = TALL_CELLS_FLOOD_GRID;
const WIDTH_M = NX * TALL_CELLS_FLOOD_CELL_SIZE_M;
const HEIGHT_M = NY * TALL_CELLS_FLOOD_CELL_SIZE_M;
const DEPTH_M = NZ * TALL_CELLS_FLOOD_CELL_SIZE_M;

test("Tall Cells hillside scene preserves the published Flood footprint and paper step", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.sceneId, "tall-cells-hillside-dam-break");
  assert.deepEqual(scene.container, {
    ...scene.container,
    width_m: WIDTH_M,
    height_m: HEIGHT_M,
    depth_m: DEPTH_M,
    fillFraction: TALL_CELLS_FLOOD_RESERVOIR_M.x
      * TALL_CELLS_FLOOD_RESERVOIR_M.y
      * TALL_CELLS_FLOOD_RESERVOIR_M.z / (WIDTH_M * HEIGHT_M * DEPTH_M),
    top: "open",
    fluidWallMode: "free-slip",
    vessel: "none",
  });
  assert.equal(scene.numerics.fixedDt_s, 1 / 30);
  assert.equal(scene.numerics.maxDt_s, 1 / 30);
  assert.deepEqual(scene.scenery?.nodes.map((node) => [node.kind, node.id]), [
    ["terrain-shell", "shell"],
    ["cone", "hillside/key"],
  ], "the benchmark retains terrain plus one bounded key, not the room-sized Stage set");
  assert.deepEqual(scene.fluid.optics, {
    absorption_mInv: [0.045, 0.009, 0.006],
    scatter: [0.0012, 0.0055, 0.0049],
  }, "the metre-scale reservoir must retain the catalog's clear-water optical depth");
  assert.deepEqual(scene.fluid.initialDamBreakDimensions_m,
    TALL_CELLS_FLOOD_RESERVOIR_M);
  assert.deepEqual(scene.fluid.initialDamBreakOrigin_m,
    { x: 0, y: TALL_CELLS_FLOOD_UPHILL_HEIGHT_M, z: 0.4 });

  const definition = findSceneDefinition(scene.sceneId);
  assert.equal(definition?.audience, "study");
  assert.equal(definition?.methodProfile?.methodId, "adaptive-mass");
  assert.equal(definition?.presentationMode, "full-scene");
});

test("height samples author a flat shelf, central slope, channel banks, and runout", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  const terrain = scene.terrain!;
  const minimumX = -0.5 * WIDTH_M;
  const slopeStart = minimumX + TALL_CELLS_FLOOD_RESERVOIR_M.x;
  const slopeEnd = 0.5 * WIDTH_M - 0.8;
  assert.deepEqual(terrain.grid?.size, { nx: NX + 1, nz: NZ + 1 });
  assert.equal(terrain.grid?.spacing_m, TALL_CELLS_FLOOD_CELL_SIZE_M);
  assert.equal(terrainHeightAt(terrain, minimumX, 0),
    TALL_CELLS_FLOOD_UPHILL_HEIGHT_M);
  assert.equal(terrainHeightAt(terrain, slopeStart, 0),
    TALL_CELLS_FLOOD_UPHILL_HEIGHT_M);
  assert.ok(Math.abs(terrainHeightAt(terrain, 0.5 * (slopeStart + slopeEnd), 0)
    - 0.5 * (TALL_CELLS_FLOOD_UPHILL_HEIGHT_M
      + TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M)) < 1e-12);
  assert.ok(Math.abs(terrainHeightAt(terrain, slopeEnd, 0)
    - TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M) < 1e-12);
  assert.ok(Math.abs(terrainHeightAt(terrain, 0.5 * WIDTH_M, 0)
    - TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M) < 1e-12);
  assert.ok(terrainHeightAt(terrain, 0.5 * (slopeStart + slopeEnd), 0.5 * DEPTH_M)
    > terrainHeightAt(terrain, 0.5 * (slopeStart + slopeEnd), 0));
});

test("the scene requires a unified sparse voxel solid and starts fluid-local", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  const solids = solidWorldForScene(scene);
  assert.equal(sampleSolidWorld(solids, [NX - 1, 0, Math.floor(NZ / 2)]).materialId,
    SOLID_WORLD_TERRAIN_MATERIAL_ID);
  assert.equal(sampleSolidWorld(solids, [NX - 1, NY - 1, Math.floor(NZ / 2)]).solidFraction,
    0, "SolidWorld claims terrain voxels rather than the logical world volume");

  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: TALL_CELLS_FLOOD_GRID,
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  assert.equal(atlas.bricks.length, 3 * 2 * 6,
    "generation zero must contain only the aligned 3 x 2 x 6 reservoir tiles");
  assert.ok(atlas.bricks.length < (NX / 8) * (NY / 8) * (NZ / 8) / 40,
    "initial fluid residency must stay independent of the vast dry extent");
});

test("bounded key publishes a sparse primitive index", () => {
  const primitives = buildSvoScenePrimitives(createTallCellsHillsideDamBreakScene());
  assert.equal(primitives.descriptors.length, 1);
  assert.equal(primitives.metadata[0]?.key, "default/hillside/key");
  const candidates = primitives.primitiveCandidates
    ?? buildSvoPrimitiveCandidates(primitives.descriptors as Parameters<
      typeof buildSvoPrimitiveCandidates>[0]);
  assert.equal(candidates.primitiveCount, 1);
  assert.equal(candidates.nodes.length, 1);
  assert.equal(packSvoPrimitiveCandidateArena(primitives.packedRecords, candidates)
    .candidateNodeCount, 1);
  assert.deepEqual(querySvoPrimitiveCandidates(candidates, {
    origin_m: { x: 0, y: 3, z: 0 },
    direction: { x: 0, y: 1, z: 0 },
  }), { primitiveIndices: [0], nodeVisits: 1, maximumStackDepth: 1 });
});
