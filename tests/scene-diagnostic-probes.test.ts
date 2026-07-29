import assert from "node:assert/strict";
import test from "node:test";
import { createBrickQuadDamBreakScene, createOceanSeicheScene } from "../lib/scenes";
import { probeBrickQuadCoverage } from "../tools/brick-quad-coverage-probe";
import { probeOceanWavePropagation } from "../tools/ocean-wave-propagation-probe";
import type { SceneFieldEvidence } from "../tools/scene-diagnostic-probe";

function fillColumnLayers(
  field: Float32Array,
  grid: readonly [number, number, number],
  xRange: readonly [number, number],
  layers: number,
): void {
  const [nx, ny, nz] = grid;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < Math.min(ny, layers); y += 1) {
    for (let x = xRange[0]; x < xRange[1]; x += 1) field[x + nx * (y + ny * z)] = 1;
  }
}

test("ocean wave probe reports a far-half signal and a serializable station time series", () => {
  const scene = createOceanSeicheScene();
  scene.sceneId = "small-ocean-probe-fixture";
  scene.container = { ...scene.container, width_m: 4, height_m: 2, depth_m: 1, fillFraction: 0.5 };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.5 };
  const grid = [8, 4, 2] as const;
  const baseline = new Float32Array(grid[0] * grid[1] * grid[2]);
  fillColumnLayers(baseline, grid, [0, grid[0]], 2);
  const leadingWave = baseline.slice();
  fillColumnLayers(leadingWave, grid, [6, 8], 3);
  const evidence: SceneFieldEvidence = {
    method: "octree",
    grid,
    checkpoints: [
      { time_s: 0.5, field: baseline },
      { time_s: 1, field: baseline },
      { time_s: 1.5, field: leadingWave },
    ],
  };

  const result = probeOceanWavePropagation(scene, evidence, {
    minimumFarHalfDisturbance_cells: 0.5,
    stationCount: 4,
  });

  assert.deepEqual(result.findings, []);
  assert.equal(result.observations.baselineHeight_cells, 2);
  assert.equal(result.observations.cellHeight_m, 0.5);
  assert.equal(result.observations.farHalfDisturbance_cells, 1);
  assert.equal(result.observations.stationX_m.length, 4);
  assert.equal(result.observations.checkpoints.length, 3);
  assert.ok((result.observations.crestReach_m ?? -Infinity) > 0);
  assert.doesNotThrow(() => JSON.stringify(result));
});

test("ocean wave probe returns findings instead of throwing for absent propagation and malformed evidence", () => {
  const scene = createOceanSeicheScene();
  scene.sceneId = "small-ocean-probe-fixture";
  scene.container = { ...scene.container, width_m: 4, height_m: 2, depth_m: 1, fillFraction: 0.5 };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.5 };
  const grid = [8, 4, 2] as const;
  const baseline = new Float32Array(8 * 4 * 2);
  fillColumnLayers(baseline, grid, [0, 8], 2);

  const still = probeOceanWavePropagation(scene, {
    method: "tall-cell",
    grid,
    checkpoints: [0, 1, 2].map((time_s) => ({ time_s, field: baseline })),
  }, { minimumFarHalfDisturbance_cells: 0.25 });
  assert.deepEqual(still.findings.map(({ id }) => id), ["ocean-wave.far-half-disturbance"]);

  const malformed = probeOceanWavePropagation(scene, {
    method: "octree",
    grid: [8, 4, 3],
    checkpoints: [{ time_s: Number.NaN, field: new Float32Array(1) }],
  }, { minimumFarHalfDisturbance_cells: Number.NaN });
  assert.deepEqual(malformed.findings.map(({ id }) => id), [
    "scene-grid.mismatch",
    "field-evidence.checkpoint-invalid",
    "ocean-wave.threshold-invalid",
  ]);
});

function wetBrickColumns(
  grid: readonly [number, number, number],
  columns: readonly (readonly [number, number])[],
): Float32Array {
  const [nx, ny, nz] = grid;
  const field = new Float32Array(nx * ny * nz);
  for (const [bx, bz] of columns) {
    for (let z = bz * 8; z < Math.min(nz, (bz + 1) * 8); z += 1) {
      for (let y = 0; y < ny; y += 1) for (let x = bx * 8; x < Math.min(nx, (bx + 1) * 8); x += 1) {
        field[x + nx * (y + ny * z)] = 1;
      }
    }
  }
  return field;
}

test("brick-quad probe derives brick topology from the scene and observes all columns", () => {
  const scene = createBrickQuadDamBreakScene();
  const grid = [16, 8, 16] as const;
  const evidence: SceneFieldEvidence = {
    method: "octree",
    grid,
    checkpoints: [
      { time_s: 0.25, field: wetBrickColumns(grid, [[0, 0], [1, 0]]) },
      { time_s: 0.5, field: wetBrickColumns(grid, [[0, 0], [1, 0], [0, 1], [1, 1]]) },
    ],
  };

  const result = probeBrickQuadCoverage(scene, evidence);
  assert.deepEqual(result.findings, []);
  assert.deepEqual(result.observations.brickGrid, [2, 1, 2]);
  assert.deepEqual(result.observations.wetBrickColumns, ["0,0", "0,1", "1,0", "1,1"]);
  assert.equal(result.observations.farColumn, "1,1");
  assert.deepEqual(result.observations.checkpoints[0], {
    time_s: 0.25,
    wetBrickColumns: ["0,0", "1,0"],
  });
});

test("brick-quad probe expresses each coverage failure without side effects", () => {
  const scene = createBrickQuadDamBreakScene();
  const grid = [16, 8, 16] as const;
  const field = wetBrickColumns(grid, [[0, 0]]);
  const before = field.slice();
  const result = probeBrickQuadCoverage(scene, {
    method: "tall-cell",
    grid,
    checkpoints: [{ time_s: 0.25, field }],
  });

  assert.deepEqual(result.findings.map(({ id }) => id), [
    "brick-quad.first-boundary-crossing",
    "brick-quad.all-columns",
    "brick-quad.far-column",
  ]);
  assert.deepEqual(field, before, "the probe must not mutate normalized field evidence");
});
