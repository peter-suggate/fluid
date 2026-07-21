import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS,
  octreeProjectionShader,
  planOctreePowerBoundaryStrip,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";

const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("power boundary strip has an exact closed-wall cell and owner-page bound", () => {
  const open = planOctreePowerBoundaryStrip({ nx: 24, ny: 18, nz: 16 }, 4, false);
  assert.deepEqual(open, {
    widthCells: 4,
    unitCellUpperBound: 24 * 18 * 16 - 16 * 14 * 8,
    // The 16-cell depth has no complete 8-cubed page between two 4-cell strips.
    ownerPageUpperBound: 3 * 3 * 2,
  });

  const closed = planOctreePowerBoundaryStrip({ nx: 24, ny: 18, nz: 16 }, 4, true);
  assert.deepEqual(closed, {
    widthCells: 4,
    unitCellUpperBound: 24 * 18 * 16 - 16 * 10 * 8,
    ownerPageUpperBound: 3 * 3 * 2,
  });
  assert.ok(closed.unitCellUpperBound > open.unitCellUpperBound,
    "only an authored closed ceiling may add the upper-y strip");
});

test("strip width covers both the paper boundary support and configured trajectory band", () => {
  assert.equal(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, 3);
  assert.equal(planOctreePowerBoundaryStrip({ nx: 64, ny: 48, nz: 32 }, 0).widthCells, 3);
  assert.equal(planOctreePowerBoundaryStrip({ nx: 64, ny: 48, nz: 32 }, 7.2).widthCells, 8);
  assert.throws(() => planOctreePowerBoundaryStrip({ nx: 0, ny: 1, nz: 1 }, 4), /dimensions/);
  assert.throws(() => planOctreePowerBoundaryStrip({ nx: 1, ny: 1, nz: 1 }, -1), /interface band/);
});

test("power pressure capacity reserves the wall strip without charging rollback mode", () => {
  const dims = { nx: 288, ny: 96, nz: 64 };
  const rollback = planOctreePressureCapacity(dims, 16, 4);
  const wall = planOctreePowerBoundaryStrip(dims, 4, false);
  const power = planOctreePressureCapacity(dims, 16, 4, undefined, true, false);
  const closedPower = planOctreePressureCapacity(dims, 16, 4, undefined, true, true);
  assert.ok(power.rowCapacity >= wall.unitCellUpperBound);
  assert.ok(power.rowCapacity > rollback.rowCapacity);
  assert.ok(closedPower.rowCapacity > power.rowCapacity);
  assert.equal(planOctreePressureCapacity(dims, 16, 4, 1024, true, true).rowCapacity, 1024,
    "an explicit diagnostic override remains authoritative and fail-closed on overflow");
});

test("topology forces authoritative closed walls to unit owners before phi sizing", () => {
  assert.match(octreeProjectionShader,
    /fn powerClosedWallStripIntersects\(origin: vec3u, size: u32\) -> bool/);
  assert.match(octreeProjectionShader, /if \(\(flags & 4u\) == 0u\) \{ return false; \}/,
    "rollback topology must not pay for the power-only strip");
  assert.match(octreeProjectionShader,
    /let width = max\(3u,[\s\S]*u32\(ceil\(max\(0\.0, params\.solve\.w\)\)\)\);/);
  assert.match(octreeProjectionShader,
    /origin\.x < min\(width, d\.x\)[\s\S]*origin\.z < min\(width, d\.z\)[\s\S]*origin\.y < min\(width, d\.y\)[\s\S]*\(flags & 2u\) != 0u/);
  assert.match(octreeProjectionShader,
    /fn leafNeedsRefinement[\s\S]*if \(powerClosedWallStripIntersects\(origin, size\)\) \{ return true; \}[\s\S]*let fineSummary/,
    "wall refinement must precede sparse-phi absence and pure-solid early exits");
});

test("host flags distinguish terrain, closed ceiling, and power strip", () => {
  assert.match(source,
    /const containerFlags = \(sceneHasTerrain\(this\.scene\) \? 1 : 0\)[\s\S]*container\.top === "closed" \? 2 : 0[\s\S]*powerPolicy\.authoritative \? 4 : 0/);
  assert.match(octreeProjectionShader,
    /\(u32\(round\(params\.container\.w\)\) & 1u\) != 0u[\s\S]*textureLoad\(terrainIn/,
    "closed-top and power bits must not accidentally enable terrain sampling");
  assert.match(source,
    /planOctreePressureCapacity\([\s\S]*this\.powerPolicy\.authoritative,[\s\S]*scene\.container\.top === "closed"/,
    "the allocation bound must use the same ceiling policy as the shader");
});
