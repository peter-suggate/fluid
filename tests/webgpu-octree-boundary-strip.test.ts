import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS,
  
  planOctreePowerBoundaryStrip,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";


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

test("power pressure capacity always reserves the authoritative wall strip", () => {
  const dims = { nx: 288, ny: 96, nz: 64 };
  const wall = planOctreePowerBoundaryStrip(dims, 4, false);
  const power = planOctreePressureCapacity(dims, 16, 4);
  const closedPower = planOctreePressureCapacity(dims, 16, 4, undefined, true);
  assert.ok(power.rowCapacity >= wall.unitCellUpperBound);
  assert.ok(closedPower.rowCapacity > power.rowCapacity);
  assert.equal(planOctreePressureCapacity(dims, 16, 4, 1024, true).rowCapacity, 1024,
    "an explicit diagnostic override remains authoritative and fail-closed on overflow");
});

