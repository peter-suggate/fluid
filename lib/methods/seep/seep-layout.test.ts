/** Seep solver (lane S): CPU checks of the baked scene contract, brick classes
 * and the initial level set. GPU behaviour is exercised by tools/seep-solver-smoke.ts
 * and lane H's seep-bench/seep-compare. */
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import {
  BRICKS, BX, BY, BZ, CLASS_CHAMBER, CLASS_DEAD, CLASS_DEEP, CLASS_PARTICLE, DEEP_ROW_LIMIT, NX, NY, NZ, PHI_MAX,
  RESERVOIR_TOP, THROAT_Y0, THROAT_Z0, WALL_X0, WALL_X1, assertSeepScene, bakeSeep, brickId, cellIndex, cellOpen,
} from "./seep-layout";

const seep = () => sceneDocument(getSceneDefinition("reservoir-seep"));

test("seep layout accepts exactly the default reservoir-seep scene", () => {
  assert.doesNotThrow(() => assertSeepScene(seep()));
  assert.throws(() => assertSeepScene({ ...seep(), sceneId: "dam-break" }), /supports only the default reservoir-seep scene/);
  const moved = seep();
  const solids = [...(moved.solidVoxels ?? [])];
  solids[6] = { ...solids[6]!, minimum: [WALL_X0, THROAT_Y0 + 1, THROAT_Z0] };
  assert.throws(() => assertSeepScene({ ...moved, solidVoxels: solids }), /solid voxel operation 6/);
  assert.throws(() => assertSeepScene({ ...seep(), fluid: { ...seep().fluid, dynamicViscosity_Pa_s: 1e-3 } }), /inviscid/);
});

test("seep solids: the slab is closed except the 2x2 throat", () => {
  assert.equal(cellOpen(WALL_X0 - 1, 10, 10), true);
  assert.equal(cellOpen(WALL_X0, 10, 10), false);
  assert.equal(cellOpen(WALL_X1 - 1, THROAT_Y0, THROAT_Z0), true);
  assert.equal(cellOpen(WALL_X1 - 1, THROAT_Y0 + 2, THROAT_Z0), false);
  assert.equal(cellOpen(WALL_X1, 0, 0), true);
  assert.equal(cellOpen(NX, 0, 0), false);
  assert.equal(cellOpen(0, -1, 0), false);
});

test("seep bake: brick classes and the initial level set", () => {
  const bake = bakeSeep();
  assert.equal(bake.brickClass.length, BRICKS);
  const cls = (bx: number, by: number, bz: number) => bake.brickClass[brickId(bx, by, bz)]!;
  // Wall interior away from the throat: no vertex touches an open cell.
  assert.equal(cls(31, 2, 2), CLASS_DEAD);
  // The throat bricks keep a live level set: not DEEP, not particle.
  assert.equal(cls(30, THROAT_Y0 >> 3, THROAT_Z0 >> 3) & (CLASS_DEEP | CLASS_PARTICLE | CLASS_DEAD), 0);
  assert.equal(cls(31, THROAT_Y0 >> 3, THROAT_Z0 >> 3) & (CLASS_DEEP | CLASS_PARTICLE | CLASS_DEAD), 0);
  // Reservoir bricks wholly below the DEEP row are DEEP; the surface rows are not.
  assert.ok(cls(5, DEEP_ROW_LIMIT / 8 - 1, 3) & CLASS_DEEP);
  assert.equal(cls(5, RESERVOIR_TOP / 8 - 1, 3) & CLASS_DEEP, 0);
  // Every right-chamber brick is a particle chamber brick and never DEEP.
  for (let bz = 0; bz < BZ; bz++) for (let by = 0; by < BY; by++) for (let bx = WALL_X1 / 8; bx < BX; bx++) {
    const c = cls(bx, by, bz);
    assert.equal(c & (CLASS_PARTICLE | CLASS_CHAMBER), CLASS_PARTICLE | CLASS_CHAMBER);
    assert.equal(c & CLASS_DEEP, 0);
  }
  // Initial liquid: the reservoir x < 240, y < 192 and the wetted throat, by cell-centre sign over open cells.
  let liquid = 0, throat = 0;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    if (!cellOpen(x, y, z) || !(bake.phiCell[cellIndex(x, y, z)]! < 0)) continue;
    liquid++;
    if (x >= WALL_X0 && x < WALL_X1) throat++;
  }
  assert.equal(throat, (WALL_X1 - WALL_X0) * 4);
  assert.equal(liquid, WALL_X0 * RESERVOIR_TOP * NZ + throat);
  // The level set is clamped to the band and is a distance near the reservoir surface.
  assert.ok(bake.phi.every(v => Math.abs(v) <= PHI_MAX));
  assert.equal(bake.phi[cellIndex(100, RESERVOIR_TOP - 2, 30)], -2);
  assert.equal(bake.phi[cellIndex(100, RESERVOIR_TOP + 3, 30)], 3);
});
