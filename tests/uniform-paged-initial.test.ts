import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneFluidLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import type { LatticePages } from "../lib/core/initial-liquid-lattice";
import type { SceneDescription } from "../lib/core/model";
import { UNIFORM_PAGE_POOL_EDGE, uniformPagePoolResidency } from "../lib/methods/uniform/uniform-page-pool";
import { uniformInitialLiquidVelocity, uniformInitialLiquidVelocityPages, uniformLatticeInitialPhi, uniformLatticeInitialVolume,
  uniformLatticePagedPhi, uniformLatticePagedVolume } from "../lib/methods/uniform/uniform-volume-initial";

// Every initial source the host builds from: dam and tank fills, terrain, a
// procedural boundary, open worlds, authored boxes (one, many, seams), every
// analytic shape, height fields, replacing and additive bricks. Large scenes
// at a coarser cell, which keeps their sources.
const SCENES: readonly [string, number][] = [
  ["water-box-dam-break", 1], ["twin-dam-collision", 1], ["ocean-seiche", 4], ["tall-cells-hillside-dam-break", 4],
  ["backyard-pool-party", 8], ["garden-pond", 2], ["garden-dam-break", 2], ["dam-break-boxes", 1], ["cm12-figure-3", 2],
  ["cm12-figure-8", 4], ["falling-water-torus", 1], ["nbflip-figure-8-letters", 4], ["reservoir-seep", 4], ["nbflip-open-ground", 1],
  ["nbflip-open-ground-faces", 1], ["nbflip-open-ground-slope", 1], ["nbflip-analytic-standing-wave", 1], ["stationary-bowl", 1],
  ["thin-wall-films", 2], ["uniform-trough-dam-break", 4], ["gentle-moving-blob", 1], ["nbflip-slope-clearance", 1], ["forest-waterfall", 8],
];
function scene(id: string, coarsen: number): SceneDescription {
  const s = structuredClone(sceneDocument(getSceneDefinition(id)));
  s.voxelDomain.finestCellSize_m *= coarsen;
  return s;
}
const bits = (v: number) => new Uint32Array(Float32Array.of(v).buffer)[0]!;
/** The value of a cell (comp values a texel) or vertex of a field by page. */
function paged(p: LatticePages, comp: number, vertex: boolean, x: number, y: number, z: number, c = 0): number | undefined {
  const E = p.edge, n = vertex ? E + 1 : E, [px, py] = p.pages, e = (v: number) => vertex ? Math.max(0, Math.ceil(v / E) - 1) : Math.floor(v / E);
  // A vertex on a page's high face is that page's (its 33rd); every page's own vertices are tested below.
  const qx = e(x), qy = e(y), qz = e(z), page = p.at[qx + px * (qy + py * qz)];
  if (!page) return undefined;
  return page[comp * ((x - qx * E) + n * ((y - qy * E) + n * (z - qz * E))) + c];
}

for (const [id, coarsen] of SCENES) test(`paged initial conditions match the dense ones bit for bit: ${id}`, () => {
  const s = scene(id, coarsen);
  const dense = uniformLatticeInitialVolume(s), byPage = uniformLatticePagedVolume(s, UNIFORM_PAGE_POOL_EDGE);
  const v = byPage.volume, [vx, vy, vz] = v.cells;
  assert.deepEqual([...v.cells], [...sceneFluidLatticeDimensions(s, Number.MAX_SAFE_INTEGER)]);
  assert.equal(dense.volume.length, vx * vy * vz);
  for (let z = 0, i = 0; z < vz; z++) for (let y = 0; y < vy; y++) for (let x = 0; x < vx; x++, i++)
    if (bits(dense.volume[i]!) !== bits(paged(v, 1, false, x, y, z) ?? 0)) assert.fail(`volume (${x}, ${y}, ${z}): ${dense.volume[i]} by page ${paged(v, 1, false, x, y, z)}`);
  assert.equal(byPage.initial, dense.initial);
  assert.deepEqual([byPage.wetMinimum, byPage.wetMaximum], [dense.wetMinimum, dense.wetMaximum]);
  assert.deepEqual([...byPage.terrain], [...dense.terrain]);
  assert.deepEqual(byPage.dam, dense.dam);
  // The wet-page scan reads the same liquid.
  assert.deepEqual(uniformPagePoolResidency([vx, vy, vz], undefined, [], true, () => v), uniformPagePoolResidency([vx, vy, vz], undefined, [], true, () => dense.volume));
  // Velocity, at the scene's or a test velocity.
  const own = s.fluid.initialVelocity_m_s, velocity = own && (own.x || own.y || own.z) ? own : { x: 0.25, y: -0.5, z: 0.125 };
  const field = uniformInitialLiquidVelocity(velocity, dense.volume, [vx, vy, vz])!, pages = uniformInitialLiquidVelocityPages(velocity, v)!;
  for (let z = 0, i = 0; z < vz; z++) for (let y = 0; y < vy; y++) for (let x = 0; x < vx; x++, i++) for (let c = 0; c < 4; c++)
    if (field[4 * i + c] !== (paged(pages, 4, false, x, y, z, c) ?? 0)) assert.fail(`velocity (${x}, ${y}, ${z}).${c}`);
  // Phi at every stored page's vertices, both burial rules where the lattice tiles.
  for (const coarseSolids of [false, true]) {
    let phi: Float32Array;
    try { phi = uniformLatticeInitialPhi(s, coarseSolids); } catch (error) {
      assert.throws(() => uniformLatticePagedPhi(s, UNIFORM_PAGE_POOL_EDGE, () => true, coarseSolids), { message: (error as Error).message });
      continue;
    }
    const held = uniformLatticePagedPhi(s, UNIFORM_PAGE_POOL_EDGE, (x, y, z) => (x + y + z) % 2 === 0, coarseSolids);
    const rest = uniformLatticePagedPhi(s, UNIFORM_PAGE_POOL_EDGE, (x, y, z) => (x + y + z) % 2 === 1, coarseSolids);
    const E = UNIFORM_PAGE_POOL_EDGE, [px, py, pz] = held.pages;
    for (let qz = 0; qz < pz; qz++) for (let qy = 0; qy < py; qy++) for (let qx = 0; qx < px; qx++) {
      const mine = (qx + qy + qz) % 2 === 0 ? held : rest, other = mine === held ? rest : held, page = mine.at[qx + px * (qy + py * qz)]!;
      assert.equal(other.at[qx + px * (qy + py * qz)], undefined);
      for (let k = 0; k <= E; k++) for (let j = 0; j <= E; j++) for (let i = 0; i <= E; i++) {
        const x = qx * E + i, y = qy * E + j, z = qz * E + k;
        if (x > vx || y > vy || z > vz) continue;
        const want = phi[x + (vx + 1) * (y + (vy + 1) * z)]!, got = page[i + (E + 1) * (j + (E + 1) * k)]!;
        if (bits(want) !== bits(got)) assert.fail(`phi (${x}, ${y}, ${z})${coarseSolids ? " cut" : ""}: ${want} by page ${got}`);
      }
    }
  }
});
