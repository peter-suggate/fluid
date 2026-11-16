// The box observer on the brick engine (tools/nbflip-observe-brick.ts) against
// boxOracle (tools/nbflip-observables.ts) over the same pool: phi and face
// velocities read back from the engine, owners its live h cells. GPU only
// (WEBGPU_NODE_MODULE).
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_STRIDE, PHI_AIR } from "../lib/methods/narrow-band/brick-abi";
import { bringUpCG } from "../lib/methods/narrow-band/brick-bringup-cg";
import type { Vec3 } from "../lib/methods/narrow-band/brick-engine";
import { cellSlot, keyOf, readChannel, snapshot } from "../lib/methods/narrow-band/brick-readback";
import type { BrickSceneSetup } from "../lib/methods/narrow-band/brick-scene";
import { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { bringUpStages } from "../lib/methods/narrow-band/brick-stages";
import { boxOracle, type BoxGeometry } from "../tools/nbflip-observables";
import { createNBFlipBoxObserver } from "../tools/nbflip-observe";
import { withUniformDevice } from "./helpers/uniform-geometric";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const H_M = 1 / 32, DT = 1 / 60;

test("box observer (GPU, brick engine): matches the oracle over the pool's cells, faces and walls", GPU, async () => {
  await withUniformDevice("brick observer", async device => {
    const n: Vec3 = [64, 32, 64];
    const setup: BrickSceneSetup = { sceneId: "test", lattice: n, h_m: H_M, dt_s: DT, gravity_mps2: [0, -9.81, 0], density_kg_m3: 1000, viscosity_Pa_s: 0, surfaceTension_N_m: 0,
      capacities: { f: 400, c: 16, b: 0, k: 0, particles: 1024 }, particleBandDepth: 3, liquid: [{ lo: [-1, -1, -1], hi: [20, 10.3, 20] }], origin_m: [0, 0, 0] };
    const solver = await BrickNBFlipSolver.create(device, { setup, stages: bringUpStages, solver: L => bringUpCG(L), values: {}, quality: "balanced", farPhi_m: 2 });
    assert.equal(solver.advanceTo(DT, []), true);
    const e = solver.brickEngine, s = await snapshot(e);
    // A known face field, so every face read counts: U_a(c) = 0.01 (a + 1) + 0.002 x - 0.001 z.
    const face = (c: Vec3, a: number) => 0.01 * (a + 1) + 0.002 * c[0] - 0.001 * c[2];
    for (const [key, slot] of s.bricks) {
      const b = keyOf(key);
      for (let a = 0; a < 3; a++) {
        const w = new Float32Array(512);
        for (let l = 0; l < 512; l++) w[l] = face([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)], a);
        e.write("vel", e.layout.base("vel", "F", `U${a}`) + BRICK_STRIDE * slot, w);
      }
    }
    const phi = await readChannel(e, "phi", "PHI");
    const vertex = (x: number, y: number, z: number) => {
      if (x < 0 || y < 0 || z < 0 || x > n[0] || y > n[1] || z > n[2]) return PHI_AIR;
      const at = cellSlot(s, [x, y, z]);
      return at ? phi(at.slot)[at.local]! : PHI_AIR;
    };
    const faceAt = (c: Vec3, a: number): [number, number] => {
      if (c.some((v, i) => v < 0 || v >= n[i]!)) return [0, 1];
      return cellSlot(s, c) ? [face(c, a), 1] : [0, 0];
    };
    const owners: { q: [number, number, number]; w: number; phi: typeof vertex; u: [number, number, number] }[] = [];
    let dropped = 0;
    for (const key of s.bricks.keys()) {
      const b = keyOf(key);
      for (let l = 0; l < 512; l++) {
        const c: Vec3 = [8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)];
        if (c.some((v, i) => v >= n[i]!)) continue;
        const u = [0, 1, 2].map(a => {
          const up: Vec3 = [c[0] + (a === 0 ? 1 : 0), c[1] + (a === 1 ? 1 : 0), c[2] + (a === 2 ? 1 : 0)];
          const lo = faceAt(c, a), hi = faceAt(up, a);
          if (lo[1] + hi[1] < 2) dropped++;
          return (lo[0] + hi[0]) / Math.max(lo[1] + hi[1], 1);
        }) as [number, number, number];
        owners.push({ q: [c[0], c[1], c[2]], w: 1, phi: vertex, u });
      }
    }
    const g: BoxGeometry = { nx: n[0], ny: n[1], nz: n[2], cell_m: H_M, gravity_m_s2: 9.81, density_kg_m3: 1000, binCells: [8, 8], bins: [8, 8], column: { x: [0, 4], z: [0, 12] } };
    const want = boxOracle(g, owners);
    const observer = await createNBFlipBoxObserver(device, solver, undefined as never, g);
    const got = await observer.observe();
    // Fixed point: half a unit of 1/4096 a cell (totals), 1/64 (moments), 1/1024 (ke).
    const cells = owners.length;
    assert.ok(want.total > 20 * 10 * 20 * 0.99, `the sheet's liquid: ${want.total}`);
    assert.ok(Math.abs(got.total - want.total) <= cells / 8192, `total ${got.total} against ${want.total}`);
    for (const k of ["xMoment", "yMoment", "zMoment"] as const) assert.ok(Math.abs(got[k] - want[k]) <= cells / 128, `${k} ${got[k]} against ${want[k]}`);
    assert.ok(want.ke > 0 && Math.abs(got.ke - want.ke) <= cells / 2048, `ke ${got.ke} against ${want.ke}`);
    assert.ok(Math.abs(got.column - want.column) <= cells / 8192);
    got.bins.forEach((v, i) => assert.ok(Math.abs(v - want.bins[i]!) <= cells / 8192, `bin ${i}`));
    got.tops.forEach((v, i) => assert.ok(Math.abs(v - want.tops[i]!) <= 1 / 32, `top ${i}: ${v} against ${want.tops[i]}`));
    assert.equal(got.front, want.front); assert.equal(got.wetFloor, want.wetFloor);
    assert.ok(Math.abs(got.columnTop - want.columnTop) <= 1 / 32);
    assert.equal(got.bulkRegions, 0);
    assert.ok(dropped > 0, "some faces lie in bricks that are not resident");
    observer.destroy(); solver.destroy();
  });
});
