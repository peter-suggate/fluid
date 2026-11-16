// The box observer on the brick engine (tools/nbflip-observe-brick.ts):
// - all-h, against boxOracle (tools/nbflip-observables.ts) over the same pool: phi and face velocities read back
//   from the engine, owners its live h cells;
// - tiered (the letters' pool without its drops, wholly 4h; the dam's collapse, h and 4h), against the app's measure: the observed liquid is the
//   presentation volume (brick-present.ts, the tier authority's per-cell fill) and the gauge's, and the particle
//   count is the live bank's. Until 2026-11-17 the observer read F cells only and lost the 4h interior.
// GPU only (WEBGPU_NODE_MODULE).
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
import { H } from "../lib/methods/narrow-band/brick-abi";
import { BRICK_ENGINE_SCENES, brickSceneSetup } from "../lib/methods/narrow-band/brick-scene";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { FH, VOL_UNIT } from "../lib/methods/narrow-band/velocity-kernels.wgsl";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { boxGeometry } from "../tools/nbflip-observables";
import "../lib/methods";
import type { SceneDescription } from "../lib/core/model";

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

/** A catalog scene through the app's createBrickSolver (the scene gate opened for it), its resolved values and step. */
function appScene(id: string, edit: (scene: SceneDescription) => SceneDescription = s => s) {
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition(id), scene = edit(sceneDocument(definition));
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  return { id, scene, quality, values };
}

test("box observer (GPU, brick engine, tiered): h and 4h owners give the presentation's volume and the live bank's particles", GPU, async () => {
  const cases = [
    // The letters' pool alone (no drops): calm, it goes 4h whole within a step (F 0), so every owner is a 4h tile.
    { ...appScene("nbflip-figure-8-letters", s => ({ ...s, fluid: { ...s.fluid, scheduledDrops: [] } })), steps: 5, mixed: false },
    // The dam's collapse: h-owned band and front, 4h-owned column interior.
    { ...appScene("high-resolution-dam-break"), steps: 10, mixed: true },
  ];
  await withUniformDevice("brick observer tiered", async device => {
    for (const { id, scene, quality, values, steps, mixed } of cases) {
      const enabled = BRICK_ENGINE_SCENES.includes(id);
      if (!enabled) (BRICK_ENGINE_SCENES as string[]).push(id);
      try {
        const solver = await createBrickSolver(device, scene, values, quality as never);
        const e = solver.brickEngine, dt = brickSceneSetup(scene, values).setup!.dt_s;
        try {
          assert.equal(e.stages.tiered, true, `${id} runs tiered`);
          for (let k = 1; k <= steps; k++) { assert.ok(solver.advanceTo(k * dt + 1e-12, [])); await solver.assertSimulationHealthy(); }
          const head = await e.read("set", 0, 256), f = new Float32Array(head.buffer);
          assert.ok(head[H.LIVE_C]! > 0, `${id}: C pages live (${head[H.LIVE_C]})`);
          if (mixed) assert.ok(head[H.LIVE_F]! > 0, `${id}: F bricks live too (${head[H.LIVE_F]})`);
          // The presentation volume (cells): its texture's sum.
          const tex = solver.brickPresentation.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
          const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
          await buf.mapAsync(GPUMapMode.READ); const v = new Float32Array(buf.getMappedRange()); let presented = 0;
          for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) { const o = (z * hgt + y) * bpr / 4; for (let x = 0; x < w; x++) presented += v[o + x]!; }
          buf.unmap(); buf.destroy();
          const g = boxGeometry(scene, id === "high-resolution-dam-break" ? "dam" : "letters"), observer = await createNBFlipBoxObserver(device, solver, scene, g);
          try {
            const got = await observer.observe(), budget = head[FH.VOL_BUDGET + 1]! * (2 ** 32 / VOL_UNIT) + head[FH.VOL_BUDGET]! / VOL_UNIT;
            console.log(`${id} step ${steps}: observed ${got.total.toFixed(1)}, presented ${presented.toFixed(1)}, gauge ${f[FH.VOL_FILLED]!.toFixed(1)}, budget ${budget.toFixed(1)} cells; particles ${got.particles}; F ${head[H.LIVE_F]} C ${head[H.LIVE_C]}`);
            // The two fills agree where φ is planar inside a 4h tile; the curved band leaves second-order terms.
            assert.ok(Math.abs(got.total - presented) <= 1e-3 * presented, `${id}: observed ${got.total} against the presentation's ${presented}`);
            assert.ok(Math.abs(got.total - f[FH.VOL_FILLED]!) <= 1e-2 * budget, `${id}: observed ${got.total} against the gauge ${f[FH.VOL_FILLED]}`);
            const live = Math.min(head[H.PCOUNT0 + (head[H.STEP]! & 1)]!, e.layout.cap.particles);
            assert.equal(got.particles, live, `${id}: the observer's particles are the live bank's`);
            if (mixed) assert.ok(live > 0, `${id}: the band has particles`);
          } finally { observer.destroy(); }
        } finally { solver.destroy(); }
      } finally { if (!enabled) (BRICK_ENGINE_SCENES as string[]).pop(); }
    }
  });
});
