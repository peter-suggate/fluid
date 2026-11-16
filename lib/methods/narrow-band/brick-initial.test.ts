/**
 * brick-initial.ts (lane F): the initial liquid from any scene description.
 * - CPU: the page table is the old engine's own t = 0 φ (fluidLatticeVertexPhi, in h) at every vertex of its pages,
 *   for an analytic volume (the torus) and a height field (the standing wave); the dam keeps its closed form; the
 *   table's boot bricks are the closed form's when the table holds the closed form's own values.
 * - GPU: the dam booted from that table is the dam booted from its boxes, by brick and page key: the same bricks and
 *   pages, flags word for word, φ to an f32 ulp of the boxes' extent (the boxes' distance is f32 on the GPU at that
 *   extent, the table's f64 rounded once).
 * - GPU: an initial velocity boots as the old engine sets it (uniformInitialLiquidVelocity): face a of every F cell c
 *   is the old texel c - e_a's, and every C face the old +face of tile T - e_a (before the first kick).
 * - GPU: the first seeds carry it (the falling torus, against itself at rest).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { fluidLatticeCellVolumes, fluidLatticeVertexPhi } from "../../core/initial-liquid-lattice";
import { uniformInitialLiquidVelocity } from "../uniform/uniform-volume-initial";
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { BrickEngine, bootBricks, bootPhiAt, initialPhiPages, type BrickEngineOptions, type BrickInitialPhi, type Vec3 } from "./brick-engine";
import { BRICK_ENGINE_SCENES, brickSceneSetup } from "./brick-scene";
import { bootParticleCount, brickInitialLiquid, brickInitialPhi, brickInitialVelocity, closedFormLiquid } from "./brick-initial";
import type { BrickStagesFactory } from "./brick-stages";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import { snapshot } from "./brick-readback";
import { BRICK_STRIDE, H } from "./brick-abi";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const METHOD = "uniform-narrow-band-flip";

function sceneOf(id: string) { return sceneDocument(getSceneDefinition(id)); }

/** The table's vertex φ against the old engine's whole-lattice φ (metres / h, in f32), every vertex of its pages. */
function tableOff(id: string): { n: number; off: number } {
  const scene = sceneOf(id), cells = sceneLatticeDimensions(scene) as unknown as Vec3, h = scene.container.width_m / cells[0];
  const t = brickInitialPhi(scene, cells, h), full = fluidLatticeVertexPhi(scene), P = initialPhiPages(cells);
  let n = 0, off = 0;
  t.pages.forEach((k, i) => {
    const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
    for (let l = 0; l < 32768; l++) {
      const v = [32 * p[0]! + (l & 31), 32 * p[1]! + ((l >> 5) & 31), 32 * p[2]! + (l >> 10)];
      if (v.some((x, a) => x > cells[a]!)) continue;
      n++;
      if (t.phi[32768 * i + l] !== Math.fround(full[v[0]! + (cells[0] + 1) * (v[1]! + (cells[1] + 1) * v[2]!)]! / h)) off++;
    }
  });
  return { n, off };
}

test("brick initial: the page table is the old engine's t = 0 φ at every vertex it holds", () => {
  for (const id of ["falling-water-torus", "nbflip-analytic-standing-wave"]) {
    const scene = sceneOf(id);
    assert.equal(closedFormLiquid(scene), false, `${id} is no closed form`);
    const r = tableOff(id);
    assert.ok(r.n > 0, `${id}: the table holds vertices`);
    assert.equal(r.off, 0, `${id}: ${r.off} of ${r.n} vertices differ from fluidLatticeVertexPhi`);
  }
});

test("brick initial: the dam keeps its closed form; a table of its own values boots the same bricks", () => {
  const scene = sceneOf("high-resolution-dam-break");
  assert.equal(closedFormLiquid(scene), true);
  const cells = sceneLatticeDimensions(scene) as unknown as Vec3, g = scene.fluid.gravity_m_s2;
  const o = { lattice: cells, h_m: scene.container.width_m / cells[0], dt_s: 1 / 60, gravity_mps2: [g.x, g.y, g.z] as Vec3 };
  const closed = brickInitialLiquid(scene, o);
  assert.equal(closed.initial, undefined);
  assert.equal(closed.liquid.length, 1);
  const table = tableOf(o.lattice, closed.liquid, closed.bricks);
  assert.deepEqual(bootBricks({ ...o, liquid: [], initial: table }), closed.bricks);
});

/** The closed form's own vertex φ (bootPhiAt) as a table over the boot bricks' pages, grown by one page. */
function tableOf(lattice: Vec3, liquid: BrickEngineOptions["liquid"], bricks: readonly number[]): BrickInitialPhi {
  const P = initialPhiPages(lattice), at = bootPhiAt({ lattice, liquid }), keys = new Set<number>();
  for (const k of bricks) {
    const p = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023].map(v => v >> 2);
    for (let j = 0; j < 27; j++) {
      const q = [p[0]! + (j % 3) - 1, p[1]! + (Math.floor(j / 3) % 3) - 1, p[2]! + Math.floor(j / 9) - 1];
      if (q.every((v, a) => v >= 0 && v < P[a]!)) keys.add(q[0]! + P[0] * (q[1]! + P[1] * q[2]!));
    }
  }
  const pages = [...keys].sort((a, b) => a - b), phi = new Float32Array(32768 * pages.length);
  pages.forEach((k, i) => {
    const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
    for (let l = 0; l < 32768; l++) phi[32768 * i + l] = at([32 * p[0]! + (l & 31), 32 * p[1]! + ((l >> 5) & 31), 32 * p[2]! + (l >> 10)]);
  });
  return { pages, phi };
}

test("brick initial (GPU): the dam booted from a table of its own φ is the dam booted from its boxes", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const { setup, gaps } = brickSceneSetup(scene, values);
  assert.ok(setup, gaps.join("; "));
  const base: BrickEngineOptions = {
    lattice: setup.lattice, h_m: setup.h_m, dt_s: setup.dt_s, gravity_mps2: setup.gravity_mps2, capacities: setup.capacities, liquid: setup.liquid,
    solver: compositeMGPCG(), stages: fluidStagesAt(true), values, particles: { bandDepth: setup.particleBandDepth },
    fluid: { density_kg_m3: setup.density_kg_m3, viscosity_Pa_s: setup.viscosity_Pa_s, surfaceTension_N_m: setup.surfaceTension_N_m }, label: "dam-boxes",
  };
  const table = tableOf(setup.lattice, setup.liquid, bootBricks(base));
  await withUniformDevice("brick initial table", async device => {
    // The boot's φ and flags by brick and page key (slots are claimed in arrival order, so the words are compared
    // through the directory), and the boot's live set. A step is not compared: the engine is not bitwise
    // reproducible run to run past its lists (brick-wide-dam.test.ts).
    const bootOf = async (o: BrickEngineOptions) => {
      const e = await BrickEngine.create(device, o);
      try {
        const s = await snapshot(e), phi = new Uint32Array(await e.read("phi")), L = e.layout, out = new Map<string, number[]>();
        for (const [key, slot] of s.bricks) for (const ch of ["PHI", "PHI1", "FLAGS"]) out.set(`F${key}.${ch}`, [...phi.subarray(L.base("phi", "F", ch) + BRICK_STRIDE * slot, L.base("phi", "F", ch) + BRICK_STRIDE * slot + 512)]);
        for (const [key, slot] of s.pages) for (const ch of ["PHI", "PHI1", "FLAGS"]) out.set(`C${key}.${ch}`, [...phi.subarray(L.base("phi", "C", ch) + BRICK_STRIDE * slot, L.base("phi", "C", ch) + BRICK_STRIDE * slot + 512)]);
        return out;
      } finally { e.destroy(); }
    };
    const a = await bootOf(base), b = await bootOf({ ...base, liquid: [], initial: table, label: "dam-table" });
    assert.deepEqual([...b.keys()].sort(), [...a.keys()].sort(), "the boot's bricks and pages");
    const f = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0]!;
    // The boxes' distance is f32 on the GPU (sdBox: |p - centre| - half, at the boxes' extent, run 2 max(n) cells on
    // through the walls), so it carries an ulp of the largest box coordinate; the table's is f64 rounded once.
    const extent = Math.max(...setup.liquid.flatMap(b => [...b.lo, ...b.hi].map(Math.abs))), ulp = 2 ** (Math.floor(Math.log2(extent)) - 23);
    let flags = 0, far = 0, worst = 0, worstAbs = 0, at = "";
    for (const [k, v] of a) {
      const w = b.get(k)!;
      for (let i = 0; i < v.length; i++) {
        if (k.endsWith("FLAGS")) { if (v[i] !== w[i]) flags++; continue; }
        const x = f(v[i]!), y = f(w[i]!), e = Math.abs(x - y) / ulp;
        if (!(e <= 1)) far++; if (e > worst) { worst = e; at = `${k}[${i}] boxes ${x} table ${y}`; }
        worstAbs = Math.max(worstAbs, Math.abs(x - y));
      }
    }
    assert.equal(flags, 0, `${flags} boot flag words differ`);
    console.log(`boot φ: worst ${worst} ulp of the extent ${extent} at ${at}; worst |boxes - table| ${worstAbs} cells`);
    assert.equal(far, 0, `${far} boot φ words differ by more than an ulp of the boxes' extent ${extent} (worst ${worst})`);
  });
});

/** The engine options for a catalog scene from brick-initial.ts directly (any scene, the brick scenes' gate aside),
 * pools from the boot's counts (3x, as lane F's smoke runs size them). */
function sceneOptions(id: string, v0?: { x: number; y: number; z: number }, stages: BrickStagesFactory = fluidStagesAt(true)): BrickEngineOptions {
  const method = getMethod(METHOD), definition = getSceneDefinition(id), scene = sceneDocument(definition);
  if (v0) scene.fluid.initialVelocity_m_s = v0;
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const lattice = sceneLatticeDimensions(scene) as unknown as Vec3, h_m = scene.container.width_m / lattice[0], g = scene.fluid.gravity_m_s2;
  const o = { lattice, h_m, dt_s: scene.numerics.fixedDt_s ?? 1 / 60, gravity_mps2: [g.x, g.y, g.z] as Vec3 }, band = scene.fluid.brickTiers?.particleBandDepth ?? 3;
  const start = brickInitialLiquid(scene, o), seeds = bootParticleCount({ lattice, liquid: start.liquid, initial: start.initial }, start.bricks, band);
  const fPages = new Set(start.bricks.map(k => [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023].map(v => v >> 2).join(","))).size;
  const p = scene.fluid.brickPools ?? { f: 3 * start.bricks.length + 64, c: start.pages.length + 3 * fPages + 8, b: 16, k: 0, particles: Math.max(3 * seeds, 65536) };
  return {
    ...o, capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles }, liquid: start.liquid, initial: start.initial,
    initialVelocity: brickInitialVelocity(scene, lattice), solver: compositeMGPCG(), stages, values, particles: { bandDepth: band },
    fluid: { density_kg_m3: scene.fluid.density_kg_m3, viscosity_Pa_s: scene.fluid.dynamicViscosity_Pa_s, surfaceTension_N_m: scene.fluid.surfaceTension_N_m }, label: id,
  };
}

test("brick initial (GPU): an initial velocity boots on the faces the old engine sets", GPU, async () => {
  // The boot's last stage is the first frame's projected half kick (BrickStages.kick), which changes U; bootPhi's
  // write is compared without it.
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const v0 = { x: 0.3, y: -0.2, z: 0.1 }, unkicked: BrickStagesFactory = c => ({ ...fluidStagesAt(true)(c), kick: undefined });
  const o = sceneOptions("high-resolution-dam-break", v0, unkicked), scene = sceneDocument(getSceneDefinition("high-resolution-dam-break"));
  assert.ok(o.initialVelocity && o.liquid.length === 1, "the dam: closed form, moving");
  const n = o.lattice, old = uniformInitialLiquidVelocity(v0, fluidLatticeCellVolumes(scene).volume, [n[0], n[1], n[2]])!;
  // Face a of cell c is the old texel c - e_a's +face (zero off the lattice: the wall).
  const want = (c: number[], a: number) => { const q = c.slice(); q[a]!--; return q.every((x, b) => x >= 0 && x < n[b]!) ? old[4 * (q[0]! + n[0] * (q[1]! + n[1] * q[2]!)) + a]! : 0; };
  const coord = (k: number) => [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023];
  await withUniformDevice("brick initial velocity", async device => {
    const e = await BrickEngine.create(device, o);
    try {
      const s = await snapshot(e), vel = new Float32Array((await e.read("vel")).buffer), L = e.layout;
      let faces = 0, moving = 0, off = 0; const ex: string[] = [];
      const check = (got: number, exp: number, what: string) => { faces++; if (exp !== 0) moving++; if (got !== exp) { off++; if (ex.length < 6) ex.push(`${what} ${got} vs ${exp}`); } };
      for (const [key, slot] of s.bricks) {
        const c0 = coord(key).map(x => 8 * x);
        for (let l = 0; l < 512; l++) {
          const c = [c0[0]! + (l & 7), c0[1]! + ((l >> 3) & 7), c0[2]! + (l >> 6)];
          for (let a = 0; a < 3; a++) check(vel[L.base("vel", "F", `U${a}`) + BRICK_STRIDE * slot + l]!, Math.fround(want(c, a)), `F ${c} ${a}`);
        }
      }
      for (const [key, slot] of s.pages) {
        const t0 = coord(key).map(x => 8 * x);
        for (let l = 0; l < 512; l++) {
          const t = [t0[0]! + (l & 7), t0[1]! + ((l >> 3) & 7), t0[2]! + (l >> 6)];
          // C face (T, a): the old +face of tile T - e_a, the +face of its cell 4T - e_a.
          for (let a = 0; a < 3; a++) check(vel[L.base("vel", "C", `U${a}`) + BRICK_STRIDE * slot + l]!, Math.fround(want(t.map(x => 4 * x), a)), `C ${t} ${a}`);
        }
      }
      assert.ok(moving > 0 && moving < faces, `${moving} of ${faces} faces move`);
      assert.equal(off, 0, `${off} of ${faces} boot faces are not the old engine's: ${ex.join("; ")}`);
    } finally { e.destroy(); }
  });
});

test("brick initial (GPU): the first seeds carry the initial velocity", GPU, async () => {
  // The torus falls in air, touching no wall: no pressure acts, so its liquid moves as one body, at g t at rest and
  // v0 + g t moving (a uniform v0 on its tiles is divergence free; the kick's projection and the extension are
  // linear). The boot seeds nothing (pSeed seeds heated tiles; a body at rest has none), so step 1's seeds are
  // compared: each run's seeds all move at its mean, and the means differ by v0.
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const v0 = { x: 0.3, y: -0.2, z: 0.1 }, id = "falling-water-torus";
  await withUniformDevice("brick initial seeds", async device => {
    const seedsOf = async (o: BrickEngineOptions) => {
      const e = await BrickEngine.create(device, o);
      try {
        e.advance(1); await e.fence();
        // After step k the records are bank k & 1's: particleStride floats, position xyzw (w > 0 live), then velocity.
        const head = await e.read("set", 0, 256), S = e.layout.particleStride, P = e.layout.cap.particles, parts = new Float32Array((await e.read("parts")).buffer), out: number[][] = [];
        assert.equal(head[H.FAIL], 0, `${o.label}: FAIL ${head[H.FAIL]} info 0x${head[H.FAIL_INFO]!.toString(16)}`);
        for (let i = 0; i < Math.min(head[H.PCOUNT1]!, P); i++) { const r = S * (P + i); if (parts[r + 3]! > 0) out.push([...parts.subarray(r + 4, r + 7)]); }
        return out;
      } finally { e.destroy(); }
    };
    const moving = sceneOptions(id, v0), rest = sceneOptions(id);
    assert.ok(moving.initial && moving.initialVelocity && !rest.initialVelocity, "the torus: a table, moving and at rest");
    const a = await seedsOf(moving), b = await seedsOf({ ...rest, label: `${id} at rest` });
    const mean = (us: number[][]) => [0, 1, 2].map(k => us.reduce((t, u) => t + u[k]!, 0) / us.length);
    const spread = (us: number[][], m: number[]) => Math.max(...us.map(u => Math.hypot(u[0]! - m[0]!, u[1]! - m[1]!, u[2]! - m[2]!)));
    assert.ok(a.length > 0 && b.length > 0, `step 1 seeds: ${a.length} moving, ${b.length} at rest`);
    const ma = mean(a), mb = mean(b), sa = spread(a, ma), sb = spread(b, mb), d = Math.hypot(ma[0]! - mb[0]! - v0.x, ma[1]! - mb[1]! - v0.y, ma[2]! - mb[2]! - v0.z);
    console.log(`step 1 seeds: ${a.length} moving at ${ma.map(v => v.toFixed(5))} (spread ${sa}), ${b.length} at rest at ${mb.map(v => v.toFixed(5))} (spread ${sb}); |moving - rest - v0| ${d}`);
    assert.ok(sa <= 1e-4 && sb <= 1e-4, `the seeds move as one body: spread ${sa} moving, ${sb} at rest`);
    assert.ok(d <= 1e-4, `the moving seeds carry v0 to ${d} m/s`);
  });
});

test("brick initial: a moving liquid in a container that is not whole pages boots on the container's lattice", () => {
  // The old engine's paged volumes cover its fluid box (the container's cells padded to whole pages past the high
  // faces); the initial velocity's tiles are the container's (E's catalog sweep, 2026-11-17: these three threw).
  const method = getMethod(METHOD);
  for (const id of ["gentle-moving-blob", "geometric-uniform-translation", "coarse-surface-translation"]) {
    const scene = sceneOf(id), lattice = sceneLatticeDimensions(scene) as unknown as Vec3;
    assert.ok(sceneFluidLatticeDimensions(scene).some((n, a) => n !== lattice[a]), `${id}: the fluid box is padded`);
    const v = brickInitialVelocity(scene, lattice)!, T = lattice.map(n => n >> 2);
    assert.ok(v && v.tiles.length > 0, `${id}: the liquid moves`);
    assert.ok(v.tiles.every(k => k >= 0 && k < T[0]! * T[1]! * T[2]!), `${id}: every tile is the container's`);
    const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
    // Past the catalog gate (as brickEngineGaps's sweep runs it): the setup reaches the initial liquid.
    const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(id);
    if (!had) enabled.push(id);
    try {
      const r = brickSceneSetup(scene, values);
      assert.ok(r.setup || r.gaps.length > 0, `${id}: a setup or gaps`);
      if (r.setup) assert.deepEqual(r.setup.initialVelocity, v, `${id}: the setup's initial velocity`);
    } finally { if (!had) enabled.splice(enabled.indexOf(id), 1); }
  }
});
