/**
 * brick-source.ts and brick-source.wgsl.ts (lane F, M4): scheduled drops.
 * - CPU: the letters' drops in source form (uniformScheduledDrop's, in cells), at the engine's step: one due a step;
 *   a first step's drop is in the boot's set; two drops in one step throw; the ring words; the box holds every liquid
 *   point of its drop.
 * - GPU (parity): nbDropDistance is liquidExtrusionDistance (Float64, over h) to f32 rounding, and nbSourcePhi is the
 *   union min(φ, drop) whatever the bounding ball's early-out.
 * - GPU (engine): a ball dropped into the dam's air on step 1: the step before claims its box, its step joins its
 *   volume to the budget, and its liquid is in φ after the step.
 * - The inflow (CPU): cm12-figure-1's nozzle in source form against createInflowGridBoundary, its step words
 *   (strength, sweep, continues, inflowNeedsFine), its flow joined to ADDED, and the plug box in the boot's set.
 * - The inflow (GPU parity): nbInflowPhi, nbInflowForced and nbInflowProjected against the old engine's own WGSL
 *   (uvSourcePhi's plug, applyInflowSweptVelocity, applyInflowVelocity: inflow-boundary.ts, uniform-source.wgsl.ts),
 *   compiled on the old params (metres) at the same cells, on its first step and a later one, at full and half strength.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { liquidExtrusionDistance, type LiquidExtrusion } from "../../core/liquid-extrusion";
import type { SceneDescription } from "../../core/model";
import { BrickEngine, bootBricks, type BrickEngineOptions, type Vec3 } from "./brick-engine";
import { brickBootSourceBoxes, brickDropDue, brickInflowStep, brickSourceRing, brickSources, SRC_RING, type BrickSources } from "./brick-source";
import { createInflowGridBoundary, inflowBoundaryWGSL } from "../../core/inflow-boundary";
import { uniformSourcePhiWGSL } from "../uniform/uniform-source.wgsl";
import { BRICK_SOURCE_WGSL } from "./brick-source.wgsl";
import { brickInitialLiquid, brickInitialVelocity } from "./brick-initial";
import { BRICK_ENGINE_SCENES, BRICK_SPLASH_DROP, brickDropParticles, brickInitialExtent, brickSceneSetup, brickSourcesExtent } from "./brick-scene";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import { snapshot } from "./brick-readback";
import { BRICK_STRIDE, H, brickKey } from "./brick-abi";
import { FH, VOL_UNIT } from "./velocity-kernels.wgsl";
import { parityGPU, prng, runParity, withParityDevice } from "./parity/harness";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const METHOD = "uniform-narrow-band-flip", LETTERS = "nbflip-figure-8-letters";
const sceneOf = (id: string) => sceneDocument(getSceneDefinition(id));

function lettersSources(): { scene: SceneDescription; s: BrickSources; lattice: Vec3; h: number; origin: Vec3 } {
  // The letters as the engine runs them: brickSceneSetup's step (the method's fixed step) and sources.
  const definition = getSceneDefinition(LETTERS), scene = sceneDocument(definition), method = getMethod(METHOD);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  (BRICK_ENGINE_SCENES as string[]).push(LETTERS);
  try {
    const { setup, gaps } = brickSceneSetup(scene, values);
    assert.deepEqual(gaps, [], "the letters run on the brick engine but for the scene gate");
    return { scene, s: setup!.sources!, lattice: setup!.lattice, h: setup!.h_m, origin: setup!.origin_m };
  } finally { (BRICK_ENGINE_SCENES as string[]).pop(); }
}

test("brick sources: the letters' drops in source form, one a step; the boot holds a first step's drop", () => {
  const { scene, s } = lettersSources(), dt = s.dt_s;
  const schedule = scene.fluid.scheduledDrops!;
  assert.equal(s.drops.length, schedule.length);
  assert.ok(s.drops.every(d => d.kind === 3 && d.edgeCount > 0 && d.volume > 0), "every letter is an extrusion with edges and volume");
  assert.equal(s.edges.length, 4 * s.drops.reduce((n, d) => n + d.edgeCount, 0));
  // Each drop is due in exactly one step of the 240, and step 0 holds the first.
  const due = new Map<number, number>();
  let t = 0;
  for (let step = 0; step < 240; step++, t += dt) { const i = brickDropDue(s, t, t + dt); if (i >= 0) { assert.ok(!due.has(i), `drop ${i} due twice`); due.set(i, step); } }
  assert.equal(due.size, s.drops.length, "every drop lands within 240 steps");
  // The ring: this step's drop and the next step's box.
  const buf = new ArrayBuffer(256), u = new Uint32Array(buf), f = new Float32Array(buf), i32 = new Int32Array(buf);
  brickSourceRing(s, due.get(0)! * dt, u, f);
  assert.equal(u[SRC_RING.KIND], 3); assert.equal(f[SRC_RING.ADDED], Math.fround(s.drops[0]!.volume));
  assert.equal(u[SRC_RING.BOXES] & 1, 1);
  assert.deepEqual([...i32.subarray(SRC_RING.BOX, SRC_RING.BOX + 6)], [...s.drops[0]!.box[0], ...s.drops[0]!.box[1]]);
  const before = due.get(1)! - 1;
  brickSourceRing(s, before * dt, u, f);
  assert.equal(u[SRC_RING.KIND], 0, "the step before a drop adds nothing"); assert.equal(f[SRC_RING.ADDED], 0);
  assert.equal(u[SRC_RING.BOXES], 2, "the step before a drop claims its box ahead");
  assert.deepEqual([...i32.subarray(SRC_RING.AHEAD, SRC_RING.AHEAD + 6)], [...s.drops[1]!.box[0], ...s.drops[1]!.box[1]]);
  // A drop due in the first step: the boot's set holds its bricks (no step before it claims them).
  const lattice = sceneLatticeDimensions(scene) as unknown as Vec3, g = scene.fluid.gravity_m_s2, h_m = scene.container.width_m / lattice[0];
  const start = brickInitialLiquid(scene, { lattice, h_m, dt_s: dt, gravity_mps2: [g.x, g.y, g.z] });
  const first: BrickSources = { ...s, drops: [{ ...s.drops[0]!, time_s: 0.5 * dt }] };
  const set = new Set(bootBricks({ lattice, liquid: start.liquid, initial: start.initial, dt_s: dt, h_m, gravity_mps2: [g.x, g.y, g.z], sources: first }));
  const without = new Set(bootBricks({ lattice, liquid: start.liquid, initial: start.initial, dt_s: dt, h_m, gravity_mps2: [g.x, g.y, g.z], sources: s }));
  assert.ok(set.size > without.size, "the first step's drop adds bricks to the boot");
  const [lo, hi] = s.drops[0]!.box;
  for (let z = lo[2] >> 3; z <= (hi[2] - 1) >> 3; z++) for (let y = lo[1] >> 3; y <= (hi[1] - 1) >> 3; y++) for (let x = lo[0] >> 3; x <= (hi[0] - 1) >> 3; x++) assert.ok(set.has(brickKey(x, y, z)), `boot brick ${x},${y},${z}`);
});

test("brick sources: the pool rule covers the scheduled drops (the letters without their authored brickPools)", () => {
  const { scene, s, lattice } = lettersSources();
  const definition = getSceneDefinition(LETTERS), method = getMethod(METHOD), profile = definition.methodProfile;
  const values = resolveMethodValues(method, profile?.methodId === METHOD ? profile.quality : "balanced", { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const { brickPools: _, ...fluid } = scene.fluid, ruled = { ...scene, fluid };
  (BRICK_ENGINE_SCENES as string[]).push(LETTERS);
  try {
    const { setup, gaps } = brickSceneSetup(ruled, values);
    assert.deepEqual(gaps, [], "a drop scene without authored pools has no gap");
    // The extent: the drops' volume joins the t = 0 liquid's; the letters fall from over two bricks, so the splash is
    // theirs (the drops' volume), not the pool's.
    const h_m = setup!.h_m, g = scene.fluid.gravity_m_s2;
    const start = brickInitialLiquid(scene, { lattice, h_m, dt_s: s.dt_s, gravity_mps2: [g.x, g.y, g.z] });
    const e0 = brickInitialExtent({ lattice, liquid: start.liquid, initial: start.initial }), e = brickSourcesExtent(e0, s, lattice);
    const dropped = s.drops.reduce((n, d) => n + d.volume, 0);
    assert.ok(Math.abs(e.cells - e0.cells - dropped) < 1e-6 * e.cells, "the drops' cells join the liquid");
    assert.ok(e.drop >= BRICK_SPLASH_DROP && e0.drop < BRICK_SPLASH_DROP, "the letters splash, the pool does not");
    assert.ok(Math.abs(e.splash! - dropped) < 1e-6 * dropped, "the splash is the drops' liquid");
    assert.ok(setup!.capacities.particles >= Math.ceil(10 * dropped), "the particle pool holds the splash");
    assert.ok(brickDropParticles(s, 2) > 0 && brickDropParticles(undefined, 2) === 0);
  } finally { (BRICK_ENGINE_SCENES as string[]).pop(); }
});

test("brick sources: two drops in one step throw; a drop at t = 0 or an unknown shape is refused", () => {
  const { scene, s: letters } = lettersSources(), lattice = sceneLatticeDimensions(scene) as unknown as Vec3, h = scene.container.width_m / lattice[0];
  const dt = letters.dt_s, o = { lattice, h_m: h, origin_m: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m] as Vec3, dt_s: dt };
  const ball = { shape: "sphere" as const, center_m: { x: 0, y: 0.5 * scene.container.height_m, z: 0 }, radius_m: 4 * h };
  const two = { ...scene, fluid: { ...scene.fluid, scheduledDrops: [{ time_s: 0.5 * dt, volume: ball }, { time_s: 0.75 * dt, volume: ball }] } };
  const s = brickSources(two, o)!;
  assert.throws(() => brickDropDue(s, 0, dt), /2 scheduled drops fall in the step/);
  assert.throws(() => brickSources({ ...scene, fluid: { ...scene.fluid, scheduledDrops: [{ time_s: 0, volume: ball }] } }, o), /initial liquid/);
  assert.throws(() => brickSources({ ...scene, fluid: { ...scene.fluid, scheduledDrops: [{ time_s: dt, volume: { shape: "cube" } as never }] } }, o), /not supported/);
});

test("brick sources: a drop's box holds every point of its liquid", () => {
  const { scene, s, h, origin } = lettersSources(), rand = prng(7);
  scene.fluid.scheduledDrops!.forEach(({ volume }, i) => {
    const d = s.drops[i]!, [lo, hi] = d.box;
    for (let n = 0; n < 4000; n++) {
      // Points about the bounding ball; each liquid one lies in the box.
      const p = d.centre.map(c => c + (2 * rand() - 1) * d.radius) as unknown as Vec3;
      const world = { x: origin[0] + h * p[0], y: origin[1] + h * p[1], z: origin[2] + h * p[2] };
      if (liquidExtrusionDistance(volume as LiquidExtrusion, world) < 0) assert.ok(p.every((x, a) => x >= lo[a]! + 4 && x <= hi[a]! - 4), `drop ${i}: liquid point ${p} outside its box`);
    }
  });
});

test("brick sources (GPU parity): nbDropDistance is the extrusion's distance over h, nbSourcePhi its union", { skip: !parityGPU }, async () => {
  const { scene, s, h, origin } = lettersSources(), rand = prng(11), N = 4096;
  const prelude = /* wgsl */`
const NB_LATTICE:vec3i=vec3i(4096);
var<private> cfg:vec4u;
fn nbRing(slot:u32,k:u32)->u32{return aux[k];}
fn nbRingF(slot:u32,k:u32)->f32{return bitcast<f32>(aux[k]);}
@group(0) @binding(3) var<storage,read> src:array<f32>;
` + BRICK_SOURCE_WGSL;
  const body = /* wgsl */`let p=v3(4u*t);let value=f(4u*t+3u);put(2u*t,nbDropDistance(p,3.0e38));put(2u*t+1u,nbSourcePhi(p,value));`;
  await withParityDevice(async device => {
    let worst = 0, unionOff = 0, liquid = 0;
    for (const [i, d] of s.drops.entries()) {
      const ring = new ArrayBuffer(256), u = new Uint32Array(ring), f = new Float32Array(ring);
      brickSourceRing(s, d.time_s - 0.5 * s.dt_s, u, f);
      assert.equal(u[SRC_RING.KIND], 3, `drop ${i} is this step's`);
      const inp = new Float32Array(4 * N), want: number[] = [];
      for (let n = 0; n < N; n++) {
        const p = d.centre.map(c => Math.fround(c + 1.5 * (2 * rand() - 1) * d.radius));
        inp.set([...p, Math.fround(8 * (2 * rand() - 1))], 4 * n);
        want.push(liquidExtrusionDistance(scene.fluid.scheduledDrops![i]!.volume as LiquidExtrusion, { x: origin[0] + h * p[0]!, y: origin[1] + h * p[1]!, z: origin[2] + h * p[2]! }) / h);
      }
      const out = new Float32Array((await runParity(device, { prelude, body }, { inp, aux: u, extra: { 3: s.edges }, threads: N, outWords: 2 * N })).buffer);
      for (let n = 0; n < N; n++) {
        const dist = out[2 * n]!, w = want[n]!, value = inp[4 * n + 3]!;
        if (w < 0) liquid++;
        worst = Math.max(worst, Math.abs(dist - w));
        // The union: the ball's early-out returns a lower bound past the value, so the min is the value's there.
        if (Math.abs(out[2 * n + 1]! - Math.min(value, w)) > 1e-3) unionOff++;
      }
    }
    console.log(`letters: ${s.drops.length} x ${N} points (${liquid} liquid), |GPU - f64| ≤ ${worst} cells, union off at ${unionOff}`);
    assert.ok(liquid > 0, "some points are liquid");
    assert.ok(worst <= 1e-3, `nbDropDistance is the f64 distance to ${worst} cells`);
    assert.equal(unionOff, 0);
  });
});

/** The dam's engine options with a ball dropped into its air on step 1 (due in (dt, 2 dt]). */
function damWithDrop(): { o: BrickEngineOptions; s: BrickSources; radius: number } {
  const id = "high-resolution-dam-break", method = getMethod(METHOD), definition = getSceneDefinition(id), scene = sceneDocument(definition);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const lattice = sceneLatticeDimensions(scene) as unknown as Vec3, h_m = scene.container.width_m / lattice[0], g = scene.fluid.gravity_m_s2;
  const dt_s = scene.numerics.fixedDt_s ?? 1 / 60, radius = 6;
  const c = scene.container, origin_m: Vec3 = [-0.5 * c.width_m, 0, -0.5 * c.depth_m];
  // The far side's air, half way up: clear of the dam's liquid and the walls.
  const ball = { shape: "sphere" as const, center_m: { x: 0.3 * c.width_m, y: 0.6 * c.height_m, z: 0 }, radius_m: radius * h_m };
  const s = brickSources({ ...scene, fluid: { ...scene.fluid, scheduledDrops: [{ time_s: 1.5 * dt_s, volume: ball }] } }, { lattice, h_m, origin_m, dt_s })!;
  const start = brickInitialLiquid(scene, { lattice, h_m, dt_s, gravity_mps2: [g.x, g.y, g.z] }), p = scene.fluid.brickPools!;
  return {
    s, radius,
    o: {
      lattice, h_m, dt_s, gravity_mps2: [g.x, g.y, g.z], capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles }, liquid: start.liquid, initial: start.initial,
      initialVelocity: brickInitialVelocity(scene, lattice), solver: compositeMGPCG(), stages: fluidStagesAt(true), values, particles: { bandDepth: scene.fluid.brickTiers?.particleBandDepth ?? 3 },
      fluid: { density_kg_m3: scene.fluid.density_kg_m3, viscosity_Pa_s: scene.fluid.dynamicViscosity_Pa_s, surfaceTension_N_m: scene.fluid.surfaceTension_N_m }, label: `${id} + a drop`, sources: s,
    },
  };
}

test("brick sources (GPU): a ball dropped on step 1 is claimed the step before, joins the budget and lands in φ", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { o, s, radius } = damWithDrop(), d = s.drops[0]!, centre = d.centre.map(Math.round);
  const budget = (w: Uint32Array) => w[FH.VOL_BUDGET + 1]! * (2 ** 32 / VOL_UNIT) + w[FH.VOL_BUDGET]! / VOL_UNIT;
  await withUniformDevice("brick sources drop", async device => {
    const e = await BrickEngine.create(device, o);
    try {
      const phiAt = async (v: number[]) => {
        const snap = await snapshot(e), key = brickKey(v[0]! >> 3, v[1]! >> 3, v[2]! >> 3), slot = snap.bricks.get(key);
        if (slot === undefined) return undefined;
        const phi = new Float32Array((await e.read("phi")).buffer), l = (v[0]! & 7) + 8 * ((v[1]! & 7) + 8 * (v[2]! & 7));
        return phi[e.layout.base("phi", "F", "PHI") + BRICK_STRIDE * slot + l]!;
      };
      const b0 = budget(await e.read("set", 0, 256));
      e.advance(1); await e.fence();
      let head = await e.read("set", 0, 256);
      assert.equal(head[H.FAIL], 0, `step 0: FAIL ${head[H.FAIL]} info 0x${head[H.FAIL_INFO]!.toString(16)}`);
      assert.ok(Math.abs(budget(head) - b0) < 1e-6 * b0, "no drop in step 0: the budget holds");
      // The step before the drop claims its box: every brick of it resident.
      const snap = await snapshot(e), [lo, hi] = d.box;
      let missing = 0;
      for (let z = lo[2] >> 3; z <= (hi[2] - 1) >> 3; z++) for (let y = lo[1] >> 3; y <= (hi[1] - 1) >> 3; y++) for (let x = lo[0] >> 3; x <= (hi[0] - 1) >> 3; x++) if (!snap.bricks.has(brickKey(x, y, z))) missing++;
      assert.equal(missing, 0, `${missing} bricks of the drop's box not resident after step 0`);
      const before = await phiAt(centre);
      assert.ok(before !== undefined && before > 0, `the drop's centre is air before: φ ${before}`);
      e.advance(1); await e.fence();
      head = await e.read("set", 0, 256);
      assert.equal(head[H.FAIL], 0, `step 1: FAIL ${head[H.FAIL]} info 0x${head[H.FAIL_INFO]!.toString(16)}`);
      const grown = budget(head) - b0;
      console.log(`drop: ${d.volume.toFixed(2)} cells asked, budget grew ${grown.toFixed(2)}; φ at its centre ${before} → ${await phiAt(centre)}`);
      assert.ok(Math.abs(grown - d.volume) <= 1e-3 * d.volume + 1e-4, `the budget grows by the drop: ${grown} of ${d.volume}`);
      const after = await phiAt(centre);
      // The ball falls g dt^2 / h of a cell in its step: its centre stays deep.
      assert.ok(after !== undefined && after < -0.5 * radius, `the drop's centre is liquid after its step: φ ${after}`);
      e.advance(2); await e.fence();
      head = await e.read("set", 0, 256);
      assert.equal(head[H.FAIL], 0, `step 3: FAIL ${head[H.FAIL]} info 0x${head[H.FAIL_INFO]!.toString(16)}`);
      const filled = new Float32Array(head.buffer)[FH.VOL_FILLED]!;
      console.log(`step 3: filled ${filled.toFixed(1)} of budget ${budget(head).toFixed(1)} cells`);
      assert.ok(Math.abs(filled - budget(head)) <= 0.01 * budget(head), `the liquid holds its budget: ${filled} of ${budget(head)}`);
    } finally { e.destroy(); }
  });
});

/** cm12-figure-1 (a CFL-25 jet along +x) as the engine runs it: brickSceneSetup's step and sources. */
function jetSources(): { scene: SceneDescription; s: BrickSources; lattice: Vec3; h: number; origin: Vec3 } {
  const id = "cm12-figure-1", definition = getSceneDefinition(id), scene = sceneDocument(definition), method = getMethod(METHOD);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const listed = BRICK_ENGINE_SCENES.includes(id);
  if (!listed) (BRICK_ENGINE_SCENES as string[]).push(id);
  try {
    const { setup, gaps } = brickSceneSetup(scene, values);
    assert.deepEqual(gaps.filter(g => !g.includes("is not enabled")), [], "the jet runs on the brick engine");
    return { scene, s: setup!.sources!, lattice: setup!.lattice, h: setup!.h_m, origin: setup!.origin_m };
  } finally { if (!listed) (BRICK_ENGINE_SCENES as string[]).pop(); }
}

test("brick sources: the nozzle in source form, its step words, its flow in ADDED and its plug box in the boot", () => {
  const { scene, s, lattice, h } = jetSources(), n = s.inflow!, inflow = scene.fluid.inflow!, dt = s.dt_s;
  const grid = createInflowGridBoundary(inflow, scene.container, lattice);
  assert.equal(n.axis, grid.axis);
  assert.equal(n.face, grid.faceIndex, "the face plane, held to the lattice");
  assert.equal(n.scale, grid.apertureScale);
  assert.ok(Math.abs(n.flow - grid.flowRate_m3_s / h ** 3) <= 1e-9 * n.flow);
  // Step 0: the nozzle runs from t = 0 with no ramp: on, not continuing (its first step), the sweep |u| dt / h.
  const w0 = brickInflowStep(n, 0, dt), w1 = brickInflowStep(n, dt, dt);
  assert.equal(w0.flags & 3, 1, "the first step is on and not continuing");
  assert.equal(w1.flags & 3, 3, "a later step continues");
  assert.ok(Math.abs(w1.sweep - Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) * dt / h) < 1e-9 * w1.sweep);
  console.log(`jet: axis ${n.axis} face ${n.face} scale ${n.scale.toFixed(4)} radius ${n.radius.toFixed(2)} cells, sweep ${w1.sweep.toFixed(2)} cells, ${w1.added.toFixed(1)} cells a step, fine ${(w1.flags >> 2) & 1}, box ${n.box.map(b => b.join(",")).join(" .. ")}`);
  // The ring: the flow joins ADDED, the plug box is box 2.
  const ring = new ArrayBuffer(256), u = new Uint32Array(ring), f = new Float32Array(ring), i32 = new Int32Array(ring);
  brickSourceRing(s, dt, u, f);
  assert.ok(Math.abs(f[SRC_RING.ADDED]! - w1.added) <= 1e-6 * w1.added);
  assert.equal(u[SRC_RING.BOXES]! & 4, 4);
  assert.deepEqual([...i32.subarray(SRC_RING.INFLOW_BOX, SRC_RING.INFLOW_BOX + 6)], [...n.box[0], ...n.box[1]]);
  // The plug box holds the outlet and the plug's reach, and the boot holds it.
  for (let a = 0; a < 3; a++) assert.ok(n.box[0][a]! <= Math.floor(n.outlet[a]! - n.radius) && n.box[1][a]! >= Math.min(lattice[a]!, Math.ceil(n.outlet[a]! + n.radius)), `axis ${a}`);
  assert.deepEqual(brickBootSourceBoxes(s, dt), [n.box]);
  // Off: past the schedule, or the tap shut.
  assert.equal(brickInflowStep(n, inflow.end_s + dt, dt).flags & 1, 0);
  assert.equal(brickInflowStep({ ...n, inflow: { ...inflow, enabled: false } }, dt, dt).added, 0);
});

test("brick sources (GPU parity): the nozzle's φ, forced and projected velocity are the old engine's WGSL", { skip: !parityGPU }, async () => {
  const { s, lattice, h, origin } = jetSources(), n = s.inflow!, rand = prng(23), N = 8192, dt = s.dt_s;
  // The old engine's own functions on its params (writeParams' words, metres): the frame at the lattice's origin, h
  // cubic, the box's low cell zero; no drop (drop.w = 0).
  const old = /* wgsl */`
struct OldParams{dimsDt:vec4f,cellGravity:vec4f,frame:vec4f,physical:vec4f,boundary:vec4f,inflowPositionRadius:vec4f,inflowVelocityLength:vec4f,inflowTiming:vec4f,tuning:vec4f,drop:vec4f}
@group(0) @binding(4) var<storage,read> params:OldParams;
fn inflowGridDims()->vec3i{return vec3i(params.dimsDt.xyz);}
fn traceWorld(p:vec3f)->vec3f{return params.frame.xyz+p*params.cellGravity.xyz;}
fn dropDistance(p:vec3f,ceiling:f32)->f32{return 3.0e38;}
${inflowBoundaryWGSL}
${uniformSourcePhiWGSL}`;
  const prelude = /* wgsl */`
const NB_LATTICE:vec3i=vec3i(${lattice.join(",")});
var<private> cfg:vec4u;
fn nbRing(slot:u32,k:u32)->u32{return aux[k];}
fn nbRingF(slot:u32,k:u32)->f32{return bitcast<f32>(aux[k]);}
@group(0) @binding(3) var<storage,read> src:array<f32>;
` + BRICK_SOURCE_WGSL + old;
  // Per point: q (cells, integer), the value and axis a. Outputs: brick φ, old φ (over h), brick forced, old forced,
  // brick projected, old projected.
  const body = /* wgsl */`let q=vec3i(v3(4u*t));let p=v3(4u*t);let value=f(4u*t+3u);let a=u32(t%3u);
put(6u*t,nbSourcePhi(p,value));put(6u*t+1u,uvSourcePhi(p,value*params.cellGravity.x)/params.cellGravity.x);
let x=0.25*value;var w=vec3f(x);
put(6u*t+2u,nbInflowForced(q,a,x));put(6u*t+3u,applyInflowSweptVelocity(q,w)[a]);
put(6u*t+4u,nbInflowProjected(q,a,x));put(6u*t+5u,applyInflowVelocity(q,w)[a]);`;
  await withParityDevice(async device => {
    let worstAt = "", onCap = 0;
    let worstPhi = 0, forcedOff = 0, projectedOff = 0, swept = 0, projected = 0;
    for (const [start, strength] of [[0, 1], [dt, 1], [dt, 0.5]] as const) {
      const ring = new ArrayBuffer(256), u = new Uint32Array(ring), f = new Float32Array(ring);
      brickSourceRing(s, start, u, f);
      if (strength !== 1) { f[SRC_RING.STRENGTH] = strength; f[SRC_RING.SWEEP] *= strength; }
      const P = new Float32Array(40);
      P.set([lattice[0], lattice[1], lattice[2], dt, h, h, h, 0, origin[0], origin[1], origin[2], 0, 0, 0, 0, (u[SRC_RING.INFLOW]! & 2) ? 1 : 0], 0);
      P.set([origin[0] + h * n.outlet[0], origin[1] + h * n.outlet[1], origin[2] + h * n.outlet[2], h * n.radius, ...n.velocity_mps, n.scale, f[SRC_RING.STRENGTH]!, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20);
      const inp = new Float32Array(4 * N);
      const [lo, hi] = n.box;
      for (let k = 0; k < N; k++) {
        const c = lo.map((l, a) => Math.floor(l - 3 + (hi[a]! - l + 6) * rand()));
        inp.set([...c, Math.fround(16 * (2 * rand() - 1))], 4 * k);
      }
      const out = new Float32Array((await runParity(device, { prelude, body }, { inp, aux: u, extra: { 3: new Float32Array(4), 4: P }, threads: N, outWords: 6 * N })).buffer);
      for (let k = 0; k < N; k++) {
        const o = out.subarray(6 * k, 6 * k + 6), x = 0.25 * inp[4 * k + 3]!;
        // The plug's φ steps at its back cap on a first step (inside, max(side, front); behind, max(side, cap)): a
        // sample on that plane (the outlet's, axial 0) reads either side by the frames' rounding (metres or cells).
        const axial = [0, 1, 2].reduce((acc, a) => acc + (inp[4 * k + a]! - n.outlet[a]!) * n.velocity_mps[a]!, 0) / Math.hypot(...n.velocity_mps);
        if (Math.abs(axial) < 1e-3) onCap++;
        else if (Math.abs(o[0]! - o[1]!) > worstPhi) { worstPhi = Math.abs(o[0]! - o[1]!); worstAt = `start ${start} strength ${strength} p ${[...inp.subarray(4 * k, 4 * k + 3)]} value ${inp[4 * k + 3]} brick ${o[0]} old ${o[1]}`; }
        if (Math.abs(o[2]! - o[3]!) > 1e-5 * Math.max(1, Math.abs(o[3]!))) forcedOff++;
        if (Math.abs(o[4]! - o[5]!) > 1e-5 * Math.max(1, Math.abs(o[5]!))) projectedOff++;
        if (o[3] !== Math.fround(x)) swept++;
        if (o[5] !== Math.fround(x)) projected++;
      }
    }
    console.log(`jet: 3 x ${N} cells, |φ brick - old| ≤ ${worstPhi} cells; forced off ${forcedOff} (${swept} overridden), projected off ${projectedOff} (${projected} overridden); ${onCap} on the back cap's plane left out; worst φ at ${worstAt}`);
    assert.ok(swept > 100 && projected > 100, "the samples reach the plug and the aperture");
    assert.ok(worstPhi <= 1e-3, `nbInflowPhi is uvSourcePhi's plug to ${worstPhi} cells`);
    assert.equal(forcedOff, 0);
    assert.equal(projectedOff, 0);
  });
});
