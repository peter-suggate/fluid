/**
 * The brick engine's M1 regressions (lane E, brick-abi.md §5 to §7).
 *
 * CPU:
 * - the schedule's direct dispatches are sized from capacities, never from the lattice;
 * - the layout grows only by the directory's page window with a 4x-wider lattice;
 * - the reach rule, the pressure slot validation, the engine switch and brickPools.
 *
 * GPU (WEBGPU_NODE_MODULE):
 * - every kernel compiles cold in under 30 s;
 * - enter and leave match the CPU oracle, with the directory, records, child
 *   counts, free stack and lists consistent;
 * - FAIL.POOL_F latches when the reach needs more bricks than the pool;
 * - a static box is hydrostatic: X matches a Float64 solve of the same system
 *   and the analytic p = rho |g| depth, and the projected velocity is ~0;
 * - a 4x-wider box holds the same bricks and list counts.
 *
 * The pressure solve here is the M1 bring-up CG (brick-bringup-cg.ts): these
 * tests check the engine, the slot contract and the operator, not P's solver.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_STRIDE, BrickLayout, F_CLASS, FAIL, H, LIST, NONE, PHI_AIR, REC, ARGS, brickKey, brickKeyCoord, brickLocal, neighbourIndex } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions, type BrickLiquidBox, type Vec3, bootBricks, boxDistance, reachSet } from "./brick-engine";
import { CG_HEADER, bringUpCG } from "./brick-bringup-cg";
import { bringUpStages } from "./brick-stages";
import { PRESSURE_ARGS, validatePressureSolver } from "./brick-pressure-slot";
import { bindGroupLayoutEntries, MAX_STORAGE_BINDINGS } from "./brick-kernel";
import { brickSetKernels } from "./brick-set.wgsl";
import { brickPressureKernels } from "./brick-pressure.wgsl";
import { brickEngineGaps, brickEngineLanes, brickEngineRefusal, createBrickSolver, nbEngine } from "./brick-method";
import { brickSceneSetup, type BrickSceneSetup } from "./brick-scene";
import { BrickNBFlipSolver } from "./brick-solver";
import { brickField, brickSetOracle, pressureOracle } from "./brick-oracle";
import { keyOf, readChannel, snapshot, type BrickSnapshot } from "./brick-readback";
import { validateScene, type SceneDescription } from "../../core/model";
import { createReservoirSeepScene } from "../../core/reservoir-seep-scene";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { damBreakSignedDistanceAtNode } from "../../core/initial-fluid";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const H_M = 1 / 32, DT = 1 / 60, G = 9.81;
const asF32 = (u: Uint32Array) => new Float32Array(u.buffer, u.byteOffset, u.length);

function options(lattice: Vec3, liquid: BrickLiquidBox[], f: number, c: number, label?: string): BrickEngineOptions {
  return { lattice, h_m: H_M, dt_s: DT, gravity_mps2: [0, -G, 0], capacities: { f, c, b: 0, k: 0, particles: 1024 }, liquid, solver: L => bringUpCG(L), stages: bringUpStages, label };
}

const sheet = (hiX: number, hiY: number, hiZ: number): BrickLiquidBox => ({ lo: [-1, -1, -1], hi: [hiX, hiY, hiZ] });

/** The direct dispatches and indirect words of a schedule, in order. */
const dispatches = (L: BrickLayout) => {
  const s = BrickEngine.scheduleFor(L, bringUpCG(L), bringUpStages({ layout: L, dt_s: DT, h_m: H_M, gravity_mps2: [0, -G, 0], values: {} }), 1);
  return [...s.boot, ...s.part1, ...s.part2].map(p => `${p.kernel.label}:${JSON.stringify(p.dispatch)}:${p.z ?? 0}`);
};

test("brick engine: direct dispatches and buffers follow capacities, not the lattice (the 4x-wider box)", () => {
  const caps = { f: 300, c: 16, b: 0, k: 0, particles: 4096 };
  const A = new BrickLayout({ lattice: [64, 32, 64], ...caps }), B = new BrickLayout({ lattice: [256, 32, 64], ...caps });
  assert.deepEqual(dispatches(B), dispatches(A));
  assert.equal(B.set.words, A.set.words);
  assert.deepEqual(B.poolWords, A.poolWords);
  assert.equal(B.partsWords, A.partsWords);
  const pages = (L: BrickLayout) => L.window[0] * L.window[1] * L.window[2];
  // Only the page window (one word a 32^3 page) may grow.
  assert.ok(B.dir.words - A.dir.words <= pages(B) - pages(A) + 64, `dir grows ${B.dir.words - A.dir.words} words for ${pages(B) - pages(A)} pages`);
  assert.ok(B.bytes - A.bytes <= 4 * (pages(B) - pages(A) + 64));
});

test("brick engine: every kernel binds at most 8 storage buffers, and the layout check refuses a ninth", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
  const kernels = [...Object.values(brickSetKernels(L)), ...Object.values(brickPressureKernels(L)),
    ...[...cg.setup, ...cg.slot(0), ...cg.finish].map(p => p.kernel)];
  for (const k of kernels) assert.ok(k.bindings.filter(b => b.kind !== "uniform").length <= MAX_STORAGE_BINDINGS, k.label);
  const nine = Array.from({ length: 9 }, (_, i) => ({ name: `b${i}`, kind: "rw" as const }));
  assert.throws(() => bindGroupLayoutEntries(nine), /9 storage buffers/);
});

test("brick engine: the reach rule keeps a one-brick ring at rest and widens with the velocity bound", () => {
  const lattice: Vec3 = [64, 64, 64], one = [[2, 2, 2] as Vec3];
  const at = (bound: number) => reachSet(lattice, one, () => ({ lo: [3, 3, 3], hi: [4, 4, 4] }), () => bound, () => false, DT, H_M, [0, -G, 0]);
  assert.equal(at(0).size, 27);
  for (const key of at(0)) assert.ok(brickKeyCoord(key).every(v => v >= 1 && v <= 3));
  // 3 m/s for 1/60 s is 1.6 cells, still inside 7 + 1.6 < 8 + 3: no wider.
  assert.equal(at(3).size, 27);
  // 20 m/s: 10.7 + 7 cells of reach pass the next brick boundary (8 + 3 below, 8 + 4 above).
  assert.equal(at(20).size, 125);
  // A non-empty bin takes its whole brick: [16 - 7.1, 23 + 7.1] still lies in bricks 1..3.
  assert.equal(reachSet(lattice, one, () => undefined, () => 0, () => true, DT, H_M, [0, -G, 0]).size, 27);
  // Clipped to the lattice's bricks.
  assert.equal(reachSet(lattice, [[0, 0, 0]], () => ({ lo: [0, 0, 0], hi: [1, 1, 1] }), () => 0, () => false, DT, H_M, [0, -G, 0]).size, 8);
});

test("brick engine: the boot bricks are the reach set of the boxes' liquid vertices", () => {
  const o = options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8);
  const keys = bootBricks(o);
  // Liquid in brick rows y 0..1 (vertices 0..10), a one-brick ring above: 5 x 3 x 5.
  assert.equal(keys.length, 75);
  assert.ok(keys.every(k => brickKeyCoord(k)[1] <= 2));
  assert.equal(boxDistance([16, 10, 16], sheet(33, 10.3, 33)) < 0, true);
  assert.equal(boxDistance([16, 11, 16], sheet(33, 10.3, 33)) > 0, true);
});

test("brick engine: a pressure solver's indirect words must be the lists' or P's", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
  validatePressureSolver(cg);
  assert.equal(cg.bringUp, true);
  const kernel = cg.setup[0]!.kernel;
  assert.throws(() => validatePressureSolver({ ...cg, setup: [{ kernel, dispatch: { indirect: ARGS.LANE_F } }] }), /args word 2048, outside/);
  validatePressureSolver({ ...cg, setup: [{ kernel, dispatch: { indirect: PRESSURE_ARGS.FIRST } }] });
});

test("brick engine: the engine switch refuses unknown engines, and every scene until F and P register", async () => {
  assert.equal(nbEngine(undefined), "uniform");
  assert.equal(nbEngine("brick"), "brick");
  assert.throws(() => nbEngine("bricks"), /unknown engine "bricks"/);
  const seep = createReservoirSeepScene();
  assert.ok(seep.fluid?.brickPools, "seep authors brickPools");
  const refusal = brickEngineRefusal(seep, { timeStep: "sixtieth" });
  assert.match(refusal.message, /not enabled on the brick engine/);
  assert.match(refusal.message, /lane F's stages/);
  assert.match(refusal.message, /lane P's pressure solver/);
  assert.match(refusal.message, /engine "uniform"/);
  assert.ok(!brickEngineGaps(seep).some(g => /brickPools/.test(g)));
  // The dam: no scene gap, only the lanes'.
  const dam = sceneDocument(getSceneDefinition("high-resolution-dam-break"));
  assert.deepEqual(brickEngineGaps(dam, { timeStep: "sixtieth" }), brickEngineLanes());
  await assert.rejects(createBrickSolver({} as GPUDevice, dam, { timeStep: "sixtieth" }, "balanced"), /lane F's stages/);
  assert.deepEqual(brickEngineLanes(bringUpStages, L => bringUpCG(L)), []);
});

test("brick engine: the dam's scene setup (128^3, 1/60 s, its box continued through the closed walls)", () => {
  const dam = sceneDocument(getSceneDefinition("high-resolution-dam-break"));
  const { setup, gaps } = brickSceneSetup(dam, { timeStep: "sixtieth" });
  assert.deepEqual(gaps, []);
  assert.ok(setup);
  assert.deepEqual(setup.lattice, [128, 128, 128]);
  assert.equal(setup.dt_s, 1 / 60);
  assert.equal(setup.h_m, 0.8 / 128);
  assert.deepEqual(setup.capacities, dam.fluid.brickPools);
  // The reservoir's three wall faces drop out (damBreakSignedDistanceAtNode): its box runs past them.
  const box = setup.liquid[0]!, n = 128;
  for (let a = 0; a < 3; a++) assert.ok(box.lo[a]! <= -n, `lo ${a}`);
  for (const [x, y, z] of [[0, 0, 0], [40, 60, 40], [79, 100, 30], [100, 10, 10], [10, 120, 10], [64, 64, 127]] as const) {
    const want = damBreakSignedDistanceAtNode(dam, x, y, z, [n, n, n])! / setup.h_m;
    assert.ok(Math.abs(boxDistance([x, y, z], box) - want) < 1e-6 * n, `phi at ${[x, y, z]}: ${boxDistance([x, y, z], box)} against ${want}`);
  }
  assert.ok(bootBricks(setup).length <= setup.capacities.f);
  // An unenabled scene, and a dam with a body, are refused with their gaps.
  assert.match(brickSceneSetup(createReservoirSeepScene(), {}).gaps.join(";"), /open top/);
  assert.match(brickSceneSetup({ ...dam, rigidBodies: [{} as never] }, { timeStep: "sixtieth" }).gaps.join(";"), /rigid bodies/);
});

test("brick engine: brickPools must be integers, f/c/particles at least 1", () => {
  const scene = createReservoirSeepScene();
  const withPools = (p: Record<string, number>) => ({ ...scene, fluid: { ...scene.fluid!, brickPools: { ...scene.fluid!.brickPools!, ...p } } }) as SceneDescription;
  assert.deepEqual(validateScene(withPools({})).filter(e => /brickPools/.test(e)), []);
  assert.ok(validateScene(withPools({ f: 0 })).some(e => /brickPools\.f/.test(e)));
  assert.ok(validateScene(withPools({ c: 1.5 })).some(e => /brickPools\.c/.test(e)));
  assert.ok(validateScene(withPools({ k: -1 })).some(e => /brickPools\.k/.test(e)));
});

/**
 * The resident set is self-consistent: directory, records, child counts,
 * the free stack, F_ALL and F_LIQ, and the header counts.
 */
function assertConsistent(e: BrickEngine, s: BrickSnapshot, fail = 0) {
  const L = e.layout, d = s.dir, hd = s.header;
  assert.equal(hd[H.FAIL], fail, `FAIL 0x${hd[H.FAIL]!.toString(16)} info 0x${hd[H.FAIL_INFO]!.toString(16)}`);
  const slots = new Set(s.bricks.values());
  assert.equal(slots.size, s.bricks.size, "one slot a brick");
  for (const [key, slot] of s.bricks) {
    const r = L.dir.frec + 32 * slot, b = brickKeyCoord(key);
    assert.equal(d[r + REC.KEY], key);
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const n = [b[0] + dx, b[1] + dy, b[2] + dz];
      const want = n.some(v => v < 0) ? NONE : s.bricks.get(brickKey(n[0]!, n[1]!, n[2]!)) ?? NONE;
      assert.equal(d[r + neighbourIndex(dx, dy, dz)], want, `brick ${b} neighbour ${[dx, dy, dz]}`);
    }
    assert.equal(d[r + REC.PARENT], s.pages.get(brickKey(b[0] >> 2, b[1] >> 2, b[2] >> 2)), `brick ${b} parent`);
    assert.equal(d[r + REC.MASK], 0xff | (brickLocal(b[0], b[1], b[2]) << 8));
    const cls = d[r + REC.CLASS]!;
    assert.equal(cls & (F_CLASS.ENTERING | F_CLASS.LEAVING), 0, `brick ${b} class 0x${cls.toString(16)}`);
    assert.ok(cls & F_CLASS.LIQUID, "M1: F_LIQ = F_ALL");
  }
  for (const [, cs] of s.pages) {
    const children = [...s.bricks.values()].filter(slot => d[L.dir.frec + 32 * slot + REC.PARENT] === cs).length;
    assert.ok(children > 0, "a live page has a child");
    assert.equal(d[L.dir.crec + 32 * cs + REC.C_CHILDREN], children);
  }
  assert.equal(hd[H.LIVE_F], s.bricks.size); assert.equal(hd[H.TOP_F], L.cap.f - s.bricks.size);
  assert.equal(hd[H.LIVE_C], s.pages.size); assert.equal(hd[H.TOP_C], L.cap.c - s.pages.size);
  const free = Array.from(s.set.subarray(L.set.freeF, L.set.freeF + hd[H.TOP_F]!));
  assert.deepEqual([...new Set([...free, ...slots])].sort((a, b) => a - b), Array.from({ length: L.cap.f }, (_, i) => i), "free stack and live slots partition the pool");
  assert.deepEqual(new Set(s.list(LIST.F_ALL)), slots);
  assert.deepEqual(new Set(s.list(LIST.F_LIQ)), slots);
  assert.deepEqual(new Set(s.list(LIST.C_ALL)), new Set(s.pages.values()));
}

/** Overwrite PHI and PHI1 of every live brick with a box's distance (clamped to PHI_AIR), as bootPhi writes it. */
function writePhi(e: BrickEngine, s: BrickSnapshot, box: BrickLiquidBox) {
  const L = e.layout, phi = L.base("phi", "F", "PHI"), phi1 = L.base("phi", "F", "PHI1");
  for (const [key, slot] of s.bricks) {
    const b = brickKeyCoord(key), w = new Float32Array(512);
    for (let l = 0; l < 512; l++) w[l] = Math.min(PHI_AIR, boxDistance([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)], box));
    e.write("phi", phi + BRICK_STRIDE * slot, w); e.write("phi", phi1 + BRICK_STRIDE * slot, w);
  }
}

async function withDevice(label: string, run: (device: GPUDevice) => Promise<void>) {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice(label, run);
}

test("brick engine (GPU): every kernel compiles cold in under 30 s", GPU, async () => {
  await withDevice("brick compile", async device => {
    const e = await BrickEngine.create(device, options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8));
    const kernels = new Set([...e.schedule.boot, ...e.schedule.part2].map(p => p.kernel.label));
    assert.deepEqual(new Set(Object.keys(e.compileMs)), kernels);
    for (const [label, ms] of Object.entries(e.compileMs)) assert.ok(ms < 30_000, `${label} compiled in ${ms.toFixed(0)} ms`);
    await e.assertHealthy();
    e.destroy();
  });
});

test("brick engine (GPU): bricks enter and leave as the CPU reach oracle says, and the set stays consistent", GPU, async () => {
  await withDevice("brick enter/leave", async device => {
    const lattice: Vec3 = [64, 32, 64], o = options(lattice, [sheet(20, 10.3, 20)], 400, 16);
    const e = await BrickEngine.create(device, o);
    let s = await snapshot(e);
    assert.deepEqual([...s.bricks.keys()].sort((a, b) => a - b), bootBricks(o));
    assertConsistent(e, s);
    e.advance(1);
    s = await snapshot(e);
    assertConsistent(e, s);
    // Narrower and taller: x and z brick 3 leave, y brick row 3 enters.
    writePhi(e, s, sheet(12, 18, 12));
    const before = await snapshot(e), L = e.layout;
    const phi = brickField(before, asF32(await e.read("phi")), L.base("phi", "F", "PHI"), PHI_AIR);
    const bound = (slot: number) => asF32(before.set.subarray(L.set.boundF + slot, L.set.boundF + slot + 1))[0]!;
    const bins = (slot: number) => before.set[L.set.binCount + slot]!;
    const want = brickSetOracle(before, phi, bound, bins, lattice, DT, H_M, [0, -G, 0]);
    e.advance(1);
    const after = await snapshot(e);
    assertConsistent(e, after);
    const keys = (m: Map<number, number>) => new Set(m.keys());
    assert.deepEqual(keys(after.bricks), want);
    const entered = [...want].filter(k => !before.bricks.has(k)), left = [...before.bricks.keys()].filter(k => !want.has(k));
    assert.ok(entered.length > 0 && left.length > 0, `entered ${entered.length}, left ${left.length}`);
    assert.ok(entered.every(k => brickKeyCoord(k)[1] === 3) && left.every(k => brickKeyCoord(k)[0] === 3 || brickKeyCoord(k)[2] === 3));
    assert.equal(after.header[H.N_ENTER_F], entered.length);
    assert.equal(after.header[H.N_LEAVE_F], left.length);
    const enterList = Array.from(after.set.subarray(L.set.lists[LIST.F_ENTER]!, L.set.lists[LIST.F_ENTER]! + entered.length));
    assert.deepEqual(new Set(enterList), new Set(entered.map(k => after.bricks.get(k)!)));
    const leaveList = Array.from(after.set.subarray(L.set.lists[LIST.F_LEAVE]!, L.set.lists[LIST.F_LEAVE]! + left.length));
    assert.deepEqual(new Set(leaveList), new Set(left.map(k => before.bricks.get(k)!)));
    // Entering bricks start as air (setInit), and the set holds still after (phi is not advected in M1).
    const phiNow = brickField(after, asF32(await e.read("phi")), L.base("phi", "F", "PHI"), -1);
    for (const k of entered) { const b = brickKeyCoord(k); assert.equal(phiNow([8 * b[0] + 3, 8 * b[1] + 3, 8 * b[2] + 3]), PHI_AIR); }
    e.advance(1);
    const still = await snapshot(e);
    assertConsistent(e, still);
    assert.deepEqual(keys(still.bricks), want);
    assert.equal(still.header[H.N_ENTER_F], 0); assert.equal(still.header[H.N_LEAVE_F], 0);
    e.destroy();
  });
});

test("brick engine (GPU): FAIL.POOL_F latches when the reach needs more bricks than the pool", GPU, async () => {
  await withDevice("brick pool fail", async device => {
    const lattice: Vec3 = [64, 32, 64], liquid = [sheet(20, 10.3, 20)];
    const f = bootBricks(options(lattice, liquid, 1, 1)).length;
    const e = await BrickEngine.create(device, options(lattice, liquid, f, 16));
    e.advance(1);
    await e.assertHealthy();
    writePhi(e, await snapshot(e), sheet(20, 18, 20));
    e.advance(1);
    await assert.rejects(e.assertHealthy(), /POOL_F/);
    const s = await snapshot(e);
    assert.ok(s.header[H.FAIL]! & FAIL.POOL_F);
    assert.ok(s.header[H.REFUSED_F]! > 0);
    // A refused claim leaves its child word NONE: the set is still consistent, and full.
    assertConsistent(e, s, s.header[H.FAIL]);
    assert.equal(s.bricks.size, f);
    for (let i = 0; i < 20 && !e.failure; i++) await e.fence();
    assert.match(e.failure ?? "", /POOL_F/, "the lagged receipt latches the failure on the host");
    assert.throws(() => e.advance(1), /POOL_F/);
    e.destroy();
  });
});

test("brick engine (GPU): a static box is hydrostatic (Float64 oracle and analytic), with ~0 velocity after projection", GPU, async t => {
  await withDevice("brick hydrostatic", async device => {
    // The box's sides and floor sit 8 cells outside the lattice, so its distance is y - top wherever
    // the surface is: with sides 1 cell out, the SDF takes the side distance beside the walls.
    const lattice: Vec3 = [32, 32, 32], top = 10.3;
    const e = await BrickEngine.create(device, options(lattice, [{ lo: [-8, -8, -8], hi: [40, top, 40] }], 200, 8));
    e.advance(1);
    const s = await snapshot(e), L = e.layout;
    await e.assertHealthy();
    const pres = asF32(await e.read("pres")), vel = asF32(await e.read("vel")), phiW = asF32(await e.read("phi"));
    const phi = brickField(s, phiW, L.base("phi", "F", "PHI"), PHI_AIR);
    const us = [0, 1, 2].map(a => brickField(s, vel, L.base("vel", "F", `US${a}`), 0));
    const X = brickField(s, pres, L.base("pres", "F", "X"), 0);
    const U = [0, 1, 2].map(a => brickField(s, vel, L.base("vel", "F", `U${a}`), 0));
    const cells: Vec3[] = [];
    for (const key of s.bricks.keys()) { const b = brickKeyCoord(key); for (let l = 0; l < 512; l++) cells.push([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]); }
    const oracle = pressureOracle(lattice, H_M, cells, phi, (a, c) => us[a]!(c));
    assert.ok(oracle.X.size === 32 * 10 * 32, `liquid cells ${oracle.X.size}`);
    let scale = 0, err = 0, analytic = 0, air = 0;
    for (const c of cells) {
      if (c.some((v, a) => v >= lattice[a]!)) continue;
      const want = oracle.X.get(c.join(","));
      if (want === undefined) { air = Math.max(air, Math.abs(X(c))); continue; }
      scale = Math.max(scale, Math.abs(want)); err = Math.max(err, Math.abs(X(c) - want));
      analytic = Math.max(analytic, Math.abs(want - G * DT * (top - (c[1] + 0.5)) * H_M) / (G * DT * (top - (c[1] + 0.5)) * H_M));
    }
    let speed = 0;
    for (const c of cells) {
      if (c.some((v, a) => v >= lattice[a]!)) continue;
      if (!oracle.X.has(c.join(","))) continue;
      for (let a = 0; a < 3; a++) speed = Math.max(speed, Math.abs(U[a]!(c)));
    }
    const header = s.header;
    t.diagnostic(`oracle ${oracle.iterations} its, residual ${oracle.residual.toExponential(2)}; GPU CG ${header[CG_HEADER.ITERATIONS]} its; |X - X64|/|X| ${(err / scale).toExponential(2)}; analytic ${analytic.toExponential(2)}; max |U| ${speed.toExponential(2)} m/s; air |X| ${air}`);
    // The Float64 system's solution is the discrete hydrostatic profile (linear under the ghost-fluid
    // condition, Gibou et al. 2002), up to the f32 rounding of the phi it reads.
    assert.ok(analytic < 1e-5, `the Float64 system is hydrostatic: ${analytic}`);
    // The bring-up CG closes at max|r| <= 1e-3 h^2 (X units m^2/s), about 1e-6 against X ~ 0.05: 1e-3 relative is its bound with margin.
    assert.ok(err / scale < 1e-3, `GPU X against Float64: ${err / scale}`);
    assert.equal(air, 0, "X is 0 off the liquid");
    // g dt = 0.16 m/s is what the projection cancels; it must leave under 0.1% of it.
    assert.ok(speed < 1e-3 * G * DT, `max |U| ${speed} m/s`);
    e.destroy();
  });
});

test("brick engine (GPU): a 4x-wider box holds the same bricks and list counts", GPU, async () => {
  await withDevice("brick wider", async device => {
    const run = async (lattice: Vec3) => {
      const e = await BrickEngine.create(device, options(lattice, [sheet(20, 10.3, 20)], 400, 16));
      e.advance(2);
      const s = await snapshot(e);
      assertConsistent(e, s);
      const out = { keys: [...s.bricks.keys()].sort((a, b) => a - b), counts: Array.from(s.header.subarray(H.LIST_COUNT, H.LIST_COUNT + 17)), bytes: e.bytes, dir: e.layout.dir.words };
      e.destroy();
      return out;
    };
    const a = await run([64, 32, 64]), b = await run([256, 32, 64]);
    assert.deepEqual(b.keys, a.keys);
    assert.deepEqual(b.counts, a.counts);
    assert.equal(b.bytes - a.bytes, 4 * (b.dir - a.dir), "only the directory's page window grows");
  });
});

/** One z slice of an r32float 3D texture. */
async function readSlice(device: GPUDevice, texture: GPUTexture, z: number): Promise<(x: number, y: number) => number> {
  const w = texture.width, h = texture.height, row = Math.ceil(4 * w / 256) * 256;
  const buf = device.createBuffer({ size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture, origin: [0, 0, z] }, { buffer: buf, bytesPerRow: row, rowsPerImage: h }, [w, h, 1]);
  device.queue.submit([enc.finish()]); await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
  return (x, y) => f[(row / 4) * y + x]!;
}

test("brick engine (GPU): the app adapter steps, reports, and presents the pool (and clears bricks that leave)", GPU, async () => {
  await withDevice("brick adapter", async device => {
    const setup: BrickSceneSetup = { sceneId: "test", lattice: [64, 32, 64], h_m: H_M, dt_s: DT, gravity_mps2: [0, -G, 0], density_kg_m3: 1000, viscosity_Pa_s: 0, surfaceTension_N_m: 0,
      capacities: { f: 400, c: 16, b: 0, k: 0, particles: 1024 }, liquid: [sheet(20, 10.3, 20)], origin_m: [0, 0, 0] };
    const far = 2;
    const solver = await BrickNBFlipSolver.create(device, { setup, stages: bringUpStages, solver: L => bringUpCG(L), values: {}, quality: "balanced", farPhi_m: far });
    assert.equal(solver.advanceTo(0.5 * DT, []), false, "no partial step");
    assert.equal(solver.advanceTo(DT, []), true);
    await solver.awaitFrameCompletion();
    await solver.assertSimulationHealthy();
    const info = await solver.readStats();
    assert.equal(info.brickEngine?.liveF, bootBricks(setup).length);
    assert.equal(info.brickEngine?.step, 1);
    assert.ok(info.pressureIterations > 0);
    assert.deepEqual(solver.fluidDomain?.dimensions, [64, 32, 64]);
    let phi = await readSlice(device, solver.denseLevelSetVolumeSource.vertexPhi, 8), vol = await readSlice(device, solver.volumeTexture, 8);
    assert.ok(Math.abs(phi(8, 4) - boxDistance([8, 4, 8], sheet(20, 10.3, 20)) * H_M) < 1e-6, `phi ${phi(8, 4)}`);
    assert.equal(vol(8, 4), 1); assert.equal(vol(8, 20), 0);
    const surface = vol(8, 10);
    assert.ok(Math.abs(surface - 0.3) < 0.02, `the surface cell holds 0.3: ${surface}`);
    assert.equal(phi(60, 4), far, "no brick there: far phi");
    // Narrower: bricks with x >= 3 leave, and their texels go back to far.
    const e = solver.brickEngine;
    writePhi(e, await snapshot(e), sheet(12, 10.3, 20));
    assert.equal(solver.advanceTo(2 * DT, []), true);
    await solver.awaitFrameCompletion();
    phi = await readSlice(device, solver.denseLevelSetVolumeSource.vertexPhi, 8); vol = await readSlice(device, solver.volumeTexture, 8);
    assert.equal(phi(26, 4), far, "a brick that left reads far");
    assert.equal(vol(26, 4), 0);
    assert.ok(Math.abs(phi(8, 4) - boxDistance([8, 4, 8], sheet(12, 10.3, 20)) * H_M) < 1e-6);
    assert.equal(solver.presentations, 3);
    solver.destroy();
  });
});

test("brick engine (GPU): operator leaves zero rows (K, DIAG, RHS, X) off the liquid in every live brick", GPU, async () => {
  await withDevice("brick zero rows", async device => {
    const e = await BrickEngine.create(device, options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8));
    let s = await snapshot(e);
    // Stale values everywhere, as a reused slot would hold.
    const junk = new Float32Array(512).fill(7.5);
    for (const slot of s.bricks.values()) for (const ch of ["X", "RHS", "DIAG", "K0", "K1", "K2"]) e.write("pres", e.layout.base("pres", "F", ch) + BRICK_STRIDE * slot, junk);
    e.advance(1);
    s = await snapshot(e);
    const ch = Object.fromEntries(await Promise.all(["X", "RHS", "DIAG", "K0", "K1", "K2"].map(async c => [c, await readChannel(e, "pres", c)] as const)));
    let air = 0, rows = 0;
    for (const [key, slot] of s.bricks) {
      const b = keyOf(key), diag = ch.DIAG!(slot);
      for (let l = 0; l < 512; l++) {
        const x = 8 * b[0] + (l & 7), y = 8 * b[1] + ((l >> 3) & 7), z = 8 * b[2] + (l >> 6);
        // The last brick row holds the vertices at n (bricksPerAxis); its cells are outside the lattice: no rows.
        const outside = x >= 32 || y >= 32 || z >= 32;
        if (y >= 11 || outside) { air++; for (const c of ["X", "RHS", "DIAG", "K0", "K1", "K2"]) assert.equal(ch[c]!(slot)[l], 0, `${c} at air`); }
        else if (y <= 9) { rows++; assert.ok(diag[l]! > 0 && diag[l] !== 7.5, `DIAG ${diag[l]} at brick ${b} local ${l} (slot ${slot}), X ${ch.X!(slot)[l]}`); }
      }
    }
    assert.ok(air > 0 && rows > 0);
    await e.assertHealthy();
    e.destroy();
  });
});
