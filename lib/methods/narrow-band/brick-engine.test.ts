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
 * - enter and leave match the head's residency oracle (records and set), with the
 *   directory, records, child counts, free stack and lists consistent;
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
import { BRICK_CELLS, BRICK_STRIDE, BrickLayout, F_CLASS, FAIL, H, KEY_PAGE, K_HOLD, LIST, NONE, PHI_AIR, PHI_AIR_C, POOLS, REC, ARGS, brickAbiWGSL, brickKey, brickKeyCoord, brickLocal, neighbourIndex } from "./brick-abi";
import { BRICK_KICK_RESOLVES, BrickEngine, type BrickEngineOptions, type BrickLiquidBox, type Vec3, bootBricks, describeFail, bootCornerPages, bootPhiAt, boxDistance, reachSet } from "./brick-engine";
import { CG_HEADER, bringUpCG } from "./brick-bringup-cg";
import { bringUpStages } from "./brick-stages";
import { PRESSURE_ARGS, validatePressureSolver } from "./brick-pressure-slot";
import { bindGroupLayoutEntries, declareBindings, MAX_STORAGE_BINDINGS, type BrickBinding, type BrickKernel, type BrickPlacement } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL, brickSetKernels } from "./brick-set.wgsl";
import { brickPressureKernels } from "./brick-pressure.wgsl";
import { pressureOperatorKernels } from "./pressure-operator.wgsl";
import { brickEngineGaps, brickEngineLanes, brickEngineRefusal, nbEngine, nbEngineChoice, nbEngineStatusText } from "./brick-method";
import { BRICK_OPEN_TOP_HEADROOM, BRICK_SPLASH_DROP, brickInitialExtent, brickPoolRule, brickSceneSetup, type BrickSceneSetup } from "./brick-scene";
import { BrickNBFlipSolver } from "./brick-solver";
import { brickField, pressureOracle, residencyOracle } from "./brick-oracle";
import { keyOf, readChannel, snapshot, type BrickSnapshot } from "./brick-readback";
import { validateScene, type SceneDescription } from "../../core/model";
import { BRICK_STATIC_EDIT_K } from "./brick-static.wgsl";
import { createReservoirSeepScene } from "../../core/reservoir-seep-scene";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { resolveMethodValues } from "../../core/method-contract";
import { uniformNarrowBandMethod } from "../uniform/uniform-narrow-band-method";
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
  // Only the page window may grow: its page word and its 64 FLAT words (brick-abi.md §3) a 32^3 page.
  assert.ok(B.dir.words - A.dir.words <= 65 * (pages(B) - pages(A)) + 128, `dir grows ${B.dir.words - A.dir.words} words for ${pages(B) - pages(A)} pages`);
  assert.ok(B.bytes - A.bytes <= 4 * (65 * (pages(B) - pages(A)) + 128));
});

test("brick engine: every kernel binds at most 8 storage buffers, and the layout check refuses a ninth", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
  const kernels = [...Object.values(brickSetKernels(L)), ...Object.values(brickPressureKernels(L)), ...Object.values(pressureOperatorKernels(L)),
    ...[...cg.setup, ...cg.slot(0), ...cg.finish].map(p => p.kernel)];
  for (const k of kernels) assert.ok(k.bindings.filter(b => b.kind !== "uniform").length <= MAX_STORAGE_BINDINGS, k.label);
  const ten = Array.from({ length: 10 }, (_, i) => ({ name: `b${i}`, kind: "rw" as const }));
  assert.throws(() => bindGroupLayoutEntries(ten), /10 storage buffers/);
});

test("brick engine: the boot's reach rule keeps a one-brick ring at rest and widens with the velocity bound", () => {
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

test("brick engine: on a padded lattice the boot's φ past the box is far (no liquid φ in the padding)", () => {
  // The closed form's faces run on through the walls (a sheet to 33 in a 24 x 16 x 24 box, stored on 32^3): a vertex
  // on the box plane keeps its distance, one past it reads far, as nbVertexPhi reads past an unpadded lattice.
  const box: Vec3 = [24, 16, 24], o = { ...options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8), box }, at = bootPhiAt(o);
  const far = bootPhiAt({ ...o, liquid: [] })([5, 5, 5]);
  assert.ok(at([24, 5, 24]) < 0 && at([5, 5, 5]) < 0);
  for (const v of [[25, 5, 5], [5, 5, 25], [31, 5, 31], [5, 17, 5]] as Vec3[]) assert.equal(at(v), far, `${v}`);
  // Unpadded (no box, or the box the lattice) the closed form runs on to the lattice's edge, unchanged.
  assert.ok(bootPhiAt(options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8))([31, 5, 31]) < 0);
  assert.ok(bootPhiAt({ ...o, box: [32, 32, 32] })([31, 5, 31]) < 0);
  // The padded boot bricks: none wholly past the box (brick 3's low plane is the box plane x, z = 24).
  assert.ok(bootBricks(o).every(k => brickKeyCoord(k)[0] <= 3 && brickKeyCoord(k)[2] <= 3));
  // bootPhi emits the far line only on a padded layout, so an unpadded scene's text is unchanged.
  const code = (L: BrickLayout) => brickSetKernels(L).bootPhi.code;
  const padded = code(new BrickLayout({ lattice: [32, 32, 32], box, f: 64, c: 1, b: 0, k: 0, particles: 1024 }));
  const plain = code(new BrickLayout({ lattice: [32, 32, 32], f: 64, c: 1, b: 0, k: 0, particles: 1024 }));
  assert.ok(padded.includes("if(any(v>NB_BOX)){return NB_PHI_FAR;}") && !plain.includes("return NB_PHI_FAR;}\n  let p=vec3f(v)") && !plain.includes("v>NB_LATTICE)){return NB_PHI_FAR;}\n  let p"));
});

test("brick engine: a tiered boot also claims the pages holding its pages' high corner planes", () => {
  // The sheet's boot bricks lie in page (0, 0, 0): its seven high neighbours hold its high-plane C vertices.
  const o = options([128, 64, 128], [sheet(20, 10.3, 20)], 400, 16), bricks = bootBricks(o);
  assert.ok(bricks.every(k => brickKeyCoord(k).every(v => v < 4)));
  const pages = bootCornerPages(o.lattice, bricks);
  assert.equal(pages.length, 7);
  for (const k of pages) {
    assert.equal(k & KEY_PAGE, KEY_PAGE);
    const p = brickKeyCoord(k & ~KEY_PAGE).map(v => v >> 2);
    assert.ok(p.every(v => v <= 1) && p.some(v => v === 1), `page ${p}`);
  }
  // A page past the lattice is not claimed: on one 32^3 page there is none.
  assert.deepEqual(bootCornerPages([32, 32, 32], bootBricks(options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8))), []);
});

test("brick engine: a pressure solver's indirect words must be the lists' or P's", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
  validatePressureSolver(cg);
  assert.equal(cg.bringUp, true);
  const kernel = cg.setup[0]!.kernel;
  assert.throws(() => validatePressureSolver({ ...cg, setup: [{ kernel, dispatch: { indirect: ARGS.LANE_F } }] }), /args word 2048, outside/);
  validatePressureSolver({ ...cg, setup: [{ kernel, dispatch: { indirect: PRESSURE_ARGS.FIRST } }] });
});

test("brick engine: the engine switch refuses unknown engines, and names a gap scene's gaps", async () => {
  assert.equal(nbEngine(undefined), "auto");
  assert.equal(nbEngine("auto"), "auto");
  assert.equal(nbEngine("uniform"), "uniform");
  assert.equal(nbEngine("brick"), "brick");
  assert.throws(() => nbEngine("bricks"), /unknown engine "bricks"/);
  // No scene list (2026-11-18): the seep runs; the seep in an open world (its voxels) is refused with the box's gap.
  const seep0 = createReservoirSeepScene(), seep = { ...seep0, container: { ...seep0.container, walls: "none" as const } };
  assert.ok(seep.fluid?.brickPools, "seep authors brickPools");
  assert.deepEqual(brickEngineGaps(seep0, { timeStep: "sixtieth" }), [], "the seep has no gaps");
  const refusal = brickEngineRefusal(seep, { timeStep: "sixtieth" });
  assert.match(refusal.message, /open world/);
  // Each lane's line is there exactly while its registration is missing (F registers the tiered set, P compositeMGPCG).
  assert.equal(/lane F's stages/.test(refusal.message), brickEngineLanes().some(l => /lane F/.test(l)));
  // Lane P's line is there exactly while P's solver is unregistered (it registers compositeMGPCG at M2).
  assert.equal(/lane P's pressure solver/.test(refusal.message), brickEngineLanes().some(l => /lane P/.test(l)));
  assert.match(refusal.message, /engine "uniform"/);
  assert.ok(!brickEngineGaps(seep).some(g => /brickPools/.test(g)));
  // The dam: no scene gap, only the lanes'.
  const dam = sceneDocument(getSceneDefinition("high-resolution-dam-break"));
  assert.deepEqual(brickEngineGaps(dam, { timeStep: "sixtieth" }), brickEngineLanes());
  assert.deepEqual(brickEngineLanes(), [], "F and P are registered: the dam runs (opt-in, engine \"brick\")");
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
  // The authored pools, but K: at least the edit headroom (BRICK_STATIC_EDIT_K, the dam has no voxels), at most f.
  const authored = dam.fluid.brickPools!;
  assert.deepEqual(setup.capacities, { ...authored, k: Math.max(authored.k, Math.min(authored.f, BRICK_STATIC_EDIT_K)) });
  // The dam keeps particles 2 cells deep (scene.fluid.brickTiers); a scene that does not say keeps 3.
  assert.equal(setup.particleBandDepth, 2);
  { const plain = sceneDocument(getSceneDefinition("high-resolution-dam-break")); delete (plain.fluid as { brickTiers?: unknown }).brickTiers; assert.equal(brickSceneSetup(plain, { timeStep: "sixtieth" }).setup?.particleBandDepth, 3); }
  // The reservoir's three wall faces drop out (damBreakSignedDistanceAtNode): its box runs past them.
  const box = setup.liquid[0]!, n = 128;
  for (let a = 0; a < 3; a++) assert.ok(box.lo[a]! <= -n, `lo ${a}`);
  for (const [x, y, z] of [[0, 0, 0], [40, 60, 40], [79, 100, 30], [100, 10, 10], [10, 120, 10], [64, 64, 127]] as const) {
    const want = damBreakSignedDistanceAtNode(dam, x, y, z, [n, n, n])! / setup.h_m;
    assert.ok(Math.abs(boxDistance([x, y, z], box) - want) < 1e-6 * n, `phi at ${[x, y, z]}: ${boxDistance([x, y, z], box)} against ${want}`);
  }
  assert.ok(bootBricks(setup).length <= setup.capacities.f);
  // An open world the box cannot hold (the seep's voxels) is refused with its gap; a body scene without authored K slots takes one a F brick (k = f).
  const seep = createReservoirSeepScene();
  assert.match(brickSceneSetup({ ...seep, container: { ...seep.container, walls: "none" } }, { timeStep: "sixtieth" }).gaps.join(";"), /open world/);
  const boats = sceneDocument(getSceneDefinition("paddleboats")), { brickPools: _pools, ...boatFluid } = boats.fluid;
  const ruled = brickSceneSetup({ ...boats, fluid: boatFluid }, { timeStep: "sixtieth" });
  assert.deepEqual(ruled.gaps, []);
  assert.equal(ruled.setup!.capacities.k, ruled.setup!.capacities.f);
});

test("brick engine: brickPools must be integers, f/c/particles at least 1", () => {
  const scene = createReservoirSeepScene();
  const withPools = (p: Record<string, number>) => ({ ...scene, fluid: { ...scene.fluid!, brickPools: { ...scene.fluid!.brickPools!, ...p } } }) as SceneDescription;
  assert.deepEqual(validateScene(withPools({})).filter(e => /brickPools/.test(e)), []);
  assert.ok(validateScene(withPools({ f: 0 })).some(e => /brickPools\.f/.test(e)));
  assert.ok(validateScene(withPools({ c: 1.5 })).some(e => /brickPools\.c/.test(e)));
  assert.ok(validateScene(withPools({ k: -1 })).some(e => /brickPools\.k/.test(e)));
});

test("brick engine: the pool sizing rule (unauthored brickPools) reproduces the measured table, and authored pools win", () => {
  // The rule's terms, by hand: the window caps a 64^3 lattice at 9^3 F and 3^3 C; a 100-cell droplet takes the floor.
  assert.deepEqual(brickPoolRule([64, 64, 64], { cells: 100, top: 16, drop: 0 }, 8, 1276), { f: 128, c: 18, b: 16, k: 0, particles: 3828, t: 4096 });
  assert.deepEqual(brickPoolRule([64, 64, 64], { cells: 94208, top: 32, drop: 0 }, 324, 174856), { f: 729, c: 27, b: 16, k: 0, particles: 524568, t: 4096 });
  // TCAP: 512 c is exact (a C_ALL tile is listed once a step); 2 f where that is larger than 4096 but under 512 c.
  assert.equal(brickPoolRule([32, 32, 32], { cells: 961, top: 30, drop: 0 }, 72, 12960).t, 4096);
  assert.equal(brickPoolRule([16, 16, 16], { cells: 300, top: 8, drop: 0 }, 27, 9632).t, 512);
  assert.equal(brickPoolRule([256, 256, 256], { cells: 6400, top: 30, drop: 0 }, 48, 26224).t, Math.min(512 * 162, 2 * 4356));
  // The splash term (drop >= BRICK_SPLASH_DROP): 10 min(floorF, ceil(1.5 V/64)) bricks and 10 V particles. Below the
  // trigger nothing changes; a small liquid's film area (10 x 10 bricks) stays under the spread term SF while its
  // particles take 10 V; a large liquid takes eight layers of the floor with the margin.
  const pool = { cells: 6400, top: 30 };
  assert.deepEqual(brickPoolRule([256, 256, 256], { ...pool, drop: 15.9 }, 48, 26224), brickPoolRule([256, 256, 256], { ...pool, drop: 0 }, 48, 26224));
  assert.deepEqual([brickPoolRule([256, 256, 256], { cells: 400, top: 30, drop: BRICK_SPLASH_DROP }, 8, 1276)].map(p => [p.f, p.particles])[0], [400, 4000]);
  assert.deepEqual([brickPoolRule([128, 128, 128], { cells: 33375, top: 128, drop: 69 }, 900, 1000)].map(p => [p.f, p.particles])[0], [2890, 333750]);
  // Through brickSceneSetup on the catalog scenes the rule was measured against (progress-lane-e.md, 2026-11-17).
  const want: Record<string, [number, number, number]> = {
    // The enabled droplets and the 100- and 800-cell fills author their pools (scenes.ts: their film outgrows the rule's
    // spread).
    "power-droplet-256": [128, 128, 3828], "power-fill-256-6400": [4356, 162, 78672],
    "minimal-power-dam-break-32": [125, 8, 117096], "minimal-power-dam-break-64": [729, 27, 524568], "mass-conserving-figure-9-dam-break": [2601, 75, 720384],
    // The splash term (drop >= 16 cells; empirical, fitted to five measured peaks): the free falls take their window
    // (WF 405; they peaked at 289 and 238 F, so without it 1.25 * the boot's 210 and 245 resident latched POOL_F);
    // cm12-figure-7 takes eight layers of its 289-brick floor with the margin (peak 2437), its particles 3 bootP > 10 V.
    "nbflip-analytic-free-fall": [405, 18, 150528], "nbflip-analytic-tilted-free-fall": [405, 18, 150528],
    "cm12-figure-7": [2890, 125, 377568], "cm12-figure-7-256": [10890, 729, 2679989],
  };
  {
    for (const [id, [f, c, particles]] of Object.entries(want)) {
      const scene = sceneDocument(getSceneDefinition(id));
      // power-droplet-256 authors its pools since 2026-11-18 (the rule's F latched within 300 steps); the rule is
      // still the one measured against it, so it is checked with the authored pools removed.
      if (id === "power-droplet-256") { assert.ok(scene.fluid.brickPools, `${id} authors its pools`); delete scene.fluid.brickPools; }
      assert.equal(scene.fluid.brickPools, undefined, `${id} authors no pools`);
      const { setup, gaps } = brickSceneSetup(scene, { timeStep: "sixtieth" });
      assert.deepEqual(gaps, [], id);
      assert.deepEqual([setup!.capacities.f, setup!.capacities.c, setup!.capacities.particles], [f, c, particles], id);
      assert.ok(setup!.bootBricks!.length <= f && setup!.bootPages!.length <= c, `${id}: the boot fits its pools`);
    }
    // The closed form's volume: the 64^3 dam's box is 94208 cells (the measured table's V); the extent counts liquid
    // cells' fill of their corner φ, which a face's wall-continued row may add to, well inside the rule's margins.
    const dam64 = brickSceneSetup(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")), { timeStep: "sixtieth" }).setup!;
    const extent = brickInitialExtent({ lattice: dam64.lattice, farPhi: dam64.farPhi, liquid: dam64.liquid, initial: dam64.initial });
    assert.ok(Math.abs(extent.cells - 94208) <= 0.01 * 94208, `extent ${extent.cells} cells`);
    // A small free box counts its boundary cells (2026-11-20): an 8 x 6 x 8 box is the gauge's 364 cells (the fill of
    // its cells' corner φ), not its 7 x 5 x 7 = 245 liquid vertices. The slope clearance's former release was that box,
    // and the spread term SF took 248 F from 245 cells: POOL_F latched at step 56 against the 287 its run peaked at.
    const small = brickInitialExtent({ lattice: [128, 32, 32], liquid: [{ lo: [4, 17, 12], hi: [12, 23, 20] }] });
    assert.ok(Math.abs(small.cells - 364) < 1, `extent ${small.cells} cells`);
    assert.ok(brickPoolRule([128, 32, 32], small, small.resident, 0).f >= 1.15 * 287);
    // The slope clearance itself: its extent is its gauge's t = 0 volume, and F holds its 360-step peak of 322
    // (tools/probe-slope-clearance-brick.ts) with a margin.
    const slope = brickSceneSetup(sceneDocument(getSceneDefinition("nbflip-slope-clearance")), { timeStep: "sixtieth" }).setup!;
    assert.ok(Math.abs(brickInitialExtent({ lattice: slope.lattice, farPhi: slope.farPhi, liquid: slope.liquid, initial: slope.initial }).cells - 3314) < 1);
    assert.ok(slope.capacities.f >= 1.15 * 322, `slope clearance F ${slope.capacities.f}`);
    // D2: the free fall's liquid bricks dilated by two fill bricks x 1..6, y 2..8, z 0..4 of its 9 x 9 x 5 window
    // (6 * 7 * 5 = 210), the F box its engine holds at step 1.
    const fall = brickSceneSetup(sceneDocument(getSceneDefinition("nbflip-analytic-free-fall")), { timeStep: "sixtieth" }).setup!;
    assert.equal(brickInitialExtent({ lattice: fall.lattice, farPhi: fall.farPhi, liquid: fall.liquid, initial: fall.initial }).resident, 210);
    // A container that is whole bricks but not whole pages is padded (M3): the lattice is whole 32^3 pages and the
    // box is the container's cells, where every wall is (F's, the release readers', P's pressure rows, the C level
    // set's and the presentation's; tests/brick-padded-dam.test.ts).
    const slab = sceneDocument(getSceneDefinition("ceiling-slab-drop"));
    const padded = brickSceneSetup(slab, { timeStep: "sixtieth" });
    assert.deepEqual(padded.gaps, []);
    assert.deepEqual([padded.setup!.lattice, padded.setup!.box], [[32, 32, 32], [24, 16, 24]]);
    // A container that is not whole bricks (64 x 20 x 64): the box is its cells rounded up to bricks (64 x 24 x 64), the
    // cells between are the static mask's padding voxels (the old engine's solidWorldForFluidLattice), and the initial
    // liquid and pools read the container's own cells.
    const big = sceneDocument(getSceneDefinition("large-power-hydrostatic"));
    const cut = brickSceneSetup(big, { timeStep: "sixtieth" });
    assert.deepEqual(cut.gaps, []);
    assert.deepEqual([cut.setup!.lattice, cut.setup!.box], [[64, 32, 64], [64, 24, 64]]);
    const m = cut.setup!.staticMask!;
    assert.equal(cut.setup!.staticsDeferred, undefined, "the padding is a static solid at the build");
    for (const [x, z] of [[0, 0], [31, 17], [63, 63]] as const) {
      for (let y = 0; y < 20; y++) assert.equal(m.solid(x, y, z), false, `container cell ${x},${y},${z} open`);
      for (let y = 20; y < 24; y++) assert.equal(m.solid(x, y, z), true, `padding cell ${x},${y},${z} solid`);
    }
    // Its 1.6 x 0.05 x 1.6 m dam (32 x 1 x 32 cells), run on through the low walls.
    assert.deepEqual(cut.setup!.liquid, [{ lo: [-128, -128, -128], hi: [32, 1, 32] }]);
    assert.ok(cut.setup!.capacities.k >= 8 * 8, "K holds the padding's cut bricks");
    // An open top (M3): BRICK_OPEN_TOP_HEADROOM cells of air above the container's pages, in the box, so the x and z
    // walls rise to the lattice's top and nothing is a wall at the container's own top. The old engine's far φ stays.
    const tank = sceneDocument(getSceneDefinition("bulk-rest-tank"));
    const open = brickSceneSetup(tank, { timeStep: "sixtieth" });
    assert.deepEqual(open.gaps, []);
    assert.deepEqual([open.setup!.lattice, open.setup!.box, open.setup!.openTop, open.setup!.farPhi], [[256, 288 + BRICK_OPEN_TOP_HEADROOM, 128], undefined, true, 288]);
    // Its closed form keeps the slab's top face (no wall continuation through an open top), at the container's height.
    const boats = structuredClone(sceneDocument(getSceneDefinition("paddleboats"))); boats.rigidBodies = [];
    const slabTop = brickSceneSetup(boats, { timeStep: "sixtieth" });
    assert.deepEqual(slabTop.gaps, []);
    assert.deepEqual([slabTop.setup!.lattice, slabTop.setup!.openTop], [[512, 32 + BRICK_OPEN_TOP_HEADROOM, 192], true]);
    assert.equal(slabTop.setup!.liquid.length, 1);
    assert.equal(slabTop.setup!.liquid[0]!.hi[1], 0.75 * 32, "the slab's top is its fill, not run on through a lid");
    // A top the engine does not know fails explicitly.
    const odd = structuredClone(tank); (odd.container as { top: string }).top = "vented";
    assert.ok(brickSceneSetup(odd, { timeStep: "sixtieth" }).gaps.some(g => g.includes('container.top "vented"')));
  }
  // An authored scene keeps its pools exactly (no t unless it says), K at least the edit headroom.
  const dam = sceneDocument(getSceneDefinition("high-resolution-dam-break")), authored = dam.fluid.brickPools!;
  assert.deepEqual(brickSceneSetup(dam, { timeStep: "sixtieth" }).setup!.capacities, { ...authored, k: Math.max(authored.k, Math.min(authored.f, BRICK_STATIC_EDIT_K)) });
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
  // FLAT is the walk page -> child resolved (brick-abi.md §3), whatever the residency did.
  const flat = L.flatMismatches(d);
  assert.equal(flat.count, 0, `FLAT differs from the walk at ${flat.count} bricks (${flat.first})`);
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
  }
  for (const [, cs] of s.pages) {
    const children = [...s.bricks.values()].filter(slot => d[L.dir.frec + 32 * slot + REC.PARENT] === cs).length;
    // A page the residency entered whose bricks a full pool refused (FAIL.POOL_F) has none; it leaves at the next head.
    assert.ok(children > 0 || (fail & FAIL.POOL_F) !== 0, "a live page has a child");
    assert.equal(d[L.dir.crec + 32 * cs + REC.C_CHILDREN], children);
  }
  assert.equal(hd[H.LIVE_F], s.bricks.size); assert.equal(hd[H.TOP_F], L.cap.f - s.bricks.size);
  assert.equal(hd[H.LIVE_C], s.pages.size); assert.equal(hd[H.TOP_C], L.cap.c - s.pages.size);
  const free = Array.from(s.set.subarray(L.set.freeF, L.set.freeF + hd[H.TOP_F]!));
  assert.deepEqual([...new Set([...free, ...slots])].sort((a, b) => a - b), Array.from({ length: L.cap.f }, (_, i) => i), "free stack and live slots partition the pool");
  assert.deepEqual(new Set(s.list(LIST.F_ALL)), slots);
  // F_LIQ: the live bricks setMark classed LIQUID (an h-owned φ < 0 vertex).
  assert.deepEqual(new Set(s.list(LIST.F_LIQ)), new Set([...slots].filter(slot => d[L.dir.frec + 32 * slot + REC.CLASS]! & F_CLASS.LIQUID)));
  assert.deepEqual(new Set(s.list(LIST.C_ALL)), new Set(s.pages.values()));
}

/** Overwrite PHI and PHI1 of every live brick with a box's distance, clamped to PHI_AIR (a test state; bootPhi writes it uncapped). */
function writePhi(e: BrickEngine, s: BrickSnapshot, box: BrickLiquidBox) {
  const L = e.layout, phi = L.base("phi", "F", "PHI"), phi1 = L.base("phi", "F", "PHI1");
  for (const [key, slot] of s.bricks) {
    const b = brickKeyCoord(key), w = new Float32Array(BRICK_CELLS);
    for (let l = 0; l < BRICK_CELLS; l++) w[l] = Math.min(PHI_AIR, boxDistance([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)], box));
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
    const kernels = new Set([...e.schedule.boot, ...e.schedule.part1, ...e.schedule.part2].map(p => p.kernel.label));
    assert.deepEqual(new Set(Object.keys(e.compileMs)), kernels);
    for (const [label, ms] of Object.entries(e.compileMs)) assert.ok(ms < 30_000, `${label} compiled in ${ms.toFixed(0)} ms`);
    await e.assertHealthy();
    e.destroy();
  });
});

test("brick engine (GPU): bricks enter and leave as the head's residency oracle says, and the set stays consistent", GPU, async () => {
  await withDevice("brick enter/leave", async device => {
    // 128 wide and 64 tall: the all-h owners' 20-cell margin (RES_MARGIN.own.allH) past the sheet reaches brick 5 of 16 in x and
    // z and brick 4 of 8 in y, so the residency has room to grow and shrink.
    const lattice: Vec3 = [128, 64, 128], o = options(lattice, [sheet(20, 10.3, 20)], 400, 16);
    const e = await BrickEngine.create(device, o);
    let s = await snapshot(e);
    assert.deepEqual([...s.bricks.keys()].sort((a, b) => a - b), bootBricks(o));
    assertConsistent(e, s);
    e.advance(1);
    s = await snapshot(e);
    assertConsistent(e, s);
    // Narrower and taller: the class pass sees the new surface, and the next head's residency follows it (the owners'
    // reach is 20 cells plus the flow past the source tiles, so y brick 5 enters and x and z bricks 4 and up leave).
    writePhi(e, s, sheet(4, 18, 4));
    // The class pass of the written state (part 2's setMark): the next head's sources are its ACTIVE tiles.
    await e.run(e.schedule.part2.filter(p => p.kernel.label === "setMark"));
    const first = await headMatchesOracle(e, await snapshot(e));
    assert.ok(first.entered.length > 0, `entered ${first.entered.length}`);
    // Entering bricks start as air (setInit), at PHI_AIR_C (16 h, the old stored far air).
    const L = e.layout, phiNow = brickField(first.after, asF32(await e.read("phi")), L.base("phi", "F", "PHI"), -1);
    for (const k of first.entered) { const b = brickKeyCoord(k); assert.equal(phiNow([8 * b[0] + 3, 8 * b[1] + 3, 8 * b[2] + 3]), PHI_AIR_C); }
    // The next head follows the flow the projection left (it is not still: no hysteresis), again as the oracle says.
    // The wide bricks leave here, not at the first head: the first head's sources include the TRIG heat of the
    // census before the write (setMark reads the last census's TRIG), which saw the wide sheet.
    const next = await headMatchesOracle(e, first.after);
    assert.ok(next.left.length > 0, `left ${next.left.length}`);
    console.log(`all-h residency: ${first.after.bricks.size} F (${first.entered.length} entered, ${first.left.length} left), radius ${first.radius} tiles; next ${next.after.bricks.size} F (${next.entered.length} entered, ${next.left.length} left), radius ${next.radius}`);
    e.destroy();
  });
});

/**
 * One step, with its head's residency checked against the CPU oracle (resMark's per-source reach from the set before
 * the head: ACT, PSUPPORT, BOUND_F): the F and C sets, RES_RADIUS, the churn counts and lists, and RES_CLAIMED_F.
 */
async function headMatchesOracle(e: BrickEngine, before: BrickSnapshot) {
  const L = e.layout;
  e.advance(1);
  const after = await snapshot(e);
  assertConsistent(e, after);
  const want = residencyOracle(L, before, after.set, false, { dt: DT, h: H_M, gravity: [0, -G, 0] }), radius = want.radius;
  assert.equal(after.header[H.RES_RADIUS], radius);
  const keys = (m: Map<number, number>) => new Set(m.keys());
  assert.deepEqual(keys(after.bricks), want.bricks);
  assert.deepEqual(keys(after.pages), want.pages);
  const entered = [...want.bricks].filter(k => !before.bricks.has(k)), left = [...before.bricks.keys()].filter(k => !want.bricks.has(k));
  assert.equal(after.header[H.N_ENTER_F], entered.length);
  assert.equal(after.header[H.N_LEAVE_F], left.length);
  assert.equal(after.header[H.RES_CLAIMED_F], entered.length, "all-h: every entry is the residency's claim");
  const enterList = Array.from(after.set.subarray(L.set.lists[LIST.F_ENTER]!, L.set.lists[LIST.F_ENTER]! + entered.length));
  assert.deepEqual(new Set(enterList), new Set(entered.map(k => after.bricks.get(k)!)));
  const leaveList = Array.from(after.set.subarray(L.set.lists[LIST.F_LEAVE]!, L.set.lists[LIST.F_LEAVE]! + left.length));
  assert.deepEqual(new Set(leaveList), new Set(left.map(k => before.bricks.get(k)!)));
  return { after, entered, left, radius };
}

test("brick engine (GPU): FAIL.POOL_F latches when the residency needs more bricks than the pool", GPU, async () => {
  await withDevice("brick pool fail", async device => {
    // As the enter/leave test: the taller sheet's reach then needs bricks the first head's set lacks.
    const lattice: Vec3 = [128, 64, 128], liquid = [sheet(20, 10.3, 20)];
    // The pool: exactly the set the first head leaves (measured on a roomy pool), at least the boot's.
    const roomy = await BrickEngine.create(device, options(lattice, liquid, 400, 16));
    roomy.advance(1);
    const f = Math.max((await snapshot(roomy)).bricks.size, bootBricks(options(lattice, liquid, 1, 1)).length);
    roomy.destroy();
    const e = await BrickEngine.create(device, options(lattice, liquid, f, 16));
    e.advance(1);
    await e.assertHealthy();
    writePhi(e, await snapshot(e), sheet(20, 18, 20));
    await e.run(e.schedule.part2.filter(p => p.kernel.label === "setMark"));
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
    for (const key of s.bricks.keys()) { const b = brickKeyCoord(key); for (let l = 0; l < BRICK_CELLS; l++) cells.push([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]); }
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
      capacities: { f: 400, c: 16, b: 0, k: 0, particles: 1024 }, particleBandDepth: 3, liquid: [sheet(20, 10.3, 20)], origin_m: [0, 0, 0] };
    const far = 2;
    const solver = await BrickNBFlipSolver.create(device, { setup, stages: bringUpStages, solver: L => bringUpCG(L), values: {}, quality: "balanced", farPhi_m: far });
    assert.equal(solver.advanceTo(0.5 * DT, []), false, "no partial step");
    assert.equal(solver.advanceTo(DT, []), true);
    await solver.awaitFrameCompletion();
    await solver.assertSimulationHealthy();
    const info = await solver.readStats();
    assert.equal(info.brickEngine?.liveF, (await snapshot(solver.brickEngine)).bricks.size);
    assert.equal(info.brickEngine?.step, 1);
    assert.ok(info.pressureIterations > 0);
    assert.deepEqual(solver.fluidDomain?.dimensions, [64, 32, 64]);
    let phi = await readSlice(device, solver.denseLevelSetVolumeSource.vertexPhi, 8), vol = await readSlice(device, solver.volumeTexture, 8);
    assert.ok(Math.abs(phi(8, 4) - boxDistance([8, 4, 8], sheet(20, 10.3, 20)) * H_M) < 1e-6, `phi ${phi(8, 4)}`);
    assert.equal(vol(8, 4), 1); assert.equal(vol(8, 20), 0);
    const surface = vol(8, 10);
    assert.ok(Math.abs(surface - 0.3) < 0.02, `the surface cell holds 0.3: ${surface}`);
    assert.equal(phi(60, 4), far, "no brick there: far phi");
    // Narrower: the bricks the head's residency retires go back to far (those on the z = 8 slice are checked).
    const e = solver.brickEngine, wide = await snapshot(e);
    writePhi(e, wide, sheet(4, 10.3, 20));
    // The class pass of the written state (setMark): the next head's sources.
    await e.run(e.schedule.part2.filter(p => p.kernel.label === "setMark"));
    // Two steps: the first head's sources still hold the TRIG heat of the census that saw the wide sheet.
    assert.equal(solver.advanceTo(2 * DT, []), true);
    await solver.awaitFrameCompletion();
    assert.equal(solver.advanceTo(3 * DT, []), true);
    await solver.awaitFrameCompletion();
    const narrow = await snapshot(e);
    const left = [...wide.bricks.keys()].filter(k => !narrow.bricks.has(k)).map(k => brickKeyCoord(k)).filter(b => b[2] === 1 && 8 * b[1] + 3 < 32);
    assert.ok(left.length > 0, "bricks on the slice leave");
    phi = await readSlice(device, solver.denseLevelSetVolumeSource.vertexPhi, 8); vol = await readSlice(device, solver.volumeTexture, 8);
    for (const b of left) {
      assert.equal(phi(8 * b[0] + 3, 8 * b[1] + 3), far, `brick ${b} left: far phi`);
      assert.equal(vol(8 * b[0] + 3, 8 * b[1] + 3), 0);
    }
    assert.ok(Math.abs(phi(8, 4) - boxDistance([8, 4, 8], sheet(4, 10.3, 20)) * H_M) < 1e-6);
    assert.equal(solver.presentations, 4);
    solver.destroy();
  });
});

test("brick engine (GPU): operator leaves zero rows (K, DIAG, RHS, X) off the liquid in every live brick", GPU, async () => {
  await withDevice("brick zero rows", async device => {
    const e = await BrickEngine.create(device, options([32, 32, 32], [sheet(33, 10.3, 33)], 200, 8));
    let s = await snapshot(e);
    // Stale values everywhere, as a reused slot would hold.
    const junk = new Float32Array(BRICK_CELLS).fill(7.5);
    for (const slot of s.bricks.values()) for (const ch of ["X", "RHS", "DIAG", "K0", "K1", "K2"]) e.write("pres", e.layout.base("pres", "F", ch) + BRICK_STRIDE * slot, junk);
    e.advance(1);
    s = await snapshot(e);
    const ch = Object.fromEntries(await Promise.all(["X", "RHS", "DIAG", "K0", "K1", "K2"].map(async c => [c, await readChannel(e, "pres", c)] as const)));
    let air = 0, rows = 0;
    for (const [key, slot] of s.bricks) {
      const b = keyOf(key), diag = ch.DIAG!(slot);
      for (let l = 0; l < BRICK_CELLS; l++) {
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

test("brick engine: the boot runs the stages' projected half kick (force, 1 + BRICK_KICK_RESOLVES solves, pmgKickScale, after) last", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
  const base = bringUpStages({ layout: L, dt_s: DT, h_m: H_M, gravity_mps2: [0, -G, 0], values: {} });
  const labels = (s: ReturnType<typeof BrickEngine.scheduleFor>) => s.boot.map(p => p.kernel.label);
  const plain = labels(BrickEngine.scheduleFor(L, cg, base, 1));
  assert.ok(!plain.includes("project"), "no kick, no boot projection");
  const force = base.velocity, after = base.velocity;
  const kicked = labels(BrickEngine.scheduleFor(L, cg, { ...base, kick: { force, after } }, 1));
  const solve = ["operator", ...[...cg.setup, ...Array.from({ length: cg.slots }, (_, k) => cg.slot(k)).flat(), ...cg.finish].map(p => p.kernel.label), "project"];
  // P's seam fix (4f17012e): the kick re-solves from its own pressure, then doubles X into a step's dt units.
  const solves = Array.from({ length: 1 + BRICK_KICK_RESOLVES }, () => solve).flat();
  const tail = [...force.map(p => p.kernel.label), ...solves, "pmgKickScale", ...after.map(p => p.kernel.label), "stepEnd"];
  assert.deepEqual(kicked.slice(0, plain.length - 1), plain.slice(0, -1));
  assert.deepEqual(kicked.slice(plain.length - 1), tail);
});

test("brick engine: with the stages' seed, the boot runs a TRIG-only census between the kick's after and the seed", () => {
  for (const tiered of [false, true]) {
    const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 0, k: 0, particles: 1024 }), cg = bringUpCG(L);
    const base = { ...bringUpStages({ layout: L, dt_s: DT, h_m: H_M, gravity_mps2: [0, -G, 0], values: {} }), tiered };
    const force = base.velocity, after = base.tail, seed = base.velocity;
    const unseeded = BrickEngine.scheduleFor(L, cg, { ...base, kick: { force, after } }, 1).boot;
    const seeded = BrickEngine.scheduleFor(L, cg, { ...base, kick: { force, after, seed } }, 1).boot;
    const labels = (b: readonly { kernel: { label: string } }[]) => b.map(p => p.kernel.label);
    // The seeded boot is the unseeded one with the census and the seed between the kick's after and stepEnd.
    const census = ["censusBegin", "censusFine", "censusCoarse", "censusCubes", "censusImportance", "censusDecide"];
    assert.deepEqual(labels(seeded), [...labels(unseeded).slice(0, -1), ...census, ...labels(seed), "stepEnd"], `tiered ${tiered}`);
    const trig = seeded.slice(unseeded.length - 1, unseeded.length - 1 + census.length);
    const decide = trig[census.length - 1]!;
    assert.equal(decide.z, 2, "censusDecide is TRIG-only (cfg.z = 2): no band, WANT, candidates or keep marks");
    assert.equal(decide.w ?? 0, 0);
    // No claim, transfer, commit or WANT check between the kick's after and the seed.
    for (const p of trig) assert.ok(!/^set|^res|^transfer|censusWant/.test(p.kernel.label), p.kernel.label);
  }
});

test("brick engine: auto (the default) takes brick exactly where the scene has no gaps, and says why not", () => {
  const method = uniformNarrowBandMethod;
  assert.equal(method.params.find(p => p.key === "engine")?.default, "auto");
  // The seep in an open world: the box's gap (an open world takes no voxels).
  const dam = sceneDocument(getSceneDefinition("high-resolution-dam-break")), seep0 = createReservoirSeepScene();
  const seep = { ...seep0, container: { ...seep0.container, walls: "none" as const } };
  const defaults = resolveMethodValues(method, "balanced", { ...method.appDefaults, timeStep: "sixtieth" });
  assert.equal(defaults.engine, "auto");
  // The dam has no gaps (the lanes' registrations included): auto runs brick.
  assert.deepEqual(brickEngineGaps(dam, defaults), []);
  assert.deepEqual(nbEngineChoice(dam, defaults), { requested: "auto", ran: "brick" });
  assert.equal(nbEngineStatusText(nbEngineChoice(dam, defaults)), "Brick (auto)");
  // A scene with gaps runs uniform, and the status names the first gap and the count of the rest.
  const gaps = brickEngineGaps(seep, defaults), s = nbEngineChoice(seep, defaults);
  assert.ok(gaps.length >= 1);
  assert.deepEqual(s, { requested: "auto", ran: "uniform", gap: gaps[0], gaps: gaps.length });
  assert.equal(nbEngineStatusText(s), `Uniform (auto: ${gaps[0]}${gaps.length > 1 ? `; ${gaps.length - 1} more` : ""})`);
  assert.equal(nbEngineStatusText({ requested: "auto", ran: "uniform", gap: "a", gaps: 3 }), "Uniform (auto: a; 2 more)");
  // Explicit engines keep their behaviour: uniform always, brick always (createBrickSolver refuses a gap scene).
  for (const scene of [dam, seep]) {
    assert.deepEqual(nbEngineChoice(scene, { ...defaults, engine: "uniform" }), { requested: "uniform", ran: "uniform" });
    assert.deepEqual(nbEngineChoice(scene, { ...defaults, engine: "brick" }), { requested: "brick", ran: "brick" });
  }
  assert.equal(nbEngineStatusText({ requested: "uniform", ran: "uniform" }), "Uniform");
  assert.equal(nbEngineStatusText({ requested: "brick", ran: "brick" }), "Brick");
  assert.throws(() => nbEngineChoice(dam, { ...defaults, engine: "bricks" }), /unknown engine "bricks"/);
});

// ---- A1, A5, A6 (m5-bodies-design-2026-11-17.md §8.1): the cut pool's channels, the K slots and the stamp stage. ----

test("brick ABI (A1, A5): the cut pool holds FRAC, AP0-2, W0-2 a K slot and one BMASK word a slot after them", () => {
  const L = new BrickLayout({ lattice: [64, 32, 64], f: 100, c: 8, b: 0, k: 40, particles: 1024 });
  assert.deepEqual([...POOLS.cut], ["FRAC", "AP0", "AP1", "AP2", "W0", "W1", "W2"]);
  assert.equal(L.base("cut", "K", "W2"), 6 * 40 * BRICK_STRIDE);
  assert.equal(L.cutBmask, 7 * 40 * BRICK_STRIDE);
  assert.equal(L.poolWords.cut, 7 * 40 * BRICK_STRIDE + 40);
  assert.equal(K_HOLD, F_CLASS.CUT_STATIC | F_CLASS.NEAR_BODY | F_CLASS.BODY_WAS);
  const bits = Object.values(F_CLASS);
  assert.equal(new Set(bits).size, bits.length, "class bits are distinct");
  assert.equal(F_CLASS.CUT_STATIC, 1 << 9); assert.equal(F_CLASS.BODY_WAS, 1 << 24);
});

/** A stand-in stamp (tests only): an entering brick with an even x coordinate takes a K slot and is CUT_STATIC. */
function testStamp(L: BrickLayout): BrickPlacement {
  const bindings: BrickBinding[] = [{ name: "cfg", kind: "uniform" }, { name: "set", kind: "rw", type: "array<atomic<u32>>" }, { name: "dir", kind: "rw", type: "array<atomic<u32>>" }];
  const kernel: BrickKernel = { label: "testStamp", entry: "testStamp", bindings, code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, /* wgsl */`
@compute @workgroup_size(64) fn testStamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=64u*nbGroup(wid)+lid;if(e>=nbListCount(NB_LIST_F_ENTER)){return;}
  let s=nbListSlot(NB_LIST_F_ENTER,e);let r=NB_D_FREC+32u*s;
  if((nbKeyCoord(atomicLoad(&dir[r+NB_REC_KEY])).x&1)!=0){return;}
  if(nbKTake(s)!=NB_NONE){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_CUT_STATIC);}
}`].join("\n") };
  return { kernel, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ENTER + ARGS.G64 } };
}

test("brick engine (A6): the stamp stage closes the head and follows the boot's census; no K pool and no stamp, no change", () => {
  const L0 = new BrickLayout({ lattice: [64, 32, 64], f: 100, c: 8, b: 0, k: 0, particles: 1024 });
  const base = (L: BrickLayout) => bringUpStages({ layout: L, dt_s: DT, h_m: H_M, gravity_mps2: [0, -G, 0], values: {} });
  const s0 = BrickEngine.scheduleFor(L0, bringUpCG(L0), base(L0), 1);
  assert.ok(![...s0.boot, ...s0.part1, ...s0.part2].some(p => p.kernel.label === "setKRelease"), "k = 0 (the dam): no release");
  const L = new BrickLayout({ lattice: [64, 32, 64], f: 100, c: 8, b: 0, k: 40, particles: 1024 }), stamp = testStamp(L);
  const s = BrickEngine.scheduleFor(L, bringUpCG(L), { ...base(L), stamp: [stamp] }, 1);
  const labels = (ps: readonly BrickPlacement[]) => ps.map(p => `${p.kernel.label}${p.z ? `:${p.z}` : ""}`);
  const p1 = labels(s.part1), at = p1.indexOf("testStamp");
  assert.deepEqual(p1.slice(at, at + 4), ["testStamp", "setKRelease", "setKRelease:1", "stepEnd:1"], "stamp, the releases, then the churn lists clear");
  assert.ok(p1.lastIndexOf("setArgs") < at, "after the head's lists");
  assert.ok(!labels(s.part2).includes("testStamp"));
  const b = labels(s.boot), bs = b.indexOf("testStamp");
  assert.ok(bs > b.indexOf("bootPhi") && bs > b.lastIndexOf("censusBegin"), "boot: after the boot phi, its lists and its census");
  assert.equal(b[bs + 1], "setKRelease");
});

test("brick engine (GPU, A5): K slots are taken once a brick, refused past KCAP (POOL_K), and given back after the stamps", GPU, async () => {
  await withDevice("brick K slots", async device => {
    const lattice: Vec3 = [64, 32, 64], liquid = [sheet(20, 10.3, 20)];
    const make = async (k: number) => {
      const o = options(lattice, liquid, 200, 16);
      return BrickEngine.create(device, { ...o, capacities: { ...o.capacities, k }, stages: x => ({ ...bringUpStages(x), stamp: [testStamp(x.layout)] }) });
    };
    const even = bootBricks(options(lattice, liquid, 200, 16)).filter(key => (brickKeyCoord(key)[0] & 1) === 0);
    const K = even.length + 5, e = await make(K), L = e.layout;
    let s = await snapshot(e);
    const rec = (slot: number, w: number) => s.dir[L.dir.frec + 32 * slot + w]!;
    assert.equal(s.header[H.FAIL], 0);
    const held = new Map<number, number>();
    for (const [key, slot] of s.bricks) {
      const isEven = (brickKeyCoord(key)[0] & 1) === 0, ks = rec(slot, REC.CUT), cls = rec(slot, REC.CLASS);
      assert.equal((cls & F_CLASS.CUT_STATIC) !== 0, isEven, `brick ${brickKeyCoord(key)} class 0x${cls.toString(16)} (kept by the class pass)`);
      if (isEven) { assert.ok(ks < K); held.set(slot, ks); } else assert.equal(ks, NONE);
    }
    assert.equal(held.size, even.length);
    assert.equal(s.header[H.LIVE_K], even.length); assert.equal(s.header[H.TOP_K], K - even.length); assert.equal(s.header[H.PEAK_K], even.length);
    const free = () => Array.from(s.set.subarray(L.set.freeK, L.set.freeK + s.header[H.TOP_K]!));
    assert.deepEqual([...free(), ...held.values()].sort((a, b) => a - b), Array.from({ length: K }, (_, i) => i), "free stack and held slots partition K");
    // F_ALL: a brick with no K_HOLD bit gives its slot back; the others keep theirs.
    const slots = [...held.keys()], cleared = slots.slice(0, 2), leaving = slots.slice(2, 5);
    for (const slot of cleared) e.write("dir", L.dir.frec + 32 * slot + REC.CLASS, new Uint32Array([0]));
    const set = brickSetKernels(L);
    await e.run([{ kernel: set.setArgs, dispatch: { direct: [1, 1, 1] }, z: 1 }, { kernel: set.setKRelease, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G64 }, z: 0 }]);
    s = await snapshot(e);
    for (const slot of cleared) assert.equal(rec(slot, REC.CUT), NONE);
    for (const slot of slots.slice(2)) assert.equal(rec(slot, REC.CUT), held.get(slot));
    assert.equal(s.header[H.TOP_K], K - even.length + 2);
    // F_LEAVE: a leaving brick gives its slot back whatever its class.
    e.write("set", L.set.lists[LIST.F_LEAVE]!, new Uint32Array(leaving));
    e.write("set", H.LIST_COUNT + LIST.F_LEAVE, new Uint32Array([leaving.length]));
    await e.run([{ kernel: set.setArgs, dispatch: { direct: [1, 1, 1] }, z: 1 }, { kernel: set.setKRelease, dispatch: { indirect: ARGS.PER_LIST * LIST.F_LEAVE + ARGS.G64 }, z: 1 }]);
    s = await snapshot(e);
    for (const slot of leaving) assert.equal(rec(slot, REC.CUT), NONE);
    assert.equal(s.header[H.TOP_K], K - even.length + 5);
    assert.deepEqual([...free(), ...slots.slice(5).map(x => held.get(x)!)].sort((a, b) => a - b), Array.from({ length: K }, (_, i) => i));
    e.destroy();
    // Past KCAP: the pop is refused, POOL_K latches with the key, and exactly KCAP bricks are cut.
    const small = await make(even.length - 3);
    const t = await snapshot(small);
    assert.equal(t.header[H.FAIL]! & FAIL.POOL_K, FAIL.POOL_K);
    assert.equal(t.header[H.REFUSED_K], 3); assert.equal(t.header[H.TOP_K], 0);
    assert.equal([...t.bricks.values()].filter(slot => (t.dir[small.layout.dir.frec + 32 * slot + REC.CLASS]! & F_CLASS.CUT_STATIC) !== 0).length, even.length - 3);
    small.destroy();
  });
});

test("brick engine (GPU, M3 2b): solidStamp cuts exactly the bricks the mask meets, with the hooks' FRAC and apertures", GPU, async () => {
  const { SolidOccupancyMask, solidOccupancyMaskVoxel } = await import("../../core/solid-occupancy-mask");
  const { createSolidWorld } = await import("../../core/solid-world");
  const { brickStaticMaskWalls, brickStaticMirror } = await import("./brick-static.wgsl");
  const lattice: Vec3 = [64, 32, 64], liquid = [sheet(20, 10.3, 20)];
  // A box in brick 1, a round fill across bricks 2 and 3 (partial coverage), and a column at x = 15 whose voxels
  // sit below brick 2's lower x faces only (brick 2 is cut by its faces, not its cells).
  const walls = brickStaticMaskWalls(lattice, false);
  const mask = new SolidOccupancyMask({ walls, tileKeys: 0, slots: 64, separateTerrain: true });
  mask.update(createSolidWorld([
    { operation: "fill", minimum: [10, 0, 10], maximumExclusive: [14, 5, 14] },
    { operation: "fill", kind: "ellipsoid", minimum: [20, 2, 4], maximumExclusive: [27, 9, 11] },
    { operation: "fill", minimum: [15, 0, 30], maximumExclusive: [16, 3, 33] },
  ]));
  mask.seal(mask.slotsHeld);
  const m = brickStaticMirror({ box: lattice, openTop: false, frame: [0, 0, 0], h: [1, 1, 1], mask: mask.words });
  const inBox = (c: readonly number[]) => c.every((v, a) => v >= 0 && v < lattice[a]!);
  const maskVoxel = (c: readonly number[]) => inBox(c) && solidOccupancyMaskVoxel(mask.words, c[0]!, c[1]!, c[2]!);
  const down = (c: readonly number[], a: number) => c.map((v, k) => v - (k === a ? 1 : 0));
  await withDevice("brick solidStamp", async device => {
    const o = options(lattice, liquid, 200, 16);
    const e = await BrickEngine.create(device, { ...o, capacities: { ...o.capacities, k: 64 }, staticMask: { words: mask.words, walls } });
    const L = e.layout, s = await snapshot(e), cut = asF32(await e.read("cut")), phi = await e.read("phi");
    assert.equal(s.header[H.FAIL], 0);
    let cutBricks = 0, faceCut = 0, bad = 0, wallHalf = 0;
    for (const [key, slot] of s.bricks) {
      const b = brickKeyCoord(key), r = L.dir.frec + 32 * slot, ks = s.dir[r + REC.CUT]!, cls = s.dir[r + REC.CLASS]!;
      const cells = Array.from({ length: BRICK_CELLS }, (_, l) => [8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]);
      const byCell = cells.some(maskVoxel), want = byCell || cells.some(c => [0, 1, 2].some(a => maskVoxel(down(c, a))));
      assert.equal((cls & F_CLASS.CUT_STATIC) !== 0, want, `brick ${b}`);
      if (!want) { assert.equal(ks, NONE); continue; }
      cutBricks++; if (!byCell) faceCut++;
      assert.ok(ks < 64);
      cells.forEach((c, l) => {
        const at = BRICK_STRIDE * ks + l;
        // At a box wall the face is the old shell voxel's, 0.5 FRAC of the cell inside (the hook's own).
        const want = [m.cellOpen(c), ...[0, 1, 2].map(a => m.faceOpen(down(c, a), a)), 0, 0, 0];
        for (let a = 0; a < 3; a++) if (!inBox(down(c, a)) && want[1 + a] === 0.5) wallHalf++;
        const have = ["FRAC", "AP0", "AP1", "AP2", "W0", "W1", "W2"].map(ch => cut[L.base("cut", "K", ch) + at]!);
        const open = (phi[L.base("phi", "F", "FLAGS") + BRICK_STRIDE * slot + l]! & 1) !== 0;
        if (want.some((v, j) => Math.fround(v) !== have[j]) || open !== (want[0]! > 0)) { if (bad++ < 3) console.error(b, c, want, have, open); }
      });
      assert.equal(new Uint32Array(cut.buffer)[L.cutBmask + ks], 0);
    }
    assert.equal(bad, 0);
    assert.ok(cutBricks >= 4 && faceCut >= 1 && wallHalf > 0, `${cutBricks} cut bricks, ${faceCut} cut by a face alone, ${wallHalf} wall faces at 0.5`);
    assert.equal(s.header[H.LIVE_K], cutBricks);
    e.destroy();
  });
});

test("brick engine: an open top's latch is setMark's alone, and its failure names the headroom", () => {
  const base = { lattice: [32, 64, 32] as const, f: 64, c: 8, b: 16, k: 0, particles: 1024 };
  for (const tiered of [false, true]) {
    const closed = brickSetKernels(new BrickLayout(base), { tiered }).setMark.code, open = brickSetKernels(new BrickLayout({ ...base, openTop: true }), { tiered }).setMark.code;
    assert.ok(!closed.includes("0x80000000u|nbKey"), "a closed top's setMark has no latch (its text unchanged)");
    assert.equal(open.split("nbFail(NB_FAIL_REACH,0x80000000u|nbKey(o>>vec3u(3u)))").length - 1, 2, "the particle/4h-corner latch and the h-vertex latch");
    assert.equal(open.split("nbCVertex(T+").length - closed.split("nbCVertex(T+").length, tiered ? 1 : 0, "tiered, the latch reads the 4h corners");
  }
  assert.throws(() => new BrickLayout({ ...base, box: [32, 32, 32], openTop: true }), /open top's box/);
  assert.match(describeFail(FAIL.REACH, (0x80000000 | (3 | (7 << 10) | (2 << 20))) >>> 0), /open top's last brick layer \(brick 3,7,2\)/);
  assert.doesNotMatch(describeFail(FAIL.REACH, 3 | (7 << 10)), /open top/);
});
