/**
 * The analytic cloth scenes (lib/cloth/scenes/analytic/): acceptance gates for
 * the GPU cloth solver against closed-form answers. CPU: the references
 * themselves, provenance, catalog cards, valid documents, solids on the shared
 * hit-test path and their contact authoring. GPU (skips without
 * WEBGPU_NODE_MODULE): each scene run on the scene clock at its K, measured
 * against its answer with the tolerance its card states.
 *
 * For lane C1's solver sweeps:
 *   CLOTH_ANALYTIC_NEWTON=8x200      run every scene at these Newton x PCG caps instead of its own (its tolerances)
 *   CLOTH_ANALYTIC_ONLY=free-fall,swinging-sheet   run only these kinds
 *   CLOTH_ANALYTIC_CONVERGED=0      skip the swinging sheet's 4K reference run
 * A failing gate is a finding about the solver, never a reason to change the scene.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { validateScene, type SceneDescription } from '../lib/core/model';
import { findSceneDefinition, sceneCatalogCards } from '../lib/core/scenes';
import { sceneDocument } from '../lib/core/scene-definition';
import { visibleToolSurface } from '../lib/core/voxel-editor/surface';
import { clothSolidSourceFromScene } from '../lib/cloth/contact/scene-solids';
import { buildClothSolidField } from '../lib/cloth/contact/solid-field';
import { prepareClothTopology, validateClothDocument } from '../lib/cloth/topology';
import { buildClothSolverModel } from '../lib/cloth/oracle';
import { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { CLOTH_BINDING } from '../lib/cloth/gpu-solver-wgsl';
import { buildClothSceneField, clothContactRadius, createClothSceneContact } from '../lib/cloth/scene-contact';
import type { ClothDocument, ClothNewtonSettings, ClothSurfacePublication } from '../lib/cloth/types';
import {
  CLOTH_ANALYTIC_PARAMETERS, CLOTH_ANALYTIC_SCENES, CLOTH_ANALYTIC_STEP_S, GRAVITY, INCLINE, PEIRCE, SWING, bdf2FreeFall, bdf2StartOffset_m, bendingLength_m,
  clothAnalyticSceneDescription, hangingBandStrain, hangingStrainContinuum, heavyElastica, inclineAcceleration_m_s2, peirceChordAngle_deg,
  clothAnalyticFreeFall, clothAnalyticFrictionIncline, clothAnalyticHangingCurtain, clothAnalyticPeirceCantilever, clothAnalyticSwingingSheet,
  type ClothAnalyticKind, type ClothAnalyticScene, type ClothAnalyticParameter,
} from '../lib/cloth/scenes/analytic';
import { clothAnalyticSceneSolids } from '../lib/cloth/scenes/analytic/contact-solids';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const caps = process.env.CLOTH_ANALYTIC_NEWTON?.split('x').map(Number);
if (caps && !(caps.length === 2 && caps.every(c => Number.isInteger(c) && c >= 1))) throw new Error(`CLOTH_ANALYTIC_NEWTON must be <newton>x<pcg>, got ${process.env.CLOTH_ANALYTIC_NEWTON}`);
const only = process.env.CLOTH_ANALYTIC_ONLY?.split(',').map(s => s.trim()).filter(Boolean);
const runs = (kind: ClothAnalyticKind) => gpu && (!only || only.includes(kind));
const g = GRAVITY.value, h = CLOTH_ANALYTIC_STEP_S;
const ray = (o: [number, number, number], d: [number, number, number]) => ({ origin: { x: o[0], y: o[1], z: o[2] }, direction: { x: d[0], y: d[1], z: d[2] } });
/** The Newton settings a scene runs at: its own, or CLOTH_ANALYTIC_NEWTON's caps with its tolerances. */
const newtonOf = (s: ClothAnalyticScene): ClothNewtonSettings => { const own = s.cloth[0]!.solver!.newton!; return caps ? { ...own, iterations: caps[0]!, pcgIterations: caps[1]! } : own; };
const override = caps ? (s: ClothAnalyticScene) => ({ newton: newtonOf(s) }) : () => ({});

// ---------------------------------------------------------------- CPU

test('every analytic cloth parameter names its source', () => {
  const sources = new Set(['bridson', 'c1', 'textbook', 'astm', 'choice']);
  for (const [group, table] of Object.entries(CLOTH_ANALYTIC_PARAMETERS)) for (const [name, entry] of Object.entries(table)) {
    for (const p of (Array.isArray(entry) ? entry : [entry]) as ClothAnalyticParameter[]) {
      assert.ok(sources.has(p.source), `${group}.${name} source`);
      assert.ok(p.ref.length > 10 && Number.isFinite(p.value) && p.unit.length > 0, `${group}.${name} provenance`);
    }
  }
});

test('the references: BDF2 free fall, the hanging equilibrium, the heavy elastica and the incline', () => {
  // BDF2 is exact for constant gravity once started: velocity g t, position g t^2/2 plus the Euler start's (3/4) g h^2 (1 - 3^-n).
  const { displacement_m: x, velocity_m_s: v } = bdf2FreeFall(60, h, -g);
  for (let n = 1; n <= 60; n++) {
    const t = n * h;
    assert.ok(Math.abs(v[n - 1]! + g * t) < 1e-12, `velocity at step ${n}`);
    assert.ok(Math.abs(x[n - 1]! - (-g * t * t / 2 - 0.75 * g * h * h * (1 - 3 ** -n))) < 1e-12, `position at step ${n}`);
  }
  assert.ok(Math.abs(bdf2StartOffset_m(h, g) - 2.04375e-3) < 1e-8);
  // The hanging sheet's band strain is the continuum rho g (L - s) / k to first order (StVK's lambda (lambda^2 - 1) / 2 = e + 3e^2/2 + ...).
  for (const j of [10, 50, 90]) {
    const exact = hangingBandStrain(j, 100, 0.01, 200, 1, g), continuum = hangingStrainContinuum((j + 0.5) * 0.01, 99 * 0.01, 200, 1, g);
    assert.ok(exact < continuum && continuum - exact < 1.6 * continuum * continuum, `band ${j}: ${exact} vs ${continuum}`);
  }
  // ASTM D1388: the tip of a strip twice its bending length overhanging falls to the 41.5 degree line.
  assert.ok(Math.abs(heavyElastica(2).chordAngle_deg - 41.5) < 0.05, `l = 2c: ${heavyElastica(2).chordAngle_deg}`);
  assert.ok(heavyElastica(0.3).chordAngle_deg < 0.3 && heavyElastica(3).chordAngle_deg > 60);
  // Peirce's fit agrees with the elastica at small droop and drifts high at large.
  assert.ok(Math.abs(peirceChordAngle_deg(1) - heavyElastica(1).chordAngle_deg) < 0.05);
  assert.ok(peirceChordAngle_deg(2.5) > heavyElastica(2.5).chordAngle_deg + 2);
  assert.ok(Math.abs(bendingLength_m(1e-3, 0.2, g) - 0.0799) < 1e-4);
  assert.equal(inclineAcceleration_m_s2(20, 0.5, g), 0);
  assert.ok(Math.abs(inclineAcceleration_m_s2(35, 0.5, g) - 1.609) < 1e-3);
});

test('the analytic cloth scenes are registered validation cards on their own shelf and open as valid dry cloth scenes', () => {
  const ids = CLOTH_ANALYTIC_SCENES.map(make => make().id);
  assert.deepEqual(ids, ['cloth-analytic-free-fall', 'cloth-analytic-hanging-curtain', 'cloth-analytic-swinging-sheet', 'cloth-analytic-peirce-cantilever', 'cloth-analytic-friction-incline']);
  for (const make of CLOTH_ANALYTIC_SCENES) {
    const source = make(), card = sceneCatalogCards.find(c => c.id === source.id);
    assert.ok(card, `${source.id} has a library card`);
    assert.equal(card.shelf, 'Cloth · Analytic');
    assert.match(card.status ?? '', /Gate/);
    const scene = sceneDocument(findSceneDefinition(source.id)!);
    assert.deepEqual(validateScene(scene), [], source.id);
    assert.equal(scene.systems?.fluid, false);
    assert.equal(scene.cloth?.length, source.cloth.length);
    for (const d of scene.cloth!) {
      assert.deepEqual(validateClothDocument(d), [], d.id);
      assert.equal(d.solver?.integrator, 'bdf2');
      assert.ok(d.materials[0]!.baseColor, `${d.id} has its own colour`);
      // The cloth starts inside the container, clear of the floor.
      const p = d.geometry.data.restPositions_m;
      for (let i = 0; i < p.length; i += 3) {
        assert.ok(Math.abs(p[i]!) < scene.container.width_m / 2 && Math.abs(p[i + 2]!) < scene.container.depth_m / 2, `${d.id} inside x/z`);
        assert.ok(p[i + 1]! > 0.05 && p[i + 1]! < scene.container.height_m, `${d.id} inside y`);
      }
    }
  }
});

test('the ledge and the inclines are voxel solids on the shared hit-test path, and their contact authoring covers every patch', () => {
  const peirce = clothAnalyticSceneDescription(clothAnalyticPeirceCantilever());
  // The ledge top (0.5 m), hit from above beside the strips.
  const ledge = visibleToolSurface(peirce, ray([-0.2537, 0.75, 0.2013], [0, -1, 0]));
  assert.ok(ledge && !ledge.cloth && Math.abs(ledge.point.y - 0.5) < 0.011, `ledge at ${ledge?.point.y}`);
  const incline = clothAnalyticFrictionIncline(), scene = clothAnalyticSceneDescription(incline);
  for (const [s, solid] of incline.solids.entries()) {
    if (solid.shape.kind !== 'wedge') throw new Error('incline solids are wedges');
    const { x0_m, top_m, angle_deg, zc_m } = solid.shape, x = x0_m + 0.4037;
    const expected = top_m - (x - x0_m) * Math.tan(angle_deg * Math.PI / 180);
    const hit = visibleToolSurface(scene, ray([x, 0.9, zc_m + 0.0123], [0, -1, 0]));
    assert.ok(hit && Math.abs(hit.point.y - expected) < 0.012, `incline ${s} at ${hit?.point.y}, expected ${expected}`);
  }
  // The contact lane's source build accepts the authoring: every box fill has a friction, every baked node its analytic solid.
  for (const doc of [peirce, scene]) {
    const src = clothSolidSourceFromScene(doc, clothAnalyticSceneSolids(doc), 0.004);
    assert.ok(src.parts!.some(p => p.mu === 0.5), `${doc.sceneId} carries mu 0.5`);
  }
  assert.throws(() => clothAnalyticSceneSolids({ ...scene, sceneId: 'cloth-analytic-none' }), /no analytic cloth scene/);
});

// ---------------------------------------------------------------- GPU

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

/**
 * Run a scene on the scene clock and capture every document at each of `ticks`.
 * CLOTH_ANALYTIC_CONTACT=0 runs each document through the bare solver instead
 * (no self or solid contact pass): the diagnosis that separates the two.
 */
async function capture(device: GPUDevice, scene: SceneDescription, ticks: readonly number[]): Promise<{ frames: ClothSurfacePublication[][]; ms: number }> {
  if (process.env.CLOTH_ANALYTIC_CONTACT === '0') return captureSolverOnly(device, scene, ticks);
  const cloth = await ClothSystem.create(device, scene);
  try {
    const t0 = performance.now();
    const frames = await cloth.capture(ticks.map(n => n * CLOTH_SCENE_STEP_S));
    return { frames, ms: (performance.now() - t0) / Math.max(1, ticks.at(-1)!) };
  } finally { cloth.destroy(); }
}

async function captureSolverOnly(device: GPUDevice, scene: SceneDescription, ticks: readonly number[]): Promise<{ frames: ClothSurfacePublication[][]; ms: number }> {
  const solvers = await Promise.all((scene.cloth ?? []).map(d => solverFor(device, d)));
  try {
    const frames: ClothSurfacePublication[][] = [];
    let tick = 0, stepping = 0;
    for (const target of ticks) {
      const t0 = performance.now();
      for (const solver of solvers) solver.step(target - tick);
      await device.queue.onSubmittedWorkDone();
      stepping += performance.now() - t0; tick = target;
      frames.push(await Promise.all(solvers.map(async (solver, k) => {
        const state = await solver.readState();
        return { documentId: scene.cloth![k]!.id, positions_m: state.positions_m, velocities_m_s: state.velocities_m_s } as unknown as ClothSurfacePublication;
      })));
    }
    return { frames, ms: stepping / Math.max(1, ticks.at(-1)!) };
  } finally { for (const solver of solvers) solver.destroy(); }
}

const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const label = (s: ClothAnalyticScene, n = newtonOf(s)) => `${s.kind}, Newton ${n.iterations} x ${n.pcgIterations}`;

test('free fall: the sheet falls exactly as the BDF2 recurrence, flat and unstrained', { skip: !runs('free-fall') }, async () => withDevice(async device => {
  const source = clothAnalyticFreeFall(), n = source.probe.gateSteps as number, y0 = source.probe.y0 as number;
  const scene = clothAnalyticSceneDescription(source, override(source));
  const { frames, ms } = await capture(device, scene, range(1, n));
  const data = source.cloth[0]!.geometry.data, rest = data.restPositions_m, tri = data.triangles;
  const ref = bdf2FreeFall(n, h, -g);
  let centroid = 0, velocity = 0, flat = 0, strain = 0, continuum = 0;
  frames.forEach(([pose], k) => {
    const x = pose!.positions_m, v = pose!.velocities_m_s!, N = x.length / 3;
    let yc = 0, vc = 0;
    for (let i = 0; i < N; i++) { yc += x[3 * i + 1]! / N; vc += v[3 * i + 1]! / N; }
    for (let i = 0; i < N; i++) flat = Math.max(flat, Math.abs(x[3 * i + 1]! - yc), Math.abs(x[3 * i]! - rest[3 * i]!), Math.abs(x[3 * i + 2]! - rest[3 * i + 2]!));
    centroid = Math.max(centroid, Math.abs(yc - (y0 + ref.displacement_m[k]!)));
    velocity = Math.max(velocity, Math.abs(vc - ref.velocity_m_s[k]!));
    const t = (k + 1) * h; continuum = yc - (y0 - g * t * t / 2);
    for (let f = 0; f < tri.length; f += 3) for (const [a, b] of [[tri[f]!, tri[f + 1]!], [tri[f + 1]!, tri[f + 2]!], [tri[f + 2]!, tri[f]!]]) {
      const l = Math.hypot(x[3 * a!]! - x[3 * b!]!, x[3 * a! + 1]! - x[3 * b! + 1]!, x[3 * a! + 2]! - x[3 * b! + 2]!);
      const l0 = Math.hypot(rest[3 * a!]! - rest[3 * b!]!, rest[3 * a! + 1]! - rest[3 * b! + 1]!, rest[3 * a! + 2]! - rest[3 * b! + 2]!);
      strain = Math.max(strain, Math.abs(l / l0 - 1));
    }
  });
  console.log(`[${label(source)}] ${n} steps, ${ms.toFixed(1)} ms/tick: centroid off the BDF2 recurrence ${(1000 * centroid).toFixed(4)} mm, velocity off g t ${(1000 * velocity).toFixed(4)} mm/s, `
    + `largest vertex departure from rigid translation ${(1000 * flat).toFixed(4)} mm, max edge strain ${strain.toExponential(2)}; centroid minus y0 - g t^2/2 at step ${n}: ${(1000 * continuum).toFixed(3)} mm (Euler start: ${(-1000 * bdf2StartOffset_m(h, g) * (1 - 3 ** -n)).toFixed(3)} mm)`);
  assert.ok(centroid < 1e-4, `centroid ${centroid} m off the recurrence`);
  assert.ok(velocity < 1e-3, `velocity ${velocity} m/s off g t`);
  assert.ok(flat < 1e-4, `sheet departs from rigid translation by ${flat} m`);
  assert.ok(strain < 1e-4, `edge strain ${strain}`);
}));

/** The free-fall sheet's drop onto the floor: per tick its mean height and mean vertical velocity. */
function dropTrack(frames: readonly (readonly { positions_m: ArrayLike<number>; velocities_m_s?: ArrayLike<number> }[])[]) {
  return frames.map(([pose]) => {
    const x = pose!.positions_m, v = pose!.velocities_m_s!, N = x.length / 3;
    let y = 0, vy = 0;
    for (let i = 0; i < N; i++) { y += x[3 * i + 1]! / N; vy += v[3 * i + 1]! / N; }
    return { y, vy };
  });
}
/** The largest upward mean velocity after the fastest mean fall, and that fall. */
function rebound(track: readonly { vy: number }[]) {
  let k = 0;
  track.forEach((t, i) => { if (t.vy < track[k]!.vy) k = i; });
  return { fall: track[k]!.vy, up: Math.max(0, ...track.slice(k + 1).map(t => t.vy)) };
}
/** The free-fall scene with its sheet turned by `tilt` about the z axis through its centre (one edge lands first). */
function tiltedDrop(tilt: number): ClothAnalyticScene {
  const source = clothAnalyticFreeFall(), cy = source.probe.y0 as number, c = Math.cos(tilt), s = Math.sin(tilt);
  const turn = (p: readonly number[]) => p.map((u, i) => { const k = i % 3, x = p[i - k]!, y = p[i - k + 1]! - cy; return k === 0 ? c * x - s * y : k === 1 ? cy + s * x + c * y : u; });
  return { ...source, cloth: source.cloth.map(d => ({ ...d, geometry: { kind: 'inline' as const, data: { ...d.geometry.data, restPositions_m: turn(d.geometry.data.restPositions_m),
    ...(d.geometry.data.initialPositions_m ? { initialPositions_m: turn(d.geometry.data.initialPositions_m) } : {}) } } })) };
}
/**
 * The reference for a landing (test only, not a product path): the same document, floor and contact, stepped as implicit
 * Euler at h / substeps, read at each tick.
 */
async function landingOracle(device: GPUDevice, scene: SceneDescription, document: ClothDocument, ticks: number, substeps: number) {
  const data = document.geometry.data, topology = prepareClothTopology(data, document.materials);
  const model = buildClothSolverModel(topology, document.materials, []);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  data.restPositions_m.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3]!, v); hi[i % 3] = Math.max(hi[i % 3]!, v); });
  const origin = [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as unknown as [number, number, number], dt = CLOTH_SCENE_STEP_S / substeps;
  const field = buildClothSceneField(scene, origin, clothContactRadius(document));
  const contact = await createClothSceneContact(device, model, document, origin, field, dt, [0, -g, 0], undefined, true), hooks = contact.stepHooks;
  const solver = await ClothGPUSolver.create(device, model, { origin_m: origin, dt_s: dt, gravity_m_s2: [0, -g, 0], integrator: 'implicit-euler',
    initialGuess: 'inertial', positions_m: data.initialPositions_m ?? data.restPositions_m, newton: document.solver!.newton!,
    contact: { ...contact.solverContact(CLOTH_BINDING.contact, true), alpha: contact.alpha, state: contact.contactState },
    hooks: { afterFrame: hooks.afterFrame, filter: hooks.filter, afterNewtonStep: hooks.afterNewtonStep, stepEnd: hooks.stepEnd } });
  contact.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: solver.stepStart, target: solver.inertialTarget, direction: solver.newtonDirection });
  contact.setSolidField(field);
  const frames: { positions_m: Float64Array; velocities_m_s: Float64Array }[][] = [];
  try {
    for (let t = 0; t < ticks; t++) {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      contact.encodeFrameBegin(pass);
      for (let k = 0; k < substeps; k++) solver.encodeStep(pass);
      pass.end(); device.queue.submit([encoder.finish()]);
      frames.push([await solver.readState()]);
    }
  } finally { solver.destroy(); contact.destroy(); }
  return frames;
}

test('free fall onto the floor: the sheet lands without a rebound (the BDF2 history restarts at the contact onset)', { skip: !runs('free-fall') }, async () => withDevice(async device => {
  // BDF2's velocity (3x_{n+1} - 4x_n + x_{n-1}) / 2h straddles the impact's velocity jump and replays half of it as a
  // rebound (2.98 m/s from 4.74 m/s before the restart, RESULT (23)); a frictionless floor and an inelastic barrier leave none.
  const source = clothAnalyticFreeFall(), scene = clothAnalyticSceneDescription(source, override(source));
  const { frames, ms } = await capture(device, scene, range(1, 45));
  const track = dropTrack(frames), r = rebound(track);
  console.log(`[${label(source)}] landing, ${ms.toFixed(1)} ms/tick: fastest fall ${r.fall.toFixed(3)} m/s, largest upward mean velocity after it ${r.up.toFixed(4)} m/s, rests at ${(1000 * track.at(-1)!.y).toFixed(2)} mm`);
  assert.ok(r.fall < -4, `the sheet reached the floor (${r.fall} m/s)`);
  assert.ok(r.up <= 0.05, `rebound ${r.up} m/s`);
}));

test('a tilted sheet slaps down onto the floor as implicit Euler at h / 8 does (the whole cloth restarts at an onset)', { skip: !runs('free-fall') }, async () => withDevice(async device => {
  // One edge lands first and the membrane carries the jump to vertices no contact touches, so a per-vertex restart left
  // them a rebound of 0.575 m/s (RESULT (25)). Part of the slap's upward motion is membrane recoil, so the reference is
  // the same landing at an eighth of the step, not zero; both run converged, so the gate judges the restart, not the caps.
  const own = newtonOf(clothAnalyticFreeFall()), converged = { iterations: 32, pcgIterations: 800, pcgTolerance: own.pcgTolerance! / 16, residualTolerance: own.residualTolerance! / 16 };
  const ticks = 60, source = tiltedDrop(15 * Math.PI / 180), scene = clothAnalyticSceneDescription(source, { newton: converged });
  const { frames, ms } = await capture(device, scene, range(1, ticks));
  const oracle = await landingOracle(device, scene, scene.cloth![0]!, ticks, 8);
  const got = dropTrack(frames), want = dropTrack(oracle), a = rebound(got), b = rebound(want);
  console.log(`[${label(source)}] tilted 15 deg landing, ${ms.toFixed(1)} ms/tick: rebound ${a.up.toFixed(4)} m/s (h/8 implicit Euler ${b.up.toFixed(4)}), `
    + `settled mean height ${(1000 * got.at(-1)!.y).toFixed(2)} mm (${(1000 * want.at(-1)!.y).toFixed(2)})`);
  assert.ok(Math.abs(a.up - b.up) <= 0.02, `rebound ${a.up} m/s, reference ${b.up}`);
  assert.ok(Math.abs(got.at(-1)!.y - want.at(-1)!.y) <= 0.002, `settled at ${got.at(-1)!.y} m, reference ${want.at(-1)!.y}`);
}));

/** The free-fall sheet with every vertex given `slide` m/s along x, its restarted steps over `ticks` ticks (the tick
 * each one ran), and the tick of its fastest mean fall. */
async function landingRestarts(device: GPUDevice, slide: number, ticks: number) {
  const source = clothAnalyticFreeFall(), doc = source.cloth[0]!, data = doc.geometry.data, n = data.restPositions_m.length / 3;
  // A sliding sheet starts back from the middle by the distance it covers (the 1 m sheet has 0.3 m to the walls of the
  // 1.6 m domain each side): it never reaches a wall, which would be a new contact of its own.
  const back = -slide * ticks * CLOTH_SCENE_STEP_S / 2, shift = (p: readonly number[]) => p.map((u, i) => i % 3 === 0 ? u + back : u);
  const moving = { ...source, cloth: [{ ...doc, geometry: { kind: 'inline' as const, data: { ...data, restPositions_m: shift(data.restPositions_m),
    ...(data.initialPositions_m ? { initialPositions_m: shift(data.initialPositions_m) } : {}), initialVelocities_m_s: Array.from({ length: 3 * n }, (_, i) => i % 3 === 0 ? slide : 0) } } }] };
  const cloth = await ClothSystem.create(device, clothAnalyticSceneDescription(moving, override(moving)), { statusInterval: 1 }), solver = cloth.solverFor(doc.id)!;
  const at: number[] = [], vy: number[] = [];
  try {
    let last = 0;
    for (let k = 1; k <= ticks; k++) {
      assert.ok(cloth.stepTo(k * CLOTH_SCENE_STEP_S), `${slide} m/s: tick ${k} refused: ${cloth.failure?.message ?? 'pending'}`);
      const { restarts } = await solver.readStatus(), v = (await solver.readState()).velocities_m_s;
      if (restarts !== last) at.push(k); last = restarts;
      let m = 0; for (let i = 0; i < n; i++) m += v[3 * i + 1]! / n; vy.push(m);
    }
    assert.equal(cloth.failure, undefined, cloth.failure?.message);
  } finally { cloth.destroy(); }
  const fall = 1 + vy.reduce((k, y, i) => y < vy[k]! ? i : k, 0), line = `[${label(source)}] landing at ${slide} m/s along the floor: fastest fall at tick ${fall}, restarted steps ${at.length} (ticks ${at.join(', ')})`;
  console.log(line);
  // Exactly one restart, within the landing: the barrier layer stops the sheet over the few steps after its fastest fall.
  return { ok: at.length === 1 && at[0]! > fall && at[0]! <= fall + 4, line };
}

test('a sheet landing on the floor, resting or sliding along it, restarts its BDF2 history exactly once, at the landing', { skip: !runs('free-fall') }, async () => withDevice(async device => {
  // The restart (implicit Euler for the step after an onset) is for a velocity jump: contact arriving that was not in
  // range at the step start. Contact that persists (resting) or slides (the frictionless floor at 0.5 m/s along x, 60
  // ticks) is no new contact, so the history restarts once, the step after the landing (Claude's restarts ruling, C1
  // RESULTs 47-48; RESULT 47's 37 sliding restarts were the sheet reaching the domain's wall).
  const fails: string[] = [];
  for (const slide of [0, 0.5]) { const r = await landingRestarts(device, slide, 60); if (!r.ok) fails.push(r.line); }
  assert.deepEqual(fails, []);
}));

/**
 * The rest value c of a lightly damped ringdown y(t) = c + e^(-a t) (p cos w t + q sin w t), sampled at t: the decay a
 * and frequency w by least squares over a grid (shared), then c, p, q linear for each series. A window mean leaves
 * the part of the sway its length does not cancel; this does not.
 */
function ringdown(t: readonly number[], series: readonly (readonly number[])[]): { rest: number[]; w: number; a: number; amplitude: number; rms: number } {
  const basis = (a: number, w: number) => t.map(s => [1, Math.exp(-a * s) * Math.cos(w * s), Math.exp(-a * s) * Math.sin(w * s)] as const);
  const fit = (B: readonly (readonly [number, number, number])[], y: readonly number[]) => {
    const M = [0, 1, 2].map(i => [0, 1, 2].map(j => B.reduce((q, r) => q + r[i]! * r[j]!, 0))), v = [0, 1, 2].map(i => B.reduce((q, r, k) => q + r[i]! * y[k]!, 0));
    for (let i = 0; i < 3; i++) for (let r = 0; r < 3; r++) if (r !== i) { const f = M[r]![i]! / M[i]![i]!; for (let c = i; c < 3; c++) M[r]![c]! -= f * M[i]![c]!; v[r]! -= f * v[i]!; }
    const c = [v[0]! / M[0]![0]!, v[1]! / M[1]![1]!, v[2]! / M[2]![2]!];
    return { c, res: B.reduce((q, r, k) => q + (r[0] * c[0]! + r[1] * c[1]! + r[2] * c[2]! - y[k]!) ** 2, 0) };
  };
  let best = { res: Infinity, a: 0, w: 0, c: [0, 0, 0] };
  for (let w = 2; w <= 60; w += 0.01) for (let a = 0; a <= 2; a += 0.025) { const f = fit(basis(a, w), series[0]!); if (f.res < best.res) best = { ...f, a, w }; }
  const B = basis(best.a, best.w);
  return { rest: series.map(y => fit(B, y).c[0]!), w: best.w, a: best.a, amplitude: Math.hypot(best.c[1]!, best.c[2]!), rms: Math.sqrt(best.res / t.length) };
}

test('hanging curtain: the pinned sheet comes to rest at its exact uniaxial equilibrium', { skip: !runs('hanging-curtain') }, async () => withDevice(async device => {
  // Released unstretched, the near-undamped sheet still rings in its first longitudinal mode at the settle time
  // (period ~0.45 s, ~8 % of the hem drop at 10 s, decaying at ~0.2 / s), so the rest state is the centre of that
  // ringdown fitted over the last second, not its window mean (which leaves 0.85 % of it: RESULT (26)).
  const source = clothAnalyticHangingCurtain(), m = source.probe.columns as number, n = source.probe.rows as number, hc = source.probe.cell as number, top = source.probe.top as number;
  const settle = Math.round((source.probe.settle as number) / h), window = Math.round(1 / h), first = settle - window + 1;
  const scene = clothAnalyticSceneDescription(source, override(source));
  const { frames, ms } = await capture(device, scene, range(first, settle));
  const material = source.cloth[0]!.materials[0]!, k = material.stretchWeft_N_m, rho = material.arealDensity_kg_m2;
  const mass = prepareClothTopology(source.cloth[0]!.geometry.data, source.cloth[0]!.materials).mass_kg;
  const bottom = Math.floor(m / 2) + m * (n - 1), gated = range(2, n - 5);
  const hem: number[] = [], bands = gated.map(() => [] as number[]);
  let maxEdge = 0, ke = 0;
  for (const [pose] of frames) {
    const x = pose!.positions_m;
    gated.forEach((j, q) => {
      let e = 0;
      for (let i = 1; i < m - 1; i++) {
        const a = i + m * j, b = a + m, s = Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / hc - 1;
        e += s / (m - 2); maxEdge = Math.max(maxEdge, s);
      }
      bands[q]!.push(e);
    });
    hem.push(top - x[3 * bottom + 1]! - (n - 1) * hc);
  }
  const v = frames.at(-1)![0]!.velocities_m_s!;
  for (let i = 0; i < mass.length; i++) ke += 0.5 * mass[i]! * (v[3 * i]! ** 2 + v[3 * i + 1]! ** 2 + v[3 * i + 2]! ** 2);
  const fit = ringdown(range(0, frames.length - 1).map(i => i * h), [hem, ...bands]), stretch = fit.rest[0]!;
  let worst = 0, drop = 0;
  for (let j = 0; j + 1 < n; j++) drop += hangingBandStrain(j, n, hc, k, rho, g) * hc;
  gated.forEach((j, q) => { const exact = hangingBandStrain(j, n, hc, k, rho, g); worst = Math.max(worst, Math.abs(fit.rest[q + 1]! - exact) / exact); });
  const windowMean = hem.reduce((s, y) => s + y, 0) / hem.length, L = (n - 1) * hc;
  console.log(`[${label(source)}] ${settle} steps, ${ms.toFixed(1)} ms/tick, ringdown over the last ${window} steps: worst band strain error ${(100 * worst).toFixed(3)} %, hem drop ${(1000 * stretch).toFixed(3)} mm (exact ${(1000 * drop).toFixed(3)}; `
    + `window mean ${(1000 * windowMean).toFixed(2)}); sway ${(1000 * fit.amplitude).toFixed(2)} mm at ${fit.w.toFixed(2)} rad/s (linear fixed-free ${(Math.PI / 2 * Math.sqrt(k / rho) / L).toFixed(2)}), decay ${fit.a.toFixed(3)} / s, `
    + `fit rms ${(1e6 * fit.rms).toFixed(1)} um; max edge strain ${(100 * maxEdge).toFixed(2)} %; kinetic energy at step ${settle} ${ke.toExponential(2)} J`);
  assert.ok(fit.rms < 0.01 * drop, `the hem is no ringdown: fit rms ${fit.rms} m`);
  assert.ok(worst < 0.01, `band strain off its equilibrium by ${worst}`);
  assert.ok(Math.abs(stretch - drop) < 0.01 * drop, `hem drop ${stretch} m, exact ${drop} m`);
}));

/** The swinging sheet's free edge: mean height, and the first time (s, interpolated) its mean x passes under the pin (x = 0). */
async function swing(device: GPUDevice, newton: ClothNewtonSettings, ticks: number) {
  const source = clothAnalyticSwingingSheet(), edge = source.probe.freeEdge as readonly number[];
  const scene = clothAnalyticSceneDescription(source, { newton });
  const { frames, ms } = await capture(device, scene, range(1, ticks));
  const y: number[] = [], x: number[] = [];
  for (const [pose] of frames) {
    const p = pose!.positions_m;
    y.push(edge.reduce((s, v) => s + p[3 * v + 1]!, 0) / edge.length);
    x.push(edge.reduce((s, v) => s + p[3 * v]!, 0) / edge.length);
  }
  let crossing = NaN;
  for (let k = 1; k < x.length; k++) if (x[k]! <= 0 && x[k - 1]! > 0) { crossing = (k + x[k - 1]! / (x[k - 1]! - x[k]!)) * h; break; }
  return { source, y, x, crossing, ms };
}

test('swinging sheet: the flat outer part falls exactly as free fall, and the swing reaches the pin as the converged run does', { skip: !runs('swinging-sheet') }, async () => withDevice(async device => {
  const n0 = newtonOf(clothAnalyticSwingingSheet()), ticks = 45, run = await swing(device, n0, ticks), { source } = run, n = source.probe.gateSteps as number, y0 = source.probe.y0 as number;
  const ref = bdf2FreeFall(n, h, -g);
  let worst = 0, at = 0;
  for (let k = 0; k < n; k++) { const e = Math.abs(run.y[k]! - (y0 + ref.displacement_m[k]!)); if (e > worst) { worst = e; at = k + 1; } }
  console.log(`[${label(source)}] ${run.ms.toFixed(1)} ms/tick: free edge off free fall by up to ${(1000 * worst).toFixed(3)} mm (step ${at}) over ${n} steps; `
    + `per-step lag at steps 6/12/18: ${[6, 12, 18].map(s => (1000 * (run.y[s - 1]! - y0 - ref.displacement_m[s - 1]!)).toFixed(2)).join('/')} mm; passes under the pin at ${run.crossing.toFixed(4)} s`);
  let converged: Awaited<ReturnType<typeof swing>> | undefined;
  if (process.env.CLOTH_ANALYTIC_CONVERGED !== '0') {
    const f = SWING.convergedFactor.value, n1: ClothNewtonSettings = { iterations: f * n0.iterations, pcgIterations: f * n0.pcgIterations,
      pcgTolerance: n0.pcgTolerance! / f ** 2, residualTolerance: n0.residualTolerance! / f ** 2 };
    converged = await swing(device, n1, ticks);
    let gap = 0;
    for (let k = 0; k < ticks; k++) gap = Math.max(gap, Math.hypot(run.x[k]! - converged.x[k]!, run.y[k]! - converged.y[k]!));
    console.log(`[${label(source, n1)}] passes under the pin at ${converged.crossing.toFixed(4)} s; free edge differs from ${n0.iterations} x ${n0.pcgIterations} by up to ${(1000 * gap).toFixed(2)} mm over ${ticks} steps`);
  }
  assert.ok(worst < 1e-3, `free edge ${worst} m off free fall at step ${at}`);
  assert.ok(Number.isFinite(run.crossing), 'the free edge never passed under the pin');
  if (converged) assert.ok(Math.abs(run.crossing - converged.crossing) <= h, `passes the pin at ${run.crossing} s, ${converged.crossing} s converged`);
}));

test('Peirce cantilever: each strip droops to the exact heavy elastica of its bending length', { skip: !runs('peirce-cantilever') }, async () => withDevice(async device => {
  // The strips never reach the ledge (their clamped parts are pinned above it, the overhangs droop past its edge), so
  // the gate runs each document through the solver exactly as ClothSystem sets it up, without the contact pass.
  const source = clothAnalyticPeirceCantilever(), settle = Math.round(PEIRCE.settle.value / h);
  const lines: string[] = [], misses: string[] = [];
  for (const [s, document] of source.cloth.entries()) {
    const [clampCells, cols, rows, overhang, exact, peirce] = source.probe[`strip${s}`] as number[];
    const solver = await solverFor(device, document, caps ? newtonOf(source) : undefined);
    let tickMs = performance.now();
    solver.step(settle);
    const x = await solver.readPositions();
    tickMs = (performance.now() - tickMs) / settle;
    const status = await solver.readStatus();
    solver.destroy();
    let dx = 0, dy = 0;
    for (let j = 0; j < rows!; j++) {
      const root = clampCells! + cols! * j, tip = cols! - 1 + cols! * j;
      dx += (x[3 * tip]! - x[3 * root]!) / rows!; dy += (x[3 * root + 1]! - x[3 * tip + 1]!) / rows!;
    }
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    lines.push(`l = ${(1000 * overhang!).toFixed(0)} mm (l/c ${(overhang! / (source.probe.bendingLength as number)).toFixed(3)}): chord ${angle.toFixed(2)} deg, exact ${exact!.toFixed(2)}, Peirce ${peirce!.toFixed(2)}; ${tickMs.toFixed(1)} ms/tick, status ${status.status}`);
    if (!(Math.abs(angle - exact!) < 1)) misses.push(`strip ${s}: ${angle.toFixed(2)} deg against ${exact!.toFixed(2)}`);
  }
  console.log(`[${label(source)}] after ${settle} steps:\n  ${lines.join('\n  ')}`);
  assert.deepEqual(misses, []);
}));

/** A document's solver exactly as ClothSystem sets it up, without the contact pass. */
async function solverFor(device: GPUDevice, document: ClothDocument, newtonOverride?: ClothNewtonSettings) {
  const data = document.geometry.data, topology = prepareClothTopology(data, document.materials);
  const model = buildClothSolverModel(topology, document.materials, document.scripts ?? []);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  data.restPositions_m.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3]!, v); hi[i % 3] = Math.max(hi[i % 3]!, v); });
  const newton = newtonOverride ?? document.solver!.newton!;
  return ClothGPUSolver.create(device, model, { origin_m: [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as [number, number, number], dt_s: CLOTH_SCENE_STEP_S,
    gravity_m_s2: [0, -g, 0], integrator: 'bdf2', initialGuess: 'previous', newton });
}

test('friction on an incline: the patch below the friction angle sticks, the one above slides at g (sin - mu cos)', { skip: !runs('friction-incline') }, async () => withDevice(async device => {
  const source = clothAnalyticFrictionIncline(), t0 = Math.round(INCLINE.gateStart.value / h), t1 = Math.round(INCLINE.gateEnd.value / h);
  const scene = clothAnalyticSceneDescription(source, override(source));
  const { frames, ms } = await capture(device, scene, range(t0, t1));
  const along = source.cloth.map((_, s) => {
    const [, , dx, dy] = source.probe[`slope${s}`] as number[];
    return frames.map(f => { const p = f[s]!.positions_m, N = p.length / 3; let a = 0; for (let i = 0; i < N; i++) a += (p[3 * i]! * dx! + p[3 * i + 1]! * dy!) / N; return a; });
  });
  // Least-squares s(t) = s0 + v0 t + a t^2 / 2 over the gate.
  const fit = (s: number[]) => {
    const rows = s.map((_, k) => { const t = (t0 + k) * h; return [1, t, t * t / 2]; });
    const A = [0, 1, 2].map(i => [0, 1, 2].map(j => rows.reduce((q, r) => q + r[i]! * r[j]!, 0))), b = [0, 1, 2].map(i => rows.reduce((q, r, k) => q + r[i]! * s[k]!, 0));
    for (let i = 0; i < 3; i++) for (let r = i + 1; r < 3; r++) { const f = A[r]![i]! / A[i]![i]!; for (let c = i; c < 3; c++) A[r]![c]! -= f * A[i]![c]!; b[r]! -= f * b[i]!; }
    const x = [0, 0, 0]; for (let i = 2; i >= 0; i--) x[i] = (b[i]! - [0, 1, 2].slice(i + 1).reduce((q, c) => q + A[i]![c]! * x[c]!, 0)) / A[i]![i]!;
    return x[2]!;
  };
  const creep = Math.abs(along[0]!.at(-1)! - along[0]![0]!), a = fit(along[1]!), exact = (source.probe.slope1 as number[])[1]!;
  console.log(`[${label(source)}] ${ms.toFixed(1)} ms/tick: sticking patch moved ${(1000 * creep).toFixed(3)} mm from ${INCLINE.gateStart.value} to ${INCLINE.gateEnd.value} s; `
    + `sliding patch a = ${a.toFixed(4)} m/s^2 (exact ${exact.toFixed(4)}, ${(100 * (a / exact - 1)).toFixed(2)} %)`);
  assert.ok(creep < 2e-3, `sticking patch moved ${creep} m`);
  assert.ok(Math.abs(a / exact - 1) < 0.05, `sliding acceleration ${a} against ${exact}`);
}));
