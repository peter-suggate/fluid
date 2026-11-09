/**
 * The analytic cloth scenes (lib/cloth/scenes/analytic/): acceptance gates for
 * the GPU cloth solver against closed-form answers. CPU: the references
 * themselves, provenance, catalog cards, valid documents, solids on the shared
 * hit-test path and their contact authoring. GPU (skips without
 * WEBGPU_NODE_MODULE): each scene run on the scene clock at its K, measured
 * against its answer with the tolerance its card states.
 *
 * For lane C1's K and solver sweeps:
 *   CLOTH_ANALYTIC_ITERATIONS=8      run every scene at this K instead of its own
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
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology, validateClothDocument } from '../lib/cloth/topology';
import { buildClothSolverModel } from '../lib/cloth/oracle';
import { buildClothCoarseSpace } from '../lib/cloth/coarse';
import { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import type { ClothDocument, ClothSurfacePublication } from '../lib/cloth/types';
import {
  CLOTH_ANALYTIC_PARAMETERS, CLOTH_ANALYTIC_SCENES, CLOTH_ANALYTIC_STEP_S, GRAVITY, INCLINE, PEIRCE, SWING, bdf2FreeFall, bdf2StartOffset_m, bendingLength_m,
  clothAnalyticSceneDescription, hangingBandStrain, hangingStrainContinuum, heavyElastica, inclineAcceleration_m_s2, peirceChordAngle_deg,
  clothAnalyticFreeFall, clothAnalyticFrictionIncline, clothAnalyticHangingCurtain, clothAnalyticPeirceCantilever, clothAnalyticSwingingSheet,
  type ClothAnalyticKind, type ClothAnalyticScene, type ClothAnalyticParameter,
} from '../lib/cloth/scenes/analytic';
import { clothAnalyticSceneSolids } from '../lib/cloth/scenes/analytic/contact-solids';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const K = process.env.CLOTH_ANALYTIC_ITERATIONS ? Number(process.env.CLOTH_ANALYTIC_ITERATIONS) : undefined;
const only = process.env.CLOTH_ANALYTIC_ONLY?.split(',').map(s => s.trim()).filter(Boolean);
const runs = (kind: ClothAnalyticKind) => gpu && (!only || only.includes(kind));
const g = GRAVITY.value, h = CLOTH_ANALYTIC_STEP_S;
const ray = (o: [number, number, number], d: [number, number, number]) => ({ origin: { x: o[0], y: o[1], z: o[2] }, direction: { x: d[0], y: d[1], z: d[2] } });
const iterationsOf = (s: ClothAnalyticScene) => K ?? s.cloth[0]!.solver!.iterations!;

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

/** Run a scene on the scene clock and capture every document at each of `ticks`. */
async function capture(device: GPUDevice, scene: SceneDescription, ticks: readonly number[]): Promise<{ frames: ClothSurfacePublication[][]; ms: number }> {
  const cloth = await ClothSystem.create(device, scene);
  try {
    const t0 = performance.now();
    const frames = await cloth.capture(ticks.map(n => n * CLOTH_SCENE_STEP_S));
    return { frames, ms: (performance.now() - t0) / Math.max(1, ticks.at(-1)!) };
  } finally { cloth.destroy(); }
}

const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const label = (s: ClothAnalyticScene) => `${s.kind}, K = ${iterationsOf(s)}`;

test('free fall: the sheet falls exactly as the BDF2 recurrence, flat and unstrained', { skip: !runs('free-fall') }, async () => withDevice(async device => {
  const source = clothAnalyticFreeFall(), n = source.probe.gateSteps as number, y0 = source.probe.y0 as number;
  const scene = clothAnalyticSceneDescription(source, K ? { bake: true, iterations: K } : {});
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

test('hanging curtain: the pinned sheet comes to rest at its exact uniaxial equilibrium', { skip: !runs('hanging-curtain') }, async () => withDevice(async device => {
  const source = clothAnalyticHangingCurtain(), m = source.probe.columns as number, n = source.probe.rows as number, hc = source.probe.cell as number, top = source.probe.top as number;
  const settle = Math.round((source.probe.settle as number) / h);
  const scene = clothAnalyticSceneDescription(source, K ? { iterations: K } : {});
  const { frames, ms } = await capture(device, scene, [settle]);
  const { positions_m: x, velocities_m_s: v } = frames[0]![0]!;
  const material = source.cloth[0]!.materials[0]!, k = material.stretchWeft_N_m, rho = material.arealDensity_kg_m2;
  const mass = prepareClothTopology(source.cloth[0]!.geometry.data, source.cloth[0]!.materials).mass_kg;
  let worst = 0, drop = 0, maxEdge = 0, ke = 0;
  for (let j = 0; j + 1 < n; j++) {
    const exact = hangingBandStrain(j, n, hc, k, rho, g);
    drop += exact * hc;
    let sum = 0;
    for (let i = 1; i < m - 1; i++) {
      const a = i + m * j, b = a + m;
      const e = Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / hc - 1;
      sum += e; maxEdge = Math.max(maxEdge, e);
    }
    if (j >= 2 && j < n - 4) worst = Math.max(worst, Math.abs(sum / (m - 2) - exact) / exact);
  }
  for (let i = 0; i < mass.length; i++) ke += 0.5 * mass[i]! * (v![3 * i]! ** 2 + v![3 * i + 1]! ** 2 + v![3 * i + 2]! ** 2);
  const bottom = Math.floor(m / 2) + m * (n - 1), stretch = top - x[3 * bottom + 1]! - (n - 1) * hc;
  console.log(`[${label(source)}] ${settle} steps, ${ms.toFixed(1)} ms/tick: worst band strain error ${(100 * worst).toFixed(2)} %, hem drop ${(1000 * stretch).toFixed(2)} mm (exact ${(1000 * drop).toFixed(2)}), `
    + `max edge strain ${(100 * maxEdge).toFixed(2)} %, kinetic energy ${ke.toExponential(2)} J`);
  assert.ok(worst < 0.01, `band strain off its equilibrium by ${worst}`);
  assert.ok(Math.abs(stretch - drop) < 0.01 * drop, `hem drop ${stretch} m, exact ${drop} m`);
  assert.ok(ke < 1e-6, `kinetic energy ${ke} J`);
}));

/** The swinging sheet's free edge: mean height, and the first time (s, interpolated) its mean x passes under the pin (x = 0). */
async function swing(device: GPUDevice, iterations: number | undefined, ticks: number) {
  const source = clothAnalyticSwingingSheet(), edge = source.probe.freeEdge as readonly number[];
  const scene = clothAnalyticSceneDescription(source, iterations ? { iterations } : {});
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
  const ticks = 45, run = await swing(device, K, ticks), { source } = run, n = source.probe.gateSteps as number, y0 = source.probe.y0 as number;
  const ref = bdf2FreeFall(n, h, -g);
  let worst = 0, at = 0;
  for (let k = 0; k < n; k++) { const e = Math.abs(run.y[k]! - (y0 + ref.displacement_m[k]!)); if (e > worst) { worst = e; at = k + 1; } }
  const k0 = iterationsOf(source);
  console.log(`[${label(source)}] ${run.ms.toFixed(1)} ms/tick: free edge off free fall by up to ${(1000 * worst).toFixed(3)} mm (step ${at}) over ${n} steps; `
    + `per-step lag at steps 6/12/18: ${[6, 12, 18].map(s => (1000 * (run.y[s - 1]! - y0 - ref.displacement_m[s - 1]!)).toFixed(2)).join('/')} mm; passes under the pin at ${run.crossing.toFixed(4)} s`);
  let converged: Awaited<ReturnType<typeof swing>> | undefined;
  if (process.env.CLOTH_ANALYTIC_CONVERGED !== '0') {
    converged = await swing(device, SWING.convergedFactor.value * k0, ticks);
    let gap = 0;
    for (let k = 0; k < ticks; k++) gap = Math.max(gap, Math.hypot(run.x[k]! - converged.x[k]!, run.y[k]! - converged.y[k]!));
    console.log(`[swinging-sheet, K = ${SWING.convergedFactor.value * k0}] passes under the pin at ${converged.crossing.toFixed(4)} s; free edge differs from K = ${k0} by up to ${(1000 * gap).toFixed(2)} mm over ${ticks} steps`);
  }
  assert.ok(worst < 1e-3, `free edge ${worst} m off free fall at step ${at}`);
  assert.ok(Number.isFinite(run.crossing), 'the free edge never passed under the pin');
  if (converged) assert.ok(Math.abs(run.crossing - converged.crossing) <= h, `passes the pin at ${run.crossing} s, ${converged.crossing} s at 4K`);
}));

test('Peirce cantilever: each strip droops to the exact heavy elastica of its bending length', { skip: !runs('peirce-cantilever') }, async () => withDevice(async device => {
  // The strips never reach the ledge (their clamped parts are pinned above it, the overhangs droop past its edge), so
  // the gate runs each document through the solver exactly as ClothSystem sets it up, without the contact pass.
  const source = clothAnalyticPeirceCantilever(), settle = Math.round(PEIRCE.settle.value / h);
  const lines: string[] = [], misses: string[] = [];
  for (const [s, document] of source.cloth.entries()) {
    const [clampCells, cols, rows, overhang, exact, peirce] = source.probe[`strip${s}`] as number[];
    const solver = await solverFor(device, document, K);
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

async function solverFor(device: GPUDevice, document: ClothDocument, iterations?: number) {
  const data = document.geometry.data, topology = prepareClothTopology(data, document.materials);
  const model = buildClothSolverModel(topology, document.materials, colourClothVertices(clothVertexNeighbours(topology)), document.scripts ?? []);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  data.restPositions_m.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3]!, v); hi[i % 3] = Math.max(hi[i % 3]!, v); });
  return ClothGPUSolver.create(device, model, { origin_m: [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as [number, number, number], dt_s: CLOTH_SCENE_STEP_S,
    gravity_m_s2: [0, -g, 0], integrator: 'bdf2', iterations: iterations ?? document.solver!.iterations!, chebyshevRho: 0, initialGuess: 'previous', coarse: buildClothCoarseSpace(topology) });
}

test('friction on an incline: the patch below the friction angle sticks, the one above slides at g (sin - mu cos)', { skip: !runs('friction-incline') }, async () => withDevice(async device => {
  const source = clothAnalyticFrictionIncline(), t0 = Math.round(INCLINE.gateStart.value / h), t1 = Math.round(INCLINE.gateEnd.value / h);
  const scene = clothAnalyticSceneDescription(source, K ? { iterations: K } : {});
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
