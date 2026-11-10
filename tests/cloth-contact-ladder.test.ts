/**
 * Lane C2's small-case contact ladder (GPU; skips without WEBGPU_NODE_MODULE): each contact
 * behaviour on the smallest mesh that shows it, seconds per case, against an analytic answer
 * (rest at the contact radius, no rebound, co-moving layers need no planes) or the brute-force
 * invariants (no intersection, nothing inside a solid, no failing status). The 150 x 150 scenes
 * are milestone acceptance only. Every case prints its per-tick (and, with CLOTH_LADDER_ROUNDS=1,
 * per-round) report: rounds, the Newton step alpha, envelope, the broad phase's attribution
 * (bucket entries scanned, hash collisions, box rejections, candidates), DAT pairs, plane records
 * (per vertex and summed over the filters) and records held for overflow.
 *
 *   CLOTH_LADDER_ONLY=flat,stacked   run only these cases (flat, stacked, sliding, plate, sphere, strip, determinism)
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { validateScene, type SceneDescription } from '../lib/core/model';
import { CLOTH_NEWTON_SCENE_DEFAULTS, CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { OgcContactDetector } from '../lib/cloth/contact/ogc-detector';
import { CLOTH_SCENE_CONTACT, clothContactRadius } from '../lib/cloth/scene-contact';
import type { ClothTopologyData } from '../lib/cloth/types';
import { herringBoneSheet } from '../lib/cloth/scenes/bridson-2002/mesh';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { clothAnalyticFreeFall, clothAnalyticPeirceCantilever, clothAnalyticSceneDescription, type ClothAnalyticScene } from '../lib/cloth/scenes/analytic';
import { prepareClothTopology } from '../lib/cloth/topology';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const only = process.env.CLOTH_LADDER_ONLY?.split(',').map(s => s.trim()).filter(Boolean);
const runs = (name: string) => gpu && (!only || only.includes(name));
const rounds = process.env.CLOTH_LADDER_ROUNDS === '1';

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

/** Sheets laid out as one cloth document's data (disjoint pieces in one mesh: they collide with each other). */
function sheets(...parts: ClothTopologyData[]): ClothTopologyData {
  const rest: number[] = [], uv: number[] = [], groups: number[] = [], triangles: number[] = [], vel: number[] = [];
  let base = 0;
  for (const p of parts) {
    rest.push(...p.restPositions_m); uv.push(...(p.uv ?? []));
    groups.push(...(p.groups ?? p.restPositions_m.filter((_, i) => i % 3 === 0).map(() => 0)));
    triangles.push(...p.triangles.map(t => t + base));
    vel.push(...(p.initialVelocities_m_s ?? p.restPositions_m.map(() => 0)));
    base += p.restPositions_m.length / 3;
  }
  return { ...parts[0]!, restPositions_m: rest, uv, groups, triangles, ...(vel.some(v => v !== 0) ? { initialVelocities_m_s: vel } : {}) };
}
const square = (n: number, cell: number, origin: [number, number, number]) => herringBoneSheet({ columns: n, rows: n, width_m: (n - 1) * cell, depth_m: (n - 1) * cell, origin_m: origin });
const moving = (d: ClothTopologyData, v: [number, number, number]): ClothTopologyData => ({ ...d, initialVelocities_m_s: d.restPositions_m.map((_, i) => v[i % 3]!) });

/** The analytic free-fall scene (its material, floor, Newton settings) with `data` for its sheet. */
function analyticWith(data: ClothTopologyData, duration_s: number, extent_m = 0.64): SceneDescription {
  const ff = clothAnalyticFreeFall(), doc = ff.cloth[0]!;
  const { contactPlaneRecordsPerVertex: _interim, ...solver } = doc.solver!;
  const source: ClothAnalyticScene = { ...ff, duration_s, extents_m: { x: extent_m, y: extent_m, z: extent_m }, cloth: [{ ...doc, geometry: { kind: 'inline', data }, solver }] };
  const scene = clothAnalyticSceneDescription(source);
  assert.deepEqual(validateScene(scene), []);
  return scene;
}

const e2 = (x: number) => (x === undefined ? '-' : x.toExponential(1));
interface TickReport { tick: number; y: number; vy: number; minY: number; intersections: number; solidIntersections: number; records: number; recordsTotal: number; held: number; heldVertices: number; conflicts: number; datPairs: number; rounds: number; status: string }

/**
 * Step `scene` tick by tick to `ticks`, reading the detector's frame counters (and round log) and the solver's
 * Newton control after each, and the intersections at each tick's end. Halts (and reports) on a failing status.
 */
async function ladder(device: GPUDevice, label: string, scene: SceneDescription, ticks: number,
  each?: (cloth: ClothSystem, c: Awaited<ReturnType<OgcContactDetector['readCounters']>>) => Promise<void>): Promise<{ track: TickReport[]; failure?: string; N: number }> {
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const doc = scene.cloth![0]!, det = cloth.contactFor(doc.id)!, solver = cloth.solverFor(doc.id)!;
  const N = doc.geometry.data.restPositions_m.length / 3;
  const track: TickReport[] = [];
  let failure: string | undefined;
  console.log(`[${label}] ${N} vertices; per tick: rounds (mandatory+re), alpha (last Newton step), env, scanned / collisions / box-rejected / candidates (longest scan), DAT pairs, plane records peak per vertex / summed over filters, held for overflow, conflicts`);
  try {
    for (let k = 1; k <= ticks; k++) {
      let pose;
      try { [pose] = (await cloth.capture([k * CLOTH_SCENE_STEP_S]))[0]!; }
      catch (e) { failure = `tick ${k}: ${String(e).slice(0, 240)}`; break; }
      const c = await det.readCounters(), ctl = Array.from(await solver.readNewtonControl() as ArrayLike<number>);
      const { intersections, solidIntersections } = await det.countIntersections();
      const x = pose!.positions_m, v = pose!.velocities_m_s!;
      let y = 0, vy = 0, minY = Infinity;
      for (let i = 0; i < N; i++) { y += x[3 * i + 1]! / N; vy += v[3 * i + 1]! / N; minY = Math.min(minY, x[3 * i + 1]!); }
      const status = [OgcContactDetector.describeStatus(c.status)].flat().join(",");
      track.push({ tick: k, y, vy, minY, intersections, solidIntersections, records: c.framePeakPlaneRecords / N, recordsTotal: c.planeRecordsTotal, held: c.planeRecordHeld, heldVertices: c.held, conflicts: c.conflicts, datPairs: c.datPairs, rounds: c.detections, status });
      console.log(`  tick ${k}: y ${(1000 * y).toFixed(2)} mm vy ${vy.toFixed(3)}; rounds ${c.detections - c.redetections}+${c.redetections} (levelled ${c.levelled}, fallback ${c.levelFallback}; conflicts no plane / closing ${c.levelConflicts.map(w => `${w & 0xffff} (${w >>> 16} solid)`).join(' / ')}) alpha ${+Number(ctl[2]).toPrecision(3)} newton ${ctl[3]} env ${e2(c.peakEnvelope)}; `
        + `scan ${c.scanned}/${c.collisions}/${c.boxRejected}/${c.candidates} (max ${c.scanMax}); dat ${c.datPairs}; records ${(c.framePeakPlaneRecords / N).toFixed(1)}/v, ${c.planeRecordsTotal} summed; held ${c.held} (overflow ${c.planeRecordHeld}); conflicts ${c.conflicts}; `
        + `x ${intersections}/${solidIntersections}; ${status}`);
      if (each) await each(cloth, c);
      if (rounds) for (const [i, r] of (await det.readRoundLog()).entries())
        console.log(`    round ${i}: |w| ${e2(r.maxPrediction)} env ${e2(r.maxEnvelope)} scan ${r.scanned}/${r.collisions}/${r.boxRejected}/${r.candidates} dat ${r.datPairs} held ${r.held} filters ${r.filters} records ${r.planeRecords} (summed ${r.planeRecordsTotal}) |d| ${e2(r.maxDirection)}`);
      if (cloth.failure) { failure = `tick ${k}: ${cloth.failure.message}`; break; }
    }
  } finally { cloth.destroy(); }
  if (failure) console.log(`  HALT ${failure}`);
  return { track, failure, N };
}
const clean = (label: string, r: { track: TickReport[]; failure?: string }) => {
  assert.equal(r.failure, undefined, `${label}: ${r.failure}`);
  for (const t of r.track) assert.equal(t.intersections + t.solidIntersections, 0, `${label}: tick ${t.tick} intersects (${t.intersections} faces, ${t.solidIntersections} in the solid)`);
};

test('ladder: a flat 16 x 16 sheet dropped from 1.2 m (the analytic free fall) lands on the floor, rests at the contact radius without a rebound, with few plane records', { skip: !runs('flat') }, async () => withDevice(async device => {
  const scene = analyticWith(square(16, 0.01, [-0.075, 1.2, -0.075]), 1, 1.6), r = clothContactRadius(scene.cloth![0]!);
  const out = await ladder(device, 'flat 16^2', scene, 50);
  clean('flat', out);
  const t = out.track;
  let k = 0;
  t.forEach((s, i) => { if (s.vy < t[k]!.vy) k = i; });
  const up = Math.max(0, ...t.slice(k + 1).map(s => s.vy)), rest = t.at(-1)!.y, peak = Math.max(...t.map(s => s.records));
  console.log(`  flat: fastest fall ${t[k]!.vy.toFixed(3)} m/s, rebound ${up.toFixed(4)} m/s, rests at ${(1000 * rest).toFixed(3)} mm (r = ${(1000 * r).toFixed(3)} mm), plane records peak ${peak.toFixed(1)} per vertex`);
  assert.ok(t[k]!.vy < -4, 'the sheet fell the 1.2 m');
  assert.ok(up <= 0.05, `rebound ${up} m/s`);
  assert.ok(rest > 0 && rest < 1.5 * r, `rests at ${rest} m, contact radius ${r}`);
  assert.ok(peak <= 9, `plane records ${peak} per vertex`);
}));

test('ladder: two co-moving stacked 8 x 8 sheets in free fall need no cloth-cloth plane records', { skip: !runs('stacked') }, async () => withDevice(async device => {
  const scene = analyticWith(sheets(square(8, 0.01, [-0.035, 0.4, -0.035]), square(8, 0.01, [-0.035, 0.406, -0.035])), 1);
  const out = await ladder(device, 'stacked 2 x 8^2', scene, 15);
  clean('stacked', out);
  const peak = Math.max(...out.track.map(s => s.records)), total = out.track.reduce((s, t) => s + t.recordsTotal, 0);
  console.log(`  stacked: plane records peak ${peak.toFixed(2)} per vertex, ${total} summed over the run`);
  // From the first step: an unmeasured vertex predicts its inertial motion like any other.
  assert.ok(out.track[0]!.recordsTotal <= out.N / 8, `the first step keeps ${out.track[0]!.recordsTotal} plane records`);
  assert.ok(total <= out.N, `co-moving layers keep ${total} plane records over ${out.track.length} ticks`);
}));

test('ladder: a sheet sliding over and onto another is divided: pairs exist and nothing intersects', { skip: !runs('sliding') }, async () => withDevice(async device => {
  // Both fall freely; the upper one slides across at 0.6 m/s and closes the 4 mm gap at 0.3 m/s.
  const scene = analyticWith(sheets(square(8, 0.01, [-0.035, 0.4, -0.035]), moving(square(8, 0.01, [-0.06, 0.404, -0.035]), [0.6, -0.3, 0])), 1);
  const out = await ladder(device, 'sliding 2 x 8^2', scene, 20);
  clean('sliding', out);
  assert.ok(out.track.some(t => t.datPairs > 0), 'the sliding sheets are paired');
}));

test('ladder: an 8 x 8 sheet lands on a slowly moving large sphere (the plate) and rides it', { skip: !runs('plate') }, async () => withDevice(async device => {
  // A box moving solid fails explicitly (scene-contact bakes spheres only): the plate is a sphere of radius 0.2 m
  // whose top the 10.5 cm sheet lands on from 2 cm, carried 4 cm sideways over the run.
  const ticks = 40, end_s = ticks * CLOTH_SCENE_STEP_S, cy = 0.25, R = 0.2;
  const base = analyticWith(square(8, 0.015, [-0.0525, cy + R + 0.02, -0.0525]), 1);
  const at = (x: number) => ({ x, y: cy, z: 0 }), zero = { x: 0, y: 0, z: 0 }, q = { w: 1, x: 0, y: 0, z: 0 };
  base.rigidBodies = [{ id: 'plate', name: 'plate', shape: 'sphere', dimensions_m: { x: R, y: R, z: R }, density_kg_m3: 1000, position_m: at(0), orientation: q,
    linearVelocity_m_s: zero, angularVelocity_rad_s: zero, restitution: 0, friction: 0.5, motion: 'static',
    poseTrack: { keyframes: [0, 1].map(f => ({ time_s: f * end_s, position_m: at(0.04 * f), orientation: q })) } }];
  base.cloth = [{ ...base.cloth![0]!, pendingSolids: [{ id: 'plate', start_s: 0 }] }];
  assert.deepEqual(validateScene(base), []);
  const out = await ladder(device, 'plate 8^2', base, ticks);
  clean('plate', out);
  assert.ok(out.track.at(-1)!.minY > cy + R - 0.03, 'the sheet rests on the sphere');
}));

// Soundness regression (envelope and pair-bound changes): a cloth-cloth pair bound taken on world-straight
// paths without the stray sigma_v of the vertices moving with the solid left a vertex inside the solid at
// tick 37 (item 2 without strayOf, 2026-11-10). Every tick to 60 is checked, faces and solid alike.
test('ladder: a 16 x 16 sheet lands on the rotating sphere: nothing intersects or enters the solid to tick 60 (the carried stray regression)', { skip: !runs('sphere') }, async () => withDevice(async device => {
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 }));
  const out = await ladder(device, 'sphere 16^2', scene, 60);
  clean('sphere', out);
  assert.equal(out.track.length, 60, 'sphere: every tick to 60 is checked');
}));

test('ladder: a 3 x 49 strip starting stretched 1.5x lands two thirds over a ledge (C1 ladder 2) at the default capacity: no holds, the duals within the momentum bound, the strain settling as in free space', { skip: !runs('strip') }, async () => withDevice(async device => {
  // C1's ladder-2 strip (tests/cloth-ladder.test.ts) with the default plane-record capacity: the Peirce ledge (top 0.5 m,
  // edge at x = 0), 5 mm cells, 5 mm above it, stretched 1.5x along its length about the ledge edge. Twin: the same
  // strip in free space (the free-fall scene, 1.2 m above its floor), where only the strain limit acts.
  const cell = 0.005, cols = 49, rows = 3, top = 0.5, ticks = Number(process.env.CLOTH_LADDER_STRIP_TICKS ?? 60), stretch = 1.5;
  const material = { ...clothAnalyticFreeFall().cloth[0]!.materials[0]! }, limit = material.strainLimit!;
  const newton = clothAnalyticFreeFall().cloth[0]!.solver!.newton!, x0 = -(2 / 3) * (cols - 1) * cell;
  const strip = (y: number) => {
    const flat = herringBoneSheet({ columns: cols, rows, width_m: (cols - 1) * cell, depth_m: (rows - 1) * cell, origin_m: [x0, y, -0.12 - (rows - 1) * cell / 2] });
    return { flat, data: { ...flat, initialPositions_m: flat.restPositions_m.map((v, i) => i % 3 === 0 ? stretch * v : v) } };
  };
  const peirce = clothAnalyticPeirceCantilever(), { flat, data } = strip(top + 0.005);
  const scene = clothAnalyticSceneDescription({ ...peirce, cloth: [{ ...peirce.cloth[0]!, id: 'ladder-strip', materials: [material],
    solver: { integrator: 'bdf2', newton }, geometry: { kind: 'inline', data } }] });
  assert.deepEqual(validateScene(scene), []);
  const free = analyticWith(strip(1.2).data, 1, 1.6);
  free.cloth = [{ ...free.cloth![0]!, materials: [material] }];
  const topology = prepareClothTopology(flat, [material]), mass = topology.mass_kg.reduce((a, b) => a + b, 0);
  // No edge force beyond moving the whole strip's mass through the whole initial excess length within one step (C1's bound).
  const h = CLOTH_SCENE_STEP_S, bound = mass * (stretch - 1) * (cols - 1) * cell / (h * h);
  const probe = (id: string, lambdas: number[], strains: number[]) => async (cloth: ClothSystem) => {
    const st = await cloth.solverFor(id)!.readState(), du = st.solverState.edgeDuals, p = st.positions_m;
    let l = 0; for (let e = 0; e < du.length; e += 2) l = Math.max(l, Math.abs(du[e]!));
    let s = 0;
    for (let e = 0; e < topology.edgeCount; e++) {
      const a = topology.edges[4 * e]!, b = topology.edges[4 * e + 1]!;
      s = Math.max(s, Math.hypot(p[3 * a]! - p[3 * b]!, p[3 * a + 1]! - p[3 * b + 1]!, p[3 * a + 2]! - p[3 * b + 2]!) / topology.edgeRestLength_m[e]! - 1);
    }
    lambdas.push(l); strains.push(s);
  };
  const lam: number[] = [], strain: number[] = [], freeLam: number[] = [], freeStrain: number[] = [];
  const out = await ladder(device, 'strip 3 x 49', scene, ticks, probe('ladder-strip', lam, strain));
  const twin = await ladder(device, 'strip 3 x 49 free', free, Math.min(ticks, 25), probe(free.cloth![0]!.id, freeLam, freeStrain));
  clean('strip', out); clean('strip free', twin);
  const at = (a: number[], ks: number[]) => ks.filter(k => k < a.length).map(k => a[k]!);
  const ks = [0, 1, 2, 4, 8, 12, 16, 24, ticks - 1];
  console.log(`  strip: max |lambda| ${at(lam, ks).map(e2).join(' ')} N (free ${at(freeLam, ks).map(e2).join(' ')}) at ticks ${ks.map(k => k + 1).join(' ')}; peak ${e2(Math.max(...lam))} N, momentum bound ${e2(bound)} N; `
    + `max strain ${at(strain, ks).map(s => s.toFixed(3)).join(' ')} (free ${at(freeStrain, ks).map(s => s.toFixed(3)).join(' ')}), limit ${limit}; `
    + `held ${out.track.reduce((s, t) => s + t.heldVertices, 0)}, held for overflow ${out.track.reduce((s, t) => s + t.held, 0)}`);
  for (const t of [...out.track, ...twin.track]) assert.equal(t.heldVertices + t.held, 0, `strip: tick ${t.tick} holds ${t.heldVertices} vertices (${t.held} for plane-record overflow)`);
  assert.ok(Math.max(...lam) <= bound, `a dual ramped past the strip's momentum bound: ${Math.max(...lam)} N > ${bound} N`);
  // As in free space: wherever the free strip is back within 5 % of the limit, so is this one, and it stays there.
  const settled = freeStrain.findIndex(s => s < 1.05 * limit);
  assert.ok(settled >= 0, `the free strip never settled to the strain limit: ${freeStrain.map(s => s.toFixed(3)).join(' ')}`);
  for (let k = settled; k < strain.length; k++) assert.ok(strain[k]! < 1.05 * limit, `tick ${k + 1}: strain ${strain[k]} past the limit ${limit} (free in space by tick ${settled + 1})`);
}));

/**
 * Determinism: the same scene stepped `runs` times in one process must give the same contact decisions every
 * tick. Per tick it keeps the cloth state (positions, velocities, edge duals), the detector's per-vertex state
 * (pred, rho, envelope, held flags) and its integer counters, and reports the first tick and buffer where a
 * rerun departs from the first run: for float buffers the largest difference, for counters the fields.
 */
async function determinism(device: GPUDevice, label: string, scene: SceneDescription, ticks: number, runs = 3) {
  type Snap = { floats: Record<string, Float32Array | Float64Array>; ints: Record<string, Uint32Array>; counters: Record<string, number> };
  const all: Snap[][] = [];
  for (let r = 0; r < runs; r++) {
    const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
    const doc = scene.cloth![0]!, det = cloth.contactFor(doc.id)!, solver = cloth.solverFor(doc.id)!;
    const track: Snap[] = [];
    try {
      for (let k = 1; k <= ticks; k++) {
        const [pose] = (await cloth.capture([k * CLOTH_SCENE_STEP_S]))[0]!;
        const c = await det.readCounters(), d = await det.readDiagnosis(), st = await solver.readState();
        const counters: Record<string, number> = {};
        // heldConflictPair is an example pair for the report, stored by whichever conflicting thread writes last.
        for (const [key, v] of Object.entries(c)) {
          if (key === 'heldConflictPair') continue;
          if (typeof v === 'number' && Number.isInteger(v)) counters[key] = v;
          else if (Array.isArray(v)) v.forEach((x, i) => { counters[`${key}[${i}]`] = x; });
        }
        track.push({
          floats: { positions: Float64Array.from(pose!.positions_m), velocities: Float64Array.from(pose!.velocities_m_s!), duals: Float64Array.from(st.solverState.edgeDuals),
            xdet: d.xdet, pred: d.pred, rho: d.rho, env: d.env },
          ints: { held: d.held }, counters,
        });
        if (cloth.failure) break;
      }
    } finally { cloth.destroy(); }
    all.push(track);
  }
  const diverged: string[] = [];
  for (let r = 1; r < runs; r++) {
    const a = all[0]!, b = all[r]!;
    for (let k = 0; k < Math.min(a.length, b.length); k++) {
      const out: string[] = [];
      for (const [key, x] of Object.entries(a[k]!.ints)) { const y = b[k]!.ints[key]!; let n = 0; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) n++; if (n) out.push(`${key}: ${n} words differ`); }
      for (const [key, x] of Object.entries(a[k]!.counters)) if (b[k]!.counters[key] !== x) out.push(`counter ${key} ${x} vs ${b[k]!.counters[key]}`);
      for (const [key, x] of Object.entries(a[k]!.floats)) {
        const y = b[k]!.floats[key]!; let m = 0, n = 0;
        for (let i = 0; i < x.length; i++) if (!Object.is(x[i], y[i])) { n++; m = Math.max(m, Math.abs(x[i]! - y[i]!)); }
        if (n) out.push(`${key}: ${n} values differ, max ${m.toExponential(2)}`);
      }
      if (out.length) { diverged.push(`run ${r + 1} vs 1: first divergence at tick ${k + 1}: ${out.join('; ')}`); break; }
    }
  }
  console.log(`  [${label}] determinism over ${runs} runs of ${ticks} ticks: ${diverged.length ? diverged.join(' | ') : 'bit-identical'}`);
  return diverged;
}

test('ladder: the flat 16 x 16 drop and the 16 x 16 sphere are deterministic over three runs (the first diverging tick and buffer are reported)', { skip: !runs('determinism') }, async () => withDevice(async device => {
  const flat = await determinism(device, 'flat 16^2', analyticWith(square(16, 0.01, [-0.075, 1.2, -0.075]), 1, 1.6), Number(process.env.CLOTH_LADDER_DET_TICKS ?? 50));
  const sphere = await determinism(device, 'sphere 16^2', bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 })), Number(process.env.CLOTH_LADDER_DET_TICKS ?? 60));
  assert.deepEqual([...flat, ...sphere], []);
}));

test('the scene round cap covers the Newton iteration cap: a re-detection round can follow every Newton iteration (the strip lagged at 4 against 8)', () => {
  assert.ok(CLOTH_SCENE_CONTACT.roundCap >= CLOTH_NEWTON_SCENE_DEFAULTS.iterations, `roundCap ${CLOTH_SCENE_CONTACT.roundCap} < Newton ${CLOTH_NEWTON_SCENE_DEFAULTS.iterations}`);
});
