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
 *   CLOTH_LADDER_ONLY=flat,stacked   run only these cases (flat, stacked, sliding, plate, sphere, strip, determinism, frozen)
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { validateScene, type SceneDescription } from '../lib/core/model';
import { CLOTH_NEWTON_SCENE_DEFAULTS, CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { OgcContactDetector, decodeOgcContactEntry } from '../lib/cloth/contact/ogc-detector';
import { ogcDecideWGSL } from '../lib/cloth/contact/ogc-detector-wgsl';
import { OGC_POSE_WORD, OGC_POSE_WORDS, ogcVertexBlockWGSL } from '../lib/cloth/contact/ogc-wgsl';
import { SOLID_SAMPLE_WORDS, clothSolidSampleEdges, clothSolidSamples, sampleClothSolidField, type ClothSolidField } from '../lib/cloth/contact/solid-field';
import { CLOTH_SCENE_CONTACT, clothContactRadius } from '../lib/cloth/scene-contact';
import type { ClothTopologyData } from '../lib/cloth/types';
import { herringBoneSheet } from '../lib/cloth/scenes/bridson-2002/mesh';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription, bridson2002SceneLayout } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { bridsonSolidDistance } from '../lib/cloth/scenes/bridson-2002/solids';
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

/**
 * The free strip twin's authored plane records per vertex: its measured peak, 16.04 per vertex (2358 records in one
 * filter at tick 2 on 9aa9b596, under a pool that does not bind; 16.0 with 96 per vertex), plus the scenes' margin of
 * about 10 % (18 x 147 = 2646, 12 % over the peak). Snapping back from 1.5x within one frame the strip compresses below
 * rest, so its neighbourhood comes within the rest-altitude envelope cap.
 */
const FREE_STRIP_RECORDS = 18;
/**
 * The sliding and plate cases' authored plane records per vertex, each its measured peak (one filter, under a pool that
 * does not bind, on b85f074e) plus the scenes' margin of about 10 %: sliding 10451 records (81.65 per vertex) -> 90,
 * plate 1396 (21.81 per vertex) -> 24. At the default 16 both held vertices for overflow and passed: clean fails that now.
 */
const SLIDING_RECORDS = 90, PLATE_RECORDS = 24;
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
  let failure: string | undefined, peakContacts = 0, newtonSum = 0, verifyMs = NaN;
  const ver = { verifies: 0, verifyMissing: 0, missingPairs: 0, lagPairs: 0, lagSteps: 0, uncovered: 0 };
  console.log(`[${label}] ${N} vertices; per tick: rounds (mandatory+re), alpha (last Newton step), env, scanned / collisions / box-rejected / candidates (longest scan), DAT pairs, plane records peak per vertex / summed over filters, held for overflow, conflicts`);
  try {
    for (let k = 1; k <= ticks; k++) {
      let pose;
      try { [pose] = (await cloth.capture([k * CLOTH_SCENE_STEP_S]))[0]!; }
      catch (e) { failure = `tick ${k}: ${String(e).slice(0, 240)}`; break; }
      const c = await det.readCounters(), ctl = Array.from(await solver.readNewtonControl() as ArrayLike<number>);
      const { intersections, solidIntersections, solidHit } = await det.countIntersections();
      const x = pose!.positions_m, v = pose!.velocities_m_s!;
      if (solidHit) console.log(`  tick ${k}: ${await describeSolidHit(cloth, doc.id, doc.geometry.data.triangles, x, solidHit)}`);
      let y = 0, vy = 0, minY = Infinity;
      for (let i = 0; i < N; i++) { y += x[3 * i + 1]! / N; vy += v[3 * i + 1]! / N; minY = Math.min(minY, x[3 * i + 1]!); }
      const status = [OgcContactDetector.describeStatus(c.status)].flat().join(",");
      peakContacts = Math.max(peakContacts, c.peakContacts); newtonSum += Number(ctl[3]);
      for (const key of Object.keys(ver) as (keyof typeof ver)[]) ver[key] += c[key];
      track.push({ tick: k, y, vy, minY, intersections, solidIntersections, records: c.framePeakPlaneRecords / N, recordsTotal: c.planeRecordsTotal, held: c.planeRecordHeld, heldVertices: c.held, conflicts: c.conflicts, datPairs: c.datPairs, rounds: c.detections, status });
      console.log(`  tick ${k}: y ${(1000 * y).toFixed(2)} mm vy ${vy.toFixed(3)}; rounds ${c.detections - c.redetections}+${c.redetections} (levelled ${c.levelled}, fallback ${c.levelFallback}; conflicts no plane / closing ${c.levelConflicts.map(w => `${w & 0xffff} (${w >>> 16} solid)`).join(' / ')}) alpha ${+Number(ctl[2]).toPrecision(3)} newton ${ctl[3]} |g|/|mg| ${Math.sqrt(ctl[5]! / ctl[6]!).toExponential(1)} travel-held ${ctl[11]} env ${e2(c.peakEnvelope)}; `
        + `scan ${c.scanned}/${c.collisions}/${c.boxRejected}/${c.candidates} (max ${c.scanMax}); dat ${c.datPairs}; records ${(c.framePeakPlaneRecords / N).toFixed(1)}/v, ${c.planeRecordsTotal} summed; held ${c.held} (overflow ${c.planeRecordHeld}); conflicts ${c.conflicts}; facets ${c.hfPairs}/${c.hfDropped} dropped; `
        + `verify ${c.verifies} (missing ${c.verifyMissing}: ${c.missingPairs} pairs; lag ${c.lagPairs}; uncovered ${c.uncovered}); x ${intersections}/${solidIntersections}; ${status}`);
      if (each) await each(cloth, c);
      if (rounds) for (const [i, r] of (await det.readRoundLog()).entries())
        console.log(`    round ${i}: |w| ${e2(r.maxPrediction)} env ${e2(r.maxEnvelope)} scan ${r.scanned}/${r.collisions}/${r.boxRejected}/${r.candidates} dat ${r.datPairs} held ${r.held} filters ${r.filters} records ${r.planeRecords} (summed ${r.planeRecordsTotal}) |d| ${e2(r.maxDirection)}`);
      if (cloth.failure) { failure = `tick ${k}: ${cloth.failure.message}`; break; }
    }
    if (!failure && track.length) verifyMs = await verifyCost(device, cloth, doc.id);
  } finally { cloth.destroy(); }
  console.log(`  [${label}] ${track.length} ticks; verification query ${(1000 * verifyMs).toFixed(1)} us at the end state; Newton iterations ${newtonSum}; verifications ${ver.verifies}, ${ver.verifyMissing} found ${ver.missingPairs} missing pairs, lag ${ver.lagPairs} pairs in ${ver.lagSteps} steps, uncovered ${ver.uncovered}; most entries on one vertex in any round ${peakContacts} (capacity ${CLOTH_SCENE_CONTACT.contactCapacity})`);
  if (failure) console.log(`  HALT ${failure}`);
  return { track, failure, N };
}
/**
 * The end-of-step verification's own cost at the current state (ms per query): K queries (arming and the pass) in one
 * submission against an empty one, with the solver's done word set to a candidate and restored after.
 */
async function verifyCost(device: GPUDevice, cloth: ClothSystem, id: string, K = 200): Promise<number> {
  const det = cloth.contactFor(id)!, ctl = cloth.solverFor(id)!.newtonControl, off = ctl.offset ?? 0;
  const wall = async (queries: number) => {
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    for (let k = 0; k < queries; k++) det.encodeVerify(pass);
    pass.end(); const t0 = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); return performance.now() - t0;
  };
  const saved = new Float32Array(await (det as unknown as { readBuffer(b: GPUBuffer, o: number, n: number): Promise<ArrayBuffer> }).readBuffer(ctl.buffer, off, 16));
  device.queue.writeBuffer(ctl.buffer, off, new Float32Array([2, saved[1]!, saved[2]!, saved[3]!]));
  await wall(K);
  const ms = (Math.min(await wall(K), await wall(K)) - Math.min(await wall(0), await wall(0))) / K;
  device.queue.writeBuffer(ctl.buffer, off, saved);
  return ms;
}
type V3 = [number, number, number];
/** The closest point of triangle abc to p (Ericson) and its feature: 0..2 a vertex, 3 ab, 4 bc, 5 ca, 6 the face. */
function closestOnTriangle(p: V3, a: V3, b: V3, c: V3): [V3, number] {
  const sub = (u: V3, v: V3): V3 => [u[0] - v[0], u[1] - v[1], u[2] - v[2]], dot = (u: V3, v: V3) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const along = (o: V3, e: V3, t: number): V3 => [o[0] + t * e[0], o[1] + t * e[1], o[2] + t * e[2]];
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return [a, 0];
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return [b, 1];
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return [along(a, ab, d1 / (d1 - d3)), 3];
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return [c, 2];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return [along(a, ac, d2 / (d2 - d6)), 5];
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return [along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))), 4];
  const den = 1 / (va + vb + vc), q = along(a, ab, vb * den);
  return [[q[0] + ac[0] * vc * den, q[1] + ac[1] * vc * den, q[2] + ac[2] * vc * den], 6];
}
/**
 * Ruling B's soundness at one step's end, from the Float64 mirror's missing pairs `m`: a step the verification let end
 * done (the done word 1) has none, and a step that ended at the Newton cap has no more than the verification counted as
 * its one-step lag (lagPairs, which also counts edge-edge and hull-facet pairs).
 */
interface Soundness { missing: string[]; unsound: string[]; lag: number }
const soundness = () => ({ missing: [], unsound: [], lag: 0 }) as Soundness;
function judge(out: Soundness, tick: number, m: string[], done: boolean, lag: number) {
  for (const x of m) out.missing.push(`tick ${tick}: ${x}`);
  out.lag += lag;
  if (done && m.length) out.unsound.push(`tick ${tick}: verified done with ${m.length} pairs within r without an entry: ${m.slice(0, 4).join('; ')}`);
  if (!done && m.length > lag) out.unsound.push(`tick ${tick}: ${m.length} pairs within r without an entry at the cap, the verification's lag ${lag}: ${m.slice(0, 4).join('; ')}`);
}
const doneOf = async (cloth: ClothSystem, id: string) => Number((await cloth.solverFor(id)!.readNewtonControl() as ArrayLike<number>)[0]) === 1;
/**
 * OGC's completeness at a step's end (C1's RESULT 46), in Float64: the exact mirror of the end-of-step verification's
 * vertex-face and sample-face queries (OGC_VERIFY_WGSL pointFaces). Every pair of a cloth vertex (not on the face) or a
 * solid sample (static, or the moving solid's at its end pose) with a face, within r of the end iterate and accepted by
 * Algorithm 1 there (vfAccept's ownership and Voronoi blocks), must be held by an entry of the step's last round whose
 * feature contains the pair's closest feature: the vertex's own (vertex-face) entry, or the sample's in the list of
 * that feature's first vertex. Edge-edge and hull-facet pairs are not checked (the GPU's lag counts them too, so it
 * bounds this count from above). Returns the missing pairs.
 */
async function missingEntries(cloth: ClothSystem, doc: { id: string; geometry: { data: ClothTopologyData } }, x: ArrayLike<number>, r: number): Promise<string[]> {
  const det = cloth.contactFor(doc.id)! as unknown as { det: GPUBuffer; pred: GPUBuffer; solidSamples: number; bodySamples: number; facetCapacity: number; readBuffer(b: GPUBuffer, o: number, n: number): Promise<ArrayBuffer>; readContactEntries(): ReturnType<OgcContactDetector['readContactEntries']> };
  const origin = (cloth as unknown as { entries: Map<string, { origin: readonly number[] }> }).entries.get(doc.id)!.origin;
  const tris = doc.geometry.data.triangles, N = x.length / 3, T = tris.length / 3, S = det.solidSamples, SB = det.bodySamples;
  const xd = new Float32Array(await det.readBuffer(det.det, 0, 16 * (N + S + SB))), pr = new Float32Array(await det.readBuffer(det.pred, 0, 32 * N + 16 * SB));
  const P = (v: number): V3 => [x[3 * v]! - origin[0]!, x[3 * v + 1]! - origin[1]!, x[3 * v + 2]! - origin[2]!];
  const sub = (u: V3, v: V3): V3 => [u[0] - v[0], u[1] - v[1], u[2] - v[2]], dot = (u: V3, v: V3) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  // Points: the cloth vertices (0..N-1), the static samples, then the moving solid's at its end pose (x_det plus its prediction).
  const pts: V3[] = [];
  for (let v = 0; v < N; v++) pts.push(P(v));
  for (let s = 0; s < S; s++) pts.push([xd[4 * (N + s)]!, xd[4 * (N + s) + 1]!, xd[4 * (N + s) + 2]!]);
  for (let s = 0; s < SB; s++) { const i = N + S + s; pts.push([xd[4 * i]! + pr[8 * N + 4 * s]!, xd[4 * i + 1]! + pr[8 * N + 4 * s + 1]!, xd[4 * i + 2]! + pr[8 * N + 4 * s + 2]!]); }
  const tri = (t: number) => [tris[3 * t]!, tris[3 * t + 1]!, tris[3 * t + 2]!];
  const featVerts = (tv: number[], f: number) => f < 3 ? [tv[f]!] : f < 6 ? [tv[f - 3]!, tv[(f - 2) % 3]!] : tv;
  // Each list's vertex-face and sample-face entries: the point id and the feature's vertices.
  const ce = await det.readContactEntries(), K = ce.header[1]!, lists: { id: number; w: number; f: number[] }[][] = Array.from({ length: N }, () => []);
  for (let v = 0; v < N; v++) for (let k = 0; k < ce.counts[v]!; k++) {
    const e = decodeOgcContactEntry(ce.entries.subarray(4 * (v * K + k), 4 * (v * K + k) + 4), N, S, det.facetCapacity);
    if (e.kind === 'sampleFace') lists[v]!.push({ id: N + (e.body ? S : 0) + e.sample, w: 1, f: featVerts(tri(e.face), e.feature) });
    else if (e.kind === 'vf') lists[v]!.push({ id: e.vertex, w: 0, f: featVerts(tri(e.face), e.feature) });
  }
  const facesOf: number[][] = Array.from({ length: N }, () => []), ring: Set<number>[] = Array.from({ length: N }, () => new Set()), edgeFaces = new Map<string, number[]>();
  const ek = (a: number, b: number) => a < b ? `${a},${b}` : `${b},${a}`;
  for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) {
    const a = tris[3 * t + k]!, b = tris[3 * t + (k + 1) % 3]!;
    facesOf[a]!.push(t); ring[a]!.add(b); ring[b]!.add(a);
    const fs = edgeFaces.get(ek(a, b)); if (fs) fs.push(t); else edgeFaces.set(ek(a, b), [t]);
  }
  // vfAcceptAt: a vertex feature belongs to the lowest face at it not holding v, and v must be in its Voronoi cone; an edge
  // feature to the lower of its faces not holding v, with v within its slab and on the outside of both faces' planes.
  const accept = (v: number, q: V3, t: number, tv: number[], f: number) => {
    if (f === 6) return true;
    if (f < 3) {
      const k = tv[f]!, xk = P(k);
      if (facesOf[k]!.some(u => u < t && !tri(u).includes(v))) return false;
      for (const o of ring[k]!) if (dot(sub(q, xk), sub(xk, P(o))) < 0) return false;
      return true;
    }
    const i = tv[f - 3]!, j = tv[(f - 2) % 3]!, fs = edgeFaces.get(ek(i, j))!;
    const other = fs.find(u => u !== t);
    if (other !== undefined && other < t && !tri(other).includes(v)) return false;
    const xi = P(i), xj = P(j), dir = sub(xj, xi);
    if (dot(sub(q, xi), dir) <= 0 || dot(sub(q, xj), [-dir[0], -dir[1], -dir[2]]) <= 0) return false;
    for (const u of fs) {
      const o = tri(u).find(w => w !== i && w !== j)!, xo = P(o), s = dot(sub(xo, xi), dir) / dot(dir, dir);
      const pp: V3 = [xi[0] + dir[0] * s, xi[1] + dir[1] * s, xi[2] + dir[2] * s];
      if (dot(sub(q, pp), sub(pp, xo)) < 0) return false;
    }
    return true;
  };
  // A uniform grid over the points, the cell the longest face edge.
  let cell = 0;
  for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) { const a = P(tris[3 * t + k]!), b = P(tris[3 * t + (k + 1) % 3]!); cell = Math.max(cell, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])); }
  const key = (i: number, j: number, k: number) => `${i},${j},${k}`, grid = new Map<string, number[]>();
  pts.forEach((p, i) => { const k = key(Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell)); let g = grid.get(k); if (!g) grid.set(k, g = []); g.push(i); });
  const missing: string[] = [];
  for (let t = 0; t < T; t++) {
    const tv = tri(t), [a, b, c] = tv.map(P) as [V3, V3, V3];
    const lo = [0, 1, 2].map(k => Math.floor((Math.min(a[k]!, b[k]!, c[k]!) - r) / cell)), hi = [0, 1, 2].map(k => Math.floor((Math.max(a[k]!, b[k]!, c[k]!) + r) / cell));
    for (let i = lo[0]!; i <= hi[0]!; i++) for (let j = lo[1]!; j <= hi[1]!; j++) for (let k = lo[2]!; k <= hi[2]!; k++) for (const p of grid.get(key(i, j, k)) ?? []) {
      if (p < N && tv.includes(p)) continue;
      const [q, f] = closestOnTriangle(pts[p]!, a, b, c), d = Math.hypot(pts[p]![0] - q[0], pts[p]![1] - q[1], pts[p]![2] - q[2]);
      if (!(d > 0) || !(d < r) || !accept(p < N ? p : -1, pts[p]!, t, tv, f)) continue;
      const fv = featVerts(tv, f), list = lists[p < N ? p : fv[0]!]!;
      if (list.some(e => e.id === p && e.w === (p < N ? 0 : 1) && fv.every(u => e.f.includes(u)))) continue;
      const what = p < N ? `vertex ${p}` : p < N + S ? `static sample ${p - N}` : `moving sample ${p - N - S}`;
      missing.push(`${what} - face ${t} (${tv.join('/')}) ${(1e6 * d).toFixed(0)} um, feature ${f}`);
    }
  }
  return missing;
}
/**
 * The intersection count's first face inside a solid: its vertices, the GPU's phi, and the CPU field (the entry's static
 * field, or the moving solid's in its body frame at the end pose the detector holds) at the same lattice in the detector's
 * cloth frame, with the face's least phi there and where.
 */
async function describeSolidHit(cloth: ClothSystem, id: string, tris: ArrayLike<number>, x: ArrayLike<number>, hit: { face: number; body: boolean; phi: number }): Promise<string> {
  const e = (cloth as unknown as { entries: Map<string, { field: ClothSolidField; origin: readonly number[] }> }).entries.get(id)!;
  const det = cloth.contactFor(id)!, d = det as unknown as { bodyField?: ClothSolidField; readBuffer(b: GPUBuffer, o: number, n: number): Promise<ArrayBuffer>; contacts: GPUBuffer };
  const V = [0, 1, 2].map(k => [0, 1, 2].map(a => x[3 * tris[3 * hit.face + k]! + a]!) as [number, number, number]);
  let text = `solid hit: face ${hit.face} (${[0, 1, 2].map(k => tris[3 * hit.face + k]).join(', ')}) in the ${hit.body ? 'moving' : 'static'} solid, GPU phi ${hit.phi.toExponential(3)}; scene vertices ${V.map(p => p.map(c => c.toFixed(5)).join(' ')).join(' | ')}`;
  // The detector's cloth frame, and the field's frame within it.
  let Y = V.map(p => p.map((c, a) => c - e.origin[a]!) as [number, number, number]), field = e.field;
  if (hit.body) {
    if (!d.bodyField) return text + '; no body field';
    field = d.bodyField;
    const f = new Float32Array(await d.readBuffer(d.contacts, 4 * OGC_POSE_WORD, 4 * OGC_POSE_WORDS));
    const c = [f[0]!, f[1]!, f[2]!], R = [[f[4]!, f[5]!, f[6]!], [f[8]!, f[9]!, f[10]!], [f[12]!, f[13]!, f[14]!]];
    // y = R^T (x - c), R's columns R[0..2].
    Y = Y.map(p => R.map(col => col[0]! * (p[0] - c[0]!) + col[1]! * (p[1] - c[1]!) + col[2]! * (p[2] - c[2]!)) as [number, number, number]);
    text += `; pose c ${c.map(v => v.toFixed(5)).join(' ')}`;
  }
  const longest = Math.max(...[0, 1, 2].map(k => Math.hypot(...[0, 1, 2].map(a => Y[(k + 1) % 3]![a]! - Y[k]![a]!))));
  const n = Math.min(32, Math.max(1, Math.ceil(4 * longest / field.cell)));
  const at = (i: number, j: number, m: number) => [0, 1, 2].map(a => (1 - i / m - j / m) * Y[0]![a]! + i / m * Y[1]![a]! + j / m * Y[2]![a]!) as [number, number, number];
  let least = Infinity, where = '';
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
    const q = at(i, j, n), s = sampleClothSolidField(field, q);
    if (s.stored && s.phi < least) { least = s.phi; where = `(${i}, ${j}) of ${n}, field frame ${q.map(v => v.toFixed(5)).join(' ')}`; }
  }
  const fine = 128;
  let fineLeast = Infinity;
  for (let i = 0; i <= fine; i++) for (let j = 0; j <= fine - i; j++) { const s = sampleClothSolidField(field, at(i, j, fine)); if (s.stored) fineLeast = Math.min(fineLeast, s.phi); }
  const vphi = Y.map(q => sampleClothSolidField(field, q).phi.toExponential(2)).join(' / ');
  // The field's surface samples and hull edges near the face: the sample-face and hull-edge pairs that should keep it out.
  const smp = clothSolidSamples(field), edges = clothSolidSampleEdges(field), SW = SOLID_SAMPLE_WORDS;
  const S = (k: number) => [smp[SW * k]!, smp[SW * k + 1]!, smp[SW * k + 2]!] as [number, number, number];
  const sub = (a: number[], b: number[]) => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
  const dot = (a: number[], b: number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const triPoint = (p: number[]) => { let m = Infinity; for (let i = 0; i <= 64; i++) for (let j = 0; j <= 64 - i; j++) m = Math.min(m, Math.hypot(...sub(at(i, j, 64), p))); return m; };
  const segSeg = (a: number[], b: number[], c: number[], e: number[]) => {
    let m = Infinity;
    for (let i = 0; i <= 64; i++) { const t = i / 64, p = a.map((v, k) => v + t * (b[k]! - v)), u = sub(e, c), l2 = dot(u, u);
      const sP = l2 > 0 ? Math.min(1, Math.max(0, dot(sub(p, c), u) / l2)) : 0; m = Math.min(m, Math.hypot(...sub(p, c.map((v, k) => v + sP * u[k]!)))); }
    return m;
  };
  const cen = [0, 1, 2].map(a => (Y[0]![a]! + Y[1]![a]! + Y[2]![a]!) / 3), reach = longest;
  const near: string[] = [];
  let sMin = Infinity;
  for (let k = 0; k < smp.length / SW; k++) {
    const q = S(k);
    if (Math.hypot(...sub(q, cen)) > reach) continue;
    const dq = triPoint(q); sMin = Math.min(sMin, dq);
    if (dq < 0.01) near.push(`s${k} ${q.map(v => v.toFixed(4)).join(' ')} phi ${sampleClothSolidField(field, q).phi.toExponential(2)} to face ${dq.toExponential(2)}`);
  }
  let eMin = Infinity, eAt = '';
  for (let k = 0; k < edges.length / 2; k++) {
    const a = S(edges[2 * k]!), b = S(edges[2 * k + 1]!);
    if (Math.hypot(...sub(a, cen)) > reach && Math.hypot(...sub(b, cen)) > reach) continue;
    for (let f = 0; f < 3; f++) { const dq = segSeg(Y[f]!, Y[(f + 1) % 3]!, a, b); if (dq < eMin) { eMin = dq; eAt = `cloth edge ${f} vs hull edge s${edges[2 * k]}-s${edges[2 * k + 1]}`; } }
  }
  text += `; samples within the face's reach: nearest to the face ${sMin.toExponential(2)} m (${near.length} within 1 cm: ${near.slice(0, 12).join('; ')}); hull edges: nearest to a cloth edge ${eMin.toExponential(2)} m (${eAt})`;
  return text + `; CPU field: vertices ${vphi}, lattice least ${least.toExponential(3)} at ${where}, 128-lattice least ${fineLeast.toExponential(3)} (cell ${field.cell}, band ${field.band})`;
}
const clean = (label: string, r: { track: TickReport[]; failure?: string }) => {
  assert.equal(r.failure, undefined, `${label}: ${r.failure}`);
  for (const t of r.track) assert.equal(t.intersections + t.solidIntersections, 0, `${label}: tick ${t.tick} intersects (${t.intersections} faces, ${t.solidIntersections} in the solid)`);
  // Plane-record overflow holds its vertices (planeRecordOverflow): no case passes with one.
  for (const t of r.track) assert.equal(t.held, 0, `${label}: tick ${t.tick} holds ${t.held} vertices for plane-record overflow (peak ${t.records.toFixed(2)} per vertex)`);
};
/**
 * `scene` with each cloth document's plane-record pool authored at `perVertex` (a case's measured peak plus the
 * scenes' margin of about 10 %); CLOTH_LADDER_RECORDS overrides it (to measure a peak under a pool that does not bind).
 */
const authored = (scene: SceneDescription, perVertex: number): SceneDescription => {
  const n = Number(process.env.CLOTH_LADDER_RECORDS ?? perVertex);
  return { ...scene, cloth: scene.cloth!.map(d => ({ ...d, solver: { ...d.solver!, contactPlaneRecordsPerVertex: n } })) };
};
const peakRecords = (label: string, r: { track: TickReport[]; N: number }) =>
  console.log(`  ${label}: plane-record peak ${Math.round(Math.max(...r.track.map(t => t.records)) * r.N)} records in one filter (${Math.max(...r.track.map(t => t.records)).toFixed(2)} per vertex)`);

/** The scene with every cloth's trial projection in Newton's H_vv metric (solver.contactProjectionMetric 'hvv'). */
const hvv = (scene: SceneDescription): SceneDescription => ({ ...scene, cloth: scene.cloth!.map(d => ({ ...d, solver: { ...d.solver!, contactProjectionMetric: 'hvv' as const } })) });
for (const metric of [false, true])
test(`ladder${metric ? ' (H_vv metric)' : ''}: a flat 16 x 16 sheet dropped from 1.2 m (the analytic free fall) lands on the floor, rests at the contact radius without a rebound, with few plane records`, { skip: !runs('flat') }, async () => withDevice(async device => {
  const flat = analyticWith(square(16, 0.01, [-0.075, 1.2, -0.075]), 1, 1.6), scene = metric ? hvv(flat) : flat, r = clothContactRadius(scene.cloth![0]!);
  const out = await ladder(device, `flat 16^2${metric ? ' hvv' : ''}`, scene, 50);
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

/**
 * The last round's DAT pairs broken down (CLOTH_LADDER_PAIRS=1): within one piece or across two (pieces of
 * `pieceSize` vertices), vertex-face or edge-edge, each pair's distance at X and at X + w against its keep
 * threshold thr = 0.5 gamma_r (env_a + env_b) + sepEps (strays and rounding left out) and its prediction spread
 * L; and per vertex the pairs it is in (its plane records), its envelope, |w| and rho.
 */
async function pairBreakdown(det: OgcContactDetector, pieceSize: number) {
  const [{ ids, slots }, d] = [await det.readPairs(), await det.readDiagnosis()];
  const N = d.env.length, X = (i: number) => [d.xdet[4 * i]!, d.xdet[4 * i + 1]!, d.xdet[4 * i + 2]!];
  const W = (i: number) => (i < N ? [d.pred[4 * i]!, d.pred[4 * i + 1]!, d.pred[4 * i + 2]!] : [0, 0, 0]);
  const Y = (i: number) => X(i).map((x, k) => x + W(i)[k]!);
  const sub = (a: number[], b: number[]) => a.map((x, k) => x - b[k]!), dot = (a: number[], b: number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const segPt = (p: number[], a: number[], b: number[]) => { const ab = sub(b, a), t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / Math.max(dot(ab, ab), 1e-30))); return Math.hypot(...sub(p, a.map((x, k) => x + t * ab[k]!))); };
  const ptTri = (p: number[], a: number[], b: number[], c: number[]) => {
    const n = [(b[1]! - a[1]!) * (c[2]! - a[2]!) - (b[2]! - a[2]!) * (c[1]! - a[1]!), (b[2]! - a[2]!) * (c[0]! - a[0]!) - (b[0]! - a[0]!) * (c[2]! - a[2]!), (b[0]! - a[0]!) * (c[1]! - a[1]!) - (b[1]! - a[1]!) * (c[0]! - a[0]!)];
    const nl = Math.hypot(...n), h = dot(sub(p, a), n) / nl, q = p.map((x, k) => x - h * n[k]! / nl);
    const inside = [[a, b], [b, c], [c, a]].every(([u, v]) => { const e = sub(v!, u!), m = [e[1]! * (q[2]! - u![2]!) - e[2]! * (q[1]! - u![1]!), e[2]! * (q[0]! - u![0]!) - e[0]! * (q[2]! - u![2]!), e[0]! * (q[1]! - u![1]!) - e[1]! * (q[0]! - u![0]!)]; return dot(m, n) >= 0; });
    return inside ? Math.abs(h) : Math.min(segPt(p, a, b), segPt(p, b, c), segPt(p, c, a));
  };
  const segSeg = (p0: number[], p1: number[], q0: number[], q1: number[]) => {
    let best = Math.min(segPt(p0, q0, q1), segPt(p1, q0, q1), segPt(q0, p0, p1), segPt(q1, p0, p1));
    const u = sub(p1, p0), v = sub(q1, q0), w0 = sub(p0, q0), a = dot(u, u), b = dot(u, v), c = dot(v, v), dd = dot(u, w0), e = dot(v, w0), den = a * c - b * b;
    if (den > 1e-30) { const s = (b * e - c * dd) / den, t = (a * e - b * dd) / den; if (s > 0 && s < 1 && t > 0 && t < 1) best = Math.min(best, Math.hypot(...sub(p0.map((x, k) => x + s * u[k]!), q0.map((x, k) => x + t * v[k]!)))); }
    return best;
  };
  const rec = new Uint32Array(N), recCross = new Uint32Array(N);
  const kinds = new Map<string, { n: number; dX: number[]; dY: number[]; thr: number[]; L: number[]; closes: number }>();
  for (let k = 0; k < slots; k++) {
    const x = ids[4 * k]!;
    if (x === 0xffffffff) continue;
    const ee = (x & 0x80000000) !== 0, v = [x & 0x7fffffff, ids[4 * k + 1]!, ids[4 * k + 2]!, ids[4 * k + 3]!], nA = ee ? 2 : 1;
    const piece = (i: number) => (i < N ? Math.floor(i / pieceSize) : -1);
    const cross = piece(v[0]!) !== piece(v[nA]!);
    const envOf = (i: number) => (i < N ? d.env[i]! : 0);
    const eA = Math.max(...v.slice(0, nA).map(envOf)), eB = Math.max(...v.slice(nA).map(envOf));
    const thr = 0.5 * det.gammaR * (eA + eB) + det.sepEps;
    let L = 0;
    for (const i of v.slice(0, nA)) for (const j of v.slice(nA)) L = Math.max(L, Math.hypot(...sub(W(i), W(j))));
    const dist = (P: (i: number) => number[]) => (ee ? segSeg(P(v[0]!), P(v[1]!), P(v[2]!), P(v[3]!)) : ptTri(P(v[0]!), P(v[1]!), P(v[2]!), P(v[3]!)));
    const dX = dist(X), dY = dist(Y), key = `${cross ? 'cross' : 'same'} ${ee ? 'EE' : 'VF'}`;
    const s = kinds.get(key) ?? { n: 0, dX: [], dY: [], thr: [], L: [], closes: 0 };
    s.n++; s.dX.push(dX); s.dY.push(dY); s.thr.push(thr); s.L.push(L); if (Math.min(dX, dY) <= thr) s.closes++;
    kinds.set(key, s);
    for (const i of v) if (i < N) { rec[i]!++; if (cross) recCross[i]!++; }
  }
  const q = (a: ArrayLike<number>) => { const t = Float64Array.from(a).sort(); return [0, 0.5, 0.9, 1].map(f => t[Math.min(t.length - 1, Math.floor(f * (t.length - 1)))]!); };
  const f = (a: number[]) => a.map(x => x.toExponential(1)).join('/');
  console.log(`    pairs: ${slots} slots; per vertex (min/med/90%/max) records ${q(rec).join('/')} (across pieces ${q(recCross).join('/')}); env ${f(q(d.env))} |w| ${f(q(Float64Array.from({ length: N }, (_, i) => Math.hypot(...W(i)))))} rho ${f(q(d.rho))}; flagged (held or in contact) ${d.held.reduce((a, h) => a + ((h & 15) !== 0 ? 1 : 0), 0)} of ${N}`);
  for (const [key, s] of kinds) console.log(`      ${key}: ${s.n} pairs, ${s.closes} within thr at X or X + w; d(X) ${f(q(s.dX))} d(X + w) ${f(q(s.dY))} thr ${f(q(s.thr))} L ${f(q(s.L))}`);
}

test('ladder: a sheet sliding over and onto another is divided: pairs exist and nothing intersects', { skip: !runs('sliding') }, async () => withDevice(async device => {
  // Both fall freely; the upper one slides across at 0.6 m/s and closes the 4 mm gap at 0.3 m/s.
  const scene = analyticWith(sheets(square(8, 0.01, [-0.035, 0.4, -0.035]), moving(square(8, 0.01, [-0.06, 0.404, -0.035]), [0.6, -0.3, 0])), 1);
  const out = await ladder(device, 'sliding 2 x 8^2', authored(scene, SLIDING_RECORDS), 20,
    process.env.CLOTH_LADDER_PAIRS === '1' ? async cloth => pairBreakdown(cloth.contactFor(scene.cloth![0]!.id)!, 64) : undefined);
  peakRecords('sliding', out);
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
  const out = await ladder(device, 'plate 8^2', authored(base, PLATE_RECORDS), ticks);
  peakRecords('plate', out);
  clean('plate', out);
  assert.ok(out.track.at(-1)!.minY > cy + R - 0.03, 'the sheet rests on the sphere');
}));

// Soundness regression (envelope and pair-bound changes): a cloth-cloth pair bound taken on world-straight
// paths without the stray sigma_v of the vertices moving with the solid left a vertex inside the solid at
// tick 37 (item 2 without strayOf, 2026-11-10). Every tick to 60 is checked, faces and solid alike.
for (const metric of [false, true])
test(`ladder${metric ? ' (H_vv metric)' : ''}: a 16 x 16 sheet lands on the rotating sphere: nothing intersects or enters the solid to tick 60 (the carried stray regression)`, { skip: !runs('sphere') }, async () => withDevice(async device => {
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 })), scene = metric ? hvv(base) : base;
  // Each tick, over every face (a 32-per-side barycentric lattice) and over the vertices, the least value of the static
  // field the detector holds, and of the true frustum's signed distance (scene frame); printed with
  // CLOTH_LADDER_SPHEREDIAG=1. The true distance is the (C) gate: the cloth stays out of the true frustum, not only out
  // of the field that rounds its rim in (face 360 at tick 49 was 1.79 mm inside it).
  const faces = scene.cloth![0]!.geometry.data.triangles;
  const source = bridson2002RotatingSphere({ nodesAcross: 16 }), t = bridson2002SceneLayout(source).translation_m;
  const frustum = source.solids.find(sd => sd.id === 'frustum')!;
  const trueFr = (q: readonly number[]) => bridsonSolidDistance(frustum, { x: q[0]! - t.x, y: q[1]! - t.y, z: q[2]! - t.z }, 0);
  const print = process.env.CLOTH_LADDER_SPHEREDIAG === '1';
  let leastTrue = Infinity, leastTick = 0;
  const sound = soundness();
  const diag = async (cloth: ClothSystem, c: Awaited<ReturnType<OgcContactDetector['readCounters']>>) => {
    const id = scene.cloth![0]!.id, x = (await cloth.capture([cloth.time_s]))[0]![0]!.positions_m;
    const e = (cloth as unknown as { entries: Map<string, { field: ClothSolidField; origin: readonly number[] }> }).entries.get(id)!;
    const P = (i: number) => [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!];
    const phi = (q: readonly number[]) => sampleClothSolidField(e.field, [q[0]! - e.origin[0]!, q[1]! - e.origin[1]!, q[2]! - e.origin[2]!]).phi;
    let vField = Infinity, vTrue = Infinity, fField = Infinity, fTrue = Infinity, fAt = -1;
    for (let i = 0; i < x.length / 3; i++) { vField = Math.min(vField, phi(P(i))); vTrue = Math.min(vTrue, trueFr(P(i))); }
    for (let f = 0; f < faces.length / 3; f++) {
      const [A, B, C] = [P(faces[3 * f]!), P(faces[3 * f + 1]!), P(faces[3 * f + 2]!)];
      for (let i = 0; i <= 32; i++) for (let j = 0; j <= 32 - i; j++) {
        const u = i / 32, w = j / 32, q = [0, 1, 2].map(k => (1 - u - w) * A[k]! + u * B[k]! + w * C[k]!);
        const pf = phi(q); if (pf < fField) { fField = pf; fAt = f; }
        fTrue = Math.min(fTrue, trueFr(q));
      }
    }
    if (Math.min(vTrue, fTrue) < leastTrue) { leastTrue = Math.min(vTrue, fTrue); leastTick = Math.round(cloth.time_s / CLOTH_SCENE_STEP_S); }
    judge(sound, Math.round(cloth.time_s / CLOTH_SCENE_STEP_S), await missingEntries(cloth, scene.cloth![0]! as never, x, clothContactRadius(scene.cloth![0]!)), await doneOf(cloth, id), c.lagPairs);
    if (print) console.log(`    static field: vertices >= ${vField.toExponential(2)}, faces >= ${fField.toExponential(2)} m (face ${fAt}); true frustum: vertices >= ${vTrue.toExponential(2)}, faces >= ${fTrue.toExponential(2)} m`);
  };
  const out = await ladder(device, `sphere 16^2${metric ? ' hvv' : ''}`, scene, 60, diag);
  console.log(`  sphere: the cloth's least true frustum distance ${leastTrue.toExponential(3)} m (tick ${leastTick}); pairs within r without an entry at a step's end ${sound.missing.length} (the verification's lag ${sound.lag})${sound.missing.length ? `: ${sound.missing.slice(0, 8).join('; ')}` : ''}`);
  clean('sphere', out);
  assert.deepEqual(sound.unsound, [], 'a verified step ends with every pair within r held, and the cap lag counts every one it leaves');
  assert.equal(out.track.length, 60, 'sphere: every tick to 60 is checked');
  assert.ok(leastTrue > 0, `the cloth reaches ${leastTrue} m into the true frustum at tick ${leastTick}`);
}));

// OGC's completeness on the 32 x 32 sphere (C1's RESULT 46): its faces are 60 to 87 mm across, so a face's interior
// sags onto the rotating body between vertices the field holds 6 mm off, and the body's samples carry that contact.
// At 32^2 kc 9668 t5 the last round paired 5 of them 1.0 to 1.3 mm from face 464/496/465, which the round's plane
// let close to 0.76 to 0.99 mm with no entry. Each step's end, every sample-face and vertex-face pair within r has one.
test('ladder: the 32 x 32 sheet on the rotating sphere: a verified step ends with every pair within r held, and the cap lag counts the pairs it leaves (OGC completeness, the first 12 ticks)', { skip: !runs('complete') }, async () => withDevice(async device => {
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 32 })), doc = scene.cloth![0]!, r = clothContactRadius(doc);
  const sound = soundness();
  let entries = 0;
  const out = await ladder(device, 'sphere 32^2 complete', scene, Number(process.env.CLOTH_LADDER_COMPLETE_TICKS ?? 12), async (cloth, c) => {
    const x = (await cloth.capture([cloth.time_s]))[0]![0]!.positions_m, tick = Math.round(cloth.time_s / CLOTH_SCENE_STEP_S);
    const ce = await cloth.contactFor(doc.id)!.readContactEntries();
    entries = Math.max(entries, ...Array.from(ce.counts));
    judge(sound, tick, await missingEntries(cloth, doc as never, x, r), await doneOf(cloth, doc.id), c.lagPairs);
  });
  console.log(`  sphere 32^2: ${out.track.length} ticks, most entries on one vertex ${entries}; pairs within r without an entry at a step's end ${sound.missing.length} (the verification's lag ${sound.lag})${sound.missing.length ? `: ${sound.missing.slice(0, 12).join('; ')}` : ''}`);
  clean('sphere 32^2', out);
  assert.deepEqual(sound.unsound, [], 'a verified step ends with every pair within r held, and the cap lag counts every one it leaves');
}));

// C1's RESULT 46 (1) repro: one step from the 32^2 sphere at tick 5 (and at tick 15, ladder 1e's state), kc 9668. At
// HEAD before the end-of-step verification the t5 step ended with 10 body samples within r of faces 464/496/465 and no
// entry. The verification (ruling B) runs one radius-r query whenever Newton would declare done and re-detects about
// the iterate where a pair has none, so the step ends with every pair within r held, or at the Newton cap counts the
// pairs it leaves as a one-step lag (OGC's guarantee is no penetration, not completeness within r at a step's end).
for (const tick of [5, 15]) test(`ladder: one step from the 32 x 32 sphere at tick ${tick} (C1's RESULT 46 (1) repro): verified done with every pair within r held, or the cap lag counts the pairs it leaves`, { skip: !runs('complete') }, async () => withDevice(async device => {
  const F = JSON.parse(readFileSync(new URL(`./fixtures/cloth-sphere-32-t${tick}.json`, import.meta.url), 'utf8'));
  const f64 = (b: string) => Array.from(new Float64Array(Uint8Array.from(Buffer.from(b, 'base64')).buffer)), ss = F.solverState;
  const solverState = { ...ss, previousVelocities_m_s: f64(ss.previousVelocities_m_s), previousDisplacements_m: f64(ss.previousDisplacements_m), pinDuals: f64(ss.pinDuals), edgeDuals: f64(ss.edgeDuals) };
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 32 })), doc = base.cloth![0]!;
  if (doc.geometry.kind !== 'inline') throw new Error('the Bridson sphere document is inline');
  const scene = { ...base, cloth: [{ ...doc, geometry: { kind: 'inline' as const, data: { ...doc.geometry.data, initialPositions_m: f64(F.positions_m), initialVelocities_m_s: f64(F.velocities_m_s), solverState } } }] } as SceneDescription;
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  try {
    assert.ok(cloth.stepTo(solverState.time_s + CLOTH_SCENE_STEP_S), `the step from tick ${tick} was refused: ${cloth.failure?.message ?? 'pending'}`);
    const x = (await cloth.capture([cloth.time_s]))[0]![0]!.positions_m;
    const c = await cloth.contactFor(doc.id)!.readCounters(), ctl = Array.from(await cloth.solverFor(doc.id)!.readNewtonControl() as ArrayLike<number>);
    const missing = await missingEntries(cloth, doc as never, x, clothContactRadius(doc)), sound = soundness();
    judge(sound, tick + 1, missing, await doneOf(cloth, doc.id), c.lagPairs);
    console.log(`  32^2 t${tick}: newton ${ctl[3]} done ${ctl[0]}; rounds ${c.detections - c.redetections}+${c.redetections}; verifications ${c.verifies}, ${c.verifyMissing} found ${c.missingPairs} missing pairs, lag ${c.lagPairs}, uncovered ${c.uncovered}; `
      + `pairs within r without an entry at the end ${missing.length}${missing.length ? `: ${missing.slice(0, 8).join('; ')}` : ''}; ${[OgcContactDetector.describeStatus(c.status)].flat().join(',')}`);
    const ms = await verifyCost(device, cloth, doc.id);
    console.log(`  32^2 t${tick}: one verification query ${(1000 * ms).toFixed(1)} us (200 in one submission, less an empty one)`);
    assert.equal(cloth.failure, undefined, cloth.failure?.message);
    assert.deepEqual(sound.unsound, [], 'the verified step ends with every pair within r held, or the cap lag counts every one it leaves');
  } finally { cloth.destroy(); }
}));

test('ladder: a 3 x 49 strip starting stretched 1.5x lands two thirds over a ledge (C1 ladder 2) at the default capacity, its free twin at its authored capacity: no holds, the duals within the momentum bound, the strain settling as in free space', { skip: !runs('strip') }, async () => withDevice(async device => {
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
  // The free twin's authored pool (FREE_STRIP_RECORDS); overflow still holds vertices, which the test fails.
  const freeRecords = Number(process.env.CLOTH_LADDER_STRIP_FREE_RECORDS ?? FREE_STRIP_RECORDS);
  free.cloth = [{ ...free.cloth![0]!, materials: [material], solver: { ...free.cloth![0]!.solver!, contactPlaneRecordsPerVertex: freeRecords } }];
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
  console.log(`  strip free: plane-record peak ${Math.round(Math.max(...twin.track.map(t => t.records)) * rows * cols)} records in one filter, against ${freeRecords} x ${rows * cols}`);
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
// In the metric: the flat drop over its 50 ticks and the sphere over its 60.
test('ladder (H_vv metric): the flat 16 x 16 drop and the 16 x 16 sphere are deterministic over three runs', { skip: !runs('determinism') }, async () => withDevice(async device => {
  const flat = await determinism(device, 'flat 16^2 hvv', hvv(analyticWith(square(16, 0.01, [-0.075, 1.2, -0.075]), 1, 1.6)), Number(process.env.CLOTH_LADDER_DET_TICKS ?? 50));
  const sphere = await determinism(device, 'sphere 16^2 hvv', hvv(bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 }))), Number(process.env.CLOTH_LADDER_DET_TICKS ?? 60));
  assert.deepEqual([...flat, ...sphere], []);
}));

/** The scene with every cloth's contact set frozen after the mandatory round, and with travel 'off' its projection's travel bounds dropped. */
const frozen = (scene: SceneDescription, travel: 'on' | 'off'): SceneDescription => ({ ...scene, cloth: scene.cloth!.map(d => ({ ...d, solver: { ...d.solver!, contactFreeze: 'mandatory' as const, contactTravelBound: travel } })) });
// C1's stage (i)/(ii) switches (diagnosis only): the contract, not the physics. Each tick runs its mandatory round and
// nothing else, and the per-vertex entries read back and decode.
for (const travel of ['on', 'off'] as const)
test(`ladder (frozen contact set${travel === 'off' ? ', no travel bounds' : ''}): the 16 x 16 sphere runs on its mandatory rounds alone and its entries decode`, { skip: !runs('frozen') }, async () => withDevice(async device => {
  const scene = frozen(bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 })), travel);
  assert.deepEqual(validateScene(scene), []);
  const kinds: Record<string, number> = {};
  let checked = 0;
  const out = await ladder(device, `sphere 16^2 frozen${travel === 'off' ? ' travel off' : ''}`, scene, Number(process.env.CLOTH_LADDER_FROZEN_TICKS ?? 40), async (cloth, c) => {
    assert.equal(c.redetections, 0, 'no round after the mandatory one');
    const det = cloth.contactFor(scene.cloth![0]!.id)!, { header, counts, entries } = await det.readContactEntries();
    const N = header[0]!, K = header[1]!;
    assert.equal(counts.length, N);
    for (let v = 0; v < N; v++) for (let k = 0; k < counts[v]!; k++) {
      const e = decodeOgcContactEntry(entries.subarray(4 * (v * K + k), 4 * (v * K + k) + 4), N, det.solidSamples, det.facetCapacity);
      kinds[e.kind] = (kinds[e.kind] ?? 0) + 1; checked++;
    }
  });
  console.log(`  frozen: ${out.track.length} ticks${out.failure ? ` (halt: ${out.failure})` : ''}; entries over the ticks ${checked}: ${JSON.stringify(kinds)}`);
  assert.ok(out.track.length > 0 && checked > 0, 'the sheet met the solid and its entries were read');
}));

test('the frozen-set switches validate: unknown values and travel bounds off without the freeze fail explicitly; off is the kernels unchanged', () => {
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 8 }));
  const withSolver = (solver: Record<string, unknown>): SceneDescription => ({ ...base, cloth: base.cloth!.map(d => ({ ...d, solver: { ...d.solver!, ...solver } as never })) });
  assert.deepEqual(validateScene(withSolver({ contactFreeze: 'mandatory', contactTravelBound: 'off' })), []);
  assert.ok(validateScene(withSolver({ contactTravelBound: 'off' })).some(e => /needs contactFreeze 'mandatory'/.test(e)));
  assert.ok(validateScene(withSolver({ contactFreeze: 'always' })).some(e => /Unknown cloth contact freeze/.test(e)));
  assert.ok(validateScene(withSolver({ contactTravelBound: 'maybe' })).some(e => /Unknown cloth contact travel bound/.test(e)));
  assert.equal(ogcDecideWGSL(true, false), ogcDecideWGSL(true));
  assert.notEqual(ogcDecideWGSL(true, true), ogcDecideWGSL(true));
  for (const metric of [{}, { metric: 'nwVertexInv' }]) {
    const on = ogcVertexBlockWGSL({ group: 0, binding: 3, newton: true, ...metric });
    assert.equal(ogcVertexBlockWGSL({ group: 0, binding: 3, newton: true, travelBound: true, ...metric }), on);
    const off = ogcVertexBlockWGSL({ group: 0, binding: 3, newton: true, travelBound: false, ...metric });
    assert.ok(!off.includes('ogcBallExit(zc, ec, cap.w)') && off.includes('fn ogcPieceExitFaces') && !off.includes('OGC_CON_CAP, ogcConViol'));
  }
});

test('the scene round cap covers the Newton iteration cap: a re-detection round can follow every Newton iteration (the strip lagged at 4 against 8)', () => {
  assert.ok(CLOTH_SCENE_CONTACT.roundCap >= CLOTH_NEWTON_SCENE_DEFAULTS.iterations, `roundCap ${CLOTH_SCENE_CONTACT.roundCap} < Newton ${CLOTH_NEWTON_SCENE_DEFAULTS.iterations}`);
});
