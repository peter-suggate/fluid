/**
 * The detector's locality switch (OgcContactConfig.locality; docs/plans/cloth-m1-max-locality-2026-11-11.md,
 * Experiment 2): compiled face records, per-round geometry records and the materialized bucket offsets only change
 * where the query kernels load the same bits from, so a round under any variant must equal the round with the switch
 * off: counters (scanned, key collisions, box rejections, candidates, pairs, rounds), every vertex's contact entries,
 * the DAT division pairs and the per-vertex round state, bit for bit. Also: the face records against the topology
 * chain they stand for, explicit failure of unknown values, and the records after a cut, a load and an undo.
 * The GPU tests skip without WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { cloneScene, validateScene, type SceneDescription } from '../lib/core/model';
import { clothGridData } from '../lib/cloth/grid';
import { sceneWithClothStroke } from '../lib/cloth/scene-edit';
import { publishSceneClothSurfaces, snapshotSceneClothPose } from '../lib/cloth/scene-geometry';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import type { ClothDocument, ClothMaterial } from '../lib/cloth/types';
import { OgcContactDetector, type OgcContactConfig } from '../lib/cloth/contact/ogc-detector';
import { OGC_LOCALITY_PARTS, ogcQueryBindings, ogcQueryWGSL, parseOgcLocality } from '../lib/cloth/contact/ogc-detector-wgsl';
import { OGC_FACE_RECORD_WORDS, OGC_NONE, buildOgcEdges, buildOgcFaceRecords, buildOgcTopology, type OgcTopology } from '../lib/cloth/contact/ogc-topology';
import { buildClothSolidField, type ClothSolidField } from '../lib/cloth/contact/solid-field';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { boxSdf, merge, rng, sheet, sphereSdf, type OracleMesh, type V3 } from './cloth-contact-oracle';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const VARIANTS = ['topology', 'geometry', 'offsets', 'topology+geometry', 'topology+geometry+offsets'];

const vec4s = (pts: V3[]) => new Float32Array(pts.flatMap(p => [p[0], p[1], p[2], 0]));
const f32 = (pts: V3[]) => pts.map(p => p.map(Math.fround) as V3);
const bits = (a: Float32Array | Uint32Array) => new Uint32Array(a.buffer, a.byteOffset, a.length);

function meshTopology(mesh: OracleMesh) {
  const triangles = new Uint32Array(mesh.tris.flatMap(t => [...t, 0]));
  return { vertexCount: mesh.x.length, triangles, edges: buildOgcEdges(triangles) };
}

/** Three jittered sheets within r of each other and one arched across them (tests/cloth-contact-gpu.test.ts's stack). */
function stack(seed: number, gap = 1.3e-3): OracleMesh {
  const R = rng(seed), h = 0.01;
  const layers = [0, gap, 2 * gap].map((z, i) => {
    const s = sheet(7, 7, h, [i * 0.0031, i * 0.0023, z]);
    return { ...s, x: s.x.map(p => [p[0] + (R() - 0.5) * 1e-3, p[1] + (R() - 0.5) * 1e-3, p[2] + (R() - 0.5) * 4e-4] as V3) };
  });
  const fold = sheet(6, 3, h, [0.004, 0.01, 0]);
  const folded = { ...fold, x: fold.x.map(p => [p[0], p[1], 3.2 * gap + 1.5 * (p[1] - 0.025) * (p[1] - 0.025) * 40 + (R() - 0.5) * 2e-4] as V3) };
  const m = merge(...layers, folded);
  return { ...m, x: f32(m.x) };
}

/** The chain a face record stands for, read from the topology words as the baseline kernels read them. */
function chainRecord(topology: OgcTopology, t: number) {
  const L = topology.layout, w = topology.words;
  const v = [0, 1, 2].map(k => w[L.trisOffset + 4 * t + k]!), e = [0, 1, 2].map(k => w[L.triEdgesOffset + 4 * t + k]!);
  return {
    v, e, owned: w[L.triEdgesOffset + 4 * t + 3]!,
    ends: e.map(id => [w[L.edgesOffset + 4 * id]!, w[L.edgesOffset + 4 * id + 1]!]),
    ownsVertex: v.map(id => w[L.vtListOffset + w[L.vtOffOffset + id]!] === t),
  };
}

function assertFaceRecords(records: Uint32Array, topology: OgcTopology, label: string) {
  const T = topology.layout.triangleCount;
  assert.equal(records.length, OGC_FACE_RECORD_WORDS * T, `${label}: one record a face`);
  for (let t = 0; t < T; t++) {
    const r = records.subarray(OGC_FACE_RECORD_WORDS * t, OGC_FACE_RECORD_WORDS * (t + 1)), want = chainRecord(topology, t), flags = r[3]!;
    assert.deepEqual([r[0], r[1], r[2]], want.v, `${label}: face ${t} vertices`);
    assert.deepEqual([r[4], r[5], r[6]], want.e, `${label}: face ${t} edges`);
    assert.equal(r[7], 0);
    assert.equal(flags & 7, want.owned, `${label}: face ${t} ownership mask`);
    assert.equal(flags >>> 9, 0, `${label}: face ${t} has no unknown flag`);
    for (let k = 0; k < 3; k++) {
      // The query's faceEdge: the endpoints in the edge table's own order.
      const a = r[k]!, b = r[(k + 1) % 3]!, ends = flags & (8 << k) ? [b, a] : [a, b];
      assert.deepEqual(ends, want.ends[k], `${label}: face ${t} edge ${k} endpoints in table order`);
      assert.equal((flags & (64 << k)) !== 0, want.ownsVertex[k], `${label}: face ${t} owns vertex ${k}`);
    }
  }
}

test('face records stand for the topology chain exactly: vertices, edge ids, endpoints in table order, edge and vertex ownership', () => {
  // Sheets with boundaries, a merged stack (several components) and an edge table in the reverse order of the faces'.
  for (const mesh of [sheet(5, 4, 0.01), stack(3), sheet(1, 1, 0.01)]) {
    const m = meshTopology(mesh), topology = buildOgcTopology(m, vec4s(mesh.x));
    assertFaceRecords(buildOgcFaceRecords(topology), topology, 'sheet');
    const flipped = m.edges.slice();
    for (let e = 0; e < flipped.length / 4; e += 2) { const a = flipped[4 * e]!; flipped[4 * e] = flipped[4 * e + 1]!; flipped[4 * e + 1] = a; }
    const other = buildOgcTopology({ ...m, edges: flipped }, vec4s(mesh.x));
    const records = buildOgcFaceRecords(other);
    assertFaceRecords(records, other, 'flipped edges');
    assert.ok(mesh.tris.length < 2 || records.some((w, i) => i % 8 === 3 && (w & 56) !== 0) && records.some((w, i) => i % 8 === 3 && (w & 56) !== 56), 'both edge orders occur');
  }
  // Every edge is owned by exactly one face and every vertex by exactly one.
  const mesh = stack(5), m = meshTopology(mesh), topology = buildOgcTopology(m, vec4s(mesh.x)), records = buildOgcFaceRecords(topology);
  const edgeOwners = new Uint32Array(m.edges.length / 4), vertexOwners = new Uint32Array(m.vertexCount);
  for (let t = 0; t < mesh.tris.length; t++) for (let k = 0; k < 3; k++) {
    if (records[8 * t + 3]! & (1 << k)) edgeOwners[records[8 * t + 4 + k]!]!++;
    if (records[8 * t + 3]! & (64 << k)) vertexOwners[records[8 * t + k]!]!++;
  }
  assert.ok(edgeOwners.every(c => c === 1) && vertexOwners.every(c => c === 1));
});

test('face records fail explicitly on topology words they cannot stand for', () => {
  const mesh = sheet(3, 3, 0.01), topology = buildOgcTopology(meshTopology(mesh), vec4s(mesh.x)), L = topology.layout;
  const corrupt = (edit: (w: Uint32Array) => void): OgcTopology => { const words = topology.words.slice(); edit(words); return { layout: L, words }; };
  assert.throws(() => buildOgcFaceRecords(corrupt(w => { w[L.triEdgesOffset + 3] = 8; })), /unknown ownership mask/);
  assert.throws(() => buildOgcFaceRecords(corrupt(w => { w[L.triEdgesOffset] = w[L.triEdgesOffset + 4 * 5 + 1]!; })), /is not face 0's edge 0/);
  assert.throws(() => buildOgcFaceRecords(corrupt(w => { w[L.edgesOffset + 4 * w[L.triEdgesOffset + 1]!] = OGC_NONE; })), /is not face 0's edge 1/);
});

test('the locality switch validates explicitly, and off leaves the query kernels and their bindings as they were', () => {
  assert.deepEqual(parseOgcLocality(undefined), { topology: false, geometry: false, offsets: false });
  assert.deepEqual(parseOgcLocality('off'), parseOgcLocality(undefined));
  assert.deepEqual(parseOgcLocality('offsets+topology'), { topology: true, geometry: false, offsets: true });
  assert.deepEqual(parseOgcLocality(OGC_LOCALITY_PARTS.join('+')), { topology: true, geometry: true, offsets: true });
  for (const bad of ['', 'on', 'all', 'topology+', 'topology+topology', 'off+offsets', 'Topology', 'geometry,offsets', 3, null, true])
    assert.throws(() => parseOgcLocality(bad as string), /OGC: unknown locality/, `locality ${JSON.stringify(bad)}`);
  const off = parseOgcLocality('off');
  for (const role of ['queryVertex', 'queryEdge', 'querySample'] as const) {
    const base = ogcQueryWGSL(role);
    assert.equal(ogcQueryWGSL(role, off), base);
    assert.ok(!/@binding\((24|25)\)/.test(base) && !/\b(loc|geo)\[/.test(base) && !base.includes('faceRec'), `${role}: no locality code when off`);
    for (const variant of VARIANTS) {
      const L = parseOgcLocality(variant), source = ogcQueryWGSL(role, L);
      assert.notEqual(source, base, `${role} ${variant}`);
      assert.equal(source.includes('@binding(24)'), L.topology || L.offsets);
      assert.equal(source.includes('@binding(25)'), L.geometry);
      assert.equal(source.includes('faceRec('), L.topology);
    }
  }
  assert.ok(!ogcQueryBindings(off).some(b => b >= 24));
  assert.deepEqual(ogcQueryBindings(parseOgcLocality('topology+geometry+offsets')).filter(b => b >= 24), [24, 25]);
  // Additive: every variant keeps the baseline's bindings (the grid, the entries, the live buffers).
  for (const variant of VARIANTS) assert.deepEqual(ogcQueryBindings(parseOgcLocality(variant)).filter(b => b < 24), ogcQueryBindings(off), variant);
});

export async function withDevice(run: (device: GPUDevice) => Promise<void>) {
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

async function readWords(device: GPUDevice, buffer: GPUBuffer, offset = 0, bytes = buffer.size - offset): Promise<Uint32Array> {
  const read = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(read.getMappedRange(0, bytes).slice(0)); read.destroy();
  return out;
}

const storage = (device: GPUDevice, data: Float32Array<ArrayBuffer>) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(b, 0, data);
  return b;
};

/**
 * Everything a round leaves that the solver or a report reads, in an order-free form: a vertex's contact entries and
 * the DAT pairs are appended by racing threads, so each list is compared as a sorted multiset of its raw words.
 */
async function roundSnapshot(det: OgcContactDetector) {
  const c = await det.readCounters(), counters: Record<string, number> = {};
  for (const [key, v] of Object.entries(c)) {
    // An example pair for the report, stored by whichever conflicting thread writes last.
    if (key === 'heldConflictPair') continue;
    if (typeof v === 'number') counters[key] = v;
    else if (Array.isArray(v)) v.forEach((x, i) => { counters[`${key}[${i}]`] = x; });
  }
  const { header, counts, entries } = await det.readContactEntries(), K = header[1]!;
  const lists = Array.from(counts, (n, v) => Array.from({ length: n }, (_, k) => Array.from(entries.subarray(4 * (v * K + k), 4 * (v * K + k) + 4)).join(',')).sort());
  const p = await det.readPairs(), planes = bits(p.planes), pairs: string[] = [];
  for (let k = 0; k < p.slots; k++) if (p.ids[4 * k] !== OGC_NONE) pairs.push(`${Array.from(p.ids.subarray(4 * k, 4 * k + 4))}|${Array.from(planes.subarray(4 * k, 4 * k + 4))}`);
  pairs.sort();
  const d = await det.readDiagnosis();
  return { counters, lists, pairs, xdet: bits(d.xdet), pred: bits(d.pred), env: bits(d.env), held: d.held };
}
type RoundSnapshot = Awaited<ReturnType<typeof roundSnapshot>>;

/** Equal element for element (bits for the float arrays' u32 views); a failure names the first difference, not the arrays. */
function assertSameArray(got: ArrayLike<number>, want: ArrayLike<number>, label: string) {
  assert.equal(got.length, want.length, `${label}: length`);
  let first = -1, count = 0;
  for (let i = 0; i < want.length; i++) if (!Object.is(got[i], want[i])) { if (first < 0) first = i; count++; }
  assert.equal(count, 0, `${label}: ${count} of ${want.length} differ, the first at ${first}: ${got[first]} against ${want[first]}`);
}

export function assertSameRound(got: RoundSnapshot, want: RoundSnapshot, label: string) {
  const differing = Object.keys(want.counters).filter(k => !Object.is(got.counters[k], want.counters[k])).map(k => `${k} ${got.counters[k]} against ${want.counters[k]}`);
  assert.deepEqual(differing, [], `${label}: counters`);
  assert.equal(got.lists.length, want.lists.length);
  const lists = want.lists.flatMap((l, v) => l.join(';') === got.lists[v]!.join(';') ? [] : [v]);
  assert.equal(lists.length, 0, `${label}: contact entries of ${lists.length} vertices differ, the first vertex ${lists[0]}: [${got.lists[lists[0]!]?.join('; ')}] against [${want.lists[lists[0]!]?.join('; ')}]`);
  const extra = got.pairs.filter(p => !want.pairs.includes(p)), missing = want.pairs.filter(p => !got.pairs.includes(p));
  assert.ok(got.pairs.length === want.pairs.length && !extra.length && !missing.length, `${label}: DAT pairs ${got.pairs.length} against ${want.pairs.length}; extra ${extra.slice(0, 3).join(' ')}; missing ${missing.slice(0, 3).join(' ')}`);
  for (const key of ['xdet', 'pred', 'env', 'held'] as const) assertSameArray(got[key], want[key], `${label}: ${key}`);
}

/** The locality buffers hold what they claim: the face records and, after a round (`round`), its final offsets and geometry. */
async function assertLocalityBuffers(device: GPUDevice, det: OgcContactDetector, round: RoundSnapshot | undefined, label: string) {
  const L = det.locality, T = det.T, N = det.N;
  assert.equal(det.locBuffer !== undefined, L.topology || L.offsets, `${label}: the offset and face table exists only when used`);
  assert.equal(det.geoBuffer !== undefined, L.geometry, `${label}: the geometry records exist only when used`);
  if (det.locBuffer) {
    // The offsets alone carry no face records.
    const words = await readWords(device, det.locBuffer), H = det.locBuffer.size / 4 - 8 - (L.topology ? OGC_FACE_RECORD_WORDS * T : 0);
    assert.ok(Number.isInteger(H) && (H & (H - 1)) === 0, `${label}: ${H} buckets`);
    if (L.topology) assert.deepEqual(words.subarray(H + 8), buildOgcFaceRecords(det.topology), `${label}: face records on the GPU`);
    if (!round) return;
    if (L.offsets) {
      assert.equal(words[0], 0);
      for (let h = 0; h < H; h++) if (words[h + 1]! < words[h]!) assert.fail(`${label}: bucket ${h} ends before it starts`);
      assert.ok(words[H]! > 0 && words[H]! <= round.counters.peakGrid!, `${label}: ${words[H]} entries in the round's grid`);
    } else assert.ok(words.subarray(0, H + 8).every(w => w === 0), `${label}: no offsets written without the variant`);
  }
  if (det.geoBuffer && round) {
    const geo = await readWords(device, det.geoBuffer, 0, 32 * N);
    for (let v = 0; v < N; v++) {
      for (let k = 0; k < 3; k++) {
        assert.equal(geo[8 * v + k], round.xdet[4 * v + k], `${label}: vertex ${v} x_det`);
        assert.equal(geo[8 * v + 4 + k], round.pred[4 * v + k], `${label}: vertex ${v} prediction`);
      }
      assert.equal(geo[8 * v + 3], round.env[v], `${label}: vertex ${v} envelope`);
    }
  }
}

const baseConfig = (N: number): OgcContactConfig => ({
  r: 2e-3, kc: 1e5, mu: 0.3, epsV: 1e-2, dt: 1 / 60, gravity: [0, 0, -9.81],
  rqBase: 3e-3, rqMax: 2e-2, envelopeScale: 1.5, contactCapacity: 96, pairCapacity: 512 * N, gridCapacity: 128 * N, bucketCount: 4096, roundCap: 2,
  prediction: 'none',
});

type Pass = (fn: (pass: GPUComputePassEncoder) => void) => void;
interface RoundCase {
  label: string; mesh: OracleMesh; vel: V3[]; config: OgcContactConfig; field?: ClothSolidField;
  /** The first positions (default the mesh's) and what follows the first detection. */
  start?: V3[];
  then?: (det: OgcContactDetector, write: (x: V3[]) => void, pass: Pass) => void;
  expect: (c: Record<string, number>) => boolean;
}

export async function runRound(device: GPUDevice, c: RoundCase, locality: string | undefined) {
  const positions = storage(device, vec4s(c.start ?? c.mesh.x)), velocities = storage(device, vec4s(c.vel));
  const config = locality === undefined ? c.config : { ...c.config, locality };
  const det = await OgcContactDetector.create(device, meshTopology(c.mesh), vec4s(c.mesh.x), config, { positions, velocities });
  if (c.field) det.setSolidField(c.field);
  const pass: Pass = fn => { const enc = device.createCommandEncoder(), p = enc.beginComputePass(); fn(p); p.end(); device.queue.submit([enc.finish()]); };
  pass(p => { det.encodeFrameBegin(p); det.encodeStepBegin(p); det.encodeDetection(p); });
  c.then?.(det, x => device.queue.writeBuffer(positions, 0, vec4s(x)), pass);
  const round = await roundSnapshot(det);
  await assertLocalityBuffers(device, det, round, `${c.label} ${locality ?? 'default'}`);
  det.destroy(); positions.destroy(); velocities.destroy();
  return round;
}

export function roundCases(): RoundCase[] {
  const a = stack(9), b = stack(23, 4e-3), R = rng(41), still = (m: OracleMesh) => m.x.map(() => [0, 0, 0] as V3);
  const solidBox = boxSdf([0.02, 0.02, 0], [0.08, 0.06, 0.03]), solidBall = sphereSdf([0.11, 0.05, 0], 0.025);
  const field = buildClothSolidField({
    parts: [
      { kind: 'sdf', sdf: p => solidBox(p), lo: [0.02, 0.02, 0], hi: [0.08, 0.06, 0.03], mu: 0.3 },
      { kind: 'sdf', sdf: p => solidBall(p), lo: [0.085, 0.025, -0.025], hi: [0.135, 0.075, 0.025], mu: 0.5 },
    ], lo: [-0.02, -0.02, -0.03], hi: [0.18, 0.12, 0.1], cell: 0.005, band: 0.06 });
  const over = sheet(15, 10, 0.01, [0.003, 0.004, 0]), draped: OracleMesh = { ...over, x: f32(over.x.map(p => [p[0], p[1], 0.0305 + 0.004 * R()] as V3)) };
  // The top sheet predicted 7 mm down at the others, which stay (tests/cloth-contact-gpu.test.ts's least-shift scene).
  const crossing = b.x.map(p => [0, 0, p[2] > 6e-3 ? (-7e-3 + 9.81 / 3600) * 60 : 9.81 / 60] as V3);
  const predicted: OgcContactConfig = { ...baseConfig(b.x.length), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: 'inertial', rqMax: 4e-2, gridCapacity: 2048 * b.x.length };
  const w0 = (m: OracleMesh, vel: V3[], c: OgcContactConfig) => f32(m.x.map((x, i) => [0, 1, 2].map(k => x[k]! - (vel[i]![k]! * c.dt + c.gravity[k]! * c.dt ** 2)) as V3));
  /** The step after: from X - w to X, so the second step's mandatory round measures no residual. */
  const secondStep = (m: OracleMesh): RoundCase['then'] => (det, write, pass) => { write(m.x); pass(p => { det.encodeFrameBegin(p); det.encodeStepBegin(p); det.encodeDetection(p); }); };
  return [
    { label: 'stack, every prediction held', mesh: a, vel: still(a), config: baseConfig(a.x.length), expect: c => c.candidates! > 0 && c.datPairs! > 0 },
    // A fifth of the vertices leave their bounds: the truncation, the GPU's decision and the re-detection round it arms.
    { label: 'stack, isotropic truncation and an armed re-detection', mesh: a, vel: still(a), config: { ...baseConfig(a.x.length), roundCap: 2, truncation: 'isotropic' },
      then: (det, write, pass) => {
        const Q = rng(5);
        write(a.x.map(p => Q() < 0.2 ? [p[0] + 0.01, p[1], p[2]] as V3 : p));
        pass(p => { det.encodeTruncate(p); det.encodeDecide(p); det.encodeDetection(p, false); });
      }, expect: c => c.redetections === 1 && c.detections === 2 && c.truncated! > 0 },
    // Crossing predictions over two steps: velocity envelopes, the predicted-motion DAT, conflicts and their redo rounds.
    { label: 'spaced stack, crossing inertial predictions', mesh: b, vel: crossing, config: predicted, start: w0(b, crossing, predicted), then: secondStep(b),
      expect: c => c.redos! >= 1 && c.conflicts! > 0 },
    // The static solid's samples, hull edges and hull facets (the sample and edge roles' solid branches).
    { label: 'sheet over a box and a ball', mesh: draped, vel: still(draped),
      config: { ...baseConfig(draped.x.length), rqBase: 0.012, gridCapacity: 256 * draped.x.length, solidCapacityWords: field.words.length + 64, solidSampleCapacity: 8192 }, field,
      expect: c => c.solidContacts! > 0 && c.sfPairs! + c.hfPairs! > 0 },
  ];
}

test('GPU detection round: every locality variant leaves the counters, contact entries, DAT pairs and round state of the switch off', { skip: !gpu }, async () => withDevice(async device => {
  assert.equal(OgcContactDetector.defaultLocality, 'off');
  for (const c of roundCases()) {
    const off = await runRound(device, c, undefined), k = off.counters;
    const summary = `scanned ${k.scanned} key collisions ${k.collisions} box-rejected ${k.boxRejected} candidates ${k.candidates} DAT pairs ${off.pairs.length} entries ${off.lists.reduce((s, l) => s + l.length, 0)} `
      + `solid contacts ${k.solidContacts} sample-facet ${k.sfPairs} hull-facet ${k.hfPairs} detections ${k.detections} redetections ${k.redetections} redos ${k.redos} conflicts ${k.conflicts} truncated ${k.truncated} status ${k.status}`;
    assert.ok(k.scanned! > 0 && c.expect(k), `${c.label} exercises what it is for: ${summary}`);
    assertSameRound(await runRound(device, c, 'off'), off, `${c.label}, off again`);
    for (const variant of VARIANTS) assertSameRound(await runRound(device, c, variant), off, `${c.label}, ${variant}`);
    console.log(`locality ${c.label}: ${summary}: equal under ${VARIANTS.length} variants`);
  }
  // An unknown value fails at construction, before any GPU object.
  const m = stack(9);
  await assert.rejects(OgcContactDetector.create(device, meshTopology(m), vec4s(m.x), { ...baseConfig(m.x.length), locality: 'faces' }), /OGC: unknown locality/);
}));

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };

/** A 24 x 24 sheet 4 cm above a sphere that slides and turns under it (tests/cloth-system.test.ts's moving solid). */
function movingSolidScene() {
  const R = 0.1, y = 0.76, end_s = 36 * CLOTH_SCENE_STEP_S;
  const scene = createEmptyScene({ sceneId: 'cloth-locality-test', finestCellSize_m: 0.01, extents_m: { x: 1.2, y: 1.6, z: 1.2 } });
  const data = clothGridData({ columns: 24, rows: 24, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2] });
  const cloth: ClothDocument = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton],
    solver: { integrator: 'bdf2', contactPlaneRecordsPerVertex: 200 }, pendingSolids: [{ id: 'turner', start_s: 0 }] };
  scene.cloth = [cloth];
  const at = (x: number) => ({ x, y, z: 0 }), zero = { x: 0, y: 0, z: 0 }, q = (a: number) => ({ w: Math.cos(a / 2), x: 0, y: Math.sin(a / 2), z: 0 });
  scene.rigidBodies = [{ id: 'turner', name: 'turner', shape: 'sphere', dimensions_m: { x: R, y: R, z: R }, density_kg_m3: 1000, position_m: at(0), orientation: q(0),
    linearVelocity_m_s: zero, angularVelocity_rad_s: zero, restitution: 0, friction: 0.5, motion: 'static',
    poseTrack: { keyframes: [0, 0.5, 1].map(f => ({ time_s: f * end_s, position_m: at(f * 0.03), orientation: q(f * 0.6) })) } }];
  assert.deepEqual(validateScene(scene), []);
  return { scene };
}

/** The halt message (a call, so a passed check does not narrow later reads). */
const failureOf = (cloth: ClothSystem) => cloth.failure?.message;

/**
 * The rounds of a step's mandatory detection, before its first Newton step: the round log's entries up to the first
 * that a line-search filter followed. A step in contact is not bit-reproducible run to run, with the switch off too
 * (the DAT pairs land in the pair buffer in the order racing threads append them, and the plane records and sums
 * built from them follow it: performance over parity), but these rounds read only the state the step starts from.
 */
async function mandatoryRounds(det: OgcContactDetector) {
  const log = await det.readRoundLog(), end = log.findIndex(r => r.filters > 0);
  return log.slice(0, end < 0 ? log.length : end + 1).map(r => ({ scanned: r.scanned, collisions: r.collisions, boxRejected: r.boxRejected,
    candidates: r.candidates, datPairs: r.datPairs, held: r.held, maxPrediction: r.maxPrediction, maxEnvelope: r.maxEnvelope }));
}
const total = (rounds: Awaited<ReturnType<typeof mandatoryRounds>>, key: 'scanned' | 'collisions' | 'boxRejected' | 'candidates' | 'datPairs') => rounds.reduce((s, r) => s + r[key], 0);

test('GPU system on a moving solid: from one restored state in contact, the step\'s mandatory detection under each variant is the one with the switch off', { skip: !gpu }, async () => withDevice(async device => {
  const dt = CLOTH_SCENE_STEP_S, tick = 10;
  const { scene } = movingSolidScene();
  const first = await ClothSystem.create(device, scene, { statusInterval: 1 });
  await first.capture([tick * dt]);
  assert.equal(failureOf(first), undefined);
  publishSceneClothSurfaces(scene, first.surfaces);
  // The history endpoint (what save, load and undo carry): every system below restores exactly this state.
  const endpoint = snapshotSceneClothPose(scene);
  first.destroy();
  const run = async (contactLocality?: string) => {
    const cloth = await ClothSystem.create(device, cloneScene(endpoint), { statusInterval: 1, ...(contactLocality ? { contactLocality } : {}) });
    try {
      const det = cloth.contactFor('sheet')!;
      assert.deepEqual(det.locality, parseOgcLocality(contactLocality));
      const [later] = await cloth.capture([(tick + 1) * dt]);
      assert.equal(failureOf(cloth), undefined);
      assert.equal(later![0]!.tick, tick + 1);
      const round = await roundSnapshot(det);
      if (contactLocality) await assertLocalityBuffers(device, det, round, `moving solid ${contactLocality}`);
      return { rounds: await mandatoryRounds(det), counters: round.counters, status: round.counters.status };
    } finally { cloth.destroy(); }
  };
  const off = await run();
  assert.ok(off.counters.bodyContacts! > 0 && total(off.rounds, 'candidates') > 0 && total(off.rounds, 'datPairs') > 0, `the step is in contact with the solid: ${off.counters.bodyContacts} body contacts`);
  assert.deepEqual((await run()).rounds, off.rounds, 'off again');
  // The status is compared, not required clear: the envelope cap's bit is a report, and the step did not halt.
  for (const variant of VARIANTS) { const on = await run(variant); assert.deepEqual(on.rounds, off.rounds, variant); assert.equal(on.status, off.status, variant); }
  console.log(`locality moving solid, the step from tick ${tick}: ${off.rounds.length} mandatory round(s), scanned ${total(off.rounds, 'scanned')} key collisions ${total(off.rounds, 'collisions')} `
    + `box-rejected ${total(off.rounds, 'boxRejected')} candidates ${total(off.rounds, 'candidates')} DAT pairs ${total(off.rounds, 'datPairs')}; body contacts in the step ${off.counters.bodyContacts}: equal under ${VARIANTS.length} variants`);
}));

test('GPU cut, load and undo under the full switch: every rebuilt detector carries the face records of its own topology and its first step detects as the switch off does', { skip: !gpu }, async () => withDevice(async device => {
  const dt = CLOTH_SCENE_STEP_S;
  const sheetScene = () => {
    const scene = createEmptyScene({ sceneId: 'cloth-locality-edit-test' });
    const data = clothGridData({ columns: 12, rows: 12, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2], group: (i, j) => j === 0 && (i === 0 || i === 11) ? 1 : 0 });
    const cloth: ClothDocument = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton],
      solver: { integrator: 'bdf2' }, pendingSolids: [{ id: 'probe', start_s: 1e9 }] };
    scene.cloth = [cloth];
    assert.deepEqual(validateScene(scene), []);
    return scene;
  };
  const run = async (contactLocality?: string) => {
    const scene = sheetScene(), cloth = await ClothSystem.create(device, scene, { statusInterval: 1, ...(contactLocality ? { contactLocality } : {}) });
    const stages: { label: string; faces: number; failure: string | undefined; rounds: Awaited<ReturnType<typeof mandatoryRounds>> }[] = [];
    /** The document just built: its detector's records, then one step (to tick 5) and that step's mandatory detection. */
    const stage = async (label: string, step: boolean) => {
      const det = cloth.contactFor('sheet')!;
      assert.deepEqual(det.locality, parseOgcLocality(contactLocality), `${label}: the rebuilt detector keeps the switch`);
      if (contactLocality) await assertLocalityBuffers(device, det, undefined, label);
      let failure: string | undefined;
      if (step) {
        // A halted capture throws; the halt is compared, not required (the cut sheet's first step is a known open defect).
        try { await cloth.capture([5 * dt]); } catch (e) { failure = String((e as Error).message); }
        if (contactLocality && !failure) await assertLocalityBuffers(device, det, await roundSnapshot(det), `${label}, stepped`);
      }
      stages.push({ label, faces: det.T, failure, rounds: step ? await mandatoryRounds(det) : [] });
      return det;
    };
    try {
      await cloth.capture([4 * dt]);
      assert.equal(failureOf(cloth), undefined);
      publishSceneClothSurfaces(scene, cloth.surfaces);
      // The history endpoint before the cut (what save, load and undo carry).
      const before = snapshotSceneClothPose(scene);
      const built = await stage('built', false);
      // Cut: a slot through the middle of the sheet. The document is rebuilt with new vertices, faces and edges.
      const edited = sceneWithClothStroke(scene, [{ kind: 'box', box: { minimum_m: [-0.06, -1, -0.06], maximum_m: [0.06, 2, 0.06] } }]);
      assert.notEqual(edited, scene);
      cloth.adopt(edited); await cloth.settled();
      const cut = await stage('cut', true);
      assert.ok(cut !== built && cut.T !== built.T, `the cut remeshes the sheet (${built.T} -> ${cut.T} faces)`);
      // Load: the endpoint through JSON, as a saved scene comes back.
      cloth.adopt(JSON.parse(JSON.stringify(before)) as SceneDescription); await cloth.settled();
      const loaded = await stage('loaded', true);
      assert.ok(loaded !== cut && loaded.T === built.T);
      assert.equal(stages[2]!.failure, undefined, 'the loaded sheet steps on');
      // Redo the cut, then undo it: the endpoint staged again.
      cloth.adopt(edited); await cloth.settled();
      await stage('cut again', false);
      cloth.adopt(cloneScene(before)); await cloth.settled();
      const undone = await stage('undone', true);
      assert.ok(undone.T === built.T);
      assert.equal(stages[4]!.failure, undefined, 'the undone sheet steps on');
    } finally { cloth.destroy(); }
    return stages;
  };
  const off = await run(), on = await run('topology+geometry+offsets');
  assert.deepEqual(on, off);
  assert.ok(off[2]!.rounds.length > 0 && total(off[2]!.rounds, 'scanned') > 0, 'the steps scan the grid');
  console.log(`locality edits: ${off.map(s => `${s.label} ${s.faces} faces${s.rounds.length ? `, scanned ${total(s.rounds, 'scanned')} candidates ${total(s.rounds, 'candidates')}` : ''}${s.failure ? ` (${s.failure})` : ''}`).join('; ')}`);
}));

// A whole trajectory, where the baseline has one to compare: the 16 x 16 H_vv rotating sphere is bit-deterministic
// with the switch off (tests/cloth-contact-ladder.test.ts), so every tick's positions and integer counters under the
// switch must be the ones with it off. (With x from the round geometry record in resolvePlane's support projection,
// the compiler folded (x + w) - o differently, a conflict test flipped at a tie at tick 4 and the run failed as
// heldConflict at tick 46; that projection and the least-shift bisection keep the live reads.)
test('GPU trajectory: the deterministic 16 x 16 H_vv rotating sphere is bit-identical to the switch off over 60 ticks, geometry records alone and the full switch', { skip: !gpu }, async () => withDevice(async device => {
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 }));
  const scene: SceneDescription = { ...base, cloth: base.cloth!.map(d => ({ ...d, solver: { ...d.solver!, contactProjectionMetric: 'hvv' as const } })) };
  const run = async (locality?: string) => {
    const cloth = await ClothSystem.create(device, scene, { statusInterval: 1, ...(locality ? { contactLocality: locality } : {}) });
    const det = cloth.contactFor(scene.cloth![0]!.id)!, track: { positions: number[]; counters: Record<string, number> }[] = [];
    try {
      for (let k = 1; k <= 60; k++) {
        const [pose] = (await cloth.capture([k * CLOTH_SCENE_STEP_S]))[0]!;
        const counters: Record<string, number> = {};
        // heldConflictPair is an example pair, stored by whichever conflicting thread writes last.
        for (const [key, v] of Object.entries(await det.readCounters())) if (key !== 'heldConflictPair' && typeof v === 'number' && Number.isInteger(v)) counters[key] = v;
        track.push({ positions: Array.from(pose!.positions_m), counters });
      }
    } finally { cloth.destroy(); }
    return track;
  };
  const off = await run();
  assert.ok(off.some(s => s.counters.redos! > 0) && off.some(s => s.counters.bodyContacts! > 0), 'the run redoes rounds and touches the moving solid');
  for (const variant of ['geometry', 'topology+geometry+offsets']) {
    const on = await run(variant);
    for (let k = 0; k < off.length; k++) {
      assert.deepEqual(on[k]!.counters, off[k]!.counters, `${variant}: tick ${k + 1} counters`);
      assert.deepEqual(on[k]!.positions, off[k]!.positions, `${variant}: tick ${k + 1} positions`);
    }
  }
}));
