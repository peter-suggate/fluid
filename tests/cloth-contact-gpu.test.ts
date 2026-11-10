/**
 * The GPU OGC detector and vertex contact block against the Float64 oracle
 * (tests/cloth-contact-oracle.ts): contact pair sets (no missed or duplicated
 * pair), d_min terms and bounds with velocity envelopes, the DAT division
 * pairs (every pair within the larger envelope), the per-vertex force, Hessian
 * and energy, Planar-DAT and isotropic truncation with the GPU-side
 * re-detection decision, the triangle-triangle intersection counter, and the
 * static solid field's entries, truncation terms and vertex block.
 * Skips without WEBGPU_NODE_MODULE.
 */
import { OGC_SHIFT_WORDS } from '../lib/cloth/contact/ogc-detector-wgsl';
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { OgcContactDetector, type OgcContactConfig } from "../lib/cloth/contact/ogc-detector";
import { buildOgcEdges } from "../lib/cloth/contact/ogc-topology";
import { OGC_STATUS } from "../lib/cloth/contact/ogc-wgsl";
import { buildClothSolidField, clothSolidClearance, clothSolidSampleFacets, sampleClothSolidField, type ClothSolidField } from "../lib/cloth/contact/solid-field";
import { CLOTH_SCENE_STEP_S, ClothSystem } from "../lib/cloth/system";
import { clothAnalyticFreeFall } from "../lib/cloth/scenes/analytic/scenes";
import { clothAnalyticSceneDescription } from "../lib/cloth/scenes/analytic/scene-description";
import {
  activation, boxSdf, solidBallBound, solidContact, solidTruncate, solidKeep, solidTermScale, sphereSdf, bruteForceDatPairs, bruteForceDetection, contactEnergy, countIntersections, datSeparator, dot, globalFeature, merge,
  motionPenetrates, norm, planarTruncate, pointTriangleDistance, rng, scale, sheet, sub, vertexForceHessian, add,
  bruteForcePredictedPairs, predictedConflict, predictedTruncate,
  bruteForceProjection, conViolation, metricProjection, projectionCase, projectionCaseCons, projectionKkt, randomMetric, shrinkCons, solidPieceCons, type ProjectionCase,
  type DatPair, type OracleContact, type OracleMesh, type V3,
} from "./cloth-contact-oracle";

const gpu = !!process.env.WEBGPU_NODE_MODULE;

export async function withContactDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => {
    e.preventDefault();
    const message = (e as GPUUncapturedErrorEvent).error.message;
    if (!errors.length) console.error(`first WebGPU error: ${message}`);
    errors.push(message);
  });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

export function storage(device: GPUDevice, data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, extra = 0) {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra });
  device.queue.writeBuffer(b, 0, data);
  return b;
}

export async function readBuffer(device: GPUDevice, buffer: GPUBuffer, bytes = buffer.size): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, 0, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = read.getMappedRange().slice(0); read.destroy();
  return out;
}

const vec4s = (pts: V3[]) => new Float32Array(pts.flatMap(p => [p[0], p[1], p[2], 0]));
const f32 = (pts: V3[]) => pts.map(p => p.map(Math.fround) as V3);

/** Three jittered, offset sheets stacked within r of each other, plus a sheet folded over them. */
function stackScene(seed: number): OracleMesh {
  const R = rng(seed), h = 0.01;
  const layers = [0, 1.3e-3, 2.5e-3].map((z, i) => {
    const s = sheet(7, 7, h, [i * 0.0031, i * 0.0023, z]);
    return { ...s, x: s.x.map(p => [p[0] + (R() - 0.5) * 1e-3, p[1] + (R() - 0.5) * 1e-3, p[2] + (R() - 0.5) * 4e-4] as V3) };
  });
  const fold = sheet(6, 3, h, [0.004, 0.01, 0]);
  // Arc over the stack: z rises from below the bottom to above the top layer across its width.
  const folded = { ...fold, x: fold.x.map(p => [p[0], p[1], 4.2e-3 + 1.5 * (p[1] - 0.025) * (p[1] - 0.025) * 40 + (R() - 0.5) * 2e-4] as V3) };
  const m = merge(...layers, folded);
  return { ...m, x: f32(m.x) };
}

function topologyOf(mesh: OracleMesh) {
  const triangles = new Uint32Array(mesh.tris.flatMap(t => [...t, 0]));
  const edges = buildOgcEdges(triangles);
  // The oracle indexes edges in buildOgcEdges' order.
  const oracleEdges: [number, number, number, number][] = [];
  for (let e = 0; e < edges.length / 4; e++) oracleEdges.push([edges[4 * e], edges[4 * e + 1], edges[4 * e + 2], edges[4 * e + 3]]);
  return { triangles, edges, oracle: { ...mesh, edges: oracleEdges } };
}

/** Planar-DAT alone (every prediction held), as the oracle's planarTruncate; the predicted tests opt in. */
const baseConfig = (N: number): OgcContactConfig => ({
  r: 2e-3, kc: 1e5, mu: 0.3, epsV: 1e-2, dt: 1 / 60, gravity: [0, 0, -9.81],
  rqBase: 3e-3, rqMax: 2e-2, envelopeScale: 1.5, contactCapacity: 96, pairCapacity: 512 * N, gridCapacity: 128 * N, bucketCount: 4096, roundCap: 2,
  prediction: "none",
});

/** The round's DAT pairs as the oracle's, keyed "v|t" / "e|f" (e < f), with the stored plane. */
function pairsOf(raw: Uint32Array, slots: number, mesh: OracleMesh) {
  const f = new Float32Array(raw.buffer, raw.byteOffset, raw.length);
  const edgeId = new Map(mesh.edges.map((e, i) => [`${e[0]},${e[1]}`, i]));
  const triId = new Map(mesh.tris.map((t, i) => [t.join(","), i]));
  const out: { key: string; pair: DatPair; n: V3; gap: number }[] = [];
  for (let i = 0; i < slots; i++) {
    const b = 8 * i;
    if (raw[b] === 0xffffffff) continue;
    const ee = (raw[b] & 0x80000000) !== 0, a = raw[b] & 0x7fffffff;
    const ids: [number, number, number, number] = [a, raw[b + 1], raw[b + 2], raw[b + 3]];
    const key = ee ? `${edgeId.get(`${ids[0]},${ids[1]}`)}|${edgeId.get(`${ids[2]},${ids[3]}`)}` : `${ids[0]}|${triId.get(ids.slice(1).join(","))}`;
    out.push({ key, pair: { kind: ee ? "ee" : "vt", ids, nA: ee ? 2 : 1 }, n: [f[b + 4], f[b + 5], f[b + 6]], gap: f[b + 7] });
  }
  return out;
}

function envelope(c: OgcContactConfig, v: V3): number {
  const d = Math.hypot(v[0] * c.dt + c.gravity[0] * c.dt ** 2, v[1] * c.dt + c.gravity[1] * c.dt ** 2, v[2] * c.dt + c.gravity[2] * c.dt ** 2);
  return Math.min(c.rqMax, Math.max(c.rqBase, c.r + c.envelopeScale * d));
}

interface Entry { x: number; y: number; z: number; w: number }
function listsOf(contacts: Uint32Array, N: number) {
  const C = contacts[1], countsBase = contacts[2], entriesBase = contacts[3];
  const lists: Entry[][] = [];
  for (let v = 0; v < N; v++) {
    const n = contacts[4 * countsBase + v];
    assert.ok(n <= C, `vertex ${v} list overflow (${n} > ${C})`);
    const l: Entry[] = [];
    for (let i = 0; i < n; i++) { const b = 4 * (entriesBase + v * C + i); l.push({ x: contacts[b], y: contacts[b + 1], z: contacts[b + 2], w: contacts[b + 3] }); }
    lists.push(l);
  }
  return lists;
}

function asOracleContact(e: Entry, mesh: OracleMesh): OracleContact {
  const lambda = new Float32Array(new Uint32Array([e.z]).buffer)[0];
  if (e.y & 0x80000000) {
    const e2 = e.y & 0x0fffffff, a = mesh.edges[e.x], b = mesh.edges[e2];
    return { kind: "ee", ids: [a[0], a[1], b[0], b[1]], feature: 0, s: (e.w & 0xffff) / 65535, t: (e.w >>> 16) / 65535, lambda };
  }
  const t = e.y & 0x0fffffff, tri = mesh.tris[t];
  return { kind: "vf", ids: [e.x, tri[0], tri[1], tri[2]], feature: (e.y >>> 28) & 7, s: 0, t: 0, lambda };
}

/**
 * Run the vertex contact block at the current positions: per vertex (f, e), (H[0], count), H[1], H[2].
 * With lanes > 1 it runs the lane-split forms over lanes 0 .. lanes - 1 and sums them, as a lane-split
 * solver does; H[1].w then holds ogcVertexContactLanes's summed energy.
 */
async function evalVertexBlock(device: GPUDevice, det: OgcContactDetector, positions: GPUBuffer, Xt: V3[], N: number, lanes = 1): Promise<Float32Array> {
  const xt = storage(device, vec4s(Xt));
  const L = det.topology.layout;
  const out = device.createBuffer({ size: 64 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const code = /* wgsl */`
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Xt: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> topo: array<u32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;
fn ogcX(i: u32) -> vec3<f32> { return X[i].xyz; }
fn ogcXt(i: u32) -> vec3<f32> { return Xt[i].xyz; }
fn ogcTri(t: u32) -> vec3<u32> { let b = ${L.trisOffset}u + 4u * t; return vec3<u32>(topo[b], topo[b + 1u], topo[b + 2u]); }
fn ogcEdgeVerts(e: u32) -> vec2<u32> { let b = ${L.edgesOffset}u + 4u * e; return vec2<u32>(topo[b], topo[b + 1u]); }
${OgcContactDetector.vertexBlockWGSL(0, 4)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= ${N}u) { return; }
  var f = vec3<f32>(0.0); var H = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var e = 0.0; var eLanes = 0.0;
  if (${lanes}u == 1u) {
    ogcAccumulateVertexContact(v, ogcX(v), &f, &H);
    e = ogcVertexContactEnergy(v, ogcX(v));
  } else {
    for (var lane = 0u; lane < ${lanes}u; lane++) {
      eLanes += ogcVertexContactLanes(v, ogcX(v), lane, ${lanes}u, &f, &H);
      e += ogcVertexContactEnergyLanes(v, ogcX(v), lane, ${lanes}u);
    }
  }
  outBuf[4u * v] = vec4<f32>(f, e);
  outBuf[4u * v + 1u] = vec4<f32>(H[0], f32(ogcContactCount(v)));
  outBuf[4u * v + 2u] = vec4<f32>(H[1], eLanes);
  outBuf[4u * v + 3u] = vec4<f32>(H[2], 0.0);
}`;
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [positions, xt, det.topo, out, det.solverBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
  device.queue.submit([enc.finish()]);
  return new Float32Array(await readBuffer(device, out));
}

/**
 * The projection's metric as the Newton solver stores it (nwVertexInv: H_vv^-1, two vec4 per vertex, the upper
 * triangle a = (m00, m01, m02, m11), b = (m12, m22)), the identity where Hinv is omitted; and the host function reading it.
 */
const METRIC_WGSL = /* wgsl */`
@group(0) @binding(2) var<storage, read> Minv: array<vec4<f32>>;
fn testMetricInv(v: u32) -> mat3x3<f32> { let a = Minv[2u * v]; let b = Minv[2u * v + 1u]; return mat3x3<f32>(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y); }`;
function metricBuffer(device: GPUDevice, N: number, Hinv?: number[][][]): GPUBuffer {
  const f = new Float32Array(8 * N);
  for (let v = 0; v < N; v++) {
    const M = Hinv?.[v] ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    f.set([M[0]![0]!, M[0]![1]!, M[0]![2]!, M[1]![1]!, M[1]![2]!, M[2]![2]!, 0, 0], 8 * v);
  }
  return storage(device, f);
}

/** ogcProject(v, X_v, Y_v) of the vertex block (the projected arc's projection) for every vertex; in the metric Hinv (H_vv^-1, OgcVertexBlockOptions.metric) where given. */
async function evalProject(device: GPUDevice, det: OgcContactDetector | GPUBuffer, X: V3[], Y: V3[], Hinv?: number[][][]): Promise<V3[]> {
  const N = X.length, xb = storage(device, vec4s(X)), yb = storage(device, vec4s(Y)), mb = Hinv && metricBuffer(device, N, Hinv);
  const out = device.createBuffer({ size: 16 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const code = /* wgsl */`
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;
fn ogcX(i: u32) -> vec3<f32> { return X[i].xyz; }
fn ogcXt(i: u32) -> vec3<f32> { return X[i].xyz; }
// The projection reads no topology.
fn ogcTri(t: u32) -> vec3<u32> { return vec3<u32>(t); }
fn ogcEdgeVerts(e: u32) -> vec2<u32> { return vec2<u32>(e); }
${Hinv ? `${METRIC_WGSL}\n${OgcContactDetector.vertexBlockWGSL(0, 4, false, 'testMetricInv')}` : OgcContactDetector.vertexBlockWGSL(0, 4)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= ${N}u) { return; }
  outBuf[v] = vec4<f32>(ogcProject(v, X[v].xyz, Y[v].xyz), 0.0);
}`;
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: ([[0, xb], [1, yb], ...(mb ? [[2, mb] as const] : []), [3, out], [4, 'solverBuffer' in det ? det.solverBuffer : det]] as const).map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
  device.queue.submit([enc.finish()]);
  const got = toV3(new Float32Array(await readBuffer(device, out)), N);
  xb.destroy(); yb.destroy(); mb?.destroy(); out.destroy();
  return got;
}

async function detect(device: GPUDevice, mesh: OracleMesh, vel: V3[], config: OgcContactConfig) {
  const { triangles, edges, oracle } = topologyOf(mesh);
  const positions = storage(device, vec4s(mesh.x)), velocities = storage(device, vec4s(vel));
  const det = await OgcContactDetector.create(device, { vertexCount: mesh.x.length, triangles, edges }, vec4s(mesh.x), config, { positions, velocities });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass);
  pass.end(); device.queue.submit([enc.finish()]);
  return { det, positions, velocities, oracle };
}

test("GPU detection matches the Float64 brute force: pairs, d_min and bounds with velocity envelopes", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    for (const seed of [3, 11]) {
      const mesh = stackScene(seed), N = mesh.x.length, R = rng(seed + 100);
      const vel: V3[] = mesh.x.map(() => [(R() - 0.5) * 0.4, (R() - 0.5) * 0.4, (R() - 0.5) * 0.4]);
      const config: OgcContactConfig = { ...baseConfig(N), truncation: "isotropic" };
      const { det, oracle } = await detect(device, mesh, vel, config);
      const rq = vel.map(v => envelope(config, v));
      const ref = bruteForceDetection(oracle, config.r, rq, 0.45, 2e-6);
      const contacts = new Uint32Array(await readBuffer(device, det.contacts));
      const xdet = new Float32Array(await readBuffer(device, det.det));
      const counters = await det.readCounters();
      assert.equal(counters.status, 0, `status ${OgcContactDetector.describeStatus(counters.status)}`);
      const lists = listsOf(contacts, N);
      // Vertex-facet: each contact sits once in its vertex's list and once in each feature vertex's list.
      const vf = new Map<string, number>(), ee = new Map<string, number>();
      let entries = 0;
      lists.forEach((l, v) => l.forEach(e => {
        entries++;
        const c = asOracleContact(e, oracle);
        if (c.kind === "vf") {
          if (e.x !== v) return;
          const key = `${v}|${globalFeature(oracle, e.y & 0x0fffffff, c.feature)}`;
          vf.set(key, (vf.get(key) ?? 0) + 1);
          const dRef = ref.dminV[v];
          assert.ok(c.lambda > 0 && Number.isFinite(c.lambda));
          assert.ok(c.lambda <= -activation(Math.max(dRef, 1e-9), config.r, config.kc)[1] * (1 + 1e-3) + 1e-3);
        } else if (oracle.edges[e.x][0] === v) {
          const key = `${e.x}|${e.y & 0x0fffffff}`;
          ee.set(key, (ee.get(key) ?? 0) + 1);
        }
      }));
      for (const [k, n] of vf) assert.equal(n, 1, `vertex-facet pair ${k} duplicated`);
      for (const [k, n] of ee) assert.equal(n, 1, `edge-edge pair ${k} duplicated`);
      const strict = (s: Set<string>, amb: Set<string>) => new Set([...s].filter(k => !amb.has(k)));
      const gpuVf = new Set([...vf.keys()].filter(k => !ref.ambiguousVF.has(k)));
      assert.deepEqual([...gpuVf].sort(), [...strict(ref.vf, ref.ambiguousVF)].sort(), "vertex-facet pair set");
      const gpuEe = new Set([...ee.keys()].filter(k => !ref.ambiguousEE.has(k)));
      assert.deepEqual([...gpuEe].sort(), [...strict(ref.ee, ref.ambiguousEE)].sort(), "edge-edge pair set");
      assert.ok(ref.vf.size > 50 && ref.ee.size > 5, `the scene exercises contact (${ref.vf.size} vf, ${ref.ee.size} ee)`);
      const featureVertices = (k: string) => k.includes("|f:") ? 3 : k.includes("|e:") ? 2 : 1;
      const expected = [...vf.keys()].reduce((s, k) => s + 1 + featureVertices(k), 0) + 2 * ee.size;
      assert.equal(entries, expected, "every contact is in each participant's list exactly once");
      for (let v = 0; v < N; v++) {
        const b = xdet[4 * v + 3];
        assert.ok(Math.abs(b - ref.bound[v]) <= 1e-5 * ref.bound[v] + 1e-9, `bound ${v}: ${b} vs ${ref.bound[v]}`);
        assert.ok(b <= 0.45 * rq[v] * (1 + 1e-6));
      }
      assert.equal(counters.detections, 1); assert.equal(counters.steps, 1);
      assert.equal(counters.pairEntries, entries);
      assert.equal(counters.datPairs, 0, "isotropic mode records no division pairs");
      det.destroy();
    }
  });
});

test("GPU DAT division pairs: every pair within its two caps (with sepEps and the rounding), once, with a certified plane", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    for (const seed of [3, 11]) {
      const mesh = stackScene(seed), N = mesh.x.length, R = rng(seed + 200);
      const vel: V3[] = mesh.x.map(() => [(R() - 0.5) * 0.8, (R() - 0.5) * 0.8, (R() - 0.5) * 0.8]);
      // Cells stay at the mean edge length, so 4 cm envelopes reach many cells each.
      const config: OgcContactConfig = { ...baseConfig(N), rqMax: 4e-2, gridCapacity: 2048 * N };
      const { det, oracle } = await detect(device, mesh, vel, config);
      const rq = vel.map(v => envelope(config, v));
      const ref = bruteForceDatPairs(oracle, rq, 2e-6, det);
      const c = await det.readCounters();
      assert.equal(c.status, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
      const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c.peakPairSlots, oracle);
      assert.equal(got.length, c.datPairs);
      const seen = new Map<string, number>();
      for (const g of got) seen.set(`${g.pair.kind}:${g.key}`, (seen.get(`${g.pair.kind}:${g.key}`) ?? 0) + 1);
      for (const [k, n] of seen) assert.equal(n, 1, `pair ${k} recorded once`);
      const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
      assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "vertex-triangle pairs");
      assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "edge-edge pairs");
      assert.ok(ref.vt.size > 500 && ref.ee.size > 500, `the scene exercises DAT (${ref.vt.size} vt, ${ref.ee.size} ee)`);
      for (const g of got) {
        const sep = datSeparator(g.pair, oracle.x, 1e-6)!;
        assert.ok(sep && g.gap > 0, "certified");
        assert.ok(Math.abs(g.gap - sep.gap) <= 1e-6 + 1e-4 * sep.gap, `gap ${g.gap} vs ${sep.gap}`);
        assert.ok(dot(g.n, sep.n) > 1 - 1e-4, `plane normal of ${g.key}`);
      }
      const xdet = new Float32Array(await readBuffer(device, det.det));
      for (let v = 0; v < N; v++) assert.ok(Math.abs(xdet[4 * v + 3] - 0.45 * rq[v]) <= 1e-6 * rq[v], "planar cap 0.5 gamma_r r_q,v");
      const planesArgs = new Uint32Array(await readBuffer(device, det.planesArgs));
      assert.equal(planesArgs[0], Math.ceil(c.peakPairSlots / 64));
      det.destroy();
    }
  });
});

test("GPU envelope cap (E12 iii): with envelopeMeshCap every envelope is at most gamma_r h_min, and a contact radius above it fails", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(3), N = mesh.x.length, R = rng(1204);
    let hmin = Infinity;
    for (const [a, b, c] of mesh.tris) {
      const ab = sub(mesh.x[b], mesh.x[a]), ac = sub(mesh.x[c], mesh.x[a]), bc = sub(mesh.x[c], mesh.x[b]);
      const cr = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]] as V3;
      hmin = Math.min(hmin, norm(cr) / Math.max(norm(ab), norm(ac), norm(bc)));
    }
    // 0.8 m/s gives envelopes of r + 1.5 |v dt| ~ 2 cm, the cap ~ 6 mm.
    const vel: V3[] = mesh.x.map(() => [(R() - 0.5) * 1.6, (R() - 0.5) * 1.6, (R() - 0.5) * 1.6]);
    const config: OgcContactConfig = { ...baseConfig(N), envelopeMeshCap: true };
    const { det, oracle } = await detect(device, mesh, vel, config);
    const top = Math.min(config.rqMax, 0.9 * hmin);
    assert.ok(Math.abs(det.rqTop - top) <= 1e-6 * top && top < config.rqMax, `rqTop ${det.rqTop} vs gamma_r h_min ${top}`);
    const c = await det.readCounters();
    // The cap is reported (status envelopeCap), and nothing else.
    assert.equal(c.status, OGC_STATUS.envelopeCap, `status ${OgcContactDetector.describeStatus(c.status)}`);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const rq = vel.map(v => Math.min(envelope(config, v), det.rqTop));
    let capped = 0;
    for (let v = 0; v < N; v++) {
      assert.ok(Math.abs(xdet[4 * v + 3] - 0.45 * rq[v]) <= 1e-6 * rq[v], `vertex ${v}: cap ${xdet[4 * v + 3]} vs 0.45 min(envelope, rqTop)`);
      if (envelope(config, vel[v]) > det.rqTop) capped++;
    }
    assert.ok(capped > N / 2, `${capped} of ${N} envelopes capped`);
    // The pairs are the oracle's at the capped envelopes, and each spans two sheets: a sheet at its build
    // (rest) shape pairs with nothing of itself (sheets of 64 vertices, then the 28 of the arc).
    const ref = bruteForceDatPairs(oracle, rq, 2e-6, det);
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c.peakPairSlots, oracle);
    const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "vertex-triangle pairs");
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "edge-edge pairs");
    const sheetOf = (i: number) => Math.min(3, Math.floor(i / 64));
    assert.ok(got.length > 0 && got.every(g => g.pair.ids.some(i => sheetOf(i) !== sheetOf(g.pair.ids[0]))), "no pair within one resting sheet");
    det.destroy();
    // A contact radius above gamma_r h_min cannot be met by any envelope: explicit failure.
    const { triangles, edges } = topologyOf(mesh);
    const bad: OgcContactConfig = { ...config, r: 1.2 * top, rqBase: 1.3 * top, rqMax: 2e-2 };
    await assert.rejects(OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(mesh.x), bad, { positions: storage(device, vec4s(mesh.x)), velocities: storage(device, vec4s(vel)) }), /envelope cap gamma_r h_min/);
  });
});

test("GPU vertex contact block: force, Hessian and energy against the oracle, and its lane-split forms sum to it", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(5), N = mesh.x.length, R = rng(77);
    const config: OgcContactConfig = { ...baseConfig(N), truncation: "isotropic" };
    const { det, positions, oracle } = await detect(device, mesh, mesh.x.map(() => [0, 0, 0]), config);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    // Evaluate inside the bounds, with a step start elsewhere so friction acts.
    const X = f32(mesh.x.map((p, v) => { const b = xdet[4 * v + 3] * 0.9; return [p[0] + (R() - 0.5) * b, p[1] + (R() - 0.5) * b, p[2] + (R() - 0.5) * b] as V3; }));
    const Xt = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 2e-4, p[1] + (R() - 0.5) * 2e-4, p[2]] as V3));
    device.queue.writeBuffer(positions, 0, vec4s(X));
    const res = await evalVertexBlock(device, det, positions, Xt, N);
    const lists = listsOf(new Uint32Array(await readBuffer(device, det.contacts)), N);
    const P = { r: config.r, k: config.kc, mu: config.mu, epsV: config.epsV, dt: config.dt };
    let active = 0;
    for (let v = 0; v < N; v++) {
      let f: V3 = [0, 0, 0], e = 0;
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const ent of lists[v]) {
        const c = asOracleContact(ent, oracle), role = c.ids.indexOf(v);
        const q = vertexForceHessian(c, X, Xt, P, role);
        f = [f[0] + q.f[0], f[1] + q.f[1], f[2] + q.f[2]];
        for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i][k] += q.H[i][k];
        e += contactEnergy(c, X, Xt, P);
      }
      if (lists[v].length) active++;
      assert.equal(res[16 * v + 7], lists[v].length);
      const fs = Math.max(1, Math.hypot(...f)), hs = Math.max(1, ...H.flat().map(Math.abs));
      for (let k = 0; k < 3; k++) assert.ok(Math.abs(res[16 * v + k] - f[k]) <= 2e-3 * fs, `vertex ${v} f[${k}] ${res[16 * v + k]} vs ${f[k]}`);
      assert.ok(Math.abs(res[16 * v + 3] - e) <= 2e-3 * Math.max(1e-6, Math.abs(e)) + 1e-9, `vertex ${v} energy ${res[16 * v + 3]} vs ${e}`);
      for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++)
        assert.ok(Math.abs(res[16 * v + 4 * (col + 1) + row] - H[row][col]) <= 2e-3 * hs, `vertex ${v} H`);
    }
    assert.ok(active > 100);
    // Lane-split: three lanes over entries 0, 3, ... / 1, 4, ... / 2, 5, ... sum to the whole block.
    const split = await evalVertexBlock(device, det, positions, Xt, N, 3);
    for (let v = 0; v < N; v++) {
      const scale = Math.max(1, ...Array.from(res.subarray(16 * v, 16 * v + 16)).map(Math.abs));
      for (const k of [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]) assert.ok(Math.abs(split[16 * v + k]! - res[16 * v + k]!) <= 1e-5 * scale, `vertex ${v} lanes [${k}] ${split[16 * v + k]} vs ${res[16 * v + k]}`);
      assert.ok(Math.abs(split[16 * v + 11]! - res[16 * v + 3]!) <= 1e-5 * Math.max(1e-6, Math.abs(res[16 * v + 3]!)), `vertex ${v} lane energies ${split[16 * v + 11]} vs ${res[16 * v + 3]}`);
    }
    det.destroy();
  });
});

const runPass = async (device: GPUDevice, fn: (pass: GPUComputePassEncoder) => void) => {
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass(); fn(pass); pass.end(); device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
};
const toV3 = (a: Float32Array, N: number) => Array.from({ length: N }, (_, v) => [a[4 * v], a[4 * v + 1], a[4 * v + 2]] as V3);

test("GPU Planar-DAT truncation matches the oracle and keeps large motions penetration-free", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(9), N = mesh.x.length, R = rng(5);
    // The step's motion: layers driven into each other by several gaps, plus sliding.
    const motion = mesh.x.map(p => [(R() - 0.5) * 6e-3, (R() - 0.5) * 6e-3, (p[2] < 1.9e-3 ? 1 : -1) * R() * 5e-3] as V3);
    const config = baseConfig(N);
    const vel = motion.map(d => [d[0] / config.dt, d[1] / config.dt, d[2] / config.dt - config.gravity[2] * config.dt] as V3);
    const { det, positions, oracle } = await detect(device, mesh, vel, config);
    const X = f32(mesh.x), Xnew = f32(X.map((p, v) => [p[0] + motion[v][0], p[1] + motion[v][1], p[2] + motion[v][2]] as V3));
    assert.ok(motionPenetrates(oracle, X, Xnew), "the raw motion penetrates");
    const c0 = await det.readCounters();
    const pairs = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.peakPairSlots, oracle).map(g => g.pair);
    const rq = vel.map(v => envelope(config, v));
    device.queue.writeBuffer(positions, 0, vec4s(Xnew));
    await runPass(device, pass => { det.encodeTruncate(pass); det.encodeDecide(pass); });
    const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
    const ref = planarTruncate(X, Xnew, pairs, rq.map(q => 0.45 * q), 0.9, 1e-6);
    let cut = 0;
    for (let v = 0; v < N; v++) {
      const full = norm(sub(Xnew[v], X[v]));
      assert.ok(norm(sub(got[v], ref.x[v])) <= 2e-7 + 2e-4 * full, `vertex ${v}: ${got[v]} vs ${ref.x[v]}`);
      if (norm(sub(ref.x[v], Xnew[v])) > 1e-6 * full) cut++;
    }
    assert.equal(motionPenetrates(oracle, X, got), false, "the truncated motion is penetration-free");
    assert.equal(countIntersections({ ...oracle, x: got }), 0);
    const c = await det.readCounters();
    assert.ok(Math.abs(c.truncated - cut) <= 2, `truncated ${c.truncated} vs ${cut}`);
    assert.ok(cut > 0.2 * N && cut < N, `the motion truncates part of the cloth (${cut} of ${N})`);
    // Truncating again re-places the planes for the shorter dX: motion only shrinks, and stays penetration-free.
    await runPass(device, pass => det.encodeTruncate(pass));
    const again = toV3(new Float32Array(await readBuffer(device, positions)), N);
    for (let v = 0; v < N; v++) assert.ok(norm(sub(again[v], X[v])) <= norm(sub(got[v], X[v])) * (1 + 1e-6) + 1e-9, `monotone at ${v}`);
    assert.equal(motionPenetrates(oracle, X, again), false);
    det.destroy();
  });
});

/**
 * The Newton forms of the vertex block at the current positions, with p in dir: per vertex
 * ((H p)_v, owned energy), (f, per-vertex energy).
 */
async function evalNewtonBlock(device: GPUDevice, det: OgcContactDetector, positions: GPUBuffer, xt: GPUBuffer, dir: GPUBuffer, N: number): Promise<Float32Array> {
  const L = det.topology.layout;
  const out = device.createBuffer({ size: 32 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const code = /* wgsl */`
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Xt: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> topo: array<u32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> D: array<vec4<f32>>;
fn ogcX(i: u32) -> vec3<f32> { return X[i].xyz; }
fn ogcXt(i: u32) -> vec3<f32> { return Xt[i].xyz; }
fn ogcDir(i: u32) -> vec3<f32> { return D[i].xyz; }
fn ogcTri(t: u32) -> vec3<u32> { let b = ${L.trisOffset}u + 4u * t; return vec3<u32>(topo[b], topo[b + 1u], topo[b + 2u]); }
fn ogcEdgeVerts(e: u32) -> vec2<u32> { let b = ${L.edgesOffset}u + 4u * e; return vec2<u32>(topo[b], topo[b + 1u]); }
${OgcContactDetector.vertexBlockWGSL(0, 4, true)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= ${N}u) { return; }
  var f = vec3<f32>(0.0); var H = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  let e = ogcVertexContactLanes(v, ogcX(v), 0u, 1u, &f, &H);
  // Two lanes for the global forms, as a lane-split solver sums them.
  let hv = ogcVertexContactHvLanes(v, 0u, 2u) + ogcVertexContactHvLanes(v, 1u, 2u);
  let eo = ogcContactEnergyOwnedLanes(v, 0u, 2u) + ogcContactEnergyOwnedLanes(v, 1u, 2u);
  outBuf[2u * v] = vec4<f32>(hv, eo);
  outBuf[2u * v + 1u] = vec4<f32>(f, e);
}`;
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: ([[0, positions], [1, xt], [2, det.topo], [3, out], [4, det.solverBuffer], [5, dir]] as [number, GPUBuffer][]).map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
  device.queue.submit([enc.finish()]);
  const res = new Float32Array(await readBuffer(device, out));
  out.destroy();
  return res;
}

test("GPU Newton contact terms: H p is symmetric, PSD and has the vertex block's diagonal; the owned energy counts each pair once and its gradient is the block's force", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(5), N = mesh.x.length, R = rng(91);
    const config: OgcContactConfig = { ...baseConfig(N), truncation: "isotropic" };
    const { det, positions, oracle } = await detect(device, mesh, mesh.x.map(() => [0, 0, 0]), config);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const X = f32(mesh.x.map((p, v) => { const b = xdet[4 * v + 3] * 0.9; return [p[0] + (R() - 0.5) * b, p[1] + (R() - 0.5) * b, p[2] + (R() - 0.5) * b] as V3; }));
    const Xt = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 2e-4, p[1] + (R() - 0.5) * 2e-4, p[2]] as V3));
    device.queue.writeBuffer(positions, 0, vec4s(X));
    const xt = storage(device, vec4s(Xt));
    const randomDir = () => f32(mesh.x.map(() => [R() - 0.5, R() - 0.5, R() - 0.5] as V3));
    const P = randomDir(), Q = randomDir();
    const dir = storage(device, vec4s(P));
    const at = async (d: V3[]) => { device.queue.writeBuffer(dir, 0, vec4s(d)); return evalNewtonBlock(device, det, positions, xt, dir, N); };
    const hv = (res: Float32Array, v: number): V3 => [res[8 * v]!, res[8 * v + 1]!, res[8 * v + 2]!];
    const rp = await at(P), rq = await at(Q);
    // Symmetric and PSD: q.(H p) = p.(H q), p.(H p) >= 0.
    let qHp = 0, pHq = 0, pHp = 0, scale = 0;
    for (let v = 0; v < N; v++) { qHp += dot(Q[v], hv(rp, v)); pHq += dot(P[v], hv(rq, v)); pHp += dot(P[v], hv(rp, v)); scale += norm(P[v]) * norm(hv(rp, v)); }
    assert.ok(scale > 0, "contacts act");
    assert.ok(Math.abs(qHp - pHq) <= 1e-4 * scale, `symmetric: ${qHp} vs ${pHq} (scale ${scale})`);
    assert.ok(pHp >= -1e-5 * scale, `PSD: ${pHp}`);
    // The diagonal: p = e_a at vertices sharing no pair, so (H p)_v = H_vv e_a, the vertex block's column.
    const lists = listsOf(new Uint32Array(await readBuffer(device, det.contacts)), N);
    const ids = (ent: Entry) => asOracleContact(ent, oracle).ids;
    const taken = new Set<number>(), probe: number[] = [];
    for (let v = 0; v < N; v++) {
      if (!lists[v]!.length) continue;
      const nb = new Set(lists[v]!.flatMap(ids));
      if ([...nb].some(i => taken.has(i))) continue;
      probe.push(v); for (const i of nb) taken.add(i);
    }
    assert.ok(probe.length > 10, `${probe.length} probe vertices`);
    const block = await evalVertexBlock(device, det, positions, Xt, N);
    for (let a = 0; a < 3; a++) {
      const E = mesh.x.map(() => [0, 0, 0] as V3);
      for (const v of probe) E[v]![a] = 1;
      const r = await at(E);
      for (const v of probe) {
        const col: V3 = [block[16 * v + 4 * (a + 1)]!, block[16 * v + 4 * (a + 1) + 1]!, block[16 * v + 4 * (a + 1) + 2]!];
        const s = Math.max(1, norm(col));
        assert.ok(norm(sub(hv(r, v), col)) <= 1e-4 * s, `vertex ${v} column ${a}: ${hv(r, v)} vs ${col}`);
      }
    }
    // Each pair once: the owned energies sum to the unique pairs' oracle energies; an EE pair sits in four lists.
    const params = { r: config.r, k: config.kc, mu: config.mu, epsV: config.epsV, dt: config.dt };
    const unique = new Map<string, number>(), eeCopies = new Map<string, number>();
    for (let v = 0; v < N; v++) for (const ent of lists[v]!) {
      const ee = (ent.y & 0x80000000) !== 0, key = ee ? [ent.x, ent.y & 0x0fffffff].sort((x, y) => x - y).join("|") : `${ent.x}|${ent.y}`;
      if (ee) eeCopies.set(key, (eeCopies.get(key) ?? 0) + 1);
      unique.set(key, contactEnergy(asOracleContact(ent, oracle), X, Xt, params));
    }
    for (const [key, n] of eeCopies) assert.equal(n, 4, `EE pair ${key} in ${n} lists`);
    assert.ok(eeCopies.size > 10, `${eeCopies.size} EE pairs`);
    const owned = (res: Float32Array) => { let e = 0; for (let v = 0; v < N; v++) e += res[8 * v + 3]!; return e; };
    const ref = [...unique.values()].reduce((x, y) => x + y, 0);
    assert.ok(Math.abs(owned(rp) - ref) <= 2e-3 * ref, `owned energy ${owned(rp)} vs ${ref} over ${unique.size} pairs`);
    // Its directional derivative is -f.p (central difference along a short p).
    const h = 2e-6;
    let fp = 0;
    for (let v = 0; v < N; v++) fp -= dot([rp[8 * v + 4]!, rp[8 * v + 5]!, rp[8 * v + 6]!], P[v]!);
    const shifted = async (s: number) => { device.queue.writeBuffer(positions, 0, vec4s(X.map((x, v) => [x[0] + s * P[v]![0], x[1] + s * P[v]![1], x[2] + s * P[v]![2]] as V3))); return owned(await at(P)); };
    const fd = (await shifted(h) - await shifted(-h)) / (2 * h);
    assert.ok(Math.abs(fd - fp) <= 2e-2 * Math.abs(fp), `dE/ds ${fd} vs -f.p ${fp}`);
    xt.destroy(); dir.destroy(); det.destroy();
  });
});

test("GPU line-search filter: x + alpha d is certified for every alpha up to alpha_v, alpha_min is their min, and the whole-step filter stretches no edge", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(9), N = mesh.x.length, R = rng(5);
    const motion = f32(mesh.x.map(p => [(R() - 0.5) * 6e-3, (R() - 0.5) * 6e-3, (p[2] < 1.9e-3 ? 1 : -1) * R() * 5e-3] as V3));
    // Three sheets 2 mm apart and 5 mm motions through each other: some 170 plane records a vertex at
    // the first filter (records past the capacity hold their vertex and fail the step).
    const config: OgcContactConfig = { ...baseConfig(N), newton: true, planeRecordCapacity: 256 * N };
    const vel = motion.map(d => [d[0] / config.dt, d[1] / config.dt, d[2] / config.dt - config.gravity[2] * config.dt] as V3);
    const { triangles, edges, oracle } = topologyOf(mesh);
    const positions = storage(device, vec4s(mesh.x)), velocities = storage(device, vec4s(vel)), direction = storage(device, vec4s(motion));
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(mesh.x), config, { positions, velocities, direction });
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
    const X = f32(mesh.x), full = X.map((x, v) => [x[0] + motion[v]![0], x[1] + motion[v]![1], x[2] + motion[v]![2]] as V3);
    assert.ok(motionPenetrates(oracle, X, full), "the whole step penetrates");
    await runPass(device, pass => det.encodeFilter(pass));
    const alpha = new Float32Array(await readBuffer(device, det.alpha, 4 * (N + 1)));
    let lo = 1, bound = 0;
    for (let v = 0; v < N; v++) { assert.ok(alpha[v]! >= 0 && alpha[v]! <= 1, `alpha_${v} ${alpha[v]}`); lo = Math.min(lo, alpha[v]!); if (alpha[v]! < 1) bound++; }
    assert.equal(alpha[N], lo, "alpha_min");
    assert.ok(lo > 0 && lo < 1 && bound > 0.05 * N, `alpha_min ${lo}, ${bound} vertices bound`);
    // A certified point: the truncation leaves it (to f32 rounding), no motion to it penetrates, no faces cross.
    const certified = async (Y: V3[], what: string) => {
      device.queue.writeBuffer(positions, 0, vec4s(Y));
      await runPass(device, pass => det.encodeTruncate(pass));
      const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
      for (let v = 0; v < N; v++) assert.ok(norm(sub(got[v]!, Y[v]!)) <= 2e-8 + 1e-5 * norm(motion[v]!), `${what}: truncation moves vertex ${v} by ${norm(sub(got[v]!, Y[v]!))}`);
      assert.equal(motionPenetrates(oracle, X, Y), false, `${what}: penetration-free`);
      assert.equal(countIntersections({ ...oracle, x: Y }), 0, `${what}: no intersections`);
    };
    const along = (k: (v: number) => number) => f32(X.map((x, v) => [x[0] + k(v) * motion[v]![0], x[1] + k(v) * motion[v]![1], x[2] + k(v) * motion[v]![2]] as V3));
    const global = along(() => lo);
    await certified(global, "alpha_min");
    await certified(along(() => 0.5 * lo), "alpha_min / 2");
    // Each alpha_v certifies its own vertex: every vertex at its own alpha_v at once is certified too.
    await certified(along(v => alpha[v]!), "alpha_v");
    // The strain-limit breaks of per-vertex truncation: an edge is convex along the segment, so the
    // global step never stretches an edge past both ends of the step; the per-vertex truncation of
    // the whole step (each residual towards its own anchor) does.
    const len = (Y: V3[], e: [number, number, number, number]) => norm(sub(Y[e[0]]!, Y[e[1]]!));
    device.queue.writeBuffer(positions, 0, vec4s(full));
    await runPass(device, pass => det.encodeTruncate(pass));
    const truncated = toV3(new Float32Array(await readBuffer(device, positions)), N);
    let globalOver = 0, truncOver = 0;
    for (const e of oracle.edges) {
      const limit = Math.max(len(X, e), len(full, e)) * (1 + 1e-6) + 1e-9;
      if (len(global, e) > limit) globalOver++;
      if (len(truncated, e) > limit) truncOver++;
    }
    assert.equal(globalOver, 0, "the global step stretches no edge past both ends");
    console.log(`filter: alpha_min ${lo.toFixed(4)}, ${bound} of ${N} vertices bound; per-vertex truncation of the whole step stretches ${truncOver} of ${oracle.edges.length} edges past both ends`);
    // Fixed planes: from the filtered iterate the rest of the direction filters again, from a certified start.
    device.queue.writeBuffer(positions, 0, vec4s(global));
    device.queue.writeBuffer(direction, 0, vec4s(f32(motion.map(m => [m[0] * (1 - lo), m[1] * (1 - lo), m[2] * (1 - lo)] as V3))));
    await runPass(device, pass => det.encodeFilter(pass));
    const again = new Float32Array(await readBuffer(device, det.alpha, 4 * (N + 1)));
    const counters = await det.readCounters();
    assert.equal(counters.status & OGC_STATUS.planeRecordOverflow, 0, `plane records within the capacity: peak ${counters.peakPlaneRecords} of ${256 * N}`);
    console.log(`filter plane records: peak ${counters.peakPlaneRecords} (${(counters.peakPlaneRecords / N).toFixed(1)} a vertex)`);
    assert.ok(again[N]! > 0, `the second filter moves on: alpha_min ${again[N]}`);
    await certified(f32(global.map((x, v) => [x[0] + again[N]! * (1 - lo) * motion[v]![0], x[1] + again[N]! * (1 - lo) * motion[v]![1], x[2] + again[N]! * (1 - lo) * motion[v]![2]] as V3)), "second step");
    // The projected arc (ogcProject onto this filter's records, about the certified iterate): every
    // vertex's trial projected onto its certificates at once is certified too; the projection never
    // moves a vertex further than its trial, a vertex its certificates keep whole comes back bit-exactly, and
    // the bound ones are carried on past their alpha_v.
    for (const k of [1, 0.5]) {
      const Y = f32(global.map((x, v) => [x[0] + k * (1 - lo) * motion[v]![0], x[1] + k * (1 - lo) * motion[v]![1], x[2] + k * (1 - lo) * motion[v]![2]] as V3));
      const Pk = await evalProject(device, det, global, Y);
      let kept = 0, free = 0, projected = 0, keptArc = 0, keptFilter = 0;
      for (let v = 0; v < N; v++) {
        const m = norm(sub(Pk[v]!, global[v]!)), dl = norm(sub(Y[v]!, global[v]!));
        assert.ok(m <= dl * (1 + 1e-5) + 1e-9, `vertex ${v}: the projection moves ${m}, past the trial's ${dl}`);
        if (again[v]! >= 1) { free++; if (Pk[v]!.every((u, i) => u === Y[v]![i])) kept++; }
        else if (dl > 0) { projected++; keptArc += m / dl; keptFilter += Math.min(again[v]! / k, 1); }
      }
      await certified(Pk, `projected arc at alpha ${k}`);
      assert.equal(kept, free, `alpha ${k}: ${kept} of ${free} unbound vertices kept bit-exactly (a solver reads any change as a cut)`);
      console.log(`projected arc at alpha ${k}: ${kept} of ${free} unbound vertices kept; the ${projected} bound ones move ${(keptArc / Math.max(projected, 1)).toFixed(3)} of their trial on average (the filter's alpha_v keeps ${(keptFilter / Math.max(projected, 1)).toFixed(3)})`);
    }
    positions.destroy(); velocities.destroy(); direction.destroy(); det.destroy();
  });
});

/**
 * A contact buffer holding only the filter's records for ogcProject (its sub-header at vec4 12, the
 * level header, each vertex's fixed records, segments and plane records), one vertex per case, each
 * case's anchor at c[v] in the world. The moving solid's end pose carries the frame of every body
 * piece, so all the cases with one share it (bodyRb). order: each vertex's plane records reversed.
 */
/**
 * The projection's contact buffer for `cases`; with `extra`, each plane record's far bit (after the pool) and one
 * field image for both solids (the moving one's read in its end-pose frame, R = Rb^T), at contact radius r.
 */
function projectionBuffer(cases: ProjectionCase[], c: V3[], bodyRb: number[][], levelled: boolean, reversed = false,
  extra?: { far: boolean[][]; field: ClothSolidField; r: number }): Uint32Array<ArrayBuffer> {
  const N = cases.length, fixed = 14, segVec = fixed + 8 * N, recVec = segVec + Math.ceil(N / 2);
  const recCap = cases.reduce((s, k) => s + k.planes.length, 0), farVec = recVec + recCap, fieldVec = farVec + Math.ceil(recCap / 128);
  const total = fieldVec + (extra ? Math.ceil(extra.field.words.length / 4) : 0);
  const u = new Uint32Array(4 * total), f = new Float32Array(u.buffer);
  const put = (vec: number, q: number[]) => q.forEach((x, i) => { f[4 * vec + i] = x; });
  u[2 * 4 + 3] = 12;
  u[4 * 12 + 3] = recCap;
  if (extra) { u.set(extra.field.words, 4 * fieldVec); u[8] = 4 * fieldVec; u[12] = 4 * fieldVec; f[4] = extra.r; }
  // Pose 1 (words 32..47): R = Rb^T, column j at word 36 + 4j is row j of Rb.
  for (let j = 0; j < 3; j++) put(9 + j, bodyRb[j]!);
  for (let j = 0; j < 3; j++) put(5 + j, bodyRb[j]!);
  u[4 * 12] = fixed; u[4 * 12 + 1] = 4 * segVec; u[4 * 12 + 2] = recVec;
  u[4 * 13 + 1] = levelled ? 1 : 0;
  let next = 0;
  cases.forEach((k, v) => {
    const fb = fixed + 8 * v, cv = c[v]!;
    put(fb, [...cv, k.R]);
    for (const [slot, piece] of [[1, k.solid], [4, k.body]] as const) {
      if (!piece) { put(fb + slot, [0, 0, 0, 0]); continue; }
      put(fb + slot, [...add(cv, piece.o), 1]); put(fb + slot + 1, [...piece.q, piece.Rq]); put(fb + slot + 2, [...piece.D, 0]);
    }
    put(fb + 7, [0, 0, 0, 0]);
    u[4 * segVec + 2 * v] = next; u[4 * segVec + 2 * v + 1] = k.planes.length;
    const order = k.planes.map((_, i) => reversed ? k.planes.length - 1 - i : i);
    order.forEach((j, i) => {
      const p = k.planes[j]!;
      put(recVec + next + i, [...p.m, p.b]);
      const slot = next + i;
      if (extra?.far[v]![j]) u[4 * farVec + (slot >> 5)] |= 1 << (slot & 31);
    });
    next += k.planes.length;
  });
  return u;
}

test("GPU exact projection (ogcProjectTrial): the Euclidean projection onto the cap, plane records and solid pieces, against the Float64 brute force with its KKT check; bit-identical for any record order", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const R = rng(77), N = 2048;
    const unitRand = (): V3 => { for (;;) { const v: V3 = [2 * R() - 1, 2 * R() - 1, 2 * R() - 1]; const l = norm(v); if (l > 0.1 && l <= 1) return scale(v, 1 / l); } };
    // One frame for every moving piece (the end pose is the solid's, not the vertex's).
    const a = unitRand(), th = 2.1, cs = Math.cos(th), sn = Math.sin(th), K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];
    const Rb = [0, 1, 2].map(i => [0, 1, 2].map(j => Math.fround((i === j ? cs : 0) + sn * K[i]![j]! + (1 - cs) * a[i]! * a[j]!)));
    // The inputs as the GPU holds them (f32), the anchors up to 1.5 m from the origin.
    const fr = (v: V3) => v.map(Math.fround) as V3;
    const cases = Array.from({ length: N }, () => {
      const k = projectionCase(R, Rb);
      const piece = (p: typeof k.solid) => p && { o: fr(p.o), q: fr(p.q), Rq: Math.fround(p.Rq), D: fr(p.D), Rb: p.Rb };
      return { R: Math.fround(k.R), planes: k.planes.map(p => ({ m: fr(p.m), b: Math.fround(p.b) })), solid: piece(k.solid), body: piece(k.body), y: k.y };
    });
    const c = cases.map(() => fr(scale(unitRand(), 1.5 * R())));
    const Y = cases.map((k, v) => fr(add(c[v]!, k.y)));
    let worst = 0, worstRel = 0, cut = 0, kept = 0, maxViol = 0, maxRes = 0;
    const bad: string[] = [], rels: number[] = [];
    for (const levelled of [false, true]) {
      const buf = storage(device, projectionBuffer(cases, c, Rb, levelled));
      const got = await evalProject(device, buf, c, Y);
      const rev = storage(device, projectionBuffer(cases, c, Rb, levelled, true));
      const back = await evalProject(device, rev, c, Y);
      buf.destroy(); rev.destroy();
      cases.forEach((k, v) => {
        assert.ok(got[v]!.every((x, i) => Object.is(x, back[v]![i])), `vertex ${v}: the projection depends on the records' order (${got[v]} vs ${back[v]})`);
        // The oracle on the GPU's own inputs: the anchor-relative trial as f32 holds it.
        // ogcProjectExact's target: every certificate pulled in by 2 tau = 2^-16 (R + |y - c|).
        const cons = projectionCaseCons(k), zy = sub(Y[v]!, c[v]!), s = k.R + norm(zy);
        const ref = bruteForceProjection(shrinkCons(cons, 2 ** -16 * s), zy, 1e-13 * s)!;
        const kkt = projectionKkt(shrinkCons(cons, 2 ** -16 * s), zy, ref.p, 1e-10 * s);
        maxRes = Math.max(maxRes, kkt.residual);
        assert.ok(kkt.residual < 1e-7 && kkt.violation <= 1e-12 * s, `vertex ${v}: the oracle's KKT ${JSON.stringify(kkt)}`);
        const zg = sub(got[v]!, c[v]!);
        const off = norm(sub(zg, ref.p));
        if (ref.W.length) cut++; else if (Object.is(got[v]![0], Y[v]![0]) && Object.is(got[v]![1], Y[v]![1]) && Object.is(got[v]![2], Y[v]![2])) kept++;
        worst = Math.max(worst, off); worstRel = Math.max(worstRel, off / s); rels.push(off / s);
        // Certified: inside every certificate (to the f32 rounding of world positions).
        const viol = Math.max(...cons.map(q => conViolation(q, zg)));
        maxViol = Math.max(maxViol, viol);
        if (viol > 4e-7) bad.push(`vertex ${v}: the projection leaves a certificate by ${viol.toExponential(2)}: ${cons.map(q => conViolation(q, zg)).map((x, i) => [i, x] as const).filter(([, x]) => x > 4e-7).map(([i, x]) => `${i}:${x.toExponential(1)}`).join(' ')} (planes ${k.planes.length} solid ${!!k.solid} body ${!!k.body})`);
        // f32 rounding of the anchor-relative trial, amplified by the corners' conditioning (measured 1.1e-7 m).
        if (off > 5e-7) {
          bad.push(`vertex ${v} (levelled ${levelled}): ${off.toExponential(2)} m off the exact projection (scale ${s.toExponential(2)}, active ${JSON.stringify(ref.W)} of ${cons.length}; |y - p| GPU ${norm(sub(zg, zy)).toExponential(4)} vs ${norm(sub(ref.p, zy)).toExponential(4)}; GPU violations ${cons.map(q => conViolation(q, zg)).map((x, i) => [i, x] as const).filter(([, x]) => x > -1e-7).map(([i, x]) => `${i}:${x.toExponential(1)}`).join(' ')}; planes ${k.planes.length} solid ${!!k.solid} body ${!!k.body})`);
        }
      });
    }
    rels.sort((a, b) => a - b);
    console.log(`off the projection over the scale R + |y - c|: median ${rels[rels.length >> 1]!.toExponential(2)}, 99 % ${rels[Math.floor(0.99 * rels.length)]!.toExponential(2)}, 99.9 % ${rels[Math.floor(0.999 * rels.length)]!.toExponential(2)}, max ${rels[rels.length - 1]!.toExponential(2)}`);
    if (bad.length) console.log(`${bad.length} off the projection:\n${bad.slice(0, 12).join('\n')}`);
    assert.equal(bad.length, 0, `${bad.length} trials off the exact projection by more than 0.5 um`);
    console.log(`exact projection on the GPU: ${2 * N} trials, ${cut} cut, ${kept} kept bit-exactly; off the Float64 projection by ${worst.toExponential(2)} m at most (${worstRel.toExponential(2)} of the scale), certificates left by ${maxViol.toExponential(2)} m at most; oracle KKT residual ${maxRes.toExponential(2)}`);
    assert.equal(kept + cut, 2 * N, "every trial the certificates keep comes back bit-exactly");
  });
});

test("GPU projection (ogcProjectTrial) in the H_vv metric: the exact metric projection onto the plane records and the solid pieces' faces, then the exit keeping the cap and the balls, against the Float64 brute force with its KKT check, identity and random metrics to condition 1e4; bit-identical for any record order", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const R = rng(77), N = 2048;
    const unitRand = (): V3 => { for (;;) { const v: V3 = [2 * R() - 1, 2 * R() - 1, 2 * R() - 1]; const l = norm(v); if (l > 0.1 && l <= 1) return scale(v, 1 / l); } };
    // One frame for every moving piece (the end pose is the solid's, not the vertex's).
    const a = unitRand(), th = 2.1, cs = Math.cos(th), sn = Math.sin(th), K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];
    const Rb = [0, 1, 2].map(i => [0, 1, 2].map(j => Math.fround((i === j ? cs : 0) + sn * K[i]![j]! + (1 - cs) * a[i]! * a[j]!)));
    // The inputs as the GPU holds them (f32), the anchors up to 1.5 m from the origin.
    const fr = (v: V3) => v.map(Math.fround) as V3;
    const cases = Array.from({ length: N }, () => {
      const k = projectionCase(R, Rb);
      const piece = (p: typeof k.solid) => p && { o: fr(p.o), q: fr(p.q), Rq: Math.fround(p.Rq), D: fr(p.D), Rb: p.Rb };
      return { R: Math.fround(k.R), planes: k.planes.map(p => ({ m: fr(p.m), b: Math.fround(p.b) })), solid: piece(k.solid), body: piece(k.body), y: k.y };
    });
    const c = cases.map(() => fr(scale(unitRand(), 1.5 * R())));
    const Y = cases.map((k, v) => fr(add(c[v]!, k.y)));
    // Every third vertex the identity (the Euclidean projection onto the half-spaces), the rest H_vv^-1 of condition
    // 10^(4 u) (a contact's kc n n^T over the mass and membrane reaches 1e4), as f32 holds them.
    const kappa = cases.map((_, v) => v % 3 === 0 ? 1 : 10 ** (4 * R()));
    const Hinv = kappa.map(kp => kp === 1 ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : randomMetric(R, kp).map(r => r.map(Math.fround)));
    let worst = 0, worstRel = 0, cut = 0, kept = 0, maxViol = 0, maxRes = 0, capCut = 0, excessOver = -Infinity;
    const bad: string[] = [], rels: number[] = [], bands = [[], [], [], [], []] as number[][], wBands = [[], [], [], [], []] as number[][], fBands = [[], [], [], [], []] as number[][];
    for (const levelled of [false, true]) {
      const buf = storage(device, projectionBuffer(cases, c, Rb, levelled));
      const got = await evalProject(device, buf, c, Y, Hinv);
      const rev = storage(device, projectionBuffer(cases, c, Rb, levelled, true));
      const back = await evalProject(device, rev, c, Y, Hinv);
      buf.destroy(); rev.destroy();
      cases.forEach((k, v) => {
        assert.ok(got[v]!.every((x, i) => Object.is(x, back[v]![i])), `vertex ${v}: the projection depends on the records' order (${got[v]} vs ${back[v]})`);
        // The oracle on the GPU's own inputs: the anchor-relative trial as f32 holds it.
        // ogcProjectExact's target: every half-space pulled in by 2 tau = 2^-16 (R + |y - c|), from the anchor (x = c here).
        const cons = projectionCaseCons(k), zy = sub(Y[v]!, c[v]!), s = k.R + norm(zy);
        const ref = metricProjection(cons, zy, Hinv[v]!, 2 ** -16 * s, 1e-13 * s * Math.sqrt(kappa[v]!));
        maxRes = Math.max(maxRes, ref.kkt.residual);
        assert.ok(ref.kkt.residual < 1e-7, `vertex ${v}: the oracle's KKT ${JSON.stringify(ref.kkt)}`);
        const zg = sub(got[v]!, c[v]!);
        const off = norm(sub(zg, ref.p));
        // In the metric's own norm (w = K^-1 z, M = K K^T = 3 H^-1 / tr H^-1, |z| <= sqrt 3 |w|) and its objective.
        const offW = norm(ref.toW(sub(zg, ref.p))), fG = norm(ref.toW(sub(zg, zy))), fR = norm(ref.toW(sub(ref.p, zy)));
        wBands[Math.min(4, Math.floor(Math.log10(kappa[v]!)))]!.push(offW / s);
        fBands[Math.min(4, Math.floor(Math.log10(kappa[v]!)))]!.push((fG - fR) / s);
        if (ref.t < 1) capCut++;
        if (ref.W.length || ref.t < 1) cut++; else if (Object.is(got[v]![0], Y[v]![0]) && Object.is(got[v]![1], Y[v]![1]) && Object.is(got[v]![2], Y[v]![2])) kept++;
        worst = Math.max(worst, off); worstRel = Math.max(worstRel, off / s); rels.push(off / s);
        bands[Math.min(4, Math.floor(Math.log10(kappa[v]!)))]!.push(off / s);
        // Certified: inside every certificate (to the f32 rounding of world positions).
        const viol = Math.max(...cons.map(q => conViolation(q, zg)));
        maxViol = Math.max(maxViol, viol);
        if (viol > 4e-7) bad.push(`vertex ${v}: the projection leaves a certificate by ${viol.toExponential(2)}: ${cons.map(q => conViolation(q, zg)).map((x, i) => [i, x] as const).filter(([, x]) => x > 4e-7).map(([i, x]) => `${i}:${x.toExponential(1)}`).join(' ')} (planes ${k.planes.length} solid ${!!k.solid} body ${!!k.body})`);
        // The GPU solves to feasibility tau = 2^-17 s, so in a stiff metric it is the exact projection of a set moved by
        // up to tau, which a wedge of near-parallel faces amplifies (its vertex moves by tau / sin(angle / 2)): the bound
        // is then its objective, at most that of the exact projection onto the set pulled in by one more tau (3 tau),
        // plus the metric norm of the output's f32 rounding, where it is not the exact point to 0.5 um (under the identity
        // it always is).
        const ref3 = metricProjection(cons, zy, Hinv[v]!, 3 * 2 ** -17 * s, 1e-13 * s * Math.sqrt(kappa[v]!));
        const f3 = norm(ref3.toW(sub(ref3.p, zy)));
        const ulp = 2 ** -23 * Math.max(...Y[v]!.map(Math.abs), ...c[v]!.map(Math.abs));
        const allow = ulp * Math.max(...[0, 1, 2].map(i => norm(ref.toW([0, 1, 2].map(j => i === j ? 1 : 0) as V3))));
        if (off > 5e-7) excessOver = Math.max(excessOver, (fG - f3) / Math.max(allow, 1e-30));
        if (off > 5e-7 && (kappa[v] === 1 || fG > f3 + allow)) {
          bad.push(`vertex ${v} (levelled ${levelled}, condition ${kappa[v]!.toExponential(1)}): objective ${fG.toExponential(4)} vs ${f3.toExponential(4)} at 3 tau; ${off.toExponential(2)} m off the exact projection (scale ${s.toExponential(2)}, active ${JSON.stringify(ref.W)} of ${cons.length}; |y - p| GPU ${norm(sub(zg, zy)).toExponential(4)} vs ${norm(sub(ref.p, zy)).toExponential(4)}; GPU violations ${cons.map(q => conViolation(q, zg)).map((x, i) => [i, x] as const).filter(([, x]) => x > -1e-7).map(([i, x]) => `${i}:${x.toExponential(1)}`).join(' ')}; planes ${k.planes.length} solid ${!!k.solid} body ${!!k.body})`);
        }
      });
    }
    rels.sort((a, b) => a - b);
    bands.forEach((b, i) => { if (!b.length) return; b.sort((x, y) => x - y); console.log(`metric condition 1e${i}-1e${i + 1}: ${b.length} trials, off over the scale: median ${b[b.length >> 1]!.toExponential(2)}, 99 % ${b[Math.floor(0.99 * b.length)]!.toExponential(2)}, max ${b[b.length - 1]!.toExponential(2)}`); });
    const st = (b: number[]) => { const x = [...b].sort((p, q) => p - q); return `median ${x[x.length >> 1]!.toExponential(2)}, 99 % ${x[Math.floor(0.99 * x.length)]!.toExponential(2)}, max ${x[x.length - 1]!.toExponential(2)}`; };
    wBands.forEach((b, i) => { if (b.length) console.log(`metric condition 1e${i}-1e${i + 1}: off in the metric's norm over the scale ${st(b)}; objective excess over the scale ${st(fBands[i]!)}`); });
    console.log(`the exit cut ${capCut} trials (a ball binding the plane projection)`);
    console.log(`off the projection over the scale R + |y - c|: median ${rels[rels.length >> 1]!.toExponential(2)}, 99 % ${rels[Math.floor(0.99 * rels.length)]!.toExponential(2)}, 99.9 % ${rels[Math.floor(0.999 * rels.length)]!.toExponential(2)}, max ${rels[rels.length - 1]!.toExponential(2)}`);
    if (bad.length) console.log(`${bad.length} off the projection:\n${bad.slice(0, 12).join('\n')}`);
    console.log(`the GPU's metric objective over that of the exact projection onto the set pulled in by 3 tau, in units of the output's f32 rounding, where it is more than 0.5 um off the exact point: ${excessOver.toFixed(2)} at most (<= 1 passes)`);
    assert.equal(bad.length, 0, `${bad.length} trials past the exact projection's objective at 3 tau (or, under the identity, 0.5 um off it)`);
    console.log(`exact projection on the GPU: ${2 * N} trials, ${cut} cut, ${kept} kept bit-exactly; off the Float64 projection by ${worst.toExponential(2)} m at most (${worstRel.toExponential(2)} of the scale), certificates left by ${maxViol.toExponential(2)} m at most; oracle KKT residual ${maxRes.toExponential(2)}`);
    assert.equal(kept + cut, 2 * N, "every trial the certificates keep comes back bit-exactly");
  });
});

/** ogcProjectActive(v, X_v, Y_v, 1) for every vertex: q (w: count + 4 held + 8 capOn + 16 travel), n0..n2, nc. */
async function evalActive(device: GPUDevice, det: GPUBuffer, X: V3[], Y: V3[], Hinv?: number[][][]): Promise<Float32Array> {
  const N = X.length, xb = storage(device, vec4s(X)), yb = storage(device, vec4s(Y)), mb = Hinv && metricBuffer(device, N, Hinv);
  const out = device.createBuffer({ size: 80 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const code = /* wgsl */`
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<vec4<f32>>;
fn ogcX(i: u32) -> vec3<f32> { return X[i].xyz; }
fn ogcXt(i: u32) -> vec3<f32> { return X[i].xyz; }
fn ogcTri(t: u32) -> vec3<u32> { return vec3<u32>(t); }
fn ogcEdgeVerts(e: u32) -> vec2<u32> { return vec2<u32>(e); }
${Hinv ? `${METRIC_WGSL}\n${OgcContactDetector.vertexBlockWGSL(0, 4, false, 'testMetricInv')}` : OgcContactDetector.vertexBlockWGSL(0, 4)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= ${N}u) { return; }
  let a = ogcProjectActive(v, X[v].xyz, Y[v].xyz, 1.0);
  outBuf[5u * v] = vec4<f32>(a.q, f32(a.count + select(0u, 4u, a.held) + select(0u, 8u, a.capOn) + select(0u, 16u, a.travel)));
  outBuf[5u * v + 1u] = vec4<f32>(a.n0, 0.0); outBuf[5u * v + 2u] = vec4<f32>(a.n1, 0.0);
  outBuf[5u * v + 3u] = vec4<f32>(a.n2, 0.0); outBuf[5u * v + 4u] = vec4<f32>(a.nc, 0.0);
}`;
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: ([[0, xb], [1, yb], ...(mb ? [[2, mb] as const] : []), [3, out], [4, det]] as const).map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
  device.queue.submit([enc.finish()]);
  const got = new Float32Array(await readBuffer(device, out));
  xb.destroy(); yb.destroy(); mb?.destroy(); out.destroy();
  return got;
}

/** The active set's check, on the Euclidean projection or (metric) on the H_vv metric's. */
function activeSetCheck(metric: boolean) {
  return async () => {
    await withContactDevice(async device => {
      const R = rng(91), N = 2048;
      const unitRand = (): V3 => { for (;;) { const v: V3 = [2 * R() - 1, 2 * R() - 1, 2 * R() - 1]; const l = norm(v); if (l > 0.1 && l <= 1) return scale(v, 1 / l); } };
      const a = unitRand(), th = 1.3, cs = Math.cos(th), sn = Math.sin(th), K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];
      const Rb = [0, 1, 2].map(i => [0, 1, 2].map(j => Math.fround((i === j ? cs : 0) + sn * K[i]![j]! + (1 - cs) * a[i]! * a[j]!)));
      const fr = (v: V3) => v.map(Math.fround) as V3;
      const cases = Array.from({ length: N }, () => {
        const k = projectionCase(R, Rb);
        const piece = (p: typeof k.solid) => p && { o: fr(p.o), q: fr(p.q), Rq: Math.fround(p.Rq), D: fr(p.D), Rb: p.Rb };
        return { R: Math.fround(k.R), planes: k.planes.map(p => ({ m: fr(p.m), b: Math.fround(p.b) })), solid: piece(k.solid), body: piece(k.body), y: k.y };
      });
      const c = cases.map(() => fr(scale(unitRand(), 1.5 * R())));
      const Y = cases.map((k, v) => fr(add(c[v]!, k.y)));
      // Travel's inputs: a ball's field (radius 0.9 m, so the anchors straddle it) at r = 0.1 m, and a far bit on about half the records.
      const field = buildClothSolidField({ sdf: p => Math.hypot(p[0], p[1], p[2]) - 0.9, lo: [-1.7, -1.7, -1.7], hi: [1.7, 1.7, 1.7], cell: 0.05, band: 0.3 });
      const extra = { far: cases.map(k => k.planes.map(() => R() < 0.5)), field, r: 0.1 };
      // In the H_vv metric: identity on every third vertex, else condition up to 1e4 (the binding set is the metric point's).
      const Hinv = metric ? cases.map((_, v) => v % 3 === 0 ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : randomMetric(R, 10 ** (4 * R())).map(r => r.map(Math.fround))) : undefined;
      const counts = [0, 0, 0, 0];
      let capOn = 0, worstNormal = 0, worstSpan = 0, binding = 0;
      const travels = { cap: 0, ball: 0, faceFar: 0, faceNear: 0, recordFar: 0, recordNear: 0, set: 0 };
      const bad: string[] = [];
      for (const levelled of [false, true]) {
        const buf = storage(device, projectionBuffer(cases, c, Rb, levelled, false, extra));
        const qs = await evalProject(device, buf, c, Y, Hinv);
        const got = await evalActive(device, buf, c, Y, Hinv);
        const rev = storage(device, projectionBuffer(cases, c, Rb, levelled, true, extra));
        const back = await evalActive(device, rev, c, Y, Hinv);
        buf.destroy(); rev.destroy();
        const orderDep = got.findIndex((x, i) => !Object.is(x, back[i]));
        if (orderDep >= 0) bad.push(`vertex ${Math.floor(orderDep / 20)} (levelled ${levelled}): the active set depends on the records' order`);
        cases.forEach((k, v) => {
          const at = (i: number): V3 => [got[20 * v + 4 * i]!, got[20 * v + 4 * i + 1]!, got[20 * v + 4 * i + 2]!];
          const q = at(0), flags = got[20 * v + 3]!, count = flags % 4, held = (flags & 4) !== 0, cap = (flags & 8) !== 0;
          if (!q.every((x, i) => Object.is(x, qs[v]![i]))) bad.push(`vertex ${v}: q ${q} is not ogcProjectTrial's ${qs[v]}`);
          if (held) bad.push(`vertex ${v}: held with no inside marker`);
          counts[count]!++; if (cap) capOn++;
          const zq = sub(q, c[v]!), zy = sub(Y[v]!, c[v]!), am = (p: V3) => Math.max(...p.map(Math.abs));
          // The GPU's binding slack: 4 tau + 2^-20 (|q|_inf + |c|_inf), tau = 2^-17 (R + |y - c|).
          const tol = 2 ** -15 * (k.R + norm(zy)) + 2 ** -20 * (am(q) + am(c[v]!));
          const cons = projectionCaseCons(k);
          const outward = (j: number): V3 => { const q2 = cons[j]!; return q2.ball ? scale(sub(zq, q2.c), 1 / norm(sub(zq, q2.c))) : scale(q2.n, 1 / norm(q2.n)); };
          const viol = cons.map(q2 => conViolation(q2, zq));
          // Each normal's f32 rounding: a ball's (z - o)/|z - o| carries z = q - c's world rounding (|dz| <= sqrt(3)
          // 2^-24 (|q|_inf + |c|_inf) < 2^-23 (...)) twice over |z - o|; an orthant face's D s - q (and its turn) a few
          // ulps of |D| + |q| over |D s - q|; a plane's is the record's own.
          const round = 2 ** -22 * (am(q) + am(c[v]!));
          const normTol: number[] = [0, ...k.planes.map(() => 1e-6)];
          for (const piece of [k.solid, k.body]) {
            if (!piece) continue;
            const pc = solidPieceCons(piece.o, piece.q, piece.Rq, piece.D, piece.Rb), dq = norm(piece.D) + norm(piece.q);
            pc.forEach(q2 => normTol.push(q2.ball ? round / Math.max(norm(sub(zq, q2.c)), 1e-30) + 1e-6 : 16 * 2 ** -24 * dq / norm(q2.n) + 1e-6));
          }
          assert.equal(normTol.length, cons.length);
          const ns = [1, 2, 3].slice(0, count).map(at);
          // Reported: the outward normal of a non-cap certificate within the slack, to that normal's rounding.
          ns.forEach((n, i) => {
            let best = Infinity;
            for (let j = 1; j < cons.length; j++) if (viol[j]! >= -1.5 * tol) best = Math.min(best, norm(sub(n, outward(j))) / normTol[j]!);
            worstNormal = Math.max(worstNormal, best);
            if (!(best <= 1)) bad.push(`vertex ${v}: normal ${i} ${n} is no binding certificate's (nearest at ${best.toFixed(2)} of its rounding)`);
          });
          // Every certificate binding q well inside the slack lies in the reported normals' span.
          const basis: V3[] = [];
          for (const n of ns) { let e = n; for (const b of basis) e = sub(e, scale(b, dot(e, b))); const l = norm(e); if (l > 1e-9) basis.push(scale(e, 1 / l)); }
          for (let j = 1; j < cons.length; j++) {
            if (!(viol[j]! >= -0.5 * tol)) continue;
            binding++;
            let e = outward(j); for (const b of basis) e = sub(e, scale(b, dot(e, b)));
            worstSpan = Math.max(worstSpan, norm(e));
            if (!(norm(e) <= 1e-4 + normTol[j]!)) bad.push(`vertex ${v} (levelled ${levelled}): certificate ${j} binds q (slack ${(-viol[j]!).toExponential(2)}, tol ${tol.toExponential(2)}) outside the ${count} normals' span by ${norm(e).toExponential(2)}`);
          }
          const capV = viol[0]!;
          if (capV >= -0.5 * tol && !cap) bad.push(`vertex ${v}: the cap binds (slack ${(-capV).toExponential(2)}) but capOn is clear`);
          if (capV < -1.5 * tol && cap) bad.push(`vertex ${v}: capOn with the cap's slack ${(-capV).toExponential(2)}`);
          if (cap) { const nc = at(4); const off = norm(sub(nc, scale(zq, 1 / norm(zq)))); if (!(off <= 1e-5)) bad.push(`vertex ${v}: nc off the cap's normal by ${off.toExponential(2)}`); }
          // Travel: each certificate's kind in cons order (the cap, the records, then per piece its ball and its faces).
          const travel = (flags & 16) !== 0;
          const phiAt = (p: V3) => sampleClothSolidField(field, p).phi;
          const solidPhi = phiAt(q), bodyPhi = phiAt([0, 1, 2].map(i => dot(Rb[i] as V3, q)) as V3);
          const kinds: ('cap' | 'recordFar' | 'recordNear' | 'ball' | 'face')[] = ['cap', ...k.planes.map((_, j) => extra.far[v]![j] ? 'recordFar' as const : 'recordNear' as const)];
          const facePhi: number[] = new Array(kinds.length).fill(NaN);
          for (const [piece, phi] of [[k.solid, solidPhi], [k.body, bodyPhi]] as const) {
            if (!piece) continue;
            solidPieceCons(piece.o, piece.q, piece.Rq, piece.D, piece.Rb).forEach(q2 => { kinds.push(q2.ball ? 'ball' : 'face'); facePhi.push(phi); });
          }
          assert.equal(kinds.length, cons.length);
          // A face's field within f32 rounding of r is undecided.
          const isTravel = (j: number) => kinds[j] === 'cap' || kinds[j] === 'ball' || kinds[j] === 'recordFar' || (kinds[j] === 'face' && facePhi[j]! >= 0.1);
          const undecided = (j: number) => kinds[j] === 'face' && Math.abs(facePhi[j]! - 0.1) < 1e-5;
          let sure = false, maybe = false;
          for (let j = 0; j < cons.length; j++) {
            const bindsSurely = viol[j]! >= -0.5 * tol, bindsMaybe = viol[j]! >= -1.5 * tol;
            if (bindsSurely && kinds[j] === 'face') travels[facePhi[j]! >= 0.1 ? 'faceFar' : 'faceNear']++;
            else if (bindsSurely) travels[kinds[j] as 'cap' | 'ball' | 'recordFar' | 'recordNear']++;
            if (bindsMaybe && (isTravel(j) || undecided(j))) maybe = true;
            if (bindsSurely && isTravel(j) && !undecided(j)) sure = true;
          }
          if (travel) travels.set++;
          if (sure && !travel) bad.push(`vertex ${v} (levelled ${levelled}): a travel bound binds q, travel is clear (kinds ${kinds.filter((_, j) => viol[j]! >= -0.5 * tol).join(',')})`);
          if (!maybe && travel) bad.push(`vertex ${v} (levelled ${levelled}): travel is set with only contacts binding q (kinds ${kinds.filter((_, j) => viol[j]! >= -1.5 * tol).join(',')})`);
        });
      }
      if (bad.length) console.log(`${bad.length} active-set failures:\n${bad.slice(0, 12).join('\n')}`);
      console.log(`projection active set: ${2 * N} trials, counts 0/1/2/3 = ${counts.join('/')}, cap binding ${capOn}; ${binding} binding certificates, all in the span to ${worstSpan.toExponential(2)}; reported normals off their certificate's by ${worstNormal.toFixed(2)} of its rounding at most`);
      console.log(`projection active set travel: set on ${travels.set} trials; surely binding: cap ${travels.cap}, ball ${travels.ball}, faces with the field >= r ${travels.faceFar} / < r ${travels.faceNear}, records far ${travels.recordFar} / near ${travels.recordNear}`);
      assert.equal(bad.length, 0, `${bad.length} active-set failures`);
      assert.ok(counts[1]! + counts[2]! + counts[3]! > N / 4, "the cases bind certificates");
      for (const [kind, n] of Object.entries(travels)) assert.ok(n > 20, `the cases exercise travel's ${kind} (${n})`);
    });
  };
}
test("GPU projection active set (ogcProjectActive): q is ogcProjectTrial's bit for bit, every reported normal is a non-cap certificate binding q, every certificate binding q lies in their span, travel is set exactly when a binding certificate is no contact at q (the cap, a piece's ball, an orthant face where the field is at least r, a record with its far bit), and the result is bit-identical for any record order", { skip: !gpu }, activeSetCheck(false));
test("GPU projection active set in the H_vv metric (OgcVertexBlockOptions.metric): the same checks with ogcProjectActive on the metric projection, identity and random metrics to condition 1e4", { skip: !gpu }, activeSetCheck(true));

/** Three sheets 4 mm apart, out of contact (r = 1 mm), lightly jittered. */
function spacedStack(seed: number): OracleMesh {
  const R = rng(seed), h = 0.01;
  const layers = [0, 4e-3, 8e-3].map((z, i) => {
    const s = sheet(7, 7, h, [i * 0.0031, i * 0.0023, z]);
    return { ...s, x: s.x.map(p => [p[0] + (R() - 0.5) * 1e-3, p[1] + (R() - 0.5) * 1e-3, p[2] + (R() - 0.5) * 4e-4] as V3) };
  });
  const m = merge(...layers);
  return { ...m, x: f32(m.x) };
}

/**
 * Two steps: the first from X - w, so that its predicted end is X and the second
 * measures no residual (r_q = rqBase); the second's mandatory round is checked.
 */
async function detectPredicted(device: GPUDevice, mesh: OracleMesh, vel: V3[], config: OgcContactConfig) {
  const { triangles, edges, oracle } = topologyOf(mesh);
  const W0 = vel.map(v => [0, 1, 2].map(k => v[k] * config.dt + config.gravity[k] * config.dt ** 2) as V3);
  const positions = storage(device, vec4s(f32(mesh.x.map((x, i) => sub(x, W0[i]))))), velocities = storage(device, vec4s(vel));
  const det = await OgcContactDetector.create(device, { vertexCount: mesh.x.length, triangles, edges }, vec4s(mesh.x), config, { positions, velocities });
  await runPass(device, pass => { det.encodeStepBegin(pass); det.encodeDetection(pass); });
  device.queue.writeBuffer(positions, 0, vec4s(mesh.x));
  await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
  const pred = new Float32Array(await readBuffer(device, det.pred));
  const W = Array.from({ length: mesh.x.length }, (_, v) => [pred[4 * v], pred[4 * v + 1], pred[4 * v + 2]] as V3);
  return { det, positions, oracle, W };
}

test("GPU predicted-motion DAT: pairs, translating planes and the truncated residual match the Float64 oracle", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = spacedStack(17), N = mesh.x.length, R = rng(18);
    // Falling and drifting together at about 1 m/s, each vertex a little differently.
    const vel = mesh.x.map(() => [0.3 + (R() - 0.5) * 0.06, -0.2 + (R() - 0.5) * 0.06, -0.9 + (R() - 0.5) * 0.06] as V3);
    // The first step has no residual measured yet: its envelopes cover the whole ~16 mm, and pair everything.
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 5e-2, pairCapacity: 1024 * N, gridCapacity: 2048 * N };
    const { det, positions, oracle, W } = await detectPredicted(device, mesh, vel, config);
    const c0 = await det.readCounters();
    assert.equal(c0.status, 0, `status ${OgcContactDetector.describeStatus(c0.status)}`);
    assert.equal(c0.held, 0, "nothing is held"); assert.equal(c0.redos, 0);
    W.forEach((w, v) => assert.ok(norm(sub(w, [vel[v][0] * config.dt, vel[v][1] * config.dt, vel[v][2] * config.dt + config.gravity[2] * config.dt ** 2])) < 1e-8, `prediction ${v}`));
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const rq = mesh.x.map(() => config.rqBase);
    for (let v = 0; v < N; v++) assert.ok(Math.abs(xdet[4 * v + 3] - 0.45 * config.rqBase) < 1e-9, "the residual envelope is rqBase");
    const ref = bruteForcePredictedPairs(oracle, rq, W, 2e-6, det);
    // The frame's peak: the first step's all-pairs round is the detector-lifetime peak.
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.framePeakPairSlots, oracle);
    const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "vertex-triangle pairs");
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "edge-edge pairs");
    // Absolute envelopes would pair the layers 4 mm apart; the co-moving predictions pair far fewer.
    const absolute = bruteForceDatPairs(oracle, mesh.x.map((_, v) => Math.min(config.rqMax, config.r + 2.5 * norm(W[v]))), 0, det).pairs.length;
    assert.ok(got.length < 0.25 * absolute, `${got.length} predicted pairs vs ${absolute} absolute`);
    assert.ok(got.every(g => !predictedConflict(g.pair, oracle.x, W, 1e-6)));
    // Residuals drive the layers into each other by several gaps.
    const X = oracle.x, res = X.map(p => [(R() - 0.5) * 2e-3, (R() - 0.5) * 2e-3, (p[2] < 2e-3 ? 1 : p[2] > 6e-3 ? -1 : (R() < 0.5 ? 1 : -1)) * R() * 6e-3] as V3);
    const Xnew = f32(X.map((x, v) => add(add(x, W[v]), res[v])));
    assert.ok(motionPenetrates(oracle, X, Xnew), "the raw motion penetrates");
    device.queue.writeBuffer(positions, 0, vec4s(Xnew));
    await runPass(device, pass => { det.encodeTruncate(pass); det.encodeDecide(pass); });
    const out = toV3(new Float32Array(await readBuffer(device, positions)), N);
    const oracleOut = predictedTruncate(X, Xnew, W, got.map(g => g.pair), rq.map(q => 0.45 * q), 0.9, 1e-6);
    for (let v = 0; v < N; v++) assert.ok(norm(sub(out[v], oracleOut.x[v])) <= 2e-7 + 2e-4 * norm(res[v]), `vertex ${v}: ${out[v]} vs ${oracleOut.x[v]}`);
    assert.equal(motionPenetrates(oracle, X, out), false, "the truncated motion is penetration-free");
    const kept = out.reduce((s, p, v) => s + dot(sub(p, X[v]), W[v]) / dot(W[v], W[v]), 0) / N;
    assert.ok(kept > 0.9, `vertices keep their predictions (${kept})`);
    det.destroy();
  });
});

test("GPU predicted-motion DAT: a plane translating along its normal past the second primitive's support is placed where the oracle places it", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // The layers fall together along their normal by about 4 times their 4 mm spacing, so every
    // cross-layer pair's predicted projections lie wholly on one side of its X: the support of the
    // second primitive is then negative along n (a support clamped at 0 would close the predicted gap).
    const mesh = spacedStack(19), N = mesh.x.length, R = rng(20);
    const vel = mesh.x.map(() => [(R() - 0.5) * 0.02, (R() - 0.5) * 0.02, -0.9 + (R() - 0.5) * 0.02] as V3);
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 6e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 5e-2, pairCapacity: 1024 * N, gridCapacity: 2048 * N };
    const { det, positions, oracle, W } = await detectPredicted(device, mesh, vel, config);
    const c0 = await det.readCounters();
    assert.equal(c0.status, 0, `status ${OgcContactDetector.describeStatus(c0.status)}`);
    assert.equal(c0.redos, 0);
    const rq = mesh.x.map(() => config.rqBase);
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.framePeakPairSlots, oracle);
    const X = oracle.x;
    const across = got.filter(g => new Set(g.pair.ids.map(i => Math.round(X[i]![2] / 4e-3))).size > 1).length;
    assert.ok(across > 100, `${across} pairs across the layers`);
    // Residuals push the layers together by up to 1.5 of their gaps.
    const res = X.map(p => [(R() - 0.5) * 1e-3, (R() - 0.5) * 1e-3, (p[2] < 2e-3 ? 1 : p[2] > 6e-3 ? -1 : (R() < 0.5 ? 1 : -1)) * R() * 6e-3] as V3);
    const Xnew = f32(X.map((x, v) => add(add(x, W[v]), res[v])));
    device.queue.writeBuffer(positions, 0, vec4s(Xnew));
    await runPass(device, pass => { det.encodeTruncate(pass); det.encodeDecide(pass); });
    const out = toV3(new Float32Array(await readBuffer(device, positions)), N);
    const oracleOut = predictedTruncate(X, Xnew, W, got.map(g => g.pair), rq.map(q => 0.45 * q), 0.9, 1e-6);
    for (let v = 0; v < N; v++) assert.ok(norm(sub(out[v], oracleOut.x[v])) <= 2e-7 + 2e-4 * norm(res[v]), `vertex ${v}: ${out[v]} vs ${oracleOut.x[v]}`);
    assert.equal(motionPenetrates(oracle, X, out), false, "the truncated motion is penetration-free");
    det.destroy();
  });
});

test("GPU predicted-motion DAT: crossing predictions are shifted back along their paths only as far as the pairs need (least shift)", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = spacedStack(23), N = mesh.x.length;
    // The top sheet is predicted 7 mm down at the others, which stay.
    const vel = mesh.x.map(p => [0, 0, p[2] > 6e-3 ? (-7e-3 + 9.81 / 3600) * 60 : 9.81 / 60] as V3);
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 4e-2, gridCapacity: 2048 * N };
    const { det, oracle, W } = await detectPredicted(device, mesh, vel, config);
    const c = await det.readCounters();
    assert.equal(c.status & ~OGC_STATUS.envelopeCap, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
    assert.ok(c.redos >= 1 && c.conflicts > 0, `${c.redos} redos, ${c.conflicts} conflicts`);
    // Each prediction is its inertial one scaled by the round's shift s in [0, 1] (anchor X: no moving solid).
    const work = new Float32Array(await readBuffer(device, det.work)), scaleOff = 12 * N + det.T + det.E;
    const W0 = vel.map(v => [0, 1, 2].map(k => v[k]! * config.dt + config.gravity[k]! * config.dt ** 2) as V3);
    let shifted = 0;
    for (let v = 0; v < N; v++) {
      const sv = work[scaleOff + OGC_SHIFT_WORDS * v + 1]!;
      assert.ok(sv >= 0 && sv <= 1, `vertex ${v} shift ${sv}`);
      if (sv < 1 && norm(W0[v]) > 0) shifted++;
      assert.ok(norm(sub(W[v], scale(W0[v], sv))) <= 1e-8 + 1e-5 * norm(W0[v]), `vertex ${v}: ${W[v]} vs ${sv} x ${W0[v]}`);
    }
    const top = W0.filter(w => norm(w) > 0).length;
    assert.ok(shifted > 0 && shifted <= top, `${shifted} of the top sheet's ${top} predictions shifted`);
    // The round redone is the oracle's for the shifted predictions, conflict-free, and its motion penetration-free.
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const rq = Array.from({ length: N }, (_, v) => xdet[4 * v + 3]! / 0.45);
    const ref = bruteForcePredictedPairs(oracle, rq, W, 2e-6, det);
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c.framePeakPairSlots, oracle);
    const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "shifted round: vertex-triangle pairs");
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "shifted round: edge-edge pairs");
    assert.ok(got.every(g => !predictedConflict(g.pair, oracle.x, W, 1e-6)), "no pair of the final round conflicts");
    assert.equal(motionPenetrates(oracle, oracle.x, oracle.x.map((x, v) => add(x, W[v]))), false, "the predicted motion is penetration-free");
    // A shift holds nothing: no vertex carries a contact flag into the next step.
    const held = new Uint32Array(work.buffer), off = 7 * N + det.T + det.E;
    assert.ok(Array.from(held.subarray(off, off + N)).every(f => (f & 1) === 0), "nothing held");
    det.destroy();
  });
});

test("GPU carried state: the per-vertex reference, rho_v and held flags a step hands the next restore exactly, so a replayed step equals the original", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = spacedStack(23), N = mesh.x.length;
    // The crossing-prediction scene: its first steps leave residuals behind, and a residual driving
    // the layers into each other the truncation cuts, holding those vertices.
    const vel = mesh.x.map(p => [0, 0, p[2] > 6e-3 ? (-7e-3 + 9.81 / 3600) * 60 : 9.81 / 60] as V3);
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 4e-2, gridCapacity: 2048 * N };
    const { det, positions, oracle, W } = await detectPredicted(device, mesh, vel, config);
    device.queue.writeBuffer(positions, 0, vec4s(f32(oracle.x.map((x, v) => add(add(x, W[v]), [0, 0, (x[2] > 6e-3 ? -1 : x[2] < 2e-3 ? 1 : 0) * 6e-3] as V3)))));
    await runPass(device, pass => det.encodeTruncate(pass));
    const saved = await det.readCarriedState();
    assert.ok(Array.from(saved.held).some(f => f & 1), "the step leaves held flags");
    assert.ok(Array.from({ length: N }, (_, v) => saved.reference[4 * v + 3]!).every(w => w >= 0), "and measured residuals");
    // The next step from moved positions, then the same step replayed from the restored state.
    const X1 = vec4s(mesh.x.map((x, v) => [x[0], x[1], x[2] + 1e-4 * (v % 7)] as V3));
    const step = async () => {
      device.queue.writeBuffer(positions, 0, X1);
      await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
      return { pred: new Float32Array(await readBuffer(device, det.pred)), carried: await det.readCarriedState(), counters: await det.readCounters() };
    };
    const first = await step();
    assert.notDeepEqual(first.carried.reference, saved.reference, "the step changes the carried state");
    det.writeCarriedState(saved);
    assert.deepEqual(await det.readCarriedState(), saved, "written back exactly");
    const replay = await step();
    assert.deepEqual(replay.pred, first.pred, "predictions bit-identical");
    assert.deepEqual(replay.carried, first.carried, "carried state bit-identical");
    assert.equal(replay.counters.held, first.counters.held); assert.equal(replay.counters.redos, first.counters.redos);
    // With no state: the state before any step.
    det.writeCarriedState();
    const fresh = await det.readCarriedState();
    assert.ok(Array.from({ length: N }, (_, v) => fresh.reference[4 * v + 3]!).every(w => w === -1) && fresh.held.every(f => f === 0), "no residual, nothing held");
    assert.throws(() => det.writeCarriedState({ reference: new Float32Array(4), held: new Uint32Array(1) }), /carried state for 1 vertices/);
    det.destroy();
  });
});

test("GPU isotropic truncation keeps |x - x_det| <= b_v and arms re-detection on the GPU, up to the round cap", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(9), N = mesh.x.length, R = rng(5);
    const config: OgcContactConfig = { ...baseConfig(N), roundCap: 1, truncation: "isotropic" };
    const { det, positions } = await detect(device, mesh, mesh.x.map(() => [0, 0, 0]), config);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const moved = new Set<number>();
    const X = mesh.x.map((p, v) => { if (R() < 0.2) { moved.add(v); return [p[0] + 0.01, p[1], p[2]] as V3; } return p; });
    const run = (fn: (pass: GPUComputePassEncoder) => void) => runPass(device, fn);
    device.queue.writeBuffer(positions, 0, vec4s(X));
    await run(pass => { det.encodeTruncate(pass); det.encodeDecide(pass); });
    const after = new Float32Array(await readBuffer(device, positions));
    for (let v = 0; v < N; v++) {
      const d = Math.hypot(after[4 * v] - xdet[4 * v], after[4 * v + 1] - xdet[4 * v + 1], after[4 * v + 2] - xdet[4 * v + 2]);
      assert.ok(d <= xdet[4 * v + 3] * (1 + 1e-5), `vertex ${v} within its bound`);
    }
    let args = new Uint32Array(await readBuffer(device, det.args));
    let c = await det.readCounters();
    assert.equal(c.truncated, moved.size); assert.equal(c.redetections, 1);
    assert.ok(args[0] > 0 && args[21] > 0, "the armed round has workgroups");
    assert.equal(new Uint32Array(await readBuffer(device, det.planesArgs))[0], 0, "isotropic mode dispatches no planes");
    // The armed round re-detects at the truncated positions; bounds refresh around them.
    await run(pass => det.encodeDetection(pass));
    device.queue.writeBuffer(positions, 0, vec4s(X));
    await run(pass => { det.encodeTruncate(pass); det.encodeDecide(pass); });
    c = await det.readCounters();
    assert.equal(c.redetections, 1, "the round cap holds");
    assert.ok(c.status & OGC_STATUS.roundCap, "the round cap raises its status bit");
    args = new Uint32Array(await readBuffer(device, det.args));
    assert.equal(args[0] + args[21], 0, "a capped round dispatches nothing");
    // Without truncations nothing is armed.
    await run(pass => det.encodeDecide(pass));
    args = new Uint32Array(await readBuffer(device, det.args));
    assert.equal(args[0], 0);
    det.destroy();
  });
});

test("GPU intersection counter matches the brute force on crossing and separated sheets", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // Offsets keep every crossing generic, away from the other sheet's edges, so f32 and f64 agree.
    const a = sheet(9, 9, 0.01, [0.0031, 0.0017, 0]), b = sheet(9, 9, 0.01, [0.013, 0.007, 0]);
    const crossing = merge(a, { ...b, x: b.x.map(p => [p[0], p[1], (p[0] - 0.05) * 0.3 + 0.0001 * Math.sin(p[1] * 300)] as V3) });
    crossing.x = f32(crossing.x);
    const stacked = stackScene(4);
    for (const [mesh, name] of [[crossing, "crossing"], [stacked, "stack"]] as const) {
      const config = baseConfig(mesh.x.length);
      const { det, oracle } = await detect(device, mesh, mesh.x.map(() => [0, 0, 0]), config);
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      det.encodeIntersectionCount(pass); pass.end(); device.queue.submit([enc.finish()]);
      const c = await det.readCounters();
      assert.equal(c.intersections, countIntersections(oracle), name);
      if (name === "crossing") assert.ok(c.intersections > 10);
      det.destroy();
    }
  });
});

test("GPU envelope cap and contact-list overflow fail explicitly through the status word", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = stackScene(13), N = mesh.x.length;
    const fast = mesh.x.map((_, v) => (v % 7 ? [0, 0, 0] : [0, 0, 5]) as V3);
    let { det } = await detect(device, mesh, fast, { ...baseConfig(N), rqMax: 8e-3 });
    let c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.envelopeCap);
    assert.equal(c.envelopeCapHits, fast.filter(v => v[2]).length);
    det.destroy();
    ({ det } = await detect(device, mesh, fast.map(() => [0, 0, 0]), { ...baseConfig(N), contactCapacity: 2 }));
    c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.contactOverflow);
    assert.ok(c.peakContacts > 2, "the measured peak is reported so a scene can author its capacity");
    det.destroy();
    ({ det } = await detect(device, mesh, fast.map(() => [0, 0, 0]), { ...baseConfig(N), pairCapacity: 16 }));
    c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.pairOverflow);
    assert.ok(c.peakPairSlots > 16, "the measured pair peak is reported");
    det.destroy();
  });
});

test("GPU coincident primitives fail explicitly (status coincident), and the vertex block reads a zero-distance entry as infinite energy, never NaN", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // Two separate sheets 1 mm apart, each vertex over one of the other's (its contact is with that
    // vertex feature); then one vertex of the upper sheet placed exactly on the lower one's.
    const lower = sheet(4, 4, 0.01, [0, 0, 0]), upper = sheet(4, 4, 0.01, [0, 0, 1e-3]);
    const mesh = merge(lower, upper), N = mesh.x.length, v = lower.x.length + 5, on = 5;
    const apart: OracleMesh = { ...mesh, x: f32(mesh.x) };
    let { det, positions } = await detect(device, apart, apart.x.map(() => [0, 0, 0]), baseConfig(N));
    assert.equal((await det.readCounters()).status & OGC_STATUS.coincident, 0, "separate sheets are not coincident");
    // The vertex block at a trial position where an entry's two primitives coincide: no NaN.
    const res = await evalVertexBlock(device, det, positions, apart.x, N);
    assert.ok(res[16 * v + 4 * 1 + 3]! > 0, "the upper vertex has contact entries");
    const trial = apart.x.map((p, i) => (i === v ? apart.x[on]! : p) as V3);
    device.queue.writeBuffer(positions, 0, vec4s(trial));
    const at = await evalVertexBlock(device, det, positions, apart.x, N);
    const block = Array.from(at.subarray(16 * v, 16 * v + 16));
    assert.ok(block.every(x => !Number.isNaN(x)), `vertex block at a coincident trial: ${block.join(", ")}`);
    assert.equal(at[16 * v + 3], Infinity, "a coincident entry lies outside the barrier's domain");
    det.destroy();
    // Detection at coincident primitives fails explicitly.
    const touching: OracleMesh = { ...mesh, x: f32(trial) };
    ({ det } = await detect(device, touching, touching.x.map(() => [0, 0, 0]), baseConfig(N)));
    const c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.coincident, `status ${OgcContactDetector.describeStatus(c.status)}`);
    det.destroy();
  });
});

// ------------------------------------------------------------------ static solid field

const solidBox = boxSdf([0.02, 0.02, 0], [0.08, 0.06, 0.03]), solidBall = sphereSdf([0.11, 0.05, 0], 0.025);
const solidField = (band = 0.06) => buildClothSolidField({
  parts: [
    { kind: "sdf", sdf: p => solidBox(p), lo: [0.02, 0.02, 0], hi: [0.08, 0.06, 0.03], mu: 0.3 },
    { kind: "sdf", sdf: p => solidBall(p), lo: [0.085, 0.025, -0.025], hi: [0.135, 0.075, 0.025], mu: 0.5 },
  ], lo: [-0.02, -0.02, -0.03], hi: [0.18, 0.12, 0.1], cell: 0.005, band });

/** A sheet over the box's top and the sphere, 0.5-4.5 mm above the box. */
function solidSheet(seed: number): OracleMesh {
  const R = rng(seed), s = sheet(15, 10, 0.01, [0.003, 0.004, 0]);
  return { ...s, x: f32(s.x.map(p => [p[0], p[1], 0.0305 + 0.004 * R()] as V3)) };
}

async function detectWithSolid(device: GPUDevice, mesh: OracleMesh, vel: V3[], config: OgcContactConfig, field: ClothSolidField) {
  const { triangles, edges, oracle } = topologyOf(mesh);
  const positions = storage(device, vec4s(mesh.x)), velocities = storage(device, vec4s(vel));
  const det = await OgcContactDetector.create(device, { vertexCount: mesh.x.length, triangles, edges }, vec4s(mesh.x), config, { positions, velocities });
  det.setSolidField(field);
  await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
  return { det, positions, oracle };
}

test("GPU hull facets: vertices across an off-lattice box edge keep the nearest of their field and facet entries, and the vertex block matches the oracle", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // A box turned 30 degrees about y, so its edges along y lie off the lattice and the field (cell 5 mm) rounds them
    // in by 1.65 mm: the samples stand on the true edge and their hull facets span it. A flat sheet across the
    // top-front edge's bisector, 1.0-1.4 mm off it (r = 2 mm), its rows 1.25 mm apart across the edge.
    const th = Math.PI / 6, C = Math.cos(th), Sn = Math.sin(th), ctr: V3 = [0.05, 0.04, 0.02], hb: V3 = [0.03, 0.025, 0.015];
    const local = boxSdf([-hb[0], -hb[1], -hb[2]], hb);
    const sdf = (p: readonly number[]) => { const d = [p[0]! - ctr[0], p[1]! - ctr[1], p[2]! - ctr[2]]; return local([C * d[0]! - Sn * d[2]!, d[1]!, Sn * d[0]! + C * d[2]!]); };
    const field = buildClothSolidField({ parts: [{ kind: "sdf", sdf, lo: [0.0, 0.01, -0.02], hi: [0.09, 0.07, 0.06], mu: 0.3 }], lo: [-0.02, -0.02, -0.03], hi: [0.18, 0.12, 0.1], cell: 0.005, band: 0.06 });
    // The bisector of the edge's two face normals (C, 0, -S) and (S, 0, C), and the direction across the edge.
    const nb: V3 = [(C + Sn) / Math.SQRT2, 0, (C - Sn) / Math.SQRT2], tb: V3 = [(C - Sn) / Math.SQRT2, 0, (-Sn - C) / Math.SQRT2];
    const base = sheet(8, 4, 0.01), R = rng(31), N = base.x.length;
    const mesh: OracleMesh = { ...base, x: f32(base.x.map((_, k) => {
      const i = k % 9, j = Math.floor(k / 9), s = (j - 2) * 1.25e-3, off = 1.0e-3 + 4e-4 * R();
      const e: V3 = [ctr[0] + C * hb[0] + Sn * hb[2], ctr[1] - 0.01 + 0.0025 * i, ctr[2] - Sn * hb[0] + C * hb[2]];
      return [0, 1, 2].map(a => e[a]! + off * nb[a]! + s * tb[a]!) as unknown as V3;
    })) };
    // The samples stand about a cell apart along the rounded edge and the reach (4 mm) takes in several hull edges and
    // facets per vertex: the pair slots are authored for that.
    const config: OgcContactConfig = { ...baseConfig(N), rqBase: 0.004, pairCapacity: 4096 * N, gridCapacity: 256 * N, solidCapacityWords: field.words.length + 64, solidSampleCapacity: 8192 };
    const { det, positions, oracle } = await detectWithSolid(device, mesh, mesh.x.map(() => [0, 0, 0]), config, field);
    const c0 = await det.readCounters();
    assert.equal(c0.status, 0, `status ${OgcContactDetector.describeStatus(c0.status)}`);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const lists = listsOf(new Uint32Array(await readBuffer(device, det.contacts)), N);
    const S = det.solidSamples, samples: V3[] = [];
    for (let s = 0; s < S; s++) samples.push([xdet[4 * (N + s)], xdet[4 * (N + s) + 1], xdet[4 * (N + s) + 2]]);
    const facets = clothSolidSampleFacets(field, { r: config.r, edge: det.topology.layout.maxEdgeLength });
    assert.equal(det.solidSampleFacets, facets.length / 3);
    const corner = (k: number, c: number) => samples[facets[3 * k + c]];
    const isFacet = (e: Entry) => e.w === 2 && (e.y & 0x80000000) === 0;
    let withFacet = 0, withField = 0, nearer = 0;
    for (let v = 0; v < N; v++) {
      const x = mesh.x[v], d = Array.from({ length: facets.length / 3 }, (_, k) => pointTriangleDistance(x, corner(k, 0), corner(k, 1), corner(k, 2)));
      const dmin = Math.min(...d), phi = sampleClothSolidField(field, x).phi;
      const fe = lists[v].filter(isFacet), so = lists[v].filter(e => e.y === 0x70000000);
      // One representation of the static solid.
      assert.ok(fe.length + so.length <= 1, `vertex ${v}: ${fe.length} facet and ${so.length} field entries`);
      if (fe.length) {
        withFacet++;
        const k = fe[0].y & 0x0fffffff;
        assert.ok(k < det.facetCapacity && d[k] < config.r && d[k] <= dmin + 1e-7 && phi >= d[k] - 1e-6, `vertex ${v}: facet ${k} at ${d[k]} (nearest ${dmin}, phi ${phi})`);
        // mu lambda: the barrier's force at d times the solid's mu (0.3) at the samples.
        const lambda = new Float32Array(new Uint32Array([fe[0].z]).buffer)[0], want = -activation(d[k], config.r, config.kc)[1] * 0.3;
        assert.ok(Math.abs(lambda - want) <= 2e-3 * want, `vertex ${v}: mu lambda ${lambda} vs ${want}`);
      }
      if (so.length) withField++;
      if (dmin < config.r - 1e-6 && phi > dmin + 1e-6) { nearer++; assert.equal(fe.length, 1, `vertex ${v}: facet at ${dmin} nearer than the field's ${phi}, no facet entry`); }
    }
    console.log(`  hull facets: ${facets.length / 3} facets, ${S} samples; ${withFacet} vertices hold a facet entry, ${withField} a field entry (of ${N}); counters ${c0.hfPairs} facet contacts, ${c0.hfDropped} dropped; ${c0.peakPairSlots} pair slots`);
    assert.ok(nearer >= 10 && withFacet === nearer && c0.hfPairs >= withFacet && c0.hfDropped > 0, `${nearer} vertices nearer a facet, ${withFacet} with its entry`);
    // The vertex block at positions nudged within a tenth of the clearance, against the oracle with the facet's samples.
    const X = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 1e-4, p[1] + (R() - 0.5) * 1e-4, p[2] + (R() - 0.5) * 1e-4] as V3));
    const Xt = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 2e-4, p[1] + (R() - 0.5) * 2e-4, p[2]] as V3));
    const XS = [...X, ...samples], XtS = [...Xt, ...samples];
    device.queue.writeBuffer(positions, 0, vec4s(X));
    const res = await evalVertexBlock(device, det, positions, Xt, N);
    const P = { r: config.r, k: config.kc, mu: config.mu, epsV: config.epsV, dt: config.dt };
    for (let v = 0; v < N; v++) {
      let f: V3 = [0, 0, 0], e = 0;
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const ent of lists[v]) {
        const lambda = new Float32Array(new Uint32Array([ent.z]).buffer)[0];
        let q: { f: V3; H: number[][] }, en: number;
        if (ent.y === 0x70000000) { const sc = solidContact(field, X[v], Xt[v], lambda, P); q = sc; en = sc.e; }
        else {
          const k = ent.y & 0x0fffffff;
          const c: OracleContact = isFacet(ent)
            ? { kind: "vf", ids: [v, N + facets[3 * k], N + facets[3 * k + 1], N + facets[3 * k + 2]], feature: (ent.y >>> 28) & 7, s: 0, t: 0, lambda: lambda / config.mu }
            : asOracleContact(ent, oracle);
          // A sample-face entry carries mu lambda with the solid's mu too.
          if (ent.w === 1 && (ent.y & 0x80000000) === 0) c.lambda /= config.mu;
          q = vertexForceHessian(c, XS, XtS, P, c.ids.indexOf(v)); en = contactEnergy(c, XS, XtS, P);
        }
        f = [f[0] + q.f[0], f[1] + q.f[1], f[2] + q.f[2]]; e += en;
        for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i][k] += q.H[i][k];
      }
      const fs = Math.max(1, Math.hypot(...f)), hs = Math.max(1, ...H.flat().map(Math.abs));
      for (let k = 0; k < 3; k++) assert.ok(Math.abs(res[16 * v + k] - f[k]) <= 2e-3 * fs, `vertex ${v} f[${k}] ${res[16 * v + k]} vs ${f[k]}`);
      assert.ok(Math.abs(res[16 * v + 3] - e) <= 2e-3 * Math.max(1e-6, Math.abs(e)) + 1e-9, `vertex ${v} energy ${res[16 * v + 3]} vs ${e}`);
      for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++)
        assert.ok(Math.abs(res[16 * v + 4 * (col + 1) + row] - H[row][col]) <= 2e-3 * hs, `vertex ${v} H`);
    }
    // Every DAT pair with a facet puts the vertex first and the facet's samples as its face.
    const pairs = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.peakPairSlots, oracle).map(g => g.pair);
    assert.ok(pairs.some(p => p.ids[0] < N && p.ids[1] >= N && p.ids[2] >= N && p.ids[3] >= N), "vertex-facet DAT pairs");
    det.destroy();
  });
});

test("GPU static solid: entries, truncation terms, the vertex block and the truncated motion against the oracle", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const field = solidField(), mesh = solidSheet(21), N = mesh.x.length, R = rng(22);
    const config: OgcContactConfig = { ...baseConfig(N), rqBase: 0.012, gridCapacity: 256 * N, solidCapacityWords: field.words.length + 64, solidSampleCapacity: 8192 };
    const { det, positions, oracle } = await detectWithSolid(device, mesh, mesh.x.map(() => [0, 0, 0]), config, field);
    const c0 = await det.readCounters();
    assert.equal(c0.status, 0, `status ${OgcContactDetector.describeStatus(c0.status)}`);
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const work = new Float32Array(await readBuffer(device, det.work));
    const lists = listsOf(new Uint32Array(await readBuffer(device, det.contacts)), N);
    // Static surface samples at x_det[N + s], on the zero set: the box's edges, and the ball,
    // whose curvature 40/m exceeds 3 r / e^2 = 30/m for the sheet's 1.41 cm diagonals.
    const S = det.solidSamples, samples: V3[] = [];
    for (let s = 0; s < S; s++) samples.push([xdet[4 * (N + s)], xdet[4 * (N + s) + 1], xdet[4 * (N + s) + 2]]);
    const onBox = samples.filter(p => solidBox(p) < solidBall(p)).length;
    assert.ok(onBox > 100 && S - onBox > 100 && samples.every(p => sampleClothSolidField(field, p).phi > -1e-6 && sampleClothSolidField(field, p).phi < 0.005 / 8), `${S} samples, ${onBox} on the box`);
    // Sample-facet entries: brute force over the sheet with the samples as free points.
    const isSample = (e: Entry) => e.w === 1 && (e.y & 0x80000000) === 0;
    const withSamples: OracleMesh = { ...oracle, x: [...mesh.x, ...samples] };
    const bf = bruteForceDetection(withSamples, config.r, withSamples.x.map(() => config.rqBase), 0.9);
    const sampleKeys = new Set(lists.flatMap(l => l.filter(isSample).map(e => `${e.x}|${globalFeature(oracle, e.y & 0x0fffffff, (e.y >>> 28) & 7)}`)));
    const wantKeys = new Set([...bf.vf].filter(k => +k.split("|")[0] >= N));
    for (const k of wantKeys) assert.ok(sampleKeys.has(k) || bf.ambiguousVF.has(k), `sample-facet pair ${k} missing`);
    for (const k of sampleKeys) assert.ok(wantKeys.has(k) || bf.ambiguousVF.has(k), `sample-facet pair ${k} not in the oracle`);
    assert.ok(sampleKeys.size > 20 && c0.sfPairs === sampleKeys.size, `${sampleKeys.size} sample-facet pairs, counter ${c0.sfPairs}`);
    const solidOff = 3 * N + det.T + det.E, solidDevOff = 8 * N + det.T + det.E;
    // The ball holds the round's whole reach, |w| + r_q / 2 (w = 0 here); the cap is 0.45 r_q.
    const bounds = mesh.x.map((x, v) => solidBallBound(field, x, xdet[4 * v + 3] / 0.9, config.r));
    // A ball whose box ends on a cell face (x0 + rho = 0.01 here) reads one more cell row in float32 or not.
    const ambiguous = new Set(mesh.x.flatMap((x, v) => {
      const rb = xdet[4 * v + 3] / 0.9, lo = solidBallBound(field, x, rb * (1 - 1e-5), config.r), hi = solidBallBound(field, x, rb * (1 + 1e-5), config.r);
      return lo.D.some((d, a) => d !== hi.D[a]) || lo.far !== hi.far ? [v] : [];
    }));
    let near = 0, entries = 0;
    for (let v = 0; v < N; v++) {
      const b = bounds[v], rho = xdet[4 * v + 3];
      const solid = lists[v].filter(e => e.y === 0x70000000);
      assert.ok(solid.length <= 1);
      entries += solid.length;
      const margin = Math.min(b.phi0, field.band) - field.maxLipschitz * rho - config.r;
      if (Math.abs(margin) < 1e-6) continue;
      // A ball whose box ends on a cell face (x0 + rho = 0.01 here) reads one more cell row in float32 or not.
      if (ambiguous.has(v)) continue;
      assert.equal(solid.length, b.far ? 0 : 1, `vertex ${v} solid entry (far ${b.far}, margin ${margin})`);
      const q = [0, 1, 2, 3, 4, 5, 6].map(k => k < 4 ? work[solidOff + 4 * v + k] : work[solidDevOff + 4 * v + k - 4]);
      if (b.far) { assert.deepEqual(q, [0, 0, 0, 0, 0, 0, 0]); continue; }
      near++;
      // q.w is the affine bound's ball, the round's reach (within rqMax / 2 here).
      const rc = Math.min(xdet[4 * v + 3] / 0.9, 0.5 * config.rqMax);
      assert.ok(Math.abs(q[3] - rc) <= 1e-6 * rc, `vertex ${v} affine ball ${q[3]} vs ${rc}`);
      const s = solidTermScale(b.phi0, 0.9, config.r), qr = [b.g0[0] * s, b.g0[1] * s, b.g0[2] * s, q[3], b.D[0] * s, b.D[1] * s, b.D[2] * s];
      const qs = Math.max(...qr.map(Math.abs));
      for (let k = 0; k < 7; k++) assert.ok(Math.abs(q[k] - qr[k]) <= 2e-3 * qs, `vertex ${v} q/D[${k}] ${q[k]} vs ${qr[k]}`);
      const lambda = new Float32Array(new Uint32Array([solid[0].z]).buffer)[0];
      const lr = (b.phi0 < config.r ? -activation(b.phi0, config.r, config.kc)[1] * norm(b.g0) : 0) * sampleClothSolidField(field, mesh.x[v]).mu;
      assert.ok(Math.abs(lambda - lr) <= 2e-3 * Math.max(1, lr), `vertex ${v} lambda ${lambda} vs ${lr}`);
    }
    assert.ok(near > 30 && c0.solidContacts === entries, `${near} checked near-solid vertices, ${entries} entries, counter ${c0.solidContacts}`);

    // The vertex block at positions nudged within the solid-safe ball, with friction from a shifted step start.
    const X = f32(mesh.x.map((p, v) => { const h = 0.3 * Math.min(xdet[4 * v + 3], bounds[v].phi0); return [p[0] + (R() - 0.5) * h, p[1] + (R() - 0.5) * h, p[2] + (R() - 0.5) * h] as V3; }));
    const Xt = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 2e-4, p[1] + (R() - 0.5) * 2e-4, p[2]] as V3));
    const XS = [...X, ...samples], XtS = [...Xt, ...samples];
    device.queue.writeBuffer(positions, 0, vec4s(X));
    const res = await evalVertexBlock(device, det, positions, Xt, N);
    const P = { r: config.r, k: config.kc, mu: config.mu, epsV: config.epsV, dt: config.dt };
    for (let v = 0; v < N; v++) {
      let f: V3 = [0, 0, 0], e = 0;
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const ent of lists[v]) {
        const lambda = new Float32Array(new Uint32Array([ent.z]).buffer)[0];
        let q: { f: V3; H: number[][] }, en: number;
        if (ent.y === 0x70000000) { const sc = solidContact(field, X[v], Xt[v], lambda, P); q = sc; en = sc.e; }
        else {
          // A sample-facet entry carries mu lambda with the solid's mu; the static sample sits at X = Xt.
          const c = asOracleContact(ent, oracle);
          if (isSample(ent)) c.lambda /= config.mu;
          q = vertexForceHessian(c, XS, XtS, P, c.ids.indexOf(v)); en = contactEnergy(c, XS, XtS, P);
        }
        f = [f[0] + q.f[0], f[1] + q.f[1], f[2] + q.f[2]]; e += en;
        for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i][k] += q.H[i][k];
      }
      const fs = Math.max(1, Math.hypot(...f)), hs = Math.max(1, ...H.flat().map(Math.abs));
      for (let k = 0; k < 3; k++) assert.ok(Math.abs(res[16 * v + k] - f[k]) <= 2e-3 * fs, `vertex ${v} f[${k}] ${res[16 * v + k]} vs ${f[k]}`);
      assert.ok(Math.abs(res[16 * v + 3] - e) <= 2e-3 * Math.max(1e-6, Math.abs(e)) + 1e-9, `vertex ${v} energy ${res[16 * v + 3]} vs ${e}`);
      for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++)
        assert.ok(Math.abs(res[16 * v + 4 * (col + 1) + row] - H[row][col]) <= 2e-3 * hs, `vertex ${v} H`);
    }

    // The step's motion drives the sheet down into the box and the sphere and slides it.
    const X0 = f32(mesh.x);
    const Xnew = f32(X0.map(p => [p[0] + (R() - 0.5) * 8e-3, p[1] + (R() - 0.5) * 8e-3, p[2] - 6e-3 * R()] as V3));
    const pairs = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.peakPairSlots, oracle).map(g => g.pair);
    device.queue.writeBuffer(positions, 0, vec4s(Xnew));
    await runPass(device, pass => det.encodeTruncate(pass));
    const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
    assert.ok(pairs.some(p => p.ids[0] >= N), "sample-facet DAT pairs");
    const planar = planarTruncate([...X0, ...samples], [...Xnew, ...samples], pairs, [...mesh.x.map((_, v) => xdet[4 * v + 3]), ...samples.map(() => 0)], 0.9, 1e-6);
    let cut = 0;
    for (let v = 0; v < N; v++) {
      const d = solidTruncate(bounds[v], sub(planar.x[v], X0[v]), 0.9, config.r, field.maxLipschitz), ref: V3 = [X0[v][0] + d[0], X0[v][1] + d[1], X0[v][2] + d[2]];
      if (norm(sub(ref, planar.x[v])) > 1e-9) cut++;
      if (!ambiguous.has(v)) assert.ok(norm(sub(got[v], ref)) <= 2e-7 + 2e-4 * norm(sub(Xnew[v], X0[v])), `vertex ${v}: ${got[v]} vs ${ref}`);
      const phi = sampleClothSolidField(field, got[v]).phi;
      assert.ok(phi >= (bounds[v].far ? config.r : solidKeep(bounds[v].phi0, 0.9, config.r)) * (1 - 1e-3), `vertex ${v} phi ${phi} after truncation`);
    }
    assert.equal(motionPenetrates(oracle, X0, got), false);
    const c1 = await det.readCounters();
    assert.ok(cut >= 10 && Math.abs(c1.solidCut - cut) <= 2 + ambiguous.size, `solid cut ${c1.solidCut} vs ${cut} (${ambiguous.size} on a cell face)`);
    // No face of the truncated sheet reaches into a solid; one pushed through the box's top edge does.
    await runPass(device, pass => det.encodeIntersectionCount(pass));
    const c2 = await det.readCounters();
    assert.ok(c2.intersections === 0 && c2.solidIntersections === 0, `${c2.intersections} face pairs, ${c2.solidIntersections} faces in a solid`);
    const edgeV = mesh.x.reduce((b, p, v) => Math.hypot(p[0] - 0.08, p[1] - 0.04) < Math.hypot(mesh.x[b][0] - 0.08, mesh.x[b][1] - 0.04) ? v : b, 0);
    const pushed = got.map((p, v) => (v === edgeV ? [0.078, 0.04, 0.028] : p) as V3);
    device.queue.writeBuffer(positions, 0, vec4s(pushed));
    await runPass(device, pass => det.encodeIntersectionCount(pass));
    const c3 = await det.readCounters();
    const faces = mesh.tris.filter(t => t.includes(edgeV)).length;
    assert.ok(c3.solidIntersections >= 1 && c3.solidIntersections <= faces, `${c3.solidIntersections} faces in the box (vertex ${edgeV} in ${faces} faces)`);
    det.destroy();
  });
});

test("GPU static solid: a cut never closes a gap below sepEps, and a vertex rounding left within it may only move away", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const field = solidField(), mesh = solidSheet(29), N = mesh.x.length, sepEps = 2e-6;
    // Over the box's flat top (z = 0.03, on the lattice, so phi is the height exactly): one vertex
    // rounding has left within sepEps, one 3 sepEps up, one 1 mm up; all pushed 2 mm down.
    const within = 51, near = 53, far = 55, top = 0.03;
    mesh.x[within] = [mesh.x[within]![0], mesh.x[within]![1], Math.fround(top + 0.5 * sepEps)];
    mesh.x[near] = [mesh.x[near]![0], mesh.x[near]![1], Math.fround(top + 3 * sepEps)];
    mesh.x[far] = [mesh.x[far]![0], mesh.x[far]![1], Math.fround(top + 1e-3)];
    const phi0 = (v: number) => sampleClothSolidField(field, mesh.x[v]!).phi;
    for (const v of [within, near, far]) assert.ok(Math.abs(phi0(v) - (mesh.x[v]![2] - top)) < 1e-9, `vertex ${v} over the flat top`);
    const config: OgcContactConfig = { ...baseConfig(N), solidCapacityWords: field.words.length, solidSampleCapacity: 8192, separationEps: sepEps };
    const { det, positions } = await detectWithSolid(device, mesh, mesh.x.map(() => [0, 0, 0]), config, field);
    assert.equal((await det.readCounters()).status & OGC_STATUS.solidPenetration, 0);
    device.queue.writeBuffer(positions, 0, vec4s(mesh.x.map(p => [p[0], p[1], p[2] - 2e-3] as V3)));
    await runPass(device, pass => det.encodeTruncate(pass));
    const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
    const phi = (v: number) => sampleClothSolidField(field, got[v]!).phi;
    assert.ok(phi(within) >= phi0(within) - 1e-9, `within sepEps: phi ${phi(within)} from ${phi0(within)}, no approach`);
    assert.ok(phi(near) >= sepEps - 1e-9 && phi(near) < phi0(near), `3 sepEps up: phi ${phi(near)} closes to sepEps, not below`);
    assert.ok(phi(far) >= solidKeep(phi0(far), 0.9, config.r, sepEps) - 1e-9 && phi(far) < phi0(far), `1 mm up: phi ${phi(far)} approaches, to (1 - gamma_r) phi0 at most`);
    // Away from the solid the vertex within sepEps moves freely.
    device.queue.writeBuffer(positions, 0, vec4s(mesh.x.map(p => [p[0], p[1], p[2] + 1e-4] as V3)));
    await runPass(device, pass => det.encodeTruncate(pass));
    const up = toV3(new Float32Array(await readBuffer(device, positions)), N);
    assert.ok(Math.abs(up[within]![2] - (mesh.x[within]![2] + 1e-4)) < 1e-9, `within sepEps it moves away: z ${up[within]![2]}`);
    det.destroy(); positions.destroy();
  });
});

test("GPU static solid fails explicitly: a vertex inside it is held, one outside the field's box is reported, a thin band and an over-capacity image are refused", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const field = solidField(), mesh = solidSheet(23), N = mesh.x.length;
    const inside = 37, outside = 0;
    mesh.x[inside] = [0.05, 0.04, 0.025];
    mesh.x[outside] = [-0.05, 0.004, 0.0305];
    const config: OgcContactConfig = { ...baseConfig(N), solidCapacityWords: field.words.length, solidSampleCapacity: 8192 };
    const { det, positions } = await detectWithSolid(device, mesh, mesh.x.map(() => [0, 0, 0]), config, field);
    const c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.solidPenetration, "inside");
    assert.ok(c.status & OGC_STATUS.solidBounds, "outside the box");
    const moved = mesh.x.map(p => [p[0] + 1e-3, p[1], p[2]] as V3);
    device.queue.writeBuffer(positions, 0, vec4s(moved));
    await runPass(device, pass => det.encodeTruncate(pass));
    const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
    assert.deepEqual(got[inside], mesh.x[inside].map(Math.fround), "the penetrating vertex holds");
    assert.throws(() => det.setSolidField(solidField(0.03)), /band/);
    // The box's edges hold more surface samples than an under-authored detector takes.
    const few = await OgcContactDetector.create(device, { vertexCount: N, triangles: topologyOf(mesh).triangles, edges: topologyOf(mesh).edges }, vec4s(mesh.x), { ...config, solidSampleCapacity: 8 }, { positions, velocities: positions });
    assert.throws(() => few.setSolidField(field), /surface samples .* \(solidSampleCapacity\)/);
    few.destroy();
    assert.throws(() => det.setSolidField(buildClothSolidField({ sdf: p => Math.min(solidBox(p), solidBall(p), boxSdf([0.2, 0, 0], [0.35, 0.1, 0.05])(p)), lo: [-0.02, -0.02, -0.03], hi: [0.38, 0.12, 0.1], cell: 0.005, band: 0.06 })), /authored/);
    det.destroy();
  });
});

test("GPU free flight: the solver's inertial target as the prediction and the solid's clearance table leave a fast sheet far above a floor untouched", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // A floor at z = 0 and a sheet 0.8 m above it moving 7 cm a step: longer than the bricks'
    // own rule certifies, (band - r) / L_max = 6 cm, so only the clearance table keeps the
    // solid out of the round.
    const r = 2e-3, rqMax = 0.04, band = r + 1.5 * Math.sqrt(3) * rqMax + 1e-6;
    const field = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.3, -0.3, -0.1], hi: [0.5, 0.5, 0], mu: 0.3 }], lo: [-0.3, -0.3, -0.1], hi: [0.5, 0.5, 1.2], cell: 0.01, band });
    const s = sheet(15, 10, 0.01, [0, 0, 0.8]), mesh: OracleMesh = { ...s, x: f32(s.x) }, N = mesh.x.length, X = mesh.x;
    const D: V3 = [0.008, -0.005, -0.07];
    assert.ok(norm(D) > (band - r) / field.maxLipschitz, "the step is longer than the brick rule certifies");
    const { triangles, edges } = topologyOf(mesh);
    const at = (d: V3) => vec4s(f32(X.map(x => add(x, d))));
    const positions = storage(device, at(D)), xt = storage(device, at([0, 0, 0])), target = storage(device, at(D));
    const velocities = storage(device, vec4s(X.map(() => scale(D, 60) as V3)));
    const config: OgcContactConfig = { ...baseConfig(N), r, rqMax, prediction: "target", solidCapacityWords: field.words.length, solidSampleCapacity: 8192 };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(X), config, { positions, velocities, stepStart: xt, target });
    assert.throws(() => det.bind({ positions, velocities, stepStart: xt, target }), /already bound/);
    det.setSolidField(field);
    // The last step ended at X as predicted (residual 0, nothing held); this one starts there,
    // predicted to its target X + D, where the solver's guess also starts: no cut, nothing held.
    det.writeCarriedState({ reference: vec4s(X), held: new Uint32Array(N) });
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.stepHooks.afterFrame(pass); });
    const guess = toV3(new Float32Array(await readBuffer(device, positions)), N);
    for (let v = 0; v < N; v++) assert.ok(norm(sub(guess[v], add(X[v], D))) < 1e-6, `vertex ${v} keeps its guess: ${guess[v]}`);
    const dg = await det.readDiagnosis(), c = await det.readCounters();
    assert.equal(c.status, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
    assert.equal(c.solidContacts, 0, "no vertex has a floor entry");
    assert.equal(c.pairEntries, 0, "a flat sheet in free flight has no self entries");
    assert.ok(dg.held.every(h => h === 0), "no vertex is held");
    for (let v = 0; v < N; v++) assert.ok(Math.abs(dg.rho[v]!) < 1e-6 && Math.abs(dg.env[v]! - config.rqBase) < 1e-9, `vertex ${v}: residual ${dg.rho[v]}, envelope ${dg.env[v]}`);
    // The solve ends at the prediction: no cut, no re-detection, and the next step is free as well.
    await runPass(device, pass => det.stepHooks.afterIteration(pass));
    const c2 = await det.readCounters(), out = toV3(new Float32Array(await readBuffer(device, positions)), N);
    assert.equal(c2.redetections, 0); assert.equal(c2.solidCut, 0);
    for (let v = 0; v < N; v++) assert.deepEqual(out[v], guess[v]);
    det.destroy();
  });
});

test("GPU solid round over its chord: a long prediction along a floor is certified by the ball about the chord's midpoint, not truncated", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // A sheet 20 cm above a floor predicted 25 cm along it, diagonally: the ball |w| + env / 2 about the start
    // reaches the floor, the ball about the chord's midpoint (radius |w| / 2 + env / 2) is clear by
    // the clearance table.
    const r = 2e-3, rqMax = 0.04, band = r + 1.5 * Math.sqrt(3) * rqMax + 1e-6, z0 = 0.2;
    const field = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.3, -0.3, -0.1], hi: [0.5, 0.5, 0], mu: 0.3 }], lo: [-0.3, -0.3, -0.1], hi: [0.5, 0.5, 1.2], cell: 0.01, band });
    const s = sheet(15, 10, 0.01, [0, 0, z0]), mesh: OracleMesh = { ...s, x: f32(s.x) }, N = mesh.x.length, X = mesh.x;
    const D: V3 = [0.18, 0.18, 0], env = 3e-3, rho = norm(D) + env / 2, rr = norm(D) / 2 + env / 2;
    assert.ok(rho > z0, "the start ball reaches the floor");
    for (const x of X) {
      const mid = add(x, scale(D, 0.5));
      assert.ok(clothSolidClearance(field, x) - rho < r && Math.min(sampleClothSolidField(field, x).phi, field.band) - field.maxLipschitz * rho < r, "the start ball is certified by neither rule");
      assert.ok(clothSolidClearance(field, mid) - rr >= r + 1e-3, "the chord ball is clear by the table");
    }
    const { triangles, edges } = topologyOf(mesh);
    const at = (d: V3) => vec4s(f32(X.map(x => add(x, d))));
    const positions = storage(device, at(D)), xt = storage(device, at([0, 0, 0])), target = storage(device, at(D));
    const velocities = storage(device, vec4s(X.map(() => scale(D, 60) as V3)));
    // The scenes' grid, 32 entries per primitive: the cell is at least the longest prediction, so a
    // face's swept box spans a few cells however far it moves (no gridOverflow).
    const gridCapacity = 32 * (N + triangles.length / 4 + edges.length / 4);
    const config: OgcContactConfig = { ...baseConfig(N), r, rqMax, prediction: "target", gridCapacity, solidCapacityWords: field.words.length, solidSampleCapacity: 8192 };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(X), config, { positions, velocities, stepStart: xt, target });
    det.setSolidField(field);
    det.writeCarriedState({ reference: vec4s(X), held: new Uint32Array(N) });
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.stepHooks.afterFrame(pass); });
    const guess = toV3(new Float32Array(await readBuffer(device, positions)), N), c = await det.readCounters(), dg = await det.readDiagnosis();
    assert.equal(c.status, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
    assert.equal(c.solidContacts, 0, "no vertex has a floor entry");
    assert.ok(dg.held.every(h => h === 0), "no vertex is held");
    for (let v = 0; v < N; v++) assert.ok(norm(sub(guess[v], add(X[v], D))) < 1e-6, `vertex ${v} keeps its guess: ${guess[v]}`);
    det.destroy();
  });
});

test("GPU solid round over its chord: a midpoint past a scene field's box certifies nothing, so a fast vertex is truncated at the floor", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    // C1's tilted drop: a whip vertex 3 cm above the floor predicted 10.5 cm down. The chord's midpoint lies past the
    // field's box, where a scene field holds nothing (the container's shell or the rest of a solid lies there): read as
    // band it passed the Lipschitz rule, the round had no terms and the vertex tunnelled the floor.
    const r = 2e-3, rqMax = 0.04, band = r + 1.5 * Math.sqrt(3) * rqMax + 1e-6, z0 = 0.03;
    const field = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.3, -0.3, -0.3], hi: [0.5, 0.5, 0], mu: 0.3 }], lo: [-0.05, -0.05, -0.005], hi: [0.25, 0.15, 0.12], cell: 0.0025, band });
    const s = sheet(15, 10, 0.01, [0, 0, z0]), mesh: OracleMesh = { ...s, x: f32(s.x) }, N = mesh.x.length, X = mesh.x;
    const D: V3 = [0, 0, -0.105], env = 3e-3, rr = norm(D) / 2 + env / 2;
    assert.equal(field.exteriorClear, false, "the floor runs on past the box");
    for (const x of X) {
      const mid = add(x, scale(D, 0.5)), b = field.origin.map((o, a) => Math.floor((mid[a]! - o) / (8 * field.cell)));
      assert.ok(b.some((v, a) => v < 0 || v >= field.dims[a]!), "the chord's midpoint lies past the box");
      assert.ok(field.band - field.maxLipschitz * rr >= r, "read as band, the midpoint would clear the chord ball");
    }
    const { triangles, edges } = topologyOf(mesh);
    const at = (d: V3) => vec4s(f32(X.map(x => add(x, d))));
    const positions = storage(device, at(D)), xt = storage(device, at([0, 0, 0])), target = storage(device, at(D));
    const velocities = storage(device, vec4s(X.map(() => scale(D, 60) as V3)));
    const gridCapacity = 32 * (N + triangles.length / 4 + edges.length / 4);
    const config: OgcContactConfig = { ...baseConfig(N), r, rqMax, prediction: "target", gridCapacity, solidCapacityWords: field.words.length, solidSampleCapacity: 8192 };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(X), config, { positions, velocities, stepStart: xt, target });
    det.setSolidField(field);
    det.writeCarriedState({ reference: vec4s(X), held: new Uint32Array(N) });
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.stepHooks.afterFrame(pass); });
    const guess = toV3(new Float32Array(await readBuffer(device, positions)), N), c = await det.readCounters();
    assert.equal(c.status & ~OGC_STATUS.envelopeCap, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
    for (let v = 0; v < N; v++) assert.ok(guess[v]![2] > 0 && sampleClothSolidField(field, guess[v]!).phi > 0, `vertex ${v} stays above the floor: ${guess[v]}`);
    det.destroy();
  });
});

test("GPU tilted drop (C1's slap): a sheet turned 15 degrees falls onto the floor; its whip edge never enters it and nothing halts", { skip: !gpu }, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  try {
    const tilt = 15 * Math.PI / 180, source = clothAnalyticFreeFall(), cy = source.probe.y0 as number, c = Math.cos(tilt), sn = Math.sin(tilt);
    const turn = (p: readonly number[]) => p.map((u, i) => { const k = i % 3, x = p[i - k]!, y = p[i - k + 1]! - cy; return k === 0 ? c * x - sn * y : k === 1 ? cy + sn * x + c * y : u; });
    const tilted = { ...source, cloth: source.cloth.map(d => ({ ...d, geometry: { kind: "inline" as const, data: { ...d.geometry.data, restPositions_m: turn(d.geometry.data.restPositions_m),
      ...(d.geometry.data.initialPositions_m ? { initialPositions_m: turn(d.geometry.data.initialPositions_m) } : {}) } } })) };
    const scene = clothAnalyticSceneDescription(tilted), cloth = await ClothSystem.create(device, scene);
    try {
      // Tick 29 lands the whip edge at about 6 m/s (10 cm a step); the sheet has settled by tick 60.
      const frames = await cloth.capture(Array.from({ length: 60 }, (_, i) => (i + 1) * CLOTH_SCENE_STEP_S));
      let lowest = Infinity, fastest = 0;
      for (const [pose] of frames) {
        const x = pose!.positions_m, v = pose!.velocities_m_s!;
        for (let i = 0; i < x.length / 3; i++) { lowest = Math.min(lowest, x[3 * i + 1]!); fastest = Math.max(fastest, Math.hypot(v[3 * i]!, v[3 * i + 1]!, v[3 * i + 2]!)); }
      }
      assert.equal(cloth.failure, undefined, `halted: ${cloth.failure?.message}`);
      assert.ok(fastest > 5, `the whip edge lands fast (${fastest} m/s)`);
      assert.ok(lowest >= 0, `no vertex below the floor (lowest ${lowest} m)`);
    } finally { cloth.destroy(); }
  } finally { device.destroy(); }
});

test("GPU contact state: touching is a normal force at the step's end, onset its first step, carried with the held flags", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const r = 2e-3, rqMax = 0.02, band = r + 1.5 * Math.sqrt(3) * rqMax + 1e-6;
    const field = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.2, -0.2, -0.1], hi: [0.3, 0.3, 0], mu: 0.3 }], lo: [-0.2, -0.2, -0.1], hi: [0.3, 0.3, 0.3], cell: 0.01, band });
    // Half the sheet rests 1.5 mm above the floor (inside r), the other half 2.6 mm (outside).
    const s = sheet(9, 6, 0.01, [0, 0, 0]), N = s.x.length;
    const X = f32(s.x.map(p => [p[0], p[1], p[0] < 0.035 ? 1.5e-3 : 2.6e-3] as V3)), mesh: OracleMesh = { ...s, x: X };
    const { triangles, edges } = topologyOf(mesh);
    const positions = storage(device, vec4s(X)), xt = storage(device, vec4s(X)), velocities = storage(device, vec4s(X.map(() => [0, 0, 0] as V3)));
    const config: OgcContactConfig = { ...baseConfig(N), r, rqMax, gravity: [0, 0, 0], solidCapacityWords: field.words.length, solidSampleCapacity: 8192 };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(X), config, { positions, velocities });
    det.setSolidField(field);
    const state = async () => new Uint32Array(await readBuffer(device, det.contactState, 4 * N));
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.stepHooks.afterFrame(pass); det.stepHooks.stepEnd(pass); });
    const x = toV3(new Float32Array(await readBuffer(device, positions)), N), s1 = await state();
    assert.equal((await det.readCounters()).status, 0);
    for (let v = 0; v < N; v++) assert.equal(s1[v], x[v]![2] < r ? 3 : 0, `vertex ${v} at ${x[v]![2]}: state ${s1[v]}`);
    assert.ok(s1.some(b => b === 3) && s1.some(b => b === 0), "both halves");
    // The next step end: still touching, no onset.
    await runPass(device, pass => det.stepHooks.stepEnd(pass));
    const s2 = await state();
    for (let v = 0; v < N; v++) assert.equal(s2[v], s1[v]! & 1, `vertex ${v} after a second step`);
    // Carried: a restored detector reports the last step's state, and a step end after it sees no new onset.
    const carried = await det.readCarriedState();
    det.writeCarriedState({ reference: carried.reference, held: carried.held });
    assert.deepEqual(await state(), s2);
    // Lifted clear: nothing touches; set back down: onset again.
    device.queue.writeBuffer(positions, 0, vec4s(X.map(p => [p[0], p[1], 0.01] as V3)));
    await runPass(device, pass => det.stepHooks.stepEnd(pass));
    assert.ok((await state()).every(b => b === 0), "lifted clear");
    device.queue.writeBuffer(positions, 0, vec4s(x));
    await runPass(device, pass => det.stepHooks.stepEnd(pass));
    assert.deepEqual(await state(), s1, "set down: onset again");
    det.destroy();
  });
});
