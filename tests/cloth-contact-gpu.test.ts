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
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { OgcContactDetector, type OgcContactConfig } from "../lib/cloth/contact/ogc-detector";
import { buildOgcEdges } from "../lib/cloth/contact/ogc-topology";
import { OGC_STATUS } from "../lib/cloth/contact/ogc-wgsl";
import { buildClothSolidField, sampleClothSolidField, type ClothSolidField } from "../lib/cloth/contact/solid-field";
import {
  activation, boxSdf, solidBallBound, solidContact, solidTruncate, solidKeep, solidTermScale, sphereSdf, bruteForceDatPairs, bruteForceDetection, contactEnergy, countIntersections, datSeparator, dot, globalFeature, merge,
  motionPenetrates, norm, planarTruncate, rng, sheet, sub, vertexForceHessian, add,
  bruteForcePredictedPairs, predictedConflict, predictedTruncate,
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

test("GPU DAT division pairs: every pair within the larger envelope, once, with a certified plane", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    for (const seed of [3, 11]) {
      const mesh = stackScene(seed), N = mesh.x.length, R = rng(seed + 200);
      const vel: V3[] = mesh.x.map(() => [(R() - 0.5) * 0.8, (R() - 0.5) * 0.8, (R() - 0.5) * 0.8]);
      // Cells stay at the mean edge length, so 4 cm envelopes reach many cells each.
      const config: OgcContactConfig = { ...baseConfig(N), rqMax: 4e-2, gridCapacity: 2048 * N };
      const { det, oracle } = await detect(device, mesh, vel, config);
      const rq = vel.map(v => envelope(config, v));
      const ref = bruteForceDatPairs(oracle, rq, 2e-6);
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
    const ref = bruteForcePredictedPairs(oracle, rq, W, 2e-6);
    // The frame's peak: the first step's all-pairs round is the detector-lifetime peak.
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c0.framePeakPairSlots, oracle);
    const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "vertex-triangle pairs");
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "edge-edge pairs");
    // Absolute envelopes would pair the layers 4 mm apart; the co-moving predictions pair far fewer.
    const absolute = bruteForceDatPairs(oracle, mesh.x.map((_, v) => Math.min(config.rqMax, config.r + 2.5 * norm(W[v]))), 0).pairs.length;
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

test("GPU predicted-motion DAT: crossing predictions redo the round held, which is Planar-DAT", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = spacedStack(23), N = mesh.x.length;
    // The top sheet is predicted 7 mm down at the others, which stay.
    const vel = mesh.x.map(p => [0, 0, p[2] > 6e-3 ? (-7e-3 + 9.81 / 3600) * 60 : 9.81 / 60] as V3);
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 4e-2, gridCapacity: 2048 * N };
    const { det, oracle, W } = await detectPredicted(device, mesh, vel, config);
    const c = await det.readCounters();
    assert.equal(c.status & ~OGC_STATUS.envelopeCap, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
    assert.equal(c.redos, 1, "the round is redone"); assert.ok(c.conflicts > 0);
    assert.ok(W.every(w => norm(w) === 0), "the redone round holds every prediction");
    const rq = vel.map(v => envelope(config, v));
    const ref = bruteForceDatPairs(oracle, rq, 2e-6);
    const got = pairsOf(new Uint32Array(await readBuffer(device, det.pairs)), c.framePeakPairSlots, oracle);
    const strict = (keys: Iterable<string>) => [...keys].filter(k => !ref.ambiguous.has(k)).sort();
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "vt").map(g => g.key)), strict(ref.vt), "held round: vertex-triangle pairs");
    assert.deepEqual(strict(got.filter(g => g.pair.kind === "ee").map(g => g.key)), strict(ref.ee), "held round: edge-edge pairs");
    // The conflict's vertices stay held into the next step.
    const held = new Uint32Array(await readBuffer(device, det.work));
    const off = 7 * N + det.T + det.E;
    assert.ok(Array.from(held.subarray(off, off + N)).some(f => f & 1), "conflicting vertices are flagged");
    det.destroy();
  });
});

test("GPU carried state: the per-vertex reference, rho_v and held flags a step hands the next restore exactly, so a replayed step equals the original", { skip: !gpu }, async () => {
  await withContactDevice(async device => {
    const mesh = spacedStack(23), N = mesh.x.length;
    // The crossing-prediction scene: its first steps leave held flags and residuals behind.
    const vel = mesh.x.map(p => [0, 0, p[2] > 6e-3 ? (-7e-3 + 9.81 / 3600) * 60 : 9.81 / 60] as V3);
    const config: OgcContactConfig = { ...baseConfig(N), r: 1e-3, rqBase: 1.5e-3, envelopeScale: 2.5, prediction: "inertial", rqMax: 4e-2, gridCapacity: 2048 * N };
    const { det, positions } = await detectPredicted(device, mesh, vel, config);
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
