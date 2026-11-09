/**
 * Lane C2 M6, one-way moving solids: one solid with prescribed poses, its field
 * baked once in its own body frame, against the OGC detector (GPU only).
 *
 *   detector  a sheet over a plate that rises, slides and turns in one step:
 *             predictions pushed towards a_v by the bound, body entries and terms against the Float64
 *             ball bound in the body frame, the posed surface samples, the vertex
 *             block (friction against the carried surface point) against the oracle,
 *             and the truncation keeping phi_body >= (1 - gamma_r) phi0 at the end pose
 *   failures  cloth pressed between a descending plate and a static floor fails as
 *             bodySqueeze; a frame without poses, or a step past them, fails explicitly
 *   plate     C1's solver: a sheet lands on a plate, which then rises, slides and turns;
 *             no intersection, no vertex or face inside the plate at any step, and
 *             friction carries the sheet at the plate's rigid velocity
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { OgcContactDetector, type OgcBodyPose, type OgcContactConfig } from "../lib/cloth/contact/ogc-detector";
import { buildOgcEdges } from "../lib/cloth/contact/ogc-topology";
import { OGC_STATUS } from "../lib/cloth/contact/ogc-wgsl";
import { buildClothSolidField, sampleClothSolidField, type ClothSolidField } from "../lib/cloth/contact/solid-field";
import { clothGridData } from "../lib/cloth/grid";
import { prepareClothTopology } from "../lib/cloth/topology";
import { CLOTH_NEWTON_SCENE_DEFAULTS } from "../lib/cloth/system";
import type { ClothMaterial } from "../lib/cloth/types";
import { buildClothSolverModel } from "../lib/cloth/oracle";
import { ClothGPUSolver } from "../lib/cloth/gpu-solver";
import { CLOTH_BINDING, CLOTH_STATE } from "../lib/cloth/gpu-solver-wgsl";
import {
  activation, boxSdf, contactEnergy, facesInsideSolid, motionPenetrates, norm, rng, sheet, solidBallBound, solidContact, solidKeep, solidTermScale, sub, vertexForceHessian,
  type OracleContact, type OracleMesh, type V3,
} from "./cloth-contact-oracle";

const gpu = !!process.env.WEBGPU_NODE_MODULE;

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}
function storage(device: GPUDevice, data: Float32Array<ArrayBuffer>) {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(b, 0, data);
  return b;
}
async function readBuffer(device: GPUDevice, buffer: GPUBuffer, bytes = buffer.size): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, 0, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = read.getMappedRange().slice(0); read.destroy();
  return out;
}
const runPass = async (device: GPUDevice, fn: (pass: GPUComputePassEncoder) => void) => {
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass(); fn(pass); pass.end(); device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
};
const vec4s = (pts: V3[]) => new Float32Array(pts.flatMap(p => [p[0], p[1], p[2], 0]));
const f32 = (pts: V3[]) => pts.map(p => p.map(Math.fround) as V3);
const toV3 = (a: Float32Array, N: number, at = 0) => Array.from({ length: N }, (_, v) => [a[4 * (at + v)]!, a[4 * (at + v) + 1]!, a[4 * (at + v) + 2]!] as V3);
const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// ---- poses, as the GPU holds them (rotation entries rounded to f32)
interface Pose { c: V3; R: number[][] }
const quat = (axis: V3, angle: number): [number, number, number, number] => {
  const s = Math.sin(angle / 2), n = norm(axis);
  return [axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, Math.cos(angle / 2)];
};
function poseOf(p: OgcBodyPose): Pose {
  const [x, y, z, w] = p.orientation;
  const R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]].map(r => r.map(Math.fround));
  return { c: p.position.map(Math.fround) as V3, R };
}
const mul = (R: number[][], x: V3): V3 => [0, 1, 2].map(i => R[i]![0]! * x[0] + R[i]![1]! * x[1] + R[i]![2]! * x[2]) as V3;
const mulT = (R: number[][], x: V3): V3 => [0, 1, 2].map(i => R[0]![i]! * x[0] + R[1]![i]! * x[1] + R[2]![i]! * x[2]) as V3;
const toBody = (p: Pose, x: V3) => mulT(p.R, sub(x, p.c));
const toWorld = (p: Pose, y: V3) => add3(mul(p.R, y), p.c);

function topologyOf(mesh: OracleMesh) {
  const triangles = new Uint32Array(mesh.tris.flatMap(t => [...t, 0]));
  const edges = buildOgcEdges(triangles);
  const oracleEdges: [number, number, number, number][] = [];
  for (let e = 0; e < edges.length / 4; e++) oracleEdges.push([edges[4 * e]!, edges[4 * e + 1]!, edges[4 * e + 2]!, edges[4 * e + 3]!]);
  return { triangles, edges, oracle: { ...mesh, edges: oracleEdges } };
}
interface Entry { x: number; y: number; z: number; w: number }
function listsOf(contacts: Uint32Array, N: number) {
  const C = contacts[1]!, countsBase = contacts[2]!, entriesBase = contacts[3]!, lists: Entry[][] = [];
  for (let v = 0; v < N; v++) {
    const n = contacts[4 * countsBase + v]!, l: Entry[] = [];
    for (let i = 0; i < n; i++) { const b = 4 * (entriesBase + v * C + i); l.push({ x: contacts[b]!, y: contacts[b + 1]!, z: contacts[b + 2]!, w: contacts[b + 3]! }); }
    lists.push(l);
  }
  return lists;
}
const BODY_Y = 0x70000001, SOLID_Y = 0x70000000;
const lambdaOf = (e: Entry) => new Float32Array(new Uint32Array([e.z]).buffer)[0]!;

/** The vertex block at the current positions: per vertex (f, e), (H[0], count), H[1], H[2]. */
async function evalVertexBlock(device: GPUDevice, det: OgcContactDetector, positions: GPUBuffer, Xt: V3[], N: number): Promise<Float32Array> {
  const xt = storage(device, vec4s(Xt)), L = det.topology.layout;
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
  ogcAccumulateVertexContact(v, ogcX(v), &f, &H);
  outBuf[4u * v] = vec4<f32>(f, ogcVertexContactEnergy(v, ogcX(v)));
  outBuf[4u * v + 1u] = vec4<f32>(H[0], f32(ogcContactCount(v)));
  outBuf[4u * v + 2u] = vec4<f32>(H[1], 0.0);
  outBuf[4u * v + 3u] = vec4<f32>(H[2], 0.0);
}`;
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [positions, xt, det.topo, out, det.solverBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  await runPass(device, pass => { pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); });
  const res = new Float32Array(await readBuffer(device, out));
  xt.destroy(); out.destroy();
  return res;
}

const baseConfig = (N: number): OgcContactConfig => ({
  r: 2e-3, kc: 1e5, mu: 0.3, epsV: 1e-2, dt: 1 / 60, gravity: [0, 0, -9.81],
  rqBase: 3e-3, rqMax: 2e-2, envelopeScale: 1.5, contactCapacity: 96, pairCapacity: 512 * N, gridCapacity: 128 * N, bucketCount: 4096, roundCap: 2,
  prediction: "none",
});

/** A 6 x 6 x 2 cm plate in its body frame (sharp edges: surface samples), mu 0.4. */
const plateHalf: V3 = [0.03, 0.03, 0.01];
const plateField = (band = 0.09) => buildClothSolidField({
  parts: [{ kind: "sdf", sdf: boxSdf([-plateHalf[0], -plateHalf[1], -plateHalf[2]], plateHalf), lo: [-plateHalf[0], -plateHalf[1], -plateHalf[2]], hi: plateHalf, mu: 0.4 }],
  lo: [-0.13, -0.13, -0.11], hi: [0.13, 0.13, 0.11], cell: 0.005, band });

test("GPU moving solid: pushed predictions, body entries and terms, posed samples, the vertex block and the truncation against the oracle", { skip: !gpu }, async () => {
  await withDevice(async device => {
    const field = plateField(), R = rng(31);
    // The sheet 0.5-4.5 mm over the plate's top (z = 0.03); the plate rises 3 mm, slides 2 mm and turns 3 degrees over the step.
    const s = sheet(15, 10, 0.01, [0.003, 0.004, 0]);
    const mesh: OracleMesh = { ...s, x: f32(s.x.map(p => [p[0], p[1], 0.0305 + 0.004 * R()] as V3)) };
    const N = mesh.x.length, { triangles, edges, oracle } = topologyOf(mesh);
    const poses: OgcBodyPose[] = [
      { position: [0.07, 0.05, 0.02], orientation: [0, 0, 0, 1] },
      { position: [0.072, 0.05, 0.023], orientation: quat([0, 0, 1], 3 * Math.PI / 180) },
    ];
    const [p0, p1] = poses.map(poseOf) as [Pose, Pose];
    // rqMax 3 cm: the turn carries the sheet's far corners 8 mm, inside the band's room 0.5 (rqMax - r_q).
    const config: OgcContactConfig = { ...baseConfig(N), rqMax: 0.03, body: { capacityWords: field.words.length, sampleCapacity: 4096, poseCapacity: 4 } };
    const positions = storage(device, vec4s(mesh.x)), velocities = storage(device, vec4s(mesh.x.map(() => [0, 0, 0])));
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(mesh.x), config, { positions, velocities });
    det.setBodyField(field);
    assert.throws(() => det.encodeFrameBegin(device.createCommandEncoder().beginComputePass()), /setBodyPoses/);
    assert.throws(() => det.setBodyPoses(poses.slice(0, 1)), /poses/);
    det.setBodyPoses(poses);
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
    const c0 = await det.readCounters();
    assert.equal(c0.status, 0, `status ${OgcContactDetector.describeStatus(c0.status)}, ${c0.conflicts} conflicts, held pair ${c0.heldConflictPair.map(i => (i & 0x7fffffff) - N).join(", ")} past N`);
    const diag = await det.readDiagnosis();
    const work = new Float32Array(await readBuffer(device, det.work));
    const pred = new Float32Array(await readBuffer(device, det.pred));
    const xdet = new Float32Array(await readBuffer(device, det.det));
    const contacts = new Uint32Array(await readBuffer(device, det.contacts)), lists = listsOf(contacts, N);
    const bodyOff = 8 * N + det.T + det.E, bodyDevOff = 20 * N + det.T + det.E, scaleOff = 24 * N + det.T + det.E;
    // The plate turns theta over the step; a vertex's straight motion strays from straight in its
    // frame by sigma = (theta^2 (|x - c0| + Q) + 2 theta Q) / 8, Q = max(|w - dc|, |a - dc|, |dc|) + env / 2
    // (w = 0 or a_v: the same Q), which the field bound's room loses L_max sigma of.
    let trace = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) trace += p1.R[i]![j]! * p0.R[i]![j]!;
    const theta = 2 * Math.asin(Math.min(1, 0.5 * Math.sqrt(Math.max(3 - trace, 0)))), dc = sub(p1.c, p0.c), L = field.maxLipschitz;
    // Every vertex is held (prediction "none"): w = 0, moved towards a_v as far as the plate's bound
    // needs; a conflict's least shift then moves it back towards its anchor; one carried with the plate
    // (bit 2: the fail-safe's impact zone, never in this scene) is a_v.
    let pushedCount = 0, carriedCount = 0, entries = 0, shifted = 0;
    const bounds = mesh.x.map((x, v) => {
      const carried = (diag.held[v]! & 4) !== 0;
      const env = diag.env[v]!, y0 = toBody(p0, x), a = sub(toWorld(p1, y0), x);
      const Q = Math.max(norm(sub(a, dc)), norm(dc)) + 0.5 * env;
      const sigma = (theta * theta * (norm(sub(x, p0.c)) + Q) + 2 * theta * Q) / 8;
      // A vertex carried before its ball was taken moves 0 relative to the plate; else |a_v|.
      const rho = norm(a) + 0.5 * env;
      const room = Math.max(0.5 * (config.rqMax - env), 0);
      if (norm(y0) - rho - sigma >= det.bodyReach) return { y0, a, carried, sigma, rho, reach: false, b: { far: true, phi0: field.band, g0: [0, 0, 0] as V3, D: [0, 0, 0] as V3, stored: false } };
      assert.ok(norm(a) <= room, `vertex ${v}: the step's carry ${norm(a)} fits the band`);
      // The oracle's ball bound is far when the ball rc clears r without sigma: then its terms are 0.
      return { y0, a, carried, sigma, rho, reach: true, b: solidBallBound(field, y0, Math.min(rho, 0.5 * config.rqMax), config.r) };
    });
    for (let v = 0; v < N; v++) {
      const { y0, a, carried, sigma, rho, reach, b } = bounds[v]!;
      const body = lists[v]!.filter(e => e.y === BODY_Y);
      entries += body.length;
      const w = [0, 1, 2].map(k => pred[4 * v + k]!) as V3;
      const aGpu = [0, 1, 2].map(k => work[bodyOff + 8 * v + 4 + k]!) as V3;
      assert.ok(norm(sub(aGpu, a)) <= 2e-7 + 1e-4 * norm(a), `vertex ${v} a ${aGpu} vs ${a}`);
      // Moving with the plate (bit 6), its round is straight in the plate's frame; its sigma is how far
      // that path c(t) + R(t) l(t) strays from straight in the world, Q = |w - a_v| + |w| + env / 2.
      const Qs = norm(sub(w, aGpu)) + norm(w) + 0.5 * diag.env[v]!;
      const sigmaV = (diag.held[v]! & 64) ? (theta * theta * (norm(sub(mesh.x[v]!, p0.c)) + Qs) + 2 * theta * Qs) / 8 : sigma;
      assert.ok(Math.abs(work[bodyOff + 8 * v + 7]! - sigmaV) <= 1e-9 + 1e-4 * sigmaV, `vertex ${v} sigma ${work[bodyOff + 8 * v + 7]} vs ${sigmaV}`);
      if (carried) {
        carriedCount++;
        assert.deepEqual(w, aGpu, `vertex ${v} carried: prediction ${w} vs a ${aGpu}`);
        continue;
      }
      // The vertex has an entry when the ball rho, widened by sigma, can reach r.
      const margin = Math.min(b.phi0, field.band) - L * (rho + sigma) - config.r;
      if (Math.abs(margin) < 1e-5) continue;
      assert.equal(body.length, !reach || margin > 0 ? 0 : 1, `vertex ${v} body entry (far ${b.far}, reach gap ${norm(y0) - rho - sigma - det.bodyReach}, margin ${margin}, held ${diag.held[v]}, w ${[0, 1, 2].map(k => pred[4 * v + k])}, env ${diag.env[v]})`);
      const q = [0, 1, 2, 3, 4, 5, 6].map(k => k < 4 ? work[bodyOff + 8 * v + k]! : work[bodyDevOff + 4 * v + k - 4]!);
      if (!reach || margin > 0) {
        assert.deepEqual(q, [0, 0, 0, 0, 0, 0, 0]);
        assert.deepEqual(w, [0, 0, 0], `vertex ${v} prediction beyond the plate's reach`);
        continue;
      }
      // The terms are in the body frame at the end pose: (g0 s, rc) and D s with s = 1 / (room - L sigma),
      // rc = min(rho, rqMax / 2) the affine bound's ball.
      const room = Math.max(0.9 * b.phi0, b.phi0 - config.r) - L * sigma, sc = 1 / room, rc = Math.min(rho, 0.5 * config.rqMax);
      assert.ok(Math.abs(q[3]! - rc) <= 1e-5 * rc, `vertex ${v} affine ball ${q[3]} vs ${rc}`);
      const g0 = b.far ? [0, 0, 0] : b.g0, D = b.far ? [0, 0, 0] : b.D;
      const qr = [g0[0]! * sc, g0[1]! * sc, g0[2]! * sc, rc, D[0]! * sc, D[1]! * sc, D[2]! * sc];
      const qs = Math.max(...qr.map(Math.abs));
      for (let k = 0; k < 7; k++) assert.ok(Math.abs(q[k]! - qr[k]!) <= 2e-3 * qs, `vertex ${v} q/D[${k}] ${q[k]} vs ${qr[k]}`);
      // Moved towards a_v as far as either certificate needs (predictionScale on d = R1^T (0 - a_v)):
      // the affine f(d) = D s.|d| - g0 s.d <= 1 within rc, or the Lipschitz ball |d| <= rl = room / L_max.
      const rl = room / L;
      const scaleOf = (d: V3) => {
        const l = norm(d), f = qr[4]! * Math.abs(d[0]) + qr[5]! * Math.abs(d[1]) + qr[6]! * Math.abs(d[2]) - (qr[0]! * d[0] + qr[1]! * d[1] + qr[2]! * d[2]);
        return { f, l, k: l > rl && !(f <= 1 && l <= rc) ? Math.max(f > 1 ? Math.min(rc / l, 1 / f) : rc / l, rl / l) : 1 };
      };
      const { f, l, k: kb } = scaleOf(mulT(p1.R, scale3(a, -1)));
      let want = kb < 1 ? scale3(a, 1 - kb) : [0, 0, 0] as V3;
      // A round redone shifts a conflict's predictions (least shift, OGC_REDO_WGSL): towards the
      // anchor (a_v with bit 5, else X) by the round's scale s, then bounded again about a_v.
      const shift = work[scaleOff + 2 * v + 1]!, anc = (diag.held[v]! & 32) ? a : [0, 0, 0] as V3;
      if (shift < 1) {
        shifted++;
        want = add3(anc, scale3(sub(want, anc), shift));
        const k2 = scaleOf(mulT(p1.R, sub(want, a))).k;
        if (k2 < 1) want = add3(a, scale3(sub(want, a), k2));
      }
      if (kb < 1) pushedCount++;
      if (Math.abs(f - 1) > 1e-3 && Math.abs(l - rl) > 1e-3 * rl && Math.abs(l - rc) > 1e-3 * rc)
        assert.ok(norm(sub(w, want)) <= 2e-7 + 1e-3 * norm(a), `vertex ${v} prediction ${w} vs ${want} (phi0 ${b.phi0}, f(-a) ${f}, kb ${kb})`);
      const lr = (b.phi0 < config.r ? -activation(b.phi0, config.r, config.kc)[1] * norm(b.g0) : 0) * sampleClothSolidField(field, y0).mu;
      assert.ok(Math.abs(lambdaOf(body[0]!) - lr) <= 2e-3 * Math.max(1, lr), `vertex ${v} lambda ${lambdaOf(body[0]!)} vs ${lr}`);
    }
    // The frame counter sums its rounds (the redos of conflicts with the plate's samples too).
    assert.ok(pushedCount + carriedCount > 30 && (c0.redos ? c0.bodyContacts >= entries : c0.bodyContacts === entries), `${pushedCount} vertices the plate's bound pushed and ${carriedCount} carried, ${entries} body entries, counter ${c0.bodyContacts} over ${1 + c0.redos} rounds`);
    assert.ok(pushedCount + carriedCount < N, "the plate's reach ends inside the sheet");
    assert.ok(c0.redos === 0 || shifted > 0, `${c0.redos} redos shifted ${shifted} vertices`);
    assert.equal(c0.zoneVertices, carriedCount, "the zone counter counts the carried vertices");

    // The plate's surface samples: x_det at the start pose, predicted exactly to the end pose.
    const SB = det.bodySamples;
    assert.ok(SB > 50, `${SB} plate samples`);
    const local = toV3(new Float32Array(contacts.buffer), SB, det.bodySampleBase);
    const xs = toV3(xdet, SB, N), ws = toV3(pred, SB, 2 * N);
    for (let k = 0; k < SB; k++) {
      assert.ok(norm(sub(xs[k]!, toWorld(p0, local[k]!))) <= 1e-6, `sample ${k} x_det`);
      assert.ok(norm(sub(add3(xs[k]!, ws[k]!), toWorld(p1, local[k]!))) <= 1e-6, `sample ${k} prediction`);
    }
    const sampled = lists.flatMap(l => l.filter(e => e.w === 1 && (e.y & 0x80000000) === 0));
    assert.ok(sampled.length > 0 && sampled.every(e => e.x >= N && e.x < N + SB), `${sampled.length} sample-facet entries on the plate's samples`);

    // The vertex block, near the co-moving positions, against the Float64 contact in the body frame at the end pose.
    const X = f32(mesh.x.map((x, v) => {
      const { a, b } = bounds[v]!, h = 0.3 * Math.min(xdet[4 * v + 3]!, b.phi0);
      return [x[0] + (b.far ? 0 : a[0]) + (R() - 0.5) * h, x[1] + (b.far ? 0 : a[1]) + (R() - 0.5) * h, x[2] + (b.far ? 0 : a[2]) + (R() - 0.5) * h] as V3;
    }));
    const Xt = mesh.x;
    device.queue.writeBuffer(positions, 0, vec4s(X));
    const res = await evalVertexBlock(device, det, positions, Xt, N);
    const P = { r: config.r, k: config.kc, mu: config.mu, epsV: config.epsV, dt: config.dt };
    // A sample-facet entry's sample sits at the end pose and started the step at the start pose.
    const XS = [...X, ...local.map(l => toWorld(p1, l))], XtS = [...Xt, ...local.map(l => toWorld(p0, l))];
    let checked = 0;
    for (let v = 0; v < N; v++) {
      let f: V3 = [0, 0, 0], e = 0;
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const ent of lists[v]!) {
        let q: { f: V3; H: number[][] }, en: number;
        if (ent.y === BODY_Y) {
          // The slip is against the plate's material point that sat at x_t: in the end pose's body frame it is R0^T (x_t - c0).
          const bc = solidContact(field, toBody(p1, X[v]!), toBody(p0, Xt[v]!), lambdaOf(ent), P);
          q = { f: mul(p1.R, bc.f), H: [0, 1, 2].map(i => [0, 1, 2].map(k => [0, 1, 2].reduce((acc, m) => acc + [0, 1, 2].reduce((s2, n) => s2 + p1.R[i]![m]! * bc.H[m]![n]! * p1.R[k]![n]!, 0), 0))) };
          en = bc.e;
        } else {
          assert.ok(ent.w === 1 && ent.y !== SOLID_Y, `vertex ${v}: only body and sample-facet entries`);
          const t = ent.y & 0x0fffffff, tri = oracle.tris[t]!;
          const c: OracleContact = { kind: "vf", ids: [ent.x, tri[0], tri[1], tri[2]], feature: (ent.y >>> 28) & 7, s: 0, t: 0, lambda: lambdaOf(ent) / config.mu };
          q = vertexForceHessian(c, XS, XtS, P, c.ids.indexOf(v)); en = contactEnergy(c, XS, XtS, P);
        }
        f = add3(f, q.f); e += en;
        for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i]![k]! += q.H[i]![k]!;
      }
      if (!lists[v]!.length) continue;
      checked++;
      const fs = Math.max(1, Math.hypot(...f)), hs = Math.max(1, ...H.flat().map(Math.abs));
      for (let k = 0; k < 3; k++) assert.ok(Math.abs(res[16 * v + k]! - f[k]!) <= 2e-3 * fs, `vertex ${v} f[${k}] ${res[16 * v + k]} vs ${f[k]}`);
      assert.ok(Math.abs(res[16 * v + 3]! - e) <= 2e-3 * Math.max(1e-6, Math.abs(e)) + 1e-9, `vertex ${v} energy ${res[16 * v + 3]} vs ${e}`);
      for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++)
        assert.ok(Math.abs(res[16 * v + 4 * (col + 1) + row]! - H[row]![col]!) <= 2e-3 * hs, `vertex ${v} H`);
    }
    assert.ok(checked > 30, `${checked} vertices with contact checked`);

    // A step's motion that drives the sheet down into the rising plate: the truncation holds
    // phi_body >= (1 - gamma_r) phi0 at the end pose, and r for a vertex the plate cannot reach.
    const Xnew = f32(mesh.x.map(p => [p[0] + (R() - 0.5) * 6e-3, p[1] + (R() - 0.5) * 6e-3, p[2] - 6e-3 * R()] as V3));
    device.queue.writeBuffer(positions, 0, vec4s(Xnew));
    await runPass(device, pass => det.encodeTruncate(pass));
    const got = toV3(new Float32Array(await readBuffer(device, positions)), N);
    for (let v = 0; v < N; v++) {
      const { b } = bounds[v]!, phi = sampleClothSolidField(field, toBody(p1, got[v]!)).phi;
      assert.ok(phi >= (b.far ? config.r : solidKeep(b.phi0, 0.9, config.r)) * (1 - 1e-3), `vertex ${v} phi_body ${phi} after truncation (far ${b.far}, phi0 ${b.phi0})`);
    }
    assert.equal(motionPenetrates(oracle, mesh.x, got), false);
    const c1 = await det.readCounters();
    assert.ok(c1.bodyCut > 10, `${c1.bodyCut} body cuts`);
    // Faces in the plate at its end pose; a vertex pushed into it puts its faces there.
    await runPass(device, pass => det.encodeIntersectionCount(pass));
    const c2 = await det.readCounters();
    const exact = facesInsideSolid(field, Float64Array.from(got.flatMap(x => toBody(p1, x))), new Uint32Array(mesh.tris.flat()));
    // Truncation alone keeps each sample apart from each face and each cloth edge apart from
    // each of the samples' hull edges, so no face edge slips between two edge samples: no face
    // is in the plate at its end pose, exactly or by the lattice.
    assert.ok(det.bodySampleEdges > det.bodySamples, `${det.bodySampleEdges} hull edges for ${det.bodySamples} samples`);
    assert.equal(c2.intersections, 0, `${c2.intersections} face pairs`);
    assert.equal(exact.inside.length, 0, `${exact.inside.length} faces in the plate after truncation (exact)`);
    assert.equal(c2.solidIntersections, 0, `${c2.solidIntersections} faces in the plate by the lattice`);
    const pushed = got.map((x, v) => (v === 60 ? toWorld(p1, [0.0, 0.0, 0.008]) : x) as V3);
    device.queue.writeBuffer(positions, 0, vec4s(pushed));
    await runPass(device, pass => det.encodeIntersectionCount(pass));
    assert.ok((await det.readCounters()).solidIntersections >= 1, "a vertex inside the plate puts its faces in it");

    // A frame of two steps with poses for one fails explicitly.
    det.setBodyPoses(poses);
    await runPass(device, pass => { det.encodeFrameBegin(pass); for (let k = 0; k < 2; k++) det.encodeStepBegin(pass); });
    assert.ok((await det.readCounters()).status & OGC_STATUS.bodyPose, "a step past the frame's poses");
    det.destroy(); positions.destroy(); velocities.destroy();
  });
});

test("GPU moving solid fails explicitly: cloth pressed between a descending plate and a static floor is a squeeze", { skip: !gpu }, async () => {
  await withDevice(async device => {
    const body = plateField();
    const floor = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.04, -0.04, -0.04], hi: [0.2, 0.15, 0.0], mu: 0.3 }], lo: [-0.1, -0.1, -0.1], hi: [0.26, 0.21, 0.1], cell: 0.005, band: 0.06 });
    // The sheet 1.2 mm over the floor (z = 0), the plate's bottom 3 mm over the sheet, descending 5 mm.
    const s = sheet(15, 10, 0.01, [0.003, 0.004, 0]);
    const mesh: OracleMesh = { ...s, x: f32(s.x.map(p => [p[0], p[1], 0.0012] as V3)) };
    const N = mesh.x.length, { triangles, edges } = topologyOf(mesh);
    const config: OgcContactConfig = { ...baseConfig(N), solidCapacityWords: floor.words.length, solidSampleCapacity: 8192,
      body: { capacityWords: body.words.length, sampleCapacity: 4096, poseCapacity: 2 } };
    const positions = storage(device, vec4s(mesh.x)), velocities = storage(device, vec4s(mesh.x.map(() => [0, 0, 0])));
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges }, vec4s(mesh.x), config, { positions, velocities });
    det.setSolidField(floor); det.setBodyField(body);
    det.setBodyPoses([{ position: [0.07, 0.05, 0.0142], orientation: [0, 0, 0, 1] }, { position: [0.07, 0.05, 0.0092], orientation: [0, 0, 0, 1] }]);
    await runPass(device, pass => { det.encodeFrameBegin(pass); det.encodeStepBegin(pass); det.encodeDetection(pass); });
    const c = await det.readCounters();
    assert.ok(c.status & OGC_STATUS.bodySqueeze, `status ${OgcContactDetector.describeStatus(c.status)}`);
    det.destroy(); positions.destroy(); velocities.destroy();
  });
});

// ---------------------------------------------------------------- plate, inside C1's solver

const cotton: ClothMaterial = { id: "cotton", arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };

test("GPU moving plate in C1's solver: the sheet rides it without entering it, and friction carries it at the plate's rigid velocity", { skip: !gpu }, async t => {
  await withDevice(async device => {
    const DT = 1 / 60, RQ_MAX = 0.04, R_C = 2e-3, STILL = 15, MOVING = 45;
    const half: V3 = [0.08, 0.01, 0.08], v: V3 = [0.3, 0.2, 0], omega = 1.0, c0: V3 = [0, -0.01, 0];
    const field = buildClothSolidField({
      parts: [{ kind: "sdf", sdf: boxSdf([-half[0], -half[1], -half[2]], half), lo: [-half[0], -half[1], -half[2]], hi: half, mu: 0.5 }],
      lo: [-0.2, -0.13, -0.2], hi: [0.2, 0.13, 0.2], cell: 0.01, band: R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005 });
    // Rest for STILL steps, then rise and slide at v while turning at omega about y.
    const poseAt = (step: number): OgcBodyPose => {
      const tm = Math.max(0, step - STILL) * DT;
      return { position: [c0[0] + v[0] * tm, c0[1] + v[1] * tm, c0[2] + v[2] * tm], orientation: quat([0, 1, 0], omega * tm) };
    };
    // The sheet overhangs the plate's sharp edges by 2 cm (its surface samples hold the faces off them).
    const n = 26, W = 0.2;
    const data = clothGridData({ columns: n, rows: n, width_m: W, height_m: W, origin_m: [-0.5 * W, 0.003, -0.5 * W] });
    const topology = prepareClothTopology(data, [cotton]);
    const m = buildClothSolverModel(topology, [cotton], []);
    const T = m.topology.faceCount, N = m.topology.vertexCount, E = m.topology.edgeCount;
    const triangles = new Uint32Array(4 * T);
    for (let f = 0; f < T; f++) triangles.set(m.topology.faces.subarray(3 * f, 3 * f + 3), 4 * f);
    const rest = new Float32Array(4 * N);
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) rest[4 * i + k] = m.topology.data.restPositions_m[3 * i + k]!;
    const config: OgcContactConfig = { r: R_C, kc: 100, mu: 0.2, epsV: 1e-2, dt: DT, gravity: [0, -9.81, 0], rqBase: 3e-3, rqMax: RQ_MAX, envelopeScale: 2.5,
      contactCapacity: 128, pairCapacity: 840 * N, gridCapacity: 32 * (N + T + E), bucketCount: 1 << 16, roundCap: 4, prediction: "guess", newton: true, planeRecordCapacity: 32 * N,
      body: { capacityWords: field.words.length, sampleCapacity: 8192, poseCapacity: 2 } };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges: m.topology.edges }, rest, config);
    const solver = await ClothGPUSolver.create(device, m, { origin_m: [0, 0, 0], dt_s: DT, gravity_m_s2: [0, -9.81, 0], integrator: "bdf2", newton: CLOTH_NEWTON_SCENE_DEFAULTS,
      diagnostics: true, label: "plate", contact: { ...det.solverContact(CLOTH_BINDING.contact, true), alpha: det.alpha, state: det.contactState }, hooks: { afterFrame: det.stepHooks.afterFrame, filter: det.stepHooks.filter, afterNewtonStep: det.stepHooks.afterNewtonStep, stepEnd: det.stepHooks.stepEnd } });
    const stride = solver.tables.stride;
    det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: { buffer: solver.state, offset: 16 * CLOTH_STATE.xt * stride, size: 16 * stride }, ...(det.config.prediction === "target" ? { target: solver.inertialTarget } : {}), direction: solver.newtonDirection });
    const t0 = performance.now();
    det.setBodyField(field);
    t.diagnostic(`plate field ${field.words.length} words, ${det.bodySamples} samples, ${det.bodySampleEdges} hull edges, ${(performance.now() - t0).toFixed(0)} ms`);
    const FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration
      | OGC_STATUS.bodySqueeze | OGC_STATUS.heldConflict | OGC_STATUS.bodyPose | OGC_STATUS.planeRecordOverflow;
    let worstPhi = Infinity, faceLower = Infinity;
    const t1 = performance.now();
    let redoSteps = 0, redosMax = 0, conflictsAll = 0, zoneSteps = 0, zoneAll = 0, zonePeak = 0, capHitSteps = 0, recordPeak = 0;
    for (let step = 0; step < STILL + MOVING; step++) {
      det.setBodyPoses([poseAt(step), poseAt(step + 1)]);
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      det.encodeFrameBegin(pass); solver.encodeStep(pass); det.encodeIntersectionCount(pass);
      pass.end(); device.queue.submit([enc.finish()]);
      const c = await det.readCounters();
      if (c.redos > 0) redoSteps++;
      redosMax = Math.max(redosMax, c.redos); conflictsAll += c.conflicts;
      if (c.roundCapHits) capHitSteps++;
      recordPeak = c.peakPlaneRecords;
      assert.equal(c.status & FAILING, 0, `step ${step}: status ${OgcContactDetector.describeStatus(c.status)} (a held conflict's pair, past N: ${c.heldConflictPair.map(i => (i & 0x7fffffff) - N).join(", ")})`);
      const end = poseOf(poseAt(step + 1)), x = await solver.readPositions();
      const body = new Float64Array(3 * N);
      for (let i = 0; i < N; i++) {
        const y = toBody(end, [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!]);
        body.set(y, 3 * i);
        worstPhi = Math.min(worstPhi, sampleClothSolidField(field, y).phi);
      }
      assert.ok(worstPhi > 0, `step ${step}: a vertex inside the plate (phi ${worstPhi})`);
      if (c.intersections !== 0 || c.solidIntersections !== 0) {
        const r = facesInsideSolid(field, body, m.topology.faces);
        const what = r.inside.slice(0, 4).map(f => [0, 1, 2].map(k => { const i = m.topology.faces[3 * f + k]!, y = [body[3 * i]!, body[3 * i + 1]!, body[3 * i + 2]!] as V3; return `${i}:(${y.map(u => u.toFixed(5))}) phi ${sampleClothSolidField(field, y).phi.toExponential(3)}`; }).join(' ')).join(' | ');
        assert.fail(`step ${step}: ${c.intersections} face pairs, ${c.solidIntersections} faces in the plate (status ${OgcContactDetector.describeStatus(c.status)}, uncertified ${c.uncertified}, rounds ${c.detections}, conflicts ${c.conflicts}, redos ${c.redos}, held ${c.held}); exact ${r.inside.length}: ${what}`);
      }
      if (step % 10 === 9 || step === STILL + MOVING - 1) {
        const r = facesInsideSolid(field, body, m.topology.faces);
        faceLower = Math.min(faceLower, r.lower);
        assert.equal(r.inside.length, 0, `step ${step}: ${r.inside.length} faces inside the plate (exact)`);
      }
      if (c.zoneVertices) { zoneSteps++; zoneAll += c.zoneVertices; zonePeak = Math.max(zonePeak, c.zonePeak); }
      if (step % 15 === 14) t.diagnostic(`step ${step}: zone ${c.zoneVertices} peak ${c.zonePeak}, rounds ${c.detections}, body contacts ${c.bodyContacts}, body cuts ${c.bodyCut}, sample-facet ${c.sfPairs}, truncated ${c.truncated}, conflicts ${c.conflicts} redos ${c.redos}, held ${c.held}, envelope ${c.peakEnvelope.toExponential(2)}`);
    }
    const ms = (performance.now() - t1) / (STILL + MOVING);
    // Friction carries the sheet on the plate's top at its rigid velocity v + omega x (x - c).
    const state = await solver.readState(), end = poseAt(STILL + MOVING), c = end.position as V3, w: V3 = [0, omega, 0];
    let err = 0, ref = 0, onTop = 0;
    for (let i = 0; i < N; i++) {
      const x: V3 = [state.positions_m[3 * i]!, state.positions_m[3 * i + 1]!, state.positions_m[3 * i + 2]!];
      const y = toBody(poseOf(end), x);
      if (!(Math.abs(y[0]) < half[0] - 0.01 && Math.abs(y[2]) < half[2] - 0.01)) continue;
      onTop++;
      const rigid = add3(v, cross(w, sub(x, c))), got: V3 = [state.velocities_m_s[3 * i]!, state.velocities_m_s[3 * i + 1]!, state.velocities_m_s[3 * i + 2]!];
      err += norm(sub(got, rigid)) ** 2; ref += norm(rigid) ** 2;
    }
    const rel = Math.sqrt(err / ref);
    assert.ok(onTop > N / 3, `${onTop} vertices on the plate's top`);
    t.diagnostic(`plate: ${N} vertices (${onTop} on the top), ${STILL + MOVING} steps, ${ms.toFixed(1)} ms/step with readback; min vertex phi ${worstPhi.toExponential(2)} m, faces (exact) phi >= ${faceLower.toExponential(2)} m; velocity vs rigid ${(100 * rel).toFixed(2)}% rms; ${redoSteps} steps redone, ${conflictsAll} conflicts; impact zones in ${zoneSteps} steps, ${zoneAll} vertices carried in all, largest ${zonePeak}; round cap hit in ${capHitSteps} steps; plane records peak ${recordPeak} (${(recordPeak / N).toFixed(1)} per vertex)`);
    assert.ok(rel < 0.05, `the sheet moves with the plate: ${(100 * rel).toFixed(2)}% rms from its rigid velocity`);
    solver.destroy(); det.destroy();
  });
});

test("GPU moving plate 3r over a sheet on the floor: the gap is wider than the cloth, so the sheet stays put while the plate comes down and slides", { skip: !gpu }, async t => {
  await withDevice(async device => {
    const DT = 1 / 60, RQ_MAX = 0.04, R_C = 2e-3, DOWN = 10, SLIDE = 30, GAP = 3 * R_C, START = 8 * R_C;
    const half: V3 = [0.15, 0.01, 0.15], band = R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005;
    const plate = buildClothSolidField({
      parts: [{ kind: "sdf", sdf: boxSdf([-half[0], -half[1], -half[2]], half), lo: [-half[0], -half[1], -half[2]], hi: half, mu: 0.5 }],
      lo: [-0.25, -0.13, -0.25], hi: [0.25, 0.13, 0.25], cell: 0.01, band });
    const floor = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [-0.4, -0.1, -0.4], hi: [0.4, 0, 0.4], mu: 0.5 }], lo: [-0.5, -0.2, -0.5], hi: [0.5, 0.1, 0.5], cell: 0.01, band });
    // The plate's bottom comes down from 8 r to 3 r over the floor's top (y = 0), then slides in x at 0.2 m/s.
    const poseAt = (step: number): OgcBodyPose => {
      const down = Math.min(step, DOWN) / DOWN, slide = Math.max(0, step - DOWN) * DT * 0.2;
      return { position: [slide, half[1] + START + (GAP - START) * down, 0], orientation: [0, 0, 0, 1] };
    };
    const n = 21, W = 0.2;
    const data = clothGridData({ columns: n, rows: n, width_m: W, height_m: W, origin_m: [-0.5 * W, 0.9 * R_C, -0.5 * W] });
    const topology = prepareClothTopology(data, [cotton]);
    const m = buildClothSolverModel(topology, [cotton], []);
    const T = m.topology.faceCount, N = m.topology.vertexCount, E = m.topology.edgeCount;
    const triangles = new Uint32Array(4 * T);
    for (let f = 0; f < T; f++) triangles.set(m.topology.faces.subarray(3 * f, 3 * f + 3), 4 * f);
    const rest = new Float32Array(4 * N);
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) rest[4 * i + k] = m.topology.data.restPositions_m[3 * i + k]!;
    const config: OgcContactConfig = { r: R_C, kc: 100, mu: 0.2, epsV: 1e-2, dt: DT, gravity: [0, -9.81, 0], rqBase: 3e-3, rqMax: RQ_MAX, envelopeScale: 2.5,
      contactCapacity: 128, pairCapacity: 840 * N, gridCapacity: 32 * (N + T + E), bucketCount: 1 << 16, roundCap: 4, prediction: "guess", newton: true, planeRecordCapacity: 32 * N,
      solidCapacityWords: floor.words.length, solidSampleCapacity: 8192, body: { capacityWords: plate.words.length, sampleCapacity: 8192, poseCapacity: 2 } };
    const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges: m.topology.edges }, rest, config);
    const solver = await ClothGPUSolver.create(device, m, { origin_m: [0, 0, 0], dt_s: DT, gravity_m_s2: [0, -9.81, 0], integrator: "bdf2", newton: CLOTH_NEWTON_SCENE_DEFAULTS,
      diagnostics: true, label: "gap", contact: { ...det.solverContact(CLOTH_BINDING.contact, true), alpha: det.alpha, state: det.contactState }, hooks: { afterFrame: det.stepHooks.afterFrame, filter: det.stepHooks.filter, afterNewtonStep: det.stepHooks.afterNewtonStep, stepEnd: det.stepHooks.stepEnd } });
    const stride = solver.tables.stride;
    det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: { buffer: solver.state, offset: 16 * CLOTH_STATE.xt * stride, size: 16 * stride }, ...(det.config.prediction === "target" ? { target: solver.inertialTarget } : {}), direction: solver.newtonDirection });
    det.setSolidField(floor); det.setBodyField(plate);
    const FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration
      | OGC_STATUS.bodySqueeze | OGC_STATUS.heldConflict | OGC_STATUS.bodyPose | OGC_STATUS.planeRecordOverflow;
    const x0 = await solver.readPositions();
    let worstFloor = Infinity, worstPlate = Infinity, drift = 0, bodyCuts = 0;
    for (let step = 0; step < DOWN + SLIDE; step++) {
      det.setBodyPoses([poseAt(step), poseAt(step + 1)]);
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      det.encodeFrameBegin(pass); solver.encodeStep(pass); det.encodeIntersectionCount(pass); pass.end(); device.queue.submit([enc.finish()]);
      const c = await det.readCounters();
      assert.equal(c.status & FAILING, 0, `step ${step}: status ${OgcContactDetector.describeStatus(c.status)}`);
      assert.ok(c.intersections === 0 && c.solidIntersections === 0, `step ${step}: ${c.intersections} face pairs, ${c.solidIntersections} faces in a solid`);
      bodyCuts += c.bodyCut;
      const end = poseOf(poseAt(step + 1)), x = await solver.readPositions();
      for (let i = 0; i < N; i++) {
        const p: V3 = [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!];
        worstFloor = Math.min(worstFloor, sampleClothSolidField(floor, p).phi);
        worstPlate = Math.min(worstPlate, sampleClothSolidField(plate, toBody(end, p)).phi);
        drift = Math.max(drift, Math.hypot(p[0] - x0[3 * i]!, p[2] - x0[3 * i + 2]!));
      }
    }
    t.diagnostic(`gap 3 r: ${N} vertices, ${DOWN + SLIDE} steps; min phi floor ${worstFloor.toExponential(2)} m, plate ${worstPlate.toExponential(2)} m; largest sideways drift ${drift.toExponential(2)} m; ${bodyCuts} plate cuts`);
    assert.ok(worstFloor > 0 && worstPlate > 0, "no vertex inside either solid");
    assert.ok(drift < R_C, `the plate drags the sheet ${drift} m sideways without touching it`);
    solver.destroy(); det.destroy();
  });
});
