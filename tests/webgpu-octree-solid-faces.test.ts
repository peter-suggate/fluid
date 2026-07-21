import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import type { OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";
import {
  OCTREE_SOLID_DIAGNOSTIC_BYTES,
  OCTREE_SOLID_FACE_RECORD_BYTES,
  OCTREE_SOLID_IMPULSE_WORDS_PER_BODY,
  constrainOctreeFaceVelocity,
  octreeSolidFaceShader,
  octreeSolidExchangeShader,
  planOctreeSolidFaces,
  WebGPUOctreeSolidFaces,
} from "../lib/webgpu-octree-solid-faces";

test("adaptive solid-face storage scales with canonical faces, not domain cells", () => {
  const plan = planOctreeSolidFaces(4096);
  assert.equal(plan.apertureBytes, 4096 * 16);
  assert.equal(plan.impulseBytes, 12 * 8 * 4);
  assert.equal(plan.diagnosticBytes, 48);
  assert.equal(plan.allocatedBytes, plan.apertureBytes + plan.impulseBytes + plan.diagnosticBytes);
  assert.equal(OCTREE_SOLID_FACE_RECORD_BYTES, 16);
  assert.equal(OCTREE_SOLID_IMPULSE_WORDS_PER_BODY, 8);
  assert.equal(OCTREE_SOLID_DIAGNOSTIC_BYTES, 48);
  assert.throws(() => planOctreeSolidFaces(0), /positive/);
  assert.throws(() => planOctreeSolidFaces(1, 13), /between/);
  assert.equal(constrainOctreeFaceVelocity(4, 0.75, 0), 3);
  assert.equal(constrainOctreeFaceVelocity(4, 0, -2), -2);
});

test("adaptive solid shader owns aperture, velocity, and paired impulse operators", () => {
  assert.match(octreeSolidFaceShader, /struct ApertureRecord \{ openFraction:f32, solidNormalVelocity:f32, dominantOwner:i32, sampleMask:u32 \}/);
  assert.match(octreeSolidFaceShader, /fn bodySdf/);
  assert.match(octreeSolidFaceShader, /fn worldFaceSample/);
  assert.match(octreeSolidFaceShader, /body\.linearVelocity\.xyz\+cross\(body\.angularVelocity\.xyz/);
  assert.match(octreeSolidFaceShader, /aperture\.openFraction\*face\.normalVelocity\+\(1\.0-aperture\.openFraction\)\*aperture\.solidNormalVelocity/);
  assert.match(octreeSolidFaceShader, /pressureAt\(face\.negativeRow\)-pressureAt\(face\.positiveRow\)/);
  assert.match(octreeSolidFaceShader, /checkedFixed\(-linear\.x\)/, "fluid diagnostic must be the opposite body impulse");
  assert.match(octreeSolidFaceShader, /faceControl\[1\]!=0u\|\|faceControl\[0\]>faceControl\[2\]/, "upstream face overflow must fail closed");
  assert.match(octreeSolidFaceShader, /params\.control\.w>12u/);
  assert.doesNotMatch(octreeSolidFaceShader, /texture_3d|solidCells/, "solid classification must not depend on a box-sized field");
  assert.match(octreeSolidExchangeShader, /body\*12u\+word/);
});

test("Dawn classifies an adaptive cut face and conserves its pressure reaction", async (t) => {
  const runtime = process.env.WEBGPU_NODE_MODULE;
  if (!runtime) { t.skip("set WEBGPU_NODE_MODULE for GPU solid-face checks"); return; }
  let dawn: { create(options: string[]): GPU; globals: Record<string, unknown> };
  try {
    dawn = await import(pathToFileURL(runtime).href) as typeof dawn;
  } catch {
    t.skip("local Dawn runtime is unavailable");
    return;
  }
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const control = device.createBuffer({ size: 16, usage: storage });
  const faces = device.createBuffer({ size: 4 * 24, usage: storage });
  const incidence = device.createBuffer({ size: 4, usage: storage });
  const parity = device.createBuffer({ size: 16, usage: storage });
  device.queue.writeBuffer(control, 0, new Uint32Array([1, 0, 4, 2]));
  const face = new ArrayBuffer(24); const faceU32 = new Uint32Array(face); const faceF32 = new Float32Array(face);
  faceU32.set([0, 1, 1, 2 << 2]); // x-normal face at x=1, covering a 2x2 fragment
  faceF32[4] = 0; faceF32[5] = 4;
  device.queue.writeBuffer(faces, 0, face);
  const source: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 2, faceCapacity: 4, faceBytes: 96, incidenceBytes: 4, allocatedBytes: 132 },
    control, faces, incidence, parity,
  };
  const bodies = device.createBuffer({ size: 12 * 128, usage: storage });
  const body = new Float32Array(32);
  body.set([0, 1, 0, 0], 0); // sphere centered on the face
  body.set([0.75, 0, 0, 0.75], 4);
  body.set([1, 0, 0, 0], 8);
  body.set([3, 0, 0, 0], 12);
  device.queue.writeBuffer(bodies, 0, body);
  const params = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const paramsBytes = new ArrayBuffer(160); const paramsF32 = new Float32Array(paramsBytes); const paramsU32 = new Uint32Array(paramsBytes);
  paramsU32.set([2, 2, 2, 0], 0);
  paramsF32.set([1, 1, 1, 0], 4);
  paramsU32.set([0, 0, 0, 1], 8);
  paramsF32.set([2, 2, 2, 0], 16);
  paramsF32.set([1, 0, 0, 0], 28);
  device.queue.writeBuffer(params, 0, paramsBytes);
  const pressureA = device.createBuffer({ size: 8, usage: storage });
  const pressureB = device.createBuffer({ size: 8, usage: storage });
  const rigidExchange = device.createBuffer({ size: 12 * 12 * 4, usage: storage });
  device.queue.writeBuffer(pressureA, 0, new Float32Array([2, 1]));
  await device.queue.onSubmittedWorkDone();
  const solids = new WebGPUOctreeSolidFaces(device, { faces: source, rigidBodies: bodies, rigidExchange, params, pressureA, pressureB });
  const encoder = device.createCommandEncoder();
  solids.encode(encoder, true);
  const apertureReadback = device.createBuffer({ size: solids.plan.apertureBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const impulseReadback = device.createBuffer({ size: solids.plan.impulseBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const diagnosticReadback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const controlReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const exchangeReadback = device.createBuffer({ size: 12 * 12 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const faceReadback = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(solids.apertures, 0, apertureReadback, 0, solids.plan.apertureBytes);
  encoder.copyBufferToBuffer(solids.bodyImpulses, 0, impulseReadback, 0, solids.plan.impulseBytes);
  encoder.copyBufferToBuffer(solids.diagnostics, 0, diagnosticReadback, 0, 48);
  encoder.copyBufferToBuffer(control, 0, controlReadback, 0, 16);
  encoder.copyBufferToBuffer(rigidExchange, 0, exchangeReadback, 0, 12 * 12 * 4);
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, 24);
  device.queue.submit([encoder.finish()]);
  await Promise.all([apertureReadback.mapAsync(GPUMapMode.READ), impulseReadback.mapAsync(GPUMapMode.READ), diagnosticReadback.mapAsync(GPUMapMode.READ), controlReadback.mapAsync(GPUMapMode.READ), exchangeReadback.mapAsync(GPUMapMode.READ), faceReadback.mapAsync(GPUMapMode.READ)]);
  const apertureBytes = apertureReadback.getMappedRange().slice(0); const apertureF32 = new Float32Array(apertureBytes); const apertureI32 = new Int32Array(apertureBytes);
  const diagnosticBytes = diagnosticReadback.getMappedRange().slice(0);
  const controlBytes = controlReadback.getMappedRange().slice(0);
  assert.ok(Math.abs(apertureF32[0] - 0.75) < 1e-6, `aperture=${apertureF32[0]}, control=${[...new Uint32Array(controlBytes)].join(",")}, diagnostics=${[...new Uint32Array(diagnosticBytes)].join(",")}, errors=${errors.join(" | ")}`);
  assert.ok(Math.abs(apertureF32[1] - 3) < 1e-6);
  assert.equal(apertureI32[2], 0);
  assert.equal(new Uint32Array(apertureBytes)[3], 0x0660);
  const impulses = new Int32Array(impulseReadback.getMappedRange().slice(0));
  assert.equal(impulses[0], 1_000_000);
  assert.equal(impulses[1], 0); assert.equal(impulses[2], 0);
  const diagnostics = new Uint32Array(diagnosticBytes); const signedDiagnostics = new Int32Array(diagnosticBytes);
  assert.equal(diagnostics[0], 0); assert.equal(diagnostics[1], 1); assert.equal(diagnostics[2], 1); assert.equal(diagnostics[3], 4);
  assert.equal(signedDiagnostics[4], 1_000_000); assert.equal(signedDiagnostics[7], -1_000_000);
  assert.equal(signedDiagnostics[4] + signedDiagnostics[7], 0);
  assert.equal(new Int32Array(exchangeReadback.getMappedRange())[0], 1_000_000);
  assert.ok(Math.abs(new Float32Array(faceReadback.getMappedRange())[4] - 0.75) < 1e-6, "face flux must blend 75% fluid velocity with 25% solid velocity");
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  apertureReadback.unmap(); impulseReadback.unmap(); diagnosticReadback.unmap(); controlReadback.unmap(); exchangeReadback.unmap(); faceReadback.unmap();
  apertureReadback.destroy(); impulseReadback.destroy(); diagnosticReadback.destroy(); controlReadback.destroy(); exchangeReadback.destroy(); faceReadback.destroy(); solids.destroy();
  control.destroy(); faces.destroy(); incidence.destroy(); parity.destroy(); bodies.destroy(); params.destroy(); pressureA.destroy(); pressureB.destroy(); rigidExchange.destroy(); device.destroy();
});
