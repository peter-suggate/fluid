import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { OCTREE_GPU_FACE_INCIDENCE_PER_ROW, type OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";
import type { OctreePowerFaceSource } from "../lib/webgpu-octree-power-faces";
import {
  OCTREE_POWER_FACE_SEED_VALID,
  WebGPUOctreePowerFaceSeed,
  octreePowerFaceSeedShader,
  planOctreePowerFaceSeed,
} from "../lib/webgpu-octree-power-face-seed";

test("power-face seed planner is bounded by compact row and face capacities", () => {
  assert.deepEqual(planOctreePowerFaceSeed(100, 800, 400), {
    rowCapacity: 100, faceCapacity: 800, axisFaceCapacity: 400,
    velocityBytes: 1_600, rowStatusBytes: 400, axisVelocityBytes: 1_600,
    allocatedBytes: 1_600 + 400 + 1_600 + 64 + 16,
  });
  assert.match(octreePowerFaceSeedShader, /weighted\[axis\]\+=face\.area\*face\.normalVelocity/);
  assert.match(octreePowerFaceSeedShader, /powerFaces\[index\]\.normalVelocity=normalVelocity/);
  assert.equal(OCTREE_GPU_FACE_INCIDENCE_PER_ROW, 48);
  assert.match(WebGPUOctreePowerFaceSeed.toString(), /OCTREE_GPU_FACE_INCIDENCE_PER_ROW/,
    "the power bridge must index the axis incidence slab with the mirror ABI stride");
  assert.match(octreePowerFaceSeedShader, /determinant<=1e-3\*trace\*trace\*trace/,
    "ill-conditioned power-to-axis least squares must preserve the axis rollback");
  assert.match(octreePowerFaceSeedShader, /length\(velocity\)>max\(1e-4,4\.0\*maxNormalSpeed\)/,
    "reverse publication must reject least-squares amplification instead of clamping it");
  assert.match(octreePowerFaceSeedShader, /face\.negativeRow==INVALID&&face\.positiveRow==INVALID/,
    "reverse publication must accept either canonical one-sided axis-face orientation and reject only an empty face");
});

test("Dawn seeds a non-axis power face from transferred compact face velocity", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-face seed checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create(["backend=metal"]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength)
      .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer;
  };
  const axisControl = upload(new Uint32Array([6, 0, 6, 1, 6, 0]));
  const axisWords = new Uint32Array(6 * 6); const axisFloats = new Float32Array(axisWords.buffer);
  [2, 2, 4, 4, 6, 6].forEach((velocity, face) => {
    const at = face * 6; axisWords[at] = 0; axisWords[at + 1] = 0xffff_ffff;
    axisWords[at + 3] = Math.floor(face / 2); axisFloats[at + 4] = velocity; axisFloats[at + 5] = 1;
  });
  // The compact mirror canonically emits both one-sided orientations.  Keep a
  // negative-boundary face in this round-trip fixture so reverse publication
  // cannot accidentally assume that negativeRow is always live.
  axisWords[6] = 0xffff_ffff; axisWords[7] = 0;
  const axisFaces = upload(axisWords); const incidenceWords = new Uint32Array(25); incidenceWords[0] = 6;
  incidenceWords.set([0, 1, 2, 3, 4, 5], 1); const axisIncidence = upload(incidenceWords);
  const placeholder = upload(new Uint32Array([0]));
  const axis: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 1, faceCapacity: 6, faceBytes: 144, incidenceBytes: 100, offsetBytes: 8, allocatedBytes: 252 },
    control: axisControl, faces: axisFaces, incidence: axisIncidence, parity: placeholder,
  };
  const powerControlWords = new Uint32Array(16); powerControlWords[0] = 1; powerControlWords[1] = 4;
  powerControlWords[7] = 9; powerControlWords[8] = 0x8000_0000;
  const powerControl = upload(powerControlWords); const powerFaceWords = new Uint32Array(4 * 8);
  const powerFaceFloats = new Float32Array(powerFaceWords.buffer);
  for (let face = 0; face < 4; face += 1) {
    powerFaceWords[face * 8] = 0; powerFaceWords[face * 8 + 1] = 0xffff_ffff; powerFaceFloats[face * 8 + 5] = 1;
  }
  const powerFaces = upload(powerFaceWords); const inverseRootTwo = Math.SQRT1_2;
  const inverseRootThree = 1 / Math.sqrt(3);
  const normalValues = new Float32Array([
    inverseRootTwo, inverseRootTwo, 0, 0,
    1, 0, 0, 0,
    0, 1, 0, 0,
    inverseRootThree, inverseRootThree, inverseRootThree, 0,
  ]);
  const normals = upload(normalValues);
  const powerRowsWords = new Uint32Array(8); powerRowsWords[7] = 4; const powerRows = upload(powerRowsWords);
  const powerIncidenceWords = new Uint32Array(8);
  for (let face = 0; face < 4; face += 1) { powerIncidenceWords[face * 2] = face; powerIncidenceWords[face * 2 + 1] = 1; }
  const powerIncidence = upload(powerIncidenceWords);
  const power: OctreePowerFaceSource = {
    plan: { rowCapacity: 1, faceCapacity: 4, incidenceCapacity: 4, faceBytes: 128, normalBytes: 64, centroidBytes: 64, quadratureBytes: 320,
      incidenceBytes: 32, workspaceBytes: 32, hashCapacity: 2, hashBytes: 32, scanBlockCount: 1,
      scanBytes: 16, maximumHashProbes: 32, allocatedBytes: 0 },
    faces: powerFaces, faceNormals: normals, faceCentroids: placeholder, faceQuadrature: placeholder, incidenceRows: powerRows, incidenceOffsets: powerRows,
    incidence: powerIncidence, control: powerControl, siteIndex: placeholder,
  };
  const seed = new WebGPUOctreePowerFaceSeed(device, axis, power);
  const readback = device.createBuffer({ size: 160, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); seed.encode(encoder);
  encoder.copyBufferToBuffer(seed.control, 0, readback, 0, 32);
  encoder.copyBufferToBuffer(powerFaces, 0, readback, 32, 128);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
  const result = readback.getMappedRange().slice(0); readback.unmap();
  const control = new Uint32Array(result, 0, 8); const face = new Float32Array(result, 32, 8);
  assert.equal(control[0], 0); assert.equal(control[2], 1); assert.equal(control[3], 4);
  assert.equal(control[4], 4); assert.equal(control[5], 9); assert.equal(control[6], OCTREE_POWER_FACE_SEED_VALID);
  assert.ok(Math.abs(face[4] - 6 * Math.SQRT1_2) < 2e-5);
  // Simulate a successfully projected power field for v=(3,5,7), then
  // conservatively republish its least-squares reconstruction to axis faces.
  const target: readonly [number, number, number] = [3, 5, 7];
  for (let index = 0; index < 4; index += 1) {
    const normal = normalValues.subarray(index * 4, index * 4 + 3);
    device.queue.writeBuffer(powerFaces, index * 32 + 16,
      new Float32Array([target[0] * normal[0] + target[1] * normal[1] + target[2] * normal[2]]));
  }
  const operatorControl = upload(new Uint32Array([0xc000_0000, 0xffff_ffff, 1, 4, 4, 0, 4, 0]));
  const reverseReadback = device.createBuffer({ size: 32 + axisWords.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const reverse = device.createCommandEncoder(); seed.encodePowerToAxis(reverse, operatorControl);
  reverse.copyBufferToBuffer(seed.control, 0, reverseReadback, 0, 32);
  reverse.copyBufferToBuffer(axisFaces, 0, reverseReadback, 32, axisWords.byteLength);
  device.queue.submit([reverse.finish()]); await device.queue.onSubmittedWorkDone(); await reverseReadback.mapAsync(GPUMapMode.READ);
  const reverseResult = reverseReadback.getMappedRange().slice(0); reverseReadback.unmap();
  const reverseControl = new Uint32Array(reverseResult, 0, 8);
  const axisResult = new Float32Array(reverseResult, 32, axisWords.length);
  assert.equal(reverseControl[0], 0); assert.equal(reverseControl[4], 6);
  assert.equal(reverseControl[6], OCTREE_POWER_FACE_SEED_VALID);
  [3, 3, 5, 5, 7, 7].forEach((expected, index) => assert.ok(Math.abs(axisResult[index * 6 + 4] - expected) < 2e-5));
  seed.destroy(); readback.destroy(); axisControl.destroy(); axisFaces.destroy(); axisIncidence.destroy();
  reverseReadback.destroy(); operatorControl.destroy(); powerControl.destroy(); powerFaces.destroy(); normals.destroy();
  powerRows.destroy(); powerIncidence.destroy(); placeholder.destroy(); device.destroy();
});
