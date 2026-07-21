import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { OCTREE_POWER_FACE_VALID, type OctreePowerFaceSource } from "../lib/webgpu-octree-power-faces";
import { OCTREE_SOLID_VERTEX_SDF_VALID } from "../lib/webgpu-octree-solid-vertex-sdf";
import {
  OCTREE_POWER_SOLID_VALID,
  WebGPUOctreePowerSolidFaces,
  octreePowerSolidFaceShader,
  octreePowerSolidImpulseShader,
  planOctreePowerSolidFaces,
} from "../lib/webgpu-octree-power-solid-faces";

const retainedNativeGPUs: GPU[] = [];
const retainedDevices: GPUDevice[] = [];

test("generalized solid-face planner is compact-capacity-scaled", () => {
  const plan = planOctreePowerSolidFaces(100, 3);
  assert.equal(plan.apertureBytes, 1_600);
  assert.equal(plan.impulseBytes, 96);
  assert.equal(plan.allocatedBytes, 1_600 + 96 + 64 + 64);
  assert.throws(() => planOctreePowerSolidFaces(0), /positive integer/);
});

test("Dawn classifies oblique rigid and terrain apertures and publishes paired reaction", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for generalized solid-face checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); retainedNativeGPUs.push(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(); retainedDevices.push(device);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  for (const code of [octreePowerSolidFaceShader, octreePowerSolidImpulseShader]) {
    const info = await device.createShaderModule({ code }).getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error"), []);
  }
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength)
      .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const controlWords = new Uint32Array(16); controlWords[0] = 1; controlWords[1] = 1;
  controlWords[7] = 7; controlWords[8] = OCTREE_POWER_FACE_VALID;
  const faceWords = new Uint32Array(8); const faceFloats = new Float32Array(faceWords.buffer);
  faceWords[1] = 0xffff_ffff; faceWords[3] = OCTREE_POWER_FACE_VALID;
  faceFloats[4] = 4; faceFloats[5] = 1; faceFloats[6] = 1; faceFloats[7] = 1;
  const root2 = Math.sqrt(0.5);
  const faceControl = upload(controlWords); const faces = upload(faceWords);
  const normals = upload(new Float32Array([root2, root2, 0, 0]));
  const centroids = upload(new Float32Array([1, 1, 1, 0]));
  const halfBits = (value: number) => {
    const magnitude = Math.abs(value);
    const bits = magnitude === 0.125 ? 0x3000 : magnitude === 0.375 ? 0x3600 : 0;
    return bits | (value < 0 ? 0x8000 : 0);
  };
  const quadratureWords = new Uint32Array(20);
  new Float32Array(quadratureWords.buffer).set([1, 1, 1, 1], 0);
  for (let sample = 0; sample < 16; sample += 1) {
    const u = (sample % 4 + 0.5) / 4 - 0.5;
    const v = (Math.floor(sample / 4) + 0.5) / 4 - 0.5;
    quadratureWords[4 + sample] = (halfBits(u) | (halfBits(v) << 16)) >>> 0;
  }
  const quadrature = upload(quadratureWords);
  const placeholder = upload(new Uint32Array(4));
  const source: OctreePowerFaceSource = {
    plan: { rowCapacity: 1, faceCapacity: 1, incidenceCapacity: 2, faceBytes: 32, normalBytes: 16,
      centroidBytes: 16, quadratureBytes: 80, incidenceBytes: 16, workspaceBytes: 32, boundaryQueryBytes: 32, hashCapacity: 2, hashBytes: 32,
      maximumHashProbes: 32, scanBlockCount: 1, scanBytes: 16, allocatedBytes: 0 },
    faces, faceNormals: normals, faceCentroids: centroids, faceQuadrature: quadrature, incidenceRows: placeholder,
    incidenceOffsets: placeholder, incidence: placeholder, control: faceControl, siteIndex: placeholder,
    boundaryPhiQueries: placeholder,
  };
  const bodyWords = new Uint32Array(12 * 32); const bodyFloats = new Float32Array(bodyWords.buffer);
  bodyFloats.set([0, 1, 0, 0], 0); // sphere centered at the physical face centroid after x/z recentering
  bodyFloats.set([0.45, 0.45, 0.45, 0], 4);
  bodyFloats.set([1, 0, 0, 0], 8);
  bodyFloats.set([2, 4, 0, 0], 12);
  const bodies = upload(bodyWords);
  const pressures = upload(new Float32Array([2, 0]));
  const solidVertexArenaWords = new Uint32Array(24);
  const solidVertexControlWords = solidVertexArenaWords.subarray(0, 16);
  solidVertexControlWords.set([0, 1, 1, 8, 7, OCTREE_SOLID_VERTEX_SDF_VALID, 0xffff_ffff, 1]);
  solidVertexControlWords[8] = 7;
  new Float32Array(solidVertexArenaWords.buffer, 64, 8).set([-1, -1, 1, 1, -1, -1, 1, 1]);
  const solidVertexArena = upload(solidVertexArenaWords);
  const terrain = device.createTexture({ size: [2, 2], format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: terrain }, new Float32Array([1, 1, 1, 1]),
    { bytesPerRow: 8, rowsPerImage: 2 }, [2, 2]);
  const stage = new WebGPUOctreePowerSolidFaces(device, {
    faces: source, rigidBodies: bodies, terrain, pressureA: pressures, pressureB: pressures,
    solidVertices: { plan: { rowCapacity: 1, sampleCapacity: 8, sdfBytes: 32, arenaBytes: 96, allocatedBytes: 160 },
      arena: solidVertexArena },
  }, 1);
  const read = async (bodyCount: number, terrainEnabled: boolean) => {
    const bytes = 64 + 16 + 32 + stage.plan.impulseBytes;
    const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    stage.encodeClassifyAndConstrain(encoder, { dimensions: [2, 2, 2], physicalSpacing: [1, 1, 1],
      container: [2, 2, 2], rigidBodyCount: bodyCount, terrainEnabled, pressureImpulseScale: 1 });
    stage.encodePressureImpulses(encoder, true);
    let offset = 0; encoder.copyBufferToBuffer(stage.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(stage.apertures, 0, readback, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(faces, 0, readback, offset, 32); offset += 32;
    encoder.copyBufferToBuffer(stage.bodyImpulses, 0, readback, offset, stage.plan.impulseBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0);
    readback.unmap(); readback.destroy(); return result;
  };

  const rigid = await read(1, false);
  const rigidControl = new Uint32Array(rigid, 0, 16); const rigidSigned = new Int32Array(rigid, 0, 16);
  assert.deepEqual(Array.from(rigidControl.slice(0, 8)), [0, 1, 1, 12, 1, 1, 0, OCTREE_POWER_SOLID_VALID]);
  assert.deepEqual(Array.from(rigidSigned.slice(8, 14)), [1_060_656, 1_060_656, 0, -1_060_656, -1_060_656, 0]);
  const aperture = new Float32Array(rigid, 64, 4);
  assert.equal(aperture[0], 0.25); assert.ok(Math.abs(aperture[1] - 6 * root2) < 1e-5);
  assert.equal(new Int32Array(rigid, 64, 4)[2], 0); assert.equal(new Uint32Array(rigid, 64, 4)[3], 0x6ff6);
  const constrained = new Float32Array(rigid, 80, 8);
  assert.ok(Math.abs(constrained[4] - (0.25 * 4 + 0.75 * 6 * root2)) < 1e-5);

  device.queue.writeBuffer(faces, 16, new Float32Array([4, 1, 1, 1]));
  const terrainOnly = await read(0, true);
  const terrainControl = new Uint32Array(terrainOnly, 0, 16);
  assert.deepEqual(Array.from(terrainControl.slice(0, 8)), [0, 1, 1, 8, 1, 0, 1, OCTREE_POWER_SOLID_VALID]);
  const terrainAperture = new Float32Array(terrainOnly, 64, 4);
  assert.equal(terrainAperture[0], 0.5); assert.equal(terrainAperture[1], 0);
  assert.equal(new Int32Array(terrainOnly, 64, 4)[2], -2);
  assert.deepEqual(errors, []);

  stage.destroy(); terrain.destroy(); faceControl.destroy(); faces.destroy(); normals.destroy(); centroids.destroy(); quadrature.destroy();
  placeholder.destroy(); bodies.destroy(); pressures.destroy(); solidVertexArena.destroy();
  await device.queue.onSubmittedWorkDone(); device.destroy(); retainedDevices.length = 0;
});

test("solid aperture shaders consume clipped power-polygon quadrature, never an equivalent square", () => {
  assert.match(octreePowerSolidFaceShader, /var<storage,read> quadrature:array<FaceQuadrature>/);
  assert.match(octreePowerSolidFaceShader, /unpack2x16float\(q\.sampleUV\[sample\]\)/);
  assert.match(octreePowerSolidImpulseShader, /unpack2x16float\(q\.sampleUV\[sample\]\)/);
  assert.match(octreePowerSolidFaceShader, /source\[4\]==faceControl\[7\]/,
    "terrain vertex SDF must match the live power generation");
  assert.match(octreePowerSolidFaceShader, /source\[8\]==faceControl\[7\]/,
    "terrain authority must retain a same-generation compact-face rollback seed");
  assert.doesNotMatch(`${octreePowerSolidFaceShader}\n${octreePowerSolidImpulseShader}`, /SAMPLE_AXIS|equivalent area square|sample%/);
});
