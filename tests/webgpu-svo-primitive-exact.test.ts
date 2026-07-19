import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  SVO_PRIMITIVE_RECORD_WORDS,
  intersectSvoPrimitive,
  packSvoPrimitiveRecords,
  svoPrimitiveWGSL,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRay,
} from "../lib/svo-primitive-abi";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const identity = { w: 1, x: 0, y: 0, z: 0 };
const quarterTurn = { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 };

interface OracleCase {
  name: string;
  primitive: SvoFinitePrimitiveDescriptor;
  ray: SvoPrimitiveRay;
  expectedStatus?: 0 | 1 | 2;
  corruptDimensions?: boolean;
}

const cases: OracleCase[] = [
  {
    name: "sphere scaled direction",
    primitive: { kind: "sphere", primitiveId: 1, materialId: 16, center_m: { x: 1, y: 0, z: 0 }, radius_m: 2 },
    ray: { origin_m: { x: -4, y: 0, z: 0 }, direction: { x: 20, y: 0, z: 0 } },
  },
  {
    name: "sphere tangent",
    primitive: { kind: "sphere", primitiveId: 2, materialId: 16, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    ray: { origin_m: { x: -2, y: 1, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
  },
  {
    name: "sphere grazing miss",
    primitive: { kind: "sphere", primitiveId: 3, materialId: 16, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    ray: { origin_m: { x: -2, y: 1.001, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
    expectedStatus: 0,
  },
  {
    name: "oriented box face",
    primitive: {
      kind: "box", primitiveId: 4, materialId: 17, center_m: { x: 0, y: 0, z: 0 },
      halfExtents_m: { x: 2, y: 1, z: 1 }, orientation: quarterTurn,
    },
    ray: { origin_m: { x: 0, y: -4, z: 0 }, direction: { x: 0, y: 3, z: 0 } },
  },
  {
    name: "box corner tie",
    primitive: {
      kind: "box", primitiveId: 5, materialId: 17, center_m: { x: 0, y: 0, z: 0 },
      halfExtents_m: { x: 1, y: 1, z: 1 }, orientation: identity,
    },
    ray: { origin_m: { x: 2, y: 2, z: 2 }, direction: { x: -1, y: -1, z: -1 } },
  },
  {
    name: "box inside exit",
    primitive: {
      kind: "box", primitiveId: 6, materialId: 17, center_m: { x: 0, y: 0, z: 0 },
      halfExtents_m: { x: 1, y: 1, z: 1 }, orientation: identity,
    },
    ray: { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: -1, z: 0 } },
  },
  {
    name: "capsule barrel",
    primitive: {
      kind: "capsule", primitiveId: 7, materialId: 18, center_m: { x: 0, y: 0, z: 0 },
      radius_m: 0.5, segmentHalfLength_m: 1, orientation: identity,
    },
    ray: { origin_m: { x: -2, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
  },
  {
    name: "capsule cap tangent",
    primitive: {
      kind: "capsule", primitiveId: 8, materialId: 18, center_m: { x: 0, y: 0, z: 0 },
      radius_m: 0.5, segmentHalfLength_m: 1, orientation: identity,
    },
    ray: { origin_m: { x: -2, y: 1.5, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
  },
  {
    name: "cylinder rim cap tie",
    primitive: {
      kind: "cylinder", primitiveId: 9, materialId: 19, center_m: { x: 0, y: 0, z: 0 },
      radius_m: 1, halfHeight_m: 2, orientation: identity,
    },
    ray: { origin_m: { x: 1, y: 4, z: 0 }, direction: { x: 0, y: -1, z: 0 } },
  },
  {
    name: "rotated cylinder inside exit",
    primitive: {
      kind: "cylinder", primitiveId: 10, materialId: 19, center_m: { x: 0, y: 0, z: 0 },
      radius_m: 1, halfHeight_m: 2, orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: -Math.SQRT1_2 },
    },
    ray: { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: -1, y: 0, z: 0 } },
  },
  {
    name: "rotated ellipsoid",
    primitive: {
      kind: "ellipsoid", primitiveId: 11, materialId: 32, center_m: { x: 0, y: 0, z: 0 },
      radii_m: { x: 2, y: 1, z: 0.5 }, orientation: quarterTurn,
    },
    ray: { origin_m: { x: 0, y: 5, z: 0 }, direction: { x: 0, y: -9, z: 0 } },
  },
  {
    name: "ellipsoid tangent",
    primitive: {
      kind: "ellipsoid", primitiveId: 12, materialId: 32, center_m: { x: 0, y: 0, z: 0 },
      radii_m: { x: 2, y: 1, z: 0.5 }, orientation: identity,
    },
    ray: { origin_m: { x: -3, y: 1, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
  },
  {
    name: "ellipsoid inside exit",
    primitive: {
      kind: "ellipsoid", primitiveId: 13, materialId: 32, center_m: { x: 0, y: 0, z: 0 },
      radii_m: { x: 2, y: 1, z: 0.5 }, orientation: identity,
    },
    ray: { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 0, z: -4 } },
  },
  {
    name: "invalid zero ellipsoid radius",
    primitive: {
      kind: "ellipsoid", primitiveId: 14, materialId: 32, center_m: { x: 0, y: 0, z: 0 },
      radii_m: { x: 2, y: 1, z: 0.5 }, orientation: identity,
    },
    ray: { origin_m: { x: -3, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 } },
    expectedStatus: 2,
    corruptDimensions: true,
  },
];

const oracleShader = `${svoPrimitiveWGSL}
struct PrimitiveRayInput { originMin: vec4f, directionMax: vec4f }
@group(0) @binding(0) var<storage,read> oraclePrimitives: array<SvoPrimitiveRecord>;
@group(0) @binding(1) var<storage,read> oracleRays: array<PrimitiveRayInput>;
@group(0) @binding(2) var<storage,read_write> oracleResults: array<SvoPrimitiveRayResult>;
@compute @workgroup_size(1)
fn primitiveOracle(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&oracleResults)) { return; }
  let ray = oracleRays[id.x];
  oracleResults[id.x] = svoIntersectPrimitiveExact(
    oraclePrimitives[id.x], ray.originMin.xyz, ray.directionMax.xyz,
    ray.originMin.w, ray.directionMax.w,
  );
}`;

test("shared primitive ray WGSL exposes exact bounded status and normal results", () => {
  assert.match(svoPrimitiveWGSL, /fn svoIntersectPrimitiveExact/);
  assert.match(svoPrimitiveWGSL, /SVO_PRIMITIVE_RAY_INVALID/);
  assert.match(svoPrimitiveWGSL, /kind == SVO_KIND_CAPSULE/);
  assert.match(svoPrimitiveWGSL, /kind == SVO_KIND_ELLIPSOID/);
  assert.match(svoPrimitiveWGSL, /bestFeature = SVO_FEATURE_CYLINDER_CAP/);
});

test("real GPU exact primitive hits agree with the CPU oracle", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU primitive checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no WebGPU adapter");
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });

  const primitiveWords = packSvoPrimitiveRecords(cases.map((entry) => entry.primitive));
  const primitiveFloats = new Float32Array(primitiveWords.buffer);
  cases.forEach((entry, index) => {
    if (entry.corruptDimensions) primitiveFloats[index * SVO_PRIMITIVE_RECORD_WORDS + 4] = 0;
  });
  const rayFloats = new Float32Array(cases.length * 8);
  cases.forEach((entry, index) => {
    const base = index * 8;
    rayFloats.set([entry.ray.origin_m.x, entry.ray.origin_m.y, entry.ray.origin_m.z, entry.ray.tMin_m ?? 0], base);
    rayFloats.set([entry.ray.direction.x, entry.ray.direction.y, entry.ray.direction.z, entry.ray.tMax_m ?? 1_000], base + 4);
  });
  const resultBytes = cases.length * 32;
  const primitiveBuffer = device.createBuffer({ size: primitiveWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const rayBuffer = device.createBuffer({ size: rayFloats.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const resultBuffer = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(primitiveBuffer, 0, primitiveWords);
  device.queue.writeBuffer(rayBuffer, 0, rayFloats);

  const shaderModule = device.createShaderModule({ label: "Exact SVO primitive GPU oracle", code: oracleShader });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter((message) => message.type === "error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "primitiveOracle" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: primitiveBuffer } },
    { binding: 1, resource: { buffer: rayBuffer } },
    { binding: 2, resource: { buffer: resultBuffer } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(cases.length);
  pass.end();
  encoder.copyBufferToBuffer(resultBuffer, 0, readback, 0, resultBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copy = readback.getMappedRange().slice(0);
  const words = new Uint32Array(copy);
  const floats = new Float32Array(copy);

  cases.forEach((entry, index) => {
    const base = index * 8;
    const expected = entry.corruptDimensions ? null : intersectSvoPrimitive(entry.primitive, entry.ray);
    const expectedStatus = entry.expectedStatus ?? (expected ? 1 : 0);
    assert.equal(words[base + 2], expectedStatus, `${entry.name}: status`);
    if (!expected) return;
    assert.ok(Math.abs(floats[base] - expected.t_m) <= 3e-4, `${entry.name}: t ${floats[base]} vs ${expected.t_m}`);
    assert.equal(words[base + 1], expected.featureId, `${entry.name}: feature`);
    const gpuNormal = [floats[base + 4], floats[base + 5], floats[base + 6]];
    const cpuNormal = [expected.normal.x, expected.normal.y, expected.normal.z];
    for (let axis = 0; axis < 3; axis += 1) {
      assert.ok(Math.abs(gpuNormal[axis] - cpuNormal[axis]) <= 3e-4,
        `${entry.name}: normal axis ${axis} ${gpuNormal[axis]} vs ${cpuNormal[axis]}`);
    }
  });
  readback.unmap();
  assert.deepEqual(validationErrors, []);
  primitiveBuffer.destroy();
  rayBuffer.destroy();
  resultBuffer.destroy();
  readback.destroy();
  device.destroy();
});
