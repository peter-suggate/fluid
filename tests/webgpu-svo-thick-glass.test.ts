import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  packSvoThickGlassVolumes,
  querySvoThickGlassVolume,
  SVO_THICK_GLASS_RECORD_WORDS,
  SVO_THICK_GLASS_QUERY_STATUS,
  svoThickGlassWGSL,
  type SvoThickGlassRay,
  type SvoThickGlassVolume,
} from "../lib/svo-thick-glass";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const identity = { w: 1, x: 0, y: 0, z: 0 };

interface OracleCase {
  name: string;
  volume: SvoThickGlassVolume;
  ray: SvoThickGlassRay;
  expectedRevision: number;
  corruptRadius?: boolean;
}

const base: SvoThickGlassVolume = {
  glassId: 1, materialId: 7, ownerId: 9, revision: 4, shape: "sphere",
  center_m: [0, 0, 0], radii_m: [1, 1, 1], orientation: identity,
  indexOfRefraction: 1.5, absorption_mInv: [.8, .2, .1], surfaceEpsilon_m: 1e-5, maximumOpticalPath_m: 4,
};

const cases: OracleCase[] = [
  { name: "sphere outside", volume: base, ray: { origin_m: [-3, 0, 0], direction: [2, 0, 0], tMax_m: 6 }, expectedRevision: 4 },
  { name: "sphere inside", volume: base, ray: { origin_m: [0, 0, 0], direction: [1, 0, 0], tMax_m: 2 }, expectedRevision: 4 },
  { name: "sphere tangent", volume: base, ray: { origin_m: [-2, 1, 0], direction: [1, 0, 0], tMax_m: 4 }, expectedRevision: 4 },
  {
    name: "rotated ellipsoid",
    volume: { ...base, glassId: 2, shape: "ellipsoid", radii_m: [2, 1, .5], orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 } },
    ray: { origin_m: [0, 3, 0], direction: [0, -3, 0], tMax_m: 6 }, expectedRevision: 4,
  },
  { name: "miss", volume: base, ray: { origin_m: [-3, 2, 0], direction: [1, 0, 0], tMax_m: 6 }, expectedRevision: 4 },
  { name: "stale", volume: base, ray: { origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6 }, expectedRevision: 5 },
  { name: "invalid radius", volume: base, ray: { origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6 }, expectedRevision: 4, corruptRadius: true },
];

const oracleShader = `${svoThickGlassWGSL}
struct ThickGlassRayInput{originMin:vec4f,directionMax:vec4f,expectedRevision:vec4u}
struct ThickGlassOracleResult{identity:vec4u,distances:vec4f,firstNormal:vec4f,exitNormal:vec4f}
@group(0) @binding(0) var<storage,read> oracleVolumes:array<SvoThickGlassRecord>;
@group(0) @binding(1) var<storage,read> oracleRays:array<ThickGlassRayInput>;
@group(0) @binding(2) var<storage,read_write> oracleResults:array<ThickGlassOracleResult>;
@compute @workgroup_size(1)
fn thickGlassOracle(@builtin(global_invocation_id) id:vec3u){if(id.x>=arrayLength(&oracleResults)){return;}let ray=oracleRays[id.x];let hit=svoThickGlassIntersect(oracleVolumes[id.x],ray.originMin.xyz,ray.directionMax.xyz,ray.originMin.w,ray.directionMax.w,ray.expectedRevision.x);var first=hit.exit;if(hit.hasEntry==1u){first=hit.entry;}oracleResults[id.x]=ThickGlassOracleResult(vec4u(hit.status,hit.insideAtStart,hit.tangent,hit.hasEntry),vec4f(first.t_m,hit.exit.t_m,hit.opticalPath_m,0.0),vec4f(first.normal,0.0),vec4f(hit.exit.normal,0.0));}`;

test("real GPU thick-glass interval and normals agree with the CPU oracle", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU thick-glass checks",
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

  const volumeWords = packSvoThickGlassVolumes(cases.map(({ volume }) => volume));
  const volumeFloats = new Float32Array(volumeWords.buffer);
  cases.forEach(({ corruptRadius }, index) => {
    if (corruptRadius) volumeFloats[index * SVO_THICK_GLASS_RECORD_WORDS + 3] = 0;
  });
  const rayWords = new Uint32Array(cases.length * 12);
  const rayFloats = new Float32Array(rayWords.buffer);
  cases.forEach(({ ray, expectedRevision }, index) => {
    const baseWord = index * 12;
    rayFloats.set([...ray.origin_m, ray.tMin_m ?? 0], baseWord);
    rayFloats.set([...ray.direction, ray.tMax_m], baseWord + 4);
    rayWords[baseWord + 8] = expectedRevision;
  });
  const resultBytes = cases.length * 64;
  const volumeBuffer = device.createBuffer({ size: volumeWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const rayBuffer = device.createBuffer({ size: rayWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const resultBuffer = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(volumeBuffer, 0, volumeWords);
  device.queue.writeBuffer(rayBuffer, 0, rayWords);
  const shaderModule = device.createShaderModule({ code: oracleShader });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "thickGlassOracle" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: volumeBuffer } },
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
    const baseWord = index * 16;
    const expected = entry.corruptRadius
      ? { status: "invalid" as const }
      : querySvoThickGlassVolume(entry.volume, entry.ray, entry.expectedRevision);
    const expectedStatus = SVO_THICK_GLASS_QUERY_STATUS[expected.status];
    assert.equal(words[baseWord], expectedStatus, `${entry.name}: status`);
    if (expected.status !== "hit") return;
    assert.equal(words[baseWord + 1], Number(expected.interval.insideAtStart), `${entry.name}: inside`);
    assert.equal(words[baseWord + 2], Number(expected.interval.tangent), `${entry.name}: tangent`);
    assert.equal(words[baseWord + 3], Number(Boolean(expected.interval.entry)), `${entry.name}: entry`);
    assert.ok(Math.abs(floats[baseWord + 4] - expected.interval.first.t_m) <= 3e-4, `${entry.name}: first distance`);
    assert.ok(Math.abs(floats[baseWord + 5] - expected.interval.exit.t_m) <= 3e-4, `${entry.name}: exit distance`);
    assert.ok(Math.abs(floats[baseWord + 6] - expected.interval.opticalPath_m) <= 3e-4, `${entry.name}: optical path`);
    for (let axis = 0; axis < 3; axis += 1) {
      assert.ok(Math.abs(floats[baseWord + 8 + axis] - expected.interval.first.geometricNormal[axis]) <= 3e-4,
        `${entry.name}: first normal ${axis}`);
      assert.ok(Math.abs(floats[baseWord + 12 + axis] - expected.interval.exit.geometricNormal[axis]) <= 3e-4,
        `${entry.name}: exit normal ${axis}`);
    }
  });
  readback.unmap();
  assert.deepEqual(validationErrors, []);
  volumeBuffer.destroy();
  rayBuffer.destroy();
  resultBuffer.destroy();
  readback.destroy();
  device.destroy();
});
