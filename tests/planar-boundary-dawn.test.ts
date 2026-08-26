import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  packPlanarBoundaryPatches,
  planarBoundaryWGSL,
  type PlanarBoundaryPatch,
} from "../lib/core/planar-boundary";
import { createSvoDrySceneFragmentWGSL } from
  "../lib/svo/webgpu-svo-dry-scene";
import { createSvoScenePrimitiveBandWGSL } from
  "../lib/svo/svo-scene-primitive-band";

const dawnModule = process.env.WEBGPU_NODE_MODULE;

test("Dawn compiles the production path and executes the planar-boundary ABI", {
  skip: !dawnModule && "set WEBGPU_NODE_MODULE for Dawn planar-boundary parity",
}, async () => {
  const dawn = await import(pathToFileURL(dawnModule!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter, "Dawn did not expose the requested adapter");
  const device = await adapter.requestDevice();
  try {
    const boundary: PlanarBoundaryPatch = {
      center_m: [0, 0, 0],
      normal: [0, 1, 0],
      tangentU: [1, 0, 0],
      tangentV: [0, 0, 1],
      halfExtentU_m: 2,
      halfExtentV_m: 3,
      halfThickness_m: 0.125,
      materialId: 7,
      ownerId: 11,
    };
    const records = packPlanarBoundaryPatches([boundary]);
    const rays = new Float32Array([
      0, 2, 0, 0, 0, -1, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0,
      3, 2, 0, 0, 0, -1, 0, 0,
      0, 0, 5, 0, 0, 0, -1, 0,
    ]);
    const shader = `${planarBoundaryWGSL}
struct TestRay { origin:vec4f, direction:vec4f }
struct TestAnswer { interval:vec4f, surface:vec4f }
@group(0) @binding(0) var<storage,read> boundaries:array<PlanarBoundaryPatch>;
@group(0) @binding(1) var<storage,read> rays:array<TestRay>;
@group(0) @binding(2) var<storage,read_write> answers:array<TestAnswer>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=arrayLength(&rays)){return;}
  let ray=rays[id.x];
  let hit=intersectPlanarBoundary(boundaries[0],ray.origin.xyz,ray.direction.xyz,0.0,10.0);
  answers[id.x].interval=vec4f(f32(hit.valid),hit.tHit,hit.tEnter,hit.tExit);
  answers[id.x].surface=vec4f(hit.normal,f32(hit.featureAxis));
}`;

    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "Planar boundary Dawn parity", code: shader });
    const compilation = await module.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(({ type }) => type === "error")
      .map(({ lineNum, message }) => `${lineNum}: ${message}`), []);
    const productionModule = device.createShaderModule({
      label: "Canonical dry-scene planar-terminal compilation gate",
      code: createSvoDrySceneFragmentWGSL(1, "canonical", "bounds", "inline"),
    });
    const productionCompilation = await productionModule.getCompilationInfo();
    assert.deepEqual(productionCompilation.messages.filter(({ type }) => type === "error")
      .map(({ lineNum, message }) => `${lineNum}: ${message}`), []);
    const rasterPrimaryModule = device.createShaderModule({
      label: "Raster-primary planar seam arbitration compilation gate",
      code: createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split",
        0, false, true, true),
    });
    const rasterPrimaryCompilation = await rasterPrimaryModule.getCompilationInfo();
    assert.deepEqual(rasterPrimaryCompilation.messages.filter(({ type }) => type === "error")
      .map(({ lineNum, message }) => `${lineNum}: ${message}`), []);
    const rasterBandModule = device.createShaderModule({
      label: "Raster-primary near-field band compilation gate",
      code: createSvoScenePrimitiveBandWGSL({ primitiveWordOffset: 0 }),
    });
    const rasterBandCompilation = await rasterBandModule.getCompilationInfo();
    assert.deepEqual(rasterBandCompilation.messages.filter(({ type }) => type === "error")
      .map(({ lineNum, message }) => `${lineNum}: ${message}`), []);
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const recordBuffer = device.createBuffer({
      size: records.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const rayBuffer = device.createBuffer({
      size: rays.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const answerBytes = 4 * 8 * Float32Array.BYTES_PER_ELEMENT;
    const answerBuffer = device.createBuffer({
      size: answerBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: answerBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(recordBuffer, 0, records);
    device.queue.writeBuffer(rayBuffer, 0, rays);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: recordBuffer } },
        { binding: 1, resource: { buffer: rayBuffer } },
        { binding: 2, resource: { buffer: answerBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(answerBuffer, 0, readback, 0, answerBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validation = await device.popErrorScope();
    assert.equal(validation, null);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const epsilon = 1e-6;
    const approximately = (actual: number, expected: number) =>
      assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
    assert.deepEqual([...result.slice(0, 1)], [1]);
    approximately(result[1]!, 1.875);
    approximately(result[2]!, 1.875);
    approximately(result[3]!, 2.125);
    assert.deepEqual([...result.slice(4, 8)], [0, 1, 0, 2]);

    assert.equal(result[8], 1, "an inside ray hits the physical exit face");
    approximately(result[9]!, 2);
    assert.deepEqual([...result.slice(12, 16)], [1, 0, 0, 0]);

    assert.equal(result[16], 0, "a ray outside the finite U extent misses");

    assert.equal(result[24], 1);
    approximately(result[25]!, 2);
    assert.deepEqual([...result.slice(28, 32)], [0, 0, 1, 1]);
  } finally {
    device.destroy();
  }
});
