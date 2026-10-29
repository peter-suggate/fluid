import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createSvoDrySceneFragmentWGSL, SVO_DRY_TRAVERSAL_MODES } from "../lib/svo/features/shading/program";
import { sparseVoxelDrySceneBindGroupLayoutEntries, svoSurfaceMeshComputeBindGroupLayoutEntries } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

const computeStorageCount = (entries: GPUBindGroupLayoutEntry[]) => entries.filter(entry =>
  (entry.visibility & GPUShaderStage.COMPUTE) !== 0
  && (entry.buffer?.type === "storage" || entry.buffer?.type === "read-only-storage")).length;

test("shared specular diagnostics leave the ten-buffer compute budget intact", () => {
  Object.assign(globalThis, { GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  for (const traversal of SVO_DRY_TRAVERSAL_MODES) {
    const scene = sparseVoxelDrySceneBindGroupLayoutEntries(traversal);
    assert.equal(scene.find(entry => entry.binding === 29)?.visibility, GPUShaderStage.FRAGMENT);
    assert.ok(computeStorageCount([...scene, ...svoSurfaceMeshComputeBindGroupLayoutEntries()]) <= 10, traversal);
  }
});

test("GPU production mesh pipelines compile with a granted ten-storage-buffer limit", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: { ...requiredFluidDeviceLimits(adapter.limits), maxStorageBuffersPerShaderStage: 10 },
  });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  try {
    assert.equal(device.limits.maxStorageBuffersPerShaderStage, 10);
    // The hybrid layout is a superset of raster-primary, including the fifth
    // traversal buffer. It exercises the tightest shared scene + mesh budget.
    const sceneEntries = sparseVoxelDrySceneBindGroupLayoutEntries("hybrid");
    const meshEntries = svoSurfaceMeshComputeBindGroupLayoutEntries();
    assert.equal(computeStorageCount([...sceneEntries, ...meshEntries]), 10);
    // Reproduce the old declaration without sending an invalid layout to the
    // device: adding compute visibility to diagnostics exceeds the same limit.
    assert.equal(computeStorageCount([...sceneEntries.map(entry => entry.binding === 29
      ? { ...entry, visibility: entry.visibility | GPUShaderStage.COMPUTE } : entry), ...meshEntries]), 11);
    const scene = device.createBindGroupLayout({ entries: sceneEntries });
    const mesh = device.createBindGroupLayout({ entries: meshEntries });
    for (const scale of [1, .5] as const) {
      const module = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(scale,
        "raster-primary", "bounds", "split", 0, false, false, false,
        { surfaceMesh: true, surfaceMeshSubgroups: false }) });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error"), []);
      // Mesh kernels do not read the reduced fragment-only prepass group.
      const middle = scale === 1 ? [] : [device.createBindGroupLayout({ entries: [] })];
      const layout = device.createPipelineLayout({ bindGroupLayouts: [scene, ...middle, mesh] });
      for (const entryPoint of ["surfaceMeshPrepare", "surfaceMeshBoxes", "surfaceMeshIngest", "surfaceMeshSchedule",
        "surfaceMeshCount", "surfaceMeshAllocate", "surfaceMeshEmit", "surfaceMeshPublish", "surfaceMeshSelect",
        "surfaceMeshCull", "surfaceMeshCullDispatch", "surfaceMeshPrimaryCandidates"]) {
        await device.createComputePipelineAsync({ label: `${entryPoint} x${scale}`, layout, compute: { module, entryPoint } });
      }
    }
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(errors, []);
  } finally { device.destroy(); }
});
