import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GridOverlayPipeline, gridOverlayViewRecordsWGSL } from "../lib/core/webgpu-grid-overlay";
import {
  TILE_CLASS_FINE, TILE_CLASS_RECORD_CLASS_WORD, TILE_CLASS_RECORD_WORDS, TILE_CLASS_SHELL,
} from "../lib/core/method-view-records";

async function readBuffer(device: GPUDevice, binding: GPUBufferBinding): Promise<Uint32Array> {
  const size = binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Uint32Array(staging.getMappedRange().slice(0));
  } finally { staging.unmap(); staging.destroy(); }
}

async function readVolume(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const row = Math.ceil(texture.width * 4 / 256) * 256;
  const layers = texture.height * texture.depthOrArrayLayers;
  const staging = device.createBuffer({ size: row * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: row, rowsPerImage: texture.height },
      [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(staging.getMappedRange());
    const result = new Float32Array(texture.width * layers);
    for (let r = 0; r < layers; r++) result.set(mapped.subarray(r * row / 4, r * row / 4 + texture.width), r * texture.width);
    return result;
  } finally { staging.unmap(); staging.destroy(); }
}

/** The view's own WGSL lookup, run once per cell against whatever is bound at 23. */
async function probeClasses(device: GPUDevice, records: GPUBufferBinding, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const module = device.createShaderModule({ code: `${gridOverlayViewRecordsWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(4,4,4) fn probe(@builtin(global_invocation_id) gid:vec3u){
  let d=vec3i(${nx},${ny},${nz});let c=vec3i(gid);if(any(c>=d)){return;}
  result[u32(c.x+d.x*(c.y+d.y*c.z))]=tileClassAt(c,d);
}` });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
  const output = device.createBuffer({ size: 4 * nx * ny * nz, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  try {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: output } }, { binding: 23, resource: records },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    device.queue.submit([encoder.finish()]);
    return await readBuffer(device, { buffer: output });
  } finally { output.destroy(); }
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("fine-tiles view reads Uniform Geometric's tile classes", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform-volume fine-tiles view");
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
  const destroy: { destroy(): void }[] = [];
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn as NodeDawnProvider, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    // A slab with a ball in it: mini32's eight tiles a side are all within the
    // two-tile fine reach of its liquid, so it has no far air to tell apart.
    solver = await uniformVolumeMethod.createSolverAsync!(device, sceneDocument(getSceneDefinition("cm12-figure-2")),
      "balanced", resolveMethodValues(uniformVolumeMethod, "balanced", {}), undefined, () => {}) as WebGPUUniformReferenceSolver;
    const { nx, ny, nz } = solver.info; const dims = [nx, ny, nz] as const;
    // Read through a call: the getter changes with the solver, not with this scope.
    const published = () => solver!.tileClassSource;
    assert.equal(published(), undefined, "no step has classified a tile yet");

    for (let frame = 1; frame <= 10; frame++) assert.ok(solver.advanceTo(frame / 30));
    await device.queue.onSubmittedWorkDone();
    const source = published();
    assert.ok(source, "the sampler is on by default, so its classes are published");

    // The lookup must land on the class word of the cell's own tile record.
    const records = await readBuffer(device, source.records);
    const tiles = dims.map(n => n / 4);
    assert.equal(records.length, TILE_CLASS_RECORD_WORDS * tiles[0]! * tiles[1]! * tiles[2]!);
    const classes = await probeClasses(device, source.records, dims);
    const volume = await readVolume(device, solver.volumeTexture);
    const counts = { fine: 0, shell: 0, far: 0 };
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const cell = x + nx * (y + ny * z);
      const tile = (x >> 2) + tiles[0]! * ((y >> 2) + tiles[1]! * (z >> 2));
      assert.equal(classes[cell], records[TILE_CLASS_RECORD_WORDS * tile + TILE_CLASS_RECORD_CLASS_WORD], `cell ${x},${y},${z}`);
      // What the legend promises: liquid is sampled on the finest lattice.
      if (volume[cell]! > 1e-6) assert.ok(classes[cell]! & TILE_CLASS_FINE, `liquid cell ${x},${y},${z} outside a fine tile`);
      if (x % 4 || y % 4 || z % 4) continue;
      if (classes[cell]! & TILE_CLASS_FINE) counts.fine++;
      else if (classes[cell]! & TILE_CLASS_SHELL) counts.shell++;
      else counts.far++;
    }
    assert.ok(counts.fine > 0 && counts.far > 0, `a picture with something to separate: ${JSON.stringify(counts)}`);

    // The overlay binds the published range as-is: offset alignment and size.
    const target = device.createTexture({ size: [16, 16], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const uniforms = device.createBuffer({ size: 416, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bodies = device.createBuffer({ size: 768, usage: GPUBufferUsage.STORAGE });
    const texture3d = (format: GPUTextureFormat) => device!.createTexture({ size: [1, 1, 1], dimension: "3d", format, usage: GPUTextureUsage.TEXTURE_BINDING });
    const columns = device.createTexture({ size: [1, 1], format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const cells = texture3d("r32uint"), samples = texture3d("rg32uint"), velocity = texture3d("rgba32float"), scalar = texture3d("r32float");
    destroy.push(target, uniforms, bodies, columns, cells, samples, velocity, scalar);
    const overlay = new GridOverlayPipeline(device, "rgba8unorm", uniforms, bodies);
    destroy.push(overlay);
    device.pushErrorScope("validation");
    await overlay.initialize();
    overlay.setDenseLevelSetVolumeSource(solver.denseLevelSetVolumeSource);
    overlay.setViewRecords(source);
    overlay.setVolume(solver.volumeTexture, columns, cells, velocity, samples, scalar, scalar, solver.volumeTexture);
    const encoder = device.createCommandEncoder();
    assert.ok(overlay.encode(encoder, target.createView()));
    device.queue.submit([encoder.finish()]);
    assert.equal((await device.popErrorScope())?.message, undefined);

    // Off withdraws the source, and the one-word dummy then reads as all fine.
    solver.applyRuntimeValues({ twoLevelVelocity: "off" });
    assert.ok(solver.advanceTo(11 / 30));
    await device.queue.onSubmittedWorkDone();
    assert.equal(published(), undefined);
    const dummy = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    destroy.push(dummy);
    const dense = await probeClasses(device, { buffer: dummy }, dims);
    assert.ok(dense.every(value => value === (TILE_CLASS_FINE | TILE_CLASS_SHELL)));
    assert.deepEqual(errors, []);
  } finally {
    for (const resource of destroy) resource.destroy();
    solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  }
});
