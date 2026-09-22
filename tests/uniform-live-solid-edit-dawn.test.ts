import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { SceneDescription } from "../lib/core/model";

async function readBuffer(device: GPUDevice, source: GPUBuffer, offset: number, size: number): Promise<Uint32Array> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, offset, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Uint32Array(staging.getMappedRange().slice(0));
  } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
}

async function readVolume(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const row = Math.ceil(texture.width * 4 / 256) * 256, layers = texture.depthOrArrayLayers;
  const staging = device.createBuffer({ size: row * texture.height * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: row, rowsPerImage: texture.height },
      [texture.width, texture.height, layers]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(staging.getMappedRange()), result = new Float32Array(texture.width * texture.height * layers);
    for (let line = 0; line < texture.height * layers; line += 1) {
      result.set(source.subarray(line * row / 4, line * row / 4 + texture.width), line * texture.width);
    }
    return result;
  } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
}

/** The per-cell sampler is the definition of the solver's solid mask. */
function sampledMask(scene: SceneDescription, nx: number, ny: number, nz: number): Uint32Array {
  const world = solidWorldForScene(scene), sx = nx + 2, sy = ny + 2, sz = nz + 2;
  const words = new Uint32Array(4 + Math.ceil(sx * sy * sz / 32));
  words.set([0x53565731, sx, sy, sz]);
  for (let z = -1; z <= nz; z += 1) for (let y = -1; y <= ny; y += 1) for (let x = -1; x <= nx; x += 1) {
    if (sampleSolidWorld(world, [x, y, z]).solidFraction <= 0) continue;
    const index = (x + 1) + sx * ((y + 1) + sy * (z + 1));
    words[4 + (index >>> 5)]! |= 1 << (index & 31);
  }
  return words;
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("Uniform Geometric adopts a voxel stroke on the running solver", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform live solid edit");
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", (event) => { event.preventDefault(); errors.push(event.error.message); });
    let scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
    solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined, uniformGeometricSolverOptions({}, scene), () => {});
    const { nx, ny, nz } = solver.info;
    const internals = solver as unknown as { activeScratch: GPUBuffer; solidVoxelScratchOffsetWords: number };
    const gpuMask = (words: number) => readBuffer(device!, internals.activeScratch, internals.solidVoxelScratchOffsetWords * 4, words * 4);
    let frame = 0;
    const advance = async (frames: number) => {
      for (let step = 0; step < frames; step += 1) { frame += 1; assert.ok(solver!.advanceTo(frame / 30)); await solver!.awaitFrameCompletion(); }
    };
    const wetColumns = async () => {
      const volume = await readVolume(device!, solver!.volumeTexture), width = solver!.volumeTexture.width, height = solver!.volumeTexture.height;
      const columns = new Float64Array(nx);
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) columns[x]! += volume[x + width * (y + height * z)]!;
      return columns;
    };
    await advance(2);
    const before = await wetColumns();
    const damLow = before.slice(0, nx / 2).reduce((sum, value) => sum + value, 0) > before.slice(nx / 2).reduce((sum, value) => sum + value, 0);
    // A full-height wall two cells thick, three quarters of the way to the dry end.
    const wall = damLow ? [24, 26] : [6, 8];
    const beyond = (columns: Float64Array) => (damLow ? columns.slice(wall[1]) : columns.slice(0, wall[0])).reduce((sum, value) => sum + value, 0);
    assert.ok(beyond(before) < 1e-6, "the far end starts dry");
    const base = scene;
    scene = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [wall[0]!, 0, 0], maximumExclusive: [wall[1]!, ny, nz] }]);
    solver.validateLiveSolidEdit(scene);
    solver.applySceneUniforms(scene);
    const expected = sampledMask(scene, nx, ny, nz);
    assert.deepEqual(await gpuMask(expected.length), expected, "the GPU mask after a stroke is the sampled mask");
    await advance(45);
    assert.ok(beyond(await wetColumns()) < 1e-3, "no water passes a wall drawn ahead of the front");
    // Undo restores the base document; the wall's pages leave the world and its bits must clear.
    solver.validateLiveSolidEdit(base);
    solver.applySceneUniforms(base);
    assert.deepEqual(await gpuMask(expected.length), sampledMask(base, nx, ny, nz), "the GPU mask after undo is the base mask");
    await advance(45);
    assert.ok(beyond(await wetColumns()) > 1, "water floods the far end once the wall is removed");
    assert.throws(() => solver!.validateLiveSolidEdit({ ...base, container: { ...base.container, width_m: base.container.width_m * 2 } }));
    await solver.readStats();
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
