/** Paired renderer-world preparation; run under run-webgpu-exclusive.ts. */
import assert from "node:assert/strict";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";
import { WebGPULiveSvoScene, type LiveSvoSceneOptions } from "../lib/svo/webgpu-live-svo-scene";
import { getScenePreset } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import type { OctreeSparseBrickWorldProgress } from "../lib/svo/webgpu-svo-sparse-bricks";
import type { SparseBrickPublicationSource } from "../lib/svo/sparse-brick-octree";

const { device, validationErrors } = await createDawnRenderDevice();
// Inspect the actual upload image before ordinary publication. This entry
// exercises all production planning and allocation, without unrelated render
// specialization in the measured interval.
const build = WebGPULiveSvoScene as unknown as {
  buildWorld(device: GPUDevice, scene: SceneDescription, options: LiveSvoSceneOptions,
    progress: OctreeSparseBrickWorldProgress, interrupt: object): Promise<{
      world: { source: SparseBrickPublicationSource; destroy(): void };
    }>;
};
async function prepare(scene: SceneDescription, cpu: boolean) {
  const start = performance.now();
  const { world } = await build.buildWorld(device, scene,
    { cpuBrickSelection: cpu, environmentRefinementDepth: 0 }, () => {}, {});
  const elapsed_ms = performance.now() - start;
  const source = world.source;
  const readback = device.createBuffer({size: Math.max(4, source.topology.size),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.topology, 0, readback, 0, source.topology.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const topology = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return { elapsed_ms, topology, capacities: source.capacities };
  } finally {readback.destroy();world.destroy();}
}
try {
  for (const id of (process.env.FLUID_PROBE_SCENES ?? "water-box-dam-break,garden-pond,garden-dam-break").split(",")) {
    const scene = getScenePreset(id).create();
    const cpu = await prepare(scene, true);
    const gpu = await prepare(scene, false);
    assert.deepEqual(gpu.capacities, cpu.capacities, `${id}: GPU capacity must match CPU oracle`);
    assert.deepEqual(gpu.topology, cpu.topology, `${id}: GPU selection/refinement must preserve every topology word`);
    const warm = await prepare(scene, false);
    assert.deepEqual(warm.topology, gpu.topology, `${id}: warmed GPU preparation must be deterministic`);
    const cpuWarm = await prepare(scene, true);
    assert.deepEqual(cpuWarm.topology, cpu.topology, `${id}: CPU oracle must remain deterministic`);
    console.log(JSON.stringify({id, cpu_cold_ms: cpu.elapsed_ms, cpu_warm_ms: cpuWarm.elapsed_ms,
      gpu_cold_ms: gpu.elapsed_ms, gpu_warm_ms: warm.elapsed_ms,
      speedup: cpuWarm.elapsed_ms / warm.elapsed_ms, capacities: gpu.capacities}));
  }
  assert.deepEqual(validationErrors, []);
} finally {device.destroy();}
