import assert from "node:assert/strict";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { WebGpuSvoFluidCoverage } from "../lib/svo/features/scene-publication/webgpu-svo-fluid-coverage";

/** Deterministic moving coverage; exercises production fill/mips/composition,
 * without conflating the renderer comparison with solver advancement. */
export async function createSunlightFluidFixture(device: GPUDevice, renderer: SparseVoxelDrySceneRenderer) {
  const size = 32;
  const phi = device.createTexture({ size: [size, size, size], dimension: "3d", format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const coverage = new WebGpuSvoFluidCoverage(device, {
    fieldDimensions: [size, size, size], worldOrigin_m: [-0.9, 0, -0.6],
    cellSize_m: [1.8 / size, 1.6 / size, 1.2 / size],
  }, { kind: "dense", coarsePhi: phi.createView() });
  await coverage.initializePipelines();
  return {
    async update(filled: boolean | undefined) {
      if (filled === undefined) { renderer.setFluidCoverage(undefined); return; }
      const values = new Float32Array(size ** 3);
      for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
        values[x + size * (y + size * z)] = filled ? Math.abs((y + 0.5) * 1.6 / size - 0.6) - 0.2 : 1;
      }
      device.queue.writeTexture({ texture: phi }, values, { bytesPerRow: size * 4, rowsPerImage: size }, [size, size, size]);
      const encoder = device.createCommandEncoder({ label: "Sunlight moving fluid coverage" });
      assert.ok(coverage.encode(encoder));
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      renderer.setFluidCoverage(coverage);
    },
    async counters() {
      const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      assert.ok(renderer.copyVoxelLightCacheCounters(encoder, buffer));
      device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const result = Array.from(new Uint32Array(buffer.getMappedRange()));
      buffer.unmap(); buffer.destroy();
      return result;
    },
    destroy() { renderer.setFluidCoverage(undefined); coverage.destroy(); phi.destroy(); },
  };
}
