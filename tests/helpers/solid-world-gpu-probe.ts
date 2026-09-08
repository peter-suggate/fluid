import assert from "node:assert/strict";
import { WEBGPU_SOLID_WORLD_MAGIC, type WebgpuSolidWorldPageLayout } from "../../lib/core/webgpu-solid-world-pages";

/** QA read of the actual fluid SolidWorld GPU authority, including dry cells
 * that have no active fluid owner and therefore do not appear in field exports.
 * The private resident access is observation only; publication still goes
 * through the production editor/worker/renderer path. */
export async function readGpuSolidFractions(device: GPUDevice, sparseWorld: unknown,
  dimensions: readonly [number, number, number]): Promise<Float32Array> {
  const resident = (sparseWorld as { resident: { topologyArena: GPUBuffer;
    solidOccupancyLayout: WebgpuSolidWorldPageLayout } }).resident;
  const layout = resident.solidOccupancyLayout;
  assert.ok(layout, "fluid resident must own a SolidWorld image");
  const buffer = device.createBuffer({ label: "QA authored solid authority", size: 4 * (layout.totalWords - layout.baseWords),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(resident.topologyArena, 4 * layout.baseWords, buffer, 0, buffer.size);
    device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(buffer.getMappedRange());
    assert.equal(words[0], WEBGPU_SOLID_WORLD_MAGIC);
    const [nx, ny, nz] = dimensions;
    const result = new Float32Array(nx * ny * nz);
    for (let slot = 0; slot < layout.directoryCapacity; slot++) {
      const entry = layout.directoryBaseWords + slot * 6;
      if (words[entry] !== 2) continue;
      const origin = [words[entry + 2]! | 0, words[entry + 3]! | 0, words[entry + 4]! | 0].map(value => value * 8);
      const page = layout.pageBaseWords + words[entry + 5]! * layout.pageWords;
      for (let local = 0; local < 512; local++) {
        const x = origin[0]! + (local & 7), y = origin[1]! + ((local >>> 3) & 7), z = origin[2]! + (local >>> 6);
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;
        result[x + nx * (y + ny * z)] = ((words[page + (local >>> 2)]! >>> (8 * (local & 3))) & 255) / 255;
      }
    }
    for (let index = 0; index < layout.regionCapacity; index++) {
      const at = layout.regionBaseWords + index * 8;
      const lo = [words[at + 1]! | 0, words[at + 2]! | 0, words[at + 3]! | 0];
      const hi = [words[at + 4]! | 0, words[at + 5]! | 0, words[at + 6]! | 0];
      for (let z = Math.max(0, lo[2]!); z < Math.min(nz, hi[2]!); z++)
        for (let y = Math.max(0, lo[1]!); y < Math.min(ny, hi[1]!); y++)
          for (let x = Math.max(0, lo[0]!); x < Math.min(nx, hi[0]!); x++) result[x + nx * (y + ny * z)] = words[at] === 1 ? 1 : 0;
    }
    buffer.unmap(); return result;
  } finally { buffer.destroy(); }
}
