/** The detail ring's mask of a field's logical values: outside the ring an h
 * texel that is not canonical holds no state (the remap or the seam kernels
 * write it before any reader), so two storages may differ there and still be
 * the same simulation. maskOutsideRing zeroes those texels: a cell field's
 * non-origin cells, a face field's texels other than a tile's origin and +face
 * anchors, a vertex field's vertices other than tile corners, in every tile
 * whose ring bit (second stencil word, bit 0, of the presentation topology)
 * is clear. A vertex belongs to the tile of min(v >> 2, tiles - 1). */
import { uniformTileKey, type UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";

/** The ring bit of every tile key of a solver's simulation ownership. */
export async function readRingBits(device: GPUDevice, solver: unknown): Promise<Uint8Array> {
  const ownership = (solver as { mixedFrame: { ownership: { presentation: { buffer: GPUBuffer }; capacity: { tiles: number } } } }).mixedFrame.ownership;
  const n = ownership.capacity.tiles, staging = device.createBuffer({ size: 8 * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(ownership.presentation.buffer, 8 * n, staging, 0, 8 * n); device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ); const words = new Uint32Array(staging.getMappedRange().slice(0)); staging.destroy();
  const ring = new Uint8Array(n); for (let t = 0; t < n; t++) ring[t] = words[2 * t + 1]! & 1; return ring;
}

/** values: a field's logical values (readMixedTexture), kind by its length: cells x comp or vertices. */
export function maskOutsideRing(values: Float32Array, address: UniformLatticeAddress, ring: Uint8Array, cells: readonly number[]): Float32Array {
  const [cx, cy, cz] = cells as [number, number, number], [tx, ty, tz] = [cx >> 2, cy >> 2, cz >> 2];
  const vertices = (cx + 1) * (cy + 1) * (cz + 1), vertex = values.length === vertices, comp = vertex ? 1 : values.length / (cx * cy * cz);
  const [lx, ly, lz] = vertex ? [cx + 1, cy + 1, cz + 1] : [cx, cy, cz], out = values.slice();
  for (let z = 0; z < lz; z++) for (let y = 0; y < ly; y++) for (let x = 0; x < lx; x++) {
    const key = uniformTileKey(address, Math.min(x >> 2, tx - 1), Math.min(y >> 2, ty - 1), Math.min(z >> 2, tz - 1));
    if (ring[key]) continue;
    const jx = (x & 3) === 3 ? 1 : 0, jy = (y & 3) === 3 ? 1 : 0, jz = (z & 3) === 3 ? 1 : 0;
    const canonical = vertex ? !((x | y | z) & 3) : comp === 4 ? ((x & 3) % 3 | (y & 3) % 3 | (z & 3) % 3) === 0 && jx + jy + jz <= 1 : !((x | y | z) & 3);
    if (!canonical) out.fill(0, (x + lx * (y + ly * z)) * comp, (x + lx * (y + ly * z) + 1) * comp);
  }
  return out;
}
