import { uniformDetailExtent, uniformDetailField } from "../../lib/methods/uniform/uniform-detail-fields";
import type { UniformLatticeAddress } from "../../lib/methods/uniform/uniform-lattice-address";
/** A solver field's lattice extent; its texture may be a packed atlas. */
export const mixedExtent = (texture: GPUTexture): [number, number, number] => uniformDetailExtent(texture);
/** The texture holding a solver field's texels now, for raw whole-texture copies. */
export const mixedPhysical = (texture: GPUTexture): GPUTexture => uniformDetailField(texture)?.storage.physical(texture) ?? texture;
/** Tightly packed texture readback for transport-stage comparisons.
 * address: the residency to place a detail field's texels by, in place of
 * its storage's (whose directory is the seed's: a live pool claims and
 * retires pages on the GPU). The storage is not changed. */
export async function readMixedTexture(device: GPUDevice, texture: GPUTexture, address?: UniformLatticeAddress): Promise<Float32Array> {
  const field = uniformDetailField(texture);
  if (field) {
    if (!address) return field.storage.read(texture);
    // The storage seen through another layout address; its field identity
    // checks (kindOf, physical) still run on the storage itself.
    const storage = field.storage, layout = { ...storage.layout, address };
    const view = new Proxy(storage, { get: (target, key, receiver) =>
      key === "layout" ? layout : key === "kindOf" || key === "physical" ? (target[key] as (t: GPUTexture) => unknown).bind(target) : Reflect.get(target, key, receiver) });
    return view.read(texture);
  }
  const components = texture.format === "rgba32float" ? 4 : 1;
  const row = texture.width * components * 4, pitch = Math.ceil(row / 256) * 256;
  const staging = device.createBuffer({ size: pitch * texture.height * texture.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: pitch, rowsPerImage: texture.height }, [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange()), dst = new Uint8Array(row * texture.height * texture.depthOrArrayLayers);
    for (let y = 0; y < texture.height * texture.depthOrArrayLayers; y++) dst.set(src.subarray(y * pitch, y * pitch + row), y * row);
    return new Float32Array(dst.buffer);
  } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
}

export async function readMixedBuffer(device: GPUDevice, source: GPUBuffer): Promise<Float32Array> {
  const read=device.createBuffer({size:source.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,0,read,0,source.size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()).slice();read.unmap();return values;
  } finally {if(read.mapState==="mapped")read.unmap();read.destroy();}
}

/** Current mixed tile words (bit 31 = h tile), read from the GPU ownership:
 * a generation adopted on the GPU leaves no host mirror to read. */
export async function readMixedTileWords(device: GPUDevice, solver: unknown): Promise<Uint32Array> {
  const ownership = (solver as { mixedFrame: { ownership: { presentation: { buffer: GPUBuffer }; capacity: { tiles: number } } } }).mixedFrame.ownership;
  const bytes = ownership.capacity.tiles * 4, read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(ownership.presentation.buffer, 0, read, 0, bytes); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}
