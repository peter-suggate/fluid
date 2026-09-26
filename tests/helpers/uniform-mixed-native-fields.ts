/** Tightly packed texture readback for transport-stage comparisons. */
export async function readMixedTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
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
