// Some Dawn releases expose an 8-byte-rounded mapped range for writes larger
// than the 4 MiB upload ring. A valid WebGPU upload whose byte length is 4 mod
// 8 then aborts inside the native staging copy. Keeping writes at or below the
// ring size is byte-exact there and remains ordinary writeBuffer behavior in
// browsers.
export const GPU_WRITE_BUFFER_CHUNK_BYTES = 4 * 1024 * 1024;

export function writeGPUBufferBytes(
  queue: GPUQueue,
  destination: GPUBuffer,
  destinationOffset: number,
  source: ArrayBuffer,
  sourceOffset: number,
  byteLength: number,
): void {
  for (let offset = 0; offset < byteLength; offset += GPU_WRITE_BUFFER_CHUNK_BYTES) {
    const chunkBytes = Math.min(GPU_WRITE_BUFFER_CHUNK_BYTES, byteLength - offset);
    queue.writeBuffer(destination, destinationOffset + offset, source,
      sourceOffset + offset, chunkBytes);
  }
}

export function writeGPUBufferView(
  queue: GPUQueue,
  destination: GPUBuffer,
  destinationOffset: number,
  source: ArrayBufferView,
): void {
  writeGPUBufferBytes(queue, destination, destinationOffset,
    source.buffer as ArrayBuffer, source.byteOffset, source.byteLength);
}
