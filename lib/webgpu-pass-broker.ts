/**
 * Lazily owns the compute pass for one GPU command encoder.
 *
 * A broker may be threaded through several encoding helpers. Consecutive
 * compute stages then share one pass until a command that cannot be recorded
 * inside a compute pass, or an explicit semantic fence, closes it.
 */
export class PassBroker {
  private openComputePass?: GPUComputePassEncoder;
  private openedComputePassCount = 0;
  private latestFence?: string;
  private readonly indirectBufferRoles = new Map<GPUBuffer, number>();

  constructor(private readonly encoder: GPUCommandEncoder) {}

  /** Return the open compute pass, creating it only when necessary. */
  compute(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    if (!this.openComputePass) {
      this.openComputePass = this.encoder.beginComputePass(descriptor);
      this.openedComputePassCount += 1;
    }
    return this.openComputePass;
  }

  /**
   * Acquire the compute pass while respecting WebGPU's pass-wide prohibition
   * on using one buffer as both writable storage and indirect commands.
   * Consecutive users of the same role remain collapsed into the open pass;
   * only an actual role transition creates a synchronization boundary.
   */
  computeForIndirectBuffer(
    buffer: GPUBuffer,
    role: "storage-write" | "indirect",
    descriptor?: GPUComputePassDescriptor,
  ): GPUComputePassEncoder {
    const bit = role === "storage-write" ? 1 : 2;
    const prior = this.indirectBufferRoles.get(buffer) ?? 0;
    if (prior !== 0 && (prior & bit) === 0) {
      this.fence("indirect buffer storage/command role transition");
    }
    const pass = this.compute(descriptor);
    this.indirectBufferRoles.set(buffer, (this.indirectBufferRoles.get(buffer) ?? 0) | bit);
    return pass;
  }

  /** End the current pass. `reason` documents semantic boundaries at callers. */
  fence(reason?: string): void {
    const pass = this.openComputePass;
    if (!pass) return;
    this.openComputePass = undefined;
    this.latestFence = reason;
    this.indirectBufferRoles.clear();
    pass.end();
  }

  clearBuffer(buffer: GPUBuffer, offset?: GPUSize64, size?: GPUSize64): void {
    this.fence("clear buffer");
    this.encoder.clearBuffer(buffer, offset, size);
  }

  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: GPUSize64,
    destination: GPUBuffer,
    destinationOffset: GPUSize64,
    size: GPUSize64,
  ): void {
    this.fence("copy buffer");
    this.encoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
  }

  /** Copy GPU-authored command arguments into an INDIRECT-only buffer. */
  updateIndirectBuffer(
    source: GPUBuffer,
    sourceOffset: GPUSize64,
    destination: GPUBuffer,
    destinationOffset: GPUSize64,
    size: GPUSize64,
  ): void {
    this.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
  }

  /** Escape to commands not represented here only after closing compute. */
  commandEncoder(): GPUCommandEncoder {
    this.fence("raw command encoder access");
    return this.encoder;
  }

  /** Finish the underlying encoder without leaving an unterminated pass. */
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer {
    this.fence("finish command encoder");
    return this.encoder.finish(descriptor);
  }

  /** Useful to command audits and focused unit tests; not a synchronization query. */
  get computePassCount(): number { return this.openedComputePassCount; }

  get hasOpenComputePass(): boolean { return this.openComputePass !== undefined; }

  get lastFenceReason(): string | undefined { return this.latestFence; }
}
