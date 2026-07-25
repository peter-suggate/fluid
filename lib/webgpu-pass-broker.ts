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

  constructor(private readonly encoder: GPUCommandEncoder) {}

  /** Return the open compute pass, creating it only when necessary. */
  compute(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    if (!this.openComputePass) {
      this.openComputePass = this.encoder.beginComputePass(descriptor);
      this.openedComputePassCount += 1;
    }
    return this.openComputePass;
  }

  /** End the current pass. `reason` documents semantic boundaries at callers. */
  fence(reason?: string): void {
    const pass = this.openComputePass;
    if (!pass) return;
    this.openComputePass = undefined;
    this.latestFence = reason;
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
