/**
 * Diagnostic only: end the open pass whenever `compute()` asks for a *different*
 * label, so a labelled pass brackets exactly the dispatches recorded under that
 * label.
 *
 * Without it, a `compute({label})` call whose pass is already open silently
 * DROPS the label. A per-pass timestamp report then names each pass after the
 * first `compute()` since the last fence while charging it everything encoded
 * until the next one. Measured on the mini dam lane 2026-07-28: the five-
 * dispatch workset publication, worth 0.19 ms, reported 3.670 ms on
 * `Workset scatter rows` because that pass stayed open across the downstream
 * stages. With this on it reports 0.011 ms, and the number follows the kernel
 * when the two scatter dispatches are swapped instead of following the slot.
 *
 * Splitting a pass is semantically neutral -- WebGPU already orders storage
 * accesses within a pass, and a pass boundary is a strictly stronger barrier --
 * so this only ever costs pass launches. It measured 39.07 ms/advance against
 * 39.09 ms unisolated on the mini lane, but it does change the command stream,
 * so it stays off unless asked for.
 */
export function passBrokerLabelIsolationRequested(
  environment: Record<string, string | undefined> | undefined
    // The broker encodes in the browser as well as under Dawn, where there is
    // no `process`.
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_GPU_ISOLATE_PASS_LABELS === "1";
}

/**
 * Lazily owns the compute pass for one GPU command encoder.
 *
 * A broker may be threaded through several encoding helpers. Consecutive
 * compute stages then share one pass until a command that cannot be recorded
 * inside a compute pass, or an explicit semantic fence, closes it.
 */
export class PassBroker {
  private openComputePass?: GPUComputePassEncoder;
  private openComputePassLabel?: string;
  private openedComputePassCount = 0;
  private latestFence?: string;
  /** Sampled per broker so a test can drive it; production reads the env once
   * per command encoder, which is not on any hot path. */
  private readonly isolateLabels: boolean;

  constructor(private readonly encoder: GPUCommandEncoder, options?: { isolateLabels?: boolean }) {
    this.isolateLabels = options?.isolateLabels ?? passBrokerLabelIsolationRequested();
  }

  /** Return the open compute pass, creating it only when necessary. */
  compute(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    if (this.isolateLabels && this.openComputePass && descriptor?.label !== undefined
      && descriptor.label !== this.openComputePassLabel) {
      this.fence("pass label isolation");
    }
    if (!this.openComputePass) {
      this.openComputePass = this.encoder.beginComputePass(descriptor);
      this.openComputePassLabel = descriptor?.label;
      this.openedComputePassCount += 1;
    }
    return this.openComputePass;
  }

  /** End the current pass. `reason` documents semantic boundaries at callers. */
  fence(reason?: string): void {
    const pass = this.openComputePass;
    if (!pass) return;
    this.openComputePass = undefined;
    this.openComputePassLabel = undefined;
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
