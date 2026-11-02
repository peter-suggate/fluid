/**
 * One GPU → host readback through a staging buffer:
 *
 *   idle ─encode→ copied ─read→ mapping ─(consumed | failed)→ idle
 *                   └─discard→ idle            destroy from any → destroyed
 *
 * It replaces the `xCopied`/`xReading`/`xPending` flag pairs each readback
 * site kept by hand. The copy is encoded into the caller's encoder; the caller
 * submits as usual and then calls `read`. A read that resolves after `destroy`
 * yields undefined rather than stale bytes, and destroying mid-map releases
 * the staging buffer.
 */
export type StagedReadbackState = "idle" | "copied" | "mapping" | "destroyed";

export interface StagedReadbackOptions {
  readonly label: string;
  readonly size: number;
}

export class StagedReadback {
  private readonly staging: GPUBuffer;
  private currentState: StagedReadbackState = "idle";

  constructor(device: GPUDevice, readonly options: StagedReadbackOptions) {
    this.staging = device.createBuffer({
      label: options.label,
      size: options.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  get state(): StagedReadbackState { return this.currentState; }
  get idle(): boolean { return this.currentState === "idle"; }
  get size(): number { return this.options.size; }

  /**
   * Encode the copy into the staging buffer. Refused (false) unless idle, so a
   * site can call it every frame and only the first frame after a read stages.
   */
  encode(encoder: GPUCommandEncoder, copy: (encoder: GPUCommandEncoder, staging: GPUBuffer) => void): boolean {
    if (this.currentState !== "idle") return false;
    copy(encoder, this.staging);
    this.currentState = "copied";
    return true;
  }

  /** Forget a copy whose encoder was never submitted. */
  discard(): void {
    if (this.currentState === "copied") this.currentState = "idle";
  }

  /**
   * Map the submitted copy and hand its bytes to `consume` while mapped. The
   * bytes are only valid inside `consume`. Returns undefined when nothing was
   * copied, a read is already mapping, or the readback was destroyed meanwhile.
   */
  async read<T>(consume: (bytes: ArrayBuffer) => T): Promise<T | undefined> {
    if (this.currentState !== "copied") return undefined;
    this.currentState = "mapping";
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
    } catch (error) {
      if (this.destroyed()) return undefined;
      this.currentState = "idle";
      throw error;
    }
    if (this.destroyed()) return undefined;
    try {
      return consume(this.staging.getMappedRange());
    } finally {
      this.staging.unmap();
      this.currentState = "idle";
    }
  }

  /** A method, so the check after an await is not narrowed away. */
  private destroyed(): boolean { return this.currentState === "destroyed"; }

  destroy(): void {
    if (this.currentState === "destroyed") return;
    this.currentState = "destroyed";
    this.staging.destroy();
  }
}
