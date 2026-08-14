import type { PerformanceTrace } from "./performance-trace";

/**
 * Fence-partitioned band walls for one presentation frame.
 *
 * Metal render-pass timestamps bracket tiler windows, so the per-pass
 * hardware trace can only price compute passes and the render-pass majority
 * of the frame goes unmeasured. This sampler takes the one measurement this
 * program already trusts — submit-to-fence queue wall — and applies it at
 * band grain: on a sampling frame the encode is split at band boundaries
 * into several submits, `performance.now()` is read as each fence retires,
 * and consecutive fence times difference into real wall costs per band.
 *
 * The queue completes submissions in order, so the fences are a prefix
 * chain: no await between submits, no CPU/GPU stall — the cost of a sampling
 * frame is a handful of extra submits and whatever pass overlap Dawn loses
 * across submit boundaries. That loss is also why a sampling frame must be
 * excluded from the rolling frame mean: it prices the bands, not the frame.
 *
 * A baseline fence is registered before the first band's submit. It retires
 * only after everything already in the queue — in-flight solver advances —
 * completes, so the first band is not billed for simulation work the way the
 * whole-frame queue wall is.
 */

export type FrameBandId =
  | "source"
  | "water-surface"
  | "svo-primary"
  | "svo-lighting"
  | "svo-shading"
  | "composite-present";

/** Band names → the closed phase-id union, so consumers aggregate as usual. */
const FRAME_BAND_PHASES: Record<FrameBandId, { id: PerformanceTrace["phases"][number]["id"]; label: string }> = {
  source: { id: "scene-upload", label: "Source: world maintenance + fluid coverage" },
  "water-surface": { id: "surface-extraction", label: "Water surface extraction + caustics" },
  "svo-primary": { id: "svo-primary", label: "Primary visibility" },
  "svo-lighting": { id: "svo-cone-lighting", label: "Lighting visibility" },
  "svo-shading": { id: "dry-scene", label: "Deferred shading" },
  "composite-present": { id: "optical-composite", label: "Interfaces + composite + present" },
};

/**
 * What the encode path sees. Callers whose callees may cross a boundary must
 * resynchronize their local encoder from `current` afterwards — an encoder is
 * finished by the boundary that submits it, and encoding into a finished
 * encoder is a validation error.
 */
export interface FrameBandPartitioner {
  /** Submit everything since the previous boundary as `band`; encode on into the returned encoder. */
  boundary(band: FrameBandId): GPUCommandEncoder;
  /** The encoder the frame is currently accumulating into. */
  readonly current: GPUCommandEncoder;
}

interface BandFence {
  band: FrameBandId;
  completedAt_ms?: number;
}

export class FencePartitionedFrameSampler implements FrameBandPartitioner {
  private encoder: GPUCommandEncoder;
  private readonly fences: BandFence[] = [];
  private baselineAt_ms?: number;

  constructor(
    private readonly device: GPUDevice,
    firstEncoder: GPUCommandEncoder,
    private readonly sampleId: number,
    private readonly context: string,
  ) {
    this.encoder = firstEncoder;
    // Everything already queued (solver advances) retires before this does,
    // so the first band's wall starts after the simulation, not around it.
    void device.queue.onSubmittedWorkDone().then(() => {
      this.baselineAt_ms = performance.now();
    }).catch(() => { /* Device loss voids the sample; finish() returns undefined. */ });
  }

  get current(): GPUCommandEncoder {
    return this.encoder;
  }

  boundary(band: FrameBandId): GPUCommandEncoder {
    this.device.queue.submit([this.encoder.finish()]);
    const fence: BandFence = { band };
    this.fences.push(fence);
    void this.device.queue.onSubmittedWorkDone().then(() => {
      fence.completedAt_ms = performance.now();
    }).catch(() => { /* Device loss voids the sample. */ });
    this.encoder = this.device.createCommandEncoder({ label: `Fluid Lab frame · after ${band}` });
    return this.encoder;
  }

  /**
   * Close the ledger over the frame's own final submission. The caller keeps
   * ownership of that submit (startup publication and advance retirement hang
   * off its fence); this only rides the completion it already awaits.
   */
  finish(finalBand: FrameBandId, completion: Promise<unknown>): Promise<PerformanceTrace | undefined> {
    const fence: BandFence = { band: finalBand };
    this.fences.push(fence);
    return completion.then(() => {
      fence.completedAt_ms = performance.now();
      return this.read();
    }).catch(() => undefined);
  }

  private read(): PerformanceTrace | undefined {
    const baseline = this.baselineAt_ms;
    if (baseline === undefined) return undefined;
    let previous = baseline;
    const phases: PerformanceTrace["phases"] = [];
    for (const fence of this.fences) {
      if (fence.completedAt_ms === undefined) return undefined;
      const named = FRAME_BAND_PHASES[fence.band];
      phases.push({
        id: named.id,
        label: named.label,
        duration_ms: Math.max(0, fence.completedAt_ms - previous),
      });
      previous = fence.completedAt_ms;
    }
    if (phases.length === 0) return undefined;
    return {
      sampleId: this.sampleId,
      domain: "gpu",
      lane: "presentation",
      context: this.context,
      capturedAt_ms: previous,
      measurementSource: "gpu-queue-wall",
      // The exact additive identity holds by construction: the total is the
      // last fence minus the baseline, and the phases are its differences.
      total_ms: phases.reduce((sum, phase) => sum + phase.duration_ms, 0),
      phases,
    };
  }
}
