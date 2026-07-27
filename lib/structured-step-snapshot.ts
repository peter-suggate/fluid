/**
 * Step-coherent structured-authority snapshot ring.
 *
 * The Dawn smoke harness proves the pattern: enqueue plain buffer-to-buffer
 * copies of the accepted structured controls immediately after a step's
 * command buffer, and in-order queue semantics make every record an exact
 * end-of-step observation with zero fences and zero extra submissions when the
 * copies ride the step's own encoder. The browser previously sampled the SAME
 * buffers from an independent readback encoder racing in-flight steps, which
 * is how a legally mid-pipeline sample read a cleared row counter as zero and
 * how "authority lag" mixed a fenced GPU epoch with a live host counter.
 *
 * This ring encodes the copies inside the step encoder itself (the last
 * commands of the advance), so a mapped record always describes exactly one
 * step, stamped host-side with that step's clock. Diagnostics consumers decode
 * the record instead of racing live buffers; the record layout is the shared
 * `STRUCTURED_GENERATION_AUDIT_SNAPSHOT` ABI, byte-identical to the harness.
 */
import {
  STRUCTURED_GENERATION_AUDIT_SNAPSHOT,
  encodeStructuredAuditRecordCopies,
  unpackStructuredGenerationAuditSnapshot,
  type StructuredAuditRecordSources,
  type StructuredGenerationAuditSnapshot,
} from "./structured-authority-audit";

export type StructuredStepSnapshotSources = StructuredAuditRecordSources;

/** Host clock captured at encode time for the step the record describes. */
export interface StructuredStepSnapshotStamp {
  readonly step: number;
  readonly dt_s: number;
  readonly submittedTime_s: number;
  readonly hostFineGeneration: number;
}

export interface StructuredStepSnapshotRecord {
  readonly stamp: StructuredStepSnapshotStamp;
  readonly snapshot: StructuredGenerationAuditSnapshot;
}

type SlotState = "free" | "encoded" | "mapping";

/** WebGPU spec constants; the `GPUBufferUsage` namespace object is a browser /
 * Dawn global and does not exist under plain Node test runs. */
const BUFFER_USAGE_MAP_READ = 0x0001;
const BUFFER_USAGE_COPY_DST = 0x0008;
const MAP_MODE_READ = 0x0001;

interface SnapshotSlot {
  readonly buffer: GPUBuffer;
  stamp?: StructuredStepSnapshotStamp;
  sequence: number;
  state: SlotState;
}

/**
 * Fixed ring of tiny MAP_READ buffers, one record each. `encode` claims the
 * stalest non-mapping slot so a slow diagnostics consumer can never starve the
 * per-step producer; `readLatest` maps the freshest encoded slot, which by
 * mapAsync semantics resolves only after the step that wrote it completed.
 */
export class StructuredStepSnapshotRing {
  private readonly slots: SnapshotSlot[];
  private sequence = 0;
  private disposed = false;

  constructor(private readonly device: GPUDevice, slotCount = 3) {
    this.slots = Array.from({ length: Math.max(2, slotCount) }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Structured step snapshot slot ${index}`,
        size: STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes,
        usage: BUFFER_USAGE_COPY_DST | BUFFER_USAGE_MAP_READ,
      }),
      sequence: 0,
      state: "free",
    }));
  }

  /** Encode this step's record into the step's own command encoder. */
  encode(
    encoder: GPUCommandEncoder,
    sources: StructuredStepSnapshotSources,
    stamp: StructuredStepSnapshotStamp,
  ): boolean {
    if (this.disposed) return false;
    let slot: SnapshotSlot | undefined;
    for (const candidate of this.slots) {
      if (candidate.state === "mapping") continue;
      if (!slot || candidate.sequence < slot.sequence) slot = candidate;
    }
    if (!slot) return false;
    encodeStructuredAuditRecordCopies(encoder, sources, slot.buffer, 0);
    slot.stamp = stamp;
    slot.sequence = ++this.sequence;
    slot.state = "encoded";
    return true;
  }

  /** Map and decode the freshest encoded record; undefined when none is ready. */
  async readLatest(): Promise<StructuredStepSnapshotRecord | undefined> {
    if (this.disposed) return undefined;
    let slot: SnapshotSlot | undefined;
    for (const candidate of this.slots) {
      if (candidate.state !== "encoded") continue;
      if (!slot || candidate.sequence > slot.sequence) slot = candidate;
    }
    if (!slot?.stamp) return undefined;
    slot.state = "mapping";
    try {
      await slot.buffer.mapAsync(MAP_MODE_READ);
      const bytes = new Uint8Array(slot.buffer.getMappedRange().slice(0));
      return { stamp: slot.stamp, snapshot: unpackStructuredGenerationAuditSnapshot(bytes, 0) };
    } catch {
      // Device loss invalidates the ring; the renderer owns recovery.
      return undefined;
    } finally {
      if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
      slot.state = this.disposed ? "mapping" : "free";
      if (this.disposed) slot.buffer.destroy();
    }
  }

  destroy() {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots) {
      if (slot.state !== "mapping") slot.buffer.destroy();
    }
  }
}

/** Health of one step's accepted structured authority, decoded coherently. */
export interface StructuredAuthorityStepHealth {
  readonly step: number;
  readonly submittedTime_s: number;
  readonly acceptedEpoch: number;
  readonly acceptedRows: number;
  readonly publishedFineGeneration: number;
  /**
   * Exact whole-step authority lag: a healthy step publishes fine generation
   * `acceptedEpoch + 1`, so zero is current and each unit is one step the
   * accepted velocity authority is stale. Unlike the legacy computation this
   * compares two words copied at the SAME step boundary, so pipeline depth
   * and diagnostics cadence cannot inflate it.
   */
  readonly authorityLagSteps: number;
  readonly velocityValid: boolean;
  readonly boundaryValid: boolean;
  readonly receiptValid: boolean;
}

export function structuredAuthorityStepHealth(
  record: StructuredStepSnapshotRecord,
): StructuredAuthorityStepHealth {
  const { structured, boundary, fineHeader } = record.snapshot;
  const publishedFineGeneration = fineHeader[0] ?? 0;
  const velocityValid = structured.flags === 0 && structured.rowCount > 0
    && structured.epoch > 0 && structured.activeBank <= 1;
  const boundaryValid = boundary.flags === 0
    && boundary.rowCount === structured.rowCount
    && boundary.epoch === structured.epoch
    && boundary.activeBank === structured.activeBank
    && boundary.publishedEpoch === structured.epoch;
  return Object.freeze({
    step: record.stamp.step,
    submittedTime_s: record.stamp.submittedTime_s,
    acceptedEpoch: structured.epoch,
    acceptedRows: structured.rowCount,
    publishedFineGeneration,
    authorityLagSteps: Math.max(0, publishedFineGeneration - structured.epoch - 1),
    velocityValid,
    boundaryValid,
    receiptValid: velocityValid && boundaryValid,
  });
}
