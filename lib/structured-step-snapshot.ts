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
  unpackAirSupportFailure,
  unpackFineTransportGovernor,
  unpackFineWorklistHeader,
  unpackSPGridLevelDeltas,
  unpackStructuredGenerationAuditSnapshot,
  unpackTopologyEpochState,
  type StructuredAuditRecordSegment,
  type StructuredAuditRecordSources,
  type StructuredGenerationAuditSnapshot,
} from "./structured-authority-audit";

export type StructuredStepSnapshotSources = StructuredAuditRecordSources;

/**
 * Maximum supported advances a driver may keep in flight. Interactive drivers
 * can use a lower latency-oriented depth, and the Dawn harness fences well
 * inside it. The ring is sized from this rather than
 * from a literal because an under-sized ring is not a diagnostics degradation:
 * every slot can legally be mapping, `encode` then SKIPS the record, and a
 * missing `step-snapshot` stage latches a permanent step-sequence fault
 * (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A4.2).
 */
export const MAXIMUM_PENDING_PHYSICS_ADVANCES = 8;

/** One slot per in-flight advance plus one for the consumer's mapped record. */
export function structuredStepSnapshotSlotCount(
  maxPendingAdvances = MAXIMUM_PENDING_PHYSICS_ADVANCES,
): number {
  const bound = Number.isFinite(maxPendingAdvances)
    ? Math.floor(maxPendingAdvances) : MAXIMUM_PENDING_PHYSICS_ADVANCES;
  return Math.max(2, Math.min(MAXIMUM_PENDING_PHYSICS_ADVANCES, Math.max(1, bound)) + 1);
}

/** Host clock captured at encode time for the step the record describes. */
export interface StructuredStepSnapshotStamp {
  readonly step: number;
  readonly dt_s: number;
  readonly submittedTime_s: number;
  readonly hostFineGeneration: number;
  /** Coarse-1 stores its compact-directory header in the fixed scalar slot;
   * it never allocates a fine worklist merely to satisfy diagnostics. */
  readonly surfaceKind?: "fine" | "coarse";
}

export interface StructuredStepSnapshotRecord {
  readonly stamp: StructuredStepSnapshotStamp;
  readonly snapshot: StructuredGenerationAuditSnapshot;
  /**
   * Segments the step's encoder actually copied. A segment that is absent
   * decodes as zero, which is indistinguishable from a real zero counter —
   * and a zero counter is exactly what a Part-B skip predicate wants to see.
   * Consumers must therefore require presence before treating a zero as
   * evidence.
   */
  readonly presentSegments: readonly StructuredAuditRecordSegment[];
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
  segments: readonly StructuredAuditRecordSegment[];
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
  private skipped = 0;
  private due = true;

  constructor(
    private readonly device: GPUDevice,
    slotCount = structuredStepSnapshotSlotCount(),
  ) {
    this.slots = Array.from({ length: Math.max(2, slotCount) }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Structured step snapshot slot ${index}`,
        size: STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes,
        usage: BUFFER_USAGE_COPY_DST | BUFFER_USAGE_MAP_READ,
      }),
      segments: Object.freeze([]),
      sequence: 0,
      state: "free",
    }));
  }

  get slotCount() { return this.slots.length; }

  /**
   * Records the producer had to drop because every slot was mapping. The ring
   * is sized so this stays zero; a non-zero value is a contract break, not a
   * diagnostics gap, because the skipped step encodes no `step-snapshot`
   * stage and therefore latches a step-sequence deviation.
   */
  get skippedRecords() { return this.skipped; }

  /**
   * Whether the next step should spend its tail copying a record. Arming
   * follows the consumer: see `WebGPUOctreeLosassoStepSnapshotRing.recordDue`
   * for why a per-step producer was serving a 250 ms consumer.
   */
  get recordDue() { return this.due; }

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
    if (!slot) { this.skipped += 1; return false; }
    slot.segments = encodeStructuredAuditRecordCopies(encoder, sources, slot.buffer, 0);
    slot.stamp = stamp;
    slot.sequence = ++this.sequence;
    slot.state = "encoded";
    this.due = false;
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
    // Re-arm on every consumer visit, including one that found nothing: a
    // reader asking for a record is what makes the next step owe one.
    this.due = true;
    if (!slot?.stamp) return undefined;
    slot.state = "mapping";
    try {
      await slot.buffer.mapAsync(MAP_MODE_READ);
      const bytes = new Uint8Array(slot.buffer.getMappedRange().slice(0));
      return {
        stamp: slot.stamp,
        snapshot: unpackStructuredGenerationAuditSnapshot(bytes, 0),
        presentSegments: slot.segments,
      };
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
  /**
   * Fine active-brick count, worklist header word ONE. Word zero is the
   * generation; decoding it as a count is the bug named in A3/A4.3 — it
   * reported a rising generation as band occupancy and masked the
   * 0xFFFFFFFF capacity-overflow sentinel, which no-ops the solver while the
   * run still prints PASS.
   */
  readonly activeFineBricks: number;
  readonly fineBandCapacityOverflow: boolean;
  /** Non-zero epoch error poisons the NEXT flip: the accepted epoch is
   * retained and transport keeps consuming the old velocity bank. */
  readonly topologyEpochError: number;
  readonly topologyFlipReady: boolean;
  readonly airSupportFailed: boolean;
  readonly transportScheduleValid: boolean;
}

export function structuredAuthorityStepHealth(
  record: StructuredStepSnapshotRecord,
): StructuredAuthorityStepHealth {
  const { structured, boundary, fineHeader } = record.snapshot;
  const fine = unpackFineWorklistHeader(fineHeader);
  const coarseOnly = record.stamp.surfaceKind === "coarse";
  const epochState = unpackTopologyEpochState(record.snapshot.topologyEpochState);
  const airSupport = unpackAirSupportFailure(record.snapshot.airSupportFailure);
  const governor = unpackFineTransportGovernor(record.snapshot.fineTransportGovernor);
  const publishedFineGeneration = coarseOnly
    // The compact coarse-directory generation shares its high two bits with
    // the publication state. Compare only the 30-bit generation payload.
    ? (Number(fineHeader[1] ?? 0) & 0x3fff_ffff) >>> 0 : fine.generation;
  const velocityValid = structured.flags === 0 && structured.rowCount > 0
    && structured.epoch > 0 && structured.activeBank <= 1;
  const boundaryValid = boundary.flags === 0
    && boundary.rowCount === structured.rowCount
    && boundary.epoch === structured.epoch
    && boundary.activeBank === structured.activeBank
    && boundary.publishedEpoch === structured.epoch;
  const has = (segment: StructuredAuditRecordSegment) =>
    record.presentSegments.includes(segment);
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
    activeFineBricks: coarseOnly ? 0 : fine.capacityOverflow ? 0 : fine.activeBricks,
    fineBandCapacityOverflow: coarseOnly ? false : fine.capacityOverflow,
    topologyEpochError: epochState.error,
    topologyFlipReady: epochState.flipReady,
    // An absent segment must never read as "healthy zero": the producer only
    // reports failure words it actually copied.
    airSupportFailed: has("airSupportFailure") && airSupport.failed,
    transportScheduleValid: !has("fineTransportGovernor") || governor.scheduleValid,
  });
}

/**
 * GPU counters for one step, decoded from that step's own record — the
 * evidence side of the P0.5 predicted-work check. Every field is `undefined`
 * when its record segment was not copied, so a conductor can never mistake an
 * uncopied segment for a zero-work verdict.
 */
export interface StructuredStepWorkObservation {
  readonly step: number;
  /** MGPCG control word 2: outer iterations the GPU actually executed. */
  readonly executedSolveIterations?: number;
  /** MGPCG control word 1. */
  readonly solveConverged?: boolean;
  readonly acceptedEpoch: number;
  readonly topologyHash?: number;
  /** Topology-epoch verdict: the flip token and its poison word. */
  readonly topologyFlipReady?: boolean;
  readonly topologyEpochError?: number;
  /** Per-level SPGrid setup-delta dirty words; all zero means no rebuilt level. */
  readonly spgridLevelDirty?: readonly number[];
  readonly airSupportErrorFlags?: number;
  readonly fineActiveBricks?: number;
  readonly fineBandCapacityOverflow: boolean;
  /** Fine-transport governor: schedule validity and the substeps it planned. */
  readonly transportScheduleValid?: boolean;
  readonly transportActiveSubsteps?: number;
}

const MGPCG_CONTROL = Object.freeze({ flags: 0, converged: 1, iterations: 2, rows: 4 } as const);

export function structuredStepWorkObservation(
  record: StructuredStepSnapshotRecord,
): StructuredStepWorkObservation {
  const has = (segment: StructuredAuditRecordSegment) =>
    record.presentSegments.includes(segment);
  const fine = unpackFineWorklistHeader(record.snapshot.fineHeader);
  const coarseOnly = record.stamp.surfaceKind === "coarse";
  const epochState = unpackTopologyEpochState(record.snapshot.topologyEpochState);
  const airSupport = unpackAirSupportFailure(record.snapshot.airSupportFailure);
  const governor = unpackFineTransportGovernor(record.snapshot.fineTransportGovernor);
  const mgpcg = record.snapshot.mgpcgControl;
  return Object.freeze({
    step: record.stamp.step,
    executedSolveIterations: has("mgpcg")
      ? Number(mgpcg[MGPCG_CONTROL.iterations] ?? 0) >>> 0 : undefined,
    solveConverged: has("mgpcg")
      ? Number(mgpcg[MGPCG_CONTROL.converged] ?? 0) !== 0 : undefined,
    acceptedEpoch: record.snapshot.structured.epoch,
    topologyHash: has("topologyEpoch") ? epochState.topologyHash : undefined,
    topologyFlipReady: has("topologyEpoch") ? epochState.flipReady : undefined,
    topologyEpochError: has("topologyEpoch") ? epochState.error : undefined,
    spgridLevelDirty: has("spgridLevelDelta")
      ? unpackSPGridLevelDeltas(record.snapshot.spgridLevelDelta)
        .map((level) => level.dirtyFlags)
      : undefined,
    airSupportErrorFlags: has("airSupportFailure") ? airSupport.errorFlags : undefined,
    fineActiveBricks: has("fine") && !coarseOnly
      ? (fine.capacityOverflow ? 0 : fine.activeBricks) : undefined,
    fineBandCapacityOverflow: has("fine") && !coarseOnly && fine.capacityOverflow,
    transportScheduleValid: has("fineTransportGovernor")
      ? governor.scheduleValid : undefined,
    transportActiveSubsteps: has("fineTransportGovernor")
      ? governor.activeSubsteps : undefined,
  });
}
