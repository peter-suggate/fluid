/** Exact direct-structured publication audit used by smoke and performance gates. */
export interface StructuredVelocityControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly epoch: number;
  readonly activeBank: number;
  readonly slotCount: number;
}

export interface StructuredBoundaryControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly slotCount: number;
  readonly epoch: number;
  readonly activeBank: number;
  readonly publishedEpoch: number;
}

export function unpackStructuredVelocityControl(words: ArrayLike<number>): StructuredVelocityControlSnapshot {
  if (words.length < 6) throw new RangeError("structured velocity control requires six words");
  return Object.freeze({ flags: Number(words[0]) >>> 0, firstError: Number(words[1]) >>> 0,
    rowCount: Number(words[2]) >>> 0, epoch: Number(words[3]) >>> 0,
    activeBank: Number(words[4]) >>> 0, slotCount: Number(words[5]) >>> 0 });
}

export function unpackStructuredBoundaryControl(words: ArrayLike<number>): StructuredBoundaryControlSnapshot {
  if (words.length < 7) throw new RangeError("structured boundary control requires seven words");
  return Object.freeze({ flags: Number(words[0]) >>> 0, firstError: Number(words[1]) >>> 0,
    rowCount: Number(words[2]) >>> 0, slotCount: Number(words[3]) >>> 0,
    epoch: Number(words[4]) >>> 0, activeBank: Number(words[5]) >>> 0,
    publishedEpoch: Number(words[6]) >>> 0 });
}

export interface ExactStructuredGenerationAudit {
  readonly publishedFineGeneration: number;
  readonly expectedStructuredEpoch: number;
  readonly previousFineGeneration: number;
  readonly previousStructuredEpoch: number;
  readonly structured: StructuredVelocityControlSnapshot;
  readonly boundary: StructuredBoundaryControlSnapshot;
}

export function exactStructuredGenerationAuditFailures(
  audit: ExactStructuredGenerationAudit,
): readonly string[] {
  const failures: string[] = [];
  const positive = (value: number) => Number.isSafeInteger(value) && value > 0;
  if (!positive(audit.publishedFineGeneration)) failures.push("invalid published fine generation");
  if (!positive(audit.expectedStructuredEpoch)) failures.push("invalid structured epoch");
  if (audit.publishedFineGeneration !== audit.expectedStructuredEpoch + 1) {
    failures.push("published fine generation is not the structured successor");
  }
  if (audit.publishedFineGeneration <= audit.previousFineGeneration) {
    failures.push("fine generation did not advance");
  }
  if (audit.expectedStructuredEpoch <= audit.previousStructuredEpoch) {
    failures.push("structured epoch did not advance");
  }
  if (audit.structured.flags !== 0 || audit.structured.firstError !== 0xffff_ffff
    || audit.structured.epoch !== audit.expectedStructuredEpoch
    || audit.structured.activeBank > 1 || audit.structured.rowCount === 0
    || audit.structured.slotCount === 0) {
    failures.push("structured velocity publication is invalid");
  }
  if (audit.boundary.flags !== 0 || audit.boundary.firstError !== 0xffff_ffff
    || audit.boundary.epoch !== audit.expectedStructuredEpoch
    || audit.boundary.publishedEpoch !== audit.expectedStructuredEpoch
    || audit.boundary.activeBank !== audit.structured.activeBank
    || audit.boundary.rowCount !== audit.structured.rowCount
    || audit.boundary.slotCount !== audit.structured.slotCount) {
    failures.push("structured boundary publication is invalid or incoherent");
  }
  return failures;
}

/** Words per SPGrid level in the exact setup-delta publication
 * (`LEVEL_DELTA_WORDS` in `webgpu-octree-spgrid-vcycle.ts`). */
export const SPGRID_LEVEL_DELTA_WORDS_PER_LEVEL = 8;
/** `planOctreeSPGridVCycle` rejects deeper hierarchies, so reserving twelve
 * levels makes the record segment a fixed ABI even though the live buffer is
 * sized to the scene's level count. */
export const SPGRID_LEVEL_DELTA_RECORD_LEVELS = 12;

/** Fixed queue-copy ABI for one accepted-step authority snapshot. Keeping the
 * record tightly packed lets a long smoke run enqueue copies without mapping
 * or allocating a staging buffer per step.
 *
 * P0.4 (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A4) appended the four segments
 * below the projection-energy block. They are the GPU's own verdict words for
 * the subsystems Part B intends to gate: without them a host-side skip has no
 * step-coherent evidence that the work it deleted was actually zero. */
export const STRUCTURED_GENERATION_AUDIT_SNAPSHOT = Object.freeze({
  structuredOffsetBytes: 0,
  structuredBytes: 24,
  boundaryOffsetBytes: 24,
  boundaryBytes: 28,
  fineOffsetBytes: 52,
  fineBytes: 28,
  mgpcgOffsetBytes: 80,
  mgpcgBytes: 64,
  fineVolumeOffsetBytes: 144,
  fineVolumeBytes: 64,
  projectionEnergyOffsetBytes: 208,
  projectionEnergyBytes: 128,
  /** `WebGPUOctreeTopologyEpoch.state`: the whole 16-word Epoch struct. */
  topologyEpochOffsetBytes: 336,
  topologyEpochBytes: 64,
  /** SPGrid per-level setup delta, twelve reserved levels of eight words. */
  spgridLevelDeltaOffsetBytes: 400,
  spgridLevelDeltaBytes: SPGRID_LEVEL_DELTA_RECORD_LEVELS * SPGRID_LEVEL_DELTA_WORDS_PER_LEVEL * 4,
  /** Air-support producer `scratch[0..1]`: error flags + first failing item. */
  airSupportFailureOffsetBytes: 784,
  airSupportFailureBytes: 8,
  /** Fine-transport `governor` words 0..3: schedule validity + substeps. */
  fineTransportGovernorOffsetBytes: 792,
  fineTransportGovernorBytes: 16,
  strideBytes: 808,
} as const);

export interface StructuredGenerationAuditSnapshot {
  readonly structured: StructuredVelocityControlSnapshot;
  readonly boundary: StructuredBoundaryControlSnapshot;
  readonly fineHeader: Uint32Array;
  readonly mgpcgControl: Uint32Array;
  readonly fineVolumeControl: Uint32Array;
  readonly projectionEnergyControl: Uint32Array;
  /** P0.4 segments. Absent sources leave these all-zero; the producer reports
   * which segments it actually copied out of band (see the snapshot ring's
   * `presentSegments`), so a zero is never mistaken for evidence. */
  readonly topologyEpochState: Uint32Array;
  readonly spgridLevelDelta: Uint32Array;
  readonly airSupportFailure: Uint32Array;
  readonly fineTransportGovernor: Uint32Array;
}

/** Live control buffers one record is copied from. The optional members are
 * absent only before the corresponding subsystem is constructed; their record
 * segments then stay zero and decode as empty controls. */
export interface StructuredAuditRecordSources {
  readonly structuredVelocityControl: GPUBuffer;
  readonly structuredBoundaryControl: GPUBuffer;
  readonly fineWorklist: GPUBuffer;
  readonly mgpcgControl?: GPUBuffer;
  readonly fineVolumeControl?: GPUBuffer;
  readonly projectionEnergyStats?: GPUBuffer;
  /** P0.4: `WebGPUOctreeTopologyEpoch.state`. */
  readonly topologyEpochState?: GPUBuffer;
  /** P0.4: the SPGrid V-cycle's per-level setup delta buffer. */
  readonly spgridLevelDelta?: GPUBuffer;
  /** P0.4: `WebGPUOctreeAirVelocitySupportProducer.scratch`. */
  readonly airSupportScratch?: GPUBuffer;
  /** P0.4: `WebGPUFineLevelSetTransport.governor`. */
  readonly fineTransportGovernor?: GPUBuffer;
}

/** Segment ids a producer reports as actually copied for one record. */
export type StructuredAuditRecordSegment =
  | "structured" | "boundary" | "fine" | "mgpcg" | "fineVolume" | "projectionEnergy"
  | "topologyEpoch" | "spgridLevelDelta" | "airSupportFailure" | "fineTransportGovernor";

/** The ONE writer of the record ABI. Both consumers — the smoke harness's
 * per-step capacity buffer and the browser's step-coherent snapshot ring —
 * enqueue exactly these copies, so layout drift is structurally impossible.
 *
 * Every copy reads a producer's end-of-step value because the whole record is
 * encoded as the advance's last commands; appending a segment here is only
 * legal while that stays true (`step-snapshot` is the final program stage).
 * Returns the segments it copied so a consumer can tell an absent segment
 * from a genuinely zero one.
 */
export function encodeStructuredAuditRecordCopies(
  encoder: GPUCommandEncoder,
  sources: StructuredAuditRecordSources,
  target: GPUBuffer,
  recordBaseBytes: number,
): readonly StructuredAuditRecordSegment[] {
  const layout = STRUCTURED_GENERATION_AUDIT_SNAPSHOT;
  const copied: StructuredAuditRecordSegment[] = [];
  const copy = (
    segment: StructuredAuditRecordSegment,
    source: GPUBuffer | undefined,
    offsetBytes: number,
    segmentBytes: number,
  ) => {
    if (!source) return;
    // A variable-length producer (the SPGrid delta is sized by the scene's
    // level count) copies only what it owns; the reserved tail stays zero.
    const bytes = Math.min(segmentBytes, Math.floor((source.size ?? segmentBytes) / 4) * 4);
    if (bytes <= 0) return;
    encoder.copyBufferToBuffer(source, 0, target, recordBaseBytes + offsetBytes, bytes);
    copied.push(segment);
  };
  copy("structured", sources.structuredVelocityControl,
    layout.structuredOffsetBytes, layout.structuredBytes);
  copy("boundary", sources.structuredBoundaryControl,
    layout.boundaryOffsetBytes, layout.boundaryBytes);
  copy("fine", sources.fineWorklist, layout.fineOffsetBytes, layout.fineBytes);
  copy("mgpcg", sources.mgpcgControl, layout.mgpcgOffsetBytes, layout.mgpcgBytes);
  copy("fineVolume", sources.fineVolumeControl,
    layout.fineVolumeOffsetBytes, layout.fineVolumeBytes);
  copy("projectionEnergy", sources.projectionEnergyStats,
    layout.projectionEnergyOffsetBytes, layout.projectionEnergyBytes);
  copy("topologyEpoch", sources.topologyEpochState,
    layout.topologyEpochOffsetBytes, layout.topologyEpochBytes);
  copy("spgridLevelDelta", sources.spgridLevelDelta,
    layout.spgridLevelDeltaOffsetBytes, layout.spgridLevelDeltaBytes);
  copy("airSupportFailure", sources.airSupportScratch,
    layout.airSupportFailureOffsetBytes, layout.airSupportFailureBytes);
  copy("fineTransportGovernor", sources.fineTransportGovernor,
    layout.fineTransportGovernorOffsetBytes, layout.fineTransportGovernorBytes);
  return copied;
}

/** Decode one record copied by the smoke harness after an accepted step. */
export function unpackStructuredGenerationAuditSnapshot(
  bytes: Uint8Array,
  recordIndex: number,
): StructuredGenerationAuditSnapshot {
  if (!Number.isSafeInteger(recordIndex) || recordIndex < 0) {
    throw new RangeError("structured generation audit record index must be non-negative");
  }
  const layout = STRUCTURED_GENERATION_AUDIT_SNAPSHOT;
  const base = recordIndex * layout.strideBytes;
  if (base + layout.strideBytes > bytes.byteLength) {
    throw new RangeError("structured generation audit snapshot is truncated");
  }
  const words = (offsetBytes: number, byteLength: number) => new Uint32Array(
    bytes.buffer,
    bytes.byteOffset + base + offsetBytes,
    byteLength / 4,
  );
  return Object.freeze({
    structured: unpackStructuredVelocityControl(
      words(layout.structuredOffsetBytes, layout.structuredBytes)),
    boundary: unpackStructuredBoundaryControl(
      words(layout.boundaryOffsetBytes, layout.boundaryBytes)),
    fineHeader: words(layout.fineOffsetBytes, layout.fineBytes).slice(),
    mgpcgControl: words(layout.mgpcgOffsetBytes, layout.mgpcgBytes).slice(),
    fineVolumeControl: words(layout.fineVolumeOffsetBytes, layout.fineVolumeBytes).slice(),
    projectionEnergyControl:
      words(layout.projectionEnergyOffsetBytes, layout.projectionEnergyBytes).slice(),
    topologyEpochState:
      words(layout.topologyEpochOffsetBytes, layout.topologyEpochBytes).slice(),
    spgridLevelDelta:
      words(layout.spgridLevelDeltaOffsetBytes, layout.spgridLevelDeltaBytes).slice(),
    airSupportFailure:
      words(layout.airSupportFailureOffsetBytes, layout.airSupportFailureBytes).slice(),
    fineTransportGovernor:
      words(layout.fineTransportGovernorOffsetBytes, layout.fineTransportGovernorBytes).slice(),
  });
}

/** Word map of the copied fine workset publication header.
 *
 * The active-brick count is word ONE. Word zero is the generation; reading it
 * as a count is the decode bug called out in A3/A4.3 — it silently reported a
 * monotonically rising "brick count" and hid both an empty band and the
 * 0xFFFFFFFF capacity-overflow sentinel. */
export const STRUCTURED_FINE_WORKLIST_HEADER = Object.freeze({
  generation: 0, activeBricks: 1, capacity: 2, flags: 3,
  dispatchX: 4, dispatchY: 5, dispatchZ: 6,
} as const);

/** Sentinel the fine band writes when the active set overflows its capacity.
 * A run that reports it is not a fast run: the solver silently no-ops. */
export const FINE_WORKLIST_CAPACITY_OVERFLOW = 0xffff_ffff;

export interface FineWorklistHeaderSnapshot {
  readonly generation: number;
  readonly activeBricks: number;
  readonly capacity: number;
  readonly flags: number;
  /** True when the count is the overflow sentinel rather than a real count. */
  readonly capacityOverflow: boolean;
  /** Both publication bits set and the count inside capacity. */
  readonly published: boolean;
}

export function unpackFineWorklistHeader(
  words: ArrayLike<number>,
): FineWorklistHeaderSnapshot {
  const word = (index: number): number => Number(words[index] ?? 0) >>> 0;
  const activeBricks = word(STRUCTURED_FINE_WORKLIST_HEADER.activeBricks);
  const capacity = word(STRUCTURED_FINE_WORKLIST_HEADER.capacity);
  const flags = word(STRUCTURED_FINE_WORKLIST_HEADER.flags);
  const capacityOverflow = activeBricks === FINE_WORKLIST_CAPACITY_OVERFLOW;
  return Object.freeze({
    generation: word(STRUCTURED_FINE_WORKLIST_HEADER.generation),
    activeBricks, capacity, flags, capacityOverflow,
    published: (flags & 3) === 3 && !capacityOverflow
      && activeBricks > 0 && activeBricks <= capacity,
  });
}

/** Word map of the `Epoch` struct in `octreeTopologyEpochWGSL`. */
export const OCTREE_TOPOLOGY_EPOCH_RECORD = Object.freeze({
  activeEpoch: 0, activeGeneration: 1, readyEpoch: 2, readyGeneration: 3,
  error: 4, ready: 5, rowCount: 6, topologyHash: 7,
} as const);

export interface TopologyEpochStateSnapshot {
  readonly activeEpoch: number;
  readonly activeGeneration: number;
  readonly readyEpoch: number;
  readonly readyGeneration: number;
  /** Non-zero poisons the next flip; the accepted epoch is then retained. */
  readonly error: number;
  /** The clean-commit token: the next `ready-topology-flip` will adopt. */
  readonly flipReady: boolean;
  readonly rowCount: number;
  readonly topologyHash: number;
}

export function unpackTopologyEpochState(
  words: ArrayLike<number>,
): TopologyEpochStateSnapshot {
  const word = (index: number): number => Number(words[index] ?? 0) >>> 0;
  return Object.freeze({
    activeEpoch: word(OCTREE_TOPOLOGY_EPOCH_RECORD.activeEpoch),
    activeGeneration: word(OCTREE_TOPOLOGY_EPOCH_RECORD.activeGeneration),
    readyEpoch: word(OCTREE_TOPOLOGY_EPOCH_RECORD.readyEpoch),
    readyGeneration: word(OCTREE_TOPOLOGY_EPOCH_RECORD.readyGeneration),
    error: word(OCTREE_TOPOLOGY_EPOCH_RECORD.error),
    flipReady: word(OCTREE_TOPOLOGY_EPOCH_RECORD.ready) !== 0,
    rowCount: word(OCTREE_TOPOLOGY_EPOCH_RECORD.rowCount),
    topologyHash: word(OCTREE_TOPOLOGY_EPOCH_RECORD.topologyHash),
  });
}

/** Per-level SPGrid setup delta: word 0 is the level's capture generation and
 * word 1 its dirty flags. A settled step rebuilds no level, so every level's
 * dirty word is zero — the exact evidence a Part-B rebuild skip needs. */
export const SPGRID_LEVEL_DELTA_RECORD = Object.freeze({
  generation: 0, dirtyFlags: 1, firstRow: 2, rowEnd: 3,
  candidateError: 4, commitGeneration: 5, topologyGeneration: 6, spectralBits: 7,
} as const);

export interface SPGridLevelDeltaSnapshot {
  readonly level: number;
  readonly generation: number;
  readonly dirtyFlags: number;
  readonly candidateError: number;
  readonly topologyGeneration: number;
}

/** Decode `levelCount` levels out of the reserved twelve-level segment. */
export function unpackSPGridLevelDeltas(
  words: ArrayLike<number>,
  levelCount = SPGRID_LEVEL_DELTA_RECORD_LEVELS,
): readonly SPGridLevelDeltaSnapshot[] {
  const levels = Math.max(0, Math.min(SPGRID_LEVEL_DELTA_RECORD_LEVELS, Math.floor(levelCount)));
  const word = (level: number, offset: number): number =>
    Number(words[level * SPGRID_LEVEL_DELTA_WORDS_PER_LEVEL + offset] ?? 0) >>> 0;
  return Object.freeze(Array.from({ length: levels }, (_, level) => Object.freeze({
    level,
    generation: word(level, SPGRID_LEVEL_DELTA_RECORD.generation),
    dirtyFlags: word(level, SPGRID_LEVEL_DELTA_RECORD.dirtyFlags),
    candidateError: word(level, SPGRID_LEVEL_DELTA_RECORD.candidateError),
    topologyGeneration: word(level, SPGRID_LEVEL_DELTA_RECORD.topologyGeneration),
  })));
}

export interface AirSupportFailureSnapshot {
  /** `scratch[0]`: OR of every producer error flag raised this step. */
  readonly errorFlags: number;
  /** `scratch[1]`: `(stage << 24) | item` of the first failing item. */
  readonly firstFailure: number;
  readonly failed: boolean;
  readonly stage: number;
  readonly item: number;
}

export function unpackAirSupportFailure(
  words: ArrayLike<number>,
): AirSupportFailureSnapshot {
  const errorFlags = Number(words[0] ?? 0) >>> 0;
  const firstFailure = Number(words[1] ?? 0) >>> 0;
  return Object.freeze({
    errorFlags, firstFailure, failed: errorFlags !== 0,
    stage: firstFailure >>> 24, item: firstFailure & 0x00ff_ffff,
  });
}

/**
 * Fine-transport governor words 0..3, written by
 * `planStructuredFineTransportSubsteps`. Word 0 is a REJECT word, not a valid
 * word: the kernel stores `select(1u, 0u, scheduleValid)` and every consumer
 * gates on `state[0] == 0u`. Word 1 is the active substep count, word 2 the
 * per-substep dt as f32 bits, word 3 the CFL displacement in cells.
 */
export interface FineTransportGovernorSnapshot {
  readonly words: readonly number[];
  readonly scheduleValid: boolean;
  readonly activeSubsteps: number;
  readonly substepDt_s: number;
  readonly displacementCells: number;
}

export function unpackFineTransportGovernor(
  words: ArrayLike<number>,
): FineTransportGovernorSnapshot {
  const word = (index: number): number => Number(words[index] ?? 0) >>> 0;
  const bits = Uint32Array.of(word(2));
  return Object.freeze({
    words: Object.freeze([word(0), word(1), word(2), word(3)]),
    scheduleValid: word(0) === 0,
    activeSubsteps: word(1),
    substepDt_s: new Float32Array(bits.buffer)[0] ?? 0,
    displacementCells: word(3),
  });
}

export interface FinalPerformanceAuthorityAudit {
  readonly expectedSteps: number;
  readonly observedSteps: number;
  readonly expectedTime_s: number;
  readonly targetTime_s: number;
  readonly submittedTime_s: number;
  readonly fineSourceGeneration: number;
  /** Seven-word [epoch,count,capacity,flags,dispatch xyz] publication header. */
  readonly fineWorklistHeader: ArrayLike<number>;
  readonly finePageCapacity: number;
  readonly structured: StructuredVelocityControlSnapshot;
  readonly boundary: StructuredBoundaryControlSnapshot;
}

export function finalPerformanceAuthorityFailures(
  audit: FinalPerformanceAuthorityAudit,
): readonly string[] {
  const failures: string[] = [];
  const exactTime = (observed: number, expected: number) =>
    Number.isFinite(observed) && Math.abs(observed - expected) <= 1e-9;
  if (!Number.isSafeInteger(audit.expectedSteps) || audit.expectedSteps < 1) {
    failures.push("missing exact positive step contract");
    return failures;
  }
  if (audit.observedSteps !== audit.expectedSteps) failures.push("accepted step count is not exact");
  if (!Number.isFinite(audit.expectedTime_s) || audit.expectedTime_s <= 0) {
    failures.push("invalid exact time contract");
  } else if (!exactTime(audit.targetTime_s, audit.expectedTime_s)
    || !exactTime(audit.submittedTime_s, audit.expectedTime_s)) {
    failures.push("submitted time is not the exact step checkpoint");
  }

  const expectedStructuredEpoch = audit.expectedSteps + 1;
  const expectedFineGeneration = expectedStructuredEpoch + 1;
  if (audit.fineSourceGeneration !== expectedFineGeneration) {
    failures.push("fine source generation is not current for the exact step");
  }
  if (audit.fineWorklistHeader.length < 7) {
    failures.push("fine worklist publication header is missing");
  } else {
    const epoch = Number(audit.fineWorklistHeader[0]) >>> 0;
    const count = Number(audit.fineWorklistHeader[1]) >>> 0;
    const capacity = Number(audit.fineWorklistHeader[2]) >>> 0;
    const flags = Number(audit.fineWorklistHeader[3]) >>> 0;
    const groups = Number(audit.fineWorklistHeader[4]) >>> 0;
    if (epoch !== audit.fineSourceGeneration) failures.push("fine worklist epoch is stale");
    if ((flags & 3) !== 3) failures.push("fine worklist publication is invalid");
    if (capacity !== audit.finePageCapacity || count === 0 || count > capacity) {
      failures.push("fine worklist active-page count is invalid");
    }
    if (groups !== Math.ceil(count / 64)
      || (Number(audit.fineWorklistHeader[5]) >>> 0) !== 1
      || (Number(audit.fineWorklistHeader[6]) >>> 0) !== 1) {
      failures.push("fine worklist indirect dispatch is invalid");
    }
  }
  failures.push(...exactStructuredGenerationAuditFailures({
    publishedFineGeneration: audit.fineSourceGeneration,
    expectedStructuredEpoch,
    previousFineGeneration: expectedFineGeneration - 1,
    previousStructuredEpoch: expectedStructuredEpoch - 1,
    structured: audit.structured,
    boundary: audit.boundary,
  }));
  return failures;
}
