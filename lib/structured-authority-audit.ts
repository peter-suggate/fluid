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

/** Fixed queue-copy ABI for one accepted-step authority snapshot. Keeping the
 * record tightly packed lets a long smoke run enqueue copies without mapping
 * or allocating a staging buffer per step. */
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
  strideBytes: 336,
} as const);

export interface StructuredGenerationAuditSnapshot {
  readonly structured: StructuredVelocityControlSnapshot;
  readonly boundary: StructuredBoundaryControlSnapshot;
  readonly fineHeader: Uint32Array;
  readonly mgpcgControl: Uint32Array;
  readonly fineVolumeControl: Uint32Array;
  readonly projectionEnergyControl: Uint32Array;
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
}

/** The ONE writer of the record ABI. Both consumers — the smoke harness's
 * per-step capacity buffer and the browser's step-coherent snapshot ring —
 * enqueue exactly these copies, so layout drift is structurally impossible. */
export function encodeStructuredAuditRecordCopies(
  encoder: GPUCommandEncoder,
  sources: StructuredAuditRecordSources,
  target: GPUBuffer,
  recordBaseBytes: number,
): void {
  const layout = STRUCTURED_GENERATION_AUDIT_SNAPSHOT;
  encoder.copyBufferToBuffer(sources.structuredVelocityControl, 0,
    target, recordBaseBytes + layout.structuredOffsetBytes, layout.structuredBytes);
  encoder.copyBufferToBuffer(sources.structuredBoundaryControl, 0,
    target, recordBaseBytes + layout.boundaryOffsetBytes, layout.boundaryBytes);
  encoder.copyBufferToBuffer(sources.fineWorklist, 0,
    target, recordBaseBytes + layout.fineOffsetBytes, layout.fineBytes);
  if (sources.mgpcgControl) encoder.copyBufferToBuffer(sources.mgpcgControl, 0,
    target, recordBaseBytes + layout.mgpcgOffsetBytes, layout.mgpcgBytes);
  if (sources.fineVolumeControl) encoder.copyBufferToBuffer(sources.fineVolumeControl, 0,
    target, recordBaseBytes + layout.fineVolumeOffsetBytes, layout.fineVolumeBytes);
  if (sources.projectionEnergyStats) encoder.copyBufferToBuffer(sources.projectionEnergyStats, 0,
    target, recordBaseBytes + layout.projectionEnergyOffsetBytes, layout.projectionEnergyBytes);
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
