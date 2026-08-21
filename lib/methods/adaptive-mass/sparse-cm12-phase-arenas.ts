import {
  sparseCM12PressureJournalLayout,
  type SparseCM12PressureJournalLayout,
} from "./sparse-cm12-pressure-journal";
import type { SparseCM12HotTopologyLayout } from "./sparse-cm12-hot-topology";

export const SPARSE_CM12_PHASE_ARENA_BRICK_FINE_RESOLUTION = 8;
export const SPARSE_CM12_PHASE_ARENA_PRESENTATION_PAGE_RESOLUTION = 8;
export const SPARSE_CM12_PHASE_ARENA_WORKGROUP_SIZE = 64;
export const SPARSE_CM12_PHASE_ARENA_PRESSURE_SCALAR_BYTES = 80;
export const SPARSE_CM12_PHASE_ARENA_DIAGNOSTIC_READBACK_BYTES = 300;

export type SparseCM12PhaseArenaOwner =
  | "PhysicsState"
  | "ScalarScratch"
  | "PressureCache"
  | "TopologyCandidate"
  | "Presentation"
  | "Diagnostics"
  | "HTP1";

export type SparseCM12BufferUsageName =
  | "STORAGE"
  | "UNIFORM"
  | "COPY_SRC"
  | "COPY_DST"
  | "MAP_READ";

export interface SparseCM12PhaseArenaRegion {
  readonly name: string;
  readonly offsetBytes: number;
  readonly sizeBytes: number;
  readonly element: "f32" | "u32" | "i32" | "bytes";
  readonly count: number;
  readonly persistence: "frame" | "topology-epoch" | "diagnostic-session" | "immutable";
}

export interface SparseCM12PhaseArenaBuffer {
  /** Unique physical-allocation identity. Equal ids would be an illegal alias. */
  readonly id: string;
  readonly owner: SparseCM12PhaseArenaOwner;
  readonly immutable: boolean;
  readonly usage: readonly SparseCM12BufferUsageName[];
  readonly sizeBytes: number;
  readonly regions: readonly SparseCM12PhaseArenaRegion[];
}

export type SparseCM12PhaseArenaAccess = "uniform" | "read-only-storage" | "storage";

export interface SparseCM12PhaseArenaBinding {
  readonly binding: number;
  readonly buffer: string;
  readonly owner: SparseCM12PhaseArenaOwner;
  readonly access: SparseCM12PhaseArenaAccess;
}

export type SparseCM12PhaseArenaPhase =
  | "transport-conditioning"
  | "pressure-preparation"
  | "pressure-solve"
  | "velocity-projection"
  | "topology-candidate"
  | "presentation-publication"
  | "diagnostics-copy";

export interface SparseCM12PhaseArenaBindGroup {
  readonly phase: SparseCM12PhaseArenaPhase;
  readonly bindings: readonly SparseCM12PhaseArenaBinding[];
}

export interface SparseCM12PressureHierarchyCapacity {
  readonly groupCount: number;
  readonly edgeCount: number;
}

export interface SparseCM12PhaseArenaPlannerInput {
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly brickCount: number;
  readonly candidateBrickCount: number;
  /** Must equal HTP1 directedEdgeCount. */
  readonly pressureEdgeCount: number;
  readonly pressureCoarseEdgeCount: number;
  readonly pressureHierarchy: readonly SparseCM12PressureHierarchyCapacity[];
  readonly pressureMembershipWords?: number;
  readonly pressureStaticTopologyWords?: number;
  /** Mutable accepted/shadow lists, page allocator, provenance, and activity. */
  readonly topologyCandidateControlWords?: number;
  readonly tracerCount?: number;
  readonly rigidCoupling?: boolean;
  readonly pressureJournal?: {
    /** Current resident capacity, including record zero's seed slot. */
    readonly iterationCapacity: number;
    readonly snapshotCapacity: number;
  };
  readonly presentation: {
    readonly metadataBytes: number;
    readonly worklistBytes: number;
    readonly payloadBytes: number;
  };
  readonly diagnosticsReadbackBytes?: number;
  readonly htp1: SparseCM12HotTopologyLayout;
}

export interface SparseCM12LegacyResidentStateLayout {
  readonly floatCount: number;
  readonly fields: Readonly<Record<string, { readonly offsetFloats: number; readonly floatCount: number }>>;
  readonly journalLayout: SparseCM12PressureJournalLayout;
}

export interface SparseCM12LegacyCapacityReceipt {
  readonly residentState: SparseCM12LegacyResidentStateLayout;
  readonly residentStateBytes: number;
  readonly conditioningBytes: number;
  readonly pressureScratchBytes: number;
  readonly candidateFieldBytes: number;
  readonly overlaidCandidateStateBytes: number;
  readonly partialReductionBytes: number;
  readonly scalarReductionBytes: number;
  readonly presentationBytes: number;
  readonly diagnosticsReadbackBytes: number;
}

export interface SparseCM12PhaseArenaMigrationEntry {
  readonly semantic: string;
  readonly legacyOffsetBytes: number;
  readonly targetBuffer: string;
  readonly targetOffsetBytes: number;
  readonly sizeBytes: number;
  readonly offsetPreserved: boolean;
}

export interface SparseCM12PhaseArenaPlan {
  readonly version: 1;
  readonly input: SparseCM12PhaseArenaPlannerInput;
  readonly buffers: readonly SparseCM12PhaseArenaBuffer[];
  readonly bindGroups: readonly SparseCM12PhaseArenaBindGroup[];
  readonly legacy: SparseCM12LegacyCapacityReceipt;
  readonly migration: readonly SparseCM12PhaseArenaMigrationEntry[];
  readonly allocatedBytes: number;
}

const checkedCount = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_fffe) {
    throw new RangeError(`${label} must be a non-negative addressable u32 count`);
  }
  return value;
};

const align = (value: number, alignment: number): number =>
  Math.ceil(value / alignment) * alignment;
const align4Floats = (value: number): number => align(value, 4);
const nonEmptyBufferBytes = (value: number): number => Math.max(4, align(value, 4));

class BufferBuilder {
  private at = 0;
  private readonly mutableRegions: SparseCM12PhaseArenaRegion[] = [];

  constructor(
    readonly id: string,
    readonly owner: SparseCM12PhaseArenaOwner,
    readonly immutable: boolean,
    readonly usage: readonly SparseCM12BufferUsageName[],
  ) {}

  add(
    name: string,
    count: number,
    element: SparseCM12PhaseArenaRegion["element"],
    persistence: SparseCM12PhaseArenaRegion["persistence"],
    alignmentBytes = 16,
  ): SparseCM12PhaseArenaRegion {
    checkedCount(count, `${this.id}.${name}.count`);
    const bytesPerElement = element === "bytes" ? 1 : 4;
    this.at = align(this.at, alignmentBytes);
    const region = Object.freeze({
      name, offsetBytes: this.at, sizeBytes: count * bytesPerElement,
      element, count, persistence,
    });
    this.mutableRegions.push(region);
    this.at += region.sizeBytes;
    return region;
  }

  finish(): SparseCM12PhaseArenaBuffer {
    return Object.freeze({
      id: this.id, owner: this.owner, immutable: this.immutable,
      usage: Object.freeze([...this.usage]), sizeBytes: nonEmptyBufferBytes(this.at),
      regions: Object.freeze([...this.mutableRegions]),
    });
  }
}

const candidateFloatsPerBrick = (brickFineResolution: number): number =>
  6 * brickFineResolution ** 3 + 6 * brickFineResolution ** 2;

export function sparseCM12LegacyResidentStateLayout(
  input: SparseCM12PhaseArenaPlannerInput,
): SparseCM12LegacyResidentStateLayout {
  const cells = checkedCount(input.cellCount, "cellCount");
  const rows = checkedCount(input.rowCount, "rowCount");
  const tracerCount = checkedCount(input.tracerCount ?? 0, "tracerCount");
  let at = 0;
  const fields: Record<string, { offsetFloats: number; floatCount: number }> = {};
  const add = (name: string, count: number): void => {
    fields[name] = { offsetFloats: at, floatCount: count };
    at += align4Floats(count);
  };
  add("densityA", cells); add("densityB", cells);
  add("gammaA", cells); add("gammaB", cells);
  add("cellVelocityA", 4 * cells); add("cellVelocityB", 4 * cells);
  add("faceA", rows); add("faceB", rows);
  add("pressure", cells); add("rhs", cells); add("diagonal", cells);
  add("liquid", cells); add("theta", rows); add("residual", cells);
  add("preconditioned", cells); add("direction", cells); add("applied", cells);
  add("divergence", cells); add("sharpeningDelta", cells); add("symmetryGamma", cells);
  if (input.rigidCoupling) {
    add("solidCellOpen", cells); add("solidRowData", 3 * rows);
  } else {
    fields.solidCellOpen = { offsetFloats: 0, floatCount: 0 };
    fields.solidRowData = { offsetFloats: 0, floatCount: 0 };
  }
  add("tracers", 4 * tracerCount);
  const journalLayout = sparseCM12PressureJournalLayout({
    iterationCapacity: input.pressureJournal?.iterationCapacity ?? 0,
    snapshotCapacity: input.pressureJournal?.snapshotCapacity ?? 0,
    cellStride: input.pressureJournal ? cells : 0,
  });
  if (journalLayout.floatCount > 0) add("journal", journalLayout.floatCount);
  else fields.journal = { offsetFloats: 0, floatCount: 0 };
  return Object.freeze({ floatCount: at, fields: Object.freeze(fields), journalLayout });
}

export function sparseCM12LegacyCapacityReceipt(
  input: SparseCM12PhaseArenaPlannerInput,
): SparseCM12LegacyCapacityReceipt {
  const cells = checkedCount(input.cellCount, "cellCount");
  const rows = checkedCount(input.rowCount, "rowCount");
  const bricks = checkedCount(input.brickCount, "brickCount");
  const candidateBricks = checkedCount(input.candidateBrickCount, "candidateBrickCount");
  const pressureEdgeCount = checkedCount(input.pressureEdgeCount, "pressureEdgeCount");
  const pressureCoarseEdgeCount = checkedCount(input.pressureCoarseEdgeCount,
    "pressureCoarseEdgeCount");
  let hierarchyWords = 0;
  input.pressureHierarchy.forEach((level, index) => {
    const groups = checkedCount(level.groupCount, `pressureHierarchy[${index}].groupCount`);
    const edges = checkedCount(level.edgeCount, `pressureHierarchy[${index}].edgeCount`);
    hierarchyWords += 4 * groups + edges;
  });
  const pressureScratchBytes = 4 * (pressureEdgeCount + pressureCoarseEdgeCount
    + 5 * bricks + hierarchyWords + cells);
  const candidateFieldBytes = 4 * candidateFloatsPerBrick(16) * candidateBricks;
  const conditioningBytes = 4 * Math.max(4 * cells, cells + rows + 8);
  const residentState = sparseCM12LegacyResidentStateLayout(input);
  const presentationBytes = Math.max(4, input.presentation.metadataBytes)
    + Math.max(4, input.presentation.worklistBytes)
    + Math.max(4, input.presentation.payloadBytes) + 16 + 4 + 4 + 4;
  return Object.freeze({
    residentState, residentStateBytes: Math.max(4, 4 * residentState.floatCount),
    conditioningBytes, pressureScratchBytes, candidateFieldBytes,
    overlaidCandidateStateBytes: Math.max(4, pressureScratchBytes, candidateFieldBytes),
    partialReductionBytes: Math.max(16,
      16 * Math.ceil(cells / SPARSE_CM12_PHASE_ARENA_WORKGROUP_SIZE)),
    scalarReductionBytes: SPARSE_CM12_PHASE_ARENA_PRESSURE_SCALAR_BYTES,
    presentationBytes,
    diagnosticsReadbackBytes: input.diagnosticsReadbackBytes
      ?? SPARSE_CM12_PHASE_ARENA_DIAGNOSTIC_READBACK_BYTES,
  });
}

const regionByName = (
  buffer: SparseCM12PhaseArenaBuffer,
  name: string,
): SparseCM12PhaseArenaRegion => {
  const region = buffer.regions.find((candidate) => candidate.name === name);
  if (!region) throw new Error(`${buffer.id} has no ${name} region`);
  return region;
};

const binding = (
  bindingIndex: number,
  buffer: SparseCM12PhaseArenaBuffer,
  access: SparseCM12PhaseArenaAccess,
): SparseCM12PhaseArenaBinding => Object.freeze({
  binding: bindingIndex, buffer: buffer.id, owner: buffer.owner, access,
});

export function createSparseCM12PhaseArenaPlan(
  source: SparseCM12PhaseArenaPlannerInput,
): SparseCM12PhaseArenaPlan {
  const input = Object.freeze({ ...source,
    brickFineResolution: source.brickFineResolution ?? source.htp1.brickFineResolution,
    presentationPageResolution:
      source.presentationPageResolution ?? source.brickFineResolution
        ?? source.htp1.presentationPageResolution,
  });
  if ((input.brickFineResolution !== 4 && input.brickFineResolution !== 8
      && input.brickFineResolution !== 16)
    || input.presentationPageResolution !== input.brickFineResolution
    || input.htp1.brickFineResolution !== input.brickFineResolution
    || input.htp1.presentationPageResolution !== input.presentationPageResolution) {
    throw new Error("phase arenas require a matched B4/P4, B8/P8, or B16/P16 physical ABI");
  }
  if (input.htp1.totalBytes !== 4 * input.htp1.totalWords
    || input.htp1.totalWords < 1) {
    throw new Error("phase arenas require a complete word-addressed HTP1 allocation");
  }
  if (input.pressureHierarchy.length < 1) {
    throw new Error("pressure cache requires at least one aggregate hierarchy level");
  }
  const cells = checkedCount(input.cellCount, "cellCount");
  const rows = checkedCount(input.rowCount, "rowCount");
  const bricks = checkedCount(input.brickCount, "brickCount");
  const candidateBricks = checkedCount(input.candidateBrickCount, "candidateBrickCount");
  if (input.htp1.cellCount !== cells || input.htp1.rowCount !== rows
    || input.htp1.directedEdgeCount !== input.pressureEdgeCount) {
    throw new Error("phase capacities do not match immutable HTP1 counts");
  }
  const legacy = sparseCM12LegacyCapacityReceipt(input);

  const physicsBuilder = new BufferBuilder("cm12.physics-state", "PhysicsState", false,
    ["STORAGE", "COPY_SRC", "COPY_DST"]);
  const physics = (name: string, count: number) => {
    return physicsBuilder.add(name, count, "f32", "frame");
  };
  physics("densityA", cells); physics("densityB", cells);
  physics("gammaA", cells); physics("gammaB", cells);
  physics("cellVelocityA", 4 * cells); physics("cellVelocityB", 4 * cells);
  physics("faceA", rows); physics("faceB", rows); physics("pressure", cells);
  if (input.rigidCoupling) {
    physics("solidCellOpen", cells); physics("solidRowData", 3 * rows);
  }
  const physicsBuffer = physicsBuilder.finish();

  const scratchBuilder = new BufferBuilder("cm12.scalar-scratch", "ScalarScratch", false,
    ["STORAGE", "COPY_SRC", "COPY_DST"]);
  for (const name of ["rhs", "diagonal", "residual", "preconditioned", "direction",
    "applied", "divergence", "sharpeningDelta", "symmetryGamma"] as const) {
    scratchBuilder.add(name, cells, "f32", "frame");
  }
  scratchBuilder.add("conditioning", legacy.conditioningBytes / 4, "i32", "frame");
  scratchBuilder.add("partialReductions", legacy.partialReductionBytes / 4, "f32", "frame");
  scratchBuilder.add("scalarReductions", legacy.scalarReductionBytes / 4, "f32", "frame");
  const scratchBuffer = scratchBuilder.finish();

  const pressureBuilder = new BufferBuilder("cm12.pressure-cache", "PressureCache", false,
    ["STORAGE", "COPY_SRC", "COPY_DST"]);
  pressureBuilder.add("liquid", cells, "f32", "topology-epoch");
  pressureBuilder.add("theta", rows, "f32", "topology-epoch");
  // Everything after this marker retains candidateState's former pressure
  // scratch offsets relative to pressureScratchBase.
  const pressureScratchBase = pressureBuilder.add("pressureScratchBase", 0, "bytes",
    "topology-epoch").offsetBytes;
  pressureBuilder.add("effectiveEdgeWeights", input.pressureEdgeCount,
    "f32", "topology-epoch", 4);
  pressureBuilder.add("brickAggregateEdgeWeights", input.pressureCoarseEdgeCount,
    "f32", "topology-epoch", 4);
  for (const name of ["brickAggregateRhs", "brickAggregateDiagonal", "brickAggregateA",
    "brickAggregateB", "brickAggregateRange"] as const) {
    pressureBuilder.add(name, bricks, name === "brickAggregateRange" ? "u32" : "f32",
      "topology-epoch", 4);
  }
  input.pressureHierarchy.forEach((level, index) => {
    pressureBuilder.add(`hierarchy${index}.edgeWeights`, level.edgeCount, "f32",
      "topology-epoch", 4);
    for (const name of ["diagonal", "rhs", "a", "b"] as const) {
      pressureBuilder.add(`hierarchy${index}.${name}`, level.groupCount, "f32",
        "topology-epoch", 4);
    }
  });
  pressureBuilder.add("densityCache", cells, "f32", "topology-epoch", 4);
  pressureBuilder.add("membership", input.pressureMembershipWords ?? 0,
    "u32", "topology-epoch");
  pressureBuilder.add("aggregateHierarchyTopology", input.pressureStaticTopologyWords ?? 0,
    "u32", "topology-epoch");
  const pressureBuffer = pressureBuilder.finish();
  const pressureDensity = regionByName(pressureBuffer, "densityCache");
  if (pressureDensity.offsetBytes + pressureDensity.sizeBytes - pressureScratchBase
    !== legacy.pressureScratchBytes) {
    throw new Error("pressure-cache dynamic prefix drifted from candidateState scratch ABI");
  }

  const candidateBuilder = new BufferBuilder("cm12.topology-candidate", "TopologyCandidate",
    false, ["STORAGE", "COPY_SRC", "COPY_DST"]);
  const candidateCells = candidateBricks * 16 ** 3;
  const candidateFaces = candidateBricks * 16 ** 2;
  for (const name of ["density", "gamma", "velocityX", "velocityY", "velocityZ", "pressure"])
    candidateBuilder.add(`cell.${name}`, candidateCells, "f32", "topology-epoch", 4);
  for (let side = 0; side < 6; side += 1)
    candidateBuilder.add(`face.side${side}`, candidateFaces, "f32", "topology-epoch", 4);
  candidateBuilder.add("controlAndPagePool", input.topologyCandidateControlWords ?? 0,
    "u32", "topology-epoch");
  const candidateBuffer = candidateBuilder.finish();

  const presentationParametersBuilder = new BufferBuilder("cm12.presentation-parameters",
    "Presentation", false, ["UNIFORM", "COPY_DST"]);
  presentationParametersBuilder.add("parameters", 16, "bytes", "frame", 4);
  const presentationParametersBuffer = presentationParametersBuilder.finish();
  const presentationBuilder = new BufferBuilder("cm12.presentation", "Presentation", false,
    ["STORAGE", "COPY_SRC", "COPY_DST"]);
  presentationBuilder.add("tracers", 4 * (input.tracerCount ?? 0), "f32", "frame", 16);
  presentationBuilder.add("metadata", input.presentation.metadataBytes,
    "bytes", "topology-epoch", 16);
  presentationBuilder.add("worklist", input.presentation.worklistBytes,
    "bytes", "topology-epoch", 16);
  presentationBuilder.add("samples", Math.max(4, input.presentation.payloadBytes),
    "bytes", "frame", 16);
  presentationBuilder.add("workA", 4, "bytes", "frame", 4);
  presentationBuilder.add("workB", 4, "bytes", "frame", 4);
  presentationBuilder.add("rollback", 4, "bytes", "frame", 4);
  const presentationBuffer = presentationBuilder.finish();

  const diagnosticsDeviceBuilder = new BufferBuilder("cm12.diagnostics-device", "Diagnostics",
    false, ["STORAGE", "COPY_SRC", "COPY_DST"]);
  diagnosticsDeviceBuilder.add("pressureJournal", legacy.residentState.journalLayout.floatCount,
    "f32", "diagnostic-session");
  const diagnosticsDeviceBuffer = diagnosticsDeviceBuilder.finish();
  const diagnosticsReadbackBuilder = new BufferBuilder("cm12.diagnostics-readback", "Diagnostics",
    false, ["COPY_DST", "MAP_READ"]);
  diagnosticsReadbackBuilder.add("readback", legacy.diagnosticsReadbackBytes,
    "bytes", "diagnostic-session", 4);
  const diagnosticsReadbackBuffer = diagnosticsReadbackBuilder.finish();

  const htpBuffer: SparseCM12PhaseArenaBuffer = Object.freeze({
    id: "cm12.htp1", owner: "HTP1", immutable: true,
    usage: Object.freeze(["STORAGE", "COPY_DST"] as const),
    sizeBytes: input.htp1.totalBytes,
    regions: Object.freeze([Object.freeze({ name: "HTP1", offsetBytes: 0,
      sizeBytes: input.htp1.totalBytes, element: "u32" as const,
      count: input.htp1.totalWords, persistence: "immutable" as const })]),
  });
  const buffers = Object.freeze([physicsBuffer, scratchBuffer, pressureBuffer, candidateBuffer,
    presentationParametersBuffer, presentationBuffer, diagnosticsDeviceBuffer,
    diagnosticsReadbackBuffer, htpBuffer]);

  const rawBindGroups: SparseCM12PhaseArenaBindGroup[] = [
    { phase: "transport-conditioning", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "storage"), binding(2, scratchBuffer, "storage")] },
    { phase: "pressure-preparation", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "storage"), binding(2, scratchBuffer, "storage"),
      binding(3, pressureBuffer, "storage")] },
    { phase: "pressure-solve", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "storage"), binding(2, scratchBuffer, "storage"),
      binding(3, pressureBuffer, "storage")] },
    { phase: "velocity-projection", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "storage"), binding(2, scratchBuffer, "storage"),
      binding(3, pressureBuffer, "read-only-storage")] },
    { phase: "topology-candidate", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "read-only-storage"),
      binding(3, pressureBuffer, "read-only-storage"),
      binding(4, candidateBuffer, "storage")] },
    { phase: "presentation-publication", bindings: [binding(0, htpBuffer, "read-only-storage"),
      binding(1, physicsBuffer, "read-only-storage"),
      binding(4, candidateBuffer, "read-only-storage"),
      binding(5, presentationParametersBuffer, "uniform"),
      binding(6, presentationBuffer, "storage")] },
    { phase: "diagnostics-copy", bindings: [binding(1, physicsBuffer, "read-only-storage"),
      binding(2, scratchBuffer, "read-only-storage"),
      binding(3, pressureBuffer, "read-only-storage"),
      binding(4, candidateBuffer, "read-only-storage"),
      binding(6, presentationBuffer, "read-only-storage"),
      binding(7, diagnosticsDeviceBuffer, "storage")] },
  ];
  const bindGroups: readonly SparseCM12PhaseArenaBindGroup[] = Object.freeze(
    rawBindGroups.map((descriptor) => Object.freeze({ ...descriptor,
    bindings: Object.freeze(descriptor.bindings) })));

  const targetForLegacy: Readonly<Record<string, readonly [string, string]>> = Object.freeze({
    densityA: [physicsBuffer.id, "densityA"], densityB: [physicsBuffer.id, "densityB"],
    gammaA: [physicsBuffer.id, "gammaA"], gammaB: [physicsBuffer.id, "gammaB"],
    cellVelocityA: [physicsBuffer.id, "cellVelocityA"],
    cellVelocityB: [physicsBuffer.id, "cellVelocityB"],
    faceA: [physicsBuffer.id, "faceA"], faceB: [physicsBuffer.id, "faceB"],
    pressure: [physicsBuffer.id, "pressure"], rhs: [scratchBuffer.id, "rhs"],
    diagonal: [scratchBuffer.id, "diagonal"], liquid: [pressureBuffer.id, "liquid"],
    theta: [pressureBuffer.id, "theta"], residual: [scratchBuffer.id, "residual"],
    preconditioned: [scratchBuffer.id, "preconditioned"],
    direction: [scratchBuffer.id, "direction"], applied: [scratchBuffer.id, "applied"],
    divergence: [scratchBuffer.id, "divergence"],
    sharpeningDelta: [scratchBuffer.id, "sharpeningDelta"],
    symmetryGamma: [scratchBuffer.id, "symmetryGamma"],
    solidCellOpen: [physicsBuffer.id, "solidCellOpen"],
    solidRowData: [physicsBuffer.id, "solidRowData"],
    tracers: [presentationBuffer.id, "tracers"],
    journal: [diagnosticsDeviceBuffer.id, "pressureJournal"],
  });
  const bufferById = new Map(buffers.map((buffer) => [buffer.id, buffer]));
  const migration: SparseCM12PhaseArenaMigrationEntry[] = [];
  for (const [semantic, legacyField] of Object.entries(legacy.residentState.fields)) {
    if (legacyField.floatCount === 0) continue;
    const target = targetForLegacy[semantic];
    if (!target) throw new Error(`legacy state field ${semantic} has no phase owner`);
    const targetBuffer = bufferById.get(target[0])!;
    const targetRegion = regionByName(targetBuffer, target[1]);
    if (targetRegion.sizeBytes < 4 * legacyField.floatCount) {
      throw new Error(`${semantic} target is smaller than its legacy field`);
    }
    migration.push(Object.freeze({
      semantic, legacyOffsetBytes: 4 * legacyField.offsetFloats,
      targetBuffer: targetBuffer.id, targetOffsetBytes: targetRegion.offsetBytes,
      sizeBytes: 4 * legacyField.floatCount,
      offsetPreserved: targetRegion.offsetBytes === 4 * legacyField.offsetFloats,
    }));
  }
  const plan: SparseCM12PhaseArenaPlan = Object.freeze({
    version: 1, input, buffers, bindGroups, legacy,
    migration: Object.freeze(migration),
    allocatedBytes: buffers.reduce((sum, buffer) => sum + buffer.sizeBytes, 0),
  });
  assertSparseCM12PhaseArenaPlan(plan);
  return plan;
}

/** Assert physical separation, internal non-overlap, phase binding, and migration coverage. */
export function assertSparseCM12PhaseArenaPlan(plan: SparseCM12PhaseArenaPlan): void {
  if (plan.version !== 1) throw new Error("unsupported phase-arena ABI");
  const ids = new Set<string>();
  for (const buffer of plan.buffers) {
    if (ids.has(buffer.id)) throw new Error(`physical allocation alias ${buffer.id}`);
    ids.add(buffer.id);
    let end = 0;
    const names = new Set<string>();
    for (const region of buffer.regions) {
      if (names.has(region.name)) throw new Error(`${buffer.id} duplicates ${region.name}`);
      names.add(region.name);
      if (region.offsetBytes < end) throw new Error(`${buffer.id}.${region.name} overlaps prior data`);
      if (region.offsetBytes + region.sizeBytes > buffer.sizeBytes) {
        throw new Error(`${buffer.id}.${region.name} exceeds its allocation`);
      }
      end = region.offsetBytes + region.sizeBytes;
    }
  }
  const owners = new Set(plan.buffers.map((buffer) => buffer.owner));
  for (const required of ["PhysicsState", "ScalarScratch", "PressureCache",
    "TopologyCandidate", "Presentation", "Diagnostics", "HTP1"] as const) {
    if (!owners.has(required)) throw new Error(`missing ${required} phase owner`);
  }
  for (const descriptor of plan.bindGroups) {
    const bindingIndices = new Set<number>(), boundBuffers = new Set<string>();
    for (const entry of descriptor.bindings) {
      if (!ids.has(entry.buffer)) throw new Error(`${descriptor.phase} binds unknown ${entry.buffer}`);
      if (bindingIndices.has(entry.binding)) throw new Error(`${descriptor.phase} reuses binding ${entry.binding}`);
      if (boundBuffers.has(entry.buffer)) throw new Error(`${descriptor.phase} aliases ${entry.buffer}`);
      bindingIndices.add(entry.binding); boundBuffers.add(entry.buffer);
      const buffer = plan.buffers.find((candidate) => candidate.id === entry.buffer)!;
      if (buffer.immutable && entry.access !== "read-only-storage") {
        throw new Error(`${descriptor.phase} requests write access to immutable ${entry.buffer}`);
      }
      if (entry.access === "uniform" && !buffer.usage.includes("UNIFORM")) {
        throw new Error(`${descriptor.phase} binds non-uniform ${entry.buffer} as uniform`);
      }
      if (entry.access !== "uniform" && !buffer.usage.includes("STORAGE")) {
        throw new Error(`${descriptor.phase} binds non-storage ${entry.buffer} as storage`);
      }
    }
  }
  const semantics = new Set<string>();
  for (const entry of plan.migration) {
    if (semantics.has(entry.semantic)) throw new Error(`duplicate migration ${entry.semantic}`);
    semantics.add(entry.semantic);
  }
  for (const [semantic, field] of Object.entries(plan.legacy.residentState.fields)) {
    if (field.floatCount > 0 && !semantics.has(semantic)) {
      throw new Error(`unmapped legacy field ${semantic}`);
    }
  }
  const physics = plan.buffers.find((buffer) => buffer.owner === "PhysicsState")!;
  for (const semantic of ["densityA", "densityB", "gammaA", "gammaB", "cellVelocityA",
    "cellVelocityB", "faceA", "faceB", "pressure"]) {
    const mapping = plan.migration.find((entry) => entry.semantic === semantic);
    if (!mapping?.offsetPreserved || mapping.targetBuffer !== physics.id) {
      throw new Error(`PhysicsState prefix offset drift for ${semantic}`);
    }
  }
  const candidate = plan.buffers.find((buffer) => buffer.owner === "TopologyCandidate")!;
  const candidateFields = candidate.regions.filter((region) =>
    region.name.startsWith("cell.") || region.name.startsWith("face."))
    .reduce((sum, region) => sum + region.sizeBytes, 0);
  if (candidateFields !== plan.legacy.candidateFieldBytes) {
    throw new Error("candidate field capacity differs from current candidateState need");
  }
  const scratch = plan.buffers.find((buffer) => buffer.owner === "ScalarScratch")!;
  if (regionByName(scratch, "conditioning").sizeBytes !== plan.legacy.conditioningBytes) {
    throw new Error("conditioning capacity differs from current resident formula");
  }
  const pressure = plan.buffers.find((buffer) => buffer.owner === "PressureCache")!;
  const effective = regionByName(pressure, "effectiveEdgeWeights");
  const density = regionByName(pressure, "densityCache");
  if (density.offsetBytes + density.sizeBytes - effective.offsetBytes
    !== plan.legacy.pressureScratchBytes) {
    throw new Error("persistent pressure cache differs from current pressure scratch need");
  }
}

export function sparseCM12PhaseArenaByteMap(
  plan: SparseCM12PhaseArenaPlan,
): readonly ({ readonly buffer: string; readonly owner: SparseCM12PhaseArenaOwner }
  & SparseCM12PhaseArenaRegion)[] {
  return Object.freeze(plan.buffers.flatMap((buffer) => buffer.regions.map((region) =>
    Object.freeze({ buffer: buffer.id, owner: buffer.owner, ...region }))));
}
