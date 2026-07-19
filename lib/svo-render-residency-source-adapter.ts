import {
  planSvoRenderResidency,
  type SvoRenderBrickRequest,
  type SvoRenderResidentBrick,
  type SvoRenderResidencyPlan,
} from "./svo-render-residency";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
  SPARSE_VOXEL_VALID_FIELDS,
  decodeSparseVoxelFluidResidencyState,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelFluidResidencySource,
  type SparseVoxelPublicationWord,
  type SparseVoxelStructuralRenderSource,
} from "./webgpu-voxel-debug";

const UINT32_MAX = 0xffff_ffff;

export interface SvoRenderResidencySourceSnapshot {
  /** Words copied from `structural.publication.state`. */
  publicationWords: Uint32Array;
  /** Complete producer header and active/retired entry arenas. */
  worklistWords: Uint32Array;
  /** One producer state word per solver brick. */
  stateWords: Uint32Array;
}

export interface SvoRenderResidencySourceEntry {
  brickIndex: number;
  leafIndex: number;
  coordinate: readonly [number, number, number];
  stateWord: number;
  state: "core" | "halo" | "retired";
  activated: boolean;
  dryFrames: number;
}

export interface SvoRenderResidencyGpuListInput {
  count: SparseVoxelPublicationWord;
  entries: GPUBufferBinding;
  entryOffsetBytes: number;
  entryStrideBytes: number;
  capacity: number;
  requiredStateBit?: number;
}

/** Binding-only inputs for the later GPU-native renderer worklist consumer. */
export interface SvoRenderResidencyGpuInputs {
  bindGroupEntries: readonly [GPUBindGroupEntry, GPUBindGroupEntry, GPUBindGroupEntry];
  fence: Readonly<{
    completeGeneration: SparseVoxelPublicationWord;
    coarseFluidRevision: SparseVoxelPublicationWord;
    residencyRevision: SparseVoxelPublicationWord;
    listGeneration: SparseVoxelPublicationWord;
  }>;
  lists: Readonly<{
    active: SvoRenderResidencyGpuListInput;
    core: SvoRenderResidencyGpuListInput;
    halo: SvoRenderResidencyGpuListInput;
    retired: SvoRenderResidencyGpuListInput;
  }>;
  /** A prepare pass must clamp each count before authoring indirect dispatch. */
  dispatch: Readonly<{
    workgroupSize: 64;
    maximumEntryWorkgroups: number;
    requiresIndirectPrepare: true;
  }>;
}

export interface SvoRenderResidencySourceTelemetry {
  completeGeneration: number;
  coarseFluidRevision: number;
  listGeneration: number;
  sourceActiveCount: number;
  sourceCoreCount: number;
  sourceHaloCount: number;
  sourceRetiredCount: number;
  decodedActiveCount: number;
  decodedRetiredCount: number;
  sourceOverflowCount: number;
  rendererOverflowCount: number;
  desiredRequestCount: number;
  dirtyRequestCount: number;
  dirtyRetiredCount: number;
}

export interface SvoRenderResidencySourceAdapterInput {
  structural: SparseVoxelStructuralRenderSource;
  snapshot: SvoRenderResidencySourceSnapshot;
  rendererCapacity: number;
  coarseCoverageComplete: boolean;
  previousCompleteGeneration?: number;
  previousResidents?: readonly SvoRenderResidentBrick[];
  retireAfterFrames?: number;
}

export interface SvoRenderResidencySourceAdapterPlan {
  status: "ready" | "unchanged";
  gpuInputs: SvoRenderResidencyGpuInputs;
  activeEntries: readonly SvoRenderResidencySourceEntry[];
  coreEntries: readonly SvoRenderResidencySourceEntry[];
  haloEntries: readonly SvoRenderResidencySourceEntry[];
  retiredEntries: readonly SvoRenderResidencySourceEntry[];
  desiredRequests: readonly SvoRenderBrickRequest[];
  dirtyRequests: readonly SvoRenderBrickRequest[];
  dirtyRetiredEntries: readonly SvoRenderResidencySourceEntry[];
  residency: SvoRenderResidencyPlan;
  telemetry: SvoRenderResidencySourceTelemetry;
}

export interface SvoRenderResidencySourceAdapterRejection {
  status: "rejected";
  reason: "unpublished" | "missing-fields" | "generation-mismatch" | "revision-mismatch";
  gpuInputs: SvoRenderResidencyGpuInputs;
  telemetry: SvoRenderResidencySourceTelemetry;
}

export type SvoRenderResidencySourceAdapterResult =
  | SvoRenderResidencySourceAdapterPlan
  | SvoRenderResidencySourceAdapterRejection;

function safeUint(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function bindingOffset(binding: GPUBufferBinding): number { return binding.offset ?? 0; }

function bindingSize(binding: GPUBufferBinding): number {
  return binding.size ?? Math.max(0, binding.buffer.size - bindingOffset(binding));
}

function sameBinding(left: GPUBufferBinding, right: GPUBufferBinding): boolean {
  return left.buffer === right.buffer && bindingOffset(left) === bindingOffset(right);
}

function requireWordBinding(
  word: SparseVoxelPublicationWord,
  binding: GPUBufferBinding,
  expectedWord: number,
  label: string,
): void {
  if (!sameBinding(word.binding, binding) || word.word !== expectedWord) {
    throw new RangeError(`${label} does not match its producer buffer word`);
  }
}

function validateDomain(structural: SparseVoxelStructuralRenderSource, residency: SparseVoxelFluidResidencySource): void {
  const dimensions = residency.domain.dimensionsBricks;
  const origin = residency.domain.originBricks;
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Fluid residency brick dimensions must be positive integers");
  }
  if (origin.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Fluid residency brick origin must be non-negative integers");
  }
  const capacity = residency.active.capacity;
  if (dimensions[0] * dimensions[1] * dimensions[2] !== capacity) {
    throw new RangeError("Fluid residency brick domain does not match its capacity");
  }
  const structuralDimensions = structural.domain.dimensionsCells.map(
    (value) => Math.ceil(value / structural.domain.brickSize),
  );
  if (origin.some((value, axis) => value + dimensions[axis] > structuralDimensions[axis])) {
    throw new RangeError("Fluid residency brick domain exceeds the structural domain");
  }
}

/** Validate and expose immutable source bindings for a future GPU-native pass. */
export function buildSvoRenderResidencyGpuInputs(
  structural: SparseVoxelStructuralRenderSource,
): SvoRenderResidencyGpuInputs {
  const residency = structural.fluidResidency;
  if (!residency) throw new RangeError("Structural source has no fluid residency publication");
  validateDomain(structural, residency);
  const capacity = residency.active.capacity;
  const layout = sparseVoxelFluidResidencyLayout(capacity);
  if (residency.stateStrideBytes !== layout.stateStrideBytes
    || residency.active.entryOffsetBytes !== layout.activeEntryOffsetBytes
    || residency.retired.entryOffsetBytes !== layout.retiredEntryOffsetBytes
    || residency.active.entryStrideBytes !== layout.entryStrideBytes
    || residency.core.entryStrideBytes !== layout.entryStrideBytes
    || residency.halo.entryStrideBytes !== layout.entryStrideBytes
    || residency.retired.entryStrideBytes !== layout.entryStrideBytes) {
    throw new RangeError("Fluid residency source layout does not match the producer ABI");
  }
  if (residency.core.capacity !== capacity || residency.halo.capacity !== capacity || residency.retired.capacity !== capacity) {
    throw new RangeError("Fluid residency list capacities must match");
  }
  if (residency.core.entryOffsetBytes !== residency.active.entryOffsetBytes
    || residency.halo.entryOffsetBytes !== residency.active.entryOffsetBytes) {
    throw new RangeError("Core and halo lists must be filtered views of the active list");
  }
  if (residency.core.requiredStateBit !== residency.stateBits.core
    || residency.halo.requiredStateBit !== residency.stateBits.halo
    || residency.stateBits.resident !== SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident
    || residency.stateBits.core !== SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core
    || residency.stateBits.halo !== SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo
    || residency.stateBits.activated !== SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.activated
    || residency.stateBits.wasResident !== SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.wasResident) {
    throw new RangeError("Fluid residency state-bit ABI is malformed");
  }
  if (bindingSize(residency.states) < capacity * layout.stateStrideBytes
    || bindingSize(residency.worklist) < layout.worklistByteLength
    || bindingSize(structural.publication.state) < 32) {
    throw new RangeError("Fluid residency source buffers are smaller than their declared layout");
  }
  requireWordBinding(structural.publication.completeGeneration, structural.publication.state, 0, "Complete generation");
  requireWordBinding(structural.publication.validFields, structural.publication.state, 1, "Valid fields");
  requireWordBinding(structural.publication.revisions.coarseFluid, structural.publication.state, 5, "Coarse-fluid revision");
  requireWordBinding(residency.revision, structural.publication.state, 5, "Residency revision");
  requireWordBinding(residency.generation, residency.worklist, SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.generation, "Residency generation");
  requireWordBinding(residency.active.count, residency.worklist, SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount, "Active count");
  requireWordBinding(residency.core.count, residency.worklist, SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.coreCount, "Core count");
  requireWordBinding(residency.halo.count, residency.worklist, SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.haloCount, "Halo count");
  requireWordBinding(residency.retired.count, residency.worklist, SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount, "Retired count");
  const list = (view: typeof residency.active, requiredStateBit?: number): SvoRenderResidencyGpuListInput => ({
    count: view.count,
    entries: residency.worklist,
    entryOffsetBytes: view.entryOffsetBytes,
    entryStrideBytes: view.entryStrideBytes,
    capacity: view.capacity,
    ...(requiredStateBit === undefined ? {} : { requiredStateBit }),
  });
  return {
    bindGroupEntries: [
      { binding: 0, resource: structural.publication.state },
      { binding: 1, resource: residency.states },
      { binding: 2, resource: residency.worklist },
    ],
    fence: {
      completeGeneration: structural.publication.completeGeneration,
      coarseFluidRevision: structural.publication.revisions.coarseFluid,
      residencyRevision: residency.revision,
      listGeneration: residency.generation,
    },
    lists: {
      active: list(residency.active),
      core: list(residency.core, residency.core.requiredStateBit),
      halo: list(residency.halo, residency.halo.requiredStateBit),
      retired: list(residency.retired),
    },
    dispatch: {
      workgroupSize: 64,
      maximumEntryWorkgroups: Math.ceil(capacity / 64),
      requiresIndirectPrepare: true,
    },
  };
}

function coordinateForBrick(
  brickIndex: number,
  residency: SparseVoxelFluidResidencySource,
): readonly [number, number, number] {
  const [width, height] = residency.domain.dimensionsBricks;
  const [ox, oy, oz] = residency.domain.originBricks;
  return [
    ox + brickIndex % width,
    oy + Math.floor(brickIndex / width) % height,
    oz + Math.floor(brickIndex / (width * height)),
  ];
}

function emptyTelemetry(
  completeGeneration: number,
  coarseFluidRevision: number,
  listGeneration: number,
): SvoRenderResidencySourceTelemetry {
  return {
    completeGeneration,
    coarseFluidRevision,
    listGeneration,
    sourceActiveCount: 0,
    sourceCoreCount: 0,
    sourceHaloCount: 0,
    sourceRetiredCount: 0,
    decodedActiveCount: 0,
    decodedRetiredCount: 0,
    sourceOverflowCount: 0,
    rendererOverflowCount: 0,
    desiredRequestCount: 0,
    dirtyRequestCount: 0,
    dirtyRetiredCount: 0,
  };
}

/**
 * Decode one completed producer snapshot into renderer-owned planning data.
 * This function never writes source buffers or state; stale fences are
 * rejected before any desired/dirty work is exposed.
 */
export function adaptSparseVoxelRenderResidencySource(
  input: SvoRenderResidencySourceAdapterInput,
): SvoRenderResidencySourceAdapterResult {
  const gpuInputs = buildSvoRenderResidencyGpuInputs(input.structural);
  const residency = input.structural.fluidResidency!;
  const { publicationWords, worklistWords, stateWords } = input.snapshot;
  const capacity = residency.active.capacity;
  const layout = sparseVoxelFluidResidencyLayout(capacity);
  if (publicationWords.length * 4 < 32 || worklistWords.length * 4 < layout.worklistByteLength || stateWords.length < capacity) {
    throw new RangeError("Fluid residency readback is smaller than the declared source layout");
  }
  safeUint(input.rendererCapacity, "Renderer residency capacity");
  if (input.previousCompleteGeneration !== undefined) safeUint(input.previousCompleteGeneration, "Previous complete generation");
  const completeGeneration = publicationWords[input.structural.publication.completeGeneration.word] >>> 0;
  const validFields = publicationWords[input.structural.publication.validFields.word] >>> 0;
  const coarseFluidRevision = publicationWords[input.structural.publication.revisions.coarseFluid.word] >>> 0;
  const residencyRevision = publicationWords[residency.revision.word] >>> 0;
  const listGeneration = worklistWords[residency.generation.word] >>> 0;
  const rejected = (reason: SvoRenderResidencySourceAdapterRejection["reason"]): SvoRenderResidencySourceAdapterRejection => ({
    status: "rejected",
    reason,
    gpuInputs,
    telemetry: emptyTelemetry(completeGeneration, coarseFluidRevision, listGeneration),
  });
  if (completeGeneration === 0) return rejected("unpublished");
  const requiredFields = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  if ((validFields & requiredFields) !== requiredFields) return rejected("missing-fields");
  if (residencyRevision !== coarseFluidRevision) return rejected("revision-mismatch");
  if (completeGeneration !== coarseFluidRevision || completeGeneration !== listGeneration) return rejected("generation-mismatch");

  const sourceActiveCount = worklistWords[residency.active.count.word] >>> 0;
  const sourceCoreCount = worklistWords[residency.core.count.word] >>> 0;
  const sourceHaloCount = worklistWords[residency.halo.count.word] >>> 0;
  const sourceRetiredCount = worklistWords[residency.retired.count.word] >>> 0;
  const retiredStatsCount = worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredStatsCount] >>> 0;
  if (sourceCoreCount + sourceHaloCount !== sourceActiveCount || retiredStatsCount !== sourceRetiredCount) {
    throw new RangeError("Fluid residency header counters are inconsistent");
  }
  const activeCount = Math.min(sourceActiveCount, capacity);
  const retiredCount = Math.min(sourceRetiredCount, capacity);
  const activeEntries: SvoRenderResidencySourceEntry[] = [];
  const retiredEntries: SvoRenderResidencySourceEntry[] = [];
  const seenBricks = new Set<number>();
  const seenLeaves = new Set<number>();
  const decodeEntry = (entryOffsetBytes: number, index: number, retired: boolean) => {
    const base = entryOffsetBytes / 4 + index * (layout.entryStrideBytes / 4);
    const brickIndex = worklistWords[base] >>> 0;
    const leafIndex = worklistWords[base + 1] >>> 0;
    if (brickIndex >= capacity || leafIndex >= input.structural.capacities.leaves) {
      throw new RangeError("Fluid residency entry exceeds its brick or leaf capacity");
    }
    if (seenBricks.has(brickIndex) || seenLeaves.has(leafIndex)) {
      throw new RangeError("Fluid residency worklists contain duplicate brick or leaf entries");
    }
    seenBricks.add(brickIndex);
    seenLeaves.add(leafIndex);
    const stateWord = stateWords[brickIndex] >>> 0;
    const decoded = decodeSparseVoxelFluidResidencyState(stateWord);
    if (retired) {
      if (decoded.resident || !decoded.wasResident) throw new RangeError("Retired residency entry has incompatible state flags");
      return {
        brickIndex,
        leafIndex,
        coordinate: coordinateForBrick(brickIndex, residency),
        stateWord,
        state: "retired" as const,
        activated: decoded.activated,
        dryFrames: decoded.dryFrames,
      };
    }
    if (!decoded.resident || decoded.core === decoded.halo) {
      throw new RangeError("Active residency entry must be exactly one of core or halo");
    }
    return {
      brickIndex,
      leafIndex,
      coordinate: coordinateForBrick(brickIndex, residency),
      stateWord,
      state: decoded.core ? "core" as const : "halo" as const,
      activated: decoded.activated,
      dryFrames: decoded.dryFrames,
    };
  };
  for (let index = 0; index < activeCount; index += 1) {
    activeEntries.push(decodeEntry(residency.active.entryOffsetBytes, index, false));
  }
  for (let index = 0; index < retiredCount; index += 1) {
    retiredEntries.push(decodeEntry(residency.retired.entryOffsetBytes, index, true));
  }
  const coreEntries = activeEntries.filter((entry) => entry.state === "core");
  const haloEntries = activeEntries.filter((entry) => entry.state === "halo");
  if (sourceActiveCount <= capacity
    && (coreEntries.length !== sourceCoreCount || haloEntries.length !== sourceHaloCount)) {
    throw new RangeError("Fluid residency entries do not match core and halo counters");
  }
  const desiredRequests = activeEntries.map((entry): SvoRenderBrickRequest => {
    if (entry.state === "retired") throw new RangeError("Retired entry leaked into the active residency list");
    return {
      coordinate: entry.coordinate,
      key: entry.coordinate.join(","),
      state: entry.state,
      layers: ["fluid"],
      causes: [entry.activated ? "solver-residency-activated" : `solver-residency-${entry.state}`],
    };
  });
  const basePlan = planSvoRenderResidency({
    desiredRequests,
    previousResidents: input.previousResidents,
    capacity: input.rendererCapacity,
    retireAfterFrames: input.retireAfterFrames,
    coarseCoverageComplete: input.coarseCoverageComplete,
  });
  const sourceOverflowCount = Math.max(0, sourceActiveCount - capacity);
  const totalOverflow = basePlan.overflowCount + sourceOverflowCount;
  const residencyPlan: SvoRenderResidencyPlan = totalOverflow === basePlan.overflowCount ? basePlan : {
    ...basePlan,
    overflowCount: totalOverflow,
    coverage: input.coarseCoverageComplete ? "coarse-fallback" : "incomplete",
    publishable: input.coarseCoverageComplete,
  };
  const unchanged = input.previousCompleteGeneration === completeGeneration;
  const dirtyRequests = unchanged ? [] : desiredRequests;
  const dirtyRetiredEntries = unchanged ? [] : retiredEntries;
  return {
    status: unchanged ? "unchanged" : "ready",
    gpuInputs,
    activeEntries,
    coreEntries,
    haloEntries,
    retiredEntries,
    desiredRequests,
    dirtyRequests,
    dirtyRetiredEntries,
    residency: residencyPlan,
    telemetry: {
      completeGeneration,
      coarseFluidRevision,
      listGeneration,
      sourceActiveCount,
      sourceCoreCount,
      sourceHaloCount,
      sourceRetiredCount,
      decodedActiveCount: activeEntries.length,
      decodedRetiredCount: retiredEntries.length,
      sourceOverflowCount,
      rendererOverflowCount: basePlan.overflowCount,
      desiredRequestCount: desiredRequests.length,
      dirtyRequestCount: dirtyRequests.length,
      dirtyRetiredCount: dirtyRetiredEntries.length,
    },
  };
}
