import {
  FACTOR_ONE_DENSE_CELL_ROLE,
  FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD,
  type FactorOneDenseCoordinate,
  type FactorOneDenseGpuChannelName,
  type FactorOneDenseGpuPhysicalPlan,
  factorOneDenseCoordinate,
  factorOneDenseSlot,
  planFactorOneDenseGpuPhysicalImage,
  planFactorOneDensePressureHierarchy,
} from "./factor-one-dense-pressure-hierarchy";
import { PassBroker } from "./webgpu-pass-broker";

const SPARSE_DISPATCH_WORDS = 12;
const PARAM_WORDS = 80;
const PARAM_BYTES = PARAM_WORDS * 4;
const MAX_LEVELS = 12;

export const FACTOR_ONE_DENSE_SHADOW_ERROR = Object.freeze({
  coordinate: 1,
  role: 2,
  owner: 4,
  diagonal: 8,
  count: 16,
  parentOrBottom: 32,
  transaction: 64,
} as const);

export interface FactorOneDenseSparseShadowPlan {
  readonly levelCount: number;
  readonly levelCapacities: readonly number[];
  readonly levelOffsets: readonly number[];
  readonly totalLevelSlots: number;
  readonly worklistBaseWords: number;
}

export interface FactorOneDenseSparseShadowSource {
  readonly dimensions: FactorOneDenseCoordinate;
  readonly sparsePlan: FactorOneDenseSparseShadowPlan;
  readonly acceptedTopology: GPUBuffer;
  readonly acceptedState: GPUBuffer;
  readonly acceptedDispatch: GPUBuffer;
  readonly candidateTopology: GPUBuffer;
  readonly candidateState: GPUBuffer;
  readonly candidateDispatch: GPUBuffer;
  readonly levelDelta: GPUBuffer;
  /** Raw CapturePageState words; generation=0, publishedGeneration=6, error=8. */
  readonly captureControl: GPUBuffer;
}

export interface FactorOneDenseAcceptedView {
  readonly arena: GPUBuffer;
  readonly physicalPlan: FactorOneDenseGpuPhysicalPlan;
  readonly rowGeometry: GPUBuffer;
  readonly worklistIndexKind: "level-local";
}

export interface FactorOneDenseShadowDiscrepancy {
  readonly level: number;
  readonly coordinate: FactorOneDenseCoordinate;
  readonly field: "flags" | "owner" | "diagonal" | "worklist"
    | "parent" | "neighbour" | "spectral" | "publication";
  readonly expected: number;
  readonly actual: number;
}

export interface FactorOneDenseShadowDifferential {
  readonly acceptedValid: number;
  readonly acceptedEpoch: number;
  readonly acceptedError: number;
  readonly occupiedCounts: readonly number[];
  readonly discrepancies: readonly FactorOneDenseShadowDiscrepancy[];
}

function channel(
  plan: FactorOneDenseGpuPhysicalPlan,
  name: FactorOneDenseGpuChannelName,
) {
  const result = plan.channels.find((candidate) => candidate.name === name);
  if (!result) throw new RangeError(`Missing factor-1 dense channel ${name}`);
  return result;
}

function checkedSource(
  dimensions: FactorOneDenseCoordinate,
  sparse: FactorOneDenseSparseShadowPlan,
): void {
  if (sparse.levelCount < 1 || sparse.levelCount > MAX_LEVELS
    || sparse.levelCapacities.length !== sparse.levelCount
    || sparse.levelOffsets.length !== sparse.levelCount
    || sparse.levelOffsets.some((base, level) =>
      !Number.isSafeInteger(base) || base < 0
      || !Number.isSafeInteger(sparse.levelCapacities[level])
      || sparse.levelCapacities[level]! < 1)
    || sparse.levelOffsets.at(-1)! + sparse.levelCapacities.at(-1)!
      !== sparse.totalLevelSlots
    || !Number.isSafeInteger(sparse.worklistBaseWords) || sparse.worklistBaseWords < 0) {
    throw new RangeError("Invalid sparse source plan for the factor-1 dense shadow");
  }
  const dense = planFactorOneDensePressureHierarchy(dimensions);
  if (dense.levelCount !== sparse.levelCount) {
    throw new RangeError("Sparse and dense factor-1 hierarchy level counts disagree");
  }
}

function writeTable(words: Uint32Array, offset: number, values: readonly number[]): void {
  values.forEach((value, index) => { words[offset + index] = value; });
}

export class WebGPUFactorOneDensePressureShadow {
  readonly encodedCandidateDispatchCount = 4;
  readonly encodedCommitDispatchCount = 2;
  readonly encodedSetupDispatchCount =
    this.encodedCandidateDispatchCount + this.encodedCommitDispatchCount;
  readonly hierarchyPlan;
  readonly physicalPlan: FactorOneDenseGpuPhysicalPlan;
  readonly arena: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly groups: Readonly<Record<
    "clearCandidate" | "scatterCandidate" | "buildCandidateWorklists",
    GPUBindGroup
  >>;
  private readonly transactionGroups: Array<Readonly<{
    solverControl: GPUBuffer;
    finalizeCandidate: GPUBindGroup;
    copyCandidateToAccepted: GPUBindGroup;
    finalizeAccepted: GPUBindGroup;
  }>> = [];
  private readonly pipelines: Readonly<Record<
    "clearCandidate" | "scatterCandidate" | "buildCandidateWorklists"
    | "finalizeCandidate" | "copyCandidateToAccepted" | "finalizeAccepted",
    GPUComputePipeline
  >>;
  private readonly scatterDispatch: readonly [number, number, number];
  private readonly denseDispatch: readonly [number, number, number];
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: FactorOneDenseSparseShadowSource,
  ) {
    checkedSource(source.dimensions, source.sparsePlan);
    this.hierarchyPlan = planFactorOneDensePressureHierarchy(source.dimensions);
    this.physicalPlan = planFactorOneDenseGpuPhysicalImage(this.hierarchyPlan);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
    this.arena = device.createBuffer({
      label: "Factor-1 dense pressure shadow x-fast SoA arena",
      size: this.physicalPlan.totalBytes,
      usage: storage,
    });
    this.params = device.createBuffer({
      label: "Factor-1 dense pressure shadow physical plan",
      size: PARAM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const words = new Uint32Array(PARAM_WORDS);
    words.set([
      source.dimensions[0], source.dimensions[1], source.dimensions[2],
      this.hierarchyPlan.levelCount,
      source.sparsePlan.totalLevelSlots, this.hierarchyPlan.totalSlots,
      source.sparsePlan.worklistBaseWords, this.physicalPlan.channelStrideElements,
      channel(this.physicalPlan, "acceptedFlags").offsetBytes / 4,
      channel(this.physicalPlan, "acceptedOwners").offsetBytes / 4,
      channel(this.physicalPlan, "acceptedDiagonal").offsetBytes / 4,
      this.physicalPlan.spectralUpper.offsetBytes / 4,
      channel(this.physicalPlan, "candidateFlags").offsetBytes / 4,
      channel(this.physicalPlan, "candidateOwners").offsetBytes / 4,
      channel(this.physicalPlan, "candidateDiagonal").offsetBytes / 4,
      this.physicalPlan.candidateSpectralUpper.offsetBytes / 4,
      this.physicalPlan.acceptedOccupiedIndices.offsetBytes / 4,
      this.physicalPlan.candidateOccupiedIndices.offsetBytes / 4,
      this.physicalPlan.acceptedOccupiedCounts.offsetBytes / 4,
      this.physicalPlan.candidateOccupiedCounts.offsetBytes / 4,
      this.physicalPlan.publicationControl.offsetBytes / 4,
      0, 0, 0,
    ]);
    writeTable(words, 32, this.hierarchyPlan.levelBases);
    writeTable(words, 44, this.hierarchyPlan.levelVolumes);
    writeTable(words, 56, source.sparsePlan.levelOffsets);
    writeTable(words, 68, source.sparsePlan.levelCapacities);
    device.queue.writeBuffer(this.params, 0, words);

    const shaderModule = device.createShaderModule({
      label: "Factor-1 dense pressure shadow publisher",
      code: factorOneDensePressureShadowShader,
    });
    const names = [
      "clearCandidate", "scatterCandidate", "buildCandidateWorklists",
      "finalizeCandidate", "copyCandidateToAccepted", "finalizeAccepted",
    ] as const;
    this.pipelines = Object.freeze(Object.fromEntries(names.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({
        label: `Factor-1 dense shadow · ${entryPoint}`,
        layout: "auto",
        compute: { module: shaderModule, entryPoint },
      }),
    ])) as Record<typeof names[number], GPUComputePipeline>);
    const resources = new Map<number, GPUBuffer>([
      [0, this.params],
      [1, source.acceptedTopology],
      [2, source.acceptedState],
      [3, source.acceptedDispatch],
      [4, source.candidateTopology],
      [5, source.candidateState],
      [6, source.candidateDispatch],
      [7, source.levelDelta],
      [8, source.captureControl],
      [10, this.arena],
    ]);
    const bindings = {
      clearCandidate: [0, 8, 10],
      scatterCandidate: [0, 1, 2, 3, 4, 5, 6, 7, 10],
      buildCandidateWorklists: [0, 2, 7, 10],
    } as const;
    const staticNames = [
      "clearCandidate", "scatterCandidate", "buildCandidateWorklists",
    ] as const;
    this.groups = Object.freeze(Object.fromEntries(staticNames.map((name) => [
      name,
      device.createBindGroup({
        label: `Factor-1 dense shadow · ${name} sources`,
        layout: this.pipelines[name].getBindGroupLayout(0),
        entries: bindings[name].map((binding) => ({
          binding,
          resource: { buffer: resources.get(binding)! },
        })),
      }),
    ])) as Record<typeof staticNames[number], GPUBindGroup>);
    const scatterGroups = Math.ceil(Math.max(...source.sparsePlan.levelCapacities) / 64);
    const denseGroups = Math.ceil(Math.max(
      this.hierarchyPlan.totalSlots,
      this.hierarchyPlan.levelCount * 4,
    ) / 64);
    if (scatterGroups > 65_535 || denseGroups > 65_535) {
      throw new RangeError("Factor-1 dense shadow exceeds the prototype dispatch width");
    }
    this.scatterDispatch = [Math.max(1, scatterGroups), this.hierarchyPlan.levelCount, 1];
    this.denseDispatch = [Math.max(1, denseGroups), 1, 1];
    this.allocatedBytes = this.arena.size + this.params.size;
  }

  acceptedView(rowGeometry: GPUBuffer): FactorOneDenseAcceptedView {
    this.assertLive();
    return Object.freeze({
      arena: this.arena,
      physicalPlan: this.physicalPlan,
      rowGeometry,
      worklistIndexKind: "level-local",
    });
  }

  encodeCandidate(broker: PassBroker, solverControl: GPUBuffer): void {
    this.assertLive();
    const pass = broker.compute({ label: "Factor-1 dense shadow · build candidate" });
    pass.setPipeline(this.pipelines.clearCandidate);
    pass.setBindGroup(0, this.groups.clearCandidate);
    pass.dispatchWorkgroups(...this.denseDispatch);
    pass.setPipeline(this.pipelines.scatterCandidate);
    pass.setBindGroup(0, this.groups.scatterCandidate);
    pass.dispatchWorkgroups(...this.scatterDispatch);
    pass.setPipeline(this.pipelines.buildCandidateWorklists);
    pass.setBindGroup(0, this.groups.buildCandidateWorklists);
    pass.dispatchWorkgroups(this.hierarchyPlan.levelCount);
    pass.setPipeline(this.pipelines.finalizeCandidate);
    pass.setBindGroup(0, this.transactionGroup(solverControl).finalizeCandidate);
    pass.dispatchWorkgroups(1);
  }

  encodeCommit(broker: PassBroker, solverControl: GPUBuffer): void {
    this.assertLive();
    const pass = broker.compute({ label: "Factor-1 dense shadow · commit accepted" });
    pass.setPipeline(this.pipelines.copyCandidateToAccepted);
    const transaction = this.transactionGroup(solverControl);
    pass.setBindGroup(0, transaction.copyCandidateToAccepted);
    pass.dispatchWorkgroups(...this.denseDispatch);
    pass.setPipeline(this.pipelines.finalizeAccepted);
    pass.setBindGroup(0, transaction.finalizeAccepted);
    pass.dispatchWorkgroups(1);
  }

  async readDifferential(): Promise<FactorOneDenseShadowDifferential> {
    this.assertLive();
    const sparse = this.source.sparsePlan;
    const denseBytes = this.physicalPlan.totalBytes;
    const sparseChannelBytes = sparse.totalLevelSlots * 4;
    const sparseWorklistBytes = sparse.totalLevelSlots * 4;
    const sparseDispatchBytes = sparse.levelCount * SPARSE_DISPATCH_WORDS * 4;
    const totalBytes = denseBytes + 5 * sparseChannelBytes
      + sparseWorklistBytes + sparseDispatchBytes;
    const readback = this.device.createBuffer({
      label: "Factor-1 dense shadow coordinate differential readback",
      size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read factor-1 dense shadow and sparse authority",
    });
    let offset = 0;
    encoder.copyBufferToBuffer(this.arena, 0, readback, offset, denseBytes);
    offset += denseBytes;
    for (const sparseChannel of [0, 1, 2, 24, 25]) {
      encoder.copyBufferToBuffer(
        this.source.acceptedState,
        sparseChannel * sparseChannelBytes,
        readback,
        offset,
        sparseChannelBytes,
      );
      offset += sparseChannelBytes;
    }
    encoder.copyBufferToBuffer(
      this.source.acceptedTopology,
      sparse.worklistBaseWords * 4,
      readback,
      offset,
      sparseWorklistBytes,
    );
    offset += sparseWorklistBytes;
    encoder.copyBufferToBuffer(
      this.source.acceptedDispatch, 0, readback, offset, sparseDispatchBytes,
    );
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return decodeFactorOneDenseShadowDifferential(
        this.physicalPlan,
        sparse,
        new Uint32Array(readback.getMappedRange()),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.arena.destroy();
    this.params.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Factor-1 dense pressure shadow is destroyed");
  }

  private transactionGroup(solverControl: GPUBuffer) {
    const old = this.transactionGroups.find((entry) =>
      entry.solverControl === solverControl);
    if (old) return old;
    const resources = new Map<number, GPUBuffer>([
      [0, this.params],
      [3, this.source.acceptedDispatch],
      [6, this.source.candidateDispatch],
      [7, this.source.levelDelta],
      [8, this.source.captureControl],
      [9, solverControl],
      [10, this.arena],
    ]);
    const make = (
      name: "finalizeCandidate" | "copyCandidateToAccepted" | "finalizeAccepted",
      bindings: readonly number[],
    ) => this.device.createBindGroup({
      label: `Factor-1 dense shadow · ${name} transaction sources`,
      layout: this.pipelines[name].getBindGroupLayout(0),
      entries: bindings.map((binding) => ({
        binding,
        resource: { buffer: resources.get(binding)! },
      })),
    });
    const created = Object.freeze({
      solverControl,
      finalizeCandidate: make("finalizeCandidate", [0, 3, 6, 7, 8, 9, 10]),
      copyCandidateToAccepted: make("copyCandidateToAccepted", [0, 3, 8, 9, 10]),
      finalizeAccepted: make("finalizeAccepted", [0, 3, 8, 9, 10]),
    });
    this.transactionGroups.push(created);
    return created;
  }
}

export function decodeFactorOneDenseShadowDifferential(
  physical: FactorOneDenseGpuPhysicalPlan,
  sparse: FactorOneDenseSparseShadowPlan,
  readbackWords: Uint32Array,
): FactorOneDenseShadowDifferential {
  const denseWordCount = physical.totalBytes / 4;
  const sparseWords = sparse.totalLevelSlots;
  const required = denseWordCount + 6 * sparseWords
    + sparse.levelCount * SPARSE_DISPATCH_WORDS;
  if (readbackWords.length < required) {
    throw new RangeError("Factor-1 dense shadow differential readback is truncated");
  }
  const dense = readbackWords.subarray(0, denseWordCount);
  let cursor = denseWordCount;
  const keys = readbackWords.subarray(cursor, cursor += sparseWords);
  const flags = readbackWords.subarray(cursor, cursor += sparseWords);
  const diagonals = readbackWords.subarray(cursor, cursor += sparseWords);
  const owners = readbackWords.subarray(cursor, cursor += sparseWords);
  const spectral = readbackWords.subarray(cursor, cursor += sparseWords);
  const sparseWorklists = readbackWords.subarray(cursor, cursor += sparseWords);
  const sparseDispatch = readbackWords.subarray(cursor);
  const word = (bytes: number) => bytes / 4;
  const acceptedFlags = word(channel(physical, "acceptedFlags").offsetBytes);
  const acceptedOwners = word(channel(physical, "acceptedOwners").offsetBytes);
  const acceptedDiagonal = word(channel(physical, "acceptedDiagonal").offsetBytes);
  const acceptedIndices = word(physical.acceptedOccupiedIndices.offsetBytes);
  const control = word(physical.publicationControl.offsetBytes);
  const discrepancies: FactorOneDenseShadowDiscrepancy[] = [];
  const occupiedCounts: number[] = [];
  const push = (
    level: number,
    local: number,
    field: FactorOneDenseShadowDiscrepancy["field"],
    expected: number,
    actual: number,
  ) => discrepancies.push(Object.freeze({
    level,
    coordinate: factorOneDenseCoordinate(
      physical.hierarchy, level, physical.hierarchy.levelBases[level]! + local,
    ),
    field,
    expected,
    actual,
  }));

  for (let level = 0; level < physical.hierarchy.levelCount; level += 1) {
    const sparseBase = sparse.levelOffsets[level]!;
    const denseBase = physical.hierarchy.levelBases[level]!;
    const sparseCount = sparseDispatch[level * SPARSE_DISPATCH_WORDS]!;
    const denseRecord = physical.worklistLevels[level]!;
    const denseCount = dense[word(denseRecord.acceptedCountOffsetBytes)]!;
    occupiedCounts.push(denseCount);
    if (denseCount !== sparseCount) push(level, 0, "worklist", sparseCount, denseCount);
    const sparseByLocal = new Map<number, number>();
    for (let item = 0; item < sparseCount; item += 1) {
      const slot = sparseWorklists[sparseBase + item]!;
      const key = keys[sparseBase + slot]!;
      if (key !== 0) sparseByLocal.set(key - 1, slot);
    }
    const denseListed = new Set<number>();
    for (let item = 0; item < denseCount; item += 1) {
      denseListed.add(dense[acceptedIndices + denseBase + item]!);
    }
    for (const [local, slot] of sparseByLocal) {
      const semantic = denseBase + local;
      if (!denseListed.has(local)) push(level, local, "worklist", 1, 0);
      for (const [field, expected, actual] of [
        ["flags", flags[sparseBase + slot]!, dense[acceptedFlags + semantic]!],
        ["owner", owners[sparseBase + slot]!, dense[acceptedOwners + semantic]!],
        ["diagonal", diagonals[sparseBase + slot]!, dense[acceptedDiagonal + semantic]!],
      ] as const) if (expected !== actual) push(level, local, field, expected, actual);
      const role = dense[acceptedFlags + semantic]!;
      const owner = dense[acceptedOwners + semantic]!;
      const owned = role === FACTOR_ONE_DENSE_CELL_ROLE.active
        || role === FACTOR_ONE_DENSE_CELL_ROLE.ghost;
      if (owned !== (owner !== 0)) push(level, local, "owner", Number(owned), Number(owner !== 0));
      if (level + 1 < physical.hierarchy.levelCount) {
        const coordinate = factorOneDenseCoordinate(physical.hierarchy, level, semantic);
        const parent = factorOneDenseSlot(
          physical.hierarchy,
          level + 1,
          coordinate.map((value) => Math.floor(value / 2)) as [number, number, number],
        );
        if (dense[acceptedFlags + parent] === 0) push(level, local, "parent", 1, 0);
      }
      const coordinate = factorOneDenseCoordinate(physical.hierarchy, level, semantic);
      const dimensions = physical.hierarchy.levelDimensions[level]!;
      for (const direction of [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
      ] as const) {
        const neighbour = coordinate.map((value, axis) =>
          value + direction[axis]) as [number, number, number];
        if (neighbour.some((value, axis) => value < 0 || value >= dimensions[axis])) continue;
        const neighbourLocal = factorOneDenseSlot(
          physical.hierarchy, level, neighbour,
        ) - denseBase;
        const expected = Number(sparseByLocal.has(neighbourLocal));
        const actual = Number(dense[acceptedFlags + denseBase + neighbourLocal] !== 0);
        if (expected !== actual) push(level, local, "neighbour", expected, actual);
      }
    }
    const sparseSpectral = spectral[sparseBase]!;
    const denseSpectral = dense[word(physical.spectralUpper.offsetBytes) + level]!;
    if (sparseSpectral !== denseSpectral) {
      push(level, 0, "spectral", sparseSpectral, denseSpectral);
    }
  }
  return Object.freeze({
    acceptedValid: dense[control + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedValid]!,
    acceptedEpoch: dense[control + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedEpoch]!,
    acceptedError: dense[control + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedError]!,
    occupiedCounts: Object.freeze(occupiedCounts),
    discrepancies: Object.freeze(discrepancies),
  });
}

export const factorOneDensePressureShadowShader = /* wgsl */ `
struct Params {
  dimsLevels: vec4u,
  totals: vec4u,
  accepted: vec4u,
  candidate: vec4u,
  worklists: vec4u,
  control: vec4u,
  reserved0: vec4u,
  reserved1: vec4u,
  denseBases: array<vec4u,3>,
  denseVolumes: array<vec4u,3>,
  sparseBases: array<vec4u,3>,
  sparseCaps: array<vec4u,3>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> acceptedTopology: array<u32>;
@group(0) @binding(2) var<storage,read> acceptedState: array<u32>;
@group(0) @binding(3) var<storage,read> acceptedDispatch: array<u32>;
@group(0) @binding(4) var<storage,read> candidateTopology: array<u32>;
@group(0) @binding(5) var<storage,read> candidateState: array<u32>;
@group(0) @binding(6) var<storage,read> candidateDispatch: array<u32>;
@group(0) @binding(7) var<storage,read> levelDelta: array<u32>;
@group(0) @binding(8) var<storage,read> capture: array<u32>;
@group(0) @binding(9) var<storage,read> solverControl: array<u32>;
@group(0) @binding(10) var<storage,read_write> dense: array<atomic<u32>>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const OWNER=24u;const SPECTRAL=25u;
const STATE_CHANNELS=26u;const DISPATCH_WORDS=12u;const DELTA_WORDS=8u;
const DELTA_TOPOLOGY=1u;const DELTA_STENCIL=2u;const RECORD_WORDS=4u;
const C_ACCEPTED_VALID=0u;const C_ACCEPTED_EPOCH=1u;const C_ACCEPTED_ERROR=2u;
const C_CANDIDATE_VALID=4u;const C_CANDIDATE_EPOCH=5u;const C_CANDIDATE_ERROR=6u;
const E_COORD=1u;const E_ROLE=2u;const E_OWNER=4u;const E_DIAG=8u;
const E_COUNT=16u;const E_PARENT=32u;const E_TRANSACTION=64u;
fn table(values:ptr<uniform,array<vec4u,3>>,l:u32)->u32{return (*values)[l>>2u][l&3u];}
fn levels()->u32{return p.dimsLevels.w;}
fn denseBase(l:u32)->u32{return table(&p.denseBases,l);}
fn denseVolume(l:u32)->u32{return table(&p.denseVolumes,l);}
fn sparseBase(l:u32)->u32{return table(&p.sparseBases,l);}
fn sparseCap(l:u32)->u32{return table(&p.sparseCaps,l);}
fn sparseAt(channel:u32,l:u32,slot:u32)->u32{return channel*p.totals.x+sparseBase(l)+slot;}
fn topologyDirty(l:u32)->bool{return (levelDelta[l*DELTA_WORDS+1u]&DELTA_TOPOLOGY)!=0u;}
fn stencilDirty(l:u32)->bool{return (levelDelta[l*DELTA_WORDS+1u]&DELTA_STENCIL)!=0u;}
fn selectedCount(l:u32)->u32{return select(acceptedDispatch[l*DISPATCH_WORDS],
 candidateDispatch[l*DISPATCH_WORDS],topologyDirty(l));}
fn selectedSlot(l:u32,item:u32)->u32{let at=p.totals.z+sparseBase(l)+item;
 return select(acceptedTopology[at],candidateTopology[at],topologyDirty(l));}
fn selectedTopologyState(channel:u32,l:u32,slot:u32)->u32{
 let at=sparseAt(channel,l,slot);return select(acceptedState[at],candidateState[at],topologyDirty(l));}
fn selectedDiagonal(l:u32,slot:u32)->u32{let at=sparseAt(DIAG,l,slot);
 return select(acceptedState[at],candidateState[at],stencilDirty(l));}
fn selectedSpectral(l:u32)->u32{return select(acceptedState[sparseAt(SPECTRAL,l,0u)],
 levelDelta[l*DELTA_WORDS+7u],stencilDirty(l));}
fn candidateControl(word:u32)->u32{return p.control.x+C_CANDIDATE_VALID+word;}
fn report(error:u32){atomicOr(&dense[p.control.x+C_CANDIDATE_ERROR],error);}
fn dims(l:u32)->vec3u{let scale=1u<<l;return (p.dimsLevels.xyz+vec3u(scale-1u))/vec3u(scale);}
fn localCoordinate(local:u32,l:u32)->vec3u{let d=dims(l);let row=local/d.x;
 return vec3u(local-row*d.x,row%d.y,row/d.y);}
fn localIndex(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z);}
fn validRole(role:u32)->bool{return role==ACTIVE||role==GHOST||role==MG_ONLY;}
fn finiteBits(bits:u32)->bool{let value=bitcast<f32>(bits);
 return value==value&&abs(value)<=3.402823e38;}
@compute @workgroup_size(64) fn clearCandidate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i<p.totals.y){
  atomicStore(&dense[p.candidate.x+i],0u);atomicStore(&dense[p.candidate.y+i],0u);
  atomicStore(&dense[p.candidate.z+i],0u);atomicStore(&dense[p.worklists.y+i],0u);}
 if(i<levels()*RECORD_WORDS){atomicStore(&dense[p.worklists.w+i],0u);}
 if(i<levels()){atomicStore(&dense[p.candidate.w+i],0u);}
 if(i==0u){atomicStore(&dense[p.control.x+C_CANDIDATE_VALID],0u);
  atomicStore(&dense[p.control.x+C_CANDIDATE_EPOCH],select(0u,capture[0],arrayLength(&capture)>0u));
  atomicStore(&dense[p.control.x+C_CANDIDATE_ERROR],0u);}
}
@compute @workgroup_size(64) fn scatterCandidate(@builtin(global_invocation_id) gid:vec3u){
 let item=gid.x;let l=gid.y;if(l>=levels()||item>=selectedCount(l)){return;}
 let slot=selectedSlot(l,item);if(slot>=sparseCap(l)){report(E_COORD);return;}
 let key=selectedTopologyState(KEY,l,slot);if(key==0u||key-1u>=denseVolume(l)){report(E_COORD);return;}
 let role=selectedTopologyState(FLAGS,l,slot);let owner=selectedTopologyState(OWNER,l,slot);
 let diagonal=selectedDiagonal(l,slot);
 if(!validRole(role)){report(E_ROLE);}
 if(((role==ACTIVE||role==GHOST)!=(owner!=0u))){report(E_OWNER);}
 if(!finiteBits(diagonal)||!(bitcast<f32>(diagonal)>0.0)){report(E_DIAG);}
 let semantic=denseBase(l)+key-1u;
 atomicStore(&dense[p.candidate.x+semantic],role);
 atomicStore(&dense[p.candidate.y+semantic],owner);
 atomicStore(&dense[p.candidate.z+semantic],diagonal);
}
@compute @workgroup_size(1) fn buildCandidateWorklists(@builtin(workgroup_id) wid:vec3u){
 let l=wid.x;if(l>=levels()){return;}let base=denseBase(l);var count=0u;
 for(var local=0u;local<denseVolume(l);local+=1u){
  if(atomicLoad(&dense[p.candidate.x+base+local])!=0u){
   atomicStore(&dense[p.worklists.y+base+count],local);count+=1u;}}
 let record=p.worklists.w+l*RECORD_WORDS;atomicStore(&dense[record],count);
 atomicStore(&dense[record+1u],(count+63u)/64u);
 atomicStore(&dense[record+2u],1u);atomicStore(&dense[record+3u],1u);
 atomicStore(&dense[p.candidate.w+l],selectedSpectral(l));
}
@compute @workgroup_size(1) fn finalizeCandidate(){
 var error=atomicLoad(&dense[p.control.x+C_CANDIDATE_ERROR]);
 for(var l=0u;l<levels();l+=1u){
  let base=denseBase(l);let count=atomicLoad(&dense[p.worklists.w+l*RECORD_WORDS]);
  if(count!=selectedCount(l)){error|=E_COUNT;}
  for(var item=0u;item<count;item+=1u){let local=atomicLoad(&dense[p.worklists.y+base+item]);
   let role=atomicLoad(&dense[p.candidate.x+base+local]);
   let owner=atomicLoad(&dense[p.candidate.y+base+local]);
   let diagonal=atomicLoad(&dense[p.candidate.z+base+local]);
   if(!validRole(role)){error|=E_ROLE;}
   if(((role==ACTIVE||role==GHOST)!=(owner!=0u))){error|=E_OWNER;}
   if(!finiteBits(diagonal)||!(bitcast<f32>(diagonal)>0.0)){error|=E_DIAG;}
   if(l+1u<levels()){let parent=localIndex(localCoordinate(local,l)/vec3u(2u),l+1u);
    if(atomicLoad(&dense[p.candidate.x+denseBase(l+1u)+parent])==0u){error|=E_PARENT;}}}}
 if(atomicLoad(&dense[p.worklists.w+(levels()-1u)*RECORD_WORDS])!=1u){error|=E_PARENT;}
 if(arrayLength(&capture)<=8u||arrayLength(&solverControl)==0u||capture[0]==0u
  ||capture[8]!=0u||solverControl[0]!=0u){error|=E_TRANSACTION;}
 atomicStore(&dense[p.control.x+C_CANDIDATE_ERROR],error);
 atomicStore(&dense[p.control.x+C_CANDIDATE_VALID],select(1u,0u,error!=0u));
}
fn readyToCommit()->bool{return atomicLoad(&dense[p.control.x+C_CANDIDATE_VALID])==1u
 &&arrayLength(&capture)>8u&&arrayLength(&solverControl)>0u
 &&capture[6]!=0u&&capture[6]==atomicLoad(&dense[p.control.x+C_CANDIDATE_EPOCH])
 &&capture[8]==0u&&solverControl[0]==0u
 &&acceptedDispatch[levels()*DISPATCH_WORDS]==1u;}
@compute @workgroup_size(64) fn copyCandidateToAccepted(@builtin(global_invocation_id) gid:vec3u){
 if(!readyToCommit()){return;}let i=gid.x;if(i<p.totals.y){
  atomicStore(&dense[p.accepted.x+i],atomicLoad(&dense[p.candidate.x+i]));
  atomicStore(&dense[p.accepted.y+i],atomicLoad(&dense[p.candidate.y+i]));
  atomicStore(&dense[p.accepted.z+i],atomicLoad(&dense[p.candidate.z+i]));
  atomicStore(&dense[p.worklists.x+i],atomicLoad(&dense[p.worklists.y+i]));}
 if(i<levels()){atomicStore(&dense[p.accepted.w+i],atomicLoad(&dense[p.candidate.w+i]));
  for(var word=0u;word<RECORD_WORDS;word+=1u){
   atomicStore(&dense[p.worklists.z+i*RECORD_WORDS+word],
    atomicLoad(&dense[p.worklists.w+i*RECORD_WORDS+word]));}}
}
@compute @workgroup_size(1) fn finalizeAccepted(){
 if(!readyToCommit()){return;}
 atomicStore(&dense[p.control.x+C_ACCEPTED_ERROR],0u);
 atomicStore(&dense[p.control.x+C_ACCEPTED_EPOCH],
  atomicLoad(&dense[p.control.x+C_CANDIDATE_EPOCH]));
 atomicStore(&dense[p.control.x+C_ACCEPTED_VALID],1u);
}
`;
