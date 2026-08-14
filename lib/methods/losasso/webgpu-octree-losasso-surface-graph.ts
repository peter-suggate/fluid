import { WebGPURadixSortU32 } from "../../core/webgpu-radix-sort-u32";
import { octreeLosassoSurfaceGraphWGSL } from "./webgpu-octree-losasso-surface-graph.wgsl";
import type { PassBroker } from "../../core/webgpu-pass-broker";

export const LOSASSO_SURFACE_GRAPH_CONTROL_WORDS = 36;
export const LOSASSO_SURFACE_GRAPH_PUBLISHED = 0x8000_0000;
export const LOSASSO_SURFACE_GRAPH_LEAF_WORDS = 16;
export const LOSASSO_SURFACE_GRAPH_NODE_WORDS = 4;
export const LOSASSO_SURFACE_GRAPH_NODE_DIRECTORY_WORDS = 2;
export const LOSASSO_SURFACE_GRAPH_LEAF_DIRECTORY_WORDS = 4;
export const LOSASSO_SURFACE_GRAPH_CONSTRAINT_WORDS = 12;
export const LOSASSO_SURFACE_GRAPH_ADJACENCY_WORDS = 12;
export const LOSASSO_SURFACE_GRAPH_INCIDENT_LEAF_WORDS = 8;
/** Transaction header followed by one graph-leaf slot per cell in each
 * resident 8^3 owner page. The existing owner directory supplies the sparse
 * top-level page lookup; this graph never allocates a finest-domain lattice. */
export const LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_HEADER_WORDS = 16;
export const LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_MAGIC = 0x4c4f_4341;
export const LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR = Object.freeze({
  magic: 0, epoch: 1, ownerPageCount: 2, supportCellCount: 3,
  leafCount: 4, nodeCount: 5, published: 6, errors: 7,
  ownerCapacity: 8, logicalBrickCount: 9, ownerDirectoryOffset: 10,
  payloadOffset: 11, compiledCells: 12,
} as const);

export function planLosassoSurfaceGraphLeafLocatorBytes(ownerPageCapacity: number): number {
  return 4 * (LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_HEADER_WORDS
    + 512 * positiveU32(ownerPageCapacity, "owner page capacity"));
}

export const LOSASSO_SURFACE_GRAPH_CONTROL = Object.freeze({
  epoch: 0, leafCount: 1, nodeCount: 2, published: 3, errors: 4,
  surfaceGeneration: 5, velocityGeneration: 6, constraintCount: 7,
  independentNodeCount: 8, edgeHangingNodeCount: 9, faceHangingNodeCount: 10,
  missingLookupCount: 11, maximumDirectoryProbes: 12,
  /** Bitcast f32 compatibility threshold carried across topology handoffs. */
  reconstructionThresholdBits: 13,
  leafCapacity: 14, nodeCapacity: 15, constraintCapacity: 16,
  leafDispatch: 17, nodeDispatch: 20, coverageErrors: 24,
  reciprocalAdjacencyErrors: 25, leafClosureErrors: 26, capacityErrors: 27,
  pressureRowCount: 28, pressureRowMappingErrors: 29,
  ownerPageCount: 30, supportCellCount: 31,
  /** One workgroup per 256 live nodes for dense topology packet compilation. */
  topologyBlockDispatch: 32,
} as const);

export const LOSASSO_SURFACE_GRAPH_ERROR = Object.freeze({
  capacity: 1 << 0, topology: 1 << 1, geometry: 1 << 2,
  ordering: 1 << 3, lookup: 1 << 4, constraint: 1 << 5,
  adjacency: 1 << 6, generation: 1 << 7, coverage: 1 << 8,
} as const);

export const LOSASSO_SURFACE_GRAPH_NODE_FLAG = Object.freeze({
  independent: 1 << 0, edgeHanging: 1 << 1, faceHanging: 1 << 2,
  boundary: 1 << 3,
} as const);

/** Four-word readiness ABI used by the topology and joint-ready publishers. */
export interface LosassoSurfaceGraphControlLayout {
  readonly epochWord: number;
  readonly countWord: number;
  readonly validWord: number;
  readonly errorWord: number;
  readonly validValue?: number;
}

export interface WebGPUOctreeLosassoSurfaceGraphInput {
  /** Candidate 48-byte wet pressure-row headers; graph leaves are not sourced here. */
  readonly candidateLeafHeaders: GPUBuffer;
  /** Candidate topology authority. Default layout is [epoch,count,*,valid,error]. */
  readonly candidateTopologyControl: GPUBuffer;
  /** Stable owner arena containing the fully refined inactive candidate bank. */
  readonly candidateOwnerArena: GPUBuffer;
  /** Owner transaction: epoch word 24, candidate page count 27, valid word 28. */
  readonly candidateOwnerTransaction: GPUBuffer;
  /** Joint topology/row/face ready publication. Defaults to candidateTopologyControl. */
  readonly readyControl?: GPUBuffer;
  readonly topologyControlLayout?: LosassoSurfaceGraphControlLayout;
  readonly readyControlLayout?: LosassoSurfaceGraphControlLayout;
}

export interface WebGPUOctreeLosassoSurfaceGraphOptions {
  /** Wet pressure-row capacity. */
  readonly rowCapacity: number;
  readonly ownerPages: Readonly<{
    readonly capacity: number;
    readonly logicalBrickCount: number;
    readonly ownerDirectoryOffsetWords: number;
    readonly ownerPagesOffsetWords: number;
  }>;
  /** Defaults to the exact finest-leaf bound over resident owner pages. */
  readonly leafCapacity?: number;
  /** Defaults to a conservative four compact nodes per support leaf, capped by the lattice. */
  readonly nodeCapacity?: number;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly physicalCellSize: readonly [number, number, number];
}

/** Exact sparse-support bound: every cell in a resident owner page can name a
 * distinct finest leaf. The graph already emits all 512 cells per page, so a
 * smaller wet-row/sheet heuristic is not an allocation bound and can reject a
 * geometrically valid candidate before pressure-row mapping begins. */
export function planLosassoSurfaceGraphLeafCapacity(
  pressureRowCapacity: number,
  ownerPageCapacity: number,
): number {
  positiveU32(pressureRowCapacity, "pressure row capacity");
  const pages = positiveU32(ownerPageCapacity, "owner page capacity");
  return 512 * pages;
}

/**
 * Compact-node planning deliberately does not reserve the eight emitted radix
 * items per leaf: most of those items are shared by the closed owner support.
 * A pathological sparse support fails closed during candidate publication and
 * can opt into a larger explicit nodeCapacity without changing buffer identity.
 */
export function planLosassoSurfaceGraphNodeCapacity(
  leafCapacity: number,
  dimensions: readonly [number, number, number],
): number {
  const leaves = positiveU32(leafCapacity, "leaf capacity");
  const nodeVolume = dimensions.reduce((volume, dimension) =>
    volume * (positiveU32(dimension, "dimension") + 1), 1);
  if (!Number.isSafeInteger(nodeVolume) || nodeVolume > 0xffff_ffff) {
    throw new RangeError("Losasso surface graph node lattice must fit u32");
  }
  return Math.min(4 * leaves, nodeVolume);
}

export interface LosassoSurfaceGraphBankSource {
  /** See LOSASSO_SURFACE_GRAPH_CONTROL. Includes indirect leaf/node dispatch records. */
  readonly control: GPUBuffer;
  /** 64-byte records: origin.xyz, span, flags/meta, then eight compact corner slots. */
  readonly leaves: GPUBuffer;
  /** Sorted 16-byte (linear cell, span, leaf slot, flags) records. */
  readonly leafDirectory: GPUBuffer;
  /** 16-byte (finest-lattice item, flags, local span, constraint offset) records. */
  readonly nodes: GPUBuffer;
  /** Sorted 8-byte (finest-lattice item, compact slot) records. */
  readonly nodeDirectory: GPUBuffer;
  /** One 48-byte exact rational record per node: kind/count/denominator, masters, numerators. */
  readonly constraints: GPUBuffer;
  /** One 48-byte record per node: six slots followed by six physical f32 spans. */
  readonly adjacency: GPUBuffer;
  /** Eight compiled containing-leaf slots, one for each signed octant at the node. */
  readonly incidentLeaves: GPUBuffer;
  /** Header plus owner-page-capacity * 512 direct cell-to-leaf slots. */
  readonly leafLocator: GPUBuffer;
  /** Per-leaf integrated surface mass. Never reconstructed from nodal phi. */
  readonly surfaceMass: GPUBuffer;
  /** Per-leaf cumulative conservative-advection compression factor. */
  readonly surfaceCompression: GPUBuffer;
  /** Per-node (old,new) raw-fluid phi. Field publishers own its generations. */
  readonly phi: GPUBuffer;
  /** Per-node accepted and predictor vec4 velocity records (32 bytes per node). */
  readonly nodalVelocity: GPUBuffer;
  /** Per-node component/field-valid bits. */
  readonly nodeValidity: GPUBuffer;
  /** Pressure-row-indexed graph leaf slot. */
  readonly pressureRowToGraphLeaf: GPUBuffer;
  readonly pressureRowCapacity: number;
  readonly leafCapacity: number;
  readonly nodeCapacity: number;
  readonly leafDispatchOffsetBytes: number;
  readonly nodeDispatchOffsetBytes: number;
  readonly topologyBlockDispatchOffsetBytes: number;
}

export interface LosassoSurfaceGraphSources {
  readonly accepted: LosassoSurfaceGraphBankSource;
  readonly candidate: LosassoSurfaceGraphBankSource;
}

export interface LosassoSurfaceGraphReceipt {
  readonly epoch: number;
  readonly leafCount: number;
  readonly nodeCount: number;
  readonly published: boolean;
  readonly errors: number;
  readonly surfaceGeneration: number;
  readonly velocityGeneration: number;
  readonly constraintCount: number;
  readonly independentNodeCount: number;
  readonly edgeHangingNodeCount: number;
  readonly faceHangingNodeCount: number;
  readonly missingLookupCount: number;
  readonly maximumDirectoryProbes: number;
  readonly coverageErrors: number;
  readonly reciprocalAdjacencyErrors: number;
  readonly leafClosureErrors: number;
  readonly pressureRowCount: number;
  readonly pressureRowMappingErrors: number;
  readonly ownerPageCount: number;
  readonly supportCellCount: number;
  readonly leafDispatch: readonly [number, number, number];
  readonly nodeDispatch: readonly [number, number, number];
  readonly topologyBlockDispatch: readonly [number, number, number];
}

type EntryPoint = "prepareSurfaceGraph" | "emitSurfaceGraphItems"
  | "prepareSurfaceGraphLeaves" | "buildSurfaceGraphLeavesAndNodeItems"
  | "prepareSurfaceGraphRecords" | "buildSurfaceGraphNodes"
  | "resolveSurfaceGraphLeaves" | "classifySurfaceGraphNodes"
  | "resolveSurfaceGraphPressureRows"
  | "compileSurfaceGraphLeafLocator"
  | "validateSurfaceGraphNodes" | "finishSurfaceGraph"
  | "prepareSurfaceGraphCommit" | "commitSurfaceGraphLeaves"
  | "commitSurfaceGraphSurfaceState"
  | "commitSurfaceGraphNodes" | "commitSurfaceGraphRelations"
  | "commitSurfaceGraphFields" | "commitSurfaceGraphPressureRows"
  | "commitSurfaceGraphLeafLocator"
  | "finishSurfaceGraphCommit";

const ENTRY_POINTS: readonly EntryPoint[] = [
  "prepareSurfaceGraph", "emitSurfaceGraphItems", "prepareSurfaceGraphLeaves",
  "buildSurfaceGraphLeavesAndNodeItems", "prepareSurfaceGraphRecords",
  "buildSurfaceGraphNodes", "resolveSurfaceGraphLeaves", "classifySurfaceGraphNodes",
  "resolveSurfaceGraphPressureRows", "compileSurfaceGraphLeafLocator",
  "validateSurfaceGraphNodes", "finishSurfaceGraph",
  "prepareSurfaceGraphCommit",
  "commitSurfaceGraphLeaves", "commitSurfaceGraphSurfaceState",
  "commitSurfaceGraphNodes", "commitSurfaceGraphRelations",
  "commitSurfaceGraphFields", "commitSurfaceGraphPressureRows",
  "commitSurfaceGraphLeafLocator", "finishSurfaceGraphCommit",
];

const BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  prepareSurfaceGraph: [0, 1, 3, 4, 8, 28, 29, 30, 31, 35],
  emitSurfaceGraphItems: [0, 5, 8, 29, 30],
  prepareSurfaceGraphLeaves: [0, 4, 6, 7, 8, 28],
  buildSurfaceGraphLeavesAndNodeItems: [0, 5, 6, 8, 9, 10, 29, 30],
  prepareSurfaceGraphRecords: [0, 7, 8, 28],
  buildSurfaceGraphNodes: [0, 6, 8, 11, 12, 13, 14, 17],
  resolveSurfaceGraphLeaves: [0, 8, 9, 10, 12],
  classifySurfaceGraphNodes: [0, 8, 9, 10, 11, 12, 13, 14, 33],
  resolveSurfaceGraphPressureRows: [0, 3, 8, 10, 31],
  compileSurfaceGraphLeafLocator: [0, 8, 9, 10, 29, 30, 35],
  validateSurfaceGraphNodes: [0, 8, 11, 13, 14],
  finishSurfaceGraph: [0, 8, 35],
  prepareSurfaceGraphCommit: [0, 2, 8, 28, 35],
  commitSurfaceGraphLeaves: [0, 8, 9, 10, 19, 20],
  commitSurfaceGraphSurfaceState: [0, 8, 37, 38, 39, 40],
  commitSurfaceGraphNodes: [0, 8, 11, 12, 21, 22],
  commitSurfaceGraphRelations: [0, 8, 13, 14, 23, 24, 33, 34],
  commitSurfaceGraphFields: [0, 8, 15, 16, 17, 25, 26, 27],
  commitSurfaceGraphPressureRows: [0, 8, 31, 32],
  commitSurfaceGraphLeafLocator: [0, 8, 30, 35, 36],
  finishSurfaceGraphCommit: [8, 18, 28, 35, 36],
});

function positiveU32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso surface graph ${label} must be a positive u32`);
  }
  return value;
}

function layout(value?: LosassoSurfaceGraphControlLayout): Required<LosassoSurfaceGraphControlLayout> {
  const result = { epochWord: value?.epochWord ?? 0, countWord: value?.countWord ?? 1,
    validWord: value?.validWord ?? 3, errorWord: value?.errorWord ?? 4,
    validValue: value?.validValue ?? 1 };
  for (const [name, word] of Object.entries(result)) {
    if (!Number.isSafeInteger(word) || word < 0 || word > 0xffff_ffff) {
      throw new RangeError(`Losasso surface graph ${name} must be a u32`);
    }
  }
  return result;
}

function unpackReceipt(words: Uint32Array): LosassoSurfaceGraphReceipt {
  const c = LOSASSO_SURFACE_GRAPH_CONTROL;
  return Object.freeze({
    epoch: words[c.epoch]!, leafCount: words[c.leafCount]!, nodeCount: words[c.nodeCount]!,
    published: words[c.published] === words[c.epoch] && words[c.epoch] !== 0,
    errors: words[c.errors]!, surfaceGeneration: words[c.surfaceGeneration]!,
    velocityGeneration: words[c.velocityGeneration]!, constraintCount: words[c.constraintCount]!,
    independentNodeCount: words[c.independentNodeCount]!,
    edgeHangingNodeCount: words[c.edgeHangingNodeCount]!,
    faceHangingNodeCount: words[c.faceHangingNodeCount]!,
    missingLookupCount: words[c.missingLookupCount]!,
    maximumDirectoryProbes: words[c.maximumDirectoryProbes]!,
    coverageErrors: words[c.coverageErrors]!,
    reciprocalAdjacencyErrors: words[c.reciprocalAdjacencyErrors]!,
    leafClosureErrors: words[c.leafClosureErrors]!,
    pressureRowCount: words[c.pressureRowCount]!,
    pressureRowMappingErrors: words[c.pressureRowMappingErrors]!,
    ownerPageCount: words[c.ownerPageCount]!, supportCellCount: words[c.supportCellCount]!,
    leafDispatch: Object.freeze([words[c.leafDispatch]!, words[c.leafDispatch + 1]!,
      words[c.leafDispatch + 2]!] as const),
    nodeDispatch: Object.freeze([words[c.nodeDispatch]!, words[c.nodeDispatch + 1]!,
      words[c.nodeDispatch + 2]!] as const),
    topologyBlockDispatch: Object.freeze([words[c.topologyBlockDispatch]!,
      words[c.topologyBlockDispatch + 1]!, words[c.topologyBlockDispatch + 2]!] as const),
  });
}

/**
 * Candidate-built deterministic adaptive node graph.
 *
 * Candidate and accepted GPUBuffer identities never rotate. `encodeCandidate`
 * publishes structure only; phi/velocity publishers fill the exposed candidate
 * fields and stamp control words 5/6. `encodeReadyCommit` then copies the whole
 * coherent tuple or retains the previous accepted tuple unchanged.
 */
export class WebGPUOctreeLosassoSurfaceGraph {
  readonly sources: LosassoSurfaceGraphSources;
  readonly candidateControl: GPUBuffer;
  readonly initializationTasks: readonly { readonly label: string; readonly run: () => Promise<void> }[];
  readonly allocatedBytes: number;
  readonly allocatedBytesBreakdown: Readonly<{
    radix: number; graphTopology: number; graphFields: number; total: number;
  }>;

  private readonly params: GPUBuffer;
  private readonly radixHeader: GPUBuffer;
  private readonly commitDispatch: GPUBuffer;
  private readonly radix: WebGPURadixSortU32;
  private readonly owned: readonly GPUBuffer[];
  private pipelines?: Readonly<Record<EntryPoint, GPUComputePipeline>>;
  private bindGroups?: Readonly<Record<EntryPoint, GPUBindGroup>>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    options: WebGPUOctreeLosassoSurfaceGraphOptions,
    private readonly input: WebGPUOctreeLosassoSurfaceGraphInput) {
    const rows = positiveU32(options.rowCapacity, "pressure row capacity");
    const ownerCapacity = positiveU32(options.ownerPages.capacity, "owner page capacity");
    const logicalBrickCount = positiveU32(options.ownerPages.logicalBrickCount,
      "logical brick count");
    const ownerDirectoryOffset = positiveU32(options.ownerPages.ownerDirectoryOffsetWords,
      "owner directory offset");
    const dimensions = options.dimensions.map((value, axis) =>
      positiveU32(value, `dimension ${axis}`)) as unknown as readonly [number, number, number];
    if (dimensions.some((value) => value > 1024)) {
      throw new RangeError("Losasso surface graph leaf ordering supports dimensions through 1024");
    }
    const expectedLogicalBricks = dimensions.reduce((count, value) =>
      count * Math.ceil(value / 8), 1);
    if (logicalBrickCount !== expectedLogicalBricks) {
      throw new RangeError("Losasso surface graph logical brick count does not match dimensions");
    }
    const cellVolume = dimensions[0] * dimensions[1] * dimensions[2];
    const nodeVolume = (dimensions[0] + 1) * (dimensions[1] + 1) * (dimensions[2] + 1);
    if (!Number.isSafeInteger(cellVolume) || cellVolume > 0xffff_ffff
      || !Number.isSafeInteger(nodeVolume) || nodeVolume > 0xffff_ffff) {
      throw new RangeError("Losasso surface graph linear identities must fit u32");
    }
    const maximumSupportLeaves = 512 * ownerCapacity;
    const leaves = positiveU32(options.leafCapacity
      ?? planLosassoSurfaceGraphLeafCapacity(rows, ownerCapacity), "leaf capacity");
    const nodes = positiveU32(options.nodeCapacity
      ?? planLosassoSurfaceGraphNodeCapacity(leaves, dimensions), "node capacity");
    if (leaves > maximumSupportLeaves) {
      throw new RangeError("Losasso surface graph leaf capacity exceeds owner support cells");
    }
    if (nodes > 8 * leaves) {
      throw new RangeError("Losasso surface graph node capacity exceeds eight corners per leaf");
    }
    const maximumLeafSize = positiveU32(options.maximumLeafSize, "maximum leaf size");
    if ((maximumLeafSize & (maximumLeafSize - 1)) !== 0 || maximumLeafSize > 1024) {
      throw new RangeError("Losasso surface graph maximum leaf size must be dyadic and at most 1024");
    }
    for (const [axis, size] of options.physicalCellSize.entries()) {
      if (!Number.isFinite(size) || size <= 0) {
        throw new RangeError(`Losasso surface graph physical cell size ${axis} must be positive`);
      }
    }
    if (input.candidateLeafHeaders.size < rows * 48) {
      throw new RangeError("Losasso surface graph LeafHeader source is smaller than row capacity");
    }
    // Transaction layout is 32 metadata words, a non-empty radix span, then
    // seven owner-capacity arrays. Candidate keys/pages are read from this
    // transaction rather than from either accepted arena record bank.
    if (input.candidateOwnerTransaction.size < (33 + 7 * ownerCapacity) * 4
      || input.candidateOwnerArena.size < (options.ownerPages.ownerPagesOffsetWords
        + 2 * ownerCapacity * 512) * 4) {
      throw new RangeError("Losasso surface graph owner candidate source is smaller than its ABI");
    }
    const topologyLayout = layout(input.topologyControlLayout);
    const readyLayout = layout(input.readyControlLayout ?? input.topologyControlLayout);
    const requiredTopologyBytes = 4 * (1 + Math.max(topologyLayout.epochWord,
      topologyLayout.countWord, topologyLayout.validWord, topologyLayout.errorWord));
    const requiredReadyBytes = 4 * (1 + Math.max(readyLayout.epochWord,
      readyLayout.countWord, readyLayout.validWord, readyLayout.errorWord));
    if (input.candidateTopologyControl.size < requiredTopologyBytes
      || (input.readyControl ?? input.candidateTopologyControl).size < requiredReadyBytes) {
      throw new RangeError("Losasso surface graph control source does not cover its declared layout");
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const make = (label: string, size: number, usage = storage): GPUBuffer => {
      if (!Number.isSafeInteger(size) || size < 1 || size > maximumBinding) {
        throw new RangeError(`Losasso surface graph ${label} exceeds a storage binding`);
      }
      return device.createBuffer({ label: `Losasso surface graph ${label}`,
        size: Math.max(16, size), usage });
    };
    this.params = make("parameters", 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.radixHeader = make("radix producer header", 16);
    this.commitDispatch = make("ready commit dispatch", 64, storage | GPUBufferUsage.INDIRECT);
    const locatorBytes = planLosassoSurfaceGraphLeafLocatorBytes(ownerCapacity);
    const createBank = (name: string): LosassoSurfaceGraphBankSource => {
      const control = make(`${name} control`, LOSASSO_SURFACE_GRAPH_CONTROL_WORDS * 4,
        storage | GPUBufferUsage.INDIRECT);
      return Object.freeze({
        control,
        leaves: make(`${name} leaves`, leaves * LOSASSO_SURFACE_GRAPH_LEAF_WORDS * 4),
        leafDirectory: make(`${name} leaf directory`, leaves * LOSASSO_SURFACE_GRAPH_LEAF_DIRECTORY_WORDS * 4),
        nodes: make(`${name} nodes`, nodes * LOSASSO_SURFACE_GRAPH_NODE_WORDS * 4),
        nodeDirectory: make(`${name} node directory`, nodes * LOSASSO_SURFACE_GRAPH_NODE_DIRECTORY_WORDS * 4),
        constraints: make(`${name} constraints`, nodes * LOSASSO_SURFACE_GRAPH_CONSTRAINT_WORDS * 4),
        adjacency: make(`${name} directional adjacency`, nodes * LOSASSO_SURFACE_GRAPH_ADJACENCY_WORDS * 4),
        incidentLeaves: make(`${name} compiled incident leaves`, nodes * LOSASSO_SURFACE_GRAPH_INCIDENT_LEAF_WORDS * 4),
        leafLocator: make(`${name} owner-page leaf locator`, locatorBytes),
        surfaceMass: make(`${name} surface mass`, leaves * 4),
        surfaceCompression: make(`${name} surface compression`, leaves * 4),
        phi: make(`${name} phi banks`, nodes * 8),
        nodalVelocity: make(`${name} nodal velocity banks`, nodes * 32),
        nodeValidity: make(`${name} node validity`, nodes * 4),
        pressureRowToGraphLeaf: make(`${name} pressure row to graph leaf`, rows * 4),
        pressureRowCapacity: rows, leafCapacity: leaves, nodeCapacity: nodes,
        leafDispatchOffsetBytes: LOSASSO_SURFACE_GRAPH_CONTROL.leafDispatch * 4,
        nodeDispatchOffsetBytes: LOSASSO_SURFACE_GRAPH_CONTROL.nodeDispatch * 4,
        topologyBlockDispatchOffsetBytes:
          LOSASSO_SURFACE_GRAPH_CONTROL.topologyBlockDispatch * 4,
      });
    };
    const accepted = createBank("accepted");
    const candidate = createBank("candidate");
    this.sources = Object.freeze({ accepted, candidate });
    this.candidateControl = candidate.control;
    this.radix = new WebGPURadixSortU32(device,
      Math.max(512 * ownerCapacity, 8 * leaves), this.radixHeader);
    const words = new Uint32Array(32);
    words.set([...dimensions, maximumLeafSize], 0);
    words.set([leaves, nodes, rows, device.limits.maxComputeWorkgroupsPerDimension], 4);
    new Float32Array(words.buffer).set([...options.physicalCellSize, 0], 8);
    words.set([topologyLayout.epochWord, topologyLayout.countWord,
      topologyLayout.validWord, topologyLayout.errorWord], 12);
    words.set([readyLayout.epochWord, readyLayout.countWord,
      readyLayout.validWord, readyLayout.errorWord], 16);
    words.set([topologyLayout.validValue, readyLayout.validValue, 0, 0], 20);
    words.set([ownerCapacity, logicalBrickCount, ownerDirectoryOffset,
      options.ownerPages.ownerPagesOffsetWords], 24);
    words.set([this.radix.capacity, LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_MAGIC,
      LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_HEADER_WORDS, 0], 28);
    device.queue.writeBuffer(this.params, 0, words);
    const initialControl = new Uint32Array(LOSASSO_SURFACE_GRAPH_CONTROL_WORDS);
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.leafCapacity] = leaves;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.nodeCapacity] = nodes;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.constraintCapacity] = nodes;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.leafDispatch + 1] = 1;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.leafDispatch + 2] = 1;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.nodeDispatch + 1] = 1;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.nodeDispatch + 2] = 1;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.topologyBlockDispatch + 1] = 1;
    initialControl[LOSASSO_SURFACE_GRAPH_CONTROL.topologyBlockDispatch + 2] = 1;
    device.queue.writeBuffer(accepted.control, 0, initialControl);
    device.queue.writeBuffer(candidate.control, 0, initialControl);
    const bankBuffers = (bank: LosassoSurfaceGraphBankSource) => [bank.control,
      bank.leaves, bank.leafDirectory, bank.nodes, bank.nodeDirectory, bank.constraints,
      bank.adjacency, bank.incidentLeaves, bank.leafLocator,
      bank.surfaceMass, bank.surfaceCompression,
      bank.phi, bank.nodalVelocity, bank.nodeValidity,
      bank.pressureRowToGraphLeaf];
    this.owned = Object.freeze([this.params, this.radixHeader, this.commitDispatch,
      ...bankBuffers(accepted), ...bankBuffers(candidate)]);
    this.allocatedBytes = this.radix.allocatedBytes
      + this.owned.reduce((sum, buffer) => sum + buffer.size, 0);
    const graphFields = [accepted, candidate].reduce((sum, bank) => sum
      + bank.surfaceMass.size + bank.surfaceCompression.size
      + bank.phi.size + bank.nodalVelocity.size + bank.nodeValidity.size, 0);
    this.allocatedBytesBreakdown = Object.freeze({
      radix: this.radix.allocatedBytes,
      graphFields,
      graphTopology: this.allocatedBytes - this.radix.allocatedBytes - graphFields,
      total: this.allocatedBytes,
    });
    this.initializationTasks = Object.freeze([
      { label: "Compile Losasso surface graph radix sort", run: () => this.radix.initializePipelines() },
      { label: "Compile Losasso adaptive surface graph", run: () => this.initialize() },
    ]);
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso adaptive surface graph",
      code: octreeLosassoSurfaceGraphWGSL });
    const entries = await Promise.all(ENTRY_POINTS.map(async (entryPoint) => [entryPoint,
      await this.device.createComputePipelineAsync({ label: `Losasso surface graph - ${entryPoint}`,
        layout: "auto", compute: { module: shaderModule, entryPoint } })] as const));
    this.pipelines = Object.freeze(Object.fromEntries(entries) as Record<EntryPoint, GPUComputePipeline>);
    const a = this.sources.accepted, c = this.sources.candidate;
    const resources: Readonly<Record<number, GPUBuffer>> = Object.freeze({
      0: this.params, 1: this.input.candidateTopologyControl,
      2: this.input.readyControl ?? this.input.candidateTopologyControl,
      3: this.input.candidateLeafHeaders, 4: this.radixHeader, 5: this.radix.keys,
      6: this.radix.runs, 7: this.radix.control, 8: c.control, 9: c.leaves,
      10: c.leafDirectory, 11: c.nodes, 12: c.nodeDirectory, 13: c.constraints,
      14: c.adjacency, 15: c.phi, 16: c.nodalVelocity, 17: c.nodeValidity,
      18: a.control, 19: a.leaves, 20: a.leafDirectory, 21: a.nodes,
      22: a.nodeDirectory, 23: a.constraints, 24: a.adjacency, 25: a.phi,
      26: a.nodalVelocity, 27: a.nodeValidity, 28: this.commitDispatch,
      29: this.input.candidateOwnerArena, 30: this.input.candidateOwnerTransaction,
      31: c.pressureRowToGraphLeaf, 32: a.pressureRowToGraphLeaf,
      33: c.incidentLeaves, 34: a.incidentLeaves,
      35: c.leafLocator, 36: a.leafLocator,
      37: c.surfaceMass, 38: c.surfaceCompression,
      39: a.surfaceMass, 40: a.surfaceCompression,
    });
    this.bindGroups = Object.freeze(Object.fromEntries(ENTRY_POINTS.map((entryPoint) => {
      const pipeline = this.pipelines![entryPoint];
      return [entryPoint, this.device.createBindGroup({
        label: `Losasso surface graph bindings - ${entryPoint}`,
        layout: pipeline.getBindGroupLayout(0), entries: BINDINGS[entryPoint].map((binding) =>
          ({ binding, resource: { buffer: resources[binding]! } })),
      })];
    })) as Record<EntryPoint, GPUBindGroup>);
  }

  encodeCandidate(broker: PassBroker): void {
    this.assertReady();
    this.direct(broker, "prepareSurfaceGraph", 1, "Losasso graph - prepare candidate");
    broker.fence("Losasso surface graph owner-cell dispatch prepared");
    this.indirect(broker, "emitSurfaceGraphItems", this.commitDispatch,
      0, "Losasso graph - emit inactive owner leaf identities");
    this.radix.encode(broker);
    this.direct(broker, "prepareSurfaceGraphLeaves", 1,
      "Losasso graph - publish deterministic support leaves");
    broker.fence("Losasso surface graph leaf dispatch prepared");
    this.indirect(broker, "buildSurfaceGraphLeavesAndNodeItems", this.commitDispatch,
      0, "Losasso graph - build leaves and emit eight corners");
    this.radix.encode(broker);
    this.direct(broker, "prepareSurfaceGraphRecords", 1, "Losasso graph - publish compact counts");
    broker.fence("Losasso surface graph compact dispatch prepared");
    this.indirect(broker, "buildSurfaceGraphNodes", this.commitDispatch,
      12, "Losasso graph - build deterministic nodes");
    this.indirect(broker, "resolveSurfaceGraphLeaves", this.commitDispatch,
      0, "Losasso graph - resolve leaf corners");
    this.indirect(broker, "compileSurfaceGraphLeafLocator", this.commitDispatch,
      48, "Losasso graph - compile owner-page leaf locator");
    this.indirect(broker, "resolveSurfaceGraphPressureRows", this.commitDispatch,
      32, "Losasso graph - map pressure rows to support leaves");
    this.indirect(broker, "classifySurfaceGraphNodes", this.commitDispatch,
      12, "Losasso graph - constraints and atomic adjacency");
    this.indirect(broker, "validateSurfaceGraphNodes", this.commitDispatch,
      12, "Losasso graph - validate structural receipts");
    this.direct(broker, "finishSurfaceGraph", 1, "Losasso graph - publish candidate receipt");
  }

  /** Requires candidate control words 5 and 6 to be stamped by field publishers. */
  encodeReadyCommit(broker: PassBroker): void {
    this.assertReady();
    this.direct(broker, "prepareSurfaceGraphCommit", 1, "Losasso graph - validate ready commit");
    broker.fence("Losasso surface graph ready dispatch prepared");
    this.indirect(broker, "commitSurfaceGraphLeaves", this.commitDispatch, 0,
      "Losasso graph - commit leaves");
    this.indirect(broker, "commitSurfaceGraphSurfaceState", this.commitDispatch, 0,
      "Losasso graph - commit conserved surface state");
    this.indirect(broker, "commitSurfaceGraphNodes", this.commitDispatch, 12,
      "Losasso graph - commit nodes");
    this.indirect(broker, "commitSurfaceGraphRelations", this.commitDispatch, 12,
      "Losasso graph - commit constraints and adjacency");
    this.indirect(broker, "commitSurfaceGraphFields", this.commitDispatch, 12,
      "Losasso graph - commit fields");
    this.indirect(broker, "commitSurfaceGraphPressureRows", this.commitDispatch, 32,
      "Losasso graph - commit pressure row mapping");
    this.indirect(broker, "commitSurfaceGraphLeafLocator", this.commitDispatch, 48,
      "Losasso graph - commit owner-page leaf locator");
    broker.fence("Losasso surface graph ready dispatch retired");
    this.direct(broker, "finishSurfaceGraphCommit", 1, "Losasso graph - publish accepted tuple");
  }

  async readReceipt(bank: "accepted" | "candidate" = "candidate"):
  Promise<LosassoSurfaceGraphReceipt> {
    this.assertLive();
    const readback = this.device.createBuffer({ label: `Losasso ${bank} graph receipt`,
      size: LOSASSO_SURFACE_GRAPH_CONTROL_WORDS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: `Read Losasso ${bank} graph receipt` });
    encoder.copyBufferToBuffer(this.sources[bank].control, 0, readback, 0, readback.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return unpackReceipt(Uint32Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.radix.destroy();
    for (const buffer of this.owned) buffer.destroy();
    this.pipelines = undefined; this.bindGroups = undefined;
  }

  private direct(broker: PassBroker, entryPoint: EntryPoint, count: number, label: string): void {
    const pass = broker.compute({ label }); pass.setPipeline(this.pipelines![entryPoint]);
    pass.setBindGroup(0, this.bindGroups![entryPoint]); pass.dispatchWorkgroups(count);
  }

  private indirect(broker: PassBroker, entryPoint: EntryPoint, dispatch: GPUBuffer,
    offset: number, label: string): void {
    const pass = broker.compute({ label }); pass.setPipeline(this.pipelines![entryPoint]);
    pass.setBindGroup(0, this.bindGroups![entryPoint]); pass.dispatchWorkgroupsIndirect(dispatch, offset);
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso surface graph is destroyed");
  }

  private assertReady(): void {
    this.assertLive();
    if (!this.pipelines || !this.bindGroups) throw new Error("Losasso surface graph is not initialized");
  }
}
