#!/usr/bin/env node
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_EDGE,
  SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS,
  createSparseCM12HotTopology,
  type SparseCM12HotTopology,
} from "../lib/methods/adaptive-mass/sparse-cm12-hot-topology";
import { createSparseCM12HotTopologyWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-hot-topology.wgsl";
import { createSparseCM12PhaseArenaPlan } from
  "../lib/methods/adaptive-mass/sparse-cm12-phase-arenas";
import {
  SPARSE_CM12_PRESSURE_CACHE_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_MAGIC,
  bakeSparseCM12PressureAggregateAuthorityQA,
  bakeSparseCM12PressureAuthorityQA,
  compareSparseCM12PressureAggregateAuthorityQA,
  compareSparseCM12PressureAuthorityQA,
  createSparseCM12PersistentPressureCacheLayout,
  initializeSparseCM12PersistentPressureCacheWords,
  sparseCM12PressureAggregateDirtyClosure,
  sparseCM12PressureCacheDirtyCellClosure,
  type SparseCM12PressureAuthorityQAInput,
  type SparseCM12PressureAuthorityQAOutput,
  type SparseCM12PressureAggregateAuthorityQAOutput,
  type SparseCM12PressureAggregateTopology,
} from "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache";
import { createSparseCM12PersistentPressureCacheWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache.wgsl";

const BITS_BUFFER = new ArrayBuffer(4);
const BITS_FLOAT = new Float32Array(BITS_BUFFER);
const BITS_WORD = new Uint32Array(BITS_BUFFER);
const f32Bits = (value: number): number => {
  BITS_FLOAT[0] = Math.fround(value);
  return BITS_WORD[0]!;
};

function fixture(): { readonly topology: SparseCM12HotTopology;
  readonly grid: ReturnType<typeof buildSparseAtlasCompositeGrid> } {
  const dimensions = [32, 16, 16] as const;
  const logical = [2, 1, 1] as const;
  const brick = (coordinate: readonly [number, number, number],
    resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey(coordinate, logical), coordinate, resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas(dimensions,
    [brick([0, 0, 0], 8), brick([1, 0, 0], 16)], 31, undefined, 16);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  return { topology: createSparseCM12HotTopology(grid), grid };
}

const setBit = (bits: Uint32Array, id: number, enabled: boolean) => {
  if (enabled) bits[id >>> 5]! |= 1 << (id & 31);
  else bits[id >>> 5]! &= ~(1 << (id & 31));
};
const hasBit = (bits: Uint32Array, id: number) =>
  (bits[id >>> 5]! & (1 << (id & 31))) !== 0;

function authorityInput(
  topology: SparseCM12HotTopology,
  grid: ReturnType<typeof buildSparseAtlasCompositeGrid>,
): SparseCM12PressureAuthorityQAInput {
  const { cellCount, rowCount } = topology.layout;
  const activeCellBits = new Uint32Array(Math.ceil(cellCount / 32));
  // A broad deterministic liquid body crosses the mixed seam and leaves air.
  for (let cell = 0; cell < cellCount; cell += 1) {
    setBit(activeCellBits, cell, cell % 7 !== 0 && cell % 11 !== 0);
  }
  const activeRowBits = new Uint32Array(Math.ceil(rowCount / 32));
  const thetaBits = new Uint32Array(rowCount);
  const solidScaleBits = new Uint32Array(rowCount);
  const fluxWeightBits = new Uint32Array(rowCount);
  const faceVelocityBits = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row += 1) {
    const source = grid.gradientRows[row]!;
    const active = source.terms.some((term) => hasBit(activeCellBits, term.cellId));
    setBit(activeRowBits, row, active);
    thetaBits[row] = f32Bits(active ? 0.5 + (row % 5) * 0.125 : 0);
    solidScaleBits[row] = f32Bits(1);
    fluxWeightBits[row] = f32Bits(source.dualWeight);
    faceVelocityBits[row] = f32Bits(((row % 13) - 6) / 17);
  }
  const rhsCorrectionBits = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell += 1) {
    rhsCorrectionBits[cell] = f32Bits(((cell % 9) - 4) / 1024);
  }
  return { activeCellBits, activeRowBits, thetaBits, solidScaleBits,
    fluxWeightBits, faceVelocityBits, rhsCorrectionBits };
}

const cloneInput = (input: SparseCM12PressureAuthorityQAInput):
SparseCM12PressureAuthorityQAInput => ({
  activeCellBits: input.activeCellBits.slice(), activeRowBits: input.activeRowBits.slice(),
  thetaBits: input.thetaBits.slice(), solidScaleBits: input.solidScaleBits.slice(),
  fluxWeightBits: input.fluxWeightBits.slice(), faceVelocityBits: input.faceVelocityBits.slice(),
  rhsCorrectionBits: input.rhsCorrectionBits.slice(),
});
const cloneOutput = (output: SparseCM12PressureAuthorityQAOutput):
SparseCM12PressureAuthorityQAOutput => ({ edgeBits: output.edgeBits.slice(),
  diagonalBits: output.diagonalBits.slice(), rhsBits: output.rhsBits.slice() });

const cloneAggregateOutput = (output: SparseCM12PressureAggregateAuthorityQAOutput):
SparseCM12PressureAggregateAuthorityQAOutput => ({
  aggregateEdgeBits: output.aggregateEdgeBits.slice(),
  brickDiagonalBits: output.brickDiagonalBits.slice(),
  hierarchyEdgeBits: output.hierarchyEdgeBits.map((level) => level.slice()),
  hierarchyDiagonalBits: output.hierarchyDiagonalBits.map((level) => level.slice()),
});

function aggregateTopology(topology: SparseCM12HotTopology):
SparseCM12PressureAggregateTopology {
  const brickOf = (cell: number) => topology.words[topology.layout.cellBaseWords
    + SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS * cell
    + SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution]! >>> 5;
  const contributions = new Map<number, number[]>();
  for (let cell = 0; cell < topology.layout.cellCount; cell += 1) {
    const source = brickOf(cell);
    const begin = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell]!;
    const end = topology.words[topology.layout.directedEdgeOffsetBaseWords + cell + 1]!;
    for (let edge = begin; edge < end; edge += 1) {
      const at = topology.layout.directedEdgeBaseWords
        + SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS * edge;
      const target = brickOf(topology.words[at + SPARSE_CM12_HOT_TOPOLOGY_EDGE.neighbor]!);
      if (source === target) continue;
      const key = source * 2 + target;
      const list = contributions.get(key);
      if (list) list.push(edge); else contributions.set(key, [edge]);
    }
  }
  const aggregateEdges = [...contributions.entries()].sort(([a], [b]) => a - b)
    .map(([key, contributionFineEdges]) => ({ sourceBrick: Math.floor(key / 2),
      targetBrick: key % 2, contributionFineEdges }));
  return { brickCount: 2, aggregateEdges, hierarchy: [
    { parentsByBrick: [0, 1], childrenByGroup: [[0], [1]],
      internalAggregateEdgesByGroup: [[], []],
      edges: aggregateEdges.map((edge, id) => ({ sourceGroup: edge.sourceBrick,
        targetGroup: edge.targetBrick, contributionAggregateEdges: [id] })) },
    { parentsByBrick: [0, 0], childrenByGroup: [[0, 1]],
      internalAggregateEdgesByGroup: [aggregateEdges.map((_, id) => id)], edges: [] },
  ] };
}

function main(): void {
  const { topology, grid } = fixture();
  const membershipWords = 3 * topology.layout.cellCount + 8192;
  const phaseArena = createSparseCM12PhaseArenaPlan({
    cellCount: topology.layout.cellCount, rowCount: topology.layout.rowCount,
    brickCount: 2, candidateBrickCount: 2,
    pressureEdgeCount: topology.layout.directedEdgeCount,
    pressureCoarseEdgeCount: 17,
    pressureHierarchy: [{ groupCount: 2, edgeCount: 1 },
      { groupCount: 1, edgeCount: 0 }],
    pressureMembershipWords: membershipWords,
    pressureStaticTopologyWords: 256,
    topologyCandidateControlWords: 8192,
    presentation: { metadataBytes: 32, worklistBytes: 40, payloadBytes: 32768 },
    htp1: topology.layout,
  });
  // Reserve an arbitrary prefix to prove PCM1 and PCF1 can share the opaque
  // membership region without aliasing either control layout.
  const layout = createSparseCM12PersistentPressureCacheLayout({
    phaseArena, htp1: topology.layout, controlOffsetWords: 4096,
    rigidScaleEnabled: true,
  });
  const pressureBuffer = phaseArena.buffers.find((entry) =>
    entry.id === "cm12.pressure-cache")!;
  const words = new Uint32Array(pressureBuffer.sizeBytes / 4);
  initializeSparseCM12PersistentPressureCacheWords(words, layout);
  if (words[layout.headerBaseWords + SPARSE_CM12_PRESSURE_CACHE_HEADER.magic]
      !== SPARSE_CM12_PRESSURE_CACHE_MAGIC
    || layout.headerBaseWords < layout.membershipRegionBaseWords + 4096
    || layout.controlEndWords > layout.membershipRegionBaseWords + layout.membershipRegionWords
    || layout.effectiveEdgeBaseWords + layout.directedEdgeCount > layout.headerBaseWords) {
    throw new Error("PCF1 non-aliasing/layout initialization contract failed");
  }

  const initialInput = authorityInput(topology, grid);
  const initial = bakeSparseCM12PressureAuthorityQA(topology, initialInput);
  const aggregate = aggregateTopology(topology);
  const initialAggregate = bakeSparseCM12PressureAggregateAuthorityQA(topology,
    initialInput, initial, aggregate);
  const next = cloneInput(initialInput);
  const membershipCells = [17, 103, 701].filter((cell) => cell < layout.cellCount);
  for (const cell of membershipCells) {
    setBit(next.activeCellBits, cell, !hasBit(next.activeCellBits, cell));
  }
  const coefficientRows = grid.gradientRows
    .map((row, id) => ({ row, id }))
    .filter(({ row }) => row.terms.length >= 2
      && row.terms.every((term) => hasBit(next.activeCellBits, term.cellId)))
    .slice(0, 3).map(({ id }) => id);
  if (coefficientRows.length < 3) throw new Error("PCF1 fixture lacks coefficient rows");
  const [thetaRow, membershipRow, solidRow] = coefficientRows;
  next.thetaBits[thetaRow!] = f32Bits(0.3125);
  setBit(next.activeRowBits, membershipRow!, false);
  next.thetaBits[membershipRow!] = 0;
  next.solidScaleBits[solidRow!] = f32Bits(0.625);
  // RHS is a frame value: change every face/correction. The local coefficient
  // repair still prepares RHS through the unchanged stable PCM cell order.
  for (let row = 0; row < next.faceVelocityBits.length; row += 1) {
    next.faceVelocityBits[row] = f32Bits(((row % 17) - 8) / 19);
  }
  for (let cell = 0; cell < next.rhsCorrectionBits.length; cell += 1) {
    next.rhsCorrectionBits[cell] = f32Bits(((cell % 7) - 3) / 2048);
  }
  const dirty = sparseCM12PressureCacheDirtyCellClosure(topology, {
    membershipCells, rows: [thetaRow!, membershipRow!], solidRows: [solidRow!],
  });
  const local = bakeSparseCM12PressureAuthorityQA(topology, next,
    cloneOutput(initial), dirty);
  const full = bakeSparseCM12PressureAuthorityQA(topology, next);
  const receipt = compareSparseCM12PressureAuthorityQA(topology, next, local, full);
  if (!receipt.exact) throw new Error(`PCF1 local/full mismatch ${JSON.stringify(receipt)}`);
  const aggregateDirty = sparseCM12PressureAggregateDirtyClosure(topology, aggregate, dirty);
  const localAggregate = bakeSparseCM12PressureAggregateAuthorityQA(topology, next, local,
    aggregate, cloneAggregateOutput(initialAggregate), aggregateDirty);
  const fullAggregate = bakeSparseCM12PressureAggregateAuthorityQA(topology, next, full,
    aggregate);
  const aggregateReceipt = compareSparseCM12PressureAggregateAuthorityQA(localAggregate,
    fullAggregate);
  if (!aggregateReceipt.exact) {
    throw new Error(`PCF1 aggregate local/full mismatch ${JSON.stringify(aggregateReceipt)}`);
  }
  if (aggregateDirty.aggregateEdges.size > 0) {
    const missingAggregateEdges = new Set(aggregateDirty.aggregateEdges);
    missingAggregateEdges.delete(missingAggregateEdges.values().next().value!);
    const incompleteAggregate = bakeSparseCM12PressureAggregateAuthorityQA(topology, next,
      local, aggregate, cloneAggregateOutput(initialAggregate), {
        ...aggregateDirty, aggregateEdges: missingAggregateEdges,
      });
    const rejectedAggregate = compareSparseCM12PressureAggregateAuthorityQA(
      incompleteAggregate, fullAggregate);
    if (rejectedAggregate.exact || rejectedAggregate.firstAggregateEdgeMismatch < 0) {
      throw new Error("PCF1 aggregate oracle did not reject a missing aggregate-edge owner");
    }
  }

  // Prove the QA oracle detects a missing event rather than silently accepting
  // incomplete provenance. A solid-scale change requires all row owners.
  const incomplete = bakeSparseCM12PressureAuthorityQA(topology, next,
    cloneOutput(initial), sparseCM12PressureCacheDirtyCellClosure(topology, {
      membershipCells, rows: [thetaRow!, membershipRow!],
    }));
  const rejected = compareSparseCM12PressureAuthorityQA(topology, next, incomplete, full);
  if (rejected.exact || rejected.firstEdgeMismatch < 0) {
    throw new Error("PCF1 QA oracle did not reject a missing solid coefficient event");
  }

  const wgsl = createSparseCM12PersistentPressureCacheWGSL({ layout,
    arenaName: "pressureCache", solidRowScaleFunction: "fixtureSolidScale" });
  for (const token of ["pcfRecordCellMembershipEvent", "pcfRecordTopologyCellEvent",
    "pcfStoreThetaAndRecord", "pcfRecordSolidRowEvent", "repairPersistentPressureCache",
    "finalizePersistentPressureFineCache", "repairPersistentPressureAggregateEdges",
    "repairPersistentPressureBrickDiagonals", "repairPersistentPressureHierarchyEdges",
    "repairPersistentPressureHierarchyDiagonals", "finalizePersistentPressureCache"]) {
    if (!wgsl.includes(token)) throw new Error(`PCF1 WGSL omitted ${token}`);
  }
  const hot = createSparseCM12HotTopologyWGSL({ layout: topology.layout,
    arenaName: "hotTopology", brickActiveFunction: "fixtureBrickActive",
    acceptedBrickResolutionFunction: "fixtureAcceptedResolution",
    templateBrickCellRangeFunction: "fixtureCellRange",
    cellResolutionFunction: "fixtureCellResolution",
    cellOpenVolumeFunction: "fixtureCellOpenVolume" });
  const sourceFor = (aggregateSource: string) => /* wgsl */ `
@group(0) @binding(0) var<storage,read> hotTopology:array<u32>;
@group(0) @binding(1) var<storage,read_write> pressureCache:array<atomic<u32>>;
fn fixtureBrickActive(brick:u32)->bool{return brick<2u;}
fn fixtureAcceptedResolution(brick:u32)->u32{return select(8u,16u,brick==1u);}
fn fixtureCellRange(brick:u32,resolution:u32)->vec2u{
  _=resolution;return select(vec2u(0u,512u),vec2u(512u,4096u),brick==1u);}
fn fixtureCellResolution(cell:u32)->u32{return select(8u,16u,cell>=512u);}
fn fixtureCellOpenVolume(cell:u32)->f32{_=cell;return 1.0;}
fn pcmCellContains(cell:u32)->bool{return (cell&7u)!=0u;}
fn pcmRowContains(row:u32)->bool{return (row&15u)!=0u;}
fn fixtureSolidScale(row:u32)->f32{_=row;return 1.0;}
fn pcfTopologyGeneration()->u32{return 7u;}
fn pcfPCMGeneration()->u32{return 8u;}
fn pcfAggregateTopologyGeneration()->u32{return 9u;}
fn pcfCellBrick(cell:u32)->u32{return select(0u,1u,cell>=512u);}
fn pcfBrickCellRange(brick:u32)->vec2u{return select(vec2u(0u,512u),
  vec2u(512u,4096u),brick==1u);}
fn pcfAggregateEdgeForFineEdge(edge:u32)->u32{return edge%17u;}
fn pcfAggregateEdgeSourceBrick(edge:u32)->u32{return edge&1u;}
fn pcfAggregateEdgeContributionRange(edge:u32)->vec2u{return vec2u(edge,1u);}
fn pcfAggregateEdgeContribution(at:u32)->u32{return at%${topology.layout.directedEdgeCount}u;}
fn pcfHierarchyParent(level:u32,brick:u32)->u32{return select(brick,0u,level==1u);}
fn pcfHierarchyEdgeForAggregate(level:u32,edge:u32)->u32{
  return select(0xffffffffu,0u,level==0u&&edge==0u);}
fn pcfHierarchyChildRange(level:u32,group:u32)->vec2u{
  if(level==0u){return vec2u(group,1u);}return vec2u(0u,2u);}
fn pcfHierarchyChild(level:u32,at:u32)->u32{_=level;return at;}
fn pcfHierarchyInternalEdgeRange(level:u32,group:u32)->vec2u{
  _=level;_=group;return vec2u(0u);}
fn pcfHierarchyInternalEdge(level:u32,at:u32)->u32{_=level;return at;}
fn pcfHierarchyEdgeContributionRange(level:u32,edge:u32)->vec2u{
  _=level;_=edge;return vec2u(0u,1u);}
fn pcfHierarchyEdgeContribution(level:u32,at:u32)->u32{_=level;return at;}
${hot}
${aggregateSource}
`;
  const directory = mkdtempSync(join(tmpdir(), "fluid-pcf1-wgsl-"));
  try {
    const qaLayout = createSparseCM12PersistentPressureCacheLayout({
      phaseArena, htp1: topology.layout, controlOffsetWords: 4096,
      rigidScaleEnabled: true, qaFullOracle: true,
    });
    const variants = [wgsl, createSparseCM12PersistentPressureCacheWGSL({
      layout: qaLayout, arenaName: "pressureCache",
      solidRowScaleFunction: "fixtureSolidScale",
    })];
    for (let variant = 0; variant < variants.length; variant += 1) {
      const path = join(directory, variant === 0
        ? "pcf1-b16-p16.wgsl" : "pcf1-b16-p16-qa.wgsl");
      writeFileSync(path, sourceFor(variants[variant]!));
      const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
      if (result.status !== 0) {
        throw new Error(`PCF1 Naga validation failed:\n${result.stderr || result.stdout}`);
      }
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
  console.log(`Sparse CM12 PCF1: B16/P16 ${layout.controlWords} control words, `
    + `${layout.dirtyLeafCount} dirty leaves; ${dirty.size}/${layout.cellCount} cells repaired; `
    + "active fine + aggregate/hierarchy coefficient/diagonal/RHS bytes match full bake; "
    + "incomplete provenance rejected");
}

main();
