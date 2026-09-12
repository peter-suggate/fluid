import {
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER as PCM_D,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE,
  createSparseCM12CanonicalMembershipInitialWords,
  createSparseCM12CanonicalMembershipLayout,
  type SparseCM12CanonicalMembershipLayout,
} from "../sparse-cm12-canonical-membership";
import {
  SPARSE_CM12_PRESSURE_CACHE_HEADER as PCF_H,
  SPARSE_CM12_PRESSURE_CACHE_PHASE,
  createSparseCM12ResidentPersistentPressureCacheLayout,
  initializeSparseCM12PersistentPressureCacheWords,
  type SparseCM12PersistentPressureCacheLayout,
} from "../sparse-cm12-persistent-pressure-cache";
import {
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER as PEI_H,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE,
  createSparseCM12PressureExecutionImageInitialWords,
  createSparseCM12PressureExecutionImageLayout,
  type SparseCM12PressureExecutionImageLayout,
} from "../sparse-cm12-pressure-execution-image";
import type {
  SliceNumericalFields,
  SliceNumericalTopology,
  SlicePressureRows,
} from "./slice-stage-numerics";

const INVALID = 0xffff_ffff;
const floatBits = (value: number): number => new Uint32Array(new Float32Array([value]).buffer)[0]!;

export interface SlicePressureAuthorityReceipt {
  readonly topologyGeneration: number;
  readonly pcmCellGeneration: number;
  readonly pcmRowGeneration: number;
  readonly coefficientGeneration: number;
  readonly executionGeneration: number;
  readonly dirtyCellLeaves: Uint32Array;
  readonly dirtyRowTiles: Uint32Array;
  readonly changedDiagonalCount: number;
  readonly pressureCellCount: number;
  readonly pressureRowCount: number;
  readonly fault: number;
  readonly firstFaultId: number;
}

/**
 * CPU-resident image of PCM1/PCF1/PEI1. The word arrays use the production
 * layouts and phases; dense executionOrder is the local-array translation of
 * PEI's stable cell stream.
 */
export interface SlicePressureAuthority {
  readonly cellCapacity: number;
  readonly rowCapacity: number;
  readonly pcmLayout: SparseCM12CanonicalMembershipLayout;
  readonly pcmWords: Uint32Array;
  readonly pcfLayout: SparseCM12PersistentPressureCacheLayout;
  readonly pcfWords: Uint32Array;
  readonly peiLayout: SparseCM12PressureExecutionImageLayout;
  readonly peiWords: Uint32Array;
  readonly acceptedCellBits: Uint32Array;
  readonly acceptedRowBits: Uint32Array;
  readonly densityBits: Uint32Array;
  readonly capacityBits: Uint32Array;
  readonly normalXBits: Uint32Array;
  readonly normalYBits: Uint32Array;
  readonly diagonalBits: Uint32Array;
  readonly rowTheta: Float32Array;
  readonly executionOrder: Uint32Array;
  readonly receipt: SlicePressureAuthorityReceipt;
}

export interface SlicePressureAuthorityCapacity {
  readonly cells: number;
  readonly rows: number;
  readonly bricks: number;
}

export interface SlicePressureAuthorityPublishOptions {
  /** Production globally invalidates PCM rows when SolidWorld is present. */
  readonly globalRowInvalidation?: boolean;
}

function stableCellId(topology: SliceNumericalTopology, dense: number): number {
  return topology.cells[dense]!.stableId ?? dense;
}

function allocate(capacity: SlicePressureAuthorityCapacity): SlicePressureAuthority {
  const pcmLayout = createSparseCM12CanonicalMembershipLayout({
    cellCapacity: capacity.cells, rowCapacity: capacity.rows,
  });
  const pcmWords = createSparseCM12CanonicalMembershipInitialWords(pcmLayout);
  const pcfLayout = createSparseCM12ResidentPersistentPressureCacheLayout({ baseWords: 0,
    cellCount: capacity.cells, rowCount: capacity.rows,
    // The 2-D compact graph publishes incidence coefficients directly. Keep
    // one address per possible directed incidence, as PCF1 does in 3-D.
    directedEdgeCount: Math.max(1, 8 * capacity.cells),
    brickCount: capacity.bricks, aggregateEdgeCount: 0,
    hierarchyLevelCounts: [0], hierarchyEdgeLevelCounts: [0],
  });
  const pcfWords = new Uint32Array(pcfLayout.bufferSizeWords);
  initializeSparseCM12PersistentPressureCacheWords(pcfWords, pcfLayout);
  const peiLayout = createSparseCM12PressureExecutionImageLayout({ baseWords: 0,
    cellCapacity: capacity.cells, brickCapacity: capacity.bricks,
    hierarchyCapacity: 1, brickFineResolution: 8, presentationPageResolution: 8,
  });
  const peiWords = createSparseCM12PressureExecutionImageInitialWords(peiLayout);
  return { cellCapacity: capacity.cells, rowCapacity: capacity.rows,
    pcmLayout, pcmWords, pcfLayout, pcfWords, peiLayout, peiWords,
    acceptedCellBits: new Uint32Array(Math.ceil(capacity.cells / 32)),
    acceptedRowBits: new Uint32Array(Math.ceil(capacity.rows / 32)),
    densityBits: new Uint32Array(capacity.cells), capacityBits: new Uint32Array(capacity.cells),
    normalXBits: new Uint32Array(capacity.cells), normalYBits: new Uint32Array(capacity.cells),
    diagonalBits: new Uint32Array(capacity.cells), rowTheta: new Float32Array(capacity.rows),
    executionOrder: new Uint32Array(),
    receipt: { topologyGeneration: 0, pcmCellGeneration: 0, pcmRowGeneration: 0,
      coefficientGeneration: 0, executionGeneration: 0,
      dirtyCellLeaves: new Uint32Array(), dirtyRowTiles: new Uint32Array(),
      changedDiagonalCount: 0, pressureCellCount: 0, pressureRowCount: 0,
      fault: 0, firstFaultId: INVALID } };
}

export function createSlicePressureAuthority(topology: SliceNumericalTopology,
  capacity?: Partial<SlicePressureAuthorityCapacity>): SlicePressureAuthority {
  const stableMaximum = topology.cells.reduce((maximum, cell) =>
    Math.max(maximum, cell.stableId ?? cell.id), -1) + 1;
  return allocate({ cells: Math.max(1, capacity?.cells ?? stableMaximum),
    rows: Math.max(1, capacity?.rows ?? topology.rows.length),
    bricks: Math.max(1, capacity?.bricks ?? Math.ceil(stableMaximum / 64)) });
}

function setBit(words: Uint32Array, id: number): void {
  words[id >>> 5] = (words[id >>> 5]! | (1 << (id & 31))) >>> 0;
}

function countBits(word: number): number {
  let value = word >>> 0, result = 0;
  while (value) { value &= value - 1; result++; }
  return result;
}

function changedLeaves(before: Uint32Array, after: Uint32Array,
  leafBits: number): Uint32Array {
  const wordsPerLeaf = leafBits / 32, result: number[] = [];
  for (let leaf = 0; leaf * wordsPerLeaf < after.length; leaf++) {
    let changed = false;
    for (let word = 0; word < wordsPerLeaf; word++) {
      const at = leaf * wordsPerLeaf + word;
      changed ||= (before[at] ?? 0) !== (after[at] ?? 0);
    }
    if (changed) result.push(leaf);
  }
  return Uint32Array.from(result);
}

/** Publish the accepted pressure epoch through the production PCM/PCF/PEI phases. */
export function publishSlicePressureAuthority(previous: SlicePressureAuthority,
  topology: SliceNumericalTopology, fields: SliceNumericalFields,
  rows: SlicePressureRows, topologyGeneration: number,
  options: SlicePressureAuthorityPublishOptions = {}): SlicePressureAuthority {
  let firstFaultId = INVALID, fault = 0;
  const cells = new Uint32Array(previous.acceptedCellBits.length);
  const rowBits = new Uint32Array(previous.acceptedRowBits.length);
  const densityBits = previous.densityBits.slice(), capacityBits = previous.capacityBits.slice();
  const normalXBits = previous.normalXBits.slice(), normalYBits = previous.normalYBits.slice();
  const diagonalBits = previous.diagonalBits.slice();
  const denseOrder: number[] = [], stableOrder: number[] = [];
  const rowTheta = previous.rowTheta.slice();
  for (const cell of topology.cells) {
    const stable = cell.stableId ?? cell.id;
    if (stable >= previous.cellCapacity) { fault = 3; firstFaultId = stable; break; }
    densityBits[stable] = floatBits(fields.density[cell.id]!);
    capacityBits[stable] = floatBits(fields.capacity[cell.id]!);
    normalXBits[stable] = floatBits(fields.interfaceNormal[2 * cell.id]!);
    normalYBits[stable] = floatBits(fields.interfaceNormal[2 * cell.id + 1]!);
    diagonalBits[stable] = floatBits(fields.pressureDiagonal[cell.id]!);
    if (fields.pressureMember[cell.id]) {
      setBit(cells, stable); denseOrder.push(cell.id); stableOrder.push(stable);
    }
  }
  for (const row of topology.rows) {
    if (row.id >= previous.rowCapacity) { fault ||= 3; firstFaultId = row.id; break; }
    if (rows.active[row.id]) setBit(rowBits, row.id);
    rowTheta[row.id] = rows.theta[row.id]!;
  }
  if (fault) return { ...previous, receipt: { ...previous.receipt, topologyGeneration,
    fault, firstFaultId } };

  const pcmWords = previous.pcmWords.slice();
  const cellGeneration = previous.receipt.pcmCellGeneration + 1;
  const rowGeneration = previous.receipt.pcmRowGeneration + 1;
  if (cellGeneration >= 0x7fff_ffff || rowGeneration >= 0x7fff_ffff) {
    return { ...previous, receipt: { ...previous.receipt, topologyGeneration,
      fault: 2, firstFaultId: INVALID } };
  }
  const dirtyCellLeaves = changedLeaves(previous.acceptedCellBits, cells,
    SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS);
  const changedStableCells = new Set<number>();
  for (const cell of topology.cells) {
    const stable = stableCellId(topology, cell.id), word = stable >>> 5, bit = 1 << (stable & 31);
    if (densityBits[stable] !== previous.densityBits[stable]
      || capacityBits[stable] !== previous.capacityBits[stable]
      || normalXBits[stable] !== previous.normalXBits[stable]
      || normalYBits[stable] !== previous.normalYBits[stable]
      || ((cells[word]! ^ previous.acceptedCellBits[word]!) & bit) !== 0) {
      changedStableCells.add(stable);
    }
  }
  const dirtyRows: number[] = [];
  const topologyChanged = previous.receipt.topologyGeneration !== topologyGeneration;
  for (let tile = 0; tile * 2 < rowBits.length; tile++) {
    let changed = topologyChanged || options.globalRowInvalidation
      || previous.acceptedRowBits[2 * tile] !== rowBits[2 * tile]
      || previous.acceptedRowBits[2 * tile + 1] !== rowBits[2 * tile + 1];
    if (!changed) for (let local = 0; local < 64; local++) {
      const row = topology.rows[64 * tile + local];
      if (row?.terms.some(term => changedStableCells.has(stableCellId(topology, term.cellId)))) {
        changed = true; break;
      }
    }
    if (changed) dirtyRows.push(tile);
  }
  const dirtyRowTiles = Uint32Array.from(dirtyRows);
  const cellHeader = previous.pcmLayout.cell.headerBaseWords;
  pcmWords[cellHeader + PCM_D.phase] = SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted;
  pcmWords[cellHeader + PCM_D.candidateGeneration] = cellGeneration;
  pcmWords[cellHeader + PCM_D.acceptedGeneration] = cellGeneration;
  pcmWords[cellHeader + PCM_D.dirtyCount] = dirtyCellLeaves.length;
  pcmWords[cellHeader + PCM_D.directWriteCount] = topology.cells.length;
  pcmWords[cellHeader + PCM_D.directCauseMask] = 1;
  pcmWords[cellHeader + PCM_D.totalCount] = denseOrder.length;
  pcmWords[cellHeader + PCM_D.repairIndirectX] = Math.ceil(denseOrder.length / 64);
  pcmWords.set(cells, previous.pcmLayout.cell.activeBitsBaseWords);
  for (const cell of topology.cells) {
    const stable = stableCellId(topology, cell.id);
    pcmWords[previous.pcmLayout.cell.candidateTokenBaseWords + stable] =
      (cellGeneration << 1) | (fields.pressureMember[cell.id] ? 1 : 0);
  }
  pcmWords.set(dirtyCellLeaves, previous.pcmLayout.cell.dirtyListBaseWords);
  const leafCounts = previous.pcmLayout.cell.treeLevelBaseWords[0]!;
  for (let leaf = 0; leaf < previous.pcmLayout.cell.leafCount; leaf++) {
    let total = 0;
    const begin = leaf * (SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS / 32);
    const end = Math.min(cells.length, begin + SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS / 32);
    for (let word = begin; word < end; word++) total += countBits(cells[word]!);
    pcmWords[leafCounts + leaf] = total;
  }
  // PCM's rank-select tree uses branch-32 sums at every level.
  for (let level = 1; level < previous.pcmLayout.cell.treeLevelBaseWords.length; level++) {
    const source = previous.pcmLayout.cell.treeLevelBaseWords[level - 1]!;
    const target = previous.pcmLayout.cell.treeLevelBaseWords[level]!;
    const targetCount = previous.pcmLayout.cell.treeLevelCounts[level]!;
    const sourceCount = previous.pcmLayout.cell.treeLevelCounts[level - 1]!;
    for (let node = 0; node < targetCount; node++) {
      let total = 0;
      for (let child = 0; child < 32 && node * 32 + child < sourceCount; child++) {
        total += pcmWords[source + node * 32 + child]!;
      }
      pcmWords[target + node] = total;
    }
  }
  const rowHeader = previous.pcmLayout.row.headerBaseWords;
  pcmWords[rowHeader + PCM_D.phase] = SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted;
  pcmWords[rowHeader + PCM_D.candidateGeneration] = rowGeneration;
  pcmWords[rowHeader + PCM_D.acceptedGeneration] = rowGeneration;
  pcmWords[rowHeader + PCM_D.dirtyCount] = dirtyRowTiles.length;
  pcmWords[rowHeader + PCM_D.directWriteCount] = rowBits.length;
  pcmWords[rowHeader + PCM_D.totalCount] = rowBits.reduce((sum, word) => sum + countBits(word), 0);
  pcmWords[rowHeader + PCM_D.expectedClosureCount] = topologyGeneration;
  pcmWords[rowHeader + PCM_D.coveredClosureCount] = topologyGeneration;
  pcmWords.set(rowBits, previous.pcmLayout.row.activeBitsBaseWords);
  pcmWords.set(dirtyRowTiles, previous.pcmLayout.row.dirtyTileListBaseWords);

  let changedDiagonalCount = 0;
  for (let id = 0; id < diagonalBits.length; id++) {
    if (diagonalBits[id] !== previous.diagonalBits[id]) changedDiagonalCount++;
  }
  const coefficientGeneration = previous.receipt.coefficientGeneration + 1;
  const pcfWords = previous.pcfWords.slice(), pcfBase = previous.pcfLayout.headerBaseWords;
  pcfWords[pcfBase + PCF_H.phase] = SPARSE_CM12_PRESSURE_CACHE_PHASE.accepted;
  pcfWords[pcfBase + PCF_H.candidateGeneration] = coefficientGeneration;
  pcfWords[pcfBase + PCF_H.acceptedGeneration] = coefficientGeneration;
  pcfWords[pcfBase + PCF_H.changedDiagonalCount] = changedDiagonalCount;

  const peiWords = previous.peiWords.slice(), peiBase = previous.peiLayout.baseWords;
  const executionGeneration = previous.receipt.executionGeneration + 1;
  peiWords[peiBase + PEI_H.phase] = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.accepted;
  peiWords[peiBase + PEI_H.fault] = 0; peiWords[peiBase + PEI_H.firstFaultId] = INVALID;
  peiWords[peiBase + PEI_H.generation] = executionGeneration;
  peiWords[peiBase + PEI_H.topologyGeneration] = topologyGeneration;
  peiWords[peiBase + PEI_H.pcmGeneration] = cellGeneration;
  peiWords[peiBase + PEI_H.pcmRowGeneration] = rowGeneration;
  peiWords[peiBase + PEI_H.coefficientGeneration] = coefficientGeneration;
  peiWords[peiBase + PEI_H.pressureCellCount] = stableOrder.length;
  peiWords[peiBase + PEI_H.cellIndirectX] = Math.ceil(stableOrder.length / 64);
  peiWords[peiBase + PEI_H.cellIndirectY] = 1; peiWords[peiBase + PEI_H.cellIndirectZ] = 1;
  peiWords[peiBase + PEI_H.acceptedReceipts] = previous.receipt.executionGeneration + 1;
  peiWords.fill(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
    previous.peiLayout.pressureCellBaseWords,
    previous.peiLayout.pressureCellBaseWords + previous.peiLayout.cellCapacity);
  peiWords.set(stableOrder, previous.peiLayout.pressureCellBaseWords);
  peiWords.set(cells, previous.peiLayout.pressureMembershipBaseWords);

  return { ...previous, pcmWords, pcfWords, peiWords, acceptedCellBits: cells,
    acceptedRowBits: rowBits, densityBits, capacityBits, normalXBits, normalYBits,
    diagonalBits, rowTheta, executionOrder: Uint32Array.from(denseOrder),
    receipt: { topologyGeneration, pcmCellGeneration: cellGeneration,
      pcmRowGeneration: rowGeneration, coefficientGeneration, executionGeneration,
      dirtyCellLeaves, dirtyRowTiles, changedDiagonalCount,
      pressureCellCount: denseOrder.length,
      pressureRowCount: rowBits.reduce((sum, word) => sum + countBits(word), 0),
      fault: 0, firstFaultId: INVALID } };
}
