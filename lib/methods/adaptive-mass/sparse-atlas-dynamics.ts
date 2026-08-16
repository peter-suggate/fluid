/**
 * General CPU authority step for the sparse adaptive-mass atlas.
 *
 * Coordinates and velocities are expressed in finest-cell units and seconds.
 * The caller converts physical acceleration with `aFine = aMetres / hMetres`.
 * The implementation deliberately shares the composite pressure row list:
 * every regular face and every 2:1 port has one velocity, one pressure row, and
 * one paired conservative transport transaction.
 */

import {
  cm12VolumeCorrectionDivergence,
} from "../../core/cm12-numerics";
import {
  buildSparseAtlasCompositeGrid,
  createSparseAtlasCompositeGridBuildWorkspace,
  collocateSparseAtlasVelocity,
  createSparseAtlasProjectionWorkspace,
  materializeSparseAtlasCollocatedVelocity,
  projectSparseAtlasVelocity,
  type SparseAtlasCompositeGrid,
  type SparseAtlasCompositeGridBuildWorkspace,
  type SparseAtlasGradientRow,
  type SparseAtlasProjectionOptions,
  type SparseAtlasProjectionResult,
  type SparseAtlasProjectionWorkspace,
} from "./sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import {
  conditionSparseAtlasSurface,
  createSparseAtlasSurfaceConditioningWorkspace,
  sparseAtlasHasHorizontalD4Topology,
  sparseAtlasScalarsHaveHorizontalD4Symmetry,
  type SparseAtlasSurfaceConditioningWorkspace,
} from "./sparse-atlas-surface-conditioning";
import {
  createSparseAtlasCM12Workspace,
  createSparseAtlasCellExtrapolationWorkspace,
  createSparseAtlasFaceExtrapolationWorkspace,
  createSparseAtlasFaceTopologyWorkspace,
  extrapolateSparseAtlasFaceVelocity,
  extrapolateSparseAtlasVelocity,
  transportSparseAtlasCM12,
  type SparseAtlasCM12Workspace,
  type SparseAtlasCellExtrapolationWorkspace,
  type SparseAtlasFaceExtrapolationWorkspace,
} from "./sparse-atlas-cm12-transport";

interface SparseAtlasDynamicsWorkspace {
  readonly transport: SparseAtlasCM12Workspace;
  readonly cellExtrapolation: SparseAtlasCellExtrapolationWorkspace;
  readonly faceExtrapolation: SparseAtlasFaceExtrapolationWorkspace;
  readonly surfaceConditioning: SparseAtlasSurfaceConditioningWorkspace;
  readonly projection: SparseAtlasProjectionWorkspace;
  workDensity: Float64Array;
  workGamma: Float64Array;
  fallbackFaces: Float64Array;
  remappedFaces: Float64Array;
  reconstructedFaces: Float64Array;
  faceCollocatedVelocity: Float64Array;
  faceCollocationWeights: Float64Array;
  forcedFaces: Float64Array;
  forcedCells: Float64Array;
  targetDivergence: Float64Array;
  statsDensity: Float64Array;
  readonly maximumFaceComponent: [number, number, number];
  readonly extrema: [number, number];
  readonly inputFields: TransportFields;
  readonly surfaceOptions: {
    gammaDiffusionIterations: number;
    timeStep_s: number;
    finestCellSize_m: number;
    preserveHorizontalD4: boolean;
  };
  readonly projectionOptions: {
    normalVelocity?: ArrayLike<number>;
    targetDivergence?: ArrayLike<number>;
    initialPressure?: ArrayLike<number>;
    phi?: ArrayLike<number>;
    relativeTolerance?: number;
    absoluteTolerance?: number;
    maximumIterations?: number;
    denominatorEpsilon?: number;
    sparseAirPhi?: number;
    onStageComplete?: SparseAtlasProjectionOptions["onStageComplete"];
  };
  publishedVelocityXyz: Float32Array;
  publishedVelocityRgba: Float32Array;
  readonly supportGridBuild: SparseAtlasCompositeGridBuildWorkspace;
  readonly outputGridBuild: readonly [
    SparseAtlasCompositeGridBuildWorkspace,
    SparseAtlasCompositeGridBuildWorkspace,
  ];
  outputGridBuildIndex: 0 | 1;
  readonly faceRemap: FaceRemapWorkspace;
  state?: SparseAtlasDynamicsState;
  stats?: SparseAtlasDynamicsStats;
  result?: SparseAtlasDynamicsStepResult;
}

function exactFloat64(values: Float64Array, length: number): Float64Array {
  if (values.length === length) return values;
  const available = values.byteOffset === 0 ? values.buffer.byteLength / 8 : 0;
  if (available >= length) return new Float64Array(values.buffer, 0, length);
  let capacity = 1;
  while (capacity < length) capacity *= 2;
  const grown = new Float64Array(capacity);
  return length === capacity ? grown : grown.subarray(0, length);
}

class NumericLookup {
  private keys = new Int32Array(16).fill(-1);
  private values = new Int32Array(16);
  private count = 0;

  clear(): void { this.keys.fill(-1); this.count = 0; }
  get(key: number): number | undefined {
    const mask = this.keys.length - 1;
    let slot = (Math.imul(key, -1640531527) >>> 0) & mask;
    while (this.keys[slot] !== -1) {
      if (this.keys[slot] === key) return this.values[slot];
      slot = (slot + 1) & mask;
    }
    return undefined;
  }
  set(key: number, value: number): void {
    if (4 * (this.count + 1) >= 3 * this.keys.length) this.grow();
    this.insert(key, value);
  }
  private insert(key: number, value: number): void {
    const mask = this.keys.length - 1;
    let slot = (Math.imul(key, -1640531527) >>> 0) & mask;
    while (this.keys[slot] !== -1 && this.keys[slot] !== key) slot = (slot + 1) & mask;
    if (this.keys[slot] === -1) { this.keys[slot] = key; this.count += 1; }
    this.values[slot] = value;
  }
  private grow(): void {
    const oldKeys = this.keys, oldValues = this.values;
    this.keys = new Int32Array(2 * oldKeys.length).fill(-1);
    this.values = new Int32Array(2 * oldValues.length);
    this.count = 0;
    for (let slot = 0; slot < oldKeys.length; slot += 1) {
      if (oldKeys[slot] !== -1) this.insert(oldKeys[slot], oldValues[slot]);
    }
  }
}

interface FaceRemapWorkspace {
  readonly bins: readonly [NumericLookup, NumericLookup, NumericLookup];
  minimumU: Float64Array;
  maximumU: Float64Array;
  minimumV: Float64Array;
  maximumV: Float64Array;
  marks: Uint32Array;
  candidates: Int32Array;
  stamp: number;
}

function createFaceRemapWorkspace(): FaceRemapWorkspace {
  return {
    bins: [new NumericLookup(), new NumericLookup(), new NumericLookup()],
    minimumU: new Float64Array(0), maximumU: new Float64Array(0),
    minimumV: new Float64Array(0), maximumV: new Float64Array(0),
    marks: new Uint32Array(0), candidates: new Int32Array(0), stamp: 0,
  };
}
import {
  initializeSparseAtlasResolutionPolicy,
  planSparseAtlasResolution,
  retainSparseAtlasResolutionPolicy,
  type SparseAtlasResolutionPolicyReceipt,
  type SparseAtlasResolutionPolicyState,
} from "./sparse-atlas-resolution-policy";

export interface SparseAtlasDynamicsState {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  /** Interleaved XYZ, in finest cells / second, in `grid.cells` order. */
  readonly cellVelocity: Float64Array;
  /** One oriented normal velocity for every authoritative composite row. */
  readonly faceNormalVelocity: Float64Array;
  /** Latest composite pressure potential in `grid.cells` order. */
  readonly cellPressure: Float64Array;
  readonly resolutionPolicy: SparseAtlasResolutionPolicyState;
  /** Exact invariant inherited only while both accepted state and topology are D4. */
  readonly preservesHorizontalD4: boolean;
  readonly time_s: number;
  /** Retained frame scratch; never part of the physical state or publications. */
  readonly workspace: SparseAtlasDynamicsWorkspace;
}

export interface SparseAtlasDynamicsInitializationOptions {
  readonly time_s?: number;
  readonly cellVelocity?: ArrayLike<number> | ((input: {
    readonly stableLeafId: number;
    readonly centerFine: SparseBrickVec3;
  }) => SparseBrickVec3);
}

export type SparseAtlasDynamicsStageId =
  | "receiver-topology"
  | "coupled-transport"
  | "surface-conditioning"
  | "activity-resolution"
  | "retain-rebuild"
  | "force"
  | "projection";

export interface SparseAtlasDynamicsStepOptions {
  readonly dt_s: number;
  /** Physical finest-cell size used by CM12's calibrated volume correction. */
  readonly finestCellSize_m?: number;
  /** Fixed parity modes keep every resident and receiver brick on one rung. */
  readonly resolutionMode?: "adaptive" | "all-fine" | "all-coarse";
  /** Finest-cell units / second squared. Defaults to zero. */
  readonly accelerationFinePerSecond2?: SparseBrickVec3;
  /** CM12 Sec. 3.5 dry-cell threshold; defaults to the paper's 1e-5. */
  readonly emptyEpsilon?: number;
  /** Pressure is enabled by default. */
  readonly project?: boolean;
  readonly projection?: Omit<SparseAtlasProjectionOptions, "normalVelocity" | "phi">;
  /** Optional level-set override in the retained, post-transport grid's cell order. */
  readonly phi?: ArrayLike<number>;
  /** Pure stage-boundary signal for external timing; dynamics owns no clock. */
  readonly onStageComplete?: (stage: SparseAtlasDynamicsStageId) => void;
}

export interface SparseAtlasDynamicsStats {
  readonly dt_s: number;
  readonly transportSubsteps: number;
  readonly maximumOutgoingCfl: number;
  readonly sourceBrickCount: number;
  readonly transientSupportBrickCount: number;
  readonly retainedBrickCount: number;
  readonly workCellCount: number;
  readonly workFaceCount: number;
  readonly mixedSeamFaceCount: number;
  readonly massBeforeFineCells: number;
  readonly massAfterFineCells: number;
  readonly massAbsoluteErrorFineCells: number;
  readonly gammaIntegralBeforeFineCells: number;
  readonly gammaIntegralAfterFineCells: number;
  readonly gammaIntegralAbsoluteErrorFineCells: number;
  readonly kineticEnergyBefore: number;
  readonly kineticEnergyAfter: number;
  readonly minimumDensity: number;
  readonly maximumDensity: number;
  readonly maximumDensityAfterTransport: number;
  readonly minimumGamma: number;
  readonly maximumGamma: number;
  readonly resolutionPolicy: SparseAtlasResolutionPolicyReceipt;
}

export interface SparseAtlasDynamicsStepResult {
  readonly state: SparseAtlasDynamicsState;
  readonly atlas: SparseAdaptiveMassAtlas;
  /** Transient halo grid used by conservative transport. */
  readonly workGrid: SparseAtlasCompositeGrid;
  readonly projection?: SparseAtlasProjectionResult;
  readonly stats: SparseAtlasDynamicsStats;
}

function assertFiniteVector(value: SparseBrickVec3, label: string): void {
  if (!Number.isFinite(value[0]) || !Number.isFinite(value[1])
    || !Number.isFinite(value[2])) {
    throw new RangeError(`${label} must contain finite values`);
  }
}

function zeroBrick(
  key: number,
  coordinate: SparseBrickVec3,
  resolution: SparseBrickResolution,
): SparseAdaptiveMassBrick {
  return {
    key,
    coordinate,
    resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  };
}

// A retained wet atlas is mutated in place between topology epochs. Its halo
// closure therefore has the same identity and can retain both the sparse-air
// payloads and the directory instead of rebuilding empty receiver bricks on
// every accepted frame. A topology change creates a new source atlas and
// naturally starts a new cache epoch.
const transportSupportCache = new WeakMap<
  SparseAdaptiveMassAtlas,
  Map<number, SparseAdaptiveMassAtlas>
>();

/**
 * Add only the characteristic/trilinear closure reachable this step around
 * resident tiles. Far empty space remains absent. With the method's 4³/8³
 * levels every created adjacency is automatically at most 2:1.
 */
function transportSupport(
  source: SparseAdaptiveMassAtlas,
  receiverResolution: SparseBrickResolution,
  haloBricks: number,
): SparseAdaptiveMassAtlas {
  if (source.bricks.length === 0) return source;
  let variants = transportSupportCache.get(source);
  if (!variants) {
    variants = new Map();
    transportSupportCache.set(source, variants);
  }
  // Both quantities are positive integers; this is collision-free for any
  // practical halo and avoids allocating a string key on every frame.
  const variantKey = receiverResolution + 16 * haloBricks;
  const cached = variants.get(variantKey);
  if (cached) {
    (cached as { generation: number }).generation = source.generation;
    return cached;
  }
  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  for (const brick of source.bricks) for (let dz = -haloBricks; dz <= haloBricks; dz += 1) {
    for (let dy = -haloBricks; dy <= haloBricks; dy += 1) {
      for (let dx = -haloBricks; dx <= haloBricks; dx += 1) {
        const coordinate = [brick.coordinate[0] + dx, brick.coordinate[1] + dy,
          brick.coordinate[2] + dz] as const;
        if (coordinate.some((value, axis) =>
          value < 0 || value >= source.brickDimensions[axis])) continue;
        const key = sparseBrickKey(coordinate, source.brickDimensions);
        if (!bricks.has(key)) bricks.set(key,
          zeroBrick(key, coordinate, receiverResolution));
      }
    }
  }
  if (bricks.size === source.bricks.length) {
    variants.set(variantKey, source);
    return source;
  }
  const support = createSparseAdaptiveMassAtlas(source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key), source.generation,
    source.boundary, source.brickFineResolution);
  variants.set(variantKey, support);
  return support;
}

function collocatedFaceVelocity(
  row: SparseAtlasGradientRow,
  cellVelocity: ArrayLike<number>,
): number {
  let negative = 0, negativeWeight = 0, positive = 0, positiveWeight = 0;
  for (const term of row.terms) {
    const weight = Math.abs(term.coefficient);
    const value = cellVelocity[3 * term.cellId + row.axis];
    if (term.coefficient < 0) {
      negative += weight * value;
      negativeWeight += weight;
    } else {
      positive += weight * value;
      positiveWeight += weight;
    }
  }
  if (negativeWeight > 0) negative /= negativeWeight;
  if (positiveWeight > 0) positive /= positiveWeight;
  if (negativeWeight > 0 && positiveWeight > 0) return 0.5 * (negative + positive);
  return negativeWeight > 0 ? negative : positive;
}

function facesFromCells(
  grid: SparseAtlasCompositeGrid,
  cellVelocity: ArrayLike<number>,
  result: Float64Array = new Float64Array(grid.gradientRows.length),
): Float64Array {
  if (cellVelocity.length !== 3 * grid.cells.length) {
    throw new RangeError("cellVelocity must contain interleaved XYZ for every grid cell");
  }
  if (result.length !== grid.gradientRows.length) {
    throw new RangeError("face output must contain one value per gradient row");
  }
  for (const row of grid.gradientRows) {
    result[row.id] = collocatedFaceVelocity(row, cellVelocity);
  }
  return result;
}

const TANGENTIAL_AXES = [[1, 2], [0, 2], [0, 1]] as const;

function tangentialAxes(axis: 0 | 1 | 2): readonly [0 | 1 | 2, 0 | 1 | 2] {
  return TANGENTIAL_AXES[axis];
}

/**
 * Area-average surviving face state when an omitted region becomes a coarse
 * receiver. Geometry-key remapping alone cannot map four former fine
 * sparse-air ports to their new 2x2 mixed port; this conservative overlap does.
 */
function remapFaceVelocity(
  previousGrid: SparseAtlasCompositeGrid,
  previousVelocity: ArrayLike<number>,
  nextGrid: SparseAtlasCompositeGrid,
  fallback: ArrayLike<number>,
  output: Float64Array = new Float64Array(nextGrid.gradientRows.length),
  workspace?: FaceRemapWorkspace,
): Float64Array {
  if (output.length !== nextGrid.gradientRows.length) {
    throw new RangeError("remapped face output must match the next grid");
  }
  if (workspace) return remapFaceVelocityRetained(
    previousGrid, previousVelocity, nextGrid, fallback, output, workspace,
  );
  const previousTopologyKey = (previousGrid.topologyKey
    ?? previousGrid.gradientRows) as object;
  const nextTopologyKey = (nextGrid.topologyKey ?? nextGrid.gradientRows) as object;
  let byDestination = faceRemapTopologyCache.get(previousTopologyKey);
  if (!byDestination) {
    byDestination = new WeakMap();
    faceRemapTopologyCache.set(previousTopologyKey, byDestination);
  }
  const retained = byDestination.get(nextTopologyKey);
  if (retained) {
    for (let rowId = 0; rowId < retained.length; rowId += 1) {
      const entry = retained[rowId];
      let weighted = 0;
      for (let index = 0; index < entry.ids.length; index += 1) {
        weighted += entry.overlaps[index] * previousVelocity[entry.ids[index]];
      }
      output[rowId] = entry.overlapArea > 0
        ? weighted / entry.overlapArea : fallback[rowId];
    }
    return output;
  }
  const bins: readonly Map<number, number[]>[] = [new Map(), new Map(), new Map()];
  const binKey = (axis: 0 | 1 | 2, plane: number, u: number, v: number) => {
    const tangents = tangentialAxes(axis);
    const planeCount = 2 * previousGrid.atlas.dimensions[axis] + 1;
    const uCount = previousGrid.atlas.dimensions[tangents[0]];
    return Math.round(2 * plane) + planeCount * (u + uCount * v);
  };
  const bounds = (row: SparseAtlasGradientRow) => {
    const tangents = tangentialAxes(row.axis);
    const width = Math.sqrt(row.area);
    return {
      tangents,
      minimumU: row.centerFine[tangents[0]] - 0.5 * width,
      maximumU: row.centerFine[tangents[0]] + 0.5 * width,
      minimumV: row.centerFine[tangents[1]] - 0.5 * width,
      maximumV: row.centerFine[tangents[1]] + 0.5 * width,
    };
  };
  const previousBounds = previousGrid.gradientRows.map(bounds);
  for (const row of previousGrid.gradientRows) {
    const box = previousBounds[row.id];
    for (let v = Math.floor(box.minimumV); v < Math.ceil(box.maximumV); v += 1) {
      for (let u = Math.floor(box.minimumU); u < Math.ceil(box.maximumU); u += 1) {
        const key = binKey(row.axis, row.centerFine[row.axis], u, v);
        const entries = bins[row.axis].get(key);
        if (entries) entries.push(row.id);
        else bins[row.axis].set(key, [row.id]);
      }
    }
  }
  const topology: FaceRemapEntry[] = [];
  for (const row of nextGrid.gradientRows) {
    const box = bounds(row);
    const candidates = new Set<number>();
    for (let v = Math.floor(box.minimumV); v < Math.ceil(box.maximumV); v += 1) {
      for (let u = Math.floor(box.minimumU); u < Math.ceil(box.maximumU); u += 1) {
        for (const id of bins[row.axis].get(
          binKey(row.axis, row.centerFine[row.axis], u, v),
        ) ?? []) {
          candidates.add(id);
        }
      }
    }
    let weighted = 0, overlapArea = 0;
    const ids: number[] = [];
    const overlaps: number[] = [];
    for (const candidateId of candidates) {
      const source = previousBounds[candidateId];
      const overlapU = Math.max(0,
        Math.min(box.maximumU, source.maximumU) - Math.max(box.minimumU, source.minimumU));
      const overlapV = Math.max(0,
        Math.min(box.maximumV, source.maximumV) - Math.max(box.minimumV, source.minimumV));
      const overlap = overlapU * overlapV;
      ids.push(candidateId);
      overlaps.push(overlap);
      weighted += overlap * previousVelocity[candidateId];
      overlapArea += overlap;
    }
    topology.push({ ids, overlaps, overlapArea });
    output[row.id] = overlapArea > 0 ? weighted / overlapArea : fallback[row.id];
  }
  byDestination.set(nextTopologyKey, topology);
  return output;
}

function remapFaceVelocityRetained(
  previousGrid: SparseAtlasCompositeGrid,
  previousVelocity: ArrayLike<number>,
  nextGrid: SparseAtlasCompositeGrid,
  fallback: ArrayLike<number>,
  output: Float64Array,
  workspace: FaceRemapWorkspace,
): Float64Array {
  const previousCount = previousGrid.gradientRows.length;
  let capacity = workspace.minimumU.length;
  if (capacity < previousCount) {
    capacity = 1;
    while (capacity < previousCount) capacity *= 2;
    workspace.minimumU = new Float64Array(capacity);
    workspace.maximumU = new Float64Array(capacity);
    workspace.minimumV = new Float64Array(capacity);
    workspace.maximumV = new Float64Array(capacity);
    workspace.marks = new Uint32Array(capacity);
    workspace.candidates = new Int32Array(capacity);
  }
  workspace.bins[0].clear();
  workspace.bins[1].clear();
  workspace.bins[2].clear();
  for (let rowId = 0; rowId < previousCount; rowId += 1) {
    const row = previousGrid.gradientRows[rowId];
    const tangents = tangentialAxes(row.axis);
    const width = Math.sqrt(row.area);
    const minimumU = row.centerFine[tangents[0]] - 0.5 * width;
    const minimumV = row.centerFine[tangents[1]] - 0.5 * width;
    workspace.minimumU[rowId] = minimumU;
    workspace.maximumU[rowId] = minimumU + width;
    workspace.minimumV[rowId] = minimumV;
    workspace.maximumV[rowId] = minimumV + width;
    const planeCount = 2 * previousGrid.atlas.dimensions[row.axis] + 1;
    const uCount = previousGrid.atlas.dimensions[tangents[0]];
    for (let v = Math.floor(minimumV); v < Math.ceil(minimumV + width); v += 1) {
      for (let u = Math.floor(minimumU); u < Math.ceil(minimumU + width); u += 1) {
        workspace.bins[row.axis].set(
          Math.round(2 * row.centerFine[row.axis]) + planeCount * (u + uCount * v),
          rowId,
        );
      }
    }
  }
  for (let rowId = 0; rowId < nextGrid.gradientRows.length; rowId += 1) {
    const row = nextGrid.gradientRows[rowId];
    const tangents = tangentialAxes(row.axis);
    const width = Math.sqrt(row.area);
    const minimumU = row.centerFine[tangents[0]] - 0.5 * width;
    const maximumU = minimumU + width;
    const minimumV = row.centerFine[tangents[1]] - 0.5 * width;
    const maximumV = minimumV + width;
    workspace.stamp = (workspace.stamp + 1) >>> 0;
    if (workspace.stamp === 0) {
      workspace.marks.fill(0);
      workspace.stamp = 1;
    }
    let candidateCount = 0;
    const planeCount = 2 * previousGrid.atlas.dimensions[row.axis] + 1;
    const uCount = previousGrid.atlas.dimensions[tangents[0]];
    for (let v = Math.floor(minimumV); v < Math.ceil(maximumV); v += 1) {
      for (let u = Math.floor(minimumU); u < Math.ceil(maximumU); u += 1) {
        const id = workspace.bins[row.axis].get(
          Math.round(2 * row.centerFine[row.axis]) + planeCount * (u + uCount * v),
        );
        if (id === undefined || workspace.marks[id] === workspace.stamp) continue;
        workspace.marks[id] = workspace.stamp;
        workspace.candidates[candidateCount++] = id;
      }
    }
    let weighted = 0, overlapArea = 0;
    for (let index = 0; index < candidateCount; index += 1) {
      const candidateId = workspace.candidates[index];
      const overlapU = Math.max(0, Math.min(maximumU, workspace.maximumU[candidateId])
        - Math.max(minimumU, workspace.minimumU[candidateId]));
      const overlapV = Math.max(0, Math.min(maximumV, workspace.maximumV[candidateId])
        - Math.max(minimumV, workspace.minimumV[candidateId]));
      const overlap = overlapU * overlapV;
      weighted += overlap * previousVelocity[candidateId];
      overlapArea += overlap;
    }
    output[rowId] = overlapArea > 0 ? weighted / overlapArea : fallback[rowId];
  }
  return output;
}

interface FaceRemapEntry {
  readonly ids: readonly number[];
  readonly overlaps: readonly number[];
  readonly overlapArea: number;
}

const faceRemapTopologyCache = new WeakMap<
  object,
  WeakMap<object, readonly FaceRemapEntry[]>
>();

function remapCellVelocity(
  previous: SparseAtlasDynamicsState,
  next: SparseAtlasCompositeGrid,
): Float64Array {
  const result = new Float64Array(3 * next.cells.length);
  let previousIndex = 0;
  for (const cell of next.cells) {
    while (previousIndex < previous.grid.cells.length
      && previous.grid.cells[previousIndex].stableLeafId < cell.stableLeafId) {
      previousIndex += 1;
    }
    const sourceCell = previous.grid.cells[previousIndex];
    if (!sourceCell || sourceCell.stableLeafId !== cell.stableLeafId) continue;
    result[3 * cell.id] = previous.cellVelocity[3 * sourceCell.id];
    result[3 * cell.id + 1] = previous.cellVelocity[3 * sourceCell.id + 1];
    result[3 * cell.id + 2] = previous.cellVelocity[3 * sourceCell.id + 2];
  }
  return result;
}

function sameAtlasTopology(
  left: SparseAdaptiveMassAtlas,
  right: SparseAdaptiveMassAtlas,
): boolean {
  if (left.bricks.length !== right.bricks.length) return false;
  for (let index = 0; index < left.bricks.length; index += 1) {
    const a = left.bricks[index], b = right.bricks[index];
    if (a.key !== b.key || a.resolution !== b.resolution) return false;
  }
  return true;
}

/** Reuse geometry/row topology when only density and gamma changed. */
function rebindCompositeGrid(
  topology: SparseAtlasCompositeGrid,
  atlas: SparseAdaptiveMassAtlas,
): SparseAtlasCompositeGrid {
  for (const cell of topology.cells) {
    const brick = atlas.directory.get(cell.brickKey);
    if (!brick || brick.resolution !== cell.brickResolution) {
      throw new Error("cannot rebind composite grid across a topology change");
    }
    const mutable = cell as { density: number; gamma: number };
    mutable.density = brick.density[cell.localIndex];
    mutable.gamma = brick.gamma[cell.localIndex];
  }
  (topology as { atlas: SparseAdaptiveMassAtlas }).atlas = atlas;
  return topology;
}

const transportGridCache = new WeakMap<object, Map<number, SparseAtlasCompositeGrid>>();

function transportGrid(
  sourceGrid: SparseAtlasCompositeGrid,
  supportAtlas: SparseAdaptiveMassAtlas,
  receiverResolution: SparseBrickResolution,
  haloBricks: number,
  workspace?: SparseAtlasCompositeGridBuildWorkspace,
): SparseAtlasCompositeGrid {
  const topologyKey = (sourceGrid.topologyKey ?? sourceGrid.gradientRows) as object;
  let variants = transportGridCache.get(topologyKey);
  if (!variants) {
    variants = new Map();
    transportGridCache.set(topologyKey, variants);
  }
  const variantKey = receiverResolution + 16 * haloBricks;
  const cached = variants.get(variantKey);
  if (cached && sameAtlasTopology(cached.atlas, supportAtlas)) {
    return rebindCompositeGrid(cached, supportAtlas);
  }
  if (workspace) variants.clear();
  const built = buildSparseAtlasCompositeGrid(supportAtlas, 0.5, workspace);
  variants.set(variantKey, built);
  return built;
}

function remapCellScalar(
  previousGrid: SparseAtlasCompositeGrid,
  previousValues: ArrayLike<number>,
  nextGrid: SparseAtlasCompositeGrid,
): Float64Array {
  if (previousValues.length !== previousGrid.cells.length) {
    throw new RangeError("previous scalar must contain one value per grid cell");
  }
  if (previousGrid === nextGrid && previousValues instanceof Float64Array) {
    return previousValues;
  }
  const byStableLeaf = new Map<number, number>();
  const byBrick = new Map<number, SparseAtlasCompositeGrid["cells"][number][]>();
  for (const cell of previousGrid.cells) {
    byStableLeaf.set(cell.stableLeafId, previousValues[cell.id]);
    const cells = byBrick.get(cell.brickKey) ?? [];
    cells.push(cell);
    byBrick.set(cell.brickKey, cells);
  }
  return Float64Array.from(nextGrid.cells, (cell) => {
    const exact = byStableLeaf.get(cell.stableLeafId);
    if (exact !== undefined && previousGrid.atlas.directory.get(cell.brickKey)?.resolution
      === cell.brickResolution) return exact;
    let weighted = 0;
    let volume = 0;
    for (const source of byBrick.get(cell.brickKey) ?? []) {
      const overlap = overlapVolume(source, cell);
      weighted += overlap * previousValues[source.id];
      volume += overlap;
    }
    return volume > 0 ? weighted / volume : 0;
  });
}

function overlapVolume(
  left: SparseAtlasCompositeGrid["cells"][number],
  right: SparseAtlasCompositeGrid["cells"][number],
): number {
  let result = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    result *= Math.max(0, Math.min(left.maximumFine[axis], right.maximumFine[axis])
      - Math.max(left.minimumFine[axis], right.minimumFine[axis]));
  }
  return result;
}

export function initializeSparseAtlasDynamics(
  atlas: SparseAdaptiveMassAtlas,
  options: SparseAtlasDynamicsInitializationOptions = {},
): SparseAtlasDynamicsState {
  const outputGridBuild0 = createSparseAtlasCompositeGridBuildWorkspace();
  const outputGridBuild1 = createSparseAtlasCompositeGridBuildWorkspace();
  const faceTopologyWorkspace = createSparseAtlasFaceTopologyWorkspace();
  const grid = buildSparseAtlasCompositeGrid(atlas, 0.5, outputGridBuild0);
  const cellVelocity = new Float64Array(3 * grid.cells.length);
  if (typeof options.cellVelocity === "function") {
    for (const cell of grid.cells) {
      const value = options.cellVelocity({
        stableLeafId: cell.stableLeafId,
        centerFine: cell.centerFine,
      });
      assertFiniteVector(value, "initial cell velocity");
      cellVelocity.set(value, 3 * cell.id);
    }
  } else if (options.cellVelocity) {
    if (options.cellVelocity.length !== cellVelocity.length) {
      throw new RangeError(`initial cell velocity has ${options.cellVelocity.length} values; expected ${cellVelocity.length}`);
    }
    cellVelocity.set(Array.from(options.cellVelocity));
  }
  const time_s = options.time_s ?? 0;
  if (!Number.isFinite(time_s)) throw new RangeError("initial time must be finite");
  const initialDensity = Float64Array.from(grid.cells, (cell) => cell.density);
  const initialGamma = Float64Array.from(grid.cells, (cell) => cell.gamma);
  const zeroInitialVelocity = cellVelocity.every((value) => Math.abs(value) <= 1e-14);
  const state: SparseAtlasDynamicsState = {
    atlas,
    grid,
    cellVelocity,
    faceNormalVelocity: facesFromCells(grid, cellVelocity),
    cellPressure: new Float64Array(grid.cells.length),
    resolutionPolicy: initializeSparseAtlasResolutionPolicy(atlas),
    preservesHorizontalD4: zeroInitialVelocity
      && sparseAtlasScalarsHaveHorizontalD4Symmetry(grid, initialDensity, initialGamma),
    time_s,
    workspace: {
      transport: createSparseAtlasCM12Workspace(faceTopologyWorkspace),
      cellExtrapolation: createSparseAtlasCellExtrapolationWorkspace(),
      faceExtrapolation: createSparseAtlasFaceExtrapolationWorkspace(
        faceTopologyWorkspace,
      ),
      surfaceConditioning: createSparseAtlasSurfaceConditioningWorkspace(),
      projection: createSparseAtlasProjectionWorkspace(),
      workDensity: new Float64Array(0),
      workGamma: new Float64Array(0),
      fallbackFaces: new Float64Array(0),
      remappedFaces: new Float64Array(0),
      reconstructedFaces: new Float64Array(0),
      faceCollocatedVelocity: new Float64Array(0),
      faceCollocationWeights: new Float64Array(0),
      forcedFaces: new Float64Array(0),
      forcedCells: new Float64Array(0),
      targetDivergence: new Float64Array(0),
      statsDensity: new Float64Array(0),
      maximumFaceComponent: [0, 0, 0],
      extrema: [0, 0],
      inputFields: {
        density: new Float64Array(0), gamma: new Float64Array(0),
        velocity: new Float64Array(0), faceNormalVelocity: new Float64Array(0),
      },
      surfaceOptions: {
        gammaDiffusionIterations: 1,
        timeStep_s: 0,
        finestCellSize_m: 1,
        preserveHorizontalD4: false,
      },
      projectionOptions: {},
      publishedVelocityXyz: new Float32Array(0),
      publishedVelocityRgba: new Float32Array(0),
      supportGridBuild: createSparseAtlasCompositeGridBuildWorkspace(),
      outputGridBuild: [outputGridBuild0, outputGridBuild1],
      outputGridBuildIndex: 0,
      faceRemap: createFaceRemapWorkspace(),
    },
  };
  state.workspace.state = state;
  return state;
}

/** A ball of liquid to add to a live state, in finest-cell index space. */
export interface SparseAtlasLiquidInjection {
  /** Centre in finest cells, where cell (i, j, k) spans [i, i + 1) on each axis. */
  readonly centerFine: SparseBrickVec3;
  /** Per-axis radii: a metric sphere is an ellipsoid whenever the lattice is anisotropic. */
  readonly radiusFine: SparseBrickVec3;
}

/**
 * Add liquid to a running state instead of restarting one.
 *
 * The alternative is authoring the ball into the scene document, which re-seeds
 * the solver at t = 0: the drop lands, but the run it landed in is gone. This
 * writes density straight into the atlas. Bricks the ball reaches that the
 * atlas does not hold are created 8³ — a ball is all interface, the same rule
 * the initial atlas applies to an interface brick — while bricks it already
 * holds keep the resolution the activity policy chose for them, because a drop
 * is not evidence about the region it lands in. Every touched cell takes
 * `max(existing, coverage)`, so the ball adds water and never erases any.
 *
 * Velocity is left alone: created cells arrive at rest and existing cells keep
 * what they had, which is what a ball released from a standstill looks like.
 * The mass arrives divergent and the next step's projection resolves it in the
 * same global solve as everything else — there is no separate correction here.
 */
export function injectSparseAtlasLiquid(
  state: SparseAtlasDynamicsState,
  injection: SparseAtlasLiquidInjection,
): SparseAtlasDynamicsState {
  assertFiniteVector(injection.centerFine, "injected centre");
  assertFiniteVector(injection.radiusFine, "injected radius");
  const source = state.atlas;
  const minimumFine: number[] = [];
  const maximumFine: number[] = [];
  for (let axis = 0; axis < 3; axis += 1) {
    if (!(injection.radiusFine[axis] > 0)) return state;
    minimumFine.push(Math.max(0,
      Math.floor(injection.centerFine[axis] - injection.radiusFine[axis])));
    maximumFine.push(Math.min(source.dimensions[axis] - 1,
      Math.ceil(injection.centerFine[axis] + injection.radiusFine[axis])));
    if (minimumFine[axis] > maximumFine[axis]) return state;
  }

  // Sub-sampled coverage rather than a centre-in/centre-out test, because the
  // ball is a few cells across at interactive radii and a hard test would make
  // its surface the lattice's staircase. Two samples per finest cell on every
  // axis, so a 4³ brick's larger cells are sampled at the same density.
  const coverage = (origin: readonly number[], span: number): number => {
    const perAxis = 2 * span;
    let inside = 0;
    let counted = 0;
    for (let sz = 0; sz < perAxis; sz += 1) {
      const z = origin[2] + (sz + 0.5) * span / perAxis;
      if (z < 0 || z > source.dimensions[2]) continue;
      for (let sy = 0; sy < perAxis; sy += 1) {
        const y = origin[1] + (sy + 0.5) * span / perAxis;
        if (y < 0 || y > source.dimensions[1]) continue;
        for (let sx = 0; sx < perAxis; sx += 1) {
          const x = origin[0] + (sx + 0.5) * span / perAxis;
          if (x < 0 || x > source.dimensions[0]) continue;
          counted += 1;
          const dx = (x - injection.centerFine[0]) / injection.radiusFine[0];
          const dy = (y - injection.centerFine[1]) / injection.radiusFine[1];
          const dz = (z - injection.centerFine[2]) / injection.radiusFine[2];
          if (dx * dx + dy * dy + dz * dz <= 1) inside += 1;
        }
      }
    }
    return counted > 0 ? inside / counted : 0;
  };

  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  let touched = false;
  const brickFineResolution = source.brickFineResolution;
  const first = minimumFine.map((value) => Math.floor(value / brickFineResolution));
  const last = maximumFine.map((value) => Math.floor(value / brickFineResolution));
  for (let bz = first[2]; bz <= last[2]; bz += 1) {
    for (let by = first[1]; by <= last[1]; by += 1) {
      for (let bx = first[0]; bx <= last[0]; bx += 1) {
        const coordinate: SparseBrickVec3 = [bx, by, bz];
        const key = sparseBrickKey(coordinate, source.brickDimensions);
        const brick = bricks.get(key) ?? zeroBrick(
          key, coordinate, brickFineResolution,
        );
        const span = brickFineResolution / brick.resolution;
        let density: Float64Array | undefined;
        for (let lz = 0; lz < brick.resolution; lz += 1) {
          for (let ly = 0; ly < brick.resolution; ly += 1) {
            for (let lx = 0; lx < brick.resolution; lx += 1) {
              const fraction = coverage([
                brickFineResolution * bx + span * lx,
                brickFineResolution * by + span * ly,
                brickFineResolution * bz + span * lz,
              ], span);
              const local = lx + brick.resolution * (ly + brick.resolution * lz);
              if (fraction <= brick.density[local]) continue;
              density ??= Float64Array.from(brick.density);
              density[local] = fraction;
            }
          }
        }
        if (!density) continue;
        touched = true;
        bricks.set(key, { ...brick, density });
      }
    }
  }
  if (!touched) return state;

  const atlas = createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation + 1,
    source.boundary,
    source.brickFineResolution,
  );
  const retained = sameAtlasTopology(source, atlas);
  const grid = retained
    ? rebindCompositeGrid(state.grid, atlas)
    : buildSparseAtlasCompositeGrid(atlas);
  const cellVelocity = retained ? state.cellVelocity : remapCellVelocity(state, grid);
  return {
    ...state,
    atlas,
    grid,
    cellVelocity,
    faceNormalVelocity: retained ? state.faceNormalVelocity : remapFaceVelocity(
      state.grid,
      state.faceNormalVelocity,
      grid,
      facesFromCells(grid, cellVelocity),
    ),
    cellPressure: retained
      ? state.cellPressure
      : remapCellScalar(state.grid, state.cellPressure, grid),
    resolutionPolicy: retainSparseAtlasResolutionPolicy(state.resolutionPolicy, atlas),
  };
}

interface TransportFields {
  density: Float64Array;
  gamma: Float64Array;
  velocity: Float64Array;
  faceNormalVelocity?: Float64Array;
}

function integral(
  grid: SparseAtlasCompositeGrid,
  values: ArrayLike<number>,
): number {
  let sum = 0, correction = 0;
  for (const cell of grid.cells) {
    const value = cell.volume * values[cell.id];
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

function kineticEnergy(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  velocity: ArrayLike<number>,
): number {
  let result = 0;
  for (const cell of grid.cells) {
    const offset = 3 * cell.id;
    const speedSquared = velocity[offset] ** 2 + velocity[offset + 1] ** 2
      + velocity[offset + 2] ** 2;
    result += 0.5 * cell.volume * density[cell.id] * speedSquared;
  }
  return result;
}

function extremaInto(
  values: ArrayLike<number>,
  empty: number,
  result: [number, number],
): void {
  if (values.length === 0) {
    result[0] = empty;
    result[1] = empty;
    return;
  }
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  result[0] = minimum;
  result[1] = maximum;
}

function retainedAtlas(
  source: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  gamma: ArrayLike<number>,
  epsilon: number,
  targetResolutionByBrick: ReadonlyMap<number, SparseBrickResolution>,
): SparseAdaptiveMassAtlas {
  // Ordinary frames keep the same wet brick set and rung even though the
  // transport grid includes transient dry receiver tiles. Update the retained
  // brick payloads in place in that case: the previous state is consumed by
  // this step and keeping a second immutable copy would allocate every frame.
  let sourceIndex = 0;
  let scanCell = 0;
  let topologyRetained = true;
  for (const brick of grid.atlas.bricks) {
    let wet = false;
    while (scanCell < grid.cells.length && grid.cells[scanCell].brickKey < brick.key) {
      scanCell += 1;
    }
    let brickCell = scanCell;
    while (brickCell < grid.cells.length && grid.cells[brickCell].brickKey === brick.key) {
      wet ||= density[grid.cells[brickCell].id] > epsilon;
      brickCell += 1;
    }
    scanCell = brickCell;
    if (!wet) continue;
    const expected = source.bricks[sourceIndex++];
    const target = targetResolutionByBrick.get(brick.key) ?? brick.resolution;
    if (!expected || expected.key !== brick.key
      || expected.resolution !== brick.resolution || target !== brick.resolution) {
      topologyRetained = false;
    }
  }
  topologyRetained &&= sourceIndex === source.bricks.length;
  if (topologyRetained) {
    for (const brick of source.bricks) {
      brick.density.fill(0);
      brick.gamma.fill(1);
    }
    for (const cell of grid.cells) {
      const brick = source.directory.get(cell.brickKey);
      if (!brick) continue;
      brick.density[cell.localIndex] = density[cell.id];
      brick.gamma[cell.localIndex] = gamma[cell.id];
    }
    (source as { generation: number }).generation += 1;
    return source;
  }

  const retained: SparseAdaptiveMassBrick[] = [];
  let cellCursor = 0;
  for (const brick of grid.atlas.bricks) {
    const nextDensity = new Float64Array(brick.resolution ** 3);
    const nextGamma = new Float64Array(brick.resolution ** 3).fill(1);
    let wet = false;
    while (cellCursor < grid.cells.length
      && grid.cells[cellCursor].brickKey < brick.key) cellCursor += 1;
    while (cellCursor < grid.cells.length
      && grid.cells[cellCursor].brickKey === brick.key) {
      const cell = grid.cells[cellCursor];
      nextDensity[cell.localIndex] = density[cell.id];
      nextGamma[cell.localIndex] = gamma[cell.id];
      wet ||= nextDensity[cell.localIndex] > epsilon;
      cellCursor += 1;
    }
    if (!wet) continue;
    const target = targetResolutionByBrick.get(brick.key) ?? brick.resolution;
    if (target === brick.resolution) {
      retained.push({ ...brick, density: nextDensity, gamma: nextGamma });
      continue;
    }
    retained.push({
      ...brick,
      resolution: target,
      density: resampleBrickScalar(
        source.dimensions, source.brickFineResolution, brick, nextDensity, target, 0,
      ),
      gamma: resampleBrickScalar(
        source.dimensions, source.brickFineResolution, brick, nextGamma, target, 1,
      ),
    });
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    retained,
    source.generation + 1,
    source.boundary,
    source.brickFineResolution,
  );
}

function localCellVolume(
  dimensions: SparseBrickVec3,
  brickFineResolution: number,
  brick: SparseAdaptiveMassBrick,
  resolution: SparseBrickResolution,
  x: number,
  y: number,
  z: number,
): number {
  const scale = brickFineResolution / resolution;
  const local = [x, y, z] as const;
  let volume = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = brick.coordinate[axis] * brickFineResolution + local[axis] * scale;
    volume *= Math.max(0, Math.min(scale, dimensions[axis] - lower));
  }
  return volume;
}

function resampleBrickScalar(
  dimensions: SparseBrickVec3,
  brickFineResolution: number,
  brick: SparseAdaptiveMassBrick,
  source: ArrayLike<number>,
  targetResolution: SparseBrickResolution,
  emptyValue: number,
): Float64Array {
  const output = new Float64Array(targetResolution ** 3).fill(emptyValue);
  if (targetResolution > brick.resolution
    && targetResolution % brick.resolution === 0) {
    const factor = targetResolution / brick.resolution;
    for (let z = 0; z < targetResolution; z += 1)
      for (let y = 0; y < targetResolution; y += 1)
        for (let x = 0; x < targetResolution; x += 1) {
          const sx = Math.floor(x / factor), sy = Math.floor(y / factor);
          const sz = Math.floor(z / factor);
          output[x + targetResolution * (y + targetResolution * z)] = source[
            sx + brick.resolution * (sy + brick.resolution * sz)
          ];
        }
    return output;
  }
  if (brick.resolution > targetResolution
    && brick.resolution % targetResolution === 0) {
    const factor = brick.resolution / targetResolution;
    for (let z = 0; z < targetResolution; z += 1)
      for (let y = 0; y < targetResolution; y += 1)
        for (let x = 0; x < targetResolution; x += 1) {
          let weighted = 0, volume = 0;
          for (let dz = 0; dz < factor; dz += 1)
            for (let dy = 0; dy < factor; dy += 1)
              for (let dx = 0; dx < factor; dx += 1) {
                const sx = factor * x + dx, sy = factor * y + dy;
                const sz = factor * z + dz;
                const childVolume = localCellVolume(
                  dimensions, brickFineResolution, brick, brick.resolution, sx, sy, sz,
                );
                weighted += childVolume * source[sx + brick.resolution
                  * (sy + brick.resolution * sz)];
                volume += childVolume;
              }
          output[x + targetResolution * (y + targetResolution * z)] = volume > 0
            ? weighted / volume : emptyValue;
        }
    return output;
  }
  if (brick.resolution === targetResolution) {
    for (let index = 0; index < output.length; index += 1) {
      output[index] = source[index] ?? emptyValue;
    }
    return output;
  }
  throw new Error(`unsupported sparse resolution transfer ${brick.resolution} -> ${targetResolution}`);
}

function remapWorkVelocityToOutput(
  workGrid: SparseAtlasCompositeGrid,
  workVelocity: ArrayLike<number>,
  workDensity: ArrayLike<number>,
  outputGrid: SparseAtlasCompositeGrid,
): Float64Array {
  const result = new Float64Array(3 * outputGrid.cells.length);
  const byBrick = new Map<number, SparseAtlasCompositeGrid["cells"][number][]>();
  const byStableLeaf = new Map<number, SparseAtlasCompositeGrid["cells"][number]>();
  for (const cell of workGrid.cells) {
    byStableLeaf.set(cell.stableLeafId, cell);
    const cells = byBrick.get(cell.brickKey) ?? [];
    cells.push(cell);
    byBrick.set(cell.brickKey, cells);
  }
  for (const cell of outputGrid.cells) {
    const exact = byStableLeaf.get(cell.stableLeafId);
    if (exact && exact.brickResolution === cell.brickResolution) {
      for (let axis = 0; axis < 3; axis += 1) {
        result[3 * cell.id + axis] = workVelocity[3 * exact.id + axis];
      }
      continue;
    }
    const momentum = [0, 0, 0];
    const fallback = [0, 0, 0];
    let mass = 0;
    let volume = 0;
    for (const source of byBrick.get(cell.brickKey) ?? []) {
      const overlap = overlapVolume(source, cell);
      if (!(overlap > 0)) continue;
      const weightedMass = overlap * Math.max(0, workDensity[source.id]);
      mass += weightedMass;
      volume += overlap;
      for (let axis = 0; axis < 3; axis += 1) {
        momentum[axis] += weightedMass * workVelocity[3 * source.id + axis];
        fallback[axis] += overlap * workVelocity[3 * source.id + axis];
      }
    }
    for (let axis = 0; axis < 3; axis += 1) {
      result[3 * cell.id + axis] = mass > 1e-30
        ? momentum[axis] / mass
        : volume > 0 ? fallback[axis] / volume : 0;
    }
  }
  return result;
}

export function stepSparseAtlasDynamics(
  source: SparseAtlasDynamicsState,
  options: SparseAtlasDynamicsStepOptions,
): SparseAtlasDynamicsStepResult {
  const { dt_s } = options;
  if (!Number.isFinite(dt_s) || dt_s <= 0) {
    throw new RangeError("dt_s must be finite and positive");
  }
  const workspace = source.workspace;
  const sourceBrickCount = source.atlas.bricks.length;
  const acceleration = options.accelerationFinePerSecond2 ?? [0, 0, 0];
  assertFiniteVector(acceleration, "accelerationFinePerSecond2");
  const epsilon = options.emptyEpsilon ?? 1e-5;
  if (!(epsilon >= 0)) {
    throw new RangeError("emptyEpsilon must be nonnegative");
  }

  const receiverResolution: SparseBrickResolution =
    options.resolutionMode === "all-fine"
      ? source.atlas.brickFineResolution : source.atlas.ladder.coarseResolution;
  const receiverCellSpanFine = source.atlas.brickFineResolution / receiverResolution;
  const maximumFaceComponent = workspace.maximumFaceComponent;
  maximumFaceComponent[0] = 0;
  maximumFaceComponent[1] = 0;
  maximumFaceComponent[2] = 0;
  for (let rowIndex = 0; rowIndex < source.grid.gradientRows.length; rowIndex += 1) {
    const row = source.grid.gradientRows[rowIndex];
    const value = source.faceNormalVelocity[row.id];
    const magnitude = value < 0 ? -value : value;
    if (magnitude > maximumFaceComponent[row.axis]) {
      maximumFaceComponent[row.axis] = magnitude;
    }
  }
  const maximumCharacteristicDisplacementFine = dt_s * Math.hypot(
    maximumFaceComponent[0], maximumFaceComponent[1], maximumFaceComponent[2],
  );
  // CM12 is intentionally useful at large CFL. Cover every brick reachable
  // by the characteristic plus one finest cell for trilinear interpolation;
  // never scan or allocate the rest of the authored domain.
  const transportHaloBricks = Math.max(1, Math.ceil(
    (maximumCharacteristicDisplacementFine + receiverCellSpanFine)
      / source.atlas.brickFineResolution,
  ));
  const supportAtlas = transportSupport(
    source.atlas, receiverResolution, transportHaloBricks,
  );
  const receiverTopologyUnchanged = supportAtlas === source.atlas;
  const workGrid = receiverTopologyUnchanged
    ? source.grid
    : transportGrid(
      source.grid, supportAtlas, receiverResolution, transportHaloBricks,
      workspace.supportGridBuild,
    );
  const remappedCellVelocity = receiverTopologyUnchanged
    ? source.cellVelocity
    : remapCellVelocity(source, workGrid);
  options.onStageComplete?.("receiver-topology");
  const workDensity = workspace.workDensity = exactFloat64(
    workspace.workDensity, workGrid.cells.length,
  );
  for (let cellIndex = 0; cellIndex < workGrid.cells.length; cellIndex += 1) {
    const cell = workGrid.cells[cellIndex];
    workDensity[cell.id] = cell.density;
  }
  const transportVelocity = extrapolateSparseAtlasVelocity(
    workGrid, workDensity, remappedCellVelocity, 8,
    source.workspace.cellExtrapolation,
  );
  workspace.fallbackFaces = exactFloat64(
    workspace.fallbackFaces, workGrid.gradientRows.length,
  );
  facesFromCells(workGrid, transportVelocity, workspace.fallbackFaces);
  const remappedFaces = receiverTopologyUnchanged
    ? source.faceNormalVelocity
    : remapFaceVelocity(
      source.grid,
      source.faceNormalVelocity,
      workGrid,
      workspace.fallbackFaces,
      workspace.remappedFaces = exactFloat64(
        workspace.remappedFaces, workGrid.gradientRows.length,
      ),
      workspace.faceRemap,
    );
  const transportFaces = extrapolateSparseAtlasFaceVelocity(
    workGrid, workDensity, remappedFaces,
    workspace.fallbackFaces,
    Math.max(
      2 * receiverCellSpanFine,
      maximumCharacteristicDisplacementFine + receiverCellSpanFine,
    ),
    source.workspace.faceExtrapolation,
  );
  workspace.faceCollocatedVelocity = exactFloat64(
    workspace.faceCollocatedVelocity, 3 * workGrid.cells.length,
  );
  workspace.faceCollocationWeights = exactFloat64(
    workspace.faceCollocationWeights, 3 * workGrid.cells.length,
  );
  const faceCollocatedVelocity = collocateSparseAtlasVelocity(
    workGrid, transportFaces,
    workspace.faceCollocatedVelocity, workspace.faceCollocationWeights,
  );
  const workGamma = workspace.workGamma = exactFloat64(
    workspace.workGamma, workGrid.cells.length,
  );
  for (let cellIndex = 0; cellIndex < workGrid.cells.length; cellIndex += 1) {
    const cell = workGrid.cells[cellIndex];
    workGamma[cell.id] = cell.gamma;
  }
  let fields = workspace.inputFields;
  fields.density = workDensity;
  fields.gamma = workGamma;
  fields.velocity = faceCollocatedVelocity;
  fields.faceNormalVelocity = transportFaces;
  const massBefore = integral(workGrid, fields.density);
  const gammaBefore = integral(workGrid, fields.gamma);
  const energyBefore = kineticEnergy(workGrid, fields.density, fields.velocity);
  let maximumSpeedSquared = 0;
  let maximumSpeedX = 0, maximumSpeedY = 0, maximumSpeedZ = 0;
  for (let id = 0; id < workGrid.cells.length; id += 1) {
    const x = fields.velocity[3 * id];
    const y = fields.velocity[3 * id + 1];
    const z = fields.velocity[3 * id + 2];
    const speedSquared = x * x + y * y + z * z;
    if (speedSquared > maximumSpeedSquared) {
      maximumSpeedSquared = speedSquared;
      maximumSpeedX = x;
      maximumSpeedY = y;
      maximumSpeedZ = z;
    }
  }
  const outflowRate = Math.hypot(maximumSpeedX, maximumSpeedY, maximumSpeedZ);
  const transportSubsteps = 1;
  fields = transportSparseAtlasCM12(
    workGrid, fields, dt_s, source.workspace.transport,
  ).fields;
  extremaInto(fields.density, 0, workspace.extrema);
  const maximumDensityAfterTransport = workspace.extrema[1];
  options.onStageComplete?.("coupled-transport");

  // CM12 Secs. 3.4-3.5 conditioning is part of the method, not presentation
  // polish. Run it on resident composite rows with the paper's 3dt dose.
  workspace.surfaceOptions.timeStep_s = dt_s;
  workspace.surfaceOptions.finestCellSize_m = options.finestCellSize_m ?? 1;
  workspace.surfaceOptions.preserveHorizontalD4 = source.preservesHorizontalD4;
  const conditioned = conditionSparseAtlasSurface(
    workGrid, fields, workspace.surfaceOptions, source.workspace.surfaceConditioning,
  );
  (fields as { density: Float64Array }).density = conditioned.fields.density;
  (fields as { gamma: Float64Array }).gamma = conditioned.fields.gamma;
  options.onStageComplete?.("surface-conditioning");

  const resolutionDecision = planSparseAtlasResolution(
    workGrid,
    fields.density,
    fields.velocity,
    source.resolutionPolicy,
    dt_s,
    options.resolutionMode,
  );
  options.onStageComplete?.("activity-resolution");
  const atlas = retainedAtlas(
    source.atlas,
    workGrid,
    fields.density,
    fields.gamma,
    epsilon,
    resolutionDecision.targetResolutionByBrick,
  );
  const sourceTopologyRetained = sameAtlasTopology(source.atlas, atlas);
  const workTopologyRetained = sameAtlasTopology(workGrid.atlas, atlas);
  let grid: SparseAtlasCompositeGrid;
  if (sourceTopologyRetained) grid = rebindCompositeGrid(source.grid, atlas);
  else if (workTopologyRetained) grid = rebindCompositeGrid(workGrid, atlas);
  else {
    const buildIndex: 0 | 1 = workspace.outputGridBuildIndex === 0 ? 1 : 0;
    grid = buildSparseAtlasCompositeGrid(
      atlas, 0.5, workspace.outputGridBuild[buildIndex],
    );
    workspace.outputGridBuildIndex = buildIndex;
  }
  const advectedCellVelocity = workTopologyRetained
    ? fields.velocity
    : remapWorkVelocityToOutput(workGrid, fields.velocity, fields.density, grid);
  let advectedFaces: Float64Array;
  if (workTopologyRetained && fields.faceNormalVelocity) {
    advectedFaces = fields.faceNormalVelocity;
  } else {
    workspace.reconstructedFaces = exactFloat64(
      workspace.reconstructedFaces, grid.gradientRows.length,
    );
    const reconstructedFaces = facesFromCells(
      grid, advectedCellVelocity, workspace.reconstructedFaces,
    );
    advectedFaces = workTopologyRetained
      ? reconstructedFaces
      : remapFaceVelocity(
        workGrid,
        fields.faceNormalVelocity ?? facesFromCells(workGrid, fields.velocity),
        grid,
        reconstructedFaces,
        workspace.remappedFaces = exactFloat64(
          workspace.remappedFaces, grid.gradientRows.length,
        ),
        workspace.faceRemap,
      );
  }
  options.onStageComplete?.("retain-rebuild");
  const forcedFaces = workspace.forcedFaces = exactFloat64(
    workspace.forcedFaces, grid.gradientRows.length,
  );
  for (let rowIndex = 0; rowIndex < grid.gradientRows.length; rowIndex += 1) {
    const row = grid.gradientRows[rowIndex];
    forcedFaces[row.id] = advectedFaces[row.id] + dt_s * acceleration[row.axis];
  }
  let forcedCells = advectedCellVelocity;
  if (options.project === false) {
    forcedCells = workspace.forcedCells = exactFloat64(
      workspace.forcedCells, advectedCellVelocity.length,
    );
    forcedCells.set(advectedCellVelocity);
    for (let cellIndex = 0; cellIndex < grid.cells.length; cellIndex += 1) {
      const cell = grid.cells[cellIndex];
      for (let axis = 0; axis < 3; axis += 1) {
        forcedCells[3 * cell.id + axis] += dt_s * acceleration[axis];
      }
    }
  }
  options.onStageComplete?.("force");
  const initialPressure = remapCellScalar(source.grid, source.cellPressure, grid);
  // Projection is deliberately last: the persistent/public state is the same
  // divergence-free state accepted by the pressure gate, never a pre-advection
  // diagnostic that transport has subsequently invalidated.
  const targetDivergence = workspace.targetDivergence = exactFloat64(
    workspace.targetDivergence, grid.cells.length,
  );
  for (let cellIndex = 0; cellIndex < grid.cells.length; cellIndex += 1) {
    const cell = grid.cells[cellIndex];
    targetDivergence[cell.id] = cm12VolumeCorrectionDivergence(
      cell.density,
      (options.finestCellSize_m ?? 1) * Math.min(
        cell.widthsFine[0], cell.widthsFine[1], cell.widthsFine[2],
      ),
      dt_s,
    );
  }
  const projectionOptions = workspace.projectionOptions;
  projectionOptions.normalVelocity = forcedFaces;
  projectionOptions.initialPressure = initialPressure;
  projectionOptions.phi = options.phi;
  projectionOptions.targetDivergence = targetDivergence;
  projectionOptions.relativeTolerance = options.projection?.relativeTolerance;
  projectionOptions.absoluteTolerance = options.projection?.absoluteTolerance;
  projectionOptions.maximumIterations = options.projection?.maximumIterations;
  projectionOptions.denominatorEpsilon = options.projection?.denominatorEpsilon;
  projectionOptions.sparseAirPhi = options.projection?.sparseAirPhi;
  projectionOptions.onStageComplete = options.projection?.onStageComplete;
  const projection = options.project === false ? undefined : projectSparseAtlasVelocity(
    grid, projectionOptions, source.workspace.projection,
  );
  if (projection) options.onStageComplete?.("projection");
  const cellVelocity = projection?.leafCollocatedVelocity ?? forcedCells;
  const faceNormalVelocity = projection?.projectedFaceVelocity ?? forcedFaces;
  const nextResolutionPolicy = options.resolutionMode === "adaptive"
    ? retainSparseAtlasResolutionPolicy(resolutionDecision.state, atlas)
    : resolutionDecision.state;
  const state = workspace.state ?? source;
  workspace.state = state;
  const mutableState = state as {
    atlas: SparseAdaptiveMassAtlas;
    grid: SparseAtlasCompositeGrid;
    cellVelocity: Float64Array;
    faceNormalVelocity: Float64Array;
    cellPressure: Float64Array;
    resolutionPolicy: SparseAtlasResolutionPolicyState;
    preservesHorizontalD4: boolean;
    time_s: number;
  };
  mutableState.atlas = atlas;
  mutableState.grid = grid;
  mutableState.cellVelocity = cellVelocity;
  mutableState.faceNormalVelocity = faceNormalVelocity;
  mutableState.cellPressure = projection?.leafPressure ?? initialPressure;
  mutableState.resolutionPolicy = nextResolutionPolicy;
  mutableState.preservesHorizontalD4 = source.preservesHorizontalD4
    && sparseAtlasHasHorizontalD4Topology(grid);
  mutableState.time_s = source.time_s + dt_s;
  const massAfter = integral(workGrid, fields.density);
  const gammaAfter = integral(workGrid, fields.gamma);
  extremaInto(fields.density, 0, workspace.extrema);
  const minimumDensity = workspace.extrema[0];
  const maximumDensity = workspace.extrema[1];
  extremaInto(fields.gamma, 1, workspace.extrema);
  const minimumGamma = workspace.extrema[0];
  const maximumGamma = workspace.extrema[1];
  const statsDensity = workspace.statsDensity = exactFloat64(
    workspace.statsDensity, grid.cells.length,
  );
  for (let cellIndex = 0; cellIndex < grid.cells.length; cellIndex += 1) {
    const cell = grid.cells[cellIndex];
    statsDensity[cell.id] = cell.density;
  }
  const kineticEnergyAfter = kineticEnergy(grid, statsDensity, cellVelocity);
  let stats = workspace.stats;
  if (!stats) {
    stats = workspace.stats = {
      dt_s,
      transportSubsteps,
      maximumOutgoingCfl: dt_s * outflowRate / transportSubsteps,
      sourceBrickCount,
      transientSupportBrickCount: supportAtlas.bricks.length - sourceBrickCount,
      retainedBrickCount: atlas.bricks.length,
      workCellCount: workGrid.cells.length,
      workFaceCount: workGrid.gradientRows.length,
      mixedSeamFaceCount: workGrid.mixedSeamRowCount,
      massBeforeFineCells: massBefore,
      massAfterFineCells: massAfter,
      massAbsoluteErrorFineCells: Math.abs(massAfter - massBefore),
      gammaIntegralBeforeFineCells: gammaBefore,
      gammaIntegralAfterFineCells: gammaAfter,
      gammaIntegralAbsoluteErrorFineCells: Math.abs(gammaAfter - gammaBefore),
      kineticEnergyBefore: energyBefore,
      kineticEnergyAfter,
      minimumDensity,
      maximumDensity,
      maximumDensityAfterTransport,
      minimumGamma,
      maximumGamma,
      resolutionPolicy: resolutionDecision.receipt,
    };
  } else {
    const mutable = stats as { -readonly [Key in keyof SparseAtlasDynamicsStats]:
    SparseAtlasDynamicsStats[Key] };
    mutable.dt_s = dt_s;
    mutable.transportSubsteps = transportSubsteps;
    mutable.maximumOutgoingCfl = dt_s * outflowRate / transportSubsteps;
    mutable.sourceBrickCount = sourceBrickCount;
    mutable.transientSupportBrickCount = supportAtlas.bricks.length - sourceBrickCount;
    mutable.retainedBrickCount = atlas.bricks.length;
    mutable.workCellCount = workGrid.cells.length;
    mutable.workFaceCount = workGrid.gradientRows.length;
    mutable.mixedSeamFaceCount = workGrid.mixedSeamRowCount;
    mutable.massBeforeFineCells = massBefore;
    mutable.massAfterFineCells = massAfter;
    mutable.massAbsoluteErrorFineCells = Math.abs(massAfter - massBefore);
    mutable.gammaIntegralBeforeFineCells = gammaBefore;
    mutable.gammaIntegralAfterFineCells = gammaAfter;
    mutable.gammaIntegralAbsoluteErrorFineCells = Math.abs(gammaAfter - gammaBefore);
    mutable.kineticEnergyBefore = energyBefore;
    mutable.kineticEnergyAfter = kineticEnergyAfter;
    mutable.minimumDensity = minimumDensity;
    mutable.maximumDensity = maximumDensity;
    mutable.maximumDensityAfterTransport = maximumDensityAfterTransport;
    mutable.minimumGamma = minimumGamma;
    mutable.maximumGamma = maximumGamma;
    mutable.resolutionPolicy = resolutionDecision.receipt;
  }
  let result = workspace.result;
  if (!result) {
    result = workspace.result = { state, atlas, workGrid, projection, stats };
  } else {
    const mutable = result as {
      state: SparseAtlasDynamicsState;
      atlas: SparseAdaptiveMassAtlas;
      workGrid: SparseAtlasCompositeGrid;
      projection?: SparseAtlasProjectionResult;
      stats: SparseAtlasDynamicsStats;
    };
    mutable.state = state;
    mutable.atlas = atlas;
    mutable.workGrid = workGrid;
    mutable.projection = projection;
    mutable.stats = stats;
  }
  return result;
}

/** Dense XYZ finest-cell publication, still in finest cells / second. */
export function materializeSparseAtlasDynamicsVelocity(
  state: SparseAtlasDynamicsState,
  output?: Float32Array,
): Float32Array {
  return materializeSparseAtlasCollocatedVelocity(state.grid, state.cellVelocity, output);
}

/** Dense RGBA publication convenient for rgba32float GPU textures. */
export function materializeSparseAtlasDynamicsVelocityRgba(
  state: SparseAtlasDynamicsState,
  output?: Float32Array,
): Float32Array {
  const xyzCount = 3 * state.atlas.dimensions[0] * state.atlas.dimensions[1]
    * state.atlas.dimensions[2];
  state.workspace.publishedVelocityXyz = state.workspace.publishedVelocityXyz.length === xyzCount
    ? state.workspace.publishedVelocityXyz : new Float32Array(xyzCount);
  const xyz = materializeSparseAtlasDynamicsVelocity(
    state, state.workspace.publishedVelocityXyz,
  );
  const resultCount = 4 * xyz.length / 3;
  state.workspace.publishedVelocityRgba = state.workspace.publishedVelocityRgba.length === resultCount
    ? state.workspace.publishedVelocityRgba : new Float32Array(resultCount);
  const result = output?.length === resultCount
    ? output : state.workspace.publishedVelocityRgba;
  result.fill(0);
  for (let source = 0, destination = 0; source < xyz.length; source += 3, destination += 4) {
    result[destination] = xyz[source];
    result[destination + 1] = xyz[source + 1];
    result[destination + 2] = xyz[source + 2];
  }
  return result;
}
