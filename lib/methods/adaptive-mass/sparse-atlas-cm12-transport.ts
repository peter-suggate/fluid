/**
 * CM12 characteristic transport on the compact sparse-atlas cell space.
 *
 * Grid addressing lives here; the conditioned-row and volume-weighted beta
 * formulas remain shared with Uniform CM12 through core/cm12-numerics. When
 * every brick is 8³, each sample is the ordinary eight-cell trilinear stencil.
 */
import {
  cm12ConditionedGamma,
  cm12ConditionedRowCoefficient,
  cm12VolumeScaledDeficitCoefficient,
  cm12VolumeWeightedBetaContribution,
} from "../../core/cm12-numerics";
import type { SparseBrickVec3 } from "./sparse-brick-atlas";
import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { collocateSparseAtlasVelocity } from "./sparse-atlas-composite-projection";

class Row {
  ids: Int32Array;
  coefficients: Float64Array;
  count = 0;

  constructor(capacity = 8) {
    this.ids = new Int32Array(capacity);
    this.coefficients = new Float64Array(capacity);
  }

  clear(): void {
    this.count = 0;
  }

  add(id: number, value: number): void {
    if (value === 0) return;
    for (let index = 0; index < this.count; index += 1) {
      if (this.ids[index] !== id) continue;
      this.coefficients[index] += value;
      return;
    }
    if (this.count === this.ids.length) {
      const ids = new Int32Array(2 * this.ids.length);
      const coefficients = new Float64Array(2 * this.coefficients.length);
      ids.set(this.ids);
      coefficients.set(this.coefficients);
      this.ids = ids;
      this.coefficients = coefficients;
    }
    this.ids[this.count] = id;
    this.coefficients[this.count] = value;
    this.count += 1;
  }
}

type MutableVec3 = [number, number, number];
interface NumericCellLookup { get(key: number): number | undefined }

export interface SparseAtlasCM12Workspace {
  readonly backward: Row[];
  readonly unconditioned: Row[];
  readonly rows: Row[];
  readonly forward: Row;
  readonly sampleRow: Row;
  readonly traceInitial: MutableVec3;
  readonly traceFirst: MutableVec3;
  readonly traceMiddle: MutableVec3;
  readonly tracePosition: MutableVec3;
  readonly traceMidpoint: MutableVec3;
  advectedGamma: Float64Array;
  backwardBeta: Float64Array;
  nextGamma: Float64Array;
  conditionedBeta: Float64Array;
  finalBeta: Float64Array;
  density: Float64Array;
  faceNormalVelocity: Float64Array;
  velocity: Float64Array;
  collocationWeights: Float64Array;
  faceSampler?: RetainedFaceSampler;
  outputFields?: SparseAtlasCM12TransportFields;
  result?: SparseAtlasCM12TransportResult;
  topologyKey?: object;
  readonly cellLookup: Int32Lookup;
  readonly faceTopologyWorkspace: SparseAtlasFaceTopologyWorkspace;
}

export function createSparseAtlasCM12Workspace(
  faceTopologyWorkspace = createSparseAtlasFaceTopologyWorkspace(),
): SparseAtlasCM12Workspace {
  return {
    backward: [], unconditioned: [], rows: [], forward: new Row(), sampleRow: new Row(),
    traceInitial: [0, 0, 0], traceFirst: [0, 0, 0], traceMiddle: [0, 0, 0],
    tracePosition: [0, 0, 0], traceMidpoint: [0, 0, 0],
    advectedGamma: new Float64Array(0),
    backwardBeta: new Float64Array(0),
    nextGamma: new Float64Array(0),
    conditionedBeta: new Float64Array(0),
    finalBeta: new Float64Array(0),
    density: new Float64Array(0),
    faceNormalVelocity: new Float64Array(0),
    velocity: new Float64Array(0),
    collocationWeights: new Float64Array(0),
    cellLookup: new Int32Lookup(),
    faceTopologyWorkspace,
  };
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

function rowsFor(storage: Row[], count: number, rowCapacity = 8): Row[] {
  const capacity = Math.ceil(count / 4096) * 4096;
  while (storage.length < capacity) storage.push(new Row(rowCapacity));
  for (let index = 0; index < count; index += 1) storage[index].clear();
  return storage;
}

export interface SparseAtlasCM12TransportFields {
  readonly density: Float64Array;
  readonly gamma: Float64Array;
  readonly velocity: Float64Array;
  readonly faceNormalVelocity?: Float64Array;
}

export interface SparseAtlasCM12TransportResult {
  readonly fields: SparseAtlasCM12TransportFields;
  readonly faceNormalVelocity: Float64Array;
  readonly finalBetaMaximumAbsoluteError: number;
}

function add(row: Row, id: number, value: number): void {
  row.add(id, value);
}

function sumRow(row: Row): number {
  let result = 0, correction = 0;
  for (let index = 0; index < row.count; index += 1) {
    const value = row.coefficients[index];
    const adjusted = value - correction;
    const next = result + adjusted;
    correction = next - result - adjusted;
    result = next;
  }
  return result;
}

function fineCellKey(
  grid: SparseAtlasCompositeGrid,
  coordinate: readonly [number, number, number],
): number {
  return coordinate[0] + grid.atlas.dimensions[0]
    * (coordinate[1] + grid.atlas.dimensions[1] * coordinate[2]);
}

const compactFineCellLookupCache = new WeakMap<object, NumericCellLookup>();
const uniformSamplingSpansCache = new WeakMap<object, readonly [number, number, number] | null>();
const FINE_SAMPLING_SPANS = [1, 1, 1] as const;

function compactFineCellLookup(
  grid: SparseAtlasCompositeGrid,
  result?: Int32Lookup,
): NumericCellLookup {
  const key = (grid.topologyKey ?? grid.gradientRows) as object;
  if (result) result.clear();
  const cached = compactFineCellLookupCache.get(key);
  if (!result && cached) return cached;
  const output = result ?? new Map<number, number>();
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1) {
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1) {
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          output.set(x + grid.atlas.dimensions[0]
            * (y + grid.atlas.dimensions[1] * z), cell.id);
        }
      }
    }
  }
  if (!result) compactFineCellLookupCache.set(key, output);
  return output;
}

function uniformSamplingSpans(
  grid: SparseAtlasCompositeGrid,
): readonly [number, number, number] | null {
  const topologyKey = (grid.topologyKey ?? grid.gradientRows) as object;
  let uniform = uniformSamplingSpansCache.get(topologyKey);
  if (uniform === undefined) {
    const first = grid.cells[0];
    uniform = first && grid.cells.every((cell) =>
      cell.widthsFine.every((width, axis) => width === first.widthsFine[axis]))
      ? [...first.widthsFine] as [number, number, number]
      : null;
    uniformSamplingSpansCache.set(topologyKey, uniform);
  }
  return uniform;
}

function samplingSpans(
  grid: SparseAtlasCompositeGrid,
  position: SparseBrickVec3,
  cellLookup: NumericCellLookup,
): readonly [number, number, number] {
  const uniform = uniformSamplingSpans(grid);
  if (uniform) return uniform;
  const coordinate0 = Math.max(0, Math.min(
    grid.atlas.dimensions[0] - 1, Math.floor(position[0])));
  const coordinate1 = Math.max(0, Math.min(
    grid.atlas.dimensions[1] - 1, Math.floor(position[1])));
  const coordinate2 = Math.max(0, Math.min(
    grid.atlas.dimensions[2] - 1, Math.floor(position[2])));
  const id = cellLookup.get(coordinate0 + grid.atlas.dimensions[0]
    * (coordinate1 + grid.atlas.dimensions[1] * coordinate2));
  if (id === undefined) return FINE_SAMPLING_SPANS;
  return grid.cells[id].widthsFine;
}

/** Cell-centred nonnegative trilinear weights, collapsed onto compact leaves. */
function sampleWeights(
  grid: SparseAtlasCompositeGrid,
  position: SparseBrickVec3,
  cellLookup?: NumericCellLookup,
  topologySpans?: readonly [number, number, number] | null,
  row = new Row(),
): Row {
  const lookup = cellLookup ?? compactFineCellLookup(grid);
  const spans = topologySpans ?? samplingSpans(grid, position, lookup);
  const halfSpan0 = 0.5 * spans[0];
  const halfSpan1 = 0.5 * spans[1];
  const halfSpan2 = 0.5 * spans[2];
  const coordinate0 = Math.max(halfSpan0, Math.min(
    grid.atlas.dimensions[0] - halfSpan0, position[0])) / spans[0] - 0.5;
  const coordinate1 = Math.max(halfSpan1, Math.min(
    grid.atlas.dimensions[1] - halfSpan1, position[1])) / spans[1] - 0.5;
  const coordinate2 = Math.max(halfSpan2, Math.min(
    grid.atlas.dimensions[2] - halfSpan2, position[2])) / spans[2] - 0.5;
  const base0 = Math.floor(coordinate0);
  const base1 = Math.floor(coordinate1);
  const base2 = Math.floor(coordinate2);
  const fraction0 = coordinate0 - base0;
  const fraction1 = coordinate1 - base1;
  const fraction2 = coordinate2 - base2;
  row.clear();
  for (let dz = 0; dz <= 1; dz += 1) for (let dy = 0; dy <= 1; dy += 1) {
    for (let dx = 0; dx <= 1; dx += 1) {
      const fine0 = Math.max(0, Math.min(grid.atlas.dimensions[0] - 1,
        Math.floor(spans[0] * (base0 + dx + 0.5))));
      const fine1 = Math.max(0, Math.min(grid.atlas.dimensions[1] - 1,
        Math.floor(spans[1] * (base1 + dy + 0.5))));
      const fine2 = Math.max(0, Math.min(grid.atlas.dimensions[2] - 1,
        Math.floor(spans[2] * (base2 + dz + 0.5))));
      const fineKey = fine0 + grid.atlas.dimensions[0]
        * (fine1 + grid.atlas.dimensions[1] * fine2);
      let id = lookup.get(fineKey);
      if (id === undefined) {
        const brick0 = Math.floor(fine0 / 8);
        const brick1 = Math.floor(fine1 / 8);
        const brick2 = Math.floor(fine2 / 8);
        const key = brick0 + grid.atlas.brickDimensions[0]
          * (brick1 + grid.atlas.brickDimensions[1] * brick2);
        const brick = grid.atlas.directory.get(key);
        const cellBase = grid.cellBaseByBrick.get(key);
        if (brick && cellBase !== undefined) {
          const scale = 8 / brick.resolution;
          const local0 = Math.floor((fine0 - 8 * brick0) / scale);
          const local1 = Math.floor((fine1 - 8 * brick1) / scale);
          const local2 = Math.floor((fine2 - 8 * brick2) / scale);
          id = cellBase + local0 + brick.resolution
            * (local1 + brick.resolution * local2);
        }
      }
      if (id === undefined) continue;
      const weight = (dx ? fraction0 : 1 - fraction0)
        * (dy ? fraction1 : 1 - fraction1)
        * (dz ? fraction2 : 1 - fraction2);
      add(row, id, weight);
    }
  }
  return row;
}

interface FaceSampler {
  readonly sample: (position: SparseBrickVec3, component: 0 | 1 | 2) => number;
}

class Int32Lookup {
  private keys = new Int32Array(16).fill(-1);
  private values = new Int32Array(16);
  private count = 0;

  clear(): void {
    this.keys.fill(-1);
    this.count = 0;
  }

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
    if (this.keys[slot] === -1) {
      this.keys[slot] = key;
      this.count += 1;
    }
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

class StencilCache {
  private readonly lookup = new Int32Lookup();
  private ids = new Int32Array(8 * 16).fill(-1);
  private count = 0;

  clear(): void {
    this.lookup.clear();
    this.count = 0;
  }

  /** Existing slot, or bitwise complement of a freshly inserted slot. */
  slot(key: number): number {
    const found = this.lookup.get(key);
    if (found !== undefined) return found;
    const slot = this.count++;
    if (8 * this.count > this.ids.length) {
      const grown = new Int32Array(2 * this.ids.length).fill(-1);
      grown.set(this.ids);
      this.ids = grown;
    }
    this.ids.fill(-1, 8 * slot, 8 * slot + 8);
    this.lookup.set(key, slot);
    return ~slot;
  }

  get(slot: number, index: number): number {
    return this.ids[8 * slot + index];
  }

  set(slot: number, index: number, value: number): void {
    this.ids[8 * slot + index] = value;
  }
}

type FaceBins = [Int32Lookup, Int32Lookup, Int32Lookup];

interface FaceTopology {
  bins: FaceBins;
  /** Negative then positive neighbor for x, y and z. */
  neighbors: Int32Array[];
  spacing: Float64Array[];
}

export interface SparseAtlasFaceTopologyWorkspace {
  topologyKey?: object;
  topology?: FaceTopology;
}

export function createSparseAtlasFaceTopologyWorkspace():
SparseAtlasFaceTopologyWorkspace {
  return {};
}

export interface SparseAtlasFaceExtrapolationWorkspace {
  values: Float64Array;
  distance: Float64Array;
  nextValues: Float64Array;
  nextDistance: Float64Array;
  readonly candidateAxes: Int8Array;
  readonly candidateDistance: Float64Array;
  readonly candidateSpacing: Float64Array;
  readonly topologyWorkspace: SparseAtlasFaceTopologyWorkspace;
}

export function createSparseAtlasFaceExtrapolationWorkspace(
  topologyWorkspace = createSparseAtlasFaceTopologyWorkspace(),
):
SparseAtlasFaceExtrapolationWorkspace {
  return {
    values: new Float64Array(0),
    distance: new Float64Array(0),
    nextValues: new Float64Array(0),
    nextDistance: new Float64Array(0),
    candidateAxes: new Int8Array(3),
    candidateDistance: new Float64Array(3),
    candidateSpacing: new Float64Array(3),
    topologyWorkspace,
  };
}

const FACE_TANGENTS = [[1, 2], [0, 2], [0, 1]] as const;

const faceTangents = (axis: 0 | 1 | 2): readonly [0 | 1 | 2, 0 | 1 | 2] =>
  FACE_TANGENTS[axis];

function buildFaceBins(grid: SparseAtlasCompositeGrid, bins?: FaceBins): FaceBins {
  bins ??= [new Int32Lookup(), new Int32Lookup(), new Int32Lookup()];
  bins[0].clear();
  bins[1].clear();
  bins[2].clear();
  for (const row of grid.gradientRows) {
    const [uAxis, vAxis] = faceTangents(row.axis);
    const width = Math.sqrt(row.area);
    const minimumU = row.centerFine[uAxis] - 0.5 * width;
    const minimumV = row.centerFine[vAxis] - 0.5 * width;
    for (let v = Math.floor(minimumV); v < Math.ceil(minimumV + width); v += 1) {
      for (let u = Math.floor(minimumU); u < Math.ceil(minimumU + width); u += 1) {
        const planeCount = grid.atlas.dimensions[row.axis] + 1;
        const uCount = grid.atlas.dimensions[uAxis];
        bins[row.axis].set(Math.round(row.centerFine[row.axis])
          + planeCount * (u + uCount * v), row.id);
      }
    }
  }
  return bins;
}

const faceTopologyCache = new WeakMap<object, FaceTopology>();

function faceTopology(
  grid: SparseAtlasCompositeGrid,
  workspace?: SparseAtlasFaceTopologyWorkspace,
): FaceTopology {
  const key = (grid.topologyKey ?? grid.gradientRows) as object;
  if (workspace?.topologyKey === key && workspace.topology) return workspace.topology;
  const cached = faceTopologyCache.get(key);
  if (!workspace && cached) return cached;
  const topology = workspace?.topology ?? {
    bins: [new Int32Lookup(), new Int32Lookup(), new Int32Lookup()] as FaceBins,
    neighbors: [], spacing: [],
  };
  const bins = buildFaceBins(grid, topology.bins);
  const neighbors = topology.neighbors;
  const spacing = topology.spacing;
  while (neighbors.length < 6) neighbors.push(new Int32Array(0));
  while (spacing.length < 6) spacing.push(new Float64Array(0));
  for (let slot = 0; slot < 6; slot += 1) {
    const old = neighbors[slot];
    const available = old.byteOffset === 0 ? old.buffer.byteLength / 4 : 0;
    if (available >= grid.gradientRows.length) {
      neighbors[slot] = old.length === grid.gradientRows.length
        ? old : new Int32Array(old.buffer, 0, grid.gradientRows.length);
    } else {
      let capacity = 1;
      while (capacity < grid.gradientRows.length) capacity *= 2;
      const grown = new Int32Array(capacity);
      neighbors[slot] = capacity === grid.gradientRows.length
        ? grown : grown.subarray(0, grid.gradientRows.length);
    }
    spacing[slot] = exactFloat64(spacing[slot], grid.gradientRows.length);
    neighbors[slot].fill(-1);
    spacing[slot].fill(0);
  }
  for (let rowIndex = 0; rowIndex < grid.gradientRows.length; rowIndex += 1) {
    const row = grid.gradientRows[rowIndex];
    for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
      const axis = axisIndex as 0 | 1 | 2;
      for (let signIndex = 0; signIndex < 2; signIndex += 1) {
        const sign = signIndex === 0 ? -1 : 1;
      const slot = 2 * axis + (sign > 0 ? 1 : 0);
      // A compact coarse face occupies multiple finest-lattice bins. Walk
      // through those aliases until the first distinct face is reached.
      for (let offset = 1; offset <= 4; offset += 1) {
        const coordinate0 = row.centerFine[0] + (axis === 0 ? sign * offset : 0);
        const coordinate1 = row.centerFine[1] + (axis === 1 ? sign * offset : 0);
        const coordinate2 = row.centerFine[2] + (axis === 2 ? sign * offset : 0);
        const candidate = faceBinAtValues(
          grid, bins, row.axis, coordinate0, coordinate1, coordinate2,
        );
        if (candidate === undefined || candidate === row.id) continue;
        neighbors[slot][row.id] = candidate;
        spacing[slot][row.id] = Math.abs(
          grid.gradientRows[candidate].centerFine[axis] - row.centerFine[axis],
        );
        break;
      }
    }
  }
  }
  topology.bins = bins;
  topology.neighbors = neighbors;
  topology.spacing = spacing;
  if (workspace) {
    workspace.topology = topology;
    workspace.topologyKey = key;
  } else faceTopologyCache.set(key, topology);
  return topology;
}

function faceBinAtValues(
  grid: SparseAtlasCompositeGrid,
  bins: FaceBins,
  axis: 0 | 1 | 2,
  coordinate0: number,
  coordinate1: number,
  coordinate2: number,
): number | undefined {
  const coordinate = axis === 0 ? coordinate0 : axis === 1 ? coordinate1 : coordinate2;
  const u = axis === 0 ? coordinate1 : coordinate0;
  const v = axis === 2 ? coordinate1 : coordinate2;
  const planeCount = grid.atlas.dimensions[axis] + 1;
  const uCount = grid.atlas.dimensions[axis === 0 ? 1 : 0];
  return bins[axis].get(Math.round(coordinate) + planeCount
    * (Math.floor(u) + uCount * Math.floor(v)));
}

function faceBinAt(
  grid: SparseAtlasCompositeGrid,
  bins: FaceBins,
  axis: 0 | 1 | 2,
  coordinate: readonly number[],
): number | undefined {
  const [uAxis, vAxis] = faceTangents(axis);
  const planeCount = grid.atlas.dimensions[axis] + 1;
  const uCount = grid.atlas.dimensions[uAxis];
  return bins[axis].get(Math.round(coordinate[axis]) + planeCount
    * (Math.floor(coordinate[uAxis]) + uCount * Math.floor(coordinate[vAxis])));
}

/**
 * CM12 Sec. 3.3 / CM11b narrow-band MAC extension.
 *
 * The dense reference solves the same Eikonal/upwind equations for two cells
 * and uses its hierarchy outside that band. Sparse can instead continue the
 * accurate solve through the locally reachable characteristic band without
 * touching the authored empty domain; `fallback` supplies any farther values
 * that no trace can consume this step. Work is proportional to represented
 * faces and the immutable face graph is cached per topology epoch.
 */
export function extrapolateSparseAtlasFaceVelocity(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  input: ArrayLike<number>,
  fallback: ArrayLike<number> = input,
  bandCells = 2,
  workspace: SparseAtlasFaceExtrapolationWorkspace =
    createSparseAtlasFaceExtrapolationWorkspace(),
): Float64Array {
  const count = grid.gradientRows.length;
  if (input.length !== count || fallback.length !== count) {
    throw new RangeError("face extrapolation fields do not match the sparse grid");
  }
  const topology = faceTopology(grid, workspace.topologyWorkspace);
  let values = workspace.values = exactFloat64(workspace.values, count);
  let distance = workspace.distance = exactFloat64(workspace.distance, count);
  let nextValues = workspace.nextValues = exactFloat64(workspace.nextValues, count);
  let nextDistance = workspace.nextDistance = exactFloat64(workspace.nextDistance, count);
  values.fill(0);
  distance.fill(Number.POSITIVE_INFINITY);
  for (const row of grid.gradientRows) {
    let liquid = false;
    for (let index = 0; index < row.terms.length; index += 1) {
      if (density[row.terms[index].cellId] > 0.5) {
        liquid = true;
        break;
      }
    }
    if (liquid) {
      values[row.id] = input[row.id];
      distance[row.id] = 0;
    }
  }
  const maximumSweeps = Math.max(16, Math.ceil(bandCells) + 2);
  for (let sweep = 0; sweep < maximumSweeps; sweep += 1) {
    nextValues.set(values);
    nextDistance.set(distance);
    let changed = false;
    for (const row of grid.gradientRows) {
      if (distance[row.id] === 0) continue;
      let axisCount = 0;
      for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
        const axis = axisIndex as 0 | 1 | 2;
        const negativeSlot = 2 * axis, positiveSlot = negativeSlot + 1;
        const negativeId = topology.neighbors[negativeSlot][row.id];
        const positiveId = topology.neighbors[positiveSlot][row.id];
        const negativeDistance = negativeId >= 0
          ? distance[negativeId] : Number.POSITIVE_INFINITY;
        const positiveDistance = positiveId >= 0
          ? distance[positiveId] : Number.POSITIVE_INFINITY;
        const slot = negativeDistance <= positiveDistance ? negativeSlot : positiveSlot;
        const candidateDistance = Math.min(negativeDistance, positiveDistance);
        const candidateSpacing = topology.spacing[slot][row.id];
        if (!Number.isFinite(candidateDistance) || !(candidateSpacing > 0)) continue;
        let insertion = axisCount;
        while (insertion > 0
          && workspace.candidateDistance[insertion - 1] > candidateDistance) {
          workspace.candidateAxes[insertion] = workspace.candidateAxes[insertion - 1];
          workspace.candidateDistance[insertion] = workspace.candidateDistance[insertion - 1];
          workspace.candidateSpacing[insertion] = workspace.candidateSpacing[insertion - 1];
          insertion -= 1;
        }
        workspace.candidateAxes[insertion] = axis;
        workspace.candidateDistance[insertion] = candidateDistance;
        workspace.candidateSpacing[insertion] = candidateSpacing;
        axisCount += 1;
      }
      if (axisCount === 0) continue;
      let solved = workspace.candidateDistance[0] + workspace.candidateSpacing[0];
      for (let used = 2; used <= axisCount; used += 1) {
        if (solved <= workspace.candidateDistance[used - 1]) break;
        let a = 0, b = 0, c = -1;
        for (let index = 0; index < used; index += 1) {
          const candidateDistance = workspace.candidateDistance[index];
          const inverseSpacingSquared = 1 / (workspace.candidateSpacing[index] ** 2);
          a += inverseSpacingSquared;
          b += candidateDistance * inverseSpacingSquared;
          c += candidateDistance ** 2 * inverseSpacingSquared;
        }
        solved = (b + Math.sqrt(Math.max(0, b * b - a * c))) / a;
      }
      solved = Math.min(distance[row.id], solved);
      if (solved > bandCells + 1e-12) continue;
      let weightedTerm0 = 0, weightedTerm1 = 0, weightedTerm2 = 0;
      let weightTerm0 = 0, weightTerm1 = 0, weightTerm2 = 0;
      // Match the dense f32 FIM convergence/tie tolerance.
      const epsilon = Math.max(2, Math.abs(solved)) * 1.1920929e-7;
      for (let candidate = 0; candidate < axisCount; candidate += 1) {
        const axis = workspace.candidateAxes[candidate];
        const axisDistance = workspace.candidateDistance[candidate];
        const axisSpacing = workspace.candidateSpacing[candidate];
        if (axisDistance >= solved - epsilon) continue;
        let minimizerCount = 0, value = 0;
        const negativeSlot = 2 * axis;
        for (let slot = negativeSlot; slot <= negativeSlot + 1; slot += 1) {
          const id = topology.neighbors[slot][row.id];
          const minimizerDistance = id >= 0
            ? distance[id] : Number.POSITIVE_INFINITY;
          if (id < 0 || Math.abs(minimizerDistance - axisDistance) > epsilon) continue;
          value += values[id];
          minimizerCount += 1;
        }
        if (minimizerCount === 0) continue;
        const weight = (solved - axisDistance) / (axisSpacing ** 2);
        const weighted = weight * value / minimizerCount;
        if (axis === 0) {
          weightedTerm0 = weighted;
          weightTerm0 = weight;
        } else if (axis === 1) {
          weightedTerm1 = weighted;
          weightTerm1 = weight;
        } else {
          weightedTerm2 = weighted;
          weightTerm2 = weight;
        }
      }
      const weightedValue = (weightedTerm0 + weightedTerm2) + weightedTerm1;
      const weightSum = (weightTerm0 + weightTerm2) + weightTerm1;
      if (weightSum <= 0) continue;
      nextDistance[row.id] = solved;
      nextValues[row.id] = weightedValue / weightSum;
      changed ||= Math.abs(solved - distance[row.id]) > epsilon;
    }
    const previousValues = values, previousDistance = distance;
    values = nextValues;
    distance = nextDistance;
    nextValues = previousValues;
    nextDistance = previousDistance;
    if (!changed) break;
  }
  for (let rowId = 0; rowId < count; rowId += 1) {
    if (!Number.isFinite(distance[rowId])) values[rowId] = fallback[rowId];
  }
  workspace.values = values;
  workspace.distance = distance;
  workspace.nextValues = nextValues;
  workspace.nextDistance = nextDistance;
  return values;
}

class RetainedFaceSampler implements FaceSampler {
  private grid!: SparseAtlasCompositeGrid;
  private velocity!: ArrayLike<number>;
  private bins!: FaceBins;
  private cellLookup!: NumericCellLookup;
  private topologySpans!: readonly [number, number, number] | null;
  private topologyKey?: object;
  private readonly stencilIds: readonly StencilCache[] = [
    new StencilCache(), new StencilCache(), new StencilCache(),
  ];

  configure(
    grid: SparseAtlasCompositeGrid,
    velocity: ArrayLike<number>,
    cellLookup: NumericCellLookup,
    faceTopologyWorkspace: SparseAtlasFaceTopologyWorkspace,
  ): void {
    const topologyKey = (grid.topologyKey ?? grid.gradientRows) as object;
    if (this.topologyKey !== topologyKey) {
      this.topologyKey = topologyKey;
      this.bins = faceTopology(grid, faceTopologyWorkspace).bins;
      this.cellLookup = cellLookup;
      this.topologySpans = uniformSamplingSpans(grid);
      this.stencilIds[0].clear();
      this.stencilIds[1].clear();
      this.stencilIds[2].clear();
    }
    this.grid = grid;
    this.velocity = velocity;
  }

  sample(position: SparseBrickVec3, component: 0 | 1 | 2): number {
      const grid = this.grid;
      const velocity = this.velocity;
      const bins = this.bins;
      const topologySpans = this.topologySpans;
      const spans = topologySpans ?? samplingSpans(grid, position, this.cellLookup);
      const planeCount = grid.atlas.dimensions[component] + 1;
      const uCount = grid.atlas.dimensions[component === 0 ? 1 : 0];
      const clamped0 = Math.max(0.5 * spans[0], Math.min(
        grid.atlas.dimensions[0] - 0.5 * spans[0], position[0]));
      const clamped1 = Math.max(0.5 * spans[1], Math.min(
        grid.atlas.dimensions[1] - 0.5 * spans[1], position[1]));
      const clamped2 = Math.max(0.5 * spans[2], Math.min(
        grid.atlas.dimensions[2] - 0.5 * spans[2], position[2]));
      const coordinate0 = clamped0 / spans[0] - (component === 0 ? 0 : 0.5);
      const coordinate1 = clamped1 / spans[1] - (component === 1 ? 0 : 0.5);
      const coordinate2 = clamped2 / spans[2] - (component === 2 ? 0 : 0.5);
      const base0 = Math.floor(coordinate0);
      const base1 = Math.floor(coordinate1);
      const base2 = Math.floor(coordinate2);
      const fraction0 = coordinate0 - base0;
      const fraction1 = coordinate1 - base1;
      const fraction2 = coordinate2 - base2;
      const baseCount0 = Math.ceil(grid.atlas.dimensions[0] / spans[0]) + 1;
      const baseCount1 = Math.ceil(grid.atlas.dimensions[1] / spans[1]) + 1;
      const stencilKey = base0 + baseCount0 * (base1 + baseCount1 * base2);
      let stencilSlot = -1;
      let freshStencil = false;
      if (topologySpans) {
        const encodedSlot = this.stencilIds[component].slot(stencilKey);
        freshStencil = encodedSlot < 0;
        stencilSlot = freshStencil ? ~encodedSlot : encodedSlot;
      }
      if (topologySpans && freshStencil) {
        for (let dz = 0; dz <= 1; dz += 1) for (let dy = 0; dy <= 1; dy += 1) {
          for (let dx = 0; dx <= 1; dx += 1) {
            const x = spans[0] * (base0 + dx + (component === 0 ? 0 : 0.5));
            const y = spans[1] * (base1 + dy + (component === 1 ? 0 : 0.5));
            const z = spans[2] * (base2 + dz + (component === 2 ? 0 : 0.5));
            const key = component === 0
              ? Math.round(x) + planeCount * (Math.floor(y) + uCount * Math.floor(z))
              : component === 1
                ? Math.round(y) + planeCount * (Math.floor(x) + uCount * Math.floor(z))
                : Math.round(z) + planeCount * (Math.floor(x) + uCount * Math.floor(y));
            this.stencilIds[component].set(
              stencilSlot, dx + 2 * dy + 4 * dz, bins[component].get(key) ?? -1,
            );
          }
        }
      }
      let term0 = 0, term1 = 0, term2 = 0, term3 = 0;
      let term4 = 0, term5 = 0, term6 = 0, term7 = 0;
      for (let dz = 0; dz <= 1; dz += 1) for (let dy = 0; dy <= 1; dy += 1) {
        for (let dx = 0; dx <= 1; dx += 1) {
          const weight = (dx ? fraction0 : 1 - fraction0)
            * (dy ? fraction1 : 1 - fraction1)
            * (dz ? fraction2 : 1 - fraction2);
          const termIndex = dx + 2 * dy + 4 * dz;
          let id = topologySpans
            ? this.stencilIds[component].get(stencilSlot, termIndex) : undefined;
          if (id === undefined) {
            const x = spans[0] * (base0 + dx + (component === 0 ? 0 : 0.5));
            const y = spans[1] * (base1 + dy + (component === 1 ? 0 : 0.5));
            const z = spans[2] * (base2 + dz + (component === 2 ? 0 : 0.5));
            const key = component === 0
              ? Math.round(x) + planeCount * (Math.floor(y) + uCount * Math.floor(z))
              : component === 1
                ? Math.round(y) + planeCount * (Math.floor(x) + uCount * Math.floor(z))
                : Math.round(z) + planeCount * (Math.floor(x) + uCount * Math.floor(y));
            id = bins[component].get(key) ?? -1;
          }
          const term = weight * (id < 0 ? 0 : velocity[id]);
          switch (termIndex) {
            case 0: term0 = term; break;
            case 1: term1 = term; break;
            case 2: term2 = term; break;
            case 3: term3 = term; break;
            case 4: term4 = term; break;
            case 5: term5 = term; break;
            case 6: term6 = term; break;
            default: term7 = term;
          }
        }
      }
      const y0 = (term0 + term5) + (term1 + term4);
      const y1 = (term2 + term7) + (term3 + term6);
      return y0 + y1;
  }
}

function makeFaceSampler(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  workspace: SparseAtlasCM12Workspace,
): FaceSampler {
  const sampler = workspace.faceSampler ??= new RetainedFaceSampler();
  sampler.configure(
    grid, velocity, workspace.cellLookup, workspace.faceTopologyWorkspace,
  );
  return sampler;
}

function sampleVelocity(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  position: SparseBrickVec3,
  faceSampler?: FaceSampler,
  result: MutableVec3 = [0, 0, 0],
  sampleRow?: Row,
): MutableVec3 {
  if (faceSampler) {
    result[0] = faceSampler.sample(position, 0);
    result[1] = faceSampler.sample(position, 1);
    result[2] = faceSampler.sample(position, 2);
    return result;
  }
  const weights = sampleWeights(grid, position, undefined, undefined, sampleRow);
  const total = sumRow(weights);
  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  if (total <= 1e-15) return result;
  for (let term = 0; term < weights.count; term += 1) {
    const id = weights.ids[term], weight = weights.coefficients[term];
    for (let axis = 0; axis < 3; axis += 1) {
      result[axis] += weight * velocity[3 * id + axis] / total;
    }
  }
  return result;
}

function clampTracePosition(
  grid: SparseAtlasCompositeGrid,
  value: number,
  axis: 0 | 1 | 2,
): number {
  return Math.max(0.5, Math.min(grid.atlas.dimensions[axis] - 0.5, value));
}

function trace(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  origin: SparseBrickVec3,
  dt_s: number,
  direction: -1 | 1,
  faceSampler?: FaceSampler,
  workspace: SparseAtlasCM12Workspace = createSparseAtlasCM12Workspace(),
): MutableVec3 {
  const initial = sampleVelocity(
    grid, velocity, origin, faceSampler, workspace.traceInitial, workspace.sampleRow,
  );
  const scaledSpeedSquared = dt_s * dt_s
    * (initial[0] * initial[0] + initial[1] * initial[1] + initial[2] * initial[2]);
  let substeps = 1;
  while (substeps < 16 && scaledSpeedSquared > substeps * substeps) substeps += 1;
  const subDt = dt_s / substeps;
  const position = workspace.tracePosition;
  position[0] = origin[0];
  position[1] = origin[1];
  position[2] = origin[2];
  for (let step = 0; step < substeps; step += 1) {
    const first = sampleVelocity(
      grid, velocity, position, faceSampler, workspace.traceFirst, workspace.sampleRow,
    );
    const midpoint = workspace.traceMidpoint;
    midpoint[0] = clampTracePosition(
      grid, position[0] + direction * 0.5 * subDt * first[0], 0);
    midpoint[1] = clampTracePosition(
      grid, position[1] + direction * 0.5 * subDt * first[1], 1);
    midpoint[2] = clampTracePosition(
      grid, position[2] + direction * 0.5 * subDt * first[2], 2);
    const middle = sampleVelocity(
      grid, velocity, midpoint, faceSampler, workspace.traceMiddle, workspace.sampleRow,
    );
    position[0] = clampTracePosition(
      grid, position[0] + direction * subDt * middle[0], 0);
    position[1] = clampTracePosition(
      grid, position[1] + direction * subDt * middle[1], 1);
    position[2] = clampTracePosition(
      grid, position[2] + direction * subDt * middle[2], 2);
  }
  return position;
}

function beta(
  grid: SparseAtlasCompositeGrid,
  rows: readonly Row[],
  result: Float64Array = new Float64Array(grid.cells.length),
): Float64Array {
  result.fill(0);
  for (const receiver of grid.cells) {
    const row = rows[receiver.id];
    for (let index = 0; index < row.count; index += 1) {
      const donorId = row.ids[index];
      result[donorId] += cm12VolumeWeightedBetaContribution(
        receiver.volume, grid.cells[donorId].volume, row.coefficients[index],
      );
    }
  }
  return result;
}

interface CellNeighborLists {
  readonly neighbors: number[][];
  readonly counts: number[];
}

const cellNeighborListsCache = new WeakMap<object, CellNeighborLists>();

export interface SparseAtlasCellExtrapolationWorkspace {
  velocity: Float64Array;
  known: Uint8Array;
  nextVelocity: Float64Array;
  nextKnown: Uint8Array;
  topologyKey?: object;
  readonly neighbors: number[][];
  readonly neighborCounts: number[];
}

export function createSparseAtlasCellExtrapolationWorkspace():
SparseAtlasCellExtrapolationWorkspace {
  return {
    velocity: new Float64Array(0),
    known: new Uint8Array(0),
    nextVelocity: new Float64Array(0),
    nextKnown: new Uint8Array(0),
    neighbors: [],
    neighborCounts: [],
  };
}

function exactUint8(values: Uint8Array, length: number): Uint8Array {
  if (values.length === length) return values;
  const available = values.byteOffset === 0 ? values.buffer.byteLength : 0;
  if (available >= length) return new Uint8Array(values.buffer, 0, length);
  let capacity = 1;
  while (capacity < length) capacity *= 2;
  const grown = new Uint8Array(capacity);
  return length === capacity ? grown : grown.subarray(0, length);
}

function cellNeighborLists(
  grid: SparseAtlasCompositeGrid,
  workspace?: SparseAtlasCellExtrapolationWorkspace,
): CellNeighborLists {
  const key = (grid.topologyKey ?? grid.gradientRows) as object;
  if (workspace?.topologyKey === key) return {
    neighbors: workspace.neighbors, counts: workspace.neighborCounts,
  };
  const cached = cellNeighborListsCache.get(key);
  if (!workspace && cached) return cached;
  const neighbors: number[][] = workspace?.neighbors ?? [];
  const counts: number[] = workspace?.neighborCounts ?? [];
  while (neighbors.length < grid.cells.length) neighbors.push([]);
  for (let id = 0; id < grid.cells.length; id += 1) counts[id] = 0;
  for (const row of grid.gradientRows) {
    for (let leftIndex = 0; leftIndex < row.terms.length; leftIndex += 1) {
      const left = row.terms[leftIndex];
      if (left.coefficient >= 0) continue;
      for (let rightIndex = 0; rightIndex < row.terms.length; rightIndex += 1) {
        const right = row.terms[rightIndex];
        if (right.coefficient <= 0) continue;
        let count = counts[left.cellId];
        neighbors[left.cellId][count] = right.cellId;
        counts[left.cellId] = count + 1;
        count = counts[right.cellId];
        neighbors[right.cellId][count] = left.cellId;
        counts[right.cellId] = count + 1;
      }
    }
  }
  const result = { neighbors, counts };
  if (workspace) workspace.topologyKey = key;
  else cellNeighborListsCache.set(key, result);
  return result;
}

/**
 * Extend collocated liquid velocity through the transient receiver closure.
 * Each sweep reads one frozen front, which keeps the operation D4-equivariant.
 */
export function extrapolateSparseAtlasVelocity(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  input: ArrayLike<number>,
  sweeps = 8,
  workspace: SparseAtlasCellExtrapolationWorkspace =
    createSparseAtlasCellExtrapolationWorkspace(),
): Float64Array {
  const count = grid.cells.length;
  let velocity = workspace.velocity = exactFloat64(workspace.velocity, 3 * count);
  let known = workspace.known = exactUint8(workspace.known, count);
  let next = workspace.nextVelocity = exactFloat64(workspace.nextVelocity, 3 * count);
  let nextKnown = workspace.nextKnown = exactUint8(workspace.nextKnown, count);
  for (let index = 0; index < 3 * count; index += 1) velocity[index] = input[index];
  for (let id = 0; id < count; id += 1) known[id] = density[id] > 1e-5 ? 1 : 0;
  const neighborLists = cellNeighborLists(grid, workspace);
  const neighbors = neighborLists.neighbors;
  const neighborCounts = neighborLists.counts;
  for (let sweep = 0; sweep < sweeps; sweep += 1) {
    next.set(velocity);
    nextKnown.set(known);
    let changed = false;
    for (let id = 0; id < count; id += 1) {
      if (known[id]) continue;
      const adjacent = neighbors[id];
      const adjacentCount = neighborCounts[id];
      let sourceCount = 0;
      for (let index = 0; index < adjacentCount; index += 1) {
        const neighbor = adjacent[index];
        if (!known[neighbor]) continue;
        let duplicate = false;
        for (let prior = 0; prior < index; prior += 1) {
          if (adjacent[prior] === neighbor) { duplicate = true; break; }
        }
        if (!duplicate) sourceCount += 1;
      }
      if (sourceCount === 0) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        let total = 0, correction = 0;
        for (let index = 0; index < adjacentCount; index += 1) {
          const neighbor = adjacent[index];
          if (!known[neighbor]) continue;
          let duplicate = false;
          for (let prior = 0; prior < index; prior += 1) {
            if (adjacent[prior] === neighbor) { duplicate = true; break; }
          }
          if (duplicate) continue;
          const value = velocity[3 * neighbor + axis];
          const adjusted = value - correction;
          const summed = total + adjusted;
          correction = summed - total - adjusted;
          total = summed;
        }
        next[3 * id + axis] = total / sourceCount;
      }
      nextKnown[id] = 1;
      changed = true;
    }
    const previousVelocity = velocity, previousKnown = known;
    velocity = next;
    known = nextKnown;
    next = previousVelocity;
    nextKnown = previousKnown;
    if (!changed) break;
  }
  workspace.velocity = velocity;
  workspace.known = known;
  workspace.nextVelocity = next;
  workspace.nextKnown = nextKnown;
  return velocity;
}

export function transportSparseAtlasCM12(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasCM12TransportFields,
  dt_s: number,
  workspace: SparseAtlasCM12Workspace = createSparseAtlasCM12Workspace(),
): SparseAtlasCM12TransportResult {
  const count = grid.cells.length;
  if (fields.density.length !== count || fields.gamma.length !== count
    || fields.velocity.length !== 3 * count) {
    throw new RangeError("CM12 transport fields do not match the sparse composite grid");
  }
  const topologyKey = (grid.topologyKey ?? grid.gradientRows) as object;
  if (workspace.topologyKey !== topologyKey) {
    compactFineCellLookup(grid, workspace.cellLookup);
    workspace.topologyKey = topologyKey;
  }
  const cellLookup = workspace.cellLookup;
  const faceSampler = fields.faceNormalVelocity
    ? makeFaceSampler(grid, fields.faceNormalVelocity, workspace) : undefined;
  const topologySpans = uniformSamplingSpans(grid);
  const backward = rowsFor(workspace.backward, count);
  for (const receiver of grid.cells) sampleWeights(
    grid, trace(
      grid, fields.velocity, receiver.centerFine, dt_s, -1, faceSampler, workspace,
    ),
    cellLookup, topologySpans, backward[receiver.id],
  );
  const advectedGamma = workspace.advectedGamma = exactFloat64(workspace.advectedGamma, count);
  for (let receiverId = 0; receiverId < count; receiverId += 1) {
    const weights = backward[receiverId];
    const visible = sumRow(weights);
    let sampled = 0, correction = 0;
    for (let index = 0; index < weights.count; index += 1) {
      const value = weights.coefficients[index] * fields.gamma[weights.ids[index]];
      const adjusted = value - correction;
      const next = sampled + adjusted;
      correction = next - sampled - adjusted;
      sampled = next;
    }
    advectedGamma[receiverId] = cm12ConditionedGamma(sampled, visible);
  }
  const unconditioned = rowsFor(workspace.unconditioned, count);
  for (let receiverId = 0; receiverId < count; receiverId += 1) {
    const weights = backward[receiverId];
    const row = unconditioned[receiverId];
    const visible = sumRow(weights);
    if (visible > 1e-15) for (let index = 0; index < weights.count; index += 1) {
      add(row, weights.ids[index],
        advectedGamma[receiverId] * weights.coefficients[index] / visible);
    }
  }
  const backwardBeta = workspace.backwardBeta = exactFloat64(workspace.backwardBeta, count);
  beta(grid, unconditioned, backwardBeta);
  // A receiver starts with at most eight backward donors. With the per-substep
  // CFL bound, forward deficit redistribution can reach at most the surrounding
  // 4 x 4 x 4 donor centres. Reserve that geometric bound up front so changing
  // flow does not grow transport rows in otherwise steady topology.
  const rows = rowsFor(workspace.rows, count, 64);
  for (let receiverId = 0; receiverId < count; receiverId += 1) {
    const weights = backward[receiverId];
    const row = rows[receiverId];
    const visible = sumRow(weights);
    if (visible > 1e-15) for (let index = 0; index < weights.count; index += 1) {
      const id = weights.ids[index];
      add(row, id, cm12ConditionedRowCoefficient(
        advectedGamma[receiverId], weights.coefficients[index] / visible, backwardBeta[id],
      ));
    }
  }
  const nextGamma = workspace.nextGamma = exactFloat64(workspace.nextGamma, count);
  for (let receiverId = 0; receiverId < count; receiverId += 1) {
    nextGamma[receiverId] = sumRow(rows[receiverId]);
  }
  const conditionedBeta = workspace.conditionedBeta = exactFloat64(
    workspace.conditionedBeta, count,
  );
  beta(grid, rows, conditionedBeta);
  for (const donor of grid.cells) {
    const deficit = Math.max(0, 1 - conditionedBeta[donor.id]);
    if (deficit <= 1e-15) continue;
    const forward = sampleWeights(
      grid, trace(
        grid, fields.velocity, donor.centerFine, dt_s, 1, faceSampler, workspace,
      ),
      cellLookup, topologySpans, workspace.forward,
    );
    const total = sumRow(forward);
    if (total <= 1e-15) continue;
    for (let index = 0; index < forward.count; index += 1) {
      const receiverId = forward.ids[index];
      const weight = forward.coefficients[index];
      const coefficient = cm12VolumeScaledDeficitCoefficient(
        donor.volume, grid.cells[receiverId].volume, deficit, weight / total,
      );
      add(rows[receiverId], donor.id, coefficient);
      nextGamma[receiverId] += coefficient * fields.gamma[donor.id];
    }
  }
  const finalBeta = workspace.finalBeta = exactFloat64(workspace.finalBeta, count);
  beta(grid, rows, finalBeta);
  const density = workspace.density = exactFloat64(workspace.density, count);
  density.fill(0);
  for (const receiver of grid.cells) {
    const row = rows[receiver.id];
    for (let index = 0; index < row.count; index += 1) {
      density[receiver.id] += row.coefficients[index] * fields.density[row.ids[index]];
    }
  }
  // Uniform CM12 initializes gamma to one throughout air and restores that
  // value whenever transport leaves a cell numerically dry. Sparse receiver
  // tiles may retire after this step, so carrying conditioned gamma through
  // dry cells would make reactivation depend on residency history.
  for (let id = 0; id < count; id += 1) {
    if (density[id] < 1e-5) nextGamma[id] = 1;
  }
  const faceNormalVelocity = workspace.faceNormalVelocity = exactFloat64(
    workspace.faceNormalVelocity, grid.gradientRows.length,
  );
  if (faceSampler) {
    for (const row of grid.gradientRows) {
      const departure = trace(
        grid, fields.velocity, row.centerFine, dt_s, -1, faceSampler, workspace,
      );
      faceNormalVelocity[row.id] = faceSampler.sample(departure, row.axis);
    }
  } else {
    faceNormalVelocity.fill(0);
  }
  const velocity = workspace.velocity = exactFloat64(workspace.velocity, 3 * count);
  if (faceSampler) {
    workspace.collocationWeights = exactFloat64(workspace.collocationWeights, 3 * count);
    collocateSparseAtlasVelocity(
      grid, faceNormalVelocity, velocity, workspace.collocationWeights,
    );
  } else {
    for (let index = 0; index < 3 * count; index += 1) {
      const receiverId = Math.floor(index / 3);
      const component = index % 3;
      const weights = backward[receiverId];
      const total = sumRow(weights);
      if (total <= 1e-15) {
        velocity[index] = 0;
        continue;
      }
      let sampled = 0, correction = 0;
      for (let term = 0; term < weights.count; term += 1) {
        const value = weights.coefficients[term]
          * fields.velocity[3 * weights.ids[term] + component];
        const adjusted = value - correction;
        const next = sampled + adjusted;
        correction = next - sampled - adjusted;
        sampled = next;
      }
      velocity[index] = sampled / total;
    }
  }
  let finalBetaMaximumAbsoluteError = 0;
  for (const value of finalBeta) {
    finalBetaMaximumAbsoluteError = Math.max(finalBetaMaximumAbsoluteError,
      Math.abs(value - 1));
  }
  let outputFields = workspace.outputFields;
  if (!outputFields) {
    outputFields = workspace.outputFields = {
      density, gamma: nextGamma, velocity, faceNormalVelocity,
    };
  } else {
    const mutable = outputFields as {
      density: Float64Array;
      gamma: Float64Array;
      velocity: Float64Array;
      faceNormalVelocity?: Float64Array;
    };
    mutable.density = density;
    mutable.gamma = nextGamma;
    mutable.velocity = velocity;
    mutable.faceNormalVelocity = faceNormalVelocity;
  }
  let result = workspace.result;
  if (!result) {
    result = workspace.result = {
      fields: outputFields, faceNormalVelocity, finalBetaMaximumAbsoluteError,
    };
  } else {
    const mutable = result as {
      fields: SparseAtlasCM12TransportFields;
      faceNormalVelocity: Float64Array;
      finalBetaMaximumAbsoluteError: number;
    };
    mutable.fields = outputFields;
    mutable.faceNormalVelocity = faceNormalVelocity;
    mutable.finalBetaMaximumAbsoluteError = finalBetaMaximumAbsoluteError;
  }
  return result;
}
