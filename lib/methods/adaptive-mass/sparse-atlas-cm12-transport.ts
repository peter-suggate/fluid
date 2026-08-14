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
import { sparseBrickKey, type SparseBrickVec3 } from "./sparse-brick-atlas";
import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { collocateSparseAtlasVelocity } from "./sparse-atlas-composite-projection";

type Row = Map<number, number>;

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
  if (value !== 0) row.set(id, (row.get(id) ?? 0) + value);
}

function sum(values: Iterable<number>): number {
  let result = 0, correction = 0;
  for (const value of values) {
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

const compactFineCellLookupCache = new WeakMap<object, ReadonlyMap<number, number>>();
const uniformSamplingSpansCache = new WeakMap<object, readonly [number, number, number] | null>();

function compactFineCellLookup(grid: SparseAtlasCompositeGrid): ReadonlyMap<number, number> {
  const key = grid.gradientRows as object;
  const cached = compactFineCellLookupCache.get(key);
  if (cached) return cached;
  const result = new Map<number, number>();
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1) {
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1) {
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          result.set(fineCellKey(grid, [x, y, z]), cell.id);
        }
      }
    }
  }
  compactFineCellLookupCache.set(key, result);
  return result;
}

function uniformSamplingSpans(
  grid: SparseAtlasCompositeGrid,
): readonly [number, number, number] | null {
  const topologyKey = grid.gradientRows as object;
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
  cellLookup: ReadonlyMap<number, number>,
): readonly [number, number, number] {
  const uniform = uniformSamplingSpans(grid);
  if (uniform) return uniform;
  const coordinate = position.map((value, axis) => Math.max(0, Math.min(
    grid.atlas.dimensions[axis] - 1,
    Math.floor(value),
  ))) as [number, number, number];
  const id = cellLookup.get(fineCellKey(grid, coordinate));
  if (id === undefined) return [1, 1, 1];
  return grid.cells[id].widthsFine;
}

/** Cell-centred nonnegative trilinear weights, collapsed onto compact leaves. */
function sampleWeights(
  grid: SparseAtlasCompositeGrid,
  position: SparseBrickVec3,
  cellLookup?: ReadonlyMap<number, number>,
  topologySpans?: readonly [number, number, number] | null,
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
  const row: Row = new Map();
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
      const id = lookup.get(fineKey) ?? (() => {
        const brickCoordinate = [
          Math.floor(fine0 / 8), Math.floor(fine1 / 8), Math.floor(fine2 / 8),
        ] as const;
        const key = sparseBrickKey(brickCoordinate, grid.atlas.brickDimensions);
        const brick = grid.atlas.directory.get(key);
        const cellBase = grid.cellBaseByBrick.get(key);
        if (!brick || cellBase === undefined) return undefined;
        const scale = 8 / brick.resolution;
        const local0 = Math.floor((fine0 - 8 * brickCoordinate[0]) / scale);
        const local1 = Math.floor((fine1 - 8 * brickCoordinate[1]) / scale);
        const local2 = Math.floor((fine2 - 8 * brickCoordinate[2]) / scale);
        return cellBase + local0 + brick.resolution
          * (local1 + brick.resolution * local2);
      })();
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

type FaceBins = [Map<number, number>, Map<number, number>, Map<number, number>];

interface FaceTopology {
  readonly bins: FaceBins;
  /** Negative then positive neighbor for x, y and z. */
  readonly neighbors: readonly Int32Array[];
  readonly spacing: readonly Float64Array[];
}

const FACE_TANGENTS = [[1, 2], [0, 2], [0, 1]] as const;

const faceTangents = (axis: 0 | 1 | 2): readonly [0 | 1 | 2, 0 | 1 | 2] =>
  FACE_TANGENTS[axis];

function buildFaceBins(grid: SparseAtlasCompositeGrid): FaceBins {
  const bins: FaceBins = [new Map(), new Map(), new Map()];
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

function faceTopology(grid: SparseAtlasCompositeGrid): FaceTopology {
  const key = grid.gradientRows as object;
  const cached = faceTopologyCache.get(key);
  if (cached) return cached;
  const bins = buildFaceBins(grid);
  const neighbors = Array.from({ length: 6 }, () =>
    new Int32Array(grid.gradientRows.length).fill(-1));
  const spacing = Array.from({ length: 6 }, () =>
    new Float64Array(grid.gradientRows.length));
  for (const row of grid.gradientRows) for (const axis of [0, 1, 2] as const) {
    for (const sign of [-1, 1] as const) {
      const slot = 2 * axis + (sign > 0 ? 1 : 0);
      const coordinate = [...row.centerFine];
      // A compact coarse face occupies multiple finest-lattice bins. Walk
      // through those aliases until the first distinct face is reached.
      for (let offset = 1; offset <= 4; offset += 1) {
        coordinate[axis] = row.centerFine[axis] + sign * offset;
        const candidate = faceBinAt(grid, bins, row.axis, coordinate);
        if (candidate === undefined || candidate === row.id) continue;
        neighbors[slot][row.id] = candidate;
        spacing[slot][row.id] = Math.abs(
          grid.gradientRows[candidate].centerFine[axis] - row.centerFine[axis],
        );
        break;
      }
    }
  }
  const topology = { bins, neighbors, spacing } satisfies FaceTopology;
  faceTopologyCache.set(key, topology);
  return topology;
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
): Float64Array {
  const count = grid.gradientRows.length;
  if (input.length !== count || fallback.length !== count) {
    throw new RangeError("face extrapolation fields do not match the sparse grid");
  }
  const topology = faceTopology(grid);
  let values = new Float64Array(count);
  let distance = new Float64Array(count).fill(Number.POSITIVE_INFINITY);
  for (const row of grid.gradientRows) {
    if (row.terms.some((term) => density[term.cellId] > 0.5)) {
      values[row.id] = input[row.id];
      distance[row.id] = 0;
    }
  }
  const distanceAt = (slot: number, rowId: number): number => {
    const neighbor = topology.neighbors[slot][rowId];
    return neighbor >= 0 ? distance[neighbor] : Number.POSITIVE_INFINITY;
  };
  const maximumSweeps = Math.max(16, Math.ceil(bandCells) + 2);
  for (let sweep = 0; sweep < maximumSweeps; sweep += 1) {
    const nextValues = values.slice();
    const nextDistance = distance.slice();
    let changed = false;
    for (const row of grid.gradientRows) {
      if (distance[row.id] === 0) continue;
      const axes: Array<{
        axis: number;
        distance: number;
        spacing: number;
        slots: readonly [number, number];
      }> = [];
      for (const axis of [0, 1, 2] as const) {
        const negativeSlot = 2 * axis, positiveSlot = negativeSlot + 1;
        const negativeDistance = distanceAt(negativeSlot, row.id);
        const positiveDistance = distanceAt(positiveSlot, row.id);
        const slot = negativeDistance <= positiveDistance ? negativeSlot : positiveSlot;
        const entry = {
          axis,
          distance: Math.min(negativeDistance, positiveDistance),
          spacing: topology.spacing[slot][row.id],
          slots: [negativeSlot, positiveSlot] as const,
        };
        if (Number.isFinite(entry.distance) && entry.spacing > 0) axes.push(entry);
      }
      axes.sort((left, right) => left.distance - right.distance);
      if (axes.length === 0) continue;
      let solved = axes[0].distance + axes[0].spacing;
      for (let used = 2; used <= axes.length; used += 1) {
        if (solved <= axes[used - 1].distance) break;
        let a = 0, b = 0, c = -1;
        for (let index = 0; index < used; index += 1) {
          const inverseSpacingSquared = 1 / (axes[index].spacing ** 2);
          a += inverseSpacingSquared;
          b += axes[index].distance * inverseSpacingSquared;
          c += axes[index].distance ** 2 * inverseSpacingSquared;
        }
        solved = (b + Math.sqrt(Math.max(0, b * b - a * c))) / a;
      }
      solved = Math.min(distance[row.id], solved);
      if (solved > bandCells + 1e-12) continue;
      let weightedTerm0 = 0, weightedTerm1 = 0, weightedTerm2 = 0;
      let weightTerm0 = 0, weightTerm1 = 0, weightTerm2 = 0;
      // Match the dense f32 FIM convergence/tie tolerance.
      const epsilon = Math.max(2, Math.abs(solved)) * 1.1920929e-7;
      for (const axis of axes) {
        if (axis.distance >= solved - epsilon) continue;
        let minimizerCount = 0, value = 0;
        for (const slot of axis.slots) {
          const id = topology.neighbors[slot][row.id];
          const minimizerDistance = distanceAt(slot, row.id);
          if (id < 0 || Math.abs(minimizerDistance - axis.distance) > epsilon) continue;
          value += values[id];
          minimizerCount += 1;
        }
        if (minimizerCount === 0) continue;
        const weight = (solved - axis.distance) / (axis.spacing ** 2);
        const weighted = weight * value / minimizerCount;
        if (axis.axis === 0) {
          weightedTerm0 = weighted;
          weightTerm0 = weight;
        } else if (axis.axis === 1) {
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
    values = nextValues;
    distance = nextDistance;
    if (!changed) break;
  }
  return Float64Array.from(values, (value, rowId) =>
    Number.isFinite(distance[rowId]) ? value : fallback[rowId]);
}

function makeFaceSampler(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
): FaceSampler {
  const bins = faceTopology(grid).bins;
  const cellLookup = compactFineCellLookup(grid);
  const topologySpans = uniformSamplingSpans(grid);
  const stencilIds: readonly Map<number, Int32Array>[] = [
    new Map(), new Map(), new Map(),
  ];
  return {
    sample: (position, component) => {
      const spans = topologySpans ?? samplingSpans(grid, position, cellLookup);
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
      let ids = topologySpans ? stencilIds[component].get(stencilKey) : undefined;
      if (topologySpans && ids === undefined) {
        ids = new Int32Array(8).fill(-1);
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
            ids[dx + 2 * dy + 4 * dz] = bins[component].get(key) ?? -1;
          }
        }
        stencilIds[component].set(stencilKey, ids);
      }
      let term0 = 0, term1 = 0, term2 = 0, term3 = 0;
      let term4 = 0, term5 = 0, term6 = 0, term7 = 0;
      for (let dz = 0; dz <= 1; dz += 1) for (let dy = 0; dy <= 1; dy += 1) {
        for (let dx = 0; dx <= 1; dx += 1) {
          const weight = (dx ? fraction0 : 1 - fraction0)
            * (dy ? fraction1 : 1 - fraction1)
            * (dz ? fraction2 : 1 - fraction2);
          const termIndex = dx + 2 * dy + 4 * dz;
          let id = ids?.[termIndex];
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
    },
  };
}

function sampleVelocity(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  position: SparseBrickVec3,
  faceSampler?: FaceSampler,
): [number, number, number] {
  if (faceSampler) return [
    faceSampler.sample(position, 0),
    faceSampler.sample(position, 1),
    faceSampler.sample(position, 2),
  ];
  const weights = sampleWeights(grid, position);
  const total = sum(weights.values());
  const result: [number, number, number] = [0, 0, 0];
  if (total <= 1e-15) return result;
  for (const [id, weight] of weights) for (let axis = 0; axis < 3; axis += 1) {
    result[axis] += weight * velocity[3 * id + axis] / total;
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
): [number, number, number] {
  const initial = sampleVelocity(grid, velocity, origin, faceSampler);
  const substeps = Math.max(1, Math.min(16, Math.ceil(
    dt_s * Math.hypot(initial[0], initial[1], initial[2]),
  )));
  const subDt = dt_s / substeps;
  let position = [...origin] as [number, number, number];
  for (let step = 0; step < substeps; step += 1) {
    const first = sampleVelocity(grid, velocity, position, faceSampler);
    const midpoint: [number, number, number] = [
      clampTracePosition(grid, position[0] + direction * 0.5 * subDt * first[0], 0),
      clampTracePosition(grid, position[1] + direction * 0.5 * subDt * first[1], 1),
      clampTracePosition(grid, position[2] + direction * 0.5 * subDt * first[2], 2),
    ];
    const middle = sampleVelocity(grid, velocity, midpoint, faceSampler);
    position = [
      clampTracePosition(grid, position[0] + direction * subDt * middle[0], 0),
      clampTracePosition(grid, position[1] + direction * subDt * middle[1], 1),
      clampTracePosition(grid, position[2] + direction * subDt * middle[2], 2),
    ];
  }
  return position;
}

function beta(grid: SparseAtlasCompositeGrid, rows: readonly Row[]): Float64Array {
  const result = new Float64Array(grid.cells.length);
  for (const receiver of grid.cells) for (const [donorId, coefficient] of rows[receiver.id]) {
    result[donorId] += cm12VolumeWeightedBetaContribution(
      receiver.volume, grid.cells[donorId].volume, coefficient,
    );
  }
  return result;
}

const cellNeighborListsCache = new WeakMap<object, readonly number[][]>();

function cellNeighborLists(grid: SparseAtlasCompositeGrid): readonly number[][] {
  const key = grid.gradientRows as object;
  const cached = cellNeighborListsCache.get(key);
  if (cached) return cached;
  const neighbors: number[][] = Array.from({ length: grid.cells.length }, () => []);
  for (const row of grid.gradientRows) {
    const negative = row.terms.filter((term) => term.coefficient < 0);
    const positive = row.terms.filter((term) => term.coefficient > 0);
    for (const left of negative) for (const right of positive) {
      neighbors[left.cellId].push(right.cellId);
      neighbors[right.cellId].push(left.cellId);
    }
  }
  cellNeighborListsCache.set(key, neighbors);
  return neighbors;
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
): Float64Array {
  const count = grid.cells.length;
  let velocity = Float64Array.from(input);
  let known = Uint8Array.from({ length: count }, (_, id) => density[id] > 1e-5 ? 1 : 0);
  const neighbors = cellNeighborLists(grid);
  for (let sweep = 0; sweep < sweeps; sweep += 1) {
    const next = velocity.slice();
    const nextKnown = known.slice();
    let changed = false;
    for (let id = 0; id < count; id += 1) {
      if (known[id]) continue;
      const sources = [...new Set(neighbors[id])].filter((neighbor) => known[neighbor]);
      if (sources.length === 0) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        next[3 * id + axis] = sum(sources.map((source) =>
          velocity[3 * source + axis])) / sources.length;
      }
      nextKnown[id] = 1;
      changed = true;
    }
    velocity = next;
    known = nextKnown;
    if (!changed) break;
  }
  return velocity;
}

export function transportSparseAtlasCM12(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasCM12TransportFields,
  dt_s: number,
): SparseAtlasCM12TransportResult {
  const count = grid.cells.length;
  if (fields.density.length !== count || fields.gamma.length !== count
    || fields.velocity.length !== 3 * count) {
    throw new RangeError("CM12 transport fields do not match the sparse composite grid");
  }
  const faceSampler = fields.faceNormalVelocity
    ? makeFaceSampler(grid, fields.faceNormalVelocity) : undefined;
  const cellLookup = compactFineCellLookup(grid);
  const topologySpans = uniformSamplingSpans(grid);
  const backward: Row[] = grid.cells.map((receiver) => sampleWeights(
    grid, trace(grid, fields.velocity, receiver.centerFine, dt_s, -1, faceSampler),
    cellLookup, topologySpans,
  ));
  const advectedGamma = Float64Array.from(backward, (weights) => {
    const visible = sum(weights.values());
    const sampled = sum([...weights].map(([id, weight]) => weight * fields.gamma[id]));
    return cm12ConditionedGamma(sampled, visible);
  });
  const unconditioned = backward.map((weights, receiverId) => {
    const row: Row = new Map();
    const visible = sum(weights.values());
    if (visible > 1e-15) for (const [id, weight] of weights) {
      add(row, id, advectedGamma[receiverId] * weight / visible);
    }
    return row;
  });
  const backwardBeta = beta(grid, unconditioned);
  const rows = backward.map((weights, receiverId) => {
    const row: Row = new Map();
    const visible = sum(weights.values());
    if (visible > 1e-15) for (const [id, weight] of weights) add(row, id,
      cm12ConditionedRowCoefficient(
        advectedGamma[receiverId], weight / visible, backwardBeta[id],
      ));
    return row;
  });
  const nextGamma = Float64Array.from(rows, (row) => sum(row.values()));
  const conditionedBeta = beta(grid, rows);
  for (const donor of grid.cells) {
    const deficit = Math.max(0, 1 - conditionedBeta[donor.id]);
    if (deficit <= 1e-15) continue;
    const forward = sampleWeights(
      grid, trace(grid, fields.velocity, donor.centerFine, dt_s, 1, faceSampler),
      cellLookup, topologySpans,
    );
    const total = sum(forward.values());
    if (total <= 1e-15) continue;
    for (const [receiverId, weight] of forward) {
      const coefficient = cm12VolumeScaledDeficitCoefficient(
        donor.volume, grid.cells[receiverId].volume, deficit, weight / total,
      );
      add(rows[receiverId], donor.id, coefficient);
      nextGamma[receiverId] += coefficient * fields.gamma[donor.id];
    }
  }
  const finalBeta = beta(grid, rows);
  const density = new Float64Array(count);
  for (const receiver of grid.cells) for (const [donorId, coefficient] of rows[receiver.id]) {
    density[receiver.id] += coefficient * fields.density[donorId];
  }
  // Uniform CM12 initializes gamma to one throughout air and restores that
  // value whenever transport leaves a cell numerically dry. Sparse receiver
  // tiles may retire after this step, so carrying conditioned gamma through
  // dry cells would make reactivation depend on residency history.
  for (let id = 0; id < count; id += 1) {
    if (density[id] < 1e-5) nextGamma[id] = 1;
  }
  const faceNormalVelocity = faceSampler
    ? Float64Array.from(grid.gradientRows, (row) => {
      const departure = trace(
        grid, fields.velocity, row.centerFine, dt_s, -1, faceSampler,
      );
      return faceSampler.sample(departure, row.axis);
    })
    : new Float64Array(grid.gradientRows.length);
  const velocity = faceSampler
    ? collocateSparseAtlasVelocity(grid, faceNormalVelocity)
    : Float64Array.from({ length: 3 * count }, (_, index) => {
      const receiverId = Math.floor(index / 3);
      const component = index % 3;
      const weights = backward[receiverId];
      const total = sum(weights.values());
      if (total <= 1e-15) return 0;
      return sum([...weights].map(([donorId, weight]) =>
        weight * fields.velocity[3 * donorId + component])) / total;
    });
  let finalBetaMaximumAbsoluteError = 0;
  for (const value of finalBeta) {
    finalBetaMaximumAbsoluteError = Math.max(finalBetaMaximumAbsoluteError,
      Math.abs(value - 1));
  }
  return {
    fields: { density, gamma: nextGamma, velocity, faceNormalVelocity },
    faceNormalVelocity,
    finalBetaMaximumAbsoluteError,
  };
}
