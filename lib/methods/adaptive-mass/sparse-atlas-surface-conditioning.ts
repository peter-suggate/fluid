/**
 * Conservative CM12 surface conditioning on an arbitrary resident brick atlas.
 *
 * Every edge comes from the composite projection rows, so work and storage are
 * proportional to resident cells/faces. Omitted bricks remain omitted: this
 * module never materializes or scans the logical finest lattice.
 */

import {
  cm12GammaDiffusionFluxInto,
  cm12SharpeningWeight,
  type Cm12GammaDiffusionFlux,
} from "../../core/cm12-numerics";
import type {
  SparseAtlasAxis,
  SparseAtlasCompositeGrid,
} from "./sparse-atlas-composite-projection";

export interface SparseAtlasSurfaceFields {
  readonly density: Float64Array;
  readonly gamma: Float64Array;
}

export interface SparseAtlasSurfaceConditioningOptions {
  readonly gammaDiffusionIterations?: number;
  /** Fraction of one paper-step gamma diffusion dose. */
  readonly gammaDiffusionScale?: number;
  /** Physical step used by CM12 Sec. 3.5's 3dt pseudo-time dose. */
  readonly timeStep_s?: number;
  /** Metres per finest-cell coordinate used by the composite grid. */
  readonly finestCellSize_m?: number;
  /** Explicit sharpening pseudo-time in finest-cell coordinates for probes. */
  readonly sharpeningCourant?: number;
  readonly sharpeningDistanceCells?: number;
  /** Retain an already-proven horizontal D4 invariant across topology rebuilds. */
  readonly preserveHorizontalD4?: boolean;
}

export interface SparseAtlasSurfaceConditioningResult {
  readonly fields: SparseAtlasSurfaceFields;
  readonly edgeCount: number;
  readonly gammaPairUpdates: number;
  readonly removedIntegratedMass: number;
  readonly returnedIntegratedMass: number;
  readonly fallbackIntegratedMass: number;
  readonly massAbsoluteError: number;
  readonly gammaIntegralAbsoluteError: number;
}

export interface SparseAtlasSurfaceConditioningWorkspace {
  density: Float64Array;
  gamma: Float64Array;
  forwardDensity0: Float64Array;
  forwardDensity1: Float64Array;
  forwardGamma0: Float64Array;
  forwardGamma1: Float64Array;
  reverseDensity0: Float64Array;
  reverseDensity1: Float64Array;
  reverseGamma0: Float64Array;
  reverseGamma1: Float64Array;
  deltas: Float64Array;
  conditionedDensity: Float64Array;
  candidateIds: Int32Array;
  candidateWeights: Float64Array;
  active: Int32Array;
  available: Int32Array;
  tracePosition: Float64Array;
  traceGradient: Float64Array;
  readonly diffusionFlux: Cm12GammaDiffusionFlux;
  readonly averageBefore: { value: number; distance: number };
  readonly averageAfter: { value: number; distance: number };
  outputFields?: SparseAtlasSurfaceFields;
  result?: SparseAtlasSurfaceConditioningResult;
  topologyKey?: object;
  topology?: SurfaceTopology;
  readonly edgePool: ScalarEdge[];
  readonly neighborPool: Neighbor[];
}

export function createSparseAtlasSurfaceConditioningWorkspace():
SparseAtlasSurfaceConditioningWorkspace {
  return {
    density: new Float64Array(0), gamma: new Float64Array(0),
    forwardDensity0: new Float64Array(0), forwardDensity1: new Float64Array(0),
    forwardGamma0: new Float64Array(0), forwardGamma1: new Float64Array(0),
    reverseDensity0: new Float64Array(0), reverseDensity1: new Float64Array(0),
    reverseGamma0: new Float64Array(0), reverseGamma1: new Float64Array(0),
    deltas: new Float64Array(0), conditionedDensity: new Float64Array(0),
    candidateIds: new Int32Array(8), candidateWeights: new Float64Array(8),
    active: new Int32Array(8), available: new Int32Array(8),
    tracePosition: new Float64Array(3), traceGradient: new Float64Array(3),
    diffusionFlux: { rho: 0, gamma: 0 },
    averageBefore: { value: 0, distance: 1 },
    averageAfter: { value: 0, distance: 1 },
    edgePool: [], neighborPool: [],
  };
}

function ensureSurfaceVectorLength(
  workspace: SparseAtlasSurfaceConditioningWorkspace,
  length: number,
): void {
  if (workspace.density.length === length) return;
  workspace.density = new Float64Array(length);
  workspace.gamma = new Float64Array(length);
  workspace.forwardDensity0 = new Float64Array(length);
  workspace.forwardDensity1 = new Float64Array(length);
  workspace.forwardGamma0 = new Float64Array(length);
  workspace.forwardGamma1 = new Float64Array(length);
  workspace.reverseDensity0 = new Float64Array(length);
  workspace.reverseDensity1 = new Float64Array(length);
  workspace.reverseGamma0 = new Float64Array(length);
  workspace.reverseGamma1 = new Float64Array(length);
  workspace.deltas = new Float64Array(length);
  workspace.conditionedDensity = new Float64Array(length);
}

function ensureCandidateCapacity(
  workspace: SparseAtlasSurfaceConditioningWorkspace,
  count: number,
): void {
  if (workspace.candidateIds.length >= count) return;
  let capacity = workspace.candidateIds.length;
  while (capacity < count) capacity *= 2;
  workspace.candidateIds = new Int32Array(capacity);
  workspace.candidateWeights = new Float64Array(capacity);
  workspace.active = new Int32Array(capacity);
  workspace.available = new Int32Array(capacity);
}

interface ScalarEdge {
  axis: SparseAtlasAxis;
  negativeCellId: number;
  positiveCellId: number;
  area: number;
  distance: number;
}

interface Neighbor {
  cellId: number;
  area: number;
  distance: number;
}

interface DirectionalAdjacency {
  readonly negative: Neighbor[][][];
  readonly positive: Neighbor[][][];
  readonly all: Neighbor[][];
  readonly negativeCount: number[][];
  readonly positiveCount: number[][];
  readonly allCount: number[];
}

function integratedScalar(
  grid: SparseAtlasCompositeGrid,
  values: ArrayLike<number>,
): number {
  let sum = 0, correction = 0;
  for (const cell of grid.cells) {
    const adjusted = cell.volume * values[cell.id] - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

function buildD4SymmetryOrbits(
  grid: SparseAtlasCompositeGrid,
): readonly (readonly number[])[] | undefined {
  const [nx, , nz] = grid.atlas.dimensions;
  if (nx !== nz) return undefined;
  const key = (
    minimum: readonly number[],
    maximum: readonly number[],
  ): string => `${minimum.join(",")}:${maximum.join(",")}`;
  const byExtent = new Map(grid.cells.map((cell) => [
    key(cell.minimumFine, cell.maximumFine), cell.id,
  ]));
  const orbitFor = (cellId: number): number[] | undefined => {
    const cell = grid.cells[cellId];
    const [x0, y0, z0] = cell.minimumFine;
    const [x1, y1, z1] = cell.maximumFine;
    const xr: readonly [number, number] = [nx - x1, nx - x0];
    const zr: readonly [number, number] = [nz - z1, nz - z0];
    const variants = [
      [x0, x1, z0, z1], [xr[0], xr[1], z0, z1],
      [x0, x1, zr[0], zr[1]], [xr[0], xr[1], zr[0], zr[1]],
      [z0, z1, x0, x1], [zr[0], zr[1], x0, x1],
      [z0, z1, xr[0], xr[1]], [zr[0], zr[1], xr[0], xr[1]],
    ] as const;
    const ids = new Set<number>();
    for (const [a0, a1, b0, b1] of variants) {
      const id = byExtent.get(key([a0, y0, b0], [a1, y1, b1]));
      if (id === undefined) return undefined;
      ids.add(id);
    }
    return Array.from(ids).sort((left, right) => left - right);
  };
  const orbits: number[][] = [], visited = new Set<number>();
  for (const cell of grid.cells) {
    if (visited.has(cell.id)) continue;
    const orbit = orbitFor(cell.id);
    if (!orbit) return undefined;
    orbit.forEach((id) => visited.add(id));
    orbits.push(orbit);
  }
  return orbits;
}

export function sparseAtlasHasHorizontalD4Topology(
  grid: SparseAtlasCompositeGrid,
): boolean {
  return buildD4SymmetryOrbits(grid) !== undefined;
}

export function sparseAtlasScalarsHaveHorizontalD4Symmetry(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  gamma: ArrayLike<number>,
): boolean {
  if (density.length !== grid.cells.length || gamma.length !== grid.cells.length) {
    return false;
  }
  return activeD4SymmetryOrbits(buildD4SymmetryOrbits(grid), {
    density: density as Float64Array,
    gamma: gamma as Float64Array,
  }) !== undefined;
}

function activeD4SymmetryOrbits(
  candidate: readonly (readonly number[])[] | undefined,
  fields: SparseAtlasSurfaceFields,
): readonly (readonly number[])[] | undefined {
  if (!candidate) return undefined;
  for (const orbit of candidate) {
    const rho0 = fields.density[orbit[0]], gamma0 = fields.gamma[orbit[0]];
    if (orbit.some((id) => Math.abs(fields.density[id] - rho0) > 1e-10
      || Math.abs(fields.gamma[id] - gamma0) > 1e-10)) return undefined;
  }
  return candidate;
}

function preserveD4Symmetry(
  orbits: readonly (readonly number[])[] | undefined,
  density: Float64Array,
  gamma: Float64Array,
): void {
  if (!orbits) return;
  for (const orbit of orbits) {
    let densitySum = 0, densityCorrection = 0;
    let gammaSum = 0, gammaCorrection = 0;
    for (const id of orbit) {
      const densityAdjusted = density[id] - densityCorrection;
      const densityNext = densitySum + densityAdjusted;
      densityCorrection = densityNext - densitySum - densityAdjusted;
      densitySum = densityNext;
      const gammaAdjusted = gamma[id] - gammaCorrection;
      const gammaNext = gammaSum + gammaAdjusted;
      gammaCorrection = gammaNext - gammaSum - gammaAdjusted;
      gammaSum = gammaNext;
    }
    const meanDensity = densitySum / orbit.length;
    const meanGamma = gammaSum / orbit.length;
    for (const id of orbit) {
      density[id] = meanDensity;
      gamma[id] = meanGamma;
    }
  }
}

/** Expand aggregate mixed-resolution ports into physical scalar subfaces. */
function scalarEdges(
  grid: SparseAtlasCompositeGrid,
  edges: ScalarEdge[] = [],
  pool: ScalarEdge[] = [],
): ScalarEdge[] {
  let edgeCount = 0;
  for (const row of grid.gradientRows) {
    let negativeCount = 0, positiveCount = 0;
    for (let index = 0; index < row.terms.length; index += 1) {
      if (row.terms[index].coefficient < 0) negativeCount += 1;
      else if (row.terms[index].coefficient > 0) positiveCount += 1;
    }
    // A one-sided sparse-air row is a pressure boundary, not a resident scalar
    // diffusion/sharpening edge. Receiver activation owns material entering it.
    if (negativeCount === 0 || positiveCount === 0) continue;
    const subfaceArea = row.area / (negativeCount * positiveCount);
    for (let leftIndex = 0; leftIndex < row.terms.length; leftIndex += 1) {
      const left = row.terms[leftIndex];
      if (left.coefficient >= 0) continue;
      for (let rightIndex = 0; rightIndex < row.terms.length; rightIndex += 1) {
        const right = row.terms[rightIndex];
        if (right.coefficient <= 0) continue;
        const edgeIndex = edgeCount++;
        let edge = pool[edgeIndex];
        if (!edge) edge = pool[edgeIndex] = {
          axis: row.axis, negativeCellId: left.cellId,
          positiveCellId: right.cellId, area: subfaceArea, distance: row.distance,
        };
        edge.axis = row.axis;
        edge.negativeCellId = left.cellId;
        edge.positiveCellId = right.cellId;
        edge.area = subfaceArea;
        edge.distance = row.distance;
        edges[edgeIndex] = edge;
      }
    }
  }
  edges.length = edgeCount;
  return edges;
}

function width(grid: SparseAtlasCompositeGrid, cellId: number): number {
  return grid.cells[cellId].widthsFine[0];
}

function diffuseGammaAxis(
  grid: SparseAtlasCompositeGrid,
  edges: readonly ScalarEdge[],
  scale: number,
  sourceDensity: Float64Array,
  sourceGamma: Float64Array,
  sweptDensity: Float64Array,
  sweptGamma: Float64Array,
  axis: SparseAtlasAxis,
  flux: Cm12GammaDiffusionFlux,
): number {
  sweptDensity.set(sourceDensity);
  sweptGamma.set(sourceGamma);
  let pairUpdates = 0;
  for (const edge of edges) {
    if (edge.axis !== axis) continue;
    const negative = grid.cells[edge.negativeCellId];
    const positive = grid.cells[edge.positiveCellId];
    const conductedVolume = scale * Math.min(
      edge.area * width(grid, negative.id),
      edge.area * width(grid, positive.id),
    );
    const intoNegative = cm12GammaDiffusionFluxInto(
      sourceDensity[negative.id], sourceGamma[negative.id],
      sourceDensity[positive.id], sourceGamma[positive.id],
      conductedVolume / negative.volume,
      flux,
    );
    const integratedRho = negative.volume * intoNegative.rho;
    const integratedGamma = negative.volume * intoNegative.gamma;
    sweptDensity[negative.id] += intoNegative.rho;
    sweptGamma[negative.id] += intoNegative.gamma;
    sweptDensity[positive.id] -= integratedRho / positive.volume;
    sweptGamma[positive.id] -= integratedGamma / positive.volume;
    pairUpdates += 1;
  }
  return pairUpdates;
}

function diffuseGamma(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasSurfaceFields,
  edges: readonly ScalarEdge[],
  iterations: number,
  scale: number,
  workspace: SparseAtlasSurfaceConditioningWorkspace,
): number {
  const count = grid.cells.length;
  const density = workspace.density;
  const gamma = workspace.gamma;
  density.set(fields.density);
  gamma.set(fields.gamma);
  let pairUpdates = 0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    // The paper declares Gauss-Seidel between dimensional sweeps. Averaging
    // the two mirrored sweep orders retains that operator while avoiding an
    // arbitrary x-before-z bias in symmetric sparse scenes.
    pairUpdates += diffuseGammaAxis(grid, edges, scale, density, gamma,
      workspace.forwardDensity0, workspace.forwardGamma0, 0, workspace.diffusionFlux);
    pairUpdates += diffuseGammaAxis(grid, edges, scale,
      workspace.forwardDensity0, workspace.forwardGamma0,
      workspace.forwardDensity1, workspace.forwardGamma1, 1, workspace.diffusionFlux);
    pairUpdates += diffuseGammaAxis(grid, edges, scale,
      workspace.forwardDensity1, workspace.forwardGamma1,
      workspace.forwardDensity0, workspace.forwardGamma0, 2, workspace.diffusionFlux);
    pairUpdates += diffuseGammaAxis(grid, edges, scale, density, gamma,
      workspace.reverseDensity0, workspace.reverseGamma0, 2, workspace.diffusionFlux);
    pairUpdates += diffuseGammaAxis(grid, edges, scale,
      workspace.reverseDensity0, workspace.reverseGamma0,
      workspace.reverseDensity1, workspace.reverseGamma1, 1, workspace.diffusionFlux);
    pairUpdates += diffuseGammaAxis(grid, edges, scale,
      workspace.reverseDensity1, workspace.reverseGamma1,
      workspace.reverseDensity0, workspace.reverseGamma0, 0, workspace.diffusionFlux);
    for (let id = 0; id < count; id += 1) {
      density[id] = 0.5 * (workspace.forwardDensity0[id] + workspace.reverseDensity0[id]);
      gamma[id] = 0.5 * (workspace.forwardGamma0[id] + workspace.reverseGamma0[id]);
    }
  }
  return pairUpdates;
}

function adjacency(
  grid: SparseAtlasCompositeGrid,
  edges: readonly ScalarEdge[],
  result?: DirectionalAdjacency,
  pool: Neighbor[] = [],
): DirectionalAdjacency {
  const axisLists = (): Neighbor[][][] => [[], [], []];
  const negative = result?.negative ?? axisLists();
  const positive = result?.positive ?? axisLists();
  const all: Neighbor[][] = result?.all ?? [];
  const negativeCount = result?.negativeCount ?? [[], [], []];
  const positiveCount = result?.positiveCount ?? [[], [], []];
  const allCount = result?.allCount ?? [];
  for (let axis = 0; axis < 3; axis += 1) {
    while (negative[axis].length < grid.cells.length) negative[axis].push([]);
    while (positive[axis].length < grid.cells.length) positive[axis].push([]);
    for (let id = 0; id < grid.cells.length; id += 1) {
      negativeCount[axis][id] = 0;
      positiveCount[axis][id] = 0;
    }
  }
  while (all.length < grid.cells.length) all.push([]);
  for (let id = 0; id < grid.cells.length; id += 1) allCount[id] = 0;
  let neighborIndex = 0;
  for (const edge of edges) {
    let towardPositive = pool[neighborIndex++];
    if (!towardPositive) towardPositive = pool[neighborIndex - 1] = {
      cellId: edge.positiveCellId, area: edge.area, distance: edge.distance,
    };
    towardPositive.cellId = edge.positiveCellId;
    towardPositive.area = edge.area;
    towardPositive.distance = edge.distance;
    let towardNegative = pool[neighborIndex++];
    if (!towardNegative) towardNegative = pool[neighborIndex - 1] = {
      cellId: edge.negativeCellId, area: edge.area, distance: edge.distance,
    };
    towardNegative.cellId = edge.negativeCellId;
    towardNegative.area = edge.area;
    towardNegative.distance = edge.distance;
    let count = positiveCount[edge.axis][edge.negativeCellId];
    positive[edge.axis][edge.negativeCellId][count] = towardPositive;
    positiveCount[edge.axis][edge.negativeCellId] = count + 1;
    count = negativeCount[edge.axis][edge.positiveCellId];
    negative[edge.axis][edge.positiveCellId][count] = towardNegative;
    negativeCount[edge.axis][edge.positiveCellId] = count + 1;
    count = allCount[edge.negativeCellId];
    all[edge.negativeCellId][count] = towardPositive;
    allCount[edge.negativeCellId] = count + 1;
    count = allCount[edge.positiveCellId];
    all[edge.positiveCellId][count] = towardNegative;
    allCount[edge.positiveCellId] = count + 1;
  }
  return { negative, positive, all, negativeCount, positiveCount, allCount };
}

interface SurfaceTopology {
  edges: ScalarEdge[];
  graph: DirectionalAdjacency;
  d4Orbits: readonly (readonly number[])[] | undefined;
  ownerByFine: Int32Array;
}

function buildFineOwnerTable(grid: SparseAtlasCompositeGrid): Int32Array {
  const [nx, ny, nz] = grid.atlas.dimensions;
  const result = new Int32Array(nx * ny * nz).fill(-1);
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1) {
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1) {
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          result[x + nx * (y + ny * z)] = cell.id;
        }
      }
    }
  }
  return result;
}

// `rebindCompositeGrid` deliberately reuses the immutable gradient-row array
// when only leaf fields change. Keying on that array makes all connectivity
// construction zero-work on ordinary frames while topology epochs naturally
// miss and rebuild the cache.
const surfaceTopologyCache = new WeakMap<object, SurfaceTopology>();
const surfaceD4Authority = new WeakMap<object, boolean>();

function surfaceTopology(
  grid: SparseAtlasCompositeGrid,
  workspace?: SparseAtlasSurfaceConditioningWorkspace,
): SurfaceTopology {
  const key = (grid.topologyKey ?? grid.gradientRows) as object;
  if (workspace?.topologyKey === key && workspace.topology) return workspace.topology;
  if (workspace) {
    const topology = workspace.topology ?? (workspace.topology = {
      edges: [],
      graph: adjacency(grid, [], undefined, workspace.neighborPool),
      d4Orbits: undefined,
      ownerByFine: new Int32Array(0),
    });
    scalarEdges(grid, topology.edges, workspace.edgePool);
    topology.graph = adjacency(grid, topology.edges, topology.graph, workspace.neighborPool);
    topology.d4Orbits = buildD4SymmetryOrbits(grid);
    topology.ownerByFine = buildFineOwnerTable(grid);
    workspace.topologyKey = key;
    return topology;
  }
  const cached = surfaceTopologyCache.get(key);
  if (cached) return cached;
  const edges = scalarEdges(grid);
  const built = {
    edges,
    graph: adjacency(grid, edges),
    d4Orbits: buildD4SymmetryOrbits(grid),
    ownerByFine: buildFineOwnerTable(grid),
  } satisfies SurfaceTopology;
  surfaceTopologyCache.set(key, built);
  return built;
}

function areaAverage(
  own: number,
  neighbors: readonly Neighbor[],
  neighborCount: number,
  values: ArrayLike<number>,
  result: { value: number; distance: number } = { value: own, distance: 1 },
): { value: number; distance: number } {
  if (neighborCount === 0) {
    result.value = own;
    result.distance = 1;
    return result;
  }
  let totalArea = 0, weightedValue = 0, weightedDistance = 0;
  for (let index = 0; index < neighborCount; index += 1) {
    const neighbor = neighbors[index];
    totalArea += neighbor.area;
    weightedValue += neighbor.area * values[neighbor.cellId];
    weightedDistance += neighbor.area * neighbor.distance;
  }
  result.value = weightedValue / totalArea;
  result.distance = weightedDistance / totalArea;
  return result;
}

function sharpeningDeltas(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  graph: DirectionalAdjacency,
  courant: number,
  result: Float64Array = new Float64Array(grid.cells.length),
  workspace?: SparseAtlasSurfaceConditioningWorkspace,
): Float64Array {
  for (const cell of grid.cells) {
    const rho = density[cell.id];
    let maximumDifference = 0;
    for (let index = 0; index < graph.allCount[cell.id]; index += 1) {
      const neighbor = graph.all[cell.id][index];
      maximumDifference = Math.max(
        maximumDifference, Math.abs(rho - density[neighbor.cellId]),
      );
    }
    const weight = cm12SharpeningWeight(rho, maximumDifference);
    let plusSquared = 0, minusSquared = 0;
    for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
      const axis = axisIndex as SparseAtlasAxis;
      const before = areaAverage(
        rho, graph.negative[axis][cell.id], graph.negativeCount[axis][cell.id],
        density, workspace?.averageBefore,
      );
      const after = areaAverage(
        rho, graph.positive[axis][cell.id], graph.positiveCount[axis][cell.id],
        density, workspace?.averageAfter,
      );
      const backward = -(rho - before.value) * courant / before.distance;
      const forward = -(after.value - rho) * courant / after.distance;
      plusSquared += Math.max(Math.max(backward, 0) ** 2, Math.min(forward, 0) ** 2);
      minusSquared += Math.max(Math.min(backward, 0) ** 2, Math.max(forward, 0) ** 2);
    }
    let delta = weight * Math.sqrt(weight >= 0 ? plusSquared : minusSquared);
    if (rho + delta < 0 || rho < 1e-5) delta = -rho;
    else if (rho > 0.5) delta = 0;
    result[cell.id] = Math.min(0, delta);
  }
  return result;
}

function sharpeningDestinations(
  grid: SparseAtlasCompositeGrid,
  sourceCellId: number,
  density: ArrayLike<number>,
  topology: SurfaceTopology,
  maximumDistanceCells: number,
  workspace: SparseAtlasSurfaceConditioningWorkspace,
): number {
  ensureCandidateCapacity(workspace, 8);
  const dimensions = grid.atlas.dimensions;
  const ownerAt = (x: number, y: number, z: number): number => {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    if (ix < 0 || iy < 0 || iz < 0
      || ix >= dimensions[0] || iy >= dimensions[1] || iz >= dimensions[2]) return -1;
    return topology.ownerByFine[ix + dimensions[0] * (iy + dimensions[1] * iz)];
  };
  const sample = (x: number, y: number, z: number): number => {
    const probe = ownerAt(
      Math.min(dimensions[0] - 1e-4, Math.max(0, x)),
      Math.min(dimensions[1] - 1e-4, Math.max(0, y)),
      Math.min(dimensions[2] - 1e-4, Math.max(0, z)),
    );
    if (probe < 0) return 0;
    const spans = grid.cells[probe].widthsFine;
    const lowerX = Math.floor(x / spans[0] - 0.5);
    const lowerY = Math.floor(y / spans[1] - 0.5);
    const lowerZ = Math.floor(z / spans[2] - 0.5);
    const fractionX = x / spans[0] - 0.5 - lowerX;
    const fractionY = y / spans[1] - 0.5 - lowerY;
    const fractionZ = z / spans[2] - 0.5 - lowerZ;
    let result = 0;
    for (let corner = 0; corner < 8; corner += 1) {
      const ox = corner & 1, oy = (corner >> 1) & 1, oz = (corner >> 2) & 1;
      const owner = ownerAt(
        spans[0] * (lowerX + ox + 0.5),
        spans[1] * (lowerY + oy + 0.5),
        spans[2] * (lowerZ + oz + 0.5),
      );
      if (owner < 0) continue;
      result += (ox ? fractionX : 1 - fractionX)
        * (oy ? fractionY : 1 - fractionY)
        * (oz ? fractionZ : 1 - fractionZ) * density[owner];
    }
    return result;
  };

  const source = grid.cells[sourceCellId];
  const position = workspace.tracePosition;
  position.set(source.centerFine);
  const maximumDistance = maximumDistanceCells * Math.min(...source.widthsFine);
  let travelled = 0;
  for (let step = 0; step < 40 && travelled < maximumDistance; step += 1) {
    if (sample(position[0], position[1], position[2]) >= 0.5) break;
    const owner = ownerAt(position[0], position[1], position[2]);
    if (owner < 0) break;
    const localWidth = Math.min(...grid.cells[owner].widthsFine);
    const halfDistance = 0.5 * localWidth;
    const gradient = workspace.traceGradient;
    gradient[0] = (sample(position[0] + halfDistance, position[1], position[2])
      - sample(position[0] - halfDistance, position[1], position[2])) / (2 * halfDistance);
    gradient[1] = (sample(position[0], position[1] + halfDistance, position[2])
      - sample(position[0], position[1] - halfDistance, position[2])) / (2 * halfDistance);
    gradient[2] = (sample(position[0], position[1], position[2] + halfDistance)
      - sample(position[0], position[1], position[2] - halfDistance)) / (2 * halfDistance);
    const magnitude = Math.hypot(gradient[0], gradient[1], gradient[2]);
    if (magnitude < 1e-12) break;
    const distance = Math.min(halfDistance, maximumDistance - travelled);
    const nextX = position[0] + gradient[0] * distance / magnitude;
    const nextY = position[1] + gradient[1] * distance / magnitude;
    const nextZ = position[2] + gradient[2] * distance / magnitude;
    if (ownerAt(nextX, nextY, nextZ) < 0) break;
    position[0] = nextX; position[1] = nextY; position[2] = nextZ;
    travelled += distance;
  }

  const probe = ownerAt(position[0], position[1], position[2]);
  if (probe < 0) {
    workspace.candidateIds[0] = sourceCellId;
    workspace.candidateWeights[0] = 1;
    return 1;
  }
  const spans = grid.cells[probe].widthsFine;
  const lowerX = Math.floor(position[0] / spans[0] - 0.5);
  const lowerY = Math.floor(position[1] / spans[1] - 0.5);
  const lowerZ = Math.floor(position[2] / spans[2] - 0.5);
  const fractionX = position[0] / spans[0] - 0.5 - lowerX;
  const fractionY = position[1] / spans[1] - 0.5 - lowerY;
  const fractionZ = position[2] / spans[2] - 0.5 - lowerZ;
  let count = 0, totalWeight = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const ox = corner & 1, oy = (corner >> 1) & 1, oz = (corner >> 2) & 1;
    const owner = ownerAt(
      spans[0] * (lowerX + ox + 0.5),
      spans[1] * (lowerY + oy + 0.5),
      spans[2] * (lowerZ + oz + 0.5),
    );
    if (owner < 0) continue;
    const weight = (ox ? fractionX : 1 - fractionX)
      * (oy ? fractionY : 1 - fractionY)
      * (oz ? fractionZ : 1 - fractionZ);
    if (!(weight > 0)) continue;
    let index = 0;
    while (index < count && workspace.candidateIds[index] !== owner) index += 1;
    if (index === count) {
      workspace.candidateIds[count++] = owner;
      workspace.candidateWeights[index] = 0;
    }
    workspace.candidateWeights[index] += weight;
    totalWeight += weight;
  }
  if (!(totalWeight > 0)) {
    workspace.candidateIds[0] = sourceCellId;
    workspace.candidateWeights[0] = 1;
    return 1;
  }
  for (let index = 0; index < count; index += 1) {
    workspace.candidateWeights[index] /= totalWeight;
  }
  return count;
}

export function conditionSparseAtlasSurface(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasSurfaceFields,
  options: SparseAtlasSurfaceConditioningOptions = {},
  workspace: SparseAtlasSurfaceConditioningWorkspace =
    createSparseAtlasSurfaceConditioningWorkspace(),
): SparseAtlasSurfaceConditioningResult {
  if (fields.density.length !== grid.cells.length
    || fields.gamma.length !== grid.cells.length) {
    throw new RangeError("surface fields must contain one value per resident cell");
  }
  ensureSurfaceVectorLength(workspace, grid.cells.length);
  const iterations = options.gammaDiffusionIterations ?? 1;
  const timeStep_s = options.timeStep_s ?? 1 / 30;
  const finestCellSize_m = options.finestCellSize_m ?? 1;
  const gammaScale = options.gammaDiffusionScale
    ?? Math.min(1, timeStep_s / (1 / 30));
  // Grid distances are expressed in finest-cell coordinates. CM12 Eqs. 6-15
  // produce a density increment of 3 dt |grad rho|, so the pseudo-time must be
  // expressed in the same coordinate units before dividing by edge distance.
  const courant = options.sharpeningCourant ?? 3 * timeStep_s / finestCellSize_m;
  const maximumDistanceCells = options.sharpeningDistanceCells ?? 2.1;
  if (!Number.isInteger(iterations) || iterations < 0
    || !Number.isFinite(timeStep_s) || timeStep_s < 0
    || !Number.isFinite(finestCellSize_m) || finestCellSize_m <= 0
    || !Number.isFinite(gammaScale) || gammaScale < 0 || gammaScale > 1
    || !Number.isFinite(courant) || courant < 0
    || !Number.isFinite(maximumDistanceCells) || maximumDistanceCells < 0) {
    throw new RangeError("invalid sparse CM12 surface-conditioning options");
  }
  const massBefore = integratedScalar(grid, fields.density);
  const gammaBefore = integratedScalar(grid, fields.gamma);
  const topologyKey = (grid.topologyKey ?? grid.gradientRows) as object;
  const topology = surfaceTopology(grid, workspace);
  let preservesD4 = options.preserveHorizontalD4
    ? topology.d4Orbits !== undefined
    : surfaceD4Authority.get(topologyKey);
  if (preservesD4 === undefined) {
    preservesD4 = activeD4SymmetryOrbits(topology.d4Orbits, fields) !== undefined;
    surfaceD4Authority.set(topologyKey, preservesD4);
  }
  const symmetryOrbits = preservesD4 ? topology.d4Orbits : undefined;
  const edges = topology.edges;
  const gammaPairUpdates = diffuseGamma(
    grid, fields, edges, iterations, gammaScale, workspace,
  );
  const frozenDensity = workspace.density;
  const graph = topology.graph;
  const deltas = workspace.deltas;
  sharpeningDeltas(grid, frozenDensity, graph, courant, deltas, workspace);
  const density = workspace.conditionedDensity;
  for (let cellId = 0; cellId < density.length; cellId += 1) {
    density[cellId] = frozenDensity[cellId] + deltas[cellId];
  }
  let removedIntegratedMass = 0, returnedIntegratedMass = 0;
  let fallbackIntegratedMass = 0;
  for (const source of grid.cells) {
    if (deltas[source.id] >= 0) continue;
    const removedMass = -deltas[source.id] * source.volume;
    const destinationCount = sharpeningDestinations(
      grid, source.id, frozenDensity, topology, maximumDistanceCells, workspace,
    );
    for (let index = 0; index < destinationCount; index += 1) {
      const destinationId = workspace.candidateIds[index];
      const destination = grid.cells[destinationId];
      density[destinationId] += removedMass * workspace.candidateWeights[index]
        / destination.volume;
    }
    if (destinationCount === 1 && workspace.candidateIds[0] === source.id) {
      fallbackIntegratedMass += removedMass;
    }
    removedIntegratedMass += removedMass;
    returnedIntegratedMass += removedMass;
  }
  const gamma = workspace.gamma;
  preserveD4Symmetry(symmetryOrbits, density, gamma);
  const massAbsoluteError = Math.abs(integratedScalar(grid, density) - massBefore);
  const gammaIntegralAbsoluteError = Math.abs(
    integratedScalar(grid, gamma) - gammaBefore,
  );
  let outputFields = workspace.outputFields;
  if (!outputFields) workspace.outputFields = outputFields = { density, gamma };
  else {
    (outputFields as { density: Float64Array }).density = density;
    (outputFields as { gamma: Float64Array }).gamma = gamma;
  }
  let result = workspace.result;
  if (!result) {
    workspace.result = result = {
      fields: outputFields, edgeCount: edges.length, gammaPairUpdates,
      removedIntegratedMass, returnedIntegratedMass, fallbackIntegratedMass,
      massAbsoluteError, gammaIntegralAbsoluteError,
    };
  } else {
    const mutable = result as {
      fields: SparseAtlasSurfaceFields;
      edgeCount: number;
      gammaPairUpdates: number;
      removedIntegratedMass: number;
      returnedIntegratedMass: number;
      fallbackIntegratedMass: number;
      massAbsoluteError: number;
      gammaIntegralAbsoluteError: number;
    };
    mutable.fields = outputFields;
    mutable.edgeCount = edges.length;
    mutable.gammaPairUpdates = gammaPairUpdates;
    mutable.removedIntegratedMass = removedIntegratedMass;
    mutable.returnedIntegratedMass = returnedIntegratedMass;
    mutable.fallbackIntegratedMass = fallbackIntegratedMass;
    mutable.massAbsoluteError = massAbsoluteError;
    mutable.gammaIntegralAbsoluteError = gammaIntegralAbsoluteError;
  }
  return result;
}
