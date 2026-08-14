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
  /** Explicit dimensionless sharpening pseudo-time multiplier for probes. */
  readonly sharpeningCourant?: number;
  readonly sharpeningDistanceCells?: number;
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

interface ScalarEdge {
  readonly axis: SparseAtlasAxis;
  readonly negativeCellId: number;
  readonly positiveCellId: number;
  readonly area: number;
  readonly distance: number;
}

interface Neighbor {
  readonly cellId: number;
  readonly area: number;
  readonly distance: number;
}

interface DirectionalAdjacency {
  readonly negative: Neighbor[][][];
  readonly positive: Neighbor[][][];
  readonly all: Neighbor[][];
}

function compensatedSum(values: Iterable<number>): number {
  let sum = 0, correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
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
    const meanDensity = compensatedSum(orbit.map((id) => density[id])) / orbit.length;
    const meanGamma = compensatedSum(orbit.map((id) => gamma[id])) / orbit.length;
    for (const id of orbit) {
      density[id] = meanDensity;
      gamma[id] = meanGamma;
    }
  }
}

/** Expand aggregate mixed-resolution ports into physical scalar subfaces. */
function scalarEdges(grid: SparseAtlasCompositeGrid): ScalarEdge[] {
  const edges: ScalarEdge[] = [];
  for (const row of grid.gradientRows) {
    const negative = row.terms.filter((term) => term.coefficient < 0);
    const positive = row.terms.filter((term) => term.coefficient > 0);
    // A one-sided sparse-air row is a pressure boundary, not a resident scalar
    // diffusion/sharpening edge. Receiver activation owns material entering it.
    if (negative.length === 0 || positive.length === 0) continue;
    const subfaceArea = row.area / (negative.length * positive.length);
    for (const left of negative) for (const right of positive) {
      edges.push({
        axis: row.axis,
        negativeCellId: left.cellId,
        positiveCellId: right.cellId,
        area: subfaceArea,
        distance: row.distance,
      });
    }
  }
  return edges;
}

function width(grid: SparseAtlasCompositeGrid, cellId: number): number {
  return grid.cells[cellId].widthsFine[0];
}

function diffuseGamma(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasSurfaceFields,
  edges: readonly ScalarEdge[],
  iterations: number,
  scale: number,
): { fields: SparseAtlasSurfaceFields; pairUpdates: number } {
  let density = fields.density.slice();
  let gamma = fields.gamma.slice();
  let pairUpdates = 0;
  const sweep = (
    sourceDensity: Float64Array,
    sourceGamma: Float64Array,
    axes: readonly SparseAtlasAxis[],
  ): SparseAtlasSurfaceFields => {
    let sweptDensity = sourceDensity, sweptGamma = sourceGamma;
    for (const axis of axes) {
      const oldDensity = sweptDensity, oldGamma = sweptGamma;
      sweptDensity = oldDensity.slice();
      sweptGamma = oldGamma.slice();
      for (const edge of edges) {
        if (edge.axis !== axis) continue;
        const negative = grid.cells[edge.negativeCellId];
        const positive = grid.cells[edge.positiveCellId];
        const conductedVolume = scale * Math.min(
          edge.area * width(grid, negative.id),
          edge.area * width(grid, positive.id),
        );
        const intoNegative = cm12GammaDiffusionFluxInto(
          oldDensity[negative.id], oldGamma[negative.id],
          oldDensity[positive.id], oldGamma[positive.id],
          conductedVolume / negative.volume,
        );
        const integratedRho = negative.volume * intoNegative.rho;
        const integratedGamma = negative.volume * intoNegative.gamma;
        sweptDensity[negative.id] += intoNegative.rho;
        sweptGamma[negative.id] += intoNegative.gamma;
        sweptDensity[positive.id] -= integratedRho / positive.volume;
        sweptGamma[positive.id] -= integratedGamma / positive.volume;
        pairUpdates += 1;
      }
    }
    return { density: sweptDensity, gamma: sweptGamma };
  };
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    // The paper declares Gauss-Seidel between dimensional sweeps. Averaging
    // the two mirrored sweep orders retains that operator while avoiding an
    // arbitrary x-before-z bias in symmetric sparse scenes.
    const forward = sweep(density, gamma, [0, 1, 2]);
    const reverse = sweep(density, gamma, [2, 1, 0]);
    density = Float64Array.from(forward.density,
      (value, id) => 0.5 * (value + reverse.density[id]));
    gamma = Float64Array.from(forward.gamma,
      (value, id) => 0.5 * (value + reverse.gamma[id]));
  }
  return { fields: { density, gamma }, pairUpdates };
}

function adjacency(
  grid: SparseAtlasCompositeGrid,
  edges: readonly ScalarEdge[],
): DirectionalAdjacency {
  const axisLists = (): Neighbor[][][] => [0, 1, 2].map(() =>
    Array.from({ length: grid.cells.length }, () => []));
  const negative = axisLists(), positive = axisLists();
  const all: Neighbor[][] = Array.from({ length: grid.cells.length }, () => []);
  for (const edge of edges) {
    const towardPositive = {
      cellId: edge.positiveCellId, area: edge.area, distance: edge.distance,
    };
    const towardNegative = {
      cellId: edge.negativeCellId, area: edge.area, distance: edge.distance,
    };
    positive[edge.axis][edge.negativeCellId].push(towardPositive);
    negative[edge.axis][edge.positiveCellId].push(towardNegative);
    all[edge.negativeCellId].push(towardPositive);
    all[edge.positiveCellId].push(towardNegative);
  }
  return { negative, positive, all };
}

interface SurfaceTopology {
  readonly edges: readonly ScalarEdge[];
  readonly graph: DirectionalAdjacency;
  readonly d4Orbits: readonly (readonly number[])[] | undefined;
}

// `rebindCompositeGrid` deliberately reuses the immutable gradient-row array
// when only leaf fields change. Keying on that array makes all connectivity
// construction zero-work on ordinary frames while topology epochs naturally
// miss and rebuild the cache.
const surfaceTopologyCache = new WeakMap<object, SurfaceTopology>();
const surfaceD4Authority = new WeakMap<object, boolean>();

function surfaceTopology(grid: SparseAtlasCompositeGrid): SurfaceTopology {
  const key = grid.gradientRows as object;
  const cached = surfaceTopologyCache.get(key);
  if (cached) return cached;
  const edges = scalarEdges(grid);
  const built = {
    edges,
    graph: adjacency(grid, edges),
    d4Orbits: buildD4SymmetryOrbits(grid),
  } satisfies SurfaceTopology;
  surfaceTopologyCache.set(key, built);
  return built;
}

function areaAverage(
  own: number,
  neighbors: readonly Neighbor[],
  values: ArrayLike<number>,
): { value: number; distance: number } {
  if (neighbors.length === 0) return { value: own, distance: 1 };
  let totalArea = 0, weightedValue = 0, weightedDistance = 0;
  for (const neighbor of neighbors) {
    totalArea += neighbor.area;
    weightedValue += neighbor.area * values[neighbor.cellId];
    weightedDistance += neighbor.area * neighbor.distance;
  }
  return {
    value: weightedValue / totalArea,
    distance: weightedDistance / totalArea,
  };
}

function sharpeningDeltas(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  graph: DirectionalAdjacency,
  courant: number,
): Float64Array {
  return Float64Array.from(grid.cells, (cell) => {
    const rho = density[cell.id];
    let maximumDifference = 0;
    for (const neighbor of graph.all[cell.id]) {
      maximumDifference = Math.max(
        maximumDifference, Math.abs(rho - density[neighbor.cellId]),
      );
    }
    const weight = cm12SharpeningWeight(rho, maximumDifference);
    let plusSquared = 0, minusSquared = 0;
    const cellWidth = width(grid, cell.id);
    for (const axis of [0, 1, 2] as const) {
      const before = areaAverage(rho, graph.negative[axis][cell.id], density);
      const after = areaAverage(rho, graph.positive[axis][cell.id], density);
      const backward = -(rho - before.value) * courant * cellWidth / before.distance;
      const forward = -(after.value - rho) * courant * cellWidth / after.distance;
      plusSquared += Math.max(Math.max(backward, 0) ** 2, Math.min(forward, 0) ** 2);
      minusSquared += Math.max(Math.min(backward, 0) ** 2, Math.max(forward, 0) ** 2);
    }
    let delta = weight * Math.sqrt(weight >= 0 ? plusSquared : minusSquared);
    if (rho + delta < 0 || rho < 1e-5) delta = -rho;
    else if (rho > 0.5) delta = 0;
    return Math.min(0, delta);
  });
}

function sharpeningDestinations(
  grid: SparseAtlasCompositeGrid,
  sourceCellId: number,
  density: ArrayLike<number>,
  graph: DirectionalAdjacency,
  maximumDistanceCells: number,
): ReadonlyMap<number, number> {
  const maximumDistance = maximumDistanceCells * width(grid, sourceCellId);
  const candidates = graph.all[sourceCellId].map((neighbor) => ({
    ...neighbor,
    conductance: neighbor.area * Math.max(0,
      (density[neighbor.cellId] - density[sourceCellId]) / neighbor.distance),
  })).filter((candidate) => candidate.conductance > 0
    && candidate.distance <= maximumDistance);
  const total = compensatedSum(candidates.map((candidate) => candidate.conductance));
  if (total <= 1e-30) return new Map([[sourceCellId, 1]]);
  // A simultaneous split across the local uphill gradient is the discrete
  // counterpart of trilinear TraceAlongField scatter. Repeating the stage
  // advances returned mass farther without selecting one arbitrary axis on
  // plateaus, which would break D4 symmetry.
  return new Map(candidates.map((candidate) => [
    candidate.cellId, candidate.conductance / total,
  ]));
}

export function conditionSparseAtlasSurface(
  grid: SparseAtlasCompositeGrid,
  fields: SparseAtlasSurfaceFields,
  options: SparseAtlasSurfaceConditioningOptions = {},
): SparseAtlasSurfaceConditioningResult {
  if (fields.density.length !== grid.cells.length
    || fields.gamma.length !== grid.cells.length) {
    throw new RangeError("surface fields must contain one value per resident cell");
  }
  const iterations = options.gammaDiffusionIterations ?? 1;
  const timeStep_s = options.timeStep_s ?? 1 / 30;
  const gammaScale = options.gammaDiffusionScale
    ?? Math.min(1, timeStep_s / (1 / 30));
  const courant = options.sharpeningCourant ?? 3 * timeStep_s;
  const maximumDistanceCells = options.sharpeningDistanceCells ?? 2.1;
  if (!Number.isInteger(iterations) || iterations < 0
    || !Number.isFinite(timeStep_s) || timeStep_s < 0
    || !Number.isFinite(gammaScale) || gammaScale < 0 || gammaScale > 1
    || !Number.isFinite(courant) || courant < 0 || courant > 1
    || !Number.isFinite(maximumDistanceCells) || maximumDistanceCells < 0) {
    throw new RangeError("invalid sparse CM12 surface-conditioning options");
  }
  const massBefore = integratedScalar(grid, fields.density);
  const gammaBefore = integratedScalar(grid, fields.gamma);
  const topologyKey = grid.gradientRows as object;
  const topology = surfaceTopology(grid);
  let preservesD4 = surfaceD4Authority.get(topologyKey);
  if (preservesD4 === undefined) {
    preservesD4 = activeD4SymmetryOrbits(topology.d4Orbits, fields) !== undefined;
    surfaceD4Authority.set(topologyKey, preservesD4);
  }
  const symmetryOrbits = preservesD4 ? topology.d4Orbits : undefined;
  const edges = topology.edges;
  const diffusion = diffuseGamma(grid, fields, edges, iterations, gammaScale);
  const frozenDensity = diffusion.fields.density;
  const graph = topology.graph;
  const deltas = sharpeningDeltas(grid, frozenDensity, graph, courant);
  const density = Float64Array.from(frozenDensity,
    (rho, cellId) => rho + deltas[cellId]);
  let removedIntegratedMass = 0, returnedIntegratedMass = 0;
  let fallbackIntegratedMass = 0;
  for (const source of grid.cells) {
    if (deltas[source.id] >= 0) continue;
    const removedMass = -deltas[source.id] * source.volume;
    const destinations = sharpeningDestinations(
      grid, source.id, frozenDensity, graph, maximumDistanceCells,
    );
    let remainingMass = removedMass;
    let active = [...destinations.entries()];
    // Capacity-aware redistribution is the discrete CM12 limiter at the
    // scatter end. Revisit saturated branches so one full receiver cannot
    // strand mass while another uphill branch still has room.
    for (let pass = 0; pass < active.length && remainingMass > 1e-30; pass += 1) {
      const available = active.filter(([destinationId]) => {
        const destination = grid.cells[destinationId];
        return (1 - density[destinationId]) * destination.volume > 1e-30;
      });
      const totalWeight = compensatedSum(available.map(([, weight]) => weight));
      if (totalWeight <= 1e-30) break;
      let acceptedThisPass = 0;
      for (const [destinationId, weight] of available) {
        const destination = grid.cells[destinationId];
        const capacity = Math.max(0,
          (1 - density[destinationId]) * destination.volume);
        const accepted = Math.min(capacity, remainingMass * weight / totalWeight);
        density[destinationId] += accepted / destination.volume;
        acceptedThisPass += accepted;
      }
      remainingMass -= acceptedThisPass;
      if (acceptedThisPass <= 1e-30) break;
      active = available;
    }
    if (remainingMass > 0) {
      // Deletion created this exact capacity at the source. Returning an
      // unaccepted remainder is therefore always bounded and conservative.
      density[source.id] += remainingMass / source.volume;
      fallbackIntegratedMass += remainingMass;
    }
    removedIntegratedMass += removedMass;
    returnedIntegratedMass += removedMass;
  }
  preserveD4Symmetry(symmetryOrbits, density, diffusion.fields.gamma);
  return {
    fields: { density, gamma: diffusion.fields.gamma },
    edgeCount: edges.length,
    gammaPairUpdates: diffusion.pairUpdates,
    removedIntegratedMass,
    returnedIntegratedMass,
    fallbackIntegratedMass,
    massAbsoluteError: Math.abs(integratedScalar(grid, density) - massBefore),
    gammaIntegralAbsoluteError: Math.abs(
      integratedScalar(grid, diffusion.fields.gamma) - gammaBefore,
    ),
  };
}
