/**
 * Conservative CM12 surface conditioning on the frozen two-tile composite grid.
 *
 * Gamma diffusion is a direct control-volume extension of the shared CM12 pair
 * flux. Density sharpening is intentionally a graph-form oracle: it keeps the
 * paper's limiter and air-side-only deletion, then returns every removed unit of
 * integrated mass along a bounded, frozen-density uphill path. The production
 * shader still needs continuous adaptive sampling for TraceAlongField.
 */

import {
  cm12GammaDiffusionFluxInto,
  cm12SharpeningWeight,
} from "../../core/cm12-numerics";
import {
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
  type Vec3,
} from "./two-tile-composite-grid";
import {
  advanceTwoTileConservativeTransport,
  integratedScalar,
  type TwoTileTransportFields,
} from "./two-tile-conservative-transport";

export interface CompositeScalarEdge {
  id: number;
  axis: CompositeAxis;
  negativeCellId: number;
  positiveCellId: number;
  area: number;
  distance: number;
  negativeFaceFraction: number;
  positiveFaceFraction: number;
  seam: boolean;
}

export interface GammaDiffusionResult {
  fields: TwoTileTransportFields;
  pairUpdates: number;
  massAbsoluteError: number;
  gammaIntegralAbsoluteError: number;
}

export interface TwoTileSharpeningOptions {
  /** Dimensionless fictitious step multiplier. Defaults to 0.2. */
  courant?: number;
  /** Maximum graph-trace distance in source-cell widths. Defaults to 2.1. */
  maximumDistanceCells?: number;
}

export interface SharpeningResult {
  fields: TwoTileTransportFields;
  removedIntegratedMass: number;
  returnedIntegratedMass: number;
  fallbackIntegratedMass: number;
  crossResolutionReturnedMass: number;
  returnBalanceAbsoluteError: number;
  massAbsoluteError: number;
}

export interface SurfaceConditioningVariantReceipt {
  axis: CompositeAxis;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  direction: -1 | 1;
  steps: number;
  maximumTransportMassError: number;
  maximumDiffusionMassError: number;
  maximumDiffusionGammaIntegralError: number;
  maximumSharpeningMassError: number;
  maximumSharpeningReturnBalanceError: number;
  finalMassAbsoluteError: number;
  minimumDensity: number;
  maximumDensity: number;
  minimumGamma: number;
  maximumGamma: number;
  totalSharpeningReturnMass: number;
  totalSharpeningFallbackMass: number;
  totalCrossResolutionReturnMass: number;
}

export interface SurfaceConditioningReflectionReceipt {
  axis: CompositeAxis;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  direction: -1 | 1;
  maximumDensityDifference: number;
  maximumGammaDifference: number;
  massDifference: number;
}

export interface SurfaceConditioningProbeResult {
  passed: boolean;
  tolerance: number;
  steps: number;
  variants: readonly SurfaceConditioningVariantReceipt[];
  reflections: readonly SurfaceConditioningReflectionReceipt[];
  failures: readonly string[];
}

interface DirectionalNeighbor {
  cellId: number;
  area: number;
  distance: number;
}

interface DirectionalAdjacency {
  negative: DirectionalNeighbor[][][];
  positive: DirectionalNeighbor[][][];
  all: DirectionalNeighbor[][];
}

function compensatedSum(values: Iterable<number>): number {
  let sum = 0;
  let correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

/** Expand aggregate seam ports into their physical fine/coarse subfaces. */
export function buildCompositeScalarEdges(grid: TwoTileCompositeGrid): CompositeScalarEdge[] {
  const edges: CompositeScalarEdge[] = [];
  const emit = (
    axis: CompositeAxis,
    negativeCellId: number,
    positiveCellId: number,
    area: number,
    distance: number,
    seam: boolean,
  ): void => {
    const negative = grid.cells[negativeCellId];
    const positive = grid.cells[positiveCellId];
    edges.push({
      id: edges.length,
      axis,
      negativeCellId,
      positiveCellId,
      area,
      distance,
      negativeFaceFraction: area / (negative.width * negative.width),
      positiveFaceFraction: area / (positive.width * positive.width),
      seam,
    });
  };

  for (const face of grid.regularFaces) {
    const negative = face.terms.find((term) => term.coefficient < 0);
    const positive = face.terms.find((term) => term.coefficient > 0);
    if (negative === undefined || positive === undefined) {
      throw new Error(`regular face ${face.id} is not an oriented pair`);
    }
    emit(face.axis, negative.cellId, positive.cellId, face.area, face.distance, false);
  }
  for (const port of grid.seamPorts) {
    const negative = port.terms.filter((term) => term.coefficient < 0);
    const positive = port.terms.filter((term) => term.coefficient > 0);
    const subfaceArea = port.area / (negative.length * positive.length);
    for (const left of negative) {
      for (const right of positive) {
        emit(port.axis, left.cellId, right.cellId, subfaceArea, port.distance, true);
      }
    }
  }
  return edges;
}

function validateFields(grid: TwoTileCompositeGrid, fields: TwoTileTransportFields): void {
  if (fields.density.length !== grid.cells.length || fields.gamma.length !== grid.cells.length) {
    throw new RangeError(`surface fields must contain ${grid.cells.length} cells`);
  }
}

/**
 * Snapshot-Jacobi within an axis and Gauss-Seidel between axis sweeps, matching
 * CM12/LAF11. Pair transfers are conservative in physical cell-volume measure.
 */
export function diffuseTwoTileGamma(
  grid: TwoTileCompositeGrid,
  fields: TwoTileTransportFields,
  iterations = 2,
): GammaDiffusionResult {
  validateFields(grid, fields);
  if (!Number.isInteger(iterations) || iterations < 0) {
    throw new RangeError(`iterations must be a nonnegative integer; received ${iterations}`);
  }
  const massBefore = integratedScalar(grid, fields.density);
  const gammaBefore = integratedScalar(grid, fields.gamma);
  let density = fields.density.slice();
  let gamma = fields.gamma.slice();
  const edges = buildCompositeScalarEdges(grid);
  let pairUpdates = 0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const axis of [0, 1, 2] as const) {
      const oldDensity = density;
      const oldGamma = gamma;
      density = oldDensity.slice();
      gamma = oldGamma.slice();
      for (const edge of edges) {
        if (edge.axis !== axis) continue;
        const negative = grid.cells[edge.negativeCellId];
        const positive = grid.cells[edge.positiveCellId];
        const conductedVolume = Math.min(
          negative.volume * edge.negativeFaceFraction,
          positive.volume * edge.positiveFaceFraction,
        );
        const negativeOpen = conductedVolume / negative.volume;
        const intoNegative = cm12GammaDiffusionFluxInto(
          oldDensity[negative.id],
          oldGamma[negative.id],
          oldDensity[positive.id],
          oldGamma[positive.id],
          negativeOpen,
        );
        const integratedRho = negative.volume * intoNegative.rho;
        const integratedGamma = negative.volume * intoNegative.gamma;
        density[negative.id] += intoNegative.rho;
        gamma[negative.id] += intoNegative.gamma;
        density[positive.id] -= integratedRho / positive.volume;
        gamma[positive.id] -= integratedGamma / positive.volume;
        pairUpdates += 1;
      }
    }
  }

  return {
    fields: { density, gamma },
    pairUpdates,
    massAbsoluteError: Math.abs(integratedScalar(grid, density) - massBefore),
    gammaIntegralAbsoluteError: Math.abs(integratedScalar(grid, gamma) - gammaBefore),
  };
}

function buildDirectionalAdjacency(
  grid: TwoTileCompositeGrid,
  edges: readonly CompositeScalarEdge[],
): DirectionalAdjacency {
  const makeAxisLists = (): DirectionalNeighbor[][][] => [
    Array.from({ length: grid.cells.length }, () => []),
    Array.from({ length: grid.cells.length }, () => []),
    Array.from({ length: grid.cells.length }, () => []),
  ];
  const negative = makeAxisLists();
  const positive = makeAxisLists();
  const all: DirectionalNeighbor[][] = Array.from({ length: grid.cells.length }, () => []);
  for (const edge of edges) {
    const towardPositive = { cellId: edge.positiveCellId, area: edge.area, distance: edge.distance };
    const towardNegative = { cellId: edge.negativeCellId, area: edge.area, distance: edge.distance };
    positive[edge.axis][edge.negativeCellId].push(towardPositive);
    negative[edge.axis][edge.positiveCellId].push(towardNegative);
    all[edge.negativeCellId].push(towardPositive);
    all[edge.positiveCellId].push(towardNegative);
  }
  return { negative, positive, all };
}

function areaAverage(
  own: number,
  neighbors: readonly DirectionalNeighbor[],
  values: ArrayLike<number>,
): { value: number; distance: number } {
  if (neighbors.length === 0) return { value: own, distance: 1 };
  const totalArea = compensatedSum(neighbors.map((neighbor) => neighbor.area));
  return {
    value: compensatedSum(neighbors.map(
      (neighbor) => neighbor.area * values[neighbor.cellId],
    )) / totalArea,
    distance: compensatedSum(neighbors.map(
      (neighbor) => neighbor.area * neighbor.distance,
    )) / totalArea,
  };
}

function sharpeningDeltas(
  grid: TwoTileCompositeGrid,
  density: ArrayLike<number>,
  adjacency: DirectionalAdjacency,
  courant: number,
): Float64Array {
  return Float64Array.from(grid.cells, (cell) => {
    const rho = density[cell.id];
    let maximumDifference = 0;
    for (const neighbor of adjacency.all[cell.id]) {
      maximumDifference = Math.max(maximumDifference, Math.abs(rho - density[neighbor.cellId]));
    }
    const weight = cm12SharpeningWeight(rho, maximumDifference);
    let plusSquared = 0;
    let minusSquared = 0;
    for (const axis of [0, 1, 2] as const) {
      const before = areaAverage(rho, adjacency.negative[axis][cell.id], density);
      const after = areaAverage(rho, adjacency.positive[axis][cell.id], density);
      const backwardChange = -(rho - before.value) * courant * cell.width / before.distance;
      const forwardChange = -(after.value - rho) * courant * cell.width / after.distance;
      plusSquared += Math.max(
        Math.max(backwardChange, 0) ** 2,
        Math.min(forwardChange, 0) ** 2,
      );
      minusSquared += Math.max(
        Math.min(backwardChange, 0) ** 2,
        Math.max(forwardChange, 0) ** 2,
      );
    }
    let delta = weight * Math.sqrt(weight >= 0 ? plusSquared : minusSquared);
    if (rho + delta < 0 || rho < 1e-5) delta = -rho;
    else if (rho > 0.5) delta = 0;
    // The deletion stage is air-side only. Numerical sign surprises become a
    // no-op here rather than manufacturing a return with negative mass.
    return Math.min(0, delta);
  });
}

function traceSharpeningDestination(
  grid: TwoTileCompositeGrid,
  sourceCellId: number,
  density: ArrayLike<number>,
  adjacency: DirectionalAdjacency,
  maximumDistanceCells: number,
): number {
  const maximumDistance = maximumDistanceCells * grid.cells[sourceCellId].width;
  let travelled = 0;
  let current = sourceCellId;
  const visited = new Set<number>([current]);
  while (density[current] < 0.5) {
    let best: DirectionalNeighbor | undefined;
    let bestSlope = 0;
    for (const candidate of adjacency.all[current]) {
      if (visited.has(candidate.cellId) || travelled + candidate.distance > maximumDistance) continue;
      const slope = (density[candidate.cellId] - density[current]) / candidate.distance;
      if (slope > bestSlope + 1e-15) {
        best = candidate;
        bestSlope = slope;
      }
    }
    if (best === undefined) break;
    current = best.cellId;
    visited.add(current);
    travelled += best.distance;
  }
  return current;
}

/** Apply one conservative, frozen-field graph sharpening pass. */
export function sharpenTwoTileDensity(
  grid: TwoTileCompositeGrid,
  fields: TwoTileTransportFields,
  options: TwoTileSharpeningOptions = {},
): SharpeningResult {
  validateFields(grid, fields);
  const courant = options.courant ?? 0.2;
  const maximumDistanceCells = options.maximumDistanceCells ?? 2.1;
  if (!Number.isFinite(courant) || courant < 0 || courant > 1) {
    throw new RangeError(`courant must be in [0,1]; received ${courant}`);
  }
  if (!Number.isFinite(maximumDistanceCells) || maximumDistanceCells < 0) {
    throw new RangeError(`maximumDistanceCells must be nonnegative; received ${maximumDistanceCells}`);
  }
  const massBefore = integratedScalar(grid, fields.density);
  const edges = buildCompositeScalarEdges(grid);
  const adjacency = buildDirectionalAdjacency(grid, edges);
  const frozenDensity = fields.density;
  const deltas = sharpeningDeltas(grid, frozenDensity, adjacency, courant);
  const density = Float64Array.from(frozenDensity, (rho, cellId) => rho + deltas[cellId]);
  let removedIntegratedMass = 0;
  let returnedIntegratedMass = 0;
  let fallbackIntegratedMass = 0;
  let crossResolutionReturnedMass = 0;

  for (const source of grid.cells) {
    if (deltas[source.id] >= 0) continue;
    const removedMass = -deltas[source.id] * source.volume;
    const destinationId = traceSharpeningDestination(
      grid,
      source.id,
      frozenDensity,
      adjacency,
      maximumDistanceCells,
    );
    const destination = grid.cells[destinationId];
    density[destinationId] += removedMass / destination.volume;
    removedIntegratedMass += removedMass;
    returnedIntegratedMass += removedMass;
    if (destinationId === source.id) fallbackIntegratedMass += removedMass;
    if (destination.resolution !== source.resolution) crossResolutionReturnedMass += removedMass;
  }

  return {
    fields: { density, gamma: fields.gamma.slice() },
    removedIntegratedMass,
    returnedIntegratedMass,
    fallbackIntegratedMass,
    crossResolutionReturnedMass,
    returnBalanceAbsoluteError: Math.abs(returnedIntegratedMass - removedIntegratedMass),
    massAbsoluteError: Math.abs(integratedScalar(grid, density) - massBefore),
  };
}

function initialPulse(grid: TwoTileCompositeGrid, direction: -1 | 1): TwoTileTransportFields {
  const normalCenter = grid.tileWidth - direction * 0.16 * grid.tileWidth;
  const density = Float64Array.from(grid.cells, (cell) => {
    const normal = (cell.center[grid.axis] - normalCenter) / (0.42 * grid.tileWidth);
    const tangent0 = (cell.center[grid.tangentialAxes[0]] - 0.5 * grid.tileWidth)
      / (0.4 * grid.tileWidth);
    const tangent1 = (cell.center[grid.tangentialAxes[1]] - 0.5 * grid.tileWidth)
      / (0.45 * grid.tileWidth);
    // A resolved diffuse droplet rather than a sub-cell impulse: the probe is
    // intended to keep a meaningful 0.5 interface while repeatedly crossing
    // the seam, not to claim that sharpening can resurrect unresolved mass.
    return Math.min(1, Math.max(
      0,
      1.15 - normal * normal - tangent0 * tangent0 - tangent1 * tangent1,
    ));
  });
  return { density, gamma: new Float64Array(grid.cells.length).fill(1) };
}

function minimum(values: ArrayLike<number>): number {
  let result = Number.POSITIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) result = Math.min(result, values[index]);
  return result;
}

function maximum(values: ArrayLike<number>): number {
  let result = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) result = Math.max(result, values[index]);
  return result;
}

interface VariantRun {
  receipt: SurfaceConditioningVariantReceipt;
  grid: TwoTileCompositeGrid;
  fields: TwoTileTransportFields;
}

function runSurfaceVariant(
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  direction: -1 | 1,
  steps: number,
): VariantRun {
  const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
  let fields = initialPulse(grid, direction);
  const initialMass = integratedScalar(grid, fields.density);
  let maximumTransportMassError = 0;
  let maximumDiffusionMassError = 0;
  let maximumDiffusionGammaIntegralError = 0;
  let maximumSharpeningMassError = 0;
  let maximumSharpeningReturnBalanceError = 0;
  let totalSharpeningReturnMass = 0;
  let totalSharpeningFallbackMass = 0;
  let totalCrossResolutionReturnMass = 0;
  let minimumDensity = minimum(fields.density);
  let maximumDensity = maximum(fields.density);
  let minimumGamma = minimum(fields.gamma);
  let maximumGamma = maximum(fields.gamma);
  const displacement = direction * 0.5 / Math.max(negativeResolution, positiveResolution);
  const observe = (): void => {
    minimumDensity = Math.min(minimumDensity, minimum(fields.density));
    maximumDensity = Math.max(maximumDensity, maximum(fields.density));
    minimumGamma = Math.min(minimumGamma, minimum(fields.gamma));
    maximumGamma = Math.max(maximumGamma, maximum(fields.gamma));
  };

  for (let step = 0; step < steps; step += 1) {
    const beforeTransport = integratedScalar(grid, fields.density);
    const transport = advanceTwoTileConservativeTransport({
      axis,
      negativeResolution,
      positiveResolution,
      displacement,
    }, fields);
    fields = transport.after;
    observe();
    maximumTransportMassError = Math.max(maximumTransportMassError,
      Math.abs(integratedScalar(grid, fields.density) - beforeTransport));
    const diffusion = diffuseTwoTileGamma(grid, fields, 2);
    fields = diffusion.fields;
    observe();
    maximumDiffusionMassError = Math.max(maximumDiffusionMassError, diffusion.massAbsoluteError);
    maximumDiffusionGammaIntegralError = Math.max(
      maximumDiffusionGammaIntegralError,
      diffusion.gammaIntegralAbsoluteError,
    );
    const sharpening = sharpenTwoTileDensity(grid, fields);
    fields = sharpening.fields;
    observe();
    maximumSharpeningMassError = Math.max(maximumSharpeningMassError,
      sharpening.massAbsoluteError);
    maximumSharpeningReturnBalanceError = Math.max(
      maximumSharpeningReturnBalanceError,
      sharpening.returnBalanceAbsoluteError,
    );
    totalSharpeningReturnMass += sharpening.returnedIntegratedMass;
    totalSharpeningFallbackMass += sharpening.fallbackIntegratedMass;
    totalCrossResolutionReturnMass += sharpening.crossResolutionReturnedMass;
  }

  return {
    receipt: {
      axis,
      negativeResolution,
      positiveResolution,
      direction,
      steps,
      maximumTransportMassError,
      maximumDiffusionMassError,
      maximumDiffusionGammaIntegralError,
      maximumSharpeningMassError,
      maximumSharpeningReturnBalanceError,
      finalMassAbsoluteError: Math.abs(integratedScalar(grid, fields.density) - initialMass),
      minimumDensity,
      maximumDensity,
      minimumGamma,
      maximumGamma,
      totalSharpeningReturnMass,
      totalSharpeningFallbackMass,
      totalCrossResolutionReturnMass,
    },
    grid,
    fields,
  };
}

function reflectedCellMap(source: TwoTileCompositeGrid, reflected: TwoTileCompositeGrid): Int32Array {
  const scale = 1e9;
  const key = (resolution: number, center: Vec3): string =>
    `${resolution}:${center.map((value) => Math.round(value * scale)).join(",")}`;
  const reflectedByKey = new Map<string, number>();
  for (const cell of reflected.cells) reflectedByKey.set(key(cell.resolution, cell.center), cell.id);
  return Int32Array.from(source.cells, (cell) => {
    const center = [...cell.center] as [number, number, number];
    center[source.axis] = 2 * source.tileWidth - center[source.axis];
    const match = reflectedByKey.get(key(cell.resolution, center));
    if (match === undefined) throw new Error("surface reflection lookup failed");
    return match;
  });
}

export function probeTwoTileSurfaceConditioning(
  tolerance = 1e-10,
  steps = 8,
): SurfaceConditioningProbeResult {
  if (!Number.isFinite(tolerance) || tolerance <= 0) {
    throw new RangeError(`tolerance must be finite and positive; received ${tolerance}`);
  }
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new RangeError(`steps must be a positive integer; received ${steps}`);
  }
  const variants: SurfaceConditioningVariantReceipt[] = [];
  const runs = new Map<string, VariantRun>();
  for (const axis of [0, 1, 2] as const) {
    for (const [negativeResolution, positiveResolution] of [
      [8, 8],
      [4, 4],
      [8, 4],
      [4, 8],
    ] as const) {
      for (const direction of [-1, 1] as const) {
        const run = runSurfaceVariant(
          axis,
          negativeResolution,
          positiveResolution,
          direction,
          steps,
        );
        variants.push(run.receipt);
        runs.set(`${axis}:${negativeResolution}:${positiveResolution}:${direction}`, run);
      }
    }
  }

  const reflections: SurfaceConditioningReflectionReceipt[] = [];
  for (const axis of [0, 1, 2] as const) {
    for (const [negativeResolution, positiveResolution] of [
      [8, 8],
      [4, 4],
      [8, 4],
      [4, 8],
    ] as const) {
      for (const direction of [-1, 1] as const) {
        const original = runs.get(
          `${axis}:${negativeResolution}:${positiveResolution}:${direction}`,
        )!;
        const reflected = runs.get(
          `${axis}:${positiveResolution}:${negativeResolution}:${-direction}`,
        )!;
        const mapping = reflectedCellMap(original.grid, reflected.grid);
        let maximumDensityDifference = 0;
        let maximumGammaDifference = 0;
        for (let source = 0; source < mapping.length; source += 1) {
          maximumDensityDifference = Math.max(maximumDensityDifference,
            Math.abs(original.fields.density[source] - reflected.fields.density[mapping[source]]));
          maximumGammaDifference = Math.max(maximumGammaDifference,
            Math.abs(original.fields.gamma[source] - reflected.fields.gamma[mapping[source]]));
        }
        reflections.push({
          axis,
          negativeResolution,
          positiveResolution,
          direction,
          maximumDensityDifference,
          maximumGammaDifference,
          massDifference: Math.abs(
            integratedScalar(original.grid, original.fields.density)
              - integratedScalar(reflected.grid, reflected.fields.density),
          ),
        });
      }
    }
  }

  const failures: string[] = [];
  for (const variant of variants) {
    const name = `axis ${variant.axis} ${variant.negativeResolution}+${variant.positiveResolution}`
      + ` direction ${variant.direction}`;
    if (variant.maximumTransportMassError > tolerance) {
      failures.push(`${name}: transport mass error ${variant.maximumTransportMassError}`);
    }
    if (variant.maximumDiffusionMassError > tolerance) {
      failures.push(`${name}: gamma-diffusion mass error ${variant.maximumDiffusionMassError}`);
    }
    if (variant.maximumDiffusionGammaIntegralError > tolerance) {
      failures.push(`${name}: gamma integral error ${variant.maximumDiffusionGammaIntegralError}`);
    }
    if (variant.maximumSharpeningMassError > tolerance) {
      failures.push(`${name}: sharpening mass error ${variant.maximumSharpeningMassError}`);
    }
    if (variant.maximumSharpeningReturnBalanceError > tolerance) {
      failures.push(`${name}: sharpening return imbalance `
        + `${variant.maximumSharpeningReturnBalanceError}`);
    }
    if (variant.finalMassAbsoluteError > tolerance) {
      failures.push(`${name}: accumulated mass error ${variant.finalMassAbsoluteError}`);
    }
    if (variant.minimumDensity < -tolerance || variant.minimumGamma < -tolerance) {
      failures.push(`${name}: negative state rho=${variant.minimumDensity}, gamma=${variant.minimumGamma}`);
    }
    if (!Number.isFinite(variant.maximumDensity) || !Number.isFinite(variant.maximumGamma)
      || variant.maximumDensity > 8 || variant.maximumGamma > 8) {
      failures.push(`${name}: unstable maximum rho=${variant.maximumDensity}, gamma=${variant.maximumGamma}`);
    }
  }
  for (const reflection of reflections) {
    if (reflection.maximumDensityDifference > tolerance
      || reflection.maximumGammaDifference > tolerance
      || reflection.massDifference > tolerance) {
      failures.push(`axis ${reflection.axis}: reflected surface state differs by rho `
        + `${reflection.maximumDensityDifference}, gamma ${reflection.maximumGammaDifference}, `
        + `mass ${reflection.massDifference}`);
    }
  }
  return { passed: failures.length === 0, tolerance, steps, variants, reflections, failures };
}
