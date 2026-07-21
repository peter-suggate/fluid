import {
  FINE_LEVELSET_SAMPLE_FLAGS,
  type FineLevelSetBrickOracle,
  type FineLevelSetVec3,
  unpackFineLevelSetBrickKey,
} from "./octree-fine-levelset-bricks";

export type FineLevelSetVelocitySampler = (position: FineLevelSetVec3) => FineLevelSetVec3;
export type FineLevelSetBoundaryPolicy = "strict" | "closed-neumann";

export interface FineLevelSetTrace {
  departure: [number, number, number];
  segments: number;
  finite: boolean;
}

export interface FineLevelSetAdvectionDiagnostics {
  samples: number;
  segmentsPerSample: number;
  departureOutsideResidentBand: number;
  nonfiniteVelocitySamples: number;
  committed: boolean;
}

export interface FineLevelSetRedistanceOptions {
  physicalBandWidth: number;
  residualTolerance?: number;
  maximumSweeps?: number;
}

export interface FineLevelSetRedistanceDiagnostics {
  samples: number;
  seeds: number;
  sweeps: number;
  converged: boolean;
  unresolvedCells: number;
  maximumEikonalResidual: number;
  committed: boolean;
}

export function traceFineLevelSetDeparture(
  start: FineLevelSetVec3,
  dt: number,
  segments: number,
  sampleVelocity: FineLevelSetVelocitySampler,
): FineLevelSetTrace {
  if (!Number.isFinite(dt) || dt < 0) throw new RangeError("Fine level-set timestep must be finite and non-negative");
  if (!Number.isSafeInteger(segments) || segments < 1 || segments > 64) {
    throw new RangeError("Fine level-set trace segments must be an integer in [1, 64]");
  }
  const departure: [number, number, number] = [start[0], start[1], start[2]];
  const segmentDt = dt / segments;
  for (let segment = 0; segment < segments; segment += 1) {
    const velocity = sampleVelocity(departure);
    if (velocity.some((value) => !Number.isFinite(value))) return { departure, segments: segment, finite: false };
    for (let axis = 0; axis < 3; axis += 1) departure[axis] -= segmentDt * velocity[axis];
  }
  return { departure, segments, finite: departure.every(Number.isFinite) };
}

/**
 * One semi-Lagrangian update with m piecewise-linear trace segments and one
 * final phi interpolation, as prescribed by paper section 5.  The update is
 * atomic: any trajectory outside sparse support leaves authoritative phi
 * untouched and reports a non-authoritative generation.
 */
export function advectFineLevelSet(
  oracle: FineLevelSetBrickOracle,
  dt: number,
  sampleVelocity: FineLevelSetVelocitySampler,
  segmentCount: number = oracle.plan.fineFactor,
  boundaryPolicy: FineLevelSetBoundaryPolicy = "strict",
): FineLevelSetAdvectionDiagnostics {
  if (segmentCount < oracle.plan.fineFactor) {
    throw new RangeError("Fine level-set trace cannot use fewer segments than the fine ratio");
  }
  const pages = oracle.residentPages();
  let samples = 0;
  let departureOutsideResidentBand = 0;
  let nonfiniteVelocitySamples = 0;
  for (const page of pages) {
    const brick = unpackFineLevelSetBrickKey(oracle.plan, page.key);
    for (let z = 0; z < oracle.plan.brickResolution; z += 1) {
      for (let y = 0; y < oracle.plan.brickResolution; y += 1) {
        for (let x = 0; x < oracle.plan.brickResolution; x += 1) {
          const index = x + oracle.plan.brickResolution * (y + oracle.plan.brickResolution * z);
          if ((page.flags[index] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
          samples += 1;
          const q = [brick[0] * oracle.plan.brickResolution + x,
            brick[1] * oracle.plan.brickResolution + y,
            brick[2] * oracle.plan.brickResolution + z] as const;
          const position = q.map((value, axis) =>
            oracle.plan.domainOrigin[axis] + (value + 0.5) * oracle.plan.fineCellWidth) as [number, number, number];
          const trace = traceFineLevelSetDeparture(position, dt, segmentCount, sampleVelocity);
          if (!trace.finite) { nonfiniteVelocitySamples += 1; page.workA[index] = Number.NaN; continue; }
          // Cell-centred phi has a constant (Neumann) ghost over the half-cell
          // between each boundary sample centre and its solid domain wall.
          // Keep genuinely out-of-domain and internally missing sparse pages
          // fail-closed; only clamp departures that remain inside the domain.
          const domainMaximum = oracle.plan.sampleDimensions.map((count, axis) =>
            oracle.plan.domainOrigin[axis] + count * oracle.plan.fineCellWidth);
          const insideDomain = trace.departure.every((value, axis) =>
            value >= oracle.plan.domainOrigin[axis] && value <= domainMaximum[axis]);
          const samplingDeparture = insideDomain && boundaryPolicy === "closed-neumann"
            ? trace.departure.map((value, axis) => Math.max(
              oracle.plan.domainOrigin[axis] + 0.5 * oracle.plan.fineCellWidth,
              Math.min(domainMaximum[axis] - 0.5 * oracle.plan.fineCellWidth, value),
            )) as [number, number, number]
            : trace.departure;
          const transported = oracle.sampleResidentTrilinear(samplingDeparture);
          if (transported === undefined) { departureOutsideResidentBand += 1; page.workA[index] = Number.NaN; continue; }
          page.workA[index] = transported;
        }
      }
    }
  }
  const committed = departureOutsideResidentBand === 0 && nonfiniteVelocitySamples === 0;
  if (committed) {
    for (const page of pages) page.phi.set(page.workA);
  }
  return { samples, segmentsPerSample: segmentCount, departureOutsideResidentBand, nonfiniteVelocitySamples, committed };
}

interface DistanceEntry {
  pageIndex: number;
  sampleIndex: number;
  q: [number, number, number];
  originalPhi: number;
  distance: number;
  seed: boolean;
}

const DIRECTIONS = [
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
] as const;

function coordinateKey(q: FineLevelSetVec3): string { return `${q[0]},${q[1]},${q[2]}`; }

function solveEikonal(axisDistances: readonly number[], h: number): number {
  const finite = axisDistances.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return Number.POSITIVE_INFINITY;
  let solution = finite[0] + h;
  if (finite.length >= 2 && solution > finite[1]) {
    const discriminant = Math.max(0, 2 * h * h - (finite[0] - finite[1]) ** 2);
    solution = (finite[0] + finite[1] + Math.sqrt(discriminant)) * 0.5;
  }
  if (finite.length >= 3 && solution > finite[2]) {
    const sum = finite[0] + finite[1] + finite[2];
    const squareSum = finite[0] ** 2 + finite[1] ** 2 + finite[2] ** 2;
    const discriminant = Math.max(0, sum * sum - 3 * (squareSum - h * h));
    solution = (sum + Math.sqrt(discriminant)) / 3;
  }
  return solution;
}

/** Ordered eight-direction sweeping on the uniform sparse fine lattice. */
export function redistanceFineLevelSet(
  oracle: FineLevelSetBrickOracle,
  options: FineLevelSetRedistanceOptions,
): FineLevelSetRedistanceDiagnostics {
  if (!Number.isFinite(options.physicalBandWidth) || options.physicalBandWidth <= 0) {
    throw new RangeError("Fine redistance band width must be finite and positive");
  }
  const residualTolerance = options.residualTolerance ?? 0.05;
  if (!Number.isFinite(residualTolerance) || residualTolerance <= 0) {
    throw new RangeError("Fine redistance residual tolerance must be finite and positive");
  }
  const maximumSweeps = options.maximumSweeps
    ?? Math.max(8, Math.ceil(3 * options.physicalBandWidth / oracle.plan.fineCellWidth) + 2);
  if (!Number.isSafeInteger(maximumSweeps) || maximumSweeps < 1 || maximumSweeps > 512) {
    throw new RangeError("Fine redistance maximum sweeps must be an integer in [1, 512]");
  }
  const pages = oracle.residentPages();
  const entries: DistanceEntry[] = [];
  const byCoordinate = new Map<string, DistanceEntry>();
  pages.forEach((page, pageIndex) => {
    const brick = unpackFineLevelSetBrickKey(oracle.plan, page.key);
    for (let z = 0; z < oracle.plan.brickResolution; z += 1) for (let y = 0; y < oracle.plan.brickResolution; y += 1) {
      for (let x = 0; x < oracle.plan.brickResolution; x += 1) {
        const sampleIndex = x + oracle.plan.brickResolution * (y + oracle.plan.brickResolution * z);
        if ((page.flags[sampleIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
        const q: [number, number, number] = [brick[0] * oracle.plan.brickResolution + x,
          brick[1] * oracle.plan.brickResolution + y, brick[2] * oracle.plan.brickResolution + z];
        const originalPhi = page.phi[sampleIndex];
        const entry = { pageIndex, sampleIndex, q, originalPhi,
          distance: originalPhi === 0 ? 0 : options.physicalBandWidth,
          seed: originalPhi === 0 };
        entries.push(entry); byCoordinate.set(coordinateKey(q), entry);
      }
    }
  });
  for (const entry of entries) {
    for (const direction of DIRECTIONS) {
      const neighbor = byCoordinate.get(coordinateKey(entry.q.map((value, axis) => value + direction[axis]) as [number, number, number]));
      if (!neighbor || (entry.originalPhi < 0) === (neighbor.originalPhi < 0)) continue;
      const denominator = Math.abs(entry.originalPhi) + Math.abs(neighbor.originalPhi);
      const seedDistance = denominator === 0 ? 0
        : oracle.plan.fineCellWidth * Math.abs(entry.originalPhi) / denominator;
      entry.distance = Math.min(entry.distance, seedDistance);
      entry.seed = true;
    }
    pages[entry.pageIndex].workB[entry.sampleIndex] = entry.originalPhi;
  }
  const seedCount = entries.reduce((count, entry) => count + (entry.seed ? 1 : 0), 0);
  const sweepOrders = Array.from({ length: 8 }, (_, order) => entries.slice().sort((a, b) => {
    for (let axis = 2; axis >= 0; axis -= 1) {
      const sign = (order & (1 << axis)) === 0 ? 1 : -1;
      const difference = sign * (a.q[axis] - b.q[axis]);
      if (difference !== 0) return difference;
    }
    return 0;
  }));
  let sweeps = 0;
  let converged = false;
  const changeTolerance = residualTolerance * oracle.plan.fineCellWidth;
  for (; sweeps < maximumSweeps; sweeps += 1) {
    let maximumChange = 0;
    for (const order of sweepOrders) for (const entry of order) {
      if (entry.seed) continue;
      const axisDistances = [0, 1, 2].map((axis) => {
        const minus = [...entry.q] as [number, number, number]; minus[axis] -= 1;
        const plus = [...entry.q] as [number, number, number]; plus[axis] += 1;
        return Math.min(byCoordinate.get(coordinateKey(minus))?.distance ?? Number.POSITIVE_INFINITY,
          byCoordinate.get(coordinateKey(plus))?.distance ?? Number.POSITIVE_INFINITY);
      });
      const candidate = Math.min(options.physicalBandWidth, solveEikonal(axisDistances, oracle.plan.fineCellWidth));
      if (candidate < entry.distance) {
        maximumChange = Math.max(maximumChange, Number.isFinite(entry.distance) ? entry.distance - candidate : options.physicalBandWidth);
        entry.distance = candidate;
      }
    }
    if (maximumChange <= changeTolerance && entries.every((entry) => Number.isFinite(entry.distance))) {
      converged = true; sweeps += 1; break;
    }
  }
  let unresolvedCells = 0;
  let maximumEikonalResidual = 0;
  for (const entry of entries) {
    if (!Number.isFinite(entry.distance)) { unresolvedCells += 1; continue; }
    if (!entry.seed && entry.distance < options.physicalBandWidth) {
      let sumSquares = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const minus = [...entry.q] as [number, number, number]; minus[axis] -= 1;
        const plus = [...entry.q] as [number, number, number]; plus[axis] += 1;
        const neighbor = Math.min(byCoordinate.get(coordinateKey(minus))?.distance ?? entry.distance,
          byCoordinate.get(coordinateKey(plus))?.distance ?? entry.distance);
        sumSquares += (Math.max(0, entry.distance - neighbor) / oracle.plan.fineCellWidth) ** 2;
      }
      const residual = Math.abs(Math.sqrt(sumSquares) - 1);
      maximumEikonalResidual = Math.max(maximumEikonalResidual, residual);
      if (residual > residualTolerance) unresolvedCells += 1;
    }
    pages[entry.pageIndex].workA[entry.sampleIndex] = entry.distance;
  }
  const committed = converged && unresolvedCells === 0 && seedCount > 0;
  if (committed) for (const entry of entries) {
    const page = pages[entry.pageIndex];
    page.phi[entry.sampleIndex] = (entry.originalPhi < 0 ? -1 : 1) * entry.distance;
    page.flags[entry.sampleIndex] = entry.distance >= options.physicalBandWidth
      ? 0
      : (page.flags[entry.sampleIndex]
        & ~(FINE_LEVELSET_SAMPLE_FLAGS.interface | FINE_LEVELSET_SAMPLE_FLAGS.negative))
        | (entry.seed ? FINE_LEVELSET_SAMPLE_FLAGS.interface : 0)
        | (entry.originalPhi < 0 ? FINE_LEVELSET_SAMPLE_FLAGS.negative : 0);
  }
  return { samples: entries.length, seeds: seedCount, sweeps, converged, unresolvedCells,
    maximumEikonalResidual, committed };
}
