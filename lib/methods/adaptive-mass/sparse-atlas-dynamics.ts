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
  buildSparseAtlasCompositeGrid,
  materializeSparseAtlasCollocatedVelocity,
  projectSparseAtlasVelocity,
  type SparseAtlasCompositeGrid,
  type SparseAtlasGradientRow,
  type SparseAtlasProjectionOptions,
  type SparseAtlasProjectionResult,
} from "./sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";

export interface SparseAtlasDynamicsState {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  /** Interleaved XYZ, in finest cells / second, in `grid.cells` order. */
  readonly cellVelocity: Float64Array;
  /** One oriented normal velocity for every authoritative composite row. */
  readonly faceNormalVelocity: Float64Array;
  /** Latest composite pressure potential in `grid.cells` order. */
  readonly cellPressure: Float64Array;
  readonly time_s: number;
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
  | "retain-rebuild"
  | "force"
  | "projection";

export interface SparseAtlasDynamicsStepOptions {
  readonly dt_s: number;
  /** Finest-cell units / second squared. Defaults to zero. */
  readonly accelerationFinePerSecond2?: SparseBrickVec3;
  /** Donor-cell outgoing-volume ceiling per transport substep. */
  readonly maximumCfl?: number;
  readonly emptyEpsilon?: number;
  /** Pressure is enabled by default. */
  readonly project?: boolean;
  readonly projection?: Omit<SparseAtlasProjectionOptions, "normalVelocity" | "phi">;
  /** Optional level-set override in the retained, post-transport grid's cell order. */
  readonly phi?: ArrayLike<number>;
  readonly maximumTransportSubsteps?: number;
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
  readonly minimumGamma: number;
  readonly maximumGamma: number;
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
  if (value.some((component) => !Number.isFinite(component))) {
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

/**
 * Add transient receiver bricks only where a resident sparse-air face has
 * outward flow. New support is always 4³: interface bricks remain 8³ while
 * low-activity receiver space is genuinely coarse, so the live expansion path
 * exercises the same 2:1 ports as projection instead of silently refining its
 * entire halo. Face-local activation is D4-invariant and performs no transport
 * work for omitted bricks whose boundary flux is zero or points inward.
 */
function transportSupport(
  source: SparseAdaptiveMassAtlas,
  sourceGrid: SparseAtlasCompositeGrid,
  sourceFaceVelocity: ArrayLike<number>,
): SparseAdaptiveMassAtlas {
  if (source.bricks.length === 0) {
    return createSparseAdaptiveMassAtlas(source.dimensions, [], source.generation);
  }
  const bricks = new Map<number, SparseAdaptiveMassBrick>();
  for (const brick of source.bricks) bricks.set(brick.key, brick);
  const requested = new Map<number, SparseBrickVec3>();
  for (const row of sourceGrid.gradientRows) {
    if (row.kind !== "sparse-air" || row.terms.length !== 1) continue;
    const velocity = sourceFaceVelocity[row.id];
    const term = row.terms[0];
    // The sign of G's sole coefficient identifies the resident side. Flow is
    // outward precisely when oriented velocity and that coefficient disagree.
    if (!(velocity * term.coefficient < -1e-14)
      || sourceGrid.cells[term.cellId].density <= 0) continue;
    const residentKey = row.negativeBrickKey ?? row.positiveBrickKey;
    if (residentKey === undefined) continue;
    const resident = source.directory.get(residentKey);
    if (!resident) continue;
    const coordinate = [...resident.coordinate] as [number, number, number];
    coordinate[row.axis] += row.negativeBrickKey === residentKey ? 1 : -1;
    if (coordinate.some((value, axis) =>
      value < 0 || value >= source.brickDimensions[axis])) continue;
    const key = sparseBrickKey(coordinate, source.brickDimensions);
    if (!bricks.has(key)) requested.set(key, coordinate);
  }
  if (requested.size === 0) return source;
  for (const [key, coordinate] of requested) {
    bricks.set(key, zeroBrick(key, coordinate, 4));
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation,
  );
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
): Float64Array {
  if (cellVelocity.length !== 3 * grid.cells.length) {
    throw new RangeError("cellVelocity must contain interleaved XYZ for every grid cell");
  }
  return Float64Array.from(grid.gradientRows, (row) =>
    collocatedFaceVelocity(row, cellVelocity));
}

function tangentialAxes(axis: 0 | 1 | 2): readonly [0 | 1 | 2, 0 | 1 | 2] {
  if (axis === 0) return [1, 2];
  if (axis === 1) return [0, 2];
  return [0, 1];
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
): Float64Array {
  const bins = new Map<string, number[]>();
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
  for (const row of previousGrid.gradientRows) {
    const box = bounds(row);
    for (let v = Math.floor(box.minimumV); v < Math.ceil(box.maximumV); v += 1) {
      for (let u = Math.floor(box.minimumU); u < Math.ceil(box.maximumU); u += 1) {
        const key = `${row.axis}:${row.centerFine[row.axis]}:${u}:${v}`;
        const entries = bins.get(key);
        if (entries) entries.push(row.id);
        else bins.set(key, [row.id]);
      }
    }
  }
  return Float64Array.from(nextGrid.gradientRows, (row) => {
    const box = bounds(row);
    const candidates = new Set<number>();
    for (let v = Math.floor(box.minimumV); v < Math.ceil(box.maximumV); v += 1) {
      for (let u = Math.floor(box.minimumU); u < Math.ceil(box.maximumU); u += 1) {
        for (const id of bins.get(`${row.axis}:${row.centerFine[row.axis]}:${u}:${v}`) ?? []) {
          candidates.add(id);
        }
      }
    }
    let weighted = 0, overlapArea = 0;
    for (const candidateId of candidates) {
      const candidate = previousGrid.gradientRows[candidateId];
      const source = bounds(candidate);
      const overlapU = Math.max(0,
        Math.min(box.maximumU, source.maximumU) - Math.max(box.minimumU, source.minimumU));
      const overlapV = Math.max(0,
        Math.min(box.maximumV, source.maximumV) - Math.max(box.minimumV, source.minimumV));
      const overlap = overlapU * overlapV;
      weighted += overlap * previousVelocity[candidateId];
      overlapArea += overlap;
    }
    return overlapArea > 0 ? weighted / overlapArea : fallback[row.id];
  });
}

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

/** Reuse immutable geometry/row topology when only density and gamma changed. */
function rebindCompositeGrid(
  topology: SparseAtlasCompositeGrid,
  atlas: SparseAdaptiveMassAtlas,
): SparseAtlasCompositeGrid {
  const cells = topology.cells.map((cell) => {
    const brick = atlas.directory.get(cell.brickKey);
    if (!brick || brick.resolution !== cell.brickResolution) {
      throw new Error("cannot rebind composite grid across a topology change");
    }
    return {
      ...cell,
      density: brick.density[cell.localIndex],
      gamma: brick.gamma[cell.localIndex],
    };
  });
  return { ...topology, atlas, cells };
}

function remapCellScalar(
  previousGrid: SparseAtlasCompositeGrid,
  previousValues: ArrayLike<number>,
  nextGrid: SparseAtlasCompositeGrid,
): Float64Array {
  if (previousValues.length !== previousGrid.cells.length) {
    throw new RangeError("previous scalar must contain one value per grid cell");
  }
  const byStableLeaf = new Map<number, number>();
  for (const cell of previousGrid.cells) {
    byStableLeaf.set(cell.stableLeafId, previousValues[cell.id]);
  }
  return Float64Array.from(nextGrid.cells, (cell) =>
    byStableLeaf.get(cell.stableLeafId) ?? 0);
}

export function initializeSparseAtlasDynamics(
  atlas: SparseAdaptiveMassAtlas,
  options: SparseAtlasDynamicsInitializationOptions = {},
): SparseAtlasDynamicsState {
  const grid = buildSparseAtlasCompositeGrid(atlas);
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
  return {
    atlas,
    grid,
    cellVelocity,
    faceNormalVelocity: facesFromCells(grid, cellVelocity),
    cellPressure: new Float64Array(grid.cells.length),
    time_s,
  };
}

interface SideTerm {
  readonly cellId: number;
  readonly fraction: number;
}

function rowSides(row: SparseAtlasGradientRow): {
  negative: readonly SideTerm[];
  positive: readonly SideTerm[];
} {
  const negative: SideTerm[] = [], positive: SideTerm[] = [];
  let negativeTotal = 0, positiveTotal = 0;
  for (const term of row.terms) {
    const raw = Math.abs(term.coefficient) * row.distance;
    if (term.coefficient < 0) {
      negative.push({ cellId: term.cellId, fraction: raw });
      negativeTotal += raw;
    } else {
      positive.push({ cellId: term.cellId, fraction: raw });
      positiveTotal += raw;
    }
  }
  return {
    negative: negativeTotal > 0
      ? negative.map((term) => ({ ...term, fraction: term.fraction / negativeTotal })) : [],
    positive: positiveTotal > 0
      ? positive.map((term) => ({ ...term, fraction: term.fraction / positiveTotal })) : [],
  };
}

function maximumOutflowRate(
  grid: SparseAtlasCompositeGrid,
  faceVelocity: ArrayLike<number>,
): number {
  const rates = new Float64Array(grid.cells.length);
  for (const row of grid.gradientRows) {
    const velocity = faceVelocity[row.id];
    if (velocity === 0) continue;
    const sides = rowSides(row);
    const donors = velocity > 0 ? sides.negative : sides.positive;
    if (donors.length === 0 || (velocity > 0 ? sides.positive : sides.negative).length === 0) continue;
    const rate = Math.abs(velocity) * row.area;
    for (const donor of donors) rates[donor.cellId] += rate * donor.fraction;
  }
  let maximum = 0;
  for (const cell of grid.cells) maximum = Math.max(maximum, rates[cell.id] / cell.volume);
  return maximum;
}

interface TransportFields {
  density: Float64Array;
  gamma: Float64Array;
  velocity: Float64Array;
}

function transportSubstep(
  grid: SparseAtlasCompositeGrid,
  faceVelocity: ArrayLike<number>,
  fields: TransportFields,
  dt_s: number,
): TransportFields {
  const count = grid.cells.length;
  const mass = Float64Array.from(grid.cells, (cell) =>
    cell.volume * fields.density[cell.id]);
  const gammaIntegral = Float64Array.from(grid.cells, (cell) =>
    cell.volume * fields.gamma[cell.id]);
  const momentum = new Float64Array(3 * count);
  for (const cell of grid.cells) {
    const cellMass = mass[cell.id];
    for (let axis = 0; axis < 3; axis += 1) {
      momentum[3 * cell.id + axis] = cellMass * fields.velocity[3 * cell.id + axis];
    }
  }
  const massDelta = new Float64Array(count);
  const gammaDelta = new Float64Array(count);
  const momentumDelta = new Float64Array(3 * count);

  for (const row of grid.gradientRows) {
    const normalVelocity = faceVelocity[row.id];
    if (normalVelocity === 0) continue;
    const sides = rowSides(row);
    const donors = normalVelocity > 0 ? sides.negative : sides.positive;
    const receivers = normalVelocity > 0 ? sides.positive : sides.negative;
    // A one-sided sparse-air row is the exterior of the transient support,
    // never a mass sink. The one-brick halo and CFL bound keep wet material
    // from reaching it during this step.
    if (donors.length === 0 || receivers.length === 0) continue;
    const sweptVolume = Math.abs(normalVelocity) * row.area * dt_s;
    let transferredMass = 0, transferredGamma = 0;
    const transferredMomentum = [0, 0, 0];
    for (const donor of donors) {
      const volume = sweptVolume * donor.fraction;
      const density = fields.density[donor.cellId];
      const gamma = fields.gamma[donor.cellId];
      const donorMass = volume * density;
      const donorGamma = volume * gamma;
      massDelta[donor.cellId] -= donorMass;
      gammaDelta[donor.cellId] -= donorGamma;
      transferredMass += donorMass;
      transferredGamma += donorGamma;
      for (let axis = 0; axis < 3; axis += 1) {
        const amount = donorMass * fields.velocity[3 * donor.cellId + axis];
        momentumDelta[3 * donor.cellId + axis] -= amount;
        transferredMomentum[axis] += amount;
      }
    }
    for (const receiver of receivers) {
      massDelta[receiver.cellId] += receiver.fraction * transferredMass;
      gammaDelta[receiver.cellId] += receiver.fraction * transferredGamma;
      for (let axis = 0; axis < 3; axis += 1) {
        momentumDelta[3 * receiver.cellId + axis]
          += receiver.fraction * transferredMomentum[axis];
      }
    }
  }

  const density = new Float64Array(count);
  const gamma = new Float64Array(count);
  const velocity = new Float64Array(3 * count);
  for (const cell of grid.cells) {
    const nextMass = mass[cell.id] + massDelta[cell.id];
    density[cell.id] = nextMass / cell.volume;
    gamma[cell.id] = (gammaIntegral[cell.id] + gammaDelta[cell.id]) / cell.volume;
    if (nextMass <= 1e-30) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      velocity[3 * cell.id + axis] =
        (momentum[3 * cell.id + axis] + momentumDelta[3 * cell.id + axis]) / nextMass;
    }
  }
  return { density, gamma, velocity };
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

function extrema(values: ArrayLike<number>, empty: number): readonly [number, number] {
  if (values.length === 0) return [empty, empty];
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  return [minimum, maximum];
}

function retainedAtlas(
  source: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  gamma: ArrayLike<number>,
  epsilon: number,
): SparseAdaptiveMassAtlas {
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
    if (wet) retained.push({ ...brick, density: nextDensity, gamma: nextGamma });
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    retained,
    source.generation + 1,
  );
}

function remapWorkVelocityToOutput(
  workGrid: SparseAtlasCompositeGrid,
  workVelocity: ArrayLike<number>,
  outputGrid: SparseAtlasCompositeGrid,
): Float64Array {
  const result = new Float64Array(3 * outputGrid.cells.length);
  let workIndex = 0;
  for (const cell of outputGrid.cells) {
    while (workIndex < workGrid.cells.length
      && workGrid.cells[workIndex].stableLeafId < cell.stableLeafId) workIndex += 1;
    const source = workGrid.cells[workIndex];
    if (!source || source.stableLeafId !== cell.stableLeafId) continue;
    result[3 * cell.id] = workVelocity[3 * source.id];
    result[3 * cell.id + 1] = workVelocity[3 * source.id + 1];
    result[3 * cell.id + 2] = workVelocity[3 * source.id + 2];
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
  const acceleration = options.accelerationFinePerSecond2 ?? [0, 0, 0];
  assertFiniteVector(acceleration, "accelerationFinePerSecond2");
  const maximumCfl = options.maximumCfl ?? 0.8;
  const maximumTransportSubsteps = options.maximumTransportSubsteps ?? 256;
  const epsilon = options.emptyEpsilon ?? 0;
  if (!(maximumCfl > 0 && maximumCfl <= 1)
    || !Number.isInteger(maximumTransportSubsteps) || maximumTransportSubsteps <= 0
    || !(epsilon >= 0)) {
    throw new RangeError("invalid transport stability options");
  }

  const supportAtlas = transportSupport(
    source.atlas,
    source.grid,
    source.faceNormalVelocity,
  );
  const receiverTopologyUnchanged = supportAtlas === source.atlas;
  const workGrid = receiverTopologyUnchanged
    ? source.grid
    : buildSparseAtlasCompositeGrid(supportAtlas);
  const remappedCellVelocity = receiverTopologyUnchanged
    ? source.cellVelocity
    : remapCellVelocity(source, workGrid);
  const remappedFaces = receiverTopologyUnchanged
    ? source.faceNormalVelocity
    : remapFaceVelocity(
      source.grid,
      source.faceNormalVelocity,
      workGrid,
      facesFromCells(workGrid, remappedCellVelocity),
    );
  options.onStageComplete?.("receiver-topology");
  let fields: TransportFields = {
    density: Float64Array.from(workGrid.cells, (cell) => cell.density),
    gamma: Float64Array.from(workGrid.cells, (cell) => cell.gamma),
    velocity: remappedCellVelocity,
  };
  const massBefore = integral(workGrid, fields.density);
  const gammaBefore = integral(workGrid, fields.gamma);
  const energyBefore = kineticEnergy(workGrid, fields.density, fields.velocity);
  const outflowRate = maximumOutflowRate(workGrid, remappedFaces);
  const transportSubsteps = Math.max(1, Math.ceil(dt_s * outflowRate / maximumCfl));
  if (transportSubsteps > maximumTransportSubsteps) {
    throw new RangeError(`transport requires ${transportSubsteps} CFL substeps; cap is ${maximumTransportSubsteps}`);
  }
  const subDt_s = dt_s / transportSubsteps;
  for (let substep = 0; substep < transportSubsteps; substep += 1) {
    fields = transportSubstep(workGrid, remappedFaces, fields, subDt_s);
  }
  options.onStageComplete?.("coupled-transport");

  const atlas = retainedAtlas(source.atlas, workGrid, fields.density, fields.gamma, epsilon);
  const sourceTopologyRetained = sameAtlasTopology(source.atlas, atlas);
  const workTopologyRetained = sameAtlasTopology(workGrid.atlas, atlas);
  const grid = sourceTopologyRetained
    ? rebindCompositeGrid(source.grid, atlas)
    : workTopologyRetained
      ? rebindCompositeGrid(workGrid, atlas)
      : buildSparseAtlasCompositeGrid(atlas);
  const advectedCellVelocity = workTopologyRetained
    ? fields.velocity
    : remapWorkVelocityToOutput(workGrid, fields.velocity, grid);
  const advectedFaces = facesFromCells(grid, advectedCellVelocity);
  options.onStageComplete?.("retain-rebuild");
  const forcedFaces = Float64Array.from(advectedFaces, (value, rowId) =>
    value + dt_s * acceleration[grid.gradientRows[rowId].axis]);
  const forcedCells = advectedCellVelocity.slice();
  for (const cell of grid.cells) {
    for (let axis = 0; axis < 3; axis += 1) {
      forcedCells[3 * cell.id + axis] += dt_s * acceleration[axis];
    }
  }
  options.onStageComplete?.("force");
  const initialPressure = remapCellScalar(source.grid, source.cellPressure, grid);
  // Projection is deliberately last: the persistent/public state is the same
  // divergence-free state accepted by the pressure gate, never a pre-advection
  // diagnostic that transport has subsequently invalidated.
  const projection = options.project === false ? undefined : projectSparseAtlasVelocity(grid, {
    ...options.projection,
    normalVelocity: forcedFaces,
    initialPressure,
    phi: options.phi,
  });
  if (projection) options.onStageComplete?.("projection");
  const cellVelocity = projection?.leafCollocatedVelocity ?? forcedCells;
  const faceNormalVelocity = projection?.projectedFaceVelocity ?? forcedFaces;
  const state: SparseAtlasDynamicsState = {
    atlas,
    grid,
    cellVelocity,
    faceNormalVelocity,
    cellPressure: projection?.leafPressure ?? initialPressure,
    time_s: source.time_s + dt_s,
  };
  const massAfter = integral(workGrid, fields.density);
  const gammaAfter = integral(workGrid, fields.gamma);
  const [minimumDensity, maximumDensity] = extrema(fields.density, 0);
  const [minimumGamma, maximumGamma] = extrema(fields.gamma, 1);
  return {
    state,
    atlas,
    workGrid,
    projection,
    stats: {
      dt_s,
      transportSubsteps,
      maximumOutgoingCfl: dt_s * outflowRate / transportSubsteps,
      sourceBrickCount: source.atlas.bricks.length,
      transientSupportBrickCount: supportAtlas.bricks.length - source.atlas.bricks.length,
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
      kineticEnergyAfter: kineticEnergy(
        grid,
        Float64Array.from(grid.cells, (cell) => cell.density),
        cellVelocity,
      ),
      minimumDensity,
      maximumDensity,
      minimumGamma,
      maximumGamma,
    },
  };
}

/** Dense XYZ finest-cell publication, still in finest cells / second. */
export function materializeSparseAtlasDynamicsVelocity(
  state: SparseAtlasDynamicsState,
): Float32Array {
  return materializeSparseAtlasCollocatedVelocity(state.grid, state.cellVelocity);
}

/** Dense RGBA publication convenient for rgba32float GPU textures. */
export function materializeSparseAtlasDynamicsVelocityRgba(
  state: SparseAtlasDynamicsState,
): Float32Array {
  const xyz = materializeSparseAtlasDynamicsVelocity(state);
  const result = new Float32Array(4 * xyz.length / 3);
  for (let source = 0, destination = 0; source < xyz.length; source += 3, destination += 4) {
    result[destination] = xyz[source];
    result[destination + 1] = xyz[source + 1];
    result[destination + 2] = xyz[source + 2];
  }
  return result;
}
