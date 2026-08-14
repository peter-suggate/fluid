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
  collocateSparseAtlasVelocity,
  materializeSparseAtlasCollocatedVelocity,
  projectSparseAtlasVelocity,
  type SparseAtlasCompositeGrid,
  type SparseAtlasGradientRow,
  type SparseAtlasProjectionOptions,
  type SparseAtlasProjectionResult,
} from "./sparse-atlas-composite-projection";
import {
  BRICK_FINE_RESOLUTION,
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import { conditionSparseAtlasSurface } from "./sparse-atlas-surface-conditioning";
import {
  extrapolateSparseAtlasFaceVelocity,
  extrapolateSparseAtlasVelocity,
  transportSparseAtlasCM12,
} from "./sparse-atlas-cm12-transport";
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
  if (bricks.size === source.bricks.length) return source;
  return createSparseAdaptiveMassAtlas(source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key), source.generation);
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
): Float64Array {
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
  return Float64Array.from(nextGrid.gradientRows, (row) => {
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
    for (const candidateId of candidates) {
      const source = previousBounds[candidateId];
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

const transportGridCache = new WeakMap<object, Map<string, SparseAtlasCompositeGrid>>();

function transportGrid(
  sourceGrid: SparseAtlasCompositeGrid,
  supportAtlas: SparseAdaptiveMassAtlas,
  receiverResolution: SparseBrickResolution,
  haloBricks: number,
): SparseAtlasCompositeGrid {
  const topologyKey = sourceGrid.gradientRows as object;
  let variants = transportGridCache.get(topologyKey);
  if (!variants) {
    variants = new Map();
    transportGridCache.set(topologyKey, variants);
  }
  const variantKey = `${receiverResolution}:${haloBricks}`;
  const cached = variants.get(variantKey);
  if (cached && sameAtlasTopology(cached.atlas, supportAtlas)) {
    return rebindCompositeGrid(cached, supportAtlas);
  }
  const built = buildSparseAtlasCompositeGrid(supportAtlas);
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
    resolutionPolicy: initializeSparseAtlasResolutionPolicy(atlas),
    time_s,
  };
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
  const first = minimumFine.map((value) => Math.floor(value / BRICK_FINE_RESOLUTION));
  const last = maximumFine.map((value) => Math.floor(value / BRICK_FINE_RESOLUTION));
  for (let bz = first[2]; bz <= last[2]; bz += 1) {
    for (let by = first[1]; by <= last[1]; by += 1) {
      for (let bx = first[0]; bx <= last[0]; bx += 1) {
        const coordinate: SparseBrickVec3 = [bx, by, bz];
        const key = sparseBrickKey(coordinate, source.brickDimensions);
        const brick = bricks.get(key) ?? zeroBrick(key, coordinate, 8);
        const span = BRICK_FINE_RESOLUTION / brick.resolution;
        let density: Float64Array | undefined;
        for (let lz = 0; lz < brick.resolution; lz += 1) {
          for (let ly = 0; ly < brick.resolution; ly += 1) {
            for (let lx = 0; lx < brick.resolution; lx += 1) {
              const fraction = coverage([
                BRICK_FINE_RESOLUTION * bx + span * lx,
                BRICK_FINE_RESOLUTION * by + span * ly,
                BRICK_FINE_RESOLUTION * bz + span * lz,
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
  targetResolutionByBrick: ReadonlyMap<number, SparseBrickResolution>,
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
        source.dimensions, brick, nextDensity, target, 0,
      ),
      gamma: resampleBrickScalar(
        source.dimensions, brick, nextGamma, target, 1,
      ),
    });
  }
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    retained,
    source.generation + 1,
  );
}

function localCellVolume(
  dimensions: SparseBrickVec3,
  brick: SparseAdaptiveMassBrick,
  resolution: SparseBrickResolution,
  x: number,
  y: number,
  z: number,
): number {
  const scale = 8 / resolution;
  const local = [x, y, z] as const;
  let volume = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = brick.coordinate[axis] * 8 + local[axis] * scale;
    volume *= Math.max(0, Math.min(scale, dimensions[axis] - lower));
  }
  return volume;
}

function resampleBrickScalar(
  dimensions: SparseBrickVec3,
  brick: SparseAdaptiveMassBrick,
  source: ArrayLike<number>,
  targetResolution: SparseBrickResolution,
  emptyValue: number,
): Float64Array {
  const output = new Float64Array(targetResolution ** 3).fill(emptyValue);
  if (brick.resolution === 4 && targetResolution === 8) {
    for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        output[x + 8 * (y + 8 * z)] = source[Math.floor(x / 2) + 4
          * (Math.floor(y / 2) + 4 * Math.floor(z / 2))];
      }
    }
    return output;
  }
  if (brick.resolution === 8 && targetResolution === 4) {
    for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) {
        let weighted = 0;
        let volume = 0;
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const sx = 2 * x + dx, sy = 2 * y + dy, sz = 2 * z + dz;
            const childVolume = localCellVolume(dimensions, brick, 8, sx, sy, sz);
            weighted += childVolume * source[sx + 8 * (sy + 8 * sz)];
            volume += childVolume;
          }
        }
        output[x + 4 * (y + 4 * z)] = volume > 0 ? weighted / volume : emptyValue;
      }
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
  const acceleration = options.accelerationFinePerSecond2 ?? [0, 0, 0];
  assertFiniteVector(acceleration, "accelerationFinePerSecond2");
  const epsilon = options.emptyEpsilon ?? 1e-5;
  if (!(epsilon >= 0)) {
    throw new RangeError("emptyEpsilon must be nonnegative");
  }

  const receiverResolution: SparseBrickResolution =
    options.resolutionMode === "all-fine" ? 8 : 4;
  const receiverCellSpanFine = BRICK_FINE_RESOLUTION / receiverResolution;
  const maximumFaceComponent = [0, 0, 0];
  for (const row of source.grid.gradientRows) {
    maximumFaceComponent[row.axis] = Math.max(maximumFaceComponent[row.axis],
      Math.abs(source.faceNormalVelocity[row.id]));
  }
  const maximumCharacteristicDisplacementFine = dt_s * Math.hypot(
    maximumFaceComponent[0], maximumFaceComponent[1], maximumFaceComponent[2],
  );
  // CM12 is intentionally useful at large CFL. Cover every brick reachable
  // by the characteristic plus one finest cell for trilinear interpolation;
  // never scan or allocate the rest of the authored domain.
  const transportHaloBricks = Math.max(1, Math.ceil(
    (maximumCharacteristicDisplacementFine + receiverCellSpanFine)
      / BRICK_FINE_RESOLUTION,
  ));
  const supportAtlas = transportSupport(
    source.atlas, receiverResolution, transportHaloBricks,
  );
  const receiverTopologyUnchanged = supportAtlas === source.atlas;
  const workGrid = receiverTopologyUnchanged
    ? source.grid
    : transportGrid(
      source.grid, supportAtlas, receiverResolution, transportHaloBricks,
    );
  const remappedCellVelocity = receiverTopologyUnchanged
    ? source.cellVelocity
    : remapCellVelocity(source, workGrid);
  options.onStageComplete?.("receiver-topology");
  const workDensity = Float64Array.from(workGrid.cells, (cell) => cell.density);
  const transportVelocity = extrapolateSparseAtlasVelocity(
    workGrid, workDensity, remappedCellVelocity,
  );
  const remappedFaces = receiverTopologyUnchanged
    ? source.faceNormalVelocity
    : remapFaceVelocity(
      source.grid,
      source.faceNormalVelocity,
      workGrid,
      facesFromCells(workGrid, transportVelocity),
    );
  const transportFaces = extrapolateSparseAtlasFaceVelocity(
    workGrid, workDensity, remappedFaces,
    facesFromCells(workGrid, transportVelocity),
    Math.max(
      2 * receiverCellSpanFine,
      maximumCharacteristicDisplacementFine + receiverCellSpanFine,
    ),
  );
  const faceCollocatedVelocity = collocateSparseAtlasVelocity(
    workGrid, transportFaces,
  );
  let fields: TransportFields = {
    density: workDensity,
    gamma: Float64Array.from(workGrid.cells, (cell) => cell.gamma),
    velocity: faceCollocatedVelocity,
    faceNormalVelocity: transportFaces,
  };
  const massBefore = integral(workGrid, fields.density);
  const gammaBefore = integral(workGrid, fields.gamma);
  const energyBefore = kineticEnergy(workGrid, fields.density, fields.velocity);
  let outflowRate = 0;
  for (let id = 0; id < workGrid.cells.length; id += 1) {
    outflowRate = Math.max(outflowRate, Math.hypot(
      fields.velocity[3 * id], fields.velocity[3 * id + 1],
      fields.velocity[3 * id + 2],
    ));
  }
  const transportSubsteps = 1;
  fields = transportSparseAtlasCM12(workGrid, fields, dt_s).fields;
  const [, maximumDensityAfterTransport] = extrema(fields.density, 0);
  options.onStageComplete?.("coupled-transport");

  // CM12 Secs. 3.4-3.5 conditioning is part of the method, not presentation
  // polish. Run it on resident composite rows with the paper's 3dt dose.
  const conditioned = conditionSparseAtlasSurface(workGrid, fields, {
    gammaDiffusionIterations: 1,
    timeStep_s: dt_s,
  });
  fields = {
    density: conditioned.fields.density,
    gamma: conditioned.fields.gamma,
    velocity: fields.velocity,
    faceNormalVelocity: fields.faceNormalVelocity,
  };
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
  const grid = sourceTopologyRetained
    ? rebindCompositeGrid(source.grid, atlas)
    : workTopologyRetained
      ? rebindCompositeGrid(workGrid, atlas)
      : buildSparseAtlasCompositeGrid(atlas);
  const advectedCellVelocity = workTopologyRetained
    ? fields.velocity
    : remapWorkVelocityToOutput(workGrid, fields.velocity, fields.density, grid);
  const reconstructedFaces = facesFromCells(grid, advectedCellVelocity);
  const advectedFaces = workTopologyRetained
    ? fields.faceNormalVelocity ?? reconstructedFaces
    : remapFaceVelocity(
      workGrid,
      fields.faceNormalVelocity ?? facesFromCells(workGrid, fields.velocity),
      grid,
      reconstructedFaces,
    );
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
    targetDivergence: Float64Array.from(grid.cells, (cell) =>
      cm12VolumeCorrectionDivergence(
        cell.density,
        (options.finestCellSize_m ?? 1) * Math.min(...cell.widthsFine),
        dt_s,
      )),
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
    resolutionPolicy: retainSparseAtlasResolutionPolicy(
      resolutionDecision.state,
      atlas,
    ),
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
      maximumDensityAfterTransport,
      minimumGamma,
      maximumGamma,
      resolutionPolicy: resolutionDecision.receipt,
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
