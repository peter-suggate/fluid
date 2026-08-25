/**
 * Finite-volume pressure/projection algebra for an arbitrary sparse 4^3/8^3
 * brick atlas.
 *
 * Every operation is derived from `gradientRows`. A mixed 2:1 brick face is
 * represented by one row per coarse face port. The cell coefficients are the
 * area-weighted averages on each side of that port, exactly generalising the
 * frozen two-tile operator. Consequently D = -M^-1 G^T W and
 * A = G^T W G (or G_l^T W/theta G_l at a free surface) are identities rather
 * than separately maintained stencils.
 *
 * Missing bricks inside the bounded atlas are sparse air. They contribute
 * one-sided Dirichlet rows only when their resident neighbour is liquid.
 * Faces on the outer domain boundary are solid/no-flow and have no row. Thus
 * empty bricks allocate neither cells nor payload, while their boundary costs
 * only the ports of adjacent resident bricks.
 */

import {
  CM12_LIQUID_ISOVALUE,
  cm12GhostFluidTheta,
} from "../../core/cm12-numerics";
import {
  pointInsideSphericalContainerFine,
  sphericalContainerOpenFractionAtFineBox,
  sphericalContainerOpenFractionAtFineFace,
} from "../../core/spherical-container";
import { tankWallOpeningFraction } from "../../core/tank-wall-field";
import {
  sparseBrickSpan,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";

export type SparseAtlasAxis = 0 | 1 | 2;

/** Trailing seams for the composite projection's independently timed work. */
export type SparseAtlasProjectionStageId =
  | "topology"
  | "rhs"
  | "solve"
  | "projection"
  | "diagnostics";

export interface SparseAtlasCompositeCell {
  readonly id: number;
  readonly stableLeafId: number;
  readonly brickKey: number;
  readonly brickCoordinate: SparseBrickVec3;
  readonly brickResolution: SparseBrickResolution;
  readonly local: SparseBrickVec3;
  readonly localIndex: number;
  readonly minimumFine: SparseBrickVec3;
  readonly maximumFine: SparseBrickVec3;
  readonly centerFine: SparseBrickVec3;
  readonly widthsFine: SparseBrickVec3;
  readonly volume: number;
  /** Alias documenting that `volume` is measured in finest-cell volumes. */
  readonly volumeFineCells: number;
  readonly openFraction: number;
  readonly openVolume: number;
  readonly separatingPressureMinimum: boolean;
  readonly density: number;
  readonly gamma: number;
}

export interface SparseAtlasGradientTerm {
  readonly cellId: number;
  readonly coefficient: number;
}

export type SparseAtlasGradientRowKind = "intra-brick" | "brick-face" | "mixed-seam" | "sparse-air";

export interface SparseAtlasGradientRow {
  readonly id: number;
  readonly kind: SparseAtlasGradientRowKind;
  readonly axis: SparseAtlasAxis;
  readonly centerFine: SparseBrickVec3;
  readonly area: number;
  readonly geometricArea: number;
  readonly openFraction: number;
  readonly pressureDualOpenFraction: number;
  readonly distance: number;
  readonly areaFineCells2: number;
  readonly centerDistanceFine: number;
  readonly dualWeight: number;
  readonly terms: readonly SparseAtlasGradientTerm[];
  readonly negativeBrickKey?: number;
  readonly positiveBrickKey?: number;
  /** Positive level-set sample for an omitted in-domain air brick. */
  readonly exteriorPhi?: number;
}

export interface SparseAtlasCompositeGrid {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly cells: readonly SparseAtlasCompositeCell[];
  readonly gradientRows: readonly SparseAtlasGradientRow[];
  readonly cellBaseByBrick: ReadonlyMap<number, number>;
  readonly mixedSeamRowCount: number;
  readonly sparseAirRowCount: number;
  /** Identity of this topology epoch; field-only rebinds preserve it. */
  readonly topologyKey?: object;
}

type MutableCompositeCell = { -readonly [Key in keyof SparseAtlasCompositeCell]:
SparseAtlasCompositeCell[Key] };
type MutableGradientRow = { -readonly [Key in keyof SparseAtlasGradientRow]:
SparseAtlasGradientRow[Key] } & { termPool: SparseAtlasGradientTerm[] };

export interface SparseAtlasCompositeGridBuildWorkspace {
  readonly cellPool: MutableCompositeCell[];
  readonly rowPool: MutableGradientRow[];
  readonly cellBaseByBrick: Map<number, number>;
  readonly termCellScratch: Int32Array;
  readonly termCoefficientScratch: Float64Array;
  readonly cells: SparseAtlasCompositeCell[];
  readonly rows: SparseAtlasGradientRow[];
  grid?: SparseAtlasCompositeGrid;
}

export function createSparseAtlasCompositeGridBuildWorkspace():
SparseAtlasCompositeGridBuildWorkspace {
  return {
    cellPool: [], rowPool: [], cellBaseByBrick: new Map(),
    // A span-one candidate beside an immutable macro can temporarily expose
    // more than the eight terms of an ordinary 2:1 row while the template
    // library enumerates levels that candidate validation will reject. Keep
    // the builder lossless through the largest 8x8 face plus its macro term.
    termCellScratch: new Int32Array(514),
    termCoefficientScratch: new Float64Array(514),
    cells: [], rows: [],
  };
}

const COMPOSITE_CELL_POOL_CHUNK = 4096;
const COMPOSITE_ROW_POOL_CHUNK = 8192;
const pooledVector = (): [number, number, number] => [0, 0, 0];

function reserveCompositeCells(
  workspace: SparseAtlasCompositeGridBuildWorkspace | undefined,
  count: number,
): void {
  if (!workspace) return;
  const capacity = Math.ceil(count / COMPOSITE_CELL_POOL_CHUNK)
    * COMPOSITE_CELL_POOL_CHUNK;
  while (workspace.cellPool.length < capacity) {
    workspace.cellPool.push({
      id: 0, stableLeafId: 0, brickKey: 0, brickCoordinate: pooledVector(),
      brickResolution: 8, local: pooledVector(), localIndex: 0,
      minimumFine: pooledVector(), maximumFine: pooledVector(), centerFine: pooledVector(),
      widthsFine: pooledVector(), volume: 0, volumeFineCells: 0,
      openFraction: 1, openVolume: 0, separatingPressureMinimum: false,
      density: 0, gamma: 1,
    });
  }
}

function reserveCompositeRows(
  workspace: SparseAtlasCompositeGridBuildWorkspace | undefined,
  count: number,
): void {
  if (!workspace) return;
  const capacity = Math.ceil(count / COMPOSITE_ROW_POOL_CHUNK)
    * COMPOSITE_ROW_POOL_CHUNK;
  while (workspace.rowPool.length < capacity) {
    const first = { cellId: 0, coefficient: 0 };
    const second = { cellId: 0, coefficient: 0 };
    workspace.rowPool.push({
      id: 0, kind: "intra-brick", axis: 0, centerFine: pooledVector(),
      area: 0, distance: 0, areaFineCells2: 0, centerDistanceFine: 0,
      geometricArea: 0, openFraction: 1, pressureDualOpenFraction: 1,
      dualWeight: 0, terms: [], termPool: [first, second],
    });
  }
}

export interface SparseAtlasProjectionOptions {
  /** One oriented velocity per `grid.gradientRows` entry. Defaults to zero. */
  readonly normalVelocity?: ArrayLike<number>;
  /** Desired liquid divergence in 1/s, one value per compact cell. */
  readonly targetDivergence?: ArrayLike<number>;
  /** Optional warm start in compact `grid.cells` order. */
  readonly initialPressure?: ArrayLike<number>;
  /** Cell-centred level set, negative in liquid. Defaults to isovalue-density. */
  readonly phi?: ArrayLike<number>;
  readonly relativeTolerance?: number;
  readonly absoluteTolerance?: number;
  readonly maximumIterations?: number;
  readonly denominatorEpsilon?: number;
  /** Phi assigned to omitted, in-domain bricks. Defaults to +0.5. */
  readonly sparseAirPhi?: number;
  /** Optional timing seam; the projection remains independent of any clock. */
  readonly onStageComplete?: (stage: SparseAtlasProjectionStageId) => void;
}

export interface SparseAtlasProjectionReceipt {
  readonly iterations: number;
  readonly converged: boolean;
  readonly componentCount: number;
  readonly anchoredComponentCount: number;
  readonly unanchoredComponentCount: number;
  readonly liquidCellCount: number;
  readonly activeRowCount: number;
  readonly cutRowCount: number;
  readonly mixedSeamRowCount: number;
  readonly cutMixedSeamRowCount: number;
  readonly thetaClampCount: number;
  readonly minimumTheta: number;
  readonly rhsCompatibilityMaxAbs: number;
  /** True `b - A p` norm before the first PCG iteration. */
  readonly initialRelativeResidualL2: number;
  /** The same frozen system from a zero pressure seed. */
  readonly zeroSeedInitialRelativeResidualL2: number;
  readonly initialMaximumResidual: number;
  readonly relativeResidualL2: number;
  readonly maximumResidual: number;
  readonly preDivergenceVolumeL2: number;
  readonly postDivergenceVolumeL2: number;
  readonly preDivergenceMaximum: number;
  readonly postDivergenceMaximum: number;
  readonly divergenceReduction: number;
  readonly kineticEnergyBefore: number;
  readonly kineticEnergyAfter: number;
  readonly pressureCorrectionEnergy: number;
  readonly energyIdentityAbsError: number;
  /** Pressure-inactive faces are outside the liquid velocity authority. */
  readonly inactiveRowCount: number;
  readonly maximumInactiveFaceVelocityBefore: number;
  readonly maximumInactiveFaceVelocityAfter: number;
  readonly postMixedSeamDivergenceMaximum: number;
}

export interface SparseAtlasProjectionResult {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  /** Compact cell-order publications; air cells contain zero. */
  readonly leafPressure: Float64Array;
  readonly leafRhs: Float64Array;
  readonly leafDiagonal: Float64Array;
  readonly leafDivergenceBefore: Float64Array;
  readonly leafDivergence: Float64Array;
  /** XYZ per compact cell, reconstructed from incident face rows. */
  readonly leafCollocatedVelocity: Float64Array;
  readonly projectedFaceVelocity: Float64Array;
  readonly receipt: SparseAtlasProjectionReceipt;
}

export interface SparseAtlasProjectionWorkspace {
  readonly rows: SparseAtlasGradientRow[];
  readonly neighbors: number[][];
  readonly neighborCounts: number[];
  componentMembers: Int32Array;
  componentStarts: Int32Array;
  componentAnchored: Uint8Array;
  phi: Float64Array;
  liquid: Uint8Array;
  diagonal: Float64Array;
  anchored: Uint8Array;
  thetaByRow: Float64Array;
  componentByCell: Int32Array;
  componentStack: Int32Array;
  velocityBefore: Float64Array;
  rhs: Float64Array;
  pressure: Float64Array;
  residual: Float64Array;
  preconditioned: Float64Array;
  direction: Float64Array;
  applied: Float64Array;
  trueResidual: Float64Array;
  correction: Float64Array;
  activeRows: Uint8Array;
  projectedFaceVelocity: Float64Array;
  projectedEquationResidual: Float64Array;
  preDivergence: Float64Array;
  leafDivergence: Float64Array;
  collocatedVelocity: Float64Array;
  collocationWeights: Float64Array;
  system?: LiquidSystem;
  result?: SparseAtlasProjectionResult;
  receipt?: SparseAtlasProjectionReceipt;
}

export function createSparseAtlasProjectionWorkspace(): SparseAtlasProjectionWorkspace {
  return {
    rows: [], neighbors: [], neighborCounts: [],
    componentMembers: new Int32Array(0), componentStarts: new Int32Array(0),
    componentAnchored: new Uint8Array(0),
    phi: new Float64Array(0), liquid: new Uint8Array(0),
    diagonal: new Float64Array(0), anchored: new Uint8Array(0),
    thetaByRow: new Float64Array(0), componentByCell: new Int32Array(0),
    componentStack: new Int32Array(0),
    velocityBefore: new Float64Array(0), rhs: new Float64Array(0),
    pressure: new Float64Array(0), residual: new Float64Array(0),
    preconditioned: new Float64Array(0), direction: new Float64Array(0),
    applied: new Float64Array(0), trueResidual: new Float64Array(0),
    correction: new Float64Array(0), activeRows: new Uint8Array(0),
    projectedFaceVelocity: new Float64Array(0),
    projectedEquationResidual: new Float64Array(0),
    preDivergence: new Float64Array(0), leafDivergence: new Float64Array(0),
    collocatedVelocity: new Float64Array(0), collocationWeights: new Float64Array(0),
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

function exactUint8(values: Uint8Array, length: number): Uint8Array {
  if (values.length === length) return values;
  const available = values.byteOffset === 0 ? values.buffer.byteLength : 0;
  if (available >= length) return new Uint8Array(values.buffer, 0, length);
  let capacity = 1;
  while (capacity < length) capacity *= 2;
  const grown = new Uint8Array(capacity);
  return length === capacity ? grown : grown.subarray(0, length);
}

function exactInt32(values: Int32Array, length: number): Int32Array {
  if (values.length === length) return values;
  const available = values.byteOffset === 0 ? values.buffer.byteLength / 4 : 0;
  if (available >= length) return new Int32Array(values.buffer, 0, length);
  let capacity = 1;
  while (capacity < length) capacity *= 2;
  const grown = new Int32Array(capacity);
  return length === capacity ? grown : grown.subarray(0, length);
}

type Mutable<T> = { -readonly [Key in keyof T]: T[Key] };

interface LiquidSystem {
  grid: SparseAtlasCompositeGrid;
  phi: Float64Array;
  liquid: Uint8Array;
  rows: readonly SparseAtlasGradientRow[];
  thetaByRow: Float64Array;
  anchored: Uint8Array;
  componentByCell: Int32Array;
  componentMembers: Int32Array;
  componentStarts: Int32Array;
  componentAnchored: Uint8Array;
  componentCount: number;
  diagonal: Float64Array;
  cutRowCount: number;
  cutMixedSeamRowCount: number;
  thetaClampCount: number;
  minimumTheta: number;
}

function localIndex(local: SparseBrickVec3, resolution: number): number {
  return local[0] + resolution * (local[1] + resolution * local[2]);
}

function tangentialAxes(axis: SparseAtlasAxis): readonly [SparseAtlasAxis, SparseAtlasAxis] {
  if (axis === 0) return [1, 2];
  if (axis === 1) return [0, 2];
  return [0, 1];
}

function mutableVector(): [number, number, number] {
  return [0, 0, 0];
}

function assertVectorLength(values: ArrayLike<number>, expected: number, label: string): void {
  if (values.length !== expected) {
    throw new RangeError(`${label} has ${values.length} entries; expected ${expected}`);
  }
}

function overlapLength(a0: number, a1: number, b0: number, b1: number): number {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

/** Build the sole authoritative G-row topology for all resident bricks. */
export function buildSparseAtlasCompositeGrid(
  atlas: SparseAdaptiveMassAtlas,
  sparseAirPhi = 0.5,
  workspace?: SparseAtlasCompositeGridBuildWorkspace,
): SparseAtlasCompositeGrid {
  if (!Number.isFinite(sparseAirPhi) || sparseAirPhi <= 0) {
    throw new RangeError("sparseAirPhi must be finite and positive");
  }
  const brickFineWidth = atlas.brickFineResolution;
  let bricks = atlas.bricks;
  for (let index = 1; index < bricks.length; index += 1) {
    if (bricks[index - 1].key <= bricks[index].key) continue;
    bricks = [...bricks].sort((left, right) => left.key - right.key);
    break;
  }
  const cells: SparseAtlasCompositeCell[] = workspace?.cells ?? [];
  let cellCountBuilt = 0;
  const cellBaseByBrick = workspace?.cellBaseByBrick ?? new Map<number, number>();
  cellBaseByBrick.clear();

  for (const brick of bricks) {
    cellBaseByBrick.set(brick.key, cellCountBuilt);
    const scale = brickFineWidth * sparseBrickSpan(brick) / brick.resolution;
    for (let z = 0; z < brick.resolution; z += 1) {
      for (let y = 0; y < brick.resolution; y += 1) {
        for (let x = 0; x < brick.resolution; x += 1) {
          const index = x + brick.resolution * (y + brick.resolution * z);
          const minimum0 = brick.coordinate[0] * brickFineWidth + x * scale;
          const minimum1 = brick.coordinate[1] * brickFineWidth + y * scale;
          const minimum2 = brick.coordinate[2] * brickFineWidth + z * scale;
          const maximum0 = Math.min(minimum0 + scale, atlas.dimensions[0]);
          const maximum1 = Math.min(minimum1 + scale, atlas.dimensions[1]);
          const maximum2 = Math.min(minimum2 + scale, atlas.dimensions[2]);
          const width0 = maximum0 - minimum0;
          const width1 = maximum1 - minimum1;
          const width2 = maximum2 - minimum2;
          if (width0 <= 0 || width1 <= 0 || width2 <= 0) continue;
          const cellId = cellCountBuilt++;
          const stableLeafId = brick.key * atlas.brickCellCapacity + index;
          const centerFine = [0.5 * (minimum0 + maximum0),
            0.5 * (minimum1 + maximum1), 0.5 * (minimum2 + maximum2)] as const;
          const widthsFine = [width0, width1, width2] as const;
          const volume = width0 * width1 * width2;
          const openFraction = sphericalContainerOpenFractionAtFineBox(
            atlas.boundary, centerFine, widthsFine,
          );
          const separatingPressureMinimum = atlas.boundary !== undefined
            && !pointInsideSphericalContainerFine(atlas.boundary, centerFine);
          let cell = workspace?.cellPool[cellId];
          if (!cell) {
            cell = {
              id: cellId, stableLeafId, brickKey: brick.key,
              brickCoordinate: brick.coordinate, brickResolution: brick.resolution,
              local: [x, y, z], localIndex: index,
              minimumFine: [minimum0, minimum1, minimum2],
              maximumFine: [maximum0, maximum1, maximum2],
              centerFine,
              widthsFine: [width0, width1, width2],
              volume, volumeFineCells: volume,
              openFraction, openVolume: volume * openFraction,
              separatingPressureMinimum,
              density: brick.density[index], gamma: brick.gamma[index],
            };
            if (workspace) workspace.cellPool[cellId] = cell;
          } else {
            cell.id = cellId;
            cell.stableLeafId = stableLeafId;
            cell.brickKey = brick.key;
            cell.brickCoordinate = brick.coordinate;
            cell.brickResolution = brick.resolution;
            (cell.local as [number, number, number])[0] = x;
            (cell.local as [number, number, number])[1] = y;
            (cell.local as [number, number, number])[2] = z;
            cell.localIndex = index;
            const minimum = cell.minimumFine as [number, number, number];
            minimum[0] = minimum0; minimum[1] = minimum1; minimum[2] = minimum2;
            const maximum = cell.maximumFine as [number, number, number];
            maximum[0] = maximum0; maximum[1] = maximum1; maximum[2] = maximum2;
            const center = cell.centerFine as [number, number, number];
            center[0] = 0.5 * (minimum0 + maximum0);
            center[1] = 0.5 * (minimum1 + maximum1);
            center[2] = 0.5 * (minimum2 + maximum2);
            const widths = cell.widthsFine as [number, number, number];
            widths[0] = width0; widths[1] = width1; widths[2] = width2;
            cell.volume = volume;
            cell.volumeFineCells = cell.volume;
            cell.openFraction = openFraction;
            cell.openVolume = volume * openFraction;
            cell.separatingPressureMinimum = separatingPressureMinimum;
            cell.density = brick.density[index];
            cell.gamma = brick.gamma[index];
          }
          cells[cellId] = cell;
        }
      }
    }
  }
  cells.length = cellCountBuilt;

  const rows: SparseAtlasGradientRow[] = workspace?.rows ?? [];
  let rowCountBuilt = 0;
  const seamScratchCapacity = 2 * brickFineWidth ** 2 + 2;
  const termCellScratch = workspace
    && workspace.termCellScratch.length >= seamScratchCapacity
    ? workspace.termCellScratch : new Int32Array(seamScratchCapacity);
  const termCoefficientScratch = workspace
    && workspace.termCoefficientScratch.length >= seamScratchCapacity
    ? workspace.termCoefficientScratch : new Float64Array(seamScratchCapacity);
  let mixedSeamRowCount = 0, sparseAirRowCount = 0;
  const tankWallOpenFraction = (
    axis: SparseAtlasAxis,
    faceCoordinate: number,
    minimum: SparseBrickVec3,
    maximum: SparseBrickVec3,
  ): number | undefined => {
    const placement = atlas.tankWallPlacement;
    const wallField = atlas.wallField;
    if (!placement || !wallField || axis === 1) return undefined;
    const side = axis === 0
      ? Math.abs(faceCoordinate - placement.minimumFine[0]) < 1e-6 ? "left"
        : Math.abs(faceCoordinate - placement.maximumFine[0]) < 1e-6 ? "right" : undefined
      : Math.abs(faceCoordinate - placement.minimumFine[2]) < 1e-6 ? "front"
        : Math.abs(faceCoordinate - placement.maximumFine[2]) < 1e-6 ? "back" : undefined;
    if (!side) return undefined;
    const uAxis = axis === 0 ? 2 : 0;
    const vAxis = 1;
    const u0 = Math.max(minimum[uAxis], placement.minimumFine[uAxis]);
    const u1 = Math.min(maximum[uAxis], placement.maximumFine[uAxis]);
    const v0 = Math.max(minimum[vAxis], placement.minimumFine[vAxis]);
    const v1 = Math.min(maximum[vAxis], placement.maximumFine[vAxis]);
    const totalArea = Math.max(0, maximum[uAxis] - minimum[uAxis])
      * Math.max(0, maximum[vAxis] - minimum[vAxis]);
    const wallArea = Math.max(0, u1 - u0) * Math.max(0, v1 - v0);
    if (!(totalArea > 0) || !(wallArea > 0)) return 1;
    const wallOpen = tankWallOpeningFraction(
      wallField, side,
      u0 - placement.minimumFine[uAxis], u1 - placement.minimumFine[uAxis],
      v0 - placement.minimumFine[vAxis], v1 - placement.minimumFine[vAxis],
    );
    return (totalArea - wallArea + wallArea * wallOpen) / totalArea;
  };
  const appendRow = (
    kind: SparseAtlasGradientRowKind,
    axis: SparseAtlasAxis,
    center0: number,
    center1: number,
    center2: number,
    area: number,
    distance: number,
    tangentWidth0: number,
    tangentWidth1: number,
    termCount: number,
    negativeBrickKey?: number,
    positiveBrickKey?: number,
    exteriorPhi?: number,
    authoredOpenFraction = 1,
  ): void => {
    const center = [center0, center1, center2] as const;
    const boundaryOpenFraction = sphericalContainerOpenFractionAtFineFace(
      atlas.boundary, center, axis, [tangentWidth0, tangentWidth1],
    );
    const tangents = tangentialAxes(axis);
    const dualWidths = [0, 0, 0] as [number, number, number];
    dualWidths[axis] = distance;
    dualWidths[tangents[0]] = tangentWidth0;
    dualWidths[tangents[1]] = tangentWidth1;
    const boundaryDualOpenFraction = sphericalContainerOpenFractionAtFineBox(
      atlas.boundary, center, dualWidths,
    );
    const openFraction = boundaryOpenFraction * authoredOpenFraction;
    const pressureDualOpenFraction = boundaryDualOpenFraction * authoredOpenFraction;
    const geometricArea = area * authoredOpenFraction;
    area *= openFraction;
    const id = rowCountBuilt++;
    let row = workspace?.rowPool[id];
    if (!row) {
      row = {
        id, kind, axis, centerFine: [center0, center1, center2], area, distance,
        areaFineCells2: area, centerDistanceFine: distance,
        dualWeight: geometricArea * distance * pressureDualOpenFraction,
        geometricArea, openFraction, pressureDualOpenFraction,
        terms: [], negativeBrickKey,
        positiveBrickKey, exteriorPhi, termPool: [],
      };
      if (workspace) workspace.rowPool[id] = row;
    } else {
      row.id = id;
      row.kind = kind;
      row.axis = axis;
      (row.centerFine as [number, number, number])[0] = center0;
      (row.centerFine as [number, number, number])[1] = center1;
      (row.centerFine as [number, number, number])[2] = center2;
      row.area = area;
      row.distance = distance;
      row.areaFineCells2 = area;
      row.centerDistanceFine = distance;
      row.dualWeight = geometricArea * distance * pressureDualOpenFraction;
      row.geometricArea = geometricArea;
      row.openFraction = openFraction;
      row.pressureDualOpenFraction = pressureDualOpenFraction;
      row.negativeBrickKey = negativeBrickKey;
      row.positiveBrickKey = positiveBrickKey;
      row.exteriorPhi = exteriorPhi;
    }
    const terms = row.terms as SparseAtlasGradientTerm[];
    for (let index = 0; index < termCount; index += 1) {
      let term = row.termPool[index] as { cellId: number; coefficient: number } | undefined;
      if (!term) term = row.termPool[index] = {
        cellId: termCellScratch[index], coefficient: termCoefficientScratch[index],
      };
      term.cellId = termCellScratch[index];
      term.coefficient = termCoefficientScratch[index];
      terms[index] = term;
    }
    terms.length = termCount;
    rows[id] = row;
    if (kind === "mixed-seam") mixedSeamRowCount += 1;
    if (kind === "sparse-air") sparseAirRowCount += 1;
  };

  // Ordinary faces wholly inside a brick.
  for (const brick of bricks) {
    const cellBase = cellBaseByBrick.get(brick.key)!;
    const scale = brickFineWidth * sparseBrickSpan(brick) / brick.resolution;
    const validX = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[0] - brick.coordinate[0] * brickFineWidth) / scale,
    )));
    const validY = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[1] - brick.coordinate[1] * brickFineWidth) / scale,
    )));
    const validZ = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[2] - brick.coordinate[2] * brickFineWidth) / scale,
    )));
    for (const axis of [0, 1, 2] as const) {
      const tangents = tangentialAxes(axis);
      for (let z = 0; z < brick.resolution; z += 1) {
        for (let y = 0; y < brick.resolution; y += 1) {
          for (let x = 0; x < brick.resolution; x += 1) {
            if (x >= validX || y >= validY || z >= validZ) continue;
            const positiveCoordinate = axis === 0 ? x : axis === 1 ? y : z;
            if (positiveCoordinate === 0) continue;
            const positiveId = cellBase + x + validX * (y + validY * z);
            const negativeId = positiveId - (axis === 0 ? 1
              : axis === 1 ? validX : validX * validY);
            const negative = cells[negativeId];
            const positive = cells[positiveId];
            const distance = positive.centerFine[axis] - negative.centerFine[axis];
            const area = overlapLength(
              negative.minimumFine[tangents[0]], negative.maximumFine[tangents[0]],
              positive.minimumFine[tangents[0]], positive.maximumFine[tangents[0]],
            ) * overlapLength(
              negative.minimumFine[tangents[1]], negative.maximumFine[tangents[1]],
              positive.minimumFine[tangents[1]], positive.maximumFine[tangents[1]],
            );
            let center0 = 0, center1 = 0, center2 = 0;
            const faceCenter = negative.maximumFine[axis];
            const tangentCenter0 = 0.5 * (negative.minimumFine[tangents[0]]
              + negative.maximumFine[tangents[0]]);
            const tangentCenter1 = 0.5 * (negative.minimumFine[tangents[1]]
              + negative.maximumFine[tangents[1]]);
            if (axis === 0) center0 = faceCenter;
            else if (axis === 1) center1 = faceCenter;
            else center2 = faceCenter;
            if (tangents[0] === 0) center0 = tangentCenter0;
            else if (tangents[0] === 1) center1 = tangentCenter0;
            else center2 = tangentCenter0;
            if (tangents[1] === 0) center0 = tangentCenter1;
            else if (tangents[1] === 1) center1 = tangentCenter1;
            else center2 = tangentCenter1;
            termCellScratch[0] = negative.id;
            termCoefficientScratch[0] = -1 / distance;
            termCellScratch[1] = positive.id;
            termCoefficientScratch[1] = 1 / distance;
            const minimum = [
              Math.min(negative.minimumFine[0], positive.minimumFine[0]),
              Math.min(negative.minimumFine[1], positive.minimumFine[1]),
              Math.min(negative.minimumFine[2], positive.minimumFine[2]),
            ] as SparseBrickVec3;
            const maximum = [
              Math.max(negative.maximumFine[0], positive.maximumFine[0]),
              Math.max(negative.maximumFine[1], positive.maximumFine[1]),
              Math.max(negative.maximumFine[2], positive.maximumFine[2]),
            ] as SparseBrickVec3;
            const authoredOpenFraction = tankWallOpenFraction(
              axis, faceCenter, minimum, maximum,
            );
            if (authoredOpenFraction === 0) continue;
            appendRow(
              "intra-brick", axis, center0, center1, center2, area, distance,
              negative.widthsFine[tangents[0]], negative.widthsFine[tangents[1]], 2,
              brick.key, brick.key, undefined, authoredOpenFraction ?? 1,
            );
          }
        }
      }
    }
  }

  const faceCells = (
    brick: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
    side: -1 | 1,
    result: SparseAtlasCompositeCell[],
  ): SparseAtlasCompositeCell[] => {
    result.length = 0;
    const coordinate = side < 0 ? 0 : brick.resolution - 1;
    const cellBase = cellBaseByBrick.get(brick.key)!;
    const scale = brickFineWidth * sparseBrickSpan(brick) / brick.resolution;
    const validX = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[0] - brick.coordinate[0] * brickFineWidth) / scale,
    )));
    const validY = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[1] - brick.coordinate[1] * brickFineWidth) / scale,
    )));
    const validZ = Math.max(0, Math.min(brick.resolution, Math.ceil(
      (atlas.dimensions[2] - brick.coordinate[2] * brickFineWidth) / scale,
    )));
    for (let z = 0; z < brick.resolution; z += 1) {
      for (let y = 0; y < brick.resolution; y += 1) {
        for (let x = 0; x < brick.resolution; x += 1) {
          if (x >= validX || y >= validY || z >= validZ) continue;
          const axisCoordinate = axis === 0 ? x : axis === 1 ? y : z;
          if (axisCoordinate === coordinate) {
            const id = cellBase + x + validX * (y + validY * z);
            result.push(cells[id]);
          }
        }
      }
    }
    return result;
  };
  const negativeFaceCells: SparseAtlasCompositeCell[] = [];
  const positiveFaceCells: SparseAtlasCompositeCell[] = [];

  const appendBrickInterface = (
    negative: SparseAdaptiveMassBrick,
    positive: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
  ): void => {
    const tangents = tangentialAxes(axis);
    const portWidth = Math.max(
      brickFineWidth * sparseBrickSpan(negative) / negative.resolution,
      brickFineWidth * sparseBrickSpan(positive) / positive.resolution,
    );
    const negativeCells = faceCells(negative, axis, 1, negativeFaceCells);
    const positiveCells = faceCells(positive, axis, -1, positiveFaceCells);
    const faceCoordinate = (negative.coordinate[axis] + sparseBrickSpan(negative))
      * brickFineWidth;
    const overlapMinimum = [0, 0, 0] as [number, number, number];
    const overlapMaximum = [0, 0, 0] as [number, number, number];
    for (const tangent of tangents) {
      overlapMinimum[tangent] = Math.max(
        negative.coordinate[tangent], positive.coordinate[tangent],
      ) * brickFineWidth;
      overlapMaximum[tangent] = Math.min(
        negative.coordinate[tangent] + sparseBrickSpan(negative),
        positive.coordinate[tangent] + sparseBrickSpan(positive),
      ) * brickFineWidth;
    }
    for (let portV = overlapMinimum[tangents[1]];
      portV < overlapMaximum[tangents[1]]; portV += portWidth) {
      for (let portU = overlapMinimum[tangents[0]];
        portU < overlapMaximum[tangents[0]]; portU += portWidth) {
        const minimum = mutableVector();
        const maximum = mutableVector();
        minimum[axis] = maximum[axis] = faceCoordinate;
        minimum[tangents[0]] = portU;
        maximum[tangents[0]] = Math.min(
          minimum[tangents[0]] + portWidth, overlapMaximum[tangents[0]],
        );
        minimum[tangents[1]] = portV;
        maximum[tangents[1]] = Math.min(
          minimum[tangents[1]] + portWidth, overlapMaximum[tangents[1]],
        );
        const area = (maximum[tangents[0]] - minimum[tangents[0]])
          * (maximum[tangents[1]] - minimum[tangents[1]]);
        if (!(area > 0)) continue;
        const authoredOpenFraction = tankWallOpenFraction(
          axis, faceCoordinate, minimum, maximum,
        );
        if (authoredOpenFraction === 0) continue;
        let negativeCount = 0, positiveCount = 0;
        let negativeCenterSum = 0, positiveCenterSum = 0;
        for (let index = 0; index < negativeCells.length; index += 1) {
          const cell = negativeCells[index];
          const overlap = overlapLength(
            minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]],
          ) * overlapLength(
            minimum[tangents[1]], maximum[tangents[1]],
            cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]],
          );
          if (!(overlap > 0)) continue;
          negativeCount += 1;
          negativeCenterSum += overlap * cell.centerFine[axis];
        }
        for (let index = 0; index < positiveCells.length; index += 1) {
          const cell = positiveCells[index];
          const overlap = overlapLength(
            minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]],
          ) * overlapLength(
            minimum[tangents[1]], maximum[tangents[1]],
            cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]],
          );
          if (!(overlap > 0)) continue;
          positiveCount += 1;
          positiveCenterSum += overlap * cell.centerFine[axis];
        }
        if (negativeCount === 0 || positiveCount === 0) continue;
        const negativeCenter = negativeCenterSum / area;
        const positiveCenter = positiveCenterSum / area;
        const distance = positiveCenter - negativeCenter;
        let center0 = 0, center1 = 0, center2 = 0;
        if (axis === 0) center0 = faceCoordinate;
        else if (axis === 1) center1 = faceCoordinate;
        else center2 = faceCoordinate;
        const tangentCenter0 = 0.5 * (minimum[tangents[0]] + maximum[tangents[0]]);
        const tangentCenter1 = 0.5 * (minimum[tangents[1]] + maximum[tangents[1]]);
        if (tangents[0] === 0) center0 = tangentCenter0;
        else if (tangents[0] === 1) center1 = tangentCenter0;
        else center2 = tangentCenter0;
        if (tangents[1] === 0) center0 = tangentCenter1;
        else if (tangents[1] === 1) center1 = tangentCenter1;
        else center2 = tangentCenter1;
        let termCount = 0;
        for (let index = 0; index < negativeCells.length; index += 1) {
          const cell = negativeCells[index];
          const overlap = overlapLength(
            minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]],
          ) * overlapLength(
            minimum[tangents[1]], maximum[tangents[1]],
            cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]],
          );
          if (overlap > 0) {
            termCellScratch[termCount] = cell.id;
            termCoefficientScratch[termCount++] = -overlap / (area * distance);
          }
        }
        for (let index = 0; index < positiveCells.length; index += 1) {
          const cell = positiveCells[index];
          const overlap = overlapLength(
            minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]],
          ) * overlapLength(
            minimum[tangents[1]], maximum[tangents[1]],
            cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]],
          );
          if (overlap > 0) {
            termCellScratch[termCount] = cell.id;
            termCoefficientScratch[termCount++] = overlap / (area * distance);
          }
        }
        appendRow(
          brickFineWidth * sparseBrickSpan(negative) / negative.resolution
            === brickFineWidth * sparseBrickSpan(positive) / positive.resolution
            ? "brick-face" : "mixed-seam",
          axis, center0, center1, center2, area, distance,
          maximum[tangents[0]] - minimum[tangents[0]],
          maximum[tangents[1]] - minimum[tangents[1]], termCount,
          negative.key, positive.key, undefined, authoredOpenFraction ?? 1,
        );
      }
    }
  };

  const appendSparseAirFace = (
    brick: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
    side: -1 | 1,
    authoredTankOpening = false,
  ): void => {
    const tangents = tangentialAxes(axis);
    const cellsOnFace = faceCells(brick, axis, side, negativeFaceCells);
    for (let index = 0; index < cellsOnFace.length; index += 1) {
      const cell = cellsOnFace[index];
      const distance = cell.widthsFine[axis];
      const area = cell.widthsFine[tangents[0]] * cell.widthsFine[tangents[1]];
      let authoredOpenFraction = 1;
      const faceCoordinate = side < 0 ? cell.minimumFine[axis] : cell.maximumFine[axis];
      const placedOpenFraction = tankWallOpenFraction(
        axis, faceCoordinate, cell.minimumFine, cell.maximumFine,
      );
      if (placedOpenFraction !== undefined) {
        authoredOpenFraction = placedOpenFraction;
        if (!(authoredOpenFraction > 0)) continue;
      } else if (authoredTankOpening) {
        const wallField = atlas.wallField;
        if (!wallField || axis === 1) continue;
        const wallSide = axis === 0
          ? (side < 0 ? "left" : "right")
          : (side < 0 ? "front" : "back");
        const uAxis = axis === 0 ? 2 : 0;
        authoredOpenFraction = tankWallOpeningFraction(
          wallField, wallSide,
          cell.minimumFine[uAxis], cell.maximumFine[uAxis],
          cell.minimumFine[1], cell.maximumFine[1],
        );
        if (!(authoredOpenFraction > 0)) continue;
      }
      let center0 = cell.centerFine[0];
      let center1 = cell.centerFine[1];
      let center2 = cell.centerFine[2];
      const faceCenter = faceCoordinate;
      if (axis === 0) center0 = faceCenter;
      else if (axis === 1) center1 = faceCenter;
      else center2 = faceCenter;
      termCellScratch[0] = cell.id;
      termCoefficientScratch[0] = side < 0 ? 1 / distance : -1 / distance;
      appendRow(
        "sparse-air", axis, center0, center1, center2, area, distance,
        cell.widthsFine[tangents[0]], cell.widthsFine[tangents[1]], 1,
        side > 0 ? brick.key : undefined,
        side < 0 ? brick.key : undefined,
        sparseAirPhi,
        authoredOpenFraction,
      );
    }
  };

  // Exact face-coordinate indices connect arbitrary dyadic spans. Their size
  // is proportional to resident leaves and faces, never to domain volume.
  const negativeFaces = ([0, 1, 2] as const).map(() =>
    new Map<number, SparseAdaptiveMassBrick[]>());
  const positiveFaces = ([0, 1, 2] as const).map(() =>
    new Map<number, SparseAdaptiveMassBrick[]>());
  for (const brick of bricks) for (const axis of [0, 1, 2] as const) {
    const negativeFace = brick.coordinate[axis];
    let negativeBucket = negativeFaces[axis].get(negativeFace);
    if (!negativeBucket) negativeFaces[axis].set(negativeFace, negativeBucket = []);
    negativeBucket.push(brick);
    const positiveFace = brick.coordinate[axis] + sparseBrickSpan(brick);
    let positiveBucket = positiveFaces[axis].get(positiveFace);
    if (!positiveBucket) positiveFaces[axis].set(positiveFace, positiveBucket = []);
    positiveBucket.push(brick);
  }
  const tangentOverlap = (
    left: SparseAdaptiveMassBrick,
    right: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
  ): boolean => tangentialAxes(axis).every((tangent) =>
    Math.min(left.coordinate[tangent] + sparseBrickSpan(left),
      right.coordinate[tangent] + sparseBrickSpan(right))
      > Math.max(left.coordinate[tangent], right.coordinate[tangent]));

  for (const brick of bricks) for (const axis of [0, 1, 2] as const) {
    const positiveFace = brick.coordinate[axis] + sparseBrickSpan(brick);
    if (positiveFace < atlas.brickDimensions[axis]) {
      const neighbors = (negativeFaces[axis].get(positiveFace) ?? [])
        .filter((candidate) => tangentOverlap(brick, candidate, axis));
      if (neighbors.length > 0) {
        for (const neighbor of neighbors) appendBrickInterface(brick, neighbor, axis);
      } else appendSparseAirFace(brick, axis, 1);
    } else if (axis !== 1 && atlas.wallField && !atlas.boundary
      && !atlas.tankWallPlacement) {
      appendSparseAirFace(brick, axis, 1, true);
    }
    if (brick.coordinate[axis] > 0) {
      const hasNegativeNeighbor = (positiveFaces[axis].get(brick.coordinate[axis]) ?? [])
        .some((candidate) => tangentOverlap(candidate, brick, axis));
      if (!hasNegativeNeighbor) appendSparseAirFace(brick, axis, -1);
    } else if (axis !== 1 && atlas.wallField && !atlas.boundary
      && !atlas.tankWallPlacement) {
      appendSparseAirFace(brick, axis, -1, true);
    }
  }

  // A reused workspace may previously have held a larger atlas variant. Keep
  // the public compact arrays authoritative: stale pooled cells beyond this
  // build would make the final brick appear to own records that were skipped
  // by domain clipping.
  cells.length = cellCountBuilt;
  rows.length = rowCountBuilt;
  reserveCompositeCells(workspace, cellCountBuilt);
  reserveCompositeRows(workspace, rows.length);
  const topologyKey = {};
  if (!workspace?.grid) {
    const grid = {
      atlas, cells, gradientRows: rows, cellBaseByBrick, mixedSeamRowCount,
      sparseAirRowCount, topologyKey,
    };
    if (workspace) workspace.grid = grid;
    return grid;
  }
  const grid = workspace.grid as {
    atlas: SparseAdaptiveMassAtlas;
    cells: readonly SparseAtlasCompositeCell[];
    gradientRows: readonly SparseAtlasGradientRow[];
    cellBaseByBrick: ReadonlyMap<number, number>;
    mixedSeamRowCount: number;
    sparseAirRowCount: number;
    topologyKey?: object;
  };
  grid.atlas = atlas;
  grid.cells = cells;
  grid.gradientRows = rows;
  grid.cellBaseByBrick = cellBaseByBrick;
  grid.mixedSeamRowCount = mixedSeamRowCount;
  grid.sparseAirRowCount = sparseAirRowCount;
  grid.topologyKey = topologyKey;
  return workspace.grid;
}

/** Apply the global G row list, with sparse-air pressure fixed to zero. */
export function applySparseAtlasGradient(
  grid: SparseAtlasCompositeGrid,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.gradientRows.length),
): Float64Array {
  assertVectorLength(pressure, grid.cells.length, "pressure");
  assertVectorLength(output, grid.gradientRows.length, "gradient output");
  for (const row of grid.gradientRows) {
    let value = 0;
    for (const term of row.terms) value += term.coefficient * pressure[term.cellId];
    output[row.id] = value;
  }
  return output;
}

/** Apply D = -M^-1 G^T W using the same global rows. */
export function applySparseAtlasDivergence(
  grid: SparseAtlasCompositeGrid,
  normalVelocity: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.cells.length),
): Float64Array {
  assertVectorLength(normalVelocity, grid.gradientRows.length, "normalVelocity");
  assertVectorLength(output, grid.cells.length, "divergence output");
  output.fill(0);
  for (const row of grid.gradientRows) {
    const weighted = row.dualWeight * normalVelocity[row.id];
    for (const term of row.terms) output[term.cellId] -= term.coefficient * weighted;
  }
  for (const cell of grid.cells) output[cell.id] = cell.openVolume > 1e-12
    ? output[cell.id] / cell.openVolume : 0;
  return output;
}

/** Apply A = G^T W G using the same global rows. */
export function applySparseAtlasPressureOperator(
  grid: SparseAtlasCompositeGrid,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.cells.length),
): Float64Array {
  assertVectorLength(output, grid.cells.length, "pressure output");
  const gradient = applySparseAtlasGradient(grid, pressure);
  output.fill(0);
  for (const row of grid.gradientRows) {
    const weighted = row.dualWeight * gradient[row.id];
    for (const term of row.terms) output[term.cellId] += term.coefficient * weighted;
  }
  return output;
}

/** Stable identity for preserving face state across atlas generation changes. */
export function sparseAtlasRowGeometryKey(row: SparseAtlasGradientRow): string {
  return `${row.axis}:${row.centerFine[0]},${row.centerFine[1]},${row.centerFine[2]}`;
}

/**
 * Preserve velocities whose oriented face-port geometry survives a topology
 * rebuild. Newly created ports receive `fallback` (zero by default).
 */
export function remapSparseAtlasFaceVelocity(
  previousGrid: SparseAtlasCompositeGrid,
  previousVelocity: ArrayLike<number>,
  nextGrid: SparseAtlasCompositeGrid,
  fallback: number | ((row: SparseAtlasGradientRow) => number) = 0,
): Float64Array {
  assertVectorLength(previousVelocity, previousGrid.gradientRows.length, "previousVelocity");
  const previousByGeometry = new Map<string, number>();
  for (const row of previousGrid.gradientRows) {
    previousByGeometry.set(sparseAtlasRowGeometryKey(row), previousVelocity[row.id]);
  }
  return Float64Array.from(nextGrid.gradientRows, (row) => {
    const preserved = previousByGeometry.get(sparseAtlasRowGeometryKey(row));
    return preserved ?? (typeof fallback === "function" ? fallback(row) : fallback);
  });
}

function buildLiquidSystem(
  grid: SparseAtlasCompositeGrid,
  phiInput: ArrayLike<number> | undefined,
  denominatorEpsilon: number,
  sparseAirPhi: number,
  workspace: SparseAtlasProjectionWorkspace,
): LiquidSystem {
  const count = grid.cells.length;
  const phi = workspace.phi = exactFloat64(workspace.phi, count);
  if (phiInput) {
    assertVectorLength(phiInput, count, "phi");
    for (let id = 0; id < count; id += 1) phi[id] = phiInput[id];
  } else {
    for (const cell of grid.cells) phi[cell.id] = CM12_LIQUID_ISOVALUE
      - cell.density / Math.max(cell.openFraction, 1e-12);
    // Continue open-liquid density into solid-centred pressure samples. These
    // cells are the dual variables on which the separating p >= 0 constraint
    // acts; treating their authored density as zero disconnects the wall.
    const continuedDensity = new Float64Array(grid.cells.length);
    for (const row of grid.gradientRows) for (const own of row.terms) {
      if (!grid.cells[own.cellId]!.separatingPressureMinimum) continue;
      for (const term of row.terms) {
        if (term.cellId === own.cellId) continue;
        const neighbor = grid.cells[term.cellId]!;
        if (neighbor.openVolume <= 1e-12) continue;
        continuedDensity[own.cellId] = Math.max(continuedDensity[own.cellId],
          neighbor.density / Math.max(neighbor.openFraction, 1e-12));
      }
    }
    for (const cell of grid.cells) if (cell.separatingPressureMinimum) {
      phi[cell.id] = CM12_LIQUID_ISOVALUE - continuedDensity[cell.id];
    }
  }
  assertVectorLength(phi, grid.cells.length, "phi");
  const liquid = workspace.liquid = exactUint8(workspace.liquid, count);
  for (let id = 0; id < count; id += 1) liquid[id] = phi[id] <= 0 ? 1 : 0;
  const rows = workspace.rows;
  let activeRowCount = 0;
  const diagonal = workspace.diagonal = exactFloat64(workspace.diagonal, count);
  diagonal.fill(0);
  const anchored = workspace.anchored = exactUint8(workspace.anchored, count);
  anchored.fill(0);
  const thetaByRow = workspace.thetaByRow = exactFloat64(
    workspace.thetaByRow, grid.gradientRows.length,
  );
  let cutRowCount = 0;
  let cutMixedSeamRowCount = 0;
  let thetaClampCount = 0;
  let minimumTheta = 1;

  for (const source of grid.gradientRows) {
    let liquidTermCount = 0, airTermCount = 0;
    for (const term of source.terms) {
      if (liquid[term.cellId] !== 0) liquidTermCount += 1;
      else airTermCount += 1;
    }
    if (liquidTermCount === 0) continue;
    const hasExteriorAir = source.kind === "sparse-air";
    const cut = airTermCount > 0 || hasExteriorAir;
    let theta = 1;
    if (cut) {
      let liquidPhiSum = 0;
      let liquidWeight = 0;
      for (const term of source.terms) {
        if (liquid[term.cellId] === 0) continue;
        const weight = Math.abs(term.coefficient);
        liquidPhiSum += weight * phi[term.cellId];
        liquidWeight += weight;
      }
      let airPhiSum = 0;
      let airWeight = 0;
      for (const term of source.terms) {
        if (liquid[term.cellId] !== 0) continue;
        const weight = Math.abs(term.coefficient);
        airPhiSum += weight * phi[term.cellId];
        airWeight += weight;
      }
      if (hasExteriorAir) {
        let weight = 0;
        for (const term of source.terms) {
          if (liquid[term.cellId] !== 0) weight += Math.abs(term.coefficient);
        }
        airPhiSum += weight * (source.exteriorPhi ?? sparseAirPhi);
        airWeight += weight;
      }
      const liquidPhi = liquidPhiSum / liquidWeight;
      const airPhi = airPhiSum / airWeight;
      const rawTheta = Math.abs(liquidPhi)
        / Math.max(Math.abs(liquidPhi) + Math.abs(airPhi), denominatorEpsilon);
      theta = cm12GhostFluidTheta(liquidPhi, airPhi, denominatorEpsilon);
      if (theta !== rawTheta) thetaClampCount += 1;
      cutRowCount += 1;
      if (source.kind === "mixed-seam") cutMixedSeamRowCount += 1;
      for (const term of source.terms) {
        if (liquid[term.cellId] !== 0) anchored[term.cellId] = 1;
      }
    }
    minimumTheta = Math.min(minimumTheta, theta);
    rows[activeRowCount++] = source;
    thetaByRow[source.id] = theta;
    for (const term of source.terms) {
      if (liquid[term.cellId] === 0) continue;
      diagonal[term.cellId] += source.dualWeight
        * term.coefficient * term.coefficient / theta;
    }
  }
  rows.length = activeRowCount;

  // Connected liquid components establish the exact nullspace projector.
  const neighbors = workspace.neighbors;
  while (neighbors.length < count) neighbors.push([]);
  const neighborCounts = workspace.neighborCounts;
  for (let id = 0; id < count; id += 1) neighborCounts[id] = 0;
  for (const row of rows) {
    for (let left = 0; left < row.terms.length; left += 1) {
      if (liquid[row.terms[left].cellId] === 0) continue;
      for (let right = left + 1; right < row.terms.length; right += 1) {
        if (liquid[row.terms[right].cellId] === 0) continue;
        const leftId = row.terms[left].cellId;
        const rightId = row.terms[right].cellId;
        let neighborCount = neighborCounts[leftId];
        neighbors[leftId][neighborCount] = rightId;
        neighborCounts[leftId] = neighborCount + 1;
        neighborCount = neighborCounts[rightId];
        neighbors[rightId][neighborCount] = leftId;
        neighborCounts[rightId] = neighborCount + 1;
      }
    }
  }
  const componentByCell = workspace.componentByCell = exactInt32(
    workspace.componentByCell, count,
  );
  componentByCell.fill(-1);
  const componentMembers = workspace.componentMembers = exactInt32(
    workspace.componentMembers, count,
  );
  const componentStarts = workspace.componentStarts = exactInt32(
    workspace.componentStarts, count + 1,
  );
  const componentAnchored = workspace.componentAnchored = exactUint8(
    workspace.componentAnchored, count,
  );
  let componentCount = 0;
  let memberCount = 0;
  const stack = workspace.componentStack = exactInt32(workspace.componentStack, count);
  for (const cell of grid.cells) {
    if (!liquid[cell.id] || componentByCell[cell.id] >= 0) continue;
    const component = componentCount;
    componentStarts[component] = memberCount;
    let stackCount = 1;
    stack[0] = cell.id;
    componentByCell[cell.id] = component;
    let hasAnchor = false;
    while (stackCount > 0) {
      const current = stack[--stackCount];
      componentMembers[memberCount++] = current;
      hasAnchor ||= anchored[current] !== 0;
      for (let neighborIndex = 0;
        neighborIndex < neighborCounts[current]; neighborIndex += 1) {
        const neighbor = neighbors[current][neighborIndex];
        if (componentByCell[neighbor] >= 0) continue;
        componentByCell[neighbor] = component;
        stack[stackCount++] = neighbor;
      }
    }
    componentStarts[component + 1] = memberCount;
    componentAnchored[component] = hasAnchor ? 1 : 0;
    componentCount += 1;
  }
  const system = workspace.system ?? (workspace.system = {
    grid, phi, liquid, rows, thetaByRow, anchored,
    componentByCell, componentMembers, componentStarts, componentAnchored, componentCount,
    diagonal, cutRowCount, cutMixedSeamRowCount, thetaClampCount, minimumTheta,
  });
  system.grid = grid;
  system.phi = phi;
  system.liquid = liquid;
  system.rows = rows;
  system.thetaByRow = thetaByRow;
  system.anchored = anchored;
  system.componentByCell = componentByCell;
  system.componentMembers = componentMembers;
  system.componentStarts = componentStarts;
  system.componentAnchored = componentAnchored;
  system.componentCount = componentCount;
  system.diagonal = diagonal;
  system.cutRowCount = cutRowCount;
  system.cutMixedSeamRowCount = cutMixedSeamRowCount;
  system.thetaClampCount = thetaClampCount;
  system.minimumTheta = minimumTheta;
  return system;
}

function applyLiquidOperator(
  system: LiquidSystem,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(system.grid.cells.length),
): Float64Array {
  output.fill(0);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) {
      if (system.liquid[term.cellId] !== 0) {
        jump += term.coefficient * pressure[term.cellId];
      }
    }
    const weighted = row.dualWeight * jump / system.thetaByRow[row.id];
    for (const term of row.terms) {
      if (system.liquid[term.cellId] !== 0) {
        output[term.cellId] += term.coefficient * weighted;
      }
    }
  }
  return output;
}

function projectNullspace(system: LiquidSystem, values: Float64Array): number {
  let maximumRemoved = 0;
  for (let component = 0; component < system.componentCount; component += 1) {
    if (system.componentAnchored[component]) continue;
    let sum = 0;
    const begin = system.componentStarts[component];
    const end = system.componentStarts[component + 1];
    for (let member = begin; member < end; member += 1) {
      sum += values[system.componentMembers[member]];
    }
    const mean = sum / (end - begin);
    maximumRemoved = Math.max(maximumRemoved, Math.abs(sum));
    for (let member = begin; member < end; member += 1) {
      values[system.componentMembers[member]] -= mean;
    }
  }
  return maximumRemoved;
}

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let sum = 0;
  let correction = 0;
  for (let index = 0; index < left.length; index += 1) {
    const product = left[index] * right[index];
    const adjusted = product - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

function norm(values: ArrayLike<number>): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function maximumAbsolute(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(values[index]));
  }
  return maximum;
}

function assembleLiquidRhs(
  system: LiquidSystem,
  velocity: ArrayLike<number>,
  targetDivergence?: ArrayLike<number>,
  rhs: Float64Array = new Float64Array(system.grid.cells.length),
): Float64Array {
  if (targetDivergence) {
    assertVectorLength(targetDivergence, system.grid.cells.length, "targetDivergence");
  }
  rhs.fill(0);
  for (const row of system.rows) {
    const weighted = row.dualWeight * velocity[row.id];
    for (const term of row.terms) {
      if (system.liquid[term.cellId] !== 0) rhs[term.cellId] += term.coefficient * weighted;
    }
  }
  // D = -M^-1 G^T W. Solving A p = G^T W u + M q leaves
  // G^T W (u-Gp) = -M q and therefore D u' = q.
  if (targetDivergence) for (const cell of system.grid.cells) {
    if (system.liquid[cell.id]) rhs[cell.id] += (cell.separatingPressureMinimum
      ? cell.volume : cell.openVolume) * targetDivergence[cell.id];
  }
  return rhs;
}

function pressureCorrection(
  system: LiquidSystem,
  pressure: ArrayLike<number>,
  correction: Float64Array = new Float64Array(system.grid.gradientRows.length),
): Float64Array {
  correction.fill(0);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) {
      if (system.liquid[term.cellId] !== 0) {
        jump += term.coefficient * pressure[term.cellId];
      }
    }
    correction[row.id] = jump / system.thetaByRow[row.id];
  }
  return correction;
}

function faceEnergy(system: LiquidSystem, velocity: ArrayLike<number>): number {
  let energy = 0;
  for (const row of system.rows) {
    const value = velocity[row.id];
    energy += 0.5 * row.dualWeight * system.thetaByRow[row.id] * value * value;
  }
  return energy;
}

function divergenceFromEquationResidual(
  system: LiquidSystem,
  residual: ArrayLike<number>,
  result: Float64Array = new Float64Array(system.grid.cells.length),
): Float64Array {
  for (const cell of system.grid.cells) {
    const volume = cell.separatingPressureMinimum ? cell.volume : cell.openVolume;
    result[cell.id] = system.liquid[cell.id] && volume > 1e-12
      ? -residual[cell.id] / volume : 0;
  }
  return result;
}

function volumeL2(grid: SparseAtlasCompositeGrid, values: ArrayLike<number>): number {
  let squared = 0;
  for (const cell of grid.cells) squared += cell.openVolume
    * values[cell.id] * values[cell.id];
  return Math.sqrt(squared);
}

export function collocateSparseAtlasVelocity(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  result: Float64Array = new Float64Array(3 * grid.cells.length),
  weights: Float64Array = new Float64Array(3 * grid.cells.length),
): Float64Array {
  assertVectorLength(result, 3 * grid.cells.length, "collocated velocity output");
  assertVectorLength(weights, 3 * grid.cells.length, "collocated velocity weights");
  result.fill(0);
  weights.fill(0);
  for (const row of grid.gradientRows) {
    for (const term of row.terms) {
      const offset = 3 * term.cellId + row.axis;
      const weight = row.area * Math.abs(term.coefficient);
      result[offset] += weight * velocity[row.id];
      weights[offset] += weight;
    }
  }
  // Domain-wall faces are fixed zero-velocity ports and therefore do not need
  // pressure rows. They still contribute one side of the MAC-to-cell average.
  // Without their zero-valued weight, boundary cells use the sole interior
  // face at full strength (twice Uniform CM12's collocated wall velocity).
  for (const cell of grid.cells) for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
    const axis = axisIndex as SparseAtlasAxis;
    if (cell.minimumFine[axis] !== 0
      && cell.maximumFine[axis] !== grid.atlas.dimensions[axis]) continue;
    const tangents = tangentialAxes(axis);
    weights[3 * cell.id + axis] += cell.widthsFine[tangents[0]]
      * cell.widthsFine[tangents[1]] / cell.widthsFine[axis];
  }
  for (let index = 0; index < result.length; index += 1) {
    if (weights[index] > 0) result[index] /= weights[index];
  }
  return result;
}

/**
 * Project arbitrary sparse-atlas face velocities with matrix-free Jacobi-PCG.
 * Dirichlet components are solved as SPD systems; fully enclosed components
 * are solved on their exact per-component zero-mean quotient.
 */
export function projectSparseAtlasVelocity(
  atlasOrGrid: SparseAdaptiveMassAtlas | SparseAtlasCompositeGrid,
  options: SparseAtlasProjectionOptions = {},
  workspace: SparseAtlasProjectionWorkspace = createSparseAtlasProjectionWorkspace(),
): SparseAtlasProjectionResult {
  const sparseAirPhi = options.sparseAirPhi ?? 0.5;
  const grid = "gradientRows" in atlasOrGrid
    ? atlasOrGrid
    : buildSparseAtlasCompositeGrid(atlasOrGrid, sparseAirPhi);
  const atlas = grid.atlas;
  const faceCount = grid.gradientRows.length;
  const cellCount = grid.cells.length;
  const velocityBefore = workspace.velocityBefore = exactFloat64(
    workspace.velocityBefore, faceCount,
  );
  if (options.normalVelocity) {
    assertVectorLength(options.normalVelocity, faceCount, "normalVelocity");
    for (let id = 0; id < faceCount; id += 1) velocityBefore[id] = options.normalVelocity[id];
  } else velocityBefore.fill(0);
  assertVectorLength(velocityBefore, grid.gradientRows.length, "normalVelocity");
  const relativeTolerance = options.relativeTolerance ?? 1e-10;
  const absoluteTolerance = options.absoluteTolerance ?? 1e-12;
  const maximumIterations = options.maximumIterations ?? Math.max(64, 4 * grid.cells.length);
  const denominatorEpsilon = options.denominatorEpsilon ?? 1e-12;
  if (!(relativeTolerance > 0) || !(absoluteTolerance >= 0)
    || !Number.isInteger(maximumIterations) || maximumIterations <= 0
    || !(denominatorEpsilon > 0)) {
    throw new RangeError("invalid sparse-atlas projection solve options");
  }
  const system = buildLiquidSystem(
    grid, options.phi, denominatorEpsilon, sparseAirPhi, workspace,
  );
  options.onStageComplete?.("topology");
  const rhs = workspace.rhs = exactFloat64(workspace.rhs, cellCount);
  assembleLiquidRhs(system, velocityBefore, options.targetDivergence, rhs);
  const rhsCompatibilityMaxAbs = projectNullspace(system, rhs);
  const rhsNorm = norm(rhs);
  options.onStageComplete?.("rhs");
  const pressure = workspace.pressure = exactFloat64(workspace.pressure, cellCount);
  if (options.initialPressure) {
    assertVectorLength(options.initialPressure, cellCount, "initialPressure");
    if (options.initialPressure !== pressure) {
      for (let id = 0; id < cellCount; id += 1) pressure[id] = options.initialPressure[id];
    }
  } else pressure.fill(0);
  assertVectorLength(pressure, grid.cells.length, "initialPressure");
  projectNullspace(system, pressure);
  const residual = workspace.residual = exactFloat64(workspace.residual, cellCount);
  residual.set(rhs);
  const applied = workspace.applied = exactFloat64(workspace.applied, cellCount);
  if (options.initialPressure) {
    const initialImage = applyLiquidOperator(system, pressure, applied);
    for (let index = 0; index < residual.length; index += 1) {
      residual[index] -= initialImage[index];
    }
    projectNullspace(system, residual);
  }
  const preconditioned = workspace.preconditioned = exactFloat64(
    workspace.preconditioned, cellCount,
  );
  for (let cellId = 0; cellId < cellCount; cellId += 1) {
    preconditioned[cellId] = system.diagonal[cellId] > 0
      ? residual[cellId] / system.diagonal[cellId] : 0;
  }
  projectNullspace(system, preconditioned);
  const direction = workspace.direction = exactFloat64(workspace.direction, cellCount);
  direction.set(preconditioned);
  // One stable residual norm per iteration is sufficient. The former loop
  // condition recomputed the same O(n) Kahan dot after the previous tail,
  // adding a full leaf sweep to every PCG iteration.
  let residualPreconditioned = dot(residual, preconditioned);
  let residualNorm = norm(residual);
  const initialResidualNorm = residualNorm;
  const initialMaximumResidual = maximumAbsolute(residual);
  let iterations = 0;
  const target = Math.max(absoluteTolerance, relativeTolerance * rhsNorm);

  while (iterations < maximumIterations && residualNorm > target) {
    applyLiquidOperator(system, direction, applied);
    projectNullspace(system, applied);
    const curvature = dot(direction, applied);
    if (!(curvature > 0) || !Number.isFinite(curvature)) {
      throw new Error(`sparse-atlas pressure PCG lost positive curvature at iteration ${iterations}`);
    }
    const alpha = residualPreconditioned / curvature;
    for (let index = 0; index < pressure.length; index += 1) {
      pressure[index] += alpha * direction[index];
      residual[index] -= alpha * applied[index];
    }
    projectNullspace(system, pressure);
    projectNullspace(system, residual);
    iterations += 1;
    residualNorm = norm(residual);
    if (residualNorm <= target) break;
    for (let index = 0; index < preconditioned.length; index += 1) {
      preconditioned[index] = system.diagonal[index] > 0
        ? residual[index] / system.diagonal[index] : 0;
    }
    projectNullspace(system, preconditioned);
    const nextResidualPreconditioned = dot(residual, preconditioned);
    const beta = nextResidualPreconditioned / residualPreconditioned;
    for (let index = 0; index < direction.length; index += 1) {
      direction[index] = preconditioned[index] + beta * direction[index];
    }
    projectNullspace(system, direction);
    residualPreconditioned = nextResidualPreconditioned;
  }

  const trueResidual = workspace.trueResidual = exactFloat64(
    workspace.trueResidual, cellCount,
  );
  applyLiquidOperator(system, pressure, trueResidual);
  for (let index = 0; index < trueResidual.length; index += 1) {
    trueResidual[index] -= rhs[index];
  }
  projectNullspace(system, trueResidual);
  options.onStageComplete?.("solve");
  const correction = workspace.correction = exactFloat64(workspace.correction, faceCount);
  pressureCorrection(system, pressure, correction);
  const activeRows = workspace.activeRows = exactUint8(workspace.activeRows, faceCount);
  activeRows.fill(0);
  for (const row of system.rows) activeRows[row.id] = 1;
  let maximumInactiveFaceVelocityBefore = 0;
  const projectedFaceVelocity = workspace.projectedFaceVelocity = exactFloat64(
    workspace.projectedFaceVelocity, faceCount,
  );
  for (let rowId = 0; rowId < faceCount; rowId += 1) {
      const value = velocityBefore[rowId];
      if (activeRows[rowId] === 0) {
        maximumInactiveFaceVelocityBefore = Math.max(
          maximumInactiveFaceVelocityBefore,
          Math.abs(value),
        );
        // Sparse CM12 has no air-velocity authority. Carrying a forced value
        // on a pressure-inactive row stores ballistic air momentum; when that
        // row later becomes a liquid interface, the stale value is injected
        // into transport and appears as surface boiling. Velocity extension
        // can replace this zero boundary condition later, but inactive values
        // must never persist as physical state.
        projectedFaceVelocity[rowId] = 0;
        continue;
      }
      projectedFaceVelocity[rowId] = value - correction[rowId];
  }
  let maximumInactiveFaceVelocityAfter = 0;
  for (const row of grid.gradientRows) {
    if (activeRows[row.id] === 0) {
      maximumInactiveFaceVelocityAfter = Math.max(
        maximumInactiveFaceVelocityAfter,
        Math.abs(projectedFaceVelocity[row.id]),
      );
    }
  }
  options.onStageComplete?.("projection");
  // This is the physical finite-volume flux imbalance. Do not apply the
  // pressure system's quotient-space nullspace projector here: doing so would
  // hide a componentwise compatibility/net-flux defect from the published
  // divergence receipt. Nullspace projection belongs only to PCG algebra.
  const projectedEquationResidual = workspace.projectedEquationResidual = exactFloat64(
    workspace.projectedEquationResidual, cellCount,
  );
  assembleLiquidRhs(
    system, projectedFaceVelocity, options.targetDivergence, projectedEquationResidual,
  );
  const preDivergence = workspace.preDivergence = exactFloat64(
    workspace.preDivergence, cellCount,
  );
  divergenceFromEquationResidual(system, rhs, preDivergence);
  const leafDivergence = workspace.leafDivergence = exactFloat64(
    workspace.leafDivergence, cellCount,
  );
  divergenceFromEquationResidual(system, projectedEquationResidual, leafDivergence);
  let postMixedSeamDivergenceMaximum = 0;
  for (const row of system.rows) {
    if (row.kind !== "mixed-seam") continue;
    for (const term of row.terms) {
      postMixedSeamDivergenceMaximum = Math.max(
        postMixedSeamDivergenceMaximum,
        Math.abs(leafDivergence[term.cellId]),
      );
    }
  }
  const preDivergenceVolumeL2 = volumeL2(grid, preDivergence);
  const postDivergenceVolumeL2 = volumeL2(grid, leafDivergence);
  const kineticEnergyBefore = faceEnergy(system, velocityBefore);
  const kineticEnergyAfter = faceEnergy(system, projectedFaceVelocity);
  const pressureCorrectionEnergy = faceEnergy(system, correction);
  options.onStageComplete?.("diagnostics");
  let liquidCellCount = 0;
  for (let id = 0; id < system.liquid.length; id += 1) {
    liquidCellCount += system.liquid[id];
  }
  let inactiveRowCount = 0;
  for (let id = 0; id < activeRows.length; id += 1) {
    inactiveRowCount += activeRows[id] === 0 ? 1 : 0;
  }
  const converged = norm(trueResidual) <= target;
  let anchoredComponentCount = 0;
  for (let component = 0; component < system.componentCount; component += 1) {
    anchoredComponentCount += system.componentAnchored[component] !== 0 ? 1 : 0;
  }
  const unanchoredComponentCount = system.componentCount - anchoredComponentCount;
  const initialRelativeResidualL2 = rhsNorm > 0 ? initialResidualNorm / rhsNorm : 0;
  const zeroSeedInitialRelativeResidualL2 = rhsNorm > 0 ? 1 : 0;
  const relativeResidualL2 = rhsNorm > 0 ? norm(trueResidual) / rhsNorm : 0;
  const maximumResidual = maximumAbsolute(trueResidual);
  const preDivergenceMaximum = maximumAbsolute(preDivergence);
  const postDivergenceMaximum = maximumAbsolute(leafDivergence);
  const divergenceReduction = preDivergenceVolumeL2 > 0
    ? postDivergenceVolumeL2 / preDivergenceVolumeL2 : 0;
  const energyIdentityAbsError = Math.abs(
    kineticEnergyBefore - kineticEnergyAfter - pressureCorrectionEnergy,
  );
  let receipt = workspace.receipt;
  if (!receipt) {
    receipt = workspace.receipt = {
      iterations, converged, componentCount: system.componentCount,
      anchoredComponentCount, unanchoredComponentCount,
      liquidCellCount, activeRowCount: system.rows.length,
      cutRowCount: system.cutRowCount, mixedSeamRowCount: grid.mixedSeamRowCount,
      cutMixedSeamRowCount: system.cutMixedSeamRowCount,
      thetaClampCount: system.thetaClampCount, minimumTheta: system.minimumTheta,
      rhsCompatibilityMaxAbs, initialRelativeResidualL2,
      zeroSeedInitialRelativeResidualL2, initialMaximumResidual,
      relativeResidualL2, maximumResidual,
      preDivergenceVolumeL2, postDivergenceVolumeL2, preDivergenceMaximum,
      postDivergenceMaximum, divergenceReduction, kineticEnergyBefore,
      kineticEnergyAfter, pressureCorrectionEnergy, energyIdentityAbsError,
      inactiveRowCount, maximumInactiveFaceVelocityBefore,
      maximumInactiveFaceVelocityAfter, postMixedSeamDivergenceMaximum,
    };
  } else {
    const mutable = receipt as Mutable<SparseAtlasProjectionReceipt>;
    mutable.iterations = iterations;
    mutable.converged = converged;
    mutable.componentCount = system.componentCount;
    mutable.anchoredComponentCount = anchoredComponentCount;
    mutable.unanchoredComponentCount = unanchoredComponentCount;
    mutable.liquidCellCount = liquidCellCount;
    mutable.activeRowCount = system.rows.length;
    mutable.cutRowCount = system.cutRowCount;
    mutable.mixedSeamRowCount = grid.mixedSeamRowCount;
    mutable.cutMixedSeamRowCount = system.cutMixedSeamRowCount;
    mutable.thetaClampCount = system.thetaClampCount;
    mutable.minimumTheta = system.minimumTheta;
    mutable.rhsCompatibilityMaxAbs = rhsCompatibilityMaxAbs;
    mutable.initialRelativeResidualL2 = initialRelativeResidualL2;
    mutable.zeroSeedInitialRelativeResidualL2 = zeroSeedInitialRelativeResidualL2;
    mutable.initialMaximumResidual = initialMaximumResidual;
    mutable.relativeResidualL2 = relativeResidualL2;
    mutable.maximumResidual = maximumResidual;
    mutable.preDivergenceVolumeL2 = preDivergenceVolumeL2;
    mutable.postDivergenceVolumeL2 = postDivergenceVolumeL2;
    mutable.preDivergenceMaximum = preDivergenceMaximum;
    mutable.postDivergenceMaximum = postDivergenceMaximum;
    mutable.divergenceReduction = divergenceReduction;
    mutable.kineticEnergyBefore = kineticEnergyBefore;
    mutable.kineticEnergyAfter = kineticEnergyAfter;
    mutable.pressureCorrectionEnergy = pressureCorrectionEnergy;
    mutable.energyIdentityAbsError = energyIdentityAbsError;
    mutable.inactiveRowCount = inactiveRowCount;
    mutable.maximumInactiveFaceVelocityBefore = maximumInactiveFaceVelocityBefore;
    mutable.maximumInactiveFaceVelocityAfter = maximumInactiveFaceVelocityAfter;
    mutable.postMixedSeamDivergenceMaximum = postMixedSeamDivergenceMaximum;
  }
  const leafCollocatedVelocity = collocateSparseAtlasVelocity(
    grid,
    projectedFaceVelocity,
    workspace.collocatedVelocity = exactFloat64(workspace.collocatedVelocity, 3 * cellCount),
    workspace.collocationWeights = exactFloat64(workspace.collocationWeights, 3 * cellCount),
  );
  if (!workspace.result) {
    workspace.result = {
      atlas, grid, leafPressure: pressure, leafRhs: rhs,
      leafDiagonal: system.diagonal, leafDivergenceBefore: preDivergence,
      leafDivergence, leafCollocatedVelocity, projectedFaceVelocity, receipt,
    };
  } else {
    const result = workspace.result as Mutable<SparseAtlasProjectionResult>;
    result.atlas = atlas;
    result.grid = grid;
    result.leafPressure = pressure;
    result.leafRhs = rhs;
    result.leafDiagonal = system.diagonal;
    result.leafDivergenceBefore = preDivergence;
    result.leafDivergence = leafDivergence;
    result.leafCollocatedVelocity = leafCollocatedVelocity;
    result.projectedFaceVelocity = projectedFaceVelocity;
    result.receipt = receipt;
  }
  return workspace.result;
}

/** Materialise one compact leaf scalar onto the bounded finest lattice. */
export function materializeSparseAtlasLeafScalar(
  grid: SparseAtlasCompositeGrid,
  values: ArrayLike<number>,
  emptyValue = 0,
  output?: Float32Array,
): Float32Array {
  assertVectorLength(values, grid.cells.length, "leaf scalar");
  const dimensions = grid.atlas.dimensions;
  const count = dimensions[0] * dimensions[1] * dimensions[2];
  const result = output?.length === count ? output : new Float32Array(count);
  result.fill(emptyValue);
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1) {
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1) {
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          result[x + dimensions[0] * (y + dimensions[1] * z)] = values[cell.id];
        }
      }
    }
  }
  return result;
}

/** Materialise XYZ-per-leaf values as a dense interleaved finest lattice. */
export function materializeSparseAtlasCollocatedVelocity(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
  output?: Float32Array,
): Float32Array {
  assertVectorLength(velocity, 3 * grid.cells.length, "leaf collocated velocity");
  const dimensions = grid.atlas.dimensions;
  const count = 3 * dimensions[0] * dimensions[1] * dimensions[2];
  const result = output?.length === count ? output : new Float32Array(count);
  result.fill(0);
  for (const cell of grid.cells) {
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1) {
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1) {
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          const dense = 3 * (x + dimensions[0] * (y + dimensions[1] * z));
          const compact = 3 * cell.id;
          result[dense] = velocity[compact];
          result[dense + 1] = velocity[compact + 1];
          result[dense + 2] = velocity[compact + 2];
        }
      }
    }
  }
  return result;
}

export const materializeSparseAtlasPressure = (
  result: SparseAtlasProjectionResult,
  output?: Float32Array,
): Float32Array => materializeSparseAtlasLeafScalar(
  result.grid, result.leafPressure, 0, output,
);

export const materializeSparseAtlasRhs = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafRhs);

export const materializeSparseAtlasDiagonal = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafDiagonal);

export const materializeSparseAtlasDivergence = (
  result: SparseAtlasProjectionResult,
  output?: Float32Array,
): Float32Array => materializeSparseAtlasLeafScalar(
  result.grid, result.leafDivergence, 0, output,
);

export const materializeSparseAtlasDivergenceBefore = (
  result: SparseAtlasProjectionResult,
): Float32Array => materializeSparseAtlasLeafScalar(result.grid, result.leafDivergenceBefore);

export const materializeSparseAtlasVelocity = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasCollocatedVelocity(result.grid, result.leafCollocatedVelocity);
