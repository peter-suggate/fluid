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
  sparseBrickKey,
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
}

export interface SparseAtlasProjectionOptions {
  /** One oriented velocity per `grid.gradientRows` entry. Defaults to zero. */
  readonly normalVelocity?: ArrayLike<number>;
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
  readonly liquidCellCount: number;
  readonly activeRowCount: number;
  readonly cutRowCount: number;
  readonly mixedSeamRowCount: number;
  readonly cutMixedSeamRowCount: number;
  readonly thetaClampCount: number;
  readonly minimumTheta: number;
  readonly rhsCompatibilityMaxAbs: number;
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

interface ActiveTerm {
  readonly cellId: number;
  readonly coefficient: number;
}

interface ActiveRow {
  readonly source: SparseAtlasGradientRow;
  readonly terms: readonly ActiveTerm[];
  readonly theta: number;
  readonly cut: boolean;
}

interface LiquidSystem {
  readonly grid: SparseAtlasCompositeGrid;
  readonly phi: Float64Array;
  readonly liquid: Uint8Array;
  readonly rows: readonly ActiveRow[];
  readonly anchored: Uint8Array;
  readonly componentByCell: Int32Array;
  readonly componentAnchored: readonly boolean[];
  readonly componentCells: readonly (readonly number[])[];
  readonly diagonal: Float64Array;
  readonly cutRowCount: number;
  readonly cutMixedSeamRowCount: number;
  readonly thetaClampCount: number;
  readonly minimumTheta: number;
}

const BRICK_FINE_WIDTH = 8;

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

function brickInside(coordinate: SparseBrickVec3, dimensions: SparseBrickVec3): boolean {
  return coordinate[0] >= 0 && coordinate[1] >= 0 && coordinate[2] >= 0
    && coordinate[0] < dimensions[0]
    && coordinate[1] < dimensions[1]
    && coordinate[2] < dimensions[2];
}

function overlapLength(a0: number, a1: number, b0: number, b1: number): number {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

/** Build the sole authoritative G-row topology for all resident bricks. */
export function buildSparseAtlasCompositeGrid(
  atlas: SparseAdaptiveMassAtlas,
  sparseAirPhi = 0.5,
): SparseAtlasCompositeGrid {
  if (!Number.isFinite(sparseAirPhi) || sparseAirPhi <= 0) {
    throw new RangeError("sparseAirPhi must be finite and positive");
  }
  const bricks = [...atlas.bricks].sort((left, right) => left.key - right.key);
  const cells: SparseAtlasCompositeCell[] = [];
  const cellBaseByBrick = new Map<number, number>();
  const cellIdByBrickLocal = new Map<string, number>();

  for (const brick of bricks) {
    cellBaseByBrick.set(brick.key, cells.length);
    const scale = BRICK_FINE_WIDTH / brick.resolution;
    for (let z = 0; z < brick.resolution; z += 1) {
      for (let y = 0; y < brick.resolution; y += 1) {
        for (let x = 0; x < brick.resolution; x += 1) {
          const local = [x, y, z] as const;
          const index = localIndex(local, brick.resolution);
          const minimum = [
            brick.coordinate[0] * BRICK_FINE_WIDTH + x * scale,
            brick.coordinate[1] * BRICK_FINE_WIDTH + y * scale,
            brick.coordinate[2] * BRICK_FINE_WIDTH + z * scale,
          ] as const;
          const maximum = minimum.map((value, axis) =>
            Math.min(value + scale, atlas.dimensions[axis])) as [number, number, number];
          const widths = maximum.map((value, axis) =>
            value - minimum[axis]) as [number, number, number];
          if (widths.some((value) => value <= 0)) continue;
          const cellId = cells.length;
          cells.push({
            id: cellId,
            stableLeafId: brick.key * 512 + index,
            brickKey: brick.key,
            brickCoordinate: brick.coordinate,
            brickResolution: brick.resolution,
            local,
            localIndex: index,
            minimumFine: minimum,
            maximumFine: maximum,
            centerFine: maximum.map((value, axis) =>
              0.5 * (minimum[axis] + value)) as [number, number, number],
            widthsFine: widths,
            volume: widths[0] * widths[1] * widths[2],
            volumeFineCells: widths[0] * widths[1] * widths[2],
            density: brick.density[index],
            gamma: brick.gamma[index],
          });
          cellIdByBrickLocal.set(`${brick.key}:${index}`, cellId);
        }
      }
    }
  }

  const rows: SparseAtlasGradientRow[] = [];
  const cellFor = (brick: SparseAdaptiveMassBrick, local: SparseBrickVec3) => {
    const id = cellIdByBrickLocal.get(`${brick.key}:${localIndex(local, brick.resolution)}`);
    return id === undefined ? undefined : cells[id];
  };

  const appendRow = (input: Omit<
    SparseAtlasGradientRow,
    "id" | "dualWeight" | "areaFineCells2" | "centerDistanceFine"
  >): void => {
    rows.push({
      ...input,
      id: rows.length,
      dualWeight: input.area * input.distance,
      areaFineCells2: input.area,
      centerDistanceFine: input.distance,
    });
  };

  // Ordinary faces wholly inside a brick.
  for (const brick of bricks) {
    for (const axis of [0, 1, 2] as const) {
      const tangents = tangentialAxes(axis);
      for (let z = 0; z < brick.resolution; z += 1) {
        for (let y = 0; y < brick.resolution; y += 1) {
          for (let x = 0; x < brick.resolution; x += 1) {
            const positiveLocal = [x, y, z] as [number, number, number];
            if (positiveLocal[axis] === 0) continue;
            const negativeLocal = [...positiveLocal] as [number, number, number];
            negativeLocal[axis] -= 1;
            const negative = cellFor(brick, negativeLocal);
            const positive = cellFor(brick, positiveLocal);
            if (!negative || !positive) continue;
            const distance = positive.centerFine[axis] - negative.centerFine[axis];
            const area = overlapLength(
              negative.minimumFine[tangents[0]], negative.maximumFine[tangents[0]],
              positive.minimumFine[tangents[0]], positive.maximumFine[tangents[0]],
            ) * overlapLength(
              negative.minimumFine[tangents[1]], negative.maximumFine[tangents[1]],
              positive.minimumFine[tangents[1]], positive.maximumFine[tangents[1]],
            );
            const center = mutableVector();
            center[axis] = negative.maximumFine[axis];
            center[tangents[0]] = 0.5 * (negative.minimumFine[tangents[0]]
              + negative.maximumFine[tangents[0]]);
            center[tangents[1]] = 0.5 * (negative.minimumFine[tangents[1]]
              + negative.maximumFine[tangents[1]]);
            appendRow({
              kind: "intra-brick",
              axis,
              centerFine: center,
              area,
              distance,
              terms: [
                { cellId: negative.id, coefficient: -1 / distance },
                { cellId: positive.id, coefficient: 1 / distance },
              ],
              negativeBrickKey: brick.key,
              positiveBrickKey: brick.key,
            });
          }
        }
      }
    }
  }

  const faceCells = (brick: SparseAdaptiveMassBrick, axis: SparseAtlasAxis, side: -1 | 1) => {
    const coordinate = side < 0 ? 0 : brick.resolution - 1;
    const result: SparseAtlasCompositeCell[] = [];
    for (let z = 0; z < brick.resolution; z += 1) {
      for (let y = 0; y < brick.resolution; y += 1) {
        for (let x = 0; x < brick.resolution; x += 1) {
          const local = [x, y, z] as [number, number, number];
          if (local[axis] === coordinate) {
            const cell = cellFor(brick, local);
            if (cell) result.push(cell);
          }
        }
      }
    }
    return result;
  };

  const appendBrickInterface = (
    negative: SparseAdaptiveMassBrick,
    positive: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
  ): void => {
    const tangents = tangentialAxes(axis);
    const portWidth = Math.max(
      BRICK_FINE_WIDTH / negative.resolution,
      BRICK_FINE_WIDTH / positive.resolution,
    );
    const negativeCells = faceCells(negative, axis, 1);
    const positiveCells = faceCells(positive, axis, -1);
    const brickMinimum = negative.coordinate.map((value) =>
      value * BRICK_FINE_WIDTH) as [number, number, number];
    const faceCoordinate = (negative.coordinate[axis] + 1) * BRICK_FINE_WIDTH;
    for (let portV = 0; portV < BRICK_FINE_WIDTH; portV += portWidth) {
      for (let portU = 0; portU < BRICK_FINE_WIDTH; portU += portWidth) {
        const minimum = mutableVector();
        const maximum = mutableVector();
        minimum[axis] = maximum[axis] = faceCoordinate;
        minimum[tangents[0]] = brickMinimum[tangents[0]] + portU;
        maximum[tangents[0]] = Math.min(
          minimum[tangents[0]] + portWidth, atlas.dimensions[tangents[0]],
        );
        minimum[tangents[1]] = brickMinimum[tangents[1]] + portV;
        maximum[tangents[1]] = Math.min(
          minimum[tangents[1]] + portWidth, atlas.dimensions[tangents[1]],
        );
        const area = (maximum[tangents[0]] - minimum[tangents[0]])
          * (maximum[tangents[1]] - minimum[tangents[1]]);
        if (!(area > 0)) continue;
        const negativeOnPort = negativeCells.map((cell) => ({
          cell,
          overlap: overlapLength(minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]])
            * overlapLength(minimum[tangents[1]], maximum[tangents[1]],
              cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]]),
        })).filter(({ overlap }) => overlap > 0);
        const positiveOnPort = positiveCells.map((cell) => ({
          cell,
          overlap: overlapLength(minimum[tangents[0]], maximum[tangents[0]],
            cell.minimumFine[tangents[0]], cell.maximumFine[tangents[0]])
            * overlapLength(minimum[tangents[1]], maximum[tangents[1]],
              cell.minimumFine[tangents[1]], cell.maximumFine[tangents[1]]),
        })).filter(({ overlap }) => overlap > 0);
        if (negativeOnPort.length === 0 || positiveOnPort.length === 0) continue;
        const negativeCenter = negativeOnPort.reduce(
          (sum, item) => sum + item.overlap * item.cell.centerFine[axis], 0,
        ) / area;
        const positiveCenter = positiveOnPort.reduce(
          (sum, item) => sum + item.overlap * item.cell.centerFine[axis], 0,
        ) / area;
        const distance = positiveCenter - negativeCenter;
        const center = mutableVector();
        center[axis] = faceCoordinate;
        center[tangents[0]] = 0.5 * (minimum[tangents[0]] + maximum[tangents[0]]);
        center[tangents[1]] = 0.5 * (minimum[tangents[1]] + maximum[tangents[1]]);
        appendRow({
          kind: negative.resolution === positive.resolution ? "brick-face" : "mixed-seam",
          axis,
          centerFine: center,
          area,
          distance,
          terms: [
            ...negativeOnPort.map(({ cell, overlap }) => ({
              cellId: cell.id, coefficient: -overlap / (area * distance),
            })),
            ...positiveOnPort.map(({ cell, overlap }) => ({
              cellId: cell.id, coefficient: overlap / (area * distance),
            })),
          ],
          negativeBrickKey: negative.key,
          positiveBrickKey: positive.key,
        });
      }
    }
  };

  const appendSparseAirFace = (
    brick: SparseAdaptiveMassBrick,
    axis: SparseAtlasAxis,
    side: -1 | 1,
  ): void => {
    const tangents = tangentialAxes(axis);
    for (const cell of faceCells(brick, axis, side)) {
      const distance = cell.widthsFine[axis];
      const area = cell.widthsFine[tangents[0]] * cell.widthsFine[tangents[1]];
      const center = [...cell.centerFine] as [number, number, number];
      center[axis] = side < 0 ? cell.minimumFine[axis] : cell.maximumFine[axis];
      appendRow({
        kind: "sparse-air",
        axis,
        centerFine: center,
        area,
        distance,
        terms: [{ cellId: cell.id, coefficient: side < 0 ? 1 / distance : -1 / distance }],
        ...(side < 0 ? { positiveBrickKey: brick.key } : { negativeBrickKey: brick.key }),
        exteriorPhi: sparseAirPhi,
      });
    }
  };

  // Cross-brick interfaces are emitted once in +axis order. Missing in-domain
  // neighbours are sparse-air boundaries; outer-domain neighbours are walls.
  for (const brick of bricks) {
    for (const axis of [0, 1, 2] as const) {
      const positiveCoordinate = [...brick.coordinate] as [number, number, number];
      positiveCoordinate[axis] += 1;
      if (brickInside(positiveCoordinate, atlas.brickDimensions)) {
        const positive = atlas.directory.get(sparseBrickKey(positiveCoordinate, atlas.brickDimensions));
        if (positive) appendBrickInterface(brick, positive, axis);
        else appendSparseAirFace(brick, axis, 1);
      }
      const negativeCoordinate = [...brick.coordinate] as [number, number, number];
      negativeCoordinate[axis] -= 1;
      if (brickInside(negativeCoordinate, atlas.brickDimensions)) {
        const negative = atlas.directory.get(sparseBrickKey(negativeCoordinate, atlas.brickDimensions));
        if (!negative) appendSparseAirFace(brick, axis, -1);
      }
    }
  }

  return {
    atlas,
    cells,
    gradientRows: rows,
    cellBaseByBrick,
    mixedSeamRowCount: rows.filter((row) => row.kind === "mixed-seam").length,
    sparseAirRowCount: rows.filter((row) => row.kind === "sparse-air").length,
  };
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
  for (const cell of grid.cells) output[cell.id] /= cell.volume;
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
): LiquidSystem {
  const phi = phiInput
    ? Float64Array.from(phiInput)
    : Float64Array.from(grid.cells, (cell) => CM12_LIQUID_ISOVALUE - cell.density);
  assertVectorLength(phi, grid.cells.length, "phi");
  const liquid = Uint8Array.from(phi, (value) => value <= 0 ? 1 : 0);
  const rows: ActiveRow[] = [];
  const diagonal = new Float64Array(grid.cells.length);
  const anchored = new Uint8Array(grid.cells.length);
  let cutRowCount = 0;
  let cutMixedSeamRowCount = 0;
  let thetaClampCount = 0;
  let minimumTheta = 1;

  for (const source of grid.gradientRows) {
    const terms = source.terms.filter((term) => liquid[term.cellId] !== 0);
    if (terms.length === 0) continue;
    const airTerms = source.terms.filter((term) => liquid[term.cellId] === 0);
    const hasExteriorAir = source.kind === "sparse-air";
    const cut = airTerms.length > 0 || hasExteriorAir;
    let theta = 1;
    if (cut) {
      let liquidPhiSum = 0;
      let liquidWeight = 0;
      for (const term of terms) {
        const weight = Math.abs(term.coefficient);
        liquidPhiSum += weight * phi[term.cellId];
        liquidWeight += weight;
      }
      let airPhiSum = 0;
      let airWeight = 0;
      for (const term of airTerms) {
        const weight = Math.abs(term.coefficient);
        airPhiSum += weight * phi[term.cellId];
        airWeight += weight;
      }
      if (hasExteriorAir) {
        const weight = terms.reduce((sum, term) => sum + Math.abs(term.coefficient), 0);
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
      for (const term of terms) anchored[term.cellId] = 1;
    }
    minimumTheta = Math.min(minimumTheta, theta);
    const active = { source, terms, theta, cut } satisfies ActiveRow;
    rows.push(active);
    for (const term of terms) {
      diagonal[term.cellId] += source.dualWeight
        * term.coefficient * term.coefficient / theta;
    }
  }

  // Connected liquid components establish the exact nullspace projector.
  const neighbors = Array.from({ length: grid.cells.length }, () => [] as number[]);
  for (const row of rows) {
    for (let left = 0; left < row.terms.length; left += 1) {
      for (let right = left + 1; right < row.terms.length; right += 1) {
        neighbors[row.terms[left].cellId].push(row.terms[right].cellId);
        neighbors[row.terms[right].cellId].push(row.terms[left].cellId);
      }
    }
  }
  const componentByCell = new Int32Array(grid.cells.length).fill(-1);
  const componentCells: number[][] = [];
  const componentAnchored: boolean[] = [];
  for (const cell of grid.cells) {
    if (!liquid[cell.id] || componentByCell[cell.id] >= 0) continue;
    const component = componentCells.length;
    const members: number[] = [];
    const stack = [cell.id];
    componentByCell[cell.id] = component;
    let hasAnchor = false;
    while (stack.length > 0) {
      const current = stack.pop();
      if (current === undefined) break;
      members.push(current);
      hasAnchor ||= anchored[current] !== 0;
      for (const neighbor of neighbors[current]) {
        if (componentByCell[neighbor] >= 0) continue;
        componentByCell[neighbor] = component;
        stack.push(neighbor);
      }
    }
    componentCells.push(members);
    componentAnchored.push(hasAnchor);
  }
  return {
    grid, phi, liquid, rows, anchored, componentByCell, componentAnchored, componentCells,
    diagonal, cutRowCount, cutMixedSeamRowCount, thetaClampCount, minimumTheta,
  };
}

function applyLiquidOperator(
  system: LiquidSystem,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(system.grid.cells.length),
): Float64Array {
  output.fill(0);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) jump += term.coefficient * pressure[term.cellId];
    const weighted = row.source.dualWeight * jump / row.theta;
    for (const term of row.terms) output[term.cellId] += term.coefficient * weighted;
  }
  return output;
}

function projectNullspace(system: LiquidSystem, values: Float64Array): number {
  let maximumRemoved = 0;
  for (let component = 0; component < system.componentCells.length; component += 1) {
    if (system.componentAnchored[component]) continue;
    const members = system.componentCells[component];
    let sum = 0;
    for (const cellId of members) sum += values[cellId];
    const mean = sum / members.length;
    maximumRemoved = Math.max(maximumRemoved, Math.abs(sum));
    for (const cellId of members) values[cellId] -= mean;
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

function assembleLiquidRhs(system: LiquidSystem, velocity: ArrayLike<number>): Float64Array {
  const rhs = new Float64Array(system.grid.cells.length);
  for (const row of system.rows) {
    const weighted = row.source.dualWeight * velocity[row.source.id];
    for (const term of row.terms) rhs[term.cellId] += term.coefficient * weighted;
  }
  return rhs;
}

function pressureCorrection(system: LiquidSystem, pressure: ArrayLike<number>): Float64Array {
  const correction = new Float64Array(system.grid.gradientRows.length);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) jump += term.coefficient * pressure[term.cellId];
    correction[row.source.id] = jump / row.theta;
  }
  return correction;
}

function faceEnergy(system: LiquidSystem, velocity: ArrayLike<number>): number {
  let energy = 0;
  for (const row of system.rows) {
    const value = velocity[row.source.id];
    energy += 0.5 * row.source.dualWeight * row.theta * value * value;
  }
  return energy;
}

function divergenceFromEquationResidual(
  system: LiquidSystem,
  residual: ArrayLike<number>,
): Float64Array {
  return Float64Array.from(system.grid.cells, (cell) =>
    system.liquid[cell.id] ? -residual[cell.id] / cell.volume : 0);
}

function volumeL2(grid: SparseAtlasCompositeGrid, values: ArrayLike<number>): number {
  let squared = 0;
  for (const cell of grid.cells) squared += cell.volume * values[cell.id] * values[cell.id];
  return Math.sqrt(squared);
}

function collocateVelocity(
  grid: SparseAtlasCompositeGrid,
  velocity: ArrayLike<number>,
): Float64Array {
  const result = new Float64Array(3 * grid.cells.length);
  const weights = new Float64Array(3 * grid.cells.length);
  for (const row of grid.gradientRows) {
    for (const term of row.terms) {
      const offset = 3 * term.cellId + row.axis;
      const weight = row.area * Math.abs(term.coefficient);
      result[offset] += weight * velocity[row.id];
      weights[offset] += weight;
    }
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
): SparseAtlasProjectionResult {
  const sparseAirPhi = options.sparseAirPhi ?? 0.5;
  const grid = "gradientRows" in atlasOrGrid
    ? atlasOrGrid
    : buildSparseAtlasCompositeGrid(atlasOrGrid, sparseAirPhi);
  const atlas = grid.atlas;
  const velocityBefore = options.normalVelocity
    ? Float64Array.from(options.normalVelocity)
    : new Float64Array(grid.gradientRows.length);
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
    grid, options.phi, denominatorEpsilon, sparseAirPhi,
  );
  options.onStageComplete?.("topology");
  const rhs = assembleLiquidRhs(system, velocityBefore);
  const rhsCompatibilityMaxAbs = projectNullspace(system, rhs);
  const rhsNorm = norm(rhs);
  options.onStageComplete?.("rhs");
  const pressure = options.initialPressure
    ? Float64Array.from(options.initialPressure)
    : new Float64Array(grid.cells.length);
  assertVectorLength(pressure, grid.cells.length, "initialPressure");
  projectNullspace(system, pressure);
  const residual = rhs.slice();
  if (options.initialPressure) {
    const initialImage = applyLiquidOperator(system, pressure);
    for (let index = 0; index < residual.length; index += 1) {
      residual[index] -= initialImage[index];
    }
    projectNullspace(system, residual);
  }
  const preconditioned = Float64Array.from(residual, (value, cellId) =>
    system.diagonal[cellId] > 0 ? value / system.diagonal[cellId] : 0);
  projectNullspace(system, preconditioned);
  const direction = preconditioned.slice();
  // One stable residual norm per iteration is sufficient. The former loop
  // condition recomputed the same O(n) Kahan dot after the previous tail,
  // adding a full leaf sweep to every PCG iteration.
  const applied = new Float64Array(grid.cells.length);
  let residualPreconditioned = dot(residual, preconditioned);
  let residualNorm = norm(residual);
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

  const trueResidual = applyLiquidOperator(system, pressure);
  for (let index = 0; index < trueResidual.length; index += 1) {
    trueResidual[index] -= rhs[index];
  }
  projectNullspace(system, trueResidual);
  options.onStageComplete?.("solve");
  const correction = pressureCorrection(system, pressure);
  const projectedFaceVelocity = Float64Array.from(
    velocityBefore, (value, rowId) => value - correction[rowId],
  );
  options.onStageComplete?.("projection");
  // This is the physical finite-volume flux imbalance. Do not apply the
  // pressure system's quotient-space nullspace projector here: doing so would
  // hide a componentwise compatibility/net-flux defect from the published
  // divergence receipt. Nullspace projection belongs only to PCG algebra.
  const projectedEquationResidual = assembleLiquidRhs(system, projectedFaceVelocity);
  const preDivergence = divergenceFromEquationResidual(system, rhs);
  const leafDivergence = divergenceFromEquationResidual(system, projectedEquationResidual);
  const preDivergenceVolumeL2 = volumeL2(grid, preDivergence);
  const postDivergenceVolumeL2 = volumeL2(grid, leafDivergence);
  const kineticEnergyBefore = faceEnergy(system, velocityBefore);
  const kineticEnergyAfter = faceEnergy(system, projectedFaceVelocity);
  const pressureCorrectionEnergy = faceEnergy(system, correction);
  options.onStageComplete?.("diagnostics");
  return {
    atlas,
    grid,
    leafPressure: pressure,
    leafRhs: rhs,
    leafDiagonal: system.diagonal,
    leafDivergenceBefore: preDivergence,
    leafDivergence,
    leafCollocatedVelocity: collocateVelocity(grid, projectedFaceVelocity),
    projectedFaceVelocity,
    receipt: {
      iterations,
      converged: norm(trueResidual) <= target,
      liquidCellCount: system.liquid.reduce((sum, value) => sum + value, 0),
      activeRowCount: system.rows.length,
      cutRowCount: system.cutRowCount,
      mixedSeamRowCount: grid.mixedSeamRowCount,
      cutMixedSeamRowCount: system.cutMixedSeamRowCount,
      thetaClampCount: system.thetaClampCount,
      minimumTheta: system.minimumTheta,
      rhsCompatibilityMaxAbs,
      relativeResidualL2: rhsNorm > 0 ? norm(trueResidual) / rhsNorm : 0,
      maximumResidual: maximumAbsolute(trueResidual),
      preDivergenceVolumeL2,
      postDivergenceVolumeL2,
      preDivergenceMaximum: maximumAbsolute(preDivergence),
      postDivergenceMaximum: maximumAbsolute(leafDivergence),
      divergenceReduction: preDivergenceVolumeL2 > 0
        ? postDivergenceVolumeL2 / preDivergenceVolumeL2 : 0,
      kineticEnergyBefore,
      kineticEnergyAfter,
      pressureCorrectionEnergy,
      energyIdentityAbsError: Math.abs(
        kineticEnergyBefore - kineticEnergyAfter - pressureCorrectionEnergy,
      ),
    },
  };
}

/** Materialise one compact leaf scalar onto the bounded finest lattice. */
export function materializeSparseAtlasLeafScalar(
  grid: SparseAtlasCompositeGrid,
  values: ArrayLike<number>,
  emptyValue = 0,
): Float32Array {
  assertVectorLength(values, grid.cells.length, "leaf scalar");
  const dimensions = grid.atlas.dimensions;
  const result = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  if (emptyValue !== 0) result.fill(emptyValue);
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
): Float32Array {
  assertVectorLength(velocity, 3 * grid.cells.length, "leaf collocated velocity");
  const dimensions = grid.atlas.dimensions;
  const result = new Float32Array(3 * dimensions[0] * dimensions[1] * dimensions[2]);
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

export const materializeSparseAtlasPressure = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafPressure);

export const materializeSparseAtlasRhs = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafRhs);

export const materializeSparseAtlasDiagonal = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafDiagonal);

export const materializeSparseAtlasDivergence = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasLeafScalar(result.grid, result.leafDivergence);

export const materializeSparseAtlasDivergenceBefore = (
  result: SparseAtlasProjectionResult,
): Float32Array => materializeSparseAtlasLeafScalar(result.grid, result.leafDivergenceBefore);

export const materializeSparseAtlasVelocity = (result: SparseAtlasProjectionResult): Float32Array =>
  materializeSparseAtlasCollocatedVelocity(result.grid, result.leafCollocatedVelocity);
