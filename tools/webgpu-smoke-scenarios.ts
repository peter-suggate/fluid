import type { SceneDescription } from "../lib/model";
import type { MethodProfile } from "../lib/methods";
import {
  getSceneWebGPUSmokeLane,
  getSceneWebGPUSmokeSuite,
  isSceneWebGPUSmokeId,
  sceneWebGPUSmokeIds,
  type SceneWebGPUSmokeId,
} from "../lib/scene-webgpu-smoke-catalog";
import type { SceneWebGPUSmokeLane, SceneWebGPUSmokeSuite } from "../lib/scene-webgpu-smoke";

export const smokeScenarioIds = sceneWebGPUSmokeIds;

export type SmokeScenarioId = SceneWebGPUSmokeId;

/** Preserve the original 7.2 m gate's disturbance-volume bar for the same fixed slab in a wider tank. */
export function minimumOceanFarHalfDisturbanceCells(width_m: number): number {
  if (!(width_m > 0) || !Number.isFinite(width_m)) throw new RangeError("Ocean width must be positive and finite");
  return 0.5 * 7.2 / width_m;
}

export interface SmokeScenario {
  id: SmokeScenarioId;
  description: string;
  scene: SceneDescription;
  /** Number of initial, matched steps used for CPU/GPU differential output. */
  oracleSteps: number;
  /** Default GPU observation duration; FLUID_TARGET_S can override it. */
  target_s: number;
  /** UI-authored solver profile, when the scenario is a validation preset.
   * Dawn consumes this same object so the browser and native smoke cannot
   * silently drift through duplicated command-line overrides. */
  methodProfile?: MethodProfile;
  /** Complete scene-owned contract consumed by the agnostic runner. */
  suite: SceneWebGPUSmokeSuite<SmokeScenarioId>;
  lane: SceneWebGPUSmokeLane;
}

export function isSmokeScenarioId(value: string): value is SmokeScenarioId {
  return isSceneWebGPUSmokeId(value);
}

export function createSmokeScenario(id: SmokeScenarioId, laneId?: string): SmokeScenario {
  const suite = getSceneWebGPUSmokeSuite(id);
  const lane = getSceneWebGPUSmokeLane(id, laneId);
  const authoredScene = suite.createScene();
  const soleMethod = lane.methods.length === 1 ? lane.methods[0] : undefined;
  return {
    id,
    description: lane.description || suite.description,
    scene: authoredScene,
    oracleSteps: lane.oracle.matchedSteps,
    target_s: lane.stop.simulatedTime_s,
    methodProfile: soleMethod ? {
      methodId: soleMethod.id,
      quality: soleMethod.quality,
      overrides: { ...soleMethod.overrides },
    } : undefined,
    suite,
    lane,
  };
}

export interface ScalarFieldSummary {
  minimum: number;
  maximum: number;
  cellSum: number;
  wetCells: number;
  mixedCells: number;
  excessCells: number;
  meanColumnAmount: number;
  columnAmountStdDev: number;
  componentCount: number;
  largestComponent: number;
  interfaceFaceCount: number;
  enclosedAirComponentCount: number;
  enclosedAirCells: number;
  centroidCells: { x: number; y: number; z: number } | null;
}

export function summarizeScalarField(field: ArrayLike<number>, nx: number, ny: number, nz: number): ScalarFieldSummary {
  if (field.length !== nx * ny * nz) throw new Error(`Field length ${field.length} does not match ${nx}x${ny}x${nz}`);
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  let minimum = Infinity, maximum = -Infinity, cellSum = 0, wetCells = 0, mixedCells = 0, excessCells = 0;
  let weightedX = 0, weightedY = 0, weightedZ = 0;
  const columnAmounts = new Float64Array(nx * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const value = field[index(x, y, z)];
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); cellSum += value;
    weightedX += value * (x + 0.5); weightedY += value * (y + 0.5); weightedZ += value * (z + 0.5);
    if (value >= 0.5) wetCells += 1;
    if (value > 0.001 && value < 0.999) mixedCells += 1;
    if (value > 1.001) excessCells += 1;
    columnAmounts[x + nx * z] += value;
  }
  const meanColumnAmount = columnAmounts.reduce((sum, value) => sum + value, 0) / columnAmounts.length;
  const columnAmountStdDev = Math.sqrt(columnAmounts.reduce((sum, value) => sum + (value - meanColumnAmount) ** 2, 0) / columnAmounts.length);
  const visited = new Uint8Array(field.length), stack = new Int32Array(field.length);
  let interfaceFaceCount = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const wet = field[index(x, y, z)] >= 0.5;
    if (x + 1 < nx && (field[index(x + 1, y, z)] >= 0.5) !== wet) interfaceFaceCount += 1;
    if (y + 1 < ny && (field[index(x, y + 1, z)] >= 0.5) !== wet) interfaceFaceCount += 1;
    if (z + 1 < nz && (field[index(x, y, z + 1)] >= 0.5) !== wet) interfaceFaceCount += 1;
  }
  let componentCount = 0, largestComponent = 0;
  for (let start = 0; start < field.length; start += 1) {
    if (visited[start] || field[start] < 0.5) continue;
    componentCount += 1; let top = 0, size = 0; stack[top++] = start; visited[start] = 1;
    while (top > 0) {
      const current = stack[--top]; size += 1;
      const x = current % nx, yz = Math.floor(current / nx), y = yz % ny, z = Math.floor(yz / ny);
      const neighbors = [[x - 1, y, z], [x + 1, y, z], [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]];
      for (const [xx, yy, zz] of neighbors) {
        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
        const next = index(xx, yy, zz);
        if (!visited[next] && field[next] >= 0.5) { visited[next] = 1; stack[top++] = next; }
      }
    }
    largestComponent = Math.max(largestComponent, size);
  }
  // Flood exterior air from the domain boundary. Any remaining dry cells are
  // enclosed phi cavities: they add rendered zero-crossings without reducing
  // the connectivity of the surrounding liquid, which makes them invisible
  // to the ordinary dominant-liquid-component metric.
  const exteriorAir = new Uint8Array(field.length);
  let top = 0;
  const seedExterior = (x: number, y: number, z: number) => {
    const cell = index(x, y, z);
    if (!exteriorAir[cell] && field[cell] < 0.5) { exteriorAir[cell] = 1; stack[top++] = cell; }
  };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) { seedExterior(0, y, z); seedExterior(nx - 1, y, z); }
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) { seedExterior(x, 0, z); seedExterior(x, ny - 1, z); }
  for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) { seedExterior(x, y, 0); seedExterior(x, y, nz - 1); }
  while (top > 0) {
    const current = stack[--top], x = current % nx, yz = Math.floor(current / nx), y = yz % ny, z = Math.floor(yz / ny);
    for (const [xx, yy, zz] of [[x - 1, y, z], [x + 1, y, z], [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]]) {
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
      const next = index(xx, yy, zz);
      if (!exteriorAir[next] && field[next] < 0.5) { exteriorAir[next] = 1; stack[top++] = next; }
    }
  }
  let enclosedAirComponentCount = 0, enclosedAirCells = 0;
  for (let start = 0; start < field.length; start += 1) {
    if (exteriorAir[start] || field[start] >= 0.5) continue;
    enclosedAirComponentCount += 1; top = 0; stack[top++] = start; exteriorAir[start] = 1;
    while (top > 0) {
      const current = stack[--top]; enclosedAirCells += 1;
      const x = current % nx, yz = Math.floor(current / nx), y = yz % ny, z = Math.floor(yz / ny);
      for (const [xx, yy, zz] of [[x - 1, y, z], [x + 1, y, z], [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]]) {
        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
        const next = index(xx, yy, zz);
        if (!exteriorAir[next] && field[next] < 0.5) { exteriorAir[next] = 1; stack[top++] = next; }
      }
    }
  }
  return {
    minimum, maximum, cellSum, wetCells, mixedCells, excessCells, meanColumnAmount, columnAmountStdDev,
    componentCount, largestComponent, interfaceFaceCount, enclosedAirComponentCount, enclosedAirCells,
    centroidCells: cellSum > 0 ? { x: weightedX / cellSum, y: weightedY / cellSum, z: weightedZ / cellSum } : null
  };
}

export interface ScalarFieldDifference {
  meanAbsoluteError: number;
  rootMeanSquareError: number;
  volumeRelativeDifference: number;
  wetIntersectionOverUnion: number;
  centroidDistanceCells: number | null;
}

export interface LocalScalarDifference {
  sampleCount: number;
  meanAbsoluteError: number;
  rootMeanSquareError: number;
  maximumAbsoluteError: number;
  maximumLocation: { x: number; y: number; z: number } | null;
}

export interface SingleTallCellDifference {
  probeColumn: LocalScalarDifference;
  neighborColumns: LocalScalarDifference;
  farField: LocalScalarDifference;
}

export interface TallCellActivitySummary {
  totalColumns: number;
  tallColumns: number;
  ordinaryColumns: number;
  tallFraction: number;
  minimumTallHeight: number;
  maximumTallHeight: number;
  meanTallHeight: number;
  maximumAdjacentDelta?: number;
  maximumPermittedHeight: number;
  canRemeshToTall: boolean;
  classification: "none" | "mixed" | "all";
}

/** A base of zero is the ordinary-grid limit; bases of two or more are tall. */
export function summarizeTallCellActivity(
  columnBases: ArrayLike<number>,
  fineNy: number,
  regularLayers: number,
  nx?: number,
  nz?: number
): TallCellActivitySummary {
  let tallColumns = 0, tallHeightSum = 0, minimumTallHeight = Infinity, maximumTallHeight = 0, maximumAdjacentDelta = 0;
  for (let index = 0; index < columnBases.length; index += 1) {
    const height = Math.round(columnBases[index]);
    if (height < 2) continue;
    tallColumns += 1; tallHeightSum += height;
    minimumTallHeight = Math.min(minimumTallHeight, height); maximumTallHeight = Math.max(maximumTallHeight, height);
  }
  if (nx && nz && nx * nz === columnBases.length) for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height = Math.round(columnBases[x + nx * z]);
    if (x + 1 < nx) maximumAdjacentDelta = Math.max(maximumAdjacentDelta, Math.abs(height - Math.round(columnBases[x + 1 + nx * z])));
    if (z + 1 < nz) maximumAdjacentDelta = Math.max(maximumAdjacentDelta, Math.abs(height - Math.round(columnBases[x + nx * (z + 1)])));
  }
  const totalColumns = columnBases.length, maximumPermittedHeight = Math.max(0, fineNy - regularLayers);
  return {
    totalColumns, tallColumns, ordinaryColumns: totalColumns - tallColumns,
    tallFraction: totalColumns > 0 ? tallColumns / totalColumns : 0,
    minimumTallHeight: tallColumns > 0 ? minimumTallHeight : 0,
    maximumTallHeight, meanTallHeight: tallColumns > 0 ? tallHeightSum / tallColumns : 0,
    ...(nx && nz ? { maximumAdjacentDelta } : {}),
    maximumPermittedHeight, canRemeshToTall: maximumPermittedHeight >= 2,
    classification: tallColumns === 0 ? "none" : tallColumns === totalColumns ? "all" : "mixed"
  };
}

export function compareScalarFields(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  nx: number,
  ny: number,
  nz: number
): ScalarFieldDifference {
  if (left.length !== right.length || left.length !== nx * ny * nz) throw new Error("Scalar fields must share the requested dimensions");
  let absolute = 0, squared = 0, leftSum = 0, rightSum = 0, intersection = 0, union = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index], b = right[index], delta = a - b;
    absolute += Math.abs(delta); squared += delta * delta; leftSum += a; rightSum += b;
    const aWet = a >= 0.5, bWet = b >= 0.5;
    if (aWet && bWet) intersection += 1;
    if (aWet || bWet) union += 1;
  }
  const leftCenter = summarizeScalarField(left, nx, ny, nz).centroidCells;
  const rightCenter = summarizeScalarField(right, nx, ny, nz).centroidCells;
  return {
    meanAbsoluteError: absolute / left.length,
    rootMeanSquareError: Math.sqrt(squared / left.length),
    volumeRelativeDifference: Math.abs(leftSum - rightSum) / Math.max(1, Math.abs(rightSum)),
    wetIntersectionOverUnion: union > 0 ? intersection / union : 1,
    centroidDistanceCells: leftCenter && rightCenter
      ? Math.hypot(leftCenter.x - rightCenter.x, leftCenter.y - rightCenter.y, leftCenter.z - rightCenter.z)
      : null
  };
}

/** Split a differential field into the isolated tall column, its four
 * face-neighbor columns, and everything else. This makes it obvious whether
 * a defect begins in the paper's local tall stencil or arrives from a global
 * stage such as extrapolation/remeshing. */
export function compareSingleTallCellNeighborhood(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  nx: number,
  ny: number,
  nz: number,
  probeX: number,
  probeZ: number
): SingleTallCellDifference {
  if (left.length !== right.length || left.length !== nx * ny * nz) throw new Error("Scalar fields must share the requested dimensions");
  type Accumulator = { count: number; absolute: number; squared: number; maximum: number; location: LocalScalarDifference["maximumLocation"] };
  const bins: Record<keyof SingleTallCellDifference, Accumulator> = {
    probeColumn: { count: 0, absolute: 0, squared: 0, maximum: 0, location: null },
    neighborColumns: { count: 0, absolute: 0, squared: 0, maximum: 0, location: null },
    farField: { count: 0, absolute: 0, squared: 0, maximum: 0, location: null }
  };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const distance = Math.abs(x - probeX) + Math.abs(z - probeZ);
    const bin = bins[distance === 0 ? "probeColumn" : distance === 1 ? "neighborColumns" : "farField"];
    const index = x + nx * (y + ny * z), delta = Math.abs(left[index] - right[index]);
    bin.count += 1; bin.absolute += delta; bin.squared += delta * delta;
    if (delta > bin.maximum) { bin.maximum = delta; bin.location = { x, y, z }; }
  }
  const finish = (bin: Accumulator): LocalScalarDifference => ({
    sampleCount: bin.count,
    meanAbsoluteError: bin.absolute / Math.max(1, bin.count),
    rootMeanSquareError: Math.sqrt(bin.squared / Math.max(1, bin.count)),
    maximumAbsoluteError: bin.maximum,
    maximumLocation: bin.location
  });
  return { probeColumn: finish(bins.probeColumn), neighborColumns: finish(bins.neighborColumns), farField: finish(bins.farField) };
}
