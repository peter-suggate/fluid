import type { SceneDescription } from "../core/model";
import type { MethodProfile } from "../core/method-contract";
import {
  getSceneWebGPUSmokeLane,
  getSceneWebGPUSmokeSuite,
  isSceneWebGPUSmokeId,
  sceneWebGPUSmokeIds,
  type SceneWebGPUSmokeId,
} from "./scene-webgpu-smoke-catalog";
import type { SceneWebGPUSmokeLane, SceneWebGPUSmokeSuite } from "./scene-webgpu-smoke";

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
  maximumCell: { x: number; y: number; z: number } | null;
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
  wetBounds: { minimum: { x: number; y: number; z: number };
    maximum: { x: number; y: number; z: number } } | null;
  centroidCells: { x: number; y: number; z: number } | null;
}

export function summarizeScalarField(field: ArrayLike<number>, nx: number, ny: number, nz: number): ScalarFieldSummary {
  if (field.length !== nx * ny * nz) throw new Error(`Field length ${field.length} does not match ${nx}x${ny}x${nz}`);
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  let minimum = Infinity, maximum = -Infinity, maximumCell: ScalarFieldSummary["maximumCell"] = null;
  let cellSum = 0, wetCells = 0, mixedCells = 0, excessCells = 0;
  let wetMinimumX = nx, wetMinimumY = ny, wetMinimumZ = nz;
  let wetMaximumX = -1, wetMaximumY = -1, wetMaximumZ = -1;
  let weightedX = 0, weightedY = 0, weightedZ = 0;
  const columnAmounts = new Float64Array(nx * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const value = field[index(x, y, z)];
    minimum = Math.min(minimum, value);
    if (value > maximum) { maximum = value; maximumCell = { x, y, z }; }
    cellSum += value;
    weightedX += value * (x + 0.5); weightedY += value * (y + 0.5); weightedZ += value * (z + 0.5);
    if (value >= 0.5) {
      wetCells += 1;
      wetMinimumX = Math.min(wetMinimumX, x); wetMaximumX = Math.max(wetMaximumX, x);
      wetMinimumY = Math.min(wetMinimumY, y); wetMaximumY = Math.max(wetMaximumY, y);
      wetMinimumZ = Math.min(wetMinimumZ, z); wetMaximumZ = Math.max(wetMaximumZ, z);
    }
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
    minimum, maximum, maximumCell, cellSum, wetCells, mixedCells, excessCells, meanColumnAmount, columnAmountStdDev,
    componentCount, largestComponent, interfaceFaceCount, enclosedAirComponentCount, enclosedAirCells,
    wetBounds: wetCells > 0 ? {
      minimum: { x: wetMinimumX, y: wetMinimumY, z: wetMinimumZ },
      maximum: { x: wetMaximumX, y: wetMaximumY, z: wetMaximumZ },
    } : null,
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
