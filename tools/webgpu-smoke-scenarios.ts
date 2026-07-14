import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { createPaperScenario } from "../lib/paper-scenarios";

export const smokeScenarioIds = [
  "settled-tank",
  "dam-break-boxes",
  "hose-tank",
  "sphere-jet",
  "deep-water"
] as const;

export type SmokeScenarioId = typeof smokeScenarioIds[number];

export interface SmokeScenario {
  id: SmokeScenarioId;
  description: string;
  scene: SceneDescription;
  /** Number of initial, matched steps used for CPU/GPU differential output. */
  oracleSteps: number;
  /** Default GPU observation duration; FLUID_TARGET_S can override it. */
  target_s: number;
}

export function isSmokeScenarioId(value: string): value is SmokeScenarioId {
  return smokeScenarioIds.includes(value as SmokeScenarioId);
}

export function createSmokeScenario(id: SmokeScenarioId): SmokeScenario {
  if (id === "hose-tank" || id === "dam-break-boxes" || id === "sphere-jet") {
    const scene = createPaperScenario(id);
    return {
      id,
      description: id === "hose-tank"
        ? "ramped boundary inflow and a shallow receiving pool"
        : id === "dam-break-boxes"
          ? "three-dimensional dam break with immersed boxes"
          : "directed inlet jet past a fixed immersed sphere",
      scene,
      oracleSteps: 2,
      target_s: Math.max(scene.numerics.maxDt_s * 8, 0.05)
    };
  }

  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [];
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  if (id === "settled-tank") {
    scene.sceneId = "smoke-settled-tank";
    scene.fluid.initialCondition = "tank-fill";
    scene.container.fillFraction = 0.7;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
    return { id, description: "hydrostatic preservation in a closed, level pool", scene, oracleSteps: 2, target_s: 0.1 };
  }

  scene.sceneId = "smoke-deep-water";
  scene.container.height_m = 20;
  scene.container.fillFraction = 0.8;
  scene.fluid.initialCondition = "tank-fill";
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return { id, description: "extreme vertical aspect ratio and tall-cell compression", scene, oracleSteps: 1, target_s: 0.1 };
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
  return {
    minimum, maximum, cellSum, wetCells, mixedCells, excessCells, meanColumnAmount, columnAmountStdDev,
    componentCount, largestComponent,
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

export interface TallCellActivitySummary {
  totalColumns: number;
  tallColumns: number;
  ordinaryColumns: number;
  tallFraction: number;
  minimumTallHeight: number;
  maximumTallHeight: number;
  meanTallHeight: number;
  maximumPermittedHeight: number;
  canRemeshToTall: boolean;
  classification: "none" | "mixed" | "all";
}

/** A base of zero is the ordinary-grid limit; bases of two or more are tall. */
export function summarizeTallCellActivity(
  columnBases: ArrayLike<number>,
  fineNy: number,
  regularLayers: number
): TallCellActivitySummary {
  let tallColumns = 0, tallHeightSum = 0, minimumTallHeight = Infinity, maximumTallHeight = 0;
  for (let index = 0; index < columnBases.length; index += 1) {
    const height = Math.round(columnBases[index]);
    if (height < 2) continue;
    tallColumns += 1; tallHeightSum += height;
    minimumTallHeight = Math.min(minimumTallHeight, height); maximumTallHeight = Math.max(maximumTallHeight, height);
  }
  const totalColumns = columnBases.length, maximumPermittedHeight = Math.max(0, fineNy - regularLayers);
  return {
    totalColumns, tallColumns, ordinaryColumns: totalColumns - tallColumns,
    tallFraction: totalColumns > 0 ? tallColumns / totalColumns : 0,
    minimumTallHeight: tallColumns > 0 ? minimumTallHeight : 0,
    maximumTallHeight, meanTallHeight: tallColumns > 0 ? tallHeightSum / tallColumns : 0,
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
