import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { createPaperScenario } from "../lib/paper-scenarios";
import { applyGardenPool } from "../lib/garden-scene";

export const smokeScenarioIds = [
  "dam-break-ui",
  "settled-tank",
  "dam-break-boxes",
  "hose-tank",
  "sphere-jet",
  "deep-water",
  "garden-pond",
  "garden-dam-break"
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
      // Inflow jets need time to establish before the frozen-scene gate is
      // meaningful: at 0.05 s the stream is still entirely sub-threshold and
      // the gate measured ambient equilibrium noise instead.
      target_s: id === "dam-break-boxes" ? Math.max(scene.numerics.maxDt_s * 8, 0.05) : 0.5
    };
  }

  if (id === "garden-pond" || id === "garden-dam-break") {
    // Terrain heightfield scenes. The CPU reference has no static-solid
    // support, so oracle differentials are informative only; the per-method
    // invariant gates (volume, stability) remain authoritative.
    const scene = applyGardenPool(cloneScene(defaultScene), id === "garden-dam-break" ? { fillFraction: 0.16 } : {});
    scene.sceneId = `smoke-${id}`;
    scene.rigidBodies = [];
    scene.fluid.surfaceTension_N_m = 0;
    delete scene.fluid.inflow;
    scene.fluid.initialCondition = id === "garden-pond" ? "tank-fill" : "dam-break";
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = id === "garden-pond" ? 1 / 120 : 0.004;
    return {
      id,
      description: id === "garden-pond"
        ? "hydrostatic rest in an organic pool carved from a terrain heightfield"
        : "dam break released onto a lawn heightfield draining into the pool",
      scene, oracleSteps: 2, target_s: id === "garden-pond" ? 0.1 : 0.2
    };
  }

  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [];
  if (id === "dam-break-ui") {
    scene.sceneId = "smoke-ui-dam-break";
    scene.fluid.initialCondition = "dam-break";
    delete scene.fluid.inflow;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = process.env.FLUID_MAX_DT ? Number(process.env.FLUID_MAX_DT) : 0.004;
    if (process.env.FLUID_SURFACE_TENSION !== undefined) scene.fluid.surfaceTension_N_m = Number(process.env.FLUID_SURFACE_TENSION);
    return { id, description: "actual UI dam break with the default capillary and wall settings", scene, oracleSteps: 2, target_s: 0.2 };
  }
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
