import { damBreakFractions } from "./initial-fluid";
import type { SceneDescription, Vec3 } from "./model";
import type { GPUQuality, TallCellLayout, TallCellSettings } from "./tall-cell-grid";

export interface AdaptiveOpticalLayerSettings {
  alpha: number;
  minimumDilationCells: number;
  maximumDilationCells: number;
  airborneOffsetCells: number;
  airborneDilationCells: number;
  smoothingRadius: number;
  smoothingIterations: number;
  logicalRegularLayers: number;
}

export interface TallColumnFit {
  bottom: Vec3;
  top: Vec3;
  reconstructed: Vec3[];
  error: number;
}

export interface AdaptiveLayerSummary {
  minimumBase: number;
  maximumBase: number;
  meanBase: number;
  activePressureSamples: number;
  tallColumnCount: number;
  opticalCellCount: number;
  smoothingAddedCells: number;
}

export interface AdaptiveOpticalLayerField {
  nx: number;
  ny: number;
  nz: number;
  gridWidth: number;
  volume: ArrayLike<number>;
  velocity: ArrayLike<Vec3>;
}

export interface AdaptiveOpticalLayerConstruction {
  errors: Float32Array;
  dilations: Uint32Array;
  groundSurfaces: Int32Array;
  rawBases: Float32Array;
  bases: Float32Array;
}

const qualityColumns: Record<GPUQuality, number> = {
  balanced: 2_500,
  high: 7_000,
  ultra: 12_500
};

const clamp = (value: number, lower: number, upper: number) => Math.max(lower, Math.min(upper, value));
const index2 = (x: number, z: number, nx: number) => x + nx * z;

export function adaptiveOpticalLayerSettings(fineNy: number): AdaptiveOpticalLayerSettings {
  const minimumDilationCells = Math.max(4, Math.ceil(fineNy / 64));
  const maximumDilationCells = Math.max(minimumDilationCells, Math.ceil(fineNy / 8));
  const airborneOffsetCells = Math.max(4, Math.ceil(fineNy / 32));
  const airborneDilationCells = Math.max(1, Math.ceil(fineNy / 16));
  return {
    alpha: 0.5,
    minimumDilationCells,
    maximumDilationCells,
    airborneOffsetCells,
    airborneDilationCells,
    smoothingRadius: 4,
    smoothingIterations: 5,
    // One surface needs room for the maximum dilation on both sides. The
    // physical texture remains full-height so disconnected components and
    // vertical interfaces are still representable.
    logicalRegularLayers: Math.min(fineNy, Math.max(8, 2 * maximumDilationCells + 1))
  };
}

export function fitTallColumnVelocity(samples: readonly Vec3[]): TallColumnFit {
  if (samples.length === 0) {
    const zero = { x: 0, y: 0, z: 0 };
    return { bottom: zero, top: zero, reconstructed: [], error: 0 };
  }
  const n = samples.length;
  let sumT = 0, sumTT = 0, sumX = 0, sumTX = 0, sumY = 0, sumZ = 0, sumTZ = 0;
  for (let j = 0; j < n; j += 1) {
    const t = n === 1 ? 0 : j / (n - 1);
    const value = samples[j];
    sumT += t; sumTT += t * t;
    sumX += value.x; sumTX += t * value.x;
    sumY += value.y;
    sumZ += value.z; sumTZ += t * value.z;
  }
  const denominator = n * sumTT - sumT * sumT;
  const slopeX = denominator > 1e-12 ? (n * sumTX - sumT * sumX) / denominator : 0;
  const slopeZ = denominator > 1e-12 ? (n * sumTZ - sumT * sumZ) / denominator : 0;
  const interceptX = (sumX - slopeX * sumT) / n;
  const interceptZ = (sumZ - slopeZ * sumT) / n;
  const averageY = sumY / n;
  const reconstructed = samples.map((_, j) => {
    const t = n === 1 ? 0 : j / (n - 1);
    return { x: interceptX + slopeX * t, y: averageY, z: interceptZ + slopeZ * t };
  });
  let error = 0;
  for (let j = 0; j < n; j += 1) {
    error += Math.abs(samples[j].x - reconstructed[j].x)
      + Math.abs(samples[j].y - reconstructed[j].y)
      + Math.abs(samples[j].z - reconstructed[j].z);
  }
  return {
    bottom: reconstructed[0],
    top: reconstructed[reconstructed.length - 1],
    reconstructed,
    error
  };
}

export function errorToDilationCells(error: number, gridWidth: number, settings: AdaptiveOpticalLayerSettings): number {
  if (!Number.isFinite(error) || error < 0) throw new Error("Adaptive optical-layer error must be finite and non-negative");
  if (!(gridWidth > 0)) throw new Error("Adaptive optical-layer grid width must be positive");
  const continuous = clamp(settings.alpha * error * gridWidth, settings.minimumDilationCells, settings.maximumDilationCells);
  // The article leaves the integer conversion implicit. Round-to-nearest is
  // deterministic and preserves both specified clamps exactly.
  return clamp(Math.round(continuous), settings.minimumDilationCells, settings.maximumDilationCells);
}

export function manhattanDilateSeedBases(
  seedBases: ArrayLike<number>,
  seedRadii: ArrayLike<number>,
  nx: number,
  nz: number,
  maximumBase: number
): Float32Array {
  if (seedBases.length !== nx * nz || seedRadii.length !== nx * nz) throw new Error("Adaptive seed fields have the wrong dimensions");
  const result = new Float32Array(nx * nz);
  result.fill(maximumBase);
  for (let destinationZ = 0; destinationZ < nz; destinationZ += 1) for (let destinationX = 0; destinationX < nx; destinationX += 1) {
    let base = maximumBase;
    for (let sourceZ = 0; sourceZ < nz; sourceZ += 1) for (let sourceX = 0; sourceX < nx; sourceX += 1) {
      const source = index2(sourceX, sourceZ, nx);
      const radius = Math.max(0, Math.round(seedRadii[source]));
      const distance = Math.abs(destinationX - sourceX) + Math.abs(destinationZ - sourceZ);
      if (distance > radius) continue;
      base = Math.min(base, seedBases[source] + distance);
    }
    result[index2(destinationX, destinationZ, nx)] = clamp(base, 0, maximumBase);
  }
  return result;
}

export function smoothAdaptiveOpticalBases(
  rawBases: ArrayLike<number>,
  nx: number,
  nz: number,
  radius = 4,
  iterations = 5
): Float32Array {
  if (rawBases.length !== nx * nz) throw new Error("Adaptive base field has the wrong dimensions");
  const raw = Float32Array.from(rawBases);
  let current = raw.slice();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Float32Array(current.length);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      let sum = 0, count = 0;
      for (let dz = -radius; dz <= radius; dz += 1) for (let dx = -radius; dx <= radius; dx += 1) {
        const qx = x + dx, qz = z + dz;
        if (qx < 0 || qx >= nx || qz < 0 || qz >= nz) continue;
        sum += current[index2(qx, qz, nx)];
        count += 1;
      }
      next[index2(x, z, nx)] = Math.min(raw[index2(x, z, nx)], sum / Math.max(1, count));
    }
    current = next;
  }
  for (let index = 0; index < current.length; index += 1) {
    const rounded = Math.max(0, Math.round(current[index]));
    current[index] = rounded === 1 ? 0 : rounded;
  }
  return current;
}

/**
 * Deterministic cubic-grid oracle for the paper's complete optical-layer
 * construction. This deliberately favors transparency over speed: it is used
 * for tests and CPU/GPU comparisons, while the WebGPU path uses separable
 * radius-budget envelopes for the same Manhattan result.
 */
export function constructAdaptiveOpticalLayer(
  field: AdaptiveOpticalLayerField,
  settings = adaptiveOpticalLayerSettings(field.ny)
): AdaptiveOpticalLayerConstruction {
  const { nx, ny, nz, gridWidth, volume, velocity } = field;
  if (nx <= 0 || ny <= 0 || nz <= 0 || !(gridWidth > 0)) throw new Error("Adaptive optical-layer field dimensions must be positive");
  const cellCount = nx * ny * nz;
  if (volume.length !== cellCount || velocity.length !== cellCount) throw new Error("Adaptive optical-layer field arrays have the wrong dimensions");
  const index3 = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const wet = (x: number, y: number, z: number) => x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz
    && volume[index3(x, y, z)] >= 0.5;
  const surface = (x: number, y: number, z: number) => wet(x, y, z)
    && (!wet(x, y + 1, z) || !wet(x - 1, y, z) || !wet(x + 1, y, z) || !wet(x, y, z - 1) || !wet(x, y, z + 1));
  const columns = nx * nz;
  const errors = new Float32Array(columns);
  const dilations = new Uint32Array(columns);
  const groundSurfaces = new Int32Array(columns); groundSurfaces.fill(-1);
  const maximumBase = Math.max(0, ny - settings.logicalRegularLayers);
  const rawBases = new Float32Array(columns); rawBases.fill(maximumBase);

  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const column = index2(x, z, nx);
    let groundSurface = -1;
    if (wet(x, 0, z)) for (let y = 0; y < ny && wet(x, y, z); y += 1) groundSurface = y;
    groundSurfaces[column] = groundSurface;
    const samples: Vec3[] = [];
    for (let y = 0; y <= groundSurface; y += 1) samples.push(velocity[index3(x, y, z)]);
    const error = fitTallColumnVelocity(samples).error;
    errors[column] = error;
    dilations[column] = errorToDilationCells(error, gridWidth, settings);
  }

  for (let sourceZ = 0; sourceZ < nz; sourceZ += 1) for (let sourceX = 0; sourceX < nx; sourceX += 1) {
    const sourceColumn = index2(sourceX, sourceZ, nx);
    const groundSurface = groundSurfaces[sourceColumn];
    for (let y = 0; y < ny; y += 1) {
      if (!surface(sourceX, y, sourceZ)) continue;
      const radius = groundSurface >= 0 && Math.abs(y - groundSurface) <= settings.airborneOffsetCells
        ? dilations[sourceColumn]
        : settings.airborneDilationCells;
      for (let dz = -radius; dz <= radius; dz += 1) for (let dx = -(radius - Math.abs(dz)); dx <= radius - Math.abs(dz); dx += 1) {
        const x = sourceX + dx, z = sourceZ + dz;
        if (x < 0 || x >= nx || z < 0 || z >= nz) continue;
        const distance = Math.abs(dx) + Math.abs(dz);
        const destination = index2(x, z, nx);
        rawBases[destination] = Math.min(rawBases[destination], Math.max(0, y - radius + distance));
      }
    }
  }
  return {
    errors,
    dilations,
    groundSurfaces,
    rawBases,
    bases: smoothAdaptiveOpticalBases(rawBases, nx, nz, settings.smoothingRadius, settings.smoothingIterations)
  };
}

export function splitTallCellRigidCoupling(contribution: ArrayLike<number>, s: number) {
  if (!Number.isFinite(s) || s < 0 || s > 1) throw new Error("Tall-cell rigid interpolation coordinate must be in [0, 1]");
  return {
    top: Float64Array.from(contribution, (value) => (1 - s) * value),
    bottom: Float64Array.from(contribution, (value) => s * value)
  };
}

export function summarizeAdaptiveLayer(rawBases: ArrayLike<number>, smoothedBases: ArrayLike<number>, fineNy: number): AdaptiveLayerSummary {
  if (rawBases.length !== smoothedBases.length || rawBases.length === 0) throw new Error("Adaptive layer summaries require matching non-empty fields");
  let minimumBase = fineNy, maximumBase = 0, baseSum = 0, activePressureSamples = 0, tallColumnCount = 0, opticalCellCount = 0, smoothingAddedCells = 0;
  for (let index = 0; index < smoothedBases.length; index += 1) {
    const base = clamp(Math.round(smoothedBases[index]), 0, fineNy);
    minimumBase = Math.min(minimumBase, base);
    maximumBase = Math.max(maximumBase, base);
    baseSum += base;
    smoothingAddedCells += Math.max(0, Math.round(rawBases[index]) - base);
    if (base > 0) {
      tallColumnCount += 1;
      activePressureSamples += fineNy - base + 2;
      opticalCellCount += fineNy - base;
    } else {
      activePressureSamples += fineNy;
      opticalCellCount += fineNy;
    }
  }
  return { minimumBase, maximumBase, meanBase: baseSum / smoothedBases.length, activePressureSamples, tallColumnCount, opticalCellCount, smoothingAddedCells };
}

function initialWet(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  if (scene.fluid.initialCondition === "tank-fill") return (y + 0.5) / fineNy <= scene.container.fillFraction;
  const dam = damBreakFractions(scene.container.fillFraction);
  return (x + 0.5) / nx <= dam.width && (y + 0.5) / fineNy <= dam.height && (z + 0.5) / nz <= dam.depth;
}

function initialSurface(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  const wet = initialWet(scene, x, y, z, nx, fineNy, nz);
  if (!wet) return false;
  if (y + 1 >= fineNy || !initialWet(scene, x, y + 1, z, nx, fineNy, nz)) return true;
  return (x > 0 && !initialWet(scene, x - 1, y, z, nx, fineNy, nz))
    || (x + 1 < nx && !initialWet(scene, x + 1, y, z, nx, fineNy, nz))
    || (z > 0 && !initialWet(scene, x, y, z - 1, nx, fineNy, nz))
    || (z + 1 < nz && !initialWet(scene, x, y, z + 1, nx, fineNy, nz));
}

export function createAdaptiveOpticalLayerLayout(scene: SceneDescription, quality: GPUQuality, maximumTextureDimension = 2048, columnsOverride?: number): TallCellLayout {
  const container = scene.container;
  const surfaceColumns = columnsOverride ?? qualityColumns[quality];
  const targetH = Math.sqrt(container.width_m * container.depth_m / surfaceColumns);
  const nx = Math.min(maximumTextureDimension, Math.max(8, Math.round(container.width_m / targetH)));
  const nz = Math.min(maximumTextureDimension, Math.max(8, Math.round(container.depth_m / targetH)));
  const horizontalH = Math.sqrt((container.width_m / nx) * (container.depth_m / nz));
  const fineNy = Math.min(maximumTextureDimension - 2, Math.max(8, Math.round(container.height_m / horizontalH)));
  const adaptive = adaptiveOpticalLayerSettings(fineNy);
  const maximumBase = Math.max(0, fineNy - adaptive.logicalRegularLayers);
  const seedBases = new Float32Array(nx * nz); seedBases.fill(maximumBase);
  const seedRadii = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const column = index2(x, z, nx);
    for (let y = 0; y < fineNy; y += 1) {
      if (!initialSurface(scene, x, y, z, nx, fineNy, nz)) continue;
      seedBases[column] = Math.min(seedBases[column], Math.max(0, y - adaptive.minimumDilationCells));
      seedRadii[column] = Math.max(seedRadii[column], adaptive.minimumDilationCells);
    }
  }
  const rawBases = manhattanDilateSeedBases(seedBases, seedRadii, nx, nz, maximumBase);
  const columnBases = smoothAdaptiveOpticalBases(rawBases, nx, nz, adaptive.smoothingRadius, adaptive.smoothingIterations);
  // Full-height packed storage is deliberate. Only samples from each adaptive
  // base upward (plus two tall endpoints) are active, but vertical dam faces
  // and separated airborne components cannot be lost due to a global band cap.
  const packedNy = fineNy + 2;
  const initialVolume = new Float32Array(nx * packedNy * nz);
  let initialVolumeCellSum = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = columnBases[index2(x, z, nx)];
    for (let packedY = 0; packedY < packedNy; packedY += 1) {
      const worldY = packedY === 0 ? 0 : packedY === 1 ? Math.max(0, base - 1) : base + packedY - 2;
      const active = packedY >= 2 ? worldY < fineNy : base > 0;
      const wet = active && initialWet(scene, x, worldY, z, nx, fineNy, nz);
      initialVolume[x + nx * (packedY + packedNy * z)] = wet ? 1 : 0;
      if (wet) initialVolumeCellSum += packedY === 0 ? base : packedY >= 2 ? 1 : 0;
    }
  }
  const settings: TallCellSettings = {
    surfaceColumns,
    regularLayers: adaptive.logicalRegularLayers,
    liquidHalo: adaptive.maximumDilationCells,
    airHalo: adaptive.maximumDilationCells,
    maximumNeighborDelta: adaptive.maximumDilationCells,
    remeshInterval: 1
  };
  const packedSampleCount = nx * packedNy * nz;
  const equivalentUniformCellCount = nx * fineNy * nz;
  return {
    nx, fineNy, nz, packedNy,
    cellSize_m: { x: container.width_m / nx, y: container.height_m / fineNy, z: container.depth_m / nz },
    columnBases, initialVolume, initialVolumeCellSum,
    packedSampleCount, equivalentUniformCellCount,
    compressionRatio: packedSampleCount / equivalentUniformCellCount,
    settings
  };
}
