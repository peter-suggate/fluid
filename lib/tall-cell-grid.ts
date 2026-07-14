import { damBreakFractions } from "./initial-fluid";
import type { SceneDescription } from "./model";

export type GPUQuality = "balanced" | "high" | "ultra";

export interface TallCellSettings {
  surfaceColumns: number;
  regularLayers: number;
  liquidHalo: number;
  airHalo: number;
  maximumNeighborDelta: number;
  remeshInterval: number;
}

export interface TallCellLayout {
  nx: number;
  fineNy: number;
  nz: number;
  packedNy: number;
  cellSize_m: { x: number; y: number; z: number };
  columnBases: Float32Array;
  initialVolume: Float32Array;
  initialVolumeCellSum: number;
  packedSampleCount: number;
  equivalentUniformCellCount: number;
  compressionRatio: number;
  settings: TallCellSettings;
}

// The presets retain approximately the x/z resolution of the former 110k,
// 500k and 1.2m cubic grids for the default scene. Only the moving surface
// band and two samples for each bottom tall cell are stored.
export const tallCellSettings: Record<GPUQuality, TallCellSettings> = {
  balanced: { surfaceColumns: 2_500, regularLayers: 24, liquidHalo: 8, airHalo: 8, maximumNeighborDelta: 4, remeshInterval: 60 },
  high: { surfaceColumns: 7_000, regularLayers: 32, liquidHalo: 16, airHalo: 8, maximumNeighborDelta: 4, remeshInterval: 60 },
  ultra: { surfaceColumns: 12_500, regularLayers: 40, liquidHalo: 24, airHalo: 8, maximumNeighborDelta: 5, remeshInterval: 60 }
};

export function tallCellFluxSampleCount(height: number) {
  const cells = Math.max(0, Math.floor(height));
  return cells <= 48 ? cells : 48;
}

export function chooseTallCellBase(
  lowestSurfaceCell: number,
  highestSurfaceCell: number,
  fineNy: number,
  settings: TallCellSettings
): number {
  const maximumBase = Math.max(0, fineNy - settings.regularLayers);
  const lowerBound = highestSurfaceCell + 1 + settings.airHalo - settings.regularLayers;
  const upperBound = lowestSurfaceCell + 1 - settings.liquidHalo;
  const desired = lowerBound <= upperBound
    ? Math.round((lowerBound + upperBound) / 2)
    : upperBound;
  const clamped = Math.max(0, Math.min(maximumBase, desired));
  // A one-subcell "tall" cell has coincident top and bottom samples.  That
  // creates a duplicate pressure unknown and a zero endpoint distance.  The
  // paper's two-endpoint reconstruction is meaningful only for h >= 2; use
  // the ordinary-cell limit for h < 2.
  return clamped < 2 ? 0 : clamped;
}

export function limitNeighboringTallCellBases(
  source: Float32Array,
  nx: number,
  nz: number,
  maximumDelta: number,
  passes = 2
): Float32Array {
  let current = source.slice();
  for (let pass = 0; pass < passes; pass += 1) {
    const next = current.slice();
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const index = x + nx * z;
      let upper = current[index];
      if (x > 0) upper = Math.min(upper, current[index - 1] + maximumDelta);
      if (x + 1 < nx) upper = Math.min(upper, current[index + 1] + maximumDelta);
      if (z > 0) upper = Math.min(upper, current[index - nx] + maximumDelta);
      if (z + 1 < nz) upper = Math.min(upper, current[index + nx] + maximumDelta);
      next[index] = Math.round(upper);
    }
    current = next;
  }
  return current;
}

function initialWet(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  if (scene.fluid.initialCondition === "tank-fill") return (y + 0.5) / fineNy <= scene.container.fillFraction;
  const dam = damBreakFractions(scene.container.fillFraction);
  return (x + 0.5) / nx <= dam.width && (y + 0.5) / fineNy <= dam.height && (z + 0.5) / nz <= dam.depth;
}

function isInitialSurfaceCell(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  const wet = initialWet(scene, x, y, z, nx, fineNy, nz);
  // The floor is a solid boundary, not a free surface.  All other wet/dry
  // transitions must fit in the regular band: a tall cell cannot represent a
  // vertical free surface through its interior (Chentanez & Mueller, sec. 3).
  const belowWet = y === 0 ? wet : initialWet(scene, x, y - 1, z, nx, fineNy, nz);
  const aboveWet = y + 1 < fineNy && initialWet(scene, x, y + 1, z, nx, fineNy, nz);
  const sideSurface = (x > 0 && wet !== initialWet(scene, x - 1, y, z, nx, fineNy, nz))
    || (x + 1 < nx && wet !== initialWet(scene, x + 1, y, z, nx, fineNy, nz))
    || (z > 0 && wet !== initialWet(scene, x, y, z - 1, nx, fineNy, nz))
    || (z + 1 < nz && wet !== initialWet(scene, x, y, z + 1, nx, fineNy, nz));
  return wet !== belowWet || wet !== aboveWet || sideSurface;
}

function requiredInitialRegularLayers(
  scene: SceneDescription,
  nx: number,
  fineNy: number,
  nz: number,
  settings: TallCellSettings
) {
  let lowest = fineNy;
  let highest = -1;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (let y = 0; y < fineNy; y += 1) {
    if (!isInitialSurfaceCell(scene, x, y, z, nx, fineNy, nz)) continue;
    lowest = Math.min(lowest, y);
    highest = Math.max(highest, y);
  }
  const inflow = scene.fluid.inflow;
  if (inflow) {
    const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
    const directionY = speed > 0 ? inflow.velocity_m_s.y / speed : 0;
    const verticalRadius = Math.abs(directionY) * inflow.length_m / 2 + Math.sqrt(Math.max(0, 1 - directionY * directionY)) * inflow.radius_m;
    const inletLowest = Math.max(0, Math.floor((inflow.center_m.y - verticalRadius) / scene.container.height_m * fineNy));
    const inletHighest = Math.min(fineNy - 1, Math.floor((inflow.center_m.y + verticalRadius) / scene.container.height_m * fineNy));
    lowest = Math.min(lowest, inletLowest);
    highest = Math.max(highest, inletHighest);
  }
  if (highest < 0) return settings.regularLayers;

  // The inequalities in chooseTallCellBase have a solution only when the
  // entire surface range plus its available liquid/air halos fits in By.  At
  // a wall, the halo is clipped because cells outside the domain are solid.
  const liquidHalo = Math.min(settings.liquidHalo, lowest + 1);
  const airHalo = Math.min(settings.airHalo, fineNy - highest - 1);
  return highest - lowest + liquidHalo + airHalo;
}

export function createTallCellLayout(scene: SceneDescription, quality: GPUQuality, maximumTextureDimension = 2048, overrides?: Partial<TallCellSettings>): TallCellLayout {
  const c = scene.container, settings = { ...tallCellSettings[quality], ...overrides };
  const targetH = Math.sqrt(c.width_m * c.depth_m / settings.surfaceColumns);
  const nx = Math.min(maximumTextureDimension, Math.max(8, Math.round(c.width_m / targetH)));
  const nz = Math.min(maximumTextureDimension, Math.max(8, Math.round(c.depth_m / targetH)));
  const horizontalH = Math.sqrt((c.width_m / nx) * (c.depth_m / nz));
  const fineNy = Math.min(maximumTextureDimension, Math.max(8, Math.round(c.height_m / horizontalH)));
  // By is a restriction on representable geometry, not merely a quality
  // knob.  If the free surface spans more vertical cells than the requested
  // band, increase By up to the uniform-grid limit instead of silently
  // dropping liquid during remeshing.  Deep, horizontally stratified scenes
  // keep the requested compact band and therefore retain the paper's benefit.
  const requiredLayers = requiredInitialRegularLayers(scene, nx, fineNy, nz, settings);
  let regularLayers = Math.min(fineNy, Math.max(settings.regularLayers, requiredLayers));
  let effectiveSettings = { ...settings, regularLayers, liquidHalo: Math.min(settings.liquidHalo, regularLayers), airHalo: Math.min(settings.airHalo, regularLayers) };
  let packedNy = regularLayers + 2;
  const maximumBase = Math.max(0, fineNy - regularLayers);
  const rawBases = new Float32Array(nx * nz);

  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    let lowestSurface = fineNy, highestSurface = -1;
    for (let y = 0; y < fineNy; y += 1) {
      if (isInitialSurfaceCell(scene, x, y, z, nx, fineNy, nz)) {
        lowestSurface = Math.min(lowestSurface, y);
        highestSurface = Math.max(highestSurface, y);
      }
    }
    rawBases[x + nx * z] = highestSurface >= 0
      ? chooseTallCellBase(lowestSurface, highestSurface, fineNy, effectiveSettings)
      // An empty column still needs its regular band at the same elevation as
      // neighbouring water.  Setting it to zero discards the upper domain and
      // makes the D limiter pull every wet tall cell down a few cells per
      // frame.  The paper has one tall cell in every column, including air.
      : maximumBase;
  }

  let columnBases = limitNeighboringTallCellBases(rawBases, nx, nz, effectiveSettings.maximumNeighborDelta, nx + nz);
  // A zero-height tall cell is the paper's ordinary-grid limit. If every
  // column chooses that limit, the packed grid must contain every cubic row;
  // keeping only the surface-band rows would silently turn the upper domain
  // into an air boundary while still advertising a Tall solve.
  if (regularLayers < fineNy && columnBases.every((base) => base === 0)) {
    regularLayers = fineNy;
    effectiveSettings = { ...effectiveSettings, regularLayers, liquidHalo: Math.min(settings.liquidHalo, regularLayers), airHalo: Math.min(settings.airHalo, regularLayers) };
    packedNy = regularLayers + 2;
    columnBases = new Float32Array(nx * nz);
  }
  const initialVolume = new Float32Array(nx * packedNy * nz);
  let initialVolumeCellSum = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = columnBases[x + nx * z];
    for (let packedY = 0; packedY < packedNy; packedY += 1) {
      let worldY: number;
      if (packedY === 0) worldY = 0;
      else if (packedY === 1) worldY = Math.max(0, base - 1);
      else worldY = base + packedY - 2;
      const active = packedY >= 2 ? worldY < fineNy : base > 0;
      const wet = active && initialWet(scene, x, worldY, z, nx, fineNy, nz);
      initialVolume[x + nx * (packedY + packedNy * z)] = wet ? 1 : 0;
      if (wet) initialVolumeCellSum += packedY === 0 ? base : packedY >= 2 ? 1 : 0;
    }
  }

  const packedSampleCount = nx * packedNy * nz;
  const equivalentUniformCellCount = nx * fineNy * nz;
  return {
    nx, fineNy, nz, packedNy,
    cellSize_m: { x: c.width_m / nx, y: c.height_m / fineNy, z: c.depth_m / nz },
    columnBases, initialVolume, initialVolumeCellSum,
    packedSampleCount, equivalentUniformCellCount,
    compressionRatio: packedSampleCount / equivalentUniformCellCount,
    settings: effectiveSettings
  };
}
