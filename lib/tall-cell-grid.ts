import { damBreakFractions, initialFluidBrickContainsCell, initialFluidBrickSignedDistance } from "./initial-fluid";
import type { SceneDescription } from "./model";
import { sceneHasTerrain, terrainHeightAt } from "./terrain";

export type GPUQuality = "balanced" | "high" | "ultra";

export interface TallCellSettings {
  surfaceColumns: number;
  regularLayers: number;
  liquidHalo: number;
  airHalo: number;
  maximumNeighborDelta: number;
  /** Temporary parity boundary: Section 5's unmeasured interior lateral
   * faces grow with tall-cell depth. Three leaves only one such layer. */
  maximumTallHeight: number;
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
  /** Signed-distance point samples at the paper's Eq. 4 locations. Negative
   * values are liquid and the narrow band is clamped to five fine cells. */
  initialPhi: Float32Array;
  initialVolumeCellSum: number;
  /** Reference liquid volume in fine-cell units. Unlike initialVolume, this
   * name survives the tall solver's level-set cutover. */
  referenceLiquidVolume_cells: number;
  packedSampleCount: number;
  activeSampleCount: number;
  equivalentUniformCellCount: number;
  compressionRatio: number;
  activeCompressionRatio: number;
  settings: TallCellSettings;
  planning: {
    requestedRegularLayers: number;
    requiredInitialRegularLayers: number;
    storedRegularLayers: number;
    regularLayersBeforeOrdinaryFallback: number;
    maximumBaseBeforeOrdinaryFallback: number;
    ordinaryGridFallback: boolean;
  };
  /** Diagnostic layout used by the cubic-vs-one-tall-cell differential
   * probe. It is never selected by the interactive method. */
  singleTallCellProbe?: {
    x: number;
    z: number;
    height: number;
    initialState: "liquid" | "air";
    mutedHeight: 2;
    supportRadius: number;
    affectedColumns: number;
    topologyFrozen: true;
  };
}

export interface SingleTallCellProbeOptions {
  /** Tall-cell height in cubic subcells. The paper uses 3 <= D <= 6; keeping
   * this at or below D makes the isolated column satisfy its Eq. 10 bound. */
  height?: number;
  x?: number;
  z?: number;
  /** Optional Manhattan-radius support ring. Zero retains the single-cell
   * differential; positive values diagnose Eq. 10 height transitions. */
  supportRadius?: number;
}

// The presets retain approximately the x/z resolution of the former 110k,
// 500k and 1.2m cubic grids for the default scene. Only the moving surface
// band and two samples for each bottom tall cell are stored.
export const tallCellSettings: Record<GPUQuality, TallCellSettings> = {
  balanced: { surfaceColumns: 2_500, regularLayers: 24, liquidHalo: 8, airHalo: 8, maximumNeighborDelta: 4, maximumTallHeight: 4096, remeshInterval: 1 },
  high: { surfaceColumns: 7_000, regularLayers: 32, liquidHalo: 16, airHalo: 8, maximumNeighborDelta: 4, maximumTallHeight: 4096, remeshInterval: 1 },
  ultra: { surfaceColumns: 12_500, regularLayers: 40, liquidHalo: 24, airHalo: 8, maximumNeighborDelta: 5, maximumTallHeight: 4096, remeshInterval: 1 }
};

export function tallCellFluxSampleCount(height: number) {
  const cells = Math.max(0, Math.floor(height));
  return Math.min(cells, 12);
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
    // Section 8 gives the air-halo constraint priority when a vertically
    // extended interface cannot satisfy both halos in the fixed B_y band.
    : lowerBound;
  const clamped = Math.max(0, Math.min(maximumBase, desired));
  // A restricted packed column always owns one tall cell. A one-subcell tall
  // cell has coincident endpoint samples, while base zero would retain only
  // the fixed regular band rather than a complete ordinary column. The method
  // selects its separately allocated uniform backend when h >= 2 is globally
  // impossible, so every restricted column can safely keep this minimum.
  return maximumBase >= 2 ? Math.max(2, clamped) : 0;
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
  if (sceneHasTerrain(scene)) {
    const c = scene.container;
    const worldX = -0.5 * c.width_m + (x + 0.5) * c.width_m / nx;
    const worldZ = -0.5 * c.depth_m + (z + 0.5) * c.depth_m / nz;
    if ((y + 0.5) * c.height_m / fineNy <= terrainHeightAt(scene.terrain, worldX, worldZ)) return false;
  }
  const brickWet = initialFluidBrickContainsCell(scene, x, y, z, [nx, fineNy, nz]);
  if (brickWet !== undefined) return brickWet;
  if (scene.fluid.initialCondition === "tank-fill") return (y + 0.5) / fineNy <= scene.container.fillFraction;
  const dam = damBreakFractions(scene.container.fillFraction);
  return (x + 0.5) / nx <= dam.width && (y + 0.5) / fineNy <= dam.height && (z + 0.5) / nz <= dam.depth;
}

function boxSignedDistance(point: { x: number; y: number; z: number }, center: { x: number; y: number; z: number }, half: { x: number; y: number; z: number }) {
  const qx = Math.abs(point.x - center.x) - half.x;
  const qy = Math.abs(point.y - center.y) - half.y;
  const qz = Math.abs(point.z - center.z) - half.z;
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  return outside + Math.min(Math.max(qx, qy, qz), 0);
}

/** Analytic initial liquid signed distance in metres. Container walls are
 * solid contacts, not liquid-air interfaces, so a tank fill is the vertical
 * free-surface plane while a dam break uses the finite liquid block. */
export function initialLiquidPhi(scene: SceneDescription, point: { x: number; y: number; z: number }, dimensions?: readonly [number, number, number]) {
  const c = scene.container;
  if (dimensions) {
    const brickDistance = initialFluidBrickSignedDistance(scene, point, dimensions);
    if (brickDistance !== undefined) return brickDistance;
  }
  if (scene.fluid.initialCondition === "tank-fill") return point.y - c.height_m * c.fillFraction;
  const dam = damBreakFractions(c.fillFraction);
  const half = { x: 0.5 * dam.width * c.width_m, y: 0.5 * dam.height * c.height_m, z: 0.5 * dam.depth * c.depth_m };
  return boxSignedDistance(point, {
    x: -0.5 * c.width_m + half.x,
    y: half.y,
    z: -0.5 * c.depth_m + half.z
  }, half);
}

function buildInitialPhi(scene: SceneDescription, nx: number, fineNy: number, nz: number, packedNy: number, columnBases: Float32Array) {
  const c = scene.container;
  const h = { x: c.width_m / nx, y: c.height_m / fineNy, z: c.depth_m / nz };
  const limit = 5 * Math.min(h.x, h.y, h.z);
  const phi = new Float32Array(nx * packedNy * nz);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = columnBases[x + nx * z];
    for (let packedY = 0; packedY < packedNy; packedY += 1) {
      const sampleY = packedY === 0 ? 0.5 : packedY === 1 ? Math.max(0.5, base - 0.5) : base + packedY - 1.5;
      const active = packedY < 2 ? base > 0 : base + packedY - 2 < fineNy;
      const value = active ? initialLiquidPhi(scene, {
        x: -0.5 * c.width_m + (x + 0.5) * h.x,
        y: sampleY * h.y,
        z: -0.5 * c.depth_m + (z + 0.5) * h.z
      }, [nx, fineNy, nz]) : limit;
      phi[x + nx * (packedY + packedNy * z)] = Math.max(-limit, Math.min(limit, value));
    }
  }
  return phi;
}

function insideTerrainCell(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  if (!sceneHasTerrain(scene)) return false;
  const c = scene.container;
  const worldX = -0.5 * c.width_m + (x + 0.5) * c.width_m / nx;
  const worldZ = -0.5 * c.depth_m + (z + 0.5) * c.depth_m / nz;
  return (y + 0.5) * c.height_m / fineNy <= terrainHeightAt(scene.terrain, worldX, worldZ);
}

function isInitialSurfaceCell(scene: SceneDescription, x: number, y: number, z: number, nx: number, fineNy: number, nz: number) {
  const wet = initialWet(scene, x, y, z, nx, fineNy, nz);
  // The moving cubic band follows vertical crossings within a column. A
  // vertical liquid face is represented by horizontal interpolation between
  // neighbouring tall endpoint values and does not force B_y to the floor.
  // A wet/dry transition against the terrain heightfield is a solid contact
  // like the flat floor — never a liquid-air surface — so it must not drag
  // the band down to the pool bed (the paper's H excludes the ground from
  // the water column entirely).
  const belowWet = y === 0 || insideTerrainCell(scene, x, y - 1, z, nx, fineNy, nz)
    ? wet
    : initialWet(scene, x, y - 1, z, nx, fineNy, nz);
  const aboveWet = y + 1 < fineNy && initialWet(scene, x, y + 1, z, nx, fineNy, nz);
  return wet !== belowWet || wet !== aboveWet;
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
  // Grow B_y only when the vertical crossings and their halos do not fit. A
  // vertical dam face is representable across neighbouring tall columns.
  const requiredLayers = requiredInitialRegularLayers(scene, nx, fineNy, nz, settings);
  // The paper's Section 5 attributes volume artifacts to lateral fluxes that
  // enter unmeasured tall-cell interiors. Until those virtual faces are part
  // of the pressure unknowns, retain only one omitted cubic layer by default.
  const parityLayers = fineNy - Math.max(2, Math.round(settings.maximumTallHeight));
  const regularLayers = Math.min(fineNy, Math.max(settings.regularLayers, requiredLayers, parityLayers));
  const storedRegularLayers = regularLayers;
  const regularLayersBeforeOrdinaryFallback = regularLayers;
  const maximumBaseBeforeOrdinaryFallback = Math.max(0, fineNy - regularLayers);
  const ordinaryGridFallback = regularLayers >= fineNy;
  const effectiveSettings = { ...settings, regularLayers, liquidHalo: Math.min(settings.liquidHalo, regularLayers), airHalo: Math.min(settings.airHalo, regularLayers) };
  const packedNy = storedRegularLayers + 2;
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
    const worldX = -c.width_m / 2 + (x + 0.5) * c.width_m / nx;
    const worldZ = -c.depth_m / 2 + (z + 0.5) * c.depth_m / nz;
    const inflow = scene.fluid.inflow;
    if (inflow) {
      const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
      const direction = speed > 0
        ? { x: inflow.velocity_m_s.x / speed, y: inflow.velocity_m_s.y / speed, z: inflow.velocity_m_s.z / speed }
        : { x: 1, y: 0, z: 0 };
      const halfLength = inflow.length_m / 2;
      const extentX = Math.abs(direction.x) * halfLength + Math.sqrt(Math.max(0, 1 - direction.x * direction.x)) * inflow.radius_m;
      const extentY = Math.abs(direction.y) * halfLength + Math.sqrt(Math.max(0, 1 - direction.y * direction.y)) * inflow.radius_m;
      const extentZ = Math.abs(direction.z) * halfLength + Math.sqrt(Math.max(0, 1 - direction.z * direction.z)) * inflow.radius_m;
      if (Math.abs(worldX - inflow.center_m.x) <= extentX + c.width_m / nx / 2 && Math.abs(worldZ - inflow.center_m.z) <= extentZ + c.depth_m / nz / 2) {
        lowestSurface = Math.min(lowestSurface, Math.max(0, Math.floor((inflow.center_m.y - extentY) / c.height_m * fineNy)));
        highestSurface = Math.max(highestSurface, Math.min(fineNy - 1, Math.ceil((inflow.center_m.y + extentY) / c.height_m * fineNy)));
      }
    }
    let bodyUpper = maximumBase;
    for (const body of scene.rigidBodies) {
      const d = body.dimensions_m;
      const radius = body.shape === "sphere" ? d.x : Math.hypot(d.x, d.y, d.z) / 2;
      if (Math.hypot(worldX - body.position_m.x, worldZ - body.position_m.z) > radius + Math.max(c.width_m / nx, c.depth_m / nz) / 2) continue;
      // Rigid geometry is not a liquid interface and must never lift the
      // regular band. Doing so makes the bottom tall store grow as soon as a
      // body approaches the water, even though the surface band already
      // contains every cell that can exchange momentum with that body.
      // The body's bottom is only an upper bound: if the body is initially
      // submerged, keep it out of the tall store by shortening that store.
      const bodyBottom = Math.floor((body.position_m.y - radius) / c.height_m * fineNy);
      let wetTop = -1;
      for (let y = fineNy - 1; y >= 0; y -= 1) if (initialWet(scene, x, y, z, nx, fineNy, nz)) { wetTop = y; break; }
      if (bodyBottom <= wetTop + 1 + effectiveSettings.airHalo) bodyUpper = Math.min(bodyUpper, bodyBottom);
    }
    const surfaceBase = highestSurface >= 0
      ? chooseTallCellBase(lowestSurface, highestSurface, fineNy, effectiveSettings)
      // An empty column still needs its regular band at the same elevation as
      // neighbouring water.  Setting it to zero discards the upper domain and
      // makes the D limiter pull every wet tall cell down a few cells per
      // frame.  The paper has one tall cell in every column, including air.
      // In parity mode the near-full-height band already covers every dry
      // column. Keep those columns at the height-two control and introduce
      // h=3 only where liquid depth requires it; the one-cell jump satisfies
      // Eq. 10 and avoids thousands of unnecessary endpoint perturbations.
      : maximumBase <= 3 ? Math.min(2, maximumBase) : maximumBase;
    rawBases[x + nx * z] = maximumBase >= 2
      ? Math.max(2, Math.min(surfaceBase, bodyUpper))
      : 0;
  }

  const columnBases = limitNeighboringTallCellBases(rawBases, nx, nz, effectiveSettings.maximumNeighborDelta, nx + nz);
  const initialVolume = new Float32Array(nx * packedNy * nz);
  let initialVolumeCellSum = 0;
  let activeSampleCount = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = columnBases[x + nx * z];
    let tallAmount = 0;
    for (let worldY = 0; worldY < base; worldY += 1) if (initialWet(scene, x, worldY, z, nx, fineNy, nz)) tallAmount += 1;
    const tallFraction = base > 0 ? tallAmount / base : 0;
    for (let packedY = 0; packedY < packedNy; packedY += 1) {
      let worldY: number;
      if (packedY === 0) worldY = 0;
      else if (packedY === 1) worldY = Math.max(0, base - 1);
      else worldY = base + packedY - 2;
      const active = packedY >= 2 ? worldY < fineNy : base > 0;
      if (active) activeSampleCount += 1;
      const fraction = packedY < 2 ? tallFraction : active && initialWet(scene, x, worldY, z, nx, fineNy, nz) ? 1 : 0;
      initialVolume[x + nx * (packedY + packedNy * z)] = fraction;
      if (packedY === 0) initialVolumeCellSum += tallAmount;
      else if (packedY >= 2) initialVolumeCellSum += fraction;
    }
  }

  const packedSampleCount = nx * packedNy * nz;
  const equivalentUniformCellCount = nx * fineNy * nz;
  const initialPhi = buildInitialPhi(scene, nx, fineNy, nz, packedNy, columnBases);
  return {
    nx, fineNy, nz, packedNy,
    cellSize_m: { x: c.width_m / nx, y: c.height_m / fineNy, z: c.depth_m / nz },
    columnBases, initialVolume, initialPhi, initialVolumeCellSum,
    referenceLiquidVolume_cells: initialVolumeCellSum,
    packedSampleCount, activeSampleCount, equivalentUniformCellCount,
    compressionRatio: packedSampleCount / equivalentUniformCellCount,
    activeCompressionRatio: activeSampleCount / equivalentUniformCellCount,
    settings: effectiveSettings,
    planning: {
      requestedRegularLayers: settings.regularLayers,
      requiredInitialRegularLayers: requiredLayers,
      storedRegularLayers,
      regularLayersBeforeOrdinaryFallback,
      maximumBaseBeforeOrdinaryFallback,
      ordinaryGridFallback
    }
  };
}

/**
 * Build the minimal differential layout used to identify the first operation
 * that changes when a tall cell is introduced.
 *
 * The control columns use height two, the smallest distinct-endpoint tall
 * cell and an exactly measured cubic-equivalent representation. Exactly one
 * column is taller. This retains the paper's "one bottom tall cell per
 * column" invariant while muting every non-probe candidate. Remeshing is
 * frozen so Algorithm 1 cannot change the experiment before it is measured.
 *
 * Paper constraints retained by the probe:
 * - the cell is at the bottom of its column (Sec. 3.1);
 * - no liquid interface lies inside it (Sec. 3.6 constraints 1/2);
 * - its height jump does not exceed D (Sec. 3.6, Eq. 10).
 */
export function createSingleTallCellProbeLayout(
  scene: SceneDescription,
  quality: GPUQuality,
  maximumTextureDimension = 2048,
  options: SingleTallCellProbeOptions = {}
): TallCellLayout {
  const requestedHeight = Math.round(options.height ?? 4);
  const maximumNeighborDelta = Math.max(tallCellSettings[quality].maximumNeighborDelta, requestedHeight);
  const layout = createTallCellLayout(scene, quality, maximumTextureDimension, {
    regularLayers: maximumTextureDimension,
    maximumNeighborDelta,
    remeshInterval: Number.MAX_SAFE_INTEGER
  });
  const { nx, fineNy, nz, packedNy } = layout;
  const height = Math.max(2, Math.min(fineNy, requestedHeight));
  if (height > maximumNeighborDelta) throw new Error(`Single tall-cell height ${height} exceeds neighbor delta ${maximumNeighborDelta}`);
  const x = Math.max(0, Math.min(nx - 1, Math.round(options.x ?? nx / 2)));
  const z = Math.max(0, Math.min(nz - 1, Math.round(options.z ?? nz / 2)));
  const supportRadius = Math.max(0, Math.round(options.supportRadius ?? 0));
  const supported = (xx: number, zz: number) => Math.abs(xx - x) + Math.abs(zz - z) <= supportRadius;
  const bottomWet = initialWet(scene, x, 0, z, nx, fineNy, nz);
  let affectedColumns = 0;
  for (let zz = 0; zz < nz; zz += 1) for (let xx = 0; xx < nx; xx += 1) {
    if (!supported(xx, zz)) continue;
    affectedColumns += 1;
    const columnWet = initialWet(scene, xx, 0, zz, nx, fineNy, nz);
    for (let y = 1; y < height; y += 1) {
      if (initialWet(scene, xx, y, zz, nx, fineNy, nz) !== columnWet) {
        throw new Error(`Tall-cell probe support column (${xx},${zz}) height ${height} contains the initial liquid interface at y=${y}`);
      }
    }
  }

  const columnBases = new Float32Array(nx * nz).fill(2);
  for (let zz = 0; zz < nz; zz += 1) for (let xx = 0; xx < nx; xx += 1) {
    if (supported(xx, zz)) columnBases[xx + nx * zz] = height;
  }
  const initialVolume = new Float32Array(nx * packedNy * nz);
  let initialVolumeCellSum = 0;
  let activeSampleCount = 0;
  for (let zz = 0; zz < nz; zz += 1) for (let xx = 0; xx < nx; xx += 1) {
    const base = columnBases[xx + nx * zz];
    let tallAmount = 0;
    for (let worldY = 0; worldY < base; worldY += 1) if (initialWet(scene, xx, worldY, zz, nx, fineNy, nz)) tallAmount += 1;
    const tallFraction = base > 0 ? tallAmount / base : 0;
    for (let packedY = 0; packedY < packedNy; packedY += 1) {
      const worldY = packedY < 2 ? (packedY === 0 ? 0 : Math.max(0, base - 1)) : base + packedY - 2;
      const active = packedY < 2 ? base > 0 : worldY < fineNy;
      if (active) activeSampleCount += 1;
      const fraction = packedY < 2 ? tallFraction : active && initialWet(scene, xx, worldY, zz, nx, fineNy, nz) ? 1 : 0;
      initialVolume[xx + nx * (packedY + packedNy * zz)] = fraction;
      if (packedY === 0) initialVolumeCellSum += tallAmount;
      else if (packedY >= 2) initialVolumeCellSum += fraction;
    }
  }
  const equivalentUniformCellCount = nx * fineNy * nz;
  const initialPhi = buildInitialPhi(scene, nx, fineNy, nz, packedNy, columnBases);
  return {
    ...layout,
    columnBases,
    initialVolume,
    initialPhi,
    initialVolumeCellSum,
    referenceLiquidVolume_cells: initialVolumeCellSum,
    activeSampleCount,
    activeCompressionRatio: activeSampleCount / equivalentUniformCellCount,
    settings: { ...layout.settings, maximumNeighborDelta, remeshInterval: Number.MAX_SAFE_INTEGER },
    singleTallCellProbe: {
      x, z, height, initialState: bottomWet ? "liquid" : "air", mutedHeight: 2,
      supportRadius, affectedColumns, topologyFrozen: true
    }
  };
}

/** Identical resources and solver path to createSingleTallCellProbeLayout,
 * but with the probe column muted back to the cubic-equivalent height-two
 * limit. The pair differs only in one column base and that column's packing. */
export function createSingleTallCellProbeControlLayout(
  scene: SceneDescription,
  quality: GPUQuality,
  maximumTextureDimension = 2048,
  options: SingleTallCellProbeOptions = {}
): TallCellLayout {
  const candidate = createSingleTallCellProbeLayout(scene, quality, maximumTextureDimension, options);
  const columnBases = new Float32Array(candidate.columnBases.length).fill(2);
  const initialVolume = candidate.initialVolume.slice();
  for (let zz = 0; zz < candidate.nz; zz += 1) for (let xx = 0; xx < candidate.nx; xx += 1) {
    if (candidate.columnBases[xx + candidate.nx * zz] === 2) continue;
    let tallAmount = 0;
    for (let worldY = 0; worldY < 2; worldY += 1) if (initialWet(scene, xx, worldY, zz, candidate.nx, candidate.fineNy, candidate.nz)) tallAmount += 1;
    for (let packedY = 0; packedY < candidate.packedNy; packedY += 1) {
      const worldY = packedY;
      const fraction = packedY < 2 ? tallAmount / 2 : worldY < candidate.fineNy
        && initialWet(scene, xx, worldY, zz, candidate.nx, candidate.fineNy, candidate.nz) ? 1 : 0;
      initialVolume[xx + candidate.nx * (packedY + candidate.packedNy * zz)] = fraction;
    }
  }
  return {
    ...candidate,
    columnBases,
    initialVolume,
    initialPhi: buildInitialPhi(scene, candidate.nx, candidate.fineNy, candidate.nz, candidate.packedNy, columnBases),
    activeSampleCount: candidate.equivalentUniformCellCount,
    activeCompressionRatio: 1,
    singleTallCellProbe: undefined
  };
}
