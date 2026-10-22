import { cloneScene, defaultScene, type CameraState, type InitialLiquidVolume, type SceneDescription, type Vec3 } from "./model";
import type { ProceduralSolidBoundary } from "./procedural-solid";
import type { SolidWorldVoxelPatch } from "./solid-world";
import type { TerrainDescription } from "./terrain";
import { forestWaterfallScenery } from "./forest-waterfall-scenery";
import colliderBake from "./imported-assets/metadata/forest-waterfall-colliders.json";

export const FOREST_WATERFALL_ID = "forest-waterfall";
export const FOREST_WATERFALL_CELL_M = .125;
export const FOREST_WATERFALL_GRID = [256, 208, 256] as const;
export const FOREST_WATERFALL_TERRAIN_CELL_M = .25;
export const FOREST_WATERFALL_WATERLINE_M = 3.1;
export const FOREST_WATERFALL_SECONDS = 60;
export const FOREST_WATERFALL_SOURCE = { x: -1, y: 22.65, z: -7.74 } as const;
export const FOREST_WATERFALL_POOL = { x: 0, z: 2, radiusX: 10.92, radiusZ: 10.4 } as const;

/** Blender Z-up metres become application Y-up metres, with the basin raised
 * three metres so all fluid and collision geometry are above the domain floor. */
export const forestWaterfallPoint = (x: number, y: number, z: number): Vec3 => ({ x, y: z + 3, z: -y });

/** The continuous visible ground is also the solver's authoritative bed.
 * Its heightfield avoids baking the entire cliff's solid interior. */
export function forestWaterfallTerrainHeight(x: number, z: number): number {
  const by = -z;
  const r = Math.hypot(x / FOREST_WATERFALL_POOL.radiusX, (z - 2) / FOREST_WATERFALL_POOL.radiusZ);
  const undulation = .12 * Math.sin(x * .61) * Math.sin(by * .47)
    + .08 * Math.sin(x * 1.13 + by * .71);
  let ground = 3.38 + undulation + Math.max(0, Math.abs(x) - 11) * .16
    + Math.max(0, z - 12) * .20;
  if (r < 1.06) ground = 1.15 + 2.18 * Math.min(1, (r / 1.06) ** 5);
  const edge = 8.65 + .65 * Math.sin(x * .33) + .28 * Math.sin(x * 1.4);
  const plateau = 22.4 + 1.4 * Math.sin(x * .19) + .25 * Math.sin(x * .35 + by * .19);
  if (by > edge) {
    const t = Math.min(1, (by - edge) / .5);
    ground = ground * (1 - t) + plateau * t;
  }
  // A shallow watercourse through the plateau keeps every point of the
  // real source disk clear. The falling water leaves the front of this lip.
  if (Math.abs(x + 1) < 2.4 && by > 7.4 && by < 12.4) ground = Math.min(ground, 21.5);
  return Math.max(.25, Math.min(25, ground));
}

export function forestWaterfallTerrain(): TerrainDescription {
  const spacing_m = FOREST_WATERFALL_TERRAIN_CELL_M, nx = 321, nz = 385;
  return { baseHeight_m: 3.38, features: [], grid: {
    kind: "grid", origin_m: { x: -40, z: -48 }, spacing_m, size: { nx, nz },
    heights_m: Array.from({ length: nx * nz }, (_, i) =>
      forestWaterfallTerrainHeight(-40 + i % nx * spacing_m, -48 + Math.floor(i / nx) * spacing_m)),
  } };
}

/** Original layout bounds. The five ledges collide with the exported mesh
 * bake below; only the two landing-zone stones retain these ellipsoids. */
export const FOREST_WATERFALL_ROCKS = [
  { center: forestWaterfallPoint(-1, 8.7, 18.3), radius: { x: 3.3, y: .65, z: 1.1 } },
  { center: forestWaterfallPoint(-1, 7.4, 11.65), radius: { x: 2.8, y: .65, z: 1.5 } },
  { center: forestWaterfallPoint(-.5, 7.1, 4.6), radius: { x: 2.65, y: .75, z: 1.5 } },
  { center: forestWaterfallPoint(-3.6, 7.5, 7.7), radius: { x: 1.3, y: 2.6, z: 1.0 } },
  { center: forestWaterfallPoint(2.8, 7.7, 14.2), radius: { x: 1.4, y: 2.5, z: 1.1 } },
  { center: forestWaterfallPoint(-5, 5.6, .26), radius: { x: 1.5, y: 1.0, z: 1.1 } },
  { center: forestWaterfallPoint(3.8, 4.6, .26), radius: { x: 1.2, y: .8, z: .9 } },
] as const;

const heldRockGeometry = {
  solidVoxels: colliderBake.solidVoxels as unknown as SolidWorldVoxelPatch[],
  solidBoundary: colliderBake.solidBoundary as unknown as ProceduralSolidBoundary,
};

/** Offline resampling preserves the actual deformed and rotated Blender
 * shoulders at the fluid's 12.5 cm pitch. No mesh loading or CPU bake is needed
 * when a user selects the normal scene. Rebuild with bake-forest-colliders.ts. */
export function forestWaterfallRockGeometry() {
  return heldRockGeometry;
}

/** The standard initial-liquid rasterizer clips this volume against the
 * elliptical basin and rock field. The banks rise above its flat free surface,
 * so one volume fills the whole cavity without artificial seams or box edges. */
export function forestWaterfallInitialPool(): InitialLiquidVolume[] {
  return [{ shape: "box", min_m: { x: -12, y: 0, z: -10 },
    max_m: { x: 12, y: FOREST_WATERFALL_WATERLINE_M, z: 14 } }];
}

/** Live NB-FLIP scene: only the initial pool and a running source describe
 * water. Every falling sheet, impact and splash is produced by the solver. */
export function createForestWaterfallScene(): SceneDescription {
  const scene = cloneScene(defaultScene), h = FOREST_WATERFALL_CELL_M;
  scene.sceneId = FOREST_WATERFALL_ID;
  scene.randomSeed = 20261022;
  scene.duration_s = FOREST_WATERFALL_SECONDS;
  scene.systems = { fluid: true };
  scene.container = { width_m: 32, height_m: 26, depth_m: 32, fillFraction: 0,
    top: "open", fluidWallMode: "free-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: h };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
  scene.fluid = {
    density_kg_m3: 998.2, dynamicViscosity_Pa_s: .001002, surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 }, initialCondition: "tank-fill",
    initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: forestWaterfallInitialPool(),
    inflow: { center_m: { ...FOREST_WATERFALL_SOURCE }, radius_m: 1.35,
      length_m: .7 / 120,
      velocity_m_s: { x: 0, y: -.7, z: 0 }, start_s: 0,
      end_s: FOREST_WATERFALL_SECONDS, ramp_s: .5 },
    // Match the bounded-container default: all 8×7×8 domain pages, one
    // 8×8 open-top layer, and two sentinels. The initial 200-page closure
    // grows beyond 254 pages when the falling stream approaches the pond.
    pagePool: { memory_MiB: 5140 },
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.19, .045, .035], roughness: .04 },
  };
  scene.terrain = forestWaterfallTerrain();
  const rocks = forestWaterfallRockGeometry();
  scene.solidVoxels = structuredClone(rocks.solidVoxels);
  scene.solidBoundary = structuredClone(rocks.solidBoundary);
  scene.rigidBodies = [];
  scene.scenery = { ...forestWaterfallScenery(scene.terrain), solidBoundarySurface: "scenery" };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    rasterAoContrast: 1.7,
    directional: { direction: [-.56, .76, .33], colorLinear: [1, .93, .79], intensity: 8 },
    environment: { diffuseScale: .8, specularScale: 1,
      lowerRadianceLinear: [.12, .19, .08], upperRadianceLinear: [.64, .83, 1.05],
      accentRadianceLinear: [.55, .65, .42] },
    grade: { toneCurve: "agx", exposure: 1.1, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const forestWaterfallCamera: Partial<CameraState> = {
  azimuth_rad: .47, elevation_rad: .20, distance_m: 48, tanHalfFov: .40,
  target_m: { x: -1, y: 11, z: -4 },
};
