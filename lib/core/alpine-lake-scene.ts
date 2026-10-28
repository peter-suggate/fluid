import type { CameraState, SceneDescription } from "./model";
import type { TerrainDescription } from "./terrain";
import { nbflipDomain } from "./nbflip-paper-scenes";
import { alpineLakeScenery } from "./alpine-lake-scenery";

export const ALPINE_LAKE_SCENE_ID = "alpine-lake-sunrise";
export const ALPINE_LAKE_GRID = [128, 16, 160] as const;
export const ALPINE_LAKE_CELL_M = .5;
export const ALPINE_LAKE_WATERLINE_M = 2.3;

/** Shared analytic lake bed: mirrored by alpine_lake.py in Blender Z-up.
 * The banks close before the domain walls, so no artificial tank edge is visible. */
export function alpineLakeBedHeight(x: number, z: number): number {
  const r = Math.hypot(x / 30, z / 38);
  return .3 + 2.8 * Math.min(1.5, r ** 6)
    + .045 * Math.sin(x * .7) * Math.sin(z * .6);
}

export function alpineLakeTerrain(): TerrainDescription {
  const spacing_m = .5, nx = 161, nz = 193;
  return { baseHeight_m: 4.5, features: [], grid: {
    kind: "grid", origin_m: { x: -40, z: -48 }, spacing_m, size: { nx, nz },
    heights_m: Array.from({ length: nx * nz }, (_, i) =>
      alpineLakeBedHeight(-40 + i % nx * spacing_m, -48 + Math.floor(i / nx) * spacing_m)),
  } };
}

/** Calm alpine tarn. All liquid is real NB-FLIP; Cycles receives the solver's
 * exported free surface, while mountains and distant trees are scenery. */
export function createAlpineLakeScene(): SceneDescription {
  const scene = nbflipDomain(ALPINE_LAKE_SCENE_ID, ALPINE_LAKE_GRID, ALPINE_LAKE_CELL_M, 600, 0, ALPINE_LAKE_GRID, 1 / 60);
  scene.randomSeed = 20261028;
  scene.container = { ...scene.container, top: "open", vessel: "none", fillFraction: 0 };
  scene.systems = { fluid: true };
  scene.solidVoxels = [];
  scene.terrain = alpineLakeTerrain();
  scene.fluid.initialVelocity_m_s = { x: .012, y: 0, z: -.005 };
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -31, y: 0, z: -39 }, max_m: { x: 31, y: ALPINE_LAKE_WATERLINE_M, z: 39 } }];
  scene.fluid.pagePool = { memory_MiB: 2048 };
  scene.fluid.optics = { indexOfRefraction: 1.333, absorption_mInv: [.075, .024, .013], roughness: .04 };
  scene.scenery = alpineLakeScenery(scene.terrain);
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-.5, .28, .8], colorLinear: [1, .75, .51], intensity: 8 },
    environment: { cloudCoverage: .35, diffuseScale: 1.2, specularScale: 1, lowerRadianceLinear: [.24, .3, .36], upperRadianceLinear: [.75, 1.1, 1.65], accentRadianceLinear: [.9, .7, .5] },
    grade: { toneCurve: "agx", exposure: 1.3, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const alpineLakeCamera: Partial<CameraState> = {
  azimuth_rad: 0, elevation_rad: -Math.atan2(7.5, 42), distance_m: Math.hypot(42, 7.5), tanHalfFov: .4,
  target_m: { x: 0, y: 12, z: -5 },
};
