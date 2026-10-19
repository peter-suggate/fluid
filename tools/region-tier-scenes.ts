import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";

export const REGION_TIER_SCENES = ["cold", "rest", "warm", "splash", "hot"] as const;
export type RegionTierScene = typeof REGION_TIER_SCENES[number];

/** Fixed h, waterline and disturbance: length/depth add only settled water.
 * No prescribed refinement or forced heat: production policy chooses the tiers. */
export function createRegionTierScene(mode: RegionTierScene, lengthCells = 128, depthCells = 32): SceneDescription {
  if (!Number.isInteger(lengthCells) || lengthCells < 64 || lengthCells % 32) throw new Error("length must be a multiple of 32, at least 64");
  if (!Number.isInteger(depthCells) || depthCells < 32 || depthCells % 32) throw new Error("depth must be a multiple of 32, at least 32");
  const scene = cloneScene(defaultScene), h = .025, width = lengthCells * h, depth = depthCells * h;
  scene.sceneId = `region-tier-${mode}-${lengthCells}${depthCells === 32 ? "" : `-depth${depthCells}`}`;
  scene.duration_s = 10;
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  scene.container = { ...scene.container, shape: "box", vessel: "none", width_m: width,
    height_m: .8, depth_m: depth, fillFraction: .5, top: "open", fluidWallMode: "free-slip", depthBoundary: "closed" };
  scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
  scene.fluid = { density_kg_m3: 1000, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: mode === "cold" ? 0 : -9.80665, z: 0 }, initialCondition: "tank-fill",
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -width / 2, y: 0, z: -depth / 2 }, max_m: { x: width / 2, y: .4, z: depth / 2 } }] };
  if (mode === "warm") {
    delete scene.fluid.initialLiquidVolumes;
    scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: .4, amplitude_m: .005, wavelength_m: 2 * width, originX_m: -width / 2 };
  }
  if (mode === "splash") scene.fluid.scheduledDrops = [{ time_s: .5,
    volume: { shape: "sphere", center_m: { x: -width / 2 + .6, y: .6, z: 0 }, radius_m: .1 } }];
  if (mode === "hot") {
    // The breaking crest's support rises above the original box. Reserve
    // headroom so this exercises active work rather than pool exhaustion.
    scene.fluid.pagePool = { blocks: 2 + 4 * (lengthCells / 32) * (depthCells / 32) };
    scene.fluid.initialCondition = "dam-break";
    scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -width / 2, y: 0, z: -depth / 2 }, max_m: { x: 0, y: .7, z: depth / 2 } }];
  }
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  return scene;
}
