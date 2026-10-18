import { createHeroGardenHoseScene, heroGardenCamera, HERO_GARDEN_POND_CELL_M, HERO_GARDEN_SOLVER_CELL_M, type HeroGardenHoseOptions } from "./hero-garden-scene";
import type { SceneDescription } from "./model";
import type { SceneryNode } from "./scenery-graph";
import { meshAssetHeader } from "./imported-assets/asset-library";
import instances from "./imported-assets/hero-garden-instances.json";

// Twice the linear fluid resolution of the original 12.5 mm garden.
export const HERO_GARDEN_BLENDER_FLUID_CELL_M = HERO_GARDEN_SOLVER_CELL_M / 2;
export const HERO_GARDEN_BLENDER_ID = "hero-garden-hose-x10-blender";
export const heroGardenBlenderCamera = { ...heroGardenCamera, distance_m: 3.2,
  target_m: { x: -.04, y: .62, z: 0 } };

/** Local GLB origins are preserved by rotating the sampled field's centre too. */
function imported(instance: typeof instances[number]): SceneryNode {
  const h = meshAssetHeader(instance.asset);
  const [cx, cy, cz] = h.origin.map((v, i) => v + h.dimensions[i]! * h.cellSize / 2);
  const [x, y, z] = instance.position;
  const c = Math.cos(instance.yaw), s = Math.sin(instance.yaw);
  return {
    kind: "field-program", id: instance.id,
    program: { ops: [], result: 0, sampledVolume: { asset: instance.asset } },
    place: { units: "metres", position: { x: x! + c * cx! + s * cz!, y: y! + cy!, z: z! - s * cx! + c * cz! },
      orientation: { w: Math.cos(instance.yaw / 2), x: 0, y: Math.sin(instance.yaw / 2), z: 0 } },
    // Coarse lighting samples can fall back to the owner material. Keep that
    // fallback in the authored palette rather than reintroducing white clay.
    material: { colorLinear: h.materials[0]!.baseColorLinear, surface: "ceramic" },
  };
}

/** A separate, entirely Blender-authored set over the existing pond physics.
 * The closed pond mesh carries a 2 mm finish over the collision heightfield;
 * trees, stones, gravel, planting, ground and hose all use imported PBR fields.
 * Water and its jet remain live simulation, never a decorative imported mesh.
 */
export function createHeroGardenBlenderScene(options: HeroGardenHoseOptions = {}): SceneDescription {
  const scene = createHeroGardenHoseScene({ ...options, pondTank: true, water: options.water ?? true,
    cellSize_m: options.cellSize_m ?? HERO_GARDEN_BLENDER_FLUID_CELL_M,
    detailCellSize_m: options.detailCellSize_m ?? Math.min(options.cellSize_m ?? HERO_GARDEN_POND_CELL_M, HERO_GARDEN_POND_CELL_M) });
  scene.sceneId = HERO_GARDEN_BLENDER_ID;
  scene.scenery = { terrainSurface: "scenery", palettes: {}, nodes: [
    { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
    ...instances.map(imported),
  ] };
  // The surroundings are authored mesh instances as well, replacing the old
  // procedural hills/scatter so there is no mixed legacy scenery in this set.
  delete scene.backdrop;
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    rasterAoContrast: 2,
    directional: { direction: [-.50, .78, .38], colorLinear: [1, .96, .90], intensity: 8 },
    environment: { diffuseScale: 1, specularScale: 1,
      lowerRadianceLinear: [.65, .60, .50], upperRadianceLinear: [.82, .95, 1.12],
      accentRadianceLinear: [.8, .85, .9] },
    grade: { toneCurve: "agx", exposure: 1.4, whiteBalance: [1, 1, 1] },
  };
  return scene;
}
