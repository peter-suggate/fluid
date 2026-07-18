import { cloneScene, type SceneDescription } from "./model";
import { terrainHeightAt, type TerrainDescription } from "./terrain";

/**
 * The garden pond: an organic, kidney-shaped pool carved into a lawn. Three
 * overlapping basins (deep pool, side lobe, pebble beach shelf) merge through
 * the terrain smooth-union into one natural hollow; a mown knoll and a low
 * berm keep the surrounding lawn from reading as a snooker table.
 *
 * The same heights feed the solvers (as solid columns), the rigid contacts
 * and the rendered grass, so a splash that lands on the bank really lands.
 */

export const GARDEN_CONTAINER = { width_m: 3.0, height_m: 1.0, depth_m: 2.2 } as const;
/** Lawn level above the container floor. */
export const GARDEN_GRASS_M = 0.38;
/** Still-water level of the filled pond: 3.5 cm of exposed bank. */
export const GARDEN_WATERLINE_M = 0.345;
/**
 * Raised seed over the upstream lip of the deep basin. The solver fills
 * exactly the one 8^3 brick containing this point; the released body drops
 * through the basin mouth instead of sheeting around its left bank.
 */
export const GARDEN_DAM_BRICK_SEED_M = { x: -0.5, y: 0.61, z: -0.3 } as const;

export function gardenPoolTerrain(): TerrainDescription {
  return {
    baseHeight_m: GARDEN_GRASS_M,
    features: [
      // Deep pool body — floor 4 cm above the container floor.
      { kind: "basin", center_m: { x: -0.35, z: -0.12 }, radius_m: { x: 0.78, z: 0.6 }, amount_m: 0.34, rotation_rad: 0.35, flat: 0.5 },
      // Kidney lobe curling toward the back right.
      { kind: "basin", center_m: { x: 0.45, z: 0.3 }, radius_m: { x: 0.6, z: 0.48 }, amount_m: 0.3, rotation_rad: -0.4, flat: 0.45 },
      // Pebble-beach shelf: a shallow entry slope on the front edge.
      { kind: "basin", center_m: { x: 0.25, z: -0.55 }, radius_m: { x: 0.55, z: 0.42 }, amount_m: 0.2, rotation_rad: 0.2, flat: 0.3 },
      // Mown knoll rising behind the pond.
      { kind: "mound", center_m: { x: -1.1, z: 0.72 }, radius_m: { x: 0.75, z: 0.6 }, amount_m: 0.16, flat: 0.2 },
      // Low berm sheltering the beach.
      { kind: "mound", center_m: { x: 1.15, z: -0.55 }, radius_m: { x: 0.65, z: 0.55 }, amount_m: 0.11, rotation_rad: 0.6, flat: 0.25 },
      // Rockery hump at the waterline between pool and lobe.
      { kind: "mound", center_m: { x: 0.05, z: 0.66 }, radius_m: { x: 0.3, z: 0.24 }, amount_m: 0.07, flat: 0.2 }
    ]
  };
}

/**
 * Rest a body on the ground: keep its authored height above the container
 * floor as clearance and add the local lawn height underneath it, so a stack
 * built for the flat tank stands the same way on the grass.
 */
function settleBodiesOnTerrain(scene: SceneDescription): void {
  if (!scene.terrain) return;
  for (const body of scene.rigidBodies) {
    body.position_m.y += terrainHeightAt(scene.terrain, body.position_m.x, body.position_m.z);
  }
}

/**
 * General adapter: re-site any tank scene into the garden pond. Container and
 * terrain are replaced; the scene's own initial condition, inflow, timestep
 * and bodies are preserved (bodies are settled onto the lawn). Works for dam
 * break, settled fill, hose fill, jet-past-obstacle — anything expressed in
 * the shared scene contract.
 */
export function applyGardenPool(source: SceneDescription, options: { fillFraction?: number } = {}): SceneDescription {
  const scene = cloneScene(source);
  scene.sceneId = `${scene.sceneId}-garden-pool`;
  scene.container.width_m = GARDEN_CONTAINER.width_m;
  scene.container.height_m = GARDEN_CONTAINER.height_m;
  scene.container.depth_m = GARDEN_CONTAINER.depth_m;
  scene.terrain = gardenPoolTerrain();
  scene.container.fillFraction = options.fillFraction ?? GARDEN_WATERLINE_M / GARDEN_CONTAINER.height_m;
  const inflow = scene.fluid.inflow;
  if (inflow) {
    // Keep an authored jet clear of the ground: the hose mouth sits above the
    // local lawn by at least its own radius.
    const ground = terrainHeightAt(scene.terrain, inflow.center_m.x, inflow.center_m.z);
    inflow.center_m.y = Math.max(inflow.center_m.y, ground + 2 * inflow.radius_m);
  }
  settleBodiesOnTerrain(scene);
  return scene;
}
