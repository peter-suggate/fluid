import { containerContains, containerPlacementPoint } from "./editor-entity";
import { sceneryEntity, sceneryIdFromSelection } from "./editor-scenery";
import { sceneCellSizes_m } from "./scene-lattice";
import { add, dot, scale, sub } from "./math";
import type { SceneDescription, Vec3 } from "./model";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import { intersectAuthoredTerrain, sceneHasTerrain, terrainNormalAt } from "./terrain";

/**
 * What the editor cursor is over.
 *
 * Deliberately analytic and CPU-side. Hover updates on every pointermove, and
 * the GPU G-buffer pick is a fenced 1x1 readback whose latency and queue
 * pressure make it the wrong instrument for a cursor — it stays the authority
 * for click-to-select, where one round trip per click is fine.
 *
 * Terrain is a first-class hit here. `svoPickingInteractionForHit` maps
 * terrain to `{kind:"none"}` because only rigid bodies are draggable; the
 * editor needs the ground itself as a target for placement and, later, for
 * the terrain brush.
 */

export type EditorHoverKind = "body" | "fluid" | "scenery" | "terrain" | "floor";

export interface EditorHover {
  readonly kind: EditorHoverKind;
  readonly position_m: Vec3;
  readonly normal: Vec3;
  readonly distance_m: number;
  /** Present only for body hits. */
  readonly bodyId?: string;
  /** Present only for scenery hits: the described object, not the primitive. */
  readonly sceneryNodeId?: string;
  readonly label: string;
}

/**
 * Scenery, against the exact geometry the SVO traces.
 *
 * The same pick the click path uses, so the rim the cursor lights up and the
 * object a click selects can never be two different things.
 */
function hoverScenery(scene: SceneDescription, ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  const hit = sceneryEntity.pick?.({ scene, bodies: [] }, ray);
  const nodeId = hit && sceneryIdFromSelection(hit.selection.id);
  if (!hit || !nodeId) return undefined;
  return {
    kind: "scenery",
    position_m: add(ray.origin, scale(ray.direction, hit.distance_m)),
    // Scenery is not a drag target for placement, so the surface normal is not
    // wanted; up is the one answer that cannot mislead a caller that reads it.
    normal: { x: 0, y: 1, z: 0 },
    distance_m: hit.distance_m,
    sceneryNodeId: nodeId,
    label: nodeId,
  };
}

/** Bodies use their bounding sphere — exact for spheres, close enough for a cursor. */
function hoverBody(bodies: readonly RigidBodyState[], ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  let nearest: EditorHover | undefined;
  for (const body of bodies) {
    const offset = sub(ray.origin, body.position_m);
    const radius = boundingRadius(body);
    const halfB = dot(offset, ray.direction);
    const discriminant = halfB * halfB - (dot(offset, offset) - radius * radius);
    if (discriminant < 0) continue;
    const distance_m = -halfB - Math.sqrt(discriminant);
    if (!(distance_m > 0) || (nearest && distance_m >= nearest.distance_m)) continue;
    const position_m = add(ray.origin, scale(ray.direction, distance_m));
    nearest = {
      kind: "body",
      position_m,
      normal: scale(sub(position_m, body.position_m), 1 / Math.max(radius, 1e-6)),
      distance_m,
      bodyId: body.description.id,
      label: body.description.name,
    };
  }
  return nearest;
}

/** Authored water is a placement surface too: a dropped ball sits on the pool. */
function hoverFluid(scene: SceneDescription, ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  if (scene.systems?.fluid === false || scene.fluid.initialCondition !== "tank-fill"
    || (scene.fluid.initialBrickSeeds_m?.length
      && scene.fluid.initialBrickSeedsAdditive !== true)
    || !(ray.direction.y < -1e-6)) return undefined;
  const surfaceY = scene.container.fillFraction * scene.container.height_m;
  if (!(surfaceY > 0)) return undefined;
  const distance_m = (surfaceY - ray.origin.y) / ray.direction.y;
  if (!(distance_m > 0)) return undefined;
  const position_m = add(ray.origin, scale(ray.direction, distance_m));
  if (Math.abs(position_m.x) > scene.container.width_m / 2
    || Math.abs(position_m.z) > scene.container.depth_m / 2) return undefined;
  return {
    kind: "fluid",
    position_m,
    // Water is not a rigid support with a meaningful side normal. A drop rests
    // above the surface it is entering, which also keeps its visible volume
    // out of the pool until the simulation advances it.
    normal: { x: 0, y: 1, z: 0 },
    distance_m,
    label: "water",
  };
}

function hoverTerrain(scene: SceneDescription, ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  if (!sceneHasTerrain(scene)) return undefined;
  const c = scene.container;
  const hit = intersectAuthoredTerrain(
    scene.terrain, ray.origin, ray.direction, Math.max(c.width_m, c.height_m, c.depth_m),
  );
  if (!hit || !(hit.t_m > 0)) return undefined;
  return { kind: "terrain", position_m: hit.position_m, normal: hit.normal, distance_m: hit.t_m, label: "terrain" };
}

/** The container floor, so a click always lands somewhere inside the tank. */
function hoverFloor(scene: SceneDescription, ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  if (!(ray.direction.y < -1e-6)) return undefined;
  const distance_m = -ray.origin.y / ray.direction.y;
  if (!(distance_m > 0)) return undefined;
  const position_m = add(ray.origin, scale(ray.direction, distance_m));
  const c = scene.container;
  if (Math.abs(position_m.x) > c.width_m / 2 || Math.abs(position_m.z) > c.depth_m / 2) return undefined;
  return { kind: "floor", position_m, normal: { x: 0, y: 1, z: 0 }, distance_m, label: "container floor" };
}

export interface EditorHoverOptions {
  /**
   * Whether the ray is tested against scenery.
   *
   * On by default, because a tool that places something needs to know it is
   * resting it on a bench rather than through one. The viewport's own cursor
   * turns it off outside the select tool: there, scenery is not a click target,
   * and answering with it would cost every pointer-move a pick against the whole
   * set — and light a rim promising a selection the armed tool will not make.
   */
  readonly scenery?: boolean;
}

/** Nearest analytic hit under the pointer across bodies, water, terrain, and the floor. */
export function hoverSceneAt(
  scene: SceneDescription,
  bodies: readonly RigidBodyState[],
  ray: { origin: Vec3; direction: Vec3 },
  options: EditorHoverOptions = {},
): EditorHover | undefined {
  const candidates = [
    hoverBody(bodies, ray),
    hoverFluid(scene, ray),
    options.scenery === false ? undefined : hoverScenery(scene, ray),
    hoverTerrain(scene, ray),
    hoverFloor(scene, ray),
  ];
  let nearest: EditorHover | undefined;
  for (const candidate of candidates) {
    if (candidate && (!nearest || candidate.distance_m < nearest.distance_m)) nearest = candidate;
  }
  return nearest;
}

/**
 * Rest a body of the given radius on a hovered surface: offset along the
 * surface normal so it touches rather than intersects, and keep it inside the
 * container so the drop is always a legal scene.
 */
export function restOnHover(hover: EditorHover, radius_m: number, scene: SceneDescription): Vec3 {
  const c = scene.container;
  const normal = hover.kind === "body" || hover.kind === "fluid" ? hover.normal
    : hover.kind === "terrain" ? terrainSurfaceNormal(scene, hover)
      : { x: 0, y: 1, z: 0 };
  // A liquid ball touching the pool is already one connected implicit body;
  // Sparse CM12 correctly unions it into a small cap before it can look like a
  // ball. Leave one finest cell of air so the paused frame shows the drop and
  // the running frame has room to let it fall.
  const airGap_m = hover.kind === "fluid" ? Math.min(...sceneCellSizes_m(scene)) : 0;
  const placed = add(hover.position_m, scale(normal, radius_m + airGap_m));
  return {
    x: Math.min(c.width_m / 2 - radius_m, Math.max(-c.width_m / 2 + radius_m, placed.x)),
    y: Math.min(c.height_m + 0.8, Math.max(radius_m, placed.y)),
    z: Math.min(c.depth_m / 2 - radius_m, Math.max(-c.depth_m / 2 + radius_m, placed.z)),
  };
}

/**
 * Rest a liquid interaction in the open sparse world.
 *
 * Unlike authored tank contents and rigid-body placement, a live drop is not
 * constrained by the vessel after the ray has missed it. Rays addressing the
 * tank retain its established promoted placement; otherwise an actual surface
 * hit anywhere is authoritative. There is deliberately no implicit world
 * floor: unsupported fluid falls through open SparseWorld until it reaches
 * authored solid voxels or planes.
 */
export function restFluidInWorld(
  scene: SceneDescription,
  ray: { origin: Vec3; direction: Vec3 },
  hover: EditorHover | undefined,
  radius_m: number,
): Vec3 | undefined {
  // The tank is a promoted drop target. In particular, a ray through its open
  // upper volume may eventually hit stage geometry outside it; the established
  // container path deliberately resolves that ray inside the vessel instead.
  // Open-world placement is only the fallback for a genuine tank miss.
  const tankPlacement = restInContainer(scene, ray, hover, radius_m);
  if (tankPlacement) return tankPlacement;
  const surface = hover;
  if (!surface) return undefined;
  const normal = surface.kind === "body" || surface.kind === "fluid" ? surface.normal
    : surface.kind === "terrain" ? terrainSurfaceNormal(scene, surface)
      : { x: 0, y: 1, z: 0 };
  const airGap_m = surface.kind === "fluid" ? Math.min(...sceneCellSizes_m(scene)) : 0;
  const placed = add(surface.position_m, scale(normal, radius_m + airGap_m));
  return placed;
}

/**
 * Where something of this size, put down at this pixel, comes to rest *in the
 * tank* — or nothing, when the pixel is not aimed at the tank at all.
 *
 * The one answer both the preview circle and the press use, for every tool that
 * puts an object into the simulation. Two rules, in this order:
 *
 * A surface inside the container wins, and the object rests on it: dropping on
 * the floor, on a prop or on a body means what it looks like. A surface hit
 * *outside* the container is not one of those — since the house set became a
 * stage the ray leaving the top of the tank lands on the stage floor metres
 * away, and resting on that put every such placement outside the simulation,
 * where it was refused. Out there the ray is carrying no depth information
 * worth having, so the container supplies it instead; see
 * `containerPlacementPoint`, which is what makes the upper volume and the top
 * corners reachable again.
 */
export function restInContainer(
  scene: SceneDescription,
  ray: { origin: Vec3; direction: Vec3 },
  hover: EditorHover | undefined,
  radius_m: number,
): Vec3 | undefined {
  if (hover && containerContains(scene, hover.position_m)) return restOnHover(hover, radius_m, scene);
  return containerPlacementPoint(scene, ray, radius_m);
}

/** Re-evaluated rather than reused so a floor hit still rests on real ground. */
function terrainSurfaceNormal(scene: SceneDescription, hover: EditorHover): Vec3 {
  if (!sceneHasTerrain(scene)) return { x: 0, y: 1, z: 0 };
  return terrainNormalAt(scene.terrain, hover.position_m.x, hover.position_m.z);
}
