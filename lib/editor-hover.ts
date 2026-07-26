import { add, dot, scale, sub } from "./math";
import type { SceneDescription, Vec3 } from "./model";
import { boundingRadius, type RigidBodyState } from "./rigid-body";
import { sceneHasTerrain, terrainNormalAt } from "./terrain";
import { intersectSvoTerrainHeightfield } from "./webgpu-svo-dry-scene";

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

export type EditorHoverKind = "body" | "terrain" | "floor";

export interface EditorHover {
  readonly kind: EditorHoverKind;
  readonly position_m: Vec3;
  readonly normal: Vec3;
  readonly distance_m: number;
  /** Present only for body hits. */
  readonly bodyId?: string;
  readonly label: string;
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

function hoverTerrain(scene: SceneDescription, ray: { origin: Vec3; direction: Vec3 }): EditorHover | undefined {
  if (!sceneHasTerrain(scene)) return undefined;
  const c = scene.container;
  const hit = intersectSvoTerrainHeightfield(
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

/** Nearest analytic hit under the pointer across bodies, terrain, and the floor. */
export function hoverSceneAt(
  scene: SceneDescription,
  bodies: readonly RigidBodyState[],
  ray: { origin: Vec3; direction: Vec3 },
): EditorHover | undefined {
  const candidates = [hoverBody(bodies, ray), hoverTerrain(scene, ray), hoverFloor(scene, ray)];
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
  const normal = hover.kind === "body" ? hover.normal : terrainSurfaceNormal(scene, hover);
  const placed = add(hover.position_m, scale(normal, radius_m));
  return {
    x: Math.min(c.width_m / 2 - radius_m, Math.max(-c.width_m / 2 + radius_m, placed.x)),
    y: Math.min(c.height_m + 0.8, Math.max(radius_m, placed.y)),
    z: Math.min(c.depth_m / 2 - radius_m, Math.max(-c.depth_m / 2 + radius_m, placed.z)),
  };
}

/** Re-evaluated rather than reused so a floor hit still rests on real ground. */
function terrainSurfaceNormal(scene: SceneDescription, hover: EditorHover): Vec3 {
  if (!sceneHasTerrain(scene)) return { x: 0, y: 1, z: 0 };
  return terrainNormalAt(scene.terrain, hover.position_m.x, hover.position_m.z);
}
