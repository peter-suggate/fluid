import type { EditorRay } from "../editor-entity";
import { containPlacement } from "../editor-hover";
import type { SceneDescription, Vec3 } from "../model";
import { focusOrPick, type FocusPoint, type ToolFocus } from "./focus";

const UP: Vec3 = { x: 0, y: 1, z: 0 };

/**
 * Where a placed object's centre goes (BALL, DRAG): `rest` — its resting place
 * on the surface under the ray, today's placement — when no lock is up; with
 * one, the focus on the lock, anchored at that resting centre. A rest point is
 * the focus here, so `Y` lifts the object straight up off what it rested on.
 */
export function placementCentre(focus: ToolFocus | undefined, ray: EditorRay,
  rest: (ray: EditorRay) => Vec3 | undefined): { readonly centre: Vec3; readonly locked: boolean } | undefined {
  const point = focusOrPick(focus, ray, (r): FocusPoint | undefined => {
    const centre = rest(r);
    return centre ? { point: centre, normal: UP } : undefined;
  });
  return point ? { centre: point.point, locked: point.locked === true } : undefined;
}

/** A body's centre: on the lock when one is up, but always inside the container, as a resting body is. */
export function bodyPlacementCentre(scene: SceneDescription, focus: ToolFocus | undefined, ray: EditorRay,
  radius_m: number, rest: (ray: EditorRay) => Vec3 | undefined): Vec3 | undefined {
  const placed = placementCentre(focus, ray, rest);
  if (!placed) return undefined;
  return placed.locked ? containPlacement(scene, placed.centre, radius_m) : placed.centre;
}
