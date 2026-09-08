import { boxCenter, intersectBox, sceneContainerBox, type EditorEntityContext, type EditorRay } from "./editor-entity";
import { targetAtRay } from "./editor-probe-catalog";
import { add, CAMERA_DISTANCE_RANGE, cameraBasis, dot, length, scale, sub } from "./math";
import type { CameraState, SceneDescription, Vec3 } from "./model";

/**
 * How long a run of wheel events counts as one gesture.
 *
 * A trackpad flick arrives as dozens of events a few milliseconds apart and a
 * mouse notch as one, so the gap is what tells "still zooming" from "zooming
 * again" — and the pivot is held for the whole of the former, so a continuous
 * scroll is one approach to one point rather than a pivot that steps between
 * surfaces as the perspective changes under it.
 */
export const WHEEL_GESTURE_GAP_MS = 250;

/**
 * How far the cursor may wander during a wheel burst before the burst is over.
 *
 * The held pivot is the point that stays under the cursor, so a cursor that
 * has moved on is aiming at something else; a jitter of a few pixels is not.
 */
export const WHEEL_GESTURE_TRAVEL_PX = 8;

/**
 * What a camera gesture turns or zooms about: whatever is under the cursor
 * where the gesture began.
 *
 * This is the rule SketchUp, Onshape, Fusion 360 and Unity's scene view share
 * — orbit about the thing you grabbed, zoom into the thing you point at — and
 * it is resolved through the same probe catalog as the hover chip, so the
 * pivot is the thing the reader has just been told they are pointing at.
 *
 * Two answers that catalog gives are *not* good pivots, and are handled here
 * because they are only bad for this purpose:
 *
 * - **The tank wall.** `pickRoomExitFace` deliberately answers with the wall a
 *   ray *leaves* through, since that is the one you can see from inside. It is
 *   also the furthest surface in the frame, and turning about the back wall
 *   swings everything in front of it — including the water, which is what the
 *   reader was actually pointing at and which no CPU probe can pick.
 * - **The room.** A ray past the container lands on the ground plane or, aimed
 *   at the sky, a fixed way along itself. Fine as a placement point, arbitrary
 *   as a centre of rotation.
 *
 * Both fall back to {@link containerViewPoint} — the point of the ray's span
 * through the tank nearest its middle, so a press on the water pivots on the
 * water — and a ray that misses the tank altogether falls back to the point on
 * the cursor's ray at the depth the camera is already looking at, which is the
 * fallback Blender and three.js use for a cursor over nothing: still under the
 * cursor, and at the depth of the last thing the reader was working on, so the
 * gesture neither jumps nor turns about the horizon. A surface outside the
 * wheel's reach (see `CAMERA_DISTANCE_RANGE`) takes that same fallback.
 */
export function gesturePivot(context: EditorEntityContext, camera: CameraState, ray: EditorRay): Vec3 {
  const target = targetAtRay(context, ray);
  const surface = target.kind !== "tank-wall" && target.kind !== "room"
    ? target.point_m
    : containerViewPoint(context.scene, ray);
  if (surface) {
    const reach_m = length(sub(surface, ray.origin));
    if (reach_m >= CAMERA_DISTANCE_RANGE.minimum_m && reach_m <= CAMERA_DISTANCE_RANGE.maximum_m) return surface;
  }
  return viewDepthPoint(camera, ray);
}

/**
 * The point on a ray at the view depth of the camera's look-at target — the
 * pivot for a cursor over nothing.
 */
export function viewDepthPoint(camera: CameraState, ray: EditorRay): Vec3 {
  const along = dot(ray.direction, cameraBasis(camera).forward);
  return add(ray.origin, scale(ray.direction, camera.distance_m / Math.max(along, 1e-6)));
}

/**
 * The point on a ray's traversal of the container that lies nearest the middle
 * of it, or nothing when the ray misses the container altogether.
 *
 * The projection is clamped to the span rather than to the whole ray, so the
 * answer is always somewhere a viewer would call "in the tank", and `near` is
 * floored at zero so a camera already inside the container measures from
 * itself rather than from a crossing behind it.
 */
export function containerViewPoint(scene: SceneDescription, ray: EditorRay): Vec3 | undefined {
  const box = sceneContainerBox(scene);
  const span = intersectBox(ray, box);
  if (!span) return undefined;
  const near = Math.max(span.near_m, 0);
  if (span.far_m < near) return undefined;
  const toward = dot(sub(boxCenter(box), ray.origin), ray.direction);
  return add(ray.origin, scale(ray.direction, Math.min(span.far_m, Math.max(near, toward))));
}
