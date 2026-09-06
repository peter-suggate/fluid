import { boxCenter, intersectBox, sceneContainerBox, type EditorEntityContext, type EditorRay } from "./editor-entity";
import { targetAtRay } from "./editor-probe-catalog";
import { add, dot, scale, sub } from "./math";
import type { CameraState, SceneDescription, Vec3 } from "./model";
import { viewportRay, type ViewportRay } from "./webgpu-camera";

/**
 * The ray out of the middle of the viewport.
 *
 * Degenerate on purpose: at the centre of the frame both screen offsets are
 * zero, so this is the camera's forward axis and the aspect ratio and the
 * aperture both cancel. It goes through {@link viewportRay} anyway rather than
 * being written as `cameraBasis(camera).forward`, because the one host-side
 * inverse of the WGSL camera is the thing that guarantees the pixel this picks
 * out is the pixel the shader draws in the middle.
 */
export function viewportCentreRay(camera: CameraState): ViewportRay {
  return viewportRay(camera, 0, 0, 1);
}

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
 * What the camera should turn and zoom about: whatever the viewport is centred
 * on.
 *
 * Resolved through the same probe catalog as the hover chip, so the pivot and
 * the interface agree about what "that" is — orbit about the thing the reader
 * would be told they are pointing at, never about a second, private answer.
 *
 * The two targets that are *not* good pivots are handled here rather than in
 * the catalog, because they are only bad for this purpose:
 *
 * - **The tank wall.** `pickRoomExitFace` deliberately answers with the wall a
 *   ray *leaves* through, since that is the one you can see from inside. It is
 *   also the furthest surface in the frame, and orbiting about the back wall
 *   swings everything in front of it — including the water, which is what the
 *   reader was actually looking at and which no CPU probe can pick.
 * - **The room.** A ray past the container lands on the ground plane or, aimed
 *   at the sky, a fixed way along itself. Fine as a placement point, arbitrary
 *   as a centre of rotation.
 *
 * Both fall back to {@link containerViewPoint}, which is the same rule
 * `containerPlacementPoint` uses to choose a depth for a click that met
 * nothing: the point of the ray's span through the container that comes
 * nearest the middle of the tank. Aimed into the tank it gives the middle of
 * the tank, and aimed along a corner it gives that corner — so an empty tank
 * orbits about its own volume, which is where the water is.
 */
export function viewportCentrePivot(context: EditorEntityContext, ray: EditorRay): Vec3 {
  const target = targetAtRay(context, ray);
  if (target.kind !== "tank-wall" && target.kind !== "room") return target.point_m;
  return containerViewPoint(context.scene, ray) ?? target.point_m;
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
