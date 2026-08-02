import { add, dot, normalize, scale, sub } from "./math";
import type { Vec3 } from "./model";
import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";

/**
 * The axis maths every editor drag rides, kept free of DOM and WebGPU so it can
 * be tested directly.
 *
 * The gizmo itself is no longer a thing of its own: a move handle is an ordinary
 * `EditorHandle` like any other, built by `moveHandles` in editor-entity.ts and
 * drawn by the same overlay as the box handles. What survives here is what that
 * is made of — the axis directions, the on-screen arm length, and the
 * closest-approach that turns a pointer ray into a point on an axis.
 */

export type GizmoAxis = "x" | "y" | "z";

export const GIZMO_AXES: readonly GizmoAxis[] = Object.freeze(["x", "y", "z"]);

export const GIZMO_AXIS_DIRECTIONS: Readonly<Record<GizmoAxis, Vec3>> = Object.freeze({
  x: { x: 1, y: 0, z: 0 },
  y: { x: 0, y: 1, z: 0 },
  z: { x: 0, y: 0, z: 1 },
});

/** Fraction of the viewport half-height each axis arm spans. */
export const GIZMO_HANDLE_SCREEN_FRACTION = 0.11;

/**
 * Arm length that holds a constant on-screen size: the projected size of a world
 * length is proportional to its eye depth, so scaling the length by the depth
 * cancels the perspective divide.
 */
export function gizmoHandleLength_m(
  depth_m: number,
  screenFraction = GIZMO_HANDLE_SCREEN_FRACTION,
): number {
  return Math.max(depth_m, 1e-3) * CAMERA_TAN_HALF_FOV * screenFraction;
}

/**
 * Point on the infinite axis line closest to the pointer ray — the standard
 * closest-approach of two lines. Returns undefined when the ray is nearly
 * parallel to the axis, where the constraint is singular and a drag would
 * shoot the body to infinity.
 */
export function closestPointOnAxis(
  rayOrigin: Vec3,
  rayDirection: Vec3,
  axisPoint: Vec3,
  axisDirection: Vec3,
  parallelEpsilon = 2e-3,
): Vec3 | undefined {
  const axis = normalize(axisDirection);
  const ray = normalize(rayDirection);
  const alignment = dot(axis, ray);
  const denominator = 1 - alignment * alignment;
  if (!(denominator > parallelEpsilon)) return undefined;
  const offset = sub(axisPoint, rayOrigin);
  const alongAxis = dot(axis, offset);
  const alongRay = dot(ray, offset);
  return add(axisPoint, scale(axis, (alignment * alongRay - alongAxis) / denominator));
}
