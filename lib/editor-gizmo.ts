import { add, dot, normalize, scale, sub } from "./math";
import type { CameraState, Vec3 } from "./model";
import { CAMERA_TAN_HALF_FOV, projectToViewport, type ViewportProjection } from "./webgpu-camera";

/**
 * Translate-gizmo geometry, kept free of DOM and WebGPU so the drag maths can
 * be tested directly. The gizmo itself is drawn as a DOM overlay in v1; only
 * these projections and the axis constraint decide where a drag lands.
 */

export type GizmoAxis = "x" | "y" | "z";
/** The centre handle drags in the camera plane, matching the free body drag. */
export type GizmoHandle = GizmoAxis | "free";

export const GIZMO_AXES: readonly GizmoAxis[] = Object.freeze(["x", "y", "z"]);

export const GIZMO_AXIS_DIRECTIONS: Readonly<Record<GizmoAxis, Vec3>> = Object.freeze({
  x: { x: 1, y: 0, z: 0 },
  y: { x: 0, y: 1, z: 0 },
  z: { x: 0, y: 0, z: 1 },
});

/** Fraction of the viewport half-height each axis handle spans. */
export const GIZMO_HANDLE_SCREEN_FRACTION = 0.11;
export const GIZMO_AXIS_PICK_TOLERANCE_PX = 10;
export const GIZMO_CENTER_PICK_TOLERANCE_PX = 9;

export interface GizmoHandleProjection {
  readonly axis: GizmoAxis;
  readonly tip_m: Vec3;
  readonly tip: ViewportProjection;
}

export interface ProjectedGizmo {
  readonly position_m: Vec3;
  readonly origin: ViewportProjection;
  readonly handles: readonly GizmoHandleProjection[];
  /** World length of each handle, chosen to keep a constant on-screen size. */
  readonly length_m: number;
}

/**
 * Handle length that holds a constant on-screen size: the projected size of a
 * world length is proportional to its eye depth, so scaling the length by the
 * depth cancels the perspective divide.
 */
export function gizmoHandleLength_m(
  depth_m: number,
  screenFraction = GIZMO_HANDLE_SCREEN_FRACTION,
): number {
  return Math.max(depth_m, 1e-3) * CAMERA_TAN_HALF_FOV * screenFraction;
}

export function projectGizmo(
  position_m: Vec3,
  camera: CameraState,
  width: number,
  height: number,
  screenFraction = GIZMO_HANDLE_SCREEN_FRACTION,
): ProjectedGizmo {
  const origin = projectToViewport(position_m, camera, width, height);
  const length_m = gizmoHandleLength_m(origin.depth_m, screenFraction);
  const handles = GIZMO_AXES.map((axis) => {
    const tip_m = add(position_m, scale(GIZMO_AXIS_DIRECTIONS[axis], length_m));
    return { axis, tip_m, tip: projectToViewport(tip_m, camera, width, height) };
  });
  return { position_m, origin, handles, length_m };
}

/** Canvas-pixel position of a projection, for hit tests and overlay layout. */
function pixels(projection: ViewportProjection, width: number, height: number) {
  return { x: projection.leftFraction * width, y: projection.topFraction * height };
}

function distanceToSegment_px(
  point: { x: number; y: number },
  from: { x: number; y: number },
  to: { x: number; y: number },
): number {
  const dx = to.x - from.x, dy = to.y - from.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-9) return Math.hypot(point.x - from.x, point.y - from.y);
  const t = Math.max(0, Math.min(1, ((point.x - from.x) * dx + (point.y - from.y) * dy) / lengthSquared));
  return Math.hypot(point.x - (from.x + t * dx), point.y - (from.y + t * dy));
}

/**
 * Which handle a pointer grabs, in canvas pixels. The centre wins ties so a
 * click on the shared origin is a free drag rather than an arbitrary axis.
 * Handles whose origin or tip is behind the camera are not grabbable.
 */
export function gizmoHandleAtPointer(
  gizmo: ProjectedGizmo,
  pointer: { x: number; y: number },
  width: number,
  height: number,
  axisTolerance_px = GIZMO_AXIS_PICK_TOLERANCE_PX,
  centerTolerance_px = GIZMO_CENTER_PICK_TOLERANCE_PX,
): GizmoHandle | undefined {
  if (!(gizmo.origin.depth_m > 1e-6)) return undefined;
  const origin = pixels(gizmo.origin, width, height);
  if (Math.hypot(pointer.x - origin.x, pointer.y - origin.y) <= centerTolerance_px) return "free";
  let best: { axis: GizmoAxis; distance_px: number } | undefined;
  for (const handle of gizmo.handles) {
    if (!(handle.tip.depth_m > 1e-6)) continue;
    const distance_px = distanceToSegment_px(pointer, origin, pixels(handle.tip, width, height));
    if (distance_px <= axisTolerance_px && (!best || distance_px < best.distance_px)) best = { axis: handle.axis, distance_px };
  }
  return best?.axis;
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

/** Axis-constrained drag: keep the grab offset so the body does not jump. */
export function gizmoAxisDragPosition(
  rayOrigin: Vec3,
  rayDirection: Vec3,
  axis: GizmoAxis,
  axisOrigin_m: Vec3,
  grabOffset_m: Vec3,
): Vec3 | undefined {
  const point = closestPointOnAxis(rayOrigin, rayDirection, axisOrigin_m, GIZMO_AXIS_DIRECTIONS[axis]);
  return point && add(point, grabOffset_m);
}
