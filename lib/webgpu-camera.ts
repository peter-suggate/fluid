import { add, cameraBasis, dot, normalize, scale, sub } from "./math";
import type { CameraState, Vec3 } from "./model";

/**
 * Perspective scale shared by every WebGPU presentation path.
 *
 * This is tan(verticalFieldOfView / 2), not an angle in radians. Keeping the
 * semantic in the name prevents raster and analytic-ray cameras diverging.
 */
export const CAMERA_TAN_HALF_FOV = 0.72;

export interface ViewportRay {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface ViewportProjection {
  readonly leftFraction: number;
  readonly topFraction: number;
  readonly visible: boolean;
  /** Signed distance along the view direction; negative is behind the camera. */
  readonly depth_m: number;
}

/** Guarded width/height so a not-yet-measured canvas cannot produce NaN rays. */
export function viewportAspect(width: number, height: number): number {
  return Math.max(width, 1) / Math.max(height, 1);
}

/**
 * The single host-side inverse of the WGSL camera. Every picking, gizmo, and
 * drop-target ray goes through here so the analytic rays cannot drift from the
 * shaders that draw the pixels being clicked.
 *
 * Normalized device coordinates run -1..1 with +x right and +y up.
 */
export function viewportRay(camera: CameraState, ndcX: number, ndcY: number, aspect: number): ViewportRay {
  const basis = cameraBasis(camera);
  return {
    origin: basis.position,
    direction: normalize(add(basis.forward, add(
      scale(basis.right, ndcX * aspect * CAMERA_TAN_HALF_FOV),
      scale(basis.up, ndcY * CAMERA_TAN_HALF_FOV),
    ))),
  };
}

/** Viewport ray for a pointer event measured against the canvas bounding rect. */
export function viewportRayForPointer(
  camera: CameraState,
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
): ViewportRay {
  const ndcX = ((clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / Math.max(rect.height, 1)) * 2;
  return viewportRay(camera, ndcX, ndcY, viewportAspect(rect.width, rect.height));
}

/** Forward transform of {@link viewportRay}: world point to canvas fractions. */
export function projectToViewport(
  position_m: Vec3,
  camera: CameraState,
  width: number,
  height: number,
): ViewportProjection {
  const basis = cameraBasis(camera);
  const relative = sub(position_m, basis.position);
  const depth_m = dot(relative, basis.forward);
  const aspect = viewportAspect(width, height);
  const inFront = depth_m > 1e-6;
  const ndcX = inFront ? dot(relative, basis.right) / (depth_m * aspect * CAMERA_TAN_HALF_FOV) : Number.POSITIVE_INFINITY;
  const ndcY = inFront ? dot(relative, basis.up) / (depth_m * CAMERA_TAN_HALF_FOV) : Number.POSITIVE_INFINITY;
  return {
    leftFraction: 0.5 * (ndcX + 1),
    topFraction: 0.5 * (1 - ndcY),
    visible: inFront && Math.abs(ndcX) <= 1 && Math.abs(ndcY) <= 1,
    depth_m,
  };
}
