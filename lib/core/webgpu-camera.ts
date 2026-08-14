import { add, cameraBasis, dot, normalize, scale, sub } from "./math";
import type { CameraState, Vec3 } from "./model";

/**
 * Perspective scale shared by every WebGPU presentation path.
 *
 * This is tan(verticalFieldOfView / 2), not an angle in radians. Keeping the
 * semantic in the name prevents raster and analytic-ray cameras diverging.
 *
 * It is also the aperture a camera gets when the document does not author one,
 * which is why it stays 0.72 rather than moving to something photographic: this
 * number reproduces every frame ever fingerprinted against this renderer. A
 * scene that wants a longer lens says so in `CameraState.tanHalfFov`.
 */
export const CAMERA_TAN_HALF_FOV = 0.72;

/**
 * Bounds on an authored aperture.
 *
 * The floor is not a taste judgement — below it the primary rays degenerate
 * towards parallel and the reversed-Z near plane stops separating anything —
 * and the ceiling is where the corner ray passes 89 degrees off axis and the
 * frame inverts. Out-of-range values clamp rather than throw: this resolves on
 * the render path for a document a user may be dragging a slider in.
 */
export const CAMERA_TAN_HALF_FOV_RANGE = Object.freeze({ minimum: 0.02, maximum: 8 });

/**
 * The aperture this camera actually renders at.
 *
 * Absent means {@link CAMERA_TAN_HALF_FOV}. That fallback is the whole reason
 * the field is optional: every existing scene document, editor fixture and
 * golden fingerprint predates the field, and all of them must keep producing
 * the frame they produced before.
 */
export function cameraTanHalfFov(camera: Pick<CameraState, "tanHalfFov">): number {
  const authored = camera.tanHalfFov;
  if (authored === undefined || !Number.isFinite(authored)) return CAMERA_TAN_HALF_FOV;
  return Math.min(CAMERA_TAN_HALF_FOV_RANGE.maximum, Math.max(CAMERA_TAN_HALF_FOV_RANGE.minimum, authored));
}

/**
 * The aperture a 35 mm-format focal length would give.
 *
 * Photographic focal lengths are the only vocabulary anyone has for "how wide
 * is this lens", and they are quoted against the 36 x 24 mm frame, so the
 * vertical half-angle is `atan(12 / f)`. A 50 mm therefore lands on 0.24 and
 * the renderer's historical 0.72 is a 16.7 mm — an ultra-wide, which is exactly
 * what the hero garden's near coping was bowing against.
 */
export function tanHalfFovFor35mmFocalLength(focalLength_mm: number): number {
  return 12 / Math.max(1, focalLength_mm);
}

/**
 * Lane of the shared view uniform block that carries the aperture.
 *
 * `cameraPosition` has always been a `vec4f` whose `w` was written as zero
 * padding, and every shader that binds this block reads only `.xyz`, so the
 * lane was free. Threading the aperture there instead of growing the block
 * means no bind-group layout, no struct declaration and no host packing offset
 * moves — the block is declared verbatim in a dozen shader modules and one
 * added field would have to be edited into all of them in lockstep.
 *
 * The same lane already carries the same quantity in the decoration overlay's
 * private block, so the convention is not new here.
 */
export const CAMERA_APERTURE_UNIFORM_LANE = 7;

/**
 * WGSL accessor for the aperture lane, to be pasted into any module that binds
 * the shared view uniform block.
 *
 * Zero falls back to {@link CAMERA_TAN_HALF_FOV} rather than clamping to the
 * range floor. That matters: benchmark harnesses and smoke executors write this
 * block themselves and none of them know about the lane, so a fallback to the
 * historical lens keeps them rendering what they rendered, where a clamp to
 * 0.02 would collapse every one of their frames to a pinhole.
 */
export function cameraApertureShaderLibrary(uniformName = "uniforms"): string {
  return /* wgsl */ `
fn cameraTanHalfFov()->f32{let authored=${uniformName}.cameraPosition.w;return select(${CAMERA_TAN_HALF_FOV},authored,authored>0.0);}
`;
}

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
  const basis = cameraBasis(camera), tanHalfFov = cameraTanHalfFov(camera);
  return {
    origin: basis.position,
    direction: normalize(add(basis.forward, add(
      scale(basis.right, ndcX * aspect * tanHalfFov),
      scale(basis.up, ndcY * tanHalfFov),
    ))),
  };
}

/**
 * The ray behind one rendered pixel, at its centre.
 *
 * The `+0.5` and the aspect are the render target's, so this is the exact
 * inverse of what a fragment or gather shader builds for that same pixel. That
 * exactness is what makes a frozen aim possible: a selection recorded as a
 * pixel silently names a different cell the moment the camera moves, and one
 * recorded as a world ray does not.
 */
export function viewportRayForPixel(
  camera: CameraState, pixelX: number, pixelY: number, width: number, height: number,
): ViewportRay {
  const across = Math.max(width, 1), down = Math.max(height, 1);
  return viewportRay(
    camera,
    ((pixelX + 0.5) / across) * 2 - 1,
    1 - ((pixelY + 0.5) / down) * 2,
    viewportAspect(across, down),
  );
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
  const inFront = depth_m > 1e-6, tanHalfFov = cameraTanHalfFov(camera);
  const ndcX = inFront ? dot(relative, basis.right) / (depth_m * aspect * tanHalfFov) : Number.POSITIVE_INFINITY;
  const ndcY = inFront ? dot(relative, basis.up) / (depth_m * tanHalfFov) : Number.POSITIVE_INFINITY;
  return {
    leftFraction: 0.5 * (ndcX + 1),
    topFraction: 0.5 * (1 - ndcY),
    visible: inFront && Math.abs(ndcX) <= 1 && Math.abs(ndcY) <= 1,
    depth_m,
  };
}
