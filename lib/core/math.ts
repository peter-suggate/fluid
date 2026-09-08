import type { CameraState, Vec3 } from "./model";

export const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vec3, b: Vec3): Vec3 => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
export const length = (a: Vec3): number => Math.sqrt(dot(a, a));
export const normalize = (a: Vec3): Vec3 => { const l = length(a); return l > 0 ? scale(a, 1 / l) : { x: 0, y: 0, z: 0 }; };

/** The eye's offset from the look-at target for a spherical camera's angles. */
export function sphericalOffset(azimuth_rad: number, elevation_rad: number, distance_m: number): Vec3 {
  const horizontal = distance_m * Math.cos(elevation_rad);
  return {
    x: horizontal * Math.sin(azimuth_rad),
    y: distance_m * Math.sin(elevation_rad),
    z: horizontal * Math.cos(azimuth_rad)
  };
}

export function cameraPosition(camera: CameraState): Vec3 {
  return add(camera.target_m, sphericalOffset(camera.azimuth_rad, camera.elevation_rad, camera.distance_m));
}

export function cameraBasis(camera: CameraState) {
  const position = cameraPosition(camera);
  const forward = normalize(sub(camera.target_m, position));
  const right = normalize(cross(forward, { x: 0, y: 1, z: 0 }));
  const up = normalize(cross(right, forward));
  return { position, forward, right, up };
}

/**
 * Rotate `v` about a unit `axis` by `angle` (Rodrigues), right-handed: about
 * +Y a positive angle carries +Z toward +X, which is what makes it advance
 * `azimuth_rad` by exactly `angle` in {@link cameraPosition}'s convention.
 */
export function rotateAbout(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

/** Drag sensitivity, radians per pixel, shared by both orbit axes. */
const ORBIT_RATE_PER_PIXEL = 0.007;
/** The turntable never quite reaches the poles, where `right` would vanish. */
const ELEVATION_LIMIT_RAD = 1.35;

/**
 * Turntable-orbit the camera about an arbitrary point, as a rigid motion.
 *
 * This is what SketchUp, Onshape and Fusion 360 do when a drag begins on a
 * surface: the whole camera pose — eye *and* view direction — rotates about
 * the grabbed point, so that point stays exactly where it was on screen and
 * the scene turns around it. The two turns are the turntable's: azimuth about
 * the world's vertical axis through the pivot, elevation about the camera's
 * own horizontal `right` axis through it (which is the axis of the eye's
 * elevation circle, so the forward direction's elevation advances by exactly
 * the step). Because the parameters are the same angles the spherical camera
 * already stores, a pivot on the look-at target reduces this to the plain
 * `azimuth += dx, elevation += dy` orbit bit for bit.
 *
 * The look-at target travels with the rigid motion, at its old distance ahead
 * of the eye, and nothing here re-anchors on the pivot: what a *later* gesture
 * turns about is that gesture's own business (see `camera-pivot.ts`).
 */
export function orbitAbout(camera: CameraState, pivot_m: Vec3, dx: number, dy: number): CameraState {
  const azimuth_rad = camera.azimuth_rad - dx * ORBIT_RATE_PER_PIXEL;
  const elevation_rad = Math.max(-ELEVATION_LIMIT_RAD, Math.min(ELEVATION_LIMIT_RAD, camera.elevation_rad + dy * ORBIT_RATE_PER_PIXEL));
  const turned = { ...camera, azimuth_rad, elevation_rad };
  const basis = cameraBasis(camera);
  // Elevation first, about the *current* right axis: rotating about `right`
  // by +phi lowers the forward direction, so the tilt that raises elevation by
  // the step is -step. Then the yaw, about the vertical through the pivot.
  const tilted = rotateAbout(sub(basis.position, pivot_m), basis.right, -(elevation_rad - camera.elevation_rad));
  const eye = add(pivot_m, rotateAbout(tilted, { x: 0, y: 1, z: 0 }, azimuth_rad - camera.azimuth_rad));
  // The eye is a function of the target in this model, so the target is
  // wherever puts the eye there: one distance ahead along the new view axis,
  // which is the turned angles' offset pointed the other way.
  return { ...turned, target_m: sub(eye, sphericalOffset(azimuth_rad, elevation_rad, camera.distance_m)) };
}

/** Orbit about the look-at target — the persistent-pivot special case. */
export function orbit(camera: CameraState, dx: number, dy: number): CameraState {
  return orbitAbout(camera, camera.target_m, dx, dy);
}

/**
 * How near and how far the wheel may take the camera from what it is looking at.
 *
 * Wide, because the wheel closes on whatever is under the cursor rather than
 * on a fixed authored point (see `zoomToward`): zooming in is approaching
 * *that* surface, so the floor is a distance from the thing itself and 2 cm
 * leaves it two near planes clear of the 0.01 m clip in
 * `voxelViewProjectionMatrix`. The ceiling sits inside that same matrix's
 * 100 m far plane with room for whatever stands behind the pivot, and well
 * outside the widest authored view: the ocean tank is 8 m across and the
 * largest preset opens at 15 m, which the previous 12 m ceiling could not even
 * hold still.
 *
 * `url-state.ts` admits exactly this range. A link is written from whatever the
 * wheel reached, and a bound that disagreed there would silently drop a shared
 * view back to its preset distance.
 */
export const CAMERA_DISTANCE_RANGE = Object.freeze({ minimum_m: 0.02, maximum_m: 60 });

/**
 * Distance is exponential in wheel travel, so a notch is a fixed *fraction* of
 * however far away the camera already is. That is what lets one range serve a
 * cup and an ocean: 100 px — one mouse notch — is a tenth of the distance
 * whether that distance is a metre or fifty.
 */
const ZOOM_RATE_PER_PIXEL = 0.001;

/**
 * Dolly toward (or away from) a point, keeping it fixed on screen.
 *
 * The "zoom to cursor" of Unity, SketchUp, Fusion 360, Houdini and three.js:
 * the view direction does not change, and the eye slides along its line to
 * the pivot so that everything on that line — the thing under the cursor —
 * stays under the cursor while the rest of the frame scales about it. As a
 * similarity centred on the pivot it carries the look-at target too, which is
 * what makes repeated notches converge on the surface rather than on a point
 * beside it.
 *
 * Two floors: the target may not come nearer the eye than the wheel's range
 * allows (that is what the URL admits), and neither may the pivot — the wheel
 * must not push the eye through the surface it is zooming into. With the
 * pivot on the target this is the plain `distance *= factor` zoom.
 */
export function zoomToward(camera: CameraState, pivot_m: Vec3, delta: number): CameraState {
  const eye = cameraPosition(camera);
  const reach_m = length(sub(eye, pivot_m));
  const requested = camera.distance_m * Math.exp(delta * ZOOM_RATE_PER_PIXEL);
  const admitted = Math.min(CAMERA_DISTANCE_RANGE.maximum_m, Math.max(CAMERA_DISTANCE_RANGE.minimum_m, requested));
  let factor = admitted / camera.distance_m, distance_m = admitted;
  // Strictly nearer, so a pivot on the target — where the two floors are the
  // same floor — keeps the admitted distance exactly.
  if (reach_m > 0 && factor * reach_m < CAMERA_DISTANCE_RANGE.minimum_m - 1e-12) {
    factor = CAMERA_DISTANCE_RANGE.minimum_m / reach_m;
    distance_m = Math.min(CAMERA_DISTANCE_RANGE.maximum_m, Math.max(CAMERA_DISTANCE_RANGE.minimum_m, factor * camera.distance_m));
  }
  const movedEye = add(pivot_m, scale(sub(eye, pivot_m), factor));
  const forward = cameraBasis(camera).forward;
  return { ...camera, distance_m, target_m: add(movedEye, scale(forward, distance_m)) };
}

/** Dolly along the view axis toward the look-at target — the centred special case. */
export function zoom(camera: CameraState, delta: number): CameraState {
  return zoomToward(camera, camera.target_m, delta);
}

/**
 * Slide the camera in its own image plane by a pixel drag.
 *
 * `worldPerPixel_m` is the scale of the frame at the depth of whatever was
 * grabbed (see `viewportWorldPerPixel`), so the surface under the cursor moves
 * with the cursor one-for-one — the convention of every tool where pan is a
 * "grab the world" gesture. The pixel axes are the screen's: dragging right
 * carries the scene right, so the camera goes left.
 */
export function pan(camera: CameraState, dx: number, dy: number, worldPerPixel_m: number): CameraState {
  const basis = cameraBasis(camera);
  return {
    ...camera,
    target_m: add(camera.target_m, add(scale(basis.right, -dx * worldPerPixel_m), scale(basis.up, dy * worldPerPixel_m)))
  };
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
