import type { CameraState, Vec3 } from "./model";

export const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vec3, b: Vec3): Vec3 => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
export const length = (a: Vec3): number => Math.sqrt(dot(a, a));
export const normalize = (a: Vec3): Vec3 => { const l = length(a); return l > 0 ? scale(a, 1 / l) : { x: 0, y: 0, z: 0 }; };

export function cameraPosition(camera: CameraState): Vec3 {
  const horizontal = camera.distance_m * Math.cos(camera.elevation_rad);
  return add(camera.target_m, {
    x: horizontal * Math.sin(camera.azimuth_rad),
    y: camera.distance_m * Math.sin(camera.elevation_rad),
    z: horizontal * Math.cos(camera.azimuth_rad)
  });
}

export function cameraBasis(camera: CameraState) {
  const position = cameraPosition(camera);
  const forward = normalize(sub(camera.target_m, position));
  const right = normalize(cross(forward, { x: 0, y: 1, z: 0 }));
  const up = normalize(cross(right, forward));
  return { position, forward, right, up };
}

export function orbit(camera: CameraState, dx: number, dy: number): CameraState {
  return {
    ...camera,
    azimuth_rad: camera.azimuth_rad - dx * 0.007,
    elevation_rad: Math.max(-1.35, Math.min(1.35, camera.elevation_rad + dy * 0.007))
  };
}

/**
 * How near and how far the wheel may take the camera from what it is looking at.
 *
 * Wide, because the pivot is whatever the viewport is centred on rather than a
 * fixed authored point (see `retarget`): zooming in is approaching *that*
 * surface, so the floor is a distance from the thing itself and 2 cm leaves it
 * two near planes clear of the 0.01 m clip in `voxelViewProjectionMatrix`. The
 * ceiling sits inside that same matrix's 100 m far plane with room for whatever
 * stands behind the pivot, and well outside the widest authored view: the ocean
 * tank is 8 m across and the largest preset opens at 15 m, which the previous
 * 12 m ceiling could not even hold still.
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

export function zoom(camera: CameraState, delta: number): CameraState {
  const distance_m = camera.distance_m * Math.exp(delta * ZOOM_RATE_PER_PIXEL);
  return {
    ...camera,
    distance_m: Math.min(CAMERA_DISTANCE_RANGE.maximum_m,
      Math.max(CAMERA_DISTANCE_RANGE.minimum_m, distance_m)),
  };
}

/**
 * Re-anchor the camera on a point *without moving it*.
 *
 * The camera is spherical — an azimuth, an elevation and a distance about a
 * target — so orbiting and zooming both happen about `target_m` and nothing
 * else. Making the pivot "whatever is at the centre of the viewport" is
 * therefore not a change to `orbit` or `zoom` at all: it is moving the target
 * onto that point first, and it is free precisely because the point came off
 * the centre ray. The centre ray *is* the forward axis, so a pivot on it is
 * already straight ahead: re-deriving the angles and the distance from the
 * unchanged eye position reproduces the same view direction and the same eye,
 * and the rendered frame does not move by a pixel.
 *
 * Idempotent for a pivot it has already been anchored to, which is what lets
 * the wheel apply it on every event without a gesture having to remember
 * whether it did so already.
 *
 * A pivot outside {@link CAMERA_DISTANCE_RANGE} is declined rather than
 * clamped: clamping would keep the target and move the eye, which is the one
 * thing this must never do.
 */
export function retarget(camera: CameraState, pivot_m: Vec3): CameraState {
  const offset = sub(cameraPosition(camera), pivot_m);
  const distance_m = length(offset);
  if (!(distance_m >= CAMERA_DISTANCE_RANGE.minimum_m && distance_m <= CAMERA_DISTANCE_RANGE.maximum_m)) {
    return camera;
  }
  return {
    ...camera,
    target_m: pivot_m,
    distance_m,
    azimuth_rad: Math.atan2(offset.x, offset.z),
    elevation_rad: Math.asin(Math.max(-1, Math.min(1, offset.y / distance_m))),
  };
}

export function pan(camera: CameraState, dx: number, dy: number): CameraState {
  const basis = cameraBasis(camera);
  const worldPerPixel = camera.distance_m * 0.0016;
  return {
    ...camera,
    target_m: add(camera.target_m, add(scale(basis.right, -dx * worldPerPixel), scale(basis.up, dy * worldPerPixel)))
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
