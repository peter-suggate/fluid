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

export function zoom(camera: CameraState, delta: number): CameraState {
  return { ...camera, distance_m: Math.max(0.65, Math.min(12, camera.distance_m * Math.exp(delta * 0.001))) };
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
