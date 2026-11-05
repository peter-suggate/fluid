import type { CameraState, SceneDescription, Vec3 } from '../core/model';
import { cameraBasis } from '../core/math';
import { wetClothSource } from './scenes/sources';

export interface SourceCameraView {
  /** Eye, target, distance and lens match the authored camera. */
  readonly camera: CameraState;
  /** Original camera up vector retained for faithful image comparisons. */
  readonly sourceUp: Vec3;
  /** Source roll applied by the shared renderer and picking camera basis. */
  readonly roll_rad: number;
}

/** libWetCloth Camera::getLookAt rotates both +Z eye offset and +Y up by
 * the authored quaternion; coordinates here include the fixture translation. */
export function sourceCameraForScene(scene: SceneDescription): SourceCameraView | undefined {
  if (!scene.wetClothSource) return undefined;
  const source = wetClothSource(scene.wetClothSource.sourceId), camera = source.camera;
  if (!camera) return undefined;
  const q = camera.rotation, length = Math.hypot(q.x, q.y, q.z, q.w);
  const x = q.x / length, y = q.y / length, z = q.z / length, w = q.w / length;
  const outward = {x: 2 * (x * z + w * y), y: 2 * (y * z - w * x), z: 1 - 2 * (x * x + y * y)};
  const up = {x: 2 * (x * y - w * z), y: 1 - 2 * (x * x + z * z), z: 2 * (y * z + w * x)};
  const offset = scene.wetClothSource.sourceOffset_m;
  const orbit: CameraState = {
    azimuth_rad: Math.atan2(outward.x, outward.z), elevation_rad: Math.asin(Math.max(-1, Math.min(1, outward.y))),
    distance_m: camera.distance_m, target_m: {x: camera.center_m[0] + offset[0], y: camera.center_m[1] + offset[1], z: camera.center_m[2] + offset[2]},
    tanHalfFov: Math.tan(camera.fov_deg * Math.PI / 360),
  };
  const basis = cameraBasis(orbit), dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
  const roll_rad=Math.atan2(dot(up,basis.right),dot(up,basis.up));
  return {camera:{...orbit,roll_rad},sourceUp:up,roll_rad};
}
