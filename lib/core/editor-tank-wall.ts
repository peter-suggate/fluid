import { intersectBox, type EditorRay } from "./editor-entity";
import type { SceneDescription, Vec3 } from "./model";
import {
  withTankWallRectangle as editTankWallRectangle,
  type TankWallField,
  type TankWallSide,
} from "./tank-wall-field";

export interface PickedTankWallCell {
  readonly side: TankWallSide;
  readonly u: number;
  readonly v: number;
  readonly distance_m: number;
  readonly point_m: Vec3;
  readonly key: string;
}

export interface TankWallRectangle {
  readonly side: TankWallSide;
  readonly u0: number;
  readonly v0: number;
  readonly u1: number;
  readonly v1: number;
}

const tankWallBox = (scene: SceneDescription) => ({
  min: {
    x: -0.5 * scene.container.width_m,
    y: 0,
    z: -0.5 * scene.container.depth_m,
  },
  max: {
    x: 0.5 * scene.container.width_m,
    y: scene.container.height_m,
    z: 0.5 * scene.container.depth_m,
  },
});

const tankWallCellAtPoint = (
  scene: SceneDescription,
  side: TankWallSide,
  point_m: Vec3,
  distance_m: number,
): PickedTankWallCell => {
  const c = scene.container;
  const dimensions = c.wallField.faces[side];
  const along = side === "left" || side === "right"
    ? (point_m.z + 0.5 * c.depth_m) / c.depth_m
    : (point_m.x + 0.5 * c.width_m) / c.width_m;
  const u = Math.max(0, Math.min(dimensions.uCells - 1, Math.floor(along * dimensions.uCells)));
  const v = Math.max(0, Math.min(dimensions.vCells - 1,
    Math.floor(point_m.y / c.height_m * dimensions.vCells)));
  return { side, u, v, distance_m, point_m, key: `${side}:${u}:${v}` };
};

/** Pick the actual face-cell atlas, including cleared cells used by repair. */
export function pickTankWallCell(
  scene: SceneDescription,
  ray: EditorRay,
): PickedTankWallCell | undefined {
  if (scene.container.shape === "sphere") return undefined;
  const c = scene.container;
  const box = tankWallBox(scene);
  const span = intersectBox(ray, box);
  if (!span) return undefined;
  const distances = [span.near_m, span.far_m].filter((value) => value > 1e-6).sort((a, b) => a - b);
  const tolerance = Math.max(c.width_m, c.height_m, c.depth_m) * 1e-5;
  for (const distance_m of distances) {
    const point_m = {
      x: ray.origin.x + ray.direction.x * distance_m,
      y: ray.origin.y + ray.direction.y * distance_m,
      z: ray.origin.z + ray.direction.z * distance_m,
    };
    const candidates: Array<{ side: TankWallSide; distance: number }> = [
      { side: "left", distance: Math.abs(point_m.x - box.min.x) },
      { side: "right", distance: Math.abs(point_m.x - box.max.x) },
      { side: "front", distance: Math.abs(point_m.z - box.min.z) },
      { side: "back", distance: Math.abs(point_m.z - box.max.z) },
    ];
    const face = candidates.sort((a, b) => a.distance - b.distance)[0]!;
    if (face.distance > tolerance || point_m.y < -tolerance || point_m.y > c.height_m + tolerance) continue;
    return tankWallCellAtPoint(scene, face.side, point_m, distance_m);
  }
  return undefined;
}

/** Keep a wall drag on the face where it began, clamping beyond its edges. */
export function projectTankWallCellOnSide(
  scene: SceneDescription,
  ray: EditorRay,
  side: TankWallSide,
): PickedTankWallCell | undefined {
  if (scene.container.shape === "sphere") return undefined;
  const c = scene.container;
  const xFace = side === "left" || side === "right";
  const plane = side === "left" ? -0.5 * c.width_m
    : side === "right" ? 0.5 * c.width_m
      : side === "front" ? -0.5 * c.depth_m : 0.5 * c.depth_m;
  const origin = xFace ? ray.origin.x : ray.origin.z;
  const direction = xFace ? ray.direction.x : ray.direction.z;
  if (Math.abs(direction) < 1e-8) return undefined;
  const distance_m = (plane - origin) / direction;
  if (!(distance_m > 1e-6)) return undefined;
  const raw = {
    x: ray.origin.x + ray.direction.x * distance_m,
    y: ray.origin.y + ray.direction.y * distance_m,
    z: ray.origin.z + ray.direction.z * distance_m,
  };
  const point_m = {
    x: xFace ? plane : Math.max(-0.5 * c.width_m, Math.min(0.5 * c.width_m, raw.x)),
    y: Math.max(0, Math.min(c.height_m, raw.y)),
    z: xFace ? Math.max(-0.5 * c.depth_m, Math.min(0.5 * c.depth_m, raw.z)) : plane,
  };
  return tankWallCellAtPoint(scene, side, point_m, distance_m);
}

/** Cell-aligned world-space corners for the drag preview. */
export function tankWallRectangleCorners(
  scene: SceneDescription,
  rectangle: TankWallRectangle,
): readonly [Vec3, Vec3, Vec3, Vec3] {
  const c = scene.container;
  const face = c.wallField.faces[rectangle.side];
  const u0 = Math.min(rectangle.u0, rectangle.u1) / face.uCells;
  const u1 = (Math.max(rectangle.u0, rectangle.u1) + 1) / face.uCells;
  const v0 = Math.min(rectangle.v0, rectangle.v1) / face.vCells * c.height_m;
  const v1 = (Math.max(rectangle.v0, rectangle.v1) + 1) / face.vCells * c.height_m;
  const point = (u: number, y: number): Vec3 => {
    if (rectangle.side === "left" || rectangle.side === "right") return {
      x: rectangle.side === "left" ? -0.5 * c.width_m : 0.5 * c.width_m,
      y,
      z: (u - 0.5) * c.depth_m,
    };
    return {
      x: (u - 0.5) * c.width_m,
      y,
      z: rectangle.side === "front" ? -0.5 * c.depth_m : 0.5 * c.depth_m,
    };
  };
  return [point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)];
}

/** Remove or restore one inclusive, cell-aligned face rectangle. */
export function withTankWallRectangle(
  field: TankWallField,
  side: TankWallSide,
  u0: number,
  v0: number,
  u1: number,
  v1: number,
  solid: boolean,
): TankWallField {
  return editTankWallRectangle(field, side, u0, v0, u1, v1, solid);
}
