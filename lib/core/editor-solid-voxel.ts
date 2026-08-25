import { intersectBox, type EditorRay } from "./editor-entity";
import type { SceneDescription, Vec3 } from "./model";
import { sceneCellSizes_m } from "./scene-lattice";
import { solidWorldForScene, type SolidWorld, type SolidWorldCoordinate,
  type SolidWorldVoxelPatch } from "./solid-world";

export const SOLID_VOXEL_CLEAR_PREVIEW_CAPACITY = 512;

export interface PickedSolidVoxel {
  readonly coordinate: SolidWorldCoordinate;
  readonly faceAxis: 0 | 1 | 2;
  readonly faceSign: -1 | 1;
  readonly distance_m: number;
  readonly point_m: Vec3;
}

export interface SolidVoxelClearRegion {
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
}

export interface SolidVoxelClearPreview {
  readonly coordinates: readonly SolidWorldCoordinate[];
  readonly affectedCount: number;
  /** Only drawing detail is capped; the edit and exact count are never capped. */
  readonly truncated: boolean;
}

const worldOrigin = (scene: SceneDescription): readonly [number, number, number] =>
  [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];

export function solidVoxelWorldBox(scene: SceneDescription,
  coordinate: SolidWorldCoordinate): { readonly min: Vec3; readonly max: Vec3 } {
  const origin = worldOrigin(scene);
  const cell = sceneCellSizes_m(scene);
  return {
    min: { x: origin[0] + coordinate[0] * cell[0],
      y: origin[1] + coordinate[1] * cell[1],
      z: origin[2] + coordinate[2] * cell[2] },
    max: { x: origin[0] + (coordinate[0] + 1) * cell[0],
      y: origin[1] + (coordinate[1] + 1) * cell[1],
      z: origin[2] + (coordinate[2] + 1) * cell[2] },
  };
}

/** Ray-pick the authoritative occupied voxels, independent of authored shape. */
export function pickSolidVoxel(scene: SceneDescription,
  ray: EditorRay,
  world: SolidWorld = solidWorldForScene(scene)): PickedSolidVoxel | undefined {
  if (world.pages.length === 0) return undefined;
  const rayLength = Math.hypot(ray.direction.x, ray.direction.y, ray.direction.z);
  if (!(rayLength > 1e-12)) return undefined;
  const normalizedRay: EditorRay = { origin: ray.origin, direction: {
    x: ray.direction.x / rayLength, y: ray.direction.y / rayLength,
    z: ray.direction.z / rayLength,
  } };
  const origin = worldOrigin(scene);
  const cell = sceneCellSizes_m(scene);
  const candidates = world.pages.flatMap((page) => {
    const minimum = page.coordinate.map((value, axis) =>
      origin[axis]! + 8 * value * cell[axis]!);
    const maximum = page.coordinate.map((value, axis) =>
      origin[axis]! + 8 * (value + 1) * cell[axis]!);
    const span = intersectBox(normalizedRay, {
      min: { x: minimum[0]!, y: minimum[1]!, z: minimum[2]! },
      max: { x: maximum[0]!, y: maximum[1]!, z: maximum[2]! },
    });
    return span && span.far_m > 1e-6
      ? [{ page, entry_m: Math.max(0, span.near_m) }] : [];
  }).sort((left, right) => left.entry_m - right.entry_m);
  let best: PickedSolidVoxel | undefined;
  for (const candidatePage of candidates) {
    if (best && candidatePage.entry_m >= best.distance_m) break;
    const page = candidatePage.page;
    for (let local = 0; local < page.solidFraction.length; local += 1) {
      if (page.solidFraction[local]! === 0) continue;
      const coordinate = [8 * page.coordinate[0] + local % 8,
        8 * page.coordinate[1] + Math.floor(local / 8) % 8,
        8 * page.coordinate[2] + Math.floor(local / 64)] as const;
      const box = solidVoxelWorldBox(scene, coordinate);
      const span = intersectBox(normalizedRay, box);
      if (!span) continue;
      const distance_m = span.near_m > 1e-6 ? span.near_m : span.far_m;
      if (!(distance_m > 1e-6) || (best && distance_m >= best.distance_m)) continue;
      const point_m = {
        x: normalizedRay.origin.x + normalizedRay.direction.x * distance_m,
        y: normalizedRay.origin.y + normalizedRay.direction.y * distance_m,
        z: normalizedRay.origin.z + normalizedRay.direction.z * distance_m,
      };
      const point = [point_m.x, point_m.y, point_m.z] as const;
      const minimum = [box.min.x, box.min.y, box.min.z] as const;
      const maximum = [box.max.x, box.max.y, box.max.z] as const;
      let faceAxis: 0 | 1 | 2 = 0;
      let faceSign: -1 | 1 = -1;
      let faceDistance = Number.POSITIVE_INFINITY;
      for (const axis of [0, 1, 2] as const) for (const sign of [-1, 1] as const) {
        const faceCoordinate = sign < 0 ? minimum[axis] : maximum[axis];
        const distance = Math.abs(point[axis] - faceCoordinate);
        if (distance < faceDistance) {
          faceDistance = distance;
          faceAxis = axis;
          faceSign = sign;
        }
      }
      best = { coordinate, faceAxis, faceSign, distance_m, point_m };
    }
  }
  return best;
}

/** Project a drag onto the picked voxel face and return a one-voxel-deep box. */
export function projectSolidVoxelClearRegion(scene: SceneDescription,
  ray: EditorRay, anchor: PickedSolidVoxel): SolidVoxelClearRegion | undefined {
  const origin = worldOrigin(scene);
  const cell = sceneCellSizes_m(scene);
  const axis = anchor.faceAxis;
  const planeCell = anchor.coordinate[axis] + (anchor.faceSign > 0 ? 1 : 0);
  const plane_m = origin[axis] + planeCell * cell[axis];
  const rayOrigin = [ray.origin.x, ray.origin.y, ray.origin.z] as const;
  const rayDirection = [ray.direction.x, ray.direction.y, ray.direction.z] as const;
  if (Math.abs(rayDirection[axis]) < 1e-8) return undefined;
  const distance_m = (plane_m - rayOrigin[axis]) / rayDirection[axis];
  if (!(distance_m > 1e-6)) return undefined;
  const end: [number, number, number] = [...anchor.coordinate];
  for (const tangent of [0, 1, 2] as const) {
    if (tangent === axis) continue;
    const worldCoordinate = rayOrigin[tangent] + rayDirection[tangent] * distance_m;
    const voxel = Math.floor((worldCoordinate - origin[tangent]) / cell[tangent]);
    if (!Number.isSafeInteger(voxel) || voxel < -0x8000_0000 || voxel > 0x7fff_ffff) {
      return undefined;
    }
    end[tangent] = voxel;
  }
  const minimum = anchor.coordinate.map((value, component) => component === axis
    ? value : Math.min(value, end[component]!)) as unknown as SolidWorldCoordinate;
  const maximumExclusive = anchor.coordinate.map((value, component) => component === axis
    ? value + 1 : Math.max(value, end[component]!) + 1) as unknown as SolidWorldCoordinate;
  return { minimum, maximumExclusive };
}

/** Exact occupied subset highlighted by a proposed generic clear box. */
export function solidVoxelClearPreview(scene: SceneDescription,
  region: SolidVoxelClearRegion,
  world: SolidWorld = solidWorldForScene(scene),
  capacity = SOLID_VOXEL_CLEAR_PREVIEW_CAPACITY): SolidVoxelClearPreview {
  const coordinates: SolidWorldCoordinate[] = [];
  let affectedCount = 0;
  for (const page of world.pages) {
    const pageMinimum = page.coordinate.map((value) => 8 * value) as
      unknown as SolidWorldCoordinate;
    const localMinimum = pageMinimum.map((value, axis) =>
      Math.max(0, region.minimum[axis]! - value));
    const localMaximum = pageMinimum.map((value, axis) =>
      Math.min(8, region.maximumExclusive[axis]! - value));
    if (localMinimum.some((value, axis) => value >= localMaximum[axis]!)) continue;
    for (let z = localMinimum[2]!; z < localMaximum[2]!; z += 1)
      for (let y = localMinimum[1]!; y < localMaximum[1]!; y += 1)
        for (let x = localMinimum[0]!; x < localMaximum[0]!; x += 1) {
          const local = x + 8 * (y + 8 * z);
          if (page.solidFraction[local]! === 0) continue;
          affectedCount += 1;
          if (coordinates.length < capacity) coordinates.push([
            pageMinimum[0] + x, pageMinimum[1] + y, pageMinimum[2] + z,
          ]);
        }
  }
  return { coordinates, affectedCount, truncated: affectedCount > capacity };
}

/** Preserve every prior edit and append one ordinary SolidWorld clear edit. */
export function withSolidVoxelClearRegion(patches: readonly SolidWorldVoxelPatch[],
  region: SolidVoxelClearRegion): SolidWorldVoxelPatch[] {
  return [...patches, { operation: "clear", minimum: [...region.minimum],
    maximumExclusive: [...region.maximumExclusive] }];
}
