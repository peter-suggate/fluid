import { intersectBox, type EditorRay } from "./editor-entity";
import type { SceneDescription, Vec3 } from "./model";
import { terrainHeightAt } from "./terrain";
import { sceneCellSizes_m, sceneLatticeDimensions, solidVoxelShellForScene } from "./scene-lattice";
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

/**
 * Is this cell part of the container's own shell?
 *
 * The shell is authored, so it is in `scene.solidVoxels` like any other patch
 * and the solid world cannot tell it apart from a dam the reader built. The
 * editor has to, because the vessel is glass: the reader looks *through* the
 * near wall at the water and the far wall behind it, and a pick that stopped on
 * the first solid cell would answer "a voxel of the front pane" for every pixel
 * of the tank — which is how the tank and the fluid both became unselectable.
 *
 * Derived from `solidVoxelShellForScene` rather than from a rule about indices,
 * so the box shell (six slabs outside the lattice) and the spherical one (every
 * cell outside an inscribed sphere, corners included) are both exact and neither
 * can drift from the authoring. Memoized per scene document: a hover asks this
 * once per candidate cell, and the answer only changes when the lattice does.
 */
const containerShellPredicates = new WeakMap<SceneDescription, (coordinate: SolidWorldCoordinate) => boolean>();

export function containerShellContains(scene: SceneDescription, coordinate: SolidWorldCoordinate): boolean {
  let contains = containerShellPredicates.get(scene);
  if (!contains) {
    if (scene.container.shape === "sphere") {
      const dimensions = sceneLatticeDimensions(scene);
      const radius = .5 * Math.min(...dimensions);
      // Same cell-center predicate as sphericalSolidVoxelShell, including its
      // finite [-1, dimension] domain. No cubic shell materialization for a pick.
      contains = q => q.every((value, axis) => value >= -1 && value <= dimensions[axis]!)
        && Math.hypot(q[0] + .5 - .5 * dimensions[0], q[1] + .5 - .5 * dimensions[1],
          q[2] + .5 - .5 * dimensions[2]) >= radius;
    } else {
      // A box shell is five/six slabs. Keep that compact authored program
      // instead of expanding every wall cell into a string Set on first hover.
      const patches = solidVoxelShellForScene(scene);
      contains = q => patches.some(patch => q.every((value, axis) => value >= patch.minimum[axis]!
        && value < patch.maximumExclusive[axis]!));
    }
    containerShellPredicates.set(scene, contains);
  }
  return contains(coordinate);
}

export interface SolidVoxelPickOptions {
  /**
   * Cells the pick may not answer with, tested before the ray is intersected.
   *
   * A skipped cell is passed *through*, not stopped at — the search keeps going
   * to whatever stands behind it. That is the whole difference between glass and
   * a hole: the wall is still there, the editor just does not point at it.
   */
  readonly skip?: (coordinate: SolidWorldCoordinate) => boolean;
  readonly rejectBudgetExhaustion?: boolean;
}

/** Traverse only cells crossed by the input ray; never bake a terrain-sized world on pointerdown. */
function pickAuthoredVoxelRay(scene: SceneDescription, ray: EditorRay,
  options: SolidVoxelPickOptions): PickedSolidVoxel | undefined {
  const h = sceneCellSizes_m(scene), origin = worldOrigin(scene), dimensions = sceneLatticeDimensions(scene);
  const minimum = [0, 0, 0], maximum = [...dimensions];
  for (const patch of scene.solidVoxels) for (let axis = 0; axis < 3; axis++) {
    minimum[axis] = Math.min(minimum[axis]!, patch.minimum[axis]!);
    maximum[axis] = Math.max(maximum[axis]!, patch.maximumExclusive[axis]!);
  }
  const length = Math.hypot(ray.direction.x, ray.direction.y, ray.direction.z);
  if (!(length > 1e-12)) return undefined;
  const d = [ray.direction.x / length, ray.direction.y / length, ray.direction.z / length];
  const o = [ray.origin.x, ray.origin.y, ray.origin.z];
  const normalized = { origin: ray.origin, direction: { x: d[0]!, y: d[1]!, z: d[2]! } };
  const span = intersectBox(normalized, { min: { x: origin[0] + minimum[0]! * h[0], y: minimum[1]! * h[1], z: origin[2] + minimum[2]! * h[2] },
    max: { x: origin[0] + maximum[0]! * h[0], y: maximum[1]! * h[1], z: origin[2] + maximum[2]! * h[2] } });
  if (!span || span.far_m <= 1e-6) return undefined;
  let t = Math.max(0, span.near_m);
  const q = o.map((value, axis) => Math.floor((value + (t + 1e-8) * d[axis]! - origin[axis]!) / h[axis]!));
  let work = 0;
  for (let step = 0; step < 4096 && t <= span.far_m; step++) {
    const coordinate = q as unknown as SolidWorldCoordinate;
    let occupied: boolean | undefined;
    for (let index = scene.solidVoxels.length - 1; index >= 0; index--) {
      if (++work > 262144) {
        if (options.rejectBudgetExhaustion) throw new Error("This pick exceeds the bounded editing budget; point closer to the surface.");
        return undefined;
      }
      const patch = scene.solidVoxels[index]!;
      if (q.every((value, axis) => value >= patch.minimum[axis]! && value < patch.maximumExclusive[axis]!)) {
        occupied = patch.operation === "fill"; break;
      }
    }
    if (occupied === undefined && scene.terrain && q.every((value, axis) => value >= 0 && value < dimensions[axis]!)) {
      const height = Math.fround(Math.min(scene.container.height_m,
        terrainHeightAt(scene.terrain, origin[0] + (q[0]! + .5) * h[0], origin[2] + (q[2]! + .5) * h[2])));
      occupied = Math.round(255 * Math.max(0, Math.min(1, (height - q[1]! * h[1]) / h[1]))) > 0;
    }
    if (occupied && !options.skip?.(coordinate)) {
      const box = solidVoxelWorldBox(scene, coordinate), hit = intersectBox(normalized, box)!;
      const distance_m = hit.near_m > 1e-6 ? hit.near_m : hit.far_m;
      const point = o.map((value, axis) => value + distance_m * d[axis]!);
      let faceAxis: 0 | 1 | 2 = 0, faceSign: -1 | 1 = -1, nearest = Infinity;
      for (const axis of [0, 1, 2] as const) for (const sign of [-1, 1] as const) {
        const distance = Math.abs(point[axis]! - (origin[axis]! + (q[axis]! + Number(sign > 0)) * h[axis]!));
        if (distance < nearest) { nearest = distance; faceAxis = axis; faceSign = sign; }
      }
      return { coordinate: [...coordinate] as SolidWorldCoordinate, faceAxis, faceSign, distance_m,
        point_m: { x: point[0]!, y: point[1]!, z: point[2]! } };
    }
    const crossings = q.map((value, axis) => d[axis] === 0 ? Infinity
      : (origin[axis]! + (value + Number(d[axis]! > 0)) * h[axis]! - o[axis]!) / d[axis]!);
    const next = Math.min(...crossings);
    if (!Number.isFinite(next)) return undefined;
    for (let axis = 0; axis < 3; axis++) if (crossings[axis]! <= next + 1e-10) q[axis]! += Math.sign(d[axis]!);
    t = next;
  }
  if (t <= span.far_m && options.rejectBudgetExhaustion) throw new Error("This pick exceeds the bounded 4096-cell editing ray; point closer to the surface.");
  return undefined;
}

/** Ray-pick the authoritative occupied voxels, independent of authored shape. */
export function pickSolidVoxel(scene: SceneDescription,
  ray: EditorRay,
  world: SolidWorld | undefined = undefined,
  options: SolidVoxelPickOptions = {}): PickedSolidVoxel | undefined {
  if (!world) return pickAuthoredVoxelRay(scene, ray, options);
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
      if (options.skip?.(coordinate)) continue;
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
