import type { EditorRay } from "./editor-entity";
import type { SceneDescription, Vec3 } from "./model";
import { sceneCellSizes_m, sceneLatticeDimensions, solidVoxelShellForScene } from "./scene-lattice";
import { solidFractionOccupies } from "./solid-patch/kind";
import type { SolidWorld, SolidWorldCoordinate } from "./solid-world";
import { voxelReaderForScene } from "./voxel-editor/voxel-reader";

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

/**
 * Ray-pick the first solid voxel along a ray, whatever authored it.
 *
 * Answered by the scene's voxel reader: the index of its patches and the
 * analytic ground, walked a cell at a time. `world` is not consulted — it is
 * the same cells over the same ground, and a document clone may not hold one.
 * The parameter stays only until its callers stop passing one.
 */
export function pickSolidVoxel(scene: SceneDescription,
  ray: EditorRay,
  world: SolidWorld | undefined = undefined,
  options: SolidVoxelPickOptions = {}): PickedSolidVoxel | undefined {
  void world;
  return voxelReaderForScene(scene).pick(ray, options);
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

/** Cells one slab of a preview count reads at a time; the region itself is not bounded by it. */
const CLEAR_PREVIEW_SLAB_CELLS = 1 << 20;

/**
 * Exact occupied subset highlighted by a proposed generic clear box.
 *
 * Counted from the scene's voxel reader: a preview is asked for while a
 * selection is being dragged, and that is no moment to bake a terrain.
 */
export function solidVoxelClearPreview(scene: SceneDescription,
  region: SolidVoxelClearRegion,
  capacity = SOLID_VOXEL_CLEAR_PREVIEW_CAPACITY): SolidVoxelClearPreview {
  const coordinates: SolidWorldCoordinate[] = [];
  let affectedCount = 0;
  const reader = voxelReaderForScene(scene);
  const [x0, y0, z0] = region.minimum, [x1, y1, z1] = region.maximumExclusive;
  // A box with no cells clears none; the reader refuses to read one.
  if (x1 <= x0 || y1 <= y0) return { coordinates, affectedCount, truncated: false };
  const layers = Math.max(1, Math.floor(CLEAR_PREVIEW_SLAB_CELLS / Math.max(1, (x1 - x0) * (y1 - y0))));
  for (let slab = z0; slab < z1; slab += layers) {
    const block = reader.read([x0, y0, slab], [x1, y1, Math.min(z1, slab + layers)]);
    for (let cell = 0; cell < block.solidFraction.length; cell += 1) {
      if (!solidFractionOccupies(block.solidFraction[cell]!)) continue;
      affectedCount += 1;
      if (coordinates.length < capacity) coordinates.push([x0 + cell % block.size[0],
        y0 + Math.floor(cell / block.size[0]) % block.size[1],
        slab + Math.floor(cell / (block.size[0] * block.size[1]))]);
    }
  }
  return { coordinates, affectedCount, truncated: affectedCount > capacity };
}
