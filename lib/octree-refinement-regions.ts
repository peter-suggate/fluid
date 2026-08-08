import type { FluidRefinementRegion, SceneDescription, Vec3 } from "./model";
import { sceneCellSizes_m, sceneLatticeDimensions } from "./scene-lattice";

/**
 * Authored boxes that cap how finely the pressure octree refines inside them.
 *
 * The octree's refinement is entirely evidence-driven: a leaf splits because
 * the free surface, a solid, a wall band or an inflow is near it. That is the
 * right default and the wrong instrument for asking "what does this scene cost
 * if I stop resolving *that* corner". A region is that instrument — draw a box
 * over a quiet pool and declare the smallest pressure cell it may hold.
 *
 * Three properties make this a cheap experiment rather than a solver change:
 *
 *  - **It only ever refuses a split.** `leafNeedsRefinement` is the one gate
 *    every topology pass asks, and a region turns a `true` into a `false`
 *    inside its own box. Nothing invents refinement, nothing changes an
 *    operator, and a scene with no regions is bit-identical.
 *  - **It is uniform-tier.** The regions ride the projection's params buffer,
 *    so drawing one, moving it, or retuning its floor is a `writeBuffer` on the
 *    running solver — no re-seed, no rebuild, no t=0. See `gpuSceneUniformKey`.
 *  - **It is not a guarantee.** Strict 2:1 grading still splits a leaf whose
 *    neighbour is finer, so the boundary of a region grades into its
 *    surroundings the way every other size transition does. A region says
 *    "stop refining here", not "these cells are exactly this size".
 *
 * Containment is *full* containment: a candidate leaf is held at the region's
 * floor only when the whole leaf lies inside the box. A leaf straddling the
 * edge refines normally, so a region can never coarsen anything outside itself.
 * Because dyadic leaves of edge S are aligned to multiples of S in cell space,
 * a box whose bounds are multiples of its own floor holds exactly the cells it
 * covers — which is why the editor snaps a region's handles to that lattice.
 */

/**
 * Regions carried per scene.
 *
 * They live in a fixed tail of the projection's uniform buffer rather than in a
 * storage binding: the topology shader's layout is already at the device's
 * storage-buffer budget (see `losasso-resident-solve`), and an experiment
 * surface that cost a binding would not be worth its own plumbing. Eight is
 * what fits comfortably and is far past what a hand-drawn experiment uses.
 */
export const OCTREE_REFINEMENT_REGION_CAPACITY = 8;

/** Words per packed region: two `vec4f` — `(min.xyz, floor)` and `(max.xyz, 0)`. */
export const OCTREE_REFINEMENT_REGION_WORDS = 8;

/** Byte offset of the region tail inside the projection params buffer. */
export const OCTREE_REFINEMENT_REGION_PARAMS_OFFSET = 160;

/** Bytes the region tail occupies: one `vec4u` control word plus the boxes. */
export const OCTREE_REFINEMENT_REGION_PARAMS_BYTES =
  16 + 4 * OCTREE_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY;

/**
 * Floors a region may name, as an edge length in finest cells.
 *
 * Powers of two, because a leaf edge is a power of two: an octree has no cell
 * of edge 3, so offering one would be a control that silently rounds. The list
 * stops at 32 for the same reason `maximumLeafSize` does — that is the largest
 * leaf the topology can hold.
 */
export const OCTREE_REFINEMENT_REGION_CELL_SIZES: readonly number[] =
  Object.freeze([1, 2, 4, 8, 16, 32]);

export const DEFAULT_REFINEMENT_REGION_CELL_SIZE = 8;

/** The one rule a region can express today. */
export const REFINEMENT_REGION_RULES = Object.freeze([
  {
    id: "minimum-cell-size" as const,
    label: "Minimum cell size",
    hint: "The pressure octree stops refining inside this box once its cells reach the chosen edge length. Grading still splits leaves on the boundary.",
  },
]);

export type RefinementRegionRule = (typeof REFINEMENT_REGION_RULES)[number]["id"];

/** Round a requested floor down onto the dyadic ladder above. */
export function clampRefinementRegionCellSize(requested: number): number {
  if (!Number.isFinite(requested)) return DEFAULT_REFINEMENT_REGION_CELL_SIZE;
  let chosen = OCTREE_REFINEMENT_REGION_CELL_SIZES[0]!;
  for (const size of OCTREE_REFINEMENT_REGION_CELL_SIZES) {
    if (size <= requested + 1e-9) chosen = size;
  }
  return chosen;
}

/** The scene's regions, bounded by what the uniform tail can carry. */
export function sceneRefinementRegions(
  scene: SceneDescription,
): readonly FluidRefinementRegion[] {
  const authored = scene.fluid?.refinementRegions;
  if (!authored || authored.length === 0) return [];
  return authored.slice(0, OCTREE_REFINEMENT_REGION_CAPACITY);
}

/** An id no existing region uses, stable and readable in a saved document. */
export function nextRefinementRegionId(scene: SceneDescription): string {
  const taken = new Set(sceneRefinementRegions(scene).map((region) => region.id));
  for (let index = 1; ; index += 1) {
    const id = `region-${index}`;
    if (!taken.has(id)) return id;
  }
}

/** Per-axis world size of a finest cell, and where cell (0,0,0) starts. */
export interface RefinementRegionLattice {
  readonly dimensions: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  readonly origin_m: Vec3;
}

/**
 * The lattice a region's bounds are measured against.
 *
 * The same rounding the solver performs, through `scene-lattice.ts`, because a
 * region that spoke a different cell size than the octree would name boxes the
 * refinement gate does not agree with.
 */
export function refinementRegionLattice(scene: SceneDescription): RefinementRegionLattice {
  const dimensions = sceneLatticeDimensions(scene);
  const c = scene.container;
  return {
    dimensions,
    cellSize_m: sceneCellSizes_m(scene),
    origin_m: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
  };
}

/** A region's world box in cell coordinates, clamped to the lattice. */
export function refinementRegionCellBounds(
  region: FluidRefinementRegion,
  lattice: RefinementRegionLattice,
): { readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number] } {
  const axes = ["x", "y", "z"] as const;
  const toCells = (point: Vec3, axis: 0 | 1 | 2) =>
    (point[axes[axis]!] - lattice.origin_m[axes[axis]!]) / lattice.cellSize_m[axis]!;
  const clamp = (value: number, axis: 0 | 1 | 2) =>
    Math.max(0, Math.min(lattice.dimensions[axis]!, value));
  return {
    min: [clamp(toCells(region.min_m, 0), 0), clamp(toCells(region.min_m, 1), 1), clamp(toCells(region.min_m, 2), 2)],
    max: [clamp(toCells(region.max_m, 0), 0), clamp(toCells(region.max_m, 1), 1), clamp(toCells(region.max_m, 2), 2)],
  };
}

/**
 * The region tail of the params buffer, ready for one `writeBuffer`.
 *
 * `maximumLeafSize` clamps every floor because a region cannot ask for a cell
 * the topology never builds: the reset pass seeds the domain at that size, so a
 * larger floor would read as "never refine anything here", which is a different
 * (and much more surprising) instruction than the one the user gave.
 */
export function packOctreeRefinementRegions(
  regions: readonly FluidRefinementRegion[],
  lattice: RefinementRegionLattice,
  maximumLeafSize: number,
): ArrayBuffer {
  const data = new ArrayBuffer(OCTREE_REFINEMENT_REGION_PARAMS_BYTES);
  const control = new Uint32Array(data, 0, 4);
  const boxes = new Float32Array(data, 16,
    OCTREE_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY);
  let written = 0;
  for (const region of regions.slice(0, OCTREE_REFINEMENT_REGION_CAPACITY)) {
    if (region.rule !== "minimum-cell-size") continue;
    const bounds = refinementRegionCellBounds(region, lattice);
    // An empty or inverted box would still be "contained" by a zero-size leaf
    // test at the degenerate corner; drop it here instead.
    if (!(bounds.max[0]! > bounds.min[0]!) || !(bounds.max[1]! > bounds.min[1]!)
      || !(bounds.max[2]! > bounds.min[2]!)) continue;
    const floor = Math.min(Math.max(1, maximumLeafSize),
      clampRefinementRegionCellSize(region.minimumCellSize_cells));
    const base = written * OCTREE_REFINEMENT_REGION_WORDS;
    boxes.set([bounds.min[0]!, bounds.min[1]!, bounds.min[2]!, floor], base);
    boxes.set([bounds.max[0]!, bounds.max[1]!, bounds.max[2]!, 0], base + 4);
    written += 1;
  }
  control[0] = written;
  return data;
}

/**
 * The floor a candidate leaf is held at, as the GPU computes it.
 *
 * A CPU mirror of `refinementRegionFloor` in the projection shader, so the
 * containment rule can be tested without a device and so a diagnostic can
 * answer "why is this cell not refining" from the host. Overlapping regions
 * take the coarsest floor: a box drawn over a box is a request for less
 * resolution, not more, and taking the finer one would make the second box
 * do nothing.
 */
export function refinementRegionFloorForLeaf(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  size: number,
): number {
  const count = Math.min(new Uint32Array(packed, 0, 4)[0]!, OCTREE_REFINEMENT_REGION_CAPACITY);
  const boxes = new Float32Array(packed, 16,
    OCTREE_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY);
  let floor = 1;
  for (let index = 0; index < count; index += 1) {
    const base = index * OCTREE_REFINEMENT_REGION_WORDS;
    let contained = true;
    for (let axis = 0; axis < 3; axis += 1) {
      if (origin[axis]! < boxes[base + axis]! || origin[axis]! + size > boxes[base + 4 + axis]!) {
        contained = false;
        break;
      }
    }
    if (contained) floor = Math.max(floor, boxes[base + 3]!);
  }
  return floor;
}

/** Whether a region holds this candidate at its floor, blocking the split. */
export function refinementRegionBlocksSplit(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  size: number,
): boolean {
  return size <= refinementRegionFloorForLeaf(packed, origin, size);
}

/** Errors `validateScene` reports for the authored regions. */
export function validateRefinementRegions(
  regions: readonly FluidRefinementRegion[] | undefined,
  container: SceneDescription["container"],
): string[] {
  if (!regions) return [];
  const errors: string[] = [];
  if (!Array.isArray(regions)) return ["Fluid refinement regions must be an array"];
  if (regions.length > OCTREE_REFINEMENT_REGION_CAPACITY) {
    errors.push(`At most ${OCTREE_REFINEMENT_REGION_CAPACITY} fluid refinement regions are supported`);
  }
  const seen = new Set<string>();
  for (const [index, region] of regions.entries()) {
    const where = `Fluid refinement region ${region?.id ?? index}`;
    if (!region?.id?.trim()) errors.push(`${where} requires an id`);
    else if (seen.has(region.id)) errors.push(`${where} has a duplicate id`);
    else seen.add(region.id);
    if (region?.rule !== "minimum-cell-size") errors.push(`${where} has an unsupported rule`);
    const min = region?.min_m, max = region?.max_m;
    if (![min?.x, min?.y, min?.z, max?.x, max?.y, max?.z].every(Number.isFinite)) {
      errors.push(`${where} bounds must be finite`);
    } else if (!(min!.x < max!.x && min!.y < max!.y && min!.z < max!.z)) {
      errors.push(`${where} must have positive extent`);
    } else if (max!.x < -container.width_m / 2 || min!.x > container.width_m / 2
      || max!.y < 0 || min!.y > container.height_m
      || max!.z < -container.depth_m / 2 || min!.z > container.depth_m / 2) {
      errors.push(`${where} must overlap the container`);
    }
    if (!OCTREE_REFINEMENT_REGION_CELL_SIZES.includes(region?.minimumCellSize_cells)) {
      errors.push(`${where} minimum cell size must be one of ${OCTREE_REFINEMENT_REGION_CELL_SIZES.join(", ")} cells`);
    }
  }
  return errors;
}
