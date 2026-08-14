/**
 * How an authored refinement box reaches the octree's topology gate.
 *
 * The boxes themselves, their dyadic cell-size ladder and the lattice they are
 * measured against are scene-document facts and live in
 * `lib/core/refinement-regions.ts`. What is left here is the half that only a
 * solver can mean: the fixed tail these boxes occupy in the projection's params
 * buffer, and the CPU mirror of the containment tests the refinement gate runs
 * against that tail.
 *
 * Three properties make this a cheap experiment rather than a solver change:
 *
 *  - **Each bound changes only the matching decision.** The minimum may refuse
 *    a split; the optional maximum may require one. Nothing changes an
 *    operator, and a scene with no regions is bit-identical.
 *  - **It is uniform-tier.** The regions ride the projection's params buffer,
 *    so drawing one, moving it, or retuning its floor is a `writeBuffer` on the
 *    running solver — no re-seed, no rebuild, no t=0. See `gpuSceneUniformKey`.
 *  - **The minimum is not a guarantee.** Strict 2:1 grading can still split a
 *    leaf whose neighbour is finer. The maximum is a hard upper bound for
 *    fully contained leaves because grading only ever adds refinement.
 */
import type { FluidRefinementRegion, SceneDescription } from "../../core/model";
import {
  clampRefinementRegionCellSize,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  refinementRegionCellBounds,
  refinementRegionLattice,
  sceneRefinementRegions,
  type RefinementRegionLattice,
} from "../../core/refinement-regions";

/** Words per packed region: `(min.xyz, floor)` and `(max.xyz, optional ceiling)`. */
export const OCTREE_REFINEMENT_REGION_WORDS = 8;

/** Byte offset of the region tail inside the projection params buffer. */
export const OCTREE_REFINEMENT_REGION_PARAMS_OFFSET = 160;

/** Bytes the region tail occupies: one `vec4u` control word plus the boxes. */
export const OCTREE_REFINEMENT_REGION_PARAMS_BYTES =
  16 + 4 * OCTREE_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY;

/** Whether one authored box caps the entire pressure domain at finest cells.
 * Such a scene has exactly one possible octree topology: rebuilding an
 * identical candidate cannot add refinement evidence and only introduces a
 * needless field-migration boundary. */
export function sceneHasUniformFinestCellCeiling(scene: SceneDescription): boolean {
  const lattice = refinementRegionLattice(scene);
  const epsilon = 1e-4;
  return sceneRefinementRegions(scene).some((region) => {
    if (region.rule !== "minimum-cell-size" || region.maximumCellSize_cells !== 1) {
      return false;
    }
    const bounds = refinementRegionCellBounds(region, lattice);
    return bounds.min.every((value) => value <= epsilon)
      && bounds.max.every((value, axis) =>
        value >= lattice.dimensions[axis]! - epsilon);
  });
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
    const ceiling = region.maximumCellSize_cells === undefined ? 0
      : Math.min(Math.max(1, maximumLeafSize),
        clampRefinementRegionCellSize(region.maximumCellSize_cells));
    const base = written * OCTREE_REFINEMENT_REGION_WORDS;
    boxes.set([bounds.min[0]!, bounds.min[1]!, bounds.min[2]!, floor], base);
    boxes.set([bounds.max[0]!, bounds.max[1]!, bounds.max[2]!, ceiling], base + 4);
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

/** The finest authored ceiling that contains this leaf; zero means no cap. */
export function refinementRegionCeilingForLeaf(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  size: number,
): number {
  const count = Math.min(new Uint32Array(packed, 0, 4)[0]!, OCTREE_REFINEMENT_REGION_CAPACITY);
  const boxes = new Float32Array(packed, 16,
    OCTREE_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY);
  let ceiling = 0;
  for (let index = 0; index < count; index += 1) {
    const base = index * OCTREE_REFINEMENT_REGION_WORDS;
    let contained = true;
    for (let axis = 0; axis < 3; axis += 1) {
      if (origin[axis]! < boxes[base + axis]!
        || origin[axis]! + size > boxes[base + 4 + axis]!) {
        contained = false;
        break;
      }
    }
    const authored = boxes[base + 7]!;
    if (contained && authored > 0) ceiling = ceiling === 0
      ? authored : Math.min(ceiling, authored);
  }
  return ceiling;
}

/** Whether an optional largest-cell bound requires this candidate to split. */
export function refinementRegionForcesSplit(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  size: number,
): boolean {
  const ceiling = refinementRegionCeilingForLeaf(packed, origin, size);
  return ceiling > 0 && size > ceiling;
}

/** Whether a region holds this candidate at its floor, blocking the split. */
export function refinementRegionBlocksSplit(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  size: number,
): boolean {
  return size <= refinementRegionFloorForLeaf(packed, origin, size);
}
