/**
 * Sparse CM12's uniform-tier representation of authored cell-size bounds.
 *
 * Sparse CM12 chooses one dyadic resolution for a whole resident brick. A
 * region therefore constrains a brick only when the brick is fully contained
 * by the authored box. A brick crossing the box boundary keeps its ordinary
 * evidence-driven resolution, and the refine-only 2:1 closure may still make
 * a contained brick one rung finer to grade that boundary safely.
 */
import type { FluidRefinementRegion } from "../../core/model";
import {
  clampRefinementRegionCellSize,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  refinementRegionCellBounds,
  type RefinementRegionLattice,
} from "../../core/refinement-regions";

/** `(minimum.xyz, floor)` followed by `(maximum.xyz, optional ceiling)`. */
export const SPARSE_CM12_REFINEMENT_REGION_WORDS = 8;

/** One vec4 control plus two vec4 records for each fixed-capacity region. */
export const SPARSE_CM12_REFINEMENT_REGION_BYTES = 16
  + 4 * SPARSE_CM12_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY;

/** Byte offset of the region tail in Sparse CM12's Params uniform. */
export const SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET = 448;

export interface SparseCM12RefinementRegionResolutionBounds {
  /** Finest allowed brick resolution; enforces the authored smallest cell. */
  readonly maximumResolution: number;
  /** Coarsest allowed brick resolution; enforces an authored largest cell. */
  readonly minimumResolution: number;
}

/**
 * Pack scene regions for the resident topology planner.
 *
 * Bounds are finest-cell coordinates, matching every physical topology record.
 * The cell-size ladder is already capped at 32, so unlike the legacy octree
 * packer this format needs no method-specific maximum-leaf clamp.
 */
export function packSparseCM12RefinementRegions(
  regions: readonly FluidRefinementRegion[],
  lattice: RefinementRegionLattice,
): ArrayBuffer {
  const data = new ArrayBuffer(SPARSE_CM12_REFINEMENT_REGION_BYTES);
  const control = new Uint32Array(data, 0, 4);
  const boxes = new Float32Array(data, 16,
    SPARSE_CM12_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY);
  let written = 0;
  for (const region of regions.slice(0, OCTREE_REFINEMENT_REGION_CAPACITY)) {
    if (region.rule !== "minimum-cell-size") continue;
    const bounds = refinementRegionCellBounds(region, lattice);
    if (!(bounds.max[0]! > bounds.min[0]!) || !(bounds.max[1]! > bounds.min[1]!)
      || !(bounds.max[2]! > bounds.min[2]!)) continue;
    const floor = clampRefinementRegionCellSize(region.minimumCellSize_cells);
    const ceiling = region.maximumCellSize_cells === undefined ? 0
      : clampRefinementRegionCellSize(region.maximumCellSize_cells);
    const base = written * SPARSE_CM12_REFINEMENT_REGION_WORDS;
    boxes.set([bounds.min[0]!, bounds.min[1]!, bounds.min[2]!, floor], base);
    boxes.set([bounds.max[0]!, bounds.max[1]!, bounds.max[2]!, ceiling], base + 4);
    written += 1;
  }
  control[0] = written;
  return data;
}

function containingCellSizeBounds(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  extent: readonly [number, number, number],
): { readonly floor: number; readonly ceiling: number } {
  const count = Math.min(new Uint32Array(packed, 0, 4)[0]!,
    OCTREE_REFINEMENT_REGION_CAPACITY);
  const boxes = new Float32Array(packed, 16,
    SPARSE_CM12_REFINEMENT_REGION_WORDS * OCTREE_REFINEMENT_REGION_CAPACITY);
  let floor = 1;
  let ceiling = 0;
  for (let index = 0; index < count; index += 1) {
    const base = index * SPARSE_CM12_REFINEMENT_REGION_WORDS;
    const contained = origin.every((value, axis) => value >= boxes[base + axis]!
      && value + extent[axis]! <= boxes[base + 4 + axis]!);
    if (!contained) continue;
    floor = Math.max(floor, boxes[base + 3]!);
    const authoredCeiling = boxes[base + 7]!;
    if (authoredCeiling > 0) {
      ceiling = ceiling === 0 ? authoredCeiling : Math.min(ceiling, authoredCeiling);
    }
  }
  return { floor, ceiling };
}

/**
 * CPU mirror of the resident shader's per-brick resolution clamp.
 *
 * Overlapping minimum-size bounds choose the coarsest floor. Overlapping
 * maximum-size bounds choose the finest ceiling. If two overlapping boxes
 * conflict, the ceiling wins, matching the legacy octree gate's conservative
 * preference for additional resolution.
 */
export function sparseCM12RefinementRegionResolutionBoundsForBrick(
  packed: ArrayBuffer,
  origin: readonly [number, number, number],
  extent: readonly [number, number, number],
  brickFineResolution: number,
  nominalEdge = Math.max(...extent),
): SparseCM12RefinementRegionResolutionBounds {
  const { floor, ceiling } = containingCellSizeBounds(packed, origin, extent);
  const maximumResolution = Math.max(1, Math.min(brickFineResolution,
    Math.floor(nominalEdge / floor)));
  const minimumResolution = ceiling === 0 ? 1 : Math.max(1,
    Math.min(brickFineResolution, Math.ceil(nominalEdge / ceiling)));
  return { maximumResolution, minimumResolution };
}

/** Apply the bounds with the conservative (finer) ceiling winning conflicts. */
export function applySparseCM12RefinementRegionResolutionBounds(
  requestedResolution: number,
  bounds: SparseCM12RefinementRegionResolutionBounds,
): number {
  return Math.max(bounds.minimumResolution,
    Math.min(bounds.maximumResolution, requestedResolution));
}
