import type { FluidRefinementRegion } from "../../../core/model";
import {
  clampRefinementRegionCellSize, OCTREE_REFINEMENT_REGION_CAPACITY,
} from "../../../core/refinement-regions";
import type { SliceSceneSeed } from "./slice-scene-seed";

/**
 * Drawing an enforcement region on the slice.
 *
 * A region is the scene document's `FluidRefinementRegion` and nothing else:
 * the same box, the same `minimum-cell-size` rule, the same dyadic ladder the
 * 3-D editor snaps to. The slice already *obeys* one — `resolutionRegions` in
 * the solver projects every authored region whose z-span crosses the centre
 * plane, and the resolution policy takes it as a hard floor and ceiling on the
 * bricks it fully contains. What was missing was a way to author one from a
 * two-dimensional picture, which is all this file adds.
 *
 * Two things are specific to the slice and are the reason this is not simply
 * `editor-refinement-region.ts` called with a flat box:
 *
 * - **Depth.** A drag on the slice names x and y; the third axis is the one the
 *   picture does not have. A drawn region is given exactly one source cell of
 *   depth, centred on the plane it was drawn on, so it means "this, on this
 *   cut" rather than a slab through a domain the reader cannot see. That is
 *   also the thinnest box the containment test can accept, since the test is
 *   `min_m.z <= centreZ < max_m.z`.
 * - **The ladder stops at 8.** A slice brick is 8 fine cells across and the
 *   rungs are 1/2/4/8, so the 16 and 32 the octree offers have nothing to name
 *   here. Offering them would be a control that silently rounds.
 *
 * Everything else is deliberately the 3-D rule, including the outward snap:
 * a box is grown to the lattice of its own smallest allowed cell, because a
 * dyadic leaf of edge S is aligned to multiples of S and a box on that lattice
 * contains exactly the leaves it covers. Rounding inward instead loses a shell
 * of cells all the way around, which reads as the floor not being obeyed.
 */

/** Cell-edge bounds a slice region may name, in finest cells. */
export const SLICE_ENFORCEMENT_CELL_SIZES = Object.freeze([1, 2, 4, 8] as const);

export type SliceEnforcementCellSize = (typeof SLICE_ENFORCEMENT_CELL_SIZES)[number];

/** What a drawn region defaults to: one rung finer than a whole brick. */
export const DEFAULT_SLICE_ENFORCEMENT_CELL_SIZE: SliceEnforcementCellSize = 2;

/** A region's footprint in the canvas frame the lab draws and clicks in. */
export interface SliceEnforcementBox {
  readonly minFine: readonly [number, number];
  readonly maxFine: readonly [number, number];
}

/** Round a requested bound onto the slice's own ladder. */
export function clampSliceEnforcementCellSize(requested: number): SliceEnforcementCellSize {
  const clamped = clampRefinementRegionCellSize(requested);
  let chosen: SliceEnforcementCellSize = SLICE_ENFORCEMENT_CELL_SIZES[0];
  for (const size of SLICE_ENFORCEMENT_CELL_SIZES) if (size <= clamped) chosen = size;
  return chosen;
}

/**
 * Every region this seed stands under, in document order.
 *
 * The one reader, so a drawn region and an authored one are never two lists
 * the policy has to reconcile: the seed's own field is the whole answer once
 * it exists, and the production document's list is what a freshly built seed
 * falls back to.
 */
export function sliceSceneRegions(seed: SliceSceneSeed): readonly FluidRefinementRegion[] {
  return seed.refinementRegions ?? seed.production?.scene.fluid.refinementRegions ?? [];
}

/**
 * The regions this cut actually stands under.
 *
 * The same containment test the solver runs, so the picture can never show a
 * box the policy is ignoring or hide one it is obeying.
 */
export function sliceEnforcementRegions(seed: SliceSceneSeed):
readonly FluidRefinementRegion[] {
  const z = seed.viewport.centerZ;
  return sliceSceneRegions(seed).filter(region =>
    z >= region.min_m.z && z < region.max_m.z);
}

/** A region's box in canvas finest cells, or null when it misses this cut. */
export function sliceEnforcementRegionCanvasBox(seed: SliceSceneSeed,
  region: FluidRefinementRegion): SliceEnforcementBox | null {
  const z = seed.viewport.centerZ;
  if (z < region.min_m.z || z >= region.max_m.z) return null;
  const { originX, originY, sourceCellSize: h } = seed.viewport;
  const ny = seed.dimensions[1];
  const x0 = (region.min_m.x - originX) / h, x1 = (region.max_m.x - originX) / h;
  /* Source-up to canvas-down, the one reflection the lab performs: the box's
   * top edge on screen is its far edge in the lattice. */
  return {
    minFine: [Math.min(x0, x1), ny - (region.max_m.y - originY) / h],
    maxFine: [Math.max(x0, x1), ny - (region.min_m.y - originY) / h],
  };
}

/** The topmost region under a canvas point — what a right-click there means. */
export function sliceEnforcementRegionAt(seed: SliceSceneSeed,
  at: readonly [number, number]): FluidRefinementRegion | undefined {
  let found: FluidRefinementRegion | undefined;
  for (const region of sliceEnforcementRegions(seed)) {
    const box = sliceEnforcementRegionCanvasBox(seed, region);
    if (!box) continue;
    if (at[0] >= box.minFine[0] && at[0] <= box.maxFine[0]
      && at[1] >= box.minFine[1] && at[1] <= box.maxFine[1]) found = region;
  }
  return found;
}

/** An id no region in this document uses. */
export function nextSliceEnforcementRegionId(seed: SliceSceneSeed): string {
  const taken = new Set(sliceSceneRegions(seed).map(region => region.id));
  for (let index = 1; ; index += 1) {
    const id = `slice-region-${index}`;
    if (!taken.has(id)) return id;
  }
}

/** Whether another region can be drawn, or the uniform tail is already full. */
export function sliceEnforcementCapacityRemaining(seed: SliceSceneSeed): number {
  return Math.max(0, OCTREE_REFINEMENT_REGION_CAPACITY - sliceSceneRegions(seed).length);
}

/**
 * The region a rubber-band drag on the canvas describes.
 *
 * Snapped outward onto the lattice of its own minimum cell, clamped to the
 * lattice, and never thinner than one of the cells it is asking for — a region
 * that contained no whole leaf would be a box with no effect, which is the one
 * outcome a drawn instruction must not have.
 */
export function sliceEnforcementRegionFromCanvasDrag(seed: SliceSceneSeed,
  anchorFine: readonly [number, number], dragFine: readonly [number, number],
  options: {
    readonly id?: string;
    readonly minimumCellSize_cells?: number;
    readonly maximumCellSize_cells?: number;
  } = {}): FluidRefinementRegion {
  const cells = clampSliceEnforcementCellSize(
    options.minimumCellSize_cells ?? DEFAULT_SLICE_ENFORCEMENT_CELL_SIZE);
  const maximum = options.maximumCellSize_cells === undefined ? undefined
    : Math.max(cells, clampSliceEnforcementCellSize(options.maximumCellSize_cells));
  const [nx, ny] = seed.dimensions;
  const { originX, originY, sourceCellSize: h, centerZ } = seed.viewport;
  const span = (lo: number, hi: number, limit: number):
  readonly [number, number] => {
    /* Outward, with the tolerance the 3-D snap carries for the same reason: a
     * box already on the lattice arrives back here as 3.0000000001 steps after
     * a round trip through metres, and a bare ceil would grow it every time. */
    const tolerance = 1e-6;
    let min = Math.max(0, Math.floor(Math.min(lo, hi) / cells + tolerance) * cells);
    let max = Math.min(limit, Math.ceil(Math.max(lo, hi) / cells - tolerance) * cells);
    if (max - min < cells) max = Math.min(limit, min + cells);
    if (max - min < cells) min = Math.max(0, max - cells);
    return [min, max];
  };
  const [x0, x1] = span(anchorFine[0], dragFine[0], nx);
  /* The drag is in canvas cells and the document is source-up. */
  const [y0, y1] = span(ny - anchorFine[1], ny - dragFine[1], ny);
  return {
    id: options.id ?? nextSliceEnforcementRegionId(seed),
    rule: "minimum-cell-size",
    minimumCellSize_cells: cells,
    ...(maximum === undefined ? {} : { maximumCellSize_cells: maximum }),
    min_m: { x: originX + x0 * h, y: originY + y0 * h, z: centerZ - 0.5 * h },
    max_m: { x: originX + x1 * h, y: originY + y1 * h, z: centerZ + 0.5 * h },
  };
}

/**
 * The seed with one region added, replaced or removed.
 *
 * A live edit to the run's copy of the document, the same shape as re-timing
 * it: authoring this into the scene itself would rebuild the world and destroy
 * the run the reader is drawing on. Reset therefore takes a drawn region back,
 * which is the same contract a drop has — it belongs to the run, not to the
 * scene.
 */
export function withSliceEnforcementRegion(seed: SliceSceneSeed, id: string,
  next: FluidRefinementRegion | undefined): SliceSceneSeed {
  const current = sliceSceneRegions(seed);
  const replaced = current.some(region => region.id === id);
  const regions = replaced
    ? current.flatMap(region => region.id !== id ? [region] : next ? [next] : [])
    : next ? [...current, next] : [...current];
  return { ...seed, refinementRegions: regions };
}
