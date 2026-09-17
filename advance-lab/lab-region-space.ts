import {
  OCTREE_REFINEMENT_REGION_CAPACITY,
  type RefinementRegionRecord,
  type RegionBox,
  type RegionSpace,
} from "../lib/features/refinement-region/definition";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import { ADVANCE_BRICK_FINE, ADVANCE_RUNGS } from "../lib/physics-wasm/advance-view";

/**
 * The advance lab as a `RegionSpace`: two axes, a Rust world behind them.
 *
 * This file is the lab's whole half of the region capability. Everything that
 * used to be in `advance-lab/slice-regions.ts` — the outward snap, the held
 * opposite side, the translate-not-reshape move, the ceiling that follows its
 * floor, the selection-id prefix restated by hand — is the shared package's
 * now, in finest cells and N dimensions. What is left is the three things that
 * genuinely belong to a 2-D slice of a Wasm world: the field names
 * `AdvanceRefinementRegion` uses, the y-flip between the solver's frame and the
 * canvas, and which rungs this lattice has.
 *
 * ## Two frames, and which is which
 *
 * A region is *stored* the way the solver reads it — `minimumFine`/
 * `maximumFine` in lattice cells with y running **up** from the floor — and a
 * `RefinementRegionRecord` is in that same frame, because the record is defined
 * as the frame the solver reads a region in. The picture is drawn the way a
 * canvas is, with y running **down**. `labRegionCanvasBox` and
 * `labRegionFromCanvasBox` are the only two places the flip happens, so no
 * caller has to remember it and `lab-region-space.test.ts` has one pair of
 * functions to hold to it.
 */

/**
 * What the lab calls a document: the boxes, and the slice they live on.
 *
 * The lattice rides along because `RegionSpace.lattice` takes the document and
 * the lab's boxes are bounded by the *published* slice rather than by anything
 * the region list knows — a run on a different scene is a different lattice
 * with the same regions still drawn on it.
 *
 * `Patch = Doc`: `AdvanceLabController.setRefinementRegions` is a whole-list
 * command, so there is no merge to express, which is exactly the case
 * `EditorHost.commitPatch` is allowed to be absent for.
 */
export interface LabRegionDocument {
  readonly regions: readonly AdvanceRefinementRegion[];
  /** Finest cells across the published slice. */
  readonly nx: number;
  readonly ny: number;
}

/** The stored box as the shared record: solver frame, y up. */
export function labRegionRecord(region: AdvanceRefinementRegion): RefinementRegionRecord {
  return {
    id: region.id,
    rule: "minimum-cell-size",
    minimumCellSize_cells: region.minimumCellWidth,
    ...(region.maximumCellWidth === undefined ? {}
      : { maximumCellSize_cells: region.maximumCellWidth }),
    min_cells: [region.minimumFine[0], region.minimumFine[1]],
    max_cells: [region.maximumFine[0], region.maximumFine[1]],
  };
}

/** The shared record back as the controller carries it. */
export function labRegionFromRecord(record: RefinementRegionRecord): AdvanceRefinementRegion {
  return {
    id: record.id,
    minimumFine: [record.min_cells[0] ?? 0, record.min_cells[1] ?? 0],
    maximumFine: [record.max_cells[0] ?? 0, record.max_cells[1] ?? 0],
    minimumCellWidth: record.minimumCellSize_cells,
    ...(record.maximumCellSize_cells === undefined ? {}
      : { maximumCellWidth: record.maximumCellSize_cells }),
  };
}

/**
 * A canvas cell as the solver names it, and back: the y-flip on one point.
 *
 * Its own function because the *snap* has to happen in the solver's frame. A
 * leaf of edge S is aligned to multiples of S measured from the floor, and a
 * slice whose height is not a multiple of S has no such line at the top of the
 * picture — so a box snapped in canvas cells and then flipped can land between
 * two leaf boundaries, which reads as the bound not being respected along the
 * top edge alone. Flipping first and snapping once is the fix, and it is why
 * the rubber band is derived from the snapped record rather than snapped
 * separately: there is exactly one rounding, in the frame that owns it.
 */
export function labSolverCell(
  at: readonly [number, number],
  ny: number,
): readonly [number, number] {
  return [at[0], ny - at[1]];
}

/** The record as a rectangle on the canvas: y down, origin at the picture's top-left. */
export function labRegionCanvasBox(record: RefinementRegionRecord, ny: number): RegionBox {
  return {
    min: [record.min_cells[0] ?? 0, ny - (record.max_cells[1] ?? 0)],
    max: [record.max_cells[0] ?? 0, ny - (record.min_cells[1] ?? 0)],
  };
}

/** The record reshaped to a canvas rectangle, flipped back to the solver's frame. */
export function labRegionFromCanvasBox(
  record: RefinementRegionRecord,
  box: RegionBox,
  ny: number,
): RefinementRegionRecord {
  return {
    ...record,
    min_cells: [box.min[0] ?? 0, ny - (box.max[1] ?? 0)],
    max_cells: [box.max[0] ?? 0, ny - (box.min[1] ?? 0)],
  };
}

/**
 * The box under a canvas point, topmost first.
 *
 * Reverse order because a later box is drawn over an earlier one: what the
 * reader sees under the pointer is the one they mean, and two overlapping
 * regions drawn to compare two bounds are exactly the case where picking the
 * first match would answer about the wrong one.
 */
export function labRegionAt(
  records: readonly RefinementRegionRecord[],
  ny: number,
  at: readonly [number, number] | null,
): RefinementRegionRecord | undefined {
  if (!at) return undefined;
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index]!;
    const box = labRegionCanvasBox(record, ny);
    if (at[0] >= (box.min[0] ?? 0) && at[0] <= (box.max[0] ?? 0)
      && at[1] >= (box.min[1] ?? 0) && at[1] <= (box.max[1] ?? 0)) return record;
  }
  return undefined;
}

/** True when a box has any area at all — a released click draws none. */
export function labRegionIsDrawn(record: RefinementRegionRecord): boolean {
  return (record.max_cells[0] ?? 0) > (record.min_cells[0] ?? 0)
    && (record.max_cells[1] ?? 0) > (record.min_cells[1] ?? 0);
}

/**
 * The next free box id, derived from the list rather than from a counter.
 *
 * A counter in a ref survives a scene change that empties the list, so the
 * first box of a new run used to be `advance-region-7`. Reading the highest
 * suffix back means an id names its position in the list the reader is looking
 * at, which is the rule `nextRefinementRegionId` follows in the studio.
 */
export function labNextRegionId(doc: LabRegionDocument): string {
  let highest = 0;
  for (const region of doc.regions) {
    const suffix = Number(region.id.slice(region.id.lastIndexOf("-") + 1));
    if (Number.isFinite(suffix)) highest = Math.max(highest, suffix);
  }
  return `advance-region-${highest + 1}`;
}

export const labRegionSpace: RegionSpace<LabRegionDocument, LabRegionDocument> = {
  axes: 2,
  lattice: (doc) => ({ dimensions: [doc.nx, doc.ny] }),
  list: (doc) => doc.regions.map(labRegionRecord),
  write: (doc, id, next) => {
    const replaced = doc.regions.some((region) => region.id === id);
    const regions = replaced
      ? doc.regions.flatMap((region) =>
        region.id !== id ? [region] : next ? [labRegionFromRecord(next)] : [])
      : next ? [...doc.regions, labRegionFromRecord(next)] : [...doc.regions];
    return { ...doc, regions };
  },
  nextId: labNextRegionId,
  // The studio's constant rather than a second 8 written here. The lab's own
  // `ENFORCEMENT_CAPACITY` was the same number for the same reason — the
  // projection's uniform tail — and two copies of a limit is how one of them
  // silently stops being the limit.
  capacity: OCTREE_REFINEMENT_REGION_CAPACITY,
  cellSizes: ADVANCE_RUNGS,
  // Two rungs off a four-rung ladder, which is what this lab shipped: the
  // slice is a couple of hundred cells across, so holding everything at a whole
  // brick would flatten the very reading the page is for.
  defaultCellSize_cells: 2,
  brick_cells: ADVANCE_BRICK_FINE,
  // No `cellEdge_mm`: the slice has no metre scale to quote, so the shared rows
  // render without the millimetre clause rather than with a fabricated number.
};
