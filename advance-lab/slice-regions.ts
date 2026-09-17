/**
 * An enforcement region as a box you can grab, in the slice's own cells.
 *
 * The 3-D editor already knows how to edit a refinement region — draw it,
 * select it, drag its faces, snap every edge onto the lattice its own floor
 * cell defines — and `lib/core/editor-refinement-region.ts` is where that
 * behaviour lives. None of that file can be imported here: it is written in
 * metres against a `SceneDescription`, and reaching it would pull the scene
 * document, the entity protocol and the octree's region rules into a page whose
 * regions are two-tuples of finest cells on a 2-D cut. So the *behaviour* is
 * mirrored and the code is not, and this file is the whole of the mirror:
 *
 *   - the snap step is the region's own smallest allowed cell, never the finest
 *     cell (a box on that lattice contains whole leaves of that size, and
 *     snapping finer loses a shell of cells all the way round every region);
 *   - a drawn box snaps *outward*, so it covers every cell the drag touched;
 *   - a resize never inverts the box and never leaves the lattice;
 *   - the box stays inside the slice, which is this lab's container.
 *
 * Pure, and in cells: nothing here reads a store, a camera or the DOM, so every
 * rule above is exercised directly by `slice-regions.test.ts`.
 *
 * ## Two frames, and which is which
 *
 * A region is stored the way the solver reads it — `minimumFine`/`maximumFine`
 * in lattice cells with y running *up* from the floor. The picture is drawn the
 * way a canvas is, with y running down. Every function here that takes or
 * returns a `SliceBox` is in canvas cells; `regionBox` and `regionWithBox` are
 * the only two places the flip happens, so no caller has to remember it.
 */
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";

/**
 * The selection id prefix, mirrored from `editor-refinement-region.ts`.
 *
 * Mirrored rather than imported for the reason at the top of this file, and
 * pinned against the studio's own constant in `slice-regions.test.ts` so the
 * two cannot drift: a region selected in the lab and a region selected in the
 * studio are the same kind of selection, and `EditorSelection` is shared.
 */
export const SLICE_REGION_SELECTION_PREFIX = "refinement-region-";

export function sliceRegionSelectionId(regionId: string): string {
  return `${SLICE_REGION_SELECTION_PREFIX}${regionId}`;
}

export function sliceRegionIdFromSelection(selectionId: string | undefined): string | undefined {
  return selectionId?.startsWith(SLICE_REGION_SELECTION_PREFIX)
    ? selectionId.slice(SLICE_REGION_SELECTION_PREFIX.length)
    : undefined;
}

/** A rectangle in canvas finest cells: y down, origin at the picture's top-left. */
export interface SliceBox {
  readonly minFine: readonly [number, number];
  readonly maxFine: readonly [number, number];
}

/** The slice a box has to stay inside — this lab's container. */
export interface SliceLattice {
  readonly nx: number;
  readonly ny: number;
}

/** The stored region, as a box on the canvas. */
export function regionBox(region: AdvanceRefinementRegion, ny: number): SliceBox {
  return {
    minFine: [region.minimumFine[0], ny - region.maximumFine[1]],
    maxFine: [region.maximumFine[0], ny - region.minimumFine[1]],
  };
}

/** The same region, reshaped to a canvas box. */
export function regionWithBox(region: AdvanceRefinementRegion, box: SliceBox,
  ny: number): AdvanceRefinementRegion {
  return {
    ...region,
    minimumFine: [box.minFine[0], ny - box.maxFine[1]],
    maximumFine: [box.maxFine[0], ny - box.minFine[1]],
  };
}

/**
 * The box under a point, topmost first.
 *
 * Reverse order because a later box is drawn over an earlier one: what the
 * reader sees under the pointer is the one they mean, and the two overlapping
 * regions a reader draws to compare two bounds are exactly the case where
 * picking the first match would answer about the wrong one.
 */
export function regionAt(regions: readonly AdvanceRefinementRegion[], ny: number,
  at: readonly [number, number] | null): AdvanceRefinementRegion | undefined {
  if (!at) return undefined;
  for (let index = regions.length - 1; index >= 0; index -= 1) {
    const region = regions[index]!;
    const box = regionBox(region, ny);
    if (at[0] >= box.minFine[0] && at[0] <= box.maxFine[0]
      && at[1] >= box.minFine[1] && at[1] <= box.maxFine[1]) return region;
  }
  return undefined;
}

/**
 * Which part of a selected box a handle grabs.
 *
 * Eight of them, named by the corner or edge they sit on in canvas terms, plus
 * the body — the same set `boxHandles` offers in three dimensions, minus the
 * axis the slice does not have. `"body"` is the move, which the 3-D editor
 * gives its own `moveHandles`; here it is the box's own interior, because a
 * rectangle small enough to need a separate move handle is a rectangle a reader
 * would rather redraw.
 */
export type SliceHandleId =
  | "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "body";

/** The eight reshaping handles, in the order they are drawn. */
export const SLICE_RESIZE_HANDLES: readonly SliceHandleId[] =
  Object.freeze(["nw", "n", "ne", "e", "se", "s", "sw", "w"]);

/** Where a handle sits on its box, as a fraction of each side. */
export function handleAnchor(handle: SliceHandleId): readonly [number, number] {
  const x = handle.includes("w") ? 0 : handle.includes("e") ? 1 : 0.5;
  const y = handle.startsWith("n") ? 0 : handle.startsWith("s") ? 1 : 0.5;
  return [x, y];
}

/** The CSS cursor a handle promises, so the box says how it reshapes. */
export function handleCursor(handle: SliceHandleId): string {
  if (handle === "body") return "move";
  return `${handle}-resize`;
}

const snap = (value: number, step: number): number => Math.round(value / step) * step;

const clamp = (value: number, low: number, high: number): number =>
  Math.min(Math.max(value, low), high);

/**
 * The box a drag from `anchor` to `at` draws, snapped outward onto `step`.
 *
 * Outward and not nearest, for the reason the 3-D editor gives: a region is an
 * instruction about an area the reader indicated, and rounding a 1.4-cell drag
 * down to one cell hands back a box visibly smaller than the rectangle they let
 * go of. The slice wins over the lattice at the edges — a domain whose cell
 * count is not a multiple of the step still gets a box that ends at the wall.
 */
export function draftRegionBox(anchor: readonly [number, number],
  at: readonly [number, number], step: number, lattice: SliceLattice): SliceBox {
  const lowX = clamp(Math.floor(Math.min(anchor[0], at[0]) / step) * step, 0, lattice.nx);
  const highX = clamp(Math.ceil(Math.max(anchor[0], at[0]) / step) * step, 0, lattice.nx);
  const lowY = clamp(Math.floor(Math.min(anchor[1], at[1]) / step) * step, 0, lattice.ny);
  const highY = clamp(Math.ceil(Math.max(anchor[1], at[1]) / step) * step, 0, lattice.ny);
  return { minFine: [lowX, lowY], maxFine: [highX, highY] };
}

/**
 * The box after dragging one handle to `at`.
 *
 * The opposite side is held: a resize moves the edge you took hold of and
 * nothing else, which is what makes a corner drag a two-axis version of the
 * same gesture rather than a scale about the centre. The moved edge snaps to
 * the step, stays inside the slice, and is never allowed past the held one —
 * one step of thickness is the floor, because a region thinner than the cell it
 * is asking for contains none of them.
 */
export function resizedRegionBox(box: SliceBox, handle: SliceHandleId,
  at: readonly [number, number], step: number, lattice: SliceLattice): SliceBox {
  if (handle === "body") return box;
  let [minX, minY] = box.minFine;
  let [maxX, maxY] = box.maxFine;
  if (handle.includes("w")) minX = Math.min(clamp(snap(at[0], step), 0, lattice.nx), maxX - step);
  if (handle.includes("e")) maxX = Math.max(clamp(snap(at[0], step), 0, lattice.nx), minX + step);
  if (handle.startsWith("n")) minY = Math.min(clamp(snap(at[1], step), 0, lattice.ny), maxY - step);
  if (handle.startsWith("s")) maxY = Math.max(clamp(snap(at[1], step), 0, lattice.ny), minY + step);
  return { minFine: [minX, minY], maxFine: [maxX, maxY] };
}

/**
 * The box after dragging its body by `(dx, dy)` cells.
 *
 * Translated, never reshaped: the travel is snapped to the step and then the
 * whole box is pushed back inside the slice, so a box dragged into a wall stops
 * against it at its own size instead of being squashed against it. That is the
 * `moveBoxWithinLimits` bargain, in two dimensions.
 */
export function movedRegionBox(box: SliceBox, dx: number, dy: number,
  step: number, lattice: SliceLattice): SliceBox {
  const width = box.maxFine[0] - box.minFine[0];
  const height = box.maxFine[1] - box.minFine[1];
  const minX = clamp(snap(box.minFine[0] + dx, step), 0, Math.max(0, lattice.nx - width));
  const minY = clamp(snap(box.minFine[1] + dy, step), 0, Math.max(0, lattice.ny - height));
  return { minFine: [minX, minY], maxFine: [minX + width, minY + height] };
}

/** True when a box has any area at all — a released click draws none. */
export function regionBoxIsDrawn(box: SliceBox): boolean {
  return box.maxFine[0] > box.minFine[0] && box.maxFine[1] > box.minFine[1];
}

/**
 * The region with a new floor cell, and the ceiling that follows it.
 *
 * The 3-D rule, restated once here so both halves of the lab obey it: a ceiling
 * that was equal to the floor is a region *held at one tier*, and it follows
 * the floor wherever the floor goes. A wider authored ceiling is kept, and only
 * ever lifted to stay above the floor it now has to be above.
 */
export function regionWithFloor(region: AdvanceRefinementRegion,
  cells: number): AdvanceRefinementRegion {
  return {
    ...region,
    minimumCellWidth: cells,
    ...(region.maximumCellWidth === undefined ? {} : {
      maximumCellWidth: region.maximumCellWidth === region.minimumCellWidth
        ? cells : Math.max(cells, region.maximumCellWidth),
    }),
  };
}

/** The region with a ceiling, or with none — AUTO is the absence of one. */
export function regionWithCeiling(region: AdvanceRefinementRegion,
  cells: number | undefined): AdvanceRefinementRegion {
  if (cells === undefined) {
    // Deleted rather than set to `undefined`: the controller reads the field's
    // presence, and a key that is there holding nothing is not the same
    // instruction as a region that never named a ceiling.
    const rest: Record<string, unknown> = { ...region };
    delete rest.maximumCellWidth;
    return rest as unknown as AdvanceRefinementRegion;
  }
  return { ...region, maximumCellWidth: Math.max(cells, region.minimumCellWidth) };
}

/** What a box enforces, in the lattice's own words. The tag, and the tip. */
export function regionCaption(region: AdvanceRefinementRegion): string {
  return region.maximumCellWidth === region.minimumCellWidth
    ? `held at ${region.minimumCellWidth}`
    : `≥ ${region.minimumCellWidth} cell${region.minimumCellWidth === 1 ? "" : "s"}`;
}
