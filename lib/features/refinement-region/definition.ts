import { BRICK_FINE_CELLS } from "../../core/sparse-brick-geometry";
import {
  OCTREE_REFINEMENT_REGION_CAPACITY,
  OCTREE_REFINEMENT_REGION_CELL_SIZES,
  REFINEMENT_REGION_RULES,
  type RefinementRegionRule,
} from "../../core/refinement-regions";

/**
 * A refinement box, as one record both hosts store.
 *
 * Two things were duplicated before this existed, and only one of them looked
 * like duplication. The obvious one is the code: the 3-D editor snapped,
 * resized and re-tiered a box in `lib/core/editor-refinement-region.ts` and the
 * 2-D advance lab mirrored every rule by hand in `advance-lab/slice-regions.ts`,
 * down to restating the selection-id prefix. The one underneath is the *record*:
 * the studio stored metres against a `SceneDescription` and the lab stored
 * `minimumFine`/`maximumFine` tuples against a Rust world, so there was nothing
 * for one policy to be written against.
 *
 * So the record is **finest cells**, and it is **N-dimensional**. Cells because
 * that is the frame both worlds already agree in — the studio converts metres
 * to cells to hand the octree a region at all, and the lab never leaves cells.
 * N-dimensional because the only thing the policy does per axis is the same
 * thing, and a `Vec3` in the shared layer would have forced the lab to invent a
 * third axis it does not have.
 */
export interface RefinementRegionRecord {
  readonly id: string;
  /**
   * Length 2 in the lab, 3 in the studio. Axis-major, y-up, origin at the
   * lattice corner — which is to say, the frame the solver reads a region in.
   */
  readonly min_cells: readonly number[];
  readonly max_cells: readonly number[];
  readonly rule: RefinementRegionRule;
  /** Power of two. The smallest pressure cell this box allows. */
  readonly minimumCellSize_cells: number;
  /** Power of two. Absent is AUTO: evidence decides how far quiet fluid coarsens. */
  readonly maximumCellSize_cells?: number;
}

/** Finest cells across the domain, per axis. */
export interface RegionLattice {
  readonly dimensions: readonly number[];
}

/** A box in finest cells, as the policy passes one around. */
export interface RegionBox {
  readonly min: readonly number[];
  readonly max: readonly number[];
}

/**
 * How a host stores, names and writes back the records. One adapter per host.
 *
 * The adapter is where every host-specific unit lives — metres and `Vec3` on
 * one side, canvas cells and a y-flip on the other — so the policy beside this
 * file is arithmetic on integers and nothing else.
 */
export interface RegionSpace<Doc, Patch = Doc> {
  readonly axes: 2 | 3;
  lattice(doc: Doc): RegionLattice;
  list(doc: Doc): readonly RefinementRegionRecord[];
  /** The document with one record replaced, added, or — with `undefined` — dropped. */
  write(doc: Doc, id: string, next: RefinementRegionRecord | undefined): Patch;
  nextId(doc: Doc): string;
  readonly capacity: number;
  /** The ladder this host offers. S: [1,2,4,8,16,32]; L: [1,2,4,8]. */
  readonly cellSizes: readonly number[];
  /**
   * The rung a box drawn here carries while nobody has chosen one.
   *
   * On `RegionSpace` rather than in the shared draft because the two hosts
   * disagree about it for reasons that are each host's own: the studio's 8 is
   * one brick on a 32-rung ladder, the lab's 2 is the second of four rungs on a
   * lattice two orders of magnitude smaller. One number in the store would have
   * silently moved whichever host lost. See `UIState.regionDraft`.
   */
  readonly defaultCellSize_cells: number;
  /** Brick width in finest cells. Both are 8 — see `BRICK_FINE_CELLS`. */
  readonly brick_cells: number;
  /**
   * World edge of one finest cell, in millimetres, when this host has one.
   *
   * The 3-D option hints read "`8³` finest cells · 40 mm edge", which is the
   * one place a region's controls speak a physical unit. The lab's slice has no
   * metre scale to quote, so it omits this and the same rows render without the
   * clause rather than with a fabricated number.
   */
  cellEdge_mm?(doc: Doc): number | undefined;
}

/**
 * How a selected region is named in `EditorSelection`.
 *
 * Here rather than in either host because it is the one thing both hosts
 * already agreed on and neither owned: `advance-lab/slice-regions.ts` restated
 * this string by hand and pinned the copy against the studio's with a test,
 * which is the whole duplication story in one constant. `EditorSelection` is
 * shared, so a region selected in the lab and one selected in the studio are
 * the same kind of selection and must spell their ids the same way.
 */
export const REFINEMENT_REGION_SELECTION_PREFIX = "refinement-region-";

export function refinementRegionSelectionId(regionId: string): string {
  return `${REFINEMENT_REGION_SELECTION_PREFIX}${regionId}`;
}

/**
 * The region a selection names, or nothing when it names something else.
 *
 * Takes `undefined` as well as a string because every caller is reading a
 * selection that may be absent, and making each of them guard first was the
 * shape both hosts independently worked around.
 */
export function refinementRegionIdFromSelection(
  selectionId: string | undefined,
): string | undefined {
  return selectionId?.startsWith(REFINEMENT_REGION_SELECTION_PREFIX)
    ? selectionId.slice(REFINEMENT_REGION_SELECTION_PREFIX.length)
    : undefined;
}

/** Whether another region can be drawn, or the host's tail is already full. */
export function regionCapacityRemaining<Doc>(
  space: Pick<RegionSpace<Doc, unknown>, "capacity" | "list">,
  doc: Doc,
): number {
  return Math.max(0, space.capacity - space.list(doc).length);
}

export {
  BRICK_FINE_CELLS,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  OCTREE_REFINEMENT_REGION_CELL_SIZES,
  REFINEMENT_REGION_RULES,
  type RefinementRegionRule,
};
