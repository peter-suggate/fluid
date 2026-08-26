/**
 * Selections, and the click-versus-drag rule.
 *
 * This file used to be the tool model: an `EditorTool` union, an `EDITOR_TOOLS`
 * table, and a viewport pointer machine that dispatched on the armed one. All of
 * that is gone. What a press means is now decided by `editor-gesture-catalog.ts`
 * against the target under it, and the handful of strokes that genuinely need a
 * mode are gestures there rather than tools here.
 *
 * What is left is the vocabulary that outlived the tools, because none of it was
 * ever about them: what a selection can be about, and the one question that
 * separates a click from a drag.
 */

/**
 * What a selection can be about.
 *
 * The tank and the water body are selection kinds like any other even though no
 * click can reach them — BOUNDS surfaces them instead. Giving them a kind is
 * what lets one handle layer, one axis lock and one commit path serve every
 * editable thing; see lib/editor-entity.ts.
 */
export type EditorSelectionKind =
  | "body"
  | "terrain-feature"
  | "inflow"
  | "scenery"
  | "tank"
  | "fluid-body"
  | "vessel-rim"
  | "refinement-region"
  // The one selection kind that is not a thing in the document: a box of solid
  // voxels a drag swept out. It is an entity anyway — it has extents, a label
  // and a ring — because being an entity is what makes it act like everything
  // else the editor selects. See `editor-voxel-region.ts`.
  | "voxel-region";

/**
 * Generalization of the original `selectedBodyId`. Later phases add terrain
 * features and inflow nozzles without another parallel id field.
 */
export interface EditorSelection {
  readonly kind: EditorSelectionKind;
  readonly id: string;
}

export function bodySelection(id: string | undefined): EditorSelection | undefined {
  return id === undefined ? undefined : { kind: "body", id };
}

/** The body id a selection refers to, or undefined for non-body selections. */
export function selectedBodyIdOf(selection: EditorSelection | undefined): string | undefined {
  return selection?.kind === "body" ? selection.id : undefined;
}

/**
 * Pointer travel, in CSS pixels, below which a press-and-release is still a
 * click. Wide enough to absorb the jitter of a real click on a trackpad,
 * narrow enough that a deliberate camera nudge is read as a drag.
 */
export const CLICK_SLOP_PX = 4;

/**
 * Whether a press and its release were the same point — the one question that
 * separates a click from a drag, asked identically wherever a gesture can be
 * either.
 *
 * Every press in the viewport is provisional: a press on the background is a
 * camera orbit until it turns out not to have moved, and a press on a body is a
 * throw until the same. Both resolve here so a gesture cannot be a click for one
 * of them and a drag for the other.
 */
export function pointerStayedWithinClickSlop(travelX_px: number, travelY_px: number): boolean {
  return Math.hypot(travelX_px, travelY_px) <= CLICK_SLOP_PX;
}

/**
 * Whether releasing a viewport gesture on empty space should clear the
 * selection.
 *
 * The pointer machine only falls through to an orbit when nothing under the
 * cursor claimed the press, so an orbit that never moved *is* the user clicking
 * the background — the one gesture that always deselects, whatever is selected
 * and whatever tool is armed. Pans are excluded: shift or the middle button
 * asks for navigation explicitly, so a pan is never a pick.
 */
export function emptySpaceClickDeselects(
  action: "orbit" | "pan",
  travelX_px: number,
  travelY_px: number,
): boolean {
  if (action !== "orbit") return false;
  return pointerStayedWithinClickSlop(travelX_px, travelY_px);
}
