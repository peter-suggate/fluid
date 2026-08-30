import { createBodyDescription, defaultBodyDimensions_m } from "./rigid-body";
import { RIGID_MINIMUM_HALF_SIZE_M } from "./editor-rigid-body";
import { sceneShape, UNUSED_DIMENSION } from "./scene-shape";
import type { RigidBodyDescription, RigidShape, Vec3 } from "./model";

/**
 * How big the next body is, before there is a body.
 *
 * Every other size in the editor belongs to something that exists: a selected
 * body's fields patch that body's document entry, and a handle drag writes the
 * one it is attached to. A *placement* has no subject yet — it is the template
 * the next click instantiates — so its numbers live in the session's UI state
 * beside `placementShape`, which is the same kind of fact and was already there.
 *
 * Deliberately not in the document. A scene that carried "the size the next
 * sphere would be" would be a scene whose file changes when a reader nudges a
 * field and then drops nothing, and two documents that describe identical water
 * would stop comparing equal.
 */

/**
 * Sizes a reader has typed, per shape.
 *
 * Sparse on purpose: a shape absent from the record takes
 * `defaultBodyDimensions_m`, so raising a default raises it for everyone who has
 * not overridden it rather than for everyone who has never opened the row. And
 * per shape rather than one triple, because the three floats mean different
 * things for each — a cup's `z` is a wall thickness and a box's is an extent —
 * so one shared triple would silently reinterpret a number on every switch.
 */
export type PlacementDimensions = Partial<Record<RigidShape, Vec3>>;

/** The size the next body of this shape would be. */
export function placementDimensions_m(
  shape: RigidShape,
  placed: PlacementDimensions,
): Vec3 {
  return placed[shape] ?? defaultBodyDimensions_m(shape);
}

/**
 * The body a click would drop, at the size the strip is showing.
 *
 * The one seam. Three call sites want this — the hover preview's radius, the
 * rest-position solve, and the spawn itself — and each of them built its own
 * template from `createBodyDescription` before, which is exactly how a preview
 * circle ends up promising a different body from the one that lands.
 */
export function placementBodyDescription(
  shape: RigidShape,
  placed: PlacementDimensions,
  index: number,
  containerHeight_m: number,
): RigidBodyDescription {
  return {
    ...createBodyDescription(shape, index, containerHeight_m),
    dimensions_m: placementDimensions_m(shape, placed),
  };
}

/** One editable number of a shape's own size. */
export interface PlacementField {
  /** Which of the three floats it writes. */
  readonly axis: "x" | "y" | "z";
  /** The shape's own word for it — "outer radius", "extent y". */
  readonly label: string;
  /** The initial that identifies it beside its siblings on one line. */
  readonly tag: string;
  readonly value: number;
  readonly step: number;
  readonly min: number;
  /** The full triple this shape would take with that number typed in. */
  readonly apply: (value: number) => Vec3;
}

/** Nudge, in metres. The same grain the selected body's Size field steps at. */
const PLACEMENT_STEP_M = 0.005;

/**
 * The letters a dimension is labelled with when it stands beside its siblings.
 *
 * The last word of the shape's own label, which is the word that separates it
 * from the others: "outer radius" and "wall thickness" are R and T, "extent x"
 * is X, "segment length" is L. Derived rather than tabulated because a second
 * list of initials beside `dimensionLabels` is a list that can disagree with it
 * — and the initials only have to be distinct *within one shape*, which every
 * shape in the table satisfies.
 */
function dimensionTag(label: string): string {
  return label.slice(label.lastIndexOf(" ") + 1).slice(0, 1).toUpperCase();
}

/**
 * The numbers this shape actually has, in the order it stores them.
 *
 * A sphere has one, a cylinder two, a cup three, and the slots a shape does not
 * use are skipped rather than shown empty — `dimensions_m` is always three
 * floats, and offering a sphere a Y box would be offering a float nothing
 * reads. Which is why the labels are read from the shape table: a shape added
 * there arrives here with its own vocabulary and no edit to this file.
 */
export function placementFields(
  shape: RigidShape,
  placed: PlacementDimensions,
): readonly PlacementField[] {
  const dimensions = placementDimensions_m(shape, placed);
  const axes = ["x", "y", "z"] as const;
  return sceneShape(shape).dimensionLabels
    .map((label, index) => ({ label, axis: axes[index]! }))
    .filter((slot) => slot.label !== UNUSED_DIMENSION)
    .map(({ label, axis }) => ({
      axis,
      label,
      tag: dimensionTag(label),
      value: dimensions[axis],
      step: PLACEMENT_STEP_M,
      // One floor for all of them, and it is the gizmo's: nothing a handle can
      // drag a body down to should be unreachable by typing, and nothing typed
      // should make a body the collision code cannot support.
      min: RIGID_MINIMUM_HALF_SIZE_M,
      apply: (value: number) => ({ ...dimensions, [axis]: value }),
    }));
}
