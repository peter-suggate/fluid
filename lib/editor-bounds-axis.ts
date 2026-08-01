import type { FluidBodyHandle } from "./editor-fluid-body";
import type { Vec3 } from "./model";

/**
 * Blender's modal axis constraint, for the bounds handles.
 *
 * A corner handle moves three sides at once, which is the right default for
 * roughing a box out and the wrong one for "make the water shallower without
 * making it wider". The 3D editors all answer this the same way — Blender, Maya
 * and Max each take an axis letter during the transform — so while BOUNDS is
 * armed X, Y and Z name the axes, and Shift+letter locks that axis out and
 * leaves the other two: Blender's plane constraint.
 *
 * The constraint is stored as the set of *allowed* axes rather than the locked
 * one, because that is the form the drag consumes: a constrained handle is
 * exactly the handle with its disallowed sides dropped, so a Y-constrained
 * corner resolves as the Y face it stands on — same snapping, same clamps, same
 * code. Nothing downstream of `constrainBoundsHandle` knows a lock exists.
 */

export const BOUNDS_AXES = Object.freeze(["x", "y", "z"] as const);
export type BoundsAxis = (typeof BOUNDS_AXES)[number];

/** Axes a bounds drag may move. `undefined` is the unconstrained drag. */
export type BoundsAxisConstraint = readonly BoundsAxis[] | undefined;

/** The axis an unmodified keypress names, or undefined for any other key. */
export function boundsAxisForKey(key: string): BoundsAxis | undefined {
  const normalized = key.toLowerCase();
  return BOUNDS_AXES.find((axis) => axis === normalized);
}

function sameAxes(a: readonly BoundsAxis[], b: readonly BoundsAxis[]): boolean {
  return a.length === b.length && a.every((axis) => b.includes(axis));
}

/**
 * What pressing an axis key does to the current constraint.
 *
 * Pressing the constraint you already hold releases it, so the key that entered
 * the state is the key that leaves it — the same contract the tool shortcuts
 * keep. Pressing a different axis switches outright rather than accumulating:
 * "X then Y" means the user changed their mind, not that they want the XY
 * plane, which is what Shift is for.
 */
export function toggleBoundsAxisConstraint(
  current: BoundsAxisConstraint,
  axis: BoundsAxis,
  mode: "axis" | "plane",
): BoundsAxisConstraint {
  const next = mode === "axis"
    ? [axis]
    : BOUNDS_AXES.filter((candidate) => candidate !== axis);
  return current && sameAxes(current, next) ? undefined : next;
}

/** The axes a handle owns, narrowed by the constraint. */
export function boundsDragAxes(
  handle: Pick<FluidBodyHandle, "sides">,
  constraint: BoundsAxisConstraint,
): readonly BoundsAxis[] {
  const owned = BOUNDS_AXES.filter((axis) => handle.sides[axis] !== undefined);
  return constraint ? owned.filter((axis) => constraint.includes(axis)) : owned;
}

/**
 * The handle a constrained drag actually moves.
 *
 * A constraint naming no axis this handle owns — Y while dragging an x face —
 * yields a handle with no sides at all, and the drag holds the box still. That
 * is the honest outcome rather than a silent fallback to an axis the user did
 * not ask for; the viewport says so on screen instead of guessing.
 */
export function constrainBoundsHandle(
  handle: FluidBodyHandle,
  constraint: BoundsAxisConstraint,
): FluidBodyHandle {
  if (!constraint) return handle;
  const sides: Partial<Record<BoundsAxis, "min" | "max">> = {};
  for (const axis of boundsDragAxes(handle, constraint)) sides[axis] = handle.sides[axis];
  return { ...handle, sides };
}

/**
 * The world direction a drag rides when one degree of freedom is left, whether
 * because the handle is a face or because the constraint reduced an edge or a
 * corner to one axis. With two or more axes free the drag belongs in the camera
 * plane, and this returns undefined.
 *
 * Riding the axis line rather than the camera plane is the whole point of the
 * constraint: it is what turns a diagonal pointer sweep into a movement of one
 * side by the amount the pointer travelled along that side's own direction.
 */
export function boundsDragAxisDirection(
  handle: Pick<FluidBodyHandle, "sides">,
  constraint: BoundsAxisConstraint,
): Vec3 | undefined {
  const axes = boundsDragAxes(handle, constraint);
  if (axes.length !== 1) return undefined;
  const axis = axes[0]!;
  return { x: axis === "x" ? 1 : 0, y: axis === "y" ? 1 : 0, z: axis === "z" ? 1 : 0 };
}

/**
 * The constraint in Blender's own words: an axis constraint reads as the axis
 * the drag runs *along*, a plane constraint as the axis it *locks out*, because
 * that is the axis the user named when they pressed the key.
 */
export function boundsAxisConstraintLabel(constraint: BoundsAxisConstraint): string {
  if (!constraint || constraint.length === 0 || constraint.length >= BOUNDS_AXES.length) return "";
  if (constraint.length === 1) return `ALONG ${constraint[0]!.toUpperCase()}`;
  return `LOCKING ${BOUNDS_AXES.filter((axis) => !constraint.includes(axis))
    .map((axis) => axis.toUpperCase()).join("")}`;
}
