import type { Vec3 } from "./model";

/**
 * Blender's modal axis constraint, for every editor handle.
 *
 * A corner handle moves three sides at once, which is the right default for
 * roughing a box out and the wrong one for "make the water shallower without
 * making it wider". The 3D editors all answer this the same way — Blender, Maya
 * and Max each take an axis letter during the transform — so X, Y and Z name the
 * axes, and Shift+letter locks that axis out and leaves the other two: Blender's
 * plane constraint.
 *
 * The constraint is stored as the set of *allowed* axes rather than the locked
 * one, because that is the form a drag consumes: a constrained handle is exactly
 * the handle with its disallowed axes dropped, so a Y-constrained corner
 * resolves as the Y face it stands on — same snapping, same clamps, same code.
 * Nothing downstream of `constrainHandleAxes` knows a lock exists.
 *
 * The axes are the *entity's own*, not the world's. For the tank and the water
 * body those coincide; for a tumbling rigid body pressing X constrains along the
 * body's X, which is what a modal transform means in every editor that has one.
 */

export const EDITOR_AXES = Object.freeze(["x", "y", "z"] as const);
export type EditorAxis = (typeof EDITOR_AXES)[number];

/** Axes a drag may move. `undefined` is the unconstrained drag. */
export type AxisConstraint = readonly EditorAxis[] | undefined;

/** The axis an unmodified keypress names, or undefined for any other key. */
export function editorAxisForKey(key: string): EditorAxis | undefined {
  const normalized = key.toLowerCase();
  return EDITOR_AXES.find((axis) => axis === normalized);
}

function sameAxes(a: readonly EditorAxis[], b: readonly EditorAxis[]): boolean {
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
export function toggleAxisConstraint(
  current: AxisConstraint,
  axis: EditorAxis,
  mode: "axis" | "plane",
): AxisConstraint {
  const next = mode === "axis"
    ? [axis]
    : EDITOR_AXES.filter((candidate) => candidate !== axis);
  return current && sameAxes(current, next) ? undefined : next;
}

/** The axes a handle owns, narrowed by the constraint. */
export function constrainedAxes(
  axes: readonly EditorAxis[],
  constraint: AxisConstraint,
): readonly EditorAxis[] {
  return constraint ? axes.filter((axis) => constraint.includes(axis)) : axes;
}

/**
 * The world direction a drag rides when one degree of freedom is left, whether
 * because the handle owns a single axis or because the constraint reduced an
 * edge or a corner to one. With two or more axes free the drag belongs in the
 * camera plane, and this returns undefined.
 *
 * Riding the axis line rather than the camera plane is the whole point of the
 * constraint: it is what turns a diagonal pointer sweep into a movement of one
 * side by the amount the pointer travelled along that side's own direction.
 */
export function axisDragDirection(
  axes: readonly EditorAxis[],
  constraint: AxisConstraint,
): Vec3 | undefined {
  const free = constrainedAxes(axes, constraint);
  if (free.length !== 1) return undefined;
  const axis = free[0]!;
  return { x: axis === "x" ? 1 : 0, y: axis === "y" ? 1 : 0, z: axis === "z" ? 1 : 0 };
}

/**
 * The constraint in Blender's own words: an axis constraint reads as the axis
 * the drag runs *along*, a plane constraint as the axis it *locks out*, because
 * that is the axis the user named when they pressed the key.
 */
export function axisConstraintLabel(constraint: AxisConstraint): string {
  if (!constraint || constraint.length === 0 || constraint.length >= EDITOR_AXES.length) return "";
  if (constraint.length === 1) return `ALONG ${constraint[0]!.toUpperCase()}`;
  return `LOCKING ${EDITOR_AXES.filter((axis) => !constraint.includes(axis))
    .map((axis) => axis.toUpperCase()).join("")}`;
}
