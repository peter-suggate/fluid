import { EDITOR_AXES, type EditorAxis } from "../editor-axis-constraint";
import type { EditorHighlight } from "../editor-target";
import type { Vec3 } from "../model";
import type { FocusLock } from "./focus";

/** The axis a plane lock holds fixed, or the axis an axis lock runs along. */
export function focusLockAxis(axes: readonly EditorAxis[]): EditorAxis {
  return axes.length === 1 ? axes[0]! : EDITOR_AXES.find((axis) => !axes.includes(axis))!;
}

/** `ALONG Y`, `ON XZ PLANE`. */
export function focusLockLabel(axes: readonly EditorAxis[]): string {
  return axes.length === 1 ? `ALONG ${axes[0]!.toUpperCase()}`
    : `ON ${EDITOR_AXES.filter((axis) => axes.includes(axis)).join("").toUpperCase()} PLANE`;
}

/** The chord that entered the lock, which is also the chord that leaves it. */
export function focusLockKey(axes: readonly EditorAxis[]): string {
  const axis = focusLockAxis(axes).toUpperCase();
  return axes.length === 1 ? axis : `Shift+${axis}`;
}

/** The second line of the lock chip: how to leave it. */
export function focusLockHint(axes: readonly EditorAxis[]): string {
  return `${focusLockKey(axes)} or Esc releases`;
}

const along = (anchor: Vec3, axis: EditorAxis, distance: number): Vec3 => ({ ...anchor, [axis]: anchor[axis] + distance });

/**
 * The guide through the anchor: the axis line for an axis lock; for a plane
 * lock its two axis lines and a square around the anchor, so the plane reads
 * as a plane from any view. A drawing only — nothing picks against it.
 */
export function focusGuide(lock: FocusLock, extent_m: number): EditorHighlight {
  const { anchor, axes } = lock;
  const lines = axes.map((axis) => [along(anchor, axis, -extent_m), along(anchor, axis, extent_m)]);
  if (axes.length === 1) return { kind: "paths", paths: lines, anchor_m: anchor };
  const [a, b] = EDITOR_AXES.filter((axis) => axes.includes(axis)) as [EditorAxis, EditorAxis];
  const half = extent_m / 2;
  const corner = (da: number, db: number): Vec3 => along(along(anchor, a, da), b, db);
  const square = [corner(-half, -half), corner(half, -half), corner(half, half), corner(-half, half), corner(-half, -half)];
  return { kind: "paths", paths: [...lines, square], anchor_m: anchor };
}
