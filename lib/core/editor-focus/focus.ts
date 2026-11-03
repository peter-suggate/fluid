import { EDITOR_AXES, type AxisConstraint, type EditorAxis } from "../editor-axis-constraint";
import type { EditorRay } from "../editor-entity";
import type { Vec3 } from "../model";

/**
 * The scene focus: the one point every placing interaction acts at.
 *
 * Free, a focus is whatever the consumer's own pick answers — the Pen's
 * nearest visible surface, a drop's resting place — and this module does not
 * touch it. Locked, it is the pointer ray projected onto a world axis line or a
 * world plane through an anchor. The lock is the editor's `axisConstraint`,
 * read as the axes the focus may move along: one axis is a line, two are the
 * plane that holds the third fixed. See docs/plans/scene-focus.md.
 *
 * Everything here is pure; `FocusSession` is the per-stroke bookkeeping of the
 * anchor and is the only stateful piece.
 */

/** Where the hand acts. `locked` is set when a lock placed it rather than a surface. */
export interface FocusPoint {
  readonly point: Vec3;
  readonly normal: Vec3;
  readonly distance_m?: number;
  readonly locked?: boolean;
}

/** A consumer's own unlocked answer: a surface pick, a drop plane, a resting place. */
export type FocusPick = (ray: EditorRay) => FocusPoint | undefined;

/** A lock with somewhere to stand. */
export interface FocusLock {
  readonly axes: readonly EditorAxis[];
  readonly anchor: Vec3;
}

/**
 * What a gesture host hands a tool: resolve a ray through whatever lock is up,
 * falling back to the tool's own pick. A tool without one uses its pick.
 */
export interface ToolFocus {
  resolve(ray: EditorRay, pick: FocusPick): FocusPoint | undefined;
}

/** The axes a constraint lets the focus move along, or undefined when it does not lock anything. */
export function focusAxes(constraint: AxisConstraint): readonly EditorAxis[] | undefined {
  if (!constraint || constraint.length === 0 || constraint.length >= EDITOR_AXES.length) return undefined;
  return constraint;
}

const unit = (axis: EditorAxis): Vec3 => ({ x: axis === "x" ? 1 : 0, y: axis === "y" ? 1 : 0, z: axis === "z" ? 1 : 0 });
const dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
const at = (ray: EditorRay, t: number): Vec3 => ({
  x: ray.origin.x + ray.direction.x * t, y: ray.origin.y + ray.direction.y * t, z: ray.origin.z + ray.direction.z * t,
});
const finite = (v: Vec3) => Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);

/** Smallest |cos| between the ray and a plane (or 1 - cos² for an axis) that still resolves. */
const GRAZING = 1e-3;

/**
 * Project a pointer ray onto the lock through `anchor`.
 *
 * Undefined when the ray runs parallel to the plane or axis, meets it behind
 * the eye, or meets it further than `reach_m` from the anchor — a grazing plane
 * would otherwise throw the focus kilometres away.
 */
export function projectFocus(ray: EditorRay, axes: readonly EditorAxis[], anchor: Vec3,
  reach_m = Number.POSITIVE_INFINITY): FocusPoint | undefined {
  const length = Math.sqrt(dot(ray.direction, ray.direction));
  if (!(length > 0) || axes.length === 0 || axes.length >= 3) return undefined;
  const direction = { x: ray.direction.x / length, y: ray.direction.y / length, z: ray.direction.z / length };
  let point: Vec3 | undefined, normal: Vec3, t: number;
  if (axes.length === 2) {
    const fixed = EDITOR_AXES.find((axis) => !axes.includes(axis))!;
    const n = unit(fixed);
    const facing = dot(direction, n);
    if (Math.abs(facing) < GRAZING) return undefined;
    t = (anchor[fixed] - ray.origin[fixed]) / direction[fixed];
    point = { ...at({ origin: ray.origin, direction }, t), [fixed]: anchor[fixed] };
    normal = { x: 0, y: 0, z: 0, [fixed]: facing > 0 ? -1 : 1 };
  } else {
    const axis = axes[0]!;
    const u = unit(axis);
    const alignment = dot(direction, u);
    const denominator = 1 - alignment * alignment;
    if (!(denominator > GRAZING)) return undefined;
    const offset = { x: anchor.x - ray.origin.x, y: anchor.y - ray.origin.y, z: anchor.z - ray.origin.z };
    const alongAxis = dot(u, offset), alongRay = dot(direction, offset);
    // Ray parameter of the closest approach, and the axis parameter.
    t = (alongRay - alignment * alongAxis) / denominator;
    const s = (alignment * alongRay - alongAxis) / denominator;
    point = { ...anchor, [axis]: anchor[axis] + s };
    // Perpendicular to the axis, towards the eye.
    const back = { x: -direction.x + alignment * u.x, y: -direction.y + alignment * u.y, z: -direction.z + alignment * u.z };
    const size = Math.sqrt(dot(back, back));
    normal = size > 0 ? { x: back.x / size, y: back.y / size, z: back.z / size } : { x: 0, y: 1, z: 0 };
  }
  if (!(t > 0) || !finite(point)) return undefined;
  const away = { x: point.x - anchor.x, y: point.y - anchor.y, z: point.z - anchor.z };
  if (Math.sqrt(dot(away, away)) > reach_m) return undefined;
  return { point, normal, distance_m: t, locked: true };
}

/**
 * One sample: free when there is no lock, projected when there is.
 * A lock with no anchor yet is anchored by `pick` — see `FocusSession`.
 */
export function resolveFocus(ray: EditorRay, lock: FocusLock | undefined, pick: FocusPick,
  reach_m?: number): FocusPoint | undefined {
  return lock ? projectFocus(ray, lock.axes, lock.anchor, reach_m) : pick(ray);
}

const sameAxes = (a: readonly EditorAxis[] | undefined, b: readonly EditorAxis[] | undefined) =>
  a === b || (a !== undefined && b !== undefined && a.length === b.length && a.every((axis) => b.includes(axis)));

export interface FocusSessionOptions {
  /** The lock as it stands now; read on every sample so a key mid-drag takes effect on the next one. */
  readonly constraint: () => AxisConstraint;
  /** The last stroke's anchor: where a press over open air stands when a lock is up. */
  readonly fallback?: Vec3;
  /** How far from the anchor a locked focus may land. */
  readonly reach_m?: number;
}

/**
 * The anchor bookkeeping of one hover sample or one stroke.
 *
 * - The first locked sample is anchored by the consumer's own pick (the press
 *   point), or by `fallback` when that pick meets nothing.
 * - A lock engaged or switched after free or locked samples re-anchors at the
 *   last focus, so the hand does not jump.
 * - Releasing the lock returns to the free pick.
 */
export class FocusSession implements ToolFocus {
  /** The anchor the current lock stands on, once a locked sample has set it. */
  anchor: Vec3 | undefined;
  private anchoredFor: readonly EditorAxis[] | undefined;
  private last: FocusPoint | undefined;

  constructor(private readonly options: FocusSessionOptions) {}

  /** The lock in force for the last sample, if any. */
  get lock(): FocusLock | undefined {
    return this.anchor && this.anchoredFor ? { axes: this.anchoredFor, anchor: this.anchor } : undefined;
  }

  resolve(ray: EditorRay, pick: FocusPick): FocusPoint | undefined {
    const axes = focusAxes(this.options.constraint());
    if (!axes) {
      this.anchor = undefined; this.anchoredFor = undefined;
      const free = pick(ray);
      if (free) this.last = free;
      return free;
    }
    if (!this.anchor || !sameAxes(axes, this.anchoredFor)) {
      let anchor = this.last?.point;
      let first: FocusPoint | undefined;
      if (!anchor) {
        first = pick(ray);
        anchor = first?.point ?? this.options.fallback;
      }
      if (!anchor) return undefined;
      this.anchor = { ...anchor }; this.anchoredFor = axes;
      // The press point is the focus: the anchoring sample keeps its surface normal.
      if (first) { this.last = { ...first, locked: true }; return this.last; }
    }
    const focus = projectFocus(ray, axes, this.anchor, this.options.reach_m);
    if (focus) this.last = focus;
    return focus;
  }
}

/** Resolve through a tool context's focus when it has one, otherwise the pick alone. */
export function focusOrPick<T extends FocusPoint>(focus: ToolFocus | undefined, ray: EditorRay,
  pick: (ray: EditorRay) => T | undefined): T | FocusPoint | undefined {
  return focus ? focus.resolve(ray, pick) : pick(ray);
}

/** How far a locked focus may stray from its anchor, and how long its guide is drawn: a few vessels' worth. */
export function focusReach_m(container: { readonly width_m: number; readonly height_m: number; readonly depth_m: number }): number {
  const extent = Math.max(container.width_m, container.height_m, container.depth_m);
  return Number.isFinite(extent) && extent > 0 ? 8 * extent : 64;
}
