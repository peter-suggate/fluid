import { editorAxisForKey, toggleAxisConstraint, type AxisConstraint, type EditorAxis } from "../editor-axis-constraint";
import type { EditorGestureId } from "../editor-gesture-catalog";

/**
 * The lock keys. X, Y, Z lock the focus to that world axis; with Shift, to the
 * plane that holds that axis fixed (Shift+Y keeps the height). A tap latches,
 * and the same chord releases. Held for `FOCUS_KEY_HOLD_MS` or longer, the key
 * is spring-loaded: letting go releases the lock it set.
 */
export const FOCUS_KEY_HOLD_MS = 350;

/** The armed gestures that place something, and so take a focus lock. */
const PLACING_GESTURES: readonly EditorGestureId[] = ["fluid-ball", "body-drag"];

/**
 * Whether the lock keys mean anything here: a standing selection (its handles
 * have always taken them) or, in EDIT, a placing tool — any voxel tool, BALL or
 * DRAG.
 */
export function focusLockKeysLive(ui: {
  readonly selection?: unknown;
  readonly viewportMode: string;
  readonly voxelToolId?: string;
  readonly armedGesture?: EditorGestureId;
}): boolean {
  if (ui.selection) return true;
  return ui.viewportMode === "interact"
    && (ui.voxelToolId !== undefined || (ui.armedGesture !== undefined && PLACING_GESTURES.includes(ui.armedGesture)));
}

interface KeyLike { readonly key: string; readonly shiftKey?: boolean; readonly repeat?: boolean; readonly timeStamp: number }

/** Tap-or-hold state of the lock keys. One per keyboard, which is one per window. */
export class FocusKeyHold {
  private held: { readonly axis: EditorAxis; readonly at_ms: number; readonly lock: AxisConstraint } | undefined;

  /** A lock key is physically down. While it is, Shift is part of its chord, not a pan request. */
  get holding(): boolean { return this.held !== undefined; }

  /**
   * The constraint a keydown asks for, or `"ignore"` when it is not a lock key
   * or is a repeat (a held key must not flicker the lock on and off).
   */
  down(event: KeyLike, current: AxisConstraint): AxisConstraint | "ignore" {
    const axis = editorAxisForKey(event.key);
    if (!axis) return "ignore";
    if (event.repeat) return "ignore";
    const next = toggleAxisConstraint(current, axis, event.shiftKey ? "plane" : "axis");
    this.held = next ? { axis, at_ms: event.timeStamp, lock: next } : undefined;
    return next;
  }

  /** On keyup: `"release"` when a held key springs its lock back, `"keep"` otherwise. */
  up(event: KeyLike, current: AxisConstraint): "release" | "keep" {
    const held = this.held;
    if (!held || editorAxisForKey(event.key) !== held.axis) return "keep";
    this.held = undefined;
    return event.timeStamp - held.at_ms >= FOCUS_KEY_HOLD_MS && current === held.lock ? "release" : "keep";
  }

  /** Focus left the window: nothing is held any more. */
  reset(): void { this.held = undefined; }
}

/** The window's lock keys. */
export const focusKeys = new FocusKeyHold();
