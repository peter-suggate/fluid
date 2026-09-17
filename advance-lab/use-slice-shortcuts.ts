"use client";

import { useEffect, useRef } from "react";
import { editorGestureForShortcut, type EditorGestureId } from "../lib/core/editor-gesture-catalog";
import { toggledViewportMode, VIEWPORT_MODE_SHORTCUT } from "../lib/core/editor-viewport-mode";
import type { PaneSession } from "../lib/core/session/session";
import { SLICE_OVERLAY_ORDER, type SliceOverlayId } from "./lenses";

/**
 * The lab's keyboard, in the studio's order.
 *
 * Not `use-editor-shortcuts` itself: that hook reaches `simulation.undo`,
 * `simulation.redo` and `simulation.removeEntity` in three of its branches, and
 * the lab has no scene document for any of them to act on. What it does share
 * is every constant and — the part that actually matters — the *order*, because
 * the order is the behaviour a reader learns:
 *
 *   1. **An open ring swallows everything.** Its own handler walks and closes it.
 *   2. **Modified keys belong to the browser.** The lab has no undo to claim them.
 *   3. **Tab** swaps LOOK and EDIT, ahead of everything either mode contains:
 *      leaving EDIT is how you put a selection and an armed stroke down, so a
 *      guard that made you clear them first would have the dependency backwards.
 *   4. **The Escape ladder**, from the inside out: the armed stroke, then the
 *      selection, then the mode itself. With nothing armed and nothing selected
 *      there is nothing left inside EDIT to leave, so the next Escape is EDIT.
 *   5. **Delete / Backspace** removes the selected enforcement region. Nothing
 *      else in the lab is removable, so the key falls through the rest of the time.
 *   6. **`0`** refits the picture — the studio's framing key, and here as there
 *      the only way back from a view pushed into nowhere.
 *   7. **`f` and `n`** toggle the fraction and normal overlays. Instruments, not
 *      verbs: they annotate the reading and never move the water, which is why
 *      they work in LOOK as well as in EDIT.
 *   8. **`b` and `g`** arm the ball and the enforcement box, from the shared
 *      gesture catalog — so the hand that drops water in the 3-D app drops it
 *      here, and `g` draws a region in both. Asking for a stroke is asking to
 *      edit, so the key enters EDIT rather than being swallowed by LOOK. The
 *      lab's old `r` is gone: in the studio `r` is the ray probe, and one letter
 *      meaning two things across two pages is how a reader learns the shortcuts
 *      are unreliable.
 */

/** The two strokes this lab can actually run, of the catalog's five armable. */
export const SLICE_GESTURES: readonly EditorGestureId[] =
  Object.freeze(["fluid-ball", "region-draw"]);

/** One key per overlay, named for the quantity rather than its position. */
export const SLICE_OVERLAY_KEYS: Readonly<Record<SliceOverlayId, string>> =
  Object.freeze({ fraction: "f", normal: "n" });

/** What the page has to do that the stores cannot. */
export interface SliceShortcutHost {
  /** `0`: frame the whole slice, read off the live world rather than a render. */
  readonly refit: () => void;
  /** `f` / `n`, when this transport draws the overlay at all. */
  readonly toggleOverlay: (overlay: SliceOverlayId) => void;
  readonly overlayOffered: (overlay: SliceOverlayId) => boolean;
  /** Delete: the selected enforcement box, when one is selected. */
  readonly removeSelectedRegion: () => boolean;
  /** Whatever a stroke had in flight — a proposed ball, a rubber band. */
  readonly clearDrafts: () => void;
}

/** Typing in a form control must never arm a stroke or remove a region. */
function editingText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

export function useSliceShortcuts(session: PaneSession, host: SliceShortcutHost): void {
  // The host is re-made every render — it closes over this render's setters —
  // so it is read through a ref rather than listed as a dependency, which would
  // re-register the listener on every frame the water moves. Written after the
  // commit rather than during render: a key can only be pressed once a frame
  // has been painted, so the listener never sees the stale one.
  const live = useRef(host);
  useEffect(() => { live.current = host; });
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (editingText(event.target)) return;
      const ui = session.ui.getState();
      const accelerator = event.metaKey || event.ctrlKey;
      // The ring is the innermost mode: while it is open its own handler walks
      // and closes it, and nothing below may steal a key from under it.
      if (ui.radialMenu && !accelerator) return;
      if (accelerator || event.altKey) return;

      if (event.key === VIEWPORT_MODE_SHORTCUT && !event.shiftKey) {
        event.preventDefault();
        live.current.clearDrafts();
        ui.setViewportMode(toggledViewportMode(ui.viewportMode));
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        live.current.clearDrafts();
        if (!ui.armedGesture && !ui.selection) { ui.setViewportMode("camera"); return; }
        ui.setArmedGesture(undefined);
        ui.select(undefined);
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (live.current.removeSelectedRegion()) {
          event.preventDefault();
          return;
        }
      }
      if (event.key === "0") {
        event.preventDefault();
        live.current.refit();
        return;
      }
      const stroke = event.key.toLowerCase();
      const overlay = SLICE_OVERLAY_ORDER.find(id => SLICE_OVERLAY_KEYS[id] === stroke);
      if (overlay) {
        event.preventDefault();
        if (live.current.overlayOffered(overlay)) live.current.toggleOverlay(overlay);
        return;
      }
      const gesture = editorGestureForShortcut(event.key);
      if (!gesture || !SLICE_GESTURES.includes(gesture)) return;
      event.preventDefault();
      live.current.clearDrafts();
      // Order matters: `setViewportMode("camera")` disarms, so entering EDIT
      // has to happen before the gesture is armed rather than after it.
      if (ui.viewportMode !== "interact") {
        ui.setViewportMode("interact");
        ui.setArmedGesture(gesture);
        return;
      }
      // Pressing the armed stroke's own key puts it away, so every mode can be
      // left the way it was entered rather than only through Escape.
      ui.select(undefined);
      ui.setArmedGesture(ui.armedGesture === gesture ? undefined : gesture);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [session]);
}
