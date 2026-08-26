/**
 * Whether the pointer edits the scene or only looks at it.
 *
 * The one modal axis in the viewport. Everything else the editor does — what is
 * selected, what a drag means, which gesture is armed — happens *inside*
 * INTERACT; CAMERA has no state of its own, because it is the absence of all of
 * it.
 *
 * Why a mode at all, in a product whose organizing principle is that nothing
 * gets a permanent button: because "clicking does nothing" is not a default a
 * reader can discover. An always-live editor means every attempt to look at a
 * running solve is one mis-click away from moving something in it, and it means
 * the pointer must resolve a pick on every move whether or not anyone is
 * editing. Naming the two states makes both problems go away, and it is what
 * buys INTERACT the right to be aggressive: inside it there is *always*
 * something under the cursor, which is only tolerable because leaving is one
 * key away.
 *
 * CAMERA is the default. A scene opens ready to be watched.
 */
export type ViewportMode = "camera" | "interact";

export const DEFAULT_VIEWPORT_MODE: ViewportMode = "camera";

export interface ViewportModeSpec {
  readonly id: ViewportMode;
  /** Caption for the toggle, and for the chip that reports the mode. */
  readonly label: string;
  /** What this mode lets the pointer do, in one line. */
  readonly hint: string;
}

/**
 * The key that swaps them.
 *
 * Tab, because every other unmodified letter in the viewport is already spoken
 * for — nine tool letters, three instrument letters, the cell picker's `c`/`i`,
 * the axis locks and the framing digits — and because a mode swap is what Tab
 * means in the editors this one is read against. Handled with preventDefault
 * only while the shell has focus, so focus traversal survives everywhere else.
 */
export const VIEWPORT_MODE_SHORTCUT = "Tab";

export const VIEWPORT_MODES: Readonly<Record<ViewportMode, ViewportModeSpec>> = Object.freeze({
  camera: Object.freeze({
    id: "camera",
    label: "LOOK",
    hint: "drag to orbit · shift-drag to pan · wheel to zoom · nothing in the scene can be touched",
  }),
  interact: Object.freeze({
    id: "interact",
    label: "EDIT",
    hint: "everything under the cursor lights up · click to select it · right-click for what it offers · drag for what its drag means",
  }),
});

export function toggledViewportMode(mode: ViewportMode): ViewportMode {
  return mode === "interact" ? "camera" : "interact";
}
