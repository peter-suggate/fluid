/**
 * What picking a field view does to the overlay.
 *
 * Two surfaces choose field views now — the flyout's catalog list and the
 * quick bar at the tank corner — and "picking the active one hides it" is the
 * kind of rule that quietly stops being the same rule in two places. So it is
 * one pure function returning the change, and each surface applies it through
 * its own store setters.
 *
 * The semantics, unchanged from when the flyout was the only picker: choosing
 * the view that is already drawing hides the overlay without disturbing which
 * publication is selected, choosing another switches the publication and
 * leaves the reader's chosen plane alone, and only an overlay that was *off*
 * adopts the view's authored axis — because that is the one moment there is no
 * reader's choice to overrule.
 */
import type { GridOverlayConfig, GridOverlayMode } from "./webgpu-renderer";

export type FieldOverlayAxis = GridOverlayConfig["axis"];

/** The parts of a field view that decide where a pick lands. */
export interface FieldOverlayView {
  readonly mode: GridOverlayMode;
  readonly axis: FieldOverlayAxis;
  /** Draws its own geometry over the frame, so it has no plane to choose. */
  readonly planeless?: boolean;
}

export interface FieldOverlayState {
  readonly mode: GridOverlayMode;
  readonly axis: FieldOverlayAxis;
}

/** Only what changed: every absent field is one the pick does not disturb. */
export interface FieldOverlayChange {
  /**
   * Absent when the pick only hid the overlay. The publication is deliberately
   * left standing in that case, so turning the same view back on is one click
   * rather than a re-pick.
   */
  readonly mode?: GridOverlayMode;
  /** Absent when the overlay was already drawing: the plane is the reader's. */
  readonly axis?: FieldOverlayAxis;
  /** Present only when a volume view opened and needs a starting opacity. */
  readonly slice?: number;
}

/** The opacity a volume view opens at: dense enough to read, thin enough to see through. */
export const FIELD_OVERLAY_VOLUME_OPACITY = 0.42;

/**
 * Where a pick on `view` leaves the overlay.
 *
 * `defaultAxis` is the plane a volume view falls back to when the running
 * method has no volume raymarch to draw it in. A planeless view never falls
 * back: it draws over the finished frame rather than inside a march, so the
 * method's capability does not apply to it.
 */
export function pickFieldOverlay(
  current: FieldOverlayState,
  view: FieldOverlayView,
  volumeCapable: boolean,
  defaultAxis: Exclude<FieldOverlayAxis, "off">,
): FieldOverlayChange {
  if (current.mode === view.mode && current.axis !== "off") return { axis: "off" };
  if (current.axis !== "off") return { mode: view.mode };
  const axis = view.planeless || !(view.axis === "volume" && !volumeCapable)
    ? view.axis : defaultAxis;
  return {
    mode: view.mode,
    axis,
    slice: axis === "volume" ? FIELD_OVERLAY_VOLUME_OPACITY : undefined,
  };
}
