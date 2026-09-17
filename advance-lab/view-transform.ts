/**
 * Where a finest cell is, in viewport pixels — the slice's camera.
 *
 * The lab draws one 2-D cut, and until now it drew all of it, always, at a
 * whole number of pixels per cell chosen so that the whole thing fitted. That
 * is a contact sheet, not an instrument: a 256-cell slice gives every cell two
 * pixels, and the PLIC line a stage is about is thinner than the grid drawn
 * over it. So the picture gets a camera, and it is the studio's camera, in two
 * dimensions: the wheel zooms toward the cursor, shift- or middle-drag grabs
 * the water and slides it, and `0` refits.
 *
 * A `SliceView` is the whole state, and it is two numbers and a point:
 *
 *   zoom     a multiplier over the *fit* scale, so 1 is exactly the old
 *            picture — the whole slice, centred. It is never below 1: there
 *            is nothing outside the slice to look at.
 *   panFine  the finest-cell coordinate held at the viewport's CENTRE. The
 *            centre rather than the top-left because the fit view is centred,
 *            which makes `{ zoom: 1, panFine: [nx / 2, ny / 2] }` the identity
 *            and makes "the slice cannot be lost" a bound on one point.
 *
 * Everything else here is derived. `fitScale` is CSS pixels per cell at zoom
 * 1, `pixelsPerCell` is what that becomes under the zoom, and the origin is
 * where cell (0, 0) lands. Two rules are worth stating because they are the
 * only places this is not plain arithmetic:
 *
 *   — pixels per cell is snapped to a whole number *only while it is at least
 *     two*, which is the lab's standing reason (a grid line lands on one pixel
 *     rather than across two). Below two a whole number is a 50% error, so
 *     down there it stays fractional and the grid may blur.
 *   — the clamp is `panFine` inside the slice. That is the same statement as
 *     "the slice's edge never passes the viewport's edge by more than half a
 *     viewport", because the centre of the viewport is by construction the
 *     cell `panFine` names.
 *
 * Nothing in here touches the DOM or React: a rect is four numbers, and the
 * page hands it either a real `DOMRect` (for hit-testing against the live
 * canvas) or a rect at the origin (for laying out overlays inside the
 * viewport, where client coordinates and element coordinates coincide).
 */

/** Just the extent of a box — all `fitScale` needs. */
export interface ViewportBox {
  readonly width: number;
  readonly height: number;
}

/** A `DOMRect` satisfies this. */
export interface ViewportRect extends ViewportBox {
  readonly left: number;
  readonly top: number;
}

export interface SliceView {
  /** Multiplier over the fit scale. 1 is the whole slice, centred. */
  readonly zoom: number;
  /** The finest-cell coordinate held at the viewport's centre. */
  readonly panFine: readonly [number, number];
}

/**
 * The fit view is the far end of the wheel rather than a midpoint: zooming out
 * past the whole slice would only add letterbox, which is what this change
 * exists to remove.
 */
export const SLICE_ZOOM_RANGE = Object.freeze({ minimum: 1, maximum: 64 });

/**
 * Zoom is exponential in wheel travel, so a notch is a fixed *fraction* of
 * wherever the zoom already is — the same feel as the 3-D viewport.
 *
 * Mirrored from `ZOOM_RATE_PER_PIXEL` in `lib/core/math.ts` (~:112), which is
 * module-private there. The sign is inverted against that one because this is
 * a magnification and that is a camera distance: wheel down zooms out in both.
 */
export const SLICE_ZOOM_RATE_PER_PIXEL = 0.001;

/** The whole slice, centred — what `0` returns to. */
export function fitView(nx: number, ny: number): SliceView {
  return { zoom: 1, panFine: [nx / 2, ny / 2] };
}

/** CSS pixels per finest cell at zoom 1: the largest scale that still fits. */
export function fitScale(room: ViewportBox, nx: number, ny: number): number {
  const fit = Math.min(room.width / Math.max(1, nx), room.height / Math.max(1, ny));
  return Number.isFinite(fit) && fit > 0 ? fit : 1;
}

/** CSS pixels per finest cell under this view, whole while that is legible. */
export function pixelsPerCell(view: SliceView, fit: number): number {
  const raw = fit * view.zoom;
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  return raw >= 2 ? Math.floor(raw) : raw;
}

/** Where fine cell (0, 0) sits, in pixels from `rect`'s top-left corner. */
export function originPixels(
  view: SliceView, fit: number, rect: ViewportRect,
): readonly [number, number] {
  const ppc = pixelsPerCell(view, fit);
  return [rect.width / 2 - view.panFine[0] * ppc, rect.height / 2 - view.panFine[1] * ppc];
}

/**
 * The continuous finest-cell coordinate under a client point.
 *
 * Deliberately unbounded: a drag that leaves the slice keeps producing a
 * coordinate, which is what lets a region be rubber-banded past the edge and
 * snapped back by the ladder rather than stopping dead at it.
 */
export function cellFromClient(
  view: SliceView, fit: number, rect: ViewportRect, clientX: number, clientY: number,
): readonly [number, number] {
  const ppc = pixelsPerCell(view, fit);
  const [ox, oy] = originPixels(view, fit, rect);
  return [(clientX - rect.left - ox) / ppc, (clientY - rect.top - oy) / ppc];
}

/**
 * The inverse: where a finest-cell coordinate lands, in `rect`'s own frame.
 *
 * Pass a rect at the origin (`left: 0, top: 0`) to get pixels inside the
 * element, which is what an absolutely-positioned overlay wants.
 */
export function clientFromCell(
  view: SliceView, fit: number, rect: ViewportRect, fx: number, fy: number,
): readonly [number, number] {
  const ppc = pixelsPerCell(view, fit);
  const [ox, oy] = originPixels(view, fit, rect);
  return [rect.left + ox + fx * ppc, rect.top + oy + fy * ppc];
}

const clampZoom = (zoom: number): number => Number.isFinite(zoom)
  ? Math.min(SLICE_ZOOM_RANGE.maximum, Math.max(SLICE_ZOOM_RANGE.minimum, zoom))
  : SLICE_ZOOM_RANGE.minimum;

/**
 * Zoom about a point, keeping the cell under it under it.
 *
 * The pan is re-derived from the *snapped* pixels-per-cell, not from the
 * requested zoom, so the fixed point is fixed on screen and not merely in
 * theory — a notch that snapped from 7.9 to 7 px per cell would otherwise
 * slide the water under the cursor by most of a cell.
 */
export function zoomedToward(
  view: SliceView, fit: number, rect: ViewportRect,
  clientX: number, clientY: number, deltaY: number,
): SliceView {
  const held = cellFromClient(view, fit, rect, clientX, clientY);
  const zoom = clampZoom(view.zoom * Math.exp(-deltaY * SLICE_ZOOM_RATE_PER_PIXEL));
  const ppc = pixelsPerCell({ ...view, zoom }, fit);
  const px = clientX - rect.left - rect.width / 2, py = clientY - rect.top - rect.height / 2;
  return { zoom, panFine: [held[0] - px / ppc, held[1] - py / ppc] };
}

/** Grab the water and slide it: what is under the pointer stays under it. */
export function panned(view: SliceView, fit: number, dxPx: number, dyPx: number): SliceView {
  const ppc = pixelsPerCell(view, fit);
  return { ...view, panFine: [view.panFine[0] - dxPx / ppc, view.panFine[1] - dyPx / ppc] };
}

/**
 * The view, made legal. Idempotent, so the page can clamp per render and the
 * gestures need not each re-derive the bound.
 *
 * Per axis, because the snap can leave a slice that fits one way and not the
 * other: an axis that fits is centred outright (there is one correct place for
 * it), and an axis that does not keeps the viewport's centre inside the slice.
 */
export function clampedView(
  view: SliceView, fit: number, rect: ViewportRect, nx: number, ny: number,
): SliceView {
  const zoom = clampZoom(view.zoom);
  const ppc = pixelsPerCell({ ...view, zoom }, fit);
  const axis = (pan: number, n: number, extent: number): number =>
    n * ppc <= extent ? n / 2 : Math.min(n, Math.max(0, Number.isFinite(pan) ? pan : n / 2));
  return { zoom, panFine: [
    axis(view.panFine[0], nx, rect.width), axis(view.panFine[1], ny, rect.height)] };
}

const round = (value: number): string => String(Math.round(value * 1e4) / 1e4);

/**
 * The same transform as a `viewBox`, for the overlays drawn in cell units.
 *
 * The overlay SVGs are the whole viewport, so their box is the viewport
 * expressed in cells. Its aspect ratio therefore equals the element's, which
 * is what lets the default `preserveAspectRatio` be a no-op rather than a
 * second, disagreeing letterbox.
 */
export function svgViewBox(
  view: SliceView, fit: number, rect: ViewportRect, nx: number, ny: number,
): string {
  const legal = clampedView(view, fit, rect, nx, ny);
  const ppc = pixelsPerCell(legal, fit);
  const width = rect.width / ppc, height = rect.height / ppc;
  return `${round(legal.panFine[0] - width / 2)} ${round(legal.panFine[1] - height / 2)} ${round(width)} ${round(height)}`;
}
