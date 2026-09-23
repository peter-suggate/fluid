"use client";

import { useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from "react";

/**
 * How far a flyout stays clear of the viewport's own edges. The viewport shell
 * clips (`overflow: hidden`), so an edge crossed is an edge lost — a panel that
 * runs past it does not scroll into view, it disappears.
 */
export const FLYOUT_EDGE_MARGIN_PX = 10;

/**
 * The band along the bottom edge the transport cluster floats in.
 *
 * The shell's bottom edge is not the last place a panel may end: the transport
 * is centred on it, it paints under these flyouts, and an expanded selection
 * panel ran its last group across the play button. Panels stop above the band
 * instead — and a panel with more to show than fits scrolls, which is what the
 * height cap is for.
 */
export const FLYOUT_TRANSPORT_KEEPOUT_PX = 90;

export type FlyoutPlacement = { left: number; top: number; maxHeight: number };

export type FlyoutGeometry = {
  /** The anchor point, as the projector reports it: a fraction of the viewport. */
  leftFraction: number;
  topFraction: number;
  /** Horizontal clearance between the anchor point and the panel's near edge. */
  gap: number;
  /** Which point on the panel's own height meets the anchor. 0 is its top edge. */
  originY: number;
  /** Vertical nudge applied after `originY`. */
  offsetY: number;
  panelWidth: number;
  panelHeight: number;
  containerWidth: number;
  containerHeight: number;
};

/** Clamps toward `low`, so a panel larger than its container still starts on-screen. */
const clamp = (value: number, low: number, high: number) =>
  Math.min(Math.max(value, low), Math.max(low, high));

/**
 * Where an anchored flyout has to sit to stay whole.
 *
 * Right of the anchor is the authored side, and it stays the side unless it
 * genuinely does not fit *and* the other one does — a panel that flipped
 * whenever it was merely tight would jitter back and forth across the anchor as
 * the camera moved, which reads worse than one sitting slightly off its mark.
 * Vertically there is no flip to make, only a clamp: these panels are taller
 * than they are wide and the anchor is already inside them. The bottom of that
 * clamp is the transport's band rather than the shell's edge — see
 * `FLYOUT_TRANSPORT_KEEPOUT_PX`.
 */
export function resolveFlyoutPlacement({
  leftFraction, topFraction, gap, originY, offsetY,
  panelWidth, panelHeight, containerWidth, containerHeight,
}: FlyoutGeometry): FlyoutPlacement {
  const margin = FLYOUT_EDGE_MARGIN_PX;
  const floor = containerHeight - FLYOUT_TRANSPORT_KEEPOUT_PX;
  const anchorX = leftFraction * containerWidth;
  const anchorY = topFraction * containerHeight;
  const fitsRight = anchorX + gap + panelWidth <= containerWidth - margin;
  const fitsLeft = anchorX - gap - panelWidth >= margin;
  const left = fitsRight || !fitsLeft ? anchorX + gap : anchorX - gap - panelWidth;
  return {
    left: clamp(left, margin, containerWidth - margin - panelWidth),
    top: clamp(anchorY - originY * panelHeight + offsetY, margin, floor - panelHeight),
    // Container-derived rather than content-derived, so applying it cannot
    // resize the panel into a different answer on the next measurement.
    maxHeight: Math.max(0, floor - margin),
  };
}

/** The measured half of `FlyoutGeometry`: what content and layout decide, not the camera. */
type FlyoutBoxes = Pick<FlyoutGeometry, "panelWidth" | "panelHeight" | "containerWidth" | "containerHeight">;

export type AnchoredFlyoutOptions = {
  leftFraction: number;
  topFraction: number;
  gap?: number;
  originY?: number;
  offsetY?: number;
};

/**
 * Keeps a viewport-anchored flyout inside the viewport.
 *
 * Every one of these panels rides a projected world point, so where it lands is
 * the camera's decision, not the layout's: orbit until the anchor nears an edge
 * and a panel authored to sit 12px to its right is a panel half outside the
 * clip. This measures both boxes and hands the arithmetic to
 * `resolveFlyoutPlacement`.
 *
 * Measured rather than declared because the size is content: the selection
 * panel is a chip until it is expanded, and the field picker grows a row per
 * view the method registers. A `ResizeObserver` on the panel and on the shell
 * catches both of those; the camera only moves the anchor, which is arithmetic.
 */
export function useAnchoredFlyout<T extends HTMLElement>({
  leftFraction,
  topFraction,
  gap = 12,
  originY = 0.6,
  offsetY = 0,
}: AnchoredFlyoutOptions): { ref: RefObject<T | null>; style: CSSProperties } {
  const ref = useRef<T>(null);
  const [boxes, setBoxes] = useState<FlyoutBoxes>();

  // The two boxes are read when either one changes size, never per render. The
  // anchor moves with the camera, and a layout read after every commit forced a
  // synchronous reflow of the page — plus a second render to apply what it
  // found — on every orbit step, on the thread that also encodes the frame.
  // Mounting measures before paint, so the first frame is already placed; a
  // later content resize lands one frame after it, when the observer reports.
  useLayoutEffect(() => {
    const element = ref.current;
    const container = element?.offsetParent;
    if (!element || !(container instanceof HTMLElement)) return;
    const measure = () => {
      const next: FlyoutBoxes = {
        panelWidth: element.offsetWidth,
        panelHeight: element.offsetHeight,
        containerWidth: container.clientWidth,
        containerHeight: container.clientHeight,
      };
      setBoxes((previous) => (previous
        && previous.panelWidth === next.panelWidth
        && previous.panelHeight === next.panelHeight
        && previous.containerWidth === next.containerWidth
        && previous.containerHeight === next.containerHeight
        ? previous
        : next));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Pure arithmetic on the measured boxes, so following the camera is one
  // render and one ordinary layout.
  const placement = boxes && resolveFlyoutPlacement({ leftFraction, topFraction, gap, originY, offsetY, ...boxes });
  return {
    ref,
    // The fallback is the pre-measurement position, which never paints: the
    // layout effect above has already measured by the time the browser draws.
    style: placement
      ? { left: placement.left, top: placement.top, maxHeight: placement.maxHeight }
      : { left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` },
  };
}
