"use client";

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from "react";

/**
 * How far a flyout stays clear of the viewport's own edges. The viewport shell
 * clips (`overflow: hidden`), so an edge crossed is an edge lost — a panel that
 * runs past it does not scroll into view, it disappears.
 */
export const FLYOUT_EDGE_MARGIN_PX = 10;

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
 * than they are wide and the anchor is already inside them.
 */
export function resolveFlyoutPlacement({
  leftFraction, topFraction, gap, originY, offsetY,
  panelWidth, panelHeight, containerWidth, containerHeight,
}: FlyoutGeometry): FlyoutPlacement {
  const margin = FLYOUT_EDGE_MARGIN_PX;
  const anchorX = leftFraction * containerWidth;
  const anchorY = topFraction * containerHeight;
  const fitsRight = anchorX + gap + panelWidth <= containerWidth - margin;
  const fitsLeft = anchorX - gap - panelWidth >= margin;
  const left = fitsRight || !fitsLeft ? anchorX + gap : anchorX - gap - panelWidth;
  return {
    left: clamp(left, margin, containerWidth - margin - panelWidth),
    top: clamp(anchorY - originY * panelHeight + offsetY, margin, containerHeight - margin - panelHeight),
    // Container-derived rather than content-derived, so applying it cannot
    // resize the panel into a different answer on the next measurement.
    maxHeight: Math.max(0, containerHeight - 2 * margin),
  };
}

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
 * catches both of those, and re-placing after every render catches the camera.
 */
export function useAnchoredFlyout<T extends HTMLElement>({
  leftFraction,
  topFraction,
  gap = 12,
  originY = 0.6,
  offsetY = 0,
}: AnchoredFlyoutOptions): { ref: RefObject<T | null>; style: CSSProperties } {
  const ref = useRef<T>(null);
  const [placement, setPlacement] = useState<FlyoutPlacement>();

  const place = () => {
    const element = ref.current;
    const container = element?.offsetParent;
    if (!element || !(container instanceof HTMLElement)) return;
    const next = resolveFlyoutPlacement({
      leftFraction, topFraction, gap, originY, offsetY,
      panelWidth: element.offsetWidth,
      panelHeight: element.offsetHeight,
      containerWidth: container.clientWidth,
      containerHeight: container.clientHeight,
    });
    // Guarded because this runs after every render: an unconditional set would
    // be a render loop rather than a placement.
    setPlacement((previous) => (previous
      && previous.left === next.left
      && previous.top === next.top
      && previous.maxHeight === next.maxHeight
      ? previous
      : next));
  };

  // No dependency list: the anchor moves with the camera, which is a new render
  // and not a new effect. Before paint, so the first frame is already placed.
  const latest = useRef(place);
  useLayoutEffect(() => {
    latest.current = place;
    place();
  });

  useEffect(() => {
    const element = ref.current;
    const container = element?.offsetParent;
    if (!element || !(container instanceof HTMLElement)) return;
    const observer = new ResizeObserver(() => latest.current());
    observer.observe(element);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return {
    ref,
    // The fallback is the pre-measurement position, which never paints: the
    // layout effect above has already replaced it by the time the browser draws.
    style: placement
      ? { left: placement.left, top: placement.top, maxHeight: placement.maxHeight }
      : { left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` },
  };
}
