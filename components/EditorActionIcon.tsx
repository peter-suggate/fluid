"use client";

import {
  ArrowDownToLine,
  Beaker,
  Box,
  Brush,
  Circle,
  Cylinder,
  Droplet,
  Egg,
  Eraser,
  Hand,
  Pill,
  Shapes,
  SlidersHorizontal,
  SprayCan,
  Square,
  SquareDashed,
  Trash2,
  type LucideIcon,
} from "lucide-react";
import type { EditorActionIcon as EditorActionIconName } from "../lib/core/editor-action";

/**
 * The one place a verb becomes a picture.
 *
 * `EditorActionIcon` is a closed vocabulary declared in core beside the entities
 * that use it; core may not import React, so the resolution happens here. That
 * split is what lets an entity say "this action drops something" without also
 * deciding what a drop looks like, and it makes changing the icon set — or
 * drawing the same vocabulary somewhere other than the ring — one edit.
 *
 * `satisfies Record<…>` rather than an index signature: a name added to the
 * union without an icon is a compile error, so a wedge can never be silently
 * blank.
 */
const ICONS = {
  edit: SlidersHorizontal,
  delete: Trash2,
  "water-ball": Droplet,
  solid: Box,
  paint: Brush,
  erase: Eraser,
  region: SquareDashed,
  hose: SprayCan,
  prop: Shapes,
  carry: Hand,
  drop: ArrowDownToLine,
  sphere: Circle,
  box: Square,
  cylinder: Cylinder,
  capsule: Pill,
  cup: Beaker,
  ellipsoid: Egg,
} satisfies Record<EditorActionIconName, LucideIcon>;

/**
 * Draw one, centred on a point in an SVG's own coordinate system.
 *
 * Nested `<svg>` inside a `<g>` rather than a `<foreignObject>`: the icon is
 * vector art in the same document, so it inherits `currentColor` from the wedge
 * that owns it and scales with the ring instead of being a rectangle of HTML
 * pinned over it.
 */
export function EditorActionIconMark({ name, x, y, size = 21 }: {
  name: EditorActionIconName;
  x: number;
  y: number;
  size?: number;
}) {
  const Icon = ICONS[name];
  return <g transform={`translate(${x - size / 2} ${y - size / 2})`} className="radial-icon">
    <Icon width={size} height={size} strokeWidth={1.6} aria-hidden />
  </g>;
}
