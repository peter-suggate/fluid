"use client";

import {
  Aperture,
  ArrowDownToLine,
  Beaker,
  Box,
  Brush,
  Circle,
  Crosshair,
  Cylinder,
  Droplet,
  Egg,
  Eraser,
  Gauge,
  Hand,
  Pill,
  ScanSearch,
  Shapes,
  SlidersHorizontal,
  SprayCan,
  Square,
  SquareDashed,
  Trash2,
  Workflow,
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
  pipeline: Workflow,
  "render-pipeline": Aperture,
  diagnostics: Gauge,
  // The two pointer probes: a reticle for the one that aims a ray at a pixel,
  // a framed magnifier for the one that reads the cell behind it.
  "trace-ray": Crosshair,
  "inspect-cell": ScanSearch,
} satisfies Record<EditorActionIconName, LucideIcon>;

/**
 * Draw one inline, for a verb that appears in ordinary layout.
 *
 * The ring is no longer the only place a verb is offered — a toolstrip row can
 * carry one too — and a row is a `<div>`, where `EditorActionIconMark`'s `<g>`
 * is not legal. Both go through the one table above on purpose: "delete" has to
 * be the same picture whether the reader met it on a wedge or on a row, and
 * that only holds while one place decides what it looks like. Sized for a strip
 * row by default, which is smaller than a wedge's; the stroke stays a hair
 * heavier so it survives at 14px.
 */
export function EditorActionGlyph({ name, size = 14 }: {
  name: EditorActionIconName;
  size?: number;
}) {
  const Icon = ICONS[name];
  return <Icon width={size} height={size} strokeWidth={1.7} aria-hidden />;
}

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
