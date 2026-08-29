"use client";

import {
  Boxes,
  CircleGauge,
  Droplets,
  Eye,
  Gauge,
  Grid3x3,
  Move3d,
  Sparkles,
  Waves,
  type LucideIcon,
} from "lucide-react";
import { getMethod } from "../lib/core/method-registry";
import { pickFieldOverlay, type FieldOverlayView } from "../lib/core/field-overlay-pick";
import { TANK_SELECTION_ID } from "../lib/core/editor-tank";
import { VISUALIZATION_FIELDS, VISUALIZATION_QUICK_FIELDS } from "../lib/core/visualization-catalog";
import type { FieldVisualization, FieldVisualizationIcon } from "../lib/core/visualization-registry";
import { isPressureJournalOverlayMode } from "../lib/core/webgpu-pressure-journal-overlay";
import { useMethodStore } from "../lib/core/stores/method-store";
import { DEFAULT_GRID_OVERLAY_AXIS, useUIStore } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import { Toolstrip, ToolstripMoreRow, ToolstripRow, ToolstripScrub } from "./toolstrip";

/**
 * The one place a field becomes a picture. Same split as `EditorActionIcon`:
 * the vocabulary is declared in core beside the views that claim it, core may
 * not import React, so the resolution happens here — and `satisfies Record<…>`
 * makes a name added to the union without a glyph a compile error rather than
 * a blank button.
 */
const ICONS = {
  grid: Grid3x3,
  levels: Boxes,
  density: Droplets,
  surface: Waves,
  speed: Gauge,
  pressure: CircleGauge,
  flow: Move3d,
  tracers: Sparkles,
} satisfies Record<FieldVisualizationIcon, LucideIcon>;

/** A catalog field narrowed to what earned it a glyph, as the flyout narrows its own list. */
type QuickField = FieldVisualization & { mode: GridOverlayMode; icon: FieldVisualizationIcon };

const QUICK_FIELDS = VISUALIZATION_QUICK_FIELDS as readonly QuickField[];

/** Whether this method publishes anything the glyph rows can offer. */
export function methodHasQuickFields(methodId: string): boolean {
  const supported = new Set(getMethod(methodId).supportedFieldModes ?? []);
  return QUICK_FIELDS.some((view) => supported.has(view.mode));
}

/**
 * The glyph rows: one per field view worth turning on without opening anything.
 *
 * Which views those are is not decided here. A pass declares an `icon` on the
 * view it thinks is worth the room (see `FieldVisualizationIcon`), and the
 * running method's `supportedFieldModes` narrows that to what this solver can
 * honestly publish — so a method that draws no volume never grows a volume
 * glyph, and a view added beside its pass reaches this strip without this file
 * learning its name.
 *
 * Glyphs and nothing else. Names alongside them would be a paragraph standing
 * over the water, and the strip lives on the image rather than beside it — so
 * what is drawing is said by the lit glyph, and the name is said on hover.
 *
 * Exported as rows rather than as a strip because the tank's own strip grows
 * out of this one: selecting the tank adds sections underneath rather than
 * swapping these for a panel.
 */
export function FieldViewRows() {
  const methodId = useMethodStore((state) => state.methodId);
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = QUICK_FIELDS.filter((view) => supported.has(view.mode));
  // What is drawing, whether or not this strip could have chosen it: the full
  // list and a link can both leave a view on that has no glyph here, and a
  // reader who cannot see what is drawn cannot turn it off either.
  const drawing = overlayAxis !== "off"
    ? VISUALIZATION_FIELDS.find((view) => view.mode === overlayMode)
    : undefined;

  const choose = (view: FieldOverlayView) => {
    const change = pickFieldOverlay(
      { mode: overlayMode, axis: overlayAxis }, view, volumeCapable, DEFAULT_GRID_OVERLAY_AXIS);
    if (change.mode !== undefined) setOverlayMode(change.mode);
    if (change.axis !== undefined) setOverlayAxis(change.axis);
    if (change.slice !== undefined) setOverlaySlice(change.slice);
  };

  // The scrub belongs with the picking, not behind a selection: choosing a
  // slice view and then having to find the tank to sweep it is the same trip
  // this strip exists to remove. It is the plane depth on a sliced view and the
  // opacity on a volume one — and it is rendered into the row of whatever is
  // on, so the control is beside the glyph it belongs to.
  // A captured solve is the exception: its scrub steps through the iterations
  // that were snapshotted rather than sliding through a plane, and the plot that
  // makes a stop mean anything is in the film pane — so the scrub goes there
  // with it rather than standing here detached from its own curve.
  const volumetric = drawing !== undefined && (drawing.planeless || overlayAxis === "volume");
  const scrub = drawing && !isPressureJournalOverlayMode(drawing.mode) && <ToolstripScrub
    min={volumetric ? 0.05 : 0}
    max={1}
    step={volumetric ? 0.01 : 0.005}
    value={overlaySlice}
    readout={`${Math.round(overlaySlice * 100)}%`}
    ariaLabel={volumetric ? `${drawing.label} opacity` : `Field ${overlayAxis} slice position`}
    onChange={setOverlaySlice}
  />;

  return <>
    {views.map((view) => {
      const Icon = ICONS[view.icon];
      const active = drawing?.mode === view.mode;
      return <ToolstripRow
        key={view.id}
        icon={<Icon width={14} height={14} strokeWidth={1.7} aria-hidden />}
        name={view.label}
        hint={view.description}
        active={active}
        testId={`field-quick-${view.mode}`}
        onClick={() => choose(view)}
      >{active ? scrub : undefined}</ToolstripRow>;
    })}
    {/* A view this strip has no glyph for is still drawing over the scene, so
        it takes the one glyph that means "something is on" and says which in
        its tip. Without this, choosing one in the full list and then
        deselecting the tank leaves a field on the image with nothing on screen
        admitting it — and it carries the scrub for the same reason a glyph row
        does. */}
    {drawing && drawing.icon === undefined && <ToolstripRow
      icon={<Eye width={14} height={14} strokeWidth={1.7} aria-hidden />}
      name={drawing.label}
      hint="Click to hide this view."
      active
      testId="field-quick-other"
      onClick={() => setOverlayAxis("off")}
    >{scrub}</ToolstripRow>}
  </>;
}

/**
 * Turning a field on, at the corner of the thing it is drawn through.
 *
 * The field picker was reachable only by selecting the tank, which is a fact
 * about this editor nobody can be expected to guess: "show me the grid" is one
 * of the first things a reader wants and it was three unguessable steps away.
 * So the corner that hosts the picker is never empty in EDIT — at rest it
 * carries the handful of views worth turning on without opening anything, and
 * selecting the tank grows the same column into everything else it can say.
 *
 * Nothing here reports EDIT mode. The mode toggle does that, permanently, in
 * the top-right — this strip only has to be *present*, which is the whole
 * discoverability claim it makes.
 */
export function FieldQuickBar({ leftFraction, topFraction }: {
  leftFraction: number;
  topFraction: number;
}) {
  const methodId = useMethodStore((state) => state.methodId);
  const select = useUIStore((state) => state.select);
  if (!methodHasQuickFields(methodId)) return null;

  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel="Field overlays"
    testId="field-quick-bar"
  >
    <FieldViewRows />
    {/* The way to everything this strip left out — the rest of the catalog, the
        plane, the solver, the setup, and the tank's own settings. Selecting the
        tank is what opens those, so this is the existing route made visible
        rather than a second one to keep agreeing with it. */}
    <ToolstripMoreRow
      name="All field views"
      hint="Every view this solver publishes, its plane, and the solver behind them. Selects the tank."
      testId="field-quick-more"
      onClick={() => select({ kind: "tank", id: TANK_SELECTION_ID })}
    />
  </Toolstrip>;
}
