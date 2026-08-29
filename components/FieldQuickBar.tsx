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
import { useMethodStore } from "../lib/core/stores/method-store";
import { DEFAULT_GRID_OVERLAY_AXIS, useUIStore } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import { useAnchoredFlyout } from "./anchored-flyout";

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

/**
 * Turning a field on, at the corner of the thing it is drawn through.
 *
 * The field picker was reachable only by selecting the tank, which is a fact
 * about this editor nobody can be expected to guess: "show me the grid" is one
 * of the first things a reader wants and it was three unguessable steps away.
 * So the corner that hosts the picker is never empty in EDIT — at rest it
 * carries the handful of views worth turning on without opening anything, one
 * glyph each, and selecting the tank grows it into the full panel.
 *
 * Which views those are is not decided here. A pass declares an `icon` on the
 * view it thinks is worth the room (see `FieldVisualizationIcon`), and the
 * running method's `supportedFieldModes` narrows that to what this solver can
 * honestly publish — so a method that draws no volume never grows a volume
 * glyph, and a view added beside its pass reaches this strip without this file
 * learning its name.
 *
 * A column of glyphs and nothing else. Names alongside them would be a
 * paragraph standing over the water, and the strip lives on the image rather
 * than beside it — so what is drawing is said by the lit glyph, and the name
 * is said on hover. That tooltip is the panel's own, not the browser's,
 * because the browser's arrives about a second late: a strip whose only labels
 * are behind a delay is a strip a reader gives up on before it answers.
 *
 * Vertical, and hung from the corner rather than laid across it: the row ran
 * out over the water it was describing, and a column falls down the outside of
 * the tank where the scene is not. There is no panel behind it either — the
 * glyphs are the chrome, and the only one that gets a background is the one
 * that is on.
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
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const select = useUIStore((state) => state.select);
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = QUICK_FIELDS.filter((view) => supported.has(view.mode));
  // Anchored against the shell's measured box, like the panel it stands in for,
  // so orbiting until this corner nears an edge slides the strip rather than
  // clipping it. `originY: 0` hangs the column's top edge off the corner, which
  // is the corner it is about.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({ leftFraction, topFraction, originY: 0 });
  // What is drawing, whether or not this strip could have chosen it: the full
  // picker and a link can both leave a view on that has no glyph here, and a
  // reader who cannot see what is drawn cannot turn it off either.
  const drawing = overlayAxis !== "off"
    ? VISUALIZATION_FIELDS.find((view) => view.mode === overlayMode)
    : undefined;
  if (views.length === 0) return null;

  const choose = (view: FieldOverlayView) => {
    const change = pickFieldOverlay(
      { mode: overlayMode, axis: overlayAxis }, view, volumeCapable, DEFAULT_GRID_OVERLAY_AXIS);
    if (change.mode !== undefined) setOverlayMode(change.mode);
    if (change.axis !== undefined) setOverlayAxis(change.axis);
    if (change.slice !== undefined) setOverlaySlice(change.slice);
  };

  // The scrub belongs with the picking, not behind the selection: choosing a
  // slice view and then having to find the tank to sweep it is the same trip
  // this strip exists to remove. It is the plane depth on a sliced view and the
  // opacity on a volume one, exactly as it is in the panel — and it is rendered
  // into the row of whatever is on, so the control is beside the glyph it
  // belongs to rather than at the foot of a column of things it does not.
  const scrub = drawing && <label className="field-quick-slice">
    <input
      type="range"
      min={drawing.planeless || overlayAxis === "volume" ? 0.05 : 0}
      max={1}
      step={drawing.planeless || overlayAxis === "volume" ? 0.01 : 0.005}
      value={overlaySlice}
      onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
      aria-label={drawing.planeless || overlayAxis === "volume"
        ? `${drawing.label} opacity`
        : `Field ${overlayAxis} slice position`}
    />
  </label>;

  return <div
    ref={ref}
    className="field-quick-bar"
    data-testid="field-quick-bar"
    style={style}
    role="group"
    aria-label="Field overlays"
  >
    {views.map((view) => {
      const Icon = ICONS[view.icon];
      const active = drawing?.mode === view.mode;
      return <div key={view.id} className="field-quick-row">
        <button
          type="button"
          className={active ? "active" : ""}
          aria-pressed={active}
          data-testid={`field-quick-${view.mode}`}
          onClick={() => choose(view)}
        >
          <Icon width={14} height={14} strokeWidth={1.7} aria-hidden />
          <span className="field-quick-tip"><strong>{view.label}</strong><small>{view.description}</small></span>
        </button>
        {active && scrub}
      </div>;
    })}
    {/* A view this strip has no glyph for is still drawing over the scene, so
        it takes the one glyph that means "something is on" and says which in its
        tip. Without this, choosing one in the full panel and then deselecting
        the tank leaves a field on the image with nothing on screen admitting
        it — and it carries the scrub for the same reason a glyph row does. */}
    {drawing && drawing.icon === undefined && <div className="field-quick-row">
      <button
        type="button"
        className="active"
        data-testid="field-quick-other"
        onClick={() => setOverlayAxis("off")}
      >
        <Eye width={14} height={14} strokeWidth={1.7} aria-hidden />
        <span className="field-quick-tip"><strong>{drawing.label}</strong><small>Click to hide this view.</small></span>
      </button>
      {scrub}
    </div>}
    {/* The way to everything this strip left out — the rest of the catalog, the
        plane, the solver, the setup. Selecting the tank is what opens that
        panel, so this is the existing route made visible rather than a second
        one to keep agreeing with it. */}
    <div className="field-quick-row">
      <button
        type="button"
        className="field-quick-more"
        data-testid="field-quick-more"
        onClick={() => select({ kind: "tank", id: TANK_SELECTION_ID })}
      >
        <i aria-hidden="true">⋯</i>
        <span className="field-quick-tip">
          <strong>All field views</strong>
          <small>Every view this solver publishes, its plane, and the solver behind them. Selects the tank.</small>
        </span>
      </button>
    </div>
  </div>;
}
