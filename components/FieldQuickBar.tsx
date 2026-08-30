"use client";

import { useEffect, useState } from "react";
import {
  Boxes,
  ChevronDown,
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
import { VISUALIZATION_FIELDS, VISUALIZATION_QUICK_FIELDS } from "../lib/core/visualization-catalog";
import type { FieldVisualization, FieldVisualizationIcon } from "../lib/core/visualization-registry";
import { isPressureJournalOverlayMode } from "../lib/core/webgpu-pressure-journal-overlay";
import { useSession } from "../lib/core/session/session-context";
import { DEFAULT_GRID_OVERLAY_AXIS } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import {
  ToolstripChoice,
  ToolstripRow,
  ToolstripScrub,
  useToolstripSection,
} from "./toolstrip";

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

/**
 * A catalog field narrowed to the mode union the overlay store speaks, and — for
 * the short list — to what earned it a glyph. Both narrowings are assertions, as
 * the flyout's own list is: `mode` is declared as a string beside the pass that
 * publishes it, because core cannot depend on the renderer's union.
 */
type Field = FieldVisualization & { mode: GridOverlayMode };
type QuickField = Field & { icon: FieldVisualizationIcon };

const FIELDS = VISUALIZATION_FIELDS as readonly Field[];
const QUICK_FIELDS = VISUALIZATION_QUICK_FIELDS as readonly QuickField[];

/**
 * A view's mark, drawn. `Eye` is the stand-in for a view this strip has no glyph
 * for — one chosen from the full catalog, or a pass that never claimed one —
 * because whatever is drawing has to be shown on the row that turns it off.
 *
 * Returns the element rather than the component, so the glyph is resolved where
 * it is drawn instead of being bound to a name in a render body: a capitalized
 * local holding a component is a new component identity every frame as far as
 * React is concerned, and the linter is right to call it out.
 */
function glyphOf(view: FieldVisualization, size: number) {
  const Icon: LucideIcon = view.icon === undefined ? Eye : ICONS[view.icon];
  return <Icon width={size} height={size} strokeWidth={1.7} aria-hidden />;
}

/** Whether this method publishes anything the row can offer. */
export function methodHasQuickFields(methodId: string): boolean {
  const supported = new Set(getMethod(methodId).supportedFieldModes ?? []);
  return QUICK_FIELDS.some((view) => supported.has(view.mode));
}

/**
 * One row: what is drawn on the water, and the two dials that shape it.
 *
 * This was a column — one glyph per field view worth turning on without opening
 * anything — and the column was the wrong shape for what it holds. Five marks
 * stacked at the tank's corner is five rows of chrome to say one fact, only one
 * of which can ever be lit, and it left no room beside the lit one for the
 * controls that view actually needs. A field overlay is a single-choice setting;
 * it reads as one row.
 *
 * So: the mark of what is drawing (or what would come back on), a chevron onto
 * the alternatives, and — once something *is* drawing — its plane and its scrub
 * beside it, which is the whole of what a reader adjusts while watching the
 * water. The glyph is the switch, as it was: clicking the lit one puts the view
 * away without disturbing which view is selected, so turning it back on is one
 * click on the same mark.
 *
 * Which views the chevron offers is not decided here. A pass declares an `icon`
 * on the view it thinks is worth the room (see `FieldVisualizationIcon`), and
 * the running method's `supportedFieldModes` narrows that to what this solver
 * can honestly publish — so a method that draws no volume never offers a volume
 * view, and a view added beside its pass reaches this row without this file
 * learning its name. Everything the short list leaves out is behind the door at
 * the foot of the column.
 *
 * Exported as rows rather than as a strip because the tank's own strip grows out
 * of this one: selecting the tank adds sections underneath rather than swapping
 * these for a panel.
 */
export function FieldViewRows() {
  const session = useSession();
  const methodId = session.method((state) => state.methodId);
  const overlayMode = session.ui((state) => state.gridOverlayMode);
  const overlayAxis = session.ui((state) => state.gridOverlayAxis);
  const overlaySlice = session.ui((state) => state.gridOverlaySlice);
  const setOverlayMode = session.ui((state) => state.setGridOverlayMode);
  const setOverlayAxis = session.ui((state) => state.setGridOverlayAxis);
  const setOverlaySlice = session.ui((state) => state.setGridOverlaySlice);
  // Local, not stored: it is the state of one disclosure, and a list that
  // reopened itself because the camera moved would be the strip remembering the
  // wrong thing. One row open across the whole column, so opening this list
  // closes the tank's own cards rather than overlapping them at the same corner.
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("field-quick", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
  // Escape closes it, which a dropdown owes the reader and a settings card does
  // not: this one is up *while* the reader is deciding, and the gesture that
  // abandons a decision is the same everywhere.
  useEffect(() => {
    if (!picking) return;
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPicking(false);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [picking]);

  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = QUICK_FIELDS.filter((view) => supported.has(view.mode));
  // What is drawing, whether or not this strip could have chosen it: the full
  // catalog and a link can both leave a view on that has no glyph here, and a
  // reader who cannot see what is drawn cannot turn it off either.
  const drawing = overlayAxis !== "off"
    ? FIELDS.find((view) => view.mode === overlayMode)
    : undefined;
  // With nothing drawing the row still has a subject: the view the store is
  // holding, which is what the glyph would bring back. Falling through to the
  // first offered view covers the one case where that view is not on this
  // method's list — a solver switched under a selection made for another.
  const shown = drawing
    ?? views.find((view) => view.mode === overlayMode)
    ?? views[0];
  // A view chosen from the full catalog is not on the short list, so it joins
  // it: a list that omitted the thing that is lit would be a list the reader
  // cannot use to move off it.
  const offered = shown !== undefined && !views.some((view) => view.mode === shown.mode)
    ? [shown, ...views]
    : views;

  const choose = (view: FieldOverlayView) => {
    const change = pickFieldOverlay(
      { mode: overlayMode, axis: overlayAxis }, view, volumeCapable, DEFAULT_GRID_OVERLAY_AXIS);
    if (change.mode !== undefined) setOverlayMode(change.mode);
    if (change.axis !== undefined) setOverlayAxis(change.axis);
    if (change.slice !== undefined) setOverlaySlice(change.slice);
  };

  if (shown === undefined) return null;

  // The plane belongs beside the view it cuts, not behind a selection: choosing
  // a sliced view and then having to find the tank to turn it sideways is the
  // same trip this row exists to remove. `VOL` stays on the strip when the
  // method cannot draw it, disabled and saying why — a button that vanishes
  // teaches the reader nothing about why it is gone. There is no HIDE here
  // because the lit glyph is it.
  const planes = drawing !== undefined && !drawing.planeless && <ToolstripChoice
    ariaLabel="Field view plane"
    value={overlayAxis}
    options={[
      { value: "x", label: "X" },
      { value: "y", label: "Y" },
      { value: "z", label: "Z" },
      {
        value: "volume",
        label: "VOL",
        disabled: !volumeCapable,
        title: volumeCapable ? undefined : "Volume views need an adaptive octree method",
      },
    ]}
    onChange={(value) => setOverlayAxis(value as typeof overlayAxis)}
  />;

  // The plane depth on a sliced view, the opacity on a volume or planeless one.
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

  // A dropdown under its own chevron rather than the strip's pane, which is
  // measured out past the whole column's right edge — correct for a card of
  // settings a row opens, wrong for a list of alternatives to the mark beside
  // it: it left the list hanging past the scrub with nothing tying it to what
  // it was changing. It is the same material as the strip's tips, since it is
  // the same kind of thing: a small surface that appears over the water for one
  // decision and goes away.
  const menu = picking && <div
    className="toolstrip-menu fluid-field-options"
    role="menu"
    aria-label="Field view"
  >
    {offered.map((view) => {
      const active = drawing?.mode === view.mode;
      return <button
        key={view.id}
        type="button"
        className={active ? "active" : ""}
        role="menuitemradio"
        aria-checked={active}
        title={view.description}
        data-testid={`field-quick-pick-${view.mode}`}
        onClick={() => {
          choose(view);
          // Picking is the list's whole job: it stands down and hands the row
          // back its plane and its scrub, which is what the reader came to this
          // corner to move.
          pick(false);
        }}
      >
        {glyphOf(view, 13)}
        <span>{view.label}</span>
      </button>;
    })}
  </div>;

  const open = planes || scrub ? <>{planes}{scrub}</> : undefined;

  return <ToolstripRow
    icon={glyphOf(shown, 14)}
    name={shown.label}
    hint={drawing === undefined ? shown.description : "Click to hide this view."}
    active={drawing !== undefined}
    testId={`field-quick-${shown.mode}`}
    onClick={() => choose(shown)}
    after={offered.length > 1 && <span className="toolstrip-anchor">
      <button
        type="button"
        className={`toolstrip-key is-chevron${picking ? " active" : ""}`}
        aria-label="Choose a field view"
        aria-haspopup="menu"
        aria-expanded={picking}
        data-testid="field-quick-pick"
        onClick={() => pick(!picking)}
      >
        <ChevronDown width={12} height={12} strokeWidth={2} aria-hidden />
        <span className="toolstrip-tip">
          <strong>Field view</strong>
          <small>The views worth turning on while the water runs. The door below has the rest.</small>
        </span>
      </button>
      {menu}
    </span>}
  >{open}</ToolstripRow>;
}
