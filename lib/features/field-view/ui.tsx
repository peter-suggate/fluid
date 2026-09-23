"use client";

import { useState, type ReactNode } from "react";
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
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMenuRule,
  ToolstripRow,
  useToolstripSection,
} from "../../../components/toolstrip";
import { Choice, Slider, Value } from "../../../components/ui";
import {
  pickFieldOverlay, type FieldOverlayAxis, type FieldOverlayView,
} from "../../core/field-overlay-pick";
import { isPressureJournalOverlayMode } from "../pressure-inspection/gpu/overlay";
import type {
  FieldVisualization, FieldVisualizationIcon,
} from "../../core/visualization-registry";

/**
 * What a field view looks like as a row — one component, both hosts.
 *
 * The 3-D studio and the 2-D advance lab were each drawing a list of field
 * pictures over their water, and each had written its own row for it:
 * `components/FieldQuickBar.tsx` read six members of the ui store and the
 * running method's `supportedFieldModes`, and `advance-lab/SliceToolstrip.tsx`
 * held a `LensRow` and an `OverlayRow` over sixteen stage lenses and two
 * annotations. Two rows, one capability. The declarations were already shared
 * where it counted — `lib/core/fluid-fraction-view.ts` is what both of them
 * colour V/K with — and the *row* was the copy.
 *
 * So the row is here, and the two things that genuinely differ between the
 * hosts are parameters:
 *
 *   - **`views`** — the `FieldVisualization` entries this host publishes, in
 *     the order they should be offered. The studio hands over the catalog
 *     narrowed by the running method; the lab hands over its lens roster,
 *     declared with the same `fieldVisualization()` helper beside the drawing
 *     in `advance-lab/lenses.ts`.
 *   - **`state`** — where the chosen mode, plane and scrub live, and what the
 *     host can honour about them. A studio overlay is drawn *over* the water
 *     and can be put away; a lab lens *is* the picture and has no "off", and
 *     neither has a plane to choose. Both of those are declared on the state
 *     rather than sniffed from the entries, because they are facts about the
 *     host and not about the view.
 *
 * Nothing here is a panel. Both exports are toolstrip rows, the alternatives
 * are a `ToolstripMenuButton` on the row that names the current answer, and the
 * two dials only exist while something is drawing.
 */

/**
 * The one place a field's declared mark becomes a glyph.
 *
 * Same split as `EditorActionIcon`: the vocabulary is declared in core beside
 * the views that claim it, core may not import React, so the resolution happens
 * in the UI layer — and `satisfies Record<…>` makes a name added to the union
 * without a glyph a compile error rather than a blank button.
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
 * A view's mark, drawn. `Eye` is the stand-in for a view this strip has no glyph
 * for — one chosen from the full catalog, or a pass that never claimed one —
 * because whatever is drawing has to be shown on the row that turns it off.
 *
 * Returns the element rather than the component, so the glyph is resolved where
 * it is drawn instead of being bound to a name in a render body: a capitalized
 * local holding a component is a new component identity every frame as far as
 * React is concerned, and the linter is right to call it out.
 */
export function fieldViewGlyph(view: FieldVisualization, size: number) {
  const Icon: LucideIcon = view.icon === undefined ? Eye : ICONS[view.icon];
  return <Icon width={size} height={size} strokeWidth={1.7} aria-hidden />;
}

/** The plane a host falls back to when a volume view has no raymarch to draw in. */
export type FieldViewAxis = FieldOverlayAxis;

/**
 * Where the chosen view lives, and what this host can do about it.
 *
 * The three readings and their three setters are the whole of what the row
 * touches. The two optional declarations below are the host's own bargain, in
 * the same spirit as `EditorHost`'s optional capability groups: a host says
 * what it can honour, and a control it cannot honour is not rendered rather
 * than rendered dead.
 */
export interface FieldViewState {
  readonly mode: string;
  readonly axis: FieldViewAxis;
  readonly slice: number;
  readonly setMode: (mode: string) => void;
  readonly setAxis: (axis: FieldViewAxis) => void;
  readonly setSlice: (slice: number) => void;
  /**
   * The view is drawn over something else, so clicking the lit mark puts it
   * away. False on a host whose chosen view *is* the picture — the lab's stage
   * lens — where "off" would be a blank canvas rather than a plainer one.
   *
   * Defaults to true, which is the studio's overlay.
   */
  readonly dismissable?: boolean;
  /**
   * The host offers a slice plane and a scrub beside the view. False on a host
   * whose picture has one plane and no opacity — the lab's 2-D cut — where the
   * two dials would be four buttons and a slider that change nothing.
   *
   * Defaults to true.
   */
  readonly adjustable?: boolean;
  /** The plane a volume view falls back to on a host with no raymarch. */
  readonly defaultAxis?: Exclude<FieldViewAxis, "off">;
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
 * The chevron offers everything the host publishes, in two halves: the views a
 * pass claimed an `icon` for (see `FieldVisualizationIcon`) lead with their
 * marks, then a hairline, then the rest with their swatches. That second half
 * used to be a FIELD row of its own opening a pane, which was a row reporting
 * one word over a list this menu already is.
 *
 * Which views those are is not decided here. The pass declares its mark, and
 * the caller narrows the roster to what this host can honestly publish — so a
 * method that draws no volume never offers a volume view, and a view added
 * beside its pass reaches this row without this file learning its name.
 *
 * Exported as rows rather than as a strip because the tank's own strip grows
 * out of this one: selecting the tank adds sections underneath rather than
 * swapping these for a panel.
 */
export function FieldViewRows({ views, catalog, state, volumeCapable }: {
  /** What this host offers, in the order it offers them. */
  readonly views: readonly FieldVisualization[];
  /**
   * What the chosen mode may name even when this host could not have picked it
   * — a focused panel or a link can leave a hidden or unsupported view drawing,
   * and a reader who cannot see what is drawn cannot turn it off either.
   * Defaults to `views`.
   */
  readonly catalog?: readonly FieldVisualization[];
  readonly state: FieldViewState;
  readonly volumeCapable: boolean;
}) {
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

  const all = catalog ?? views;
  const dismissable = state.dismissable ?? true;
  const adjustable = state.adjustable ?? true;
  const defaultAxis = state.defaultAxis ?? "z";
  // Two lists, one menu. The first is the short list a pass claimed a glyph for
  // — the handful worth turning on while watching the water — and the second is
  // everything else this host publishes: twenty more on the Losasso methods,
  // which used to be a row of their own opening a pane. That row is gone, so
  // this menu is the whole picker, and the seam between the two halves is a
  // hairline rather than a second surface to go and find.
  const quick = views.filter((view) => view.icon !== undefined && !view.hidden);
  const rest = views.filter((view) => !view.hidden
    && !quick.some((marked) => marked.mode === view.mode));
  // What is drawing, whether or not this strip could have chosen it.
  const drawing = dismissable && state.axis === "off" ? undefined
    : all.find((view) => view.mode === state.mode);
  // With nothing drawing the row still has a subject: the view the store is
  // holding, which is what the glyph would bring back. Falling through to the
  // first offered view covers the one case where that view is not on this
  // host's list — a solver switched under a selection made for another.
  const shown = drawing
    ?? quick.find((view) => view.mode === state.mode)
    ?? quick[0];
  // A hidden view is in neither list — it is one a focused panel selected — so
  // it joins the short list while it is drawing: a menu that omitted the thing
  // that is lit is a menu the reader cannot use to move off it.
  const listed = (view: FieldVisualization) => quick.some((marked) => marked.mode === view.mode)
    || rest.some((other) => other.mode === view.mode);
  const offered = shown !== undefined && !listed(shown) ? [shown, ...quick] : quick;

  const choose = (view: FieldVisualization) => {
    // A host whose view is the picture has no overlay to hide and no plane to
    // adopt, so the pick is the whole of the change. Everywhere else the rule
    // that "picking the lit one hides it" is `pickFieldOverlay`'s, because two
    // surfaces choose field views and that rule must not be written twice.
    if (!dismissable) {
      state.setMode(view.mode);
      return;
    }
    const change = pickFieldOverlay(
      { mode: state.mode as FieldOverlayView["mode"], axis: state.axis },
      view as FieldOverlayView, volumeCapable, defaultAxis);
    if (change.mode !== undefined) state.setMode(change.mode);
    if (change.axis !== undefined) state.setAxis(change.axis);
    if (change.slice !== undefined) state.setSlice(change.slice);
  };

  if (shown === undefined) return null;

  // The plane belongs beside the view it cuts, not behind a selection: choosing
  // a sliced view and then having to find the tank to turn it sideways is the
  // same trip this row exists to remove. `VOL` stays on the strip when the
  // method cannot draw it, disabled and saying why — a button that vanishes
  // teaches the reader nothing about why it is gone. There is no HIDE here
  // because the lit glyph is it.
  const planes = adjustable && drawing !== undefined && !drawing.planeless && <Choice<FieldViewAxis>
    ariaLabel="Field view plane"
    value={state.axis}
    options={[
      { value: "x", label: "X" },
      { value: "y", label: "Y" },
      { value: "z", label: "Z" },
      {
        value: "volume",
        label: "VOL",
        disabled: !volumeCapable || drawing.sliceOnly,
        hint: drawing.sliceOnly
          ? "This diagnostic is drawn on an X, Y, or Z slice"
          : volumeCapable ? undefined : "Volume views need an adaptive octree method",
      },
    ]}
    onChange={state.setAxis}
  />;

  // The plane depth on a sliced view, the opacity on a volume or planeless one.
  // A captured solve is the exception: its scrub steps through the iterations
  // that were snapshotted rather than sliding through a plane, and the plot that
  // makes a stop mean anything is in the film pane — so the scrub goes there
  // with it rather than standing here detached from its own curve.
  const volumetric = drawing !== undefined && (drawing.planeless || state.axis === "volume");
  const scrub = adjustable && drawing && !isPressureJournalOverlayMode(drawing.mode)
    && <>
      <Slider
        min={volumetric ? 0.05 : 0}
        max={1}
        step={volumetric ? 0.01 : 0.005}
        value={state.slice}
        ariaLabel={volumetric ? `${drawing.label} opacity` : `Field ${state.axis} slice position`}
        onInput={state.setSlice}
      />
      <Value value={`${Math.round(state.slice * 100)}%`} />
    </>;

  const item = (view: FieldVisualization, glyph: boolean) => <ToolstripMenuItem
    key={view.id}
    icon={glyph ? fieldViewGlyph(view, 13) : undefined}
    swatch={view.swatch}
    label={view.label}
    note={view.figure}
    title={view.description}
    active={drawing?.mode === view.mode}
    testId={`field-quick-pick-${view.mode}`}
    onClick={() => {
      choose(view);
      // Picking is the list's whole job: it stands down and hands the row back
      // its plane and its scrub, which is what the reader came to this corner
      // to move.
      pick(false);
    }}
  />;

  const open = planes || scrub ? <>{planes}{scrub}</> : undefined;

  return <ToolstripRow
    icon={fieldViewGlyph(shown, 14)}
    name={shown.label}
    hint={drawing === undefined || !dismissable ? shown.description : "Click to hide this view."}
    active={drawing !== undefined}
    testId={`field-quick-${shown.mode}`}
    onClick={() => choose(shown)}
    after={offered.length + rest.length > 1 && <ToolstripMenuButton
      label="Field view"
      hint="Every view this solver publishes; the ones with a mark are the short list."
      open={picking}
      testId="field-quick-pick"
      onOpen={pick}
    >
      {offered.map((view) => item(view, true))}
      {offered.length > 0 && rest.length > 0 && <ToolstripMenuRule />}
      {rest.map((view) => item(view, false))}
    </ToolstripMenuButton>}
  >{open}</ToolstripRow>;
}

/** The swatch a composable annotation is identified by, in the column's grain. */
const DOT: React.CSSProperties =
  { display: "block", width: 9, height: 9, borderRadius: 2, flex: "none" };

/**
 * Whether each annotation is on, whether it can be drawn at all, and the key.
 *
 * Three closures rather than three sets because the answer is per view and the
 * host already has it: the lab's normal overlay is offered exactly while the
 * transport reconstructs a surface to take a normal from, which is a fact about
 * the run and not a list this row could be handed once.
 */
export interface FieldOverlayState {
  readonly enabled: (mode: string) => boolean;
  readonly offered: (mode: string) => boolean;
  readonly toggle: (mode: string) => void;
  /** The key that toggles it, when the host binds one. */
  readonly shortcut?: (mode: string) => string | undefined;
  /** Why an unoffered annotation cannot be drawn right now. */
  readonly unavailable?: (mode: string) => string | undefined;
}

/**
 * The annotations, as switches.
 *
 * Independent of each other and of whatever view is up, so these are rows and
 * not a choice: a reader comparing the fraction a cell holds against the normal
 * it was given wants both at once, and making them exclusive would be the page
 * deciding that question for them. It is the same split the 3-D catalog draws
 * between a *field* and a *decoration*, and for the same reason — which is why
 * the entries are the same `FieldVisualization` records `FieldViewRows` takes,
 * and why their swatches come from the quantity's own definition rather than
 * from a colour typed beside the row.
 */
export function FieldOverlayRows({ views, state }: {
  readonly views: readonly FieldVisualization[];
  readonly state: FieldOverlayState;
}): ReactNode {
  return <>{views.map((view) => {
    const on = state.enabled(view.mode);
    const offered = state.offered(view.mode);
    const key = state.shortcut?.(view.mode);
    return <ToolstripRow
      key={view.id}
      icon={<i aria-hidden
        style={{ ...DOT, background: view.swatch, opacity: on ? 1 : 0.32 }} />}
      name={key === undefined ? view.label : `${view.label} (${key})`}
      hint={offered ? view.description
        : state.unavailable?.(view.mode) ?? view.description}
      active={on}
      disabled={!offered}
      testId={`field-overlay-${view.mode}`}
      onClick={() => state.toggle(view.mode)}
    />;
  })}</>;
}
