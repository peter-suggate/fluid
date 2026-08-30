"use client";

import { useState, type KeyboardEvent } from "react";
import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import type { MethodParamSpec, SelectParamSpec } from "@/lib/core/method-contract";
import type { GPUQuality } from "../lib/core/gpu-quality";
import { RangeControl } from "./controls";
import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import type { FieldVisualization } from "../lib/core/visualization-registry";
import { simulation } from "../lib/core/simulation/controller";
import {
  stageLensOverlayMode,
  type AnyStageLens,
  type StageLensReceipt,
} from "../lib/core/stage-lens";
import { pickFieldOverlay, type FieldOverlayView } from "../lib/core/field-overlay-pick";
import { methodHasQuickFields } from "./FieldQuickBar";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { DEFAULT_GRID_OVERLAY_AXIS } from "../lib/core/stores/ui-store";
import { isPressureJournalOverlayMode } from "../lib/core/webgpu-pressure-journal-overlay";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import { LegendEntries } from "./VisualizationLegend";
import { PressureFilmStrip } from "./PressureFilmStrip";
import {
  ToolstripChoice,
  ToolstripPane,
  ToolstripRow,
  ToolstripScrub,
  useToolstripSection,
  type ToolstripTab,
} from "./toolstrip";

/**
 * Everything about the water that is not one of the glyphs above it.
 *
 * These are rows on the tank's own toolstrip rather than a panel of their own:
 * selecting the tank *grows* the strip that is already standing at that corner
 * instead of swapping it for a card. What the glyph rows leave out is the rest
 * of the catalog, the plane the overlay is drawn through, the solver behind it
 * and that solver's construction — four rows, each reporting its own current
 * answer, each opening beside itself rather than pushing the others down.
 *
 * The Performance panel's observatory remains the annotated read of these views
 * — legends, sources, live stats — but choosing one and sweeping a slice are
 * edits worth making dozens of times while watching the water, and a panel you
 * have to open first is the wrong home for that loop.
 *
 * Selection semantics match the observatory cards: picking the active view
 * hides the overlay, picking another switches the publication without touching
 * the chosen plane, and only an inactive overlay adopts the view's authored
 * axis. The rule itself is `pickFieldOverlay`, shared with the glyph rows.
 *
 * A row that has more to say than a line opens a **pane** — the dozen field
 * rows, a lens's phases, a captured solve's film, the construction settings.
 * That pane is deliberately generic and there is one at a time, so the next
 * thing worth inspecting reuses it rather than becoming a fifth stacked group.
 */
type FieldView = FieldVisualization & { mode: GridOverlayMode };

/**
 * Which secondary surface the side column is showing, if any.
 *
 * A lens carries its id because there is one pane per lens rather than one pane
 * showing whichever is selected: the pane and the selection can disagree for a
 * frame, and resolving by id means the mismatch closes the column rather than
 * drawing one lens's phases under another lens's name.
 */
type FieldDetail =
  | { readonly kind: "views" }
  | { readonly kind: "lens"; readonly id: string }
  | { readonly kind: "film" }
  | { readonly kind: "solver" }
  | { readonly kind: "setup" };

function sameDetail(a: FieldDetail | undefined, b: FieldDetail): boolean {
  if (!a || a.kind !== b.kind) return false;
  return a.kind !== "lens" || b.kind !== "lens" || a.id === b.id;
}

/**
 * The whole label, plus the hint, as one tooltip.
 *
 * The panel is a strip over the scene and several authored parameter labels are
 * longer than it is wide, so a row ellipsizes rather than forcing the panel to
 * scroll sideways — which makes the tooltip the only place the full label
 * exists. It therefore leads with the label rather than carrying only the hint.
 */
function paramTitle(spec: { label: string; hint?: string }) {
  return spec.hint ? `${spec.label} — ${spec.hint}` : spec.label;
}

/**
 * One tuning of the selected method, as the method itself declared it.
 *
 * Lifted here from the configuration popover's Method panel when that panel was
 * cut: the solver row is on this widget, and a parameter that belongs to the
 * solver has to be reachable from wherever the solver is chosen — otherwise
 * switching method and then tuning it is two surfaces again.
 *
 * Both kinds render the same row anatomy — a small tag label over a full-width
 * control, any readout right-aligned on the label's line. `RangeControl` is the
 * studio's dial and keeps its commit-on-release semantics; only its typography
 * is brought down to this panel's scale, by flyout-scoped rules rather than by
 * a second copy of the control.
 */
function MethodParamControl({ spec, methodId }: { spec: MethodParamSpec; methodId: string }) {
  const session = useSession();
  const methodState = session.method();
  const values = resolvedMethodValues(methodState);
  const overridden = spec.key in (methodState.overrides[methodId] ?? {});
  if (spec.kind === "select") {
    return (
      <label className="select-control" title={paramTitle(spec)}>
        <span>{spec.label}</span>
        <select
          value={String(values[spec.key])}
          onChange={(event) => simulation.setMethodParam(methodId, spec.key, event.currentTarget.value, session.id)}
        >
          {spec.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    );
  }
  return (
    <RangeControl
      label={spec.label}
      unit={spec.unit}
      value={Number(values[spec.key])}
      min={spec.min} max={spec.max} step={spec.step}
      displayDigits={spec.digits ?? 3}
      hint={paramTitle(spec)}
      modified={overridden}
      onReset={() => simulation.resetMethodParam(methodId, spec.key, session.id)}
      onChange={(value) => simulation.setMethodParam(methodId, spec.key, value, session.id)}
    />
  );
}

const FIELD_VIEWS: readonly FieldView[] = VISUALIZATION_FIELDS
  .filter((field) => !field.hidden) as readonly FieldView[];

/**
 * The open lens's scrubber, its counters and its key.
 *
 * A rail of ticks rather than a select or a strip of chips because the phases
 * are *ordered* — they are one reading of one stage walked from plan to accept,
 * and the gesture is sweeping along it and back, not picking an item out of a
 * list. Only the phase under the head is named: seven labels across a 200 px
 * panel is a wall of text over the water, and the whole direction here is that
 * the lens decorates the grid rather than annotating it.
 *
 * Every number the stage produced this frame is drawn, including the ones it
 * did not: a counter the receipt has no word for reads as an em dash, never as
 * zero, and a tap the frame never encoded is shown as a shortfall in the fault
 * colour. Absence is drawn.
 *
 * The rules are inline rather than in the flyout's stylesheet, but every value
 * is a design token, so the panel's theme still owns the look — keep it that
 * way, since a literal colour here would be the one thing that does not follow
 * the water into dark.
 */
function LensPhaseScrubber({
  lens, phase, receipt, onPhase,
}: {
  readonly lens: AnyStageLens;
  readonly phase: number;
  readonly receipt: StageLensReceipt | undefined;
  readonly onPhase: (index: number) => void;
}) {
  const index = Math.max(0, Math.min(lens.phases.length - 1, phase));
  const current = lens.phases[index];
  const shortfall = receipt && receipt.capturedTaps < receipt.expectedTaps;
  // Arrow keys move the head and the focus together: a scrubber whose value
  // stepped away from the button still holding focus would leave the next
  // arrow press stepping from somewhere the reader cannot see.
  const step = (event: KeyboardEvent<HTMLDivElement>) => {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (delta === 0) return;
    event.preventDefault();
    const next = Math.max(0, Math.min(lens.phases.length - 1, index + delta));
    onPhase(next);
    event.currentTarget.querySelectorAll("button")[next]?.focus();
  };
  return <div style={{ display: "grid", gap: 2 }}>
    <div
      role="group"
      aria-label={`${lens.label} phase`}
      onKeyDown={step}
      style={{ position: "relative", display: "flex", alignItems: "center", height: 14 }}
    >
      <span
        aria-hidden="true"
        style={{ position: "absolute", insetInline: 2, top: "50%", height: 1, background: "var(--line)" }}
      />
      {lens.phases.map((candidate, tick) => <button
        key={candidate.id}
        type="button"
        title={candidate.label}
        aria-label={candidate.label}
        aria-pressed={tick === index}
        tabIndex={tick === index ? 0 : -1}
        onClick={() => onPhase(tick)}
        // `display` and `minHeight` are written back over the row styling this
        // list gives every button under it: a tick is a mark on a rail, not
        // another 20 px row, and the same rules would stretch the dot too.
        style={{
          position: "relative", display: "block", flex: 1, height: "100%", minHeight: 0,
          padding: 0, border: "none", background: "transparent", cursor: "pointer",
        }}
      >
        <span style={{
          display: "block", flex: "none", margin: "0 auto", borderRadius: "50%",
          width: tick === index ? 7 : 3, height: tick === index ? 7 : 3,
          background: tick === index ? "var(--accent)" : "var(--line-strong)",
        }} />
      </button>)}
    </div>
    <div style={{ display: "flex", gap: 6, justifyContent: "space-between", font: "var(--text-micro)" }}>
      <span style={{ overflow: "hidden", color: "var(--accent)", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {current?.label ?? "—"}
      </span>
      <span style={{ flex: "none", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
        {index + 1}/{lens.phases.length}
      </span>
    </div>
    {shortfall && <p style={{ margin: 0, color: "var(--red)", font: "var(--text-micro)" }}>
      taps {receipt.capturedTaps}/{receipt.expectedTaps}
    </p>}
    {current && current.counters.length > 0 && <dl style={{ display: "grid", gap: 1, margin: 0 }}>
      {current.counters.map(({ word, lit }) => {
        const value = receipt?.counters[word];
        return <div
          key={word}
          style={{
            display: "flex", gap: 6, justifyContent: "space-between",
            color: lit ? "var(--ink)" : "var(--muted)", font: "var(--text-micro)",
          }}
        >
          <dt style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{word}</dt>
          <dd style={{ flex: "none", margin: 0, fontVariantNumeric: "tabular-nums" }}>
            {value === undefined ? "—" : value.toLocaleString()}
          </dd>
        </div>;
      })}
    </dl>}
    {lens.legend.length > 0 && <LegendEntries
      entries={lens.legend}
      leadMark
      style={{
        display: "grid", gap: 1, margin: 0, padding: 0,
        color: "var(--muted)", font: "var(--text-micro)", listStyle: "none",
      }}
    />}
  </div>;
}

export function FieldControlRows({ lenses: override }: {
  /** Overrides the running method's own roster. For tests and previews. */
  lenses?: readonly AnyStageLens[];
}) {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const overlayMode = session.ui((state) => state.gridOverlayMode);
  const overlayAxis = session.ui((state) => state.gridOverlayAxis);
  const overlaySlice = session.ui((state) => state.gridOverlaySlice);
  const lensPhase = session.ui((state) => state.gridOverlayLensPhase);
  const setOverlayMode = session.ui((state) => state.setGridOverlayMode);
  const setOverlayAxis = session.ui((state) => state.setGridOverlayAxis);
  const setOverlaySlice = session.ui((state) => state.setGridOverlaySlice);
  const setLensPhase = session.ui((state) => state.setGridOverlayLensPhase);
  const lensReceipt = session.diagnostics((state) => state.stageLensReceipt);
  // Which pane the side column holds. Local, not stored: it is the state of a
  // disclosure on one panel, and a column that reopened itself because the
  // camera moved would be the panel remembering the wrong thing.
  const [detail, setDetail] = useState<FieldDetail | undefined>(undefined);
  // Only the views this method registered: a picker offering a publication the
  // solver never produces is a button that draws nothing.
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const methodValues = resolvedMethodValues(methodState);
  // Declared on the method rather than fetched from the solver, so the section
  // is there before a device is.
  const lenses = override ?? method.stageLenses ?? [];
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = FIELD_VIEWS.filter((view) => supported.has(view.mode));
  const active = overlayAxis !== "off"
    ? views.find((view) => view.mode === overlayMode)
    : undefined;
  const activeLens = overlayAxis !== "off"
    ? lenses.find((lens) => stageLensOverlayMode(lens.stage) === overlayMode)
    : undefined;
  // The plane buttons, HIDE and the slice scrub belong to whatever is drawing,
  // and a lens draws on the same plane a field view does. Without this a
  // selected lens would take the panel's only way to turn the overlay off down
  // with the catalog row it does not have.
  const shown = active ?? (activeLens && { label: activeLens.label, planeless: false });
  // The solver row stays even when a method registers no field views: this
  // widget is also where the method is switched, and a picker that vanished
  // with the fields would strand such a method with no way back.
  const hasViews = views.length > 0;

  // A film view's slider is a scrub through the captured iterations, not an
  // opacity. The iteration each stop lands on is knowable here without asking
  // the solver anything: the snapshot schedule is a pure function of the
  // iteration ceiling and the reserved capacity, which is the same property
  // that lets the device pick its slot without the host telling it.
  const filmMode = active !== undefined && isPressureJournalOverlayMode(active.mode);
  const filmReserved = method.pressureJournal?.isReserved(methodValues) ?? false;
  const filmReserve = method.pressureJournal?.reserve;
  const filmSchedule = filmMode
    ? method.pressureJournal?.schedule(methodValues) ?? []
    : [];
  const filmSlot = filmSchedule.length > 0
    ? Math.round(Math.max(0, Math.min(1, overlaySlice)) * (filmSchedule.length - 1))
    : 0;

  // Resolved against what is true this frame rather than synchronised with it:
  // a pane whose subject went away — the lens deselected, the overlay hidden,
  // the method switched out from under it — simply stops resolving, so the
  // column closes itself instead of holding a stale reading open beside a scene
  // that no longer produces it.
  const open = detail === undefined ? undefined
    : detail.kind === "views" ? (hasViews ? detail : undefined)
      : detail.kind === "lens" ? (activeLens?.id === detail.id ? detail : undefined)
        : detail.kind === "film" ? (filmMode ? detail : undefined)
          : detail;
  // One row open across the whole strip, not one per section: these rows and
  // the selected object's own hang their cards off the same corner, so two open
  // at once overlap.
  const { claim } = useToolstripSection("field", () => setDetail(undefined));
  const openDetail = (next: FieldDetail | undefined) => {
    claim(next !== undefined);
    setDetail(next);
  };
  const toggle = (next: FieldDetail) => openDetail(sameDetail(open, next) ? undefined : next);

  // Narrowed to what selecting actually reads, so a lens goes down the same
  // path as a catalog row rather than through a second one that would have to
  // be kept agreeing with it about what "picking the active view" means. The
  // rule itself is shared with the quick bar, for the same reason.
  const selectView = (view: FieldOverlayView) => {
    const change = pickFieldOverlay(
      { mode: overlayMode, axis: overlayAxis }, view, volumeCapable, DEFAULT_GRID_OVERLAY_AXIS);
    if (change.mode !== undefined) setOverlayMode(change.mode);
    if (change.axis !== undefined) setOverlayAxis(change.axis);
    if (change.slice !== undefined) setOverlaySlice(change.slice);
  };

  const closePane = () => openDetail(undefined);
  // The plane buttons and HIDE are one chooser, not a strip plus an exception:
  // "off" is a value of the same setting, and splitting it out was what put a
  // button inside the summary line that had to argue with its own disclosure.
  const planeOptions = [
    ...(shown && !shown.planeless ? (["x", "y", "z"] as const).map((axis) => ({
      value: axis, label: axis.toUpperCase(),
    })) : []),
    ...(shown && !shown.planeless ? [{
      value: "volume",
      label: "VOL",
      disabled: !volumeCapable,
      title: volumeCapable ? undefined : "Volume views need an adaptive octree method",
    }] : []),
    { value: "off", label: "HIDE" },
  ];

  // The whole catalog, hung off the row that names what is currently drawing.
  const viewsPane = open?.kind === "views"
    && <ToolstripPane label="Field" onClose={closePane}>
        <div className="fluid-field-options">
          {views.map((view) => {
            const isActive = active?.mode === view.mode;
            return <button
              key={view.id}
              type="button"
              className={isActive ? "active" : ""}
              aria-pressed={isActive}
              title={view.description}
              onClick={() => {
                selectView(view);
                // Picking is the list's whole job, so it stands down afterwards
                // and hands the pane to the chosen view's own control surface —
                // which for a film is the curve, and for everything else is
                // nothing.
                openDetail(isPressureJournalOverlayMode(view.mode) ? { kind: "film" } : undefined);
              }}
            >
              <i style={view.swatch ? { background: view.swatch } : undefined} />
              <span>{view.label}</span>
              <small>{view.figure ?? ""}</small>
            </button>;
          })}
        </div>
      </ToolstripPane>;

  return <>
    {hasViews && <ToolstripRow
      tag="FIELD"
      value={shown?.label ?? "Hidden"}
      name="Field view"
      hint="Every view this solver publishes. The strip above is the short list; this is all of it."
      active={open?.kind === "views"}
      testId="fluid-field-row-field"
      onClick={() => toggle({ kind: "views" })}
    >{viewsPane}</ToolstripRow>}
    {/* Always shown rather than opened, because this is the control a reader
        works while the water moves — and it is the only way to put away a lens,
        which has no row of its own anywhere else.

        Withheld when the quick row above is already carrying the plane. That row
        grows the same chooser beside whatever field view is drawing, and two
        copies of one setting forty pixels apart in one column is the reader
        asking which of them is the real one. A *lens* keeps this row: the quick
        row is built from the field catalog and a lens is not in it, so nothing
        above would be offering the plane in its place. */}
    {shown && !(active !== undefined && methodHasQuickFields(methodId)) && <ToolstripRow
      tag="PLANE"
      name="Overlay plane"
      hint="Which plane the overlay is drawn through, and the way to put it away."
    >
      <ToolstripChoice
        ariaLabel="Field view plane"
        value={overlayAxis}
        options={planeOptions}
        onChange={(value) => setOverlayAxis(value as typeof overlayAxis)}
      />
    </ToolstripRow>}
    {/* A film's curve is the one pane not reachable from a row of its own kind:
        the view that produces it is chosen in the field list, which then stands
        down. Without this the pane could be closed and never reopened, and the
        scrub above would be stepping through stops with nothing plotting them. */}
    {filmMode && <ToolstripRow
      tag="FILM"
      value={filmReserved ? `${filmSchedule.length} stops` : "none reserved"}
      name="Captured solve"
      hint="The pressure solve this scrub replays, iteration by iteration."
      active={open?.kind === "film"}
      testId="fluid-field-row-film"
      onClick={() => toggle({ kind: "film" })}
    >
      {open?.kind === "film" && <ToolstripPane label="Film" onClose={closePane}>
        {/* The curve belongs beside the scrub, not under it: a frame of the film
            means something different at the third iteration than at the
            thirtieth, and knowing which needs the plot and the slider both on
            screen at once. Selecting a stop drives the same slider. */}
        {filmReserved && <ToolstripScrub
          min={0}
          max={1}
          // A film's stops are the captured iterations and nothing between them,
          // so the scrub steps between snapshots rather than sliding through a
          // continuum it cannot show.
          step={filmSchedule.length > 1 ? 1 / (filmSchedule.length - 1) : 1}
          value={overlaySlice}
          readout={filmSchedule.length > 0 ? `iter ${filmSchedule[filmSlot]}` : "—"}
          ariaLabel={`${shown?.label ?? "Film"} iteration`}
          onChange={setOverlaySlice}
        />}
        {filmReserved
          ? <PressureFilmStrip
            slot={filmSlot}
            onSelectSlot={(slot) => setOverlaySlice(filmSchedule.length > 1
              ? slot / (filmSchedule.length - 1) : 0)}
          />
          : <p className="fluid-field-film" data-testid="fluid-field-film-off">
            <span>No film reserved — this view has nothing to replay.</span>
            {filmReserve && <button
              type="button"
              data-testid="fluid-field-film-reserve"
              title="Reserve room to capture one pressure solve. Rebuilds the solver once."
              onClick={() => simulation.setMethodParam(methodId, filmReserve.parameter, filmReserve.value, session.id)}
            >RESERVE</button>}
          </p>}
      </ToolstripPane>}
    </ToolstripRow>}
    {/* What the *stages* did, as opposed to what the state is. A field row picks
        a publication; a lens row opens one pass's own reading of itself, and the
        scrubber only means anything inside one. */}
    {lenses.length > 0 && <ToolstripRow
      tag="STAGES"
      value={activeLens?.label ?? `${lenses.length} lenses`}
      name="Solver stage lenses"
      hint="What one pass did this frame, walked from plan to accept."
      active={open?.kind === "lens"}
      testId="fluid-field-row-stages"
      onClick={() => toggle({ kind: "lens", id: activeLens?.id ?? lenses[0]!.id })}
    >
      {open?.kind === "lens" && <ToolstripPane
        label={activeLens?.label ?? "Stages"}
        onClose={closePane}
      >
        <div className="fluid-field-options" role="group" aria-label="Solver stage lenses">
          {lenses.map((lens) => {
            const mode = stageLensOverlayMode(lens.stage);
            const isActive = activeLens?.id === lens.id;
            return <button
              key={lens.id}
              type="button"
              className={isActive ? "active" : ""}
              aria-pressed={isActive}
              title={lens.description}
              onClick={() => {
                selectView({ mode, axis: DEFAULT_GRID_OVERLAY_AXIS });
                openDetail({ kind: "lens", id: lens.id });
              }}
            >
              <i style={lens.legend[0] ? { background: lens.legend[0].swatch } : undefined} />
              <span>{lens.label}</span>
            </button>;
          })}
        </div>
        {/* Opening a lens opens its scrubber, because the phases *are* the lens:
            a selected lens with no scrubber on screen is one reading of a stage
            with no way to walk to the other six. */}
        {activeLens && <LensPhaseScrubber
          lens={activeLens}
          phase={lensPhase}
          receipt={lensReceipt?.lensId === activeLens.id ? lensReceipt : undefined}
          onPhase={setLensPhase}
        />}
      </ToolstripPane>}
    </ToolstripRow>}
    {/* The solver behind these fields. It lives on this strip because switching
        methods is part of the same watch-the-water loop as choosing a view. */}
    <ToolstripRow
      tag="SOLVER"
      value={method.shortLabel}
      name="Fluid solver"
      hint={method.description}
      active={open?.kind === "solver"}
      testId="fluid-field-row-solver"
      onClick={() => toggle({ kind: "solver" })}
    >
      {open?.kind === "solver" && <ToolstripChoice
        ariaLabel="Fluid solver method"
        value={methodId}
        options={interactiveSimulationMethods().map((candidate) => ({
          value: candidate.id, label: candidate.shortLabel, title: candidate.description,
        }))}
        onChange={(value) => simulation.setMethod(value, session.id)}
      />}
    </ToolstripRow>
  </>;
}

/**
 * The solver's construction settings, as the panel's first face.
 *
 * These moved off the strip. As a row they were one line reporting the quality
 * and a card holding nine controls — the tallest thing the column opened, and
 * by a distance the thing most often opened, which is a bad trade for a strip
 * whose rows are meant to be readable at a glance. They are now the tab the
 * panel opens on, with the room to say what they are.
 *
 * Quality leads because it is the method's own coarsest dial — how much grid the
 * solver is given — so it stands above the options it scales.
 */
export function MethodSetupTab() {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const method = getMethod(methodId);
  const values = resolvedMethodValues(methodState);
  const selects = method.params.filter((spec): spec is SelectParamSpec =>
    spec.tier === "coarse" && spec.kind === "select");
  const dials = method.params.filter((spec) => spec.tier === "coarse" && spec.kind !== "select");
  return <div className="fluid-field-settings" role="group" aria-label="Solver setup">
    {method.showQualityControl !== false && <label className="select-control" title={method.pressureMapping}>
      <span>Quality</span>
      <select
        aria-label="Simulation quality"
        value={methodState.quality}
        onChange={(event) => simulation.setQuality(event.currentTarget.value as GPUQuality, session.id)}
      >
        {(["balanced", "high", "ultra"] as const).map((level) => (
          <option key={level} value={level}>{level[0]!.toUpperCase() + level.slice(1)} · {method.qualityLabels[level]}</option>
        ))}
      </select>
    </label>}
    {selects.map((spec) => <label className="select-control" key={spec.key} title={paramTitle(spec)}>
      <span>{spec.label}</span>
      <select
        value={String(values[spec.key])}
        onChange={(event) => simulation.setMethodParam(methodId, spec.key, event.currentTarget.value, session.id)}
      >
        {spec.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>)}
    {dials.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
  </div>;
}

/**
 * The fine tier, as its own face.
 *
 * A disclosure at the foot of the setup list, which is where these used to be,
 * is a scroll away from the settings that are actually reached for — and it made
 * the panel's height depend on whether it had been opened. A tab of its own
 * costs one word in the bar and keeps the first face one screenful.
 */
export function MethodAdvancedTab() {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const fine = getMethod(methodId).params.filter((spec) => spec.tier === "fine");
  return <div className="fluid-field-settings" role="group" aria-label="Advanced solver parameters">
    {fine.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
  </div>;
}

/**
 * The faces this method's setup is worth showing, for the panel to lead with.
 *
 * A plain function rather than a hook: the caller decides whether the panel
 * exists at all, and a method that declares no fine parameters must not grow an
 * empty Advanced tab.
 */
export function methodSetupTabs(methodId: string): readonly ToolstripTab[] {
  const method = getMethod(methodId);
  const setup = method.showQualityControl !== false || method.params.some((spec) => spec.tier === "coarse");
  return [
    ...(setup ? [{ id: "setup", label: "Setup", content: <MethodSetupTab /> }] : []),
    ...(method.params.some((spec) => spec.tier === "fine")
      ? [{ id: "advanced", label: "Advanced", content: <MethodAdvancedTab /> }]
      : []),
  ];
}
