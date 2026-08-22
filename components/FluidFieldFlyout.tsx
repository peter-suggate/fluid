"use client";

import { useState, type KeyboardEvent, type ReactNode } from "react";
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
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { resolvedMethodValues, useMethodStore } from "../lib/core/stores/method-store";
import { DEFAULT_GRID_OVERLAY_AXIS, useUIStore } from "../lib/core/stores/ui-store";
import { isPressureJournalOverlayMode } from "../lib/core/webgpu-pressure-journal-overlay";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import { LegendEntries } from "./VisualizationLegend";
import {
  sparseCM12PressureJournalSchedule,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-journal";
import {
  SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS,
  sparseCM12PressureIterations,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { useAnchoredFlyout } from "./anchored-flyout";
import { PressureFilmStrip } from "./PressureFilmStrip";

/**
 * The field-overlay picker, riding a corner of the fluid container.
 *
 * The Performance panel's observatory remains the annotated read of these
 * views — legends, sources, live stats — but choosing one and sweeping a slice
 * are edits worth making dozens of times while watching the water, and a panel
 * you have to open first is the wrong home for that loop. Like the scale
 * overlay, this shows while the tank or the water body is selected, so "inspect
 * the fluid" is part of the same gesture as "edit the fluid" rather than a
 * fourth permanent cluster of viewport chrome.
 *
 * Selection semantics match the observatory cards: picking the active view
 * hides the overlay, picking another switches the publication without touching
 * the chosen plane, and only an inactive overlay adopts the view's authored
 * axis.
 *
 * Two columns. The **rail** is what is true right now, a line per fact: which
 * field is drawing, on which plane, at which depth, by which solver, at which
 * setup. Everything with more to say than a line — the dozen field rows, a
 * lens's phases, a captured solve's film, the construction settings — opens in
 * the **side column** instead. That column is deliberately generic: it takes
 * one occupant at a time and every rail row that owns one opens it the same
 * way, so the next thing worth inspecting reuses it rather than becoming a
 * fifth stacked group. The scene is the hero, and the resting height is the
 * price of admission.
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
  const methodState = useMethodStore();
  const values = resolvedMethodValues(methodState);
  const overridden = spec.key in (methodState.overrides[methodId] ?? {});
  if (spec.kind === "select") {
    return (
      <label className="select-control" title={paramTitle(spec)}>
        <span>{spec.label}</span>
        <select
          value={String(values[spec.key])}
          onChange={(event) => simulation.setMethodParam(methodId, spec.key, event.currentTarget.value)}
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
      onReset={() => simulation.resetMethodParam(methodId, spec.key)}
      onChange={(value) => simulation.setMethodParam(methodId, spec.key, value)}
    />
  );
}

const FIELD_VIEWS: readonly FieldView[] = VISUALIZATION_FIELDS
  .filter((field) => !field.hidden) as readonly FieldView[];

/**
 * A rail row that owns a pane in the side column.
 *
 * Tag, current answer, chevron. One anatomy for every pane there is, so which
 * rows lead somewhere is legible before any of them is clicked — and the answer
 * stays on the rail while the pane is shut, which is the whole reason a row may
 * stand in for a section at all.
 */
function DetailRow({ tag, value, title, open, onToggle }: {
  readonly tag: string;
  readonly value: string;
  readonly title: string;
  readonly open: boolean;
  readonly onToggle: () => void;
}) {
  return <button
    type="button"
    className="fluid-field-summary"
    data-testid={`fluid-field-row-${tag.toLowerCase()}`}
    aria-expanded={open}
    title={title}
    onClick={onToggle}
  >
    <span>{tag}</span>
    <strong>{value}</strong>
  </button>;
}

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

export function FluidFieldFlyout({
  leftFraction,
  topFraction,
  lenses: override,
}: {
  leftFraction: number;
  topFraction: number;
  /** Overrides the running method's own roster. For tests and previews. */
  lenses?: readonly AnyStageLens[];
}) {
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const lensPhase = useUIStore((state) => state.gridOverlayLensPhase);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const setLensPhase = useUIStore((state) => state.setGridOverlayLensPhase);
  const lensReceipt = useDiagnosticsStore((state) => state.stageLensReceipt);
  // Which pane the side column holds. Local, not stored: it is the state of a
  // disclosure on one panel, and a column that reopened itself because the
  // camera moved would be the panel remembering the wrong thing.
  const [detail, setDetail] = useState<FieldDetail | undefined>(undefined);
  // Only the views this method registered: a picker offering a publication the
  // solver never produces is a button that draws nothing.
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const methodValues = resolvedMethodValues(methodState);
  const importantOptions = method.params.filter((spec): spec is SelectParamSpec =>
    spec.tier === "coarse" && spec.kind === "select");
  // The rest of what the Method panel held. Coarse dials sit under the selects
  // they qualify; the fine tier stays behind a disclosure, as it always was.
  const coarseDials = method.params.filter((spec) => spec.tier === "coarse" && spec.kind !== "select");
  const fineParams = method.params.filter((spec) => spec.tier === "fine");
  const quality = methodState.quality;
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
  const filmReserved = methodValues.pressureJournal === "on";
  const filmSchedule = filmMode && filmReserved
    ? sparseCM12PressureJournalSchedule(
      sparseCM12PressureIterations(methodValues.pressureIterations),
      SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS)
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
  const toggle = (next: FieldDetail) =>
    setDetail((current) => (sameDetail(current, next) ? undefined : next));

  // Narrowed to what selecting actually reads, so a lens goes down the same
  // path as a catalog row rather than through a second one that would have to
  // be kept agreeing with it about what "picking the active view" means.
  const selectView = (view: Pick<FieldView, "mode" | "axis" | "planeless">) => {
    if (overlayMode === view.mode && overlayAxis !== "off") {
      setOverlayAxis("off");
      return;
    }
    setOverlayMode(view.mode);
    // A view changes the publication, not the user's chosen presentation.
    // Only an inactive overlay needs the view's authored default.
    if (overlayAxis === "off") {
      // A planeless view has no slice to fall back to: it draws its own geometry
      // over the frame, so the raymarch's volume capability does not apply.
      const axis = view.planeless || !(view.axis === "volume" && !volumeCapable)
        ? view.axis : DEFAULT_GRID_OVERLAY_AXIS;
      setOverlayAxis(axis);
      if (axis === "volume") setOverlaySlice(0.42);
    }
  };

  // Anchored against the shell's measured box, so orbiting the camera
  // until this corner nears an edge slides the panel instead of clipping it.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({ leftFraction, topFraction });

  // Built here rather than in the column's own component so the pane and the
  // rail row that opens it stay in one place per subject: adding a pane is a
  // row above and a case below, and nothing in between has to learn about it.
  let paneTitle = "";
  let pane: ReactNode = null;
  if (open?.kind === "views") {
    paneTitle = "Field";
    pane = <div className="fluid-field-options">
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
            // Picking is the list's whole job, so it stands down afterwards and
            // hands the column to the chosen view's own control surface — which
            // for a film is the curve, and for everything else is nothing.
            setDetail(isPressureJournalOverlayMode(view.mode) ? { kind: "film" } : undefined);
          }}
        >
          <i style={view.swatch ? { background: view.swatch } : undefined} />
          <span>{view.label}</span>
          <small>{view.figure ?? ""}</small>
        </button>;
      })}
    </div>;
  } else if (open?.kind === "lens" && activeLens) {
    paneTitle = activeLens.label;
    pane = <LensPhaseScrubber
      lens={activeLens}
      phase={lensPhase}
      receipt={lensReceipt?.lensId === activeLens.id ? lensReceipt : undefined}
      onPhase={setLensPhase}
    />;
  } else if (open?.kind === "film") {
    paneTitle = "Film";
    // The curve belongs beside the scrub, not under it: a frame of the film
    // means something different at the third iteration than at the thirtieth,
    // and knowing which needs the plot and the slider both on screen at once —
    // which a column beside the rail gives and a stacked panel did not.
    // Selecting a stop drives the same slider.
    pane = filmReserved
      ? <PressureFilmStrip
        slot={filmSlot}
        onSelectSlot={(slot) => setOverlaySlice(filmSchedule.length > 1
          ? slot / (filmSchedule.length - 1) : 0)}
      />
      : <p className="fluid-field-film" data-testid="fluid-field-film-off">
        <span>No film reserved — this view has nothing to replay.</span>
        <button
          type="button"
          data-testid="fluid-field-film-reserve"
          title="Reserve room to capture one pressure solve. Rebuilds the solver once."
          onClick={() => simulation.setMethodParam(methodId, "pressureJournal", "on")}
        >RESERVE</button>
      </p>;
  } else if (open?.kind === "setup") {
    paneTitle = "Setup";
    pane = <div
      className="fluid-field-settings"
      role="group"
      aria-label="Important fluid solver options"
    >
      {/* Quality is the method's own coarsest dial — how much grid it is given
          — so it leads the options it scales. */}
      {method.showQualityControl !== false && <label className="select-control" title={method.pressureMapping}>
        <span>Quality</span>
        <select
          aria-label="Simulation quality"
          value={quality}
          onChange={(event) => simulation.setQuality(event.currentTarget.value as GPUQuality)}
        >
          {(["balanced", "high", "ultra"] as const).map((level) => (
            <option key={level} value={level}>{level[0]!.toUpperCase() + level.slice(1)} · {method.qualityLabels[level]}</option>
          ))}
        </select>
      </label>}
      {importantOptions.map((spec) => {
        return <label className="select-control" key={spec.key} title={paramTitle(spec)}>
          <span>{spec.label}</span>
          <select
            value={String(methodValues[spec.key])}
            onChange={(event) => simulation.setMethodParam(
              methodId, spec.key, event.currentTarget.value)}
          >
            {spec.options.map((option) => <option key={option.value} value={option.value}>
              {option.label}
            </option>)}
          </select>
        </label>;
      })}
      {coarseDials.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      {fineParams.length > 0 && <details className="advanced-params">
        <summary>Advanced</summary>
        {fineParams.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      </details>}
    </div>;
  }

  return <div
    ref={ref}
    className="fluid-field-flyout"
    data-testid="fluid-field-flyout"
    style={style}
  >
    <div className="fluid-field-rail">
      {hasViews && <section className="fluid-field-group">
        {/* A dozen radio rows is half the panel spent on a choice made once a
            session, and the one fact worth keeping on screen — which field is
            drawing — is a single line. So the list moves to the side column,
            and the summary is the answer; the plane and the scrub stay out
            here, because those are the controls a reader works while the water
            moves. */}
        <DetailRow
          tag="FIELD"
          value={shown?.label ?? "Hidden"}
          title="Choose the field this overlay draws"
          open={open?.kind === "views"}
          onToggle={() => toggle({ kind: "views" })}
        />
        {shown && <>
          <div className="fluid-field-plane" role="group" aria-label="Field view plane">
            {!shown.planeless && <>
              {(["x", "y", "z"] as const).map((axis) => <button
                key={axis}
                type="button"
                className={overlayAxis === axis ? "active" : ""}
                onClick={() => setOverlayAxis(axis)}
              >{axis.toUpperCase()}</button>)}
              <button
                type="button"
                className={overlayAxis === "volume" ? "active" : ""}
                disabled={!volumeCapable}
                title={volumeCapable ? undefined : "Volume views need an adaptive octree method"}
                onClick={() => setOverlayAxis("volume")}
              >VOL</button>
            </>}
            {/* Turning the overlay off belongs with the presentation controls,
                not on the summary line: a button inside a disclosure's own
                header is a click that has to argue with the fold. */}
            <button
              type="button"
              className="fluid-field-off"
              data-testid="fluid-field-hide"
              title="Hide the field overlay"
              onClick={() => setOverlayAxis("off")}
            >HIDE</button>
          </div>
          <label className="fluid-field-slice">
            <input
              type="range"
              min={filmMode ? 0 : shown.planeless || overlayAxis === "volume" ? 0.05 : 0}
              max={1}
              // A film's stops are the captured iterations and nothing between
              // them, so the scrub steps between snapshots rather than sliding
              // through a continuum it cannot show.
              step={filmSchedule.length > 1 ? 1 / (filmSchedule.length - 1)
                : shown.planeless || overlayAxis === "volume" ? 0.01 : 0.005}
              value={overlaySlice}
              onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
              aria-label={filmMode
                ? `${shown.label} iteration`
                : shown.planeless
                  ? `${shown.label} opacity`
                  : overlayAxis === "volume"
                    ? "Field volume opacity"
                    : `Field ${overlayAxis} slice position`}
            />
            <output>{filmSchedule.length > 0
              ? `iter ${filmSchedule[filmSlot]}`
              : filmMode ? "—" : `${Math.round(overlaySlice * 100)}%`}</output>
          </label>
          {/* A film's curve is the one pane that is not reachable from a row of
              its own kind — the view that produces it is chosen in the field
              list, which then stands down. Without this the column could be
              closed and never reopened, and the scrub above would be stepping
              through stops with nothing plotting them. */}
          {filmMode && <DetailRow
            tag="FILM"
            value={filmReserved ? `${filmSchedule.length} stops` : "none reserved"}
            title="The captured solve this scrub replays"
            open={open?.kind === "film"}
            onToggle={() => toggle({ kind: "film" })}
          />}
        </>}
      </section>}
      {/* What the *stages* did, as opposed to what the state is. Below the
          fields rather than folded in with them because the two answer
          different questions: a field row picks a publication, a lens row opens
          one pass's own reading of itself, and the scrubber only means anything
          inside one. */}
      {lenses.length > 0 && <section className="fluid-field-group">
        <h4 className="fluid-field-heading">Stages</h4>
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
                // Opening a lens opens its scrubber, because the phases are the
                // lens: a selected lens with no scrubber on screen is one
                // reading of a stage with no way to walk to the other six.
                setDetail(isActive ? undefined : { kind: "lens", id: lens.id });
              }}
            >
              <i style={lens.legend[0] ? { background: lens.legend[0].swatch } : undefined} />
              <span>{lens.label}</span>
            </button>;
          })}
        </div>
      </section>}
      {/* The solver behind these fields. It lives on the same widget because
          switching methods is part of the same watch-the-water loop as choosing
          a view. One chip per method under one heading — the label used to sit
          in the first cell of the same grid the chips wrapped through, which
          read as a select above a segment strip rather than as one chooser. */}
      <section className="fluid-field-group">
        <h4 className="fluid-field-heading">Solver</h4>
        <div className="fluid-field-solver" role="group" aria-label="Fluid solver method">
          {interactiveSimulationMethods().map((candidate) => <button
            key={candidate.id}
            type="button"
            className={candidate.id === methodId ? "active" : ""}
            aria-pressed={candidate.id === methodId}
            title={candidate.description}
            onClick={() => { if (candidate.id !== methodId) simulation.setMethod(candidate.id); }}
          >{candidate.shortLabel}</button>)}
        </div>
      </section>
      {/* The construction choices that follow from the method. Seven selects and
          a dial ladder was the tallest thing on the panel and the least often
          touched — changing one of these rebuilds the solver — so it is a row
          reporting the coarsest of them and a pane holding the rest. */}
      <section className="fluid-field-group">
        <DetailRow
          tag="SETUP"
          value={method.showQualityControl !== false
            ? `${quality[0]!.toUpperCase()}${quality.slice(1)} · ${method.qualityLabels[quality]}`
            : `${importantOptions.length + coarseDials.length} options`}
          title="Construction choices for this solver"
          open={open?.kind === "setup"}
          onToggle={() => toggle({ kind: "setup" })}
        />
      </section>
    </div>
    {/* Gated on the pane rather than on the request for one, so an unresolvable
        detail is a closed column and never an empty one. */}
    {pane && <aside className="fluid-field-aside" data-testid="fluid-field-aside" aria-label={paneTitle}>
      <div className="fluid-field-aside-head">
        <h4>{paneTitle}</h4>
        <button
          type="button"
          title="Close"
          aria-label="Close"
          data-testid="fluid-field-aside-close"
          onClick={() => setDetail(undefined)}
        >×</button>
      </div>
      {pane}
    </aside>}
  </div>;
}
