"use client";

import { useState } from "react";
import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import type { MethodParamSpec, SelectParamSpec } from "@/lib/core/method-contract";
import type { GPUQuality } from "../lib/core/gpu-quality";
import { RangeControl } from "./controls";
import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import type { FieldVisualization } from "../lib/core/visualization-registry";
import { simulation } from "../lib/core/simulation/controller";
import { resolvedMethodValues, useMethodStore } from "../lib/core/stores/method-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { isPressureJournalOverlayMode } from "../lib/core/webgpu-pressure-journal-overlay";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
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
 * Three sections, each introduced by one small heading: the FIELD being drawn,
 * the SOLVER drawing it, and that solver's SETUP. The scene is the hero, so the
 * resting height is the price of admission — the dozen field views and the fine
 * tier are both one disclosure away, and what stays on screen is the handful of
 * facts worth reading while the water moves.
 */
type FieldView = FieldVisualization & { mode: GridOverlayMode };

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

export function FluidFieldFlyout({
  leftFraction,
  topFraction,
}: {
  leftFraction: number;
  topFraction: number;
}) {
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  // Whether the view list is unfolded. Local, not stored: it is the state of a
  // disclosure on one panel, and a picker that reopened itself because the
  // camera moved would be the panel remembering the wrong thing.
  const [pickerOpen, setPickerOpen] = useState(false);
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
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = FIELD_VIEWS.filter((view) => supported.has(view.mode));
  const active = overlayAxis !== "off"
    ? views.find((view) => view.mode === overlayMode)
    : undefined;
  // The solver row stays even when a method registers no field views: this
  // widget is also where the method is switched, and a picker that vanished
  // with the fields would strand such a method with no way back.
  const hasViews = views.length > 0;

  // A film view's slider is a scrub through the captured iterations, not an
  // opacity. The iteration each stop lands on is knowable here without asking
  // the solver anything: the snapshot schedule is a pure function of the
  // iteration ceiling and the reserved capacity, which is the same property
  // that lets the device pick its slot without the host telling it.
  const filmMode = active && isPressureJournalOverlayMode(active.mode);
  const filmReserved = methodValues.pressureJournal === "on";
  const filmSchedule = filmMode && filmReserved
    ? sparseCM12PressureJournalSchedule(
      sparseCM12PressureIterations(methodValues.pressureIterations),
      SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS)
    : [];
  const filmSlot = filmSchedule.length > 0
    ? Math.round(Math.max(0, Math.min(1, overlaySlice)) * (filmSchedule.length - 1))
    : 0;

  const selectView = (view: FieldView) => {
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
        ? view.axis : "y";
      setOverlayAxis(axis);
      if (axis === "volume") setOverlaySlice(0.42);
    }
  };

  // Anchored against the shell's measured box, so orbiting the camera
  // until this corner nears an edge slides the panel instead of clipping it.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({ leftFraction, topFraction });

  return <div
    ref={ref}
    className="fluid-field-flyout"
    data-testid="fluid-field-flyout"
    style={style}
  >
    {hasViews && <section className="fluid-field-group">
      {/* A dozen radio rows is half the panel spent on a choice made once a
          session, and the one fact worth keeping on screen — which field is
          drawing — is a single line. So the list folds, and the summary is the
          answer; the plane and the scrub stay out here, because those are the
          controls a reader works while the water moves. */}
      <details
        className="fluid-field-views"
        open={pickerOpen}
        onToggle={(event) => setPickerOpen(event.currentTarget.open)}
      >
        <summary title="Choose the field this overlay draws">
          <span>FIELD</span>
          <strong>{active?.label ?? "Hidden"}</strong>
        </summary>
        <div className="fluid-field-options">
          {views.map((view) => {
            const isActive = active?.mode === view.mode;
            return <button
              key={view.id}
              type="button"
              className={isActive ? "active" : ""}
              aria-pressed={isActive}
              title={view.description}
              onClick={() => selectView(view)}
            >
              <i style={view.swatch ? { background: view.swatch } : undefined} />
              <span>{view.label}</span>
              <small>{view.figure ?? ""}</small>
            </button>;
          })}
        </div>
      </details>
      {active && <>
        <div className="fluid-field-plane" role="group" aria-label="Field view plane">
          {!active.planeless && <>
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
          {/* Turning the overlay off belongs with the presentation controls, not
              on the summary line: a button inside a disclosure's own header is a
              click that has to argue with the fold. */}
          <button
            type="button"
            className="fluid-field-off"
            data-testid="fluid-field-hide"
            title="Hide the field overlay"
            onClick={() => setOverlayAxis("off")}
          >HIDE</button>
        </div>
        {filmMode && !filmReserved && <p className="fluid-field-film" data-testid="fluid-field-film-off">
          <span>No film reserved — this view has nothing to replay.</span>
          <button
            type="button"
            data-testid="fluid-field-film-reserve"
            title="Reserve room to capture one pressure solve. Rebuilds the solver once."
            onClick={() => simulation.setMethodParam(methodId, "pressureJournal", "on")}
          >RESERVE</button>
        </p>}
        <label className="fluid-field-slice">
          <input
            type="range"
            min={filmMode ? 0 : active.planeless || overlayAxis === "volume" ? 0.05 : 0}
            max={1}
            // A film's stops are the captured iterations and nothing between
            // them, so the scrub steps between snapshots rather than sliding
            // through a continuum it cannot show.
            step={filmSchedule.length > 1 ? 1 / (filmSchedule.length - 1)
              : active.planeless || overlayAxis === "volume" ? 0.01 : 0.005}
            value={overlaySlice}
            onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
            aria-label={filmMode
              ? `${active.label} iteration`
              : active.planeless
                ? `${active.label} opacity`
                : overlayAxis === "volume"
                  ? "Field volume opacity"
                  : `Field ${overlayAxis} slice position`}
          />
          <output>{filmSchedule.length > 0
            ? `iter ${filmSchedule[filmSlot]}`
            : filmMode ? "—" : `${Math.round(overlaySlice * 100)}%`}</output>
        </label>
        {/* The curve belongs beside the scrub, not in a panel of its own: a frame
            of the film means something different at the third iteration than at
            the thirtieth, and knowing which needs the plot and the position
            together. Selecting a stop drives the same slider. */}
        {filmMode && filmReserved && <PressureFilmStrip
          slot={filmSlot}
          onSelectSlot={(slot) => setOverlaySlice(filmSchedule.length > 1
            ? slot / (filmSchedule.length - 1) : 0)}
        />}
      </>}
    </section>}
    {/* The solver behind these fields. It lives on the same widget because
        switching methods is part of the same watch-the-water loop as choosing
        a view. One chip per method under one heading — the label used to sit in
        the first cell of the same grid the chips wrapped through, which read as
        a select above a segment strip rather than as one chooser. */}
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
    {/* The construction choices that follow from the method, under their own
        heading so seven selects read as one group rather than as a wall. */}
    <section className="fluid-field-group">
      <h4 className="fluid-field-heading">Setup</h4>
      <div
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
      </div>
    </section>
  </div>;
}
