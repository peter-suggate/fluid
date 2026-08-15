"use client";

import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import type { SelectParamSpec } from "@/lib/core/method-contract";
import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import type { FieldVisualization } from "../lib/core/visualization-registry";
import { simulation } from "../lib/core/simulation/controller";
import { resolvedMethodValues, useMethodStore } from "../lib/core/stores/method-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";
import { useAnchoredFlyout } from "./anchored-flyout";

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
 */
type FieldView = FieldVisualization & { mode: GridOverlayMode };

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
  // Only the views this method registered: a picker offering a publication the
  // solver never produces is a button that draws nothing.
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const methodValues = resolvedMethodValues(methodState);
  const importantOptions = method.params.filter((spec): spec is SelectParamSpec =>
    spec.tier === "coarse" && spec.kind === "select");
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = FIELD_VIEWS.filter((view) => supported.has(view.mode));
  const active = overlayAxis !== "off"
    ? views.find((view) => view.mode === overlayMode)
    : undefined;
  // The solver row stays even when a method registers no field views: this
  // widget is also where the method is switched, and a picker that vanished
  // with the fields would strand such a method with no way back.
  const hasViews = views.length > 0;

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
    {hasViews && <header>
      <span>FIELD</span>
      <strong>{active?.label ?? "Hidden"}</strong>
      {active && <button
        type="button"
        data-testid="fluid-field-hide"
        title="Hide the field overlay"
        onClick={() => setOverlayAxis("off")}
      >HIDE</button>}
    </header>}
    {active && <>
      {!active.planeless
      && <div className="fluid-field-plane" role="group" aria-label="Field view plane">
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
      </div>}
      <label className="fluid-field-slice">
        <input
          type="range"
          min={active.planeless || overlayAxis === "volume" ? 0.05 : 0}
          max={1}
          step={active.planeless || overlayAxis === "volume" ? 0.01 : 0.005}
          value={overlaySlice}
          onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
          aria-label={active.planeless
            ? `${active.label} opacity`
            : overlayAxis === "volume"
              ? "Field volume opacity"
              : `Field ${overlayAxis} slice position`}
        />
        <output>{Math.round(overlaySlice * 100)}%</output>
      </label>
    </>}
    {hasViews && <div className="fluid-field-options">
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
    </div>}
    {/* The solver behind these fields. It lives on the same widget because
        switching methods is part of the same watch-the-water loop as choosing
        a view. The high-value construction choices follow it so the common
        watch-adjust-repeat loop does not require opening the full method panel. */}
    <div className="fluid-field-solver" role="group" aria-label="Fluid solver method">
      <span>SOLVER</span>
      {interactiveSimulationMethods().map((candidate) => <button
        key={candidate.id}
        type="button"
        className={candidate.id === methodId ? "active" : ""}
        aria-pressed={candidate.id === methodId}
        title={candidate.description}
        onClick={() => { if (candidate.id !== methodId) simulation.setMethod(candidate.id); }}
      >{candidate.shortLabel}</button>)}
    </div>
    {importantOptions.length > 0 && <div
      className="fluid-field-settings"
      role="group"
      aria-label="Important fluid solver options"
    >
      {importantOptions.map((spec) => {
        return <label className="select-control" key={spec.key} title={spec.hint}>
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
    </div>}
  </div>;
}
