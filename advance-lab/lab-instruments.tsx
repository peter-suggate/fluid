"use client";

import type { FeatureControlViewProps } from "../lib/framework/ui/slot";
import {
  ToolstripChoice,
  ToolstripNumber,
  ToolstripRow,
} from "../components/toolstrip";
import {
  FieldOverlayRows, FieldViewRows,
} from "../lib/features/field-view/ui";
import { useEditorHost } from "../lib/core/session/host-context";
import {
  ADVANCE_PRESSURE_BUDGET_RANGE, ADVANCE_SLICE_SETTINGS, ADVANCE_SURFACE_VIEWS,
  ADVANCE_TRANSPORT_EXPERIMENTS, ADVANCE_TRANSPORT_EXPERIMENT_ORDER,
  type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import type { LabRegionDocument } from "./lab-region-space";
import {
  ADVANCE_LENS_VIEWS, REPRESENT_LENS_MODE, SLICE_OVERLAY_VIEWS,
} from "./lenses";
import { SLICE_OVERLAY_KEYS } from "./use-slice-shortcuts";

/**
 * The lab's half of `advanceSliceFeature`: one component per declared control.
 *
 * Every one of these reads and writes through `EditorHost.params`, keyed by the
 * `setting` its declaration names. That is what makes them *views of a
 * control* rather than rows with a page's state wired into them: the same
 * component can be bound in `lib/features/ui/FeatureSlot.tsx`'s
 * `applicationViews` the day the studio wants one of these dials, without this
 * file learning anything about the studio.
 *
 * Two of the five are not written here at all. The lens picker and the two
 * annotations are `lib/features/field-view/ui.tsx` — the *same* row the studio
 * mounts at the tank's corner — handed this page's lens roster and its own
 * reading of what is on. The sixteen-entry `LensRow` and the `OverlayRow` that
 * used to stand in `SliceToolstrip.tsx` restated that row's chevron, its mark,
 * its swatch and its tip against declarations the registry already had a shape
 * for; there is nothing left of them to drift.
 */

/** The lab's host: its document is the region list, and it has no patch form. */
function useLabHost() {
  return useEditorHost<LabRegionDocument, LabRegionDocument>();
}

/**
 * A declared setting's current answer, through the host.
 *
 * `undefined` means this host cannot answer for the control — the bargain
 * `EditorHost`'s optional capability groups make one level up — and a view that
 * reads one renders nothing rather than a dial that quietly does nothing.
 */
function useParam(setting: string): {
  readonly value: number | string | boolean | undefined;
  readonly set: (next: number | string | boolean) => void;
} {
  const host = useLabHost();
  return {
    value: host.params?.get(setting),
    set: (next) => host.params?.set(setting, next),
  };
}

/* ---- what is drawn on the water ------------------------------------- */

/**
 * Which lens is over the water.
 *
 * A menu rather than a segmented strip for the obvious reason — there are
 * sixteen — and the swatch beside each is the band's own tone, the same
 * colouring the bottom strip's bars carry, so the two readings of one set
 * cannot drift.
 *
 * `dismissable: false` is the one thing this host declares differently from the
 * studio: a 3-D field overlay is drawn *over* the water and clicking its lit
 * mark puts it away, and a lens here *is* the picture — "off" would be a blank
 * canvas rather than a plainer one. `adjustable: false` says the same about the
 * plane and the scrub: a 2-D cut has one plane and no opacity.
 */
export function LabLensRow() {
  const lens = useParam(ADVANCE_SLICE_SETTINGS.lens);
  const mode = typeof lens.value === "string" ? lens.value : REPRESENT_LENS_MODE;
  return <FieldViewRows
    views={ADVANCE_LENS_VIEWS}
    volumeCapable={false}
    state={{
      mode,
      axis: "z",
      slice: 0,
      setMode: (next) => lens.set(next),
      setAxis: () => {},
      setSlice: () => {},
      dismissable: false,
      adjustable: false,
    }}
  />;
}

/** The annotations that are on, as a stable comma-separated list. */
function overlaySet(value: number | string | boolean | undefined): ReadonlySet<string> {
  return new Set(typeof value === "string" && value.length > 0 ? value.split(",") : []);
}

/**
 * The two annotations, as switches.
 *
 * Whether each can be drawn at all is derived rather than carried: the direct
 * level set publishes its own surface, so there is no reconstructed normal to
 * annotate with — and "the surface is imposed" is exactly "the chosen surface
 * view declares itself unselectable". One fact, read where it is stated, rather
 * than a second availability flag travelling beside the value.
 */
export function LabOverlayRows() {
  const overlays = useParam(ADVANCE_SLICE_SETTINGS.overlays);
  const surface = useParam(ADVANCE_SLICE_SETTINGS.surface);
  const on = overlaySet(overlays.value);
  const imposed = surfaceImposed(surface.value);
  return <FieldOverlayRows
    views={SLICE_OVERLAY_VIEWS}
    state={{
      enabled: (mode) => on.has(mode),
      offered: (mode) => !(imposed && mode === "normal"),
      unavailable: () => "This transport publishes its surface directly, so there is no reconstructed normal to draw.",
      shortcut: (mode) => SLICE_OVERLAY_KEYS[mode as keyof typeof SLICE_OVERLAY_KEYS],
      toggle: (mode) => {
        const next = new Set(on);
        if (!next.delete(mode)) next.add(mode);
        overlays.set([...next].join(","));
      },
    }}
  />;
}

/* ---- how the surface is reconstructed ------------------------------- */

/** True when the transport publishes its own surface rather than the reader choosing. */
function surfaceImposed(value: number | string | boolean | undefined): boolean {
  const shown = ADVANCE_SURFACE_VIEWS.find((view) => view.id === value);
  return shown !== undefined && !shown.selectable;
}

/**
 * Which reconstruction the picture draws — stated rather than offered when the
 * transport publishes its own.
 *
 * The roster and both hints are the control's declaration, so an arm added
 * beside the method appears here without this file learning its name.
 */
export function LabSurfaceRow({ control }: FeatureControlViewProps) {
  const surface = useParam(ADVANCE_SLICE_SETTINGS.surface);
  const imposed = surfaceImposed(surface.value);
  const options = (control.options ?? []).filter((option) =>
    ADVANCE_SURFACE_VIEWS.find((view) => view.id === option.value)?.selectable);
  const shown = control.options?.find((option) => option.value === surface.value);
  return <ToolstripRow
    tag="SURFACE"
    value={shown?.label ?? "—"}
    name="Reconstructed surface"
    hint={imposed
      ? "The advected phi zero set is the published surface under this transport, so there is nothing to reconstruct."
      : control.hint}
    testId="slice-surface-row"
    after={<>
      <span className="toolstrip-gutter" aria-hidden />
      <ToolstripChoice
        ariaLabel="Which surface the picture reconstructs"
        value={typeof surface.value === "string" ? surface.value : ""}
        options={options.map((option) => ({
          value: option.value,
          label: option.label,
          title: option.hint,
          disabled: imposed,
        }))}
        onChange={(value) => surface.set(value as AdvanceSurfaceViewId)}
      />
    </>}
  />;
}

/* ---- what one advance may spend ------------------------------------- */

/**
 * The pressure budget, as a number found by sliding it and watching.
 *
 * Its range is the declaration's — `min`, `max` and `step` on the control —
 * which is what keeps the clamp here and the bound a host enforces from being
 * two numbers.
 */
export function LabBudgetRow({ control }: FeatureControlViewProps) {
  const budget = useParam(ADVANCE_SLICE_SETTINGS.budget);
  const minimum = control.min ?? ADVANCE_PRESSURE_BUDGET_RANGE.minimum;
  const maximum = control.max ?? ADVANCE_PRESSURE_BUDGET_RANGE.maximum;
  const iterations = typeof budget.value === "number" ? budget.value : minimum;
  return <ToolstripRow
    tag="SOLVE"
    value={`${iterations} ${control.unit ?? ""}`.trim()}
    name="Pressure iterations"
    hint={control.hint}
    testId="slice-budget-row"
    after={<>
      <span className="toolstrip-gutter" aria-hidden />
      <div className="toolstrip-dimensions">
        <ToolstripNumber
          value={iterations}
          step={control.step ?? ADVANCE_PRESSURE_BUDGET_RANGE.step}
          min={minimum}
          max={maximum}
          ariaLabel="Pressure iterations one advance may spend"
          onCommit={(next) => budget.set(
            Math.max(minimum, Math.min(maximum, Math.round(next))))}
        />
      </div>
    </>}
  />;
}

/* ---- which transport the run is testing ----------------------------- */

/**
 * The transport arm, in the header rather than on the edit strip.
 *
 * Its placement says so — `sim.transport` is a slot the header renders — and
 * the reason is the control's own `update: "reset"`: choosing an arm starts a
 * new *run* from the scene, so it is not an instrument on the water in front of
 * you. The arms, their names and what each one does are declared beside the
 * method; this is that roster read out.
 *
 * A label and a control rather than a label *around* one, because the flex row
 * and the gap belong to the header that mounts the slot — which is also what
 * keeps this file free of any one route's CSS module, and so renderable in a
 * test.
 */
export function LabTransportRow({ control }: FeatureControlViewProps) {
  const transport = useParam(ADVANCE_SLICE_SETTINGS.transport);
  return <>
    <label htmlFor="advance-transport">{control.label}</label>
    <select id="advance-transport" data-testid="advance-transport"
      value={typeof transport.value === "string"
        ? transport.value : ADVANCE_TRANSPORT_EXPERIMENT_ORDER[0]}
      title={control.hint}
      onChange={(event) => transport.set(event.target.value)}>
      {(control.options ?? ADVANCE_TRANSPORT_EXPERIMENT_ORDER.map((id) => ({
        value: id, label: ADVANCE_TRANSPORT_EXPERIMENTS[id].label,
        hint: ADVANCE_TRANSPORT_EXPERIMENTS[id].hint,
      }))).map((option) =>
        <option key={option.value} value={option.value} title={option.hint}>
          {option.label}</option>)}
    </select>
  </>;
}

/** Representation selection is a reset of the run, independent of its lens. */
export function LabAdaptiveSdfRow({ control }: FeatureControlViewProps) {
  const adaptive = useParam(ADVANCE_SLICE_SETTINGS.adaptiveSdf);
  const transport = useParam(ADVANCE_SLICE_SETTINGS.transport);
  if (transport.value !== "level-set-volume") return null;
  const enabled = adaptive.value !== false;
  return <ToolstripRow
    tag={control.label}
    value={enabled ? "on" : "off"}
    name={control.label}
    hint={control.hint}
    active={enabled}
    testId="advance-adaptive-sdf"
    onClick={() => adaptive.set(!enabled)}
  />;
}
