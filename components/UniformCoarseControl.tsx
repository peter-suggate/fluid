"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { simulation } from "../lib/core/simulation/controller";
import type { MethodParamValue, MethodParamValues } from "../lib/core/method-contract";
import { UNIFORM_DETAIL_CRITERION_PARAMS, UNIFORM_DETAIL_RANGES, uniformDetailSettings, uniformDetailValues } from "../lib/methods/uniform/uniform-detail-policy";
import { UNIFORM_DETAIL_CRITERIA, type UniformDetailCriterion } from "../lib/methods/uniform/uniform-stage-grids";
import { IMPORTANCE_LEGEND } from "../lib/core/visual-layers";
import { uniformDetailRequestKey } from "../lib/methods/uniform/uniform-detail-requests";
import { NARROW_BAND_ACTIVITY_CONTROLS, NARROW_BAND_PARTICLE_CAP_CONTROL as CAP, NARROW_BAND_SPARSE_CONTROL, NARROW_BAND_SPARSE_NEIGHBOURS_CONTROL, NARROW_BAND_SURFACE_ADVECTION as ADVECTION, NARROW_BAND_TUNE_CONTROLS, narrowBandControlValues, narrowBandSurfaceAdvection, type NarrowBandSurfaceAdvection } from "../lib/methods/uniform/uniform-narrow-band-controls";
import { NARROW_BAND_TUNE_DIALS, narrowBandTunePosition, narrowBandTuneValues, type NarrowBandTuneDial } from "../lib/methods/uniform/uniform-narrow-band-tuning";
import { RailChoice, RailLink, RailRange, RailReadout, RailRule } from "./rail";
import { ChoiceField, Facts, Field, FieldList, NumberInput, RangeField, Swatch, Switch, SwitchField } from "./ui";

const SHAPE_DISPLACEMENT_HINT = "Shape measures how far the surface moves at 4h, not the error in phi beside it. Off: value error.";
const SURFACE_DISTANCE_HINT = "Allowed tile layers around surface-crossing tiles, including diagonals. 0 keeps only crossing tiles. This permits existing refinement; it does not request more.";
const SURFACE_ONLY_HINT = "Only tiles within the Surface distance of a surface-crossing tile are eligible for automatic detail. Filters the existing selection, including margin and hold; never adds fine tiles. Explicit requests and required solid contact remain.";
const BULK_HINT = "Strain and Spin also judge liquid tiles with no surface in them. Off: surface tiles only.";
const BUDGET_HINT = "A hard cap on cost: the share of tiles requesting automatic fine detail that are admitted. Drives Adaptive budget, 0 to 100%; 100 admits all.";

const percent = (part: number, whole: number) => whole ? `${Math.round(100 * part / whole)}%` : "—";

/**
 * Dynamic's importance criteria as controls: a switch and the one threshold
 * it is measured against. A tile scores its measure over that threshold and
 * runs h from a score of 1; the Detail importance layer draws the scores.
 */
const CRITERIA: Readonly<Record<UniformDetailCriterion, { label: string; name: string; unit: string; group: "surface" | "flow"; hint: string; thresholdHint: string }>> = {
  shape: { label: "Shape", name: "Surface shape", unit: "h", group: "surface",
    hint: "h where 4h corners cannot carry the surface's shape: crests, creases, ripples.",
    thresholdHint: "Surface error a tile may take at 4h. 0 keeps every surface tile at h." },
  thin: { label: "Thin", name: "Thin features", unit: "h", group: "surface",
    hint: "h where the liquid, or the air gap, across the surface is thin: sheets, films, jets, necks.",
    thresholdHint: "Liquid or air thinner than this across the surface runs h." },
  impact: { label: "Impact", name: "Wall impact and lift", unit: "h/step", group: "surface",
    hint: "h where surface liquid runs into, or up, a closed wall or a solid.",
    thresholdHint: "Travel toward (or up) the wall at which the rule fires." },
  approach: { label: "Approach", name: "Approaching contact", unit: "steps", group: "surface",
    hint: "h before the surface meets a wall, a solid or another surface.",
    thresholdHint: "A surface that closes its gap within this many steps runs h." },
  strain: { label: "Strain", name: "Deformation", unit: "/step", group: "flow",
    hint: "h where the flow stretches or shears fast (dt·‖sym ∇u‖).",
    thresholdHint: "Deformation per step at which a tile runs h." },
  rotation: { label: "Spin", name: "Rotation", unit: "/step", group: "flow",
    hint: "h where the flow turns fast (dt·‖curl u‖): vortices, rolling fronts.",
    thresholdHint: "Rotation per step, in radians, at which a tile runs h." },
};
const SHAPING = [
  { key: "detailSensitivity", unit: "", label: "Sensitivity", hint: "One dial over every threshold: 0.5 leaves them as set, 1 asks for h four times sooner, 0 four times later." },
  { key: "detailMarginTiles", unit: "tiles", label: "Margin", hint: "Tiles of h kept around every tile a criterion requires, on top of the surface's travel over the step." },
  { key: "detailHoldSteps", unit: "steps", label: "Hold", hint: "Steps a tile stays h after its last trigger, and for as long as a score stays above 0.7. 0 follows the criteria step by step." },
  { key: "detailBudgetPercent", unit: "%", label: "Budget", hint: "Most of the domain the criteria may hold at h. Over it, the lowest scores are dropped first. 100 clips nothing." },
] as const;
const thresholdOf = (values: Record<string, unknown>, key: keyof typeof UNIFORM_DETAIL_RANGES) => Number(values[key]);
type SetParam = (key: string, value: MethodParamValue) => void;

/**
 * One of the Tune card's dials: a slider over several params at once
 * (uniform-narrow-band-tuning.ts). Moving it writes every param it drives and
 * its own position; it reads Custom once the full controls move one of them
 * off its curve, and dragging it from there writes them all again.
 */
function TuneDial({ dial, values, set }: { dial: NarrowBandTuneDial; values: MethodParamValues; set: SetParam }) {
  const spec = NARROW_BAND_TUNE_CONTROLS[NARROW_BAND_TUNE_DIALS.indexOf(dial)]!;
  const { position, custom } = narrowBandTunePosition(dial, values);
  const apply = (t: number) => {
    for (const [key, value] of Object.entries(narrowBandTuneValues(dial, t))) if (values[key] !== value) set(key, value);
    if (values[spec.key] !== t) set(spec.key, t);
  };
  return <RailRange label={spec.label} hint={spec.hint} testId={`scene-detail-tune-${dial}`}
    value={position} min={spec.min} max={spec.max} step={spec.step} digits={spec.digits}
    display={custom ? "Custom" : undefined} onInput={apply} onChange={apply} />;
}

/**
 * Every detail control on its own, for the Tune card's All controls: the
 * surface advection, the particle controls, the surface filter, each
 * criterion with its threshold, and the shaping dials. The card's dials write
 * these same params, so a change here moves a dial off its curve (Custom).
 * Changes apply live, as the card's do.
 */
function UniformDetailControls({ onClose }: { onClose: () => void }) {
  const session = useSession();
  const method = session.method();
  const resolved = resolvedMethodValues(method);
  const adaptive = resolved.adaptiveSurface === "on";
  const activityValues = narrowBandControlValues(resolved);
  const settings = uniformDetailSettings(resolved), values = uniformDetailValues(settings);
  const set: SetParam = (key, value) => simulation.setMethodParam(method.methodId, key, value, session.id);
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  });
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  const dial = (key: keyof typeof UNIFORM_DETAIL_RANGES, unit: string, label: string, hint: string, disabled = false) => {
    const [min, max, step, digits] = UNIFORM_DETAIL_RANGES[key];
    return <NumberInput unit={unit || undefined} value={thresholdOf(values, key)} min={min} max={max} step={step} digits={digits}
      ariaLabel={label} hint={hint} disabled={disabled} onChange={value => set(key, value)} />;
  };
  const range = (c: { key: string; label: string; unit: string; min: number; max: number; step: number; digits: number; hint: string }) =>
    <RangeField key={c.key} label={c.label} value={activityValues[c.key as keyof typeof activityValues]}
      min={c.min} max={c.max} step={c.step} digits={c.digits} unit={c.unit || undefined} hint={c.hint}
      onInput={value => set(c.key, value)} onChange={value => set(c.key, value)} />;
  return createPortal(<div className="detail-modal-backdrop" onMouseDown={event => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <section className="detail-modal" role="dialog" aria-modal="true" aria-labelledby="detail-modal-title" data-testid="scene-detail-all-controls-modal">
      <header>
        <div>
          <p className="eyebrow">SIMULATION DETAIL</p>
          <h2 id="detail-modal-title">All detail controls</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close detail controls" autoFocus>×</button>
      </header>
      <div className="detail-modal-body">
        <FieldList>
          <ChoiceField<NarrowBandSurfaceAdvection> label={ADVECTION.label} hint={ADVECTION.hint} testId="scene-detail-surface-advection"
            value={narrowBandSurfaceAdvection(resolved.surfaceAdvection)}
            options={ADVECTION.options.map(o => ({ value: o.value as NarrowBandSurfaceAdvection, label: o.label }))}
            onChange={value => set(ADVECTION.key, value)} />
        </FieldList>
        <h3>Particles</h3>
        <FieldList>
          {adaptive && NARROW_BAND_ACTIVITY_CONTROLS.map(range)}
          {[CAP, NARROW_BAND_SPARSE_CONTROL, NARROW_BAND_SPARSE_NEIGHBOURS_CONTROL].map(range)}
        </FieldList>
        <h3>Surface filter</h3>
        <FieldList>
          <SwitchField label="Surface" hint={SURFACE_ONLY_HINT} checked={settings.surfaceOnly} testId="scene-detail-surface"
            onChange={on => set("detailSurface", on ? "on" : "off")} />
          <RangeField label="Surface distance" value={settings.surfaceDistance} min={0} max={3} step={1} digits={0} unit="tiles"
            hint={SURFACE_DISTANCE_HINT} disabled={!settings.surfaceOnly} onChange={value => set("detailSurfaceDistance", value)} />
        </FieldList>
        <h3>Criteria</h3>
        <FieldList>
          {(["surface", "flow"] as const).flatMap(group => UNIFORM_DETAIL_CRITERIA.filter(id => CRITERIA[id].group === group)).map(id => {
            const c = CRITERIA[id], keys = UNIFORM_DETAIL_CRITERION_PARAMS[id], enabled = settings.criteria[id];
            return <Field key={id} label={<><Swatch color={IMPORTANCE_LEGEND[UNIFORM_DETAIL_CRITERIA.indexOf(id)]!.color} /> {c.label}</>}
              hint={`${c.name}: ${c.hint}`} testId={`scene-detail-${id}`}
              value={dial(keys.threshold, c.unit, `${c.name} threshold (${c.unit})`, c.thresholdHint, !enabled)}>
              <Switch checked={enabled} ariaLabel={c.name} onChange={on => set(keys.toggle, on ? "on" : "off")} />
            </Field>;
          })}
          <SwitchField label="Shape by displacement" hint={SHAPE_DISPLACEMENT_HINT} checked={settings.shapeMetric === "displacement"}
            disabled={!settings.criteria.shape} testId="scene-detail-shape-metric"
            onChange={on => set("detailShapeMetric", on ? "displacement" : "value")} />
          <SwitchField label="Bulk liquid" hint={BULK_HINT} checked={settings.bulk}
            disabled={!settings.criteria.strain && !settings.criteria.rotation} testId="scene-detail-bulk"
            onChange={on => set("detailBulk", on ? "on" : "off")} />
        </FieldList>
        <h3>Shaping</h3>
        <FieldList>
          {SHAPING.filter(c => !adaptive || c.key !== "detailBudgetPercent").map(control => <Field key={control.key} label={control.label} hint={control.hint}>
            {dial(control.key, control.unit, `Detail ${control.label.toLowerCase()}`, control.hint)}
          </Field>)}
        </FieldList>
      </div>
    </section>
  </div>, document.body);
}

/**
 * Dynamic detail's tuning, as the study rail's Tune card: the accepted
 * coverage, four dials, Surface accuracy, Splash particles, Persistence and
 * Budget, each a writer over the params it names, and All controls, which
 * opens every control on its own in a dialog. The method installs the card
 * only under the Dynamic policy (`uniformDetailTuningFeature`). Changes tune
 * the running simulation; the Detail importance layer draws what they ask for.
 */
export function UniformTunePanel() {
  const session = useSession();
  const method = session.method();
  const fine = session.diagnostics(state => state.gpuInfo?.uniformMixedFineTiles);
  const coarse = session.diagnostics(state => state.gpuInfo?.uniformMixedCoarseTiles);
  const [everything, setEverything] = useState(false);
  const resolved = resolvedMethodValues(method);
  const activityValues = narrowBandControlValues(resolved);
  const set: SetParam = (key, value) => simulation.setMethodParam(method.methodId, key, value, session.id);
  const budget = NARROW_BAND_ACTIVITY_CONTROLS[0];
  return <>
    <RailReadout label="Dynamic detail" value={fine === undefined ? "—" : `${percent(fine, fine + (coarse ?? 0))} h`}
      hint="Where the solver runs h cells on its 4h base: tiles of 4×4×4 h cells running h in the accepted layout."
      testId="scene-detail-coverage" />
    {NARROW_BAND_TUNE_DIALS.map(dial => <TuneDial key={dial} dial={dial} values={resolved} set={set} />)}
    <RailRange label="Budget" value={activityValues[budget.key]} min={budget.min} max={budget.max} step={budget.step} digits={budget.digits}
      unit={budget.unit} hint={BUDGET_HINT} testId="scene-detail-tune-budget"
      onInput={value => set(budget.key, value)} onChange={value => set(budget.key, value)} />
    <RailChoice label="Surface particles" value={resolved.surfaceSamples === "64" ? "64" : "8"}
      options={[{ value: "8", label: "8" }, { value: "64", label: "64" }]}
      hint="Particles a cell in the top cell and a half, where they define the surface. 64 gives a smoother, less biased surface (nothing finer than a cell) at more particle work; deeper particles are unchanged. Restarts the solver."
      testId="scene-detail-surface-samples" onChange={value => set("surfaceSamples", value)} />
    <RailRule />
    <RailLink label="All controls…" hint="Every detail control on its own, grouped as the solver reads them. Changes apply live."
      testId="scene-detail-all-controls" onClick={() => setEverything(true)} />
    {everything && <UniformDetailControls onClose={() => setEverything(false)} />}
  </>;
}

/** The accepted layout, and whether the latest request is still on its way. */
function useUniformDetail() {
  const session = useSession();
  const method = session.method();
  const scene = session.scene(state => state.scene);
  const info = session.diagnostics(state => state.gpuInfo);
  const values = resolvedMethodValues(method), settings = uniformDetailSettings(values);
  const detail = info?.uniformDetail;
  const fine = info?.uniformMixedFineTiles, coarse = info?.uniformMixedCoarseTiles;
  const ready = detail !== undefined && fine !== undefined;
  // The accepted layout answers a policy and region set; anything else is still on its way.
  const pending = ready && detail.requestKey !== uniformDetailRequestKey(settings.policy, scene.fluid.refinementRegions ?? []);
  return { values, info, detail, fine: fine ?? 0, coarse: coarse ?? 0, ready, pending };
}

/**
 * What the detail planner is doing that the reader has to see without opening
 * anything: a refused request, or pipelines compiling for the latest change.
 */
export function UniformDetailStatus() {
  const { detail, ready } = useUniformDetail();
  if (!ready) return null;
  return <>
    {detail!.rejected && <small className="rail-note" data-tone="attention" data-testid="uniform-detail-rejected"
      title={`The previous layout keeps running. ${detail!.rejected}.`}>Detail request refused</small>}
    {detail!.preparing && <small className="rail-note" data-testid="uniform-detail-preparing"
      title="The simulation keeps running until they are ready.">Compiling {detail!.preparing} detail pipelines</small>}
  </>;
}

/** Where the Uniform Geometric solver runs h on its 4h base, and the plan that put it there. */
export function UniformDetailLines() {
  const { values, info, detail, fine, coarse, ready, pending } = useUniformDetail();
  const adaptive = values.adaptiveSurface === "on";
  const activityValues = narrowBandControlValues(values);
  const tiles = fine + coarse;
  return <>
    <RailReadout label="Coverage" testId="uniform-detail-coverage"
      hint={ready ? `${fine.toLocaleString()} h · ${coarse.toLocaleString()} 4h tiles · requested ${detail!.requestedTiles.toLocaleString()}, admitted ${detail!.admittedTiles.toLocaleString()}${detail!.contactTiles ? ` + ${detail!.contactTiles.toLocaleString()} contact` : ""}` : undefined}
      value={!ready ? "—" : `${percent(fine, tiles)} h${detail!.preparing ? " · preparing" : pending ? detail!.rejected ? " · refused" : " · applying" : ""}`} />
    {ready && <details className="rail-more is-nested" data-testid="uniform-detail-diagnostics">
      <summary>Plan</summary>
      <Facts items={[
        { label: "Mapping", value: detail!.mapping },
        { label: "Requests", value: `${detail!.reasons.region.toLocaleString()} region · ${detail!.reasons.full.toLocaleString()} full · ${detail!.reasons.focus.toLocaleString()} focus · ${detail!.reasons.activity.toLocaleString()} activity` },
        { label: "Support", value: `${detail!.supportTiles.toLocaleString()} (${percent(detail!.supportTiles, detail!.admittedTiles + detail!.supportTiles)})`,
          hint: "Tiles around admitted h tiles that a patch store must also hold." },
        ...(adaptive ? [{ label: "Budget", value: `${activityValues.adaptiveBudgetPercent}% of requesting` }]
          : [{ label: "Budget", value: `${detail!.automaticCostTiles.toLocaleString()} / ${detail!.budgetTiles.toLocaleString()}${detail!.budgetClippedTiles ? ` · ${detail!.budgetClippedTiles.toLocaleString()} clipped` : ""}` }]),
        { label: "Patches", value: `${detail!.residentPatches.toLocaleString()} × ${detail!.patchCells}³ · ${percent(detail!.wastedCells, detail!.allocatedCells)} unused` },
        ...(detail!.deferredTiles ? [{ label: "Deferred", value: `${detail!.deferredTiles.toLocaleString()} over capacity` }] : []),
        ...(detail!.unconstrainedRegions ? [{ label: "Unconstrained", value: `${detail!.unconstrainedRegions} region${detail!.unconstrainedRegions === 1 ? "" : "s"}` }] : []),
        { label: "Allocated", value: `${((info?.allocatedBytes ?? 0) / 1048576).toFixed(1)} MiB` },
      ]} />
    </details>}
  </>;
}
