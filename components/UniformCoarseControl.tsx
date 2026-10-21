"use client";

import { useState } from "react";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { simulation } from "../lib/core/simulation/controller";
import type { MethodParamValue } from "../lib/core/method-contract";
import { UNIFORM_DETAIL_CRITERION_PARAMS, UNIFORM_DETAIL_RANGES, uniformDetailSettings, uniformDetailValues } from "../lib/methods/uniform/uniform-detail-policy";
import { UNIFORM_DETAIL_CRITERIA, type UniformDetailCriterion } from "../lib/methods/uniform/uniform-stage-grids";
import { IMPORTANCE_LEGEND, legacyVisualLayers, toggleVisualLayer } from "../lib/core/visual-layers";
import { uniformDetailRequestKey } from "../lib/methods/uniform/uniform-detail-requests";
import { NARROW_BAND_ACTIVITY_CONTROLS, NARROW_BAND_PARTICLE_CAP_CONTROL as CAP, narrowBandControlValues } from "../lib/methods/uniform/uniform-narrow-band-controls";
import { Grid3X3 } from "lucide-react";
import { ToolstripMenuButton, ToolstripMenuItem, ToolstripMenuRule, ToolstripRow, useToolstripSection } from "./toolstrip";
import { Facts, Field, FieldList, FieldNote, NumberInput, RangeField, ToggleButton } from "./ui";

const SHAPE_DISPLACEMENT_HINT = "Shape measures how far the surface moves at 4h, not the error in phi beside it. Off: value error.";
const SURFACE_DISTANCE_HINT = "Allowed tile layers around surface-crossing tiles, including diagonals. 0 keeps only crossing tiles. This permits existing refinement; it does not request more.";
const SURFACE_ONLY_HINT = "Only tiles within the Surface distance of a surface-crossing tile are eligible for automatic detail. Filters the existing selection, including margin and hold; never adds fine tiles. Explicit requests and required solid contact remain.";

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

/**
 * The detail controls on the scene toolstrip, under the solver row: the policy
 * on one row and, for Dynamic, a Tune dropdown beside it holding the criteria.
 * Each criterion is a check item in its layer colour with its threshold beside
 * it while it is on; the shaping dials follow under a rule. A menu because the
 * rows stood three wide over the water. Changes tune the running simulation;
 * the panel below reports the accepted layout.
 */
export function UniformDetailRow() {
  const session = useSession();
  const method = session.method();
  const fine = session.diagnostics(state => state.gpuInfo?.uniformMixedFineTiles);
  const coarse = session.diagnostics(state => state.gpuInfo?.uniformMixedCoarseTiles);
  const layers = session.ui(state => state.visualLayers);
  const overlayMode = session.ui(state => state.gridOverlayMode);
  const overlayAxis = session.ui(state => state.gridOverlayAxis);
  const [tuning, setTuning] = useState(false);
  const { claim } = useToolstripSection("uniform-detail", () => setTuning(false));
  const resolved = resolvedMethodValues(method);
  const narrowBand = true;
  const adaptive = narrowBand && resolved.adaptiveSurface === "on";
  const activityValues = narrowBandControlValues(resolved);
  const settings = uniformDetailSettings(resolved), values = uniformDetailValues(settings);
  const set = (key: string, value: MethodParamValue) => simulation.setMethodParam(method.methodId, key, value, session.id);
  const dynamic = settings.policy === "dynamic";
  // The scores layer rides the field view's layer state (FieldQuickBar).
  const drawn = layers ?? { ...legacyVisualLayers(overlayMode), visible: overlayAxis !== "off" };
  const scores = drawn.visible && drawn.enabled.includes("importance");
  const showScores = (on: boolean) => session.ui.setState({
    visualLayers: on === drawn.enabled.includes("importance") ? { ...drawn, visible: true } : toggleVisualLayer(drawn, "importance"),
    gridOverlayMode: "structure",
    gridOverlayAxis: overlayAxis === "off" || overlayAxis === "volume" ? "z" : overlayAxis,
  });
  const dial = (key: keyof typeof UNIFORM_DETAIL_RANGES, unit: string, label: string, hint: string) => {
    const [min, max, step, digits] = UNIFORM_DETAIL_RANGES[key];
    return <NumberInput unit={unit || undefined} value={thresholdOf(values, key)} min={min} max={max} step={step} digits={digits}
      ariaLabel={label} hint={hint} onChange={value => set(key, value)} />;
  };
  const on = UNIFORM_DETAIL_CRITERIA.filter(id => settings.criteria[id]).length;
  const menu = <ToolstripMenuButton
    label="Detail criteria"
    caption="Tune"
    hint={`What makes a tile run h: ${on} of ${UNIFORM_DETAIL_CRITERIA.length} on. Each scores a tile as its measure over its threshold; 1 asks for h.`}
    open={dynamic && tuning}
    testId="scene-detail-tune"
    onOpen={value => { claim(value); setTuning(value); }}
  >
    {adaptive && <>
      {NARROW_BAND_ACTIVITY_CONTROLS.map(c => <RangeField key={c.key} label={c.label} value={activityValues[c.key]}
        min={c.min} max={c.max} step={c.step} digits={c.digits} unit={c.unit} hint={c.hint}
        onInput={value => set(c.key, value)} onChange={value => set(c.key, value)} />)}
    </>}
    <RangeField label={CAP.label} value={activityValues[CAP.key]} min={CAP.min} max={CAP.max} step={CAP.step} digits={CAP.digits}
      hint={CAP.hint} onInput={value => set(CAP.key, value)} onChange={value => set(CAP.key, value)} />
    <ToolstripMenuRule />
    <ToolstripMenuItem multiple label="Surface" title={SURFACE_ONLY_HINT} active={settings.surfaceOnly}
      testId="scene-detail-surface" onClick={() => set("detailSurface", settings.surfaceOnly ? "off" : "on")} />
    {settings.surfaceOnly && <RangeField label="Surface distance" value={settings.surfaceDistance} min={0} max={3} step={1} digits={0} unit="tiles"
      hint={SURFACE_DISTANCE_HINT} onChange={value => set("detailSurfaceDistance", value)} />}
    <ToolstripMenuRule />
    {(["surface", "flow"] as const).flatMap(group => UNIFORM_DETAIL_CRITERIA.filter(id => CRITERIA[id].group === group)).map(id => {
      const c = CRITERIA[id], keys = UNIFORM_DETAIL_CRITERION_PARAMS[id], enabled = settings.criteria[id];
      return <div key={id} role="none" className="toolstrip-menu-option">
        <ToolstripMenuItem multiple label={c.label} title={`${c.name}: ${c.hint}`} active={enabled}
          swatch={IMPORTANCE_LEGEND[UNIFORM_DETAIL_CRITERIA.indexOf(id)]!.color} testId={`scene-detail-${id}`}
          onClick={() => set(keys.toggle, enabled ? "off" : "on")} />
        {enabled && dial(keys.threshold, c.unit, `${c.name} threshold (${c.unit})`, c.thresholdHint)}
      </div>;
    })}
    {settings.criteria.shape && <ToolstripMenuItem multiple label="Shape by displacement" note={settings.shapeMetric === "displacement" ? "✓" : undefined}
      title={SHAPE_DISPLACEMENT_HINT} active={settings.shapeMetric === "displacement"} testId="scene-detail-shape-metric"
      onClick={() => set("detailShapeMetric", settings.shapeMetric === "displacement" ? "value" : "displacement")} />}
    {(settings.criteria.strain || settings.criteria.rotation) && <ToolstripMenuItem multiple label="Bulk liquid" note={settings.bulk ? "✓" : undefined}
      title="Strain and Spin also judge liquid tiles with no surface in them. Off: surface tiles only."
      active={settings.bulk} testId="scene-detail-bulk" onClick={() => set("detailBulk", settings.bulk ? "off" : "on")} />}
    <ToolstripMenuRule />
    {SHAPING.filter(c => !adaptive || c.key !== "detailBudgetPercent").map(control => <div key={control.key} role="none" className="toolstrip-menu-option" title={control.hint}>
      <span>{control.label}</span>
      {dial(control.key, control.unit, `Detail ${control.label.toLowerCase()}`, control.hint)}
    </div>)}
  </ToolstripMenuButton>;
  return <ToolstripRow
    icon={<Grid3X3 width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Simulation detail"
    hint="Where the solver runs h cells on its 4h base. Changes apply at the next frame and keep the running simulation."
    testId="scene-detail-row"
    after={<>
      <span className="toolstrip-gutter" aria-hidden />
      <span className="toolstrip-name">Dynamic</span>
      {dynamic && menu}
      {dynamic && <ToggleButton pressed={scores} ariaLabel="Show detail importance scores" testId="scene-detail-scores"
        hint="Draw every criterion's score per tile on the slice (the Detail importance layer): 1 is where it starts asking for h. Criteria that are off are scored too."
        onChange={showScores}>Scores</ToggleButton>}
      {fine !== undefined && <span className="toolstrip-name" data-testid="scene-detail-coverage"
        title="Tiles of 4×4×4 h cells running h in the accepted layout.">{percent(fine, fine + (coarse ?? 0))} h</span>}
    </>}
  />;
}

/**
 * Simulation detail: where the Uniform Geometric solver runs h on its 4h base.
 * Keep this panel focused on accepted coverage and status; tuning lives in
 * the scene toolstrip.
 */
export function UniformCoarseControl() {
  const session = useSession();
  const method = session.method();
  const scene = session.scene(state => state.scene);
  const info = session.diagnostics(state => state.gpuInfo);
  const values = resolvedMethodValues(method), settings = uniformDetailSettings(values);
  const narrowBand = true;
  const adaptive = narrowBand && values.adaptiveSurface === "on";
  const activityValues = narrowBandControlValues(values);
  const regions = scene.fluid.refinementRegions ?? [];
  const detail = info?.uniformDetail;
  const fine = info?.uniformMixedFineTiles, coarse = info?.uniformMixedCoarseTiles;
  const ready = detail !== undefined && fine !== undefined;
  // The accepted layout answers a policy and region set; anything else is still on its way.
  const pending = ready && detail.requestKey !== uniformDetailRequestKey(settings.policy, regions);
  const tiles = (fine ?? 0) + (coarse ?? 0);
  return <FieldList testId="uniform-resolution-control">
    <Field label="Coverage" className="is-readout"
      hint="Tiles of 4×4×4 h cells in the accepted layout. Enabled solid-contact refinement also requests h.">
      <output className="ui-value" data-testid="uniform-detail-coverage">
        {!ready ? "Initializing" : `${percent(fine!, tiles)} h · ${percent(coarse ?? 0, tiles)} 4h`}{detail?.preparing ? " · preparing" : pending ? detail.rejected ? " · refused" : " · applying" : ""}
      </output>
    </Field>
    {ready && <FieldNote>
      {`${fine!.toLocaleString()} h · ${(coarse ?? 0).toLocaleString()} 4h tiles · requested ${detail.requestedTiles.toLocaleString()}, admitted ${detail.admittedTiles.toLocaleString()}`
        + `${detail.contactTiles ? ` + ${detail.contactTiles.toLocaleString()} contact` : ""} · ${(info!.allocatedBytes / 1048576).toFixed(1)} MiB allocated`}
    </FieldNote>}
    {ready && detail.rejected && <FieldNote><span data-testid="uniform-detail-rejected">
      {`Request refused; the layout above keeps running. ${detail.rejected}.`}</span></FieldNote>}
    {ready && detail.preparing && <FieldNote><span data-testid="uniform-detail-preparing">
      {`Compiling ${detail.preparing} pipelines for the latest change; the simulation above keeps running until they are ready.`}</span></FieldNote>}
    {ready && <details className="instrument-drawer" data-testid="uniform-detail-diagnostics">
      <summary><span>Detail plan</span><small>Accepted layout</small></summary>
      <Facts items={[
        { label: "Mapping", value: detail.mapping },
        { label: "Requests", value: `${detail.reasons.region.toLocaleString()} region · ${detail.reasons.full.toLocaleString()} full · ${detail.reasons.focus.toLocaleString()} focus · ${detail.reasons.activity.toLocaleString()} activity tiles` },
        { label: "Support overhead", value: `${detail.supportTiles.toLocaleString()} tiles (${percent(detail.supportTiles, detail.admittedTiles + detail.supportTiles)})`,
          hint: "Tiles around admitted h tiles that a patch store must also hold." },
        ...(adaptive ? [{ label: "Adaptive budget", value: `${activityValues.adaptiveBudgetPercent}% of requesting tiles`, hint: "Automatic candidates only; cooling support, sources and required contact may add tiles." }]
          : [{ label: "Domain budget", value: `${detail.automaticCostTiles.toLocaleString()} / ${detail.budgetTiles.toLocaleString()} tiles${detail.budgetClippedTiles ? ` · ${detail.budgetClippedTiles.toLocaleString()} clipped` : ""}` }]),
        { label: "Patches", value: `${detail.residentPatches.toLocaleString()} × ${detail.patchCells}³ · ${percent(detail.wastedCells, detail.allocatedCells)} unused`,
          hint: "What a 4h-first patch store would allocate for this plan; this build still holds the full h lattice." },
        ...(detail.deferredTiles ? [{ label: "Deferred", value: `${detail.deferredTiles.toLocaleString()} tiles over capacity` }] : []),
        ...(detail.unconstrainedRegions ? [{ label: "Unconstrained", value: `${detail.unconstrainedRegions} region${detail.unconstrainedRegions === 1 ? "" : "s"} allow both sizes and request nothing` }] : []),
      ]} />
    </details>}
  </FieldList>;
}
