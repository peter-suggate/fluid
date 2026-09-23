"use client";

import { ChoiceField, FieldList, RangeField } from "../../../../components/ui";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { type SvoRenderTuning } from "../../pipeline/svo-render-tuning";
import { SvoFeatureSlot } from "../../pipeline/ui-slots";
import { SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT,svoRenderStageUsesLightSlot } from "../diagnostics/svo-render-diagnostics";

export function renderConeVisibilityControls({ svoConeTracingMode, tuning, updateTuning, modified, resetTuning, svoStageView, svoStageLightSlot, setSvoStageLightSlot, lightingVisibilityStatus }: Pick<SvoFeatureControlContext, "svoConeTracingMode" | "tuning" | "updateTuning" | "modified" | "resetTuning" | "svoStageView" | "svoStageLightSlot" | "setSvoStageLightSlot" | "lightingVisibilityStatus">): SvoStageControls {
  const exactBudgets = svoConeTracingMode === "exact";
  const lightSlot = svoRenderStageUsesLightSlot(svoStageView);
  return {
    // Source, shadows, AO, lattice and rate on the rail; ten calibrations, two
    // sample counts, and the arm- and plane-specific budgets in the drawer.
    settings: 5 + 12 + (exactBudgets ? 4 : 0) + (lightSlot ? 1 : 0),
    notice: (lightingVisibilityStatus.fallback || lightingVisibilityStatus.detail)
      && <p data-testid="lighting-visibility-status" aria-live="polite" className="render-inline-warning">
        Lighting visibility: {lightingVisibilityStatus.state.toUpperCase()}
        {lightingVisibilityStatus.fallback ? " FALLBACK" : ""}
        {lightingVisibilityStatus.detail ? ` · ${lightingVisibilityStatus.detail}` : ""}
      </p>,
    node: <>
      <FieldList aria-label="SVO lighting effects">
      <SvoFeatureSlot slot="frame.lighting" />
      <ChoiceField label="Cone prepass rate" value={String(tuning.coneLightingScale)} disabled={svoConeTracingMode !== "cones"}
        onChange={(value) => updateTuning("coneLightingScale", Number(value) as SvoRenderTuning["coneLightingScale"])}
        options={[{ value: "1", label: "FULL" }, { value: "0.5", label: "2×2" }, { value: "0.25", label: "4×4" }, { value: "0.125", label: "8×8" }]} />
      </FieldList>
      {/* What the frame *is* stays on the card above; what it is *calibrated
          to* folds. Twenty budgets is a session's worth of tuning, not
          something read at a glance beside the picture they change. */}
      <details className="rp-tune"><summary>Cone calibration</summary>
      <FieldList>
        <RangeField label="Shadow cone aperture" unit="rad" value={tuning.shadowConeAperture} min={0.01} max={0.25} step={0.005} digits={3}
          onChange={(value) => updateTuning("shadowConeAperture", value)}
          modified={modified("shadowConeAperture")} onReset={resetTuning("shadowConeAperture")}
          hint="Wider cones take larger march steps and produce softer shadows; narrower cones preserve sharper shadows but need more taps. The sharp/soft decision most scenes want is the CONES/EXACT switch above; this calibrates the soft path." />
        <RangeField label="Shadow strength" unit="%" value={tuning.shadowStrength * 100} min={0} max={100} step={1} digits={0}
          onChange={(value) => updateTuning("shadowStrength", value / 100)} modified={modified("shadowStrength")} onReset={resetTuning("shadowStrength")} />
        <RangeField label="Shadow origin bias" unit="cells" value={tuning.shadowBiasCells} min={0} max={0.25} step={0.005} digits={3}
          onChange={(value) => updateTuning("shadowBiasCells", value)} modified={modified("shadowBiasCells")} onReset={resetTuning("shadowBiasCells")} />
        <RangeField label="AO cone aperture" unit="rad" value={tuning.aoConeAperture} min={0.1} max={1.4} step={0.01} digits={2}
          onChange={(value) => updateTuning("aoConeAperture", value)} modified={modified("aoConeAperture")} onReset={resetTuning("aoConeAperture")} />
        <RangeField label="AO strength" unit="%" value={tuning.aoStrength * 100} min={0} max={100} step={1} digits={0}
          onChange={(value) => updateTuning("aoStrength", value / 100)} modified={modified("aoStrength")} onReset={resetTuning("aoStrength")} />
        <RangeField label="AO radius" unit="×" value={tuning.aoRadiusScale} min={0.1} max={3} step={0.05} digits={2}
          onChange={(value) => updateTuning("aoRadiusScale", value)} modified={modified("aoRadiusScale")} onReset={resetTuning("aoRadiusScale")} />
        <RangeField label="Cone step budget" unit="steps" value={tuning.coneStepBudget} min={1} max={48} step={1} digits={0}
          onChange={(value) => updateTuning("coneStepBudget", value)} modified={modified("coneStepBudget")} onReset={resetTuning("coneStepBudget")} />
        <RangeField label="Shaded lights" unit="lights" value={tuning.maximumShadedLights} min={1} max={8} step={1} digits={0}
          onChange={(value) => updateTuning("maximumShadedLights", value)} modified={modified("maximumShadedLights")} onReset={resetTuning("maximumShadedLights")} />
        <RangeField label="Normal escape" unit="cells" value={tuning.coneNormalEscapeCells} min={0} max={2} step={0.05} digits={2}
          onChange={(value) => updateTuning("coneNormalEscapeCells", value)} modified={modified("coneNormalEscapeCells")} onReset={resetTuning("coneNormalEscapeCells")} />
        <RangeField label="Emitter clearance" unit="cells" value={tuning.coneEmitterClearanceCells} min={0} max={8} step={0.25} digits={2}
          onChange={(value) => updateTuning("coneEmitterClearanceCells", value)} modified={modified("coneEmitterClearanceCells")} onReset={resetTuning("coneEmitterClearanceCells")} />
      </FieldList>
      {/* The exact-ray budgets are the whole cost of a shadow under EXACT and
          are nearly inert under CONES, where the cone march has its own step
          budget and the exact traversal is never reached. They appear with the
          arm that spends them rather than sitting in a drawer that claims to
          govern the frame everyone actually renders. */}
      {exactBudgets && <FieldList data-testid="exact-visibility-budgets">
        <RangeField label="Visibility nodes" unit="nodes" value={tuning.visibilityNodeVisits} min={1} max={128} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityNodeVisits", value)} modified={modified("visibilityNodeVisits")} onReset={resetTuning("visibilityNodeVisits")} />
        <RangeField label="Visibility leaves" unit="bricks" value={tuning.visibilityLeafVisits} min={1} max={32} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityLeafVisits", value)} modified={modified("visibilityLeafVisits")} onReset={resetTuning("visibilityLeafVisits")} />
        <RangeField label="Visibility voxel work" unit="tests" value={tuning.visibilityWorkItems} min={16} max={1024} step={16} digits={0}
          onChange={(value) => updateTuning("visibilityWorkItems", value)} modified={modified("visibilityWorkItems")} onReset={resetTuning("visibilityWorkItems")} />
        <RangeField label="Intersections" unit="hits" value={tuning.visibilityIntersections} min={1} max={4} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityIntersections", value)} modified={modified("visibilityIntersections")} onReset={resetTuning("visibilityIntersections")} />
      </FieldList>}
      <FieldList>
        <RangeField label="Area samples" unit="rays" value={tuning.stableAreaLightSamples} min={1} max={2} step={1} digits={0}
          onChange={(value) => updateTuning("stableAreaLightSamples", value)} modified={modified("stableAreaLightSamples")} onReset={resetTuning("stableAreaLightSamples")} />
        <RangeField label="AO samples" unit="cones" value={tuning.stableAoSamples} min={1} max={4} step={1} digits={0}
          onChange={(value) => updateTuning("stableAoSamples", value)} modified={modified("stableAoSamples")} onReset={resetTuning("stableAoSamples")} />
      </FieldList>
      {/* Only the per-light plane consults the slot, so it appears with that
          plane rather than sitting inert beside the others. */}
      {lightSlot && <FieldList>
        <RangeField label="Cached light slot" unit="slot" value={svoStageLightSlot}
          min={0} max={SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT} step={1} digits={0}
          onChange={setSvoStageLightSlot}
          hint="Which of the eight cached per-light visibilities the prepass plane is decoded for." />
      </FieldList>}
      </details>
    </>,
  };
}
