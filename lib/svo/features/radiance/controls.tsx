"use client";

import { FieldList, RangeField } from "../../../../components/ui";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { SvoFeatureSlot } from "../../pipeline/ui-slots";

export function renderReducedShadeControls(): SvoStageControls {
  return { settings: 1, node: <FieldList><SvoFeatureSlot slot="frame.reconstruction" /></FieldList> };
}

export function renderGiCompositionControls({ svoGlobalIlluminationEnabled, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "svoGlobalIlluminationEnabled" | "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  return { settings: 5, node: <FieldList data-testid="gi-composition-controls"
      data-withheld={svoGlobalIlluminationEnabled ? undefined : "true"}>
      <RangeField label="GI bounce" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giBounceStrength * 100} min={0} max={400} step={5} digits={0}
        onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
        hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
      <RangeField label="GI occlusion" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
        hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
      <RangeField label="Diffuse environment" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} digits={0}
        onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
        hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
      <RangeField label="GI cone aperture" unit="rad" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} digits={2}
        onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
        hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
      <RangeField label="GI cones" unit="cones" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeCount} min={3} max={4} step={1} digits={0}
        onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
        hint="Four cones give the best hemispherical coverage; three trades the normal cone for longer marches at the same total budget." />
      </FieldList> };
}
