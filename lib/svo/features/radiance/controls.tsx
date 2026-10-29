"use client";

import { RailRange } from "../../../../components/rail";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { SvoFeatureSlot } from "../../pipeline/ui-slots";

export function renderReducedShadeControls(): SvoStageControls {
  return { node: <><SvoFeatureSlot slot="frame.reconstruction" /></> };
}

export function renderGiCompositionControls({ svoGlobalIlluminationEnabled, svoConeTracingMode, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "svoGlobalIlluminationEnabled" | "svoConeTracingMode" | "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  const active = svoGlobalIlluminationEnabled && svoConeTracingMode === "cones";
  return { node: <div className="rail-group" data-testid="gi-composition-controls"
      data-withheld={active ? undefined : "true"}>
      <RailRange label="GI bounce" unit="%" disabled={!active} value={tuning.giBounceStrength * 100} min={0} max={400} step={5} digits={0}
        onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
        hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
      <RailRange label="GI occlusion" unit="%" disabled={!active} value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
        hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
      <RailRange label="Diffuse environment" unit="%" disabled={!active} value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} digits={0}
        onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
        hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
      <RailRange label="GI cone aperture" unit="rad" disabled={!active} value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} digits={2}
        onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
        hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
      <RailRange label="GI cones" unit="cones" disabled={!active} value={tuning.giConeCount} min={3} max={4} step={1} digits={0}
        onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
        hint="Four cones give fuller hemispherical coverage; three omits the normal cone. Each cone uses the configured step budget." />
      </div> };
}
