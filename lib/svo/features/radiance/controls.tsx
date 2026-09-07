"use client";

import { PipeRange } from "../../../../components/PipeControls";
import type { SvoFeatureControlContext } from "../../pipeline/control-context";
import { SvoFeatureSlot } from "../../pipeline/ui-slots";

export function renderReducedShadeControls() {
  return <SvoFeatureSlot slot="frame.reconstruction" />;
}

export function renderGiCompositionControls({ svoGlobalIlluminationEnabled, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "svoGlobalIlluminationEnabled" | "tuning" | "updateTuning" | "modified" | "resetTuning">) {
  return (<details className="rp-tune"><summary>Bounce tuning</summary>
      <div className="pipe-fields" data-testid="gi-composition-controls"
      data-withheld={svoGlobalIlluminationEnabled ? undefined : "true"}>
      <PipeRange label="GI bounce" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giBounceStrength * 100} min={0} max={400} step={5} digits={0}
        onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
        hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
      <PipeRange label="GI occlusion" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
        hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
      <PipeRange label="Diffuse environment" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} digits={0}
        onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
        hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
      <PipeRange label="GI cone aperture" unit="rad" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} digits={2}
        onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
        hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
      <PipeRange label="GI cones" unit="cones" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeCount} min={3} max={4} step={1} digits={0}
        onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
        hint="Four cones give the best hemispherical coverage; three trades the normal cone for longer marches at the same total budget." />
      </div>
    </details> );
}
