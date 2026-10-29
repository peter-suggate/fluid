"use client";
import { SvoLatticeVisibilityControlRow } from "../lighting-visibility/ui";
import { RailRange, RailToggle } from "../../../../components/rail";
import { useSession } from "../../../core/session/session-context";
import { useSvoRenderTuningValue } from "../../pipeline/tuning-binding";
import type { SvoRenderTuning } from "../../pipeline/svo-render-tuning";

type GiNumberKey="giBounceStrength"|"giOcclusionStrength"|"giEnvironmentStrength"|"giConeAperture"|"giConeCount"|"coneStepBudget";
function GiRange({name,label,min,max,step,scale=1,unit,hint,disabled}:{name:GiNumberKey,label:string,min:number,max:number,step:number,scale?:number,unit?:string,hint:string,disabled:boolean}){
  const {value,set,modified,reset}=useSvoRenderTuningValue(name);
  return <RailRange label={label} value={value*scale} min={min} max={max} step={step} digits={step<1?2:0} unit={unit}
    disabled={disabled} onChange={next=>set(next/scale as SvoRenderTuning[GiNumberKey])} modified={modified} onReset={reset} hint={hint}/>;
}
export function GiLightingControls(){
  const session=useSession();
  const enabled=session.ui(s=>s.svoGlobalIlluminationEnabled),cones=session.ui(s=>s.svoConeTracingMode==="cones");
  const cache=session.ui(s=>s.svoWorldGiCacheEnabled);
  const setEnabled=session.ui(s=>s.setSvoGlobalIlluminationEnabled),setCache=session.ui(s=>s.setSvoWorldGiCacheEnabled);
  const active=enabled&&cones;
  return <details className="rail-more is-nested" data-testid="gi-lighting-controls"><summary>Cones and indirect light</summary>
    <RailToggle label="Global illumination" checked={enabled} onChange={setEnabled} testId="gi-enabled"
      hint="Gather diffuse bounced light from voxel radiance. Enabling selects Cones. GI bounce starts at 100%."/>
    {!cones&&enabled&&<p className="tool-options-note">Select Cones to apply indirect light.</p>}
    <RailToggle label="GI cache" checked={cache} onChange={setCache} testId="gi-cache"
      hint="Remember world-space diffuse gathers during reduced-rate full-resolution relighting. Other shading modes gather inline. Off recomputes the same light; this preference does not change the light model."/>
    <SvoLatticeVisibilityControlRow />
    <GiRange name="giBounceStrength" label="GI bounce" scale={100} min={0} max={400} step={5} unit="%" disabled={!active} hint="Strength of gathered diffuse light; direct highlights and emission are unchanged."/>
    <GiRange name="giOcclusionStrength" label="GI occlusion" scale={100} min={0} max={100} step={1} unit="%" disabled={!active} hint="Darken diffuse sky fill using the same cones. Requires AO enabled."/>
    <GiRange name="giEnvironmentStrength" label="Diffuse environment" scale={100} min={0} max={200} step={5} unit="%" disabled={!active} hint="Analytic diffuse sky fill retained alongside gathered bounce."/>
    <GiRange name="giConeAperture" label="GI cone aperture" min={.4} max={1.4} step={.01} unit="rad" disabled={!active} hint="Wider cones produce smoother, broader indirect light and occlusion."/>
    <GiRange name="giConeCount" label="GI cones" min={3} max={4} step={1} disabled={!active} hint="Four includes the normal cone; three uses the surrounding cones."/>
    <GiRange name="coneStepBudget" label="Steps per cone" min={1} max={48} step={1} disabled={!cones} hint="Each diffuse GI cone receives this full budget. Exhausted cones retain finite partial light; higher values travel farther."/>
  </details>;
}
