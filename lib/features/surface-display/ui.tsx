"use client";
import { Waves } from "lucide-react";
import { getMethod } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import { PARTICLE_DISPLAY_OPTIONS } from "../../core/visual-layers";
import { ToolstripRow } from "../../../components/toolstrip";
import { Choice } from "../../../components/ui";
import { surfaceDisplayFeature, type FluidParticleDisplay, type FluidSurfaceRenderMode } from "./definition";
export function SurfaceDisplayRow() {
  const session = useSession();
  const value = session.ui(state => state.fluidSurfaceRenderMode);
  const change = session.ui(state => state.setFluidSurfaceRenderMode);
  const particles = session.ui(state => state.fluidParticleDisplay);
  const changeParticles = session.ui(state => state.setFluidParticleDisplay);
  const methodId = session.method(state => state.methodId);
  // Only a method that publishes its particles as a composed layer has any to draw.
  const layers = getMethod(methodId).capabilities?.visualLayers;
  const keepsParticles = layers !== undefined && !layers.hidden.includes("particles");
  const control = surfaceDisplayFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} testId="fluid-surface-render-row" icon={<Waves width={14} height={14} />}>
    <Choice<FluidSurfaceRenderMode> ariaLabel="Fluid surface render mode" value={value} options={control.options} onChange={change} />
    {keepsParticles && <>
      <span className="toolstrip-name" title="The method's own particles, drawn through the liquid. Under the Simple surface they sit in the water, fogged by the liquid in front of them.">Particles</span>
      <div data-testid="fluid-particle-display">
        <Choice<FluidParticleDisplay> ariaLabel="Particle display" value={particles} options={PARTICLE_DISPLAY_OPTIONS} onChange={changeParticles} />
      </div>
    </>}
  </ToolstripRow>;
}
