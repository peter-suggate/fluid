"use client";
import { useEffect, useRef } from "react";
import { Waves } from "lucide-react";
import { getMethod } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import { PARTICLE_DISPLAY_OPTIONS } from "../../core/visual-layers";
import { simulation } from "../../core/simulation/controller";
import { resolveWaterOptics, WATER_ROUGHNESS_MIN } from "../../core/webgpu-lighting";
import { ToolstripRow } from "../../../components/toolstrip";
import { Choice, Slider, Value } from "../../../components/ui";
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
    {value === "shaded" && <WaterRoughnessControl />}
    {keepsParticles && <>
      <span className="toolstrip-name" title="The method's own particles, drawn through the liquid. Under the Simple surface they sit in the water, fogged by the liquid in front of them.">Particles</span>
      <div data-testid="fluid-particle-display">
        <Choice<FluidParticleDisplay> ariaLabel="Particle display" value={particles} options={PARTICLE_DISPLAY_OPTIONS} onChange={changeParticles} />
      </div>
    </>}
  </ToolstripRow>;
}

function WaterRoughnessControl() {
  const session = useSession();
  const roughness = session.scene(state => resolveWaterOptics(state.scene.fluid.optics).roughness);
  const editing = useRef(false);
  const finish = () => {
    if (!editing.current) return;
    editing.current = false;
    simulation.commitEdit(undefined, { reseed: false }, session.id);
  };
  useEffect(() => () => {
    if (editing.current) {
      editing.current = false;
      simulation.commitEdit(undefined, { reseed: false }, session.id);
    }
  }, [session]);
  return <>
    <span className="toolstrip-name">Roughness</span>
    <Slider ariaLabel="Water roughness" value={roughness} min={WATER_ROUGHNESS_MIN} max={1} step={0.01}
      hint="Lower values give sharper surface highlights; higher values spread them out."
      onInput={value => {
        if (!editing.current) {
          simulation.beginEdit("Change water roughness", session.id);
          editing.current = true;
        }
        const store = session.scene.getState();
        store.patchFluid({ optics: { ...store.scene.fluid.optics, roughness: value } });
      }}
      onGestureEnd={finish} />
    <Value value={roughness.toFixed(2)} />
  </>;
}
