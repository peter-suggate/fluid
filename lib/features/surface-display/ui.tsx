"use client";
import { useEffect, useRef } from "react";
import { Droplets, Waves } from "lucide-react";
import { getMethod } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import { PARTICLE_DISPLAY_OPTIONS } from "../../core/visual-layers";
import { simulation } from "../../core/simulation/controller";
import { resolveWaterOptics, WATER_ROUGHNESS_MIN } from "../../core/webgpu-lighting";
import { ToolstripRow } from "../../../components/toolstrip";
import { Choice, Slider, Value } from "../../../components/ui";
import { PARTICLE_SURFACE_SMOOTHING_RANGE, PARTICLE_SURFACE_THICKNESS_RANGE } from "../../core/webgpu-particle-surface";
import { surfaceDisplayFeature, type FineSurfaceState, type FluidParticleDisplay, type FluidSurfaceRenderMode } from "./definition";
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
  return <>
  <ToolstripRow name={control.label} hint={control.hint} testId="fluid-surface-render-row" icon={<Waves width={14} height={14} />}>
    <Choice<FluidSurfaceRenderMode> ariaLabel="Fluid surface render mode" value={value} options={control.options} onChange={change} />
    {value === "shaded" && <WaterRoughnessControl />}
    {keepsParticles && <>
      <span className="toolstrip-name" title="The method's own particles, drawn through the liquid. Under the Simple surface they sit in the water, fogged by the liquid in front of them.">Particles</span>
      <div data-testid="fluid-particle-display">
        <Choice<FluidParticleDisplay> ariaLabel="Particle display" value={particles} options={PARTICLE_DISPLAY_OPTIONS} onChange={changeParticles} />
      </div>
    </>}
  </ToolstripRow>
  {keepsParticles && particles === "fine" && <FineSurfaceRows />}
  </>;
}

/** The Fine display's sliders: presentation only, so they move live with no edit to commit. */
function FineSurfaceRows() {
  const session = useSession();
  const change = session.ui(state => state.setFineSurface);
  const slider = (key: keyof FineSurfaceState, label: string, min: number, max: number, step: number, hint: string) =>
    <FineSlider key={key} setting={key} label={label} min={min} max={max} step={step} hint={hint} onInput={value => change({ [key]: value })} />;
  return <>
    <ToolstripRow name="Fine skin" hint="How the fine particle surface is meshed from the particles." testId="fine-surface-row" icon={<Waves width={14} height={14} />}>
      {slider("fineSurfaceGreed", "Greed", 0, 1, 0.01, "How few neighbours a particle needs to be liquid. Low meshes only bulk liquid and leaves thin sheets and clusters as spray; high meshes nearly every pair.")}
      {slider("fineSurfaceSmoothing", "Smooth", ...PARTICLE_SURFACE_SMOOTHING_RANGE, 0.05, "Kernel radius in cells. Larger merges particles into smoother liquid; smaller keeps ripples and separate droplets.")}
      {slider("fineSurfaceThickness", "Thick", ...PARTICLE_SURFACE_THICKNESS_RANGE, 0.05, "How far the skin stands off the particles' weighted centre, in cells. Larger fills gaps and thickens sheets.")}
    </ToolstripRow>
    <ToolstripRow name="Spray" hint="Particles the fine skin does not enclose, drawn as droplets." testId="fine-spray-row" icon={<Droplets width={14} height={14} />}>
      {slider("fineSpraySize", "Size", 0.1, 1, 0.05, "Droplet radius as a fraction of the particle sphere's.")}
      {slider("fineSprayOpacity", "Opacity", 0, 1, 0.05, "Droplet opacity at its centre.")}
    </ToolstripRow>
  </>;
}

function FineSlider({ setting, label, min, max, step, hint, onInput }: {
  setting: keyof FineSurfaceState; label: string; min: number; max: number; step: number; hint: string; onInput: (value: number) => void;
}) {
  const session = useSession();
  const value = session.ui(state => state[setting]);
  return <>
    <span className="toolstrip-name" title={hint}>{label}</span>
    <Slider ariaLabel={`Fine surface ${label.toLowerCase()}`} value={value} min={min} max={max} step={step} hint={hint} onInput={onInput} onChange={onInput} />
    <Value value={value} step={step} />
  </>;
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
