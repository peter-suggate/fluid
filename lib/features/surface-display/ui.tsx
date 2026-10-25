"use client";
import { useEffect, useRef } from "react";
import { Waves } from "lucide-react";
import { getMethod } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import { PARTICLE_DISPLAY_OPTIONS } from "../../core/visual-layers";
import { simulation } from "../../core/simulation/controller";
import { resolveWaterOptics, WATER_ROUGHNESS_MIN } from "../../core/webgpu-lighting";
import { ToolstripRow } from "../../../components/toolstrip";
import { RailChoice, RailRange, RailSub } from "../../../components/rail";
import type { FeatureControlViewProps } from "../../framework/ui/slot";
import { Choice, Slider, Value } from "../../../components/ui";
import { PARTICLE_SURFACE_MERGE_RANGE, PARTICLE_SURFACE_SMOOTHING_RANGE, PARTICLE_SURFACE_THICKNESS_RANGE } from "../../core/webgpu-particle-surface";
import { surfaceDisplayFeature, type FineSurfaceState, type FluidParticleDisplay, type FluidSurfaceRenderMode } from "./definition";
export function SurfaceDisplayRow() {
  const session = useSession();
  const value = session.ui(state => state.fluidSurfaceRenderMode);
  const change = session.ui(state => state.setFluidSurfaceRenderMode);
  const control = surfaceDisplayFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} testId="fluid-surface-render-row" icon={<Waves width={14} height={14} />}>
    <Choice<FluidSurfaceRenderMode> ariaLabel="Fluid surface render mode" value={value} options={control.options} onChange={change} />
    {value === "shaded" && <WaterRoughnessControl />}
  </ToolstripRow>;
}

/**
 * The particle display, as a line of the study rail's layer card: the
 * method's own particles are a layer over the liquid like the field views,
 * drawn through the whole volume rather than on a plane. The Fine display's
 * sliders stand under it while it is chosen.
 */
export function ParticleDisplayLines({ control }: FeatureControlViewProps) {
  const session = useSession();
  const particles = session.ui(state => state.fluidParticleDisplay);
  const changeParticles = session.ui(state => state.setFluidParticleDisplay);
  const methodId = session.method(state => state.methodId);
  // Only a method that publishes its particles as a composed layer has any to draw.
  const layers = getMethod(methodId).capabilities?.visualLayers;
  if (layers === undefined || layers.hidden.includes("particles")) return null;
  return <>
    <RailChoice<FluidParticleDisplay> label={control.label} hint={control.hint} ariaLabel="Particle display"
      value={particles} options={PARTICLE_DISPLAY_OPTIONS} onChange={changeParticles} testId="fluid-particle-display" />
    {particles === "fine" && <FineSurfaceLines />}
  </>;
}

/** The Fine display's sliders: presentation only, so they move live with no edit to commit. */
function FineSurfaceLines() {
  const session = useSession();
  const change = session.ui(state => state.setFineSurface);
  const slider = (key: keyof FineSurfaceState, label: string, min: number, max: number, step: number, hint: string, aria = `Fine surface ${label.toLowerCase()}`) =>
    <FineSlider key={key} setting={key} label={label} ariaLabel={aria} min={min} max={max} step={step} hint={hint} onInput={value => change({ [key]: value })} />;
  return <RailSub>
    {slider("fineSurfaceGreed", "Greed", 0, 1, 0.01, "How few neighbours a particle needs to be liquid. Low meshes only bulk liquid and leaves thin sheets and clusters as spray; high meshes nearly every pair.")}
    {slider("fineSurfaceSmoothing", "Smooth", ...PARTICLE_SURFACE_SMOOTHING_RANGE, 0.05, "Kernel radius in cells. Larger merges particles into smoother liquid; smaller keeps ripples and separate droplets.")}
    {slider("fineSurfaceThickness", "Thick", ...PARTICLE_SURFACE_THICKNESS_RANGE, 0.05, "How far the skin stands off the particles' weighted centre, in cells. Larger fills gaps and thickens sheets.")}
    {slider("fineSurfaceMerge", "Merge", ...PARTICLE_SURFACE_MERGE_RANGE, 0.05, "How far, in cells, the skin must stand in front of the level-set surface to be drawn. Closer than that, the smooth surface shows instead; 0 draws the skin wherever it is nearer.")}
    {slider("fineSpraySize", "Spray size", 0.1, 1, 0.05, "Spray footprint radius as a fraction of a Motion sample's. Spray is the particles the fine skin does not enclose, drawn as droplets.", "Fine surface size")}
    {slider("fineSprayOpacity", "Spray opacity", 0, 1, 0.05, "Scales the spray's Motion opacity.", "Fine surface opacity")}
  </RailSub>;
}

function FineSlider({ setting, label, ariaLabel, min, max, step, hint, onInput }: {
  setting: keyof FineSurfaceState; label: string; ariaLabel: string; min: number; max: number; step: number; hint: string; onInput: (value: number) => void;
}) {
  const session = useSession();
  const value = session.ui(state => state[setting]);
  return <RailRange label={label} ariaLabel={ariaLabel} value={value} min={min} max={max} step={step} hint={hint} onInput={onInput} />;
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
