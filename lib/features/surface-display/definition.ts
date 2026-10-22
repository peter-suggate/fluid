import type { FeatureDefinition } from "../../framework/composition";
import { choiceQuery, numberQuery, queryRecord } from "../../framework/persistence";
import { DEFAULT_PARTICLE_SURFACE_SETTINGS as FINE, PARTICLE_SURFACE_SMOOTHING_RANGE, PARTICLE_SURFACE_THICKNESS_RANGE } from "../../core/webgpu-particle-surface";
export type FluidSurfaceRenderMode = "shaded" | "wireframe" | "simple";
/** The liquid's own particles over the surface (core/visual-layers ParticleDisplay). */
export type FluidParticleDisplay = "off" | "motion" | "speed" | "fine";
/** The Fine display's sliders (core/webgpu-particle-surface ParticleSurfaceSettings). */
export interface FineSurfaceState {
  fineSurfaceGreed: number; fineSurfaceSmoothing: number; fineSurfaceThickness: number;
  fineSpraySize: number; fineSprayOpacity: number;
}
export interface SurfaceDisplayState extends FineSurfaceState { fluidSurfaceRenderMode: FluidSurfaceRenderMode; fluidParticleDisplay: FluidParticleDisplay }
export const SURFACE_DISPLAY_OPTIONS = [{ value: "shaded", label: "Shade" }, { value: "wireframe", label: "Wire" }, { value: "simple", label: "Simple" }] as const;
export const surfaceDisplayQuery = queryRecord<SurfaceDisplayState>({
  fluidSurfaceRenderMode: choiceQuery("fluidSurface", "simple", SURFACE_DISPLAY_OPTIONS.map(option => option.value)),
  // On by default: a method that keeps particles shows where they are agitated.
  fluidParticleDisplay: choiceQuery<FluidParticleDisplay>("fluidParticles", "motion", ["off", "motion", "speed", "fine"]),
  fineSurfaceGreed: numberQuery("fineGreed", FINE.greed, 0, 1),
  fineSurfaceSmoothing: numberQuery("fineSmooth", FINE.smoothing, ...PARTICLE_SURFACE_SMOOTHING_RANGE),
  fineSurfaceThickness: numberQuery("fineThick", FINE.thickness, ...PARTICLE_SURFACE_THICKNESS_RANGE),
  fineSpraySize: numberQuery("spraySize", FINE.spraySize, 0.1, 1),
  fineSprayOpacity: numberQuery("sprayOpacity", FINE.sprayOpacity, 0, 1),
});
export const surfaceDisplayFeature = {
  id: "presentation.surface-display", controls: [{ id: "mode", label: "Fluid surface", kind: "choice", update: "live", setting: "fluidSurfaceRenderMode",
    hint: "Shade the liquid, show it as a murky body its particles are seen through, or show triangle edges.", options: SURFACE_DISPLAY_OPTIONS }],
  placements: [{ slot: "scene.surface", control: "mode", presentation: "compact", priority: "high" }, { slot: "frame.surface", control: "mode", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
