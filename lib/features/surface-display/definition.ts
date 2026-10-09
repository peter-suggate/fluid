import type { FeatureDefinition } from "../../framework/composition";
import { choiceQuery, queryRecord } from "../../framework/persistence";
export type FluidSurfaceRenderMode = "shaded" | "wireframe" | "simple";
/** The liquid's own particles over the surface (core/visual-layers ParticleDisplay). */
export type FluidParticleDisplay = "off" | "motion" | "speed";
export interface SurfaceDisplayState { fluidSurfaceRenderMode: FluidSurfaceRenderMode; fluidParticleDisplay: FluidParticleDisplay }
export const SURFACE_DISPLAY_OPTIONS = [{ value: "shaded", label: "Shade" }, { value: "wireframe", label: "Wire" }, { value: "simple", label: "Simple" }] as const;
export const surfaceDisplayQuery = queryRecord<SurfaceDisplayState>({
  fluidSurfaceRenderMode: choiceQuery("fluidSurface", "simple", SURFACE_DISPLAY_OPTIONS.map(option => option.value)),
  // On by default: a method that keeps particles shows where they are agitated.
  fluidParticleDisplay: choiceQuery<FluidParticleDisplay>("fluidParticles", "motion", ["off", "motion", "speed"]),
});
export const surfaceDisplayFeature = {
  id: "presentation.surface-display", controls: [{ id: "mode", label: "Fluid surface", kind: "choice", update: "live", setting: "fluidSurfaceRenderMode",
    hint: "Shade the liquid, show it as a murky body its particles are seen through, or show triangle edges.", options: SURFACE_DISPLAY_OPTIONS }],
  placements: [{ slot: "scene.surface", control: "mode", presentation: "compact", priority: "high" }, { slot: "frame.surface", control: "mode", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
