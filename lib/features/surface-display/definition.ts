import type { FeatureDefinition } from "../../framework/composition";
import { choiceQuery, queryRecord } from "../../framework/persistence";
export type FluidSurfaceRenderMode = "shaded" | "wireframe" | "simple";
export interface SurfaceDisplayState { fluidSurfaceRenderMode: FluidSurfaceRenderMode }
export const SURFACE_DISPLAY_OPTIONS = [{ value: "shaded", label: "Shade" }, { value: "wireframe", label: "Wire" }, { value: "simple", label: "Simple" }] as const;
export const surfaceDisplayQuery = queryRecord<SurfaceDisplayState>({
  fluidSurfaceRenderMode: choiceQuery("fluidSurface", "simple", SURFACE_DISPLAY_OPTIONS.map(option => option.value)),
});
export const surfaceDisplayFeature = {
  id: "presentation.surface-display", controls: [{ id: "mode", label: "Fluid surface", kind: "choice", update: "live", setting: "fluidSurfaceRenderMode",
    hint: "Shade the liquid, inspect a simple translucent surface, or show triangle edges.", options: SURFACE_DISPLAY_OPTIONS }],
  placements: [{ slot: "scene.surface", control: "mode", presentation: "compact", priority: "high" }, { slot: "frame.surface", control: "mode", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
