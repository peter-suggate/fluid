import { SVO_GBUFFER_PORT, SVO_SCENE_PUBLICATION_PORT } from "../../contracts/ports";
import type { FeatureDefinition } from "../../../framework/composition";

export const SVO_LIGHTING_VISIBILITY_OPTIONS = [
  { value: "cones", label: "CONES", hint: "Cone-traced soft shadows, AO and GI, fed by the reduced-rate prepass and world-GI cache." },
  { value: "raster-ao", label: "RASTER + AO", hint: "Cached sun shadow maps and screen-space contact AO. Selects Mesh primary visibility. GI is absent; optional Coarse voxel AO adds approximate offscreen occlusion." },
  { value: "exact", label: "EXACT", hint: "No cone stage runs; shadows and AO use bounded exact SVO visibility rays. Sharp reference shadows, costlier per pixel." },
  { value: "off", label: "OFF", hint: "No visibility work at all: unshadowed direct lighting, no AO, no GI." },
] as const;

export const SVO_LIGHTING_VISIBILITY_FEATURE = {
  id: "svo.lighting-visibility",
  label: "Lighting visibility",
  inputs: [
    { port: SVO_SCENE_PUBLICATION_PORT, provider: "svo.scene-publication" },
    { port: SVO_GBUFFER_PORT, provider: "svo.primary-visibility" },
  ],
  variants: [
    { id: "cones", point: "svo.lighting-visibility", update: "live", requires: ["svo.scene-publication", "svo.radiance"], provides: ["svo.lighting-visibility"] },
    { id: "raster-ao", point: "svo.lighting-visibility", default: true, update: "live", requires: ["svo.scene-publication"], provides: ["svo.lighting-visibility"] },
    { id: "exact", point: "svo.lighting-visibility", update: "live", requires: ["svo.scene-publication"], provides: ["svo.lighting-visibility"] },
    { id: "off", point: "svo.lighting-visibility", update: "live", provides: ["svo.lighting-visibility"] },
  ],
  controls: [
    { id: "mode", label: "Visibility source", kind: "choice", setting: "svoConeTracingMode", options: SVO_LIGHTING_VISIBILITY_OPTIONS, update: "live" },
    { id: "shadows", label: "Shadows", kind: "toggle", setting: "svoShadowsEnabled", update: "live" },
    { id: "ambient-occlusion", label: "AO", kind: "toggle", setting: "svoAmbientOcclusionEnabled", update: "live" },
    { id: "lattice", label: "Lattice visibility", kind: "toggle", setting: "svoLatticeVisibilityEnabled", update: "live",
      hint: "On samples the shadow and AO cones on the half-voxel lattice of each lit face, once per point, and interpolates between points. Off samples them per reduced-resolution screen texel. Serves reduced-rate cones with full-resolution relight, GI off and Smooth surface off. Smooth surfaces use screen-space lighting." },
    { id: "far-ao-taps", label: "Minimum AO taps", kind: "number", setting: "rasterFarAoTaps", unit: "taps", min: 0, max: 36, step: 4, update: "live",
      hint: "Minimum contact-AO taps in the visible scene. Clear regions use this count; curved surfaces and nearby contacts receive more, up to 36. Zero skips contact AO at the minimum. Coarse voxel AO is separate." },
    { id: "still-view-reuse", label: "Still-camera reuse", kind: "toggle", setting: "rasterStillViewReuse", update: "live",
      hint: "While paused with no bodies and the camera still, keep the last frame's shadow and AO visibility instead of recomputing it. Water shadows and stage ablations always recompute." },
    { id: "ao-tap-overlay", label: "AO tap budget", kind: "toggle", setting: "svoStageView", update: "live",
      hint: "Show the selected horizon-AO sample budget at each surface pixel. Excludes coarse AO and receiver sharing." },
    { id: "cloud-coverage", label: "Cloud coverage", kind: "number", setting: "lighting.environment.cloudCoverage", min: 0, max: 1, step: .01, update: "live", hint: "Distant clouds shared by visible sky and reflections. Zero keeps the clear gradient." },
    { id: "direct-light-samples", label: "Direct light samples", kind: "number", setting: "maximumShadedLights", min: 1, max: 32, step: 1, update: "live", hint: "Direct emitter samples per receiver. Area lights may consume two samples each." },
    { id: "native-ao", label: "Native voxel AO", kind: "toggle", setting: "svoNativeAmbientOcclusionEnabled", update: "live", hint: "Adds finer contact shading within imported voxel objects. Slower to render; other objects keep the selected AO method." },
    { id: "surface-textures", label: "Image textures", kind: "toggle", setting: "svoSurfaceTexturesEnabled", update: "live", hint: "Show authored color, roughness and normal-map detail on voxel surfaces. Geometry and visibility remain voxel based." },
    { id: "shadow-resolution", label: "Shadow resolution", kind: "choice", setting: "rasterShadowResolution", options: [{value:"2048",label:"2048"},{value:"4096",label:"4096"}], update: "live", hint: "Higher resolution preserves finer branch shadows. 4096 uses four times the shadow-map memory. Fluid continues unchanged." },
  ],
  sections: [
    { id: "ao-diagnostics", surface: "view-rail", slot: "view.ao-diagnostics", label: "AO diagnostics",
      hint: "Inspect where ambient occlusion spends samples.", icon: "M3 3h18v18H3ZM7 15v3M12 10v8M17 6v12" },
    // A sun: what the scene is lit by, and what stands in its way.
    { id: "lighting", surface: "view-rail", slot: "view.lighting", label: "Lighting", hint: "How the frame is lit and occluded.",
      icon: "M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8ZM12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" },
  ],
  placements: [
    { slot: "view.lighting", control: "mode", presentation: "compact" },
    { slot: "view.lighting", control: "shadows", presentation: "compact" },
    { slot: "view.lighting", control: "shadow-resolution", presentation: "compact" },
    { slot: "view.lighting", control: "ambient-occlusion", presentation: "compact" },
    { slot: "view.lighting", control: "native-ao", presentation: "compact" },
    { slot: "frame.lighting", control: "native-ao", presentation: "compact" },
    { slot: "view.lighting", control: "surface-textures", presentation: "compact" },
    { slot: "view.lighting", control: "direct-light-samples", presentation: "compact" },
    { slot: "view.lighting", control: "cloud-coverage", presentation: "compact" },
    { slot: "view.ao-diagnostics", control: "ao-tap-overlay", presentation: "compact" },
    { slot: "frame.lighting", control: "mode", presentation: "compact" },
    { slot: "frame.lighting", control: "shadows", presentation: "compact" },
    { slot: "frame.lighting", control: "shadow-resolution", presentation: "compact" },
    { slot: "frame.lighting", control: "ambient-occlusion", presentation: "compact" },
    { slot: "frame.lighting", control: "lattice", presentation: "compact" },
    { slot: "frame.lighting", control: "far-ao-taps", presentation: "compact" },
    { slot: "view.lighting", control: "far-ao-taps", presentation: "compact" },
    // A renderer saving, not a look: it stands with the rail's other savings.
    { slot: "view.performance", control: "still-view-reuse", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
