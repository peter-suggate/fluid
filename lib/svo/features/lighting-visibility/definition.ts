import { SVO_GBUFFER_PORT, SVO_SCENE_PUBLICATION_PORT } from "../../contracts/ports";
import type { FeatureDefinition } from "../../../framework/composition";

export const SVO_LIGHTING_VISIBILITY_OPTIONS = [
  { value: "cones", label: "CONES", hint: "Cone-traced soft shadows, AO and GI, fed by the reduced-rate prepass and world-GI cache." },
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
    { id: "cones", point: "svo.lighting-visibility", default: true, update: "live", requires: ["svo.scene-publication", "svo.radiance"], provides: ["svo.lighting-visibility"] },
    { id: "exact", point: "svo.lighting-visibility", update: "live", requires: ["svo.scene-publication"], provides: ["svo.lighting-visibility"] },
    { id: "off", point: "svo.lighting-visibility", update: "live", provides: ["svo.lighting-visibility"] },
  ],
  controls: [
    { id: "mode", label: "Visibility source", kind: "choice", setting: "svoConeTracingMode", options: SVO_LIGHTING_VISIBILITY_OPTIONS, update: "live" },
    { id: "shadows", label: "Shadows", kind: "toggle", setting: "svoShadowsEnabled", update: "live" },
    { id: "ambient-occlusion", label: "AO", kind: "toggle", setting: "svoAmbientOcclusionEnabled", update: "live" },
  ],
  placements: [
    { slot: "frame.lighting", control: "mode", presentation: "expanded" },
    { slot: "frame.lighting", control: "shadows", presentation: "expanded" },
    { slot: "frame.lighting", control: "ambient-occlusion", presentation: "expanded" },
  ],
} as const satisfies FeatureDefinition;
