import { SVO_GBUFFER_PORT, SVO_SCENE_PUBLICATION_PORT } from "../../contracts/ports";
import type { FeatureDefinition } from "../../../framework/composition";

export const SVO_PRIMARY_VISIBILITY_OPTIONS = [
  { value: "traced", label: "Ray traced", hint: "Trace the accepted voxel scene for each camera pixel." },
  { value: "mesh", label: "Rasterized", hint: "Draw cached exposed voxel faces. Disable Smooth surface to use mesh rasterization; geometry is withheld while it is enabled. Lighting is unchanged." },
] as const;

export const SVO_PRIMARY_VISIBILITY_FEATURE = {
  id: "svo.primary-visibility",
  label: "Primary visibility",
  inputs: [{ port: SVO_SCENE_PUBLICATION_PORT, provider: "svo.scene-publication" }],
  outputs: [SVO_GBUFFER_PORT],
  variants: [
    { id: "mesh", point: "svo.primary-visibility", default: true, update: "rebuild", requires: ["svo.scene-publication"], provides: ["svo.gbuffer"] },
    { id: "traced", point: "svo.primary-visibility", update: "rebuild", requires: ["svo.scene-publication"], provides: ["svo.gbuffer"] },
    // Diagnostic proxy arm remains explicitly selectable by the runtime override;
    // the product selector exposes the two supported production paths above.
    { id: "raster", point: "svo.primary-visibility", update: "rebuild", requires: ["svo.scene-publication"], provides: ["svo.gbuffer"] },
  ],
  controls: [
    { id: "mode", label: "Primary visibility", kind: "choice", setting: "svoPrimaryTraversal", options: SVO_PRIMARY_VISIBILITY_OPTIONS, update: "rebuild" },
    { id: "primary-work", label: "Primary ray work", kind: "choice", setting: "svoStageView", update: "rebuild" },
  ],
  placements: [
    { slot: "frame.options", control: "mode", presentation: "expanded" },
    { slot: "scene.visibility", control: "primary-work", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
