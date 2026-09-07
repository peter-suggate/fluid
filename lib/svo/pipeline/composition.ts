import { SVO_RADIANCE_RECONSTRUCTION_FEATURE, type SvoConeRadianceReconstruction } from "../features/radiance/definition";
import { SVO_SCENE_PUBLICATION_PORT } from "../contracts/ports";
import { composeFeatures, type FeatureDefinition } from "../../framework/composition";
import { SVO_PRIMARY_VISIBILITY_FEATURE } from "../features/primary-visibility/definition";
import { SVO_LIGHTING_VISIBILITY_FEATURE } from "../features/lighting-visibility/definition";
import type { SvoConeTracingMode, SvoPrimaryTraversalMode } from "./svo-render-options";

/** Static installed providers. Availability of a particular publication remains
 * a runtime readiness condition, not a reason to silently select another path. */
export const SVO_PIPELINE_FEATURES = [
  { id: "svo.scene-publication", provides: ["svo.scene-publication"], outputs: [SVO_SCENE_PUBLICATION_PORT] },
  { id: "svo.radiance", requires: ["svo.scene-publication"], provides: ["svo.radiance"] },
  SVO_PRIMARY_VISIBILITY_FEATURE,
  SVO_LIGHTING_VISIBILITY_FEATURE,
  SVO_RADIANCE_RECONSTRUCTION_FEATURE,
] as const satisfies readonly FeatureDefinition[];

export interface SvoPipelineSelection {
  readonly primaryTraversal?: SvoPrimaryTraversalMode;
  readonly coneTracingMode?: SvoConeTracingMode;
  readonly coneRadianceReconstruction?: SvoConeRadianceReconstruction;
}

/** Resolve before preparing or changing a pipeline. Numerical and encoder order
 * stay with the SVO host; this validates the selected capability providers. */
const resolved = new Map<string, { composition: ReturnType<typeof composeFeatures>; primaryTraversal: SvoPrimaryTraversalMode; coneTracingMode: SvoConeTracingMode; coneRadianceReconstruction: SvoConeRadianceReconstruction }>();

export function resolveSvoPipelineComposition(selection: SvoPipelineSelection = {}) {
  const primaryTraversal = selection.primaryTraversal ?? "mesh";
  const coneTracingMode = selection.coneTracingMode ?? "cones";
  const coneRadianceReconstruction = selection.coneRadianceReconstruction ?? "full-res-relight";
  const key = `${primaryTraversal}/${coneTracingMode}/${coneRadianceReconstruction}`;
  const cached = resolved.get(key);
  if (cached) return cached;
  const composition = composeFeatures({
    features: SVO_PIPELINE_FEATURES,
    selections: {
      "svo.primary-visibility": primaryTraversal,
      "svo.lighting-visibility": coneTracingMode,
      "svo.radiance-reconstruction": coneRadianceReconstruction,
    },
  });
  const result = Object.freeze({ composition, primaryTraversal, coneTracingMode, coneRadianceReconstruction });
  resolved.set(key, result);
  return result;
}
