"use client";

import { SvoRadianceReconstructionControlRow } from "../features/radiance/ui";
import { useSession } from "../../core/session/session-context";
import { ComposedFeatureSlot, type FeatureControlViews } from "../../framework/ui/slot";
import { PrimaryTraversalRow, SvoPrimaryVisibilityControlRow } from "../features/primary-visibility/ui";
import { SvoLightingVisibilityControlRow, SvoShadowsControlRow, SvoAmbientOcclusionControlRow } from "../features/lighting-visibility/ui";
import { resolveSvoPipelineComposition } from "./composition";

/** The SVO package exports its own views; application composition can install
 * them alongside other packages without the SVO host importing the app. */
export const SVO_FEATURE_VIEWS: FeatureControlViews = {
  "svo.radiance-reconstruction/mode": SvoRadianceReconstructionControlRow,
  "svo.primary-visibility/primary-work": PrimaryTraversalRow,
  "svo.primary-visibility/mode": SvoPrimaryVisibilityControlRow,
  "svo.lighting-visibility/mode": SvoLightingVisibilityControlRow,
  "svo.lighting-visibility/shadows": SvoShadowsControlRow,
  "svo.lighting-visibility/ambient-occlusion": SvoAmbientOcclusionControlRow,
};

export function SvoFeatureSlot({ slot }: { readonly slot: string }) {
  const session = useSession();
  const primaryTraversal = session.ui(state => state.svoPrimaryTraversal);
  const coneTracingMode = session.ui(state => state.svoConeTracingMode);
  const coneRadianceReconstruction = session.ui(state => state.svoRenderTuning.coneRadianceReconstruction);
  return <ComposedFeatureSlot
    composition={resolveSvoPipelineComposition({ primaryTraversal, coneTracingMode, coneRadianceReconstruction }).composition}
    views={SVO_FEATURE_VIEWS} slot={slot} />;
}
