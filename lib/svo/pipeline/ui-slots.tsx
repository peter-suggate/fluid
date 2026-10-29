"use client";

import { RenderResolutionControlRow } from "../features/presentation/controls";
import { GiLightingControls } from "../features/radiance/gi-ui";
import { LensBloomControls } from "../../features/lens-bloom/ui";
import { VoxelSpecularControls } from "../features/radiance/voxel-specular-ui";
import { SvoRadianceReconstructionControlRow } from "../features/radiance/ui";
import { useSession } from "../../core/session/session-context";
import { ComposedFeatureSlot, type FeatureControlViews } from "../../framework/ui/slot";
import { PrimaryTraversalRow, SvoPrimaryVisibilityControlRow } from "../features/primary-visibility/ui";
import { SvoShadowResolutionControlRow, SvoNativeAmbientOcclusionControlRow, SvoSurfaceTexturesControlRow, SvoDirectLightSamplesControlRow, SvoCloudCoverageControlRow, SvoLightingVisibilityControlRow, SvoShadowsControlRow, SvoAmbientOcclusionControlRow, SvoLatticeVisibilityControlRow, SvoFarAoTapsControlRow, SvoStillViewReuseControlRow } from "../features/lighting-visibility/ui";
import { resolveSvoPipelineComposition } from "./composition";
import { SvoAoTapOverlayControlRow } from "../features/lighting-visibility/ui";

/** The SVO package exports its own views; application composition can install
 * them alongside other packages without the SVO host importing the app. */
export const SVO_FEATURE_VIEWS: FeatureControlViews = {
  "svo.presentation/resolution": RenderResolutionControlRow,
  "svo.lighting-visibility/global-illumination": GiLightingControls,
  "svo.lighting-visibility/lens-bloom": LensBloomControls,
  "svo.lighting-visibility/voxel-specular": VoxelSpecularControls,
  "svo.lighting-visibility/shadow-resolution": SvoShadowResolutionControlRow,
  "svo.lighting-visibility/native-ao": SvoNativeAmbientOcclusionControlRow,
  "svo.lighting-visibility/surface-textures": SvoSurfaceTexturesControlRow,
  "svo.lighting-visibility/direct-light-samples": SvoDirectLightSamplesControlRow,
  "svo.lighting-visibility/cloud-coverage": SvoCloudCoverageControlRow,
  "svo.lighting-visibility/ao-tap-overlay": SvoAoTapOverlayControlRow,
  "svo.radiance-reconstruction/mode": SvoRadianceReconstructionControlRow,
  "svo.primary-visibility/primary-work": PrimaryTraversalRow,
  "svo.primary-visibility/mode": SvoPrimaryVisibilityControlRow,
  "svo.lighting-visibility/mode": SvoLightingVisibilityControlRow,
  "svo.lighting-visibility/shadows": SvoShadowsControlRow,
  "svo.lighting-visibility/ambient-occlusion": SvoAmbientOcclusionControlRow,
  "svo.lighting-visibility/lattice": SvoLatticeVisibilityControlRow,
  "svo.lighting-visibility/far-ao-taps": SvoFarAoTapsControlRow,
  "svo.lighting-visibility/still-view-reuse": SvoStillViewReuseControlRow,
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
