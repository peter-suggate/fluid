"use client";
import { useMemo } from "react";
import { ParticleDisplayLines, SurfaceDisplayRow } from "../surface-display/ui";
import { VisualLayerLegendControl, VisualLayersControl } from "../field-view/layers-ui";
import { UniformTunePanel } from "../../../components/UniformCoarseControl";

import { resolvedMethodValues } from "../../core/stores/method-store";
import { useSession } from "../../core/session/session-context";
import { SVO_FEATURE_VIEWS } from "../../svo/pipeline/ui-slots";
import { GravityRow, GravityYRow } from "../gravity/ui";
import { SimulationCostLines, SimulationMoreLines, SimulationStepLine } from "../simulation/ui";
import { InstrumentLines, PerformanceLoadLines } from "../view-rail/ui";
import { FrameCostLines, FrameMoreLines } from "../../svo/pipeline/ui";
import { composeFeatureUI } from "./composition";
import type { FeatureComposition } from "../../framework/composition";
import { ComposedFeatureSlot, type FeatureControlViews } from "../../framework/ui/slot";

/** Only application composition knows the installed React implementations. */
export const applicationViews: FeatureControlViews = {
  "presentation.surface-display/mode": SurfaceDisplayRow,
  "presentation.surface-display/particles": ParticleDisplayLines,
  "presentation.visual-layers/layers": VisualLayersControl,
  "presentation.visual-layers/legend": VisualLayerLegendControl,
  "uniform.detail-tuning/dials": UniformTunePanel,
  "scene.gravity/enabled": GravityRow,
  "scene.gravity/y": GravityYRow,
  "simulation/step": SimulationStepLine,
  "simulation/cost": SimulationCostLines,
  "simulation/more": SimulationMoreLines,
  "view.rail/frame": FrameCostLines,
  "view.rail/frame-more": FrameMoreLines,
  "view.rail/load": PerformanceLoadLines,
  "view.rail/instruments": InstrumentLines,
  ...SVO_FEATURE_VIEWS,
};

/** The installed features, as the session's method and selections resolve them. */
export function useFeatureComposition(): FeatureComposition {
  const session = useSession();
  const methodId = session.method(state => state.methodId);
  const quality = session.method(state => state.quality);
  const overrides = session.method(state => state.overrides);
  const primary = session.ui(state => state.svoPrimaryTraversal);
  const lighting = session.ui(state => state.svoConeTracingMode);
  const reconstruction = session.ui(state => state.svoRenderTuning.coneRadianceReconstruction);
  const fluid = session.scene(state => state.scene.systems?.fluid !== false);
  // Composing validates and deep-freezes the whole feature catalog, and its
  // inputs move with the method, not with the host. The scene toolstrip rides
  // the camera, so it re-renders on every orbit step: recomposing there was
  // most of that column's render cost, six slots over, at pointer rate.
  return useMemo(() => composeFeatureUI(methodId, fluid, {
    "svo.primary-visibility": primary,
    "svo.lighting-visibility": lighting,
    "svo.radiance-reconstruction": reconstruction,
  }, fluid ? resolvedMethodValues({ methodId, quality, overrides }) : {}),
  [methodId, quality, overrides, fluid, primary, lighting, reconstruction]);
}

export function FeatureSlot({ slot }: { readonly slot: string }) {
  return <ComposedFeatureSlot composition={useFeatureComposition()} views={applicationViews} slot={slot} />;
}
