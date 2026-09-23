"use client";
import { useMemo } from "react";
import { PressureInspectionRow } from "../pressure-inspection/ui";
import { SurfaceDisplayRow } from "../surface-display/ui";
import { TopologyFreezeRow } from "../topology-freeze/ui";

import { resolvedMethodValues } from "../../core/stores/method-store";
import { useSession } from "../../core/session/session-context";
import { adaptiveMassAdaptivityFeature } from "../../methods/adaptive-mass/features/adaptivity/definition";
import { AdaptiveMassControlRow, AdaptiveMassToolstripRow } from "../../methods/adaptive-mass/features/adaptivity/ui";
import { adaptiveMassAdaptivityFeature as adaptiveVolumeAdaptivityFeature } from "../../methods/adaptive-volume/features/adaptivity/definition";
import { AdaptiveMassControlRow as AdaptiveVolumeControlRow, AdaptiveMassToolstripRow as AdaptiveVolumeToolstripRow } from "../../methods/adaptive-volume/features/adaptivity/ui";
import { SVO_FEATURE_VIEWS } from "../../svo/pipeline/ui-slots";
import { GravityRow, GravityYRow } from "../gravity/ui";
import { composeFeatureUI } from "./composition";
import { ComposedFeatureSlot, type FeatureControlViews } from "../../framework/ui/slot";

/** Only application composition knows the installed React implementations. */
export const applicationViews: FeatureControlViews = {
  "simulation.pressure-inspection/film": PressureInspectionRow,
  "presentation.surface-display/mode": SurfaceDisplayRow,
  "simulation.topology-freeze/enabled": TopologyFreezeRow,
  "scene.gravity/enabled": GravityRow,
  "scene.gravity/y": GravityYRow,
  ...Object.fromEntries(adaptiveMassAdaptivityFeature.controls!.map(control =>
    [`${adaptiveMassAdaptivityFeature.id}/${control.id}`, AdaptiveMassControlRow])),
  "simulation.adaptive-mass.adaptivity/adaptivity": AdaptiveMassToolstripRow,
  ...Object.fromEntries(adaptiveVolumeAdaptivityFeature.controls!.map(control =>
    [`${adaptiveVolumeAdaptivityFeature.id}/${control.id}`, AdaptiveVolumeControlRow])),
  "simulation.adaptive-volume.adaptivity/adaptivity": AdaptiveVolumeToolstripRow,
  ...SVO_FEATURE_VIEWS,
};

export function FeatureSlot({ slot }: { readonly slot: string }) {
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
  const composition = useMemo(() => composeFeatureUI(methodId, fluid, {
    "svo.primary-visibility": primary,
    "svo.lighting-visibility": lighting,
    "svo.radiance-reconstruction": reconstruction,
  }, fluid ? resolvedMethodValues({ methodId, quality, overrides }) : {}),
  [methodId, quality, overrides, fluid, primary, lighting, reconstruction]);
  return <ComposedFeatureSlot composition={composition} views={applicationViews} slot={slot} />;
}
