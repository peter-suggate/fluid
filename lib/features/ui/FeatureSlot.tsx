"use client";

import { resolvedMethodValues } from "../../core/stores/method-store";
import { useSession } from "../../core/session/session-context";
import { adaptiveMassAdaptivityFeature } from "../../methods/adaptive-mass/features/adaptivity/definition";
import { AdaptiveMassControlRow, AdaptiveMassToolstripRow } from "../../methods/adaptive-mass/features/adaptivity/ui";
import { SVO_FEATURE_VIEWS } from "../../svo/pipeline/ui-slots";
import { GravityRow, GravityYRow } from "../gravity/ui";
import { composeFeatureUI } from "./composition";
import { ComposedFeatureSlot, type FeatureControlViews } from "../../framework/ui/slot";

/** Only application composition knows the installed React implementations. */
export const applicationViews: FeatureControlViews = {
  "scene.gravity/enabled": GravityRow,
  "scene.gravity/y": GravityYRow,
  ...Object.fromEntries(adaptiveMassAdaptivityFeature.controls!.map(control =>
    [`${adaptiveMassAdaptivityFeature.id}/${control.id}`, AdaptiveMassControlRow])),
  "simulation.adaptive-mass.adaptivity/adaptivity": AdaptiveMassToolstripRow,
  ...SVO_FEATURE_VIEWS,
};

export function FeatureSlot({ slot }: { readonly slot: string }) {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const primary = session.ui(state => state.svoPrimaryTraversal);
  const lighting = session.ui(state => state.svoConeTracingMode);
  const reconstruction = session.ui(state => state.svoRenderTuning.coneRadianceReconstruction);
  const fluid = session.scene(state => state.scene.systems?.fluid !== false);
  const selections: Record<string, string> = {
    "svo.primary-visibility": primary,
    "svo.lighting-visibility": lighting,
    "svo.radiance-reconstruction": reconstruction,
  };
  return <ComposedFeatureSlot composition={composeFeatureUI(methodId, fluid, selections, fluid ? resolvedMethodValues(methodState) : {})} views={applicationViews} slot={slot} />;
}
