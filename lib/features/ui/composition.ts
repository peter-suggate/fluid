import { surfaceDisplayFeature } from "../surface-display/definition";
import type { MethodParamValues } from "../../core/method-contract";
import { getMethod } from "../../core/method-registry";
import { composeFeatures } from "../../framework/composition";
import { SVO_PIPELINE_FEATURES } from "../../svo/pipeline/composition";
import { gravityFeature } from "../gravity/definition";
import { simulationFeature } from "../simulation/definition";
import { viewRailFeature } from "../view-rail/definition";
import { visualLayersFeature } from "../field-view/definition";

/** Installed methods supply their own features and resolved variation points. */
export function composeFeatureUI(
  methodId: string,
  fluid: boolean,
  selections: Readonly<Record<string, string>> = {},
  methodValues: MethodParamValues = {},
) {
  const method = fluid ? getMethod(methodId).resolveComposition(methodValues) : undefined;
  // Only a method that publishes its field views as composed layers has a layer card to open.
  const layers = fluid && getMethod(methodId).capabilities?.visualLayers !== undefined;
  return composeFeatures({
    features: [
      viewRailFeature,
      ...SVO_PIPELINE_FEATURES,
      ...(fluid ? [simulationFeature, gravityFeature, surfaceDisplayFeature] : []),
      ...(layers ? [visualLayersFeature] : []),
      ...(method?.features ?? []),
    ],
    selections: {
      ...selections,
      ...Object.fromEntries((method?.variants ?? []).map(variant => [variant.point, variant.id])),
    },
  });
}
