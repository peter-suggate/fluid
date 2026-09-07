import type { MethodParamValues } from "../../core/method-contract";
import { getMethod } from "../../core/method-registry";
import { composeFeatures } from "../../framework/composition";
import { SVO_PIPELINE_FEATURES } from "../../svo/pipeline/composition";
import { gravityFeature } from "../gravity/definition";

/** Installed methods supply their own features and resolved variation points. */
export function composeFeatureUI(
  methodId: string,
  fluid: boolean,
  selections: Readonly<Record<string, string>> = {},
  methodValues: MethodParamValues = {},
) {
  const method = fluid ? getMethod(methodId).resolveComposition(methodValues) : undefined;
  return composeFeatures({
    features: [
      ...SVO_PIPELINE_FEATURES,
      ...(fluid ? [gravityFeature] : []),
      ...(method?.features ?? []),
    ],
    selections: {
      ...selections,
      ...Object.fromEntries((method?.variants ?? []).map(variant => [variant.point, variant.id])),
    },
  });
}
