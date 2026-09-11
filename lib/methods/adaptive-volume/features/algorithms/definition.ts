import type { MethodParamSpec } from "../../../../core/method-contract";
import { parameterVariantFeature } from "../../../../core/method-parameter-variants";

export const ALGORITHM_PARAMS: MethodParamSpec[] = [
  {
    kind: "select", key: "timeStep", label: "Time step", default: "paper",
    tier: "coarse", update: "runtime",
    options: [
      { value: "paper", label: "Fixed · 1/30 s" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    hint: "Choose the outer simulation step. Geometric volume transport divides it into synchronized internal steps according to the face-flux CFL condition.",
  },
];

export const algorithmFeature = parameterVariantFeature("simulation.adaptive-volume.algorithms", ALGORITHM_PARAMS, ["simulation.sparse-atlas"]);
