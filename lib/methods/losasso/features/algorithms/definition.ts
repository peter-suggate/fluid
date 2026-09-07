import type { MethodParamSpec } from "../../../../core/method-contract";
import { parameterVariantFeature } from "../../../../core/method-parameter-variants";

export const ALGORITHM_PARAMS: MethodParamSpec[] = [
  { kind: "select", key: "losassoVelocityExtension", label: "Losasso extrapolation", default: "fixed-jacobi", tier: "fine", options: [{ value: "fixed-jacobi", label: "Fixed Jacobi · default" }, { value: "causal-front", label: "Causal layer front" }], hint: "Construction-time A/B control for Section 5 air velocity extension. Causal-front publishes one graph layer per sweep from already-valid inner layers." },
];

export const algorithmFeature = parameterVariantFeature("simulation.losasso.algorithms", ALGORITHM_PARAMS, ["simulation.octree"]);
