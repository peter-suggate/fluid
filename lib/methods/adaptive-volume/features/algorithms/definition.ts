import type { MethodParamSpec } from "../../../../core/method-contract";
import { parameterVariantFeature } from "../../../../core/method-parameter-variants";

export const ALGORITHM_PARAMS: MethodParamSpec[] = [
  {
    kind: "select",
    key: "timeStep",
    label: "Time step",
    default: "paper",
    tier: "coarse",
    options: [
      { value: "paper", label: "Paper · 1/30 s large steps" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    update: "runtime",
    hint: "Sparse CM12 defaults to the exact 1/30 s paper step. Scene mode remains available for matched-step validation and hydrostatic probes.",
  },
  {
    kind: "select",
    key: "gammaDiffusion",
    label: "Gamma diffusion",
    default: "on",
    tier: "coarse",
    update: "runtime",
    options: [
      { value: "on", label: "On" },
      { value: "off", label: "Off" },
    ],
    hint: "Enables CM12 Sec. 3.4's configured conservative gamma-diffusion passes. Turning it off keeps conservative transport and the sparse scalar-publication chain active.",
  },
  {
    kind: "select",
    key: "surfaceSharpening",
    label: "Surface sharpening",
    default: "on",
    tier: "coarse",
    update: "runtime",
    options: [
      { value: "on", label: "On" },
      { value: "off", label: "Off" },
    ],
    hint: "Enables CM12 Sec. 3.5 Algorithm 2 mass return. Turning it off still publishes the final sparse scalar masks required by pressure, topology, and presentation.",
  },
];

export const algorithmFeature = parameterVariantFeature("simulation.adaptive-volume.algorithms", ALGORITHM_PARAMS, ["simulation.sparse-atlas"]);
