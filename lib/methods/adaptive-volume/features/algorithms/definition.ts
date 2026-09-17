import type { FluidStageControl } from "../../../../core/fluid-pipeline";
import type { MethodParamSpec } from "../../../../core/method-contract";
import { parameterVariantFeature } from "../../../../core/method-parameter-variants";
import { ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS, ADAPTIVE_VOLUME_RETURN_ROUNDS, SPARSE_CM12_SHARPENING_STRENGTH } from "../../sharpening-controls";

export const ALGORITHM_PARAMS: MethodParamSpec[] = [
  {
    kind: "select", key: "airExtension", label: "Air-band velocity correction",
    default: "on", tier: "coarse", update: "runtime",
    options: [{ value: "off", label: "Off" }, { value: "on", label: "On" }],
    hint: "Uses direct face transport with an air-band correction and preserves the face field across remeshing for momentum. Adds an iterative GPU solve and snapshot storage. Compare runs from the same reset state.",
  },
  {
    kind: "select", key: "timeStep", label: "Time step", default: "paper",
    tier: "coarse", update: "runtime",
    options: [
      { value: "paper", label: "Fixed · 1/30 s" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    hint: "Choose the outer simulation step. Geometric volume transport divides it into synchronized internal steps according to the face-flux CFL condition.",
  },
  {
    kind: "select", key: "surfaceSharpening", label: "Volume sharpening",
    default: "on", tier: "coarse", update: "runtime",
    options: [
      { value: "on", label: "On" },
      { value: "off", label: "Off" },
    ],
    hint: "Redistributes liquid volume inward toward the accepted level set at the end of conservative transport, countering the interface smearing that advection accumulates. Off disables local sharpening and far-volume return; transport, pressure, topology and presentation are otherwise unchanged. Expect a softer, more diffuse surface with it off.",
  },
];

/**
 * Numeric tuning that belongs to the same stage but is deliberately not a
 * composition variant — `parameterVariantFeature` only adapts choice schemas.
 */
export const ALGORITHM_TUNING_PARAMS: MethodParamSpec[] = [
  {
    kind: "number", key: "distanceSweeps", label: "Adaptive distance sweeps",
    default: ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS, tier: "coarse", update: "runtime",
    unit: "", min: 0, max: 16, step: 1, digits: 0,
    hint: "Each sweep propagates distance across two accepted-cell neighbours, up to eight fine-cell widths. Zero disables far-volume return.",
  },
  {
    kind: "number", key: "returnPasses", label: "Far-volume return passes",
    default: ADAPTIVE_VOLUME_RETURN_ROUNDS, tier: "coarse", update: "runtime",
    unit: "", min: 0, max: 16, step: 1, digits: 0,
    hint: "Each conservative pass moves surplus volume one neighbour toward its surface patch. Zero keeps only local sharpening.",
  },
  {
    kind: "number", key: "sharpeningStrength", label: "Sharpening strength",
    default: SPARSE_CM12_SHARPENING_STRENGTH, tier: "coarse", update: "runtime",
    unit: "×", min: 0, max: 1, step: 0.05, digits: 2,
    hint: "Fraction of each cell's computed sharpening dose actually applied. 1 is the full CM12 Sec. 3.5 return; lower values sharpen more gently over more frames. Zero is equivalent to turning sharpening off — the dispatches still run but move no mass.",
  },
];

export const algorithmFeature = parameterVariantFeature("simulation.adaptive-volume.algorithms", ALGORITHM_PARAMS, ["simulation.sparse-atlas"]);

/**
 * Surfaces an algorithm parameter beside the stage that dispatches it.
 *
 * Numeric sharpening controls follow the master toggle.
 */
export function algorithmStageControl(key: string): FluidStageControl {
  const spec = [...ALGORITHM_PARAMS, ...ALGORITHM_TUNING_PARAMS].find(p => p.key === key);
  if (!spec) throw new Error(`Unknown algorithm control: ${key}`);
  if (spec.kind === "select") {
    return {kind:"param-choice",param:key,label:spec.label,hint:spec.hint,options:spec.options};
  }
  return {kind:"param-range",param:key,label:spec.label,hint:spec.hint,
    unit:spec.unit ? ` ${spec.unit}` : "",min:spec.min ?? 0,max:spec.max ?? 1,
    step:spec.step ?? 0.01,digits:spec.digits,
    ...(["sharpeningStrength", "distanceSweeps", "returnPasses"].includes(key)
      ? {enabled: context => context.values.surfaceSharpening !== "off"} : {})};
}
