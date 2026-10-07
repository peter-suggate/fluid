import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import { MAC_PARAMS, macOptions } from "../mac-shared/parameters";

export const APIC_PARAMS: MethodParamSpec[] = [
  { kind: "select", key: "transferMode", label: "Particle transfer", default: "apic", options: [
    { value: "apic", label: "APIC · affine" }, { value: "pic", label: "PIC · smooth" }, { value: "flip", label: "PIC/FLIP · blended" },
  ], tier: "coarse", update: "runtime", hint: "APIC carries local velocity gradients. PIC smooths velocity; PIC/FLIP preserves particle motion using the grid velocity change." },
  { kind: "number", key: "flipRatio", label: "FLIP blend", default: 0.95, min: 0, max: 1, step: 0.01, digits: 2, unit: "", tier: "fine", update: "runtime",
    hint: "Used in PIC/FLIP mode: 0 is PIC, 1 is pure FLIP. More FLIP preserves motion but can amplify particle noise." },
  { kind: "number", key: "pressureRelativeReduction", label: "Initial residual fraction", default: 0.1, min: 0, max: 1, step: 0.01, digits: 2, unit: "×", tier: "fine", update: "runtime",
    hint: "Also reduce the initial fresh residual to this fraction, with a 0.0001 s⁻¹ floor. 0 disables the relative criterion. Uniform Geometric uses 0.1." },
  ...MAC_PARAMS.filter(param => param.key !== "advection" && param.key !== "maxStep").map(param =>
  param.key === "cfl" && param.kind === "number" ? { ...param, default: 2, max: 4, hint: "Maximum particle travel in cells per GPU substep, bounded using both velocity and the affine gradients. Larger values trade advection accuracy for longer steps." }
    : param.key === "pressureTolerance" && param.kind === "number" ? { ...param, default: 5, max: 100, step: 0.00001, digits: 5,
      hint: "Absolute ceiling for the fresh b − Ap infinity norm. Also satisfy the initial-residual fraction; the defaults match Uniform Geometric's 5 s⁻¹ ceiling and tenfold reduction." }
    // Retain headroom for tighter user-selected tolerances and impact steps.
    // GPU convergence guards stop numerical work early.
    : param.key === "pressureLimit" && param.kind === "number" ? { ...param, default: 32, min: 1, max: 128, step: 1, label: "MG-PCG iteration limit" }
      : param),
];

/** Stable GPU mode codes shared with the particle settings uniform. */
export function particleTransferOptions(values: MethodParamValues = {}) {
  return {
    transferMode: values.transferMode === "pic" ? 1 : values.transferMode === "flip" ? 2 : 0,
    flipRatio: numberValue(values, APIC_PARAMS, "flipRatio"),
  };
}
/** One user-facing dt: the shared transport control. GPU stability limits can
 * split it, but a hidden method cap must not silently override an edit. */
export function apicOptions(scene: SceneDescription, values: MethodParamValues) {
  return { ...macOptions(values), tolerance: numberValue(values, APIC_PARAMS, "pressureTolerance"),
    relativeReduction: numberValue(values, APIC_PARAMS, "pressureRelativeReduction"), cfl: numberValue(values, APIC_PARAMS, "cfl"), pressureLimit: Math.round(numberValue(values, APIC_PARAMS, "pressureLimit")),
    maxStep: Math.max(0.0001, Math.min(0.05, scene.numerics.fixedDt_s)) };
}
export function validateApicScene(scene: SceneDescription): void {
  if (scene.rigidBodies.length) throw new Error("APIC particles currently supports fixed voxel solids, not rigid bodies. Try Minimal dam break 32³ or a scene without bodies.");
  if (scene.fluid.inflow && scene.fluid.inflow.enabled !== false)
    throw new Error("APIC particles does not yet support inflows. Disable the inflow or select a dam-break scene.");
}
