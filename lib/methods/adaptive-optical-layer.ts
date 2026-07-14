import { WebGPUAdaptiveOpticalLayerSolver } from "../webgpu-adaptive-optical-solver";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureCycles", label: "Pressure V-cycles", unit: "cycles", min: 2, max: 16, step: 1, digits: 0, default: 8, tier: "coarse", hint: "Multigrid refinement cycles. The adaptive layout needs more cycles than the fixed band to converge across dilation changes." },
  { kind: "number", key: "surfaceColumns", label: "Surface columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Target x/z column count; sets horizontal resolution. Layer dilation is planned per step from Eq. 1." }
];

export const adaptiveOpticalLayerMethod: SimulationMethod = {
  id: "adaptive-optical-layer",
  label: "Adaptive conservative VOF",
  shortLabel: "Adaptive",
  badge: "ADAPTIVE LAYER",
  description: "Narita–Kanai adaptive optical layer: velocity-error-controlled dilation of the fine band.",
  detail: "the paper's per-column velocity fit, Eq. 1 error-controlled dilation, exact variable-radius Manhattan propagation, constrained 9×9 smoothing, pairwise conservative VOF face transport, remapped tall endpoints, and restricted multigrid projection",
  backend: "webgpu",
  qualityLabels: { balanced: "adaptive d = 4…Ny/8", high: "adaptive d = 4…Ny/8", ultra: "adaptive d = 4…Ny/8" },
  params,
  pressureMapping: "Pressure accuracy scales the multigrid refinement V-cycle count. Layer dilation bounds derive from Eq. 1 and grid height; they are planned per step, not user-set.",
  presetFor: (quality) => ({ pressureCycles: 8, surfaceColumns: quality === "balanced" ? 2_500 : quality === "high" ? 7_000 : 12_500 }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUAdaptiveOpticalLayerSolver(device, scene, quality, onRigidLoads, {
    pressureCycles: numberValue(values, params, "pressureCycles"),
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") }
  })
};
