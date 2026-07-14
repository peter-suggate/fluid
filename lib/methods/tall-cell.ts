import { WebGPUEulerianSolver } from "../webgpu-eulerian";
import { tallCellSettings } from "../tall-cell-grid";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureCycles", label: "Pressure V-cycles", unit: "cycles", min: 1, max: 12, step: 1, digits: 0, default: 2, tier: "coarse", hint: "Multigrid refinement cycles after the initial full cycle. More cycles tighten divergence at impacts." },
  { kind: "number", key: "surfaceColumns", label: "Surface columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Target x/z column count; sets horizontal resolution." },
  { kind: "number", key: "regularLayers", label: "Surface band layers", unit: "cells", min: 12, max: 64, step: 4, digits: 0, default: 24, tier: "fine", hint: "Cubic cells kept around the free surface. The band grows automatically if the surface spans more." },
  { kind: "number", key: "maximumNeighborDelta", label: "Neighbor base delta", unit: "cells", min: 2, max: 6, step: 1, digits: 0, default: 4, tier: "fine", hint: "Maximum tall-cell base step between adjacent columns." },
  { kind: "number", key: "remeshInterval", label: "Remesh interval", unit: "steps", min: 20, max: 240, step: 10, digits: 0, default: 60, tier: "fine", hint: "Steps between tall-cell band re-planning passes." }
];

export const tallCellMethod: SimulationMethod = {
  id: "tall-cell",
  label: "Tall-cell conservative VOF",
  shortLabel: "Tall cells",
  badge: "TALL CELLS",
  description: "Restricted fixed-band tall-cell grid: cubic cells at the surface, one variable-height cell below.",
  detail: "one variable-height bottom cell per x/z column plus a fixed moving band of cubic surface cells, pairwise conservative VOF face transport, bounded MacCormack velocity advection, narrow-band velocity extrapolation, and a restricted full-cycle multigrid pressure solve",
  backend: "webgpu",
  qualityLabels: { balanced: "~2.5k columns · 24 layers", high: "~7k columns · 32 layers", ultra: "~12.5k columns · 40 layers" },
  params,
  pressureMapping: "Pressure accuracy scales the multigrid refinement V-cycle count (1 full cycle + N V-cycles, RBGS smoothing).",
  presetFor: (quality) => {
    const preset = tallCellSettings[quality];
    return { pressureCycles: 2, surfaceColumns: preset.surfaceColumns, regularLayers: preset.regularLayers, maximumNeighborDelta: preset.maximumNeighborDelta, remeshInterval: preset.remeshInterval };
  },
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUEulerianSolver(device, scene, quality, onRigidLoads, {
    pressureCycles: numberValue(values, params, "pressureCycles"),
    tallCellSettings: {
      surfaceColumns: numberValue(values, params, "surfaceColumns"),
      regularLayers: numberValue(values, params, "regularLayers"),
      maximumNeighborDelta: numberValue(values, params, "maximumNeighborDelta"),
      remeshInterval: numberValue(values, params, "remeshInterval")
    }
  })
};
