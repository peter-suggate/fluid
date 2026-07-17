import { WebGPUEulerianSolver } from "../webgpu-eulerian";
import { createTallCellLayout, tallCellSettings } from "../tall-cell-grid";
import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "select", key: "velocityTransport", label: "Velocity transport", default: "maccormack", tier: "coarse", options: [{ value: "maccormack", label: "Bounded MacCormack" }, { value: "semi-lagrangian", label: "Semi-Lagrangian" }], hint: "Velocity may use bounded MacCormack; the level set always follows the paper's semi-Lagrangian transport." },
  { kind: "number", key: "pressureCycles", label: "Pressure V-cycles", unit: "cycles", min: 1, max: 12, step: 1, digits: 0, default: 8, tier: "coarse", hint: "Multigrid refinement cycles after the initial full cycle. Eight cycles keep remeshed dam-break pressure residuals below the settling-energy threshold; lower values are diagnostic/performance tradeoffs." },
  { kind: "select", key: "pressureWarmStart", label: "Pressure warm start", default: "off", tier: "coarse", options: [{ value: "off", label: "Off (cold FMG)" }, { value: "on", label: "On (experimental)" }], hint: "Reuse and remap the previous pressure field, then solve with V-cycles. The cold default retains tighter impact convergence." },
  { kind: "select", key: "pressureDefectCorrection", label: "Pressure defect correction", default: "off", tier: "coarse", options: [{ value: "off", label: "Off (default)" }, { value: "on", label: "On (second projection)" }], hint: "Rebuild divergence after projection and run a second pressure correction. This targets the projected residual at the cost of a second pressure solve." },
  { kind: "number", key: "surfaceColumns", label: "Surface columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Target x/z column count; sets horizontal resolution." },
  { kind: "number", key: "regularLayers", label: "Surface band layers", unit: "cells", min: 12, max: 64, step: 4, digits: 0, default: 24, tier: "fine", hint: "Cubic cells kept around the free surface. The band grows automatically if the surface spans more." },
  { kind: "number", key: "maximumNeighborDelta", label: "Neighbor base delta", unit: "cells", min: 2, max: 6, step: 1, digits: 0, default: 4, tier: "fine", hint: "Maximum tall-cell base step between adjacent columns." },
  { kind: "number", key: "maximumTallHeight", label: "Maximum tall height", unit: "cells", min: 3, max: 4096, step: 1, digits: 0, default: 4096, tier: "fine", hint: "Diagnostic ceiling on tall-cell height. Small values force near-cubic parity (Section 5 middle-face isolation); the default leaves remeshing free to claim the full depth below the surface band." },
  { kind: "number", key: "remeshInterval", label: "Remesh interval", unit: "steps", min: 1, max: 60, step: 1, digits: 0, default: 1, tier: "fine", hint: "Paper Algorithm 1 remeshes after advection on every step; larger values are diagnostic departures." },
  { kind: "select", key: "volumeControl", label: "Volume control", default: "on", tier: "fine", options: [{ value: "on", label: "On" }, { value: "off", label: "Off (diagnostic)" }], hint: "Slow global normal level-set correction for the tall-cell paper's acknowledged water gain; disable to measure free drift." }
];

export const tallCellMethod: SimulationMethod = {
  id: "tall-cell",
  label: "Tall-cell level set",
  shortLabel: "Tall cells",
  badge: "TALL CELLS",
  description: "Restricted fixed-band tall-cell grid: cubic cells at the surface, one variable-height cell below.",
  detail: "one variable-height bottom cell per x/z column plus a moving cubic surface band, signed-distance surface tracking, bounded MacCormack velocity transport, semi-Lagrangian level-set transport, and a restricted full-cycle multigrid pressure solve",
  backend: "webgpu",
  qualityLabels: { balanced: "~2.5k columns · h≤3 parity", high: "~7k columns · h≤3 parity", ultra: "~12.5k columns · h≤3 parity" },
  params,
  pressureMapping: "Pressure accuracy scales the multigrid V-cycle count; experimental warm start is available for comparison.",
  presetFor: (quality) => {
    const preset = tallCellSettings[quality];
    return { pressureCycles: 8, pressureWarmStart: "off", pressureDefectCorrection: "off", surfaceColumns: preset.surfaceColumns, regularLayers: preset.regularLayers, maximumNeighborDelta: preset.maximumNeighborDelta, maximumTallHeight: preset.maximumTallHeight, remeshInterval: preset.remeshInterval };
  },
  createSolver: (device, scene, quality, values, onRigidLoads) => {
    const velocityTransport = values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack";
    const settings = {
      surfaceColumns: numberValue(values, params, "surfaceColumns"),
      regularLayers: numberValue(values, params, "regularLayers"),
      maximumNeighborDelta: numberValue(values, params, "maximumNeighborDelta"),
      maximumTallHeight: numberValue(values, params, "maximumTallHeight"),
      remeshInterval: numberValue(values, params, "remeshInterval")
    };
    const layout = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, settings);
    // Use the cubic backend only when h >= 2 is geometrically impossible.
    // Otherwise retain dynamic remeshing even if the initial layout happens
    // to be near the ordinary-cell limit; later frames may expose tall cells.
    if (layout.planning.maximumBaseBeforeOrdinaryFallback < 2) return new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, { velocityTransport, tallCellSettings: settings });
    return new WebGPUEulerianSolver(device, scene, quality, onRigidLoads, {
      pressureCycles: numberValue(values, params, "pressureCycles"),
      pressureWarmStart: values.pressureWarmStart !== "off",
      pressureDefectCorrection: values.pressureDefectCorrection !== "off",
      velocityTransport,
      volumeControl: values.volumeControl !== "off",
      referenceVolumeScale: typeof values.referenceVolumeScale === "number" ? values.referenceVolumeScale : undefined,
      hierarchicalExtrapolation: values.hierarchicalExtrapolation !== "off",
      tallCellSettings: settings
    });
  },
  createSolverAsync: async (device, scene, quality, values, onRigidLoads, onProgress) => {
    const velocityTransport = values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack";
    const settings = {
      surfaceColumns: numberValue(values, params, "surfaceColumns"),
      regularLayers: numberValue(values, params, "regularLayers"),
      maximumNeighborDelta: numberValue(values, params, "maximumNeighborDelta"),
      maximumTallHeight: numberValue(values, params, "maximumTallHeight"),
      remeshInterval: numberValue(values, params, "remeshInterval")
    };
    const layout = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, settings);
    if (layout.planning.maximumBaseBeforeOrdinaryFallback < 2) {
      return WebGPUUniformEulerianSolver.createAsync(device, scene, quality, onRigidLoads, { velocityTransport, tallCellSettings: settings },
        (label, completed, total) => onProgress({ phase: "solver-pipelines", label: `Uniform fallback · ${label}`, completed, total }));
    }
    return WebGPUEulerianSolver.createAsync(device, scene, quality, onRigidLoads, {
      pressureCycles: numberValue(values, params, "pressureCycles"), pressureWarmStart: values.pressureWarmStart !== "off", pressureDefectCorrection: values.pressureDefectCorrection !== "off", velocityTransport,
      volumeControl: values.volumeControl !== "off",
      referenceVolumeScale: typeof values.referenceVolumeScale === "number" ? values.referenceVolumeScale : undefined,
      hierarchicalExtrapolation: values.hierarchicalExtrapolation !== "off",
      tallCellSettings: settings
    }, (label, completed, total) => onProgress({ phase: "solver-pipelines", label, completed, total }));
  }
};
