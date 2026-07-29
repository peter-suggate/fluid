import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";

const params: MethodParamSpec[] = [
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "4", tier: "coarse", options: [{ value: "4", label: "4× fine band · paper default" }, { value: "8", label: "8× fine band · experimental" }], hint: "The paper tracks the interface on a separate 4× or 8× sparse narrow band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "16", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells" }], hint: "Largest dyadic octree cell away from interfaces. The topology remains strictly 2:1 graded for valid power-diagram stencils." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Band reach", unit: "level", min: 0, max: 4, step: 1, digits: 0, default: 4, tier: "fine", hint: "One coupled reach level for pressure refinement and Section 5 surface tracking. Experimental level 0 uses one fine brick; level 1 retains the two-finest-cell moving-surface floor while still reducing pressure reach and recurring residency. Level 4 is the paper/default reach." },
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const globalFineLevelSetFactor = (value: unknown): 4 | 8 => Number(value) >= 8 ? 8 : 4;

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => {
  const bandReachCells = numberValue(values, params, "interfaceRefinementBandCells");
  // Not a product control. Keep the fine-only override available to the Dawn
  // harness for fault injection and planner isolation without letting normal
  // configurations drift into the restriction-starvation pairing.
  const diagnosticFineBand = typeof values.fineLevelSetBandCells === "number"
    && Number.isFinite(values.fineLevelSetBandCells)
    ? Math.max(0, Math.min(32, Math.round(values.fineLevelSetBandCells)))
    : bandReachCells;
  return {
    densitySharpening: false,
    velocityTransport: "maccormack" as const,
    octree: {
      maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 16),
      // Pressure topology is a method setting, not a solver setting. Keep the
      // same adaptive octree when comparing the two pressure implementations.
      adaptivity: 1,
      interfaceRefinementBandCells: bandReachCells,
      fineLevelSetBandCells: diagnosticFineBand,
      globalFineLevelSetFactor: globalFineLevelSetFactor(values.globalFineLevelSetFactor),
      // Diagnostic-only capacity override. Not an authored UI parameter: the
      // production capacity comes from `planOctreePressureCapacity`, and this
      // exists so a harness can bisect the arena demand of a large domain
      // against the on-GPU overflow report.
      pressureRowCapacity: typeof values.pressureRowCapacity === "number" && values.pressureRowCapacity > 0
        ? values.pressureRowCapacity : undefined,
    }
  };
};

export const octreeMethod: SimulationMethod = {
  id: "octree",
  label: "Power-diagram octree",
  shortLabel: "Power octree",
  badge: "POWER OCTREE",
  description: "GPU-resident adaptive power cells with the sparse-pyramid hybrid MGPCG pressure solve and a high-resolution interface band.",
  detail: "2:1-graded adaptive pressure cells and six structured velocity families, one production Section 4.3 hybrid MGPCG authority, factor-4 signed-distance narrow band, frame-lagged variational rigid-body coupling, and no topology readbacks or alternate solver path",
  backend: "webgpu",
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  params,
  pressureMapping: "The production lane uses the matrix-free Section 4.3 hybrid MGPCG authority over the current adaptive liquid frontier. Failure is terminal for the publication; no alternate pressure solver is retained.",
  presetFor: () => ({
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "4",
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
