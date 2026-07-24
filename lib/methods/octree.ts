import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";

const params: MethodParamSpec[] = [
  { kind: "select", key: "powerPressureSolver", label: "Adaptive pressure solver", default: "galerkin", tier: "coarse", options: [{ value: "galerkin", label: "Adaptive native-L2 Galerkin" }, { value: "section43-mgpcg", label: "Adaptive Section 4.3 MGPCG" }], hint: "Both choices solve the same adaptive, 2:1-graded power-octree frontier. The dropdown changes only the pressure algorithm; Galerkin is the default and never falls back to MGPCG." },
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "4", tier: "coarse", options: [{ value: "4", label: "4× fine band · paper default" }, { value: "8", label: "8× fine band · experimental" }], hint: "The paper tracks the interface on a separate 4× or 8× sparse narrow band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "16", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells" }], hint: "Largest dyadic octree cell away from interfaces. The topology remains strictly 2:1 graded for valid power-diagram stencils." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Pressure refinement band", unit: "finest cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Keeps the octree pressure grid fine around liquid and solid interfaces. This is distinct from the separate high-resolution surface-tracking band." }
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const globalFineLevelSetFactor = (value: unknown): 4 | 8 => Number(value) >= 8 ? 8 : 4;
const powerPressureSolver = (value: unknown): "galerkin" | "section43-mgpcg" =>
  value === "section43-mgpcg" ? "section43-mgpcg" : "galerkin";

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => ({
  densitySharpening: false,
  velocityTransport: "maccormack" as const,
  octree: {
    powerPressureSolver: powerPressureSolver(values.powerPressureSolver),
    powerBoundarySmoothingIterations: 8,
    maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 16),
    // Pressure topology is a method setting, not a solver setting. Keep the
    // same adaptive octree when comparing the two pressure implementations.
    adaptivity: 1,
    interfaceRefinementBandCells: numberValue(values, params, "interfaceRefinementBandCells"),
    globalFineLevelSetFactor: globalFineLevelSetFactor(values.globalFineLevelSetFactor),
    energyLedger: values.energyLedger === true || values.energyLedger === "on",
    energyLedgerStepCapacity: typeof values.energyLedgerStepCapacity === "number"
      ? values.energyLedgerStepCapacity : undefined,
  }
});

export const octreeMethod: SimulationMethod = {
  id: "octree",
  label: "Power-diagram octree",
  shortLabel: "Power octree",
  badge: "POWER OCTREE",
  description: "GPU-resident adaptive power cells with native-L2 Galerkin or the paper's sparse-pyramid pressure solve and the required high-resolution interface band.",
  detail: "2:1-graded adaptive pressure cells and power-face velocities, adaptive native-L2 Galerkin by default, separately selectable Section 4.3 sparse-pyramid hybrid PCG, factor-4 signed-distance narrow band, frame-lagged variational rigid-body coupling, and no topology readbacks or solver fallbacks",
  backend: "webgpu",
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  params,
  pressureMapping: "The default UI lane uses an immutable adaptive-superset native-L2 Galerkin hierarchy and maps the current adaptive liquid frontier into it on GPU. The paper's Section 4.3 hybrid MGPCG remains separately selectable. Selection is immutable and both lanes fail closed without invoking the other.",
  presetFor: () => ({
    powerPressureSolver: "galerkin",
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
