import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";

const params: MethodParamSpec[] = [
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "4", tier: "coarse", options: [{ value: "1", label: "Coarse octree only · no fine band" }, { value: "4", label: "4× fine band · paper default" }, { value: "8", label: "8× fine band · experimental" }], hint: "Factor 1 transports φ directly on the adaptive octree and allocates no separate fine-band grid. Factors 4/8 maintain the sparse higher-resolution interface band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "32", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells · default" }], hint: "Largest dyadic octree cell away from interfaces. Every authored scene defaults to 32, and the topology remains strictly 2:1 graded for valid power-diagram stencils." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Band reach", unit: "level", min: 0, max: 4, step: 1, digits: 0, default: 4, tier: "fine", hint: "One coupled reach level for pressure refinement and Section 5 surface tracking. Experimental level 0 uses one fine brick; level 1 retains the two-finest-cell moving-surface floor while still reducing pressure reach and recurring residency. Level 4 is the paper/default reach." },
  { kind: "number", key: "surfaceRefinementGradingLayers", label: "Surface grading", unit: "layers", min: 1, max: 4, step: 1, digits: 0, default: 1, tier: "fine", hint: "Intermediate pressure-cell layers retained per octree level around the surface. 1 is the existing sharp 2:1 transition; 3 is the progressive-refinement experiment." },
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 32;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const globalFineLevelSetFactor = (value: unknown): 1 | 4 | 8 => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 4;
  return numeric === 1 ? 1 : numeric >= 8 ? 8 : 4;
};

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => {
  const bandReachCells = numberValue(values, params, "interfaceRefinementBandCells");
  const fineFactor = globalFineLevelSetFactor(values.globalFineLevelSetFactor);
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
      maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 32),
      // Dry boundary adaptivity is the production pressure policy at every
      // surface resolution. False remains the exact Dawn control.
      fluidGatedBoundaryRefinement: values.fluidGatedBoundaryRefinement !== false,
      environmentBrickRefinementLevels: typeof values.svoEnvironmentBrickRefinementLevels === "number"
        ? values.svoEnvironmentBrickRefinementLevels : undefined,
      // Pressure topology is a method setting, not a solver setting. Keep the
      // same adaptive octree when comparing the two pressure implementations.
      adaptivity: typeof values.octreeAdaptivity === "number"
        && Number.isFinite(values.octreeAdaptivity)
        ? Math.max(0, Math.min(1, values.octreeAdaptivity))
        : 1,
      interfaceRefinementBandCells: bandReachCells,
      surfaceRefinementGradingLayers: numberValue(values, params, "surfaceRefinementGradingLayers"),
      fineLevelSetBandCells: diagnosticFineBand,
      globalFineLevelSetFactor: fineFactor,
      // Hidden authored/harness override for scenes whose fluid footprint is
      // intentionally much smaller than the container. The generic fallback
      // remains the conservative domain-cross-section estimate.
      globalFineLevelSetMaximumBricks:
        typeof values.globalFineLevelSetMaximumBricks === "number"
          && Number.isSafeInteger(values.globalFineLevelSetMaximumBricks)
          && values.globalFineLevelSetMaximumBricks > 0
          ? values.globalFineLevelSetMaximumBricks : undefined,
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
  description: "GPU-resident adaptive power cells with persistent MGPCG pressure and selectable compact-coarse or fine-band surface tracking.",
  detail: "2:1-graded adaptive pressure cells and six structured velocity families, one persistent Section 4.3 fixed-schedule MGPCG authority, factor-4 signed-distance narrow band, frame-lagged variational rigid-body coupling, and no topology readbacks or alternate solver path",
  backend: "webgpu",
  resource: {
    id: "fluid.power-octree",
    lane: "fluid",
    label: "Power-octree fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Determining which solver resources can be reused.",
      drain: "Fencing the previous simulation generation before a reset.",
      allocation: "Allocating solver-owned buffers and textures.",
      "solver-pipelines": "Compiling the simulation programs this method can dispatch.",
      "adaptive-topology": "Preparing adaptive topology and pressure programs.",
      "secondary-particles": "Preparing optional secondary-liquid programs.",
      upload: "Uploading the authored t=0 fields.",
      warmup: "Publishing and fencing authoritative t=0 state.",
      attach: "Attaching the warmed generation atomically.",
      presentation: "Publishing the first fenced water surface.",
    },
  },
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  params,
  pressureMapping: "The production lane uses the matrix-free persistent Section 4.3 fixed-schedule MGPCG authority over the current adaptive liquid frontier. Failure is terminal for the publication; no alternate pressure solver is retained.",
  presetFor: () => ({
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    surfaceRefinementGradingLayers: 1,
    globalFineLevelSetFactor: "4",
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "planning" || phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
