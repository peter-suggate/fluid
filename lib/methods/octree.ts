import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";
import {
  DEFAULT_OCTREE_COARSE_BACKEND,
  resolveOctreeCoarseDynamics,
} from "../octree-coarse-backend";
import {
  OCTREE_RUNTIME_DIALS,
  OCTREE_RUNTIME_DIAL_KEYS,
} from "../octree-runtime-dials";

/**
 * The live accuracy/frame-time dials, as ordinary method parameters.
 *
 * They are declared from one schema (`lib/octree-runtime-dials.ts`) so the
 * performance strip, the solver, and the URL state cannot disagree about a
 * range or a default. `update: "runtime"` is what keeps a drag off the
 * structural fingerprint: the controller applies the value to the attached
 * solver instead of resetting to t=0.
 */
const runtimeDialParams: MethodParamSpec[] = OCTREE_RUNTIME_DIALS.map((dial) => ({
  kind: "number",
  key: dial.key,
  label: dial.label,
  unit: dial.unit,
  min: dial.min,
  max: dial.max,
  step: dial.step,
  digits: dial.digits,
  default: dial.default,
  tier: "fine",
  update: "runtime",
  hint: dial.hint,
}));

const params: MethodParamSpec[] = [
  { kind: "select", key: "coarseBackend", label: "Coarse dynamics", default: DEFAULT_OCTREE_COARSE_BACKEND, tier: "coarse", options: [{ value: "losasso", label: "Losasso 2004 · default" }, { value: "power2017", label: "Power 2017 · frozen reference" }], hint: "Construction-time backend choice. Each backend owns distinct pipelines, layouts, and velocity channels; the frozen Power path remains available for reference lanes." },
  { kind: "select", key: "losassoFreeSurfacePressure", label: "Losasso air pressure", default: "subcell-contact", tier: "fine", options: [{ value: "subcell-contact", label: "Subcell ghost + contact · default" }, { value: "cell-centered-air", label: "Losasso 2004 cell-centred air" }], hint: "Construction-time A/B control. The paper mode uses a cell-centred p_air=0 neighbor and closed-wall Neumann faces; the production extension retains subcell theta and unilateral overhead separation." },
  { kind: "select", key: "losassoVelocityExtension", label: "Losasso extrapolation", default: "fixed-jacobi", tier: "fine", options: [{ value: "fixed-jacobi", label: "Fixed Jacobi · default" }, { value: "causal-front", label: "Causal layer front" }], hint: "Construction-time A/B control for Section 5 air velocity extension. Causal-front publishes one graph layer per sweep from already-valid inner layers." },
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "1", tier: "coarse", options: [{ value: "1", label: "Coarse octree only · default" }, { value: "4", label: "4× fine band" }, { value: "8", label: "8× fine band · experimental" }], hint: "Factor 1 transports φ on the complete coarse lattice, restricts it onto adaptive octree rows, and allocates no fine-band pages. Factors 4/8 opt into the sparse higher-resolution interface band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "32", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells · default" }], hint: "Largest dyadic octree cell away from interfaces. Scene profiles choose the largest compatible root while preserving strict 2:1 grading." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Band reach", unit: "level", min: 0, max: 4, step: 1, digits: 0, default: 4, tier: "fine", hint: "One coupled reach level for pressure refinement and Section 5 surface tracking. Experimental level 0 uses one fine brick; level 1 retains the two-finest-cell moving-surface floor while still reducing pressure reach and recurring residency. Level 4 is the paper/default reach. This is the STRUCTURAL band: it sizes the pressure row capacity, the residency halo, the candidate dilation rings and the redistance reach, so changing it rebuilds. The performance strip carries a live Surface band dial that thins this one without a rebuild, and is clamped to it." },
  { kind: "number", key: "surfaceRefinementGradingLayers", label: "Surface grading", unit: "layers", min: 1, max: 4, step: 1, digits: 0, default: 1, tier: "fine", hint: "Intermediate pressure-cell layers retained per octree level around the surface. 1 is the existing sharp 2:1 transition; 3 is the progressive-refinement experiment." },
  { kind: "number", key: "topologyCadenceAdvances", label: "Topology cadence", unit: "advances", min: 1, max: 8, step: 1, digits: 0, default: 1, tier: "fine", hint: "Losasso candidate epochs may cover k accepted advances. Each skipped rebuild is represented spatially by an extra dilation ring; band 4 remains canonical." },
  ...runtimeDialParams,
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 32;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const globalFineLevelSetFactor = (value: unknown): 1 | 4 | 8 => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 1;
  return numeric === 1 ? 1 : numeric >= 8 ? 8 : 4;
};

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => {
  const bandReachCells = numberValue(values, params, "interfaceRefinementBandCells");
  const fineFactor = globalFineLevelSetFactor(values.globalFineLevelSetFactor);
  const coarseDynamics = resolveOctreeCoarseDynamics({
    backend: values.coarseBackend,
    globalFineLevelSetFactor: fineFactor,
    topologyCadenceAdvances: values.topologyCadenceAdvances,
    topologyDisplacementRingsPerAdvance: values.topologyDisplacementRingsPerAdvance,
    losassoFreeSurfacePressure: values.losassoFreeSurfacePressure,
    losassoVelocityExtension: values.losassoVelocityExtension,
  });
  // Not a product control. Keep the fine-only override available to the Dawn
  // harness for fault injection and planner isolation without letting normal
  // configurations drift into the restriction-starvation pairing.
  const diagnosticFineBand = typeof values.fineLevelSetBandCells === "number"
    && Number.isFinite(values.fineLevelSetBandCells)
    ? Math.max(0, Math.min(32, Math.round(values.fineLevelSetBandCells)))
    : bandReachCells;
  return {
    coarseDynamics,
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
  label: "Adaptive octree",
  shortLabel: "Octree",
  badge: "OCTREE",
  description: "GPU-resident adaptive liquid simulation with selectable Losasso coarse dynamics and a frozen Power 2017 reference backend.",
  detail: "Shared fine-band transport over a construction-time coarse-dynamics seam. Losasso uses a uniformly-fine surface shell, axis-aligned face velocities, and wide MGPCG; Power 2017 retains the frozen power-cell pipeline for reference lanes.",
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
  pressureMapping: "Losasso uses the wide, warm-started V-cycle-preconditioned MGPCG authority. The frozen Power 2017 backend retains its persistent Section 4.3 solver; neither backend falls through to the other after construction.",
  runtimeParamKeys: OCTREE_RUNTIME_DIAL_KEYS,
  presetFor: () => ({
    coarseBackend: DEFAULT_OCTREE_COARSE_BACKEND,
    losassoFreeSurfacePressure: "subcell-contact",
    losassoVelocityExtension: "fixed-jacobi",
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    surfaceRefinementGradingLayers: 1,
    globalFineLevelSetFactor: "1",
    topologyCadenceAdvances: 1,
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "planning" || phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
