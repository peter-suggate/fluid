import { WebGPUOctreeEulerianSolver } from "../webgpu-octree-eulerian";
import { VISUALIZATION_FIELDS } from "../visualization-catalog";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";
import {
  DEFAULT_OCTREE_COARSE_BACKEND,
  octreeCoarseBackend,
  resolveOctreeCoarseDynamics,
} from "../octree-coarse-backend";
import {
  OCTREE_RUNTIME_DIALS,
  OCTREE_RUNTIME_DIAL_KEYS,
  resolveOctreeRuntimeDials,
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
  { kind: "select", key: "coarseBackend", label: "Coarse dynamics", default: DEFAULT_OCTREE_COARSE_BACKEND, tier: "coarse", options: [{ value: "losasso", label: "Losasso 2004 · default" }, { value: "power2017", label: "Power 2017 · factor-4 benchmark" }], hint: "Construction-time backend choice. Each backend owns distinct pipelines, layouts, and velocity channels. Power selects the paper's separate 4× narrow-band level set and rebuilds topology every advance." },
  { kind: "select", key: "losassoVelocityExtension", label: "Losasso extrapolation", default: "fixed-jacobi", tier: "fine", options: [{ value: "fixed-jacobi", label: "Fixed Jacobi · default" }, { value: "causal-front", label: "Causal layer front" }], hint: "Construction-time A/B control for Section 5 air velocity extension. Causal-front publishes one graph layer per sweep from already-valid inner layers." },
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "1", tier: "coarse", options: [{ value: "1", label: "Adaptive finest surface · default" }, { value: "4", label: "4× subcell surface" }, { value: "8", label: "8× subcell surface · experimental" }], hint: "Ando-style factor 1 remeshes the octree itself: every leaf cut by the moving interface stays at the finest pressure/level-set tier, while pure liquid and air grade rapidly to coarse cells away from it. Fixed-point mass is handed conservatively between old and new leaves. Factors 4/8 instead opt into a separate sparse subcell interface band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "32", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells · default" }], hint: "Largest dyadic octree cell away from interfaces. Scene profiles choose the largest compatible root while preserving strict 2:1 grading." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Band reach", unit: "level", min: 0, max: 4, step: 1, digits: 0, default: 4, tier: "fine", hint: "One coupled reach level for pressure refinement and Section 5 surface tracking. Experimental level 0 uses one fine brick; level 1 retains the two-finest-cell moving-surface floor while still reducing pressure reach and recurring residency. Level 4 is the paper/default reach. This is the STRUCTURAL band: it sizes the pressure row capacity, the residency halo, the candidate dilation rings and the redistance reach, so changing it rebuilds. The performance strip carries a live Surface band dial that thins this one without a rebuild, and is clamped to it." },
  { kind: "number", key: "surfaceRefinementGradingLayers", label: "Surface grading", unit: "layers", min: 1, max: 4, step: 1, digits: 0, default: 1, tier: "fine", hint: "Intermediate pressure-cell layers retained per octree level around the surface. 1 is the existing sharp 2:1 transition; 3 is the progressive-refinement experiment." },
  { kind: "number", key: "topologyCadenceAdvances", label: "Topology cadence", unit: "advances", min: 1, max: 8, step: 1, digits: 0, default: 8, tier: "fine", hint: "Losasso candidate epochs cover eight accepted advances by default. Each skipped rebuild is represented spatially by an extra dilation ring, keeping the moving interface inside the resident band." },
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

/**
 * Power 2017 is exposed as one reproducible benchmark, not as a partially
 * compatible collection of old switches. The paper evolves a separate fine
 * SPGrid level set, typically at 4× or 8× the octree resolution; factor four
 * is this product's canonical reference lane. Normalizing here means the same
 * tuple is restored by the picker, a shared URL, and an authored scene profile.
 */
export function normalizeOctreeMethodValues(values: MethodParamValues): MethodParamValues {
  if (octreeCoarseBackend(values.coarseBackend) !== "power2017") return values;
  return {
    ...values,
    globalFineLevelSetFactor: "4",
    topologyCadenceAdvances: 1,
  };
}

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => {
  // Factories are public to browser and Dawn callers, so uphold the compound
  // backend invariant here as well as in generic method-value resolution.
  const normalizedValues = normalizeOctreeMethodValues(values);
  const bandReachCells = numberValue(normalizedValues, params, "interfaceRefinementBandCells");
  const fineFactor = globalFineLevelSetFactor(normalizedValues.globalFineLevelSetFactor);
  const coarseDynamics = resolveOctreeCoarseDynamics({
    backend: normalizedValues.coarseBackend,
    globalFineLevelSetFactor: fineFactor,
    topologyCadenceAdvances: normalizedValues.topologyCadenceAdvances,
    topologyDisplacementRingsPerAdvance: normalizedValues.topologyDisplacementRingsPerAdvance,
    losassoVelocityExtension: normalizedValues.losassoVelocityExtension,
  });
  // Not a product control. Keep the fine-only override available to the Dawn
  // harness for fault injection and planner isolation without letting normal
  // configurations drift into the restriction-starvation pairing.
  const diagnosticFineBand = typeof normalizedValues.fineLevelSetBandCells === "number"
    && Number.isFinite(normalizedValues.fineLevelSetBandCells)
    ? Math.max(0, Math.min(32, Math.round(normalizedValues.fineLevelSetBandCells)))
    : bandReachCells;
  return {
    coarseDynamics,
    octree: {
      maximumLeafSize: maximumLeafSize(normalizedValues.maximumLeafSize ?? 32),
      // Dry boundary adaptivity is the production pressure policy at every
      // surface resolution. False remains the exact Dawn control.
      fluidGatedBoundaryRefinement: normalizedValues.fluidGatedBoundaryRefinement !== false,
      environmentBrickRefinementLevels: typeof normalizedValues.svoEnvironmentBrickRefinementLevels === "number"
        ? normalizedValues.svoEnvironmentBrickRefinementLevels : undefined,
      // Pressure topology is a method setting, not a solver setting. Keep the
      // same adaptive octree when comparing the two pressure implementations.
      adaptivity: typeof normalizedValues.octreeAdaptivity === "number"
        && Number.isFinite(normalizedValues.octreeAdaptivity)
        ? Math.max(0, Math.min(1, normalizedValues.octreeAdaptivity))
        : 1,
      interfaceRefinementBandCells: bandReachCells,
      surfaceRefinementGradingLayers: numberValue(normalizedValues, params, "surfaceRefinementGradingLayers"),
      // Seed t=0 with the live thickness too. Allocations still use the two
      // authored fields above; this only prevents the fenced cold topology
      // from displaying and solving the widest mesh until its first epoch.
      initialRuntimeDials: resolveOctreeRuntimeDials(normalizedValues),
      fineLevelSetBandCells: diagnosticFineBand,
      globalFineLevelSetFactor: fineFactor,
      // Hidden authored/harness override for scenes whose fluid footprint is
      // intentionally much smaller than the container. The generic fallback
      // remains the conservative domain-cross-section estimate.
      globalFineLevelSetMaximumBricks:
        typeof normalizedValues.globalFineLevelSetMaximumBricks === "number"
          && Number.isSafeInteger(normalizedValues.globalFineLevelSetMaximumBricks)
          && normalizedValues.globalFineLevelSetMaximumBricks > 0
          ? normalizedValues.globalFineLevelSetMaximumBricks : undefined,
      // Diagnostic-only capacity override. Not an authored UI parameter: the
      // production capacity comes from `planOctreePressureCapacity`, and this
      // exists so a harness can bisect the arena demand of a large domain
      // against the on-GPU overflow report.
      pressureRowCapacity: typeof normalizedValues.pressureRowCapacity === "number"
        && normalizedValues.pressureRowCapacity > 0
        ? normalizedValues.pressureRowCapacity : undefined,
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
  // The octree publishes every catalog field: the technique overlays read its
  // compact debug source, and the generic dense-grid views read the diagnostic
  // textures it materializes on demand.
  supportedFieldModes: VISUALIZATION_FIELDS.map((field) => field.mode),
  params,
  pressureMapping: "Losasso uses the wide, warm-started V-cycle-preconditioned MGPCG authority. The frozen Power 2017 backend retains its persistent Section 4.3 solver; neither backend falls through to the other after construction.",
  runtimeParamKeys: OCTREE_RUNTIME_DIAL_KEYS,
  presetFor: () => ({
    coarseBackend: DEFAULT_OCTREE_COARSE_BACKEND,
    losassoVelocityExtension: "fixed-jacobi",
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    surfaceRefinementGradingLayers: 1,
    globalFineLevelSetFactor: "1",
    topologyCadenceAdvances: 8,
  }),
  normalizeValues: normalizeOctreeMethodValues,
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUOctreeEulerianSolver(device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUOctreeEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "planning" || phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
