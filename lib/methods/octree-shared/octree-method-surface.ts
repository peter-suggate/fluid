import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";
import type { ResourcePluginDefinition } from "../../core/resource-plugin";
import type { OctreeCoarseDynamicsConfiguration } from "./octree-coarse-backend";
import { resolveOctreeRuntimeDials } from "./octree-runtime-dials";

/**
 * What the two adaptive-octree methods share at the *identity* layer.
 *
 * Losasso and Power Liquids are separate registered methods over one shared
 * engine, so the structural knobs that size the same engine — leaf ceiling,
 * band reach, grading layers — have to mean exactly the same thing in both
 * schemas. Declaring them once is what keeps a range or a default from
 * drifting between two method modules that must stay comparable; everything
 * that genuinely differs (surface tracking, velocity extension, cadence,
 * dials) is declared by the owning method instead.
 */
export const OCTREE_STRUCTURAL_PARAMS: readonly MethodParamSpec[] = Object.freeze([
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "32", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells · default" }], hint: "Largest dyadic octree cell away from interfaces. Scene profiles choose the largest compatible root while preserving strict 2:1 grading." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Band reach", unit: "level", min: 0, max: 4, step: 1, digits: 0, default: 4, tier: "fine", hint: "One coupled reach level for pressure refinement and Section 5 surface tracking. Experimental level 0 uses one fine brick; level 1 retains the two-finest-cell moving-surface floor while still reducing pressure reach and recurring residency. Level 4 is the paper/default reach. This is the STRUCTURAL band: it sizes the pressure row capacity, the residency halo, the candidate dilation rings and the redistance reach, so changing it rebuilds. The performance strip carries a live Surface band dial that thins this one without a rebuild, and is clamped to it." },
  { kind: "number", key: "surfaceRefinementGradingLayers", label: "Surface grading", unit: "layers", min: 1, max: 4, step: 1, digits: 0, default: 1, tier: "fine", hint: "Intermediate pressure-cell layers retained per octree level around the surface. 1 is the existing sharp 2:1 transition; 3 is the progressive-refinement experiment." },
] as const satisfies readonly MethodParamSpec[]);

/**
 * The readiness copy for the shared staged constructor.
 *
 * Both methods build through `WebGPUOctreeEulerianSolver`, so they cross the
 * same phases in the same order; two hand-maintained copies of this table
 * would let one lane silently describe a phase the other renamed.
 */
export const OCTREE_RESOURCE_PHASE_COPY: ResourcePluginDefinition["phaseCopy"] = Object.freeze({
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
});

export const octreeMaximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 32;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

export const octreeGlobalFineLevelSetFactor = (value: unknown): 1 | 4 | 8 => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 1;
  return numeric === 1 ? 1 : numeric >= 8 ? 8 : 4;
};

/** Canonical browser/native construction options for an adaptive-octree method. */
export function octreeSolverOptions(
  params: readonly MethodParamSpec[],
  coarseDynamics: OctreeCoarseDynamicsConfiguration,
  values: MethodParamValues,
) {
  const bandReachCells = numberValue(values, params, "interfaceRefinementBandCells");
  const fineFactor = octreeGlobalFineLevelSetFactor(values.globalFineLevelSetFactor);
  // Not a product control. Keep the fine-only override available to the Dawn
  // harness for fault injection and planner isolation without letting normal
  // configurations drift into the restriction-starvation pairing.
  const diagnosticFineBand = typeof values.fineLevelSetBandCells === "number"
    && Number.isFinite(values.fineLevelSetBandCells)
    ? Math.max(0, Math.min(32, Math.round(values.fineLevelSetBandCells)))
    : bandReachCells;
  return {
    coarseDynamics,
    octree: {
      maximumLeafSize: octreeMaximumLeafSize(values.maximumLeafSize ?? 32),
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
      // Seed t=0 with the live thickness too. Allocations still use the two
      // authored fields above; this only prevents the fenced cold topology
      // from displaying and solving the widest mesh until its first epoch.
      initialRuntimeDials: resolveOctreeRuntimeDials(values),
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
      pressureRowCapacity: typeof values.pressureRowCapacity === "number"
        && values.pressureRowCapacity > 0
        ? values.pressureRowCapacity : undefined,
    }
  };
}
