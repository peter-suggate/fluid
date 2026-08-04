import type { SvoConeLightingScale } from "./webgpu-svo-dry-scene";

/** Full-resolution reconstruction policy for reduced visibility and opaque radiance. */
export type SvoConeRadianceReconstruction =
  | "nearest" | "gated-linear" | "joint-bilateral" | "wide-relight" | "full-res-relight";

export const SVO_CONE_RADIANCE_RECONSTRUCTION_CODES = Object.freeze({
  nearest: 0,
  "gated-linear": 1,
  "joint-bilateral": 2,
  "wide-relight": 3,
  "full-res-relight": 4,
} satisfies Record<SvoConeRadianceReconstruction, number>);

/** Audited compile-time ceiling for primary-ray leaf continuation. */
export const SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT = 256;

/**
 * Ceiling on the analytic band's budget: the authored record ceiling itself.
 *
 * Mirrors `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` deliberately rather than
 * importing it — this module is on the browser's tuning path and has no other
 * reason to pull in the candidate BVH — and a budget above the record ceiling
 * is indistinguishable from an unbounded one anyway.
 */
export const SVO_NEAR_FIELD_BAND_MAXIMUM_RECORDS = 16_384;

/** Additional authored-environment subdivision beyond the legacy SVO brick plan. */
export const SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM = 1;

/**
 * Extra octree levels dense environment regions may descend into.
 *
 * Distinct from `environmentBrickRefinementLevels`, which *coarsens* toward the
 * solver's lattice. This spends depth below it, and only on a scene the solver
 * does not own — the domain planner claims every brick of a simulated
 * container, and a solver brick pins its node. Two is the measured point where
 * the hero garden's busiest brick reaches the 64 candidates the hierarchy binds;
 * beyond that nothing more splits.
 */
export const SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM = 3;

/** Runtime-adjustable sparse-presentation controls. Shader loops retain hard caps;
 * these values only lower work or adjust quality inside those audited bounds. */
export interface SvoRenderTuning {
  readonly resolutionScale: number;
  readonly environmentBrickRefinementLevels: number;
  /** Extra octree levels for dense environment regions on an unsimulated scene. */
  readonly environmentRefinementDepth: number;
  readonly coneLightingScale: SvoConeLightingScale;
  readonly coneRadianceReconstruction: SvoConeRadianceReconstruction;
  /** Reuse an exact primary G-buffer while an eligible camera and scene remain unchanged. */
  readonly stationaryPrimaryReuseEnabled: boolean;
  readonly primaryLeafVisits: number;
  readonly coneStepBudget: number;
  readonly maximumShadedLights: number;
  readonly stableAreaLightSamples: number;
  readonly movingAreaLightSamples: number;
  readonly stableAoSamples: number;
  readonly movingAoSamples: number;
  readonly visibilityNodeVisits: number;
  readonly visibilityLeafVisits: number;
  readonly visibilityWorkItems: number;
  readonly visibilityIntersections: number;
  readonly shadowBiasCells: number;
  readonly shadowStrength: number;
  readonly aoRadiusScale: number;
  readonly aoStrength: number;
  readonly aoConeAperture: number;
  readonly shadowConeAperture: number;
  /** Artistic exposure applied only to gathered diffuse radiance. */
  readonly giBounceStrength: number;
  /** Broad visibility recovered from the diffuse GI cones. */
  readonly giOcclusionStrength: number;
  /** Analytic diffuse-environment contribution while GI is active. */
  readonly giEnvironmentStrength: number;
  /** Live analytic direct-light contribution, independent of derived-GI readiness. */
  readonly giDirectStrength: number;
  readonly giConeAperture: number;
  readonly giConeCount: number;
  readonly coneNormalEscapeCells: number;
  readonly coneEmitterClearanceCells: number;
  /**
   * Near-field analytic band: projected voxel size, in pixels, above which an
   * authored record is still drawn analytically through the coverage arena.
   *
   * Zero disables the band and every record stays analytic, which is the exact
   * historical image. A positive threshold hands everything below it to the
   * voxel primary, where the owning record's own surface is still what resolves
   * the hit — the accelerator changes, the authority does not.
   *
   * See `lib/svo-scene-primitive-band.ts` for the measure and the budget.
   */
  readonly nearFieldBandPixels: number;
  /** Exit threshold as a fraction of the entry threshold. One removes hysteresis. */
  readonly nearFieldBandHysteresis: number;
  /** Hard ceiling on analytic records per frame. Zero leaves the band unbounded. */
  readonly nearFieldBandBudget: number;
}

/**
 * Diagnostic overrides for the band, read once at module load.
 *
 * The acceptance lanes for this policy (`tools/benchmark-svo-dry-frame-gpu.ts`,
 * `tools/run-svo-dry-render-smoke.ts`) build their tuning by spreading
 * `DEFAULT_SVO_RENDER_TUNING`, so an A/B over the band would otherwise mean
 * editing a constant between arms — and the one measurement rule this program
 * has is that arms are interleaved rather than run in blocks. These are read
 * exactly like the octree solver's diagnostic switches and default to the
 * authored production values.
 */
const bandEnvironment = typeof process !== "undefined" ? process.env : undefined;
const bandOverride = (name: string, fallback: number): number => {
  const raw = bandEnvironment?.[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be a non-negative finite number`);
  return value;
};

export const DEFAULT_SVO_RENDER_TUNING: SvoRenderTuning = Object.freeze({
  resolutionScale: 0.72,
  environmentBrickRefinementLevels: 1,
  // Off by default: it changes leaf sizes, and the excursion budget a swaying
  // prop is held to is still derived from one scene-wide cell size.
  environmentRefinementDepth: 0,
  // The balanced tier keeps the accepted 2x2 visibility error bar. The 4x4
  // rate remains available in the performance preset, while full-resolution
  // relighting preserves material and edge detail at either reduced rate.
  coneLightingScale: 0.5,
  coneRadianceReconstruction: "full-res-relight",
  stationaryPrimaryReuseEnabled: false,
  primaryLeafVisits: 48,
  coneStepBudget: 48,
  maximumShadedLights: 8,
  stableAreaLightSamples: 2,
  movingAreaLightSamples: 1,
  stableAoSamples: 4,
  movingAoSamples: 1,
  visibilityNodeVisits: 96,
  visibilityLeafVisits: 24,
  visibilityWorkItems: 768,
  visibilityIntersections: 4,
  shadowBiasCells: 0.02,
  shadowStrength: 1,
  aoRadiusScale: 1,
  aoStrength: 1,
  aoConeAperture: 0.62,
  shadowConeAperture: 0.065,
  // GI is deliberately image-forward by default: the exact key light remains
  // crisp, while bounce and broad cone visibility visibly shape the scene.
  giBounceStrength: 1.8,
  // Multi-bounce compensation already restores much of the energy hidden by
  // broad cone occlusion. Retaining the old .82 contrast double-darkened room
  // corners and amplified 8-bit opacity contours into visible bands.
  giOcclusionStrength: 0.65,
  giEnvironmentStrength: 0.85,
  // Direct light is evaluated from the current scene-light arena every frame.
  // Derived radiance contains emitted energy, not a baked replacement for the
  // authored lights, so enabling GI must not silently remove ten percent of it.
  giDirectStrength: 1,
  giConeAperture: 1.05,
  giConeCount: 4,
  coneNormalEscapeCells: 0.5,
  coneEmitterClearanceCells: 3,
  // Off by default until the band's silhouette evidence is authored per scene:
  // zero reproduces the historical analytic set exactly, so enabling the voxel
  // primary underneath it (which only ever seeds a tighter depth) cannot be
  // confused with the band's own image effect.
  nearFieldBandPixels: bandOverride("FLUID_SVO_BAND_PIXELS", 0),
  nearFieldBandHysteresis: bandOverride("FLUID_SVO_BAND_HYSTERESIS", 0.8),
  nearFieldBandBudget: bandOverride("FLUID_SVO_BAND_BUDGET", 256),
});

export const SVO_RENDER_TUNING_PRESETS = Object.freeze({
  performance: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 0.5,
    coneLightingScale: 0.25 as const,
    primaryLeafVisits: 24,
    coneStepBudget: 20,
    giConeCount: 3,
    giBounceStrength: 1.35,
    giOcclusionStrength: 0.6,
    maximumShadedLights: 3,
    stableAreaLightSamples: 1,
    stableAoSamples: 2,
    visibilityNodeVisits: 48,
    visibilityLeafVisits: 12,
    visibilityWorkItems: 320,
  }),
  balanced: DEFAULT_SVO_RENDER_TUNING,
  quality: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 1,
    coneLightingScale: 0.5 as const,
    primaryLeafVisits: 128,
    coneStepBudget: 48,
    giConeCount: 4,
    visibilityNodeVisits: 128,
    visibilityLeafVisits: 32,
    visibilityWorkItems: 1024,
  }),
});

export type SvoRenderTuningPreset = keyof typeof SVO_RENDER_TUNING_PRESETS;

const bounded = (value: number, minimum: number, maximum: number) =>
  Math.max(minimum, Math.min(maximum, Number.isFinite(value) ? value : minimum));
const integer = (value: number, minimum: number, maximum: number) =>
  Math.round(bounded(value, minimum, maximum));

export function normalizeSvoRenderTuning(value: SvoRenderTuning): SvoRenderTuning {
  const coneLightingScale = value.coneLightingScale === 0.125 || value.coneLightingScale === 0.25 || value.coneLightingScale === 0.5
    ? value.coneLightingScale : 1;
  const coneRadianceReconstruction = value.coneRadianceReconstruction === "nearest"
    || value.coneRadianceReconstruction === "gated-linear"
    || value.coneRadianceReconstruction === "joint-bilateral"
    || value.coneRadianceReconstruction === "wide-relight"
    || value.coneRadianceReconstruction === "full-res-relight"
    ? value.coneRadianceReconstruction : "full-res-relight";
  return {
    resolutionScale: bounded(value.resolutionScale, 0.35, 1),
    environmentBrickRefinementLevels: integer(
      value.environmentBrickRefinementLevels,
      0,
      SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
    ),
    environmentRefinementDepth: integer(
      value.environmentRefinementDepth,
      0,
      SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
    ),
    coneLightingScale,
    coneRadianceReconstruction,
    stationaryPrimaryReuseEnabled: Boolean(value.stationaryPrimaryReuseEnabled),
    primaryLeafVisits: integer(value.primaryLeafVisits, 1, SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT),
    coneStepBudget: integer(value.coneStepBudget, 1, 48),
    maximumShadedLights: integer(value.maximumShadedLights, 1, 8),
    stableAreaLightSamples: integer(value.stableAreaLightSamples, 1, 2),
    movingAreaLightSamples: integer(value.movingAreaLightSamples, 1, 2),
    stableAoSamples: integer(value.stableAoSamples, 1, 4),
    movingAoSamples: integer(value.movingAoSamples, 1, 4),
    visibilityNodeVisits: integer(value.visibilityNodeVisits, 1, 128),
    visibilityLeafVisits: integer(value.visibilityLeafVisits, 1, 32),
    visibilityWorkItems: integer(value.visibilityWorkItems, 16, 1024),
    visibilityIntersections: integer(value.visibilityIntersections, 1, 4),
    shadowBiasCells: bounded(value.shadowBiasCells, 0, 0.25),
    shadowStrength: bounded(value.shadowStrength, 0, 1),
    aoRadiusScale: bounded(value.aoRadiusScale, 0.1, 3),
    aoStrength: bounded(value.aoStrength, 0, 1),
    aoConeAperture: bounded(value.aoConeAperture, 0.1, 1.4),
    shadowConeAperture: bounded(value.shadowConeAperture, 0.01, 0.25),
    giBounceStrength: bounded(value.giBounceStrength ?? DEFAULT_SVO_RENDER_TUNING.giBounceStrength, 0, 4),
    giOcclusionStrength: bounded(value.giOcclusionStrength ?? DEFAULT_SVO_RENDER_TUNING.giOcclusionStrength, 0, 1),
    giEnvironmentStrength: bounded(value.giEnvironmentStrength ?? DEFAULT_SVO_RENDER_TUNING.giEnvironmentStrength, 0, 2),
    giDirectStrength: bounded(value.giDirectStrength ?? DEFAULT_SVO_RENDER_TUNING.giDirectStrength, 0, 2),
    giConeAperture: bounded(value.giConeAperture ?? DEFAULT_SVO_RENDER_TUNING.giConeAperture, 0.4, 1.4),
    giConeCount: integer(value.giConeCount ?? DEFAULT_SVO_RENDER_TUNING.giConeCount, 3, 4),
    coneNormalEscapeCells: bounded(value.coneNormalEscapeCells, 0, 2),
    coneEmitterClearanceCells: bounded(value.coneEmitterClearanceCells, 0, 8),
    nearFieldBandPixels: bounded(value.nearFieldBandPixels ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandPixels, 0, 4096),
    nearFieldBandHysteresis: bounded(value.nearFieldBandHysteresis ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandHysteresis, 0.25, 1),
    nearFieldBandBudget: integer(value.nearFieldBandBudget ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandBudget, 0, SVO_NEAR_FIELD_BAND_MAXIMUM_RECORDS),
  };
}

export function svoRenderTuningKey(value: SvoRenderTuning): string {
  return Object.values(value).join(":");
}
