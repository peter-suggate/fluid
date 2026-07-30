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

/** Runtime-adjustable sparse-presentation controls. Shader loops retain hard caps;
 * these values only lower work or adjust quality inside those audited bounds. */
export interface SvoRenderTuning {
  readonly resolutionScale: number;
  readonly coneLightingScale: SvoConeLightingScale;
  readonly coneRadianceReconstruction: SvoConeRadianceReconstruction;
  readonly temporalEnabled: boolean;
  readonly checkerboardShadowsEnabled: boolean;
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
  readonly coneNormalEscapeCells: number;
  readonly coneEmitterClearanceCells: number;
  readonly temporalMaximumSamples: number;
  readonly temporalVarianceSigma: number;
  readonly temporalDepthToleranceScale: number;
}

export const DEFAULT_SVO_RENDER_TUNING: SvoRenderTuning = Object.freeze({
  resolutionScale: 0.72,
  coneLightingScale: 0.5,
  coneRadianceReconstruction: "full-res-relight",
  temporalEnabled: true,
  checkerboardShadowsEnabled: true,
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
  coneNormalEscapeCells: 0.5,
  coneEmitterClearanceCells: 3,
  temporalMaximumSamples: 64,
  temporalVarianceSigma: 2,
  temporalDepthToleranceScale: 1,
});

export const SVO_RENDER_TUNING_PRESETS = Object.freeze({
  performance: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 0.5,
    coneLightingScale: 0.25 as const,
    primaryLeafVisits: 24,
    coneStepBudget: 20,
    maximumShadedLights: 3,
    stableAreaLightSamples: 1,
    stableAoSamples: 2,
    visibilityNodeVisits: 48,
    visibilityLeafVisits: 12,
    visibilityWorkItems: 320,
    temporalMaximumSamples: 32,
  }),
  balanced: DEFAULT_SVO_RENDER_TUNING,
  quality: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 1,
    coneLightingScale: 1 as const,
    checkerboardShadowsEnabled: false,
    primaryLeafVisits: 128,
    visibilityNodeVisits: 128,
    visibilityLeafVisits: 32,
    visibilityWorkItems: 1024,
    temporalMaximumSamples: 96,
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
    coneLightingScale,
    coneRadianceReconstruction,
    temporalEnabled: Boolean(value.temporalEnabled),
    checkerboardShadowsEnabled: Boolean(value.checkerboardShadowsEnabled),
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
    coneNormalEscapeCells: bounded(value.coneNormalEscapeCells, 0, 2),
    coneEmitterClearanceCells: bounded(value.coneEmitterClearanceCells, 0, 8),
    temporalMaximumSamples: integer(value.temporalMaximumSamples, 1, 128),
    temporalVarianceSigma: bounded(value.temporalVarianceSigma, 0.5, 4),
    temporalDepthToleranceScale: bounded(value.temporalDepthToleranceScale, 0.25, 4),
  };
}

export function svoRenderTuningKey(value: SvoRenderTuning): string {
  return Object.values(value).join(":");
}
