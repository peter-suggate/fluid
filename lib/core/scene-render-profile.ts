import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM, type SvoRenderTuning } from "../svo/pipeline/svo-render-tuning";

/** Authored presentation defaults, independent of the fluid lattice and solver profile. */
export interface SceneRenderProfile {
  readonly resolutionScale?: number;
  readonly stillResolutionScale?: number;
  /** Maximum shaded pixels at the authored playback scale, independent of display density. */
  readonly playbackPixelBudget?: number;
  readonly interpolateWater?: boolean;
  readonly edgeAntialiasing?: boolean;
  readonly shadowBiasCells?: number;
  readonly environmentRefinementDepth?: number;
  readonly fluidSurfaceRenderMode?: "shaded" | "wireframe" | "simple";
  readonly fluidParticleDisplay?: "off" | "motion" | "speed" | "fine" | "water";
}

export function sceneRenderProfileErrors(profile: SceneRenderProfile): string[] {
  if (!profile || typeof profile !== "object" || Array.isArray(profile)) return ["Scene render profile must be an object"];
  if (profile.interpolateWater !== undefined && typeof profile.interpolateWater !== "boolean") return ["Water interpolation must be boolean"];
  if (profile.edgeAntialiasing !== undefined && typeof profile.edgeAntialiasing !== "boolean") return ["Scene edge antialiasing must be boolean"];
  if (profile.playbackPixelBudget !== undefined && (!Number.isSafeInteger(profile.playbackPixelBudget) || profile.playbackPixelBudget < 1)) return ["Playback pixel budget must be a positive integer"];
  if (profile.stillResolutionScale !== undefined && (!Number.isFinite(profile.stillResolutionScale) || profile.stillResolutionScale < .35 || profile.stillResolutionScale > 2)) return ["Still resolution scale must be between 0.35 and 2"];
  if (profile.resolutionScale !== undefined && (!Number.isFinite(profile.resolutionScale) || profile.resolutionScale < .35 || profile.resolutionScale > 2)) return ["Scene render resolution scale must be between 0.35 and 2"];
  const bias = profile.shadowBiasCells;
  if (bias !== undefined && (!Number.isFinite(bias) || bias < 0 || bias > .25)) return ["Shadow bias must be between 0 and .25 cells"];
  const depth = profile.environmentRefinementDepth;
  if (profile.fluidParticleDisplay !== undefined && !["off", "motion", "speed", "fine", "water"].includes(profile.fluidParticleDisplay)) return ["Scene fluid particle display must be off, motion, speed, fine or water"];
  if (profile.fluidSurfaceRenderMode !== undefined && !["shaded", "wireframe", "simple"].includes(profile.fluidSurfaceRenderMode)) return ["Scene fluid surface render mode must be shaded, wireframe or simple"];
  return depth !== undefined && (!Number.isInteger(depth) || depth < 0 || depth > SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM)
    ? [`Scene environment refinement depth must be an integer from 0 to ${SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}`] : [];
}

const defaults = new Map<string, SvoRenderTuning>();
/** An explicit complete tuning always wins, including values equal to global defaults. */
export function sceneRenderTuning(profile?: SceneRenderProfile, explicit?: SvoRenderTuning): SvoRenderTuning {
  if (explicit) return explicit;
  const depth = profile?.environmentRefinementDepth;
  const scale = profile?.resolutionScale;
  const bias = profile?.shadowBiasCells;
  if (depth === undefined && scale === undefined && bias === undefined) return DEFAULT_SVO_RENDER_TUNING;
  const errors = sceneRenderProfileErrors(profile!);
  if (errors.length) throw new RangeError(errors[0]);
  const environmentRefinementDepth = depth ?? DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth;
  const resolutionScale = scale ?? DEFAULT_SVO_RENDER_TUNING.resolutionScale;
  const shadowBiasCells = bias ?? DEFAULT_SVO_RENDER_TUNING.shadowBiasCells;
  const key = `${environmentRefinementDepth}/${resolutionScale}/${shadowBiasCells}`;
  let result = defaults.get(key);
  if (!result) {
    result = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, environmentRefinementDepth, resolutionScale, shadowBiasCells });
    defaults.set(key, result);
  }
  return result;
}
