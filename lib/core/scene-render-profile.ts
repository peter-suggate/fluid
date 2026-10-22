import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM, type SvoRenderTuning } from "../svo/pipeline/svo-render-tuning";

/** Authored presentation defaults, independent of the fluid lattice and solver profile. */
export interface SceneRenderProfile {
  readonly environmentRefinementDepth?: number;
}

export function sceneRenderProfileErrors(profile: SceneRenderProfile): string[] {
  if (!profile || typeof profile !== "object" || Array.isArray(profile)) return ["Scene render profile must be an object"];
  const depth = profile.environmentRefinementDepth;
  return depth !== undefined && (!Number.isInteger(depth) || depth < 0 || depth > SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM)
    ? [`Scene environment refinement depth must be an integer from 0 to ${SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}`] : [];
}

const defaults = new Map<number, SvoRenderTuning>();
/** An explicit complete tuning always wins, including values equal to global defaults. */
export function sceneRenderTuning(profile?: SceneRenderProfile, explicit?: SvoRenderTuning): SvoRenderTuning {
  if (explicit) return explicit;
  const depth = profile?.environmentRefinementDepth;
  if (depth === undefined) return DEFAULT_SVO_RENDER_TUNING;
  const errors = sceneRenderProfileErrors(profile!);
  if (errors.length) throw new RangeError(errors[0]);
  let result = defaults.get(depth);
  if (!result) {
    result = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, environmentRefinementDepth: depth });
    defaults.set(depth, result);
  }
  return result;
}
