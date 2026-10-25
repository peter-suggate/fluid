"use client";

import { sceneRenderTuning, type SceneRenderProfile } from "../../core/scene-render-profile";
import { useSession } from "../../core/session/session-context";
import { SVO_RENDER_TUNING_PRESETS, type SvoRenderTuning } from "./svo-render-tuning";

/** What a tuning key resets to in this scene: the balanced rung, at the scene's own refinement depth. */
export function svoSceneTuningDefaults(renderProfile: SceneRenderProfile | undefined): SvoRenderTuning {
  return renderProfile ? { ...SVO_RENDER_TUNING_PRESETS.balanced,
    environmentRefinementDepth: sceneRenderTuning(renderProfile).environmentRefinementDepth,
  } : SVO_RENDER_TUNING_PRESETS.balanced;
}

/** One tuning key as a control reads it, for views placed outside the pipeline overlay. */
export function useSvoRenderTuningValue<K extends keyof SvoRenderTuning>(key: K) {
  const session = useSession();
  const value = session.ui((state) => state.svoRenderTuning[key]);
  const renderProfile = session.scene((state) => state.scene.renderProfile);
  const fallback = svoSceneTuningDefaults(renderProfile)[key];
  const set = (next: SvoRenderTuning[K]) =>
    session.ui.getState().setSvoRenderTuning((current) => ({ ...current, [key]: next }));
  return { value, set, modified: value !== fallback, reset: () => set(fallback) };
}
