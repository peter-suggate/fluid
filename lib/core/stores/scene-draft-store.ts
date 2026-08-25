import { useMemo } from "react";
import { create } from "zustand";
import { markSceneRevision, type SceneDescription } from "../model";
import { useSceneStore } from "./scene-store";

/**
 * The client-only half of direct manipulation.
 *
 * A drag proposes a scene; the committed document is what the physics runs. The
 * two are separated because the scene document *is* the GPU solver's rebuild
 * identity (`gpuSceneSolverKey`): writing it on every pointer-move asks the
 * renderer to re-seed the solver at pointer rate, which is what turns a smooth
 * gesture into a slideshow. So a gesture writes here — free, immediate, no GPU
 * — and spends exactly one document write when the pointer comes up.
 *
 * The draft is a *patch over the scene*, not a bespoke per-gesture shape. That
 * is what makes it general: any component that already reads the scene to draw
 * something can read `useDisplayScene()` instead and immediately follow the
 * drag, with no knowledge of which gesture is open or what it is editing.
 *
 * The division of labour is strict, and it is the whole contract:
 *
 * - **Display** reads `useDisplayScene()` — handles, outlines, overlays, HUD
 *   readouts. Always current with the pointer.
 * - **Physics** reads `useSceneStore` — the solver, the rebuild keys, the CPU
 *   oracle, serialization, history. Never sees a draft.
 *
 * A draft is therefore never persisted, never undoable, and never valid input
 * to `validateScene`: it is a proposal that has not happened yet.
 */

export interface SceneDraft {
  /** What the open gesture will be called in history when it commits. */
  readonly label: string;
  /** Patch over the committed scene. */
  readonly patch: Partial<SceneDescription>;
  /**
   * What the gesture is manipulating, for overlays that style or label the
   * proposal differently from the committed scene.
   */
  readonly subject: SceneDraftSubject;
}

export type SceneDraftSubject =
  | "tank"
  | "solid-voxels"
  | "fluid-body"
  | "body"
  | "scenery"
  | "fill-level"
  | "inflow"
  | "terrain"
  | "refinement-region";

interface SceneDraftStore {
  draft?: SceneDraft;
  /** Open a gesture. Replaces any draft left behind by an interrupted one. */
  beginDraft: (subject: SceneDraftSubject, label: string) => void;
  /**
   * Revise the open gesture's proposal. Ignored when no gesture is open, so a
   * stray pointer-move after a cancel cannot resurrect a draft.
   */
  updateDraft: (patch: Partial<SceneDescription>) => void;
  clearDraft: () => void;
}

export const useSceneDraftStore = create<SceneDraftStore>((set) => ({
  draft: undefined,
  beginDraft: (subject, label) => set({ draft: { subject, label, patch: {} } }),
  updateDraft: (patch) => set((state) => state.draft
    ? { draft: { ...state.draft, patch } }
    : {}),
  clearDraft: () => set((state) => (state.draft ? { draft: undefined } : {})),
}));

/** The scene as the pointer currently proposes it. */
let cachedDraftBase: SceneDescription | undefined;
let cachedDraft: SceneDraft | undefined;
let cachedDraftScene: SceneDescription | undefined;

export function applySceneDraft(
  scene: SceneDescription,
  draft: SceneDraft | undefined,
): SceneDescription {
  if (!draft) return scene;
  if (scene === cachedDraftBase && draft === cachedDraft && cachedDraftScene) return cachedDraftScene;
  cachedDraftBase = scene;
  cachedDraft = draft;
  cachedDraftScene = markSceneRevision({ ...scene, ...draft.patch });
  return cachedDraftScene;
}

/**
 * The scene to *draw*. Returns the committed scene identity unchanged when no
 * gesture is open, so nothing downstream re-renders or re-derives while the
 * scene is at rest.
 */
export function useDisplayScene(): SceneDescription {
  const scene = useSceneStore((state) => state.scene);
  const draft = useSceneDraftStore((state) => state.draft);
  return useMemo(() => applySceneDraft(scene, draft), [scene, draft]);
}

/** Non-reactive read, for the render loop and pointer handlers. */
export function displaySceneSnapshot(): SceneDescription {
  return applySceneDraft(useSceneStore.getState().scene, useSceneDraftStore.getState().draft);
}
