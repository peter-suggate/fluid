import { create } from "zustand";
import type { SceneDescription } from "../model";
import { defaultScenePresetId, getScenePreset } from "../scenes";

/**
 * The scene description is the single source of truth for everything physical:
 * container geometry, fluid properties, initial condition, inflow, timing,
 * and the rigid-body roster. Mutations here are pure data edits; anything
 * that must also rebuild runtime state (solvers, body states) goes through
 * the simulation controller, which calls back into this store.
 */
interface SceneStore {
  scene: SceneDescription;
  presetId: string;
  setScene: (scene: SceneDescription, presetId?: string) => void;
  patchScene: (patch: Partial<SceneDescription>) => void;
  patchContainer: (patch: Partial<SceneDescription["container"]>) => void;
  patchFluid: (patch: Partial<SceneDescription["fluid"]>) => void;
  patchNumerics: (patch: Partial<SceneDescription["numerics"]>) => void;
}

export const useSceneStore = create<SceneStore>((set) => ({
  scene: getScenePreset(defaultScenePresetId).create(),
  presetId: defaultScenePresetId,
  setScene: (scene, presetId) => set((state) => ({ scene, presetId: presetId ?? state.presetId })),
  patchScene: (patch) => set((state) => ({ scene: { ...state.scene, ...patch } })),
  patchContainer: (patch) => set((state) => ({ scene: { ...state.scene, container: { ...state.scene.container, ...patch } } })),
  patchFluid: (patch) => set((state) => ({ scene: { ...state.scene, fluid: { ...state.scene.fluid, ...patch } } })),
  patchNumerics: (patch) => set((state) => ({ scene: { ...state.scene, numerics: { ...state.scene.numerics, ...patch } } }))
}));
