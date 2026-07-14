import { create } from "zustand";
import type { GPUQuality } from "../tall-cell-grid";
import { defaultMethodId, getMethod, resolveMethodValues, type MethodParamValue, type MethodParamValues } from "../methods";

/**
 * Which simulation method runs, at what quality, and any sparse per-method
 * parameter overrides on top of the quality preset. Overrides are keyed by
 * method id so switching methods round-trips user tuning.
 */
interface MethodStore {
  methodId: string;
  quality: GPUQuality;
  overrides: Record<string, MethodParamValues>;
  setMethodId: (methodId: string) => void;
  setQuality: (quality: GPUQuality) => void;
  setParam: (methodId: string, key: string, value: MethodParamValue) => void;
  resetParam: (methodId: string, key: string) => void;
  resetParams: (methodId: string) => void;
}

export const useMethodStore = create<MethodStore>((set) => ({
  methodId: defaultMethodId,
  quality: "balanced",
  overrides: {},
  setMethodId: (methodId) => set({ methodId }),
  setQuality: (quality) => set({ quality }),
  setParam: (methodId, key, value) => set((state) => ({ overrides: { ...state.overrides, [methodId]: { ...state.overrides[methodId], [key]: value } } })),
  resetParam: (methodId, key) => set((state) => {
    const rest = { ...(state.overrides[methodId] ?? {}) };
    delete rest[key];
    return { overrides: { ...state.overrides, [methodId]: rest } };
  }),
  resetParams: (methodId) => set((state) => ({ overrides: { ...state.overrides, [methodId]: {} } }))
}));

/** Effective values for the active method: defaults ← quality preset ← user overrides. */
export function resolvedMethodValues(state: Pick<MethodStore, "methodId" | "quality" | "overrides">): MethodParamValues {
  return resolveMethodValues(getMethod(state.methodId), state.quality, state.overrides[state.methodId] ?? {});
}
